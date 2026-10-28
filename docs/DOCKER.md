# GPT-Zero v3 离线镜像与运维手册

> Owner: `ops` ｜ 适用版本：GPT-Zero v3 ｜ 最后更新：2026-10-28
>
> 一句话：**一个 `python:3.11-slim` 起的小镜像，把「OpenAI 兼容 Mock 推理服务 + bench + agentx」
> 全部装进去；构建期走内网镜像源，运行期彻底断网也能跑。**

---

## 1. 镜像里有什么

| 内容 | 位置 | 说明 |
|---|---|---|
| `gptzero` 包（服务核心） | `/app/gptzero` | FastAPI + uvicorn，OpenAI 兼容 `--network none` 可跑 |
| `server.py` | `/app/server.py` | v2 兼容入口（等价 `python -m gptzero`） |
| `bench/`、`agentx/` | `/app/bench`、`/app/agentx` | 压测 harness 与 AgentX trace 回放（由队友维护，镜像按契约预留入口） |
| 本地 tokenizer | `/app/deepseek_v3_tokenizer` | 可选精确计数（需 `--build-arg WITH_TRANSFORMERS=1`） |
| 运行期依赖 | 系统 site-packages | fastapi / uvicorn / pydantic / httpx / numpy / matplotlib / pytest |
| 入口脚本 | `/app/docker/entrypoint.sh` | `mock|bench|agentx|smoke|shell` 子命令 |
| 请求日志卷 | `/data/requests.jsonl` | compose 挂 `./data`，容器重建不丢 |
| 结果目录 | `/app/results` | compose 挂 `./results`，bench/agentx 产物落宿主机 |

镜像**不含**：`.venv`、`.git`、`results/`、`dist/`、`data/`、任何 wheel/tar 包、模型权重、CUDA。

关键环境变量（镜像内已设默认值）：

```
PYTHONUNBUFFERED=1     PYTHONDONTWRITEBYTECODE=1   PIP_NO_CACHE_DIR=1
PYTHONPATH=/app        MPLBACKEND=Agg              MPLCONFIGDIR=/tmp/matplotlib
HF_HUB_OFFLINE=1       TRANSFORMERS_OFFLINE=1      GPTZERO_HOST=0.0.0.0  GPTZERO_PORT=8090
```

`HEALTHCHECK` 用 Python 标准库请求 `/health`（镜像里**没有** curl/wget，smoke 会断言这一点）。

---

## 2. 为什么「运行期零网络」

1. 依赖在 **builder 阶段**一次性装好，runtime 阶段只 `COPY --from=builder /install /usr/local`，
   没有任何在线安装动作。
2. `pip install -e . --no-deps --no-build-isolation`：只有「把本地源码挂进 site-packages」这一动作，
   不解析依赖图、不下载构建后端（setuptools/wheel 已在镜像里）。
3. tokenizer 只读仓库内 `deepseek_v3_tokenizer/`，并强制 `HF_HUB_OFFLINE=1`（transformers 不会尝试连 Hub）。
4. 图表用 `MPLBACKEND=Agg`，不弹窗、不需要字体外网下载。
5. 验证据此而来：`docker run --rm --network none gptzero:v3 smoke` **必须** 通过（见 §9 实测记录）。

---

## 3. 构建

前置：Docker Desktop 已启动，本机有 `python:3.11-slim`（或允许拉取一次）。

```bash
# 默认：不装 transformers（体积小，heuristic 分词）
docker build -t gptzero:v3 .

# 精确 token 计数：加装 transformers（约 +150~200MB）
docker build -t gptzero:v3 --build-arg WITH_TRANSFORMERS=1 .

# 强制重装依赖（改了 requirements 或怀疑缓存）
docker build --no-cache -t gptzero:v3 .
```

构建期使用的索引（可用 build-arg 覆盖）：

```bash
docker build -t gptzero:v3 \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  --build-arg PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn .
```

一键脚本（推荐）：

```powershell
.\scripts\offline_image.ps1                 # 构建 + save + sha256 + 打印 load 用法
.\scripts\offline_image.ps1 -NoBuild        # 只打包已有镜像
.\scripts\offline_image.ps1 -WithTransformers -Out dist/gptzero-v3-hf.tar
```

```bash
bash scripts/offline_image.sh               # Linux/macOS 等价物
bash scripts/offline_image.sh -n -o /tmp/gptzero-v3.tar
```

### 3.1 两种分词模式（体积 vs 精确度）

| 模式 | 构建参数 | 镜像体积（实测） | `prompt_tokens` | 适用 |
|---|---|---|---|---|
| heuristic（默认） | `WITH_TRANSFORMERS=0` | **388 MB**（`docker images`；分层压缩 88 MiB；`/app` 13 MB） | 按「CJK 单字 + 拉丁 4 字符」估算，仅用于模拟 | 只关心吞吐/延迟曲线、快速分发 |
| hf（精确） | `WITH_TRANSFORMERS=1` | **549 MB**（388 + 161 MB 的 transformers/tokenizers/huggingface_hub/safetensors） | 与真实 DeepSeek V3 tokenizer 一致（`tokenizers:deepseek_v3_tokenizer`） | 需要 `use_server_token_count` 对齐真实客户端计数 |

选择建议：

- **默认交付用精简版**。mock 的延迟/吞吐曲线由 `--ttft-ms` / `--tpot-ms`（或 `--prefill-tps` / `--decode-tps`）
  决定，与 tokenizer 无关；heuristic 对同一段文本的计数是确定性的，用来算 tok/s、缓存命中率完全够用，
  体积小 1/3，`docker load` 和分发都快。
- **需要和真实客户端对账时用精确版**：例如「客户端自己数的 prompt_tokens」与「服务端 usage.prompt_tokens」
  要逐条相等（AIPerf 的 `use_server_token_count`），或要复现真实 DeepSeek V3 的分词长度分布。
- 两种镜像的**接口、指标、smoke 检查完全一致**，只是启动横幅里的 `tokenizer` 一行不同
  （`heuristic` vs `hf:deepseek_v3_tokenizer`），因此可以随时互换。


服务启动时会打印实际生效的分词器：

```
  tokenizer    : heuristic            # 或 hf:deepseek_v3_tokenizer
```

缺 `transformers` 时**不会报错**，自动降级 heuristic 并打印 warning（契约 §2 允许）。
运行期可用 `GPTZERO_TOKENIZER=heuristic|auto|<dir>` 覆盖。

---

## 4. 离线交付（save / load）

```powershell
# 交付机上：构建并打包
.\scripts\offline_image.ps1
# → dist\gptzero-v3.tar + dist\gptzero-v3.tar.sha256

# 目标机（无外网、无 registry）：
certutil -hashfile gptzero-v3.tar SHA256      # 与 .sha256 比对
docker load -i gptzero-v3.tar
docker images | findstr gptzero

# 立刻验证纯离线可用
docker run --rm --network none gptzero:v3 smoke
```

```bash
# Linux 侧
sha256sum -c gptzero-v3.tar.sha256
docker load -i gptzero-v3.tar
docker run --rm --network none gptzero:v3 smoke
```

镜像**从不 push** 到任何 registry；`dist/` 已在 `.gitignore` 中。

---

## 5. 三种用法

### 5.1 mock（常驻服务，默认子命令）

```bash
docker run --rm -p 8090:8090 gptzero:v3
# 等价
docker run --rm -p 8090:8090 gptzero:v3 mock
```

> 端口说明：本机 `127.0.0.1:8080` 被 `piforged` 常驻占用，因此**镜像、脚本、compose 统一用 8090**。
> 想换端口：`docker run --rm -p 9090:8090 -e GPTZERO_PORT=8090 gptzero:v3`（容器内仍是 8090，
> 只改宿主机映射）；或 `-e GPTZERO_PORT=9090 -p 9090:9090`（容器内也改）。

常用参数覆盖（`GPTZERO_*` 环境变量 ↔ CLI 参数，CLI 优先级更高）：

```bash
docker run --rm -p 8090:8090 \
  -e GPTZERO_TTFT_MS=120 \
  -e GPTZERO_TPOT_MS=6 \
  -e GPTZERO_MAX_CONCURRENCY=128 \
  -e GPTZERO_TOOL_CALL_RATIO=0.8 \
  -v "$PWD/data:/data" \
  gptzero:v3 mock --model gpt-3.5-turbo,qwen3-32b
```

任意 CLI 参数都可以直接跟在 `mock` 后面：`gptzero:v3 mock --prefill-tps 20000 --decode-tps 1200`。
批量参数可用 `-e GPTZERO_ARGS="--jitter 0 --seed 7"`（会做 shell 分词后追加）。
请求日志默认写 `/data/requests.jsonl`；未挂卷时自动退化到 `/tmp/`（不会启动失败）。

### 5.2 bench（一次性压测，profile）

```bash
# 1) 起 mock
docker compose up -d mock

# 2) 跑一次短测，结果落 ./results/
docker compose --profile bench run --rm bench \
    run --url http://mock:8090 --profile coding --concurrency 8 --duration 20 \
    --out results/compose-bench.json
```

`docker compose --profile bench run --rm bench` 不带参数时，用 `docker-compose.yml` 里的默认短测
（8 并发 / 20s / `results/compose-bench.json`）。

要点：

- `bench` 服务 `depends_on: mock: condition: service_healthy`，会等 `/health` 通过再开跑。
- 走 compose 默认桥接网络，容器内用服务名 `mock` 访问，**不额外占宿主机端口**。
- `./results` 与 `./data` 双向挂载，跑完宿主机直接看到 JSON/MD/图。

### 5.3 agentx（trace 生成 / 回放，profile）

```bash
docker compose --profile agentx run --rm agentx \
    replay --url http://mock:8090 --out results/agentx.json

# 只生成 trace
docker compose --profile agentx run --rm agentx generate --out results/trace.json
```

`agentx` 服务把宿主 `./agentx` 以只读方式挂进容器，改代码后无需重建镜像即可联调。
若镜像内 `python -m agentx` 尚未就绪（队友未提交 CLI），入口会打印明确提示并 **退出码 3**，不会静默失败。

### 5.4 smoke / shell / pytest（调试用）

```bash
docker run --rm --network none gptzero:v3 smoke        # 端到端自检（0/1）
docker run --rm -it gptzero:v3 shell                  # 进容器
docker run --rm gptzero:v3 pytest                     # 容器内跑单元测试
```

`smoke` 覆盖：`/health`、`/v1/models`、chat 非流式、chat 流式（含 `include_usage`）、
`/metrics`（`vllm:prefix_cache_*` 等 14 个 vLLM 命名 + `# HELP`）、`/api/stats`（15 标量 +
6 个分位字典 + `models`/`cache`）、前缀缓存命中。

---

## 6. 用 docker compose 的一站式体验

```bash
docker compose build                     # gptzero:v3
docker compose up -d mock                # 宿主机 8090 → 容器 8090
curl http://127.0.0.1:8090/health        # {"status":"ok"}
curl -s http://127.0.0.1:8090/metrics | head

docker compose --profile bench  run --rm bench  run --url http://mock:8090 --profile coding --duration 20 --out results/c.json
docker compose --profile agentx run --rm agentx replay --url http://mock:8090 --out results/a.json
docker compose --profile smoke  run --rm smoke        # 容器内断网自检
docker compose down -v
```

`docker-compose.yml` 的 `mock` 服务显式声明了 healthcheck（`/health`，10s 间隔），
`bench`/`agentx` 依赖该条件；`smoke` 服务 `network_mode: none`，是「离线」最硬的证据。

---

## 7. 资源建议

| 场景 | CPU | 内存 | 说明 |
|---|---|---|---|
| 单机 smoke / 功能联调 | 1~2 核 | 512MB | mock 纯 CPU 事件循环，几乎不吃算力 |
| bench 常规（并发 ≤16） | 2~4 核 | 1~2GB | 服务端 + 压测端同机建议 4 核起 |
| bench sweep（并发 1..32） | 4~8 核 | 2~4GB | 客户端也要留够内存生成大 prompt |
| 带 transformers 的精确计数 | +1 核 | +300MB | 加载 tokenizer 一次性开销 |

- uvicorn 单进程：`--workers > 1` 不受支持（服务端状态在进程内），并发靠 asyncio。
- 延迟是**模拟**的：`--ttft-ms` / `--tpot-ms`（或 `--prefill-tps` / `--decode-tps`）决定曲线，
  与机器性能基本无关 —— 所以低配机器也能复现高配机器的性能图像。
- 磁盘：`/data/requests.jsonl` 增长很快（每个请求一行），长压测建议 `--log-file ""` 或限制 `--log-limit`。

---

## 8. ARM64 / 跨架构

```bash
# 一次性（需要 QEMU，Docker Desktop 自带）
docker buildx create --name gz --use --bootstrap        # 首次
docker buildx build --platform linux/arm64 --load -t gptzero:v3-arm64 .
.\scripts\offline_image.ps1 -Platform linux/arm64       # PowerShell 版封装
bash scripts/offline_image.sh -p linux/arm64            # Bash 版封装
```

注意：

- `python:3.11-slim` 有 arm64 manifest，依赖（numpy/matplotlib）也有 aarch64 manylinux wheel，
  正常**不需要编译器**；若自行换成无 wheel 的版本，需要在 builder 阶段加 `build-essential`。
- 在 x86 上 `--platform linux/arm64 --load` 生成的镜像可通过 `docker save` 直接搬到 arm64 机器
  `docker load`（`--load` 只保留单架构，别加 `--push`）。
- `WITH_TRANSFORMERS=1` 在 QEMU 下构建会明显变慢（pip 解包 + 编译 tokenizers 的备用路径），建议直接
  在目标架构上构建。

---

## 9. 故障排查

| 症状 | 原因 | 处理 |
|---|---|---|
| `docker build` 卡在 `pip install` | 清华源不可达 / 企业代理 | 加 `--build-arg PIP_INDEX_URL=<内网源>`，或先 `docker load` 基础镜像 |
| `docker compose ps` 只显示 `8090/tcp`，宿主 curl 打到别的东西 | 宿主机端口已被占用时 Docker Desktop **静默不绑定**（不报错） | `netstat -ano \| findstr :8090` 找到占用者；改用 `GPTZERO_PORT=8095 docker compose up -d mock` |
| 镜像从 388 MB 涨到 1 GB+ | `.venv-xxx` 之类本地环境被塞进构建上下文 | `.dockerignore` 已用 `.venv*` / `venv*` 通配；确认 `docker run --rm --entrypoint du gptzero:v3 -sh /app` 只有十几 MB |
| `pip install -e .` 报 `setuptools` 缺失 | 少了 builder 里的 setuptools | 别用 `--no-build-isolation` 以外的自定义改法；重建镜像 |
| 容器起不来，日志 `ModuleNotFoundError: bench` | `bench/` 尚未提交 | 正常：入口退出码 3 并提示重建；`mock` 路径不受影响 |
| 宿主机 `curl 127.0.0.1:8090` 无响应 | 端口被占 / 映射错 | `netstat -ano \| findstr :8090`；注意本机 8080 被 piforged 占用，用 8090 |
| `docker compose run bench` 报连不上 mock | mock 未 healthy / 用了 `--no-deps` | 等 healthcheck 通过；命令里 `--url http://mock:8090` 用服务名 |
| `smoke` 报 `/metrics` 缺指标 | 镜像代码旧于服务端契约 | 重新 `docker build`（镜像里代码是构建时快照） |
| 请求日志写入失败 | `/data` 无写权限 / 未挂卷 | 入口自动退化到 `/tmp`；或 `-v "$PWD/data:/data"` 且目录可写 |
| `HEALTHCHECK` 一直 unhealthy | 容器内端口与 `GPTZERO_PORT` 不一致 | `-e GPTZERO_PORT` 与 `--port` 必须一致 |
| Windows 上 `docker run ... | head` 乱码 | 终端编码 | `chcp 65001`，或 PowerShell 里用 `Select-Object -First 20` |

常用排查命令：

```bash
docker run --rm gptzero:v3 python -c "import gptzero,fastapi;print(gptzero.__version__, fastapi.__version__)"
docker run --rm gptzero:v3 mock --help
docker inspect gptzero:v3 --format '{{.Config.Healthcheck}}'
docker history gptzero:v3 --no-trunc | head -20
```

---

## 10. 实测记录

> 实测环境：Windows 11 + Docker Desktop **29.0.1**（engine `29.0.1`，linux/x86_64，28 vCPU / 15.4 GB），
> 基础镜像 `python:3.11-slim`（`sha256:e9e0bd902ae57c30d3ffdb7019b33785cee3c7860dc1eda2dfd740354911a82c`）。
> 实测时间：**2026-10-28 16:33 – 17:20 (Asia/Shanghai)**。
> （测试中途 Docker Desktop 自动升级到 **29.9.2** 并重启引擎，前后两次 `docker run --network none smoke`
> 都是 8/8，说明镜像不依赖引擎版本。）
> 全程只 `docker build` / `docker run` / `docker save`，**没有 push 到任何 registry**。

### 10.1 构建

```bash
$ docker build -t gptzero:v3 .
# 冷构建（首次，需要从清华源下载 numpy/matplotlib/pillow 等 wheel）
#8 [builder 4/4] RUN ... pip install --prefix=/install -r /build/requirements-runtime.txt
#8 167.4 Successfully installed annotated-types-0.8.0 ... fastapi-0.115.6 ... matplotlib-3.9.3 \
#    numpy-2.1.3 ... pydantic-2.10.4 ... uvicorn-0.32.1
#8 DONE 171.5s
#13 [runtime 6/6] RUN pip install --no-deps --no-build-isolation -e /app
#13 2.766 Created wheel for gptzero: filename=gptzero-3.0.0-0.editable-py3-none-any.whl size=7463
#13 2.789 Successfully installed gptzero-3.0.0
#14 naming to docker.io/library/gptzero:v3 done
#14 DONE 6.0s
# 冷构建总耗时 ≈ 3 分 05 秒（其中依赖下载/安装 171.5s）

# 增量重建（只改代码、依赖层命中缓存）
$ docker build -t gptzero:v3 .
#14 DONE 0.7s
# build elapsed: 19s（含 COPY + editable 重装 + 导出层）
```

**镜像大小（精简版，`WITH_TRANSFORMERS=0`，默认）**

```bash
$ docker images gptzero:v3 --format '{{.Repository}}:{{.Tag}} | {{.Size}}'
gptzero:v3 | 388MB              # docker images 显示的是解压后体积
$ docker image inspect gptzero:v3 --format '{{.Size}}'
92350045                        # ≈ 88 MiB，分层压缩后体积（传输用）
$ docker run --rm --entrypoint du gptzero:v3 -sh /app
13M     /app                    # 源码 + dashboard.html + deepseek_v3_tokenizer(7.5M)
```

> ⚠️ 踩坑记录：第一版镜像一度是 **1.22 GB**（Lead 复验时看到的就是这一版），原因是并行开发时仓库里出现了
> `.venv-agentx/`（569 MB），而 `.dockerignore` 当时只写了 `.venv`。已改成通配 `.venv*` / `venv*`，
> 重建后回到 **388 MB**（同一份代码、同一套 8/8 smoke）。现在任何 `.venv-xxx` 都不会再进镜像。

**镜像大小（精确计数版，`WITH_TRANSFORMERS=1`，实测）**

```bash
$ docker build -t gptzero:v3-hf --build-arg WITH_TRANSFORMERS=1 .   # 221s（含 transformers+tokenizers 下载）
$ docker images gptzero --format '{{.Repository}}:{{.Tag}} | {{.Size}}'
gptzero:v3-hf | 549MB      # 388MB 精简版 + 161MB（transformers/tokenizers/huggingface_hub/safetensors）
gptzero:v3    | 388MB

$ docker run --rm --network none --entrypoint python3 gptzero:v3-hf -c \
    "from gptzero.tokenizer import CountingTokenizer; t=CountingTokenizer('auto'); \
     print(t.name); print(t.count('hello world 你好世界'))"
tokenizers:deepseek_v3_tokenizer      # 真实加载仓库内 tokenizer.json（纯离线，HF_HUB_OFFLINE=1 生效）
5
```

两者接口/指标/smoke 完全一致，只差 tokens 计数精确度与 161 MB 体积。

**交付产物（`dist/`，已 gitignore）**

```bash
$ scripts/offline_image.ps1 -NoBuild
    镜像        : gptzero:v3
    镜像 ID     : sha256:e6ee1064ae2f
    镜像大小    : 370.3 MB          # inspect .Size；不同引擎版本口径不同（29.0.1 报 92 MB 压缩层）
    docker images: 388MB            # 解压后体积
    tar 路径    : D:\Vibe_Workspace\GPT-Zero\dist\gptzero-v3.tar
    tar 大小    : 88.1 MB           # docker save 产物，拷贝分发就用它
    sha256      : 3f3a333de7ae6d3bb709480856e3cc6045b6934e32e0120f7d06b7906ab39fec
    校验文件    : dist\gptzero-v3.tar.sha256
```

> ⚠️ 踩坑记录（续）：该 1.22 GB 版本与现在的 388 MB 版本功能一致，8/8 smoke 均通过，
> 差别仅在于构建上下文里多打了一个 569 MB 的 `.venv-agentx/`。

### 10.2 纯离线自检：`docker run --rm --network none gptzero:v3 smoke`

```text
==================================================================
 GPT-Zero v3 smoke test   target=http://127.0.0.1:8090
==================================================================
[smoke] 启动 mock：port=8090 log=/tmp/smoke-server.log
[smoke] 服务已就绪（pid=8）

[1/7] GET /health
  [PASS] /health — status=ok

[2/7] GET /v1/models（OpenAI 列表格式）
  [PASS] /v1/models — 5 个模型，首个 id=gpt-3.5-turbo object=model

[3/7] POST /v1/chat/completions（非流式）
  [PASS] chat/non-stream — finish=length completion_tokens=24

[4/7] POST /v1/chat/completions（流式 SSE + include_usage）
  [PASS] chat/stream — 19 个 chunk，16 个增量帧，usage chunk 已出现

[5/7] GET /metrics（vLLM 命名子集）
  [PASS] /metrics — 14 个必需指标齐全，含 # HELP/# TYPE
  [PASS] metrics/prefix_cache — vllm:prefix_cache_* 数值可解析

[6/7] GET /api/stats（契约 §3.5 稳定字段）
  [PASS] /api/stats — 15 标量 + 6 分位字典 + models/cache 齐全；total_requests=2

[7/7] 前缀缓存生效（两次相同长 prompt → cache_hit_tokens 增长）
  [PASS] prefix-cache — cache_hit_tokens=960 rate=0.4656

==================================================================
 结果: 8/8 项通过   (tokenizer=auto)
 结论: SMOKE PASS — 离线 mock 服务契约自检全部通过
==================================================================

[smoke] PASS (exit 0)
SMOKE_EXIT=0
```

容器内依赖版本（同样在 `--network none` 下取）：

```
3.11.18  fastapi 0.115.6  pydantic 2.10.4  numpy 2.1.3  mpl 3.9.3
```

**结论：镜像在完全无网络（无网卡 / 无 DNS / 无外网路由）下可独立完成 mock 服务 + 契约自检。**

### 10.3 compose 实测

```bash
$ docker compose up -d mock
 Network gptzero_gptzero  Created
 Container gptzero-mock   Started
$ docker compose ps
gptzero-mock | Up 6 seconds (healthy) | 0.0.0.0:8095->8090/tcp, [::]:8095->8090/tcp

$ curl http://127.0.0.1:8095/health
{"status":"ok","version":"3.0.0","uptime_s":6.0}

$ curl -s -X POST http://127.0.0.1:8095/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"compose host check"}],"max_tokens":12}'
{"id":"chatcmpl-1fa4c89d589044d9b767d1f1","object":"chat.completion", ...,
 "choices":[{"index":0,"message":{"role":"assistant","content":"the matc hing pref ix . Late ncy is domi nate d"},
 "finish_reason":"length"}],"usage":{"prompt_tokens":9,"completion_tokens":12,"total_tokens":21}}
```

> 这里用 `GPTZERO_PORT=8095 docker compose up -d mock` 是为了避开本机 8090 上另一个常驻 mock
> （并行联调时被队友占用）。compose 里容器内部始终是 8090，只映射宿主机端口。

**bench（一次性 profile，结果落宿主 `results/`）**

```bash
$ docker compose --profile bench run --rm bench \
    run --url http://mock:8090 --profile coding --concurrency 4 --duration 8 \
    --out results/compose-bench.json
bench run: http://mock:8090 profile=coding concurrency=4 duration=8.0s
server: ttft_ms=180.0 tpot_ms=8.0 max_concurrency=64 cache_blocks=8192
[c=  4] coding   req=   25 ok=100.00% rps=   2.40 ttft_p50=    220.0 p90=    245.9 \
                 tpot_p50=    7.03 out_tok/s=    461.3 cache=  70.7%
written: results/compose-bench.json
$ ls -l results/compose-bench.json     # 宿主机可见（挂卷 ./results:/app/results）
```

**agentx（一次性 profile）**

```bash
$ docker compose --profile agentx run --rm agentx \
    replay --url http://mock:8090 --concurrency 2 --duration 5 \
    --out results/agentx_docker --no-charts
[replay] 8 条 trace × 并发 [2] @ http://mock:8090（gpt-3.5-turbo）
  c=2: req=4 ok=4 fail=0 TTFT p50/p90=252.616/296.864ms out_tok/s=161.9 cache=96.1%
[replay] 完成 -> results/agentx_docker
  results/agentx_docker/replay.json / replay.md / replay.csv
```

> `agentx replay --out` 把参数当**目录**用（会在其下写 `replay.json|md|csv`），所以传 `results/agentx_docker`。

### 10.4 交叉复验

Lead 在本机独立复验同一镜像：`docker run --rm --network none gptzero:v3 smoke` → 8/8 PASS；
compose `gptzero-mock` healthy、agentx 一次性任务跑通；`results/compose-bench.json` 中
coding/c=4 共 25 请求（21 个触发 tool_calls）、TTFT p50 = 220 ms、输入吞吐 16.1k tok/s。
产物归档路径：`results/compose-bench.json`、`results/agentx_docker/replay.{json,md,csv}`。

> Lead 复验时看到的 1.22 GB 是 `.venv-agentx` 未排除的那一版镜像；修正 `.dockerignore` 后精简版
> **388 MB**，功能完全一致（同一套 8/8 smoke、同一份 compose 压测脚本）。

`scripts/verify_offline.ps1` 一次性跑完三件事（实测 exit 0）：

```text
==> 1/3 镜像存在性         [PASS] gptzero:v3（388 MB）
==> 2/3 断网自检           [PASS] smoke 通过（--network none），耗时 2.8s
==> 3/3 离线环境校验       gptzero version: 3.0.0 / tokenizer: auto / PASS
 结论: 离线验证通过 —— 镜像在 --network none 下可独立完成 mock + 全部 smoke 检查
```

### 10.5 known-good 命令清单（可直接复制）

```bash
# ---- 构建 ----
docker build -t gptzero:v3 .                                    # 精简版（388MB，默认，heuristic 分词）
docker build -t gptzero:v3-hf --build-arg WITH_TRANSFORMERS=1 . # 精确计数版（549MB，实测）

# ---- 离线自检（零网络，最硬的验收证据）----
docker run --rm --network none gptzero:v3 smoke                 # → 8/8 PASS, exit 0

# ---- 交付打包 / 落地 ----
scripts/offline_image.ps1                    # Windows: build + save + sha256 → dist/gptzero-v3.tar (88 MB)
bash scripts/offline_image.sh                # Linux/macOS 等价
docker load -i dist/gptzero-v3.tar           # 目标机
scripts/verify_offline.ps1                   # 断网 smoke + 无 curl/wget + HF 离线断言

# ---- 起服务 ----
docker run --rm -p 8090:8090 gptzero:v3                          # 宿主机 8090（若被占，改 -p 8095:8090）
docker compose up -d mock                                        # 同上，带 /data 卷与 healthcheck

# ---- 压测 / 回放（容器内同时起，结果落 ./results）----
docker compose --profile bench  run --rm bench  run --url http://mock:8090 --profile coding --concurrency 8 --duration 20 --out results/compose-bench.json
docker compose --profile agentx run --rm agentx replay --url http://mock:8090 --concurrency 2 --duration 5 --out results/agentx_docker

# ---- 收尾 ----
docker compose down -v
docker rm -f $(docker ps -aq --filter "name=gptzero-") 2>/dev/null || true
```

### 10.6 已知风险 / 未闭环项

| 项 | 现状 | 影响 |
|---|---|---|
| 精确计数版镜像（`WITH_TRANSFORMERS=1`） | ✅ 已构建并实测：**549 MB**，容器内 `CountingTokenizer('auto')` → `tokenizers:deepseek_v3_tokenizer`，`--network none` 下可加载 | 需要精确 `prompt_tokens` 时用它；默认版走 heuristic，契约允许 |
| `docker compose` 端口冲突静默 | 宿主机 8090 被占时，Docker Desktop **不报错**，容器照起但 `NetworkSettings.Ports` 为空（`docker compose ps` 只显示 `8090/tcp`） | 换 `GPTZERO_PORT` 或先确认 `netstat -ano \| findstr :8090` |
| Docker Desktop 稳定性 / C 盘空间 | 测试期间 daemon 两次卡死或崩溃（第二次是 `--build-arg WITH_TRANSFORMERS=1` 构建中 `rpc error ... EOF`）；C: 一度只剩 **3.6 GB**，`%LOCALAPPDATA%\Docker\wsl\disk\docker_data.vhdx` 达 **124 GB**；期间 Docker Desktop 自动升级 29.0.1 → 29.9.2 | 与镜像无关，但会影响构建成功率。建议：`docker builder prune -af` + `docker image prune` 后再构建；预留 ≥5 GB 空闲；跨机交付用 88 MB 的 `dist/gptzero-v3.tar` 而不是现场 build |
| 精确 token 计数在默认镜像 | tokenizer 打印 `heuristic`，且 warning 出现两次（`_candidate_paths` 把 `/app` 与 `/app/deepseek_v3_tokenizer` 各加一次，重复） | 仅影响计数精确度，不影响接口形态；已反馈 Lead |
| Windows PowerShell 编码 | `.ps1` 里含中文，必须保存为 **UTF-8 with BOM**，否则 Windows PowerShell 5.1 按 ANSI 解析会报语法错误 | 脚本已按 BOM 保存；编辑后若丢失 BOM，用 `[System.IO.File]::WriteAllText(..., New-Object System.Text.UTF8Encoding($true))` 补回 |

