# GPT-Zero · offline mock LLM for agentic benchmarks

[![CI](https://img.shields.io/badge/ci-ready-brightgreen)](#)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

一个**纯离线**、OpenAI 兼容的 Mock 推理服务，用来复现「编码 Agent（Claude Code / Aider / Codex 类）打推理后端」时的负载形状：
真实工具调用、增长的超长上下文、64-token 块的 prefix cache、vLLM 风格 Prometheus 指标、开箱即用的压测脚本与图表。

> 它**不是**模型，不产生有意义的回答。它复现的是**可观测行为**：token 计数、TTFT/TPOT、排队、缓存命中、工具调用帧、成功率——让推理栈、网关、调度器、监控面板在没有 GPU 的环境里也能被压测和回归。

```
                 ┌──────────────────────────┐
   agent loop ──▶│  /v1/chat/completions    │──▶ SSE / JSON（工具调用、精确 token 数）
   (bench/agentx)│  tools · streaming       │
                 ├──────────────────────────┤
                 │  engine: prefix cache    │──▶ 64-token 块哈希 + LRU 序列淘汰
                 │          timing model    │──▶ TTFT = 未命中 prefill + 排队
                 ├──────────────────────────┤
                 │  /metrics  /api/stats    │──▶ vLLM 命名计数器 + 分位数统计
                 └──────────────────────────┘
```

## v3 做了什么（相对 v2 单回复 mock）

| 能力 | v2 | v3 |
|---|---|---|
| 回复内容 | 固定一篇长文 | 确定性 filler，**精确**按 `max_tokens` 产出；本地 tokenizer 计数 |
| 工具调用 | 无 | OpenAI `tool_calls`（流式 delta、并行调用、合法 JSON 参数、`role=tool` 收尾） |
| 上下文 | 无状态 | 多轮增长上下文；64-token 块 prefix cache（LRU 序列淘汰） |
| 延迟 | 固定字符间 sleep | TTFT/TPOT 模型 + 未命中 prefill + 并发排队（可配 `--max-concurrency`） |
| 指标 | 只统计日志里的 token | vLLM 命名 Prometheus 指标 + `/api/stats` 分位数（TTFT/TPOT/ITL/E2E/RPS/缓存命中） |
| 压测 | 无 | `python -m bench`：闭环并发、profile、sweep、出 MD/CSV/PNG/SVG |
| AgentX | 不支持 | `docs/AGENTX.md` 参数调研 + WEKA trace 生成/离线 DAG 回放 |
| 交付 | `python server.py` | 同左 + 纯离线 Docker 镜像（`dist/gptzero-v3.tar`） |
| 管理面板 | 静态日志页 | 实时 KPI/延迟/缓存/工具调用面板（无 CDN、无外网） |

## 快速开始

```bash
# 1) 安装（离线可用：依赖只在本地 venv / 镜像里）
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"      # Windows
# source .venv/bin/activate && pip install -e ".[dev]"   # Linux/macOS

# 2) 启动 mock（8090 只是示例端口；本机 8080 可能被其它服务占用）
.venv\Scripts\python server.py --port 8090 --tokenizer deepseek_v3_tokenizer

# 3) 冒烟
curl http://127.0.0.1:8090/health
curl -X POST http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"hi"}],"max_tokens":32}'
```

打开 <http://127.0.0.1:8090/> 看实时面板；`/metrics` 给 Prometheus 抓；`/api/stats` 给脚本读。

![管理面板：吞吐 / TTFT 分位 / 前缀缓存命中 / 工具调用 / 请求明细](results/charts/dashboard.png)

### 关键参数

| 参数 | 默认 | 作用 |
|---|---|---|
| `--ttft-ms` / `--tpot-ms` | 180 / 8 | 首 token 基础延迟、每 token 解码时间 |
| `--jitter` | 0.12 | 延迟相对抖动（0 = 完全确定） |
| `--prefill-tps` | 0 | >0 时 TTFT 由**未命中** token 数 / 该速率决定（缓存收益可见） |
| `--max-concurrency` | 64 | 同时在处理的请求数，超出排队（排队时间计入 TTFT） |
| `--cache-blocks` / `--block-size` | 8192 / 64 | prefix cache 容量与块大小 |
| `--tool-call-ratio` | 0.6 | 带 tools 的轮次触发工具调用的概率 |
| `--tokenizer` | auto | `auto` 找仓库内 `deepseek_v3_tokenizer/`，否则 heuristic |
| `--log-file` | 空 | 每请求 JSONL 落盘 |

所有参数支持 `GPTZERO_*` 环境变量（如 `GPTZERO_PORT=8090`），CLI 优先。

## 压测：一条命令出表 + 出图

```bash
# 闭环并发扫点（chat / coding / agentic / longctx 四种 profile）
.venv\Scripts\python -m bench sweep ^
  --url http://127.0.0.1:8090 ^
  --profiles chat,coding,agentic ^
  --concurrency 1,2,4,8,16 ^
  --duration 15 --name sweep_mock_a

# 只重画/重生成报告（从 JSON）
.venv\Scripts\python -m bench report results/sweep_mock_a.json
```

产物：

```
results/sweep_mock_a.json     自包含原始摘要（config + 每个并发点）
results/sweep_mock_a.md       中文表格：TTFT p50/p90/p99、TPOT、E2E、RPS、tok/s、cache hit、成功率
results/sweep_mock_a.csv
results/charts/*.png / *.svg  TTFT、吞吐、缓存命中、CDF、Pareto、profile 对比
```

指标口径（与 ClawPerf / AIPerf 对齐）：TTFT 从发出请求到**第一个含 content/tool_calls 的 delta**；TPOT = `(e2e-ttft)/(completion_tokens-1)`；缓存命中率取服务端计数器**差分**（token 级，不是请求级）。

已跑出的真实结果（表格 + 图表 + 读法与复现命令）：**[results/README.md](results/README.md)**。

- `sweep_mock_a`（ttft=120ms / tpot=8ms / 64 槽位）：并发 1→16，RPS 0.63→10.12，TPOT 稳定 7.4–8.5ms，缓存命中 70.8–87.3%；
- `sweep_mock_b`（ttft=400ms / tpot=18ms / 8 槽位）：并发 16 超过槽位，TTFT p50 从 ~450ms 跳到 **3935ms**、RPS 停在 2.0 —— 复现“容量拐点”；
- `sweep_profiles`：同并发下 chat / coding / agentic 的差异，coding 缓存命中 97.6%、agentic 70.7%。

## AgentX 复现

`docs/AGENTX.md` 是完整的参数调研（`agentx-harness==1.0.7` 的 scenario 锁、数据集形状、warmup/cache-bust 规则、哪些能/不能离线复现），`docs/AGENTX_REPRO.md` 记录实测。

```bash
# 生成 AgentX 形状的 WEKA trace（主链 + 子代理 SPAWN/JOIN、64-token 块哈希）
.venv\Scripts\python -m agentx generate --sessions 8 --requests 24 --subagents 0.4 --out agentx/traces

# 离线 DAG 回放（闭环并发、warmup、cache-bust）
.venv\Scripts\python -m agentx replay --url http://127.0.0.1:8092 --traces agentx/traces --concurrency 4
```

**我们真的跑通了官方 harness**（离线 + 真实语料）：`agentx-harness==1.0.7` 加载
`semianalysisai/cc-traces-weka-062126`（8 条），对 mock 做了 c=4 / 180s 回放，产出官方
`profile_export_aiperf.json` / `profile_export.jsonl` / `server_metrics_export.json` / 1s timeslices：

| 并发 | 请求 | TTFT p50/p90/p99 | ITL p50 | out tok/s | 缓存命中（理论/服务端） |
|---|---|---|---|---|---|
| 4 | 58 | 160 / 239 / 370 ms | 8.7 ms | 103 | 95.5% / 93.4% |

完整命令、5 个坑与修复、官方产物说明见 [docs/AGENTX_REAL.md](docs/AGENTX_REAL.md)；
结果与图在 [results/agentx_real/](results/agentx_real/)。官方形式（本地 pinned 语料）等价命令：

```bash
agentx profile --url http://127.0.0.1:8092 --endpoint-type chat --streaming \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --tokenizer deepseek-ai/DeepSeek-V3 --model gpt-3.5-turbo \
  --concurrency 4 --benchmark-duration 180 --num-dataset-entries 8 \
  --unsafe-override --output-artifact-dir experiments/agentx_real/run_c4_180
```

## Docker（纯离线）

```bash
docker build -t gptzero:v3 .
docker run --rm --network none gptzero:v3 smoke          # 容器内自检
docker run --rm -p 8090:8090 gptzero:v3 mock             # 起 mock
docker run --rm --network none -v %cd%/results:/app/results gptzero:v3 bench sweep --concurrency 1,2,4
```

离线交付：`scripts/offline_image.ps1` 负责 build + `docker save` 到 `dist/gptzero-v3.tar`，目标机器 `docker load` 即用。细节见 `docs/DOCKER.md`。

## 仓库结构

```
gptzero/           mock 服务（app / engine / tokenizer / metrics / dashboard）
bench/             闭环压测 harness（client / workload / runner / aggregate / report / charts）
agentx/            AgentX 调研配套：WEKA trace 生成 + 离线 DAG 回放
docs/              AGENTX.md · AGENTX_REPRO.md · BENCHMARK.md · DOCKER.md · ARCHITECTURE.md
tests/             server 契约测试 + bench/agentx 单测
scripts/           离线镜像、容器 smoke、CI 辅助
Dockerfile         python:3.11-slim 多阶段、非 root、运行期零网络
```

## 兼容性

v2 的全部端点保留：`/`（面板）、`/v1/chat/completions`、`/v1/models`、`/health`、`/api/logs`、`/api/log/{id}`、`/api/config`、`/api/stats`、`/api/models*`、`/api/tokens`。旧的 `assets/dashboard.html` 仍可通过 `/legacy` 打开。

## 测试

```bash
.venv\Scripts\python -m pytest -q
```

## 已知限制（请如实引用）

- mock 的吞吐上限是**单机 Python event loop**，不是 GPU；高并发下 CPU 先饱和，不代表真实服务端能力。
- 长上下文有 CPU 成本：176k token 的 prompt 首次编码约 **290ms**，之后走编码缓存约 **20ms**/请求（块哈希 + 计数）。这是 mock 的开销，报告解读时要扣除。
- Windows 上默认定时器精度 ~15.6ms 会让 8ms 的 TPOT 被放大到 ~15.7ms 或偶发提前返回；`server.py` 启动时会调用 `timeBeginPeriod(1)`（日志行 `timer : 1ms high-resolution timer enabled`），**看到这行才代表 TPOT 可信**。Linux 无此问题。
- filler 文本无语义；`finish_reason=length` 是常态（没有 EOS），不要据此判断截断缺陷。
- prefix cache 是**块级模拟**，不含真实 KV 显存/调度器行为；`kv_cache_usage_perc` 是块占用比例。
- AgentX 官方分数依赖真实模型 + 厂商 chat template + speculative acceptance；本仓库只复现**负载形状与时序**，不产出可提交的 submission。

## License

Apache-2.0（见 [LICENSE](LICENSE)）。AgentX / AIPerf 的商标与数据集版权归各自所有者。
