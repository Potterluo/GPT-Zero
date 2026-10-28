# GPT-Zero v3 — 内部接口契约 (Team 共享)

> 所有成员（Lead / 队友）在同一工作区 `D:\Vibe_Workspace\GPT-Zero` 工作。
> 本文件定义模块边界、HTTP 契约与工具约束。**改动契约必须先通知 Lead。**

## 0. 项目目标（一句话）

把 GPT-Zero 从「单回复 Mock LLM」升级为**可离线复现 AgentX 类 agentic 负载的 Mock 推理服务**：
真实工具调用（function calling）、前缀缓存模拟、vLLM 风格 Prometheus 指标、统计面板、
自带 benchmark harness（出表 + 出图）、标准 Docker 镜像包（纯离线）。

## 1. 目录与写作用域（互不重叠）

| 路径 | Owner | 说明 |
|---|---|---|
| `gptzero/`、`server.py`、`tests/test_server*.py`、`README.md` | Lead | 服务核心 |
| `bench/`、`results/`、`docs/BENCHMARK.md` | `bench` 队友 | 压测 harness、结果、图表 |
| `agentx/`、`docs/AGENTX.md`、`docs/AGENTX_REPRO.md`、`tests/test_agentx*.py` | `agentx` 队友 | AgentX 调研、WEKA trace 生成/回放 |
| `Dockerfile`、`docker-compose.yml`、`docker/`、`scripts/`、`.github/`、`install.bat`、`start.bat`、`docs/DOCKER.md` | `ops` 队友 | 离线镜像与脚本 |

禁止跨域写别人 owner 的文件；需要对方改接口时发消息。

## 2. Python 运行环境

- 项目 venv：`.venv`（Python 3.11，已装 fastapi/uvicorn/pydantic/httpx/matplotlib/numpy/pytest）。
- 解释器：`D:\Vibe_Workspace\GPT-Zero\.venv\Scripts\python.exe`
- 依赖清单以 `pyproject.toml` 为准；不要引入需要联网运行时的依赖。
- 只可依赖：fastapi, uvicorn, pydantic, httpx, numpy, matplotlib（bench 图表）, pytest。
  `transformers` 可选（本地 tokenizer 存在时启用，缺失自动降级），不作为硬依赖。

## 3. 服务端 HTTP 契约（`gptzero`，默认端口 8080；本机实测统一用 **8090**）

> 本机 127.0.0.1:8080 被 piforged 常驻占用，团队实测/脚本/容器映射统一走 8090。

### 3.1 OpenAI 兼容
- `POST /v1/chat/completions`
  - 请求体：标准字段 `model/messages/stream/max_tokens/min_tokens/stop/seed/tools/tool_choice/parallel_tool_calls/stream_options/include_usage/temperature/top_p/user`。
  - 扩展字段（AIPerf 使用）：`ignore_eos: bool=True` 时必须**精确生成 max_tokens 个 token**（finish_reason=`length`）。
  - `stream=true` → SSE：`data: {chunk}` … `data: [DONE]`；chunk 形状与 OpenAI 一致（`choices[0].delta`）。
    `stream_options.include_usage=true` 时最后追加一个 `choices: []` + `usage` 的 chunk。
  - 非流式：`choices[0].message`，`finish_reason` ∈ `stop|length|tool_calls`，含 `usage{prompt_tokens,completion_tokens,total_tokens}`。
  - `tools` 存在且最后一条消息不是 `role=tool` 时，按确定性策略返回 `tool_calls`（见 §3.4）。
- `POST /v1/completions`：文本补全最小实现（同 usage 语义）。
- `GET /v1/models`：OpenAI 列表格式。
- `GET /health`：`{"status":"ok"}`。

### 3.2 OpenAI 兼容错误
- 未知路径 404；非法 JSON 400，body 形如 `{"error":{"message":...,"type":"invalid_request_error"}}`。

### 3.3 运维/观测端点（保留 v2 兼容）
- `GET /` 管理面板 HTML（`gptzero/dashboard.html`，纯静态，无外部 CDN）。
- `GET /api/stats`：见 §3.5。
- `GET /api/logs`、`GET /api/log/{id}`（id 为全局序号，1 起；倒序返回）。
- `GET /api/config`、`GET /api/models`、`GET /api/models/search?query=`、`POST /api/models/scan`、`POST /api/config/model-dir`、`POST /api/models/match`、`POST /api/tokens`：与 v2 兼容（可增强字段，不删字段）。
- `POST /api/reset`：清空统计/日志/缓存，返回 `{"ok":true}`。
- `GET /metrics`：Prometheus text，必须包含 vLLM 命名（见 §3.6）。

### 3.4 工具调用合成策略（确定性）
输入 `messages` 最后一条：
- `role=tool` → 生成**普通文本**收尾（`finish_reason=stop`），不再调用工具。
- `role=user`（或 system+user）且 `tools` 非空：
  - 依据 `sha256(model + 最后一条用户文本 + seed)` 决定是否调用工具（默认概率 `--tool-call-ratio 0.6`，可用请求头/参数覆盖）。
  - 选择 `tools[i].function.name`（轮转 + hash），生成 JSON 参数（键取自 `parameters.properties`，值按类型填充；字符串值含路径/命令/内容，内容长度由 `--tool-payload-tokens` 控制）。
  - `finish_reason="tool_calls"`，`message.tool_calls=[{id:"call_<hash8>",type:"function",function:{name,arguments}}]`。
  - 若 `parallel_tool_calls=true` 且 hash 命中，返回 2 个 tool_calls。
- 流式：先 delta.role，再按 8~16 字符切分 `delta.tool_calls[0].function.arguments` 片段（`index/id/type/name` 首帧给出），最后 finish_reason。
- 统计：`tool_calls_total{name}` 计数。

### 3.5 `/api/stats` JSON（客户端 bench 依赖，必须稳定）
```json
{
  "uptime_s": 12.3,
  "total_requests": 100,
  "active_requests": 2,
  "completed_requests": 98,
  "failed_requests": 0,
  "prompt_tokens": 12345,
  "completion_tokens": 6789,
  "total_tokens": 19134,
  "tool_calls": 12,
  "rps": 8.1,
  "prompt_tokens_per_s": 1000.0,
  "completion_tokens_per_s": 550.0,
  "cache_hit_tokens": 10000,
  "cache_query_tokens": 12345,
  "cache_hit_rate": 0.81,
  "ttft_ms":   {"p50":1,"p90":2,"p99":3,"mean":1.5,"max":9},
  "tpot_ms":   {"p50":1,"p90":2,"p99":3,"mean":1.5,"max":9},
  "itl_ms":    {"p50":1,"p90":2,"p99":3,"mean":1.5,"max":9},
  "e2e_ms":    {"p50":1,"p90":2,"p99":3,"mean":1.5,"max":9},
  "prompt_tokens_hist": {"p50":1,"p90":2,"p99":3,"mean":1.5,"max":9},
  "completion_tokens_hist": {"p50":1,"p90":2,"p99":3,"mean":1.5,"max":9},
  "models": {"<model>": {"requests": 1, "prompt_tokens": 1, "completion_tokens": 1}},
  "cache": {"blocks_cached": 1, "blocks_capacity": 8192, "evictions": 0}
}
```
字段只增不改；`*_ms` 的键固定 `p50/p90/p99/mean/max`。

### 3.6 `/metrics` 命名（vLLM 兼容子集）
```
vllm:num_requests_running
vllm:num_requests_waiting
vllm:kv_cache_usage_perc
vllm:prompt_tokens_total
vllm:generation_tokens_total
vllm:request_success_total{finished_reason="stop|length|tool_calls"}
vllm:prefix_cache_queries_total
vllm:prefix_cache_hits_total
vllm:prefix_cache_evictions_total
vllm:time_to_first_token_seconds_sum / _count / _bucket{le=...}
vllm:time_per_output_token_seconds_sum / _count
vllm:e2e_request_latency_seconds_sum / _count
gptzero:tool_calls_total{name="..."}
gptzero:cache_hit_tokens_total
```
必须带 `# HELP` / `# TYPE`。

### 3.7 CLI（`python server.py` / `python -m gptzero`）
```
--host 0.0.0.0 --port 8080
--model gpt-3.5-turbo  (可多次 / 逗号分隔)
--models-file path.json
--ttft-ms 200 --tpot-ms 8 --jitter 0.15
--prefill-tps 20000 --decode-tps 1200   # 可选：按 token 数推算延迟
--max-concurrency 64                    # 超出排队，体现 TTFT 压力
--cache-blocks 8192 --block-size 64
--tokenizer auto|heuristic|<path>       # auto: 本地 deepseek_v3_tokenizer
--log-file logs/requests.jsonl --log-limit 1000
--tool-call-ratio 0.6 --tool-payload-tokens 120
--seed 42
--model-dir <path> (v2 兼容)
```
所有参数可用环境变量覆盖：`GPTZERO_<UPPER_SNAKE>`（如 `GPTZERO_PORT`）。

## 4. bench（`bench` 队友）
- 入口：`python -m bench run|sweep|report`（也提供 `bench/__main__.py`）。
- `run --url http://127.0.0.1:8080 --profile coding --concurrency 8 --duration 20 --out results/x.json`
- 必须采集：成功/失败、TTFT、TPOT、ITL、E2E、prompt/completion tokens、服务端 `/metrics` 前后差值（prefix cache hits）、RPS、tok/s。
- `sweep`：并发 `1,2,4,8,16` 逐步跑；输出 `results/<name>.json|md|csv` 与 `results/charts/*.png|svg`。
- 图表（matplotlib，Agg）：TTFT vs 并发、吞吐 vs 并发、缓存命中率、延迟 CDF、Pareto(吞吐 vs p90 TTFT)。
- JSON schema 自包含，`report` 能由 JSON 重建 md/csv/图。
- 依赖服务端 §3；不得 import `gptzero` 内部（只走 HTTP），除非仅复用 tokenizer 计数（允许 `gptzero.tokenizer.count_tokens`）。

## 5. agentx（`agentx` 队友）
- `agentx/trace_gen.py`：生成与 AIPerf WEKA trace schema 兼容的合成语料（`hash_ids`、`t`、`out`、`type`、`subagent`）。
- `agentx/replay.py`：离线 DAG 回放（主链 + 子代理 SPAWN/JOIN、10 条 warmup/lane、cache-bust marker、闭环并发）。
- `agentx/cli.py`：`generate|replay|report`。
- 研究文档：AgentX 参数（版本 1.0.7 scenario lock）、数据集、报告指标、可复现映射表。
- 允许尝试真实 `agentx-harness==1.0.7`（wheel 已下载在 `%TEMP%\agentxw\`）离线跑 `--input-file`，失败则记录原因并走自研回放。

## 6. ops（`ops` 队友）
- `Dockerfile`：`python:3.11-slim` 为基底（本地已有），构建期用清华镜像装依赖；**运行期零网络**。
- `docker-compose.yml`：`mock` 服务 + 可选 `bench` 一次性任务，挂载 `results/`。
- `scripts/offline_image.ps1|.sh`：`docker build` + `docker save` + `docker load` 说明与校验。
- 入口 `gptzero/docker-entrypoint.sh`（或 `docker/`）：`mock|bench|agentx|shell` 子命令。

## 7. 交付验收（Lead 负责最终验收）
1. `.venv\Scripts\python.exe -m pytest tests -q` 全绿。
2. 起 mock → `bench sweep` 出 `results/sweep.{md,csv,json}` + `results/charts/*.png`。
3. `agentx replay` 对 mock 跑通，出 `results/agentx.*`。
4. `docker build` 成功，容器内 `bench` 跑通（无外网）。
5. `README.md` + `docs/*.md` 给出可复制命令与「AgentX 参数 → mock 复现」映射表。

## 8. 约定
- 中文注释/文档优先；代码标识符英文。
- 提交前 `python -m compileall` 与 pytest。
- 不做与本任务无关的大重构；保留 v2 行为兼容（旧端点/旧字段）。
