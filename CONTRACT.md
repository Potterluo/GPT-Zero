# GPT-Zero v4 — 内部接口契约 (Team 共享)

> 所有成员（Lead / 队友）在同一工作区 `D:\Vibe_Workspace\GPT-Zero` 工作。
> 本文件定义模块边界、HTTP 契约与工具约束。**改动契约必须先通知 Lead。**

## 0. 项目目标（一句话）

把 GPT-Zero 做成**功能完整的多协议 Mock LLM 推理服务**：一套引擎同时支持
OpenAI Chat Completions / OpenAI Responses / Anthropic Messages 三种调用形式
（JSON + SSE），每个协议都有真实的 fake 工具调用；外加前缀缓存模拟、vLLM 风格
Prometheus 指标、统计面板、自带 benchmark harness（出表 + 出图）、标准 Docker 镜像包（纯离线）。

## 1. 目录与写作用域（互不重叠）

| 路径 | Owner | 说明 |
|---|---|---|
| `gptzero/`、`server.py`、`tests/test_server*.py`、`tests/test_protocols.py`、`README.md` | Lead | 服务核心 + 协议适配 |
| `bench/`、`results/`、`docs/BENCHMARK.md` | `bench` 队友 | 压测 harness、结果、图表 |
| `Dockerfile`、`docker-compose.yml`、`docker/`、`scripts/`、`.github/`、`install.bat`、`start.bat`、`docs/DOCKER.md` | `ops` 队友 | 离线镜像与脚本 |

禁止跨域写别人 owner 的文件；需要对方改接口时发消息。
（v4 起不再有 `agentx/` 与 `agentx_offline/`：AgentX 调研/回放配套已整体移除。）

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
  - 扩展字段（bench harness 使用）：`ignore_eos: bool=True` 时必须**精确生成 max_tokens 个 token**（finish_reason=`length`）。
  - `stream=true` → SSE：`data: {chunk}` … `data: [DONE]`；chunk 形状与 OpenAI 一致（`choices[0].delta`）。
    `stream_options.include_usage=true` 时最后追加一个 `choices: []` + `usage` 的 chunk。
  - 非流式：`choices[0].message`，`finish_reason` ∈ `stop|length|tool_calls`，含 `usage{prompt_tokens,completion_tokens,total_tokens}`。
  - `tools` 存在且最后一条消息不是 `role=tool` 时，按确定性策略返回 `tool_calls`（见 §3.4）。
- `POST /v1/responses`（OpenAI Responses API）
  - `input` 接受字符串或条目数组（`message` / `function_call` / `function_call_output`），`instructions` 归入 system；
    `max_output_tokens`、扁平函数工具、`tool_choice`（`auto|none|required|{"type":"function","name":...}`）。
  - 非流式：`response` 对象；纯文本轮 `status="incomplete"` + `incomplete_details={"reason":"max_output_tokens"}`，
    工具轮 `status="completed"`；工具以 `output[].type="function_call"`（`call_id/name/arguments`）返回。
  - `stream=true` → SSE（`event:` + `data:`）：`response.created` → `output_item.added` →
    `output_text.delta` / `function_call_arguments.delta` → `response.completed`（或 `response.incomplete`）。
  - `GET /v1/responses/{id}` 可取回最近创建的响应；`previous_response_id` 支持多轮链式上下文。
- `POST /v1/messages`（Anthropic Messages API）
  - 请求体：`model/max_tokens/messages/system/tools/tool_choice/stop_sequences/temperature/top_p/stream/metadata`；
    `system` 与消息 `content` 接受字符串或 block 数组（`text` / `tool_use` / `tool_result`；`cache_control` 等装饰字段忽略）。
  - 非流式：`message` 对象；`stop_reason` ∈ `max_tokens|tool_use|stop_sequence`；工具以 `content[].type="tool_use"`
    （id 为 `toolu_*`、`name`、`input` 为 JSON 对象）返回；`usage` 含 `cache_read_input_tokens`。
  - `stream=true` → SSE（`event:` + `data:`）：`message_start` → `ping` → `content_block_start/delta/stop`
    （文本 `text_delta`、工具参数 `input_json_delta`）→ `message_delta` → `message_stop`。
- `POST /v1/completions`：文本补全最小实现（同 usage 语义）。
- `GET /v1/models`：OpenAI 列表格式。
- `GET /health`：`{"status":"ok"}`。

### 3.2 协议错误体
- 未知路径 404；非法 JSON 400。
- OpenAI 端点：`{"error":{"message":...,"type":"invalid_request_error"}}`。
- Anthropic `/v1/messages`：`{"type":"error","error":{"type":"invalid_request_error","message":...}}`。

### 3.3 运维/观测端点（保留 v2 兼容）
- `GET /` 管理面板 HTML（`gptzero/dashboard.html`，纯静态，无外部 CDN）。
- `GET /api/stats`：见 §3.5。
- `GET /api/logs`、`GET /api/log/{id}`（id 为全局序号，1 起；倒序返回）。
- `GET /api/config`、`GET /api/models`、`GET /api/models/search?query=`、`POST /api/models/scan`、`POST /api/config/model-dir`、`POST /api/models/match`、`POST /api/tokens`：与 v2 兼容（可增强字段，不删字段）。
- `POST /api/reset`：清空统计/日志/缓存，返回 `{"ok":true}`。
- `GET /metrics`：Prometheus text，必须包含 vLLM 命名（见 §3.6）。

### 3.4 工具调用合成策略（确定性，三协议共用）
输入 `messages` 最后一条：
- `role=tool` → 生成**普通文本**收尾（`finish_reason=stop`），不再调用工具。
- `role=user`（或 system+user）且 `tools` 非空：
  - 依据 `sha256(model + 最后一条用户文本 + seed)` 决定是否调用工具（默认概率 `--tool-call-ratio 0.6`；
    `tool_choice` 为 `required`/指定函数/Anthropic `any|tool` 时强制调用，`none` 时强制不调用）。
  - 选择 `tools[i].function.name`（轮转 + hash），生成 JSON 参数（键取自 `parameters.properties`，值按类型填充；字符串值含路径/命令/内容，内容长度由 `--tool-payload-tokens` 控制）。
  - `finish_reason="tool_calls"`，`message.tool_calls=[{id:"call_<hash8>",type:"function",function:{name,arguments}}]`。
  - 若 `parallel_tool_calls=true` 且 hash 命中，返回 2 个 tool_calls。
- 流式：先 delta.role，再按 8~24 字符切分参数片段（各协议的事件形状见 §3.1），最后终止帧。
- 协议出形：Chat/Responses 沿用引擎 `call_*` id；Messages 重映射为 `toolu_*` id、参数以 JSON 对象 `input` 呈现。
- 统计：`tool_calls_total{name}` 计数（三协议同口径）。

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

## 5. 协议适配（`gptzero/protocols.py`，Lead）
- 请求归一化：`normalize_chat` / `normalize_responses` / `normalize_messages` → `NormalizedRequest`
  （内部表示 = OpenAI chat 消息 + OpenAI 嵌套函数工具）。纯函数，不感知 asyncio/HTTP。
- 响应蓝图：各协议 `stream=true` 返回 `{"event","data","pace_s"}` 帧序列，HTTP 层负责按 `pace_s` 节流；
  `stream=false` 直接组协议 JSON 对象。
- `ResponseStore`：Responses 协议的 `previous_response_id` 上下文存储与 `GET /v1/responses/{id}`。

## 6. ops（`ops` 队友）
- `Dockerfile`：`python:3.11-slim` 为基底（本地已有），构建期用清华镜像装依赖；**运行期零网络**。
- `docker-compose.yml`：`mock` 服务 + 可选 `bench` 一次性任务，挂载 `results/`。
- `scripts/offline_image.ps1|.sh`：`docker build` + `docker save` + `docker load` 说明与校验。
- 入口 `docker/entrypoint.sh`：`mock|bench|smoke|shell|pytest` 子命令。

## 7. 交付验收（Lead 负责最终验收）
1. `.venv\Scripts\python.exe -m pytest tests -q` 全绿（含三协议契约 `tests/test_protocols.py`）。
2. 起 mock → `bench sweep` 出 `results/sweep.{md,csv,json}` + `results/charts/*.png`。
3. 三个协议端点（chat-completions / responses / messages）非流式 + 流式 + 工具调用 + 工具结果续轮全部可正确返回。
4. `docker build` 成功，容器内 `bench` 跑通（无外网）。
5. `README.md` + `docs/*.md` 给出可复制的多协议调用示例。

## 8. 约定
- 中文注释/文档优先；代码标识符英文。
- 提交前 `python -m compileall` 与 pytest。
- 不做与本任务无关的大重构；保留 v2 行为兼容（旧端点/旧字段）。
