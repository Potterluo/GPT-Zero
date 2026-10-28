# AgentX 复现记录（REPRO）

> 本文件是 `docs/AGENTX.md` 的**执行证据**：记录「跑了什么、得到什么、卡在哪、怎么绕过」。
> 所有命令可在 `D:\Vibe_Workspace\GPT-Zero` 直接复制执行（Windows / PowerShell）。

---

## 0. 一句话结论

**AgentX 的负载形状与参数可以离线复现；官方分数不可离线复现。**

- 可复现：scenario 参数锁、WEKA trace 格式、`hash_ids` 前缀语义、SPAWN/JOIN、cache-bust marker、
  trajectory 切片与 warmup primer、`ignore_eos` 精确长度、prefix cache 命中率、TTFT/TPOT 分位。
- 不可复现：真实模型质量、HF pinned 语料（`semianalysisai/cc-traces-weka-062126`）、
  厂商 chat template/tokenizer 行为、speculative acceptance、GPU 遥测、真实芯片数与 KV 容量。

---

## 1. 运行环境

| 项 | 值 |
|---|---|
| 工作区 | `D:\Vibe_Workspace\GPT-Zero` |
| 项目 venv | `.venv`（Python 3.11.9，pytest 8 / httpx / matplotlib / numpy / pydantic 2.14） |
| 真 harness venv | `.venv-agentx`（`agentx-harness==1.0.7`，见 §4） |
| mock 服务端口 | **8090**（团队统一；8080 被 piforged 常驻占用） |
| 本仓库自测端口 | 8092（避免与 `bench` 队友的 sweep 抢 8090） |
| 回放端 | `agentx/replay.py`（纯 httpx，不依赖 aiperf） |

---

## 2. 语料生成与校验

### 2.1 生成

```powershell
.venv\Scripts\python.exe -m agentx generate --sessions 8 --requests 30 --subagents 0.4 `
  --max-isl 32000 --out agentx\traces
```

实测输出：

```
[generate] 8 条 trace -> agentx\traces
  请求 784（含子代理 96 个）；输入 14,711,445 tok，输出 292,164 tok；平均 in=18,764，out=372
  主链缓存前缀 ≈ 6,277,696 tok / 主链输入 6,412,046 tok      # 97.9%，官方口径 ~98%
  [OK] schema 校验通过（与 agentx-harness 1.0.7 的 WekaTrace 字段对齐）
```

- 8 条 session × 30 主链轮 + 96 个子代理块（`--subagents 0.4` 扇出概率）
- 每条 session 的 `block_size=64`、`hash_id_scope=local`、`system_tokens`/`tool_tokens` 符合抓取量级
- `t` 单位为**秒**（与 wheel schema 一致；loader 内部 ×1000 转毫秒）

### 2.2 校验

```powershell
.venv\Scripts\python.exe -m agentx validate --traces agentx\traces
```

两层校验都过：

1. **pydantic 结构**：`tests/_weka_schema_mirror.py`（从 wheel 的
   `aiperf/dataset/loader/weka_trace_models.py` 逐字复制，仅把 `AIPerfBaseModel`
   换成本地等价基类）`model_validate()` 通过 —— 即真 harness 的 loader 会接受。
   `extra="forbid"` 多一个字段就报错，别名 `in`/`out`、`ttft` 仅流式等都对。
2. **语义/时间/缓存**：
   - `len(hash_ids) * block_size <= in` —— **全部通过**（含首轮与每个子代理的首条内层请求）
   - 同一 stream 内 `hash_ids` 严格前缀递增 —— **grow 469 次全部前缀一致**
   - **shrink 211 次**均为上下文 compaction（前缀仍一致，只是块数减少），不是断裂
   - 外层 `t` 单调、SPAWN/JOIN 时序自洽（`t_join >= t_spawn + duration_ms/1000`）
   - `duration_ms` 用 `ceil` 而非 `round`：否则 JOIN 结束时刻可能早于最后一条内层请求，
     父链会提前唤醒

> 校验统计（grow 469 / shrink 211 / 块上界全通过）由 Lead 独立复核，与本仓库
> `validate_trace()` 的判定一致。

### 2.3 首轮 `hash_ids` 必须非空（重要保真点）

`hash_ids=[]` 会让 AIPerf 走 `prompt_generator.generate(mean=input_length)` 的**随机 prompt**
路径，与后续轮由 `_build_token_sequence(input_length, hash_ids, block_size)` 重建的 prompt
**不共享任何前缀字节**。后果：真实 harness 里第 1 条永远冷、且与后面接不上，prefix cache
曲线失真。

因此生成器对**每一轮（含首轮）**都给出 `len(hash_ids)*block_size <= in` 的块 ID；子代理内部同理。
「首轮冷」由 AgentX 的 cache-bust marker 负责制造，而不是靠空 `hash_ids`。

---

## 3. 本地 mock 回放实测

### 3.1 起服务

```powershell
# 团队统一端口 8090（自测可用 8092：--port 8092）
.venv\Scripts\python.exe server.py --port 8090 --tokenizer deepseek_v3_tokenizer `
  --ttft-ms 120 --tpot-ms 8 --max-concurrency 64
```

### 3.2 回放

```powershell
.venv\Scripts\python.exe -m agentx replay --traces agentx\traces `
  --url http://127.0.0.1:8092 --concurrency 1,2,4 --duration 30 `
  --warmup 10 --out results\agentx
```

### 3.3 结果（`results/agentx/replay.{json,md,csv}`）

> **本轮结果基于修复后的服务端**（`ignore_eos` 精确长度 + `tokenizers` 版 `deepseek_v3_tokenizer`）：
> 带 `tools` 的轮次现在「先 content 帧、再 tool_calls 帧」，`usage.completion_tokens` 恒等于
> `max_tokens`，因此 TPOT 可信。

| 并发 | 请求 | 成功 | TTFT p50 | TTFT p90 | TTFT p99 | TPOT p50 | E2E p90 | req/s | out tok/s | 缓存命中 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 100% | 265 ms | 302 ms | 311 ms | 8.27 ms | 5057 ms | 0.085 | 43.5 | **99.6%** |
| 2 | 7 | 100% | 248 ms | 278 ms | 288 ms | 8.27 ms | 6545 ms | 0.193 | 93.3 | **99.3%** |
| 4 | 13 | 100% | 293 ms | 336 ms | 345 ms | 8.17 ms | 6999 ms | 0.356 | 153.8 | **99.6%** |

（`results/agentx/replay.csv` 原始行，`valid=true`；`tok/s/chip` 留空 —— 未给 `--chips`。）

观察：

- **成功率 100%**，无失败请求（`failed_request_threshold=0.10` 未触发，`valid=true`）
- **TPOT ≈ 8.2 ms**，与 `--tpot-ms 8` 一致 —— 说明服务端 `ignore_eos` 精确长度语义正确
  （修复前带 `tools` 的轮次 `completion_tokens` 远小于 `max_tokens`，TPOT 会被算成 0.06ms 的失真值）
- **prefix cache 命中 99.3%~99.6%**，官方口径 ~98%，同一量级
- 吞吐随并发近似线性增长（43.5 → 93.3 → 153.8 out tok/s），符合 mock 的固定 TTFT + 固定 TPOT 延迟模型
- **TTFT 在 c=2 处的下探是采样噪声**：每点只有 3/7/13 个请求（30s 窗口 ÷ trace 自身 3~10s 的 idle gap），
  分位数不足以支撑单调性结论。看 **p90/p99 的绝对量级**（250~350ms，≈ `--ttft-ms 120`
  加上 16k~32k prompt 的处理开销）比看趋势更可靠

图表（matplotlib，Agg）：

- `results/agentx/charts/throughput_ttft_vs_concurrency.{png,svg}`
- `results/agentx/charts/cache_hit_vs_concurrency.{png,svg}`

### 3.4 指标口径

| 指标 | 采集方式 |
|---|---|
| TTFT | 请求发出 → 第一个含 `delta.content`/`tool_calls` 的 SSE chunk |
| TPOT | `(E2E - TTFT) / (completion_tokens - 1)`，`completion_tokens` 取服务端 `usage` |
| 缓存命中 | `vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` 的**本轮差值**；缺失时退回 `/api/stats` 的 `cache_hit_tokens/cache_query_tokens` |
| 吞吐 | `completion_tokens / 墙钟`；`tok/s/chip` 需 `--chips N`，否则留空 |
| 有效性 | 成功率 < 90% 标 `valid=false`（对齐 `failed_request_threshold=0.10`） |

---

## 4. 真实 harness 冒烟（`agentx-harness==1.0.7`）

### 4.1 安装（已成功）

```powershell
# 独立 venv，不污染项目 .venv
python -m venv .venv-agentx
.venv-agentx\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple `
  agentx-harness==1.0.7
# 证据：experiments/agentx_harness/logs/01_install.log（Resolved 130 packages，含 uvicorn/pyarrow/plotly）
```

### 4.2 实跑命令

```powershell
.venv-agentx\Scripts\agentx.exe profile `
  --url http://127.0.0.1:8097 --endpoint-type chat --streaming `
  --input-file agentx\traces --custom-dataset-type weka-trace `
  --unsafe-override --tokenizer deepseek_v3_tokenizer --model gpt-3.5-turbo `
  --concurrency 1 --benchmark-duration 30 `
  --output-artifact-dir experiments\agentx_harness\out
```

### 4.3 证据：scenario 锁被**真实注入**

`experiments/agentx_harness/logs/04_run_c1_30s_norealtime.log`（与 `out/logs/aiperf.log` 一致）：

```
Scenario 'agentx': setting profiling-phase timing_mode=agentic_replay.   (validator.py:226)
Scenario 'agentx': injecting extra_inputs.ignore_eos=true (was absent).  (validator.py:285)
Scenario 'agentx': auto-set --cache-bust=first_turn_prefix (was default none).  (validator.py:538)
Scenario 'agentx': auto-set --system-idle-gap-cap-seconds=10.0 (was unset).     (validator.py:779)
GPU telemetry disabled via --no-gpu-telemetry (system_controller.py:269)
Server metrics enabled - 1/1 endpoint(s) reachable. (system_controller.py:592)
```

这**直接证实**了 `docs/AGENTX.md` §2 的参数表：`agentx profile` 在不带 `--scenario` 时注入
`agentx` spec，而 spec 的 `timing_mode / ignore_eos / cache_bust / idle_gap_cap` 由 validator
强制改写 —— 不是文档推测。

### 4.4 两条 violation（`--unsafe-override` 下降级为警告）

```
WARNING Scenario violation (override active): --custom-dataset-type / --input-file:
  got 'weka_trace (local, unpinned)',
  required 'pinned --public-dataset alias or --hf-weka-dataset semianalysisai/cc-traces-weka-062126'
  → scenario 'agentx' cannot verify corpus identity for a local weka_trace directory
WARNING Scenario violation (override active): --benchmark-duration: got 30.0, required '>=900'
```

含义与影响：

| violation | 原因 | 后果 | 官方跑法 |
|---|---|---|---|
| 本地 `weka_trace` 目录非 pinned 语料 | AgentX 要求语料身份可核验（HF 别名或 `--hf-weka-dataset`） | **`submission_valid=false`**（数值不作数，但流程可跑） | 用 `--public-dataset semianalysis_cc_traces_weka_062126` 或 `--hf-weka-dataset` |
| `--benchmark-duration 30 < 900` | 官方要求 ≥900s 才到稳态并触发 KV offloading | 同上，且曲线未进稳态 | `--benchmark-duration 1800`（默认 3600） |

**结论**：两条 violation 是"离线冒烟"的预期代价，不是环境故障。要 `submission_valid=true`
必须：pinned 语料 + ≥900s + 真实模型服务。

### 4.5 卡点：卡在 tokenizer/dataset 配置（非网络）

第二次运行（`04_run_c1_30s_norealtime.log`）走得更远：ZMQ 起来了、5 个 manager 注册完成、
服务端 `/metrics` 基线抓取成功（`Server metrics enabled - 1/1 endpoint(s) reachable.`），
然后停在：

```
timing_manager   INFO  Waiting for dataset to be configured before configuring timing
dataset_manager  INFO  Configuring tokenizer(s) for dataset manager
```

此后无输出：

- `out/profile_export.jsonl` = **0 字节**
- mock 侧 `/api/stats` 的 `total_requests` = **0**（一个请求都没发出去）

即 **卡在 dataset/tokenizer 解析阶段**，而不是 HTTP/网络。多半是 `--tokenizer deepseek_v3_tokenizer`
这个别名在 harness 侧要走 HF tokenizer 解析（本地目录未按 HF 布局注册 / 需要
`--tokenizer <path>` 或 `trust_remote_code`），在离线环境里无法完成。

**第一次运行为什么更早就挂了**：`Address in use (addr='tcp://127.0.0.1:46337')` ——
AIPerf 的 ZMQ 前端 socket 端口被占（同机还有别的 aiperf 实例）。

### 4.6 绕过建议（按代价排序）

1. **不追真 harness**：用本仓库 `agentx replay`（同一套 schedule 语义，无需 aiperf）。
   这是当前交付路径。
2. **绕过 tokenizer 解析**：把 `--tokenizer` 指向本地 HF 目录（含 `tokenizer.json`），
   或用 harness 内置别名（如 `Qwen/Qwen2.5-7B`）让它走缓存；再配
   `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` 避免联网等待。
3. **绕过语料身份**：`--custom-dataset-type weka_trace --input-file <dir> --unsafe-override`
   （= `submission_valid=false`，仅形状验证）；要 valid 则必须走 HF pinned 语料。
4. **绕过时长**：`--benchmark-duration 900` 起步（1 小时更接近官方）；本地可用
   `--benchmark-duration 30 --unsafe-override` 只看流程。
5. **绕开 ZMQ 冲突**：同机只跑一个 aiperf 实例，或用 `AIPERF_ZMQ_*` 端口环境变量错开。

### 4.7 结论（引用）

> 能起到「scenario 锁注入完成 + 服务端 metrics 连通 + dataset 配置开始」；
> 卡在 tokenizer/dataset 解析（0 请求发出）；
> 通过 `--unsafe-override` 可绕过合规性检查，但要 `submission_valid=true` 必须有
> pinned HF 语料 + ≥900s + 真实模型服务。
> **负载形状与参数可离线复现，官方分数不可离线复现。**

---

## 5. 踩坑清单（对后来者最有用）

| 现象 | 根因 | 处理 |
|---|---|---|
| `httpx.ReadTimeout` 连 127.0.0.1 | Windows 上 httpx `trust_env=True` 会从注册表读系统代理，把 localhost 请求送进代理 | `AsyncClient(trust_env=False)`（`agentx/replay.py` 已固化） |
| `UnicodeEncodeError: 'gbk' codec can't encode '\u2713'` | Windows 控制台默认 GBK，`✓/✗/⚠` 不在 GBK 字符集 | 输出统一用 `[OK]/[ERR]/[WARN]`；`sys.stdout.reconfigure(errors="replace")` 兜底；`tests/test_agentx_replay.py::test_cli_source_is_gbk_safe` 回归防护 |
| 回放"卡住"很久 | ① `warmup=10` × 每条约 3~12s 串行预热；② 子代理内层 `t` 远超窗口时曾无条件 `sleep` | ① 预热与测量两阶段分离；② `sleep_until()` 用 deadline 夹紧，越界立即返回 |
| `TPOT = 0.06ms` 失真 | 服务端带 `tools` 的轮次只回 `tool_calls`，`completion_tokens << max_tokens` | 服务端改为「先 content 帧再 tool_calls 帧，总 token 数精确 = `max_tokens`」；回放侧 TPOT 用 `usage.completion_tokens` |
| 首轮永远冷、缓存曲线失真 | `hash_ids=[]` → AIPerf 走随机 prompt 路径 | 每轮（含首轮）都给 `len(hash_ids)*64 <= in` 的块 ID |
| JOIN 提前唤醒 | `duration_ms` 四舍五入小于内层请求结束时刻 | 生成器改 `ceil`；校验器断言 `duration_ms` 不短于 `max(inner.t + api_time)` |

---

## 6. 复现清单（一条命令跑完）

```powershell
# 1. 生成 + 校验
.venv\Scripts\python.exe -m agentx generate --sessions 8 --requests 30 --subagents 0.4 `
  --max-isl 32000 --out agentx\traces
.venv\Scripts\python.exe -m agentx validate --traces agentx\traces

# 2. 起服务（另开一个终端）
.venv\Scripts\python.exe server.py --port 8090 --tokenizer deepseek_v3_tokenizer `
  --ttft-ms 120 --tpot-ms 8 --max-concurrency 64

# 3. 回放 + 出报告图表
.venv\Scripts\python.exe -m agentx replay --traces agentx\traces `
  --url http://127.0.0.1:8090 --concurrency 1,2,4 --duration 30 --warmup 10 --out results\agentx

# 4. 测试
.venv\Scripts\python.exe -m pytest tests\test_agentx_trace.py tests\test_agentx_replay.py -q
```

---

## 7. 产物索引

| 文件 | 内容 |
|---|---|
| `agentx/trace_gen.py` | WEKA trace 合成器 + `validate_trace()` |
| `agentx/replay.py` | 闭环并发回放器（SPAWN/JOIN、cache-bust、prefix cache 差值、图表） |
| `agentx/cli.py` | `generate｜replay｜report｜validate` |
| `agentx/traces/*.json` | 8 条合成语料（784 请求 / 96 子代理） |
| `results/agentx/replay.json` | 自包含 schema（runs + meta，可 `report` 重建） |
| `results/agentx/replay.md` / `.csv` | 人读表格 / 机读表格 |
| `results/agentx/charts/*.{png,svg}` | 吞吐+TTFT vs 并发、缓存命中率 vs 并发 |
| `tests/test_agentx_trace.py` | schema + 语义（前缀递增、块上界、JOIN） |
| `tests/test_agentx_replay.py` | plan/marker/prompt/统计/报告/CLI/GBK |
| `tests/_weka_schema_mirror.py` | 上游 pydantic 模型镜像（离线校验真 harness 兼容性） |
| `experiments/agentx_harness/logs/*` | 真 harness 冒烟证据 |
| `docs/AGENTX.md` | 调研报告：参数表 + schema + 复现映射表 |
