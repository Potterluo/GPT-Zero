# AgentX 真实 harness 离线跑分（真实语料）

> 运行者：真正的 `agentx-harness==1.0.7`（不是自研回放）；服务端是 GPT-Zero mock。
> 语料：`semianalysisai/cc-traces-weka-062126`（393 traces，HF 镜像预下载到本地缓存后断网运行）。
> 模型名：`gpt-3.5-turbo`（mock 不加载权重），tokenizer：`deepseek-ai/DeepSeek-V3`。

| 并发 | 时长(s) | 请求 | RPS | TTFT p50/p90/p99 (ms) | ITL p50 (ms) | E2E p50/p90 (ms) | out tok/s | ISL p50 | OSL p50 | 缓存命中(理论/服务端) |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 180 | 58 | 0.28 | 160 / 239 / 370 | 8.7 | 1175 / 2491 | 103 | 54258 | 124 | 95.53% / 93.4% |

## 说明

- `缓存命中(理论/服务端)`：AIPerf 从 trace 的 64-token 块算出的理论 LCP 命中率 / mock 的
  `vllm:prefix_cache_hits_total ÷ queries_total`（profiling 阶段）。两者有差距说明 mock 的
  `--cache-blocks` 容量不足，发生了淘汰（可调大后重跑）。
- 每点时长被压到 120~180s（官方要求 ≥900s）才在单机可跑；这会导致请求数偏少、分位数抖动，
  但**语料、scenario 锁、warmup、cache-bust、ignore_eos、闭环并发**全部是真实的。
- 这是**性能形状**验证，不是官方 submission：mock 没有真实模型、GPU 遥测与 KV offload。

## 原始产物

- `experiments/agentx_real/run_c4_180/`：profile_export*、server_metrics_export*、logs/
