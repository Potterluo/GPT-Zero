# AgentX 闭环回放报告

- 目标服务：`http://x`
- 模型：`m`
- trace 语料：`agentx/traces`（2 条）
- 每点时长：10.0s，warmup/lane：10，primer：True
- 生成时间：now
- 有效点：2/2

## 汇总

| 并发 | 请求 | 成功 | 成功率 | TTFT p50 | TTFT p90 | TTFT p99 | TPOT p50 | E2E p90 | req/s | out tok/s | tok/s/chip | 缓存命中 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 20 | 20 | 100.0% | 200 | 240 | 280 | 8.00 | 3500 | 2.00 | 100.0 | — | 95.0% |
| 2 | 20 | 20 | 100.0% | 260 | 312 | 364 | 8.00 | 3500 | 2.00 | 190.0 | — | 97.0% |

## 与 AgentX 官方口径的差异

- n1

## 指标口径

- **TTFT**：请求发出 → 第一个含内容的 SSE chunk（`delta.content` 或 `tool_calls`）。
- **TPOT**：`(E2E - TTFT) / (completion_tokens - 1)`，`completion_tokens` 取服务端 `usage`。
- **缓存命中率**：`vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` 的本轮差值；缺失时退回 `/api/stats` 的 `cache_hit_tokens/cache_query_tokens`。
- **tok/s/chip**：需要 `--chips N`；未知时留空（官方按设备芯片数折算）。
- **成功率**：`--failed-request-threshold`（默认 0.10）以下仍标 `valid=false`，不出结论。

字段级 schema 见 `results/agentx/replay.json`，逐请求明细用 `--dump-requests` 打开。
