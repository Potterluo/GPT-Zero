# AgentX 闭环回放报告

- 目标服务：`http://127.0.0.1:8092`
- 模型：`gpt-3.5-turbo`
- trace 语料：`agentx/traces`（8 条）
- 每点时长：30.0s，warmup/lane：10，primer：True
- 生成时间：2026-10-28T19:28:22+0800
- 有效点：3/3

## 汇总

| 并发 | 请求 | 成功 | 成功率 | TTFT p50 | TTFT p90 | TTFT p99 | TPOT p50 | E2E p90 | req/s | out tok/s | tok/s/chip | 缓存命中 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 3 | 100.0% | 227 | 241 | 244 | 5.63 | 5799 | 0.09 | 70.7 | — | 96.3% |
| 2 | 6 | 6 | 100.0% | 174 | 201 | 203 | 5.84 | 5422 | 0.18 | 88.3 | — | 97.9% |
| 4 | 10 | 10 | 100.0% | 207 | 287 | 287 | 6.08 | 6103 | 0.30 | 146.3 | — | 98.4% |

## 与 AgentX 官方口径的差异

- 本地 mock 无真实模型：token 内容与质量无意义，只保证 token 计数/流式时序/缓存语义。
- benchmark 时长按分钟级跑（官方下限 900s、默认 3600s），曲线形状可看，绝对值不可与官网对比。
- 语料为合成 trace（形状对齐 393 条语料：median in≈142k / out≈444 / 98% cached，规模小 2~3 个数量级）。
- tok/s/chip 需 --chips 显式给芯片数，否则留空。
- 不涉及 speculative acceptance / GPU 遥测 / 厂商 chat template。

## 指标口径

- **TTFT**：请求发出 → 第一个含内容的 SSE chunk（`delta.content` 或 `tool_calls`）。
- **TPOT**：`(E2E - TTFT) / (completion_tokens - 1)`，`completion_tokens` 取服务端 `usage`。
- **缓存命中率**：`vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` 的本轮差值；缺失时退回 `/api/stats` 的 `cache_hit_tokens/cache_query_tokens`。
- **tok/s/chip**：需要 `--chips N`；未知时留空（官方按设备芯片数折算）。
- **成功率**：`--failed-request-threshold`（默认 0.10）以下仍标 `valid=false`，不出结论。

字段级 schema 见 `results/agentx/replay.json`，逐请求明细用 `--dump-requests` 打开。
