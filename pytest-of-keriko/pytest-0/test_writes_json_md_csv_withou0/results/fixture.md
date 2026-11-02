# GPT-Zero 压测报告 — fixture

- 生成时间：2026-01-01T00:00:00+08:00
- 服务地址：`http://127.0.0.1:8090`
- 模型：`gpt-3.5-turbo`
- 工具：gptzero-bench 0.1.0

## 1. 服务端配置（模拟硬件参数）

| 参数 | 值 | 含义 |
|---|---|---|
| `ttft_ms` | 120.0 | 基础首 token 延迟（不含排队/预填充放大） |
| `tpot_ms` | 8.0 | 每个输出 token 的解码时间 |
| `max_concurrency` | 64 | 服务端并发槽位；超出后请求排队，TTFT 抬升 |
| `block_size` | - | prefix cache 块大小（token） |
| `cache_blocks` | - | prefix cache 容量（块） |

配置一致性：⚠ 存在不一致或未记录校验信息

## 2. 压测配置

| 参数 | 值 |
|---|---|
| profiles | coding |
| concurrency | 2, 4 |
| duration_s | 10.0 |
| warmup | 2 |
| seed | 42 |
| max_tokens | 按 profile 默认 |

## 3.1 profile `coding`

| 并发 | 请求数 | 成功率 | TTFT p50 (ms) | TTFT p90 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | ITL p50 (ms) | E2E p50 (ms) | E2E p90 (ms) | E2E p99 (ms) | RPS | 输入 tok/s | 输出 tok/s | Cache Hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 100.0% | 130.0 | 140.0 | 145.0 | 8.0 | 8.0 | 630.0 | 730.0 | 780.0 | 2.00 | 10000.0 | 200.0 | 50.0% |
| 4 | 20 | 100.0% | 140.0 | 150.0 | 155.0 | 8.0 | 8.0 | 640.0 | 740.0 | 790.0 | 2.00 | 10000.0 | 200.0 | 50.0% |

该 profile 共 40 条请求；表中每行为一个独立测量点。

## 4. 图表

_本次未生成图表（`--no-charts`）。_

## 5. 结论

- 并发 2 → 4（profile `coding`）：RPS 由 2.00 提升到 2.00（+0%）。
- TTFT p90 由 140 ms 变化到 150 ms（1.07×，上升）；TTFT 中包含服务端排队时间，服务端 max_concurrency=64。
- 输出吞吐由 200.0 tok/s 变化到 200.0 tok/s，输入吞吐由 10000.0 到 10000.0 tok/s。
- 服务端 token 级 prefix cache 命中率区间为 50.0% ~ 50.0%（按并发点）。
- 所有测量点失败率为 0。

_数据自检：所有测量点配置一致、TPOT 与配置相符、无异常样本数。_

## 6. 备注

- fixture note

## 7. 复现命令

```bash
python -m bench sweep --url http://127.0.0.1:8090 --profiles coding --concurrency 2,4 --duration 10.0 --name fixture
python -m bench report fixture.json
```
