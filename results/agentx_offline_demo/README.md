# AgentX 离线镜像 demo（两个 target × 两个并发点）

这是 `agentx-offline:1.0.7` 的**端到端验证数据**：镜像在 Linux 容器里，对宿主机上的两个
"推理服务"（用 gptzero mock 模拟 fast / slow 两档 infra）跑官方 AgentX 回放。

> 注意：两个 target 都是 **mock**（没有真实模型），这里只证明「离线镜像 → 真 harness → 真语料
> → 官方 artifacts → HTML 对比报告」这条链路可用；换成你的 vLLM / SGLang / API 服务即得真实数据。

## 怎么跑出来的

```bash
# 两个被测服务（示例用 gptzero 模拟两档 infra）
python server.py --port 8092 --tokenizer deepseek_v3_tokenizer --ttft-ms 100 --tpot-ms 6  --max-concurrency 64 --cache-blocks 65536
python server.py --port 8093 --tokenizer deepseek_v3_tokenizer --ttft-ms 320 --tpot-ms 14 --max-concurrency 64 --cache-blocks 65536

# target A（fast）
docker run --rm --add-host host.docker.internal:host-gateway \
  -v "$PWD/results/agentx_offline_demo:/results" agentx-offline:1.0.7 \
  run --url http://host.docker.internal:8092 --model gpt-3.5-turbo \
      --target mock-fast --concurrency 4,8 --duration 120 --entries 8

# target B（slow）
docker run --rm --add-host host.docker.internal:host-gateway \
  -v "$PWD/results/agentx_offline_demo:/results" agentx-offline:1.0.7 \
  run --url http://host.docker.internal:8093 --model gpt-3.5-turbo \
      --target mock-slow --concurrency 4,8 --duration 120 --entries 8

# 重新渲染报告（任意时候）
docker run --rm -v "$PWD/results/agentx_offline_demo:/results" \
  agentx-offline:1.0.7 report --root /results
```

每个点都在容器里跑完整 `agentx profile`（scenario `agentx`、`--unsafe-override` 仅因 120s < 900s），
`run.log` 是驱动日志，`logs/aiperf.log` 是 harness 原始日志。

## 结果摘要（真实运行，非手填）

| target | 并发 | 请求 | 成功率 | TTFT p50/p90/p99 (ms) | 交互性 tok/s | 输出 tok/s | 缓存命中（理论/实测） |
|---|---|---|---|---|---|---|---|
| mock-fast | 1 | 10 | 100% | 235 / 292 / 317 | 159 | 69 | 99.6% / 99.4% |
| mock-fast | 4 | 35 | 100% | 143 / 237 / 45,210 | 158 | 78 | 93.2% / 92.9% |
| mock-fast | 8 | 14 | 100% | 255 / 405 / 435 | 164 | 155 | 81.7% / 81.6% |
| mock-fast | 16 | 25 | 100% | 244 / 483 / 632 | 154 | 293 | 72.0% / 71.9% |
| mock-slow | 1 | 8 | 100% | 527 / 665 / 671 | 69 | 39 | 99.6% / 99.4% |
| mock-slow | 4 | 43 | 100% | 375 / 582 / 651 | 69 | 56 | 93.5% / 93.3% |
| mock-slow | 8 | 12 | 100% | 739 / 1,107 / 1,250 | 71 | 139 | 79.4% / 80.6% |
| mock-slow | 16 | 23 | 100% | 678 / 1,424 / 2,811 | 67 | 222 | 69.2% / 69.2% |

fast 的 c=4 p99=45s 是真实离群点（一条超长上下文 prefill）；报告里的“逐请求延迟 CDF”用
`profile_export.jsonl` 的 profiling 记录画，不看这个离群点也能读出分布形状。

## 文件

```
report.html              单文件交互报告（ECharts 内联，离线可开）
report_data.json         报告数据源（机器可读）
report_preview.png       报告截图
mock-{fast,slow}/c{4,8}/
  profile_export_aiperf.json|.csv   官方聚合指标
  profile_export.jsonl              逐请求记录
  profile_export_aiperf_timeslices.* 1s 切片
  server_metrics_export.json|.csv   被测服务 /metrics 快照
  profile_export_console.txt        官方 console 报告
  logs/aiperf.log, run.log
```
