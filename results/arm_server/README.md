# ARM64 服务器实测（openEuler + Ascend 910B3，2TB 内存）

在 110.138.0.3（aarch64、256 核、2014 GB、Docker 24.0.9、openEuler）上，用与镜像相同的
`agentx-harness==1.0.7` + 官方语料，对两个 mock 服务跑 AgentX；harness 跑在容器内
（`quay.io/ascend/vllm-ascend:v0.30.0rc2` 的 Python 3.12 venv），语料走 HF 缓存
（`/data/agentx/hf`，393 条，下载自 hf-mirror）。这不是镜像 demo，而是 **ARM 原生执行**验证。

## 命令（与镜像内一致）

```bash
HF_HOME=/data/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
/data/venv/bin/agentx profile \
  --url http://127.0.0.1:8090 --endpoint-type chat --streaming \
  --model gpt-3.5-turbo --tokenizer deepseek-ai/DeepSeek-V3 \
  --concurrency 4 --benchmark-duration 120 --num-dataset-entries 8 \
  --public-dataset semianalysis_cc_traces_weka_062126 --unsafe-override \
  --output-artifact-dir /data/results/mock-fast/c4
```

mock 两个 target：`gptzero-fast`（8090，ttft 120ms/tpot 6ms）与 `gptzero-slow`
（8091，ttft 320ms/tpot 14ms），都以容器方式在 host 网络运行。

## 结果（真实运行）

| target | 并发 | 请求 | TTFT p50/p90 (ms) | ITL p50 (ms) | 输出 tok/s | 缓存命中（理论/服务端） |
|---|---|---|---|---|---|---|
| mock-fast | 4 | 54 | 185 / 257 | 6.14 | 92.9 | 95.1% / 94.9% |
| mock-fast | 8 | 14 | 331 / 525 | 6.15 | 155.4 | 81.7% / 81.7% |
| mock-slow | 4 | 43 | 435 / 582 | 14.22 | 55.6 | 93.6% / 93.3% |
| mock-slow | 8 | 12 | 749 / 1,189 | 14.38 | 138.6 | 79.4% / 80.7% |

与 x86 demo（`results/agentx_offline_demo/`）对比：延迟/吞吐量级一致，差异来自 mock 所在机器
的计时器与 CPU 调度；ARM 上 c=4 的 fast TTFT 185ms（x86 143ms）符合预期。

## 文件

- `report.html`：ARM 上生成的单文件报告（ECharts 内联）
- `report_data.json`：报告数据源

原始 artifacts 留在服务器 `/data/agentx/results/<target>/c<N>/`（未随仓库提交）。
