# AgentX 离线测试镜像（agentx-offline）

一个**自带 AgentX 官方 harness + 真实语料**的离线测试镜像：在无外网的机房/实验室里，对任意
OpenAI 兼容的 LLM 推理服务跑 AgentX，并产出官方 artifacts + 可交互的 HTML 对比报告。

> 与 `gptzero:v3`（mock 服务镜像）的区别：
> `gptzero` 是**被测服务**（没有真模型，用于自测链路）；`agentx-offline` 是**测试工具**，
> 打的是你真实的 vLLM / SGLang / MindIE / TRT-LLM / 云 API 服务。

## 1. 镜像里有什么

| 组件 | 说明 |
|---|---|
| `agentx-harness==1.0.7` | 官方 harness（`aiperf` + AgentX scenario 锁） |
| 依赖 | 固定版本：`sentencepiece==0.2.0`（0.2.2 在部分机器上 DLL 崩溃）、transformers/datasets/pyarrow 等 |
| 数据集 | `semianalysisai/cc-traces-weka-062126` 的 Arrow 缓存（393 条 trace，1.77 GB） |
| Tokenizer | `deepseek-ai/DeepSeek-V3` 的 HF 缓存（8 MB） |
| 驱动 | `runner.py`：一个 target × 多个并发点，逐点跑 `agentx profile`，自动校验产物 |
| 报告 | `report.py`：多目标对比的**单文件 HTML**，ECharts 内联（无 CDN、无外网） |

镜像默认 `HF_HOME=/opt/hf` 且 `HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 / HF_DATASETS_OFFLINE=1`，
运行期**不需要任何外网**；只需要容器能访问被测服务的地址。

## 2. 构建与离线交付

```bash
# 0) 准备缓存（一次性；本仓库 data/hf 已由 experiments/agentx_real/prefetch_dataset.py 生成）
pwsh scripts/prepare_agentx_cache.ps1     # Windows
# bash scripts/prepare_agentx_cache.sh    # Linux/macOS

# 1) 构建（上下文 = agentx_offline/，约 3.6 GB）
docker build -t agentx-offline:1.0.7 agentx_offline

# 2) 导出成离线包
docker save -o dist/agentx-offline-1.0.7.tar agentx-offline:1.0.7
sha256sum dist/agentx-offline-1.0.7.tar > dist/agentx-offline-1.0.7.tar.sha256

# 3) 目标机器（无外网）
docker load -i agentx-offline-1.0.7.tar
docker run --rm agentx-offline:1.0.7 smoke     # 自检：数据集体+tokenizer 能否离线加载
```

精简版（不烤数据集，运行时挂卷）：

```bash
docker build --build-arg WITH_CACHE=0 -t agentx-offline:slim agentx_offline
docker run --rm -v /srv/hf:/opt/hf -v "$PWD/results:/results" agentx-offline:slim run ...
```

## 3. 对一个真实 LLM 服务跑

```bash
docker run --rm \
  -v "$PWD/results:/results" \
  --add-host=host.docker.internal:host-gateway \
  agentx-offline:1.0.7 run \
    --url http://10.0.0.5:8000 \
    --model Qwen3-8B \
    --tokenizer deepseek-ai/DeepSeek-V3 \
    --target qwen3-8b-vllm \
    --concurrency 4,8,16 \
    --duration 900 \
    --entries 8
```

- `--url`：OpenAI 兼容 base URL（也接受完整的 `/v1/chat/completions`）；
- `--model`：服务端真实看到的 model id（多模型服务可分别跑）；
- `--duration`：每个并发点的 profiling 秒数；官方要求 ≥900，低于 900 会打 `submission_valid=false`；
- `--entries`：使用多少条真实 trace；393 条全量需要 ~30GB+ 内存，小机器用 8~32；
- `--concurrency`：并发 **agent 会话**数（不是 HTTP 批次），可逗号分隔多个点。

镜像内也会自动处理 Windows 风格端口窗口（`AIPERF_SERVICE_WINDOWS_TCP_*`，Linux 无影响）。

## 4. 多模型 / 多 infra 对比

同一个 results 卷里，每个 target 一个子目录；重复执行 `run` 时用不同 `--target`：

```bash
docker run --rm -v "$PWD/results:/results" agentx-offline:1.0.7 run \
  --url http://10.0.0.5:8000 --model Qwen3-8B --target qwen3-vllm \
  --concurrency 4,8,16 --duration 900

docker run --rm -v "$PWD/results:/results" agentx-offline:1.0.7 run \
  --url http://10.0.0.6:8000 --model Qwen3-8B --target qwen3-sglang \
  --concurrency 4,8,16 --duration 900

docker run --rm -v "$PWD/results:/results" agentx-offline:1.0.7 report --root /results
```

`report.html` 会给出：

- 每个 target 的 KPI 卡（峰值输出吞吐、最好 TTFT p50、服务端缓存命中）；
- 汇总表（请求、成功率、RPS、TTFT p50/p90/p99、ITL、E2E、tok/s、ISL/OSL、缓存理论/实测）；
- 6 张交互式图（TTFT p50、TTFT p90、输出吞吐、请求吞吐、E2E p90、缓存命中）按 target 对比；
- 每个 target 的时序页（profiling 每秒的 TTFT 与吞吐）。

图表为 ECharts 内联渲染，**单文件、可离线打开**（发给同事也不需要联网）。

## 5. 产物目录（完整数据）

```
results/
  report.html                       # 单文件交互报告
  report_data.json                  # 报告数据（机器可读）
  <target>/c<并发>/
    profile_export_aiperf.json      # 官方聚合指标
    profile_export_aiperf.csv
    profile_export.jsonl            # 逐请求记录
    profile_export_aiperf_timeslices.csv/.json
    server_metrics_export.json/.csv # 被测服务的 /metrics 快照（若暴露）
    profile_export_console.txt      # 官方 console 报告
    logs/aiperf.log                 # 全量日志（scenario 注入、每 30s 指标）
    run.log                         # 容器内驱动日志
  <target>/run_summary.json         # 该 target 的点位、耗时、退出码
```

## 6. 已验证（本机实测，2026-10-28）

| 项 | 结果 |
|---|---|
| 镜像 | `agentx-offline:1.0.7`，**3.92 GB**（Python 3.11 + agentx 1.0.7 + 1.77GB Arrow 语料 + tokenizer） |
| 离线自检 | `docker run --rm agentx-offline:1.0.7 smoke` → `dataset rows: 393`、`tokenizer ok`、`offline smoke OK` |
| 真跑 | 2 个 target（fast / slow）× c=4、c=8 × 120s，全部 `ok=True exit=0` |
| 报告 | 单文件 `report.html` 1.05 MB，离线可开，无外部请求 |

两个 target 是同一台机器上的 gptzero mock（模拟 fast/slow 两档 infra），用于验证链路：

| target | 并发 | 请求 | TTFT p50/p90/p99 (ms) | 输出 tok/s | 缓存命中（理论/实测） |
|---|---|---|---|---|---|
| mock-fast | 4 | 35 | 143 / 237 / 45,210* | 78 | 93.2% / 92.9% |
| mock-fast | 8 | 14 | 255 / 405 / 435 | 155 | 81.7% / 81.5% |
| mock-slow | 4 | 43 | 375 / 582 / 651 | 56 | 93.5% / 93.3% |
| mock-slow | 8 | 12 | 739 / 1,107 / 1,250 | 139 | 79.4% / 80.6% |

\* 真实离群点：一条超长上下文 prefill；原始记录见 `results/agentx_offline_demo/mock-fast/c4/profile_export.jsonl`。

完整 demo（报告 + 4 个点的全部官方 artifacts）：`results/agentx_offline_demo/`；
截图：`results/agentx_offline_demo/report_preview.png`。

换成真实服务时只需改 `--url/--model/--target`，其余不变。

## 7. 已知边界

- 镜像里的语料是 Arrow 缓存，**不含**原始 1.85GB `traces.jsonl`（离线复现只需要 Arrow）；
- 393 条全量需要大内存机器（重建会并行展开会话），`--entries` 是离线小机器的必要开关；
- 官方 submission 还要求 pinned 语料 + ≥900s + 真实模型服务；本镜像满足前三者的形式要求，
  但分数的解释权仍取决于你的服务是否真实（mock 分数没有意义）；
- GPU 遥测默认关闭（`--no-gpu-telemetry`）；需要的话在 `--extra` 里打开并挂载 GPU 设备。
