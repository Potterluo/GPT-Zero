# ARM64 镜像实测（推送到腾讯云 CCR 后，在 Ascend 服务器上拉下来跑）

镜像：`ccr.ccs.tencentyun.com/keriko/agentx-offline:1.0.7-arm64`（arm64，3.57 GB，内置 393 条
语料 + DeepSeek-V3 tokenizer）。

服务器：110.138.0.3（aarch64 / openEuler / 256 核 / 2014 GB / Docker 24.0.9 / 8×Ascend 910B3）。

## 服务器上执行

```bash
docker login ccr.ccs.tencentyun.com -u 100013814339
docker pull ccr.ccs.tencentyun.com/keriko/agentx-offline:1.0.7-arm64

docker run --rm ccr.ccs.tencentyun.com/keriko/agentx-offline:1.0.7-arm64 smoke
# dataset rows: 393 / tokenizer ok / offline smoke OK

docker run --rm --net host -v /data/agentx/image_results:/results \
  ccr.ccs.tencentyun.com/keriko/agentx-offline:1.0.7-arm64 run \
  --url http://127.0.0.1:8090 --model gpt-3.5-turbo --target mock-fast \
  --concurrency 4 --duration 120 --entries 8
# ok=True exit=0 183.9s
```

两个 mock（gptzero-fast 8090 / gptzero-slow 8091）在同一台机器上以 host 网络运行。

## 结果（镜像内真实执行）

| target | 并发 | 请求 | TTFT p50/p90 (ms) | ITL p50 (ms) | 输出 tok/s | 缓存命中（理论/服务端） |
|---|---|---|---|---|---|---|
| mock-fast | 4 | 54 | 184 / 252 | 6.14 | 92.9 | 95.1% / 94.9% |
| mock-slow | 4 | 43 | 434 / 575 | 14.21 | 55.6 | 93.6% / 93.3% |

`report.html` 是镜像里的 `agentx-offline report` 直接生成后拉回的，未在本地重算。
原始 artifacts 在服务器 `/data/agentx/image_results/<target>/c4/`。
