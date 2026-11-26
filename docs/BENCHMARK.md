# GPT-Zero Benchmark Harness（`bench/`）

`bench/` 是一个只通过 HTTP 与 mock 推理服务交互的闭环压测工具：它生成多轮 agentic 会话，
精确测量流式时延，按并发扫描，输出自包含 JSON、中文 Markdown/CSV 表格和 6 组图表。

> 默认服务地址是 **`http://127.0.0.1:8090`**（本机 8080 被其他常驻服务占用）。
> 端口可用 `--url` 覆盖；`CONTRACT.md` 里的 8080 只是示例。

---

## 1. 快速开始

前提：mock 已启动。

```bash
# 终端 A：启动 mock（模拟 ttft=120ms / tpot=8ms 的“硬件”）
.venv\Scripts\python.exe server.py --port 8090 --tokenizer deepseek_v3_tokenizer \
    --ttft-ms 120 --tpot-ms 8 --max-concurrency 64

# 终端 B：单点压测
.venv\Scripts\python.exe -m bench run --url http://127.0.0.1:8090 \
    --profile coding --concurrency 8 --duration 20 --out results/x.json

# 多并发扫描 + 自动出表出图
.venv\Scripts\python.exe -m bench sweep --url http://127.0.0.1:8090 \
    --profiles coding --concurrency 1,2,4,8,16 --duration 15 --name sweep_mock_a

# 由已有 JSON 重建 md/csv/图
.venv\Scripts\python.exe -m bench report results/sweep_mock_a.json

# 一条命令：自己起停 mock 再扫描（便于复现；记得先选空闲端口）
.venv\Scripts\python.exe -m bench serve-sweep --launch-server --port 8097 \
    --ttft-ms 120 --tpot-ms 8 --max-concurrency 64 \
    --profiles coding --concurrency 1,2,4,8,16 --duration 15 --name serve_sweep
```

安装为命令后可直接用 `gptzero-bench run|sweep|report|serve-sweep`（入口同为 `bench/cli.py:main`）。

产物（以 `--name sweep_mock_a` 为例）：

```
results/sweep_mock_a.json   # 自包含：config + 每个并发点的汇总 + 样本 + 服务端快照
results/sweep_mock_a.md     # 中文报告：表 + 结论 + 自检
results/sweep_mock_a.csv    # 扁平表格，便于 pandas/Excel
results/charts/*.png|.svg   # 6 张图，PNG 嵌报告、SVG 可编辑
```

---

## 2. 工作负载 profile

所有会话由固定 `--seed` 派生（`sha256(profile:index:seed)`），可复现；每个请求都带
`ignore_eos=true` 和 `stream_options.include_usage=true`。

| profile | 形态 | 上下文 | 输出 |
|---|---|---|---|
| `chat` | 多轮闲聊 6~10 轮，短上下文 | ISL 200~800 | 默认 160 tok |
| `coding` | 工具增强编码会话，system 前缀 + 历史 + read/write/shell 工具结果回填，每轮带 4 个 tools schema | 2k 几何递增到 32k | 默认 192 tok |
| `agentic` | coding + **真实工具闭环**（assistant `tool_calls` → `role=tool` 结果 → 继续），20% 概率并行派生 subagent 子请求 | 2.5k → 32k | 默认 160 tok |
| `longctx` | 单轮长输入、短输出（半数 64k、半数 128k） | 64k/128k | 64 tok |

`coding` 每轮的上下文增长由 `_pad_to_target` 精确补到计划目标；`agentic` 的工具结果
（read_file 1.8k~3.2k tok、run_shell 0.7k~1.4k tok 等）和 20% fan-out 模拟 agentic 编码
Agent 负载的“工具结果驱动上下文膨胀”形态。可用 `--max-tokens` 覆盖输出长度。

---

## 3. 指标定义

客户端逐请求记录，失败请求计入失败率但不进入延迟分位数。

| 指标 | 定义 |
|---|---|
| **TTFT** | 请求发出 → 首个携带 `content` 或 `tool_calls` 的 SSE delta（`delta.role` 不算）。含服务端排队与预填充。 |
| **ITL** | 相邻两个内容帧的间隔（content 文本或 tool_call argument 片段）；按帧汇总。 |
| **TPOT** | `(E2E - TTFT) / (completion_tokens - 1)`；`completion_tokens` 取 usage，缺失时本地估算。 |
| **E2E** | 请求发出 → 流结束（含 `[DONE]`）。 |
| **RPS** | 测量窗口内 `请求总数 / wall_s`；窗口为所有在途请求的首尾跨度。 |
| **输入/输出 tok/s** | 成功请求的 prompt/completion token 之和 ÷ wall_s。 |
| **Cache Hit** | 服务端 `/metrics` 前后差值 `vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total`，**token 级**；`/metrics` 不可用时回退 `/api/stats` 计数。 |
| **成功率** | 成功请求 / 总请求；HTTP 非 200、超时、连接错误均算失败。 |

分位数用线性插值（与 numpy 默认一致）：p50/p90/p95/p99 + mean/max；空样本为 `null`，
报告里显示 `-`，不伪造 0。

每个并发点的 JSON 里同时保存：

* `server_config_full`：该点 `/api/config`（ttft_ms/tpot_ms/max_concurrency/...）；
* `warmup_probe`：预热请求的 TTFT/TPOT（仅诊断，不计入统计）；
* `sanity`：配置/计时自检结果与 warnings；
* `metrics_delta`：Prometheus 前后差值；
* `server_stats`：`/api/stats` 结束快照；
* `samples`：逐请求 TTFT/E2E 样本（画 CDF 用）。

---

## 4. 怎么读图

| 文件 | 内容 | 怎么读 |
|---|---|---|
| `ttft_vs_concurrency.png/svg` | TTFT p50/p90/p99 vs 并发 | 曲线开始抬升的拐点就是服务端槽位（`max_concurrency`）附近；p99 与 p50 的剪刀差代表排队尾部风险。 |
| `throughput_vs_concurrency.png/svg` | 请求/s 与输出 tok/s 双轴 | 吞吐走平而 TTFT 继续涨 = 已饱和，加并发只增加排队。 |
| `cache_hit_vs_concurrency.png/svg` | prefix cache 命中率 vs 并发 | 命中率高说明 system 前缀/工具 schema 被复用；并发升高时若下降，多是缓存被挤占或会话多样性上升。 |
| `latency_cdf.png/svg` | E2E 延迟 CDF（每个并发一条） | 曲线整体右移 = 延迟变差；尾部越长 p99 越差。 |
| `pareto.png/svg` | 输出吞吐 vs p90 TTFT 散点 | 越靠左上越好；标注了对应并发，可用来选“甜点并发”。 |
| `profile_comparison.png/svg` | 最大并发点的 profile 对比 | 左图 p90 TTFT、右图输出 tok/s，用于比较 chat/coding/agentic 的代价。 |

图表标签全英文、matplotlib Agg 后端，不依赖中文字体与网络；同名 `.svg` 可无损缩放。

---

## 5. mock 参数 → “模拟硬件”映射

mock 不是真实 GPU，它用三个参数抽象推理服务的时延形态：

| 服务端参数 | 物理含义 | 对 bench 指标的影响 |
|---|---|---|
| `--ttft-ms` | 基础首 token 延迟（固定开销） | TTFT 的下限；实际 TTFT 还会随**未缓存** prompt token 数按 `ttft_ms * (1 + uncached/8192)^0.6` 放大（未设置 `--prefill-tps` 时）。 |
| `--tpot-ms` | 每 token 解码时间 | TPOT/ITL 的下限；输出越长 E2E 越高。Windows 上受 15.6ms 定时器精度影响，实测 TPOT 有量化抬高（见第 7 节）。 |
| `--max-concurrency` | 并发槽位（类似 GPU 的并发序列上限） | 超过该值的请求在服务端信号量上排队，TTFT 上升、RPS 走平。 |
| `--cache-blocks` / `--block-size` | prefix cache 容量/块大小 | 命中率与长前缀复用；块越大，短前缀越难命中。 |
| `--prefill-tps` / `--decode-tps` | 可选的 token 速率模型 | 设置后覆盖 `ttft-ms`/`tpot-ms` 的线性部分，更接近“按 token 计费”的硬件。 |

因此：

* `mock_a`（`ttft=120 tpot=8 max_conc=64`）≈ 一张“低延迟、中等并发”的加速卡；
* `mock_b`（`ttft=400 tpot=18 max_conc=8`）≈ 一张“高延迟、低并发”的小卡，8 并发以上必然排队。

对应启动命令见第 8 节。**bench 不假设真实硬件**，报告只描述 mock 在上述参数下的可观测行为。

---

## 5.1 本次正式结果概览

两组都在**修复后的服务端**上实测（Windows 1ms 高精度定时器，启动日志含
`timer : 1ms high-resolution timer enabled`），profile=`coding`、duration=15s、warmup=3，
逐点 `/api/config` 校验全部通过、失败率 0。

**sweep_mock_a**（`ttft=120ms tpot=8ms max_concurrency=64`，≈ 低延迟中等并发加速卡）

| 并发 | TTFT p50 | p90 | p99 | TPOT p50 | ITL p50 | E2E p50 | RPS | 输出 tok/s | Cache Hit | 成功率 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 171.6 | 202.5 | 217.0 | 8.07 | 23.13 | 1708.2 | 0.63 | 121.0 | 70.8% | 100% |
| 2 | 148.5 | 185.6 | 210.1 | 8.45 | 20.39 | 1751.4 | 1.21 | 232.4 | 87.3% | 100% |
| 4 | 158.1 | 198.0 | 237.5 | 8.17 | 18.83 | 1735.7 | 2.45 | 470.6 | 83.1% | 100% |
| 8 | 155.7 | 182.7 | 230.8 | 7.69 | 17.28 | 1642.0 | 4.95 | 950.9 | 84.0% | 100% |
| 16 | 156.5 | 194.9 | 232.3 | 7.38 | 16.26 | 1568.4 | 10.12 | 1942.4 | 83.6% | 100% |

**sweep_mock_b**（`ttft=400ms tpot=18ms max_concurrency=8`，≈ 高延迟低并发小卡）

| 并发 | TTFT p50 | p90 | p99 | TPOT p50 | ITL p50 | E2E p50 | RPS | 输出 tok/s | Cache Hit | 成功率 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 494.9 | 575.9 | 593.8 | 12.56 | 22.19 | 2868.9 | 0.34 | 64.4 | 63.9% | 100% |
| 2 | 450.7 | 516.0 | 551.4 | 16.21 | 34.26 | 3557.4 | 0.58 | 111.9 | 80.4% | 100% |
| 4 | 442.2 | 501.3 | 543.1 | 15.82 | 33.76 | 3528.7 | 1.08 | 207.7 | 79.5% | 100% |
| 8 | 453.9 | 525.5 | 546.4 | 15.88 | 33.80 | 3446.3 | 2.17 | 416.7 | 82.6% | 100% |
| 16 | 3935.1 | 4408.1 | 5048.5 | 17.02 | 37.38 | 6982.8 | 2.04 | 390.8 | 76.7% | 100% |

读数提示：

* **TTFT 的基线与预填充放大**：mock_a 的 TTFT p50 在 150~172ms，高于 `ttft-ms=120ms`，
  因为 coding 的未缓存 prompt 会按 `120*(1+uncached/8192)^0.6` 放大，并叠加 HTTP/序列化开销；
  mock_b 的基线约 450~495ms，与 400ms 配置同量级。
* **ITL 是“帧间隔”而非单 token**：服务端每帧发送约 3 个 token（`max_tokens=192` 时），
  所以 ITL p50 ≈ 16~23ms（mock_a）/ 22~37ms（mock_b），而 TPOT（每 token）≈ 7~8.5ms /
  12.6~17ms，与配置 8ms / 18ms 对齐。看 TPOT 更接近“每 token 解码时间”。
* **mock_b 在并发 16 时的台阶**：`max_concurrency=8`，16 并发时一半请求排队，
  RPS 从 2.17 掉到 2.04（走平），E2E p50 从 3.4s 跳到 7.0s，TTFT p90 从 525ms 跳到 4408ms。
  这是**配置的并发槽位**造成的，不是被测系统“变慢”。
* **mock_a 在 16 并发内仍近似线性**：RPS 0.63→10.12（约 16×），E2E p50 稳定在 1.57~1.75s，
  说明本机 CPU 在 16 并发 × 64 槽位下尚未成为瓶颈；更高并发（32/64）未纳入本次交付，
  单机合理压测上限建议 ≤16~32 并发（再高会与 OS/其他进程抢 CPU，且 Python mock 的
  逐 token sleep 会放大调度噪声）。
* **补充 profile 对比**：`results/sweep_profiles.*` 在同一台 mock 上对 chat/coding/agentic
  以并发 8 各跑 8s，用来看三种负载的 p90 TTFT 与输出 tok/s 差异（图表
  `results/charts/sweep_profiles/profile_comparison.png`）。

对应产物：

```
results/sweep_mock_a.json|md|csv   results/charts/sweep_mock_a/*.png|svg
results/sweep_mock_b.json|md|csv   results/charts/sweep_mock_b/*.png|svg
results/sweep_profiles.json|md|csv results/charts/sweep_profiles/*.png|svg
```

---

## 6. 自检（数据可信度）

从 v0.1.0 起，runner 在**每个并发点**都会：

1. 读取 `/api/config`，与 sweep 首点比较 `ttft_ms/tpot_ms/max_concurrency/cache_blocks/block_size`；
   **不一致直接抛错中止**（端口被别的实例接管时宁可失败，也不产出假数据）。
2. 记录 `warmup_probe`（RTT/TTFT/TPOT，仅诊断）与实测 TPOT、ITL p50；若实测
   **TPOT 或 ITL** p50 < 配置值的 50%，在该点写 `sanity.warnings`，Markdown 表格的
   并发列标 `⚠` 并在「5.1 数据自检警告」列明原因（典型原因：服务端没启用 1ms 定时器、
   或同端口混入旧实例）。
3. 校验 `样本数 == 成功请求数`，防止记录跨点串号。
4. 每个点使用独立的 `records` 列表与独立的 before/after 快照，不存在跨点复用数组。

```bash
# 自检结果在 JSON 里
python -c "import json;d=json.load(open('results/sweep_mock_a.json',encoding='utf-8'));print(d['server']['config_verified'], d['sanity'])"
```

---

## 7. 已知限制（务必阅读）

1. **单机 Python mock 的吞吐上限**：服务端是单进程 uvicorn，每个 token 一次 `asyncio.sleep`；
   高并发下 CPU/事件循环本身会成为瓶颈。16 并发时若 RPS 不再上升而 TTFT 飙升，那是
   **mock 的 CPU 饱和**，不是被测系统的真实上限。
2. **Windows 定时器精度**：旧版服务端在 Windows 上直接 `asyncio.sleep(8~16ms)`，系统定时器
   粒度约 15.6ms，sleep 会提前返回，导致内容帧突发投递（ITL≈0.1ms、TPOT 偶发 0.06ms）。
   当前服务端已修复：启动时 `winmm.timeBeginPeriod(1)` 把精度提升到 1ms，并用绝对 deadline
   循环补睡（`_pace`）。启动日志出现 `timer : 1ms high-resolution timer enabled` 才算生效。
   修复后实测 `--tpot-ms 8` 时 ITL p50≈7~16ms、TPOT p50≈7~10ms，与配置相符；`--tpot-ms 18`
   时 TPOT≈17~20ms。bench 的 ITL 自检会拦截旧版失真数据。
3. **偶发“sleep 补发”**：事件循环被长阻塞（如超长 prompt 的 tokenizer/哈希）后，已到期
   的多个 sleep 会在恢复时立即返回，导致个别请求解码“瞬间完成”。1ms 定时器 + `_pace`
   已大幅缓解；自检仍会检查 TPOT/ITL 异常并打 `⚠`。本报告两组正式结果无异常点。
4. **非真实 GPU 行为**：没有 KV cache 显存压力、没有 continuous batching 的真实调度、
   没有网络/序列化之外的硬件效应；cache 命中是块哈希模拟，不是真实 KV 复用。
5. **长上下文成本**：`longctx` 的 128k prompt 会带来明显的 HTTP 上传与 tokenizer 开销，
   单机结果波动较大，建议单独短跑观察，不要与高并发小请求混在一张表里比较。
6. **闭环 vs 开环**：bench 固定 N 个会话闭环，RPS 由“响应时间 × 并发”决定；这与真实
   生产流量的开环泊松到达不同，不能把 RPS 当作容量上限。
7. **图表字体**：英文标签 + DejaVu Sans，避免容器内缺中文字体导致方块；文档正文中文。

---

## 8. 正式结果复现

```bash
# sweep_mock_a：低延迟、64 并发槽位
.venv\Scripts\python.exe server.py --port 8090 --tokenizer deepseek_v3_tokenizer \
    --ttft-ms 120 --tpot-ms 8 --max-concurrency 64
.venv\Scripts\python.exe -m bench sweep --url http://127.0.0.1:8090 \
    --profiles coding --concurrency 1,2,4,8,16 --duration 15 --warmup 3 --name sweep_mock_a

# sweep_mock_b：高延迟、8 并发槽位（制造排队）
.venv\Scripts\python.exe server.py --port 8093 --tokenizer deepseek_v3_tokenizer \
    --ttft-ms 400 --tpot-ms 18 --max-concurrency 8
.venv\Scripts\python.exe -m bench sweep --url http://127.0.0.1:8093 \
    --profiles coding --concurrency 1,2,4,8,16 --duration 15 --warmup 3 --name sweep_mock_b
```

> 注：若 8080 空闲，可直接把 `--url` 换成 `http://127.0.0.1:8080`；bench 默认 8090。

---

## 9. 测试

```bash
.venv\Scripts\python.exe -m pytest tests/test_bench_aggregate.py tests/test_bench_client.py \
    tests/test_bench_workload.py tests/test_bench_report.py -q
```

覆盖：分位数/空样本/单样本、SSE 解析与 usage/tool_calls 累积、HTTP 错误与无 usage 回退、
workload 确定性与工具闭环/长上下文规模、JSON 往返、Markdown/CSV 生成、6 张图落盘、
空数据与缺字段健壮性。
