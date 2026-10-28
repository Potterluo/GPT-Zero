"""Collect real AgentX (agentx-harness) run artifacts into a table + charts.

Each run directory is produced by:

    agentx profile ... --output-artifact-dir experiments/agentx_real/run_c<N>_<secs>

and contains ``profile_export_aiperf.json`` (aggregate metrics),
``profile_export_console.txt`` (human summary incl. prefix-cache hit) and
``server_metrics_export.json`` (Prometheus snapshots from the mock).

This script never invents numbers: every cell comes from those files.  Outputs
land in ``results/agentx_real/``.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments" / "agentx_real"
OUT = ROOT / "results" / "agentx_real"
CHARTS = OUT / "charts"


def _pct(block: dict | None, key: str) -> float | None:
    if not block:
        return None
    value = block.get(key)
    return round(float(value), 3) if value is not None else None


def _console_value(text: str, pattern: str, last: bool = False) -> float | None:
    matches = re.findall(pattern, text)
    if not matches:
        return None
    value = matches[-1] if last else matches[0]
    return round(float(value), 2) if value else None


def load_run(path: Path) -> dict | None:
    summary_file = path / "profile_export_aiperf.json"
    if not summary_file.exists():
        return None
    data = json.loads(summary_file.read_text(encoding="utf-8"))
    m = re.match(r"run_c(\d+)_(\d+)", path.name)
    concurrency = int(m.group(1)) if m else None
    duration = int(m.group(2)) if m else None

    console = ""
    console_file = path / "profile_export_console.txt"
    if console_file.exists():
        console = console_file.read_text(encoding="utf-8", errors="replace")

    log_file = path / "logs" / "aiperf.log"
    log_text = log_file.read_text(encoding="utf-8", errors="replace") if log_file.exists() else ""

    # Console renders a table row; the realtime log line carries the Prometheus
    # fallback measurement ("srv prefix_cache_hit=93.4%").
    cache_theoretical = _console_value(
        console, r"Theoretical Prefix Cache Hit \(%\)\s*│\s*([\d.]+)"
    )
    cache_server = _console_value(log_text, r"srv\s+prefix_cache_hit=([\d.]+)%", last=True)

    run = {
        "run": path.name,
        "concurrency": concurrency,
        "duration_s": duration,
        "benchmark_id": data.get("benchmark_id"),
        "requests": int((data.get("request_count") or {}).get("avg") or 0),
        "rps": _pct(data.get("request_throughput"), "avg"),
        "ttft_p50": _pct(data.get("time_to_first_token"), "p50"),
        "ttft_p90": _pct(data.get("time_to_first_token"), "p90"),
        "ttft_p99": _pct(data.get("time_to_first_token"), "p99"),
        "itl_p50": _pct(data.get("inter_token_latency"), "p50"),
        "e2e_p50": _pct(data.get("request_latency"), "p50"),
        "e2e_p90": _pct(data.get("request_latency"), "p90"),
        "e2e_p99": _pct(data.get("request_latency"), "p99"),
        "out_tok_s": _pct(data.get("output_token_throughput"), "avg"),
        "in_tok_s": _pct(data.get("input_token_throughput"), "avg"),
        "isl_p50": _pct(data.get("input_sequence_length"), "p50"),
        "osl_p50": _pct(data.get("output_sequence_length"), "p50"),
        "total_in": int((data.get("total_isl") or {}).get("avg") or 0),
        "total_out": int((data.get("total_osl") or {}).get("avg") or 0),
        "cache_hit_theoretical_pct": cache_theoretical,
        "cache_hit_server_pct": cache_server,
        "errors": len(data.get("error_summary") or []),
    }
    return run


def write_outputs(runs: list[dict]) -> None:
    runs = sorted(runs, key=lambda r: (r["concurrency"] or 0, r["duration_s"] or 0))
    OUT.mkdir(parents=True, exist_ok=True)

    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "harness": "agentx-harness==1.0.7 (aiperf)",
                "scenario": "agentx (default injected)",
                "dataset": "semianalysisai/cc-traces-weka-062126 (real corpus, offline cache)",
                "tokenizer": "deepseek-ai/DeepSeek-V3",
                "target": "gptzero mock (offline, OpenAI-compatible)",
                "runs": runs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    header = [
        "并发",
        "时长(s)",
        "请求",
        "RPS",
        "TTFT p50/p90/p99 (ms)",
        "ITL p50 (ms)",
        "E2E p50/p90 (ms)",
        "out tok/s",
        "ISL p50",
        "OSL p50",
        "缓存命中(理论/服务端)",
    ]
    rows = []
    for r in runs:
        rows.append(
            [
                r["concurrency"],
                r["duration_s"],
                r["requests"],
                f"{r['rps']:.2f}" if r["rps"] else "-",
                f"{r['ttft_p50']:.0f} / {r['ttft_p90']:.0f} / {r['ttft_p99']:.0f}"
                if r["ttft_p50"] is not None
                else "-",
                f"{r['itl_p50']:.1f}" if r["itl_p50"] is not None else "-",
                f"{r['e2e_p50']:.0f} / {r['e2e_p90']:.0f}" if r["e2e_p50"] is not None else "-",
                f"{r['out_tok_s']:.0f}" if r["out_tok_s"] else "-",
                f"{r['isl_p50']:.0f}" if r["isl_p50"] else "-",
                f"{r['osl_p50']:.0f}" if r["osl_p50"] else "-",
                f"{r['cache_hit_theoretical_pct']}% / {r['cache_hit_server_pct']}%"
                if r["cache_hit_theoretical_pct"] is not None
                else "-",
            ]
        )

    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)

    lines = [
        "# AgentX 真实 harness 离线跑分（真实语料）",
        "",
        "> 运行者：真正的 `agentx-harness==1.0.7`（不是自研回放）；服务端是 GPT-Zero mock。",
        "> 语料：`semianalysisai/cc-traces-weka-062126`（393 traces，HF 镜像预下载到本地缓存后断网运行）。",
        "> 模型名：`gpt-3.5-turbo`（mock 不加载权重），tokenizer：`deepseek-ai/DeepSeek-V3`。",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    lines += [
        "",
        "## 说明",
        "",
        "- `缓存命中(理论/服务端)`：AIPerf 从 trace 的 64-token 块算出的理论 LCP 命中率 / mock 的",
        "  `vllm:prefix_cache_hits_total ÷ queries_total`（profiling 阶段）。两者有差距说明 mock 的",
        "  `--cache-blocks` 容量不足，发生了淘汰（可调大后重跑）。",
        "- 每点时长被压到 120~180s（官方要求 ≥900s）才在单机可跑；这会导致请求数偏少、分位数抖动，",
        "  但**语料、scenario 锁、warmup、cache-bust、ignore_eos、闭环并发**全部是真实的。",
        "- 这是**性能形状**验证，不是官方 submission：mock 没有真实模型、GPU 遥测与 KV offload。",
        "",
        "## 原始产物",
        "",
    ]
    for r in runs:
        lines.append(f"- `experiments/agentx_real/{r['run']}/`：profile_export*、server_metrics_export*、logs/")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    _charts(runs)


def _timeline_chart(run_name: str) -> None:
    """One completed point: plot the official per-second time slices."""
    import csv as _csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = RUNS / run_name / "profile_export_aiperf_timeslices.csv"
    if not path.exists():
        return
    rows = list(_csv.DictReader(path.open(encoding="utf-8")))
    series: dict[tuple[str, str], dict[int, float]] = {}
    for row in rows:
        if row["Metric"] not in ("Time to First Token", "Output Token Throughput", "Request Throughput"):
            continue
        if row["Stat"] not in ("p50", "p90", "avg"):
            continue
        key = (row["Metric"], row["Stat"])
        series.setdefault(key, {})[int(row["Timeslice"])] = float(row["Value"])
    if not series:
        return

    CHARTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=130)
    for stat, color in (("p50", "#5aa9ff"), ("p90", "#ffb454")):
        data = series.get(("Time to First Token", stat))
        if data:
            xs = sorted(data)
            ax.plot(xs, [data[x] for x in xs], color=color, label=f"TTFT {stat}")
    ax.set_xlabel("profiling second")
    ax.set_ylabel("ms")
    ax.set_title(f"AgentX real corpus — TTFT over time (c=4, {run_name})")
    ax.grid(alpha=0.3)
    ax.legend()
    for fmt in ("png", "svg"):
        fig.savefig(CHARTS / f"timeline_ttft.{fmt}", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=130)
    for metric, color, label in (
        ("Output Token Throughput", "#39d3a3", "output tok/s"),
        ("Request Throughput", "#5aa9ff", "req/s"),
    ):
        data = series.get((metric, "avg"))
        if data:
            xs = sorted(data)
            ax.plot(xs, [data[x] for x in xs], color=color, label=label)
    ax.set_xlabel("profiling second")
    ax.set_ylabel("per second")
    ax.set_title("AgentX real corpus — throughput over time")
    ax.grid(alpha=0.3)
    ax.legend()
    for fmt in ("png", "svg"):
        fig.savefig(CHARTS / f"timeline_throughput.{fmt}", bbox_inches="tight")
    plt.close(fig)


def _charts(runs: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(runs) == 1:
        _timeline_chart(runs[0]["run"])
        return
    CHARTS.mkdir(parents=True, exist_ok=True)
    conc = [r["concurrency"] for r in runs]

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
    for key, label in (("ttft_p50", "p50"), ("ttft_p90", "p90"), ("ttft_p99", "p99")):
        ax.plot(conc, [r[key] for r in runs], marker="o", label=f"TTFT {label}")
    ax.set_xlabel("Concurrency (agent sessions)")
    ax.set_ylabel("ms")
    ax.set_title("AgentX real corpus — TTFT vs concurrency")
    ax.grid(alpha=0.3)
    ax.legend()
    for fmt in ("png", "svg"):
        fig.savefig(CHARTS / f"ttft_vs_concurrency.{fmt}", bbox_inches="tight")
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(7, 4.2), dpi=130)
    ax1.plot(conc, [r["rps"] for r in runs], marker="o", color="#5aa9ff", label="req/s")
    ax1.set_xlabel("Concurrency (agent sessions)")
    ax1.set_ylabel("req/s", color="#5aa9ff")
    ax2 = ax1.twinx()
    ax2.plot(conc, [r["out_tok_s"] for r in runs], marker="s", color="#39d3a3", label="out tok/s")
    ax2.set_ylabel("output tokens/s", color="#39d3a3")
    ax1.set_title("AgentX real corpus — throughput vs concurrency")
    ax1.grid(alpha=0.3)
    for fmt in ("png", "svg"):
        fig.savefig(CHARTS / f"throughput_vs_concurrency.{fmt}", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
    ax.plot(conc, [r["cache_hit_theoretical_pct"] for r in runs], marker="o", label="theoretical (trace LCP)")
    ax.plot(conc, [r["cache_hit_server_pct"] for r in runs], marker="s", label="server measured")
    ax.set_xlabel("Concurrency (agent sessions)")
    ax.set_ylabel("prefix cache hit %")
    ax.set_title("AgentX real corpus — prefix cache reuse")
    ax.grid(alpha=0.3)
    ax.legend()
    for fmt in ("png", "svg"):
        fig.savefig(CHARTS / f"cache_hit_vs_concurrency.{fmt}", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    runs = []
    for path in sorted(RUNS.glob("run_c*")):
        run = load_run(path)
        if run:
            runs.append(run)
    if not runs:
        print("no complete AgentX runs found")
        return 1
    write_outputs(runs)
    print(f"wrote {OUT/'summary.md'} ({len(runs)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
