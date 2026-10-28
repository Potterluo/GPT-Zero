"""报告生成：Markdown / CSV / matplotlib 图表（PNG + SVG）。

只依赖结果 JSON（自包含），不访问网络，也不依赖中文字体：图表标签全英文，
Markdown 正文中文。matplotlib 使用 Agg 后端，可在无显示环境/容器内运行。
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Iterable, Sequence

from .util import fmt_float, fmt_pct, load_json, safe_div, write_json, write_text


# ---------------------------------------------------------------------------
# 数据访问
# ---------------------------------------------------------------------------


def _points(data: dict[str, Any]) -> list[dict[str, Any]]:
    points = data.get("points")
    return points if isinstance(points, list) else []


def _config(data: dict[str, Any]) -> dict[str, Any]:
    return data.get("config") or {}


def profiles_in_order(data: dict[str, Any]) -> list[str]:
    order: list[str] = []
    for p in _config(data).get("profiles") or []:
        if p not in order:
            order.append(p)
    for point in _points(data):
        name = str(point.get("profile") or "")
        if name and name not in order:
            order.append(name)
    return order or ["bench"]


def primary_profile(data: dict[str, Any]) -> str:
    cfg = _config(data)
    if cfg.get("profile"):
        return str(cfg["profile"])
    return profiles_in_order(data)[0]


def points_for_profile(data: dict[str, Any], profile: str) -> list[dict[str, Any]]:
    return [p for p in _points(data) if str(p.get("profile") or "") == profile]


def _summary(point: dict[str, Any], name: str) -> dict[str, Any]:
    return (point.get("summary") or {}).get(name) or {}


def _val(point: dict[str, Any], *path: str) -> Any:
    cur: Any = point
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _ordered(points: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(points, key=lambda p: int(p.get("concurrency") or 0))


def _server_config(data: dict[str, Any]) -> dict[str, Any]:
    return (data.get("server") or {}).get("config") or (_points(data)[0].get("server_config") if _points(data) else {}) or {}


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

_HEADER = [
    "并发",
    "请求数",
    "成功率",
    "TTFT p50 (ms)",
    "TTFT p90 (ms)",
    "TTFT p99 (ms)",
    "TPOT p50 (ms)",
    "ITL p50 (ms)",
    "E2E p50 (ms)",
    "E2E p90 (ms)",
    "E2E p99 (ms)",
    "RPS",
    "输入 tok/s",
    "输出 tok/s",
    "Cache Hit",
]


def _row(point: dict[str, Any]) -> list[str]:
    requests = point.get("requests") or {}
    summary = point.get("summary") or {}
    rates = point.get("rates") or {}
    total = int(requests.get("total") or 0)
    success = int(requests.get("success") or 0)
    success_rate = safe_div(success, total)
    cache = (point.get("cache") or {}).get("hit_rate")

    def cell(group: str, key: str, digits: int = 1) -> str:
        return fmt_float(((summary.get(group) or {}).get(key)), digits)

    return [
        str(point.get("concurrency", "-")) + (" ⚠" if (point.get("sanity") or {}).get("warnings") else ""),
        str(total),
        fmt_pct(success_rate),
        cell("ttft_ms", "p50"),
        cell("ttft_ms", "p90"),
        cell("ttft_ms", "p99"),
        cell("tpot_ms", "p50"),
        cell("itl_ms", "p50"),
        cell("e2e_ms", "p50"),
        cell("e2e_ms", "p90"),
        cell("e2e_ms", "p99"),
        fmt_float(rates.get("rps"), 2),
        fmt_float(rates.get("input_tokens_per_s"), 1),
        fmt_float(rates.get("output_tokens_per_s"), 1),
        fmt_pct(cache),
    ]


def _md_table(points: Sequence[dict[str, Any]]) -> str:
    lines = ["| " + " | ".join(_HEADER) + " |", "|" + "|".join(["---"] * len(_HEADER)) + "|"]
    for point in _ordered(points):
        lines.append("| " + " | ".join(_row(point)) + " |")
    return "\n".join(lines)


def _conclusion(data: dict[str, Any]) -> list[str]:
    """根据实测点生成结论段落（不臆断，只描述趋势与饱和信号）。"""
    bullets: list[str] = []
    primary = primary_profile(data)
    points = _ordered(points_for_profile(data, primary))
    if not points:
        return ["- 无有效测量点。"]
    first, last = points[0], points[-1]
    cfg = _config(data)
    server_cfg = _server_config(data)

    first_rps = _val(first, "rates", "rps")
    last_rps = _val(last, "rates", "rps")
    first_ttft = _val(first, "summary", "ttft_ms", "p90")
    last_ttft = _val(last, "summary", "ttft_ms", "p90")
    first_out = _val(first, "rates", "output_tokens_per_s")
    last_out = _val(last, "rates", "output_tokens_per_s")
    failure = _val(last, "requests", "failure_rate")
    cache_rates = [
        p.get("cache", {}).get("hit_rate")
        for p in points
        if p.get("cache", {}).get("hit_rate") is not None
    ]

    if first_rps and last_rps:
        gain = safe_div(last_rps - first_rps, first_rps)
        text = (
            f"- 并发 {first.get('concurrency')} → {last.get('concurrency')}（profile `{primary}`）："
            f"RPS 由 {first_rps:.2f} 提升到 {last_rps:.2f}"
        )
        if gain is not None:
            text += f"（{gain * 100:+.0f}%）"
        text += "。"
        bullets.append(text)
    if first_ttft is not None and last_ttft is not None:
        ratio = safe_div(last_ttft, first_ttft)
        trend = "上升" if (ratio or 1.0) > 1.05 else ("下降" if (ratio or 1.0) < 0.95 else "基本持平")
        text = f"- TTFT p90 由 {first_ttft:.0f} ms 变化到 {last_ttft:.0f} ms"
        if ratio is not None:
            text += f"（{ratio:.2f}×，{trend}）"
        text += "；TTFT 中包含服务端排队时间，服务端 max_concurrency="
        text += str(server_cfg.get("max_concurrency", "?"))
        text += "。"
        bullets.append(text)
    if first_out and last_out:
        bullets.append(
            f"- 输出吞吐由 {first_out:.1f} tok/s 变化到 {last_out:.1f} tok/s，"
            f"输入吞吐由 {fmt_float(_val(first, 'rates', 'input_tokens_per_s'), 1)} "
            f"到 {fmt_float(_val(last, 'rates', 'input_tokens_per_s'), 1)} tok/s。"
        )
    if cache_rates:
        bullets.append(
            f"- 服务端 token 级 prefix cache 命中率区间为 {min(cache_rates) * 100:.1f}% ~ "
            f"{max(cache_rates) * 100:.1f}%（按并发点）。"
        )
    if failure:
        bullets.append(f"- 最高并发点失败率 {failure * 100:.2f}%；失败请求不计入延迟分位数。")
    else:
        bullets.append("- 所有测量点失败率为 0。")

    max_c = server_cfg.get("max_concurrency")
    last_c = last.get("concurrency")
    try:
        if max_c and last_c and int(last_c) > int(max_c):
            bullets.append(
                f"- 服务端并发槽位为 {max_c}，最后测量点并发 {last_c} 已超出槽位；"
                "此时 TTFT 中相当一部分是排队等待，而不是 mock 的计算时间。"
            )
    except (TypeError, ValueError):
        pass

    # 饱和判断：最后一个并发档相对前一档吞吐提升 < 10% 且 TTFT p90 明显上升。
    if len(points) >= 2:
        prev, latest = points[-2], points[-1]
        prev_rps, cur_rps = _val(prev, "rates", "rps"), _val(latest, "rates", "rps")
        prev_ttft, cur_ttft = _val(prev, "summary", "ttft_ms", "p90"), _val(latest, "summary", "ttft_ms", "p90")
        if prev_rps and cur_rps and prev_ttft and cur_ttft:
            throughput_gain = (cur_rps - prev_rps) / prev_rps
            ttft_gain = (cur_ttft - prev_ttft) / prev_ttft
            if throughput_gain < 0.10 and ttft_gain > 0.15:
                bullets.append(
                    f"- **饱和信号**：最后两级并发 RPS 仅提升 {throughput_gain * 100:.1f}%，"
                    f"而 TTFT p90 上升 {ttft_gain * 100:.1f}%——继续加并发只增加排队，不增加吞吐。"
                )
            elif throughput_gain >= 0.30:
                bullets.append(
                    f"- 未观察到饱和：最后两级并发 RPS 仍提升 {throughput_gain * 100:.1f}%，"
                    f"TTFT p90 变化 {ttft_gain * 100:+.1f}%；本次并发范围内 mock 的 CPU 尚未成为瓶颈，"
                    "继续加并发前建议先观察更高并发的 TTFT 曲线。"
                )
    return bullets


def render_markdown(data: dict[str, Any]) -> str:
    cfg = _config(data)
    server_cfg = _server_config(data)
    name = data.get("name") or "benchmark"
    profiles = profiles_in_order(data)
    lines: list[str] = []
    lines.append(f"# GPT-Zero 压测报告 — {name}")
    lines.append("")
    lines.append(f"- 生成时间：{data.get('generated_at', '-')}")
    lines.append(f"- 服务地址：`{cfg.get('url', '-')}`")
    lines.append(f"- 模型：`{cfg.get('model', '-')}`")
    lines.append(f"- 工具：{data.get('tool', {}).get('name', 'gptzero-bench')} {data.get('tool', {}).get('version', '')}")
    lines.append("")
    lines.append("## 1. 服务端配置（模拟硬件参数）")
    lines.append("")
    lines.append("| 参数 | 值 | 含义 |")
    lines.append("|---|---|---|")
    server_rows = [
        ("ttft_ms", "基础首 token 延迟（不含排队/预填充放大）"),
        ("tpot_ms", "每个输出 token 的解码时间"),
        ("max_concurrency", "服务端并发槽位；超出后请求排队，TTFT 抬升"),
        ("block_size", "prefix cache 块大小（token）"),
        ("cache_blocks", "prefix cache 容量（块）"),
    ]
    for key, desc in server_rows:
        lines.append(f"| `{key}` | {server_cfg.get(key, '-')} | {desc} |")
    tokenizer = (data.get("server") or {}).get("tokenizer")
    if tokenizer:
        lines.append(f"| `tokenizer` | {tokenizer} | 服务端 token 计数实现 |")
    lines.append("")
    verified = (data.get("server") or {}).get("config_verified")
    lines.append(
        f"配置一致性：{'✅ 所有测量点的 `/api/config` 与首点一致' if verified else '⚠ 存在不一致或未记录校验信息'}"
    )
    lines.append("")
    probe_rows = [p for p in _ordered(_points(data)) if p.get("server_config_full")]
    if probe_rows:
        lines.append("### 按点 `/api/config` 校验与计时自检")
        lines.append("")
        lines.append(
            "| 并发 | profile | ttft_ms | tpot_ms | max_concurrency | 配置一致 | 预热探针 RTT p50 (ms) | 实测 TPOT p50 (ms) | 状态 |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        baseline = probe_rows[0].get("server_config_full") or {}
        for point in probe_rows:
            full = point.get("server_config_full") or {}
            diff = []
            for key in ("ttft_ms", "tpot_ms", "max_concurrency", "cache_blocks", "block_size"):
                if key not in baseline or key not in full:
                    continue
                try:
                    same = abs(float(baseline[key]) - float(full[key])) < 1e-9
                except (TypeError, ValueError):
                    same = baseline[key] == full[key]
                if not same:
                    diff.append(key)
            sane = point.get("sanity") or {}
            unverified = bool(point.get("config_unverified"))
            if unverified:
                diff_text = "未校验"
            elif diff:
                diff_text = "否（" + ", ".join(diff) + "）"
            else:
                diff_text = "是"
            status = "OK" if not unverified and not diff and not sane.get("warnings") else "⚠ 见下"
            lines.append(
                f"| {point.get('concurrency')} | {point.get('profile')} | {full.get('ttft_ms', '-')} | "
                f"{full.get('tpot_ms', '-')} | {full.get('max_concurrency', '-')} | "
                f"{diff_text} | "
                f"{fmt_float((point.get('warmup_probe') or {}).get('e2e_ms_p50'), 0)} | "
                f"{fmt_float(sane.get('measured_tpot_p50_ms'), 2)} | {status} |"
            )
        lines.append("")
    lines.append("## 2. 压测配置")
    lines.append("")
    lines.append("| 参数 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| profiles | {', '.join(profiles)} |")
    lines.append(f"| concurrency | {', '.join(str(c) for c in cfg.get('concurrency') or [])} |")
    lines.append(f"| duration_s | {cfg.get('duration_s')} |")
    lines.append(f"| warmup | {cfg.get('warmup')} |")
    lines.append(f"| seed | {cfg.get('seed')} |")
    max_tokens = cfg.get("max_tokens")
    lines.append(f"| max_tokens | {max_tokens if max_tokens is not None else '按 profile 默认'} |")
    lines.append("")
    for profile in profiles:
        points = points_for_profile(data, profile)
        if not points:
            continue
        lines.append(f"## 3.{profiles.index(profile) + 1} profile `{profile}`")
        lines.append("")
        lines.append(_md_table(points))
        lines.append("")
        per = _val(points[0], "per_profile", profile)
        if per:
            total_requests = sum(
                int(((p.get("per_profile") or {}).get(profile) or {}).get("requests", {}).get("total") or 0)
                for p in points
            )
            sub = sum(int(_val(p, "requests", "sub_requests") or 0) for p in points)
            note = ""
            if profile == "agentic":
                note = "（agentic 含 20% 概率的并行 subagent 子请求）"
            lines.append(
                f"该 profile 共 {total_requests} 条请求"
                + (f"，其中子请求 {sub} 条" if sub else "")
                + note
                + "；表中每行为一个独立测量点。"
            )
            lines.append("")

    lines.append("## 4. 图表")
    lines.append("")
    chart_titles = {
        "ttft_vs_concurrency.png": "TTFT p50/p90/p99 vs 并发",
        "throughput_vs_concurrency.png": "吞吐（req/s 与 out tok/s）vs 并发",
        "cache_hit_vs_concurrency.png": "prefix cache 命中率 vs 并发",
        "latency_cdf.png": "E2E 延迟 CDF（每个并发一条）",
        "pareto.png": "Pareto：输出吞吐 vs p90 TTFT",
        "profile_comparison.png": "profile 对比（最大并发点）",
    }
    charts = data.get("charts") or []
    if charts:
        for path in charts:
            fname = Path(path).name
            lines.append(f"![{chart_titles.get(fname, fname)}]({path})")
            lines.append("")
    else:
        lines.append("_本次未生成图表（`--no-charts`）。_")
        lines.append("")

    lines.append("## 5. 结论")
    lines.append("")
    lines.extend(_conclusion(data))
    lines.append("")
    warnings = (data.get("sanity") or {}).get("warnings") or []
    if warnings:
        lines.append("### 5.1 数据自检与提示（异常点已标注）")
        lines.append("")
        for item in warnings:
            lines.append(
                f"- ⚠ 并发 {item.get('concurrency')} / `{item.get('profile')}`：{item.get('warning')}"
            )
        lines.append("")
    else:
        lines.append("_数据自检：所有测量点配置一致、TPOT 与配置相符、无异常样本数。_")
        lines.append("")
    if data.get("notes"):
        lines.append("## 6. 备注")
        lines.append("")
        for note in data["notes"]:
            lines.append(f"- {note}")
        lines.append("")
    lines.append("## 7. 复现命令")
    lines.append("")
    lines.append("```bash")
    profiles_arg = ",".join(profiles)
    conc_arg = ",".join(str(c) for c in cfg.get("concurrency") or [1])
    lines.append(
        f"python -m bench sweep --url {cfg.get('url', 'http://127.0.0.1:8090')} "
        f"--profiles {profiles_arg} --concurrency {conc_arg} "
        f"--duration {cfg.get('duration_s', 20)} --name {name}"
    )
    lines.append(f"python -m bench report {name}.json")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "profile",
    "concurrency",
    "requests",
    "success",
    "failed",
    "failure_rate",
    "rps",
    "ttft_p50_ms",
    "ttft_p90_ms",
    "ttft_p95_ms",
    "ttft_p99_ms",
    "tpot_p50_ms",
    "itl_p50_ms",
    "e2e_p50_ms",
    "e2e_p90_ms",
    "e2e_p99_ms",
    "input_tokens_per_s",
    "output_tokens_per_s",
    "prompt_tokens",
    "completion_tokens",
    "cache_hit_rate",
    "cache_queries_tokens",
    "server_ttft_p90_ms",
    "server_e2e_p99_ms",
]


def _csv_row(point: dict[str, Any]) -> dict[str, Any]:
    requests = point.get("requests") or {}
    summary = point.get("summary") or {}
    rates = point.get("rates") or {}
    server_stats = point.get("server_stats") or {}

    def s(group: str, key: str) -> Any:
        value = (summary.get(group) or {}).get(key)
        return round(value, 3) if isinstance(value, (int, float)) else ""

    total = int(requests.get("total") or 0)
    failed = int(requests.get("failed") or 0)
    return {
        "profile": point.get("profile", ""),
        "concurrency": point.get("concurrency", ""),
        "requests": total,
        "success": int(requests.get("success") or 0),
        "failed": failed,
        "failure_rate": round(failed / total, 4) if total else "",
        "rps": round(rates["rps"], 3) if rates.get("rps") is not None else "",
        "ttft_p50_ms": s("ttft_ms", "p50"),
        "ttft_p90_ms": s("ttft_ms", "p90"),
        "ttft_p95_ms": s("ttft_ms", "p95"),
        "ttft_p99_ms": s("ttft_ms", "p99"),
        "tpot_p50_ms": s("tpot_ms", "p50"),
        "itl_p50_ms": s("itl_ms", "p50"),
        "e2e_p50_ms": s("e2e_ms", "p50"),
        "e2e_p90_ms": s("e2e_ms", "p90"),
        "e2e_p99_ms": s("e2e_ms", "p99"),
        "input_tokens_per_s": round(rates["input_tokens_per_s"], 2) if rates.get("input_tokens_per_s") is not None else "",
        "output_tokens_per_s": round(rates["output_tokens_per_s"], 2) if rates.get("output_tokens_per_s") is not None else "",
        "prompt_tokens": (point.get("tokens") or {}).get("prompt", ""),
        "completion_tokens": (point.get("tokens") or {}).get("completion", ""),
        "cache_hit_rate": round(point["cache"]["hit_rate"], 4) if (point.get("cache") or {}).get("hit_rate") is not None else "",
        "cache_queries_tokens": (point.get("cache") or {}).get("queries_tokens", ""),
        "server_ttft_p90_ms": (server_stats.get("ttft_ms") or {}).get("p90", ""),
        "server_e2e_p99_ms": (server_stats.get("e2e_ms") or {}).get("p99", ""),
    }


def render_csv(data: dict[str, Any]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=_CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for point in _ordered(_points(data)):
        writer.writerow(_csv_row(point))
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )
    return plt


def _save(fig: Any, charts_dir: Path, stem: str) -> list[Path]:
    charts_dir.mkdir(parents=True, exist_ok=True)
    png = charts_dir / f"{stem}.png"
    svg = charts_dir / f"{stem}.svg"
    fig.savefig(png)
    fig.savefig(svg)
    return [png, svg]


def _xy(points: Sequence[dict[str, Any]], getter: Any) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for point in _ordered(points):
        value = getter(point)
        if value is None:
            continue
        xs.append(int(point.get("concurrency") or 0))
        ys.append(float(value))
    return xs, ys


def _concurrency_ticks(ax: Any, points: Sequence[dict[str, Any]]) -> None:
    ticks = [int(p.get("concurrency") or 0) for p in _ordered(points)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])


def chart_ttft(points: Sequence[dict[str, Any]], charts_dir: Path, title: str) -> list[Path]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    styles = {"p50": ("o", "-"), "p90": ("s", "-"), "p99": ("^", "--")}
    for key, (marker, line) in styles.items():
        xs, ys = _xy(points, lambda p, k=key: (p.get("summary", {}).get("ttft_ms") or {}).get(k))
        if xs:
            ax.plot(xs, ys, marker=marker, linestyle=line, label=f"TTFT {key}")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title(f"TTFT vs concurrency — {title}")
    ax.set_xscale("log", base=2)
    _concurrency_ticks(ax, points)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    paths = _save(fig, charts_dir, "ttft_vs_concurrency")
    plt.close(fig)
    return paths


def chart_throughput(points: Sequence[dict[str, Any]], charts_dir: Path, title: str) -> list[Path]:
    plt = _plt()
    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    xs, rps = _xy(points, lambda p: (p.get("rates") or {}).get("rps"))
    ax1.plot(xs, rps, marker="o", color="#1f77b4", label="requests/s")
    ax1.set_xlabel("Concurrency")
    ax1.set_ylabel("Requests / s", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    xs2, out = _xy(points, lambda p: (p.get("rates") or {}).get("output_tokens_per_s"))
    ax2.plot(xs2, out, marker="s", color="#d62728", label="output tokens/s")
    ax2.set_ylabel("Output tokens / s", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_xscale("log", base=2)
    _concurrency_ticks(ax1, points)
    ax1.set_title(f"Throughput vs concurrency — {title}")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    paths = _save(fig, charts_dir, "throughput_vs_concurrency")
    plt.close(fig)
    return paths


def chart_cache(points: Sequence[dict[str, Any]], charts_dir: Path, title: str) -> list[Path]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    xs, ys = _xy(points, lambda p: (p.get("cache") or {}).get("hit_rate"))
    ys = [y * 100.0 for y in ys]
    ax.plot(xs, ys, marker="o", color="#2ca02c")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Prefix cache hit rate (%)")
    ax.set_ylim(0, 105)
    ax.set_xscale("log", base=2)
    _concurrency_ticks(ax, points)
    ax.set_title(f"Prefix cache hit rate vs concurrency — {title}")
    paths = _save(fig, charts_dir, "cache_hit_vs_concurrency")
    plt.close(fig)
    return paths


def chart_cdf(points: Sequence[dict[str, Any]], charts_dir: Path, title: str) -> list[Path]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    plotted = 0
    for point in _ordered(points):
        samples = (point.get("samples") or {}).get("e2e_ms") or []
        if len(samples) < 2:
            continue
        xs = sorted(float(v) for v in samples)
        ys = [(i + 1) / len(xs) for i in range(len(xs))]
        ax.plot(xs, ys, label=f"c={point.get('concurrency')}")
        plotted += 1
    if plotted == 0:
        ax.text(0.5, 0.5, "no samples", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.set_xscale("log")
    ax.set_xlabel("End-to-end latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"E2E latency CDF — {title}")
    if plotted:
        ax.legend(loc="lower right")
    paths = _save(fig, charts_dir, "latency_cdf")
    plt.close(fig)
    return paths


def chart_pareto(points: Sequence[dict[str, Any]], charts_dir: Path, title: str) -> list[Path]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for point in _ordered(points):
        x = (point.get("summary", {}).get("ttft_ms") or {}).get("p90")
        y = (point.get("rates") or {}).get("output_tokens_per_s")
        if x is None or y is None:
            continue
        ax.scatter([x], [y], s=48)
        ax.annotate(
            f"c={point.get('concurrency')}",
            (x, y),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )
    ax.set_xlabel("p90 TTFT (ms)")
    ax.set_ylabel("Output tokens / s")
    ax.set_title(f"Pareto: throughput vs p90 TTFT — {title}")
    paths = _save(fig, charts_dir, "pareto")
    plt.close(fig)
    return paths


def chart_profiles(data: dict[str, Any], charts_dir: Path, title: str) -> list[Path]:
    plt = _plt()
    profiles = profiles_in_order(data)
    ttft: list[float] = []
    out: list[float] = []
    labels: list[str] = []
    for profile in profiles:
        points = _ordered(points_for_profile(data, profile))
        if not points:
            continue
        point = points[-1]  # 最大并发点
        p90 = (point.get("summary", {}).get("ttft_ms") or {}).get("p90")
        rate = (point.get("rates") or {}).get("output_tokens_per_s")
        if p90 is None and rate is None:
            continue
        labels.append(profile)
        ttft.append(float(p90) if p90 is not None else 0.0)
        out.append(float(rate) if rate is not None else 0.0)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    if labels:
        axes[0].bar(labels, ttft, color="#1f77b4")
        axes[0].set_ylabel("p90 TTFT (ms)")
        axes[0].set_title("p90 TTFT (max concurrency)")
        axes[1].bar(labels, out, color="#d62728")
        axes[1].set_ylabel("Output tokens / s")
        axes[1].set_title("Output throughput (max concurrency)")
        for ax in axes:
            ax.tick_params(axis="x", rotation=15)
    else:
        axes[0].text(0.5, 0.5, "no data", ha="center", va="center", transform=axes[0].transAxes)
    fig.suptitle(f"Profile comparison — {title}")
    paths = _save(fig, charts_dir, "profile_comparison")
    plt.close(fig)
    return paths


def build_charts(data: dict[str, Any], charts_dir: Path) -> list[str]:
    """生成全部图表，返回相对于结果 JSON 所在目录的路径字符串。"""
    primary = primary_profile(data)
    points = points_for_profile(data, primary)
    title = str(data.get("name") or "bench")
    if not points:
        points = _points(data)
    if not points:
        return []

    produced: list[Path] = []
    produced += chart_ttft(points, charts_dir, title)
    produced += chart_throughput(points, charts_dir, title)
    produced += chart_cache(points, charts_dir, title)
    produced += chart_cdf(points, charts_dir, title)
    produced += chart_pareto(points, charts_dir, title)
    if len(profiles_in_order(data)) >= 1:
        produced += chart_profiles(data, charts_dir, title)

    json_dir = Path(data.get("_json_dir") or ".")
    rel: list[str] = []
    for path in produced:
        if path.suffix.lower() != ".png":  # Markdown 只嵌入 PNG；SVG 同名同目录备用
            continue
        try:
            rel.append(path.resolve().relative_to(json_dir.resolve()).as_posix())
        except ValueError:
            rel.append(path.as_posix())
    return rel


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def write_outputs(
    data: dict[str, Any],
    json_path: str | Path,
    *,
    charts: bool = True,
    charts_dir: str | Path | None = None,
) -> dict[str, Any]:
    """由结果 JSON（内存 dict）生成 .md / .csv / 图，并回写 charts 列表。"""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path = json_path.with_suffix(".md")
    csv_path = json_path.with_suffix(".csv")
    data = dict(data)
    data["_json_dir"] = str(json_path.parent)

    if charts:
        target = Path(charts_dir) if charts_dir else json_path.parent / "charts"
        data["charts"] = build_charts(data, target)
    else:
        data["charts"] = []

    write_text(md_path, render_markdown(data))
    write_text(csv_path, render_csv(data))
    payload = {k: v for k, v in data.items() if k != "_json_dir"}
    write_json(json_path, payload)
    return {
        "json": str(json_path),
        "md": str(md_path),
        "csv": str(csv_path),
        "charts": data["charts"],
    }


def write_json_only(data: dict[str, Any], json_path: str | Path) -> Path:
    payload = {k: v for k, v in data.items() if k != "_json_dir"}
    return write_json(json_path, payload)


def report_from_file(
    json_path: str | Path,
    *,
    charts: bool = True,
    charts_dir: str | Path | None = None,
) -> dict[str, Any]:
    data = load_json(json_path)
    return write_outputs(data, json_path, charts=charts, charts_dir=charts_dir)
