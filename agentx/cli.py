"""``python -m agentx generate|replay|report|validate`` 命令行入口。

命令与 AgentX 官方 harness 的对应关系见 ``docs/AGENTX.md`` §10/§11。

示例::

    python -m agentx generate --sessions 8 --requests 30 --subagents 0.4 \
        --max-isl 32000 --out agentx/traces
    python -m agentx validate --traces agentx/traces
    python -m agentx replay --traces agentx/traces --url http://127.0.0.1:8090 \
        --concurrency 1,2,4,8 --duration 60 --out results/agentx
    python -m agentx report --json results/agentx/replay.json --out results/agentx
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from . import replay as rp
from . import trace_gen as tg

DEFAULT_URL = "http://127.0.0.1:8090"
DEFAULT_TRACES = Path("agentx/traces")
DEFAULT_OUT = Path("results/agentx")

#: 与官方口径的已知差异（会写进报告 meta.notes）
DEV_NOTES = [
    "本地 mock 无真实模型：token 内容与质量无意义，只保证 token 计数/流式时序/缓存语义。",
    "benchmark 时长按分钟级跑（官方下限 900s、默认 3600s），曲线形状可看，绝对值不可与官网对比。",
    "语料为合成 trace（形状对齐 393 条语料：median in≈142k / out≈444 / 98% cached，规模小 2~3 个数量级）。",
    "tok/s/chip 需 --chips 显式给芯片数，否则留空。",
    "不涉及 speculative acceptance / GPU 遥测 / 厂商 chat template。",
]


def _parse_ints(text: str) -> list[int]:
    out = []
    for part in str(text).replace(" ", "").split(","):
        if part:
            out.append(int(part))
    return out


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #


def cmd_generate(args: argparse.Namespace) -> int:
    cfg = tg.GenConfig(
        sessions=args.sessions,
        requests=args.requests,
        subagent_ratio=args.subagents,
        max_isl=args.max_isl,
        block_size=args.block_size,
        cached_ratio=args.cached_ratio,
        out_median=args.out_median,
        seed=args.seed,
        model=args.model,
    )
    out = Path(args.out)
    paths = tg.generate(cfg, out)
    traces = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    errors, warns = tg.validate_traces(traces)

    total_req = 0
    total_sub = 0
    total_in = 0
    total_out = 0
    for t in traces:
        for r in t["requests"]:
            if r["type"] == "subagent":
                total_sub += 1
                total_req += len(r["requests"])
                total_in += sum(c.get("in", 0) for c in r["requests"])
                total_out += sum(c.get("out", 0) for c in r["requests"])
            else:
                total_req += 1
                total_in += r.get("in", 0)
                total_out += r.get("out", 0)
    n_calls = total_req
    cached_blocks = sum(
        len(r.get("hash_ids") or [])
        for t in traces
        for r in t["requests"]
        if r["type"] in ("n", "s")
    ) * cfg.block_size
    print(f"[generate] {len(paths)} 条 trace -> {out}")
    print(
        f"  请求 {n_calls}（含子代理 {total_sub} 个）；"
        f"输入 {total_in:,} tok，输出 {total_out:,} tok；"
        f"平均 in={total_in // max(n_calls, 1):,}，out={total_out // max(n_calls, 1)}"
    )
    print(f"  主链缓存前缀 ≈ {cached_blocks:,} tok / 主链输入 {sum(r.get('in', 0) for t in traces for r in t['requests'] if r['type'] in ('n','s')):,} tok")
    if errors:
        print(f"  [ERR] 校验失败 {len(errors)} 条：")
        for e in errors[:5]:
            print(f"    - {e}")
        return 1
    if warns:
        print(f"  [WARN] 校验告警 {len(warns)} 条（前 3 条）：")
        for w in warns[:3]:
            print(f"    - {w}")
    print("  [OK] schema 校验通过（与 agentx-harness 1.0.7 的 WekaTrace 字段对齐）")
    return 0


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #


def cmd_validate(args: argparse.Namespace) -> int:
    traces = tg.load_traces(args.traces)
    errors, warns = tg.validate_traces(traces)
    print(f"[validate] {args.traces}: {len(traces)} 条 trace")
    for t in traces:
        calls = sum(1 for r in t["requests"] if r["type"] in ("n", "s"))
        subs = sum(1 for r in t["requests"] if r["type"] == "subagent")
        print(f"  {t['id']}: 主链 {calls} 轮, 子代理 {subs} 个, block_size={t['block_size']}")
    if errors:
        print(f"[ERR] {len(errors)} 个错误：")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"[OK] 通过（{len(warns)} 条告警）")
    for w in warns[: args.max_warnings]:
        print(f"  - {w}")
    if len(warns) > args.max_warnings:
        print(f"  … 其余 {len(warns) - args.max_warnings} 条省略（--max-warnings 调大）")
    return 0


# --------------------------------------------------------------------------- #
# replay
# --------------------------------------------------------------------------- #


def cmd_replay(args: argparse.Namespace) -> int:
    traces = tg.load_traces(args.traces)
    concurrencies = _parse_ints(args.concurrency)
    out_dir = Path(args.out)

    run = rp.RunConfig(
        url=args.url.rstrip("/"),
        model=args.model,
        concurrency=concurrencies[0],
        duration_s=args.duration,
        warmup=args.warmup,
        primer=not args.no_primer,
        start_ratio_min=args.start_ratio_min,
        start_ratio_max=args.start_ratio_max,
        seed=args.seed,
        system_idle_gap_cap=args.system_idle_gap_cap,
        trace_idle_gap_cap=args.trace_idle_gap_cap,
        timeout_s=args.timeout,
        tokens_per_word=args.tokens_per_word,
        chips=args.chips,
        benchmark_id=args.benchmark_id,
    )

    print(f"[replay] {len(traces)} 条 trace × 并发 {concurrencies} @ {run.url}（{run.model}）")
    print(f"  每点 {run.duration_s}s，warmup {run.warmup}/lane，primer={run.primer}，"
          f"start_ratio=[{run.start_ratio_min},{run.start_ratio_max}]")

    def on_point(point: dict) -> None:
        print(
            "  c={c}: req={n} ok={ok} fail={f} TTFT p50/p90={a}/{b}ms "
            "out_tok/s={t} cache={ch:.1%}".format(
                c=point["concurrency"],
                n=point["requests"],
                ok=point["ok"],
                f=point["failed"],
                a=point["ttft_ms"]["p50"],
                b=point["ttft_ms"]["p90"],
                t=point["output_tokens_per_s"],
                ch=point["cache"]["hit_rate"],
            )
        )

    try:
        report = asyncio.run(
            rp.replay_sweep(
                traces,
                run,
                concurrencies,
                out_dir=out_dir,
                charts=not args.no_charts,
                dump_requests=args.dump_requests,
                notes=DEV_NOTES,
                on_point=on_point,
            )
        )
    except RuntimeError as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        print("  提示：先起 mock 服务 "
              "`.venv\\Scripts\\python.exe server.py --port 8090`", file=sys.stderr)
        return 2

    files = report.get("files", {})
    print(f"[replay] 完成 -> {out_dir}")
    for key in ("json", "md", "csv"):
        if key in files:
            print(f"  {files[key]}")
    for p in files.get("charts", []):
        print(f"  {p}")

    invalid = [r for r in report["runs"] if not r["valid"]]
    if invalid:
        print(f"[WARN] {len(invalid)} 个并发点未达成功率阈值（<90%），结论不可用")
        return 1
    return 0


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def cmd_report(args: argparse.Namespace) -> int:
    src = Path(args.json)
    report = json.loads(src.read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    files = rp.write_report(report, out_dir, charts=not args.no_charts)
    print(f"[report] 由 {src} 重建 -> {out_dir}")
    for key in ("json", "md", "csv"):
        print(f"  {files[key]}")
    for p in files.get("charts", []):
        print(f"  {p}")
    return 0


# --------------------------------------------------------------------------- #
# argparse
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m agentx",
        description="AgentX（SemiAnalysis InferenceX）调研复现：WEKA trace 生成 / 闭环回放 / 报告",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="生成与 AIPerf weka_trace schema 对齐的合成语料")
    g.add_argument("--sessions", type=int, default=8, help="trace 条数（默认 8）")
    g.add_argument("--requests", type=int, default=30, help="每条 trace 主链轮数（默认 30）")
    g.add_argument("--subagents", type=float, default=0.4, help="子代理扇出概率 0~1（默认 0.4）")
    g.add_argument("--max-isl", type=int, default=32_000, help="输入 token 上限（官方 median 142k）")
    g.add_argument("--block-size", type=int, default=tg.BLOCK_SIZE_DEFAULT, help="KV 块大小（默认 64）")
    g.add_argument("--cached-ratio", type=float, default=0.98, help="缓存前缀占比（默认 0.98）")
    g.add_argument("--out-median", type=int, default=444, help="输出 token 中位数（默认 444）")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--model", default="gpt-3.5-turbo")
    g.add_argument("--out", default=str(DEFAULT_TRACES))
    g.set_defaults(func=cmd_generate)

    v = sub.add_parser("validate", help="校验 trace 是否会被 AIPerf weka_trace loader 接受")
    v.add_argument("--traces", default=str(DEFAULT_TRACES))
    v.add_argument("--max-warnings", type=int, default=10)
    v.set_defaults(func=cmd_validate)

    r = sub.add_parser("replay", help="对 mock 服务做闭环并发回放（sweep 多个并发点）")
    r.add_argument("--traces", default=str(DEFAULT_TRACES))
    r.add_argument("--url", default=DEFAULT_URL, help="mock 服务地址（默认 8090，8080 常被占用）")
    r.add_argument("--model", default="gpt-3.5-turbo")
    r.add_argument("--concurrency", default="1,2,4,8", help="逗号分隔的并发点（=lane 数）")
    r.add_argument("--duration", type=float, default=60.0, help="每个并发点测量时长（秒）")
    r.add_argument("--warmup", type=int, default=10, help="每 lane 额外 warmup 条数（官方 10）")
    r.add_argument("--no-primer", action="store_true", help="关闭 t* 快照 primer")
    r.add_argument("--start-ratio-min", type=float, default=0.25, help="轨迹起点下限（官方 0.25）")
    r.add_argument("--start-ratio-max", type=float, default=0.75, help="轨迹起点上限（官方 0.75）")
    r.add_argument("--system-idle-gap-cap", type=float, default=10.0, help="idle gap 上限秒（官方 10.0）")
    r.add_argument("--trace-idle-gap-cap", type=float, default=300.0, help="单 trace gap 上限秒（官方 300）")
    r.add_argument("--timeout", type=float, default=600.0, help="单请求超时（秒）")
    r.add_argument("--tokens-per-word", type=float, default=1.05,
                   help="词表每词 token 数（标定 1.05；调大=同样 in 用更少文字）")
    r.add_argument("--chips", type=int, default=None, help="芯片数（用于 tok/s/chip，官方指标）")
    r.add_argument("--benchmark-id", default="gptzero-agentx", help="cache-bust marker 的 benchmark_id")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--out", default=str(DEFAULT_OUT))
    r.add_argument("--dump-requests", action="store_true", help="在 json 里输出逐请求明细")
    r.add_argument("--no-charts", action="store_true")
    r.set_defaults(func=cmd_replay)

    q = sub.add_parser("report", help="由 replay.json 重建 md/csv/图表")
    q.add_argument("--json", default=str(DEFAULT_OUT / "replay.json"))
    q.add_argument("--out", default=str(DEFAULT_OUT))
    q.add_argument("--no-charts", action="store_true")
    q.set_defaults(func=cmd_report)

    return p


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台默认 GBK：只把编码错误降级为替换，**不**强制 utf-8
    # （强制 utf-8 会让中文在 GBK 终端变乱码）。输出里也刻意只用 ASCII 状态标记
    # （[OK]/[ERR]/[WARN]），避免 GBK 之外的字符把 CI/脚本打崩。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001 - 非 TTY / 被重定向时跳过
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
