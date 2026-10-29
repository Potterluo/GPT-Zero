#!/usr/bin/env python3
"""Offline AgentX runner: one target (an OpenAI-compatible LLM service) × N concurrency points.

This is the driver baked into the ``agentx-offline`` image.  It never touches
the network: the dataset Arrow cache and tokenizer are baked into the image
(``/opt/hf``), and ``HF_*_OFFLINE=1`` is set by the entrypoint.

Artifacts are laid out as::

    <out>/<target>/c<CONCURRENCY>/
        profile_export_aiperf.json      # aggregate metrics (official)
        profile_export.jsonl            # per-request records
        profile_export_aiperf_timeslices.*
        server_metrics_export.*
        profile_export_console.txt
        logs/aiperf.log
        run.log                         # our stdout/stderr capture

and ``<out>/report.html`` is regenerated after every point.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

AGENTX = os.environ.get("AGENTX_BIN", "agentx")
DEFAULT_DATASET = "semianalysis_cc_traces_weka_062126"
DEFAULT_TOKENIZER = "deepseek-ai/DeepSeek-V3"


def parse_concurrency(value: str) -> list[int]:
    points = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not points:
        raise argparse.ArgumentTypeError("at least one concurrency point is required")
    return points


def build_command(args: argparse.Namespace, concurrency: int, artifact_dir: Path) -> list[str]:
    cmd = [
        AGENTX,
        "profile",
        "--url",
        args.url,
        "--endpoint-type",
        "chat",
        "--streaming",
        "--model",
        args.model,
        "--tokenizer",
        args.tokenizer,
        "--concurrency",
        str(concurrency),
        "--benchmark-duration",
        str(args.duration),
        "--num-dataset-entries",
        str(args.entries),
        "--stats-interval",
        str(args.stats_interval),
        "--output-artifact-dir",
        str(artifact_dir),
    ]
    if args.dataset_path:
        cmd += [
            "--input-file",
            args.dataset_path,
            "--custom-dataset-type",
            "weka_trace",
        ]
    else:
        cmd += ["--public-dataset", args.dataset]
    # One override is enough: local corpus identity + short duration are both
    # downgraded by the same flag.
    if args.dataset_path or args.duration < 900:
        cmd += ["--unsafe-override"]
    if args.trust_remote_code:
        cmd += ["--tokenizer-trust-remote-code"]
    if getattr(args, "max_context", 0):
        cmd += ["--max-context-length", str(args.max_context)]
    if args.no_realtime:
        cmd += ["--no-realtime-metrics"]
    if args.extra:
        cmd += args.extra
    return cmd


def run_point(args: argparse.Namespace, target: str, concurrency: int) -> dict:
    artifact_dir = Path(args.out) / target / f"c{concurrency}"
    if artifact_dir.exists() and not args.force:
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(args, concurrency, artifact_dir)
    run_log = artifact_dir / "run.log"
    started = time.time()
    print(f"[agentx-offline] c={concurrency} -> {artifact_dir}", flush=True)
    print("[agentx-offline] " + " ".join(cmd), flush=True)
    with run_log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - started
    export = artifact_dir / "profile_export_aiperf.json"
    ok = proc.returncode == 0 and export.exists() and export.stat().st_size > 0
    tail = ""
    if not ok:
        lines = run_log.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-12:])
    return {
        "target": target,
        "concurrency": concurrency,
        "exit_code": proc.returncode,
        "ok": ok,
        "artifact_dir": str(artifact_dir),
        "elapsed_s": round(elapsed, 1),
        "tail": tail,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agentx-offline")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run AgentX against one target")
    run.add_argument("--url", required=True, help="base URL or full /v1/chat/completions URL")
    run.add_argument("--model", required=True, help="model id the server actually sees")
    run.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    run.add_argument("--target", default="", help="label for this target (default: model)")
    run.add_argument("--dataset", default=DEFAULT_DATASET, help="public dataset alias")
    run.add_argument("--dataset-path", default="", help="local WEKA trace dir (marks run non-submission)")
    run.add_argument("--concurrency", type=parse_concurrency, default=[4], help="e.g. 4,8,16")
    run.add_argument("--duration", type=int, default=900, help="profiling seconds per point (official >=900)")
    run.add_argument("--entries", type=int, default=8, help="number of traces (393 needs a big machine)")
    run.add_argument(
        "--max-context",
        type=int,
        default=0,
        help="drop traces whose peak context exceeds this (match the served model's max_model_len)",
    )
    run.add_argument("--stats-interval", type=int, default=30)
    run.add_argument("--out", default="/results", help="results root (mounted volume)")
    run.add_argument("--trust-remote-code", action="store_true")
    run.add_argument("--no-realtime", action="store_true")
    run.add_argument("--force", action="store_true", help="delete an existing point directory")
    run.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra agentx flags after --")

    rep = sub.add_parser("report", help="build report.html from an existing results root")
    rep.add_argument("--root", default="/results")
    rep.add_argument("--out", default="", help="output html path (default <root>/report.html)")

    args = p.parse_args(argv)

    if args.command == "report":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from report import build_report

        path = build_report(Path(args.root), Path(args.out) if args.out else None)
        print(f"[agentx-offline] report: {path}")
        return 0

    target = args.target or args.model
    results = [run_point(args, target, c) for c in args.concurrency]

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from report import build_report

    report_path = build_report(Path(args.out), None)

    summary = {
        "target": target,
        "url": args.url,
        "model": args.model,
        "tokenizer": args.tokenizer,
        "dataset": args.dataset_path or args.dataset,
        "entries": args.entries,
        "duration_s": args.duration,
        "points": results,
        "report": str(report_path),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (Path(args.out) / target / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[agentx-offline] report: {report_path}")
    for r in results:
        print(f"  c={r['concurrency']:<3} ok={r['ok']} exit={r['exit_code']} {r['elapsed_s']}s")
        if not r["ok"] and r["tail"]:
            print("  --- tail of run.log ---")
            print("\n".join("  " + line for line in r["tail"].splitlines()))
    return 0 if all(r["ok"] for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
