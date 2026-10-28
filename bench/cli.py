"""命令行入口：``python -m bench`` / ``gptzero-bench``。

子命令::

    python -m bench run   --url http://127.0.0.1:8090 --profile coding \
                          --concurrency 8 --duration 20 --out results/x.json
    python -m bench sweep --url http://127.0.0.1:8090 --profiles chat,coding,agentic \
                          --concurrency 1,2,4,8,16 --duration 10 --name sweep
    python -m bench report results/x.json
    python -m bench serve-sweep --launch-server --port 8099 --ttft-ms 120 --tpot-ms 8 \
                          --max-concurrency 64 --profiles coding --concurrency 1,2,4,8,16

默认端口是 **8090**（本机 8080 被其他常驻服务占用，见 CONTRACT/文档）。
"""

from __future__ import annotations

import argparse
import asyncio
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__
from .report import write_outputs
from .runner import BenchConfig, build_document, run_point, run_sweep
from .util import REPO_ROOT, RESULTS_DIR, now_iso, write_json

DEFAULT_URL = "http://127.0.0.1:8090"


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------


def _parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    try:
        items = [int(item) for item in _parse_list(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid concurrency list: {value!r}") from exc
    if not items:
        raise argparse.ArgumentTypeError("concurrency list is empty")
    return items


def _add_bench_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--url", default=DEFAULT_URL, help=f"mock 服务地址（默认 {DEFAULT_URL}）")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="模型名")
    parser.add_argument("--duration", type=float, default=20.0, help="每个并发点的测量时长（秒）")
    parser.add_argument("--warmup", type=int, default=3, help="预热请求数（不计入统计）")
    parser.add_argument("--seed", type=int, default=42, help="会话生成种子")
    parser.add_argument("--timeout", type=float, default=300.0, help="单请求读超时（秒）")
    parser.add_argument("--max-tokens", type=int, default=None, help="覆盖 profile 默认输出长度")
    parser.add_argument("--fail-fast", action="store_true", help="首个请求失败立即中止")
    parser.add_argument("--no-charts", action="store_true", help="不生成图表")
    parser.add_argument("--charts-dir", default=None, help="图表目录（默认 <结果目录>/charts）")
    parser.add_argument("--notes", default=None, help="写入结果 JSON 的备注（分号分隔）")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bench",
        description="GPT-Zero mock 推理服务压测 harness（闭环并发 + SSE 精确时延 + 图表）",
    )
    parser.add_argument("--version", action="version", version=f"gptzero-bench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="单并发点压测")
    _add_bench_options(run)
    run.add_argument("--profile", default="coding", help="chat|coding|agentic|longctx")
    run.add_argument("--concurrency", type=int, default=8, help="闭环会话数")
    run.add_argument("--out", default=str(RESULTS_DIR / "run.json"), help="结果 JSON 路径")
    run.add_argument("--report", action="store_true", help="同时生成 md/csv/图")

    sweep = sub.add_parser("sweep", help="多并发/多 profile 扫描")
    _add_bench_options(sweep)
    sweep.add_argument("--profiles", default="coding", help="逗号分隔：chat,coding,agentic,longctx")
    sweep.add_argument("--concurrency", type=_parse_int_list, default=[1, 2, 4, 8, 16], help="如 1,2,4,8,16")
    sweep.add_argument("--name", default="sweep", help="结果文件名（results/<name>.json|md|csv）")
    sweep.add_argument("--out-dir", default=str(RESULTS_DIR), help="结果目录")

    report = sub.add_parser("report", help="由结果 JSON 重新生成 md/csv/图")
    report.add_argument("json", help="结果 JSON 路径")
    report.add_argument("--no-charts", action="store_true", help="不生成图表")
    report.add_argument("--charts-dir", default=None, help="图表目录")

    serve = sub.add_parser("serve-sweep", help="自行启动 mock 再跑 sweep（一条命令复现）")
    _add_bench_options(serve)
    serve.add_argument("--launch-server", action="store_true", help="必须显式声明会启动子进程")
    serve.add_argument("--port", type=int, default=8099, help="mock 监听端口（默认 8099）")
    serve.add_argument("--ttft-ms", type=float, default=120.0, help="服务端基础 TTFT（ms）")
    serve.add_argument("--tpot-ms", type=float, default=8.0, help="服务端 TPOT（ms/token）")
    serve.add_argument("--max-concurrency", type=int, default=64, help="服务端并发槽位")
    serve.add_argument("--cache-blocks", type=int, default=8192, help="prefix cache 块数")
    serve.add_argument("--tokenizer", default="deepseek_v3_tokenizer", help="服务端 tokenizer")
    serve.add_argument("--profiles", default="coding")
    serve.add_argument("--concurrency", type=_parse_int_list, default=[1, 2, 4, 8, 16])
    serve.add_argument("--name", default="serve_sweep")
    serve.add_argument("--out-dir", default=str(RESULTS_DIR))
    return parser


# ---------------------------------------------------------------------------
# 打印
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _print_point(point: dict[str, Any]) -> None:
    summary = point.get("summary") or {}
    rates = point.get("rates") or {}
    requests = point.get("requests") or {}
    cache = point.get("cache") or {}
    total = int(requests.get("total") or 0)
    success = int(requests.get("success") or 0)
    rate = (success / total * 100.0) if total else 0.0
    print(
        f"[c={int(point.get('concurrency') or 0):>3}] {str(point.get('profile')):<8} "
        f"req={total:>5} ok={rate:>6.2f}% "
        f"rps={_fmt(rates.get('rps'), 2):>7} "
        f"ttft_p50={_fmt((summary.get('ttft_ms') or {}).get('p50')):>9} "
        f"p90={_fmt((summary.get('ttft_ms') or {}).get('p90')):>9} "
        f"tpot_p50={_fmt((summary.get('tpot_ms') or {}).get('p50'), 2):>7} "
        f"out_tok/s={_fmt(rates.get('output_tokens_per_s')):>9} "
        f"cache={_fmt((cache.get('hit_rate') or 0) * 100, 1):>6}%",
        flush=True,
    )


def _print_server(point: dict[str, Any]) -> None:
    cfg = point.get("server_config") or {}
    print(
        f"server: ttft_ms={cfg.get('ttft_ms')} tpot_ms={cfg.get('tpot_ms')} "
        f"max_concurrency={cfg.get('max_concurrency')} cache_blocks={cfg.get('cache_blocks')}",
        flush=True,
    )


def _notes(args: argparse.Namespace) -> list[str]:
    if not getattr(args, "notes", None):
        return []
    return [item.strip() for item in str(args.notes).split(";") if item.strip()]


def _config_from_args(args: argparse.Namespace, *, profiles: list[str], concurrency: list[int]) -> BenchConfig:
    return BenchConfig(
        url=str(args.url).rstrip("/"),
        model=args.model,
        profiles=profiles,
        concurrency=concurrency,
        duration=float(args.duration),
        warmup=int(args.warmup),
        seed=int(args.seed),
        timeout=float(args.timeout),
        fail_fast=bool(args.fail_fast),
        max_tokens=args.max_tokens,
        notes=_notes(args),
    )


# ---------------------------------------------------------------------------
# 子命令
# ---------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = _config_from_args(args, profiles=[args.profile], concurrency=[int(args.concurrency)])
    out = Path(args.out)
    print(f"bench run: {cfg.url} profile={args.profile} concurrency={args.concurrency} duration={cfg.duration}s")
    point = asyncio.run(run_point(cfg, int(args.concurrency), args.profile))
    document = build_document(cfg, [point], name=out.stem)
    _print_server(point)
    _print_point(point)
    if args.report:
        paths = write_outputs(document, out, charts=not args.no_charts, charts_dir=args.charts_dir)
        print(f"written: {paths['json']}\n         {paths['md']}\n         {paths['csv']}")
        for chart in paths["charts"]:
            print(f"chart:   {chart}")
    else:
        write_json(out, document)
        print(f"written: {out}")
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    profiles = _parse_list(args.profiles)
    cfg = _config_from_args(args, profiles=profiles, concurrency=list(args.concurrency))
    out = Path(args.out_dir) / f"{args.name}.json"
    print(
        f"bench sweep: {cfg.url} profiles={','.join(profiles)} "
        f"concurrency={','.join(str(c) for c in cfg.concurrency)} duration={cfg.duration}s"
    )
    points = asyncio.run(run_sweep(cfg, progress=_print_point))
    document = build_document(cfg, points, name=args.name)
    if points:
        _print_server(points[0])
    paths = write_outputs(document, out, charts=not args.no_charts, charts_dir=args.charts_dir)
    print(f"written: {paths['json']}\n         {paths['md']}\n         {paths['csv']}")
    for chart in paths["charts"]:
        print(f"chart:   {chart}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from .report import report_from_file

    paths = report_from_file(args.json, charts=not args.no_charts, charts_dir=args.charts_dir)
    print(f"written: {paths['json']}\n         {paths['md']}\n         {paths['csv']}")
    for chart in paths["charts"]:
        print(f"chart:   {chart}")
    return 0


def _port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _wait_health(url: str, timeout_s: float = 90.0) -> bool:
    import httpx

    deadline = time.monotonic() + timeout_s
    with httpx.Client(trust_env=False, timeout=3.0) as client:
        while time.monotonic() < deadline:
            try:
                response = client.get(f"{url}/health")
                if response.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(0.4)
    return False


def _cmd_serve_sweep(args: argparse.Namespace) -> int:
    if not args.launch_server:
        print("error: serve-sweep 会启动子进程，请显式加 --launch-server", file=sys.stderr)
        return 2
    server_py = REPO_ROOT / "server.py"
    if not server_py.exists():
        print(f"error: 找不到 {server_py}", file=sys.stderr)
        return 2
    if _port_in_use(args.port):
        print(f"error: 端口 {args.port} 已被占用，请换 --port", file=sys.stderr)
        return 2
    url = f"http://127.0.0.1:{args.port}"
    command = [
        sys.executable,
        str(server_py),
        "--port",
        str(args.port),
        "--tokenizer",
        str(args.tokenizer),
        "--ttft-ms",
        str(args.ttft_ms),
        "--tpot-ms",
        str(args.tpot_ms),
        "--max-concurrency",
        str(args.max_concurrency),
        "--cache-blocks",
        str(args.cache_blocks),
    ]
    print("launching:", " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not _wait_health(url):
            print("error: mock 未在 90s 内就绪（查看 server.py 输出）", file=sys.stderr)
            return 1
        profiles = _parse_list(args.profiles)
        cfg = _config_from_args(args, profiles=profiles, concurrency=list(args.concurrency))
        cfg.url = url
        out = Path(args.out_dir) / f"{args.name}.json"
        print(
            f"bench serve-sweep: {url} profiles={','.join(profiles)} "
            f"concurrency={','.join(str(c) for c in cfg.concurrency)} duration={cfg.duration}s"
        )
        points = asyncio.run(run_sweep(cfg, progress=_print_point))
        document = build_document(cfg, points, name=args.name)
        if points:
            _print_server(points[0])
        document["server"]["launched"] = {
            "command": command,
            "ttft_ms": args.ttft_ms,
            "tpot_ms": args.tpot_ms,
            "max_concurrency": args.max_concurrency,
            "cache_blocks": args.cache_blocks,
            "tokenizer": args.tokenizer,
            "started_at": now_iso(),
        }
        document["notes"].append(
            f"serve-sweep 自启 mock：ttft={args.ttft_ms}ms tpot={args.tpot_ms}ms "
            f"max_concurrency={args.max_concurrency} cache_blocks={args.cache_blocks}"
        )
        paths = write_outputs(document, out, charts=not args.no_charts, charts_dir=args.charts_dir)
        print(f"written: {paths['json']}\n         {paths['md']}\n         {paths['csv']}")
        for chart in paths["charts"]:
            print(f"chart:   {chart}")
        return 0
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "sweep":
            return _cmd_sweep(args)
        if args.command == "report":
            return _cmd_report(args)
        if args.command == "serve-sweep":
            return _cmd_serve_sweep(args)
        parser.error(f"unknown command {args.command}")  # pragma: no cover
    except KeyboardInterrupt:  # pragma: no cover - 交互中断
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # 压测/网络错误：给出简洁错误而非堆栈
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
