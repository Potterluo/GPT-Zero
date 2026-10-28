"""Server configuration: dataclass + CLI/env layering.

Precedence (highest first): CLI flag > ``GPTZERO_*`` environment variable >
default.  This mirrors what ClawPerf does for its own CLI so the two tools feel
the same.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

ENV_PREFIX = "GPTZERO_"

# Default canned output text.  The mock never "understands" anything; content is
# generated deterministically from a filler corpus so token counts are exact and
# streaming timing is realistic.
DEFAULT_MODELS = ["gpt-3.5-turbo", "qwen3-32b", "deepseek-v3", "claude-3-opus", "deepseek-r1"]


@dataclass
class ServerConfig:
    # --- network ---
    host: str = "0.0.0.0"
    port: int = 8080
    # --- models ---
    models: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    model_dir: str = "/mnt/model"
    # --- timing model ---
    ttft_ms: float = 180.0
    tpot_ms: float = 8.0
    jitter: float = 0.12
    # token-rate model (used when > 0; overrides ttft/tpot for the prefill part)
    prefill_tps: float = 0.0
    decode_tps: float = 0.0
    max_concurrency: int = 64
    # --- prefix cache ---
    block_size: int = 64
    cache_blocks: int = 8192
    # --- output ---
    default_max_tokens: int = 256
    max_tokens_cap: int = 65536
    tool_call_ratio: float = 0.6
    tool_payload_tokens: int = 120
    seed: int = 42
    # --- tokenizer ---
    tokenizer: str = "auto"  # auto | heuristic | <local path>
    # --- observability ---
    log_limit: int = 1000
    log_file: str = ""
    stats_sample_limit: int = 20000
    # --- runtime ---
    workers: int = 1


def _coerce(name: str, value: Any, annotation: str) -> Any:
    """Best-effort conversion of env strings to the annotated type."""
    if annotation.startswith("list"):
        if isinstance(value, str):
            return [x.strip() for x in value.split(",") if x.strip()]
        return list(value)
    if annotation.startswith("int"):
        return int(value)
    if annotation.startswith("float"):
        return float(value)
    return str(value)


def from_env(cfg: ServerConfig | None = None) -> ServerConfig:
    cfg = cfg or ServerConfig()
    for f in fields(ServerConfig):
        env_name = ENV_PREFIX + f.name.upper()
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        setattr(cfg, f.name, _coerce(f.name, raw, str(f.type)))
    return cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gptzero",
        description="Offline OpenAI-compatible mock LLM server for agentic benchmarks",
    )
    p.add_argument("--host", help="bind address (default 0.0.0.0)")
    p.add_argument("--port", "-p", type=int, help="bind port (default 8080)")
    p.add_argument("--model", action="append", help="served model id (repeatable or comma separated)")
    p.add_argument("--models-file", help="JSON file with a list of model ids")
    p.add_argument("--model-dir", help="directory scanned for local tokenizers/models")
    p.add_argument("--ttft-ms", type=float, help="base time-to-first-token in ms")
    p.add_argument("--tpot-ms", type=float, help="per-output-token decode time in ms")
    p.add_argument("--jitter", type=float, help="relative jitter (0 disables)")
    p.add_argument("--prefill-tps", type=float, help="prefill tokens/s (0 = use ttft-ms only)")
    p.add_argument("--decode-tps", type=float, help="decode tokens/s (0 = use tpot-ms)")
    p.add_argument("--max-concurrency", type=int, help="requests served at once; extra requests queue")
    p.add_argument("--block-size", type=int, help="prefix-cache block size in tokens")
    p.add_argument("--cache-blocks", type=int, help="prefix-cache capacity in blocks")
    p.add_argument("--default-max-tokens", type=int, help="output tokens when max_tokens is omitted")
    p.add_argument("--max-tokens-cap", type=int, help="hard cap for max_tokens")
    p.add_argument("--tool-call-ratio", type=float, help="probability that a tools-bearing turn calls a tool")
    p.add_argument("--tool-payload-tokens", type=int, help="approx tokens of synthesized tool arguments")
    p.add_argument("--seed", type=int, help="deterministic generation seed")
    p.add_argument("--tokenizer", help="auto | heuristic | /path/to/tokenizer_dir")
    p.add_argument("--log-limit", type=int, help="in-memory request log ring size")
    p.add_argument("--log-file", help="append every request as JSONL to this file")
    p.add_argument("--stats-sample-limit", type=int, help="retained latency samples for percentiles")
    p.add_argument("--workers", type=int, help="uvicorn worker count (default 1)")
    return p


def load_config(argv: list[str] | None = None) -> ServerConfig:
    """Layering: defaults -> env -> CLI."""
    cfg = from_env()
    args = build_parser().parse_args(argv)

    if args.model:
        models: list[str] = []
        for item in args.model:
            models.extend([x.strip() for x in item.split(",") if x.strip()])
        cfg.models = models
    if args.models_file:
        import json

        payload = json.loads(Path(args.models_file).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("models", [])
        cfg.models = [str(m) for m in payload]

    mapping = {
        "host": args.host,
        "port": args.port,
        "model_dir": args.model_dir,
        "ttft_ms": args.ttft_ms,
        "tpot_ms": args.tpot_ms,
        "jitter": args.jitter,
        "prefill_tps": args.prefill_tps,
        "decode_tps": args.decode_tps,
        "max_concurrency": args.max_concurrency,
        "block_size": args.block_size,
        "cache_blocks": args.cache_blocks,
        "default_max_tokens": args.default_max_tokens,
        "max_tokens_cap": args.max_tokens_cap,
        "tool_call_ratio": args.tool_call_ratio,
        "tool_payload_tokens": args.tool_payload_tokens,
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "log_limit": args.log_limit,
        "log_file": args.log_file,
        "stats_sample_limit": args.stats_sample_limit,
        "workers": args.workers,
    }
    for key, value in mapping.items():
        if value is not None:
            setattr(cfg, key, value)

    if not cfg.model_dir:
        cfg.model_dir = str(Path.cwd() / "models")
    return cfg


def config_as_dict(cfg: ServerConfig) -> dict[str, Any]:
    data = {f.name: getattr(cfg, f.name) for f in fields(ServerConfig)}
    data["version"] = __import__("gptzero").__version__
    return data
