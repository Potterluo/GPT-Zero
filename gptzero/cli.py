"""CLI entry point for the mock server."""

from __future__ import annotations

import sys

import uvicorn

from . import __version__
from .app import create_app
from .config import load_config


def _begin_high_resolution_timers():
    """Windows: raise the system timer to 1 ms while serving.

    The mock paces sub-20 ms decode steps; the default Windows timer granularity
    (~15.6 ms) would round every ``asyncio.sleep(0.008)`` up to ~15.6 ms and make
    TPOT measurements look ~2x the configured value.  ``timeBeginPeriod(1)`` is
    the documented way to get millisecond sleeps.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        winmm = ctypes.WinDLL("winmm")
        return winmm if winmm.timeBeginPeriod(1) == 0 else None
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    cfg = load_config(argv)
    app = create_app(cfg)
    winmm = _begin_high_resolution_timers()

    print("=" * 68)
    print(f"  GPT-Zero Mock LLM v{__version__}  (offline, multi-protocol)")
    print("=" * 68)
    print(f"  listen       : http://{cfg.host}:{cfg.port}")
    print(f"  dashboard    : http://{cfg.host}:{cfg.port}/")
    print("  protocols    : openai chat-completions · openai responses · anthropic messages")
    print(f"  openai       : http://{cfg.host}:{cfg.port}/v1/chat/completions")
    print(f"                 http://{cfg.host}:{cfg.port}/v1/responses")
    print(f"  anthropic    : http://{cfg.host}:{cfg.port}/v1/messages")
    print(f"  metrics      : http://{cfg.host}:{cfg.port}/metrics")
    print(f"  models       : {', '.join(cfg.models)}")
    print(f"  tokenizer    : {app.state.gptzero.tokenizer.name}")
    print(f"  timing       : ttft={cfg.ttft_ms}ms tpot={cfg.tpot_ms}ms jitter={cfg.jitter}")
    print(f"  concurrency  : max={cfg.max_concurrency}")
    print(f"  prefix cache : {cfg.cache_blocks} blocks x {cfg.block_size} tokens")
    print(f"  tool calls   : ratio={cfg.tool_call_ratio} payload~{cfg.tool_payload_tokens} tok")
    if cfg.log_file:
        print(f"  request log  : {cfg.log_file}")
    print("=" * 68)
    for warning in app.state.gptzero.tokenizer.warnings:
        print(f"  tokenizer warning: {warning}")

    if cfg.workers > 1:
        print("  note: the mock keeps state in-process; --workers > 1 is not supported, using 1")
    if winmm is not None:
        print("  timer       : 1ms high-resolution timer enabled (timeBeginPeriod)")
    try:
        uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info", access_log=False)
    except KeyboardInterrupt:
        return 0
    finally:
        if winmm is not None:
            try:
                winmm.timeEndPeriod(1)
            except Exception:
                pass
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
