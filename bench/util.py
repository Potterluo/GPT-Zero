"""Shared helpers: JSON IO, Prometheus text parsing, token estimation.

依赖仅标准库；不联网、不 import 服务端内部模块。
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"

# 与 gptzero.tokenizer.HeuristicCodec 同构的轻量计数器（仅用于 usage 缺失时兜底）。
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_WORD = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


def estimate_tokens(text: str) -> int:
    """与 mock 的 heuristic tokenizer 相同规则的 token 数估算。"""
    if not text:
        return 0
    total = 0
    for piece in _WORD.findall(text):
        if _CJK.fullmatch(piece):
            total += 1
        elif piece.isspace():
            continue
        elif len(piece) <= 4:
            total += 1
        else:
            total += math.ceil(len(piece) / 4)
    return total


def safe_div(num: float | int | None, den: float | int | None) -> float | None:
    """除法；分母为 0/None 时返回 None，避免报告里出现 inf/nan。"""
    if num is None or den is None:
        return None
    try:
        if float(den) == 0.0:
            return None
        value = float(num) / float(den)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def round_or_none(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def ensure_parent(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | os.PathLike[str], payload: Any) -> Path:
    p = ensure_parent(path)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return p


def load_json(path: str | os.PathLike[str]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text(path: str | os.PathLike[str], text: str) -> Path:
    p = ensure_parent(path)
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Prometheus text format
# ---------------------------------------------------------------------------

_METRIC_LINE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{.*\})?\s+(?P<value>[-+0-9.eE]+)\s*$")


def parse_prometheus(text: str) -> dict[str, float]:
    """把 Prometheus 文本解析成 ``{metric_name: sum(value)}``。

    同名不同 label 的序列会被求和；对 bench 使用的 counter（prefix cache /
    token 总量 / success）足够且稳定。gauge 求和没有意义，调用方按需取用。
    """
    out: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE.match(line)
        if not match:
            continue
        name = match.group("name")
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        out[name] = out.get(name, 0.0) + value
    return out


def counter_delta(before: dict[str, float] | None, after: dict[str, float] | None, name: str) -> float | None:
    """单调 counter 的差值；缺失或服务端重启（负值）时返回 None。"""
    if not before or not after:
        return None
    b = before.get(name)
    a = after.get(name)
    if b is None or a is None:
        return None
    delta = a - b
    if delta < 0:
        return None
    return delta


def env_summary() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "bench_python": sys.executable,
    }


def human_ms(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}"


def fmt_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.{digits}f}%"


def flatten(iters: Iterable[Iterable[float]]) -> list[float]:
    out: list[float] = []
    for it in iters:
        out.extend(it)
    return out


def monotonic() -> float:
    return time.perf_counter()
