"""分位数与聚合。

输入是 runner 记录的逐请求字典（``bench.client.RequestResult.to_dict()``），
输出是可直接 JSON 序列化、并驱动报告/图表的汇总结构。
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from .util import flatten, safe_div

PERCENTILES = (50.0, 90.0, 95.0, 99.0)


def percentile(values: Sequence[float], p: float) -> float | None:
    """线性插值分位数（与 numpy 默认 ``linear`` 一致）。

    * 空样本 → ``None``（报告里显示 ``-``，不伪造 0）；
    * 单样本 → 该值；
    * 非有限值会被忽略。
    """
    if not values:
        return None
    clean = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    n = len(clean)
    if n == 0:
        return None
    if n == 1:
        return clean[0]
    if p <= 0:
        return clean[0]
    if p >= 100:
        return clean[-1]
    rank = (n - 1) * p / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return clean[lo]
    weight = rank - lo
    return clean[lo] * (1.0 - weight) + clean[hi] * weight


def mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return None
    return sum(clean) / len(clean)


def summarize(values: Sequence[float] | None) -> dict[str, float | int | None]:
    """p50/p90/p95/p99/mean/max/min + count。空样本返回全 None、count=0。"""
    data = list(values or [])
    clean = [float(v) for v in data if v is not None and math.isfinite(float(v))]
    if not clean:
        return {
            "count": 0,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "mean": None,
            "max": None,
            "min": None,
        }
    return {
        "count": len(clean),
        "p50": percentile(clean, 50),
        "p90": percentile(clean, 90),
        "p95": percentile(clean, 95),
        "p99": percentile(clean, 99),
        "mean": mean(clean),
        "max": max(clean),
        "min": min(clean),
    }


def cache_hit_rate(hit_tokens: float | int | None, query_tokens: float | int | None) -> float | None:
    return safe_div(hit_tokens, query_tokens)


def _samples(records: Iterable[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for rec in records:
        value = rec.get(key)
        if value is not None:
            out.append(float(value))
    return out


def _itl_samples(records: Iterable[dict[str, Any]]) -> list[float]:
    out: list[float] = []
    for rec in records:
        out.extend(float(v) for v in (rec.get("itl_ms") or []))
    return out


def summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """对一组成功请求统计延迟（TTFT/TPOT/E2E 按请求，ITL 按帧）。"""
    return {
        "ttft_ms": summarize(_samples(records, "ttft_ms")),
        "tpot_ms": summarize(_samples(records, "tpot_ms")),
        "e2e_ms": summarize(_samples(records, "e2e_ms")),
        "itl_ms": summarize(_itl_samples(records)),
    }


def _tokens(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    prompt = sum(int(r.get("prompt_tokens") or 0) for r in records)
    completion = sum(int(r.get("completion_tokens") or 0) for r in records)
    tool_calls = sum(int(r.get("tool_calls") or 0) for r in records)
    return {"prompt": prompt, "completion": completion, "tool_calls": tool_calls}


def aggregate_records(records: Sequence[dict[str, Any]], wall_s: float | None) -> dict[str, Any]:
    """汇总一个并发点。``wall_s`` 为实测窗口（用于 RPS / tok/s）。

    失败请求计入失败率与总请求数，但不进入延迟分位数。
    """
    total = len(records)
    success = [r for r in records if r.get("ok")]
    failed = [r for r in records if not r.get("ok")]
    tokens = _tokens(success)
    wall = wall_s if wall_s and wall_s > 0 else None
    summary = summarize_records(success)

    per_profile: dict[str, Any] = {}
    for profile in sorted({str(r.get("profile") or "unknown") for r in records}):
        group_all = [r for r in records if str(r.get("profile") or "unknown") == profile]
        group_ok = [r for r in group_all if r.get("ok")]
        group_tokens = _tokens(group_ok)
        per_profile[profile] = {
            "requests": {"total": len(group_all), "success": len(group_ok), "failed": len(group_all) - len(group_ok)},
            "tokens": group_tokens,
            "summary": summarize_records(group_ok),
            "rps": safe_div(len(group_ok), wall),
            "input_tokens_per_s": safe_div(group_tokens["prompt"], wall),
            "output_tokens_per_s": safe_div(group_tokens["completion"], wall),
            "samples": {
                "ttft_ms": sorted(float(r["ttft_ms"]) for r in group_ok if r.get("ttft_ms") is not None),
                "e2e_ms": sorted(float(r["e2e_ms"]) for r in group_ok if r.get("e2e_ms") is not None),
            },
        }

    return {
        "requests": {
            "total": total,
            "success": len(success),
            "failed": len(failed),
            "failure_rate": safe_div(len(failed), total),
            "sub_requests": sum(1 for r in records if r.get("label") == "sub"),
            "tool_call_requests": sum(1 for r in success if int(r.get("tool_calls") or 0) > 0),
        },
        "tokens": tokens,
        "rates": {
            "rps": safe_div(total, wall),
            "success_rps": safe_div(len(success), wall),
            "input_tokens_per_s": safe_div(tokens["prompt"], wall),
            "output_tokens_per_s": safe_div(tokens["completion"], wall),
            "total_tokens_per_s": safe_div(tokens["prompt"] + tokens["completion"], wall),
        },
        "summary": summary,
        "per_profile": per_profile,
        "errors": _error_summary(failed),
        "samples": {
            "ttft_ms": sorted(float(r["ttft_ms"]) for r in success if r.get("ttft_ms") is not None),
            "e2e_ms": sorted(float(r["e2e_ms"]) for r in success if r.get("e2e_ms") is not None),
        },
    }


def _error_summary(failed: Sequence[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for rec in failed:
        key = str(rec.get("error") or "unknown")[:240]
        counts[key] = counts.get(key, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    return [{"error": msg, "count": count} for msg, count in ordered]


def combined_itl(records: Sequence[dict[str, Any]]) -> list[float]:
    return flatten(rec.get("itl_ms") or [] for rec in records)
