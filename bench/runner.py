"""闭环压测执行器。

* 并发 N = N 个 worker 协程，每个 worker 维持「一个会话、一条在途请求」的闭环
  （agentic fan-out 时额外派生一条 subagent 请求）。
* 预热 K 条请求不进入统计；``/metrics`` 与 ``/api/stats`` 在预热后取 before 快照，
  测量结束后取 after 快照，用差值计算服务端 token 吞吐与 prefix cache 命中率。
* 按 ``duration`` 秒运行；到点后不再发起新请求，已在途请求允许完成。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from . import __version__
from .aggregate import aggregate_records, summarize
from .client import build_payload, chat_stream, make_client
from .util import (
    counter_delta,
    env_summary,
    now_iso,
    parse_prometheus,
    safe_div,
)
from .workload import PROFILES, build_session, warmup_messages

METRIC_KEYS = (
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_evictions_total",
    "vllm:request_success_total",
    "vllm:time_to_first_token_seconds_count",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:e2e_request_latency_seconds_count",
    "gptzero:tool_calls_total",
    "gptzero:cache_hit_tokens_total",
)


@dataclass
class BenchConfig:
    url: str = "http://127.0.0.1:8090"
    model: str = "gpt-3.5-turbo"
    profiles: list[str] = field(default_factory=lambda: ["coding"])
    concurrency: list[int] = field(default_factory=lambda: [1])
    duration: float = 20.0
    warmup: int = 3
    seed: int = 42
    timeout: float = 300.0
    fail_fast: bool = False
    max_tokens: int | None = None
    notes: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.profiles:
            raise ValueError("at least one profile is required")
        for p in self.profiles:
            if p not in PROFILES:
                raise ValueError(f"unknown profile {p!r}; choose from {', '.join(PROFILES)}")
        if not self.concurrency:
            raise ValueError("at least one concurrency level is required")
        if any(c <= 0 for c in self.concurrency):
            raise ValueError("concurrency levels must be positive")
        if self.duration <= 0:
            raise ValueError("duration must be positive")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


async def fetch_json(client: httpx.AsyncClient, url: str, path: str) -> dict[str, Any]:
    response = await client.get(f"{url.rstrip('/')}{path}")
    response.raise_for_status()
    return response.json()


async def fetch_metrics(client: httpx.AsyncClient, url: str) -> dict[str, float]:
    response = await client.get(f"{url.rstrip('/')}/metrics")
    response.raise_for_status()
    return parse_prometheus(response.text)


async def fetch_config(client: httpx.AsyncClient, url: str) -> dict[str, Any] | None:
    """读取 ``/api/config``；旧版本服务端没有该端点时返回 None（不阻断压测）。"""
    try:
        payload = await fetch_json(client, url, "/api/config")
    except (httpx.HTTPError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


async def server_snapshot(client: httpx.AsyncClient, url: str) -> dict[str, Any]:
    stats = await fetch_json(client, url, "/api/stats")
    try:
        metrics = await fetch_metrics(client, url)
    except httpx.HTTPError:
        metrics = {}
    return {"stats": stats, "metrics": metrics}


async def probe(client: httpx.AsyncClient, url: str) -> dict[str, Any]:
    health = await fetch_json(client, url, "/health")
    stats = await fetch_json(client, url, "/api/stats")
    return {"health": health, "stats": stats}


CONFIG_KEYS = (
    "ttft_ms",
    "tpot_ms",
    "max_concurrency",
    "cache_blocks",
    "block_size",
    "jitter",
    "prefill_tps",
    "decode_tps",
)


def _config_diff(expected: dict[str, Any] | None, actual: dict[str, Any] | None) -> dict[str, Any]:
    """返回 key → (expected, actual)；只比较影响压测语义的参数。"""
    if not expected or not actual:
        return {}
    diff: dict[str, Any] = {}
    for key in CONFIG_KEYS:
        if key not in expected and key not in actual:
            continue
        left, right = expected.get(key), actual.get(key)
        if left is None and right is None:
            continue
        try:
            same = abs(float(left) - float(right)) < 1e-9
        except (TypeError, ValueError):
            same = left == right
        if not same:
            diff[key] = (left, right)
    return diff


def _check_config(
    expected: dict[str, Any] | None,
    actual: dict[str, Any] | None,
    concurrency: int,
    profile: str,
) -> None:
    diff = _config_diff(expected, actual)
    if diff:
        raise RuntimeError(
            f"/api/config 与 sweep 首点不一致（c={concurrency} profile={profile}）：{diff}。"
            "端口可能被另一个实例接管，结果不可信，已中止。"
        )


def _sanity(
    agg: dict[str, Any],
    config_full: dict[str, Any],
    warmup_probe: dict[str, Any],
    concurrency: int,
) -> dict[str, Any]:
    warnings: list[str] = []
    cfg_tpot = config_full.get("tpot_ms")
    measured = (agg["summary"].get("tpot_ms") or {}).get("p50")
    itl_measured = (agg["summary"].get("itl_ms") or {}).get("p50")
    probe_tpot = warmup_probe.get("tpot_ms_p50")
    try:
        cfg_tpot = float(cfg_tpot) if cfg_tpot is not None else None
    except (TypeError, ValueError):
        cfg_tpot = None
    if cfg_tpot and cfg_tpot > 0:
        if measured is not None and measured < 0.5 * cfg_tpot:
            warnings.append(
                f"TPOT p50={measured:.3f}ms 远低于服务端配置 {cfg_tpot}ms 的 50%，"
                "疑似服务端未按配置计时（同端口可能混入旧实例，或事件循环被长阻塞后 sleep 补发）"
            )
        if itl_measured is not None and itl_measured < 0.5 * cfg_tpot:
            warnings.append(
                f"ITL p50={itl_measured:.3f}ms 远低于配置 {cfg_tpot}ms 的 50%，"
                "说明内容帧被突发投递（定时器精度/提前唤醒），该点延迟数据不可信"
            )
    max_c = config_full.get("max_concurrency") or 0
    if max_c and concurrency > max_c:
        warnings.append(f"并发 {concurrency} 超过服务端槽位 {max_c}，TTFT 包含排队时间")
    success = int((agg.get("requests") or {}).get("success") or 0)
    sample_count = len((agg.get("samples") or {}).get("e2e_ms") or [])
    if success and sample_count != success:
        warnings.append(f"样本数与成功请求数不一致（samples={sample_count}, success={success}）")
    return {
        "ok": not any("TPOT" in w for w in warnings),
        "warnings": warnings,
        "configured_tpot_ms": cfg_tpot,
        "measured_tpot_p50_ms": measured,
        "probe_tpot_ms": probe_tpot,
        "sample_count": sample_count,
        "success_count": success,
    }


# ---------------------------------------------------------------------------
# 单点压测
# ---------------------------------------------------------------------------


async def _warmup(client: httpx.AsyncClient, cfg: BenchConfig) -> list[Any]:
    """顺序打 K 条短请求：不进统计、不参与失败率，仅用于预热与计时自检。"""
    payload = build_payload(
        cfg.model,
        warmup_messages(cfg.seed),
        max_tokens=32,
        seed=cfg.seed,
    )
    results = []
    for i in range(max(0, int(cfg.warmup))):
        payload["seed"] = cfg.seed + i
        results.append(await chat_stream(client, cfg.url, payload, profile="warmup", label="warmup"))
    return results


async def _worker(
    wid: int,
    cfg: BenchConfig,
    profile: str,
    deadline: float,
    client: httpx.AsyncClient,
    records: list[dict[str, Any]],
    stop: asyncio.Event,
) -> None:
    session_seq = 0
    while not stop.is_set() and time.perf_counter() < deadline:
        index = cfg.seed * 100000 + wid * 1000 + session_seq
        session = build_session(profile, index=index, seed=cfg.seed, max_tokens=cfg.max_tokens)
        while not session.done and not stop.is_set():
            turn = session.next_turn()
            if turn is None:
                break
            if time.perf_counter() >= deadline:
                break
            tasks = [
                chat_stream(
                    client,
                    cfg.url,
                    spec.to_payload(cfg.model),
                    profile=profile,
                    label=spec.label,
                )
                for spec in turn.specs
            ]
            results = await asyncio.gather(*tasks)
            for spec, result in zip(turn.specs, results):
                item = result.to_dict()
                item["est_prompt_tokens"] = spec.est_prompt_tokens
                item["turn"] = spec.turn
                records.append(item)
            session.observe(results)
            if cfg.fail_fast and any(not r.ok for r in results):
                stop.set()
                raise RuntimeError(
                    f"fail-fast: request failed at concurrency={cfg.concurrency} profile={profile}: "
                    f"{results[0].error}"
                )
        session_seq += 1


async def run_point(
    cfg: BenchConfig,
    concurrency: int,
    profile: str,
    *,
    expected_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """跑一个 (concurrency, profile) 测量点，返回自包含 dict。

    ``expected_config`` 为 sweep 首点采集到的 /api/config；后续点若与之不一致
    （例如端口被别的实例接管）直接抛错，而不是产出不可信数据。
    """
    max_conns = max(64, int(concurrency) * 4 + 16)
    client = make_client(timeout_s=cfg.timeout, max_connections=max_conns)
    started_at = now_iso()
    records: list[dict[str, Any]] = []
    stop = asyncio.Event()
    try:
        await probe(client, cfg.url)
        config_full = await fetch_config(client, cfg.url)
        config_unverified = config_full is None
        _check_config(expected_config, config_full, concurrency, profile)
        warmup_results = await _warmup(client, cfg)
        before = await server_snapshot(client, cfg.url)

        wall_t0 = time.perf_counter()
        deadline = wall_t0 + float(cfg.duration)
        tasks = [
            asyncio.create_task(_worker(i, cfg, profile, deadline, client, records, stop))
            for i in range(int(concurrency))
        ]
        failure: Exception | None = None
        try:
            await asyncio.gather(*tasks)
        except Exception as exc:  # fail-fast 冒泡；先取消其余 worker
            failure = exc
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        wall_end = time.perf_counter()
        after = await server_snapshot(client, cfg.url)
        if failure is not None:
            raise failure
    finally:
        await client.aclose()

    wall_s = wall_end - wall_t0
    if records:
        span = max(r["wall_end"] for r in records) - min(r["wall_start"] for r in records)
        wall_s = max(span, 1e-9)
    agg = aggregate_records(records, wall_s)

    metrics_delta = {
        key: counter_delta(before["metrics"], after["metrics"], key) for key in METRIC_KEYS
    }
    stats_before = before["stats"]
    stats_after = after["stats"]
    stats_delta = {
        key: (
            (stats_after.get(key) or 0) - (stats_before.get(key) or 0)
            if isinstance(stats_after.get(key), (int, float)) and isinstance(stats_before.get(key), (int, float))
            else None
        )
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "completed_requests",
            "tool_calls",
            "cache_hit_tokens",
            "cache_query_tokens",
        )
    }

    queries = metrics_delta.get("vllm:prefix_cache_queries_total")
    hits = metrics_delta.get("vllm:prefix_cache_hits_total")
    cache_source = "metrics"
    if not queries:
        queries = stats_delta.get("cache_query_tokens")
        hits = stats_delta.get("cache_hit_tokens")
        cache_source = "stats"
    hit_rate = safe_div(hits, queries)
    if hit_rate is not None:
        hit_rate = max(0.0, min(1.0, hit_rate))

    server_cfg = stats_after.get("config", {}) or {}
    if not config_full:
        # 服务端没有 /api/config 时退回 /api/stats 的 config 字段，但标记未校验。
        config_full = server_cfg
    warm_ok = [r for r in warmup_results if getattr(r, "ok", False)]
    warmup_probe = {
        "requests": len(warmup_results),
        "ok": len(warm_ok),
        "ttft_ms_p50": (summarize([r.ttft_ms for r in warm_ok if r.ttft_ms is not None]) or {}).get("p50"),
        "e2e_ms_p50": (summarize([r.e2e_ms for r in warm_ok if r.e2e_ms is not None]) or {}).get("p50"),
        "tpot_ms_p50": (summarize([r.tpot_ms for r in warm_ok if r.tpot_ms is not None]) or {}).get("p50"),
    }
    sanity = _sanity(agg, config_full, warmup_probe, concurrency)
    if config_unverified:
        sanity["warnings"].append("无法读取 /api/config，服务端配置未校验（已回退 /api/stats.config）")
        sanity["ok"] = False
    server_rates = {
        "prompt_tokens_per_s": safe_div(metrics_delta.get("vllm:prompt_tokens_total"), wall_s),
        "generation_tokens_per_s": safe_div(metrics_delta.get("vllm:generation_tokens_total"), wall_s),
    }

    return {
        "concurrency": int(concurrency),
        "profile": profile,
        "requested_duration_s": float(cfg.duration),
        "wall_s": round(wall_s, 4),
        "warmup_requests": len(warmup_results),
        "warmup_probe": warmup_probe,
        "started_at": started_at,
        "finished_at": now_iso(),
        "requests": agg["requests"],
        "tokens": agg["tokens"],
        "rates": {**agg["rates"], "server": server_rates},
        "summary": agg["summary"],
        "per_profile": agg["per_profile"],
        "cache": {
            "queries_tokens": queries,
            "hit_tokens": hits,
            "hit_rate": hit_rate,
            "source": cache_source,
            "evictions": metrics_delta.get("vllm:prefix_cache_evictions_total"),
        },
        "samples": agg["samples"],
        "errors": agg["errors"],
        "sanity": sanity,
        "config_unverified": config_unverified,
        "server_config": server_cfg,
        "server_config_full": config_full,
        "server_stats": stats_after,
        "server_stats_delta": stats_delta,
        "metrics_delta": metrics_delta,
        "records_kept": len(records),
        "bench_version": __version__,
    }


async def run_sweep(
    cfg: BenchConfig,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """按 concurrency 外层、profile 内层逐点测量（逐点独立快照）。"""
    cfg.validate()
    points: list[dict[str, Any]] = []
    baseline: dict[str, Any] | None = None
    for concurrency in cfg.concurrency:
        for profile in cfg.profiles:
            point = await run_point(cfg, int(concurrency), profile, expected_config=baseline)
            if baseline is None:
                baseline = point.get("server_config_full")
            points.append(point)
            if progress is not None:
                progress(point)
    return points


def build_document(cfg: BenchConfig, points: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """组装自包含结果 JSON（report 只需这个文件即可重建 md/csv/图）。"""
    server_cfg = points[0].get("server_config_full", {}) if points else {}
    warnings = [
        {"concurrency": p.get("concurrency"), "profile": p.get("profile"), "warning": w}
        for p in points
        for w in ((p.get("sanity") or {}).get("warnings") or [])
    ]
    return {
        "schema_version": 1,
        "tool": {"name": "gptzero-bench", "version": __version__},
        "name": name,
        "generated_at": now_iso(),
        "config": {
            "url": cfg.url,
            "model": cfg.model,
            "profiles": list(cfg.profiles),
            "profile": cfg.profiles[0] if cfg.profiles else None,
            "concurrency": list(cfg.concurrency),
            "duration_s": float(cfg.duration),
            "warmup": int(cfg.warmup),
            "seed": int(cfg.seed),
            "timeout_s": float(cfg.timeout),
            "fail_fast": bool(cfg.fail_fast),
            "max_tokens": cfg.max_tokens,
            "notes": list(cfg.notes),
            "env": env_summary(),
        },
        "server": {
            "url": cfg.url,
            "config": server_cfg,
            "tokenizer": (points[0].get("server_stats", {}) if points else {}).get("tokenizer"),
            "config_verified": all(
                (not p.get("config_unverified")) and not _config_diff(server_cfg, p.get("server_config_full"))
                for p in points
            )
            if points
            else False,
        },
        "sanity": {"ok": not warnings, "warnings": warnings},
        "points": points,
        "charts": [],
        "notes": list(cfg.notes),
    }
