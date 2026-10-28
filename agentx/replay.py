"""AgentX 风格的闭环并发回放器（不依赖 aiperf）。

复刻 ``agentx-harness==1.0.7`` 的 ``--scenario agentx`` 语义（见 ``docs/AGENTX.md`` §2）：

* 按 trace 的 ``t``（秒）调度，**流内串行**：第 k 个请求真的要等第 k-1 个返回
  才可能发（等价 AIPerf 的 end-to-start idle gap：``max(0, t_k - (t_{k-1}+api_{k-1}))``）
* 子代理 ``SPAWN`` 出独立并发流，父链的下一条请求是 ``JOIN`` 点，被
  ``entry.t + duration_ms`` 门控
* ``trajectory start ratio``：在 trace 时间轴 ``[0.25, 0.75]`` 处取快照 ``t*``，
  从那里开始跑；``t*`` 之前的最后一个 turn 作为 **primer** 先打一次
* 每条 lane 额外 ``--warmup`` 条预热请求（不计入统计）
* ``cache-bust marker``：``[rid:sha256(benchmark_id:pass:trajectory_index:trace_id)[:12]]``
* 每请求 ``max_tokens=out`` + ``ignore_eos=true`` + ``stream=true``

指标：TTFT / TPOT / E2E 分位、req/s、out tok/s、tok/s/chip、prefix cache 命中
（服务端 ``/metrics`` 前后差值）、成功率。

CLI 入口在 :mod:`agentx.cli`。
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

# --------------------------------------------------------------------------- #
# prompt 合成：hash_ids -> 文本
#
# 服务端 prefix cache 命中的判据是"token 前缀相同"。我们用「块 ID -> 确定性
# 文本」的纯函数生成内容，因此同 ID 的块在任何请求里字节一致 -> 服务端自然
# 命中；不同 session/agent 用不同 ID 段 -> 不会误命中。
# --------------------------------------------------------------------------- #

_LEXICON = (
    "the of and to in a is that for it as was with be by on not he i this are or "
    "his from at which but have an they you were her she there one all we can if "
    "will would what some man been more other very when time much may such these "
    "two how our them way even new want because any these give day most us is are "
    "system user assistant tool result context request response token block prefix "
    "cache latency throughput agent session stream generate compute memory kernel "
    "batch schedule decode prefill attention layer matrix vector weight gradient"
).split()

_LEXICON_CHARS = sum(len(w) + 1 for w in _LEXICON)

#: 实测标定（`deepseek_v3_tokenizer` + 本词表）：1 个词 ≈ 1.05 个 token。
#: 于是 ``n_words = n_tokens / TOKENS_PER_WORD`` 能让线上 prompt_tokens ≈ 目标值。
TOKENS_PER_WORD = 1.05


def _filler(n_tokens: int, seed: int | str, tokens_per_word: float = TOKENS_PER_WORD) -> str:
    """确定性地生成约 ``n_tokens`` 个 token 的填充文本。

    AIPerf 用真 tokenizer 精确构造 prompt；离线回放拿不到服务端 tokenizer 的
    每个 token id，只能按"每词 token 数"标定。这里的目标是**量级对齐**，让
    `prompt_tokens / in` ≈ 1，从而 cache 命中率、prefill 压力可比。
    """
    if n_tokens <= 0:
        return ""
    rng = random.Random(seed if isinstance(seed, int) else _stable_seed(seed))
    n_words = max(1, int(round(n_tokens / max(tokens_per_word, 0.2))))
    words = [rng.choice(_LEXICON) for _ in range(n_words)]
    return " ".join(words) + " "


def _stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def _block_text(block_id: int, block_size: int, tokens_per_word: float = TOKENS_PER_WORD) -> str:
    """一个 KV 块的可复现文本（同 id 全局一致，带缓存避免重复生成）。"""
    return _cached_block_text(block_id, block_size, tokens_per_word)


@lru_cache(maxsize=200_000)
def _cached_block_text(block_id: int, block_size: int, tokens_per_word: float) -> str:
    return _filler(block_size, block_id, tokens_per_word)


def build_prompt(
    hash_ids: Sequence[int],
    in_tokens: int,
    block_size: int,
    fresh_seed: str,
    tokens_per_word: float = TOKENS_PER_WORD,
) -> str:
    """按 AIPerf 的三种布局拼 prompt：缓存前缀块 + 未缓存尾部。

    * ``len(hash_ids)*block_size == in_tokens``：精确铺满
    * ``< in_tokens``：前缀 + fresh tail（真实抓取形态）
    * ``> in_tokens``：截断到最后一块（防止 estimated token 膨胀）
    """
    cached_tokens = len(hash_ids) * block_size
    if cached_tokens >= in_tokens:
        keep = max(1, in_tokens // max(block_size, 1))
        return "".join(_block_text(b, block_size, tokens_per_word) for b in hash_ids[:keep])
    prefix = "".join(_block_text(b, block_size, tokens_per_word) for b in hash_ids)
    tail_tokens = in_tokens - cached_tokens
    cache_key = (hash_ids[-1] if hash_ids else -1, tail_tokens)
    return prefix + _filler(tail_tokens, f"{fresh_seed}:{cache_key}", tokens_per_word)


# --------------------------------------------------------------------------- #
# cache-bust marker（与 aiperf/timing/strategies/cache_bust.py 同构）
# --------------------------------------------------------------------------- #


def build_cache_bust_marker(
    benchmark_id: str, recycle_pass: int, trajectory_index: int, trace_id: str
) -> str:
    unique = f"{benchmark_id}:{recycle_pass}:{trajectory_index}:{trace_id}"
    digest = hashlib.sha256(unique.encode()).hexdigest()[:12]
    return f"[rid:{digest}]\n\n"


# --------------------------------------------------------------------------- #
# trace -> 可回放计划
# --------------------------------------------------------------------------- #


@dataclass
class Call:
    """一次 API 请求（trace 里的一条 n/s 记录）。"""

    t: float
    in_tokens: int
    out_tokens: int
    hash_ids: list[int]
    stop: str = ""
    kind: str = "main"  # main | subagent
    agent_id: str | None = None
    #: 抓取时记录的服务端耗时（秒）。只用于 idle-gap 夹紧的 end-to-start 折算，
    #: 真实延迟由被测服务在回放时重新产生。
    api_time: float = 0.0


@dataclass
class Subagent:
    """一个 ``type="subagent"`` 块（已解析为可并发执行的子链）。"""

    agent_id: str
    spawn_t: float
    end_t: float
    calls: list[Call] = field(default_factory=list)


@dataclass
class TracePlan:
    """一条 trace 的回放计划。"""

    trace_id: str
    block_size: int
    main: list[Call]  # 主链（按 t 升序，且已剔除 JOIN 依赖表达）
    #: 主链第 i 个请求之前必须 join 完的子代理（由外层列表顺序决定）
    joins_before: dict[int, list[Subagent]] = field(default_factory=dict)
    #: 主链第 i 个请求之后立即 spawn 的子代理（并行于父链的"等待期"）
    spawns_after: dict[int, list[Subagent]] = field(default_factory=dict)
    span_s: float = 0.0


def build_plan(trace: dict[str, Any]) -> TracePlan:
    """把 WEKA trace 解析成 ``TracePlan``（含 SPAWN / JOIN 关系）。"""
    block_size = int(trace.get("block_size") or 64)
    main: list[Call] = []
    joins_before: dict[int, list[Subagent]] = {}
    spawns_after: dict[int, list[Subagent]] = {}
    pending: list[Subagent] = []
    last_end = 0.0

    for req in trace.get("requests") or []:
        if req.get("type") == "subagent":
            spawn_t = float(req.get("t") or 0.0)
            duration = req.get("duration_ms")
            if duration is not None:
                end_t = spawn_t + float(duration) / 1000.0
            else:
                # async_launched：退化为内层请求的最晚结束时刻
                end_t = max(
                    (float(c.get("t") or 0.0) + float(c.get("api_time") or 0.0)
                     for c in req.get("requests") or []),
                    default=spawn_t,
                )
            sub = Subagent(
                agent_id=str(req.get("agent_id") or "agent"),
                spawn_t=spawn_t,
                end_t=end_t,
                calls=[
                    Call(
                        t=float(c.get("t") or 0.0),
                        in_tokens=int(c.get("in") or 1),
                        out_tokens=int(c.get("out") or 1),
                        hash_ids=list(c.get("hash_ids") or []),
                        stop=str(c.get("stop") or ""),
                        kind="subagent",
                        agent_id=str(req.get("agent_id") or "agent"),
                        api_time=float(c.get("api_time") or 0.0),
                    )
                    for c in req.get("requests") or []
                    if c.get("type") in ("n", "s")
                ],
            )
            pending.append(sub)
            last_end = max(last_end, end_t)
            continue

        call = Call(
            t=float(req.get("t") or 0.0),
            in_tokens=int(req.get("in") or 1),
            out_tokens=int(req.get("out") or 1),
            hash_ids=list(req.get("hash_ids") or []),
            stop=str(req.get("stop") or ""),
            api_time=float(req.get("api_time") or 0.0),
        )
        idx = len(main)
        if pending:
            joins_before[idx] = pending
            if idx > 0:
                spawns_after.setdefault(idx - 1, []).extend(pending)
            pending = []
        main.append(call)
        last_end = max(last_end, call.t)

    if pending and main:
        # trace 尾部还有没 join 的子代理：挂在最后一个请求后面
        spawns_after.setdefault(len(main) - 1, []).extend(pending)

    span = last_end
    for sub in main:
        span = max(span, sub.t)
    for subs in list(joins_before.values()) + list(spawns_after.values()):
        for su in subs:
            span = max(span, su.end_t)
    return TracePlan(
        trace_id=str(trace.get("id") or "trace"),
        block_size=block_size,
        main=main,
        joins_before=joins_before,
        spawns_after=spawns_after,
        span_s=span,
    )


# --------------------------------------------------------------------------- #
# 服务端交互
# --------------------------------------------------------------------------- #


def _parse_prom(text: str) -> dict[str, float]:
    """解析 Prometheus text，返回 ``{metric{label}: value}``（只取计数器/仪表）。"""
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        name, value = parts
        try:
            out[name] = float(value)
        except ValueError:
            continue
    return out


def _metric_total(metrics: dict[str, float], name: str) -> float:
    """带 label 的计数器求和（如 ``vllm:request_success_total{...}``）。"""
    total = 0.0
    for key, value in metrics.items():
        base = key.split("{", 1)[0]
        if base == name:
            total += value
    return total


async def fetch_cache_snapshot(client: Any, url: str) -> dict[str, float]:
    """抓一次 prefix cache 相关累计量（``/metrics`` 优先，``/api/stats`` 兜底）。"""
    out = {
        "hits": 0.0,
        "queries": 0.0,
        "evictions": 0.0,
        "requests": 0.0,
        "generation_tokens": 0.0,
        "prompt_tokens": 0.0,
        "source": 0.0,  # 1 = /metrics, 0 = /api/stats, -1 = 都没有
    }
    try:
        resp = await client.get(f"{url}/metrics", timeout=5.0)
        if resp.status_code == 200:
            m = _parse_prom(resp.text)
            out.update(
                {
                    "hits": _metric_total(m, "vllm:prefix_cache_hits_total"),
                    "queries": _metric_total(m, "vllm:prefix_cache_queries_total"),
                    "evictions": _metric_total(m, "vllm:prefix_cache_evictions_total"),
                    "requests": _metric_total(m, "vllm:request_success_total"),
                    "generation_tokens": _metric_total(m, "vllm:generation_tokens_total"),
                    "prompt_tokens": _metric_total(m, "vllm:prompt_tokens_total"),
                    "source": 1.0,
                }
            )
            return out
    except Exception:  # noqa: BLE001 - 兜底路径，任何异常都退化
        pass
    try:
        resp = await client.get(f"{url}/api/stats", timeout=5.0)
        if resp.status_code == 200:
            s = resp.json()
            out.update(
                {
                    "hits": float(s.get("cache_hit_tokens") or 0.0),
                    "queries": float(s.get("cache_query_tokens") or 0.0),
                    "requests": float(s.get("completed_requests") or 0.0),
                    "generation_tokens": float(s.get("completion_tokens") or 0.0),
                    "prompt_tokens": float(s.get("prompt_tokens") or 0.0),
                    "source": 0.0,
                }
            )
            return out
    except Exception:  # noqa: BLE001
        pass
    return out


@dataclass
class RequestRecord:
    """单请求观测值。"""

    lane: int
    trace_id: str
    kind: str
    t: float
    start_s: float  # 相对本轮起点
    ttft_ms: float | None
    e2e_ms: float
    tpot_ms: float | None
    prompt_tokens: int
    completion_tokens: int
    ok: bool
    error: str | None = None


async def call_chat(
    client: Any,
    url: str,
    model: str,
    prompt: str,
    out_tokens: int,
    *,
    timeout_s: float,
    seed: int = 42,
) -> tuple[float | None, float, dict[str, Any], str | None]:
    """打一次流式 chat 请求，返回 ``(ttft_ms, e2e_ms, usage, error)``。"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max(1, int(out_tokens)),
        "min_tokens": max(1, int(out_tokens)),
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "seed": seed,
    }
    t0 = time.perf_counter()
    ttft: float | None = None
    usage: dict[str, Any] = {}
    try:
        async with client.stream(
            "POST", f"{url}/v1/chat/completions", json=payload, timeout=timeout_s
        ) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "replace")[:400]
                return None, (time.perf_counter() - t0) * 1000.0, {}, (
                    f"HTTP {resp.status_code}: {body}"
                )
            async for raw in resp.aiter_lines():
                line = raw.strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if ttft is None:
                    choices = obj.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        if (
                            (delta.get("content") or "")
                            or delta.get("tool_calls")
                            or choices[0].get("finish_reason")
                        ):
                            ttft = (time.perf_counter() - t0) * 1000.0
                if obj.get("usage"):
                    usage = obj["usage"]
    except asyncio.TimeoutError:
        return None, (time.perf_counter() - t0) * 1000.0, {}, f"timeout >{timeout_s}s"
    except Exception as exc:  # noqa: BLE001 - 网络错误都算失败请求
        return None, (time.perf_counter() - t0) * 1000.0, {}, f"{type(exc).__name__}: {exc}"

    e2e = (time.perf_counter() - t0) * 1000.0
    return ttft, e2e, usage, None


# --------------------------------------------------------------------------- #
# 单并发点的回放
# --------------------------------------------------------------------------- #


@dataclass
class RunConfig:
    url: str = "http://127.0.0.1:8090"  # 8080 被其它常驻服务占用；8090 为团队默认
    model: str = "gpt-3.5-turbo"
    concurrency: int = 4
    duration_s: float = 60.0
    warmup: int = 10
    primer: bool = True
    start_ratio_min: float = 0.25
    start_ratio_max: float = 0.75
    seed: int = 42
    system_idle_gap_cap: float = 10.0
    trace_idle_gap_cap: float = 300.0
    timeout_s: float = 600.0
    tokens_per_word: float = TOKENS_PER_WORD
    chips: int | None = None
    benchmark_id: str = "gptzero-agentx"
    max_requests: int | None = None
    #: 单请求输出 token 上限。官方不设（trace 的 out 直接作 max_tokens），
    #: 但本地 mock 的 tpot=8ms 下 8k 输出要 65s，会把短时回放拖死。
    max_out_tokens: int = 1024


def _apply_idle_caps(plan: TracePlan, run: RunConfig) -> TracePlan:
    """把 trace 内 >cap 的 idle gap 夹紧（AIPerf 的 system/trace idle gap cap）。

    AIPerf 的口径是 **end-to-start**（``t_k - (t_{k-1} + api_{k-1})``），不是
    start-to-start；用 start-to-start 会把上一请求的服务耗时也算进 gap，夹紧量
    偏小、并发被高估。
    """
    cap = min(run.system_idle_gap_cap, run.trace_idle_gap_cap)
    chains: list[list[Call]] = [plan.main]
    for subs in plan.joins_before.values():
        for sub in subs:
            chains.append(sub.calls)
    for chain in chains:
        prev_end = None
        for call in chain:
            if prev_end is not None and call.t - prev_end > cap:
                call.t = prev_end + cap
            prev_end = call.t + max(call.api_time, 0.0)
    return plan


def _pick_snapshot(plan: TracePlan, run: RunConfig, rng: random.Random) -> tuple[float, int]:
    """按 trajectory start ratio 取快照点 ``t*``，返回 (t*, 起始主链下标)。"""
    if plan.span_s <= 0 or not plan.main:
        return 0.0, 0
    ratio = rng.uniform(run.start_ratio_min, run.start_ratio_max)
    t_star = plan.span_s * ratio
    idx = 0
    for i, call in enumerate(plan.main):
        if call.t >= t_star:
            idx = i
            break
    return t_star, idx


async def run_point(
    plans: Sequence[TracePlan],
    run: RunConfig,
    *,
    client: Any,
) -> dict[str, Any]:
    """在一个并发点（= lane 数）上跑一轮回放，返回该点的指标。"""
    rng = random.Random(run.seed * 7919 + run.concurrency)
    traj_index_per_lane = list(range(run.concurrency))
    recycle_pass: dict[str, int] = {}
    markers: list[str] = []
    for lane in range(run.concurrency):
        plan = plans[lane % len(plans)]
        passes = recycle_pass.get(plan.trace_id, 0)
        recycle_pass[plan.trace_id] = passes + 1
        markers.append(
            build_cache_bust_marker(run.benchmark_id, passes, traj_index_per_lane[lane], plan.trace_id)
        )

    before: dict[str, float] = {}
    records: list[RequestRecord] = []
    t_loop0 = time.perf_counter()
    # deadline 在 warmup 之后才设定（见下方两阶段），这里先占位
    deadline = float("inf")
    hard_stop = asyncio.Event()

    async def dispatch(
        lane: int, plan: TracePlan, call: Call, marker: str, local_start: float
    ) -> None:
        if hard_stop.is_set():
            return
        prompt = marker + build_prompt(
            call.hash_ids, call.in_tokens, plan.block_size,
            f"{plan.trace_id}:{call.t}:{call.kind}", run.tokens_per_word,
        )
        ttft, e2e, usage, err = await call_chat(
            client, run.url, run.model, prompt,
            min(call.out_tokens, run.max_out_tokens),
            timeout_s=run.timeout_s, seed=run.seed,
        )
        in_tok = int(usage.get("prompt_tokens") or 0)
        out_tok = int(usage.get("completion_tokens") or call.out_tokens)
        tpot = None
        if ttft is not None and out_tok > 1:
            tpot = max(e2e - ttft, 0.0) / (out_tok - 1)
        records.append(
            RequestRecord(
                lane=lane,
                trace_id=plan.trace_id,
                kind=call.kind,
                t=call.t,
                start_s=time.perf_counter() - t_loop0,
                ttft_ms=ttft,
                e2e_ms=e2e,
                tpot_ms=tpot,
                prompt_tokens=in_tok,
                completion_tokens=out_tok,
                ok=err is None,
                error=err,
            )
        )

    async def sleep_until(target: float) -> bool:
        """睡到 ``target``（monotonic）。越过 deadline 返回 False（该收尾了）。

        关键：subagent 内的 ``t`` 可能远超本轮窗口，绝不能无条件 ``sleep`` 到
        未来某点，否则窗口结束后任务仍挂着，整轮 gather 永不返回。
        """
        now = time.perf_counter()
        if now > deadline:
            return False
        wait = target - now
        if wait <= 0:
            return True
        await asyncio.sleep(min(wait, max(0.0, deadline - now)))
        return time.perf_counter() <= deadline + 0.05

    async def run_subagent(lane: int, plan: TracePlan, sub: Subagent, marker: str, base_t: float, t0: float) -> None:
        """子代理内层串行执行（一条独立并发流），SPAWN 后即启动。"""
        for call in sub.calls:
            if hard_stop.is_set():
                return
            if not await sleep_until(t0 + (call.t - base_t)):
                return
            await dispatch(lane, plan, call, marker, t0)

    async def lane_warmup(lane: int, plan: TracePlan, marker: str) -> tuple[int, float]:
        """WARMUP 阶段：primer（``t*`` 前的最后一个 turn）+ ``--warmup`` 条。

        返回 ``(start_idx, base_t)`` 供 PROFILING 阶段复用。primer 的作用是把
        该 lane 的 prefix cache 预热到"跟线上一样热"，因此必须在测量窗口**之前**。
        """
        snap_rng = random.Random(run.seed * 104_729 + lane)
        _, start_idx = _pick_snapshot(plan, run, snap_rng)
        base_t = plan.main[start_idx].t if plan.main else 0.0
        t_warm = time.perf_counter()
        if run.primer and start_idx > 0:
            await dispatch(lane, plan, plan.main[start_idx - 1], marker, t_warm)
        if run.warmup > 0 and plan.main:
            primer = plan.main[max(start_idx - 1, 0)]
            for _ in range(run.warmup):
                await dispatch(lane, plan, primer, marker, t_warm)
        return start_idx, base_t

    async def lane_profile(lane: int, plan: TracePlan, marker: str, start_idx: int, base_t: float) -> None:
        """PROFILING 阶段：从 ``t*`` 起按 trace 时间轴回放主链 + 子代理。"""
        t0 = time.perf_counter()

        inflight: list[asyncio.Task] = []
        for idx in range(start_idx, len(plan.main)):
            if hard_stop.is_set() or time.perf_counter() > deadline:
                break
            call = plan.main[idx]
            if not await sleep_until(t0 + (call.t - base_t)):
                break
            # JOIN 门控：本请求之前 spawn 的子代理必须先全部结束
            if idx in plan.joins_before and inflight:
                pending_tasks, inflight = inflight, []
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            await dispatch(lane, plan, call, marker, t0)
            # 该请求之后 spawn 的子代理：立刻起飞，与父链下一段等待并行
            for sub in plan.spawns_after.get(idx, []):
                inflight.append(
                    asyncio.create_task(run_subagent(lane, plan, sub, marker, base_t, t0))
                )
        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)

    # --- 阶段 1：WARMUP（所有 lane 并行预热） ---
    warm_ctx = await asyncio.gather(
        *(lane_warmup(i, plans[i % len(plans)], markers[i]) for i in range(run.concurrency))
    )
    # --- WARMUP -> PROFILING 边界：此刻才开始计时与采缓存基线 ---
    before = await fetch_cache_snapshot(client, run.url)
    deadline = time.perf_counter() + run.duration_s
    started_at = time.perf_counter()

    # --- 阶段 2：PROFILING ---
    try:
        await asyncio.gather(
            *(
                lane_profile(i, plans[i % len(plans)], markers[i], warm_ctx[i][0], warm_ctx[i][1])
                for i in range(run.concurrency)
            )
        )
    finally:
        hard_stop.set()
    wall = time.perf_counter() - started_at


    after = await fetch_cache_snapshot(client, run.url)

    # warmup 记录剔除：按"本轮开始后第一个 profiling 请求之前的记录"近似——
    # dispatch 时无法区分，这里用 lane 的 warmup 数量做保守前缀剔除。
    warmup_per_lane = (1 if run.primer else 0) + max(run.warmup, 0)

    return summarize_point(
        records, wall, before, after, run, warmup_per_lane=warmup_per_lane
    )


# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #


def percentile(values: Sequence[float], q: float) -> float:
    """线性插值分位（与 numpy 默认一致，避免依赖 numpy）。"""
    if not values:
        return 0.0
    data = sorted(values)
    if len(data) == 1:
        return float(data[0])
    pos = (len(data) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(data[lo])
    return float(data[lo] + (data[hi] - data[lo]) * (pos - lo))


def _dist(values: Sequence[float]) -> dict[str, float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "p50": round(percentile(vals, 50), 3),
        "p90": round(percentile(vals, 90), 3),
        "p99": round(percentile(vals, 99), 3),
        "mean": round(sum(vals) / len(vals), 3),
        "max": round(max(vals), 3),
    }


def summarize_point(
    records: list[RequestRecord],
    wall_s: float,
    before: dict[str, float],
    after: dict[str, float],
    run: RunConfig,
    *,
    warmup_per_lane: int = 0,
) -> dict[str, Any]:
    """把一轮原始记录压成指标 dict（warmup 记录已剔除）。"""
    # 剔除 warmup：每个 lane 记录里最靠前的 warmup_per_lane 条
    by_lane: dict[int, list[RequestRecord]] = {}
    for rec in records:
        by_lane.setdefault(rec.lane, []).append(rec)
    measured: list[RequestRecord] = []
    for lane, recs in by_lane.items():
        recs_sorted = sorted(recs, key=lambda r: r.start_s)
        measured.extend(recs_sorted[warmup_per_lane:])
    if not measured:
        measured = list(records)

    ok = [r for r in measured if r.ok]
    failed = [r for r in measured if not r.ok]
    ttft_ok = [r.ttft_ms for r in ok if r.ttft_ms is not None]
    tpot_ok = [r.tpot_ms for r in ok if r.tpot_ms is not None]
    e2e_ok = [r.e2e_ms for r in ok]

    out_tokens = sum(r.completion_tokens for r in measured)
    in_tokens = sum(r.prompt_tokens for r in measured)
    req_rate = len(measured) / wall_s if wall_s > 0 else 0.0
    out_tps = out_tokens / wall_s if wall_s > 0 else 0.0
    total_tps = (in_tokens + out_tokens) / wall_s if wall_s > 0 else 0.0

    hits = after.get("hits", 0.0) - before.get("hits", 0.0)
    queries = after.get("queries", 0.0) - before.get("queries", 0.0)
    # /api/stats 兜底时 queries 是 prompt token 数；统一成"查询 token"
    cache_rate = (hits / queries) if queries > 0 else 0.0

    success_rate = (len(ok) / len(measured)) if measured else 0.0
    valid = success_rate >= 0.90 and len(measured) > 0

    return {
        "concurrency": run.concurrency,
        "duration_s": round(wall_s, 3),
        "requests": len(measured),
        "warmup_requests": len(records) - len(measured),
        "ok": len(ok),
        "failed": len(failed),
        "success_rate": round(success_rate, 4),
        "valid": valid,
        "ttft_ms": _dist(ttft_ok),
        "tpot_ms": _dist(tpot_ok),
        "e2e_ms": _dist(e2e_ok),
        "req_per_s": round(req_rate, 3),
        "output_tokens_per_s": round(out_tps, 2),
        "total_tokens_per_s": round(total_tps, 2),
        "output_tokens_per_s_per_chip": (
            round(out_tps / run.chips, 3) if run.chips else None
        ),
        "prompt_tokens": in_tokens,
        "completion_tokens": out_tokens,
        "cache": {
            "hits": int(hits),
            "queries": int(queries),
            "hit_rate": round(cache_rate, 4),
            "source": "metrics" if after.get("source") == 1 else ("stats" if after.get("source") == 0 else "none"),
        },
        "errors": sorted({r.error for r in failed if r.error})[:5],
    }


# --------------------------------------------------------------------------- #
# 报告产出
# --------------------------------------------------------------------------- #


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    meta = report.get("meta", {})
    runs = report.get("runs", [])
    lines = [
        "# AgentX 闭环回放报告",
        "",
        f"- 目标服务：`{meta.get('url')}`",
        f"- 模型：`{meta.get('model')}`",
        f"- trace 语料：`{meta.get('traces')}`（{meta.get('trace_count')} 条）",
        f"- 每点时长：{meta.get('duration_s')}s，warmup/lane：{meta.get('warmup')}，primer：{meta.get('primer')}",
        f"- 生成时间：{meta.get('generated_at')}",
        f"- 有效点：{sum(1 for r in runs if r.get('valid'))}/{len(runs)}",
        "",
        "## 汇总",
        "",
        "| 并发 | 请求 | 成功 | 成功率 | TTFT p50 | TTFT p90 | TTFT p99 | TPOT p50 | E2E p90 "
        "| req/s | out tok/s | tok/s/chip | 缓存命中 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        lines.append(
            "| {c} | {n} | {ok} | {sr} | {t50} | {t90} | {t99} | {p50} | {e90} | {rps} | {otps} | {chip} | {ch} |".format(
                c=r["concurrency"],
                n=r["requests"],
                ok=r["ok"],
                sr=f"{r['success_rate'] * 100:.1f}%",
                t50=_fmt(r["ttft_ms"]["p50"], 0),
                t90=_fmt(r["ttft_ms"]["p90"], 0),
                t99=_fmt(r["ttft_ms"]["p99"], 0),
                p50=_fmt(r["tpot_ms"]["p50"], 2),
                e90=_fmt(r["e2e_ms"]["p90"], 0),
                rps=_fmt(r["req_per_s"], 2),
                otps=_fmt(r["output_tokens_per_s"], 1),
                chip=_fmt(r["output_tokens_per_s_per_chip"], 2),
                ch=f"{r['cache']['hit_rate'] * 100:.1f}%",
            )
        )
    lines += ["", "## 与 AgentX 官方口径的差异", ""]
    for note in meta.get("notes", []):
        lines.append(f"- {note}")
    lines += [
        "",
        "## 指标口径",
        "",
        "- **TTFT**：请求发出 → 第一个含内容的 SSE chunk（`delta.content` 或 `tool_calls`）。",
        "- **TPOT**：`(E2E - TTFT) / (completion_tokens - 1)`，`completion_tokens` 取服务端 `usage`。",
        "- **缓存命中率**：`vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` 的本轮差值；"
        "缺失时退回 `/api/stats` 的 `cache_hit_tokens/cache_query_tokens`。",
        "- **tok/s/chip**：需要 `--chips N`；未知时留空（官方按设备芯片数折算）。",
        "- **成功率**：`--failed-request-threshold`（默认 0.10）以下仍标 `valid=false`，不出结论。",
        "",
        "字段级 schema 见 `results/agentx/replay.json`，逐请求明细用 `--dump-requests` 打开。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(report: dict[str, Any], path: Path) -> None:
    cols = [
        "concurrency", "requests", "ok", "failed", "success_rate", "valid",
        "ttft_p50", "ttft_p90", "ttft_p99", "ttft_mean",
        "tpot_p50", "tpot_p90", "tpot_p99",
        "e2e_p50", "e2e_p90", "e2e_p99",
        "req_per_s", "output_tokens_per_s", "total_tokens_per_s",
        "output_tokens_per_s_per_chip",
        "cache_hits", "cache_queries", "cache_hit_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r in report.get("runs", []):
            row = {
                "concurrency": r["concurrency"],
                "requests": r["requests"],
                "ok": r["ok"],
                "failed": r["failed"],
                "success_rate": r["success_rate"],
                "valid": r["valid"],
                "ttft_p50": r["ttft_ms"]["p50"],
                "ttft_p90": r["ttft_ms"]["p90"],
                "ttft_p99": r["ttft_ms"]["p99"],
                "ttft_mean": r["ttft_ms"]["mean"],
                "tpot_p50": r["tpot_ms"]["p50"],
                "tpot_p90": r["tpot_ms"]["p90"],
                "tpot_p99": r["tpot_ms"]["p99"],
                "e2e_p50": r["e2e_ms"]["p50"],
                "e2e_p90": r["e2e_ms"]["p90"],
                "e2e_p99": r["e2e_ms"]["p99"],
                "req_per_s": r["req_per_s"],
                "output_tokens_per_s": r["output_tokens_per_s"],
                "total_tokens_per_s": r["total_tokens_per_s"],
                "output_tokens_per_s_per_chip": r["output_tokens_per_s_per_chip"],
                "cache_hits": r["cache"]["hits"],
                "cache_queries": r["cache"]["queries"],
                "cache_hit_rate": r["cache"]["hit_rate"],
            }
            writer.writerow(row)


def make_charts(report: dict[str, Any], out_dir: Path) -> list[Path]:
    """画两张图（png + svg）：吞吐/TTFT vs 并发、缓存命中率 vs 并发。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [r for r in report.get("runs", []) if r.get("requests")]
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if not runs:
        return written

    xs = [r["concurrency"] for r in runs]

    # 图 1：吞吐（out tok/s）柱 + TTFT p50/p90 折线（双轴）
    fig, ax1 = plt.subplots(figsize=(8.0, 4.8), dpi=140)
    width = 0.5
    ax1.bar([str(x) for x in xs], [r["output_tokens_per_s"] for r in runs],
            width=width, color="#4C78A8", label="Output tok/s")
    ax1.set_xlabel("Concurrency (lanes)")
    ax1.set_ylabel("Output tokens / s", color="#4C78A8")
    ax1.tick_params(axis="y", labelcolor="#4C78A8")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot([str(x) for x in xs], [r["ttft_ms"]["p50"] for r in runs],
             "o-", color="#F58518", linewidth=2, label="TTFT p50")
    ax2.plot([str(x) for x in xs], [r["ttft_ms"]["p90"] for r in runs],
             "s--", color="#B23B3B", linewidth=1.6, label="TTFT p90")
    ax2.set_ylabel("TTFT (ms)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8, framealpha=0.9)
    ax1.set_title("AgentX replay: throughput & TTFT vs concurrency")
    fig.tight_layout()
    for ext in ("png", "svg"):
        p = out_dir / f"throughput_ttft_vs_concurrency.{ext}"
        fig.savefig(p, bbox_inches="tight")
        written.append(p)
    plt.close(fig)

    # 图 2：prefix cache 命中率 + 成功率/req/s 辅助线
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=140)
    rates = [r["cache"]["hit_rate"] * 100.0 for r in runs]
    bars = ax.bar([str(x) for x in xs], rates, color="#54A24B", width=0.5,
                  label="Prefix cache hit rate")
    ax.set_xlabel("Concurrency (lanes)")
    ax.set_ylabel("Cache hit rate (%)", color="#356B2E")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 1.5, f"{rate:.1f}%",
                ha="center", va="bottom", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot([str(x) for x in xs], [r["req_per_s"] for r in runs],
             "o-", color="#E45756", linewidth=2, label="req/s")
    ax2.set_ylabel("Requests / s", color="#E45756")
    ax2.tick_params(axis="y", labelcolor="#E45756")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title("AgentX replay: prefix cache hit rate vs concurrency")
    fig.tight_layout()
    for ext in ("png", "svg"):
        p = out_dir / f"cache_hit_vs_concurrency.{ext}"
        fig.savefig(p, bbox_inches="tight")
        written.append(p)
    plt.close(fig)

    return written


def write_report(report: dict[str, Any], out_dir: Path, *, charts: bool = True) -> dict[str, Any]:
    """把 report dict 落盘为 json/md/csv + 图表，返回落盘文件清单。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "replay.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = out_dir / "replay.md"
    write_markdown(report, md_path)
    csv_path = out_dir / "replay.csv"
    write_csv(report, csv_path)
    files = {"json": json_path, "md": md_path, "csv": csv_path}
    if charts:
        charts_list = make_charts(report, out_dir / "charts")
        files["charts"] = [Path(p) for p in charts_list]
    return files


# --------------------------------------------------------------------------- #
# 顶层：多并发点 sweep
# --------------------------------------------------------------------------- #


async def replay_sweep(
    traces: Sequence[dict[str, Any]],
    run: RunConfig,
    concurrencies: Sequence[int],
    *,
    out_dir: Path | None = None,
    charts: bool = True,
    generate_charts: bool = True,
    dump_requests: bool = False,
    notes: Iterable[str] | None = None,
    on_point: Any = None,
) -> dict[str, Any]:
    """对每个并发点跑一轮，汇总成 report（并可选落盘）。"""
    import httpx

    if not traces:
        raise ValueError("没有 trace 可回放")
    plans = [build_plan(t) for t in traces]
    plans = [_apply_idle_caps(p, run) for p in plans]

    # trust_env=False：Windows 上 httpx 会从注册表读到系统代理，把 127.0.0.1
    # 的请求也送进代理导致读超时（本机 8090 联调实测）。回放只打本地服务，
    # 一律绕开环境代理。
    async with httpx.AsyncClient(
        timeout=run.timeout_s,
        trust_env=False,
        limits=httpx.Limits(
            max_connections=max(concurrencies) + 8,
            max_keepalive_connections=max(concurrencies) + 8,
        ),
    ) as client:
        # 健康检查：无 /health 也继续（不同服务实现），但完全连不上要早失败
        try:
            await client.get(f"{run.url}/health", timeout=3.0)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"无法连接 {run.url}：{exc}") from exc

        runs: list[dict[str, Any]] = []
        per_request: list[dict[str, Any]] = []
        for conc in concurrencies:
            point_cfg = RunConfig(**{**run.__dict__, "concurrency": conc})
            point = await run_point(plans, point_cfg, client=client)
            runs.append(point)
            if on_point:
                on_point(point)

    report: dict[str, Any] = {
        "meta": {
            "url": run.url,
            "model": run.model,
            "traces": "agentx/traces",
            "trace_count": len(traces),
            "duration_s": run.duration_s,
            "warmup": run.warmup,
            "primer": run.primer,
            "start_ratio": [run.start_ratio_min, run.start_ratio_max],
            "system_idle_gap_cap_s": run.system_idle_gap_cap,
            "trace_idle_gap_cap_s": run.trace_idle_gap_cap,
            "seed": run.seed,
            "chips": run.chips,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "notes": list(notes or []),
        },
        "runs": runs,
    }
    if dump_requests:
        report["per_request"] = per_request

    if out_dir is not None:
        report["files"] = {
            k: (str(v) if not isinstance(v, list) else [str(x) for x in v])
            for k, v in write_report(report, out_dir, charts=charts and generate_charts).items()
        }
    return report
