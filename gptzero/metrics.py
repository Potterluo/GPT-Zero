"""Observability: in-memory metrics, percentile stats and Prometheus exposition.

Two consumers:

* ``GET /api/stats`` – JSON used by the dashboard and the bench harness,
* ``GET /metrics``   – Prometheus text with the vLLM metric names a real
  serving stack exposes, so ClawPerf/AIPerf-style cache-hit maths works
  unchanged against the mock.
"""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

TTFT_BUCKETS = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300]
TPOT_BUCKETS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5]
E2E_BUCKETS = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 900, 3600]


def _pct(sorted_samples: list[float], q: float) -> float:
    if not sorted_samples:
        return 0.0
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    pos = (len(sorted_samples) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return sorted_samples[int(pos)]
    return sorted_samples[lo] + (sorted_samples[hi] - sorted_samples[lo]) * (pos - lo)


class Histogram:
    def __init__(self, buckets: list[float]) -> None:
        self.buckets = sorted(buckets)
        self.counts = [0] * (len(self.buckets) + 1)  # +Inf bucket last
        self.sum = 0.0
        self.count = 0

    def observe(self, value: float) -> None:
        self.sum += value
        self.count += 1
        for i, b in enumerate(self.buckets):
            if value <= b:
                self.counts[i] += 1
                return
        self.counts[-1] += 1

    def render(self, name: str, labels: str = "", help_text: str = "") -> str:
        lines: list[str] = []
        lbl = f"{{{labels}}}" if labels else ""
        if help_text:
            lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} histogram")
        cum = 0
        for b, c in zip(self.buckets, self.counts):
            cum += c
            lines.append(f'{name}_bucket{{le="{b:g}"{"," + labels if labels else ""}}} {cum}')
        cum += self.counts[-1]
        lines.append(f'{name}_bucket{{le="+Inf"{" ," + labels if labels else ""}}} {cum}'.replace(" ,", ","))
        lines.append(f"{name}_sum{lbl} {self.sum:.6f}")
        lines.append(f"{name}_count{lbl} {self.count}")
        return "\n".join(lines)


@dataclass
class RequestRecord:
    model: str
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float
    e2e_s: float
    tpot_s: float
    finish_reason: str
    tool_calls: int = 0
    cache_hit_tokens: int = 0
    cache_query_tokens: int = 0
    queue_s: float = 0.0
    ok: bool = True
    error: str = ""
    started_at: float = field(default_factory=time.time)


class Metrics:
    def __init__(self, sample_limit: int = 20000) -> None:
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.sample_limit = max(100, sample_limit)
        self.counters: dict[str, float] = defaultdict(float)
        self.model_counters: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.tool_counters: dict[str, int] = defaultdict(int)
        self.active = 0
        self.waiting = 0
        self.max_active_seen = 0
        self.ttft_hist = Histogram(TTFT_BUCKETS)
        self.tpot_hist = Histogram(TPOT_BUCKETS)
        self.e2e_hist = Histogram(E2E_BUCKETS)
        self.ttft_samples: deque[float] = deque(maxlen=self.sample_limit)
        self.tpot_samples: deque[float] = deque(maxlen=self.sample_limit)
        self.itl_samples: deque[float] = deque(maxlen=self.sample_limit)
        self.e2e_samples: deque[float] = deque(maxlen=self.sample_limit)
        self.prompt_samples: deque[float] = deque(maxlen=self.sample_limit)
        self.completion_samples: deque[float] = deque(maxlen=self.sample_limit)
        self.success_by_reason: dict[str, int] = defaultdict(int)

    # -- lifecycle ---------------------------------------------------------
    def begin_request(self) -> None:
        with self._lock:
            self.active += 1
            self.max_active_seen = max(self.max_active_seen, self.active)

    def end_request(self) -> None:
        with self._lock:
            self.active = max(0, self.active - 1)

    def set_waiting(self, value: int) -> None:
        with self._lock:
            self.waiting = max(0, int(value))

    def record(self, rec: RequestRecord) -> None:
        with self._lock:
            self.counters["requests_total"] += 1
            if not rec.ok:
                self.counters["requests_failed"] += 1
                return
            self.counters["requests_completed"] += 1
            self.counters["prompt_tokens_total"] += rec.prompt_tokens
            self.counters["generation_tokens_total"] += rec.completion_tokens
            self.counters["cache_query_tokens_total"] += rec.cache_query_tokens
            self.counters["cache_hit_tokens_total"] += rec.cache_hit_tokens
            self.counters["tool_calls_total"] += rec.tool_calls
            self.counters["queue_seconds_total"] += rec.queue_s
            self.counters["e2e_seconds_total"] += rec.e2e_s
            self.success_by_reason[rec.finish_reason] += 1
            if rec.tool_calls:
                self.tool_counters[f"call_{rec.tool_calls}"] += 1

            self.ttft_hist.observe(rec.ttft_s)
            self.tpot_hist.observe(max(rec.tpot_s, 0.0))
            self.e2e_hist.observe(rec.e2e_s)
            self.ttft_samples.append(rec.ttft_s)
            self.tpot_samples.append(max(rec.tpot_s, 0.0))
            if rec.completion_tokens > 1:
                self.itl_samples.append((rec.e2e_s - rec.ttft_s) / (rec.completion_tokens - 1))
            self.e2e_samples.append(rec.e2e_s)
            self.prompt_samples.append(float(rec.prompt_tokens))
            self.completion_samples.append(float(rec.completion_tokens))

            mc = self.model_counters[rec.model]
            mc["requests"] += 1
            mc["prompt_tokens"] += rec.prompt_tokens
            mc["completion_tokens"] += rec.completion_tokens

    def record_tool(self, name: str, count: int = 1) -> None:
        with self._lock:
            self.tool_counters[name] += count

    # -- snapshots ---------------------------------------------------------
    @staticmethod
    def _summary(samples: deque[float], unit_scale: float = 1.0) -> dict[str, float]:
        data = sorted(samples)
        if not data:
            return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "mean": 0.0, "max": 0.0}
        return {
            "p50": round(_pct(data, 0.50) * unit_scale, 3),
            "p90": round(_pct(data, 0.90) * unit_scale, 3),
            "p99": round(_pct(data, 0.99) * unit_scale, 3),
            "mean": round(sum(data) / len(data) * unit_scale, 3),
            "max": round(data[-1] * unit_scale, 3),
        }

    def stats(self) -> dict:
        with self._lock:
            elapsed = max(time.time() - self.started_at, 1e-6)
            counters = dict(self.counters)
            completed = counters.get("requests_completed", 0)
            prompt_tokens = counters.get("prompt_tokens_total", 0)
            gen_tokens = counters.get("generation_tokens_total", 0)
            cache_q = counters.get("cache_query_tokens_total", 0)
            cache_h = counters.get("cache_hit_tokens_total", 0)
            return {
                "uptime_s": round(elapsed, 3),
                "total_requests": int(counters.get("requests_total", 0)),
                "completed_requests": int(completed),
                "failed_requests": int(counters.get("requests_failed", 0)),
                "active_requests": self.active,
                "waiting_requests": self.waiting,
                "max_active_requests": self.max_active_seen,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(gen_tokens),
                "total_tokens": int(prompt_tokens + gen_tokens),
                "tool_calls": int(counters.get("tool_calls_total", 0)),
                "rps": round(completed / elapsed, 3),
                "prompt_tokens_per_s": round(prompt_tokens / elapsed, 1),
                "completion_tokens_per_s": round(gen_tokens / elapsed, 1),
                "cache_hit_tokens": int(cache_h),
                "cache_query_tokens": int(cache_q),
                "cache_hit_rate": round(cache_h / cache_q, 4) if cache_q else 0.0,
                "ttft_ms": self._summary(self.ttft_samples, 1000.0),
                "tpot_ms": self._summary(self.tpot_samples, 1000.0),
                "itl_ms": self._summary(self.itl_samples, 1000.0),
                "e2e_ms": self._summary(self.e2e_samples, 1000.0),
                "prompt_tokens_hist": self._summary(self.prompt_samples),
                "completion_tokens_hist": self._summary(self.completion_samples),
                "finish_reasons": dict(self.success_by_reason),
                "models": {k: dict(v) for k, v in self.model_counters.items()},
                "tool_calls_by_name": dict(self.tool_counters),
            }

    def reset(self) -> None:
        with self._lock:
            self.started_at = time.time()
            self.counters.clear()
            self.model_counters.clear()
            self.tool_counters.clear()
            self.active = 0
            self.waiting = 0
            self.max_active_seen = 0
            self.ttft_hist = Histogram(TTFT_BUCKETS)
            self.tpot_hist = Histogram(TPOT_BUCKETS)
            self.e2e_hist = Histogram(E2E_BUCKETS)
            for q in (
                self.ttft_samples,
                self.tpot_samples,
                self.itl_samples,
                self.e2e_samples,
                self.prompt_samples,
                self.completion_samples,
            ):
                q.clear()
            self.success_by_reason.clear()

    # -- Prometheus --------------------------------------------------------
    def prometheus(self, cache_stats: dict[str, float] | None = None) -> str:
        with self._lock:
            c = dict(self.counters)
            active, waiting = self.active, self.waiting
            reasons = dict(self.success_by_reason)
            tools = dict(self.tool_counters)
            ttft_hist, tpot_hist, e2e_hist = self.ttft_hist, self.tpot_hist, self.e2e_hist
            stats = {
                "prompt": c.get("prompt_tokens_total", 0),
                "gen": c.get("generation_tokens_total", 0),
                "cache_q": c.get("cache_query_tokens_total", 0),
                "cache_h": c.get("cache_hit_tokens_total", 0),
            }
        cache_stats = cache_stats or {}
        used = float(cache_stats.get("blocks_cached", 0))
        cap = float(cache_stats.get("blocks_capacity", 1) or 1)
        lines: list[str] = [
            "# HELP vllm:num_requests_running Number of requests currently being served.",
            "# TYPE vllm:num_requests_running gauge",
            f"vllm:num_requests_running {active}",
            "# HELP vllm:num_requests_waiting Number of requests queued behind the concurrency limit.",
            "# TYPE vllm:num_requests_waiting gauge",
            f"vllm:num_requests_waiting {waiting}",
            "# HELP vllm:kv_cache_usage_perc Fraction of KV-cache blocks in use.",
            "# TYPE vllm:kv_cache_usage_perc gauge",
            f"vllm:kv_cache_usage_perc {min(used / cap, 1.0):.4f}",
            "# HELP vllm:prompt_tokens_total Total prompt tokens processed.",
            "# TYPE vllm:prompt_tokens_total counter",
            f'vllm:prompt_tokens_total{{model_name="mock"}} {stats["prompt"]:.0f}',
            "# HELP vllm:generation_tokens_total Total generated tokens.",
            "# TYPE vllm:generation_tokens_total counter",
            f'vllm:generation_tokens_total{{model_name="mock"}} {stats["gen"]:.0f}',
            "# HELP vllm:request_success_total Requests completed per finish reason.",
            "# TYPE vllm:request_success_total counter",
        ]
        for reason, count in sorted(reasons.items()):
            lines.append(f'vllm:request_success_total{{finished_reason="{reason}"}} {count}')
        lines += [
            "# HELP vllm:prefix_cache_queries_total Total prompt tokens queried against the prefix cache.",
            "# TYPE vllm:prefix_cache_queries_total counter",
            f'vllm:prefix_cache_queries_total{{model_name="mock",engine="0"}} {stats["cache_q"]:.0f}',
            "# HELP vllm:prefix_cache_hits_total Total prompt tokens reused from the prefix cache.",
            "# TYPE vllm:prefix_cache_hits_total counter",
            f'vllm:prefix_cache_hits_total{{model_name="mock",engine="0"}} {stats["cache_h"]:.0f}',
            "# HELP vllm:prefix_cache_evictions_total Prefix cache evictions.",
            "# TYPE vllm:prefix_cache_evictions_total counter",
            f'vllm:prefix_cache_evictions_total{{model_name="mock",engine="0"}} '
            f'{int(cache_stats.get("evictions", 0))}',
        ]
        lines.append(
            ttft_hist.render(
                "vllm:time_to_first_token_seconds",
                'model_name="mock"',
                "Time to first token in seconds.",
            )
        )
        lines.append(
            tpot_hist.render(
                "vllm:time_per_output_token_seconds",
                'model_name="mock"',
                "Time per output token in seconds.",
            )
        )
        lines.append(
            e2e_hist.render(
                "vllm:e2e_request_latency_seconds",
                'model_name="mock"',
                "End-to-end request latency in seconds.",
            )
        )
        lines += [
            "# HELP gptzero:cache_hit_tokens_total Tokens served from the prefix cache.",
            "# TYPE gptzero:cache_hit_tokens_total counter",
            f"gptzero:cache_hit_tokens_total {stats['cache_h']:.0f}",
            "# HELP gptzero:tool_calls_total Synthesized tool calls per name.",
            "# TYPE gptzero:tool_calls_total counter",
        ]
        for name, count in sorted(tools.items()):
            lines.append(f'gptzero:tool_calls_total{{name="{name}"}} {count}')
        return "\n".join(lines) + "\n"
