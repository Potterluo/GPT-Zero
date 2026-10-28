"""聚合函数测试：分位数、空样本、单样本、失败率、per-profile 汇总。"""

from __future__ import annotations

import math

import pytest

from bench.aggregate import aggregate_records, cache_hit_rate, percentile, summarize


class TestPercentile:
    def test_empty_returns_none(self) -> None:
        assert percentile([], 50) is None
        assert summarize([])["p90"] is None

    def test_single_sample(self) -> None:
        assert percentile([7.5], 50) == 7.5
        assert percentile([7.5], 99) == 7.5
        s = summarize([7.5])
        assert s["count"] == 1
        assert s["p50"] == s["p99"] == s["max"] == 7.5

    def test_known_interpolation(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        # numpy 默认 linear: p50 = 2.5, p90 = 3.7
        assert percentile(values, 50) == pytest.approx(2.5)
        assert percentile(values, 90) == pytest.approx(3.7)
        assert percentile(values, 0) == 1.0
        assert percentile(values, 100) == 4.0

    def test_ignores_none_and_nan(self) -> None:
        s = summarize([1.0, None, float("nan"), 3.0])
        assert s["count"] == 2
        assert s["p50"] == pytest.approx(2.0)

    def test_unsorted_input(self) -> None:
        assert percentile([10, 1, 5], 50) == 5


class TestSummarize:
    def test_fields_present(self) -> None:
        s = summarize([1, 2, 3, 4, 5])
        for key in ("count", "p50", "p90", "p95", "p99", "mean", "max", "min"):
            assert key in s
        assert s["mean"] == pytest.approx(3.0)
        assert s["min"] == 1 and s["max"] == 5

    def test_cache_hit_rate(self) -> None:
        assert cache_hit_rate(50, 100) == pytest.approx(0.5)
        assert cache_hit_rate(0, 0) is None
        assert cache_hit_rate(None, 100) is None
        assert cache_hit_rate(0, 100) == 0.0


def _record(
    *,
    ok: bool = True,
    profile: str = "coding",
    ttft: float | None = 100.0,
    tpot: float | None = 8.0,
    e2e: float | None = 1000.0,
    itl: list[float] | None = None,
    prompt: int = 1000,
    completion: int = 100,
    error: str = "",
) -> dict:
    return {
        "ok": ok,
        "profile": profile,
        "label": "main",
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "e2e_ms": e2e,
        "itl_ms": itl if itl is not None else [8.0, 8.5, 9.0],
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "tool_calls": 0,
        "error": error,
    }


class TestAggregate:
    def test_rates_and_failures(self) -> None:
        records = [
            _record(ttft=100, e2e=1000),
            _record(ttft=200, e2e=2000),
            _record(ok=False, error="ReadTimeout: boom", ttft=None, e2e=3000),
        ]
        agg = aggregate_records(records, wall_s=10.0)
        assert agg["requests"] == {
            "total": 3,
            "success": 2,
            "failed": 1,
            "failure_rate": pytest.approx(1 / 3),
            "sub_requests": 0,
            "tool_call_requests": 0,
        }
        assert agg["rates"]["rps"] == pytest.approx(0.3)
        assert agg["rates"]["success_rps"] == pytest.approx(0.2)
        assert agg["tokens"]["prompt"] == 2000
        assert agg["tokens"]["completion"] == 200
        assert agg["rates"]["output_tokens_per_s"] == pytest.approx(20.0)
        # 失败请求不进入延迟分位数
        assert agg["summary"]["ttft_ms"]["count"] == 2
        assert agg["summary"]["ttft_ms"]["max"] == 200
        assert agg["errors"][0]["error"].startswith("ReadTimeout")

    def test_zero_wall_does_not_crash(self) -> None:
        agg = aggregate_records([_record()], wall_s=0.0)
        assert agg["rates"]["rps"] is None
        assert agg["rates"]["output_tokens_per_s"] is None

    def test_empty_records(self) -> None:
        agg = aggregate_records([], wall_s=5.0)
        assert agg["requests"]["total"] == 0
        assert agg["requests"]["failure_rate"] is None
        assert agg["summary"]["ttft_ms"]["p50"] is None
        assert agg["samples"] == {"ttft_ms": [], "e2e_ms": []}

    def test_per_profile_and_itl_frames(self) -> None:
        records = [
            _record(profile="chat", ttft=50, itl=[10.0, 10.0]),
            _record(profile="coding", ttft=150, itl=[8.0]),
            _record(profile="chat", ttft=60, itl=[12.0]),
        ]
        agg = aggregate_records(records, wall_s=2.0)
        assert set(agg["per_profile"]) == {"chat", "coding"}
        chat = agg["per_profile"]["chat"]
        assert chat["requests"]["total"] == 2
        assert chat["summary"]["ttft_ms"]["p50"] == pytest.approx(55.0)
        # ITL 按帧汇总：chat 有 2+1 帧
        assert chat["summary"]["itl_ms"]["count"] == 3
        assert chat["samples"]["ttft_ms"] == [50.0, 60.0]

    def test_samples_are_sorted(self) -> None:
        agg = aggregate_records([_record(ttft=300), _record(ttft=100), _record(ttft=200)], 3.0)
        assert agg["samples"]["ttft_ms"] == [100.0, 200.0, 300.0]

    def test_finite_guard(self) -> None:
        agg = aggregate_records([_record(ttft=float("inf")), _record(ttft=100.0)], 1.0)
        assert agg["summary"]["ttft_ms"]["count"] == 1
        assert math.isfinite(agg["summary"]["ttft_ms"]["p50"])
