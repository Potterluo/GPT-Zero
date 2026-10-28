"""回放器 / 报告 / CLI 的单元测试（不需要起服务）。

真实 HTTP 联调不走这里：设 ``AGENTX_TEST_URL=http://127.0.0.1:8092`` 才会跑
``test_live_replay_optional``。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agentx import cli, replay, trace_gen

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# --------------------------------------------------------------------------- #
# trace -> plan（SPAWN / JOIN 结构）
# --------------------------------------------------------------------------- #


def _mini_trace() -> dict:
    """手写最小 trace：turn0 → subagent → turn1(JOIN) → turn2。"""
    return {
        "id": "mini-000",
        "models": ["m"],
        "block_size": 64,
        "hash_id_scope": "local",
        "tool_tokens": 0,
        "system_tokens": 0,
        "requests": [
            {"t": 0.0, "type": "s", "model": "m", "in": 1280, "out": 10,
             "hash_ids": [0, 1, 2], "api_time": 1.0},
            {
                "t": 2.0, "type": "subagent", "agent_id": "agent_001",
                "subagent_type": "Explore", "duration_ms": 5000,
                "total_tokens": 100, "tool_use_count": 2, "status": "completed",
                "requests": [
                    {"t": 2.1, "type": "s", "model": "m", "in": 1920, "out": 10,
                     "hash_ids": [10, 11, 12], "api_time": 0.5},
                    {"t": 5.0, "type": "s", "model": "m", "in": 2560, "out": 10,
                     "hash_ids": [10, 11, 12, 13], "api_time": 0.5},
                ],
                "models": ["m"], "tool_tokens": 0, "system_tokens": 0,
            },
            {"t": 7.0, "type": "s", "model": "m", "in": 3200, "out": 10,
             "hash_ids": [0, 1, 2, 3], "api_time": 1.0},
        ],
    }


def test_build_plan_main_chain() -> None:
    plan = replay.build_plan(_mini_trace())
    assert [c.t for c in plan.main] == [0.0, 7.0]
    assert plan.block_size == 64
    assert plan.span_s >= 7.0


def test_build_plan_spawn_and_join() -> None:
    plan = replay.build_plan(_mini_trace())
    # subagent 在 turn0 之后 spawn，在 turn1 之前 join
    assert 0 in plan.spawns_after
    assert [s.agent_id for s in plan.spawns_after[0]] == ["agent_001"]
    assert 1 in plan.joins_before
    sub = plan.joins_before[1][0]
    assert sub.spawn_t == 2.0
    assert sub.end_t == 7.0  # t + duration_ms/1000
    assert [c.t for c in sub.calls] == [2.1, 5.0]
    assert all(c.kind == "subagent" for c in sub.calls)


def test_build_plan_subagent_without_duration_falls_back_to_inner_end() -> None:
    trace = _mini_trace()
    trace["requests"][1]["duration_ms"] = None
    plan = replay.build_plan(trace)
    assert plan.joins_before[1][0].end_t == pytest.approx(5.5)


def test_idle_gap_cap_clamps_long_silence() -> None:
    trace = _mini_trace()
    trace["requests"][-1]["t"] = 999.0
    plan = replay.build_plan(trace)
    run = replay.RunConfig(system_idle_gap_cap=10.0, trace_idle_gap_cap=300.0)
    replay._apply_idle_caps(plan, run)
    # turn0 api_time=1.0 → idle gap 从 1.0 起算，被夹到 1.0+10.0
    assert plan.main[1].t == pytest.approx(11.0)


# --------------------------------------------------------------------------- #
# cache-bust marker（与 aiperf/timing/strategies/cache_bust.py 同构）
# --------------------------------------------------------------------------- #


def test_cache_bust_marker_shape_and_determinism() -> None:
    a = replay.build_cache_bust_marker("bench", 0, 3, "trace-1")
    b = replay.build_cache_bust_marker("bench", 0, 3, "trace-1")
    assert a == b
    assert a.startswith("[rid:") and a.endswith("]\n\n")
    assert len(a) == len("[rid:") + 12 + len("]\n\n")


def test_cache_bust_marker_changes_with_recycle_pass() -> None:
    """同一 trace 被第二个 lane 复用时必须换 marker，否则会跨 lane 偷热。"""
    a = replay.build_cache_bust_marker("bench", 0, 0, "t")
    b = replay.build_cache_bust_marker("bench", 1, 0, "t")
    assert a != b


def test_cache_bust_marker_matches_upstream_digest() -> None:
    """digest = sha256(f"{benchmark_id}:{pass}:{trajectory_index}:{trace_id}")[:12]。"""
    import hashlib

    expected = hashlib.sha256("bench:0:3:trace-1".encode()).hexdigest()[:12]
    assert replay.build_cache_bust_marker("bench", 0, 3, "trace-1") == f"[rid:{expected}]\n\n"


# --------------------------------------------------------------------------- #
# prompt 合成
# --------------------------------------------------------------------------- #


def test_prompt_prefix_is_shared_for_shared_block_ids() -> None:
    ids = list(range(20))
    p1 = replay.build_prompt(ids, 20 * 64, 64, "seed-a")
    p2 = replay.build_prompt(ids[:10], 10 * 64, 64, "seed-b")
    assert p1.startswith(p2)


def test_prompt_extends_with_fresh_tail() -> None:
    ids = list(range(10))
    p_short = replay.build_prompt(ids, 10 * 64, 64, "same")
    p_long = replay.build_prompt(ids, 10 * 64 + 500, 64, "same")
    assert p_long.startswith(p_short)
    assert len(p_long) > len(p_short)


def test_prompt_token_scale_is_roughly_target() -> None:
    """1 词 ≈ 1.05 token 的标定；1800 token 的块组应落在 1500~2200 词。"""
    text = replay.build_prompt(list(range(0)), 1800, 64, "x")
    n_words = len(text.split())
    assert 1500 <= n_words <= 2200


def test_prompt_deterministic() -> None:
    ids = [1, 2, 3]
    assert replay.build_prompt(ids, 1000, 64, "s") == replay.build_prompt(ids, 1000, 64, "s")


# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #


def test_percentile_matches_numpy_style() -> None:
    data = [1.0, 2.0, 3.0, 4.0]
    assert replay.percentile(data, 50) == pytest.approx(2.5)
    assert replay.percentile(data, 0) == pytest.approx(1.0)
    assert replay.percentile(data, 100) == pytest.approx(4.0)
    assert replay.percentile([], 90) == 0.0


def _rec(lane: int, start: float, ttft: float | None, e2e: float, ok: bool = True) -> replay.RequestRecord:
    return replay.RequestRecord(
        lane=lane, trace_id="t", kind="main", t=start, start_s=start,
        ttft_ms=ttft, e2e_ms=e2e, tpot_ms=(e2e - ttft) / 9 if ttft else None,
        prompt_tokens=100, completion_tokens=10, ok=ok, error=None if ok else "boom",
    )


def test_summarize_point_excludes_warmup_and_computes_rates() -> None:
    records = [
        _rec(0, 0.0, 50.0, 500.0),   # warmup (primer)
        _rec(0, 1.0, 50.0, 500.0),   # warmup
        _rec(0, 2.0, 100.0, 1000.0),  # measured
        _rec(0, 3.0, 200.0, 2000.0),  # measured
    ]
    before = {"hits": 0.0, "queries": 0.0, "source": 1.0}
    after = {"hits": 900.0, "queries": 1000.0, "source": 1.0}
    run = replay.RunConfig(concurrency=1, duration_s=10.0, chips=2)
    out = replay.summarize_point(records, 10.0, before, after, run, warmup_per_lane=2)
    assert out["requests"] == 2
    assert out["warmup_requests"] == 2
    assert out["ttft_ms"]["p50"] == pytest.approx(150.0)
    assert out["cache"]["hit_rate"] == pytest.approx(0.9)
    assert out["output_tokens_per_s"] == pytest.approx(2.0)
    assert out["output_tokens_per_s_per_chip"] == pytest.approx(1.0)
    assert out["valid"] is True


def test_summarize_point_marks_invalid_on_failures() -> None:
    records = [_rec(0, float(i), 100.0, 1000.0, ok=(i < 5)) for i in range(10)]
    before = {"hits": 0.0, "queries": 0.0, "source": 1.0}
    after = {"hits": 0.0, "queries": 0.0, "source": 1.0}
    run = replay.RunConfig(concurrency=1)
    out = replay.summarize_point(records, 5.0, before, after, run, warmup_per_lane=0)
    assert out["success_rate"] == pytest.approx(0.5)
    assert out["valid"] is False
    assert out["errors"]


# --------------------------------------------------------------------------- #
# 报告 / 图表
# --------------------------------------------------------------------------- #


def _report() -> dict:
    runs = []
    for conc, ttft, tps, hit in ((1, 200.0, 100.0, 0.95), (2, 260.0, 190.0, 0.97)):
        runs.append({
            "concurrency": conc, "duration_s": 10.0, "requests": 20, "warmup_requests": 2,
            "ok": 20, "failed": 0, "success_rate": 1.0, "valid": True,
            "ttft_ms": {"p50": ttft, "p90": ttft * 1.2, "p99": ttft * 1.4, "mean": ttft, "max": ttft * 2},
            "tpot_ms": {"p50": 8.0, "p90": 9.0, "p99": 10.0, "mean": 8.5, "max": 12.0},
            "e2e_ms": {"p50": 3000.0, "p90": 3500.0, "p99": 4000.0, "mean": 3100.0, "max": 5000.0},
            "req_per_s": 2.0, "output_tokens_per_s": tps, "total_tokens_per_s": tps * 4,
            "output_tokens_per_s_per_chip": None, "prompt_tokens": 1000,
            "completion_tokens": 500,
            "cache": {"hits": 950, "queries": 1000, "hit_rate": hit, "source": "metrics"},
            "errors": [],
        })
    return {
        "meta": {"url": "http://x", "model": "m", "traces": "agentx/traces",
                 "trace_count": 2, "duration_s": 10.0, "warmup": 10, "primer": True,
                 "generated_at": "now", "notes": ["n1"]},
        "runs": runs,
    }


def test_write_report_creates_all_artifacts(tmp_path: Path) -> None:
    files = replay.write_report(_report(), tmp_path)
    for key in ("json", "md", "csv"):
        assert Path(files[key]).exists(), key
    md = Path(files["md"]).read_text(encoding="utf-8")
    assert "AgentX 闭环回放报告" in md
    assert "缓存命中率" in md
    csv_text = Path(files["csv"]).read_text(encoding="utf-8")
    assert csv_text.splitlines()[0].startswith("concurrency,")
    assert len(csv_text.splitlines()) == 3
    charts = [Path(p) for p in files["charts"]]
    names = {p.name for p in charts}
    assert {"throughput_ttft_vs_concurrency.png", "throughput_ttft_vs_concurrency.svg",
            "cache_hit_vs_concurrency.png", "cache_hit_vs_concurrency.svg"} == names
    for p in charts:
        assert p.stat().st_size > 1000, p


def test_report_roundtrip_from_json(tmp_path: Path) -> None:
    replay.write_report(_report(), tmp_path, charts=False)
    data = json.loads((tmp_path / "replay.json").read_text(encoding="utf-8"))
    again = replay.write_report(data, tmp_path / "again", charts=False)
    assert Path(again["md"]).exists()


def test_make_charts_handles_empty_runs(tmp_path: Path) -> None:
    assert replay.make_charts({"runs": []}, tmp_path) == []


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_generate_then_validate(tmp_path: Path) -> None:
    out = tmp_path / "traces"
    assert cli.main(["generate", "--sessions", "2", "--requests", "5",
                     "--subagents", "0.5", "--max-isl", "8000", "--out", str(out)]) == 0
    assert len(list(out.glob("*.json"))) == 2
    assert cli.main(["validate", "--traces", str(out)]) == 0


def test_cli_validate_fails_on_broken_trace(tmp_path: Path) -> None:
    out = tmp_path / "traces"
    cli.main(["generate", "--sessions", "1", "--requests", "4", "--max-isl", "8000",
              "--out", str(out)])
    path = next(out.glob("*.json"))
    data = json.loads(path.read_text(encoding="utf-8"))
    data["requests"][0]["hash_ids"] = []
    path.write_text(json.dumps(data), encoding="utf-8")
    assert cli.main(["validate", "--traces", str(out)]) == 1


def test_cli_report_rebuilds_from_json(tmp_path: Path) -> None:
    replay.write_report(_report(), tmp_path)
    assert cli.main(["report", "--json", str(tmp_path / "replay.json"),
                     "--out", str(tmp_path), "--no-charts"]) == 0


@pytest.mark.skipif(
    not os.environ.get("AGENTX_TEST_URL"),
    reason="需要真实 mock 服务：设 AGENTX_TEST_URL=http://127.0.0.1:8092",
)
def test_live_replay_optional() -> None:  # pragma: no cover - 仅手动联调
    url = os.environ["AGENTX_TEST_URL"]
    code = cli.main([
        "replay", "--traces", "agentx/traces", "--url", url,
        "--concurrency", "1", "--duration", "5", "--warmup", "1",
        "--out", "results/agentx_test",
    ])
    assert code in (0, 1)


# --------------------------------------------------------------------------- #
# Windows GBK 控制台兼容（Regression: 'gbk' codec can't encode '\u2713'）
# --------------------------------------------------------------------------- #


def test_cli_source_is_gbk_safe() -> None:
    """源文件里不能出现 GBK 编不出来的字符。

    曾经 `✓`/`✗`/`⚠` 让 `python -m agentx validate` 在最后一行抛
    ``UnicodeEncodeError: 'gbk' codec can't encode character '\\u2713'``；
    CI 与脚本依赖退出码，必须回归防护。
    """
    root = Path(__file__).resolve().parent.parent / "agentx"
    for path in sorted(root.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        bad = set()
        for ch in text:
            if ord(ch) < 128:
                continue
            try:
                ch.encode("gbk")
            except UnicodeEncodeError:
                bad.add(f"U+{ord(ch):04X}")
        assert not bad, f"{path.name} 含 GBK 无法编码的字符: {sorted(bad)}"


def test_cli_validate_survives_gbk_stdout(monkeypatch, tmp_path: Path) -> None:
    """把 stdout 换成严格 GBK 编码后，validate 仍应返回 0 而不是抛编码异常。"""
    import io
    import sys

    out = tmp_path / "traces"
    cli.main(["generate", "--sessions", "1", "--requests", "4", "--max-isl", "8000",
              "--out", str(out)])

    buf = io.TextIOWrapper(io.BytesIO(), encoding="gbk", errors="strict")
    monkeypatch.setattr(sys, "stdout", buf)
    try:
        code = cli.main(["validate", "--traces", str(out)])
        buf.flush()
    finally:
        monkeypatch.undo()
    assert code == 0


# --------------------------------------------------------------------------- #
# 计划构造与生成器的一致性
# --------------------------------------------------------------------------- #


def test_plan_from_generated_trace_matches_trace() -> None:
    cfg = trace_gen.GenConfig(sessions=1, requests=8, subagent_ratio=0.8, max_isl=12_000, seed=11)
    trace = trace_gen.generate_session(0, cfg)
    n_main = sum(1 for r in trace["requests"] if r["type"] in ("n", "s"))
    n_sub = sum(1 for r in trace["requests"] if r["type"] == "subagent")
    plan = replay.build_plan(trace)
    assert len(plan.main) == n_main
    assert sum(len(v) for v in plan.joins_before.values()) == n_sub
    assert sum(len(v) for v in plan.spawns_after.values()) == n_sub
    # 所有 join 点都必须有对应的 spawn 点
    assert set(plan.joins_before) == {i + 1 for i in plan.spawns_after}
