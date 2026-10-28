"""WEKA trace 生成器的 schema / 语义测试。

两层校验：

1. **字段级**：用 ``tests/_weka_schema_mirror.py``（从 ``agentx-harness==1.0.7``
   逐字复制的 pydantic 模型）``model_validate``，确保真实 ``weka_trace`` loader
   能接受（``extra="forbid"``、``in``/``out`` 别名、``ttft`` 仅流式等）。
2. **语义级**：``hash_ids`` 前缀递增 + 块上界、``t`` 为秒且单调、SPAWN/JOIN 时序。
"""

from __future__ import annotations

import json

import pytest

from agentx import trace_gen as tg
from _weka_schema_mirror import WekaTrace  # tests/ 由 pytest 加入 sys.path


@pytest.fixture(scope="module")
def trace() -> dict:
    cfg = tg.GenConfig(sessions=1, requests=10, subagent_ratio=0.8, max_isl=16_000, seed=7)
    return tg.generate_session(0, cfg)


# --------------------------------------------------------------------------- #
# 1. pydantic schema 兼容（等价于真 harness 的 WekaTrace.model_validate）
# --------------------------------------------------------------------------- #


def test_validates_against_upstream_pydantic_model(trace: dict) -> None:
    model = WekaTrace.model_validate(trace)  # extra="forbid" 会拒绝任何多余字段
    assert model.id == trace["id"]
    assert model.hash_id_scope == "local"
    assert model.block_size == trace["block_size"]
    assert len(model.requests) == len(trace["requests"])


def test_json_roundtrip_preserves_schema(trace: dict) -> None:
    """loader 走 orjson 读文件，必须能吃我们写出的字节。"""
    raw = json.dumps(trace, ensure_ascii=False, separators=(",", ":"))
    assert WekaTrace.model_validate(json.loads(raw)).id == trace["id"]


def test_extra_field_is_rejected(trace: dict) -> None:
    bad = json.loads(json.dumps(trace))
    bad["unexpected"] = 1
    with pytest.raises(Exception):
        WekaTrace.model_validate(bad)


def test_call_field_names_match_upstream(trace: dict) -> None:
    """n/s 请求必须用 ``in``/``out`` 别名，字段集与上游完全一致。"""
    allowed = tg.ALLOWED_CALL_KEYS
    for req in trace["requests"]:
        if req["type"] == "subagent":
            assert set(req) <= tg.ALLOWED_SUBAGENT_KEYS
            for inner in req["requests"]:
                assert set(inner) <= allowed
                assert "input_length" not in inner
                assert "in" in inner and "out" in inner
                assert isinstance(inner["in"], int) and inner["in"] > 0
        else:
            assert set(req) <= allowed
            assert "in" in req and "out" in req


def test_streaming_request_has_ttft_normal_does_not(trace: dict) -> None:
    for req in trace["requests"]:
        if req["type"] == "subagent":
            continue
        if req["type"] == "s":
            assert "ttft" in req
        else:
            assert "ttft" not in req


# --------------------------------------------------------------------------- #
# 2. 语义约束
# --------------------------------------------------------------------------- #


def _iter_streams(trace: dict):
    """产出 (stream_name, [calls...])：主链一条，每个 subagent 一条。"""
    main = []
    subs = []
    for req in trace["requests"]:
        if req["type"] == "subagent":
            subs.append((req["agent_id"], req["requests"]))
        else:
            main.append(req)
    yield trace["id"], main
    for agent_id, calls in subs:
        yield f"{trace['id']}::{agent_id}", calls


def test_hash_ids_never_empty_and_within_input(trace: dict) -> None:
    """lead 的核心要求：每轮（含首轮）hash_ids 非空，且块前缀 <= in。

    空 hash_ids 会让 AIPerf 走 ``prompt_generator.generate()`` 随机 prompt 路径，
    与后续轮次不共享任何前缀字节 → prefix cache 曲线失真。
    """
    bs = trace["block_size"]
    for stream, calls in _iter_streams(trace):
        for i, call in enumerate(calls):
            assert call["hash_ids"], f"{stream}[{i}] hash_ids 为空"
            assert len(call["hash_ids"]) * bs <= call["in"], (
                f"{stream}[{i}] 缓存前缀 {len(call['hash_ids']) * bs} > in={call['in']}"
            )


def test_hash_ids_are_strict_prefix_increasing(trace: dict) -> None:
    for stream, calls in _iter_streams(trace):
        prev: list[int] = []
        for i, call in enumerate(calls):
            ids = call["hash_ids"]
            assert ids == sorted(set(ids)), f"{stream}[{i}] hash_ids 非严格递增"
            n = 0
            for a, b in zip(prev, ids):
                if a != b:
                    break
                n += 1
            assert n == min(len(prev), len(ids)), (
                f"{stream}[{i}] 与上一轮 LCP={n}，不是严格前缀扩展"
            )
            prev = ids


def test_cache_coverage_is_agentx_like(trace: dict) -> None:
    """整体缓存前缀占比应贴近官方的 ~98%（首轮不计入分母）。"""
    bs = trace["block_size"]
    calls = [r for r in trace["requests"] if r["type"] in ("n", "s")]
    covered = sum(len(c["hash_ids"]) * bs for c in calls)
    total = sum(c["in"] for c in calls)
    assert 0.90 <= covered / total <= 0.999


def test_t_is_seconds_monotonic(trace: dict) -> None:
    last = -1.0
    for req in trace["requests"]:
        assert req["t"] >= last - 1e-9, "外层 t 必须单调整（秒）"
        last = req["t"]
        if req["type"] == "subagent":
            assert req["duration_ms"] is not None
            for inner in req["requests"]:
                assert inner["t"] >= req["t"] - 1e-9, "子代理内层 t 应在根坐标系"
    # 量级检查：10 轮 agentic 会话在秒域，不该是毫秒数值
    assert 1.0 <= trace["totals"]["duration_s"] <= 24 * 3600


def test_join_ordering_holds(trace: dict) -> None:
    """subagent 之后的第一条父级请求不早于子代理结束（spec §4.4）。"""
    waiting: float | None = None
    for req in trace["requests"]:
        if req["type"] == "subagent":
            waiting = req["t"] + req["duration_ms"] / 1000.0
            continue
        if waiting is not None:
            assert req["t"] + 1e-6 >= waiting
            waiting = None


def test_subagent_duration_consistent_with_inner_calls(trace: dict) -> None:
    for req in trace["requests"]:
        if req["type"] != "subagent":
            continue
        max_end = max(
            (c["t"] + (c["api_time"] or 0.0) for c in req["requests"]), default=req["t"]
        )
        recorded_end = req["t"] + req["duration_ms"] / 1000.0
        assert recorded_end + 1e-6 >= max_end, "duration_ms 短于内层请求，JOIN 会提前"


# --------------------------------------------------------------------------- #
# 3. validate_trace / validate_traces
# --------------------------------------------------------------------------- #


def test_validate_trace_accepts_generated(trace: dict) -> None:
    result = tg.validate_trace(trace)
    assert result["errors"] == []


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda t: t.__setitem__("bogus", 1), "非法字段"),
        (lambda t: t["requests"][0].pop("hash_ids"), "hash_ids"),
        (lambda t: t["requests"][0].__setitem__("in", 10), "超过 in"),
        (lambda t: t["requests"][1].__setitem__("t", -5.0), "非单调"),
        (lambda t: t.__setitem__("hash_id_scope", "global"), "local"),
        (lambda t: t["requests"][0].pop("out"), "缺必填字段"),
    ],
)
def test_validate_trace_catches_violations(trace: dict, mutate, needle: str) -> None:
    bad = json.loads(json.dumps(trace))
    mutate(bad)
    errors = tg.validate_trace(bad)["errors"]
    assert errors, f"未捕获违规: {needle}"
    assert any(needle in e for e in errors), errors


def test_validate_traces_detects_duplicate_id() -> None:
    cfg = tg.GenConfig(sessions=1, requests=5, seed=1)
    a = tg.generate_session(0, cfg)
    b = json.loads(json.dumps(a))
    errors, _ = tg.validate_traces([a, b])
    assert any("重复" in e for e in errors)


# --------------------------------------------------------------------------- #
# 4. 落盘 / 读取
# --------------------------------------------------------------------------- #


def test_generate_writes_one_json_per_trace(tmp_path) -> None:
    cfg = tg.GenConfig(sessions=3, requests=6, subagent_ratio=0.0, max_isl=8000, seed=3)
    paths = tg.generate(cfg, tmp_path)
    assert len(paths) == 3
    assert all(p.suffix == ".json" for p in paths)
    traces = tg.load_traces(tmp_path)
    assert len(traces) == 3
    assert {t["id"] for t in traces} == {p.stem for p in paths}
    # AIPerf 用 path.glob("*.json") 发现文件：目录里不能有别的东西
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(p.name for p in paths)


def test_generation_is_deterministic_with_seed() -> None:
    cfg = tg.GenConfig(sessions=1, requests=8, subagent_ratio=0.5, seed=99)
    assert tg.generate_session(0, cfg) == tg.generate_session(0, cfg)
    other = tg.generate_session(0, tg.GenConfig(**{**cfg.__dict__, "seed": 100}))
    assert other != tg.generate_session(0, cfg)


def test_max_isl_bounds_input_lengths() -> None:
    cfg = tg.GenConfig(sessions=2, requests=12, max_isl=24_000, seed=5)
    for i in range(2):
        trace = tg.generate_session(i, cfg)
        for req in trace["requests"]:
            calls = req["requests"] if req["type"] == "subagent" else [req]
            for call in calls:
                assert call["in"] <= 24_000 + 1, "超过 --max-isl"
