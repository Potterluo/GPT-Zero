"""工作负载生成器测试：确定性、上下文增长、工具闭环、长上下文规模。"""

from __future__ import annotations

import os

import pytest

os.environ["BENCH_TOKENIZER"] = "heuristic"

from bench import workload  # noqa: E402
from bench.client import RequestResult  # noqa: E402


@pytest.fixture(autouse=True)
def _heuristic_tokenizer() -> None:
    os.environ["BENCH_TOKENIZER"] = "heuristic"
    workload._COUNTER = None


def _fake_result(*, tool_calls: int = 0, text: str = "done", ok: bool = True) -> RequestResult:
    payload = None
    if tool_calls:
        payload = [
            {
                "id": "call_test",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "src/engine.py"}'},
            }
        ]
    return RequestResult(
        ok=ok,
        profile="agentic",
        label="main",
        ttft_ms=100.0,
        e2e_ms=500.0,
        tpot_ms=8.0,
        prompt_tokens=2000,
        completion_tokens=64,
        finish_reason="tool_calls" if tool_calls else "stop",
        tool_calls=tool_calls,
        tool_call_payload=payload,
        text_preview=text,
    )


class TestDeterminism:
    def test_same_seed_same_messages(self) -> None:
        a = workload.build_session("coding", 3, seed=42)
        b = workload.build_session("coding", 3, seed=42)
        turn_a = a.next_turn()
        turn_b = b.next_turn()
        assert [m["content"] for m in turn_a.specs[0].messages] == [m["content"] for m in turn_b.specs[0].messages]
        assert turn_a.specs[0].seed == turn_b.specs[0].seed

    def test_different_index_differs(self) -> None:
        a = workload.build_session("coding", 1, seed=42).next_turn().specs[0]
        b = workload.build_session("coding", 2, seed=42).next_turn().specs[0]
        assert a.messages[-1]["content"] != b.messages[-1]["content"]

    def test_payload_contract(self) -> None:
        spec = workload.build_session("coding", 0, seed=42).next_turn().specs[0]
        payload = spec.to_payload("gpt-3.5-turbo")
        assert payload["ignore_eos"] is True
        assert payload["stream_options"]["include_usage"] is True
        assert payload["max_tokens"] == workload.DEFAULT_MAX_TOKENS["coding"]


class TestChatProfile:
    def test_turn_count_and_context_range(self) -> None:
        session = workload.build_session("chat", 0, seed=7)
        assert 6 <= session.n_turns <= 10
        count = 0
        while True:
            turn = session.next_turn()
            if turn is None:
                break
            spec = turn.specs[0]
            assert spec.tools is None
            assert spec.est_prompt_tokens >= 150
            session.observe([_fake_result()])
            count += 1
        assert count == session.n_turns


class TestCodingProfile:
    def test_context_grows_monotonic_and_tool_results_backfilled(self) -> None:
        session = workload.build_session("coding", 0, seed=42)
        sizes: list[int] = []
        saw_tool_result = False
        while True:
            turn = session.next_turn()
            if turn is None:
                break
            spec = turn.specs[0]
            sizes.append(spec.est_prompt_tokens)
            assert spec.tools is not None
            assert spec.tools[0]["function"]["name"] == "read_file"
            if any(m.get("role") == "tool" for m in spec.messages):
                saw_tool_result = True
            session.observe([_fake_result()])
        assert saw_tool_result
        assert sizes[-1] > sizes[0]
        assert sizes[0] >= 1500
        assert sizes[-1] >= 8000
        # 每轮都会补到目标上下文，允许 ±30% 偏差
        assert all(s >= 1500 for s in sizes)


class TestAgenticProfile:
    def test_tool_loop_closes(self) -> None:
        session = workload.build_session("agentic", 0, seed=11)
        turn = session.next_turn()
        assert turn.kind == "user"
        assert turn.specs[0].tools is not None
        session.observe([_fake_result(tool_calls=1)])
        assert session.pending_tool is True

        follow = session.next_turn()
        assert follow is not None and follow.kind == "tool"
        last = follow.specs[0].messages[-1]
        assert last["role"] == "tool"
        assert last["tool_call_id"] == "call_test"
        session.observe([_fake_result(tool_calls=0)])
        assert session.pending_tool is False
        assert session.turn == 1

    def test_fanout_probability(self) -> None:
        subs = 0
        turns = 0
        for index in range(120):
            session = workload.build_session("agentic", index, seed=5)
            for _ in range(3):
                turn = session.next_turn()
                if turn is None:
                    break
                turns += 1
                subs += sum(1 for spec in turn.specs if spec.label == "sub")
                session.observe([_fake_result()])
        ratio = subs / max(turns, 1)
        assert 0.05 < ratio < 0.40, f"fan-out ratio out of expected band: {ratio}"


class TestLongContext:
    @pytest.mark.parametrize(("index", "expected"), [(0, 65536), (1, 131072)])
    def test_single_turn_sizes(self, index: int, expected: int) -> None:
        session = workload.build_session("longctx", index, seed=42)
        turn = session.next_turn()
        assert turn is not None
        spec = turn.specs[0]
        assert spec.tools is None
        assert spec.est_prompt_tokens >= expected * 0.9
        session.observe([_fake_result()])
        assert session.next_turn() is None
        assert session.done

    def test_short_output(self) -> None:
        spec = workload.build_session("longctx", 0, seed=42).next_turn().specs[0]
        assert spec.max_tokens == 64


class TestUnknownProfile:
    def test_rejects_unknown(self) -> None:
        with pytest.raises(ValueError):
            workload.build_session("nope", 0)
