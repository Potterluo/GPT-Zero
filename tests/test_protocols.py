"""Multi-protocol contract tests: Responses API + Anthropic Messages API.

Covers what agent loops actually parse: response object shapes, SSE event
order, fake tool calls (arguments arrive as valid JSON fragments), tool-result
follow-ups and ``previous_response_id`` chaining.  Chat Completions keeps its
own suite in ``test_server.py``.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from gptzero.app import create_app
from gptzero.config import ServerConfig


@pytest.fixture()
def client():
    cfg = ServerConfig(
        tokenizer="heuristic",
        ttft_ms=1.0,
        tpot_ms=0.0,
        jitter=0.0,
        max_concurrency=8,
        block_size=16,
        cache_blocks=256,
        default_max_tokens=32,
        tool_call_ratio=0.0,
    )
    app = create_app(cfg)
    with TestClient(app) as c:
        yield c


def _force_tools(client):
    client.app.state.gptzero.cfg.tool_call_ratio = 1.0


def _sse_frames(text: str) -> list[tuple[str | None, dict]]:
    """Parse ``event:``/``data:`` SSE pairs into (event, payload) tuples."""
    frames: list[tuple[str | None, dict]] = []
    event: str | None = None
    for line in text.splitlines():
        if line.startswith("event: "):
            event = line[7:].strip()
        elif line.startswith("data: "):
            frames.append((event, json.loads(line[6:])))
            event = None
    return frames


RESPONSES_TOOLS = [
    {
        "type": "function",
        "name": "read_file",
        "description": "Read a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }
]

ANTHROPIC_TOOLS = [
    {
        "name": "read_file",
        "description": "Read a file",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }
]


# ---------------------------------------------------------------------------
# Responses API — non-streaming
# ---------------------------------------------------------------------------


def test_responses_basic_text(client):
    resp = client.post(
        "/v1/responses",
        json={"model": "gpt-4.1", "input": "hello", "max_output_tokens": 12},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "response"
    assert body["id"].startswith("resp_")
    assert body["model"] == "gpt-4.1"
    # No EOS in the mock: filler hits the cap -> incomplete/max_output_tokens.
    assert body["status"] == "incomplete"
    assert body["incomplete_details"] == {"reason": "max_output_tokens"}
    messages = [item for item in body["output"] if item["type"] == "message"]
    assert messages and messages[0]["role"] == "assistant"
    assert messages[0]["status"] == "incomplete"
    part = messages[0]["content"][0]
    assert part["type"] == "output_text" and part["text"]
    usage = body["usage"]
    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] == 12
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
    assert usage["input_tokens_details"]["cached_tokens"] >= 0
    assert usage["output_tokens_details"]["reasoning_tokens"] == 0


def test_responses_instructions_and_input_items(client):
    resp = client.post(
        "/v1/responses",
        json={
            "model": "gpt-4.1",
            "instructions": "You are a mock.",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            ],
            "max_output_tokens": 8,
        },
    )
    body = resp.json()
    assert body["instructions"] == "You are a mock."
    assert body["usage"]["input_tokens"] > 0


def test_responses_function_call_roundtrip(client):
    _force_tools(client)
    body = client.post(
        "/v1/responses",
        json={
            "model": "gpt-4.1",
            "input": "read src/engine.py",
            "tools": RESPONSES_TOOLS,
            "tool_choice": "required",
            "max_output_tokens": 64,
        },
    ).json()
    assert body["status"] == "completed"
    assert body["incomplete_details"] is None
    calls = [item for item in body["output"] if item["type"] == "function_call"]
    assert calls, "tool_choice=required must produce a function_call item"
    call = calls[0]
    assert call["call_id"].startswith("call_")
    assert call["name"] == "read_file"
    json.loads(call["arguments"])  # arguments must be valid JSON
    assert body["usage"]["output_tokens"] > 0

    # Follow-up: send the tool output back, get plain text.
    follow = client.post(
        "/v1/responses",
        json={
            "model": "gpt-4.1",
            "input": [
                {"type": "message", "role": "user", "content": "read src/engine.py"},
                {
                    "type": "function_call",
                    "call_id": call["call_id"],
                    "name": call["name"],
                    "arguments": call["arguments"],
                },
                {"type": "function_call_output", "call_id": call["call_id"], "output": "file body"},
            ],
            "tools": RESPONSES_TOOLS,
            "tool_choice": "required",
            "max_output_tokens": 8,
        },
    ).json()
    texts = [i for i in follow["output"] if i["type"] == "message"]
    assert texts and texts[0]["content"][0]["text"]


def test_responses_named_tool_choice(client):
    _force_tools(client)
    body = client.post(
        "/v1/responses",
        json={
            "model": "gpt-4.1",
            "input": "hi",
            "tools": RESPONSES_TOOLS,
            "tool_choice": {"type": "function", "name": "read_file"},
            "max_output_tokens": 64,
        },
    ).json()
    calls = [item for item in body["output"] if item["type"] == "function_call"]
    assert calls and calls[0]["name"] == "read_file"


def test_responses_tool_choice_none(client):
    _force_tools(client)
    body = client.post(
        "/v1/responses",
        json={
            "model": "gpt-4.1",
            "input": "hi",
            "tools": RESPONSES_TOOLS,
            "tool_choice": "none",
            "max_output_tokens": 8,
        },
    ).json()
    assert not [i for i in body["output"] if i["type"] == "function_call"]


def test_responses_non_function_tools_ignored(client):
    body = client.post(
        "/v1/responses",
        json={
            "model": "gpt-4.1",
            "input": "hi",
            "tools": [{"type": "web_search"}],
            "max_output_tokens": 8,
        },
    ).json()
    assert not [i for i in body["output"] if i["type"] == "function_call"]
    assert body["tools"] == []


def test_responses_previous_response_id_chain(client):
    _force_tools(client)
    first = client.post(
        "/v1/responses",
        json={
            "model": "gpt-4.1",
            "input": "read src/engine.py",
            "tools": RESPONSES_TOOLS,
            "tool_choice": "required",
            "max_output_tokens": 64,
        },
    ).json()
    call = next(i for i in first["output"] if i["type"] == "function_call")

    # Retrieve by id.
    got = client.get(f"/v1/responses/{first['id']}")
    assert got.status_code == 200
    assert got.json()["id"] == first["id"]

    second = client.post(
        "/v1/responses",
        json={
            "model": "gpt-4.1",
            "previous_response_id": first["id"],
            "input": [
                {"type": "function_call_output", "call_id": call["call_id"], "output": "file body"}
            ],
            "tools": RESPONSES_TOOLS,
            "tool_choice": "required",
            "max_output_tokens": 8,
        },
    )
    assert second.status_code == 200
    texts = [i for i in second.json()["output"] if i["type"] == "message"]
    assert texts and texts[0]["content"][0]["text"]
    # Context chaining is visible in the growing prompt accounting.
    assert second.json()["usage"]["input_tokens"] > first["usage"]["input_tokens"]

    missing = client.get("/v1/responses/resp_does_not_exist")
    assert missing.status_code == 404

    unknown = client.post(
        "/v1/responses",
        json={"model": "gpt-4.1", "input": "hi", "previous_response_id": "resp_missing"},
    )
    assert unknown.status_code == 404


# ---------------------------------------------------------------------------
# Responses API — streaming
# ---------------------------------------------------------------------------


def test_responses_stream_text_events(client):
    resp = client.post(
        "/v1/responses",
        json={"model": "gpt-4.1", "input": "stream me", "max_output_tokens": 12, "stream": True},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    frames = _sse_frames(resp.text)
    names = [event for event, _ in frames]
    assert names[0] == "response.created"
    assert "response.in_progress" in names
    assert "response.output_item.added" in names
    assert "response.content_part.added" in names
    deltas = [data for event, data in frames if event == "response.output_text.delta"]
    assert deltas and "".join(d["delta"] for d in deltas)
    assert names[-1] == "response.incomplete"

    final = frames[-1][1]["response"]
    assert final["status"] == "incomplete"
    assert final["usage"]["output_tokens"] == 12
    # sequence_number is monotonically increasing across frames
    seqs = [data["sequence_number"] for _, data in frames]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


def test_responses_stream_function_call_deltas(client):
    _force_tools(client)
    resp = client.post(
        "/v1/responses",
        json={
            "model": "gpt-4.1",
            "input": "read src/engine.py",
            "tools": RESPONSES_TOOLS,
            "tool_choice": "required",
            "stream": True,
            "max_output_tokens": 64,
        },
    )
    frames = _sse_frames(resp.text)
    arg_deltas = [d for e, d in frames if e == "response.function_call_arguments.delta"]
    assert arg_deltas, "arguments must stream as function_call_arguments.delta frames"
    args = "".join(d["delta"] for d in arg_deltas)
    json.loads(args)
    done = [d for e, d in frames if e == "response.function_call_arguments.done"]
    assert done and done[0]["arguments"] == args
    assert frames[-1][0] == "response.completed"
    item_added = [d for e, d in frames if e == "response.output_item.added"]
    assert any(i["item"]["type"] == "function_call" for i in item_added)


# ---------------------------------------------------------------------------
# Messages API — non-streaming
# ---------------------------------------------------------------------------


def test_messages_basic_text(client):
    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-3-5-sonnet",
            "max_tokens": 12,
            "system": "You are a mock.",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["id"].startswith("msg_")
    assert body["model"] == "claude-3-5-sonnet"
    assert body["stop_reason"] == "max_tokens"
    assert body["stop_sequence"] is None
    text_blocks = [b for b in body["content"] if b["type"] == "text"]
    assert text_blocks and text_blocks[0]["text"]
    usage = body["usage"]
    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] == 12
    assert usage["cache_read_input_tokens"] >= 0


def test_messages_system_blocks_with_cache_control(client):
    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-3-5-sonnet",
            "max_tokens": 8,
            "system": [
                {
                    "type": "text",
                    "text": "long system prompt",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    body = resp.json()
    assert body["usage"]["input_tokens"] > 0


def test_messages_tool_use_roundtrip(client):
    _force_tools(client)
    body = client.post(
        "/v1/messages",
        json={
            "model": "claude-3-5-sonnet",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "read src/engine.py"}],
            "tools": ANTHROPIC_TOOLS,
            "tool_choice": {"type": "any"},
        },
    ).json()
    assert body["stop_reason"] == "tool_use"
    tool_blocks = [b for b in body["content"] if b["type"] == "tool_use"]
    assert tool_blocks
    block = tool_blocks[0]
    assert block["id"].startswith("toolu_")
    assert block["name"] == "read_file"
    assert isinstance(block["input"], dict)

    # Follow-up with a tool_result block inside a user message.
    follow = client.post(
        "/v1/messages",
        json={
            "model": "claude-3-5-sonnet",
            "max_tokens": 8,
            "messages": [
                {"role": "user", "content": "read src/engine.py"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Reading the file."},
                        {
                            "type": "tool_use",
                            "id": block["id"],
                            "name": block["name"],
                            "input": block["input"],
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": block["id"],
                            "content": [{"type": "text", "text": "file body"}],
                        }
                    ],
                },
            ],
            "tools": ANTHROPIC_TOOLS,
        },
    ).json()
    texts = [b for b in follow["content"] if b["type"] == "text"]
    assert texts and texts[0]["text"]


def test_messages_stop_sequences(client):
    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-3-5-sonnet",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hello"}],
            "stop_sequences": ["the"],
        },
    )
    body = resp.json()
    text = next(b for b in body["content"] if b["type"] == "text")["text"]
    assert "the" not in text.lower() or body["stop_reason"] == "stop_sequence"
    if body["stop_reason"] == "stop_sequence":
        assert body["stop_sequence"] is None  # mock does not echo the sequence


def test_messages_error_shape(client):
    resp = client.post("/v1/messages", content=b"{not json", headers={"content-type": "application/json"})
    assert resp.status_code == 400
    body = resp.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"


# ---------------------------------------------------------------------------
# Messages API — streaming
# ---------------------------------------------------------------------------


def test_messages_stream_text_events(client):
    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-3-5-sonnet",
            "max_tokens": 12,
            "messages": [{"role": "user", "content": "stream me"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    frames = _sse_frames(resp.text)
    names = [event for event, _ in frames]
    assert names[0] == "message_start"
    assert "ping" in names
    start = next(d for e, d in frames if e == "content_block_start")
    assert start["index"] == 0 and start["content_block"]["type"] == "text"
    deltas = [d for e, d in frames if e == "content_block_delta"]
    text = "".join(d["delta"]["text"] for d in deltas if d["delta"]["type"] == "text_delta")
    assert text
    delta_evt = next(d for e, d in frames if e == "message_delta")
    assert delta_evt["delta"]["stop_reason"] == "max_tokens"
    assert delta_evt["usage"]["output_tokens"] == 12
    assert names[-1] == "message_stop"
    # message_start carries the full message envelope with empty content
    start_msg = next(d for e, d in frames if e == "message_start")["message"]
    assert start_msg["role"] == "assistant" and start_msg["content"] == []
    assert start_msg["usage"]["input_tokens"] > 0


def test_messages_stream_tool_use_json_deltas(client):
    _force_tools(client)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-3-5-sonnet",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "read src/engine.py"}],
            "tools": ANTHROPIC_TOOLS,
            "tool_choice": {"type": "any"},
            "stream": True,
        },
    )
    frames = _sse_frames(resp.text)
    starts = [d for e, d in frames if e == "content_block_start" and d["content_block"]["type"] == "tool_use"]
    assert starts
    block = starts[0]["content_block"]
    assert block["name"] == "read_file" and block["input"] == {}
    partials = [
        d["delta"]["partial_json"]
        for e, d in frames
        if e == "content_block_delta" and d["delta"]["type"] == "input_json_delta" and d["index"] == starts[0]["index"]
    ]
    assert partials
    json.loads("".join(partials))
    stops = [d for e, d in frames if e == "content_block_stop" and d["index"] == starts[0]["index"]]
    assert stops
    delta_evt = next(d for e, d in frames if e == "message_delta")
    assert delta_evt["delta"]["stop_reason"] == "tool_use"


# ---------------------------------------------------------------------------
# Cross-protocol observability
# ---------------------------------------------------------------------------


def test_all_protocols_count_in_stats(client):
    client.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "a"}], "max_tokens": 4})
    client.post("/v1/responses", json={"model": "m", "input": "a", "max_output_tokens": 4})
    client.post("/v1/messages", json={"model": "m", "max_tokens": 4, "messages": [{"role": "user", "content": "a"}]})
    stats = client.get("/api/stats").json()
    assert stats["total_requests"] == 3
    logs = client.get("/api/logs").json()
    paths = {entry["endpoint"] for entry in logs["logs"]}
    assert {"/v1/chat/completions", "/v1/responses", "/v1/messages"} <= paths
