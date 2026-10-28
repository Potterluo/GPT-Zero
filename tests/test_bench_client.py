"""SSE 解析与客户端测量测试（httpx.MockTransport，不依赖真实服务）。"""

from __future__ import annotations

import json

import httpx
import pytest

from bench.client import (
    DONE_SENTINEL,
    SSEParser,
    build_payload,
    chat_stream,
    extract_delta,
    make_client,
)


def _sse(*events: dict | str) -> bytes:
    lines = []
    for event in events:
        if isinstance(event, str):
            lines.append(f"data: {event}\n\n")
        else:
            lines.append(f"data: {json.dumps(event)}\n\n")
    return "".join(lines).encode()


def _content_chunk(text: str, finish: str | None = None) -> dict:
    return {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}]}


class TestSSEParser:
    def test_single_event(self) -> None:
        parser = SSEParser()
        events = parser.feed('data: {"choices": []}')
        assert events == []
        events = parser.feed("")
        assert events == [{"choices": []}]

    def test_done_sentinel(self) -> None:
        parser = SSEParser()
        parser.feed("data: [DONE]")
        assert parser.feed("") == [{DONE_SENTINEL: True}]

    def test_ignores_comments_and_other_fields(self) -> None:
        parser = SSEParser()
        assert parser.feed(": keep-alive") == []
        assert parser.feed("event: message") == []
        assert parser.feed("") == []

    def test_invalid_json_survives(self) -> None:
        parser = SSEParser()
        parser.feed("data: {not json")
        assert parser.feed("") == [{"__raw__": "{not json"}]

    def test_flush_without_trailing_blank(self) -> None:
        parser = SSEParser()
        parser.feed('data: {"x": 1}')
        assert parser.flush() == [{"x": 1}]


class TestExtractDelta:
    def test_role_chunk_has_no_content(self) -> None:
        norm = extract_delta({"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]})
        assert norm["content"] is None
        assert norm["tool_calls"] is None

    def test_usage_chunk(self) -> None:
        norm = extract_delta({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}})
        assert norm["usage"]["completion_tokens"] == 2

    def test_finish_reason(self) -> None:
        norm = extract_delta({"choices": [{"delta": {}, "finish_reason": "length"}]})
        assert norm["finish_reason"] == "length"
        assert norm["done"] is False


@pytest.mark.asyncio
async def test_chat_stream_measures_usage_and_itl() -> None:
    events = [
        {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
        _content_chunk("The build "),
        _content_chunk("succeeds. "),
        _content_chunk("Done.", "length"),
        {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}},
        "[DONE]",
    ]
    body = _sse(*events)
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.content.decode()
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        result = await chat_stream(client, "http://mock.local", {"stream": True}, profile="chat")

    assert seen["url"].endswith("/v1/chat/completions")
    assert result.ok, result.error
    assert result.ttft_ms is not None and result.ttft_ms >= 0
    assert result.e2e_ms is not None and result.e2e_ms >= result.ttft_ms
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 3
    assert result.finish_reason == "length"
    assert len(result.itl_ms) == 2  # 3 个内容帧 → 2 个间隔
    assert result.tpot_ms is not None
    assert result.text_preview.startswith("The build")
    assert result.stream_chunks == 3


@pytest.mark.asyncio
async def test_chat_stream_tool_calls_fragments() -> None:
    events = [
        {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "call_abc", "type": "function", "function": {"name": "read_file", "arguments": ""}}
                        ]
                    }
                }
            ]
        },
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"path": '}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"a.py"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 6}},
        "[DONE]",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse(*events))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
        result = await chat_stream(client, "http://mock.local/v1/chat/completions", {"stream": True}, profile="agentic")

    assert result.ok
    assert result.tool_calls == 1
    assert result.tool_call_payload[0]["function"]["name"] == "read_file"
    assert result.tool_call_payload[0]["function"]["arguments"] == '{"path": "a.py"}'
    assert result.finish_reason == "tool_calls"
    # 首帧 tool_calls 只有元信息，不算内容帧；两个参数片段之间才有 1 个 ITL
    assert len(result.itl_ms) == 1


@pytest.mark.asyncio
async def test_chat_stream_http_error_is_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, content=b'{"error": {"message": "bad"}}')

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
        result = await chat_stream(client, "http://mock.local", {"stream": True}, profile="chat")

    assert result.ok is False
    assert result.status == 400
    assert "HTTP 400" in result.error


@pytest.mark.asyncio
async def test_chat_stream_without_usage_falls_back_to_local_count() -> None:
    events = [
        _content_chunk("hello world from the fallback path ", "stop"),
        "[DONE]",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse(*events))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
        result = await chat_stream(client, "http://mock.local", {"stream": True}, profile="chat")

    assert result.ok
    assert result.completion_tokens is not None and result.completion_tokens > 0
    assert result.prompt_tokens is None


def test_build_payload_contract_fields() -> None:
    payload = build_payload(
        "gpt-3.5-turbo",
        [{"role": "user", "content": "hi"}],
        max_tokens=64,
        tools=[{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
        parallel_tool_calls=True,
        seed=7,
    )
    assert payload["stream"] is True
    assert payload["ignore_eos"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["max_tokens"] == 64
    assert payload["parallel_tool_calls"] is True
    assert payload["seed"] == 7


@pytest.mark.asyncio
async def test_make_client_disables_env_proxy() -> None:
    client = make_client()
    assert client.trust_env is False
    await client.aclose()
