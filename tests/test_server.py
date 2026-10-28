"""Server contract tests: OpenAI surface, tool calls, cache, metrics, stats."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from gptzero.app import create_app
from gptzero.config import ServerConfig

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]


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


def test_health_and_models(client):
    assert client.get("/health").json()["status"] == "ok"
    models = client.get("/v1/models").json()
    assert models["object"] == "list"
    assert any(m["id"] == "gpt-3.5-turbo" for m in models["data"])


def test_chat_non_stream_exact_tokens(client):
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 12},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["usage"]["completion_tokens"] == 12
    assert body["choices"][0]["finish_reason"] in ("length", "stop")
    assert body["choices"][0]["message"]["content"]
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + 12


def test_chat_stream_sse(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "stream please"}],
            "max_tokens": 6,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    assert resp.status_code == 200
    lines = [ln for ln in resp.text.splitlines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    payloads = [json.loads(ln[6:]) for ln in lines[:-1]]
    assert payloads[0]["choices"][0]["delta"].get("role") == "assistant"
    content = "".join(p["choices"][0]["delta"].get("content", "") for p in payloads if p.get("choices"))
    assert content
    usage = [p for p in payloads if p.get("usage")]
    assert usage and usage[0]["usage"]["completion_tokens"] == 6


def test_tool_call_then_final_answer(client):
    # force tool calls by ratio=1.0
    client.app.state.gptzero.cfg.tool_call_ratio = 1.0
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "read src/engine.py"}],
            "tools": TOOLS,
            "max_tokens": 16,
        },
    )
    body = resp.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    calls = choice["message"]["tool_calls"]
    assert calls and calls[0]["function"]["name"] in ("read_file", "run_shell")
    json.loads(calls[0]["function"]["arguments"])  # arguments must be valid JSON

    follow_up = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": "read src/engine.py"},
                {"role": "assistant", "content": None, "tool_calls": calls},
                {"role": "tool", "tool_call_id": calls[0]["id"], "content": "file body"},
            ],
            "tools": TOOLS,
            "max_tokens": 8,
        },
    )
    final = follow_up.json()["choices"][0]
    assert final["finish_reason"] in ("length", "stop")
    assert final["message"]["content"]


def test_tool_call_streaming_deltas(client):
    client.app.state.gptzero.cfg.tool_call_ratio = 1.0
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "run tests"}],
            "tools": TOOLS,
            "stream": True,
            "max_tokens": 8,
        },
    )
    payloads = [json.loads(ln[6:]) for ln in resp.text.splitlines() if ln.startswith("data: {")]
    deltas = [p["choices"][0]["delta"] for p in payloads if p["choices"]]
    tool_deltas = [d for d in deltas if d.get("tool_calls")]
    assert tool_deltas
    assert any(d["tool_calls"][0].get("id") for d in tool_deltas)
    args = "".join(
        d["tool_calls"][0].get("function", {}).get("arguments", "") for d in tool_deltas
    )
    json.loads(args)


def test_prefix_cache_hits_and_metrics(client):
    long_prefix = "system prompt " * 200
    body = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": long_prefix},
            {"role": "user", "content": "continue"},
        ],
        "max_tokens": 4,
    }
    client.post("/v1/chat/completions", json=body)
    client.post("/v1/chat/completions", json=body)
    stats = client.get("/api/stats").json()
    assert stats["cache_query_tokens"] > 0
    assert stats["cache_hit_tokens"] > 0
    assert 0 < stats["cache_hit_rate"] <= 1
    metrics = client.get("/metrics").text
    assert "vllm:prefix_cache_queries_total" in metrics
    assert "vllm:prefix_cache_hits_total" in metrics
    assert "vllm:time_to_first_token_seconds_bucket" in metrics
    assert "gptzero:tool_calls_total" in metrics


def test_stats_shape(client):
    client.post(
        "/v1/chat/completions",
        json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3},
    )
    stats = client.get("/api/stats").json()
    for key in (
        "uptime_s",
        "total_requests",
        "completed_requests",
        "prompt_tokens",
        "completion_tokens",
        "rps",
        "cache_hit_rate",
        "ttft_ms",
        "tpot_ms",
        "itl_ms",
        "e2e_ms",
        "prompt_tokens_hist",
        "completion_tokens_hist",
        "models",
        "cache",
    ):
        assert key in stats, key
    for pct in ("p50", "p90", "p99", "mean", "max"):
        assert pct in stats["ttft_ms"]


def test_logs_and_reset(client):
    client.post(
        "/v1/chat/completions",
        json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "log me"}], "max_tokens": 2},
    )
    logs = client.get("/api/logs").json()
    assert logs["total"] >= 1
    entry = logs["logs"][0]
    assert entry["prompt_tokens"] > 0
    detail = client.get(f"/api/log/{entry['id']}").json()
    assert detail["id"] == entry["id"]
    assert client.post("/api/reset").json()["ok"] is True
    assert client.get("/api/stats").json()["total_requests"] == 0


def test_max_completion_tokens_alias(client):
    """AIPerf/AgentX sends ``max_completion_tokens`` instead of ``max_tokens``."""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "alias field"}],
            "max_completion_tokens": 17,
            "ignore_eos": True,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["usage"]["completion_tokens"] == 17


def test_completions_endpoint(client):
    resp = client.post("/v1/completions", json={"model": "gpt-3.5-turbo", "prompt": "hello", "max_tokens": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "text_completion"
    assert body["usage"]["completion_tokens"] == 5


def test_api_tokens(client):
    resp = client.post("/api/tokens", json={"messages": [{"role": "user", "content": "hello world"}]})
    body = resp.json()
    assert body["tokens"]["prompt_tokens"] > 0


def test_bad_json_is_400(client):
    resp = client.post("/v1/chat/completions", content=b"{not json", headers={"content-type": "application/json"})
    assert resp.status_code == 400


def test_dashboard_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "GPT-Zero" in resp.text
