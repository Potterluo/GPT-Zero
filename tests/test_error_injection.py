"""失败注入与 API Key 校验的回归测试（验收工具失败路径的 mock 支持）。"""
from __future__ import annotations

import fastapi.testclient
import pytest

from gptzero.app import create_app
from gptzero.config import ServerConfig


def _client(**overrides) -> fastapi.testclient.TestClient:
    cfg = ServerConfig(**{"host": "127.0.0.1", "port": 0, **overrides})
    from gptzero.app import create_app
    return fastapi.testclient.TestClient(create_app(cfg))


def test_error_rate_injection_deterministic():
    """error_rate=0.5：每 2 个请求注入 1 次 500，确定性可复现。"""
    client = _client(error_rate=0.5, error_status=500)
    statuses = [
        client.post("/v1/chat/completions", json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": f"msg{i}"}],
            "max_tokens": 8,
        }).status_code
        for i in range(10)
    ]
    failed = sum(1 for s in statuses if s == 500)
    ok = sum(1 for s in statuses if s == 200)
    assert failed == 5 and ok == 5


def test_error_rate_zero_means_never():
    client = _client(error_rate=0.0)
    statuses = [
        client.post("/v1/chat/completions", json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        }).status_code
        for _ in range(8)
    ]
    assert all(s == 200 for s in statuses)


def test_require_api_key_rejects_and_accepts():
    client = _client(require_api_key="sk-ucm")
    # 缺失
    r = client.post("/v1/chat/completions", json={
        "model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 401
    # 错误 key
    r = client.post("/v1/chat/completions",
                    headers={"Authorization": "Bearer wrong"},
                    json={"model": "gpt-3.5-turbo",
                          "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 401
    # 正确 key（Bearer）
    r = client.post("/v1/chat/completions",
                    headers={"Authorization": "Bearer sk-ucm"},
                    json={"model": "gpt-3.5-turbo",
                          "messages": [{"role": "user", "content": "hi"}],
                          "max_tokens": 8})
    assert r.status_code == 200
    # 正确 key（x-api-key）
    r = client.post("/v1/chat/completions",
                    headers={"x-api-key": "sk-ucm"},
                    json={"model": "gpt-3.5-turbo",
                          "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200


def test_auth_and_error_rate_independent_of_protocol():
    """三协议都受鉴权与注入约束。"""
    client = _client(require_api_key="sk-x", error_rate=1.0, error_status=503)
    for path, payload in (
        ("/v1/chat/completions", {"model": "gpt-3.5-turbo",
                                  "messages": [{"role": "user", "content": "hi"}]}),
        ("/v1/messages", {"model": "claude-3-opus",
                          "max_tokens": 8,
                          "messages": [{"role": "user", "content": "hi"}]}),
    ):
        r = client.post(path, json=payload)
        assert r.status_code == 401
        r = client.post(path, headers={"Authorization": "Bearer sk-x"}, json=payload)
        assert r.status_code == 503
    r = client.post("/v1/responses", headers={"x-api-key": "sk-x"},
                    json={"model": "gpt-3.5-turbo", "input": "hi"})
    assert r.status_code == 503
