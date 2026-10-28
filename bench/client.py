"""异步 HTTP 客户端与精确的流式延迟测量。

指标定义（与 AIPerf / vLLM 对齐）：

* ``ttft``  : 请求发出 → 首个携带 ``content`` 或 ``tool_calls`` 的 delta；
              服务端先发的 ``delta.role`` 不计入。
* ``itl``   : 相邻两个**内容帧**（content 文本或 tool_call argument 片段）的间隔。
* ``tpot``  : ``(e2e - ttft) / (completion_tokens - 1)``；completion_tokens 取
              ``stream_options.include_usage`` 返回的 usage，缺失时本地估算。
* ``e2e``   : 请求发出 → 流结束（含 ``[DONE]``）。

只用 httpx，不 import 服务端代码。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .util import estimate_tokens

DONE_SENTINEL = "__done__"


class SSEParser:
    """增量 SSE 解析器：喂入单行，吐出完整的 ``data:`` 事件。

    SSE 以空行分帧；``data:`` 可以跨多行（这里按规范用 ``\\n`` 连接）。
    ``[DONE]`` 解析为 ``{"__done__": True}``，非法 JSON 保留为 ``{"__raw__": ...}``
    以便统计而不中断压测。
    """

    def __init__(self) -> None:
        self._data: list[str] = []

    def feed(self, line: str) -> list[dict[str, Any]]:
        if line == "":
            return self.flush()
        if line.startswith(":"):
            return []
        if line.startswith("data:"):
            self._data.append(line[5:].lstrip())
        # event:/id:/retry: 对 bench 无意义，忽略。
        return []

    def flush(self) -> list[dict[str, Any]]:
        if not self._data:
            return []
        payload = "\n".join(self._data).strip()
        self._data = []
        if payload == "[DONE]":
            return [{DONE_SENTINEL: True}]
        try:
            return [json.loads(payload)]
        except json.JSONDecodeError:
            return [{"__raw__": payload}]


def extract_delta(event: dict[str, Any]) -> dict[str, Any]:
    """把一个 SSE chunk 归一化成 bench 关心的字段。"""
    out: dict[str, Any] = {
        "content": None,
        "tool_calls": None,
        "finish_reason": None,
        "usage": None,
        "done": False,
        "raw": event,
    }
    if event.get(DONE_SENTINEL):
        out["done"] = True
        return out
    usage = event.get("usage")
    if isinstance(usage, dict):
        out["usage"] = usage
    choices = event.get("choices") or []
    if not choices:
        return out
    choice = choices[0] or {}
    delta = choice.get("delta") or {}
    if isinstance(delta.get("content"), str):
        out["content"] = delta["content"]
    if delta.get("tool_calls"):
        out["tool_calls"] = delta["tool_calls"]
    if choice.get("finish_reason"):
        out["finish_reason"] = choice["finish_reason"]
    return out


@dataclass
class RequestResult:
    """单条请求的测量结果（可 JSON 序列化）。"""

    ok: bool
    profile: str = ""
    label: str = "main"
    status: int | None = None
    error: str = ""
    ttft_ms: float | None = None
    e2e_ms: float | None = None
    tpot_ms: float | None = None
    itl_ms: list[float] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    tool_calls: int = 0
    tool_call_payload: list[dict[str, Any]] | None = None
    text_preview: str = ""
    stream_chunks: int = 0
    wall_start: float = 0.0
    wall_end: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "profile": self.profile,
            "label": self.label,
            "status": self.status,
            "error": self.error,
            "ttft_ms": self.ttft_ms,
            "e2e_ms": self.e2e_ms,
            "tpot_ms": self.tpot_ms,
            "itl_ms": self.itl_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
            "tool_calls": self.tool_calls,
            "tool_call_payload": self.tool_call_payload,
            "text_preview": self.text_preview,
            "stream_chunks": self.stream_chunks,
            "wall_start": self.wall_start,
            "wall_end": self.wall_end,
        }


def _assemble_tool_calls(acc: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for index in sorted(acc):
        item = acc[index]
        if not (item.get("name") or item.get("arguments")):
            continue
        calls.append(
            {
                "id": item.get("id", ""),
                "type": "function",
                "function": {"name": item.get("name", ""), "arguments": item.get("arguments", "")},
            }
        )
    return calls


async def chat_stream(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    *,
    profile: str = "",
    label: str = "main",
) -> RequestResult:
    """发一条流式 chat 请求并测全部时延指标。``url`` 可为 base 或完整端点。"""
    endpoint = url.rstrip("/")
    if not endpoint.endswith("/v1/chat/completions"):
        endpoint += "/v1/chat/completions"
    result = RequestResult(ok=False, profile=profile, label=label)
    result.wall_start = time.time()
    t0 = time.perf_counter()
    text_parts: list[str] = []
    tool_acc: dict[int, dict[str, Any]] = {}
    ttft_s: float | None = None
    last_content_s: float | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None

    def absorb(events: list[dict[str, Any]]) -> bool:
        """处理一帧事件；返回 True 表示收到 [DONE]。"""
        nonlocal ttft_s, last_content_s, usage, finish_reason
        for event in events:
            norm = extract_delta(event)
            if norm["done"]:
                return True
            if norm["usage"]:
                usage = norm["usage"]
            if norm["finish_reason"]:
                finish_reason = norm["finish_reason"]
            has_content = bool(norm["content"])
            tool_frames = norm["tool_calls"] or []
            has_tool = bool(tool_frames)
            if not (has_content or has_tool):
                continue
            now = time.perf_counter()
            if ttft_s is None:
                ttft_s = now - t0
            for frame in tool_frames:
                idx = int(frame.get("index", 0) or 0)
                slot = tool_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if frame.get("id"):
                    slot["id"] = frame["id"]
                fn = frame.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
            # 内容帧（真正携带文本/参数）才计入 ITL；role 帧与首帧 tool_calls 元信息不算。
            content_bearing = has_content or any((f.get("function") or {}).get("arguments") for f in tool_frames)
            if content_bearing:
                if last_content_s is not None:
                    result.itl_ms.append(round((now - last_content_s) * 1000.0, 4))
                last_content_s = now
            if norm["content"]:
                text_parts.append(norm["content"])
            result.stream_chunks += 1
        return False

    try:
        async with client.stream("POST", endpoint, json=payload) as response:
            result.status = response.status_code
            if response.status_code != 200:
                body = (await response.aread()).decode("utf-8", "replace")[:400]
                result.error = f"HTTP {response.status_code}: {body}"
                result.e2e_ms = round((time.perf_counter() - t0) * 1000.0, 4)
                result.wall_end = time.time()
                return result

            parser = SSEParser()
            done = False
            async for line in response.aiter_lines():
                if done:
                    continue
                if absorb(parser.feed(line)):
                    done = True
            if not done:
                absorb(parser.flush())
        ok = True
    except httpx.HTTPError as exc:  # 连接失败、超时、协议错误
        result.error = f"{type(exc).__name__}: {exc}"
        ok = False
    except Exception as exc:  # 防御性：任何异常都要落到失败率里
        result.error = f"{type(exc).__name__}: {exc}"
        ok = False

    e2e_s = time.perf_counter() - t0
    result.e2e_ms = round(e2e_s * 1000.0, 4)
    result.wall_end = time.time()
    result.finish_reason = finish_reason
    result.text_preview = "".join(text_parts)[:500]
    result.ok = ok
    if not ok:
        return result

    calls = _assemble_tool_calls(tool_acc)
    result.tool_call_payload = calls or None
    result.tool_calls = len(calls)
    if usage:
        result.prompt_tokens = int(usage.get("prompt_tokens") or 0) or None
        result.completion_tokens = int(usage.get("completion_tokens") or 0)
    else:
        # 回退：本地 token 估算（按服务端 heuristic 规则）。
        local = estimate_tokens("".join(text_parts))
        for call in calls:
            fn = call["function"]
            local += estimate_tokens(fn.get("name", "")) + estimate_tokens(fn.get("arguments", ""))
        result.completion_tokens = local
        result.prompt_tokens = None

    if ttft_s is not None:
        result.ttft_ms = round(ttft_s * 1000.0, 4)
    ct = result.completion_tokens
    if result.ttft_ms is not None and ct is not None and ct > 0:
        decode_window = max(e2e_s - ttft_s, 0.0)
        result.tpot_ms = round(decode_window / max(ct - 1, 1) * 1000.0, 4)
    return result


def build_payload(
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    tools: list[dict[str, Any]] | None = None,
    parallel_tool_calls: bool = False,
    seed: int | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": int(max_tokens),
        "ignore_eos": True,  # 契约 §3.1：精确生成 max_tokens，压测可复现
        "stream_options": {"include_usage": True},
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
        if parallel_tool_calls:
            payload["parallel_tool_calls"] = True
    if seed is not None:
        payload["seed"] = int(seed)
    return payload


def make_client(timeout_s: float = 300.0, max_connections: int = 256) -> httpx.AsyncClient:
    """创建压测客户端。

    ``trust_env=False`` 很关键：本机存在系统级 HTTP 代理时，httpx 会把
    ``127.0.0.1`` 的请求也发给代理并得到 502（实测踩到过）。
    """
    limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections)
    timeout = httpx.Timeout(connect=10.0, read=timeout_s, write=60.0, pool=60.0)
    return httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        trust_env=False,
        headers={"Authorization": "Bearer bench", "Content-Type": "application/json"},
    )
