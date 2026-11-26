"""Multi-protocol adapters: one mock engine, three wire formats.

The engine speaks a single internal representation (OpenAI chat messages +
OpenAI function tools).  Each adapter here translates one public protocol in and
out of that representation:

* **Chat Completions** — ``POST /v1/chat/completions`` (OpenAI)
* **Responses** — ``POST /v1/responses`` (OpenAI Responses API)
* **Messages** — ``POST /v1/messages`` (Anthropic Messages API)

Every adapter produces the same two artifacts the HTTP layer needs:

1. :class:`NormalizedRequest` — engine-ready fields (messages / tools /
   tool_choice / max_tokens ...),
2. response payloads — a JSON body for ``stream=false``, or an ordered list of
   SSE frames (``(event, data, pace_seconds)`` blueprints) for ``stream=true``.

Wire shapes follow the public API references; the framing is the same shape
deepseek-recipe's example server emits for its three endpoints (JSON for
``stream=false``, SSE for ``stream=true``, tool calls on every protocol).  The
adapters are pure functions: pacing, queueing and metrics stay in ``app.py``.
"""

from __future__ import annotations

import json
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def new_id(prefix: str, size: int = 24) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:size]}"


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


@dataclass
class NormalizedRequest:
    """Engine-facing view of a request, regardless of wire protocol."""

    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    tool_choice: str | dict[str, Any] | None
    parallel_tool_calls: bool
    max_tokens: int
    seed: int | None
    stop: str | list[str] | None
    stream: bool
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def force_tool(self) -> bool | None:
        """``True``/``False`` when tool_choice pins the outcome, else ``None``."""
        return resolve_force_tool(self.tool_choice)


def resolve_force_tool(tool_choice: str | dict[str, Any] | None) -> bool | None:
    if tool_choice is None or tool_choice == "auto":
        return None
    if tool_choice == "none":
        return False
    if tool_choice in ("required", "any"):
        return True
    if isinstance(tool_choice, dict):
        kind = tool_choice.get("type")
        if kind == "function" or kind == "tool":
            return True
        if kind in ("auto",):
            return None
        if kind in ("none",):
            return False
        if kind in ("any", "required"):
            return True
    return None


def _clamp_tokens(requested: Any, default: int, cap: int) -> int:
    try:
        value = int(requested)
    except (TypeError, ValueError):
        return default
    if value <= 0:
        return default
    return max(1, min(value, cap))


def _text_of_content(content: Any) -> str:
    """Best-effort text extraction from a message ``content`` value."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                for key in ("text", "output_text", "input_text", "refusal"):
                    value = part.get(key)
                    if isinstance(value, str):
                        parts.append(value)
                        break
        return "\n".join(p for p in parts if p)
    return ""


# ---------------------------------------------------------------------------
# Chat Completions (OpenAI) — the internal shape itself
# ---------------------------------------------------------------------------


def normalize_chat(body: dict[str, Any], default_tokens: int, token_cap: int) -> NormalizedRequest:
    requested = body.get("max_tokens")
    if requested is None:
        requested = body.get("max_completion_tokens")
    tools = []
    for tool in body.get("tools") or []:
        if isinstance(tool, dict) and tool.get("type", "function") == "function":
            tools.append(tool)
    return NormalizedRequest(
        model=str(body.get("model") or "gpt-3.5-turbo"),
        messages=list(body.get("messages") or []),
        tools=tools,
        tool_choice=body.get("tool_choice"),
        parallel_tool_calls=bool(body.get("parallel_tool_calls")),
        max_tokens=_clamp_tokens(requested, default_tokens, token_cap),
        seed=body.get("seed"),
        stop=body.get("stop"),
        stream=bool(body.get("stream")),
        raw=body,
    )


# ---------------------------------------------------------------------------
# Responses (OpenAI Responses API)
# ---------------------------------------------------------------------------


def _responses_items_to_messages(items: Any) -> list[dict[str, Any]]:
    """Flatten Responses ``input`` (string or item list) into chat messages."""
    if items is None:
        return []
    if isinstance(items, str):
        return [{"role": "user", "content": items}] if items else []

    messages: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            continue
        kind = item.get("type")
        if kind == "message" or (kind is None and item.get("role")):
            role = item.get("role") or "user"
            if role in ("system", "developer"):
                role = "system"
            text = _text_of_content(item.get("content"))
            messages.append({"role": role, "content": text})
        elif kind == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": item.get("call_id") or item.get("id") or new_id("call", 12),
                            "type": "function",
                            "function": {
                                "name": item.get("name", "tool"),
                                "arguments": item.get("arguments") or "{}",
                            },
                        }
                    ],
                }
            )
        elif kind == "function_call_output":
            output = item.get("output")
            if isinstance(output, (dict, list)):
                output = json_dumps(output)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id") or "",
                    "content": str(output or ""),
                }
            )
        elif kind == "reasoning" or kind == "item_reference":
            continue  # carries no client-visible text in the mock
        else:
            text = _text_of_content(item)
            if text:
                messages.append({"role": "user", "content": text})
    return messages


def _responses_tools(body: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue  # web_search / file_search / custom tools are not faked
        out.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", "tool"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                },
            }
        )
    return out


def _responses_tool_choice(body: dict[str, Any]) -> str | dict[str, Any] | None:
    choice = body.get("tool_choice")
    if isinstance(choice, dict) and choice.get("type") == "function":
        return {"type": "function", "function": {"name": choice.get("name", "tool")}}
    return choice if isinstance(choice, str) or choice is None else None


def normalize_responses(body: dict[str, Any], default_tokens: int, token_cap: int) -> NormalizedRequest:
    messages: list[dict[str, Any]] = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})
    messages.extend(_responses_items_to_messages(body.get("input")))
    return NormalizedRequest(
        model=str(body.get("model") or "gpt-4.1"),
        messages=messages,
        tools=_responses_tools(body),
        tool_choice=_responses_tool_choice(body),
        parallel_tool_calls=bool(body.get("parallel_tool_calls", True)),
        max_tokens=_clamp_tokens(body.get("max_output_tokens"), default_tokens, token_cap),
        seed=None,
        stop=None,
        stream=bool(body.get("stream")),
        raw=body,
    )


def responses_usage(prompt_tokens: int, completion_tokens: int, cached_tokens: int) -> dict[str, Any]:
    return {
        "input_tokens": prompt_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens},
        "output_tokens": completion_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": prompt_tokens + completion_tokens,
    }


def responses_response_object(
    *,
    rid: str,
    model: str,
    created_at: int,
    status: str,
    output_items: list[dict[str, Any]],
    usage: dict[str, Any] | None,
    body: dict[str, Any],
    incomplete_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tools_out = []
    for tool in body.get("tools") or []:
        if isinstance(tool, dict) and tool.get("type") == "function":
            tools_out.append(tool)
    return {
        "id": rid,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "error": None,
        "incomplete_details": incomplete_details,
        "instructions": body.get("instructions"),
        "metadata": body.get("metadata") or {},
        "model": model,
        "output": output_items,
        "parallel_tool_calls": body.get("parallel_tool_calls", True),
        "previous_response_id": body.get("previous_response_id"),
        "reasoning": {"effort": (body.get("reasoning") or {}).get("effort"), "summary": (body.get("reasoning") or {}).get("summary")},
        "store": body.get("store", True),
        "temperature": body.get("temperature", 1.0),
        "text": {"format": (body.get("text") or {}).get("format") or {"type": "text"}},
        "tool_choice": body.get("tool_choice", "auto"),
        "tools": tools_out,
        "top_p": body.get("top_p", 1.0),
        "truncation": body.get("truncation", "disabled"),
        "usage": usage,
        "user": None,
        "max_output_tokens": body.get("max_output_tokens"),
    }


def responses_output_items(
    text: str, tool_calls: list[dict[str, Any]], item_status: str
) -> list[dict[str, Any]]:
    """Build Responses ``output`` items: one message item + function_call items."""
    items: list[dict[str, Any]] = []
    if text:
        items.append(
            {
                "id": new_id("msg"),
                "type": "message",
                "status": item_status,
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    for call in tool_calls:
        items.append(
            {
                "id": new_id("fc"),
                "type": "function_call",
                "status": "completed",
                "call_id": call["id"],
                "name": call["function"]["name"],
                "arguments": call["function"]["arguments"],
            }
        )
    return items


def responses_stream_blueprint(
    *,
    rid: str,
    model: str,
    created_at: int,
    response_obj: dict[str, Any],
    chunks: list[tuple[str, int]],
    tpot_s: float,
) -> list[dict[str, Any]]:
    """Ordered SSE frames for ``POST /v1/responses`` with ``stream=true``.

    Each frame is ``{"event": name, "data": payload, "pace_s": seconds}``; the
    HTTP layer sleeps ``pace_s`` before emitting.  Event names/payloads follow
    the Responses streaming reference: ``response.created`` → per-item
    added/delta/done events → a terminal ``response.completed`` (or
    ``response.incomplete`` when the filler hit ``max_output_tokens``).
    """
    frames: list[dict[str, Any]] = []
    seq = 0

    def push(event: str, data: dict[str, Any], pace_s: float = 0.0) -> None:
        nonlocal seq
        data = {"type": event, "sequence_number": seq, **data}
        seq += 1
        frames.append({"event": event, "data": data, "pace_s": pace_s})

    in_progress = dict(response_obj)
    in_progress["status"] = "in_progress"
    in_progress["output"] = []
    in_progress["usage"] = None
    push("response.created", {"response": in_progress})
    push("response.in_progress", {"response": in_progress})

    item_messages = [
        item for item in response_obj["output"] if item["type"] == "message"
    ]
    item_calls = [
        item for item in response_obj["output"] if item["type"] == "function_call"
    ]

    # -- assistant text item -------------------------------------------------
    if item_messages:
        message_item = item_messages[0]
        running = dict(message_item)
        running["status"] = "in_progress"
        running["content"] = []
        push("response.output_item.added", {"output_index": 0, "item": running})

        part = {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
        push(
            "response.content_part.added",
            {"item_id": message_item["id"], "output_index": 0, "content_index": 0, "part": dict(part)},
        )
        for piece, n_tokens in chunks:
            push(
                "response.output_text.delta",
                {
                    "item_id": message_item["id"],
                    "output_index": 0,
                    "content_index": 0,
                    "delta": piece,
                    "logprobs": [],
                },
                pace_s=tpot_s * max(n_tokens, 1),
            )
        full_text = message_item["content"][0]["text"]
        push(
            "response.output_text.done",
            {
                "item_id": message_item["id"],
                "output_index": 0,
                "content_index": 0,
                "logprobs": [],
                "text": full_text,
            },
        )
        done_part = dict(part)
        done_part["text"] = full_text
        push(
            "response.content_part.done",
            {"item_id": message_item["id"], "output_index": 0, "content_index": 0, "part": done_part},
        )
        push("response.output_item.done", {"output_index": 0, "item": message_item})

    # -- function call items --------------------------------------------------
    for output_index, call_item in enumerate(item_calls, start=len(item_messages)):
        running = dict(call_item)
        running["status"] = "in_progress"
        push("response.output_item.added", {"output_index": output_index, "item": running})
        args = call_item["arguments"]
        step = 24
        for i in range(0, len(args), step):
            push(
                "response.function_call_arguments.delta",
                {
                    "item_id": call_item["id"],
                    "output_index": output_index,
                    "delta": args[i : i + step],
                },
                pace_s=tpot_s,
            )
        push(
            "response.function_call_arguments.done",
            {"item_id": call_item["id"], "output_index": output_index, "arguments": args},
        )
        push("response.output_item.done", {"output_index": output_index, "item": call_item})

    terminal = "response.completed" if response_obj.get("status") == "completed" else "response.incomplete"
    push(terminal, {"response": response_obj})
    return frames


# ---------------------------------------------------------------------------
# Messages (Anthropic Messages API)
# ---------------------------------------------------------------------------


def _anthropic_blocks_text(blocks: Any) -> str:
    if isinstance(blocks, str):
        return blocks
    parts = []
    if isinstance(blocks, list):
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
    return "\n".join(p for p in parts if p)


def normalize_messages(body: dict[str, Any], default_tokens: int, token_cap: int) -> NormalizedRequest:
    messages: list[dict[str, Any]] = []
    system_text = _anthropic_blocks_text(body.get("system"))
    if system_text:
        messages.append({"role": "system", "content": system_text})

    for entry in body.get("messages") or []:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role") or "user"
        content = entry.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        for block in content or []:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                text_parts.append(block.get("text", ""))
            elif kind == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id") or new_id("toolu", 24),
                        "type": "function",
                        "function": {
                            "name": block.get("name", "tool"),
                            "arguments": json_dumps(block.get("input") or {}),
                        },
                    }
                )
            elif kind == "tool_result":
                result_content = block.get("content")
                result_text = _anthropic_blocks_text(result_content)
                if isinstance(result_content, str):
                    result_text = result_content
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id") or "",
                        "content": result_text,
                        "is_error": bool(block.get("is_error")),
                    }
                )
            # thinking / image / server_tool_use / ... blocks: ignored by the mock
        if role == "assistant":
            message: dict[str, Any] = {"role": "assistant", "content": "\n".join(t for t in text_parts if t) or None}
            if tool_calls:
                message["tool_calls"] = tool_calls
            messages.append(message)
        else:
            # tool_result blocks arrive inside user messages; each becomes a
            # dedicated tool-role message, then any remaining text.
            for result in tool_results:
                tool_message = {"role": "tool", "tool_call_id": result["tool_call_id"], "content": result["content"]}
                messages.append(tool_message)
            user_text = "\n".join(t for t in text_parts if t)
            if user_text:
                messages.append({"role": "user", "content": user_text})

    tools = [
        {
            "type": "function",
            "function": {
                "name": tool.get("name", "tool"),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
            },
        }
        for tool in (body.get("tools") or [])
        if isinstance(tool, dict)
    ]
    return NormalizedRequest(
        model=str(body.get("model") or "claude-3-5-sonnet"),
        messages=messages,
        tools=tools,
        tool_choice=body.get("tool_choice"),
        parallel_tool_calls=True,
        max_tokens=_clamp_tokens(body.get("max_tokens"), default_tokens, token_cap),
        seed=None,
        stop=body.get("stop_sequences"),
        stream=bool(body.get("stream")),
        raw=body,
    )


ANTHROPIC_STOP_REASONS = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


def anthropic_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remap engine-synthesized OpenAI ``call_*`` ids to Anthropic ``toolu_`` ids.

    Tool-call ids are protocol-visible: Anthropic clients echo ``tool_use.id``
    back as ``tool_result.tool_use_id``, so the mock must hand out ids in the
    shape the protocol documents.
    """
    out: list[dict[str, Any]] = []
    for call in calls:
        remapped = dict(call)
        if not str(remapped.get("id", "")).startswith("toolu_"):
            remapped["id"] = new_id("toolu", 24)
        out.append(remapped)
    return out


def anthropic_usage(prompt_tokens: int, completion_tokens: int, cached_tokens: int) -> dict[str, Any]:
    return {
        "input_tokens": prompt_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cached_tokens,
        "output_tokens": completion_tokens,
    }


def anthropic_message_object(
    *,
    mid: str,
    model: str,
    content_blocks: list[dict[str, Any]],
    stop_reason: str | None,
    usage: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": mid,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }


def anthropic_content_blocks(text: str, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if text:
        blocks.append({"type": "text", "text": text})
    for call in tool_calls:
        try:
            parsed = json.loads(call["function"]["arguments"])
        except (ValueError, TypeError):
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {"value": parsed}
        blocks.append(
            {"type": "tool_use", "id": call["id"], "name": call["function"]["name"], "input": parsed}
        )
    return blocks


def anthropic_stream_blueprint(
    *,
    mid: str,
    model: str,
    chunks: list[tuple[str, int]],
    tool_calls: list[dict[str, Any]],
    stop_reason: str,
    usage: dict[str, Any],
    tpot_s: float,
) -> list[dict[str, Any]]:
    """Ordered SSE frames for ``POST /v1/messages`` with ``stream=true``.

    Event order follows the Messages streaming reference: ``message_start`` →
    ``ping`` → ``content_block_start/delta/stop`` per block → ``message_delta``
    → ``message_stop``.
    """
    frames: list[dict[str, Any]] = []

    def push(event: str, data: dict[str, Any], pace_s: float = 0.0) -> None:
        frames.append({"event": event, "data": {"type": event, **data}, "pace_s": pace_s})

    start_message = anthropic_message_object(
        mid=mid,
        model=model,
        content_blocks=[],
        stop_reason=None,
        usage={
            "input_tokens": usage["input_tokens"],
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": usage["cache_read_input_tokens"],
            "output_tokens": 1,
        },
    )
    push("message_start", {"message": start_message})
    push("ping", {})

    block_index = 0
    if chunks:
        push("content_block_start", {"index": block_index, "content_block": {"type": "text", "text": ""}})
        for piece, n_tokens in chunks:
            push(
                "content_block_delta",
                {"index": block_index, "delta": {"type": "text_delta", "text": piece}},
                pace_s=tpot_s * max(n_tokens, 1),
            )
        full_text = "".join(piece for piece, _ in chunks)
        push("content_block_stop", {"index": block_index})
        block_index += 1

    for call in tool_calls:
        try:
            parsed = json.loads(call["function"]["arguments"])
        except (ValueError, TypeError):
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {"value": parsed}
        push(
            "content_block_start",
            {
                "index": block_index,
                "content_block": {"type": "tool_use", "id": call["id"], "name": call["function"]["name"], "input": {}},
            },
        )
        args_json = json_dumps(parsed)
        step = 24
        for i in range(0, len(args_json), step):
            push(
                "content_block_delta",
                {"index": block_index, "delta": {"type": "input_json_delta", "partial_json": args_json[i : i + step]}},
                pace_s=tpot_s,
            )
        push("content_block_stop", {"index": block_index})
        block_index += 1

    push(
        "message_delta",
        {
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
            },
        },
    )
    push("message_stop", {})
    return frames


# ---------------------------------------------------------------------------
# Responses API state store (previous_response_id chaining + GET retrieval)
# ---------------------------------------------------------------------------


class ResponseStore:
    """In-memory ``previous_response_id`` store for the Responses protocol."""

    def __init__(self, limit: int = 512) -> None:
        self.limit = max(1, limit)
        self._responses: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._context: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()

    def put(
        self,
        rid: str,
        response_obj: dict[str, Any],
        context_messages: list[dict[str, Any]],
        output_messages: list[dict[str, Any]],
    ) -> None:
        self._responses[rid] = response_obj
        self._context[rid] = list(context_messages) + list(output_messages)
        while len(self._responses) > self.limit:
            self._responses.popitem(last=False)
        while len(self._context) > self.limit:
            self._context.popitem(last=False)

    def get(self, rid: str) -> dict[str, Any] | None:
        return self._responses.get(rid)

    def context_for(self, previous_response_id: str | None) -> list[dict[str, Any]]:
        if not previous_response_id:
            return []
        return list(self._context.get(previous_response_id) or [])

    def clear(self) -> None:
        self._responses.clear()
        self._context.clear()


def responses_output_to_messages(output_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn a stored response's ``output`` into internal assistant messages."""
    messages: list[dict[str, Any]] = []
    for item in output_items or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "message":
            text = _text_of_content(item.get("content"))
            messages.append({"role": "assistant", "content": text})
        elif kind == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": item.get("call_id") or new_id("call", 12),
                            "type": "function",
                            "function": {"name": item.get("name", "tool"), "arguments": item.get("arguments") or "{}"},
                        }
                    ],
                }
            )
    return messages


def now_ts() -> int:
    return int(time.time())
