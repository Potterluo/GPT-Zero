"""FastAPI application: multi-protocol mock LLM surface + observability + dashboard.

Design notes
------------
* The transport is where queueing lives: a semaphore caps in-flight requests and
  the time spent waiting for a slot shows up in the client's TTFT, exactly like
  a saturated inference server.
* Every request is logged (ring buffer + optional JSONL) and every completed
  request feeds the metrics registry, so ``/api/stats`` and ``/metrics`` agree.
* Three protocol adapters share one engine: OpenAI Chat Completions, the OpenAI
  Responses API and the Anthropic Messages API (see ``protocols.py``).  Tool
  calls arrive in each protocol's native streaming shape — OpenAI
  ``delta.tool_calls`` fragments, Responses ``function_call_arguments`` deltas,
  Anthropic ``input_json_delta`` blocks — which is what agent loops parse.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import __version__
from .config import ServerConfig, config_as_dict
from .engine import GenerationPlan, MockEngine
from .metrics import Metrics, RequestRecord
from .models import match_model, normalize_model_name, scan_model_directory
from .protocols import (
    ResponseStore,
    anthropic_content_blocks,
    anthropic_message_object,
    anthropic_stream_blueprint,
    anthropic_tool_calls,
    anthropic_usage,
    normalize_messages,
    normalize_responses,
    new_id,
    responses_output_items,
    responses_output_to_messages,
    responses_response_object,
    responses_stream_blueprint,
    responses_usage,
)
from .tokenizer import CountingTokenizer, HFTokenCodec

SSE_DONE = "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "gpt-3.5-turbo"
    messages: list[dict[str, Any]] = Field(default_factory=list)
    stream: bool = False
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    min_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool = False
    stream_options: dict[str, Any] | None = None
    ignore_eos: bool = False
    temperature: float | None = None
    top_p: float | None = None
    user: str | None = None


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "gpt-3.5-turbo"
    prompt: str | list[str] = ""
    stream: bool = False
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    stream_options: dict[str, Any] | None = None
    ignore_eos: bool = False


# ---------------------------------------------------------------------------
# Request log
# ---------------------------------------------------------------------------


@dataclass
class RequestLogStore:
    limit: int = 1000
    path: str = ""
    entries: list[dict[str, Any]] = field(default_factory=list)
    seq: int = 0

    def add(self, entry: dict[str, Any]) -> dict[str, Any]:
        self.seq += 1
        entry["id"] = self.seq
        entry["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.entries.append(entry)
        if len(self.entries) > max(1, self.limit):
            del self.entries[: len(self.entries) - self.limit]
        if self.path:
            try:
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except OSError:
                pass
        return entry

    def reversed(self) -> list[dict[str, Any]]:
        return list(reversed(self.entries))


@dataclass
class AppState:
    cfg: ServerConfig
    tokenizer: CountingTokenizer
    engine: MockEngine
    metrics: Metrics
    logs: RequestLogStore
    sem: asyncio.Semaphore | None = None
    waiting: int = 0
    extra_tokenizers: dict[str, Any] = field(default_factory=dict)
    responses: ResponseStore = field(default_factory=ResponseStore)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_max_tokens(data: ChatRequest | CompletionRequest, cfg: ServerConfig) -> int:
    # OpenAI's newer field is ``max_completion_tokens``; benchmark harnesses
    # send it for chat requests, so treat it as an alias of ``max_tokens``.
    requested = data.max_tokens
    if requested is None:
        requested = getattr(data, "max_completion_tokens", None)
    if requested is None:
        requested = cfg.default_max_tokens
    requested = int(requested)
    if requested <= 0:
        requested = cfg.default_max_tokens
    # ``ignore_eos`` means "generate exactly this many tokens" for AIPerf; a
    # client asking for more than the cap still gets a bounded answer.
    return max(1, min(requested, cfg.max_tokens_cap))


def _apply_stop(text: str, stop: str | list[str] | None) -> tuple[str, bool]:
    if not stop:
        return text, False
    stops = [stop] if isinstance(stop, str) else [s for s in stop if s]
    cut = len(text)
    for s in stops:
        idx = text.find(s)
        if 0 <= idx < cut:
            cut = idx
    return text[:cut], cut < len(text)


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _sse_event(event: str | None, payload: dict[str, Any]) -> str:
    """One SSE frame, optionally with a named event (Responses / Messages)."""
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _openai_error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": "invalid_request_error", "param": None, "code": None}},
    )


def _anthropic_error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"type": "error", "error": {"type": "invalid_request_error", "message": message}},
    )


async def _pace(delay_s: float) -> None:
    """Sleep at least ``delay_s``, tolerating spurious early wakeups.

    On Windows the Proactor loop's timer resolution (~15.6 ms) can make short
    ``asyncio.sleep`` calls return early when the loop is busy, which would
    collapse decode pacing and corrupt TPOT measurements.  Re-checking against
    an absolute deadline keeps the simulated TPOT honest at the cost of a cheap
    loop.
    """
    if delay_s <= 0:
        return
    deadline = time.perf_counter() + delay_s
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        await asyncio.sleep(remaining)


def _usage(prompt: int, completion: int, cached: int = 0) -> dict[str, Any]:
    """OpenAI usage block, including prompt-cache details.

    Benchmark harnesses read ``usage.prompt_tokens_details.cached_tokens`` to
    compute cache-hit metrics; without it a harness reports "no prompt-cache
    read tokens were seen" and falls back to scraping Prometheus.
    """
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": int(cached)},
    }


def _chunk(
    cid: str, model: str, created: int, delta: dict[str, Any] | None = None, finish: str | None = None
) -> dict[str, Any]:
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
    }


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(cfg: ServerConfig) -> FastAPI:
    tokenizer = CountingTokenizer(cfg.tokenizer)
    engine = MockEngine(cfg, tokenizer)
    metrics = Metrics(cfg.stats_sample_limit)
    logs = RequestLogStore(limit=cfg.log_limit, path=cfg.log_file)
    state = AppState(cfg=cfg, tokenizer=tokenizer, engine=engine, metrics=metrics, logs=logs)
    state.error_counter = 0
    if cfg.max_concurrency and cfg.max_concurrency > 0:
        state.sem = asyncio.Semaphore(cfg.max_concurrency)

    app = FastAPI(title="GPT-Zero Mock LLM", version=__version__)
    app.state.gptzero = state

    # -- slot accounting ---------------------------------------------------
    async def acquire_slot() -> float:
        sem = state.sem
        if sem is None:
            return 0.0
        state.waiting += 1
        metrics.set_waiting(state.waiting)
        started = time.perf_counter()
        await sem.acquire()
        waited = time.perf_counter() - started
        state.waiting -= 1
        metrics.set_waiting(state.waiting)
        return waited

    def check_auth(request: Request) -> None:
        """cfg.require_api_key 非空时校验 Bearer / x-api-key，失败抛 401。"""
        key = cfg.require_api_key
        if not key:
            return
        provided = request.headers.get("x-api-key")
        if not provided:
            auth = request.headers.get("authorization") or ""
            if auth.startswith("Bearer "):
                provided = auth[len("Bearer "):]
        if provided != key:
            raise HTTPException(status_code=401, detail={
                "error": {"message": "invalid API key", "type": "invalid_request_error"}})

    def inject_failure() -> HTTPException | None:
        """cfg.error_rate > 0 时按请求计数确定性注入失败（每 N 个请求一次）。"""
        rate = cfg.error_rate
        if rate and rate > 0:
            state.error_counter += 1
            every = max(1, round(1.0 / rate))
            if state.error_counter % every == 0:
                return HTTPException(status_code=cfg.error_status, detail={
                    "error": {"message": f"injected failure ({state.error_counter}th request)",
                              "type": "server_error"}})
        return None

    def finish_record(
        *,
        model: str,
        plan: GenerationPlan,
        ttft_s: float,
        e2e_s: float,
        queue_s: float,
        completion_tokens: int,
        finish_reason: str,
        tool_calls: int,
        ok: bool,
        error: str = "",
    ) -> None:
        metrics.record(
            RequestRecord(
                model=model,
                prompt_tokens=plan.prompt_tokens,
                completion_tokens=completion_tokens,
                ttft_s=ttft_s,
                e2e_s=e2e_s,
                tpot_s=(e2e_s - ttft_s) / max(completion_tokens - 1, 1) if completion_tokens > 1 else 0.0,
                finish_reason=finish_reason,
                tool_calls=tool_calls,
                cache_hit_tokens=plan.cache.hit_tokens,
                cache_query_tokens=plan.cache.query_tokens,
                queue_s=queue_s,
                ok=ok,
                error=error,
            )
        )

    def log_request(
        *,
        path: str,
        body: dict[str, Any],
        model: str,
        stream: bool,
        plan: GenerationPlan,
        completion_tokens: int,
        ttft_s: float,
        e2e_s: float,
        queue_s: float,
        finish_reason: str,
        tool_calls: int,
        status: str = "success",
        preview_messages: list[dict[str, Any]] | None = None,
    ) -> None:
        messages = preview_messages if preview_messages is not None else body.get("messages") or []
        preview = ""
        for msg in reversed(messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                preview = msg["content"][:80]
                break
        logs.add(
            {
                "endpoint": path,
                "status": status,
                "model": model,
                "stream": stream,
                "prompt_tokens": plan.prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": plan.prompt_tokens + completion_tokens,
                "tool_calls": tool_calls,
                "finish_reason": finish_reason,
                "cache_hit_tokens": plan.cache.hit_tokens,
                "cache_hit_rate": round(plan.cache.hit_rate, 4),
                "ttft_ms": round(ttft_s * 1000, 2),
                "queue_ms": round(queue_s * 1000, 2),
                "response_time_ms": round(e2e_s * 1000, 2),
                "request_preview": preview,
                "request_body": body,
            }
        )

    # -- OpenAI surface ----------------------------------------------------
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        started = time.perf_counter()
        check_auth(request)
        injected = inject_failure()
        if injected is not None:
            raise injected
        try:
            raw = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail={"error": {"message": "invalid JSON body"}})
        try:
            data = ChatRequest.model_validate(raw)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail={"error": {"message": str(exc)}})

        max_tokens = _resolve_max_tokens(data, cfg)
        plan = engine.plan(
            data.messages,
            data.model,
            max_tokens,
            seed=data.seed,
            tools=data.tools,
            tool_choice=data.tool_choice,
            parallel_tool_calls=data.parallel_tool_calls,
        )
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        completion_tokens = plan.output_tokens

        if not data.stream:
            queue_s = await acquire_slot()
            try:
                await _pace(plan.ttft_s + plan.decode_s)
                text = "".join(c for c, _ in plan.chunks)
                if not plan.tool_calls:
                    text, stopped = _apply_stop(text, data.stop)
                    if stopped:
                        completion_tokens = tokenizer.count(text)
                message: dict[str, Any] = {"role": "assistant", "content": text if text else None}
                if plan.tool_calls:
                    message["tool_calls"] = plan.tool_calls
                e2e_s = time.perf_counter() - started
                finish_record(
                    model=data.model,
                    plan=plan,
                    ttft_s=plan.ttft_s + queue_s,
                    e2e_s=e2e_s,
                    queue_s=queue_s,
                    completion_tokens=completion_tokens,
                    finish_reason=plan.finish_reason,
                    tool_calls=len(plan.tool_calls),
                    ok=True,
                )
                log_request(
                    path="/v1/chat/completions",
                    body=raw,
                    model=data.model,
                    stream=False,
                    plan=plan,
                    completion_tokens=completion_tokens,
                    ttft_s=plan.ttft_s + queue_s,
                    e2e_s=e2e_s,
                    queue_s=queue_s,
                    finish_reason=plan.finish_reason,
                    tool_calls=len(plan.tool_calls),
                )
                return JSONResponse(
                    {
                        "id": cid,
                        "object": "chat.completion",
                        "created": created,
                        "model": data.model,
                        "system_fingerprint": f"gptzero-{__version__}",
                        "choices": [
                            {"index": 0, "message": message, "finish_reason": plan.finish_reason}
                        ],
                        "usage": _usage(plan.prompt_tokens, completion_tokens, plan.cache.hit_tokens),
                    }
                )
            finally:
                if state.sem is not None:
                    state.sem.release()

        include_usage = bool((data.stream_options or {}).get("include_usage"))

        async def event_stream():
            queue_s = await acquire_slot()
            ttft_s = 0.0
            ok = False
            produced = 0
            emitted_tokens = completion_tokens
            error = ""
            try:
                if plan.ttft_s:
                    await _pace(plan.ttft_s)
                ttft_s = time.perf_counter() - started
                yield _sse(_chunk(cid, data.model, created, {"role": "assistant"}))
                if plan.tool_calls:
                    # Reasoning text first (padded so output length is exact),
                    # then the tool-call JSON fragments.
                    if plan.chunks:
                        for text, n_tokens in plan.chunks:
                            await _pace(plan.tpot_s * max(n_tokens, 1))
                            yield _sse(_chunk(cid, data.model, created, {"content": text}))
                            produced += n_tokens
                    for index, call in enumerate(plan.tool_calls):
                        yield _sse(
                            _chunk(
                                cid,
                                data.model,
                                created,
                                {
                                    "tool_calls": [
                                        {
                                            "index": index,
                                            "id": call["id"],
                                            "type": "function",
                                            "function": {"name": call["function"]["name"], "arguments": ""},
                                        }
                                    ]
                                },
                            )
                        )
                        args = call["function"]["arguments"]
                        step = 24
                        for i in range(0, len(args), step):
                            await _pace(plan.tpot_s)
                            yield _sse(
                                _chunk(
                                    cid,
                                    data.model,
                                    created,
                                    {
                                        "tool_calls": [
                                            {"index": index, "function": {"arguments": args[i : i + step]}}
                                        ]
                                    },
                                )
                            )
                else:
                    buffer = ""
                    stopped = False
                    for text, n_tokens in plan.chunks:
                        await _pace(plan.tpot_s * max(n_tokens, 1))
                        buffer += text
                        if data.stop:
                            trimmed, hit = _apply_stop(buffer, data.stop)
                            if hit:
                                leftover = trimmed[len(buffer) - len(text) :]
                                if leftover:
                                    yield _sse(_chunk(cid, data.model, created, {"content": leftover}))
                                produced += tokenizer.count(leftover)
                                stopped = True
                                break
                        yield _sse(_chunk(cid, data.model, created, {"content": text}))
                        produced += n_tokens
                    emitted_tokens = produced
                    if stopped:
                        plan.finish_reason = "stop"
                yield _sse(_chunk(cid, data.model, created, {}, plan.finish_reason))
                if include_usage:
                    yield _sse(
                        {
                            "id": cid,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": data.model,
                            "choices": [],
                            "usage": _usage(plan.prompt_tokens, emitted_tokens, plan.cache.hit_tokens),
                        }
                    )
                yield SSE_DONE
                ok = True
            except asyncio.CancelledError:
                error = "client disconnected"
                raise
            finally:
                e2e_s = time.perf_counter() - started
                if state.sem is not None:
                    state.sem.release()
                finish_record(
                    model=data.model,
                    plan=plan,
                    ttft_s=ttft_s or plan.ttft_s + queue_s,
                    e2e_s=e2e_s,
                    queue_s=queue_s,
                    completion_tokens=emitted_tokens,
                    finish_reason=plan.finish_reason,
                    tool_calls=len(plan.tool_calls),
                    ok=ok,
                    error=error,
                )
                log_request(
                    path="/v1/chat/completions",
                    body=raw,
                    model=data.model,
                    stream=True,
                    plan=plan,
                    completion_tokens=emitted_tokens,
                    ttft_s=ttft_s or plan.ttft_s + queue_s,
                    e2e_s=e2e_s,
                    queue_s=queue_s,
                    finish_reason=plan.finish_reason,
                    tool_calls=len(plan.tool_calls),
                    status="success" if ok else "aborted",
                )

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/v1/completions")
    async def completions(request: Request):
        started = time.perf_counter()
        check_auth(request)
        injected = inject_failure()
        if injected is not None:
            raise injected
        try:
            raw = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail={"error": {"message": "invalid JSON body"}})
        data = CompletionRequest.model_validate(raw)
        prompt = data.prompt if isinstance(data.prompt, str) else "\n".join(data.prompt)
        max_tokens = _resolve_max_tokens(data, cfg)
        plan = engine.plan([{"role": "user", "content": prompt}], data.model, max_tokens, seed=data.seed)
        queue_s = await acquire_slot()
        try:
            await _pace(plan.ttft_s + plan.decode_s)
            text = "".join(c for c, _ in plan.chunks)
            completion_tokens = plan.output_tokens
            text, stopped = _apply_stop(text, data.stop)
            if stopped:
                completion_tokens = tokenizer.count(text)
            e2e_s = time.perf_counter() - started
            finish_record(
                model=data.model,
                plan=plan,
                ttft_s=plan.ttft_s + queue_s,
                e2e_s=e2e_s,
                queue_s=queue_s,
                completion_tokens=completion_tokens,
                finish_reason="stop" if stopped else "length",
                tool_calls=0,
                ok=True,
            )
            log_request(
                path="/v1/completions",
                body=raw,
                model=data.model,
                stream=False,
                plan=plan,
                completion_tokens=completion_tokens,
                ttft_s=plan.ttft_s + queue_s,
                e2e_s=e2e_s,
                queue_s=queue_s,
                finish_reason="stop" if stopped else "length",
                tool_calls=0,
            )
            return JSONResponse(
                {
                    "id": f"cmpl-{uuid.uuid4().hex[:24]}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": data.model,
                    "choices": [
                        {
                            "index": 0,
                            "text": text,
                            "finish_reason": "stop" if stopped else "length",
                        }
                    ],
                    "usage": _usage(plan.prompt_tokens, completion_tokens, plan.cache.hit_tokens),
                }
            )
        finally:
            if state.sem is not None:
                state.sem.release()

    # -- OpenAI Responses API ----------------------------------------------
    @app.post("/v1/responses")
    async def responses_endpoint(request: Request):
        started = time.perf_counter()
        check_auth(request)
        injected = inject_failure()
        if injected is not None:
            return _openai_error(injected.detail["error"]["message"], injected.status_code)
        try:
            raw = await request.json()
        except Exception:
            return _openai_error("invalid JSON body")
        if not isinstance(raw, dict):
            return _openai_error("request body must be a JSON object")
        data = normalize_responses(raw, cfg.default_max_tokens, cfg.max_tokens_cap)

        previous_id = raw.get("previous_response_id")
        if previous_id and state.responses.get(previous_id) is None:
            return _openai_error(f"Previous response with id '{previous_id}' not found.", 404)
        data.messages = state.responses.context_for(previous_id) + data.messages

        plan = engine.plan(
            data.messages,
            data.model,
            data.max_tokens,
            tools=data.tools,
            tool_choice=data.tool_choice,
            parallel_tool_calls=data.parallel_tool_calls,
            force_tool=data.force_tool,
        )
        completion_tokens = plan.output_tokens
        rid = new_id("resp")
        created_at = int(time.time())
        status = "completed" if plan.tool_calls else "incomplete"

        if not data.stream:
            queue_s = await acquire_slot()
            try:
                await _pace(plan.ttft_s + plan.decode_s)
                text = "".join(c for c, _ in plan.chunks)
                output_items = responses_output_items(
                    text, plan.tool_calls, "completed" if plan.tool_calls else "incomplete"
                )
                usage = responses_usage(plan.prompt_tokens, completion_tokens, plan.cache.hit_tokens)
                response_obj = responses_response_object(
                    rid=rid,
                    model=data.model,
                    created_at=created_at,
                    status=status,
                    output_items=output_items,
                    usage=usage,
                    body=raw,
                    incomplete_details=None if plan.tool_calls else {"reason": "max_output_tokens"},
                )
                e2e_s = time.perf_counter() - started
                finish_record(
                    model=data.model,
                    plan=plan,
                    ttft_s=plan.ttft_s + queue_s,
                    e2e_s=e2e_s,
                    queue_s=queue_s,
                    completion_tokens=completion_tokens,
                    finish_reason=plan.finish_reason,
                    tool_calls=len(plan.tool_calls),
                    ok=True,
                )
                log_request(
                    path="/v1/responses",
                    body=raw,
                    model=data.model,
                    stream=False,
                    plan=plan,
                    completion_tokens=completion_tokens,
                    ttft_s=plan.ttft_s + queue_s,
                    e2e_s=e2e_s,
                    queue_s=queue_s,
                    finish_reason=plan.finish_reason,
                    tool_calls=len(plan.tool_calls),
                    preview_messages=data.messages,
                )
                state.responses.put(rid, response_obj, data.messages, responses_output_to_messages(output_items))
                return JSONResponse(response_obj)
            finally:
                if state.sem is not None:
                    state.sem.release()

        usage = responses_usage(plan.prompt_tokens, completion_tokens, plan.cache.hit_tokens)
        response_obj = responses_response_object(
            rid=rid,
            model=data.model,
            created_at=created_at,
            status=status,
            output_items=responses_output_items(
                "".join(c for c, _ in plan.chunks),
                plan.tool_calls,
                "completed" if plan.tool_calls else "incomplete",
            ),
            usage=usage,
            body=raw,
            incomplete_details=None if plan.tool_calls else {"reason": "max_output_tokens"},
        )
        frames = responses_stream_blueprint(
            rid=rid,
            model=data.model,
            created_at=created_at,
            response_obj=response_obj,
            chunks=plan.chunks,
            tpot_s=plan.tpot_s,
        )

        async def responses_stream():
            queue_s = await acquire_slot()
            ok = False
            error = ""
            try:
                if plan.ttft_s:
                    await _pace(plan.ttft_s)
                for frame in frames:
                    await _pace(frame["pace_s"])
                    yield _sse_event(frame["event"], frame["data"])
                ok = True
            except asyncio.CancelledError:
                error = "client disconnected"
                raise
            finally:
                e2e_s = time.perf_counter() - started
                if state.sem is not None:
                    state.sem.release()
                finish_record(
                    model=data.model,
                    plan=plan,
                    ttft_s=plan.ttft_s + queue_s,
                    e2e_s=e2e_s,
                    queue_s=queue_s,
                    completion_tokens=completion_tokens,
                    finish_reason=plan.finish_reason,
                    tool_calls=len(plan.tool_calls),
                    ok=ok,
                    error=error,
                )
                log_request(
                    path="/v1/responses",
                    body=raw,
                    model=data.model,
                    stream=True,
                    plan=plan,
                    completion_tokens=completion_tokens,
                    ttft_s=plan.ttft_s + queue_s,
                    e2e_s=e2e_s,
                    queue_s=queue_s,
                    finish_reason=plan.finish_reason,
                    tool_calls=len(plan.tool_calls),
                    status="success" if ok else "aborted",
                    preview_messages=data.messages,
                )
                state.responses.put(rid, response_obj, data.messages, responses_output_to_messages(response_obj["output"]))

        return StreamingResponse(responses_stream(), media_type="text/event-stream")

    @app.get("/v1/responses/{response_id}")
    async def get_response(response_id: str):
        stored = state.responses.get(response_id)
        if stored is None:
            return _openai_error(f"Response with id '{response_id}' not found.", 404)
        return JSONResponse(stored)

    # -- Anthropic Messages API ---------------------------------------------
    @app.post("/v1/messages")
    async def messages_endpoint(request: Request):
        started = time.perf_counter()
        check_auth(request)
        injected = inject_failure()
        if injected is not None:
            return _anthropic_error(injected.detail["error"]["message"], injected.status_code)
        try:
            raw = await request.json()
        except Exception:
            return _anthropic_error("invalid JSON body")
        if not isinstance(raw, dict):
            return _anthropic_error("request body must be a JSON object")
        data = normalize_messages(raw, cfg.default_max_tokens, cfg.max_tokens_cap)
        plan = engine.plan(
            data.messages,
            data.model,
            data.max_tokens,
            tools=data.tools,
            tool_choice=data.tool_choice,
            parallel_tool_calls=data.parallel_tool_calls,
            force_tool=data.force_tool,
        )
        mid = new_id("msg", 24)
        completion_tokens = plan.output_tokens

        chunks = plan.chunks
        tool_calls = anthropic_tool_calls(plan.tool_calls)
        stopped = False
        if data.stop:
            text, stopped = _apply_stop("".join(c for c, _ in chunks), data.stop)
            if stopped:
                tool_calls = []
                completion_tokens = tokenizer.count(text)
                chunks = [(text, max(1, completion_tokens))]
        stop_reason = (
            "stop_sequence"
            if stopped
            else ("tool_use" if plan.tool_calls else "max_tokens")
        )
        usage = anthropic_usage(plan.prompt_tokens, completion_tokens, plan.cache.hit_tokens)

        if not data.stream:
            queue_s = await acquire_slot()
            try:
                await _pace(plan.ttft_s + plan.decode_s)
                text = "".join(c for c, _ in chunks)
                blocks = anthropic_content_blocks(text, tool_calls)
                message_obj = anthropic_message_object(
                    mid=mid,
                    model=data.model,
                    content_blocks=blocks,
                    stop_reason=stop_reason,
                    usage=usage,
                )
                e2e_s = time.perf_counter() - started
                finish_record(
                    model=data.model,
                    plan=plan,
                    ttft_s=plan.ttft_s + queue_s,
                    e2e_s=e2e_s,
                    queue_s=queue_s,
                    completion_tokens=completion_tokens,
                    finish_reason=plan.finish_reason,
                    tool_calls=len(tool_calls),
                    ok=True,
                )
                log_request(
                    path="/v1/messages",
                    body=raw,
                    model=data.model,
                    stream=False,
                    plan=plan,
                    completion_tokens=completion_tokens,
                    ttft_s=plan.ttft_s + queue_s,
                    e2e_s=e2e_s,
                    queue_s=queue_s,
                    finish_reason=plan.finish_reason,
                    tool_calls=len(tool_calls),
                    preview_messages=data.messages,
                )
                return JSONResponse(message_obj)
            finally:
                if state.sem is not None:
                    state.sem.release()

        frames = anthropic_stream_blueprint(
            mid=mid,
            model=data.model,
            chunks=chunks,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=usage,
            tpot_s=plan.tpot_s,
        )

        async def messages_stream():
            queue_s = await acquire_slot()
            ok = False
            error = ""
            try:
                if plan.ttft_s:
                    await _pace(plan.ttft_s)
                for frame in frames:
                    await _pace(frame["pace_s"])
                    yield _sse_event(frame["event"], frame["data"])
                ok = True
            except asyncio.CancelledError:
                error = "client disconnected"
                raise
            finally:
                e2e_s = time.perf_counter() - started
                if state.sem is not None:
                    state.sem.release()
                finish_record(
                    model=data.model,
                    plan=plan,
                    ttft_s=plan.ttft_s + queue_s,
                    e2e_s=e2e_s,
                    queue_s=queue_s,
                    completion_tokens=completion_tokens,
                    finish_reason=plan.finish_reason,
                    tool_calls=len(tool_calls),
                    ok=ok,
                    error=error,
                )
                log_request(
                    path="/v1/messages",
                    body=raw,
                    model=data.model,
                    stream=True,
                    plan=plan,
                    completion_tokens=completion_tokens,
                    ttft_s=plan.ttft_s + queue_s,
                    e2e_s=e2e_s,
                    queue_s=queue_s,
                    finish_reason=plan.finish_reason,
                    tool_calls=len(tool_calls),
                    status="success" if ok else "aborted",
                    preview_messages=data.messages,
                )

        return StreamingResponse(messages_stream(), media_type="text/event-stream")

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {"id": m, "object": "model", "created": int(time.time()), "owned_by": "gptzero"}
                for m in cfg.models
            ],
        }

    # -- observability -----------------------------------------------------
    @app.get("/health")
    async def health():
        return {"status": "ok", "version": __version__, "uptime_s": round(time.time() - metrics.started_at, 2)}

    @app.get("/api/stats")
    async def api_stats():
        stats = metrics.stats()
        stats.setdefault("cache", {}).update(
            {
                k: v
                for k, v in engine.cache.stats().items()
                if k in ("blocks_cached", "blocks_capacity", "evictions")
            }
        )
        stats["tokenizer"] = tokenizer.name
        stats["config"] = {
            "ttft_ms": cfg.ttft_ms,
            "tpot_ms": cfg.tpot_ms,
            "max_concurrency": cfg.max_concurrency,
            "block_size": cfg.block_size,
            "cache_blocks": cfg.cache_blocks,
        }
        return stats

    @app.get("/metrics")
    async def prometheus_metrics():
        return PlainTextResponse(metrics.prometheus(engine.cache.stats()))

    @app.get("/api/logs")
    async def api_logs(limit: int = 200):
        entries = logs.reversed()
        return {"logs": entries[: max(1, limit)], "total": len(entries)}

    @app.get("/api/log/{log_id}")
    async def api_log_detail(log_id: str):
        try:
            wanted = int(log_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="log id must be an integer")
        for entry in logs.entries:
            if entry.get("id") == wanted:
                return entry
        entries = logs.reversed()
        if 0 <= wanted < len(entries):
            return entries[wanted]
        raise HTTPException(status_code=404, detail="Log not found")

    @app.post("/api/reset")
    async def api_reset():
        metrics.reset()
        engine.cache.reset()
        logs.entries.clear()
        state.responses.clear()
        return {"ok": True, "version": __version__}

    @app.get("/api/config")
    async def api_config():
        data = config_as_dict(cfg)
        data.update({"tokenizer": tokenizer.name, "tokenizer_warnings": tokenizer.warnings})
        return data

    # -- model directory (v2 compatible) -----------------------------------
    @app.get("/api/models")
    async def api_models():
        models = scan_model_directory(cfg.model_dir)
        return {"models": models, "total": len(models), "model_dir": cfg.model_dir}

    @app.get("/api/models/search")
    async def api_models_search(query: str = "", model_dir: str = ""):
        search_dir = os.path.abspath(model_dir or cfg.model_dir)
        available = scan_model_directory(search_dir)
        if not query:
            return {"results": available, "total": len(available), "query": query, "search_directory": search_dir}
        q = normalize_model_name(query)
        results = []
        for model in available:
            name = model.get("normalized_name") or normalize_model_name(model["name"])
            if q in name or name in q:
                score = round(len(q) / max(len(q), len(name)) * 100, 1)
                results.append({**model, "match_score": score})
            elif query.lower() in model["name"].lower():
                results.append({**model, "match_score": 50.0})
        results.sort(key=lambda m: m.get("match_score", 0), reverse=True)
        return {"results": results, "total": len(results), "query": query, "search_directory": search_dir}

    @app.post("/api/models/scan")
    async def api_models_scan(request: Request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        path = os.path.abspath(body.get("path") or cfg.model_dir)
        if not os.path.isdir(path):
            return {"success": False, "error": f"path not found: {path}", "models": [], "total": 0, "path": path}
        models = scan_model_directory(path)
        return {"success": True, "models": models, "total": len(models), "path": path, "is_custom": "path" in body}

    @app.post("/api/config/model-dir")
    async def api_set_model_dir(request: Request):
        body = await request.json()
        new_dir = os.path.abspath(body.get("model_dir") or "")
        if not new_dir or not os.path.isdir(new_dir):
            raise HTTPException(status_code=400, detail=f"path not found: {new_dir}")
        cfg.model_dir = new_dir
        models = scan_model_directory(new_dir)
        return {"success": True, "model_dir": new_dir, "available_models": len(models), "models": models}

    @app.post("/api/models/match")
    async def api_models_match(request: Request):
        body = await request.json()
        name = body.get("model_name")
        if not name:
            raise HTTPException(status_code=400, detail="model_name is required")
        search_dir = os.path.abspath(body.get("model_dir") or cfg.model_dir)
        available = scan_model_directory(search_dir)
        matched = match_model(name, available)
        req = normalize_model_name(name)
        suggestions = []
        for model in available:
            norm = model.get("normalized_name") or normalize_model_name(model["name"])
            if norm in req or req in norm:
                suggestions.append(
                    {
                        "name": model["name"],
                        "path": model["path"],
                        "match_score": round(len(norm) / max(len(norm), len(req)) * 100, 1),
                    }
                )
        suggestions.sort(key=lambda m: m["match_score"], reverse=True)
        return {
            "requested_model": name,
            "search_directory": search_dir,
            "matched": matched is not None,
            "model": matched,
            "suggestions": suggestions[:5],
            "total_available": len(available),
        }

    @app.post("/api/tokens")
    async def api_tokens(request: Request):
        body = await request.json()
        tokenizer_path = body.get("tokenizer_path")
        codec = tokenizer
        used = tokenizer.name
        if tokenizer_path:
            codec = state.extra_tokenizers.get(tokenizer_path)
            if codec is None:
                try:
                    codec = CountingTokenizer(tokenizer_path)
                    state.extra_tokenizers[tokenizer_path] = codec
                except Exception as exc:
                    raise HTTPException(status_code=400, detail=f"tokenizer load failed: {exc}")
            used = codec.name
        messages = body.get("messages") or []
        prompt_tokens = codec.messages_tokens(messages)
        completion = body.get("completion") or ""
        completion_tokens = codec.count(completion)
        return {
            "tokens": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "tokenizer_used": used,
            "transformers_available": not isinstance(codec.codec, type(None)),
        }

    # -- dashboard ---------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        path = Path(__file__).with_name("dashboard.html")
        if path.exists():
            return HTMLResponse(path.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>GPT-Zero</h1><p>dashboard.html missing</p>")

    @app.get("/legacy", response_class=HTMLResponse)
    async def legacy_dashboard():
        path = Path(__file__).resolve().parent.parent / "assets" / "dashboard.html"
        if path.exists():
            return HTMLResponse(path.read_text(encoding="utf-8"))
        raise HTTPException(status_code=404, detail="legacy dashboard not present")

    return app
