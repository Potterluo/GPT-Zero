"""FastAPI application: OpenAI-compatible surface + observability + dashboard.

Design notes
------------
* The transport is where queueing lives: a semaphore caps in-flight requests and
  the time spent waiting for a slot shows up in the client's TTFT, exactly like
  a saturated inference server.
* Every request is logged (ring buffer + optional JSONL) and every completed
  request feeds the metrics registry, so ``/api/stats`` and ``/metrics`` agree.
* Streaming frames are real SSE; tool calls arrive as OpenAI ``delta.tool_calls``
  fragments, which is what agent loops actually parse.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_max_tokens(data: ChatRequest | CompletionRequest, cfg: ServerConfig) -> int:
    # OpenAI's newer field is ``max_completion_tokens``; AIPerf/AgentX sends it
    # for chat requests, so treat it as an alias of ``max_tokens``.
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

    AIPerf/AgentX reads ``usage.prompt_tokens_details.cached_tokens`` to compute
    cache-hit metrics; without it the harness reports "no prompt-cache read
    tokens were seen" and falls back to scraping Prometheus.
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
    ) -> None:
        messages = body.get("messages") or []
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
