"""Mock inference engine: prefix-cache simulation, deterministic output, timing.

The engine is deliberately *not* an LLM.  It reproduces the observable shape of
an inference server under agentic load:

* prompt token accounting (exact with a local tokenizer, heuristic otherwise),
* a vLLM-style block prefix cache with LRU sequence eviction,
* deterministic filler output whose length is exactly the requested token count,
* a TTFT + TPOT timing model, with queueing added by the transport layer.

Everything is seeded, so replaying the same request against a fresh server
produces the same output text and the same cache outcomes.
"""

from __future__ import annotations

import hashlib
import random
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable

from .config import ServerConfig
from .tokenizer import CountingTokenizer

# Readable filler.  Token counts, not prose quality, is what matters; this keeps
# dashboards and logs legible.
FILLER_TEXT = (
    "The build succeeds when the failing test is fixed. "
    "Reading src/engine.py to understand how the prefix cache evicts sequences. "
    "Now I will update the handler and re-run the benchmark against the mock service. "
    "Large language models serving systems benefit from prefix caching because agentic "
    "traffic repeats long system prompts and tool schemas across every turn. "
    "A block level cache hashes fixed size chunks of tokens and reuses the matching prefix. "
    "Latency is dominated by prefill for long contexts and by decode for long answers. "
    "Measure time to first token and inter token latency separately, then report throughput. "
    "The quick brown fox jumps over the lazy dog while the benchmark keeps running. "
    "package main; import fmt; func main() { fmt.Println(computeThroughput()) } "
    "def aggregate(samples): return sorted(samples)[len(samples) // 2] "
    "SELECT model, p90_ttft FROM benchmark_runs WHERE concurrency = 16 "
)


@dataclass
class CacheOutcome:
    query_tokens: int = 0
    hit_tokens: int = 0
    evictions: int = 0
    blocks_cached: int = 0

    @property
    def hit_rate(self) -> float:
        return (self.hit_tokens / self.query_tokens) if self.query_tokens else 0.0


@dataclass
class GenerationPlan:
    prompt_tokens: int
    cached_tokens: int
    output_tokens: int
    ttft_s: float
    tpot_s: float
    chunks: list[tuple[str, int]]
    finish_reason: str = "stop"
    tool_calls: list[dict] = field(default_factory=list)
    cache: CacheOutcome = field(default_factory=CacheOutcome)
    seed: int = 0

    @property
    def decode_s(self) -> float:
        return self.tpot_s * max(self.output_tokens - 1, 0)


class PrefixCache:
    """Block-level prefix cache with LRU eviction over whole sequences.

    Mirrors vLLM's KV-block prefix cache at the level a benchmark can observe:
    a request's prompt is chunked into ``block_size``-token blocks, the longest
    cached block prefix is a hit, and inserting the request refreshes recency.
    Evicting a sequence only drops blocks whose reference count reaches zero, so
    shared system prefixes survive while unique tails churn.
    """

    def __init__(self, block_size: int = 64, capacity_blocks: int = 8192) -> None:
        self.block_size = max(1, int(block_size))
        self.capacity_blocks = max(1, int(capacity_blocks))
        self._root: dict[str, dict] = {}
        self._refs: dict[str, int] = {}
        self._seq: "OrderedDict[tuple[str, ...], None]" = OrderedDict()
        self._blocks: set[str] = set()
        self.evictions = 0
        self.hits = 0
        self.queries = 0

    # -- hashing -----------------------------------------------------------
    def block_hashes(self, token_ids: Iterable[int]) -> list[str]:
        ids = list(token_ids)
        out: list[str] = []
        bs = self.block_size
        for i in range(0, len(ids) - bs + 1, bs):
            h = hashlib.blake2b(
                b"".join(int(t).to_bytes(4, "little", signed=False) for t in ids[i : i + bs]),
                digest_size=8,
            ).hexdigest()
            out.append(h)
        return out

    def _tokens_per_block(self) -> int:
        return self.block_size

    # -- query / insert ----------------------------------------------------
    def lookup(self, hashes: list[str]) -> int:
        """Return the number of cached *tokens* in the longest block prefix."""
        node = self._root
        matched_blocks = 0
        for h in hashes:
            child = node.get(h)
            if child is None:
                break
            node = child
            matched_blocks += 1
        self.queries += matched_blocks * self.block_size
        return matched_blocks * self.block_size

    def insert(self, hashes: list[str]) -> None:
        if not hashes:
            return
        key = tuple(hashes)
        if key in self._seq:
            self._seq.move_to_end(key)
        else:
            self._seq[key] = None
        node = self._root
        for h in hashes:
            child = node.get(h)
            if child is None:
                child = {}
                node[h] = child
            self._refs[h] = self._refs.get(h, 0) + 1
            self._blocks.add(h)
            node = child
        while len(self._blocks) > self.capacity_blocks and self._seq:
            self._evict_oldest()

    def _evict_oldest(self) -> None:
        key, _ = self._seq.popitem(last=False)
        node = self._root
        detached = False
        for h in key:
            self._refs[h] = self._refs.get(h, 1) - 1
            child = node.get(h)
            if self._refs[h] <= 0:
                self._blocks.discard(h)
                self._refs.pop(h, None)
                self.evictions += 1
                if not detached:
                    node.pop(h, None)
                    detached = True
            node = child if isinstance(child, dict) else {}

    def stats(self) -> dict:
        return {
            "blocks_cached": len(self._blocks),
            "blocks_capacity": self.capacity_blocks,
            "evictions": self.evictions,
            "hit_tokens": self.hits,
            "query_tokens": self.queries,
        }

    def reset(self) -> None:
        self._root.clear()
        self._refs.clear()
        self._seq.clear()
        self._blocks.clear()
        self.evictions = 0
        self.hits = 0
        self.queries = 0

    # -- public API --------------------------------------------------------
    def observe(self, token_ids: list[int], prompt_tokens: int | None = None) -> CacheOutcome:
        """Look up then insert a prompt; returns per-request cache outcome.

        ``prompt_tokens`` (chat-template inclusive) is what vLLM reports as
        queried tokens; block matching itself only sees encoded content tokens,
        so the hit ratio stays slightly conservative.
        """
        hashes = self.block_hashes(token_ids)
        hit = self.lookup(hashes)
        before_evictions = self.evictions
        self.insert(hashes)
        return CacheOutcome(
            query_tokens=prompt_tokens if prompt_tokens is not None else len(token_ids),
            hit_tokens=hit,
            evictions=self.evictions - before_evictions,
            blocks_cached=len(self._blocks),
        )


class Filler:
    """Deterministic token stream with a legible decoded form."""

    def __init__(self, tokenizer: CountingTokenizer) -> None:
        self.tokenizer = tokenizer
        self._ids: list[int] | None = None
        self._text = FILLER_TEXT

    def _ensure(self) -> list[int]:
        if self._ids is None:
            self._ids = self.tokenizer.encode(self._text)
            if not self._ids:  # pathological tokenizer: fall back to word ids
                self._ids = list(range(1, 1024))
        return self._ids

    def token_ids(self, count: int, seed: int) -> list[int]:
        base = self._ensure()
        if count <= 0:
            return []
        offset = seed % len(base)
        n = len(base)
        reps = (count // n) + 1
        rotated = base[offset:] + base[:offset]
        stream = rotated * reps
        return stream[:count]

    def text_for(self, count: int, seed: int) -> str:
        return self.tokenizer.decode(self.token_ids(count, seed))

    def chunks(self, count: int, seed: int, chunk_tokens: int = 8) -> list[tuple[str, int]]:
        """Split the decoded answer into frames of roughly ``chunk_tokens``.

        Text is decoded once (so word spacing is correct) and then cut on
        character boundaries; the per-frame token counts are proportional to
        the character share, which is accurate enough to drive streaming timing.
        """
        if count <= 0:
            return []
        full = self.tokenizer.decode(self.token_ids(count, seed))
        if not full:
            return [("", count)]
        chunk_tokens = max(1, chunk_tokens)
        n_chunks = max(1, min(count, (count + chunk_tokens - 1) // chunk_tokens))
        step = len(full) / n_chunks
        out: list[tuple[str, int]] = []
        for i in range(n_chunks):
            start = int(round(i * step))
            end = int(round((i + 1) * step)) if i < n_chunks - 1 else len(full)
            piece = full[start:end]
            if not piece:
                continue
            tokens = max(1, round(count * len(piece) / len(full)))
            out.append((piece, tokens))
        if not out:
            return [(full, count)]
        # Make the frame token counts sum to the requested total.
        diff = count - sum(t for _, t in out)
        if diff:
            text, tokens = out[-1]
            out[-1] = (text, max(1, tokens + diff))
        return out


class LatencyModel:
    """TTFT/TPOT model with seeded jitter.

    ``ttft`` grows with the *uncached* prompt part when ``prefill_tps`` is set,
    which is what makes prefix caching visible in latency rather than only in
    the cache counters.
    """

    def __init__(self, cfg: ServerConfig) -> None:
        self.cfg = cfg

    def _jitter(self, rng: random.Random, value: float) -> float:
        if self.cfg.jitter <= 0:
            return value
        sigma = self.cfg.jitter * max(value, 1e-6)
        return max(0.0, rng.gauss(value, sigma))

    def ttft(self, prompt_tokens: int, cached_tokens: int, rng: random.Random) -> float:
        uncached = max(prompt_tokens - cached_tokens, 0)
        if self.cfg.prefill_tps > 0:
            prefill_ms = uncached / self.cfg.prefill_tps * 1000.0
            base = self.cfg.ttft_ms
        else:
            # No explicit prefill rate: scale the base TTFT sub-linearly with
            # context length so 100k-token prompts still feel expensive.
            prefill_ms = self.cfg.ttft_ms * (1.0 + uncached / 8192.0) ** 0.6
            base = 0.0
        return self._jitter(rng, base + prefill_ms) / 1000.0

    def tpot(self, rng: random.Random) -> float:
        if self.cfg.decode_tps > 0:
            value = 1000.0 / self.cfg.decode_tps
        else:
            value = self.cfg.tpot_ms
        return self._jitter(rng, value) / 1000.0


class MockEngine:
    def __init__(self, cfg: ServerConfig, tokenizer: CountingTokenizer) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.cache = PrefixCache(cfg.block_size, cfg.cache_blocks)
        self.filler = Filler(tokenizer)
        self.latency = LatencyModel(cfg)

    # -- helpers -----------------------------------------------------------
    def _prompt_ids(self, messages: list[dict]) -> list[int]:
        """Flatten messages to a token-id sequence for cache hashing.

        Using encode() (not count()) keeps block hashes stable across requests
        that share a prefix, which is the whole point of the cache simulation.
        """
        parts: list[int] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                parts.extend(self.tokenizer.encode_cached(content))
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        parts.extend(self.tokenizer.encode_cached(part["text"]))
            for call in msg.get("tool_calls") or []:
                fn = call.get("function", {}) if isinstance(call, dict) else {}
                parts.extend(self.tokenizer.encode_cached(fn.get("name", "")))
                parts.extend(self.tokenizer.encode_cached(fn.get("arguments", "")))
        return parts

    def _chunk_size(self, output_tokens: int) -> int:
        # Keep the event-stream chunk count sane for 8k-token answers, while
        # still streaming enough frames to measure ITL.
        return max(1, min(16, max(1, output_tokens // 64)))

    def plan(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int,
        *,
        seed: int | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        parallel_tool_calls: bool = False,
        force_tool: bool | None = None,
    ) -> GenerationPlan:
        cfg = self.cfg
        prompt_ids = self._prompt_ids(messages)
        prompt_tokens = self.tokenizer.messages_tokens(messages)
        cache_outcome = self.cache.observe(prompt_ids, prompt_tokens)
        self.cache.hits += cache_outcome.hit_tokens

        # Deterministic across processes: str hash() is randomized per
        # interpreter (PYTHONHASHSEED), which would silently change the filler
        # stream between runs. A stable digest of (model, prompt_tokens) keeps
        # "same request + same seed -> same output" true everywhere.
        seed_digest = hashlib.blake2b(
            f"{model}:{prompt_tokens}".encode("utf-8"), digest_size=8
        ).digest()
        rng = random.Random((seed if seed is not None else cfg.seed) ^ int.from_bytes(seed_digest, "big"))
        ttft_s = self.latency.ttft(prompt_tokens, cache_outcome.hit_tokens, rng)

        # Tool-call turn?  A tool result is always answered with plain text so
        # agent loops terminate.
        tool_calls: list[dict] = []
        finish_reason = "length"  # the mock has no natural EOS: it fills max_tokens
        output_tokens = max_tokens

        last_role = messages[-1].get("role") if messages else "user"
        wants_tools = bool(tools) and last_role != "tool" and tool_choice != "none"
        if wants_tools and force_tool is not None:
            wants_tools = force_tool
        elif wants_tools:
            wants_tools = rng.random() < cfg.tool_call_ratio

        if wants_tools:
            tool_calls = synthesize_tool_calls(
                tools or [],
                messages,
                min(cfg.tool_payload_tokens, max(16, max_tokens // 2)),
                rng,
                parallel_tool_calls,
            )
            finish_reason = "tool_calls"
            # Real assistant turns emit reasoning text *and* a tool call; the
            # total still has to honour max_tokens so ``ignore_eos`` clients can
            # compare lengths.  Pad the remainder as content.
            tool_tokens = sum(
                self.tokenizer.count(c["function"]["name"]) + self.tokenizer.count(c["function"]["arguments"]) + 8
                for c in tool_calls
            )
            content_tokens = max(0, max_tokens - tool_tokens)
            chunks = (
                self.filler.chunks(content_tokens, rng.randrange(1 << 30), self._chunk_size(content_tokens))
                if content_tokens
                else []
            )
        else:
            chunks = self.filler.chunks(
                output_tokens, rng.randrange(1 << 30), self._chunk_size(output_tokens)
            )
        tpot_s = self.latency.tpot(rng)

        return GenerationPlan(
            prompt_tokens=prompt_tokens,
            cached_tokens=cache_outcome.hit_tokens,
            output_tokens=output_tokens,
            ttft_s=ttft_s,
            tpot_s=tpot_s,
            chunks=chunks,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            cache=cache_outcome,
            seed=rng.randrange(1 << 30),
        )


def synthesize_tool_calls(
    tools: list[dict],
    messages: list[dict],
    payload_tokens: int,
    rng: random.Random,
    parallel: bool,
) -> list[dict]:
    """Build one (or two) OpenAI ``tool_calls`` with schema-plausible arguments.

    The arguments are drawn from the tool's declared JSON-schema properties so a
    client that validates arguments does not choke.  Payload size is controlled
    in *tokens* to emulate a real read/write tool result being sent back.
    """
    if not tools:
        return []
    chosen = tools[rng.randrange(len(tools))]
    fn = chosen.get("function", chosen) if isinstance(chosen, dict) else {}
    name = fn.get("name", "tool")
    params = fn.get("parameters", {}) or {}
    props: dict = params.get("properties", {}) or {}
    required = params.get("required") or list(props)

    args: dict[str, object] = {}
    payload = FillerWords(payload_tokens)
    for key in required:
        spec = props.get(key, {}) if isinstance(props, dict) else {}
        kind = spec.get("type", "string") if isinstance(spec, dict) else "string"
        if kind in ("integer", "number"):
            args[key] = rng.randrange(1, 4096 if kind == "integer" else 100)
        elif kind == "boolean":
            args[key] = rng.random() < 0.5
        elif kind == "array":
            args[key] = ["main.py", "tests"]
        elif kind == "object":
            args[key] = {"path": "src/engine.py"}
        else:
            low = key.lower()
            if low in ("path", "file", "filepath", "filename"):
                args[key] = rng.choice(
                    ["src/engine.py", "gptzero/app.py", "tests/test_server.py", "bench/report.py"]
                )
            elif low in ("command", "cmd", "shell"):
                args[key] = rng.choice(["pytest -q tests", "python -m bench run --concurrency 8", "ls -la"])
            elif low in ("content", "text", "body", "patch", "old_string", "new_string"):
                args[key] = payload.text()
            else:
                args[key] = payload.text()

    import json

    calls = [
        {
            "id": f"call_{rng.randrange(1 << 40):010x}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        }
    ]
    if parallel and len(tools) > 1:
        second = tools[(tools.index(chosen) + 1) % len(tools)]
        fn2 = second.get("function", second) if isinstance(second, dict) else {}
        calls.append(
            {
                "id": f"call_{rng.randrange(1 << 40):010x}",
                "type": "function",
                "function": {
                    "name": fn2.get("name", "tool"),
                    "arguments": json.dumps({"path": "src/engine.py", "pattern": "prefix"}, ensure_ascii=False),
                },
            }
        )
    return calls


class FillerWords:
    """Cheap word-level filler sized in tokens (~0.75 words per token)."""

    WORDS = FILLER_TEXT.split()

    def __init__(self, tokens: int) -> None:
        self.tokens = max(0, tokens)

    def text(self) -> str:
        if not self.tokens:
            return ""
        n = max(1, int(self.tokens * 0.75))
        out: list[str] = []
        for i in range(n):
            out.append(self.WORDS[i % len(self.WORDS)])
        return " ".join(out)
