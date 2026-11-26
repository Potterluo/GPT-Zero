"""Token codecs.

Two implementations share one interface:

``HFTokenCodec``
    Wraps a *local* ``transformers`` tokenizer directory (the repo ships
    ``deepseek_v3_tokenizer/``).  This gives token counts that match what a real
    client (bench harness / agent loops) computes, which is what ``use_server_token_count``
    compares against.

``HeuristicCodec``
    Dependency-free fallback that tokenizes CJK characters individually and
    latin words in ~4-character pieces.  It is deterministic and reversible for
    the generated filler corpus, which is all the mock needs.

Both expose ``encode``/``decode``/``count`` and are safe to share across async
tasks (no mutable state after construction).
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import OrderedDict
from pathlib import Path

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_WORD = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


class TokenCodec:
    name = "base"

    def encode(self, text: str) -> list[int]:  # pragma: no cover - interface
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encode(text))


class HeuristicCodec(TokenCodec):
    """Deterministic, reversible, dependency-free tokenizer."""

    name = "heuristic"

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._id_to_tok: list[str] = []

    def _tok_id(self, tok: str) -> int:
        idx = self._vocab.get(tok)
        if idx is None:
            idx = len(self._id_to_tok)
            self._vocab[tok] = idx
            self._id_to_tok.append(tok)
        return idx

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in _WORD.findall(text or ""):
            if _CJK.fullmatch(piece):
                ids.append(self._tok_id(piece))
                continue
            if piece.isspace():
                continue
            if len(piece) <= 4:
                ids.append(self._tok_id(piece))
            else:
                for i in range(0, len(piece), 4):
                    ids.append(self._tok_id(piece[i : i + 4]))
        return ids

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        for i in ids:
            if 0 <= i < len(self._id_to_tok):
                out.append(self._id_to_tok[i])
        return " ".join(out)


class HFTokenCodec(TokenCodec):
    """Local tokenizer; never touches the network.

    Prefers the standalone ``tokenizers`` runtime on ``tokenizer.json`` (exact
    encode *and* decode incl. whitespace markers).  Falls back to
    ``transformers.AutoTokenizer`` for repositories that only ship a
    ``tokenizer.model``/config, where custom remote code may be required.
    """

    name = "hf"

    def __init__(self, path: str) -> None:
        self._tok = None
        self._hf = None
        tokenizer_json = Path(path) / "tokenizer.json"
        if tokenizer_json.exists():
            try:
                from tokenizers import Tokenizer as _Tk

                self._tok = _Tk.from_file(str(tokenizer_json))
                self.name = f"tokenizers:{Path(path).name}"
                return
            except Exception:
                self._tok = None
        from transformers import AutoTokenizer  # optional dependency

        self._hf = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        self.name = f"hf:{Path(path).name}"

    def encode(self, text: str) -> list[int]:
        if self._tok is not None:
            return list(self._tok.encode(text or "").ids)
        return self._hf.encode(text or "", add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        if self._tok is not None:
            return self._tok.decode(ids, skip_special_tokens=True)
        return self._hf.decode(ids, skip_special_tokens=True)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encode(text))


def _candidate_paths(explicit: str) -> list[str]:
    here = Path(__file__).resolve().parent
    repo = here.parent
    cands: list[str] = []
    if explicit and explicit not in ("auto", "heuristic"):
        cands.append(explicit)
    if explicit in ("", "auto"):
        env = os.environ.get("GPTZERO_TOKENIZER_PATH", "")
        if env:
            cands.append(env)
        cands += [
            str(repo / "deepseek_v3_tokenizer"),
            str(Path.cwd() / "deepseek_v3_tokenizer"),
            str(here / "tokenizer"),
        ]
    return list(dict.fromkeys(c for c in cands if os.path.isdir(c)))


class CountingTokenizer:
    """Facade used by the engine: resolves the codec once, caches token counts.

    Prompt token counting is on the hot path (every request, and every block of
    a long agentic context), so a small bounded LRU saves repeated encodes of
    the same system prefix.
    """

    def __init__(self, spec: str = "auto", cache_size: int = 4096) -> None:
        self.codec: TokenCodec
        warnings: list[str] = []
        if spec == "heuristic":
            self.codec = HeuristicCodec()
        else:
            codec = None
            for path in _candidate_paths(spec):
                try:
                    codec = HFTokenCodec(path)
                    break
                except Exception as exc:  # transformers missing or bad tokenizer
                    warnings.append(f"{path}: {type(exc).__name__}: {exc}")
            self.codec = codec or HeuristicCodec()
        self.warnings = warnings
        self._cache: dict[tuple[int, int], int] = {}
        self._cache_size = cache_size
        self._cache_keys: list[tuple[int, int]] = []
        # Encoded-id cache.  Agentic traffic re-sends the same 100k+ token
        # system/tool prefix on every turn; re-encoding it per request would
        # make the *mock* the bottleneck instead of the simulated server.
        self._enc: "OrderedDict[tuple[int, int], tuple[int, ...]]" = OrderedDict()
        self._enc_tokens = 0
        self._enc_token_budget = 4_000_000

    @property
    def name(self) -> str:
        return self.codec.name

    def count(self, text: str) -> int:
        """Token count; shares the encoded-id cache with prompt hashing."""
        if not text:
            return 0
        return len(self.encode_cached(text))

    def encode(self, text: str) -> list[int]:
        return self.codec.encode(text)

    def encode_cached(self, text: str) -> tuple[int, ...]:
        if not text:
            return ()
        key = (hash(text), len(text))
        hit = self._enc.get(key)
        if hit is not None:
            self._enc.move_to_end(key)
            return hit
        ids = tuple(self.codec.encode(text))
        self._enc[key] = ids
        self._enc_tokens += len(ids)
        while self._enc_tokens > self._enc_token_budget and len(self._enc) > 1:
            _, old = self._enc.popitem(last=False)
            self._enc_tokens -= len(old)
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.codec.decode(ids)

    def messages_tokens(self, messages: list[dict]) -> int:
        """Chat-template-ish token accounting: role + content + tool framing.

        Exactness does not matter for the *simulation*; what matters is that
        long agentic contexts dominate the prefill term and that multi-turn
        growth is monotonic.
        """
        total = 0
        for msg in messages:
            total += 4  # role/format overhead
            content = msg.get("content")
            if isinstance(content, str):
                total += self.count(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if isinstance(part.get("text"), str):
                            total += self.count(part["text"])
                        elif part.get("type") == "image_url":
                            total += 1024
            for call in msg.get("tool_calls") or []:
                fn = call.get("function", {}) if isinstance(call, dict) else {}
                total += self.count(fn.get("name", "")) + self.count(fn.get("arguments", ""))
            if msg.get("tool_call_id"):
                total += 8
        return max(total, 1)


def tokenizer_fingerprint(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()[:8]
