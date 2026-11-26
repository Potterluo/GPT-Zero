"""确定性多轮工作负载：chat / coding / agentic / longctx。

设计要点
--------
* 每个会话（Session）由 ``seed`` + 会话序号唯一确定，同一 seed 重放得到相同消息。
* 会话内按轮推进：chat 6~10 轮；coding/agentic 上下文从 2k 递增到 32k；
  longctx 单轮 64k/128k 输入 + 短输出。
* coding 每轮把上一轮的 read/write/shell 工具结果回填进历史（模拟 agentic 编码
  Agent「工具结果驱动上下文增长」的形态）。
* agentic 走真实工具闭环：服务端返回 tool_calls → 客户端构造 role=tool 结果 →
  继续同一轮；并以 20% 概率并行派生一个 subagent 子请求（fan-out）。
* 所有请求带 ``ignore_eos=true``，``max_tokens`` 可按 profile 或 CLI 覆盖。

token 计数优先复用 ``gptzero.tokenizer``（契约 §4 允许），缺失时退回本文件的
heuristic 估算，保证 bench 可独立运行。
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any

from .util import estimate_tokens

PROFILES = ("chat", "coding", "agentic", "longctx")
DEFAULT_MAX_TOKENS = {"chat": 160, "coding": 192, "agentic": 160, "longctx": 64}
DEFAULT_TURNS = {"chat": (6, 10), "coding": (8, 10), "agentic": (6, 9), "longctx": (1, 1)}

# ---------------------------------------------------------------------------
# Token 计数（可注入/可降级）
# ---------------------------------------------------------------------------


class _HeuristicCounter:
    name = "heuristic"

    def count(self, text: str) -> int:
        return estimate_tokens(text)

    def messages_tokens(self, messages: list[dict[str, Any]]) -> int:
        total = 0
        for msg in messages:
            total += 4
            content = msg.get("content")
            if isinstance(content, str):
                total += self.count(content)
            for call in msg.get("tool_calls") or []:
                fn = call.get("function", {}) if isinstance(call, dict) else {}
                total += self.count(fn.get("name", "")) + self.count(fn.get("arguments", ""))
            if msg.get("tool_call_id"):
                total += 8
        return max(total, 1)


class _ServerCounter:
    """薄包装 ``gptzero.tokenizer.CountingTokenizer``，保持与 mock 同源计数。"""

    def __init__(self, tokenizer: Any) -> None:
        self._tok = tokenizer
        self.name = getattr(tokenizer, "name", "hf")

    def count(self, text: str) -> int:
        return int(self._tok.count(text))

    def messages_tokens(self, messages: list[dict[str, Any]]) -> int:
        return int(self._tok.messages_tokens(messages))


_COUNTER: Any = None


def get_counter() -> Any:
    """懒加载 tokenizer。``BENCH_TOKENIZER=heuristic`` 可强制轻量实现。"""
    global _COUNTER
    if _COUNTER is not None:
        return _COUNTER
    spec = os.environ.get("BENCH_TOKENIZER", "auto")
    if spec != "heuristic":
        try:
            from gptzero.tokenizer import CountingTokenizer  # 契约允许的复用点

            tok = CountingTokenizer(spec)
            if getattr(tok, "name", "").startswith("hf"):
                _COUNTER = _ServerCounter(tok)
                return _COUNTER
        except Exception:
            pass
    _COUNTER = _HeuristicCounter()
    return _COUNTER


# ---------------------------------------------------------------------------
# 文本工厂
# ---------------------------------------------------------------------------

_SENTENCES = (
    "The repository contains a small async service with a request router and a storage layer.",
    "Prefill cost dominates when the prompt is long, while decode cost dominates long answers.",
    "Reading src/engine.py to trace how the block level prefix cache evicts stale sequences.",
    "The failing test asserts that token accounting stays stable across retries and timeouts.",
    "Large language model serving systems reuse system prompts and tool schemas on every turn.",
    "A benchmark should report time to first token and inter token latency separately.",
    "The shell command writes the generated report into results and validates the schema.",
    "package main; import fmt; func main() { fmt.Println(computeThroughput()) }",
    "def aggregate(samples): return sorted(samples)[len(samples) // 2]",
    "SELECT model, p90_ttft, rps FROM benchmark_runs WHERE concurrency = 16 ORDER BY ts DESC",
    "Tool results are appended to the conversation so the next request carries a longer context.",
    "The agent reads the diff, runs the unit tests, and then summarizes what changed.",
    "Cache hit rate is computed on prompt tokens, not on requests, because prompts differ in size.",
    "Queueing appears once the number of active sessions exceeds the server concurrency limit.",
    "The mock replays deterministic filler so that two runs with the same seed are comparable.",
    "We keep the system prefix stable to make prefix cache reuse visible in the counters.",
    "The worker loop keeps exactly one request in flight per session, which is closed loop load.",
    "Long context requests stress memory bandwidth more than they stress the tokenizer.",
)


class TextFactory:
    """按 ``seed`` 生成可复现的填充文本，长度按 token 数近似控制。"""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.counter = get_counter()
        self._base = self._build_base()

    def _build_base(self) -> str:
        sentences = list(_SENTENCES)
        self.rng.shuffle(sentences)
        return " ".join(sentences)

    def text(self, target_tokens: int, *, salt: str = "") -> str:
        if target_tokens <= 0:
            return salt
        prefix = (salt + "\n") if salt else ""
        base_tokens = self.counter.count(self._base) or 1
        chars_per_token = len(self._base) / base_tokens
        budget = max(target_tokens - self.counter.count(prefix), 1)
        n_chars = max(1, int(budget * chars_per_token))
        reps = n_chars // max(1, len(self._base)) + 1
        body = (self._base * reps)[:n_chars]
        # 一次校准，避免 tokenizer 与字符比例偏差过大（仅对中小上下文，超大上下文交给近似）。
        if target_tokens <= 16384:
            got = self.counter.count(body)
            if got and abs(got - budget) / budget > 0.15:
                body = (self._base * (reps + 1))[: max(1, int(len(body) * budget / got))]
        return prefix + body

    def choice(self, items: tuple[str, ...] | list[str]) -> str:
        return self.rng.choice(list(items))


# ---------------------------------------------------------------------------
# 工具 schema 与提示词
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a UTF-8 file from the repository. Returns numbered lines; use start_line and "
                "end_line to page through large files without blowing up the context window."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Repository relative path, e.g. src/engine.py"},
                    "start_line": {"type": "integer", "description": "First line, 1-based, inclusive"},
                    "end_line": {"type": "integer", "description": "Last line, inclusive"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create or overwrite a file with the given content. Use after reading the current "
                "version so that edits are minimal and the diff stays reviewable."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Repository relative path"},
                    "content": {"type": "string", "description": "Full new file content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Run a shell command inside the workspace sandbox and capture stdout plus stderr. "
                "Use it for tests, linters, builds and git status; keep commands idempotent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command line to execute"},
                    "cwd": {"type": "string", "description": "Working directory, defaults to repo root"},
                    "timeout_s": {"type": "integer", "description": "Kill the process after N seconds"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_repo",
            "description": (
                "Search the repository with a regular expression and return matching files with line "
                "numbers and a short snippet so the agent can jump straight to the relevant code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Ripgrep style regular expression"},
                    "path": {"type": "string", "description": "Optional subtree to search"},
                    "max_results": {"type": "integer", "description": "Upper bound on matches"},
                },
                "required": ["query"],
            },
        },
    },
]


def tool_names() -> list[str]:
    return [t["function"]["name"] for t in TOOLS]


_SYSTEM_CODING = (
    "You are a senior software engineer working inside a local repository. "
    "You have tools: read_file, write_file, run_shell and search_repo. "
    "Work in small verified steps: inspect the relevant files, form a hypothesis, make the edit, "
    "then run the focused tests. Never invent file paths; read before writing. "
    "Report the root cause, the change you made, and the commands you ran. "
    "Keep answers concise and prefer evidence from tool output over speculation."
)

_SYSTEM_CHAT = (
    "You are a helpful assistant. Answer the user's question directly and concisely. "
    "If the question is ambiguous, state your assumption and answer anyway."
)

_SYSTEM_AGENTIC = (
    "You are an autonomous coding agent. You may call tools multiple times per turn and you may "
    "spawn subagents for independent research. Plan with a short todo list, execute one step at a "
    "time, and verify every edit with a test command. After the work is complete, summarize the "
    "diff, the test results, and any remaining risk. Do not stop while a required check is failing."
)

_SYSTEM_SUBAGENT = (
    "You are a subagent. Investigate exactly one narrow question and return findings with file "
    "paths and line numbers. Do not modify files and do not ask follow-up questions."
)

_CHAT_QUESTIONS = (
    "How does prefix caching change the TTFT distribution under agentic traffic?",
    "Explain the difference between TPOT and inter-token latency in a streaming benchmark.",
    "Why can a closed-loop benchmark overstate throughput compared with an open-loop one?",
    "What is the practical difference between p90 and p99 latency for a chat product?",
    "How should I size the KV cache block size for long tool outputs?",
    "Summarize the tradeoff between batching more requests and keeping TPOT low.",
    "When does queueing start to dominate end-to-end latency in this mock?",
    "What makes a benchmark reproducible across machines and days?",
)

_CODING_TASKS = (
    "The streaming endpoint drops the usage chunk when the client disconnects early. Find the "
    "root cause and fix it, then run the request-level tests.",
    "Prefix cache hit rate is lower than expected for the coding workload. Inspect block hashing "
    "and explain where the reuse boundary is lost.",
    "Add a regression test that asserts finish_reason=length when ignore_eos=true and max_tokens=64.",
    "The dashboard shows p99 TTFT spikes after a long-context run. Profile the queueing path and "
    "propose a fix that keeps p90 under 500 ms.",
    "Refactor the metrics registry so that request latency histograms are recorded exactly once.",
    "Investigate why tool call arguments stream in 24-character fragments and whether the client "
    "tolerates arbitrary splits.",
    "Write a script that replays the trace with a fixed seed and compares two servers byte for byte.",
    "The cache evicts shared system prefixes under load. Check reference counting in the LRU and "
    "add a test for the shared-prefix case.",
)

_SUB_TASKS = (
    "Locate every place that writes to results/ and list the file and line for each write.",
    "Read the metrics module and summarize which counters are monotonic and which are gauges.",
    "Find the tests covering SSE parsing and report whether the usage chunk is asserted.",
    "Check whether token counting is consistent between /api/stats and /metrics.",
)

_TOOL_ARGS = {
    "read_file": {"path": "gptzero/engine.py", "start_line": 180, "end_line": 260},
    "write_file": {"path": "gptzero/engine.py", "content": "def observe(...): ..."},
    "run_shell": {"command": "python -m pytest tests -q", "cwd": ".", "timeout_s": 120},
    "search_repo": {"query": "prefix_cache_hits_total", "path": "gptzero", "max_results": 20},
}

_TOOL_RESULT_TOKENS = {"read_file": (1800, 3200), "run_shell": (700, 1400), "search_repo": (500, 1000), "write_file": (80, 160)}


@dataclass
class RequestSpec:
    messages: list[dict[str, Any]]
    max_tokens: int
    tools: list[dict[str, Any]] | None = None
    parallel_tool_calls: bool = False
    label: str = "main"
    seed: int | None = None
    est_prompt_tokens: int = 0
    turn: int = 0

    def to_payload(self, model: str) -> dict[str, Any]:
        from .client import build_payload

        return build_payload(
            model,
            self.messages,
            max_tokens=self.max_tokens,
            tools=self.tools,
            parallel_tool_calls=self.parallel_tool_calls,
            seed=self.seed,
        )


@dataclass
class Turn:
    specs: list[RequestSpec]
    kind: str = "user"


@dataclass
class Session:
    """一个闭环会话；状态机式推进，供 runner 逐轮驱动。"""

    profile: str
    index: int
    seed: int = 42
    max_tokens_override: int | None = None
    _rng: random.Random = field(init=False, repr=False)
    _text: TextFactory = field(init=False, repr=False)
    _counter: Any = field(init=False, repr=False)
    messages: list[dict[str, Any]] = field(init=False, default_factory=list)
    turn: int = field(init=False, default=0)
    n_turns: int = field(init=False, default=1)
    targets: list[int] = field(init=False, default_factory=list)
    tool_rounds: int = field(init=False, default=0)
    pending_tool: bool = field(init=False, default=False)
    done: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if self.profile not in PROFILES:
            raise ValueError(f"unknown profile {self.profile!r}; expected one of {PROFILES}")
        mixed = int(hashlib.sha256(f"{self.profile}:{self.index}:{self.seed}".encode()).hexdigest()[:8], 16)
        self._rng = random.Random(mixed)
        self._text = TextFactory(mixed)
        self._counter = get_counter()
        self.max_tokens = self.max_tokens_override or DEFAULT_MAX_TOKENS[self.profile]
        self._setup()

    # -- setup -------------------------------------------------------------
    def _setup(self) -> None:
        lo, hi = DEFAULT_TURNS[self.profile]
        self.n_turns = self._rng.randint(lo, hi)
        if self.profile == "longctx":
            # 一半 64k、一半 128k，覆盖两条长上下文曲线。
            self.targets = [65536 if self.index % 2 == 0 else 131072]
            self.messages = [
                {"role": "system", "content": "You are a retrieval assistant. Answer with a short citation list."}
            ]
            return
        if self.profile == "chat":
            self.targets = [int(self._rng.uniform(200, 800)) for _ in range(self.n_turns)]
            self.messages = [{"role": "system", "content": _SYSTEM_CHAT}]
            return
        if self.profile == "coding":
            self.messages = [{"role": "system", "content": _SYSTEM_CODING}]
        else:  # agentic
            self.messages = [{"role": "system", "content": _SYSTEM_AGENTIC}]
        # 几何递增 2k → 32k，模拟 coding agent 的上下文膨胀。
        start, end = (2000, 32000) if self.profile == "coding" else (2500, 32000)
        ratio = (end / start) ** (1.0 / max(self.n_turns - 1, 1))
        self.targets = [int(start * (ratio**i)) for i in range(self.n_turns)]

    # -- helpers -----------------------------------------------------------
    def _maybe_tools(self) -> list[dict[str, Any]] | None:
        return TOOLS if self.profile in ("coding", "agentic") else None

    def _pad_to_target(self, target: int, salt: str) -> None:
        """把最后一条消息补到目标 prompt token 数（保持语义角色不变）。"""
        if not self.messages:
            return
        current = self._counter.messages_tokens(self.messages)
        needed = target - current
        if needed <= 32:
            return
        last = self.messages[-1]
        content = last.get("content")
        if not isinstance(content, str):
            content = ""
        header = f"\n\n[context {salt}]\n" if salt else "\n\n[context]\n"
        last["content"] = content + header + self._text.text(needed, salt=f"ctx-{self.index}-{self.turn}")

    def _append_assistant_text(self, text: str) -> None:
        snippet = (text or "").strip()
        if len(snippet) > 240:
            snippet = snippet[:240] + " ..."
        self.messages.append({"role": "assistant", "content": snippet or "Done."})

    def _append_simulated_history(self) -> None:
        """coding：把上一轮的 read/write/shell 结果回填进历史。"""
        if self.profile != "coding" or self.turn == 0:
            return
        name = self._rng.choice(tool_names())
        call_id = "call_" + hashlib.sha1(f"{self.index}:{self.turn}:{name}".encode()).hexdigest()[:8]
        args = _TOOL_ARGS[name]
        self.messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                ],
            }
        )
        lo, hi = _TOOL_RESULT_TOKENS[name]
        size = int(self._rng.uniform(lo, hi))
        self.messages.append(
            {"role": "tool", "tool_call_id": call_id, "name": name, "content": self._text.text(size, salt=f"tool-{self.turn}")}
        )

    def _new_user_message(self) -> str:
        if self.profile == "chat":
            return self._text.choice(_CHAT_QUESTIONS)
        return self._text.choice(_CODING_TASKS)

    def _subagent_spec(self) -> RequestSpec:
        task = self._text.choice(_SUB_TASKS)
        body = self._text.text(int(self._rng.uniform(300, 700)), salt=f"subagent-{self.index}-{self.turn}")
        messages = [
            {"role": "system", "content": _SYSTEM_SUBAGENT},
            {"role": "user", "content": f"{task}\n\nRepository notes:\n{body}"},
        ]
        return RequestSpec(
            messages=messages,
            max_tokens=max(48, self.max_tokens // 2),
            tools=None,
            label="sub",
            seed=self.seed + self.index * 31 + self.turn,
            turn=self.turn,
        )

    # -- public API --------------------------------------------------------
    def next_turn(self) -> Turn | None:
        if self.done:
            return None
        if self.profile == "longctx" and self.turn >= 1:
            self.done = True
            return None
        if self.turn >= self.n_turns:
            self.done = True
            return None

        specs: list[RequestSpec] = []
        kind = "tool" if self.pending_tool else "user"
        if self.pending_tool:
            # history 末尾已经是 role=tool，服务端会走文本收尾分支。
            spec = RequestSpec(
                messages=list(self.messages),
                max_tokens=self.max_tokens,
                tools=self._maybe_tools(),
                label="main",
                seed=self.seed + self.index * 31 + self.turn,
                turn=self.turn,
            )
            self.pending_tool = False
        else:
            self._append_simulated_history()
            user_text = self._new_user_message()
            self.messages.append({"role": "user", "content": user_text})
            target = self.targets[min(self.turn, len(self.targets) - 1)]
            self._pad_to_target(target, salt=f"{self.profile}-turn-{self.turn}")
            spec = RequestSpec(
                messages=list(self.messages),
                max_tokens=self.max_tokens,
                tools=self._maybe_tools(),
                parallel_tool_calls=(self.profile == "agentic" and self._rng.random() < 0.3),
                label="main",
                seed=self.seed + self.index * 31 + self.turn,
                turn=self.turn,
            )
            # 20% 概率并行派生 subagent 子请求（仅 agentic，且不在工具回填轮）。
            if self.profile == "agentic" and self._rng.random() < 0.2:
                specs.append(self._subagent_spec())
        specs.insert(0, spec)
        for s in specs:
            s.est_prompt_tokens = self._counter.messages_tokens(s.messages)
        return Turn(specs=specs, kind=kind)

    def observe(self, results: list[Any]) -> None:
        """根据本轮主请求的真实结果推进会话状态。"""
        main = next((r for r in results if getattr(r, "label", "") == "main"), None)
        if main is None:
            self.done = True
            return
        if not getattr(main, "ok", False):
            # 失败不推进历史，避免污染后续请求；runner 会记录失败率。
            return
        if self.profile == "agentic" and getattr(main, "tool_calls", 0):
            self._append_tool_loop(main)
            return
        self._append_assistant_text(getattr(main, "text_preview", "") or "")
        self.tool_rounds = 0
        self.turn += 1
        if self.profile == "longctx":
            self.done = True

    def _append_tool_loop(self, main: Any) -> None:
        calls = getattr(main, "tool_call_payload", None) or []
        if not calls:
            self._append_assistant_text("")
            self.turn += 1
            return
        self.messages.append({"role": "assistant", "content": None, "tool_calls": calls})
        for call in calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            lo, hi = _TOOL_RESULT_TOKENS.get(name, (400, 800))
            size = int(self._rng.uniform(lo, hi))
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "name": name,
                    "content": self._text.text(size, salt=f"result-{name}-{self.turn}"),
                }
            )
        self.tool_rounds += 1
        if self.tool_rounds >= 4:
            # 收口，避免无限工具循环；下一轮由用户消息开启。
            self.tool_rounds = 0
            self.turn += 1
        else:
            self.pending_tool = True

    def snapshot(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "index": self.index,
            "seed": self.seed,
            "turn": self.turn,
            "n_turns": self.n_turns,
            "targets": self.targets,
            "tool_rounds": self.tool_rounds,
            "pending_tool": self.pending_tool,
        }


def build_session(
    profile: str, index: int, seed: int = 42, max_tokens: int | None = None
) -> Session:
    return Session(profile=profile, index=index, seed=seed, max_tokens_override=max_tokens)


def warmup_messages(seed: int = 42) -> list[dict[str, Any]]:
    """预热用的短对话（不进入统计）。"""
    return [
        {"role": "system", "content": "You are a warmup probe. Reply with one short sentence."},
        {"role": "user", "content": f"warmup request {seed}"},
    ]
