"""AgentX / AIPerf WEKA trace 合成器。

严格对齐 wheel `agentx-harness==1.0.7` 内的 pydantic schema
（`aiperf/dataset/loader/weka_trace_models.py`，`extra="forbid"`）：

* 顶层：``id / models / block_size / hash_id_scope / tool_tokens / system_tokens
  / requests / totals``
* 普通请求 ``type="n"`` / 流式请求 ``type="s"``：
  ``t / type / model / in / out / hash_ids / input_types / output_types / stop
  / api_time / think_time / (ttft 仅 s)``
* 子代理 ``type="subagent"``：
  ``t / type / agent_id / subagent_type / duration_ms / total_tokens
  / tool_use_count / status / requests / models / tool_tokens / system_tokens``

[WARN] 时间单位：``t`` 是**秒**（相对会话起点），不是毫秒。AIPerf loader 内部
``req.t * 1000.0`` 转成毫秒后喂给 ``Turn.timestamp``。

生成的语料保持 AgentX 的真实形状：

- 每个 session = 一条主链（linear），可选插入 ``subagent`` 分支（SPAWN/JOIN）
- ``hash_ids`` 是 **64-token 块 ID**，session 内局部递增，天然共享前缀
- 首轮 ``hash_ids`` 为空（无缓存可命中），后续轮 ~98% 输入落在缓存前缀里
- ``in`` 中位数量级对齐官方（142k → 由 ``--max-isl`` 缩放）
- ``out`` 中位 444，长尾到数千
- ``t`` 按 ``api_time + think_time`` 递推，保证 START-to-START 间隔自洽，
  loader 折算出的 end-to-start idle gap 就等于 ``think_time``

用法::

    python -m agentx generate --sessions 8 --requests 30 --subagents 0.4 \
        --max-isl 32000 --out agentx/traces
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# schema 常量（与 wheel 内 pydantic 模型逐字段对齐）
# --------------------------------------------------------------------------- #

BLOCK_SIZE_DEFAULT = 64

#: 顶层允许字段（`extra="forbid"`，多一个就 ValidationError）
ALLOWED_TOP_KEYS = {
    "id",
    "models",
    "block_size",
    "hash_id_scope",
    "tool_tokens",
    "system_tokens",
    "requests",
    "totals",
}
TOP_REQUIRED = {
    "id",
    "models",
    "block_size",
    "hash_id_scope",
    "requests",
}

#: n / s 请求允许字段
ALLOWED_CALL_KEYS = {
    "t",
    "type",
    "model",
    "in",
    "out",
    "hash_ids",
    "input_types",
    "output_types",
    "stop",
    "api_time",
    "think_time",
    "ttft",
}
CALL_REQUIRED = {"t", "type", "model", "in", "out"}

#: subagent 允许字段
ALLOWED_SUBAGENT_KEYS = {
    "t",
    "type",
    "agent_id",
    "subagent_type",
    "duration_ms",
    "total_tokens",
    "tool_use_count",
    "status",
    "requests",
    "models",
    "tool_tokens",
    "system_tokens",
}
SUBAGENT_REQUIRED = {"t", "type", "agent_id", "subagent_type", "status", "requests", "models"}

#: 工具/系统前缀的合理区间（真实抓取量级）
SYSTEM_TOKENS_RANGE = (6000, 14000)
TOOL_TOKENS_RANGE = (8000, 24000)

#: 假设的硬件系数，仅用于给 ``api_time`` 一个可信量级（回放时不使用）
DEFAULT_PREFILL_TPS = 20_000.0
DEFAULT_TPOT_S = 0.008


# --------------------------------------------------------------------------- #
# 生成
# --------------------------------------------------------------------------- #


@dataclass
class GenConfig:
    """生成参数。默认值刻意贴近 AgentX 语料统计。"""

    sessions: int = 8
    requests: int = 30
    subagent_ratio: float = 0.4
    max_isl: int = 32_000
    block_size: int = BLOCK_SIZE_DEFAULT
    cached_ratio: float = 0.98
    out_median: int = 444
    seed: int = 42
    model: str = "gpt-3.5-turbo"
    prefill_tps: float = DEFAULT_PREFILL_TPS
    tpot_s: float = DEFAULT_TPOT_S


@dataclass
class _SessionState:
    """单条 trace 的生成中间态（主链 + 子代理各自的 hash id 命名空间）。"""

    trace_id: str
    rng: random.Random
    block_size: int
    system_tokens: int
    tool_tokens: int
    model: str
    chain_ids: list[int] = field(default_factory=list)
    next_id: int = 0
    clock: float = 0.0  # 当前主链"最后一个请求结束"的时刻（秒）

    def extend_ids(self, count: int) -> list[int]:
        """把 session 的块 ID 空间扩展到至少 ``count`` 个，返回前缀。"""
        while len(self.chain_ids) < count:
            self.chain_ids.append(self.next_id)
            self.next_id += 1
        return self.chain_ids[:count]


def _lognormal(rng: random.Random, median: float, sigma: float) -> float:
    return math.exp(rng.gauss(math.log(max(median, 1e-9)), sigma))


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def _out_tokens(rng: random.Random, median: int) -> int:
    """输出 token 数：lognormal 中位 = ``median``，5% 长尾。"""
    if rng.random() < 0.05:
        value = _lognormal(rng, median * 4.0, 0.8)
    else:
        value = _lognormal(rng, median, 0.75)
    return int(_clamp(round(value), 8, 8192))


def _api_time(cfg: GenConfig, in_tokens: int, out_tokens: int, rng: random.Random) -> float:
    """抓取时的服务端耗时（秒）。只影响 t 递推与 JOIN 判定。"""
    base = in_tokens / cfg.prefill_tps + out_tokens * cfg.tpot_s
    return round(max(base * rng.uniform(0.85, 1.25), 0.01), 4)


def _think_time(rng: random.Random) -> float:
    """用户/工具在两次请求之间的思考时间（秒）。多数很短，长尾到几分钟。"""
    value = _lognormal(rng, 3.0, 1.5)
    if rng.random() < 0.06:  # 长尾：开会/吃饭
        value *= rng.uniform(4.0, 12.0)
    return round(_clamp(value, 0.05, 900.0), 4)


def _cached_fraction(rng: random.Random, target: float) -> float:
    """单请求的缓存前缀占比。围绕 ``target`` 抖动，且永远 < 1（总有一段新 token）。"""
    value = target + rng.gauss(0.0, 0.012)
    return _clamp(value, 0.90, 0.995)


def _make_call(
    *,
    t: float,
    in_tokens: int,
    out_tokens: int,
    hash_ids: list[int],
    model: str,
    rng: random.Random,
    cfg: GenConfig,
    streaming: bool,
    stop: str,
    think_time: float,
) -> dict[str, Any]:
    api = _api_time(cfg, in_tokens, out_tokens, rng)
    call: dict[str, Any] = {
        "t": round(t, 4),
        "type": "s" if streaming else "n",
        "model": model,
        "in": int(in_tokens),
        "out": int(out_tokens),
        "hash_ids": list(hash_ids),
        "input_types": ["text"],
        "output_types": ["tool_use" if stop == "tool_use" else "text"],
        "stop": stop,
        "api_time": api,
        "think_time": round(think_time, 4),
    }
    if streaming:
        # 抓取记录里的 TTFT，仅作记录（回放不读）
        call["ttft"] = round(api * rng.uniform(0.05, 0.18), 4)
    return call


def _make_subagent(
    *,
    entry_t: float,
    parent: _SessionState,
    cfg: GenConfig,
    agent_index: int,
) -> tuple[dict[str, Any], float]:
    """造一个 ``type="subagent"`` 块，返回 (块, 结束时刻秒)。

    子代理有自己的 system/tool 前缀和自己的 hash id 命名空间（不同 agent 的
    KV 前缀不共享），内部是一条 3~8 轮的短链；``duration_ms`` 与内层请求的
    ``t + api_time`` 自洽 —— loader 用 ``entry.t + duration_ms/1000`` 做 JOIN
    门控，两者不一致会造成父链提前/延后唤醒。
    """
    rng = parent.rng
    inner_model = cfg.model
    n_inner = rng.randint(3, 8)
    # Keep the declared prefix small enough that the first inner request can
    # physically contain it: AIPerf reconstructs the turn-0 system prefix from
    # hash_ids and rejects the trace when ceil((system+tool)/block) exceeds the
    # recorded block count.
    sub_system = rng.randint(2000, 4000)
    sub_tool = rng.randint(3000, 6000)
    sub_prefix = sub_system + sub_tool
    sub_prefix_blocks = math.ceil(sub_prefix / cfg.block_size)

    ids: list[int] = []
    next_id = 0
    inner: list[dict[str, Any]] = []
    clock = entry_t
    total_in = 0
    total_out = 0

    for k in range(n_inner):
        # 子代理的上下文从自己前缀开始，逐轮增长（比主链短很多）。
        # 注意：即使 k==0 也必须给 hash_ids —— 空 hash_ids 会让 AIPerf 走
        # prompt_generator.generate() 的随机路径，与后续轮次不共享任何前缀字节。
        base = sub_prefix + int(k * rng.uniform(1200, 4200))
        in_tokens = min(base, max(cfg.max_isl // 3, sub_prefix + 4096), cfg.max_isl)
        in_tokens = max(in_tokens, sub_prefix + 512)
        cached_frac = _cached_fraction(rng, cfg.cached_ratio)
        cached_blocks = int((in_tokens * cached_frac) // cfg.block_size)
        cached_blocks = max(cached_blocks, sub_prefix_blocks)
        cached_blocks = min(cached_blocks, in_tokens // cfg.block_size)
        while len(ids) < cached_blocks:
            ids.append(next_id)
            next_id += 1
        out_tokens = _out_tokens(rng, max(80, cfg.out_median // 3))
        think = 0.0 if k == 0 else _think_time(rng) * 0.3
        t_k = clock + think
        call = _make_call(
            t=t_k,
            in_tokens=in_tokens,
            out_tokens=out_tokens,
            hash_ids=ids[:cached_blocks],
            model=inner_model,
            rng=rng,
            cfg=cfg,
            streaming=True,
            stop="tool_use" if k < n_inner - 1 else "end_turn",
            think_time=think,
        )
        inner.append(call)
        total_in += in_tokens
        total_out += out_tokens
        clock = t_k + call["api_time"]

    end = clock
    block: dict[str, Any] = {
        "t": round(entry_t, 4),
        "type": "subagent",
        "agent_id": f"agent_{agent_index:03d}",
        "subagent_type": rng.choice(["Explore", "Plan", "general-purpose"]),
        # ceil 而不是 round：loader 用 entry.t + duration_ms/1000 做 JOIN 门控，
        # 四舍五入可能让结束时刻早于最后一条内层请求，导致父链提前唤醒。
        "duration_ms": int(math.ceil(max(end - entry_t, 0.0) * 1000.0)),
        "total_tokens": total_in + total_out,
        "tool_use_count": n_inner,
        "status": "completed",
        "requests": inner,
        "models": [inner_model],
        "tool_tokens": sub_tool,
        "system_tokens": sub_system,
    }
    return block, end


def generate_session(index: int, cfg: GenConfig) -> dict[str, Any]:
    """生成一条 session 的 WEKA trace（dict，可直接 json.dump）。"""
    rng = random.Random((cfg.seed << 20) ^ (index * 2_654_435_761))
    trace_id = f"agentx-syn-{index:06d}"
    system_tokens = rng.randint(*SYSTEM_TOKENS_RANGE)
    tool_tokens = rng.randint(*TOOL_TOKENS_RANGE)
    # 前缀（system + tools）必须给后续增长留空间，否则 --max-isl 会被突破
    if system_tokens + tool_tokens > 0.55 * cfg.max_isl:
        scale = (0.55 * cfg.max_isl) / (system_tokens + tool_tokens)
        system_tokens = max(256, int(system_tokens * scale))
        tool_tokens = max(256, int(tool_tokens * scale))
    prefix = system_tokens + tool_tokens

    state = _SessionState(
        trace_id=trace_id,
        rng=rng,
        block_size=cfg.block_size,
        system_tokens=system_tokens,
        tool_tokens=tool_tokens,
        model=cfg.model,
    )

    n_turns = max(4, cfg.requests)
    # 每条 session 的峰值上下文在 [35%, 100%] * max_isl 之间，保证分布有层次
    peak_target = int(rng.uniform(0.35, 1.0) * cfg.max_isl)
    peak_target = min(cfg.max_isl, max(peak_target, prefix + 2048))
    prefix_blocks = math.ceil(prefix / cfg.block_size)
    # Turn 0 must carry at least the declared system+tools prefix as blocks,
    # otherwise AIPerf aborts with "system prefix requires N hash blocks".
    min_first = prefix_blocks * cfg.block_size + cfg.block_size
    first_in = min(max(prefix + rng.randint(400, 2500), min_first), peak_target)
    first_in = max(first_in, prefix + 256)

    requests: list[dict[str, Any]] = []
    agent_index = 0
    clock = 0.0

    for k in range(n_turns):
        if k == 0:
            think = 0.0
            in_tokens = first_in
            t_k = 0.0
        else:
            # 上下文增长：早期快、后期趋平（贴着 peak_target 收敛）
            frac = (k / max(n_turns - 1, 1)) ** 0.7
            target = first_in + (peak_target - first_in) * frac
            in_tokens = int(_clamp(target * rng.uniform(0.97, 1.03), prefix + 256, peak_target))
            think = _think_time(rng)
            t_k = clock + think

        # 每一轮都给出覆盖自身输入的块 ID（含首轮）。
        # 首轮虽然"服务端必定冷"（AIPerf 用 cache-bust marker 打断），但 hash_ids
        # 必须非空：`hash_ids=[]` 会退化成 prompt_generator.generate() 的随机路径，
        # 与后续轮由 `_build_token_sequence` 重建的 prompt 不共享任何前缀字节，
        # 导致 prefix cache 曲线在真实 harness 里失真。
        cached_frac = _cached_fraction(rng, cfg.cached_ratio)
        n_blocks = int((in_tokens * cached_frac) // cfg.block_size)
        n_blocks = max(n_blocks, prefix_blocks)
        n_blocks = min(n_blocks, in_tokens // cfg.block_size)
        hash_ids = state.extend_ids(n_blocks) if n_blocks else []

        out_tokens = _out_tokens(rng, cfg.out_median)
        streaming = rng.random() < 0.92
        stop = "end_turn" if k == n_turns - 1 else "tool_use"
        call = _make_call(
            t=t_k,
            in_tokens=in_tokens,
            out_tokens=out_tokens,
            hash_ids=list(hash_ids),
            model=cfg.model,
            rng=rng,
            cfg=cfg,
            streaming=streaming,
            stop=stop,
            think_time=think,
        )
        requests.append(call)
        clock = t_k + call["api_time"]

        # 子代理扇出：turn 1 之后、非末轮，按概率插入；父链下一条请求是 JOIN 点
        if (
            k >= 1
            and k < n_turns - 1
            and rng.random() < cfg.subagent_ratio
        ):
            agent_index += 1
            entry_t = clock + rng.uniform(0.05, 1.5)
            block, end = _make_subagent(
                entry_t=entry_t, parent=state, cfg=cfg, agent_index=agent_index
            )
            requests.append(block)
            clock = end  # JOIN：父链必须等子代理结束

    total_in = sum(
        r.get("in", 0)
        for r in requests
        if r["type"] in ("n", "s")
    ) + sum(
        sum(ir.get("in", 0) for ir in r.get("requests", []))
        for r in requests
        if r["type"] == "subagent"
    )
    total_out = sum(
        r.get("out", 0) for r in requests if r["type"] in ("n", "s")
    ) + sum(
        sum(ir.get("out", 0) for ir in r.get("requests", []))
        for r in requests
        if r["type"] == "subagent"
    )

    return {
        "id": trace_id,
        "models": [cfg.model],
        "block_size": cfg.block_size,
        "hash_id_scope": "local",
        "tool_tokens": tool_tokens,
        "system_tokens": system_tokens,
        "requests": requests,
        "totals": {
            "requests": sum(1 for r in requests if r["type"] in ("n", "s")),
            "subagents": sum(1 for r in requests if r["type"] == "subagent"),
            "input_tokens": total_in,
            "output_tokens": total_out,
            "duration_s": round(clock, 3),
        },
    }


def generate(cfg: GenConfig, out_dir: str | Path) -> list[Path]:
    """生成 ``cfg.sessions`` 条 trace 到 ``out_dir``（每文件一条）。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i in range(cfg.sessions):
        trace = generate_session(i, cfg)
        path = out / f"{trace['id']}.json"
        path.write_text(
            json.dumps(trace, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        written.append(path)
    return written


# --------------------------------------------------------------------------- #
# 校验（手写，不依赖 pydantic —— 项目 venv 不保证有 aiperf）
# --------------------------------------------------------------------------- #


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_call(obj: dict[str, Any], where: str, errors: list[str], warns: list[str]) -> None:
    extra = set(obj) - ALLOWED_CALL_KEYS
    if extra:
        errors.append(f"{where}: 非法字段 {sorted(extra)}（schema extra=forbid）")
    missing = CALL_REQUIRED - set(obj)
    if missing:
        errors.append(f"{where}: 缺必填字段 {sorted(missing)}")
        return
    if obj.get("type") not in ("n", "s"):
        errors.append(f"{where}: type 必须是 'n' 或 's'，实际 {obj.get('type')!r}")
    if not _is_num(obj.get("t")):
        errors.append(f"{where}: t 必须是数字（秒）")
    elif obj["t"] < 0:
        errors.append(f"{where}: t 不能为负（实际 {obj['t']}）")
    if not isinstance(obj.get("model"), str) or not obj["model"]:
        errors.append(f"{where}: model 必须是非空字符串")
    if not _is_int(obj.get("in")) or obj["in"] <= 0:
        errors.append(f"{where}: in 必须是正整数，实际 {obj.get('in')!r}")
    if not _is_int(obj.get("out")) or obj["out"] < 0:
        errors.append(f"{where}: out 必须是非负整数，实际 {obj.get('out')!r}")
    hash_ids = obj.get("hash_ids", [])
    if not isinstance(hash_ids, list) or any(not _is_int(h) for h in hash_ids):
        errors.append(f"{where}: hash_ids 必须是 int 列表")
        hash_ids = []
    for key in ("input_types", "output_types"):
        val = obj.get(key, [])
        if not isinstance(val, list) or any(not isinstance(x, str) for x in val):
            errors.append(f"{where}: {key} 必须是 str 列表")
    if not isinstance(obj.get("stop", ""), str):
        errors.append(f"{where}: stop 必须是字符串")
    for key in ("api_time", "think_time", "ttft"):
        if key in obj and obj[key] is not None and not _is_num(obj[key]):
            errors.append(f"{where}: {key} 必须是数字或 null")
    if obj.get("type") == "n" and "ttft" in obj:
        warns.append(f"{where}: type='n' 不该带 ttft（schema 只允许 s）")
    if not hash_ids and obj.get("in", 0) > 0:
        # 合法（全 fresh 请求），但值得记账：真实语料首轮之外的请求几乎都带前缀
        warns.append(f"{where}: hash_ids 为空，该请求全部走 prefill")


def validate_trace(trace: Any, *, name: str = "<trace>") -> dict[str, list[str]]:
    """校验单条 trace，返回 ``{"errors": [...], "warnings": [...]}``。

    覆盖三类问题：
    1. schema 层（字段名/类型/必填，等价 pydantic ``extra="forbid"``）
    2. 时间层（t 单调、subagent 内层 t 归属根坐标系、JOIN 时序自洽）
    3. 缓存层（hash_ids 不越界、首轮无缓存、块 ID 局部递增可共享前缀）
    """
    errors: list[str] = []
    warns: list[str] = []
    if not isinstance(trace, dict):
        return {"errors": [f"{name}: 顶层必须是 JSON object"], "warnings": []}

    extra = set(trace) - ALLOWED_TOP_KEYS
    if extra:
        errors.append(f"{name}: 顶层非法字段 {sorted(extra)}")
    missing = TOP_REQUIRED - set(trace)
    if missing:
        errors.append(f"{name}: 顶层缺必填字段 {sorted(missing)}")
        return {"errors": errors, "warnings": warns}

    if not isinstance(trace["id"], str) or not trace["id"]:
        errors.append(f"{name}: id 必须是非空字符串")
    models = trace["models"]
    if not isinstance(models, list) or not models or any(not isinstance(m, str) for m in models):
        errors.append(f"{name}: models 必须是非空 str 列表")
    block_size = trace["block_size"]
    if not _is_int(block_size) or block_size <= 0:
        errors.append(f"{name}: block_size 必须是正整数")
        block_size = BLOCK_SIZE_DEFAULT
    if trace["hash_id_scope"] != "local":
        errors.append(
            f"{name}: hash_id_scope 只能是 'local'（实际 {trace['hash_id_scope']!r}）"
        )
    for key in ("tool_tokens", "system_tokens"):
        if key in trace and (not _is_int(trace[key]) or trace[key] < 0):
            errors.append(f"{name}: {key} 必须是非负整数")

    requests = trace["requests"]
    if not isinstance(requests, list) or not requests:
        errors.append(f"{name}: requests 必须是非空列表")
        return {"errors": errors, "warnings": warns}

    last_outer_t = -1.0
    prev_call_t = -1.0
    pending_join_until: float | None = None
    prev_hash_ids: list[int] = []

    for idx, req in enumerate(requests):
        where = f"{name}[{idx}]"
        if not isinstance(req, dict):
            errors.append(f"{where}: 必须是 object")
            continue

        if req.get("type") == "subagent":
            extra = set(req) - ALLOWED_SUBAGENT_KEYS
            if extra:
                errors.append(f"{where}: 非法字段 {sorted(extra)}")
            missing = SUBAGENT_REQUIRED - set(req)
            if missing:
                errors.append(f"{where}: 缺必填字段 {sorted(missing)}")
                continue
            t = req.get("t")
            if not _is_num(t) or t < 0:
                errors.append(f"{where}: subagent.t 必须是非负数")
                continue
            if t < last_outer_t:
                errors.append(f"{where}: subagent.t={t} 早于前一个外层请求 t={last_outer_t}")
            last_outer_t = t
            duration = req.get("duration_ms")
            if duration is not None and (not _is_int(duration) or duration < 0):
                errors.append(f"{where}: duration_ms 必须是非负整数或 null")
            if not isinstance(req.get("status"), str) or not req["status"]:
                errors.append(f"{where}: status 必须是非空字符串")
            if not isinstance(req.get("models"), list) or not req["models"]:
                errors.append(f"{where}: models 必须是非空列表")
            inner = req.get("requests")
            if not isinstance(inner, list):
                errors.append(f"{where}: requests 必须是列表")
                inner = []
            inner_clock = t
            inner_prev_hash: list[int] = []
            inner_total_out = 0
            for j, sub in enumerate(inner):
                sub_where = f"{where}.requests[{j}]"
                if not isinstance(sub, dict):
                    errors.append(f"{sub_where}: 必须是 object")
                    continue
                validate_call(sub, sub_where, errors, warns)
                # 子代理是一条独立 stream：内部同样要求非空前缀 + 严格前缀递增
                inner_prev_hash = _check_hash_ids(
                    sub, sub_where, block_size, inner_prev_hash, errors, warns
                )
                st = sub.get("t")
                if _is_num(st):
                    # loader 约定：内层 t < entry.t 视为相对时间，否则视为绝对时间
                    abs_t = t + st if st < t else st
                    if abs_t < inner_clock - 1e-6:
                        warns.append(
                            f"{sub_where}: 内层 t 回退（{abs_t:.3f} < {inner_clock:.3f}），"
                            "loader 会重排但会损失原始交错"
                        )
                    inner_clock = max(inner_clock, abs_t + float(sub.get("api_time") or 0.0))
                inner_total_out += int(sub.get("out") or 0)
            if duration is None:
                end = inner_clock
                warns.append(
                    f"{where}: duration_ms=null（async_launched），JOIN 结束时刻回退到 "
                    f"max(inner.t+api_time)={end:.3f}s"
                )
            else:
                end = t + duration / 1000.0
            # JOIN：subagent 之后的第一条父级请求必须不早于 end
            pending_join_until = end
            if req.get("total_tokens") is not None and not _is_int(req["total_tokens"]):
                errors.append(f"{where}: total_tokens 必须是整数或 null")
            if req.get("tool_use_count") is not None and not _is_int(req["tool_use_count"]):
                errors.append(f"{where}: tool_use_count 必须是整数或 null")
            continue

        # ---- 普通 / 流式请求 ----
        validate_call(req, where, errors, warns)
        t = req.get("t")
        if not _is_num(t):
            continue
        if t < last_outer_t - 1e-9:
            errors.append(f"{where}: 外层 t 非单调（{t} < {last_outer_t}）")
        if pending_join_until is not None:
            if t + 1e-6 < pending_join_until:
                errors.append(
                    f"{where}: JOIN 违约 —— 父链在 {t}s 发起，但子代理到 "
                    f"{pending_join_until:.3f}s 才结束"
                )
            pending_join_until = None
        last_outer_t = t
        if prev_call_t >= 0 and t < prev_call_t:
            errors.append(f"{where}: 主链 t 回退（{t} < {prev_call_t}）")
        prev_call_t = t

        prev_hash_ids = _check_hash_ids(
            req, where, block_size, prev_hash_ids, errors, warns
        )

    if pending_join_until is not None:
        warns.append(f"{name}: 最后一个 subagent 没有 JOIN 请求（trace 在此截断，可接受）")

    return {"errors": errors, "warnings": warns}


def _check_hash_ids(
    req: dict[str, Any],
    where: str,
    block_size: int,
    prev_hash_ids: list[int],
    errors: list[str],
    warns: list[str],
) -> list[int]:
    """校验 hash_ids 的三条硬约束，返回本轮 hash_ids 供下一轮做前缀比对。

    1. 非空（空会让 AIPerf 走随机 prompt 路径，与后续轮次字节不共享）
    2. ``len(hash_ids) * block_size <= in``（块前缀不能超过输入）
    3. 同一 stream 内严格前缀递增（KV cache 的 LCP 语义）
    """
    hash_ids = req.get("hash_ids") or []
    if not hash_ids:
        in_tokens = req.get("in")
        if _is_int(in_tokens) and in_tokens >= block_size:
            errors.append(
                f"{where}: hash_ids 为空。AIPerf 对空 hash_ids 会走 "
                "prompt_generator.generate() 的随机 prompt，与后续轮次不共享"
                "任何前缀字节，prefix cache 曲线会失真 —— 每轮（含首轮）都必须"
                "给出 len(hash_ids)*block_size <= in 的块 ID"
            )
        else:
            warns.append(f"{where}: in={in_tokens} 小于一个块，无法给出 hash_ids")
        return prev_hash_ids

    if not all(_is_int(h) for h in hash_ids):
        errors.append(f"{where}: hash_ids 必须是 int 列表")
        return prev_hash_ids
    if hash_ids != sorted(set(hash_ids)):
        errors.append(f"{where}: hash_ids 必须严格递增且无重复")
    cached_tokens = len(hash_ids) * block_size
    if _is_int(req.get("in")) and cached_tokens > req["in"]:
        errors.append(
            f"{where}: 缓存前缀 {cached_tokens} token（{len(hash_ids)} 块 × "
            f"{block_size}）超过 in={req['in']}"
        )
    if prev_hash_ids:
        lcp = 0
        for a, b in zip(prev_hash_ids, hash_ids):
            if a != b:
                break
            lcp += 1
        if lcp < min(len(prev_hash_ids), len(hash_ids)):
            errors.append(
                f"{where}: hash_ids 与上一轮的最长公共前缀只有 {lcp} 块"
                "（应为严格前缀扩展，否则同一 stream 的 prefix cache 不连续）"
            )
    return list(hash_ids)


def validate_traces(traces: Iterable[dict[str, Any]]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warns: list[str] = []
    ids: set[str] = set()
    for trace in traces:
        result = validate_trace(trace, name=str(trace.get("id", "<trace>")))
        errors.extend(result["errors"])
        warns.extend(result["warnings"])
        tid = trace.get("id")
        if isinstance(tid, str):
            if tid in ids:
                errors.append(f"{tid}: trace id 重复（loader 会抛 Duplicate trace id）")
            ids.add(tid)
    return errors, warns


def load_traces(path: str | Path) -> list[dict[str, Any]]:
    """读目录（``*.json``，按文件名排序，与 AIPerf 的发现顺序一致）或单文件。"""
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.json"))
    elif p.is_file():
        files = [p]
    else:
        raise FileNotFoundError(f"找不到 trace 路径: {p}")
    out = []
    for f in files:
        out.append(json.loads(f.read_text(encoding="utf-8")))
    return out
