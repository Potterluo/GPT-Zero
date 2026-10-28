"""AgentX 调研 + WEKA trace 合成/回放（GPT-Zero v3）。

模块：

* :mod:`agentx.trace_gen` — 与 ``agentx-harness==1.0.7`` 的
  ``aiperf.dataset.loader.weka_trace_models`` **逐字段对齐**的合成 trace 生成器
  与 ``validate_trace()`` 校验器。
* :mod:`agentx.replay` — 不依赖 aiperf 的闭环并发回放器：按 trace ``t`` 调度、
  子代理 SPAWN/JOIN 门控、cache-bust marker、prefix cache 命中统计、图表。
* :mod:`agentx.cli` — ``generate | replay | report | validate``。

文档见 ``docs/AGENTX.md``（参数表与「AgentX 参数 → mock 复现」映射）。
"""

from __future__ import annotations

__version__ = "1.0.0"

from .trace_gen import (  # noqa: F401  (对外 re-export，方便 tests / cli)
    BLOCK_SIZE_DEFAULT,
    GenConfig,
    generate,
    generate_session,
    load_traces,
    validate_trace,
    validate_traces,
)

__all__ = [
    "__version__",
    "BLOCK_SIZE_DEFAULT",
    "GenConfig",
    "generate",
    "generate_session",
    "load_traces",
    "validate_trace",
    "validate_traces",
]
