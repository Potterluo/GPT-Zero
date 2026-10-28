# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""WekaTrace pydantic 模型镜像（用于离线校验合成 trace 的 schema 兼容性）。

来源：``agentx-harness==1.0.7`` 的 ``aiperf/dataset/loader/weka_trace_models.py``。
唯一改动：把 `AIPerfBaseModel`（继承 AutoRoutedModel，带自动路由）替换成本地
等价基类 —— 只保留与本 schema 相关的 pydantic 配置
（``arbitrary_types_allowed=True, extra="allow"``）。字段名、别名、默认值、
类型、``extra="forbid"`` 与 ``populate_by_name`` 全部与上游逐字一致。

这样即使环境里没装 ``aiperf``（Docker 离线构建、CI），也能断言"我们生成的
trace 能被真实 harness 的 pydantic 模型接受"。
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class AIPerfBaseModel(BaseModel):  # 上游: aiperf.common.models.AIPerfBaseModel
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class WekaNormalRequest(AIPerfBaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    t: float = Field(description="Request timestamp in seconds from conversation start.")
    type: Literal["n"] = Field(description="Discriminator: normal API call.")
    model: str = Field(description="Model identifier for this request.")
    input_length: int = Field(alias="in", description="Input token count.")
    output_length: int = Field(alias="out", description="Output token count.")
    hash_ids: list[int] = Field(default_factory=list, description="KV-cache block hash IDs.")
    input_types: list[str] = Field(default_factory=list)
    output_types: list[str] = Field(default_factory=list)
    stop: str = Field(default="")
    api_time: float | None = Field(default=None)
    think_time: float | None = Field(default=None)


class WekaStreamingRequest(AIPerfBaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    t: float = Field(description="Request timestamp in seconds from conversation start.")
    type: Literal["s"] = Field(description="Discriminator: streaming API call.")
    model: str = Field(description="Model identifier for this request.")
    input_length: int = Field(alias="in", description="Input token count.")
    output_length: int = Field(alias="out", description="Output token count.")
    hash_ids: list[int] = Field(default_factory=list)
    input_types: list[str] = Field(default_factory=list)
    output_types: list[str] = Field(default_factory=list)
    stop: str = Field(default="")
    api_time: float | None = Field(default=None)
    think_time: float | None = Field(default=None)
    ttft: float | None = Field(default=None)


class WekaSubagentEntry(AIPerfBaseModel):
    model_config = ConfigDict(extra="forbid")

    t: float = Field(description="Spawn timestamp in seconds from conversation start.")
    type: Literal["subagent"] = Field(description="Discriminator: subagent marker.")
    agent_id: str = Field(description="Opaque subagent identifier, e.g. 'agent_001'.")
    subagent_type: str = Field(description="Subagent type, e.g. 'Explore'.")
    duration_ms: int | None = Field(default=None)
    total_tokens: int | None = Field(default=None)
    tool_use_count: int | None = Field(default=None)
    status: str = Field(description="'completed' or other terminal status.")
    requests: list[
        Annotated[WekaNormalRequest | WekaStreamingRequest, Field(discriminator="type")]
    ] = Field(description="Inner requests of the subagent (normal or streaming API calls).")
    models: list[str] = Field(description="Models used by the subagent.")
    tool_tokens: int = Field(default=0)
    system_tokens: int = Field(default=0)


WekaRequest: TypeAlias = Annotated[
    WekaNormalRequest | WekaStreamingRequest | WekaSubagentEntry,
    Field(discriminator="type"),
]


class WekaTrace(AIPerfBaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Trace identifier (session ID).")
    models: list[str] = Field(description="Models used in the trace.")
    block_size: int = Field(gt=0, description="Cache block size in tokens.")
    hash_id_scope: Literal["local"] = Field(description="Hash ID namespace scope.")
    tool_tokens: int = Field(default=0)
    system_tokens: int = Field(default=0)
    requests: list[WekaRequest] = Field(description="Interleaved normal requests and subagent markers.")
    totals: dict[str, Any] | None = Field(default=None)
