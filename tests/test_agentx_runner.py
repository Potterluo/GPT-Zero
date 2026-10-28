"""Tests for the offline AgentX runner command construction (no harness needed)."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentx_offline.runner import build_command, parse_concurrency


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        url="http://llm:8000",
        model="Qwen3-8B",
        tokenizer="deepseek-ai/DeepSeek-V3",
        dataset="semianalysis_cc_traces_weka_062126",
        dataset_path="",
        duration=900,
        entries=8,
        stats_interval=30,
        trust_remote_code=False,
        no_realtime=False,
        extra=[],
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_public_dataset_official_duration() -> None:
    cmd = build_command(_args(), 8, Path("/r/c8"))
    assert "--public-dataset" in cmd
    assert cmd[cmd.index("--public-dataset") + 1] == "semianalysis_cc_traces_weka_062126"
    assert "--unsafe-override" not in cmd
    assert cmd[cmd.index("--concurrency") + 1] == "8"
    assert cmd[cmd.index("--benchmark-duration") + 1] == "900"


def test_short_duration_adds_single_override() -> None:
    cmd = build_command(_args(duration=120), 4, Path("/r/c4"))
    assert cmd.count("--unsafe-override") == 1


def test_local_trace_mode_adds_single_override() -> None:
    cmd = build_command(_args(dataset_path="/traces", duration=120), 2, Path("/r/c2"))
    assert cmd.count("--unsafe-override") == 1
    assert cmd[cmd.index("--input-file") + 1] == "/traces"
    assert cmd[cmd.index("--custom-dataset-type") + 1] == "weka_trace"
    assert "--public-dataset" not in cmd


def test_extra_passthrough_and_flags() -> None:
    cmd = build_command(
        _args(trust_remote_code=True, no_realtime=True, extra=["--chips", "8"]), 4, Path("/r/c4")
    )
    assert "--tokenizer-trust-remote-code" in cmd
    assert "--no-realtime-metrics" in cmd
    assert cmd[-2:] == ["--chips", "8"]


def test_parse_concurrency() -> None:
    assert parse_concurrency("1,4, 16") == [1, 4, 16]
