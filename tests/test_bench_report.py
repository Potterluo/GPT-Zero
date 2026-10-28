"""报告生成测试：JSON 往返、Markdown/CSV 渲染、图表落盘、空数据健壮性。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.report import (
    build_charts,
    primary_profile,
    render_csv,
    render_markdown,
    report_from_file,
    write_outputs,
)
from bench.util import load_json, write_json


def _summary(ttft: float, tpot: float = 8.0) -> dict:
    def block(base: float, spread: float) -> dict:
        return {
            "count": 10,
            "p50": base,
            "p90": base + spread,
            "p95": base + spread * 1.2,
            "p99": base + spread * 1.5,
            "mean": base + spread / 2,
            "max": base + spread * 2,
            "min": base,
        }

    return {
        "ttft_ms": block(ttft, 10.0),
        "tpot_ms": block(tpot, 1.0),
        "itl_ms": block(tpot, 1.0),
        "e2e_ms": block(ttft + 500, 100.0),
    }


def _point(concurrency: int, profile: str = "coding", *, ok: bool = True) -> dict:
    total = 20
    success = total if ok else 18
    return {
        "concurrency": concurrency,
        "profile": profile,
        "wall_s": 10.0,
        "requests": {
            "total": total,
            "success": success,
            "failed": total - success,
            "failure_rate": (total - success) / total,
            "sub_requests": 2 if profile == "agentic" else 0,
            "tool_call_requests": 3,
        },
        "tokens": {"prompt": 100000, "completion": 2000, "tool_calls": 4},
        "rates": {
            "rps": 2.0,
            "success_rps": 1.9,
            "input_tokens_per_s": 10000.0,
            "output_tokens_per_s": 200.0,
            "total_tokens_per_s": 10200.0,
        },
        "summary": _summary(120 + concurrency * 5),
        "per_profile": {
            profile: {
                "requests": {"total": total, "success": success, "failed": total - success},
                "tokens": {"prompt": 100000, "completion": 2000, "tool_calls": 4},
                "summary": _summary(120 + concurrency * 5),
                "rps": 2.0,
                "input_tokens_per_s": 10000.0,
                "output_tokens_per_s": 200.0,
                "samples": {"ttft_ms": [100.0, 120.0, 140.0], "e2e_ms": [500.0, 600.0, 700.0]},
            }
        },
        "cache": {"queries_tokens": 100000.0, "hit_tokens": 50000.0, "hit_rate": 0.5, "source": "metrics"},
        "samples": {"ttft_ms": [100.0, 120.0, 140.0], "e2e_ms": [500.0, 600.0, 700.0]},
        "errors": [] if ok else [{"error": "ReadTimeout", "count": 2}],
        "server_config": {"ttft_ms": 120.0, "tpot_ms": 8.0, "max_concurrency": 64},
        "server_stats": {"ttft_ms": {"p90": 130.0}, "e2e_ms": {"p99": 900.0}},
        "metrics_delta": {},
    }


def _document(points: list[dict], profiles: list[str]) -> dict:
    return {
        "schema_version": 1,
        "tool": {"name": "gptzero-bench", "version": "0.1.0"},
        "name": "fixture",
        "generated_at": "2026-01-01T00:00:00+08:00",
        "config": {
            "url": "http://127.0.0.1:8090",
            "model": "gpt-3.5-turbo",
            "profiles": profiles,
            "profile": profiles[0],
            "concurrency": [2, 4],
            "duration_s": 10.0,
            "warmup": 2,
            "seed": 42,
            "max_tokens": None,
        },
        "server": {"url": "http://127.0.0.1:8090", "config": {"ttft_ms": 120.0, "tpot_ms": 8.0, "max_concurrency": 64}},
        "points": points,
        "charts": [],
        "notes": ["fixture note"],
    }


class TestMarkdown:
    def test_contains_table_and_conclusion(self) -> None:
        doc = _document([_point(2), _point(4)], ["coding"])
        md = render_markdown(doc)
        assert "# GPT-Zero 压测报告 — fixture" in md
        assert "| 并发 |" in md
        assert "TTFT p90 (ms)" in md
        assert "## 5. 结论" in md
        assert "profile `coding`" in md
        assert "复现命令" in md
        assert ">80" not in md  # no HTML artifacts

    def test_multiple_profiles(self) -> None:
        doc = _document([_point(2, "chat"), _point(2, "agentic")], ["chat", "agentic"])
        md = render_markdown(doc)
        assert "profile `chat`" in md
        assert "profile `agentic`" in md

    def test_empty_points_does_not_crash(self) -> None:
        doc = _document([], ["coding"])
        md = render_markdown(doc)
        assert "无有效测量点" in md

    def test_missing_summary_is_tolerated(self) -> None:
        point = _point(2)
        point["summary"] = {}
        point["cache"] = {}
        doc = _document([point], ["coding"])
        md = render_markdown(doc)
        assert "| 2 |" in md


class TestCSV:
    def test_header_and_rows(self) -> None:
        doc = _document([_point(2), _point(4)], ["coding"])
        text = render_csv(doc)
        lines = text.strip().splitlines()
        assert lines[0].startswith("profile,concurrency,requests")
        assert len(lines) == 3
        assert lines[1].startswith("coding,2,")

    def test_failure_rate_rendered(self) -> None:
        doc = _document([_point(2, ok=False)], ["coding"])
        text = render_csv(doc)
        assert ",0.1," in text


class TestWriteOutputs:
    def test_writes_json_md_csv_without_charts(self, tmp_path: Path) -> None:
        doc = _document([_point(2), _point(4)], ["coding"])
        out = tmp_path / "results" / "fixture.json"
        paths = write_outputs(doc, out, charts=False)
        assert Path(paths["md"]).exists()
        assert Path(paths["csv"]).exists()
        data = load_json(out)
        assert data["charts"] == []
        assert len(data["points"]) == 2
        # JSON 往返
        original = json.loads(json.dumps(doc, ensure_ascii=False))
        assert data["config"] == original["config"]

    def test_charts_png_and_svg_generated(self, tmp_path: Path) -> None:
        doc = _document([_point(2), _point(4)], ["coding"])
        out = tmp_path / "results" / "fixture.json"
        paths = write_outputs(doc, out, charts=True)
        charts_dir = tmp_path / "results" / "charts"
        pngs = sorted(charts_dir.glob("*.png"))
        svgs = sorted(charts_dir.glob("*.svg"))
        assert len(pngs) == 6, [p.name for p in pngs]
        assert len(svgs) == 6
        assert all(path.endswith(".png") for path in paths["charts"])

    def test_report_from_file_regenerates(self, tmp_path: Path) -> None:
        doc = _document([_point(2), _point(4)], ["coding"])
        out = tmp_path / "fixture.json"
        write_json(out, doc)
        paths = report_from_file(out, charts=False)
        assert Path(paths["md"]).exists()
        assert "fixture" in Path(paths["md"]).read_text(encoding="utf-8")


class TestCharts:
    def test_build_charts_empty_points(self, tmp_path: Path) -> None:
        doc = _document([], ["coding"])
        assert build_charts(doc, tmp_path) == []

    def test_primary_profile(self) -> None:
        doc = _document([_point(2, "chat"), _point(2, "coding")], ["chat", "coding"])
        assert primary_profile(doc) == "chat"
