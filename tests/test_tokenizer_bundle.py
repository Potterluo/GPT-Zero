"""Tokenizer 包内置（bundled tokenizer）契约测试。

PR #2 把 deepseek_v3_tokenizer/ 的副本放进 gptzero/tokenizer/ 并打进 wheel，
配合 pyproject 的硬依赖 tokenizers>=0.19，使 `pip install openai-mock` 后
`--tokenizer auto` 无需仓库 checkout 即得精确计数。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gptzero.tokenizer import CountingTokenizer, _candidate_paths

try:
    import tokenizers  # noqa: F401

    HAVE_TOKENIZERS = True
except ImportError:  # pragma: no cover - 老环境未装 tokenizers 时跳过
    HAVE_TOKENIZERS = False

REPO_TOKENIZER = Path(__file__).resolve().parent.parent / "deepseek_v3_tokenizer"
BUNDLED_TOKENIZER = Path(__file__).resolve().parent.parent / "gptzero" / "tokenizer"

pytestmark = pytest.mark.skipif(not HAVE_TOKENIZERS, reason="tokenizers 未安装")


def test_auto_resolves_hf_codec():
    """tokenizers 可用时，auto 必须解析到 HF tokenizer 而不是 heuristic 回退。"""
    tok = CountingTokenizer("auto")
    assert tok.codec.__class__.__name__ == "HFTokenCodec"
    assert not tok.warnings or all("tokenizer" in w for w in tok.warnings)


def test_candidate_paths_include_bundled_dir():
    """包内 gptzero/tokenizer 始终作为 auto 的最后候选出现。"""
    paths = _candidate_paths("auto")
    assert paths, "auto 至少要找到一个候选目录"
    bundled = [p for p in paths if Path(p).resolve() == BUNDLED_TOKENIZER.resolve()]
    assert bundled, f"包内目录 {BUNDLED_TOKENIZER} 不在候选里: {paths}"


@pytest.mark.skipif(not REPO_TOKENIZER.is_dir(), reason="仓库根 tokenizer 副本不存在（pip 安装场景）")
def test_bundle_counts_match_repo_copy():
    """包内副本与仓库根 tokenizer 是同一份文件：计数必须一致。"""
    tok_repo = CountingTokenizer(str(REPO_TOKENIZER))
    tok_bundle = CountingTokenizer(str(BUNDLED_TOKENIZER))
    sample = "The quick brown fox 跳过懒狗 12345\ndef add(a, b): return a + b"
    assert tok_repo.count(sample) == tok_bundle.count(sample) > 0
