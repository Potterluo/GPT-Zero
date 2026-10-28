"""GPT-Zero benchmark harness.

只通过 HTTP 与 mock 服务交互（不 import ``gptzero`` 内部实现，tokenizer 计数除外，
见 CONTRACT.md §4）。入口：``python -m bench`` 或 ``gptzero-bench``。
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
