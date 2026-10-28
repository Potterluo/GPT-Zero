"""GPT-Zero: an offline, OpenAI-compatible mock inference server.

The package is intentionally dependency-light (fastapi/uvicorn/pydantic) so the
Docker image can run with no network access.  The optional ``transformers``
dependency is only used to get *exact* token counts from a local tokenizer
directory; without it a deterministic heuristic codec is used instead.
"""

__version__ = "3.0.0"

__all__ = ["__version__"]
