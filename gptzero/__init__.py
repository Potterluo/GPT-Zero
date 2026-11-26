"""GPT-Zero: an offline, multi-protocol mock inference server.

The package is intentionally dependency-light (fastapi/uvicorn/pydantic plus a
small Rust ``tokenizers`` wheel) so the Docker image can run with no network
access.  One mock engine answers three wire protocols — OpenAI Chat
Completions, the OpenAI Responses API and the Anthropic Messages API — with
fake tool calls, exact token counts and a simulated prefix cache.  Exact
counts come from the tokenizer bundled inside the package
(``gptzero/tokenizer/``); a deterministic heuristic codec remains the
fallback, and ``transformers`` stays optional for repositories that only ship
``tokenizer.model``-style tokenizers.
"""

__version__ = "4.0.0"

__all__ = ["__version__"]
