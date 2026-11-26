"""GPT-Zero: an offline, multi-protocol mock inference server.

The package is intentionally dependency-light (fastapi/uvicorn/pydantic) so the
Docker image can run with no network access.  One mock engine answers three
wire protocols — OpenAI Chat Completions, the OpenAI Responses API and the
Anthropic Messages API — with fake tool calls, exact token counts and a
simulated prefix cache.  The optional ``transformers`` dependency is only used
to get *exact* token counts from a local tokenizer directory; without it a
deterministic heuristic codec is used instead.
"""

__version__ = "4.0.0"

__all__ = ["__version__"]
