"""Local model-directory scanning helpers (v2 compatible).

The mock does not need model weights, only tokenizer files.  These helpers let
the dashboard point at a directory full of checkpoints and map a request's
``model`` field onto the best matching tokenizer directory.
"""

from __future__ import annotations

import os
import re
from typing import Any


def normalize_model_name(name: str) -> str:
    name = name.lower().replace("-", "_")
    for pattern in (r"_fp\d+", r"_bf\d+", r"_int\d+", r"_q\d+", r"_awq", r"_gptq", r"_gguf"):
        name = re.sub(pattern, "", name)
    for pattern in (r"_instruct", r"_chat", r"_v\d+", r"_revision"):
        name = re.sub(pattern, "", name)
    return name.strip("_")


def scan_model_directory(directory: str) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    if not directory or not os.path.isdir(directory):
        return models
    try:
        for item in os.listdir(directory):
            path = os.path.join(directory, item)
            if not os.path.isdir(path):
                continue
            has_tokenizer = any(
                os.path.exists(os.path.join(path, f))
                for f in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")
            )
            models.append(
                {
                    "name": item,
                    "path": path,
                    "has_tokenizer": has_tokenizer,
                    "normalized_name": normalize_model_name(item),
                    "aliases": [item, normalize_model_name(item), path],
                }
            )
    except OSError:
        return models
    return sorted(models, key=lambda m: m["name"])


def match_model(requested: str, available: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not requested or not available:
        return None
    for model in available:
        if model["name"] == requested or model["path"] == requested:
            return model
    req = normalize_model_name(requested)
    for model in available:
        if normalize_model_name(model["name"]) == req or normalize_model_name(os.path.basename(model["path"])) == req:
            return model
    for model in available:
        name = normalize_model_name(model["name"])
        path = normalize_model_name(os.path.basename(model["path"]))
        if req in name or name in req or req in path or path in req:
            return model
    return None
