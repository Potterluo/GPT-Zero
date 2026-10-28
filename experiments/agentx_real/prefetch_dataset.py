"""Prefetch the real AgentX corpus (offline prep step).

Downloads ``semianalysisai/cc-traces-weka-062126`` through the HF mirror and
materializes the Arrow cache + a dataset copy under ``data/hf`` so every later
AgentX run can use ``HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1``.

Run with the agentx venv:
    .venv-agentx\\Scripts\\python.exe experiments\\agentx_real\\prefetch_dataset.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "hf"
os.environ.setdefault("HF_HOME", str(DATA))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("HF_DATASETS_OFFLINE", None)

REPO = "semianalysisai/cc-traces-weka-062126"
EXPORT = DATA / "weka_062126"


def main() -> int:
    from datasets import load_dataset

    print(f"[prefetch] HF_HOME={DATA}")
    print(f"[prefetch] HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
    print(f"[prefetch] repo={REPO}")
    t0 = time.time()
    ds = load_dataset(REPO, split="train")
    print(f"[prefetch] loaded rows={len(ds)} in {time.time() - t0:.1f}s")
    print(f"[prefetch] features={list(ds.features)}")
    print(f"[prefetch] cache_files={ds.cache_files}")
    if not EXPORT.exists():
        t1 = time.time()
        ds.save_to_disk(str(EXPORT))
        print(f"[prefetch] saved to {EXPORT} in {time.time() - t1:.1f}s")
    print("[prefetch] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
