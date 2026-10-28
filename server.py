#!/usr/bin/env python3
"""Backwards-compatible entry point.

``python server.py`` keeps working exactly like the v2 server, but now boots the
``gptzero`` package (tool calling, prefix cache, Prometheus metrics, stats API).
"""

from __future__ import annotations

import sys


def main() -> int:
    from gptzero.cli import main as cli_main

    return cli_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
