#!/usr/bin/env python3
"""CLI for the canonical MiniMax-H3 A6000 benchmark contract v1."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from minimax_h3_benchmark_contract import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
