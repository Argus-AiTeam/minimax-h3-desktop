#!/usr/bin/env python3
"""CLI entry point for the pre-gate MiniMax-H3 verifier."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from argus_h3_verifier import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
