# SPDX-License-Identifier: Apache-2.0
"""Repeatable vLLM-Omni opt-in patch builder.

The builder writes a patch file only. It does not edit the locked
``runtime/.../src/vllm-omni`` worktree and it does not import vLLM-Omni.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .env import DEFAULT_ENV_SWITCHES

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PATCH_SOURCE = PACKAGE_ROOT / "patches" / "vllm_omni_h3_a6000_opt_in.patch"


def write_patch(output: str | Path) -> Path:
    """Copy the audited opt-in patch to ``output`` and return its path."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PATCH_SOURCE, output)
    return output


def env_switch_report() -> str:
    """Human-readable default switch table for installer logs."""

    return "\n".join(f"{key}={value}" for key, value in sorted(DEFAULT_ENV_SWITCHES.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("vllm_omni_h3_a6000_opt_in.patch"))
    parser.add_argument(
        "--print-env",
        action="store_true",
        help="print default env switches; every value is off by default",
    )
    args = parser.parse_args(argv)
    out = write_patch(args.output)
    print(out)
    if args.print_env:
        print(env_switch_report())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
