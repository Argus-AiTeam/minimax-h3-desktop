# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 A6000 Sol-Engine overlay package.

The package contains default-off exact-kernel candidates and audit references.
It does not load MiniMax-H3 weights, initialize CUDA, compile GPU kernels, or
enable any runtime integration at package import.

Imports are intentionally lazy: documentation and patch-builder tooling must
work on hosts that do not have PyTorch installed, while reference/kernel modules
retain their explicit PyTorch dependency when imported directly.
"""

from .env import DEFAULT_ENV_SWITCHES

__all__ = ["DEFAULT_ENV_SWITCHES"]
