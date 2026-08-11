#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Install the r7 vLLM-Omni patch outputs into both image import surfaces.

The r7 Dockerfile applies the current MiniMax-H3 A6000 opt-in patch to a
throwaway copy of `/app/vllm-omni`, then invokes this helper.  The helper audits
that the patch-changed file list is exactly the locked r7 list, installs every
changed file to both `/app/vllm-omni` and the active site-packages copy, and
emits source-hash manifests for image-build evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import site
from pathlib import Path
from typing import Iterable

EXPECTED_PATCH_CHANGED_FILES = (
    "vllm_omni/diffusion/attention/backends/registry.py",
    "vllm_omni/diffusion/attention/backends/sol_attn_h3_a6000.py",
    "vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py",
)


def extract_patch_changed_files(patch_text: str) -> list[str]:
    """Return b-side paths for every file touched by a git patch."""
    changed: list[str] = []
    for line in patch_text.splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) < 4 or not parts[3].startswith("b/"):
            raise ValueError(f"cannot parse patch file header: {line!r}")
        changed.append(parts[3][2:])
    if sorted(changed) != sorted(EXPECTED_PATCH_CHANGED_FILES):
        raise ValueError(
            "r7 patch changed-file audit mismatch: "
            f"patch={sorted(changed)!r} expected={sorted(EXPECTED_PATCH_CHANGED_FILES)!r}"
        )
    return changed


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _active_site_package_vllm_roots() -> list[Path]:
    roots: list[Path] = []
    for base in site.getsitepackages():
        candidate = Path(base) / "vllm_omni"
        candidate_text = str(candidate)
        if candidate.exists() and ("site-packages" in candidate_text or "dist-packages" in candidate_text):
            roots.append(candidate)
    # Deduplicate while preserving order.  The r2 image is expected to have a
    # single active site-packages installation, but installing into all detected
    # site package roots is safer than silently missing one.
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(root)
    if not unique:
        raise RuntimeError("could not locate active site-packages/vllm_omni installation")
    return unique


def install_changed_files(
    *,
    patched_root: Path,
    app_root: Path,
    patch_path: Path,
    manifest_path: Path,
    hash_json_path: Path,
    hash_sha256_path: Path,
) -> None:
    changed_files = extract_patch_changed_files(patch_path.read_text(encoding="utf-8"))
    site_roots = _active_site_package_vllm_roots()
    records: list[dict[str, object]] = []

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(changed_files) + "\n", encoding="utf-8")

    sha_lines: list[str] = []
    for rel in changed_files:
        source = patched_root / rel
        if not source.is_file():
            raise FileNotFoundError(f"patched source file missing after git apply: {rel}")
        data = source.read_bytes()
        digest = _sha256(data)
        installed_targets = [app_root / rel]
        installed_targets.extend(site_root.parent / rel for site_root in site_roots)
        for target in installed_targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            if target.read_bytes() != data:
                raise RuntimeError(f"post-install verification failed for {target}")
        records.append(
            {
                "path": rel,
                "sha256": digest,
                "bytes": len(data),
                "installed_targets": [str(target) for target in installed_targets],
            }
        )
        sha_lines.append(f"{digest}  {rel}")

    hash_json_path.parent.mkdir(parents=True, exist_ok=True)
    hash_json_path.write_text(
        json.dumps(
            {
                "schema_version": "minimax_h3_a6000_r7_patched_source_hashes_v1",
                "patch": str(patch_path),
                "patch_sha256": _sha256(patch_path.read_bytes()),
                "changed_files": records,
                "app_root": str(app_root),
                "site_package_roots": [str(root) for root in site_roots],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    hash_sha256_path.write_text("\n".join(sha_lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch", required=True, type=Path)
    parser.add_argument("--list-patch-files", action="store_true")
    parser.add_argument("--patched-root", type=Path)
    parser.add_argument("--app-root", type=Path, default=Path("/app/vllm-omni"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--hash-json", type=Path)
    parser.add_argument("--hash-sha256", type=Path)
    return parser.parse_args()


def _require_paths(args: argparse.Namespace, names: Iterable[str]) -> None:
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"missing required arguments for install mode: {', '.join(missing)}")


def main() -> int:
    args = _parse_args()
    changed_files = extract_patch_changed_files(args.patch.read_text(encoding="utf-8"))
    if args.list_patch_files:
        print("\n".join(changed_files))
        return 0
    _require_paths(args, ("patched_root", "manifest", "hash_json", "hash_sha256"))
    install_changed_files(
        patched_root=args.patched_root,
        app_root=args.app_root,
        patch_path=args.patch,
        manifest_path=args.manifest,
        hash_json_path=args.hash_json,
        hash_sha256_path=args.hash_sha256,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
