#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Install the H3 VAE spatial tile batching patch into both import surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import site
from pathlib import Path

RELATIVE_PATHS = (Path("vllm_omni/diffusion/models/minimax_h3/vae.py"),)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _active_site_package_vllm_roots() -> list[Path]:
    roots: list[Path] = []
    for base in site.getsitepackages():
        candidate = Path(base) / "vllm_omni"
        if candidate.exists() and ("site-packages" in str(candidate) or "dist-packages" in str(candidate)):
            roots.append(candidate)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patched-root", required=True, type=Path)
    parser.add_argument("--app-root", type=Path, default=Path("/app/vllm-omni"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--hash-json", required=True, type=Path)
    parser.add_argument("--hash-sha256", required=True, type=Path)
    parser.add_argument("--patch", required=True, type=Path)
    args = parser.parse_args()

    site_roots = _active_site_package_vllm_roots()
    manifest_lines: list[str] = []
    sha_lines: list[str] = []
    changed_files: list[dict[str, object]] = []
    for relative_path in RELATIVE_PATHS:
        source = args.patched_root / relative_path
        if not source.is_file():
            raise FileNotFoundError(f"patched VAE source file missing: {source}")
        data = source.read_bytes()
        digest = _sha256(data)
        targets = [args.app_root / relative_path]
        targets.extend(site_root.parent / relative_path for site_root in site_roots)
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            if target.read_bytes() != data:
                raise RuntimeError(f"post-install verification failed for {target}")
        manifest_lines.append(relative_path.as_posix())
        sha_lines.append(f"{digest}  {relative_path.as_posix()}")
        changed_files.append(
            {
                "path": relative_path.as_posix(),
                "sha256": digest,
                "bytes": len(data),
                "installed_targets": [str(target) for target in targets],
            }
        )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    args.hash_json.write_text(
        json.dumps(
            {
                "schema_version": "minimax_h3_a6000_r15_vae_spatial_tile_batching_patch_hashes_v1",
                "patch": str(args.patch),
                "patch_sha256": _sha256(args.patch.read_bytes()),
                "changed_files": changed_files,
                "app_root": str(args.app_root),
                "site_package_roots": [str(root) for root in site_roots],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    args.hash_sha256.write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
