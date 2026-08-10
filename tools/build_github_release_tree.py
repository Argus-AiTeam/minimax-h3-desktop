#!/usr/bin/env python3
"""Build a sanitized, manifest-driven GitHub release tree.

This is a CPU-only file copier/generator. It does not call Git, Docker, GPUs,
model loaders, package managers, network APIs, or publication endpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA_VERSION = "argus-github-release-manifest-v1"
BUILD_MANIFEST_VERSION = "argus-github-release-build-v1"
DEFAULT_MANIFEST = Path("release/github_release_manifest.json")
DEFAULT_DESTINATION_PARTS = ("data", "chenxi", "minimax-h3-desktop-github")
DEFAULT_MAX_BYTES = 1_000_000
MARKER_FILE = ".argus_release_tree"

GLOBAL_EXCLUDE_PARTS = {
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv-h3-prep",
    "__pycache__",
    "models",
    "upstreams",
}
GLOBAL_EXCLUDE_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".ckpt",
    ".gguf",
    ".onnx",
    ".pt",
    ".pth",
    ".npz",
    ".mp4",
    ".mov",
    ".avi",
    ".webm",
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".zip",
    ".7z",
    ".rar",
    ".pid",
    ".log",
)

PRIVATE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:"
    r"/(?:home|data)/[A-Za-z0-9._-]+(?:/[^\s\"'`<>)]*)?"
    r"|/Users/[A-Za-z0-9._-]+(?:/[^\s\"'`<>)]*)?"
    r"|[A-Za-z]:\\\\Users\\\\[A-Za-z0-9._-]+(?:\\\\[^\s\"'`<>)]*)?"
    r")",
)


class BuildError(Exception):
    """Raised when the export tree cannot be built safely."""


@dataclass(frozen=True)
class CopiedFile:
    path: str
    size_bytes: int
    sha256: str
    source: str


def default_destination() -> Path:
    path = Path(os.sep)
    for part in DEFAULT_DESTINATION_PARTS:
        path /= part
    return path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _safe_rel(path_text: str) -> Path:
    rel = Path(path_text)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise BuildError(f"manifest path must be a non-empty relative path without '..': {path_text!r}")
    return rel


def _is_excluded(rel: Path) -> bool:
    parts = set(rel.parts)
    if parts & GLOBAL_EXCLUDE_PARTS:
        return True
    lower = rel.as_posix().lower()
    return any(lower.endswith(suffix) for suffix in GLOBAL_EXCLUDE_SUFFIXES)


def _decode_text(data: bytes) -> str | None:
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _normalise_term(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", text)


def _term_regex(term: str) -> re.Pattern[str] | None:
    normalised = _normalise_term(term)
    if not normalised:
        return None
    pieces = [re.escape(ch) for ch in normalised]
    pattern = r"(?<![A-Za-z0-9])" + r"[^A-Za-z0-9]*".join(pieces) + r"(?![A-Za-z0-9])"
    return re.compile(pattern, re.IGNORECASE)


def load_terms(path: Path | None) -> list[str]:
    if path is None:
        return []
    terms: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        term = raw.strip()
        if not term or term.startswith("#"):
            continue
        terms.append(term)
    return terms


def _redact_terms(text: str, terms: Sequence[str]) -> str:
    out = text
    for term in terms:
        regex = _term_regex(term)
        if regex is not None:
            out = regex.sub("[redacted-term]", out)
    return out


def _redact_private_paths(text: str, *, source_root: Path) -> str:
    source_text = source_root.resolve().as_posix()
    out = text.replace(source_text, "${PWD}")
    out = PRIVATE_PATH_RE.sub(lambda match: "${PWD}" if match.group(0) == source_text else "<private-path>", out)
    return out


def _transform_bytes(data: bytes, *, source_root: Path, terms: Sequence[str]) -> bytes:
    text = _decode_text(data)
    if text is None:
        return data
    text = _redact_private_paths(text, source_root=source_root)
    text = _redact_terms(text, terms)
    return text.encode("utf-8")


def _write_file(dest_root: Path, rel: Path, data: bytes, *, source_label: str, executable: bool = False) -> CopiedFile:
    if _is_excluded(rel):
        raise BuildError(f"refusing to write globally excluded export path: {rel.as_posix()}")
    out = dest_root / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    if executable:
        current = out.stat().st_mode
        out.chmod(current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return CopiedFile(path=rel.as_posix(), size_bytes=len(data), sha256=sha256_bytes(data), source=source_label)


def _copy_one(source_root: Path, dest_root: Path, rel: Path, *, max_file_bytes: int, terms: Sequence[str]) -> CopiedFile:
    if _is_excluded(rel):
        raise BuildError(f"manifest attempts to copy excluded path: {rel.as_posix()}")
    src = source_root / rel
    if not src.is_file() or src.is_symlink():
        raise BuildError(f"manifest source is not a regular file: {rel.as_posix()}")
    size = src.stat().st_size
    if size > max_file_bytes:
        raise BuildError(f"manifest source exceeds max_file_bytes ({max_file_bytes}): {rel.as_posix()} size={size}")
    data = _transform_bytes(src.read_bytes(), source_root=source_root, terms=terms)
    executable = os.access(src, os.X_OK) or rel.suffix == ".sh"
    return _write_file(dest_root, rel, data, source_label=rel.as_posix(), executable=executable)


def _tree_files(source_root: Path, tree: dict[str, Any]) -> list[Path]:
    base = _safe_rel(str(tree["path"]))
    base_dir = source_root / base
    if not base_dir.is_dir():
        raise BuildError(f"manifest tree is not a directory: {base.as_posix()}")
    includes = tree.get("include") or ["**/*"]
    excludes = tree.get("exclude") or []
    files: set[Path] = set()
    for pattern in includes:
        for path in base_dir.glob(pattern):
            if path.is_file() and not path.is_symlink():
                rel = path.relative_to(source_root)
                files.add(rel)
    filtered: list[Path] = []
    for rel in sorted(files, key=lambda p: p.as_posix()):
        rel_under_base = rel.relative_to(base).as_posix()
        if _is_excluded(rel):
            continue
        if any(Path(rel_under_base).match(pattern) or rel.match(pattern) for pattern in excludes):
            continue
        filtered.append(rel)
    return filtered


def _readme_text() -> str:
    return """# MiniMax-H3 desktop code-only release tree

This is the sanitized, code-only export for the private `Argus-AiTeam/minimax-h3-desktop` repository. It contains source, patches, tests, schemas, small evidence JSON, locked runtime metadata, and reports. It does **not** contain model weights, adapters, merged checkpoints, generated media, Docker layers, caches, virtual environments, or vendored upstream/runtime source trees.

## Weight and license boundary

The first full model preparation is approximately **144 GB** and requires the official model license, authorization, and local credentials. This release tree never bundles weights and the default workflow is fail-closed: dry-run and publication-audit checks are safe; real download, container, deployment, demo, or GPU execution requires separate operator approval in the private environment.

## One-command A6000 workflow shape

Run from the export root:

```bash
bash scripts/a6000_one_command.sh --dry-run
```

The dry run prints the intended stages: preflight, model-prepare, deploy, demo, and verify. Non-dry model preparation/deploy/demo actions are placeholders until a later authorized private run supplies official credentials, model storage, runtime image access, and GPU approval.

## Local-only Git initialization for chenxi

Run only after the export has passed the publication audit. These commands are local repository setup only and do not create a remote, tag, release, or push anything:

```bash
git init
git config --local user.name chenxi
git config --local user.email cxx2216@163.com
python3 tools/publication_audit.py --root . --max-bytes 1000000 --prohibited-terms-file <local-file-not-committed>
git add .
git status --short
git commit -m "Initial private Argus MiniMax-H3 desktop release tree"
```

Remote publication is allowed only after the export passes the publication audit and `gh auth status` confirms the intended private account `Chenxxxxxx06`.

## Included evidence boundaries

- `technical_report/minimax_h3_a6000_performance.md` is a CPU-only evidence reader output; it does not claim deployment for pending lanes.
- `technical_report/final_technical_report.md` preserves accepted/rejected/blocked lane boundaries.
- `technical_report/evidence/minimax_h3_desktop/quantization_feasibility_a6000.md` records that there is no runnable no-new-download quantized A6000 candidate today.
- Runtime metadata under `runtime/single_a6000_bf16/` is lock metadata only; vendored runtime source is intentionally excluded.

## Attribution

See `NOTICE`, `LICENSE`, and the port-level `ports/minimax_h3_a6000/NOTICE`. NVLabs/Sol-Engine team attribution is preserved for Sol-Engine-derived design and kernel-candidate work; upstream projects retain their own licenses and authorship.
"""


def _license_text() -> str:
    return """Apache License
Version 2.0, January 2004
https://www.apache.org/licenses/

This code-only release tree is provided under Apache-2.0-compatible terms for the original Argus adaptation code unless a file says otherwise. Upstream projects, model providers, and third-party dependencies retain their own notices, licenses, and usage terms. Model weights are not included.

Full Apache-2.0 license text is available at https://www.apache.org/licenses/LICENSE-2.0 .
"""


def _notice_text() -> str:
    return """MiniMax-H3 desktop code-only release tree

This export contains original local adaptation, packaging, verification, and reporting code for a MiniMax-H3 A6000 desktop/workstation path. It does not include model weights, adapter weights, generated media, Docker layers, vendored upstream source trees, credentials, or private caches.

Attribution is preserved for upstream work that informed or is patched by this tree, including the NVLabs/Sol-Engine team, vLLM-Omni, MiniMax, Hugging Face ecosystem packages, PyTorch, Triton, and related open-source dependencies. The local A6000 overlay NOTICE and patch files retain additional file-level attribution. This NOTICE does not reassign upstream authorship to the Argus adapter authors.
"""


def _gitignore_text() -> str:
    return """# Code-only release tree guardrails
models/
upstreams/
runtime/**/src/
.venv/
.venv-*/
__pycache__/
.pytest_cache/
.cache/
*.safetensors
*.bin
*.ckpt
*.pt
*.pth
*.gguf
*.onnx
*.npz
*.mp4
*.mov
*.webm
*.wav
*.mp3
*.flac
*.tar
*.tar.gz
*.tgz
*.zip
*.log
*.pid
.env
*.env
"""


def _workflow_script_text() -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=1
STAGE="all"
TERMS_FILE=""

usage() {
  cat <<'EOF'
Usage: bash scripts/a6000_one_command.sh --dry-run [--stage preflight|model-prepare|deploy|demo|verify|all] [--prohibited-terms-file PATH]

Dry-run is the only enabled mode in this public-ready export. The first full
model preparation is about 144 GB and requires official license/auth and a
separate private authorization. This script never bundles weights.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --stage)
      STAGE="${2:?missing stage}"
      shift 2
      ;;
    --prohibited-terms-file)
      TERMS_FILE="${2:?missing terms file}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 64
      ;;
  esac
done

if [[ "$DRY_RUN" != "1" ]]; then
  echo "ERROR: non-dry model preparation, deploy, demo, and GPU execution are blocked in this export." >&2
  exit 2
fi

run_stage() {
  local name="$1"
  case "$name" in
    preflight)
      echo "[DRY-RUN] preflight: inspect local metadata and run publication audit; no GPU, Docker, model load, network, or download."
      ;;
    model-prepare)
      echo "[DRY-RUN] model-prepare: first full model download is about 144 GB and requires official license/auth; weights are never bundled."
      ;;
    deploy)
      echo "[DRY-RUN] deploy: locked runtime metadata is present; actual container/runtime deployment needs separate private approval."
      ;;
    demo)
      echo "[DRY-RUN] demo: real generation is disabled here; no media output is created."
      ;;
    verify)
      echo "[DRY-RUN] verify: running publication audit."
      if [[ -n "$TERMS_FILE" ]]; then
        python3 "$ROOT/tools/publication_audit.py" --root "$ROOT" --max-bytes 1000000 --prohibited-terms-file "$TERMS_FILE"
      else
        python3 "$ROOT/tools/publication_audit.py" --root "$ROOT" --max-bytes 1000000
      fi
      ;;
    *)
      echo "ERROR: unsupported stage: $name" >&2
      exit 64
      ;;
  esac
}

case "$STAGE" in
  all)
    for stage in preflight model-prepare deploy demo verify; do
      run_stage "$stage"
    done
    ;;
  preflight|model-prepare|deploy|demo|verify)
    run_stage "$STAGE"
    ;;
  *)
    echo "ERROR: unsupported --stage: $STAGE" >&2
    exit 64
    ;;
esac
"""


def _generated_text(kind: str) -> tuple[str, bool]:
    if kind == "readme":
        return _readme_text(), False
    if kind == "license":
        return _license_text(), False
    if kind == "notice":
        return _notice_text(), False
    if kind == "gitignore":
        return _gitignore_text(), False
    if kind == "a6000_workflow":
        return _workflow_script_text(), True
    raise BuildError(f"unknown generated artifact kind: {kind}")


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BuildError(f"manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BuildError(f"invalid manifest JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise BuildError("manifest must be a JSON object")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise BuildError(f"manifest schema_version must be {SCHEMA_VERSION}")
    return data


def _prepare_destination(dest: Path, *, force: bool) -> None:
    if dest.exists():
        if not dest.is_dir():
            raise BuildError(f"destination exists and is not a directory: {dest}")
        has_marker = (dest / MARKER_FILE).is_file()
        nonempty = any(dest.iterdir())
        if nonempty and not force:
            raise BuildError(f"destination is non-empty; rerun with --force to rebuild: {dest}")
        if nonempty and not has_marker:
            raise BuildError(f"refusing to delete unmarked destination: {dest}")
        if nonempty:
            shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)


def build_release_tree(
    *,
    source_root: Path,
    manifest_path: Path,
    dest: Path,
    force: bool = False,
    prohibited_terms_file: Path | None = None,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    manifest_path = manifest_path.resolve()
    dest = dest.resolve()
    if not source_root.is_dir():
        raise BuildError(f"source root is not a directory: {source_root}")
    try:
        dest.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise BuildError("destination must not be inside the source tree")

    manifest = load_manifest(manifest_path)
    max_file_bytes = int(manifest.get("max_file_bytes", DEFAULT_MAX_BYTES))
    terms = load_terms(prohibited_terms_file)
    _prepare_destination(dest, force=force)

    tmp_parent = dest.parent
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{dest.name}.tmp-", dir=tmp_parent))
    copied: list[CopiedFile] = []
    try:
        (tmp_dir / MARKER_FILE).write_text(BUILD_MANIFEST_VERSION + "\n", encoding="utf-8")
        copied.append(
            CopiedFile(
                path=MARKER_FILE,
                size_bytes=len((BUILD_MANIFEST_VERSION + "\n").encode("utf-8")),
                sha256=sha256_bytes((BUILD_MANIFEST_VERSION + "\n").encode("utf-8")),
                source="generated:marker",
            )
        )

        for item in manifest.get("generated", []):
            if not isinstance(item, dict):
                raise BuildError("generated entries must be objects")
            rel = _safe_rel(str(item["path"]))
            text, executable = _generated_text(str(item["kind"]))
            data = _transform_bytes(text.encode("utf-8"), source_root=source_root, terms=terms)
            copied.append(_write_file(tmp_dir, rel, data, source_label=f"generated:{item['kind']}", executable=executable))

        explicit_files: list[Path] = []
        for item in manifest.get("files", []):
            if not isinstance(item, dict):
                raise BuildError("files entries must be objects")
            explicit_files.append(_safe_rel(str(item["path"])))
        for tree in manifest.get("trees", []):
            if not isinstance(tree, dict):
                raise BuildError("trees entries must be objects")
            explicit_files.extend(_tree_files(source_root, tree))

        seen: set[str] = set()
        for rel in sorted(explicit_files, key=lambda p: p.as_posix()):
            rel_text = rel.as_posix()
            if rel_text in seen:
                continue
            seen.add(rel_text)
            copied.append(_copy_one(source_root, tmp_dir, rel, max_file_bytes=max_file_bytes, terms=terms))

        copied_sorted = sorted(copied, key=lambda item: item.path)
        build_manifest = {
            "schema_version": BUILD_MANIFEST_VERSION,
            "release_name": manifest.get("release_name"),
            "source_root_name": source_root.name,
            "manifest": _repo_rel(manifest_path, source_root),
            "file_count": len(copied_sorted),
            "files": [item.__dict__ for item in copied_sorted],
            "excluded_by_design": manifest.get("excluded_by_design", []),
            "publication_audit_command": "python3 tools/publication_audit.py --root . --max-bytes 1000000 --prohibited-terms-file <local-file-not-committed>",
        }
        build_data = json.dumps(build_manifest, indent=2, sort_keys=True).encode("utf-8")
        build_rel = Path("release/export_build_manifest.json")
        copied.append(_write_file(tmp_dir, build_rel, build_data + b"\n", source_label="generated:build_manifest"))

        if dest.exists():
            shutil.rmtree(dest)
        tmp_dir.rename(dest)
        final_files = sorted(copied, key=lambda item: item.path)
        return {
            "status": "built",
            "destination": dest.as_posix(),
            "file_count": len(final_files),
            "manifest": _repo_rel(manifest_path, source_root),
            "files": [item.__dict__ for item in final_files],
        }
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build sanitized GitHub release tree from an explicit manifest.")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1], help="canonical working tree")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="release/export manifest")
    parser.add_argument("--dest", type=Path, default=default_destination(), help="export destination")
    parser.add_argument("--force", action="store_true", help="rebuild an existing marked export directory")
    parser.add_argument(
        "--prohibited-terms-file",
        type=Path,
        default=None,
        help="local terms file used to redact operator-prohibited backend/product names during this build",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON summary")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    manifest = args.manifest
    if not manifest.is_absolute():
        manifest = args.source_root / manifest
    try:
        summary = build_release_tree(
            source_root=args.source_root,
            manifest_path=manifest,
            dest=args.dest,
            force=args.force,
            prohibited_terms_file=args.prohibited_terms_file,
        )
    except BuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"built sanitized release tree: {summary['destination']} ({summary['file_count']} files)")
        print("next audit: python3 tools/publication_audit.py --root <export-root> --max-bytes 1000000 --prohibited-terms-file <local-file-not-committed>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
