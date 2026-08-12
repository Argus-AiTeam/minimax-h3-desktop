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
    return """# MiniMax-H3 on A6000 — Argus desktop release tree

> **中文定位**：这是 Argus-AiTeam 为 **单卡 NVIDIA RTX A6000 48 GB（SM86）桌面/工作站**整理的 MiniMax-H3 代码发布树。它只包含源码、补丁、测试、Schema、报告和小型证据 JSON；不包含模型权重、适配器、生成视频、Docker 层、缓存、凭据或 vendored 上游源码。
>
> **English positioning**: this is the Argus-AiTeam maintained **code-only** MiniMax-H3 desktop/A6000 release tree. It documents a conservative BF16 fidelity baseline, practical Turbo candidates, DLO capacity evidence, and default-off Sol/attention diagnostics without bundling private runtime artifacts or weights.

MiniMax-H3 is an omni-modal diffusion pipeline for synchronized video + audio generation. This export is meant to be auditable first and runnable only after a private operator supplies licensed model access, local storage, and GPU approval.

## Quick start / 快速开始

From the export root, these commands are CPU-only and do not load models, start containers, touch GPUs, download files, publish, or call Git:

```bash
bash scripts/a6000_one_command.sh --dry-run
python3 tools/publication_audit.py --root . --max-bytes 1000000 --prohibited-terms-file <local-file-not-committed>
PYTHONPATH=code:ports/minimax_h3_a6000/src python3 -m pytest -q tests/test_github_release_tree_builder.py tests/test_minimax_h3_a6000_performance_report.py
```

The dry run prints the intended preflight/model-prepare/deploy/run/verify stages. A gated non-dry local lifecycle verifier is included for private clean-room checks: it inspects an existing local FL2VA model directory and locked runtime image metadata, runs the checked-in CPU verifier fixture, and audits the sanitized export root. It still does **not** download, start Docker containers, load weights, run GPU inference, generate media, publish, or claim speed/quality.

Current delivery-gate evidence is CPU/static and packaging-only except for already-written GPU evidence that is only read, not rerun: the checked-in reports record CPU/static tests, fixture verification, Turbo dry-run planning, strict aggregation, sanitized export build, publication audit with 0 issues, a terminal Sol-Attn r8 N=3 matched-workload route gate, and a terminal formal Sol-Attn N>=10 matched-workload acceptance. Independent Reviewer certification is still required for the latest formal-N10 sync before any private-main update. These gates do not certify BF16 fidelity for Sol-Attn, Turbo semantic quality, quality equivalence, human listening, public release, or tag creation.

## Hardware and asset requirements / 硬件与资产要求

| Requirement | Current release boundary |
|---|---|
| GPU | Single RTX A6000 48 GB, compute capability 8.6, verified internally as the target class. |
| Host memory | Baseline evidence observed up to about 205 GiB host memory; plan for a high-memory workstation. |
| Model assets | First full MiniMax-H3 preparation is about **144 GB** of licensed weights; weights are never included here. |
| Runtime | Locked runtime metadata is included under `runtime/single_a6000_bf16/`; vendored runtime source and Docker layers are intentionally excluded. |
| Network/credentials | Required only for a later private model/runtime preparation; no credentials belong in this tree. |

## Architecture and DLO map / 架构与 DLO 示意

```mermaid
flowchart LR
    U[Prompt / media request] --> T[Qwen3-VL conditioner]
    T --> P[Packed H3 sequence\ntext + conditions + audio + video]
    P --> D[MiniMax-H3 DiT\n50 denoising blocks]
    D --> V[Video VAE]
    D --> A[Audio VAE]
    V --> M[Muxed video]
    A --> M

    subgraph Stable[Stable evidence]
      B[BF16 dense baseline\n50-step fidelity denominator]
      TB[Turbo LoRA practical lane\n4/8-step timing + structural AV]
    end

    subgraph Diagnostics[Default-off diagnostics]
      DLO[DLO resident-layer candidates\ncapacity gates only]
      SOL[Sol-Attn r8 opt-in\nformal N>=10 accepted lane]
      EX[Exact Triton kernels\nkernel-only microbench]
    end

    D -. evidence denominator .-> B
    D -. practical candidate .-> TB
    D -. memory/layout study .-> DLO
    D -. attention experiment .-> SOL
    D -. op fusion candidates .-> EX
```

DLO means resident-layer placement/continuation tuning. The current DLO evidence is a capacity gate, not a formal N10 performance result.

## Evidence-backed 1344x768 status / 证据支持的 1344x768 状态

All rows are read from checked-in reports/evidence for the same 1344x768, 5.166667 s, 124-frame, 24 FPS workload unless marked as diagnostic. Do **not** compare rows across lanes as if they were one benchmark leaderboard.

| Lane / item | Evidence status | Timing | Quality / structure | Release meaning |
|---|---|---:|---|---|
| BF16 dense baseline, 50 steps | certified internal same-physical-device A6000 baseline | warm N10 median **1792.2021025 s** | structural AV pass 13/13 | Fidelity denominator only; not an optimized result. |
| Turbo LoRA, 8 steps | practical paired N10 | median **290.9976015 s**; **6.158820874x** vs BF16 warm median | structural AV pass; semantic quality and human listening pending | Current practical default candidate, not BF16-exact. |
| Turbo LoRA, 4 steps | practical paired N10 | median **149.6191865 s**; **11.978424321x** vs BF16 warm median | structural AV pass; stronger quality-risk boundary | Ultra-fast experimental option, not fidelity evidence. |
| DLO resident layers 13, 5-step capacity gate | present capacity gate | dense 188.098444 s → candidate 186.773476 s | hash match true | Capacity/placement signal only; formal DLO N10 pending. |
| Sol-Attn r8 opt-in, 5-step diagnostic + terminal N=3 route gate + formal N>=10 matched gate | sparse runtime candidate pass; N=3 route decision `proceed_to_formal_n10_candidate`; formal classification `accepted_formal_n10_same_gpu_sol_attn_speed_candidate` | 5-step diagnostic dense **186.498762 s**, opt-in **158.923988 s**; N=3 route-gate median HTTP-time improvement **14.782455716%**; formal N=10 median HTTP-time improvement **15.203295894%** over a >3% threshold | structural AV pass; formal pairs 10/10; every opt-in pair has `sparse_candidate_calls=192`, `sparse_calls=192`, `fallback_calls=0`, density/materialization telemetry, HTTP 200, and resource envelope within thresholds | Accepted only for the formal matched 5-step Sol-Attn opt-in lane. This is **not BF16 fidelity, release approval, Turbo/DLO/DMD evidence, human listening, or semantic quality certification**. |

## Stable vs experimental / 稳定与实验边界

- **Stable / 稳定**: code-only export builder, publication audit, BF16 baseline report, locked runtime metadata, gated local lifecycle verifier, dry-run workflow, and CPU/static tests.
- **Practical but not fidelity / 实用但非保真**: Turbo LoRA 8-step and 4-step timing. These outputs passed structural AV checks, but semantic/video/audio quality still needs separate review before product claims.
- **Diagnostic/formal lane separated / 诊断与正式通道分离**: exact Triton op kernels and DLO capacity gates remain diagnostic; Sol-Attn r8 sparse-execution metadata plumbing plus N=3 route gate led to a separate formal N>=10 matched-workload acceptance only inside the opt-in 5-step Sol-Attn lane. This still is not BF16 fidelity or release approval.
- **Blocked / 阻塞**: DMD/DMD2 remains research-only because there is no first-source H3 recipe/checkpoint basis in this tree.

## Quality limits / 质量限制

Structural AV validation means the file decodes with expected video/audio properties. It is not equivalent to semantic quality, prompt faithfulness, or human auditory approval. Turbo and diagnostic lanes must never be relabeled as BF16-exact fidelity evidence. The Sol-Attn r8 formal N>=10 acceptance is limited to the matched 5-step opt-in lane and must not be reported as BF16 fidelity, release readiness, human-auditory/semantic quality certification, or Turbo/DLO/DMD evidence.

## Reports and evidence links / 报告与证据链接

- `technical_report/minimax_h3_a6000_performance.md` — CPU-only evidence reader with lane boundaries and pending/blocker status.
- `technical_report/final_technical_report.md` — high-level accepted/rejected/blocked summary.
- `technical_report/evidence/minimax_h3_desktop/baseline_a6000/baseline_certification.json` — BF16 baseline denominator.
- `technical_report/evidence/minimax_h3_desktop/dlo_autotune/runs/a6000_dlo_candidate_50_rl16_20260810T141257Z/candidate50_summary.json` — DLO candidate-50 artifact.
- `technical_report/evidence/minimax_h3_desktop/delivery/argus_ir04_delivery_summary.json` — delivery aggregation.
- `technical_report/evidence/minimax_h3_desktop/sol_engine_port/sol_attn_h3_matched_retest_r8_n3_20260812T013544Z/decision.json` — terminal N=3 Sol-Attn route decision.
- `technical_report/evidence/minimax_h3_desktop/sol_engine_port/sol_attn_h3_formal_n10_r8_n10_20260812T031757Z/formal_n10_decision.json` — terminal formal N>=10 matched-workload Sol-Attn opt-in acceptance.
- `ports/minimax_h3_a6000/README.md` — default-off exact-kernel and Sol-Attn port details.

Raw videos, model weights, caches, Docker layers, and private run directories are intentionally outside the release manifest.

## Troubleshooting / 常见问题

| Symptom | Likely cause | Safe next step |
|---|---|---|
| Publication audit flags a private path or prohibited term | A copied text file contains local machine detail | Fix the source text or pass a local prohibited-terms file; rebuild the export. |
| Non-dry lifecycle refuses to run | The explicit `ARGUS_ALLOW_MINIMAX_H3_RUN=1`, authorization id, clean `--work-dir`, local model dir, or locked Docker image is missing | Fix the declared local resource or run `--dry-run`; do not download or mutate weights implicitly. |
| Sol-Attn appears faster in r8 5-step, N=3 route-gate, or formal N>=10 timing | Sparse path evidence exists, and the formal N>=10 gate is accepted only in the matched 5-step opt-in lane | Report only that bounded formal-lane result with its evidence; do not claim BF16 fidelity, release readiness, semantic/human quality, or quality equivalence. |
| Turbo output is fast but visually questionable | Structural AV is not semantic quality | Run human/semantic review before user-facing quality claims. |
| Model files are missing | Weights are excluded by design | Obtain official MiniMax-H3 access and prepare assets only in a private authorized environment. |

## External supervised commands for later private operators

These are documentation-only runbook commands. They are **not** run by the release builder and require a private workstation with approved GPU/runtime/model access:

```bash
# Rebuild the CPU-only performance report from already-written evidence.
python3 tools/minimax_h3_a6000_performance_report.py --evidence-root technical_report/evidence/minimax_h3_desktop --out technical_report/minimax_h3_a6000_performance.md

# Recheck the static release tree before any publication decision.
python3 tools/build_github_release_tree.py --source-root . --manifest release/github_release_manifest.json --dest <export-root> --force --json
python3 tools/publication_audit.py --root <export-root> --max-bytes 1000000 --prohibited-terms-file <local-file-not-committed>

# Optional private GPU2 diagnostic retry; keep Sol-Attn default-off unless the runbook explicitly opts in.
PYTHONPATH=ports/minimax_h3_a6000/src python3 ports/minimax_h3_a6000/gpu_exact_kernel_test.py --device cuda:0 --output <private-evidence-root>/h3_exact_correctness.json
PYTHONPATH=ports/minimax_h3_a6000/src python3 ports/minimax_h3_a6000/gpu_sol_attn_sm86_harness.py --device cuda:0 --mode both --output <private-evidence-root>/h3_sol_attn_sm86.json
```

## Roadmap / 路线图

1. Keep the release tree audit-clean and code-only.
2. Finish quality review for practical Turbo lanes before product language changes.
3. Promote DLO only after a formal same-device N10 timing result exists.
4. Keep the Sol-Attn formal N>=10 acceptance confined to its matched 5-step opt-in lane; the independent Reviewer has accepted this bounded formal-lane evidence, private-main sync is allowed only after a fresh zero-issue export/publication audit, and it must not become BF16 fidelity, public release, or human-quality language.
5. Revisit DMD/DMD2 only if a real H3 first-source recipe/checkpoint appears.

## Attribution / 致谢

See `NOTICE`, `LICENSE`, and `ports/minimax_h3_a6000/NOTICE`. NVLabs/Sol-Engine attribution is preserved for Sol-Engine-derived design/kernel-candidate work; MiniMax, vLLM-Omni, Hugging Face ecosystem packages, PyTorch, Triton, and other upstream projects retain their own licenses and authorship.
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
        raise BuildError("a6000_workflow generation is retired; copy scripts/a6000_one_command.sh from the manifest instead")
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
