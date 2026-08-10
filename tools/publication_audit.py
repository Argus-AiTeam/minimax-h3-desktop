#!/usr/bin/env python3
"""Conservative publication audit for sanitized code-only release trees.

The scanner is intentionally local and CPU-only. It reads files under an export
root and reports packaging hazards: oversized files, model-weight extensions,
media/archive artifacts, private host paths, likely credentials, and optional
operator-prohibited terms supplied from a local file that is not part of the
release tree.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_MAX_BYTES = 1_000_000

FORBIDDEN_MODEL_EXTENSIONS = {
    ".safetensors",
    ".bin",
    ".ckpt",
    ".gguf",
    ".onnx",
    ".pt",
    ".pth",
    ".npz",
}
FORBIDDEN_MEDIA_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".wav", ".mp3", ".m4a", ".flac"}
FORBIDDEN_ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".zip", ".7z", ".rar", ".whl")
FORBIDDEN_PATH_PARTS = {
    ".cache",
    ".pytest_cache",
    ".venv",
    ".venv-h3-prep",
    "__pycache__",
    "models",
    "upstreams",
}
SKIP_DIR_PARTS = {".git"}

PRIVATE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:"
    r"/(?:home|data)/[A-Za-z0-9._-]+(?:/[^\s\"'`<>)]*)?"
    r"|/Users/[A-Za-z0-9._-]+(?:/[^\s\"'`<>)]*)?"
    r"|[A-Za-z]:\\\\Users\\\\[A-Za-z0-9._-]+(?:\\\\[^\s\"'`<>)]*)?"
    r")",
)
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "secret_assignment",
        re.compile(
            r"(?i)\b(?:aws_access_key_id|aws_secret_access_key|hf_token|huggingface_token|"
            r"openai_api_key|anthropic_api_key|github_token|gh_token|api_key|secret_key|"
            r"client_secret|password|passwd|credentials?)\b\s*[:=]\s*['\"]?"
            r"[A-Za-z0-9_./+=:@-]{12,}"
        ),
    ),
    ("private_key", re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b")),
    ("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("api_secret_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
)


@dataclass(frozen=True)
class AuditIssue:
    kind: str
    path: str
    message: str
    detail: str | int | None = None


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _has_forbidden_suffix(path: Path, suffixes: Iterable[str]) -> str | None:
    lower = path.name.lower()
    for suffix in suffixes:
        if lower.endswith(suffix):
            return suffix
    return None


def _normalise_term(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


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


def _term_hits(text: str, terms: Sequence[str]) -> list[str]:
    if not terms:
        return []
    lower = text.lower()
    normalised = _normalise_term(text)
    hits: list[str] = []
    for term in terms:
        term_lower = term.lower()
        term_norm = _normalise_term(term)
        if term_lower and term_lower in lower:
            hits.append(term)
        elif term_norm and term_norm in normalised:
            hits.append(term)
    return hits


def _iter_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*")):
        rel_parts = path.relative_to(root).parts
        if any(part in SKIP_DIR_PARTS for part in rel_parts):
            continue
        if path.is_file() or path.is_symlink():
            yield path


def _decode_text(data: bytes) -> str | None:
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="ignore")


def audit_tree(root: Path, *, max_bytes: int = DEFAULT_MAX_BYTES, prohibited_terms: Sequence[str] = ()) -> list[AuditIssue]:
    root = root.resolve()
    issues: list[AuditIssue] = []
    if not root.exists():
        return [AuditIssue("missing_root", root.as_posix(), "audit root does not exist")]

    for path in _iter_files(root):
        rel = _rel(path, root)
        parts = set(path.relative_to(root).parts)

        if path.is_symlink():
            issues.append(AuditIssue("symlink", rel, "symlinks are not allowed in the sanitized export"))
            continue

        forbidden_parts = sorted(parts & FORBIDDEN_PATH_PARTS)
        if forbidden_parts:
            issues.append(
                AuditIssue(
                    "forbidden_path",
                    rel,
                    "path is inside a forbidden cache/model/vendor directory",
                    ",".join(forbidden_parts),
                )
            )

        model_ext = _has_forbidden_suffix(path, FORBIDDEN_MODEL_EXTENSIONS)
        if model_ext:
            issues.append(AuditIssue("forbidden_model_extension", rel, "model/checkpoint-like file extension is forbidden", model_ext))

        media_ext = _has_forbidden_suffix(path, FORBIDDEN_MEDIA_EXTENSIONS)
        if media_ext:
            issues.append(AuditIssue("forbidden_media_extension", rel, "generated media/audio outputs are forbidden", media_ext))

        archive_ext = _has_forbidden_suffix(path, FORBIDDEN_ARCHIVE_SUFFIXES)
        if archive_ext:
            issues.append(AuditIssue("forbidden_archive_extension", rel, "archives or Docker-layer-like bundles are forbidden", archive_ext))

        try:
            size = path.stat().st_size
        except OSError as exc:
            issues.append(AuditIssue("stat_error", rel, "could not stat file", str(exc)))
            continue
        if size > max_bytes:
            issues.append(AuditIssue("oversized_file", rel, "file exceeds conservative publication limit", size))
            continue

        try:
            data = path.read_bytes()
        except OSError as exc:
            issues.append(AuditIssue("read_error", rel, "could not read file", str(exc)))
            continue

        text = _decode_text(data)
        path_hits = _term_hits(rel, prohibited_terms)
        if path_hits:
            issues.append(AuditIssue("prohibited_term", rel, "operator-prohibited term appears in path", ",".join(path_hits)))
        if text is None:
            continue

        for match in PRIVATE_PATH_RE.finditer(text):
            issues.append(AuditIssue("private_absolute_path", rel, "private host/user absolute path found", match.group(0)))
            break

        for name, pattern in SECRET_PATTERNS:
            if pattern.search(text):
                issues.append(AuditIssue("secret_or_credential", rel, "possible secret or credential material found", name))
                break

        content_hits = _term_hits(text, prohibited_terms)
        if content_hits:
            issues.append(AuditIssue("prohibited_term", rel, "operator-prohibited term appears in file content", ",".join(content_hits)))

    return issues


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a sanitized code-only release export before publication.")
    parser.add_argument("--root", type=Path, default=Path("."), help="export root to scan")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES, help="maximum allowed file size in bytes")
    parser.add_argument(
        "--prohibited-terms-file",
        type=Path,
        default=None,
        help="local, uncommitted newline-delimited terms file to scan case-insensitively",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    terms = load_terms(args.prohibited_terms_file)
    issues = audit_tree(args.root, max_bytes=args.max_bytes, prohibited_terms=terms)
    payload = {
        "status": "pass" if not issues else "fail",
        "root": args.root.as_posix(),
        "max_bytes": args.max_bytes,
        "issue_count": len(issues),
        "issues": [asdict(issue) for issue in issues],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        if not issues:
            print(f"publication audit passed: {args.root}")
        else:
            print(f"publication audit failed: {len(issues)} issue(s)", file=sys.stderr)
            for issue in issues:
                detail = f" ({issue.detail})" if issue.detail is not None else ""
                print(f"- {issue.kind}: {issue.path}: {issue.message}{detail}", file=sys.stderr)
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
