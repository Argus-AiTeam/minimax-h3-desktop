from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from publication_audit import audit_tree, load_terms  # noqa: E402


def _kinds(issues):
    return {issue.kind for issue in issues}


def test_publication_audit_passes_minimal_safe_tree(tmp_path: Path) -> None:
    root = tmp_path / "export"
    root.mkdir()
    (root / "README.md").write_text("safe code-only export\n", encoding="utf-8")

    issues = audit_tree(root, max_bytes=1024, prohibited_terms=[])

    assert issues == []


def test_publication_audit_allows_small_curated_media_under_examples(tmp_path: Path) -> None:
    root = tmp_path / "export"
    media = root / "examples" / "demo" / "result.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"curated-demo-media")

    issues = audit_tree(root, max_bytes=1024, prohibited_terms=[])

    assert issues == []


def test_publication_audit_rejects_media_outside_examples(tmp_path: Path) -> None:
    root = tmp_path / "export"
    root.mkdir()
    (root / "result.mp4").write_bytes(b"unreviewed-media")

    issues = audit_tree(root, max_bytes=1024, prohibited_terms=[])

    assert "forbidden_media_extension" in _kinds(issues)


def test_publication_audit_detects_forbidden_model_extension(tmp_path: Path) -> None:
    root = tmp_path / "export"
    root.mkdir()
    (root / "bad.safetensors").write_text("not a real weight\n", encoding="utf-8")

    issues = audit_tree(root, max_bytes=1024, prohibited_terms=[])

    assert "forbidden_model_extension" in _kinds(issues)


def test_publication_audit_detects_oversized_file(tmp_path: Path) -> None:
    root = tmp_path / "export"
    root.mkdir()
    (root / "large.txt").write_text("x" * 11, encoding="utf-8")

    issues = audit_tree(root, max_bytes=10, prohibited_terms=[])

    assert "oversized_file" in _kinds(issues)


def test_publication_audit_detects_private_absolute_path(tmp_path: Path) -> None:
    root = tmp_path / "export"
    root.mkdir()
    private_path = "/" + "data" + "/" + "chenxi" + "/" + "private"
    (root / "note.md").write_text(f"do not publish {private_path}\n", encoding="utf-8")

    issues = audit_tree(root, max_bytes=1024, prohibited_terms=[])

    assert "private_absolute_path" in _kinds(issues)


def test_publication_audit_detects_prohibited_terms_from_external_file(tmp_path: Path) -> None:
    root = tmp_path / "export"
    root.mkdir()
    (root / "note.md").write_text("This mentions forbiddenbackend in lowercase.\n", encoding="utf-8")
    terms = tmp_path / "local_terms.txt"
    terms.write_text("ForbiddenBackend\n", encoding="utf-8")

    issues = audit_tree(root, max_bytes=1024, prohibited_terms=load_terms(terms))

    assert "prohibited_term" in _kinds(issues)


def test_publication_audit_detects_automated_generation_attribution(tmp_path: Path) -> None:
    root = tmp_path / "export"
    root.mkdir()
    phrase = "Work completed auto" + "nomously by an agent.\n"
    (root / "README.md").write_text(phrase, encoding="utf-8")

    issues = audit_tree(root, max_bytes=1024, prohibited_terms=[])

    assert "automated_generation_attribution" in _kinds(issues)


def test_publication_audit_scans_binary_media_metadata(tmp_path: Path) -> None:
    root = tmp_path / "export"
    media = root / "examples" / "demo" / "result.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"\x00\x01metadata: AI-gen" + b"erated\x00")

    issues = audit_tree(root, max_bytes=1024, prohibited_terms=[])

    assert "automated_generation_attribution" in _kinds(issues)


def test_publication_audit_cli_json_returns_nonzero_on_failure(tmp_path: Path) -> None:
    root = tmp_path / "export"
    root.mkdir()
    (root / "large.txt").write_text("x" * 11, encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "publication_audit.py"),
            "--root",
            str(root),
            "--max-bytes",
            "10",
            "--json",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert payload["status"] == "fail"
    assert payload["issues"][0]["kind"] == "oversized_file"
