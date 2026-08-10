from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_github_release_tree import (  # noqa: E402
    BuildError,
    build_release_tree,
    default_destination,
)
from publication_audit import audit_tree  # noqa: E402


def test_default_destination_is_operator_export_sibling_without_literal_in_test() -> None:
    expected = Path("/") / "data" / "chenxi" / "minimax-h3-desktop-github"

    assert default_destination() == expected


def test_real_manifest_builds_sanitized_export_to_tmp_path(tmp_path: Path) -> None:
    dest = tmp_path / "minimax-h3-desktop-github"
    summary = build_release_tree(
        source_root=ROOT,
        manifest_path=ROOT / "release" / "github_release_manifest.json",
        dest=dest,
        force=False,
    )

    assert summary["status"] == "built"
    assert (dest / "README.md").is_file()
    assert (dest / "LICENSE").is_file()
    assert (dest / "NOTICE").is_file()
    assert (dest / "scripts" / "a6000_one_command.sh").is_file()
    assert (dest / "tools" / "build_github_release_tree.py").is_file()
    assert (dest / "tools" / "publication_audit.py").is_file()
    assert (dest / "release" / "github_release_manifest.json").is_file()
    assert (dest / "release" / "export_build_manifest.json").is_file()
    assert (dest / "runtime" / "single_a6000_bf16" / "source_commit.json").is_file()
    assert (dest / "ports" / "minimax_h3_a6000" / "src" / "minimax_h3_a6000" / "exact_kernels.py").is_file()
    assert (dest / "ports" / "minimax_h3_a6000" / "integration" / "r6" / "Dockerfile").is_file()
    assert not (dest / "models").exists()
    assert not (dest / "upstreams").exists()
    assert not (dest / "runtime" / "single_a6000_bf16" / "src").exists()

    build_manifest = json.loads((dest / "release" / "export_build_manifest.json").read_text(encoding="utf-8"))
    assert build_manifest["schema_version"] == "argus-github-release-build-v1"
    assert any(item["path"] == "technical_report/minimax_h3_a6000_performance.md" for item in build_manifest["files"])
    assert any(item["path"].endswith("quantization_feasibility_a6000.md") for item in build_manifest["files"])

    issues = audit_tree(dest, max_bytes=1_000_000, prohibited_terms=[])
    assert issues == []


def test_builder_refuses_forbidden_weight_extension_in_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "bad.bin").write_text("weight-like artifact\n", encoding="utf-8")
    manifest = source / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "argus-github-release-manifest-v1",
                "release_name": "test",
                "files": [{"path": "bad.bin"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(BuildError, match="excluded path"):
        build_release_tree(source_root=source, manifest_path=manifest, dest=tmp_path / "out")


def test_builder_cli_creates_export(tmp_path: Path) -> None:
    dest = tmp_path / "export"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "build_github_release_tree.py"),
            "--source-root",
            str(ROOT),
            "--manifest",
            str(ROOT / "release" / "github_release_manifest.json"),
            "--dest",
            str(dest),
            "--json",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["status"] == "built"
    assert (dest / "README.md").is_file()
