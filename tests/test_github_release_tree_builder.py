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


def test_default_destination_is_project_sibling() -> None:
    assert default_destination() == ROOT.parent / "github-release"


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
    assert (dest / "README_EN.md").is_file()
    assert (dest / "LICENSE").is_file()
    assert (dest / "NOTICE").is_file()
    assert (dest / "Makefile").is_file()
    assert (dest / "scripts" / "a6000_one_command.sh").is_file()
    assert (dest / "scripts" / "build_runtime.sh").is_file()
    assert (dest / "scripts" / "prepare_models.sh").is_file()
    assert (dest / "scripts" / "run_turbo_demo.sh").is_file()
    assert (dest / "tools" / "turbo_lora_peft.py").is_file()
    assert (dest / "tools" / "turbo_lora_offline_merge.py").is_file()
    assert (dest / "examples" / "a6000-turbo-8step-sci-fi" / "orbital-shipyard-turbo-8step.mp4").is_file()
    assert (dest / "examples" / "a6000-turbo-8step-sci-fi" / "contact-sheet.jpg").is_file()
    assert (dest / "examples" / "a6000-turbo-8step-sci-fi" / "metadata.json").is_file()
    assert (dest / "tools" / "build_github_release_tree.py").is_file()
    assert (dest / "tools" / "publication_audit.py").is_file()
    assert (dest / "tools" / "verify_run.py").is_file()
    assert (dest / "tools" / "argus_h3_verifier.py").is_file()
    assert (dest / "release" / "github_release_manifest.json").is_file()
    assert (dest / "release" / "export_build_manifest.json").is_file()
    assert (dest / "runtime" / "single_a6000_bf16" / "source_commit.json").is_file()
    assert (dest / "schemas" / "minimax_h3_run.schema.json").is_file()
    assert (dest / "schemas" / "minimax_h3_benchmark_record_v1.schema.json").is_file()
    assert (dest / "tools" / "validate_benchmark_record.py").is_file()
    assert (dest / "benchmark_contract" / "v1" / "contract.json").is_file()
    assert (dest / "benchmark_contract" / "v1" / "README.md").is_file()
    assert len(list((dest / "benchmark_contract" / "v1" / "lane-manifests").glob("*.json"))) == 3
    assert len(list((dest / "benchmark_contract" / "v1" / "normalized-records").glob("*.json"))) == 5
    assert len(list((dest / "tests" / "fixtures" / "benchmark_contract" / "rejected").glob("*.json"))) == 6
    assert (dest / "tests" / "fixtures" / "minimal_av_case" / "run_record.json").is_file()
    readme = (dest / "README.md").read_text(encoding="utf-8")
    assert "MiniMax-H3 on a Single RTX A6000" in readme
    assert "CURRENT_WORK.md" in readme
    assert "自主" + "完成" not in readme
    assert "orbital-shipyard-turbo-8step.mp4" in readme
    assert "1792.202" in readme
    assert "290.998" in readme
    assert "6.159×" in readme
    assert "11.978×" in readme
    assert "sparse_candidate_calls=192" in readme
    assert "sparse_calls=192" in readme
    assert "15.203%" in readme
    assert "4.326%" in readme
    assert "不是 50-step BF16 fidelity speedup" in readme
    assert "git init" not in readme
    assert "gh auth" not in readme
    assert (dest / "ports" / "minimax_h3_a6000" / "src" / "minimax_h3_a6000" / "exact_kernels.py").is_file()
    assert (dest / "ports" / "minimax_h3_a6000" / "integration" / "r6" / "Dockerfile").is_file()
    assert (dest / "ports" / "minimax_h3_a6000" / "integration" / "r7" / "Dockerfile").is_file()
    assert (dest / "ports" / "minimax_h3_a6000" / "integration" / "r8" / "Dockerfile").is_file()
    assert not (dest / "models").exists()
    assert not (dest / "upstreams").exists()
    assert not (dest / "runtime" / "single_a6000_bf16" / "src").exists()

    build_manifest = json.loads((dest / "release" / "export_build_manifest.json").read_text(encoding="utf-8"))
    assert build_manifest["schema_version"] == "argus-github-release-build-v1"
    assert any(item["path"] == "technical_report/minimax_h3_a6000_performance.md" for item in build_manifest["files"])
    assert any(item["path"].endswith("quantization_feasibility_a6000.md") for item in build_manifest["files"])
    assert any(item["path"].endswith("sol_attn_h3_matched_retest_r8_n3_20260812T013544Z/decision.json") for item in build_manifest["files"])
    assert any(item["path"].endswith("sol_attn_h3_formal_n10_r8_n10_20260812T031757Z/formal_n10_decision.json") for item in build_manifest["files"])
    assert any(item["path"].endswith("sol_attn_h3_formal_n10_r8_n10_20260812T031757Z/formal_n10_summary.json") for item in build_manifest["files"])
    assert any(item["path"].endswith("final-av-30s-r10-step3-guarded-adaptive-sol-attn-formal-n10-20260816T052452Z/formal_n10_summary.json") for item in build_manifest["files"])
    assert not any("sol_attn_h3_gpu2_5step_r6_" in item["path"] for item in build_manifest["files"])
    script = dest / "scripts" / "a6000_one_command.sh"
    assert script.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash")
    assert not (dest / "technical_report" / "evidence" / "minimax_h3_desktop" / "sol_engine_port" / "sol_attn_gpu2_supervisor").exists()

    issues = audit_tree(dest, max_bytes=15_000_000, prohibited_terms=[])
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
