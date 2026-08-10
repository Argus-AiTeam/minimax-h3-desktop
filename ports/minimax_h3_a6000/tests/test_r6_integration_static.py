# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PORT = ROOT / "ports" / "minimax_h3_a6000"
R6 = PORT / "integration" / "r6"
PATCH = PORT / "patches" / "vllm_omni_h3_a6000_opt_in.patch"
EXPECTED_PATCH_CHANGED_FILES = [
    "vllm_omni/diffusion/attention/backends/registry.py",
    "vllm_omni/diffusion/attention/backends/sol_attn_h3_a6000.py",
    "vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py",
]


def _patch_changed_files() -> list[str]:
    changed: list[str] = []
    for line in PATCH.read_text(encoding="utf-8").splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            changed.append(parts[3][2:])
    return changed


def test_r6_patch_file_audit_helper_matches_current_patch() -> None:
    helper = R6 / "dual_install_patch_files.py"
    proc = subprocess.run(
        [sys.executable, str(helper), "--patch", str(PATCH), "--list-patch-files"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout.splitlines() == EXPECTED_PATCH_CHANGED_FILES
    assert _patch_changed_files() == EXPECTED_PATCH_CHANGED_FILES


def test_r6_dockerfile_builds_from_locked_r2_and_preserves_attribution() -> None:
    dockerfile = (R6 / "Dockerfile").read_text(encoding="utf-8")
    assert "argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r2" in dockerfile
    assert "FROM ${BASE_IMAGE}" in dockerfile
    assert "MiniMax-H3 A6000 r6 Sol-Attn integration overlay" in dockerfile
    assert "org.opencontainers.image.version=\"r6\"" in dockerfile
    assert "NVLabs/Sol-Engine" in dockerfile
    assert "COPY ports/minimax_h3_a6000 /opt/minimax_h3_a6000" in dockerfile
    assert "python3 -m pip install" in dockerfile
    assert "--no-deps" in dockerfile and "--no-build-isolation" in dockerfile
    assert "git -C \"$tmp\" apply \"$patch\"" in dockerfile
    assert "runtime/single_a6000_bf16/src/vllm-omni" not in dockerfile
    assert "MINIMAX_H3_A6000_ENABLE_OVERLAY=0" in dockerfile
    assert "MINIMAX_H3_A6000_ENABLE_SOL_ATTN=0" in dockerfile
    assert "MINIMAX_H3_A6000_SOL_ATTN_CACHE=0" in dockerfile
    assert "--gpus" not in dockerfile
    assert "docker run" not in dockerfile


def test_r6_dual_install_helper_installs_every_patch_file_to_both_surfaces() -> None:
    dockerfile = (R6 / "Dockerfile").read_text(encoding="utf-8")
    helper = (R6 / "dual_install_patch_files.py").read_text(encoding="utf-8")
    for rel in EXPECTED_PATCH_CHANGED_FILES:
        assert rel in dockerfile
        assert rel in helper
    assert "--list-patch-files > /opt/minimax_h3_a6000/r6_patch_changed_files.txt" in dockerfile
    assert "--app-root /app/vllm-omni" in dockerfile
    assert "--hash-json /opt/minimax_h3_a6000/r6_patched_source_hashes.json" in dockerfile
    assert "--hash-sha256 /opt/minimax_h3_a6000/r6_patched_source_hashes.sha256" in dockerfile
    assert "changed_files = extract_patch_changed_files" in helper
    assert "installed_targets = [app_root / rel]" in helper
    assert "installed_targets.extend(site_root.parent / rel for site_root in site_roots)" in helper
    assert "target.write_bytes(data)" in helper
    assert "post-install verification failed" in helper
    assert "site.getsitepackages()" in helper
    assert "minimax_h3_a6000_r6_patched_source_hashes_v1" in helper


def test_r6_build_script_is_build_only_and_emits_required_evidence() -> None:
    script = (R6 / "build_r6_overlay_image.sh").read_text(encoding="utf-8")
    assert "BASE_IMAGE=${BASE_IMAGE:-argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r2}" in script
    assert "r6-sol-attn-overlay" in script
    assert "docker build --pull=false --network=none" in script
    assert "--build-arg \"BASE_IMAGE=$BASE_IMAGE\"" in script
    assert "-f ports/minimax_h3_a6000/integration/r6/Dockerfile" in script
    assert "--iidfile \"$EVIDENCE_DIR/r6_image_iid.txt\"" in script
    assert "docker image inspect \"$TAG\" > \"$EVIDENCE_DIR/r6_image_inspect.json\"" in script
    assert "r6_patch_changed_files.txt" in script
    assert "r6_source_hashes.sha256" in script
    assert "r6_source_hashes.json" in script
    assert "r6_build_params.env" in script
    assert "r6_image_identity_summary.txt" in script
    assert "ports/minimax_h3_a6000/NOTICE" in script
    assert "ports/minimax_h3_a6000/UPSTREAM.md" in script
    assert "docker run" not in script
    assert "--gpus" not in script


def test_sol_attn_gpu2_diagnostic_requires_fresh_r6_identity_not_stale_default() -> None:
    script = (PORT / "integration" / "run_gpu2_sol_attn_h3_5step_diagnostic.sh").read_text(encoding="utf-8")
    assert "argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r6-sol-attn-overlay" in script
    assert "r3" not in script
    assert "R6_IID_FILE=${R6_IID_FILE:-$ROOT/technical_report/evidence/minimax_h3_desktop/sol_engine_port/r6_overlay_image/r6_image_iid.txt}" in script
    assert "EXPECTED_R6_IMAGE_IID=${EXPECTED_R6_IMAGE_IID:-}" in script
    assert "REQUIRED_IMAGE_VERSION_LABEL=${REQUIRED_IMAGE_VERSION_LABEL:-r6}" in script
    assert "load_expected_r6_iid" in script
    assert "verify_r6_image_identity" in script
    assert "docker image inspect --format '{{.Id}}' \"$IMAGE\"" in script
    assert "org.opencontainers.image.version" in script
    assert "org.opencontainers.image.base.name" in script
    assert "r6 Sol-Attn" in script
    assert "verify_r6_image_identity > \"$OUT_DIR/r6_image_identity.env\"" in script
    assert script.index("verify_r6_image_identity > \"$OUT_DIR/r6_image_identity.env\"") < script.index("actual_uuid=$(nvidia-smi")
    assert "external_r6_build_command=EVIDENCE_DIR=technical_report/evidence/minimax_h3_desktop/sol_engine_port/r6_overlay_image bash ports/minimax_h3_a6000/integration/r6/build_r6_overlay_image.sh" in script
    assert "DRY_RUN=1" in script
    assert "ARGUS_ALLOW_GPU2_SOL_ATTN_H3_5STEP=1" in script
    assert "diagnostic_5_step_sol_attn_metadata_gate_not_fidelity_or_performance_claim" in script
