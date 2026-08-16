# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PORT = ROOT / "ports" / "minimax_h3_a6000"
R6 = PORT / "integration" / "r6"
R7 = PORT / "integration" / "r7"
R8 = PORT / "integration" / "r8"
R9 = PORT / "integration" / "r9"
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
    for helper in (
        R6 / "dual_install_patch_files.py",
        R7 / "dual_install_patch_files.py",
        R8 / "dual_install_patch_files.py",
        R9 / "dual_install_patch_files.py",
    ):
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
    assert "MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_MATERIALIZE=0" in dockerfile
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


def test_r7_build_script_is_build_only_and_emits_required_evidence() -> None:
    script = (R7 / "build_r7_overlay_image.sh").read_text(encoding="utf-8")
    dockerfile = (R7 / "Dockerfile").read_text(encoding="utf-8")
    assert "BASE_IMAGE=${BASE_IMAGE:-argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r2}" in script
    assert "r7-sol-attn-overlay" in script
    assert "docker build --pull=false --network=none" in script
    assert "--iidfile \"$EVIDENCE_DIR/r7_image_iid.txt\"" in script
    assert "docker image inspect \"$TAG\" > \"$EVIDENCE_DIR/r7_image_inspect.json\"" in script
    assert "r7_source_hashes.sha256" in script
    assert "minimax_h3_a6000_r7_source_hashes_v1" in script
    assert "org.opencontainers.image.version=\"r7\"" in dockerfile
    assert "MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_MATERIALIZE=0" in dockerfile
    assert "docker run" not in script and "--gpus" not in script


def test_r8_build_script_is_build_only_and_emits_required_evidence() -> None:
    script = (R8 / "build_r8_overlay_image.sh").read_text(encoding="utf-8")
    dockerfile = (R8 / "Dockerfile").read_text(encoding="utf-8")
    helper = (R8 / "dual_install_patch_files.py").read_text(encoding="utf-8")
    assert "BASE_IMAGE=${BASE_IMAGE:-argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r2}" in script
    assert "r8-sol-attn-overlay" in script
    assert "--dry-run" in script and "unknown argument" in script
    assert "docker image inspect \"$BASE_IMAGE\" > \"$EVIDENCE_DIR/r8_base_image_inspect.json\"" in script
    assert "r8_base_image_blocker.json" in script
    assert "r8_image_identity_summary.txt" in script
    assert "pinned_base_image_not_inspectable_locally" in script
    assert script.index("docker image inspect \"$BASE_IMAGE\"") < script.index("docker build --pull=false --network=none")
    assert "docker build --pull=false --network=none" in script
    assert "--iidfile \"$EVIDENCE_DIR/r8_image_iid.txt\"" in script
    assert "docker image inspect \"$TAG\" > \"$EVIDENCE_DIR/r8_image_inspect.json\"" in script
    assert "r8_source_hashes.sha256" in script
    assert "minimax_h3_a6000_r8_source_hashes_v1" in script
    assert "org.opencontainers.image.version=\"r8\"" in dockerfile
    assert "MiniMax-H3 A6000 r8 Sol-Attn integration overlay" in dockerfile
    assert "MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_MATERIALIZE=0" in dockerfile
    assert "minimax_h3_a6000_r8_patched_source_hashes_v1" in helper
    assert "docker run" not in script and "--gpus" not in script


def test_r9_build_script_is_build_only_default_off_and_stride_aware() -> None:
    script = (R9 / "build_r9_overlay_image.sh").read_text(encoding="utf-8")
    dockerfile = (R9 / "Dockerfile").read_text(encoding="utf-8")
    helper = (R9 / "dual_install_patch_files.py").read_text(encoding="utf-8")
    assert "BASE_IMAGE=${BASE_IMAGE:-argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r2}" in script
    assert "r9-sol-attn-overlay" in script
    assert "--dry-run" in script and "unknown argument" in script
    assert "docker image inspect \"$BASE_IMAGE\" > \"$EVIDENCE_DIR/r9_base_image_inspect.json\"" in script
    assert "r9_base_image_blocker.json" in script
    assert "r9_image_identity_summary.txt" in script
    assert "pinned_base_image_not_inspectable_locally" in script
    assert script.index("docker image inspect \"$BASE_IMAGE\"") < script.index("docker build --pull=false --network=none")
    assert "docker build --pull=false --network=none" in script
    assert "--iidfile \"$EVIDENCE_DIR/r9_image_iid.txt\"" in script
    assert "r9_source_hashes.sha256" in script
    assert "minimax_h3_a6000_r9_source_hashes_v1" in script
    assert "org.opencontainers.image.version=\"r9\"" in dockerfile
    assert "MiniMax-H3 A6000 r9 Sol-Attn integration overlay" in dockerfile
    assert "MINIMAX_H3_A6000_SOL_ATTN_STRIDE_AWARE_V=0" in dockerfile
    assert "MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_OUTPUT_DIGEST=0" in dockerfile
    assert "MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_MATERIALIZE=0" in dockerfile
    assert "minimax_h3_a6000_r9_patched_source_hashes_v1" in helper
    assert "docker run" not in script and "--gpus" not in script


def test_stride_aware_v_n1_gate_is_matched_warm_copy_timed_and_default_dry() -> None:
    script = (PORT / "integration" / "run_sol_attn_h3_stride_aware_v_n1.sh").read_text(encoding="utf-8")
    assert "DRY_RUN=1" in script
    assert "ARGUS_ALLOW_A6000_SOL_ATTN_STRIDE_AWARE_V_N1=1" in script
    assert "argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r9-sol-attn-overlay" in script
    assert "REQUIRED_IMAGE_VERSION_LABEL=${REQUIRED_IMAGE_VERSION_LABEL:-r9}" in script
    assert "run_one materialized_reference" in script
    assert "run_one current_retained" in script
    assert "request warmup" in script and "request output" in script
    assert "SOL_ATTN_TELEMETRY_ARM_FILE=/evidence/materialized_reference/measure.arm" in script
    assert "SOL_ATTN_TELEMETRY_ARM_FILE=/evidence/current_retained/measure.arm" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_STRIDE_AWARE_V=0" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_STRIDE_AWARE_V=1" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_MATERIALIZE=1" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_MATERIALIZE=0" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_SKIP_FULL_PREFIX_BLOCKS=1" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_PAIR_VALUE_HALVES=0" in script
    assert "materialize_gpu_copy_latency_ms" in script
    assert "sparse_attention_gpu_latency_ms" in script
    assert "denoise_gpu_latency_ms" in script
    assert "observed_r8_cv_pct = 0.5072177175606011" in script
    assert "promotion_threshold_pct = max(1.5, 2.0 * observed_r8_cv_pct)" in script
    assert "both_sparse_calls_192" in script and "both_fallback_calls_zero" in script
    assert "materialized_lane_actually_materialized_v" in script
    assert "retained_zero_materialization_and_input_copies" in script
    assert "real_h3_fused_value_layout_seen_in_retained" in script
    assert "hash_equality_used_for_decision': False" in script
    assert "--init-timeout '$VLLM_OMNI_INIT_TIMEOUT_S'" in script
    assert "--stage-init-timeout '$VLLM_OMNI_STAGE_INIT_TIMEOUT_S'" in script
    assert "VLLM_OMNI_INIT_TIMEOUT_S=${VLLM_OMNI_INIT_TIMEOUT_S:-2400}" in script
    assert "VLLM_OMNI_STAGE_INIT_TIMEOUT_S=${VLLM_OMNI_STAGE_INIT_TIMEOUT_S:-1800}" in script
    assert "SERVER_READY_TIMEOUT_S=${SERVER_READY_TIMEOUT_S:-$((VLLM_OMNI_INIT_TIMEOUT_S + 600))}" in script
    assert "startup_timeout_config.env" in script
    assert "write_runtime_failure_decision" in script
    assert "no_above_noise_n1_signal" in script
    assert "promote_to_matched_n3" in script
    assert "duration=5.166667" in script and "duration=30" not in script and "duration=60" not in script


def test_prefix_skip_n1_gate_is_matched_default_dry_only_principal_variable() -> None:
    script = (PORT / "integration" / "run_sol_attn_h3_prefix_skip_n1.sh").read_text(encoding="utf-8")
    finalizer = (PORT / "integration" / "finalize_sol_attn_prefix_skip_diagnostic.py").read_text(encoding="utf-8")
    assert "DRY_RUN=1" in script
    assert "ARGUS_ALLOW_A6000_SOL_ATTN_PREFIX_SKIP_N1=1" in script
    assert "argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r9-sol-attn-overlay" in script
    assert "run_one skip_off_a" in script
    assert "run_one skip_off_b" in script
    assert "run_one skip_on" in script
    assert "request warmup" in script and "request output" in script
    assert "SOL_ATTN_TELEMETRY_ARM_FILE=/evidence/skip_off_a/measure.arm" in script
    assert "SOL_ATTN_TELEMETRY_ARM_FILE=/evidence/skip_off_b/measure.arm" in script
    assert "SOL_ATTN_TELEMETRY_ARM_FILE=/evidence/skip_on/measure.arm" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_SKIP_FULL_PREFIX_BLOCKS=0" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_SKIP_FULL_PREFIX_BLOCKS=1" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_STRIDE_AWARE_V=1" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_MATERIALIZE=0" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_OUTPUT_DIGEST=1" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_PAIR_VALUE_HALVES=0" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_EXACT_PREFIX_QUERY=0" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_STATIC_PREFIX_SINK=0" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_BITMASK_SCHEDULER=0" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_MATERIALIZE=1" not in script
    assert "principal_variable=MINIMAX_H3_A6000_SOL_ATTN_SKIP_FULL_PREFIX_BLOCKS_0_vs_1" in script
    assert "current_current_attention_output_digests_equal" in finalizer
    assert "candidate_attention_output_digests_equal" in finalizer
    assert "skip_off_modes_have_no_full_prefix_skip_marker" in finalizer
    assert "skip_on_mode_has_full_prefix_skip_marker" in finalizer
    assert "skip_on_skipped_prefix_blocks_positive" in finalizer
    assert "all_prefix_dense_overwrite_calls_192" in finalizer
    assert "all_zero_materialization_and_input_copies" in finalizer
    assert "sparse_attention_gpu_latency_ms" in finalizer and "denoise_gpu_latency_ms" in finalizer
    assert "peak_gpu_power_w" in finalizer and "host_mem_available_kib" in finalizer
    assert "ignored_policy_keys={\"skip_full_prefix_blocks\"}" in finalizer
    assert "decoded_av_comparison_current_vs_current.json" in finalizer
    assert "decoded_av_comparison_skip_off_vs_skip_on.json" in finalizer
    assert '"hash_equality_used_for_decision": False' in finalizer
    assert "no_above_noise_n1_signal" in finalizer
    assert "promote_to_matched_n3" in finalizer
    assert "duration=5.166667" in script and "duration=30" not in script and "duration=60" not in script


def test_pair_value_halves_n1_gate_is_matched_default_off_no_materialization() -> None:
    script = (PORT / "integration" / "run_sol_attn_h3_pair_value_halves_n1.sh").read_text(encoding="utf-8")
    finalizer = (PORT / "integration" / "finalize_sol_attn_pair_value_halves_diagnostic.py").read_text(encoding="utf-8")
    assert "DRY_RUN=1" in script
    assert "ARGUS_ALLOW_A6000_SOL_ATTN_PAIR_VALUE_HALVES_N1=1" in script
    assert "argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r9-sol-attn-overlay" in script
    assert "REQUIRED_IMAGE_VERSION_LABEL=${REQUIRED_IMAGE_VERSION_LABEL:-r9}" in script
    assert "run_one current_retained_a" in script
    assert "run_one current_retained_b" in script
    assert "run_one pair_value_halves" in script
    assert "request warmup" in script and "request output" in script
    assert "SOL_ATTN_TELEMETRY_ARM_FILE=/evidence/current_retained_a/measure.arm" in script
    assert "SOL_ATTN_TELEMETRY_ARM_FILE=/evidence/current_retained_b/measure.arm" in script
    assert "SOL_ATTN_TELEMETRY_ARM_FILE=/evidence/pair_value_halves/measure.arm" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_STRIDE_AWARE_V=1" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_SKIP_FULL_PREFIX_BLOCKS=1" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_MATERIALIZE=0" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_OUTPUT_DIGEST=1" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_MAX_CALLS=256" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_SHADOW_ROW_STATE_PROBE=1" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_MATERIALIZE=1" not in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_PAIR_VALUE_HALVES=0" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_PAIR_VALUE_HALVES=1" in script
    assert "all_zero_materialization_and_input_copies" in finalizer
    assert "candidate_pair_value_halves_seen" in finalizer
    assert "current_pair_value_halves_absent" in finalizer
    assert "all_skip_full_prefix_blocks_seen" in finalizer
    assert "current_current_attention_output_digests_equal" in finalizer
    assert "pair_attention_output_digests_equal" in finalizer
    assert "row_state_shadow_probe_summary" in finalizer
    assert "same_input_shadow_row_state_route_digest_equal" in finalizer
    assert "sparse_attention_gpu_latency_ms" in finalizer and "denoise_gpu_latency_ms" in finalizer
    assert "peak_gpu_power_w" in finalizer and "host_mem_available_kib" in finalizer
    assert "--init-timeout '$VLLM_OMNI_INIT_TIMEOUT_S'" in script
    assert "--stage-init-timeout '$VLLM_OMNI_STAGE_INIT_TIMEOUT_S'" in script
    assert "VLLM_OMNI_INIT_TIMEOUT_S=${VLLM_OMNI_INIT_TIMEOUT_S:-2400}" in script
    assert "VLLM_OMNI_STAGE_INIT_TIMEOUT_S=${VLLM_OMNI_STAGE_INIT_TIMEOUT_S:-1800}" in script
    assert "SERVER_READY_TIMEOUT_S=${SERVER_READY_TIMEOUT_S:-$((VLLM_OMNI_INIT_TIMEOUT_S + 600))}" in script
    assert "startup_timeout_config.env" in script
    assert "extended_startup_failure" in script
    assert "write_runtime_failure_decision" in script
    assert "finalize_sol_attn_pair_value_halves_diagnostic.py" in script
    assert "decoded_av_comparison_current_vs_current.json" in finalizer
    assert "decoded_av_comparison_current_vs_pair_value_halves.json" in finalizer
    assert "decoded_av_content_used_for_decision" in finalizer
    assert "mp4_sha256_recorded_not_gate_pair" in finalizer
    assert '"hash_equality_used_for_decision": False' in finalizer
    assert "classification = \"reject_no_promotion\"" in finalizer
    assert "current_current_decoded_av_unstable_separate_full_chain_from_kernel" in finalizer
    assert "no_above_noise_n1_signal" in finalizer
    assert "promote_to_matched_n3" in finalizer
    assert "duration=5.166667" in script and "duration=30" not in script and "duration=60" not in script


def test_adaptive_routing_finalizer_treats_zero_fallback_as_zero() -> None:
    finalizer = (PORT / "integration" / "finalize_sol_attn_adaptive_routing_diagnostic.py").read_text(encoding="utf-8")
    assert '"all_fallback_calls_zero": all(int(tel[mode].get("fallback_calls", -1)) == 0 for mode in modes),' in finalizer
    assert 'tel[mode].get("fallback_calls", -1) or -1' not in finalizer


def test_pair_value_halves_shadow_localizer_is_no_raw_tensor_reject_gate() -> None:
    localizer = (PORT / "integration" / "localize_sol_attn_pair_value_halves_shadow.py").read_text(encoding="utf-8")
    assert "raw tensors" in localizer
    assert "raw_tensor_exported" in localizer
    assert "reject_no_promotion" in localizer
    assert "bf16_probability_rounding_or_pv_dot_codegen" in localizer
    assert "route_or_exact_block_selection" in localizer
    assert "row_max_or_row_sum" in localizer
    assert "approximate_vs_exact_contribution" in localizer
    assert "v_stride_or_load" in localizer
    assert "lo_hi_store_behavior" in localizer
    assert "promote_to_matched_n3\": False" in localizer


def test_sol_attn_gpu2_diagnostic_requires_fresh_r8_identity_and_resource_telemetry() -> None:
    script = (PORT / "integration" / "run_gpu2_sol_attn_h3_5step_diagnostic.sh").read_text(encoding="utf-8")
    assert "argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r8-sol-attn-overlay" in script
    assert "r3" not in script and "r6-sol-attn-overlay" not in script and "r7-sol-attn-overlay" not in script
    assert "R8_IID_FILE=" not in script
    assert "EXPECTED_R8_IMAGE_IID" not in script
    assert "REQUIRED_IMAGE_VERSION_LABEL=${REQUIRED_IMAGE_VERSION_LABEL:-r8}" in script
    assert "load_expected_r8_iid" not in script
    assert "verify_r8_readable_image_provenance" in script
    assert "docker image inspect --format '{{.Id}}' \"$IMAGE\"" not in script
    assert "org.opencontainers.image.version" in script
    assert "org.opencontainers.image.base.name" in script
    assert "r8 Sol-Attn" in script
    assert "verify_r8_readable_image_provenance > \"$OUT_DIR/r8_image_identity.env\"" in script
    assert "install -m 0644 \"$PROMPT_FILE\" \"$OUT_DIR/prompt.txt\"" in script
    assert script.index("verify_r8_readable_image_provenance > \"$OUT_DIR/r8_image_identity.env\"") < script.index("\nrecord_gpu_hygiene_preflight\n\nif [[ -n \"$EXPECTED_UUID\" ]]")
    assert "opaque_image_identifier_policy=omitted_not_evidence" in script
    assert "external_r8_build_command=EVIDENCE_DIR=technical_report/evidence/minimax_h3_desktop/sol_engine_port/r8_overlay_image bash ports/minimax_h3_a6000/integration/r8/build_r8_overlay_image.sh" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_DENSE_FIRST_STEPS=0" in script
    assert "sol_attn_diagnostic_dense_first_steps=0_for_metadata_gate_only" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_MATERIALIZE=1" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_MATERIALIZE_MAX_BYTES=1073741824" in script
    assert "sol_attn_materialize_max_bytes=1073741824_for_full_h3_value_view" in script
    assert "sol_attn_diagnostic_materialize=on_for_r8_only" in script
    assert "gpu_hygiene_preflight.json" in script
    assert "gpu_hygiene_blocker.json" in script
    assert "nvidia_smi_compute_apps.csv" in script
    assert "gpu_lease_status.txt" in script
    assert "disk_preflight.txt" in script
    assert "host_resource_before.json" in script
    assert "host_resource_after.json" in script
    assert "gpu_resource_samples.csv" in script
    assert "wall_time.json" in script
    assert "overall_wall_time.json" in script
    assert "resource_monitor.csv" in script
    assert "missing_resource_or_wall_time_telemetry" in script
    assert "if [[ \"$OUT_DIR\" != /* ]]" in script
    assert "OUT_DIR=\"$ROOT/$OUT_DIR\"" in script
    assert "DRY_RUN=1" in script
    assert "ARGUS_ALLOW_GPU2_SOL_ATTN_H3_5STEP=1" in script
    assert "diagnostic_5_step_sol_attn_metadata_gate_not_fidelity_or_performance_claim" in script
