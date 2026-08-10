# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PORT = ROOT / "ports" / "minimax_h3_a6000"


def test_r3_dockerfile_builds_from_r2_and_does_not_touch_locked_host_tree() -> None:
    dockerfile = (PORT / "integration" / "r3" / "Dockerfile").read_text(encoding="utf-8")
    assert "argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r2" in dockerfile
    assert "FROM ${BASE_IMAGE}" in dockerfile
    assert "COPY ports/minimax_h3_a6000 /opt/minimax_h3_a6000" in dockerfile
    assert "python3 -m pip install" in dockerfile
    assert "--no-deps" in dockerfile
    assert "/opt/minimax_h3_a6000" in dockerfile
    assert "git -C \"$tmp\" apply" in dockerfile
    assert "runtime/single_a6000_bf16/src/vllm-omni" not in dockerfile
    assert "MINIMAX_H3_A6000_ENABLE_OVERLAY=0" in dockerfile
    assert "MINIMAX_H3_A6000_ENABLE_TELEMETRY=0" in dockerfile
    assert "MINIMAX_H3_A6000_EXACT_INDEXED_STRATEGY=auto" in dockerfile
    assert "--gpus" not in dockerfile


def test_r3_build_script_is_build_only_no_gpu_run() -> None:
    script = (PORT / "integration" / "r3" / "build_r3_overlay_image.sh").read_text(encoding="utf-8")
    assert "docker build" in script
    assert "--network=none" in script
    assert "8e2e9b6b53e8-r2" in script
    assert "r3-exact-overlay" in script
    assert "docker run" not in script
    assert "--gpus" not in script


def test_gpu2_five_step_integration_script_verifies_av_and_exact_telemetry() -> None:
    script = (PORT / "integration" / "run_gpu2_exact_integration_5step.sh").read_text(encoding="utf-8")
    assert "GPU_INDEX=${GPU_INDEX:-2}" in script
    assert "num_inference_steps=5" in script
    assert "MINIMAX_H3_A6000_ENABLE_FUSED_ADALN=1" in script
    assert "MINIMAX_H3_A6000_EXACT_INDEXED_STRATEGY=auto" in script
    assert "MINIMAX_H3_A6000_ENABLE_FUSED_ROPE=1" in script
    assert "MINIMAX_H3_A6000_ENABLE_FUSED_SWIGLU=1" in script
    assert "MINIMAX_H3_A6000_ENABLE_SOL_ATTN=0" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_CACHE=0" in script
    assert "MINIMAX_H3_A6000_TELEMETRY_JSON=/evidence/exact/exact_telemetry.json" in script
    assert "av_validation.json" in script
    assert "item['calls'] > 0" in script
    assert "item['candidate'] > 0" in script
    assert "strategy_summary" in script
    assert "materialize_copy_summary" in script
    assert "http_metrics" in script
    assert "5_step_same_workload_integration_not_e2e_benchmark" in script


def test_r4_build_and_integration_scripts_exist_for_next_gpu2_retest() -> None:
    dockerfile = (PORT / "integration" / "r4" / "Dockerfile").read_text(encoding="utf-8")
    build_script = (PORT / "integration" / "r4" / "build_r4_overlay_image.sh").read_text(encoding="utf-8")
    run_script = (PORT / "integration" / "run_gpu2_exact_integration_5step_r4.sh").read_text(encoding="utf-8")
    assert "r4 exact-kernel integration candidate" in dockerfile
    assert "MINIMAX_H3_A6000_EXACT_INDEXED_STRATEGY=auto" in dockerfile
    assert "r4-exact-overlay" in build_script
    assert "r4/Dockerfile" in build_script
    assert "docker run" not in build_script
    assert "--gpus" not in build_script
    assert "r4-exact-overlay" in run_script
    assert "r4_integration_$(date -u +%Y%m%dT%H%M%SZ)" in run_script
    assert "strategy_summary" in run_script
    assert "materialize_copy_summary" in run_script


def test_sol_attn_gpu2_diagnostic_is_dry_run_default_and_not_fidelity() -> None:
    script = (PORT / "integration" / "run_gpu2_sol_attn_h3_5step_diagnostic.sh").read_text(encoding="utf-8")
    assert "DRY_RUN=1" in script
    assert "no GPU execution, Docker execution, network access, downloads, model loading, inference, cache enablement, or publication" in script
    assert "--dry-run" in script and "--execute" in script
    assert "ARGUS_ALLOW_GPU2_SOL_ATTN_H3_5STEP=1" in script
    assert "GPU_INDEX=${GPU_INDEX:-2}" in script
    assert "--network none" in script
    assert "--gpus \"device=$GPU_INDEX\"" in script
    assert "H3_A6000_SOL_ATTN" in script
    assert "num_inference_steps=5" in script
    assert "MINIMAX_H3_A6000_ENABLE_SOL_ATTN=1" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_CACHE=0" in script
    assert "MINIMAX_H3_A6000_ENABLE_FUSED_ADALN=0" in script
    assert "MINIMAX_H3_A6000_ENABLE_FUSED_ROPE=0" in script
    assert "MINIMAX_H3_A6000_ENABLE_FUSED_SWIGLU=0" in script
    assert "sol_attn_telemetry.sol_attn.json" in script
    assert "missing_h3_hook_metadata" in script
    assert "diagnostic_5_step_sol_attn_metadata_gate_not_fidelity_or_performance_claim" in script


def test_r4_per_kernel_ablation_script_is_external_gpu2_diagnostic_gate() -> None:
    script = (PORT / "integration" / "run_gpu2_exact_ablation_5step_r4.sh").read_text(encoding="utf-8")
    assert "External GPU2 gate only" in script
    assert "GPU_INDEX=${GPU_INDEX:-2}" in script
    assert "--gpus \"device=$GPU_INDEX\"" in script
    assert "r4_ablation_$(date -u +%Y%m%dT%H%M%SZ)" in script
    assert "ablation_modes=dense,adaln,rope,swiglu,all_exact" in script
    assert "run_one dense" in script
    assert "run_one adaln" in script
    assert "MINIMAX_H3_A6000_ENABLE_FUSED_ADALN=1" in script
    # The consolidated AdaLN family enables both indexed wrappers through the
    # family switch; deprecated per-op switches remain explicitly off.
    assert "MINIMAX_H3_A6000_ENABLE_FUSED_ADALN_MODULATE=0" in script
    assert "MINIMAX_H3_A6000_ENABLE_FUSED_ADALN_GATE=0" in script
    assert "run_one rope" in script and "MINIMAX_H3_A6000_ENABLE_FUSED_ROPE=1" in script
    assert "run_one swiglu" in script and "MINIMAX_H3_A6000_ENABLE_FUSED_SWIGLU=1" in script
    assert "MINIMAX_H3_A6000_ENABLE_SOL_ATTN=0" in script
    assert "MINIMAX_H3_A6000_SOL_ATTN_CACHE=0" in script
    assert "quality_vs_dense.json" in script
    assert "'expected_candidates': ['indexed_modulate_bf16', 'indexed_gate_bf16']" in script
    assert "'expected_zero_calls': ['apply_rope_bf16', 'swiglu_bf16']" in script
    assert "'expected_zero_calls': ['indexed_modulate_bf16', 'indexed_gate_bf16', 'swiglu_bf16']" in script
    assert "diagnostic_5_step_per_kernel_ablation_not_fidelity_or_performance_claim" in script
    assert "video_psnr_db': 24.63" in script
    assert "audio_waveform_cosine': 0.9776" in script


def test_gpu_harness_regression_for_prior_path_and_permissions_bugs() -> None:
    source = (PORT / "gpu_exact_kernel_test.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert "Path(__file__).resolve().parent / \"src\"" in source
    assert "parents[" not in source
    assert "args.output.parent.mkdir(parents=True, exist_ok=True)" in source
    assert "chmod" not in source
