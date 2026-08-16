# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PORT = ROOT / "ports" / "minimax_h3_a6000"
SRC = PORT / "src"
sys.path.insert(0, str(SRC))

from minimax_h3_a6000.env import DEFAULT_ENV_SWITCHES  # noqa: E402


def test_exact_kernel_source_contains_real_triton_kernels_and_guards():
    source = (SRC / "minimax_h3_a6000" / "exact_kernels.py").read_text(encoding="utf-8")
    ast.parse(source)
    for name in (
        "_indexed_modulate_kernel",
        "_indexed_gate_kernel",
        "_indexed_modulate_strided_kernel",
        "_indexed_gate_strided_kernel",
        "_rope_kernel",
        "_swiglu_kernel",
        "@triton.jit",
        "_round_bf16",
        "torch.cuda.get_device_capability",
        "requires SM86/A6000",
        "freqs must be FP32",
        "freq_channel = channel % half",
        "consumes only the first rotary half",
        "indexed_modulate_bf16_reference",
        "indexed_gate_bf16_reference",
        "apply_rope_bf16_reference",
        "get_exact_kernel_telemetry",
        "write_exact_kernel_telemetry_json",
        "MINIMAX_H3_A6000_ENABLE_TELEMETRY",
        "materialize_copy_bytes",
        "materialize_copy_by_tensor",
        "tensor_layouts",
        "tensor_layout_summary",
        "tensor_layout_samples",
        "stride_aware",
        "strategy",
        "_record_outcome(op_name, \"candidate\")",
    ):
        assert name in source
    assert "float8" not in source.lower()
    assert "pruned" not in source.lower()


def test_exact_kernel_env_switches_default_off():
    for key in (
        "MINIMAX_H3_A6000_ENABLE_OVERLAY",
        "MINIMAX_H3_A6000_ENABLE_FUSED_ADALN",
        "MINIMAX_H3_A6000_ENABLE_FUSED_ROPE",
        "MINIMAX_H3_A6000_ENABLE_FUSED_SWIGLU",
        "MINIMAX_H3_A6000_ENABLE_TRITON_CANDIDATES",
        "MINIMAX_H3_A6000_ENABLE_TELEMETRY",
        "MINIMAX_H3_A6000_TELEMETRY_ATEXIT",
        "MINIMAX_H3_A6000_ENABLE_SHADOW",
        "MINIMAX_H3_A6000_SHADOW_STRICT",
        "MINIMAX_H3_A6000_SOL_ATTN_SHADOW_ROW_STATE_PROBE",
        "MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_OUTPUT_DIGEST",
        "MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_MATERIALIZE",
        "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_ROUTING",
    ):
        assert DEFAULT_ENV_SWITCHES[key] == "0"
    assert DEFAULT_ENV_SWITCHES["MINIMAX_H3_A6000_SOL_ATTN_TAU"] == "1.0"
    assert DEFAULT_ENV_SWITCHES["MINIMAX_H3_A6000_SOL_ATTN_THRESH_TYPE"] == "diag"
    assert DEFAULT_ENV_SWITCHES["MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_PROFILE"] == ""
    assert DEFAULT_ENV_SWITCHES["MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_STEP_MIN"] == ""
    assert DEFAULT_ENV_SWITCHES["MINIMAX_H3_A6000_EXACT_INDEXED_STRATEGY"] == "auto"
    assert DEFAULT_ENV_SWITCHES["MINIMAX_H3_A6000_TELEMETRY_JSON"] == ""
    assert DEFAULT_ENV_SWITCHES["MINIMAX_H3_A6000_SHADOW_CALLS"] == "3"
    assert DEFAULT_ENV_SWITCHES["MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_MAX_CALLS"] == "256"


def test_sol_attn_sm86_source_contains_real_triton_candidate_and_harness():
    source = (SRC / "minimax_h3_a6000" / "sol_attn_triton_sm86.py").read_text(encoding="utf-8")
    backend = (SRC / "minimax_h3_a6000" / "sol_attn_backend.py").read_text(encoding="utf-8")
    harness = (PORT / "gpu_sol_attn_sm86_harness.py").read_text(encoding="utf-8")
    ast.parse(source)
    ast.parse(backend)
    ast.parse(harness)
    for needle in (
        "_reduce_kv_kernel",
        "_forward_ptr_kernel",
        "@triton.jit",
        "arch[0] < 8",
        "requires SM86",
        "sink_start_block",
        "HAS_SINK",
        "V_STRIDE_B",
        "V_STRIDE_T",
        "allow_strided_value",
        "_strided_value_layout_reason",
    ):
        assert needle in source
    for needle in (
        "prefix_query_dense",
        "estimate_sparse_density",
        "static_exact_block_lower_bound",
        "unsupported_device",
        "unsupported_contiguity",
        "diagnostic_materialize_noncontiguous",
        "materialize_copy_bytes",
        "materialize_gpu_copy_latency_ms",
        "stride_aware_value_layout_reason",
        "stride_aware_value_calls",
        "layout_samples",
        "kernel_error:",
        "sol_attn_sm86",
        "adaptive_routing_policy_error",
        "adaptive_routing_guard_reason",
        "MINIMAX_H3_A6000_SOL_ATTN_TAU",
        "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_STEP_MIN",
    ):
        assert needle in backend
    assert "kernel_not_implemented" not in backend
    for needle in (
        "--mode",
        "correctness",
        "bench",
        "device_count()",
        "a6000",
        "(8, 6)",
        "model_load",
        "False",
        "prefix_rows_equal_dense",
        "kernel_candidates_only_not_h3_e2e",
        "warmup < 20",
        "fused-QKV V view",
        "materialized_same_values",
        "adaptive-routing-bench",
        "run_adaptive_routing_bench",
    ):
        assert needle in harness
    for needle in (
        "tl.dot(probability.to(vc_lo.dtype), vc_lo)",
        "tl.dot(probability.to(vc_hi.dtype), vc_hi)",
        "tl.dot(exact_probability.to(v_lo.dtype), v_lo)",
        "tl.dot(exact_probability.to(v_hi.dtype), v_hi)",
        "cast boundary per value half",
    ):
        assert needle in source
    assert "probability_bf16 = probability.to" not in source
    assert "exact_probability_bf16 = exact_probability.to" not in source


def test_gpu_harnesses_are_external_single_a6000_json_gates():
    test_source = (PORT / "gpu_exact_kernel_test.py").read_text(encoding="utf-8")
    bench_source = (PORT / "gpu_exact_kernel_bench.py").read_text(encoding="utf-8")
    for source in (test_source, bench_source):
        ast.parse(source)
        assert "--device" in source and "cuda:0" in source
        assert "--output" in source
        assert "device_count() != 1" in source or "visible != 1" in source
        assert "a6000" in source.lower() and "(8, 6)" in source
        assert "model_load" in source and "False" in source
    assert "compile_status" in test_source
    assert "parents[" not in test_source and "parents[" not in bench_source
    assert "chmod" not in test_source and "chmod" not in bench_source
    assert "max_abs" in test_source and "max_rel" in test_source and "mismatch" in test_source
    for coverage_tag in (
        "fixed_seed",
        "random_inputs",
        "explicit_extreme_values_per_op",
        "tag_index_edges",
        "non_aligned_tail_shapes",
        "representative_T_H_D_shapes",
        "sliced_table_strides",
        "expanded_broadcast_views",
        "explicit_materialize_strategy",
    ):
        assert coverage_tag in test_source
    assert "_inject_extremes" in test_source
    assert "warmup < 20" in bench_source and "repeats < 100" in bench_source
    assert "kernel_candidates_only_not_h3_e2e" in bench_source
    assert "copy_inclusive" in bench_source
    assert "candidate_telemetry_including_warmup" in bench_source


def test_r4_integration_artifacts_are_static_only_and_assert_strategy_telemetry():
    run_script = (PORT / "integration" / "run_gpu2_exact_integration_5step_r4.sh").read_text(encoding="utf-8")
    build_script = (PORT / "integration" / "r4" / "build_r4_overlay_image.sh").read_text(encoding="utf-8")
    dockerfile = (PORT / "integration" / "r4" / "Dockerfile").read_text(encoding="utf-8")
    assert "r4-exact-overlay" in run_script and "r4-exact-overlay" in build_script
    assert "r4 exact-kernel integration candidate" in dockerfile
    assert "--network=none" in build_script and "gpu_flags=none" in build_script
    assert "--gpus" not in build_script
    for needle in (
        "strategies.get('stride_aware', 0) > 0",
        "tensor_layout_samples",
        "tensor_layout_summary",
        "materialize_copy_by_tensor",
        "materialize_copy_bytes",
    ):
        assert needle in run_script


def test_vllm_patch_wires_opt_in_exact_wrappers_at_transformer_boundaries():
    patch = (PORT / "patches" / "vllm_omni_h3_a6000_opt_in.patch").read_text(encoding="utf-8")
    for needle in (
        "vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py",
        "MINIMAX_H3_A6000_ENABLE_TRITON_CANDIDATES",
        "MINIMAX_H3_A6000_ENABLE_FUSED_ADALN",
        "MINIMAX_H3_A6000_ENABLE_FUSED_ROPE",
        "MINIMAX_H3_A6000_ENABLE_FUSED_SWIGLU",
        "indexed_modulate_bf16",
        "indexed_gate_bf16",
        "apply_rope_bf16",
        "swiglu_bf16",
        "order=\"gate_up\"",
        "get_exact_kernel_telemetry",
        "MINIMAX_H3_A6000_ENABLE_TELEMETRY",
        "MINIMAX_H3_A6000_TELEMETRY_JSON",
        "MINIMAX_H3_A6000_EXACT_INDEXED_STRATEGY",
        "MiniMax-H3 A6000 exact telemetry",
        "MINIMAX_H3_A6000_ENABLE_SHADOW",
        "ShadowMismatchError",
    ):
        assert needle in patch
