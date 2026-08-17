# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

try:
    import torch
except ModuleNotFoundError:  # local host may lack PyTorch; CI/runtime gates should install it
    torch = None  # type: ignore[assignment]

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

if torch is not None:
    import minimax_h3_a6000.sol_attn_backend as sol_backend  # noqa: E402
    from minimax_h3_a6000.sol_attn_backend import (  # noqa: E402
        DEFAULT_ENV_SWITCHES,
        H3_HOOK_METADATA_SOURCE,
        PackedH3Metadata,
        SolAttnPolicy,
        SolAttnTelemetry,
        dense_attention_packed_reference,
        dense_attention_reference,
        derive_h3_sol_attn_hook_metadata,
        derive_sink_range,
        effective_sol_attn_policy_for_call,
        estimate_sparse_density,
        sm86_capability_guard,
        sol_attn_h3_reference_or_decline,
        stride_aware_value_layout_reason,
        triton_candidate_enabled,
    )
else:
    from minimax_h3_a6000.env import DEFAULT_ENV_SWITCHES  # noqa: E402

    H3_HOOK_METADATA_SOURCE = None
    PackedH3Metadata = SolAttnPolicy = SolAttnTelemetry = None
    dense_attention_packed_reference = dense_attention_reference = derive_h3_sol_attn_hook_metadata = None
    derive_sink_range = effective_sol_attn_policy_for_call = estimate_sparse_density = None
    sm86_capability_guard = sol_attn_h3_reference_or_decline = stride_aware_value_layout_reason = None
    triton_candidate_enabled = None


def _qkv(tokens=6, heads=2, dim=128):
    torch.manual_seed(13)
    q = torch.randn(1, tokens, heads, dim).to(torch.bfloat16)
    k = torch.randn(1, tokens, heads, dim).to(torch.bfloat16)
    v = torch.randn(1, tokens, heads, dim).to(torch.bfloat16)
    return q, k, v


def _noncontiguous_qkv(tokens=8, heads=2, dim=128):
    torch.manual_seed(17)
    q = torch.randn(1, heads, tokens, dim).to(torch.bfloat16).permute(0, 2, 1, 3)
    k = torch.randn(1, heads, tokens, dim).to(torch.bfloat16).permute(0, 2, 1, 3)
    v = torch.randn(1, heads, tokens, dim).to(torch.bfloat16).permute(0, 2, 1, 3)
    assert not q.is_contiguous() and not k.is_contiguous() and not v.is_contiguous()
    return q, k, v


def _fused_value_view(tokens=8, heads=2, dim=128):
    torch.manual_seed(23)
    plane = heads * dim
    fused = torch.randn(1, tokens, 3 * plane).to(torch.bfloat16).contiguous()
    value = fused[..., 2 * plane :].view(1, tokens, heads, dim)
    assert value.stride() == (tokens * 3 * plane, 3 * plane, dim, 1)
    assert value.storage_offset() == 2 * plane
    assert not value.is_contiguous()
    return value


class _HookLayout:
    def __init__(self, *, prefix_len=2, latent_grid=(1, 2, 2)):
        self.prefix_len = prefix_len
        self.latent_grid = latent_grid


class _HookMetadata:
    def __init__(self, *, video_layout=None, extra=None):
        self.video_layout = video_layout
        self.extra = {} if extra is None else extra


def test_env_switches_are_default_off():
    assert DEFAULT_ENV_SWITCHES
    assert "1" not in DEFAULT_ENV_SWITCHES.values()
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_EXACT_INDEXED_STRATEGY") == "auto"
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_TELEMETRY_JSON", "") == ""
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_TELEMETRY_JSON", "") == ""
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_DENSE_FIRST_STEPS") == "10"
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_DENSE_FIRST_LAYERS") == "2"
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_ROUTING") == "0"
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_TAU") == "1.0"
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_THRESH_TYPE") == "diag"
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_PROFILE") == ""
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_STEP_MIN") == ""
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_STEP_MAX") == ""
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_LAYER_MIN") == ""
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_LAYER_MAX") == ""
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_LAYER_RANGE_SCOPE") == "all_adaptive_steps"
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_STRIDE_AWARE_V") == "0"
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_EXACT_PREFIX_QUERY") == "0"
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_SKIP_FULL_PREFIX_BLOCKS") == "0"
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_STATIC_PREFIX_SINK") == "0"
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_BITMASK_SCHEDULER") == "0"
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_PAIR_VALUE_HALVES") == "0"
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_SHADOW_PAIR_VALUE_HALVES") == "0"
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_SHADOW_ROW_STATE_PROBE") == "0"
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_SHADOW_MAX_MISMATCHES") == "8"
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_OUTPUT_DIGEST") == "0"
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_MAX_CALLS") == "256"
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_MATERIALIZE") == "0"
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_MATERIALIZE_MAX_BYTES") == "67108864"
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_FORWARD_CONFIG") == ""


def test_sol_attn_policy_from_env_reads_diagnostic_dense_gate_override():
    env = {
        "MINIMAX_H3_A6000_ENABLE_OVERLAY": "1",
        "MINIMAX_H3_A6000_ENABLE_TRITON_CANDIDATES": "1",
        "MINIMAX_H3_A6000_ENABLE_SOL_ATTN": "1",
        "MINIMAX_H3_A6000_SOL_ATTN_DENSE_FIRST_STEPS": "0",
        "MINIMAX_H3_A6000_SOL_ATTN_DENSE_FIRST_LAYERS": "2",
        "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_ROUTING": "1",
        "MINIMAX_H3_A6000_SOL_ATTN_TAU": "1.5",
        "MINIMAX_H3_A6000_SOL_ATTN_THRESH_TYPE": "exact",
        "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_PROFILE": "late_steps_unit",
        "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_STEP_MIN": "4",
        "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_STEP_MAX": "6",
        "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_LAYER_MIN": "8",
        "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_LAYER_MAX": "47",
        "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_LAYER_RANGE_SCOPE": "step_min_only",
        "MINIMAX_H3_A6000_SOL_ATTN_EXACT_PREFIX_QUERY": "1",
        "MINIMAX_H3_A6000_SOL_ATTN_SKIP_FULL_PREFIX_BLOCKS": "1",
        "MINIMAX_H3_A6000_SOL_ATTN_STATIC_PREFIX_SINK": "1",
        "MINIMAX_H3_A6000_SOL_ATTN_BITMASK_SCHEDULER": "1",
        "MINIMAX_H3_A6000_SOL_ATTN_PAIR_VALUE_HALVES": "1",
        "MINIMAX_H3_A6000_SOL_ATTN_SHADOW_PAIR_VALUE_HALVES": "1",
        "MINIMAX_H3_A6000_SOL_ATTN_SHADOW_ROW_STATE_PROBE": "1",
        "MINIMAX_H3_A6000_SOL_ATTN_SHADOW_MAX_MISMATCHES": "3",
    }
    policy = SolAttnPolicy.from_env(env)
    assert policy.allow_sparse is True
    assert policy.dense_first_steps == 0
    assert policy.dense_first_layers == 2
    assert policy.adaptive_routing is True
    assert policy.adaptive_routing_policy_error is None
    assert policy.tau == 1.5
    assert policy.thresh_type == "exact"
    assert policy.adaptive_profile == "late_steps_unit"
    assert policy.adaptive_step_min == 4
    assert policy.adaptive_step_max == 6
    assert policy.adaptive_layer_min == 8
    assert policy.adaptive_layer_max == 47
    assert policy.adaptive_layer_range_scope == "step_min_only"
    assert policy.exact_prefix_query is True
    assert policy.skip_full_prefix_blocks is True
    assert policy.static_prefix_sink is True
    assert policy.bitmask_exact_scheduler is True
    assert policy.pair_value_halves is True
    assert policy.shadow_pair_value_halves is True
    assert policy.shadow_row_state_probe is True
    assert policy.shadow_max_mismatches == 3


def test_sol_attn_policy_from_env_reads_default_off_forward_config():
    policy = SolAttnPolicy.from_env({"MINIMAX_H3_A6000_SOL_ATTN_FORWARD_CONFIG": " g64_bv64_w4_s1 "})
    assert policy.forward_config == "g64_bv64_w4_s1"
    assert SolAttnPolicy.from_env({}).forward_config is None


def test_adaptive_routing_env_is_default_off_and_invalid_values_fail_closed():
    assert SolAttnPolicy.from_env({"MINIMAX_H3_A6000_SOL_ATTN_TAU": "2.0"}).adaptive_routing is False
    assert SolAttnPolicy.from_env({"MINIMAX_H3_A6000_SOL_ATTN_THRESH_TYPE": "exact"}).thresh_type == "diag"

    env = {
        "MINIMAX_H3_A6000_ENABLE_OVERLAY": "1",
        "MINIMAX_H3_A6000_ENABLE_TRITON_CANDIDATES": "1",
        "MINIMAX_H3_A6000_ENABLE_SOL_ATTN": "1",
        "MINIMAX_H3_A6000_SOL_ATTN_DENSE_FIRST_STEPS": "0",
        "MINIMAX_H3_A6000_SOL_ATTN_DENSE_FIRST_LAYERS": "0",
        "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_ROUTING": "1",
        "MINIMAX_H3_A6000_SOL_ATTN_TAU": "nan",
        "MINIMAX_H3_A6000_SOL_ATTN_THRESH_TYPE": "bad",
        "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_STEP_MIN": "7",
        "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_STEP_MAX": "2",
        "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_LAYER_MIN": "-1",
        "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_LAYER_RANGE_SCOPE": "bad",
    }
    policy = SolAttnPolicy.from_env(env)
    q, k, v = _qkv(tokens=6)
    metadata = PackedH3Metadata(prefix_len=2, latent_grid=(1, 2, 2), valid_length=6, total_length=6)
    reason = sol_backend.decline_reason(
        query=q,
        key=k,
        value=v,
        metadata=metadata,
        step_index=0,
        layer_index=0,
        policy=policy,
        device_capability=(8, 6),
    )
    assert reason == (
        "unsupported_adaptive_routing_policy:"
        "tau_not_finite;unsupported_threshold_type;adaptive_layer_min_negative;adaptive_step_range_invalid;unsupported_adaptive_layer_range_scope"
    )


def test_guarded_adaptive_routing_uses_retained_tau_outside_guard_and_records_active_calls():
    policy = SolAttnPolicy.from_env(
        {
            "MINIMAX_H3_A6000_ENABLE_OVERLAY": "1",
            "MINIMAX_H3_A6000_ENABLE_TRITON_CANDIDATES": "1",
            "MINIMAX_H3_A6000_ENABLE_SOL_ATTN": "1",
            "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_ROUTING": "1",
            "MINIMAX_H3_A6000_SOL_ATTN_TAU": "1.5",
            "MINIMAX_H3_A6000_SOL_ATTN_THRESH_TYPE": "diag",
            "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_PROFILE": "r9_adaptive_tau1_5_late_steps_diag",
            "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_STEP_MIN": "4",
            "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_LAYER_MIN": "2",
        }
    )
    inactive, inactive_reason = effective_sol_attn_policy_for_call(policy, step_index=3, layer_index=16)
    active, active_reason = effective_sol_attn_policy_for_call(policy, step_index=4, layer_index=16)
    protected_layer, layer_reason = effective_sol_attn_policy_for_call(policy, step_index=4, layer_index=1)

    assert inactive_reason == "step_before_adaptive_min"
    assert inactive.adaptive_routing is False
    assert inactive.tau == 1.0 and inactive.thresh_type == "diag"
    assert active_reason is None
    assert active.adaptive_routing is True and active.tau == 1.5
    assert layer_reason == "layer_before_adaptive_min"
    assert protected_layer.adaptive_routing is False and protected_layer.tau == 1.0


def test_step_min_only_layer_scope_preserves_r10_after_step2():
    policy = SolAttnPolicy.from_env(
        {
            "MINIMAX_H3_A6000_ENABLE_OVERLAY": "1",
            "MINIMAX_H3_A6000_ENABLE_TRITON_CANDIDATES": "1",
            "MINIMAX_H3_A6000_ENABLE_SOL_ATTN": "1",
            "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_ROUTING": "1",
            "MINIMAX_H3_A6000_SOL_ATTN_TAU": "1.5",
            "MINIMAX_H3_A6000_SOL_ATTN_THRESH_TYPE": "diag",
            "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_PROFILE": "r12_adaptive_tau1_5_step2_layers34_49_diag",
            "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_STEP_MIN": "2",
            "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_LAYER_MIN": "34",
            "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_LAYER_MAX": "49",
            "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_LAYER_RANGE_SCOPE": "step_min_only",
        }
    )

    early_step, early_reason = effective_sol_attn_policy_for_call(policy, step_index=1, layer_index=40)
    protected_step2, protected_reason = effective_sol_attn_policy_for_call(policy, step_index=2, layer_index=20)
    active_step2, active_step2_reason = effective_sol_attn_policy_for_call(policy, step_index=2, layer_index=40)
    r10_preserved_after_step2, after_reason = effective_sol_attn_policy_for_call(policy, step_index=3, layer_index=20)

    assert early_reason == "step_before_adaptive_min"
    assert early_step.adaptive_routing is False and early_step.tau == 1.0
    assert protected_reason == "layer_before_adaptive_min"
    assert protected_step2.adaptive_routing is False and protected_step2.tau == 1.0
    assert active_step2_reason is None
    assert active_step2.adaptive_routing is True and active_step2.tau == 1.5
    assert after_reason is None
    assert r10_preserved_after_step2.adaptive_routing is True and r10_preserved_after_step2.tau == 1.5


def test_guarded_adaptive_routing_telemetry_marks_guard_active_and_inactive(monkeypatch):
    q, k, _ = _qkv(tokens=128)
    value = _fused_value_view(tokens=128)
    metadata = PackedH3Metadata(prefix_len=70, latent_grid=(1, 1, 40), valid_length=110, total_length=128)
    telemetry = SolAttnTelemetry()
    seen_taus = []

    def fake_decline_reason(**kwargs):
        return None

    def fake_kernel(q_full, k_full, v_full, **kwargs):
        seen_taus.append(float(kwargs["tau"]))
        out = torch.zeros(q_full.shape, dtype=q_full.dtype)
        valid = int(kwargs["tokens"])
        out[:, :valid] = dense_attention_reference(
            q_full[:, :valid], k_full[:, :valid], v_full[:, :valid], softmax_scale=kwargs.get("scale")
        )
        return out

    monkeypatch.setattr(sol_backend, "decline_reason", fake_decline_reason)
    monkeypatch.setattr(sol_backend, "_load_sol_attn_sm86", lambda: fake_kernel)
    policy = SolAttnPolicy(
        allow_sparse=True,
        stride_aware_value=True,
        adaptive_routing=True,
        tau=1.5,
        thresh_type="diag",
        adaptive_profile="r9_adaptive_tau1_5_late_steps_diag",
        adaptive_step_min=4,
    )
    for step_index in (3, 4):
        sol_attn_h3_reference_or_decline(
            q,
            k,
            value,
            metadata=metadata,
            step_index=step_index,
            layer_index=2,
            policy=policy,
            telemetry=telemetry,
            device_capability=(8, 6),
        )

    assert seen_taus == [1.0, 1.5]
    inactive, active = telemetry.density_samples[-2:]
    assert inactive["adaptive_guard_active"] is False
    assert inactive["adaptive_guard_reason"] == "step_before_adaptive_min"
    assert inactive["tau"] == 1.0 and inactive["adaptive_candidate_tau"] == 1.5
    assert active["adaptive_guard_active"] is True
    assert active["adaptive_guard_reason"] == "active"
    assert active["tau"] == 1.5 and active["adaptive_profile"] == "r9_adaptive_tau1_5_late_steps_diag"


def test_sol_attn_diagnostic_output_digest_is_default_off_bounded_and_no_raw(monkeypatch: pytest.MonkeyPatch):
    q, k, v = _qkv(tokens=8)
    metadata = PackedH3Metadata(prefix_len=2, latent_grid=(1, 2, 2), valid_length=6, total_length=8)
    policy = SolAttnPolicy(pair_value_halves=True, stride_aware_value=True, skip_full_prefix_blocks=True)
    telemetry = SolAttnTelemetry()

    telemetry.record_diagnostic_output(
        query=q,
        key=k,
        value=v,
        output=v,
        metadata=metadata,
        step_index=10,
        layer_index=2,
        policy=policy,
        sink_range=(0, 2),
        stage="unit_test",
    )
    assert telemetry.diagnostic_output_records == []

    monkeypatch.setenv("MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_OUTPUT_DIGEST", "1")
    monkeypatch.setenv("MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_MAX_CALLS", "1")
    telemetry.record_diagnostic_output(
        query=q,
        key=k,
        value=v,
        output=v,
        metadata=metadata,
        step_index=10,
        layer_index=2,
        policy=policy,
        sink_range=(0, 2),
        stage="unit_test",
    )
    telemetry.record_diagnostic_output(
        query=q,
        key=k,
        value=v,
        output=v,
        metadata=metadata,
        step_index=11,
        layer_index=3,
        policy=policy,
        sink_range=(0, 2),
        stage="unit_test_over_cap",
    )

    assert len(telemetry.diagnostic_output_records) == 1
    record = telemetry.diagnostic_output_records[0]
    assert record["raw_tensor_exported"] is False
    assert record["metadata"]["step_index"] == 10
    assert record["policy"]["pair_value_halves"] is True
    assert record["output_digest"]["algorithm"] == "sha256"
    assert len(record["output_digest"]["sha256"]) == 64
    assert "values" not in record and "tensor" not in record


def test_sol_attn_shadow_pair_value_halves_records_only_sanitized_scalar_mismatch():
    q, k, v = _qkv(tokens=8)
    metadata = PackedH3Metadata(prefix_len=2, latent_grid=(1, 2, 2), valid_length=6, total_length=8)
    telemetry = SolAttnTelemetry()
    retained = torch.zeros_like(v)
    candidate = retained.clone()
    candidate[:, 3, 0, 65] = torch.tensor(0.25, dtype=candidate.dtype)
    policy = SolAttnPolicy(shadow_pair_value_halves=True, shadow_max_mismatches=1, stride_aware_value=True)

    telemetry.record_shadow_pair_value_halves(
        query=q,
        key=k,
        value=v,
        retained_output=retained,
        candidate_output=candidate,
        metadata=metadata,
        step_index=12,
        layer_index=4,
        policy=policy,
        sink_range=(0, 2),
        sparse_call_index=7,
    )

    assert telemetry.shadow_pair_value_halves_calls == 1
    assert telemetry.shadow_pair_value_halves_mismatch_count == 1
    assert telemetry.shadow_pair_value_halves_record_failures == 0
    assert len(telemetry.shadow_pair_value_halves_records) == 1
    record = telemetry.shadow_pair_value_halves_records[0]
    assert record["raw_tensor_exported"] is False
    assert record["candidate_marker"] == "pair_value_halves_shadow_candidate"
    assert record["returned_output"] == "retained_current"
    assert record["metadata"]["step_index"] == 12
    assert record["error"]["max_abs"] == 0.25
    assert record["error"]["argmax_region_bucket"]["region"] == "tail"
    assert record["error"]["argmax_region_bucket"]["value_half"] == "hi"
    assert record["region_equality"] == {"prefix_equal": True, "tail_equal": False, "padding_equal": True}
    assert record["finite_equality"]["finite_mask_equal"] is True
    assert "row_state_probe" not in record
    assert all(len(value) == 64 for value in record["code_hashes"].values())
    assert "values" not in record and "path" not in json.dumps(record).lower()


def test_shadow_row_state_probe_records_only_scalar_route_and_pv_summaries():
    q, k, v = _qkv(tokens=8)
    metadata = PackedH3Metadata(prefix_len=2, latent_grid=(1, 2, 2), valid_length=6, total_length=8)
    telemetry = SolAttnTelemetry()
    retained = torch.zeros_like(v)
    candidate = retained.clone()
    candidate[:, 3, 0, 65] = torch.tensor(0.25, dtype=candidate.dtype)
    policy = SolAttnPolicy(
        shadow_pair_value_halves=True,
        shadow_row_state_probe=True,
        shadow_max_mismatches=1,
        stride_aware_value=True,
    )

    telemetry.record_shadow_pair_value_halves(
        query=q,
        key=k,
        value=v,
        retained_output=retained,
        candidate_output=candidate,
        metadata=metadata,
        step_index=12,
        layer_index=4,
        policy=policy,
        sink_range=(0, 1),
        sparse_call_index=7,
    )

    record = telemetry.shadow_pair_value_halves_records[0]
    probe = record["row_state_probe"]
    assert probe["status"] == "pass"
    assert probe["raw_tensor_exported"] is False
    assert probe["route"]["route_digest_equal"] is True
    assert probe["route"]["exact_order_digest_equal"] is True
    assert probe["row_state"]["row_max_abs_delta"] == 0.0
    assert probe["row_state"]["row_sum_abs_delta"] == 0.0
    assert "pv_contribution_summaries" in probe
    assert "lo" in probe["pv_contribution_summaries"]["total_numerator"]
    assert probe["raw_tensor_payload_available"] is False
    assert probe["route"]["route_values_exported"] is False
    assert probe["route"]["exact_block_indices_exported"] is False
    assert probe["pv_contribution_summaries"]["lo_hi_raw_vectors_exported"] is False
    probe_json = json.dumps(probe).lower()
    assert "raw_tensor_values" not in probe_json
    assert "path" not in probe_json


def test_sol_attn_forward_config_group_hook_is_default_off_static():
    source = (SRC / "minimax_h3_a6000" / "sol_attn_triton_sm86.py").read_text(encoding="utf-8")
    assert '"g16_bv64_w4_s1": {"GROUP": 16, "BV": 64, "num_warps": 4, "num_stages": 1}' in source
    assert '"g64_bv64_w4_s1": {"GROUP": 64, "BV": 64, "num_warps": 4, "num_stages": 1}' in source
    assert 'group = int(config.get("GROUP", GROUP_SIZE))' in source
    assert 'GROUP=group' in source
    assert "_forward_ptr_pair_v64_kernel" in source
    assert "pair_value_halves" in source
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_FORWARD_CONFIG") == ""
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_ROUTING") == "0"
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_PAIR_VALUE_HALVES") == "0"
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_SHADOW_PAIR_VALUE_HALVES") == "0"
    assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SOL_ATTN_SHADOW_ROW_STATE_PROBE") == "0"


def test_h3_hook_metadata_derivation_uses_source_backed_layout_only():
    hook = _HookMetadata(
        video_layout=_HookLayout(prefix_len=2, latent_grid=(1, 2, 2)),
        extra={"valid_kv_length": 6, "h3_denoise_step_index": 10, "h3_layer_index": 2},
    )
    derived, reason = derive_h3_sol_attn_hook_metadata(hook, total_length=8)

    assert reason is None
    assert derived is not None
    assert derived.source == H3_HOOK_METADATA_SOURCE
    assert derived.step_index == 10 and derived.layer_index == 2
    assert derived.packed.prefix_len == 2
    assert derived.packed.latent_grid == (1, 2, 2)
    assert derived.packed.valid_length == 6
    assert derived.packed.total_length == 8


def test_h3_hook_metadata_fail_closed_blockers_are_stable():
    missing, reason = derive_h3_sol_attn_hook_metadata(None, total_length=8)
    assert missing is None
    assert reason == "missing_h3_hook_metadata:missing_attention_metadata"

    no_step = _HookMetadata(
        video_layout=_HookLayout(prefix_len=2, latent_grid=(1, 2, 2)),
        extra={"valid_kv_length": 6},
    )
    missing, reason = derive_h3_sol_attn_hook_metadata(no_step, total_length=8)
    assert missing is None
    assert reason == "missing_h3_hook_metadata:missing_step_layer_metadata"

    bad_layout = _HookMetadata(
        video_layout=_HookLayout(prefix_len=2, latent_grid=(1, 2, 2)),
        extra={"valid_kv_length": 5, "h3_denoise_step_index": 10, "h3_layer_index": 2},
    )
    missing, reason = derive_h3_sol_attn_hook_metadata(bad_layout, total_length=8)
    assert missing is None
    assert reason == "missing_h3_hook_metadata:invalid_packed_video_layout"


def test_packed_metadata_and_sink_contract():
    metadata = PackedH3Metadata(prefix_len=2, latent_grid=(1, 2, 2), valid_length=6, total_length=64)
    assert metadata.video_rows == 4
    assert metadata.video_start == 2
    assert metadata.video_end == 6
    assert derive_sink_range(metadata, SolAttnPolicy()) == (0, 2)
    assert derive_sink_range(metadata, SolAttnPolicy(sink_start=1, sink_tokens=3)) == (1, 4)
    with pytest.raises(ValueError):
        PackedH3Metadata(prefix_len=2, latent_grid=(1, 2, 2), valid_length=5, total_length=64)
    with pytest.raises(ValueError):
        PackedH3Metadata(prefix_len=2, latent_grid=(1, 2, 2), valid_length=7, total_length=64)
    with pytest.raises(ValueError):
        derive_sink_range(metadata, SolAttnPolicy(sink_start=5, sink_tokens=2))


def test_default_sol_attn_declines_dense_and_matches_reference():
    q, k, v = _qkv(tokens=8)
    metadata = PackedH3Metadata(prefix_len=2, latent_grid=(1, 2, 2), valid_length=6, total_length=8)
    telemetry = SolAttnTelemetry()

    out = sol_attn_h3_reference_or_decline(
        q,
        k,
        v,
        metadata=metadata,
        step_index=20,
        layer_index=10,
        telemetry=telemetry,
        device_capability=(8, 6),
    )

    assert torch.equal(out, dense_attention_packed_reference(q, k, v, metadata=metadata))
    assert torch.count_nonzero(out[:, 6:]) == 0
    assert telemetry.dense_calls == 1
    assert telemetry.decline_reasons == {"env_disabled": 1}
    assert telemetry.sparse_candidate_calls == 0


def test_dense_first_cpu_decline_and_density_telemetry_contract():
    q, k, v = _qkv(tokens=8)
    metadata = PackedH3Metadata(prefix_len=2, latent_grid=(1, 2, 2), valid_length=6, total_length=8)
    policy = SolAttnPolicy(allow_sparse=True)

    early = SolAttnTelemetry()
    sol_attn_h3_reference_or_decline(
        q, k, v, metadata=metadata, step_index=0, layer_index=10, policy=policy, telemetry=early, device_capability=(8, 6)
    )
    assert early.decline_reasons == {"dense_first_steps": 1}

    cpu_decline = SolAttnTelemetry()
    sol_attn_h3_reference_or_decline(
        q, k, v, metadata=metadata, step_index=10, layer_index=2, policy=policy, telemetry=cpu_decline, device_capability=(8, 6)
    )
    assert cpu_decline.sparse_candidate_calls == 0
    assert cpu_decline.fallback_calls == 0
    assert cpu_decline.decline_reasons == {"unsupported_device": 1}

    density = estimate_sparse_density(metadata, policy)
    assert density["kind"] == "static_exact_block_lower_bound"
    assert density["sink_start"] == 0 and density["sink_end"] == 2
    assert 0 < density["exact_density_lower_bound"] <= 1


def test_sm86_and_triton_candidate_guards_do_not_probe_cuda():
    assert sm86_capability_guard((8, 6)) is True
    assert sm86_capability_guard((8, 0)) is False
    assert sm86_capability_guard(None) is False
    assert triton_candidate_enabled(device_capability=(8, 6), env={}) is False
    on = {
        "MINIMAX_H3_A6000_ENABLE_OVERLAY": "1",
        "MINIMAX_H3_A6000_ENABLE_TRITON_CANDIDATES": "1",
    }
    assert triton_candidate_enabled(device_capability=(8, 6), env=on) is True
    assert triton_candidate_enabled(device_capability=(9, 0), env=on) is False


def test_cache_and_shape_declines():
    q, k, v = _qkv(dim=64)
    metadata = PackedH3Metadata(prefix_len=2, latent_grid=(1, 2, 2), valid_length=6, total_length=6)
    telemetry = SolAttnTelemetry()
    sol_attn_h3_reference_or_decline(
        q,
        k,
        v,
        metadata=metadata,
        step_index=10,
        layer_index=2,
        policy=SolAttnPolicy(allow_sparse=True, cache_enabled=True),
        telemetry=telemetry,
        device_capability=(8, 6),
    )
    assert telemetry.decline_reasons == {"cache_disabled_by_contract": 1}

    telemetry = SolAttnTelemetry()
    sol_attn_h3_reference_or_decline(
        q,
        k,
        v,
        metadata=metadata,
        step_index=10,
        layer_index=2,
        policy=SolAttnPolicy(allow_sparse=True),
        telemetry=telemetry,
        device_capability=(8, 6),
    )
    assert telemetry.decline_reasons == {"unsupported_qkv_layout": 1}


def test_noncontiguous_default_off_declines_with_layout_telemetry():
    q, k, v = _noncontiguous_qkv(tokens=8)
    metadata = PackedH3Metadata(prefix_len=2, latent_grid=(1, 2, 2), valid_length=6, total_length=8)
    telemetry = SolAttnTelemetry()

    out = sol_attn_h3_reference_or_decline(
        q,
        k,
        v,
        metadata=metadata,
        step_index=10,
        layer_index=2,
        policy=SolAttnPolicy(allow_sparse=True),
        telemetry=telemetry,
        device_capability=(8, 6),
    )

    assert torch.equal(out, dense_attention_packed_reference(q, k, v, metadata=metadata))
    assert telemetry.decline_reasons == {"unsupported_contiguity": 1}
    assert telemetry.sparse_candidate_calls == 0
    assert telemetry.materialize_copy_count == 0
    assert telemetry.layout_samples[0]["stage"] == "pre_decline"
    assert telemetry.layout_samples[0]["tensors"][0]["is_contiguous"] is False
    assert telemetry.layout_samples[0]["tensors"][0]["shape"] == [1, 8, 2, 128]


def test_source_backed_fused_value_layout_is_the_only_strided_layout_accepted():
    value = _fused_value_view(tokens=8)
    assert stride_aware_value_layout_reason(value) is None

    plane = value.shape[2] * value.shape[3]
    fused = value.as_strided((1, 8, 3 * plane), (8 * 3 * plane, 3 * plane, 1), storage_offset=0)
    wrong_segment = fused[..., :plane].view_as(value)
    assert stride_aware_value_layout_reason(wrong_segment) == "storage_offset"

    overlapping = torch.as_strided(
        fused,
        value.shape,
        (8 * 3 * plane, plane - 1, value.shape[3], 1),
        storage_offset=2 * plane,
    )
    assert stride_aware_value_layout_reason(overlapping) == "token_stride"


def test_stride_aware_value_matches_materialized_reference_without_input_copy(monkeypatch):
    q, k, _ = _qkv(tokens=8)
    value = _fused_value_view(tokens=8)
    metadata = PackedH3Metadata(prefix_len=2, latent_grid=(1, 2, 2), valid_length=6, total_length=8)
    telemetry = SolAttnTelemetry()

    def fake_decline_reason(**kwargs):
        assert kwargs["query"].is_contiguous() and kwargs["key"].is_contiguous()
        if not kwargs["value"].is_contiguous():
            assert kwargs["policy"].stride_aware_value is True
            reason = stride_aware_value_layout_reason(kwargs["value"])
            return None if reason is None else f"unsupported_stride_aware_v_layout:{reason}"
        return None

    def fake_kernel(q_full, k_full, v_full, **kwargs):
        assert q_full.shape[1] == 8 and kwargs["tokens"] == 6
        assert not v_full.is_contiguous()
        assert kwargs["allow_strided_value"] is True
        out = torch.zeros(q_full.shape, dtype=q_full.dtype)
        valid = int(kwargs["tokens"])
        out[:, :valid] = dense_attention_reference(
            q_full[:, :valid], k_full[:, :valid], v_full[:, :valid], softmax_scale=kwargs.get("scale")
        )
        return out

    monkeypatch.setattr(sol_backend, "decline_reason", fake_decline_reason)
    monkeypatch.setattr(sol_backend, "_load_sol_attn_sm86", lambda: fake_kernel)
    out = sol_attn_h3_reference_or_decline(
        q,
        k,
        value,
        metadata=metadata,
        step_index=10,
        layer_index=2,
        policy=SolAttnPolicy(allow_sparse=True, stride_aware_value=True),
        telemetry=telemetry,
        device_capability=(8, 6),
    )

    materialized_reference = dense_attention_packed_reference(q, k, value.contiguous(), metadata=metadata)
    assert torch.equal(out, materialized_reference)
    assert telemetry.sparse_calls == 1 and telemetry.fallback_calls == 0
    assert telemetry.stride_aware_value_calls == 1
    assert telemetry.stride_aware_value_bytes == value.numel() * value.element_size()
    assert telemetry.materialize_copy_count == 0 and telemetry.materialize_copy_bytes == 0
    assert telemetry.density_samples[-1]["stride_aware_value"] is True
    assert "diagnostic_materialized_qkv" not in telemetry.density_samples[-1]



def test_exact_prefix_query_experiment_skips_dense_overwrite_and_records_telemetry(monkeypatch):
    q, k, _ = _qkv(tokens=8)
    value = _fused_value_view(tokens=8)
    metadata = PackedH3Metadata(prefix_len=2, latent_grid=(1, 2, 2), valid_length=6, total_length=8)
    telemetry = SolAttnTelemetry()

    def fake_decline_reason(**kwargs):
        reason = stride_aware_value_layout_reason(kwargs["value"])
        return None if reason is None else f"unsupported_stride_aware_v_layout:{reason}"

    def fake_kernel(q_full, k_full, v_full, **kwargs):
        assert kwargs["prefix_exact_tokens"] == 2
        assert kwargs["allow_strided_value"] is True
        out = torch.zeros(q_full.shape, dtype=q_full.dtype)
        valid = int(kwargs["tokens"])
        out[:, :valid] = dense_attention_reference(
            q_full[:, :valid], k_full[:, :valid], v_full[:, :valid], softmax_scale=kwargs.get("scale")
        )
        return out

    monkeypatch.setattr(sol_backend, "decline_reason", fake_decline_reason)
    monkeypatch.setattr(sol_backend, "_load_sol_attn_sm86", lambda: fake_kernel)
    out = sol_attn_h3_reference_or_decline(
        q,
        k,
        value,
        metadata=metadata,
        step_index=10,
        layer_index=2,
        policy=SolAttnPolicy(allow_sparse=True, stride_aware_value=True, exact_prefix_query=True),
        telemetry=telemetry,
        device_capability=(8, 6),
    )

    assert torch.equal(out, dense_attention_packed_reference(q, k, value.contiguous(), metadata=metadata))
    assert telemetry.sparse_calls == 1 and telemetry.fallback_calls == 0
    assert telemetry.exact_prefix_query_calls == 1
    assert telemetry.prefix_query_dense_calls == 0
    assert telemetry.materialize_copy_count == 0 and telemetry.materialize_copy_bytes == 0
    assert telemetry.density_samples[-1]["exact_prefix_query"] is True



def test_skip_full_prefix_blocks_and_static_prefix_sink_preserve_dense_prefix_overwrite(monkeypatch):
    q, k, _ = _qkv(tokens=128)
    value = _fused_value_view(tokens=128)
    metadata = PackedH3Metadata(prefix_len=70, latent_grid=(1, 1, 40), valid_length=110, total_length=128)
    telemetry = SolAttnTelemetry()

    def fake_decline_reason(**kwargs):
        reason = stride_aware_value_layout_reason(kwargs["value"])
        return None if reason is None else f"unsupported_stride_aware_v_layout:{reason}"

    def fake_kernel(q_full, k_full, v_full, **kwargs):
        assert kwargs["prefix_exact_tokens"] == 70
        assert kwargs["exact_prefix_query"] is False
        assert kwargs["skip_full_prefix_blocks"] is True
        assert kwargs["static_prefix_sink"] is True
        assert kwargs["pair_value_halves"] is False
        assert kwargs["allow_strided_value"] is True
        out = torch.full(q_full.shape, -7.0, dtype=q_full.dtype)
        valid = int(kwargs["tokens"])
        out[:, :valid] = dense_attention_reference(
            q_full[:, :valid], k_full[:, :valid], v_full[:, :valid], softmax_scale=kwargs.get("scale")
        )
        # This simulates skipped/unwritten full-prefix rows.  The wrapper must
        # repair them with the dense prefix overwrite while leaving tail rows as
        # returned by the kernel.  The real kernel also zeros padding internally.
        out[:, :64] = 0
        out[:, valid:] = 0
        return out

    monkeypatch.setattr(sol_backend, "decline_reason", fake_decline_reason)
    monkeypatch.setattr(sol_backend, "_load_sol_attn_sm86", lambda: fake_kernel)
    out = sol_attn_h3_reference_or_decline(
        q,
        k,
        value,
        metadata=metadata,
        step_index=10,
        layer_index=2,
        policy=SolAttnPolicy(
            allow_sparse=True,
            stride_aware_value=True,
            skip_full_prefix_blocks=True,
            static_prefix_sink=True,
        ),
        telemetry=telemetry,
        device_capability=(8, 6),
    )

    reference = dense_attention_packed_reference(q, k, value.contiguous(), metadata=metadata)
    assert torch.equal(out[:, :70], reference[:, :70])
    assert torch.equal(out[:, 70:110], reference[:, 70:110])
    assert torch.count_nonzero(out[:, 110:]) == 0
    assert telemetry.sparse_calls == 1 and telemetry.fallback_calls == 0
    assert telemetry.prefix_query_dense_calls == 1
    assert telemetry.exact_prefix_query_calls == 0
    assert telemetry.materialize_copy_count == 0 and telemetry.materialize_copy_bytes == 0
    assert telemetry.density_samples[-1]["skip_full_prefix_blocks"] is True
    assert telemetry.density_samples[-1]["skipped_full_prefix_query_blocks_estimate"] == 1
    assert telemetry.density_samples[-1]["static_prefix_sink"] is True
    assert telemetry.density_samples[-1]["static_prefix_sink_blocks_estimate"] == 2



def test_pair_value_halves_probe_preserves_dense_prefix_overwrite_contract(monkeypatch):
    q, k, _ = _qkv(tokens=128)
    value = _fused_value_view(tokens=128)
    metadata = PackedH3Metadata(prefix_len=70, latent_grid=(1, 1, 40), valid_length=110, total_length=128)
    telemetry = SolAttnTelemetry()

    def fake_decline_reason(**kwargs):
        reason = stride_aware_value_layout_reason(kwargs["value"])
        return None if reason is None else f"unsupported_stride_aware_v_layout:{reason}"

    def fake_kernel(q_full, k_full, v_full, **kwargs):
        assert kwargs["pair_value_halves"] is True
        assert kwargs["skip_full_prefix_blocks"] is True
        assert kwargs["exact_prefix_query"] is False
        assert kwargs["allow_strided_value"] is True
        out = torch.zeros(q_full.shape, dtype=q_full.dtype)
        valid = int(kwargs["tokens"])
        out[:, :valid] = dense_attention_reference(
            q_full[:, :valid], k_full[:, :valid], v_full[:, :valid], softmax_scale=kwargs.get("scale")
        )
        return out

    monkeypatch.setattr(sol_backend, "decline_reason", fake_decline_reason)
    monkeypatch.setattr(sol_backend, "_load_sol_attn_sm86", lambda: fake_kernel)
    out = sol_attn_h3_reference_or_decline(
        q,
        k,
        value,
        metadata=metadata,
        step_index=10,
        layer_index=2,
        policy=SolAttnPolicy(
            allow_sparse=True,
            stride_aware_value=True,
            skip_full_prefix_blocks=True,
            pair_value_halves=True,
        ),
        telemetry=telemetry,
        device_capability=(8, 6),
    )

    reference = dense_attention_packed_reference(q, k, value.contiguous(), metadata=metadata)
    assert torch.equal(out[:, :70], reference[:, :70])
    assert torch.equal(out[:, 70:110], reference[:, 70:110])
    assert torch.count_nonzero(out[:, 110:]) == 0
    assert telemetry.sparse_calls == 1 and telemetry.fallback_calls == 0
    assert telemetry.prefix_query_dense_calls == 1
    assert telemetry.density_samples[-1]["pair_value_halves"] is True
    assert telemetry.materialize_copy_count == 0 and telemetry.materialize_copy_bytes == 0


def test_invalid_stride_aware_value_fails_closed_without_materializing():
    q, k, _ = _qkv(tokens=8)
    value = _fused_value_view(tokens=8)
    plane = value.shape[2] * value.shape[3]
    fused = value.as_strided((1, 8, 3 * plane), (8 * 3 * plane, 3 * plane, 1), storage_offset=0)
    invalid = torch.as_strided(
        fused,
        value.shape,
        (8 * 3 * plane, plane - 1, value.shape[3], 1),
        storage_offset=2 * plane,
    )
    metadata = PackedH3Metadata(prefix_len=2, latent_grid=(1, 2, 2), valid_length=6, total_length=8)
    telemetry = SolAttnTelemetry()
    out = sol_attn_h3_reference_or_decline(
        q,
        k,
        invalid,
        metadata=metadata,
        step_index=10,
        layer_index=2,
        policy=SolAttnPolicy(
            allow_sparse=True,
            stride_aware_value=True,
            diagnostic_materialize_noncontiguous=True,
            diagnostic_materialize_max_bytes=1_000_000,
        ),
        telemetry=telemetry,
        device_capability=(8, 6),
    )

    assert torch.equal(out, dense_attention_packed_reference(q, k, invalid, metadata=metadata))
    assert telemetry.decline_reasons == {"unsupported_stride_aware_v_layout:token_stride": 1}
    assert telemetry.sparse_candidate_calls == 0 and telemetry.fallback_calls == 0
    assert telemetry.materialize_copy_count == 0 and telemetry.materialize_copy_bytes == 0


def test_stride_aware_mode_never_repairs_nonpacked_query_or_key_with_copy():
    q, k, _ = _noncontiguous_qkv(tokens=8)
    value = _fused_value_view(tokens=8)
    metadata = PackedH3Metadata(prefix_len=2, latent_grid=(1, 2, 2), valid_length=6, total_length=8)
    telemetry = SolAttnTelemetry()
    out = sol_attn_h3_reference_or_decline(
        q,
        k,
        value,
        metadata=metadata,
        step_index=10,
        layer_index=2,
        policy=SolAttnPolicy(
            allow_sparse=True,
            stride_aware_value=True,
            diagnostic_materialize_noncontiguous=True,
            diagnostic_materialize_max_bytes=1_000_000,
        ),
        telemetry=telemetry,
        device_capability=(8, 6),
    )

    assert torch.equal(out, dense_attention_packed_reference(q, k, value, metadata=metadata))
    assert telemetry.decline_reasons == {"unsupported_contiguity": 1}
    assert telemetry.sparse_candidate_calls == 0 and telemetry.fallback_calls == 0
    assert telemetry.materialize_copy_count == 0 and telemetry.materialize_copy_bytes == 0


def test_deferred_cuda_event_telemetry_reports_device_time_separately():
    class _End:
        synchronized = False

        def synchronize(self):
            self.synchronized = True

    class _Start:
        def elapsed_time(self, end):
            assert end.synchronized
            return 3.25

    end = _End()
    telemetry = SolAttnTelemetry()
    telemetry.record_materialize(
        by_tensor={"value": 1024},
        host_enqueue_latency_ms=0.125,
        gpu_event_pair=(_Start(), end),
    )
    assert telemetry.materialize_host_enqueue_latency_ms == 0.125
    assert telemetry.materialize_gpu_copy_latency_ms == 0.0
    assert telemetry.finalize_materialize_gpu_timing() == 3.25
    assert telemetry.materialize_gpu_timing_failures == 0


def test_diagnostic_materialize_cap_declines_without_allocation():
    q, k, v = _noncontiguous_qkv(tokens=8)
    metadata = PackedH3Metadata(prefix_len=2, latent_grid=(1, 2, 2), valid_length=6, total_length=8)
    telemetry = SolAttnTelemetry()

    sol_attn_h3_reference_or_decline(
        q,
        k,
        v,
        metadata=metadata,
        step_index=10,
        layer_index=2,
        policy=SolAttnPolicy(
            allow_sparse=True,
            diagnostic_materialize_noncontiguous=True,
            diagnostic_materialize_max_bytes=1,
        ),
        telemetry=telemetry,
        device_capability=(8, 6),
    )

    assert telemetry.decline_reasons == {"diagnostic_materialize_cap_exceeded": 1}
    assert telemetry.sparse_candidate_calls == 0
    assert telemetry.materialize_copy_count == 0
    assert len(telemetry.layout_samples) == 1


def test_diagnostic_materialize_can_reach_sparse_candidate_with_monkeypatch(monkeypatch):
    q, k, v = _noncontiguous_qkv(tokens=8)
    metadata = PackedH3Metadata(prefix_len=2, latent_grid=(1, 2, 2), valid_length=6, total_length=8)
    telemetry = SolAttnTelemetry()

    def fake_decline_reason(**kwargs):
        tensors = (kwargs["query"], kwargs["key"], kwargs["value"])
        if not all(t.is_contiguous() for t in tensors):
            return "unsupported_contiguity"
        return None

    def fake_kernel(q_full, k_full, v_full, **kwargs):
        valid = int(kwargs["tokens"])
        out = torch.zeros(q_full.shape, dtype=q_full.dtype)
        out[:, :valid] = dense_attention_reference(
            q_full[:, :valid], k_full[:, :valid], v_full[:, :valid], softmax_scale=kwargs.get("scale")
        )
        return out

    monkeypatch.setattr(sol_backend, "decline_reason", fake_decline_reason)
    monkeypatch.setattr(sol_backend, "_load_sol_attn_sm86", lambda: fake_kernel)

    out = sol_attn_h3_reference_or_decline(
        q,
        k,
        v,
        metadata=metadata,
        step_index=10,
        layer_index=2,
        policy=SolAttnPolicy(
            allow_sparse=True,
            diagnostic_materialize_noncontiguous=True,
            diagnostic_materialize_max_bytes=1_000_000,
        ),
        telemetry=telemetry,
        device_capability=(8, 6),
    )

    assert torch.equal(out, dense_attention_packed_reference(q, k, v, metadata=metadata))
    assert telemetry.sparse_candidate_calls == 1
    assert telemetry.sparse_calls == 1
    assert telemetry.decline_reasons == {}
    assert telemetry.materialize_copy_count == 3
    assert telemetry.materialize_copy_bytes == q.numel() * q.element_size() * 3
    assert telemetry.materialize_copy_by_tensor == {
        "query": q.numel() * q.element_size(),
        "key": k.numel() * k.element_size(),
        "value": v.numel() * v.element_size(),
    }
    assert telemetry.materialize_latency_ms >= 0.0
    assert telemetry.layout_samples[-1]["stage"] == "post_diagnostic_materialize"
    assert telemetry.layout_samples[-1]["tensors"][0]["is_contiguous"] is True
    assert telemetry.density_samples[-1]["diagnostic_materialized_qkv"] is True


def test_diagnostic_materialize_candidate_failure_falls_back_dense(monkeypatch):
    q, k, v = _noncontiguous_qkv(tokens=8)
    metadata = PackedH3Metadata(prefix_len=2, latent_grid=(1, 2, 2), valid_length=6, total_length=8)
    telemetry = SolAttnTelemetry()

    def fake_decline_reason(**kwargs):
        tensors = (kwargs["query"], kwargs["key"], kwargs["value"])
        if not all(t.is_contiguous() for t in tensors):
            return "unsupported_contiguity"
        return None

    def failing_kernel(*args, **kwargs):
        raise ValueError("synthetic candidate failure")

    monkeypatch.setattr(sol_backend, "decline_reason", fake_decline_reason)
    monkeypatch.setattr(sol_backend, "_load_sol_attn_sm86", lambda: failing_kernel)

    out = sol_attn_h3_reference_or_decline(
        q,
        k,
        v,
        metadata=metadata,
        step_index=10,
        layer_index=2,
        policy=SolAttnPolicy(
            allow_sparse=True,
            diagnostic_materialize_noncontiguous=True,
            diagnostic_materialize_max_bytes=1_000_000,
        ),
        telemetry=telemetry,
        device_capability=(8, 6),
    )

    assert torch.equal(out, dense_attention_packed_reference(q, k, v, metadata=metadata))
    assert telemetry.sparse_candidate_calls == 1
    assert telemetry.sparse_calls == 0
    assert telemetry.fallback_calls == 1
    assert telemetry.fallback_reasons == {"kernel_error:ValueError": 1}
    assert telemetry.decline_reasons == {"kernel_error:ValueError": 1}


if torch is None:
    def test_sol_attn_backend_requires_pytorch_dependency():
        assert "1" not in DEFAULT_ENV_SWITCHES.values()
        assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_EXACT_INDEXED_STRATEGY") == "auto"

    for _name, _obj in list(globals().items()):
        if _name.startswith("test_") and _name != "test_sol_attn_backend_requires_pytorch_dependency" and callable(_obj):
            globals()[_name] = test_sol_attn_backend_requires_pytorch_dependency
