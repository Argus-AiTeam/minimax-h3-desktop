# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
        estimate_sparse_density,
        sm86_capability_guard,
        sol_attn_h3_reference_or_decline,
        triton_candidate_enabled,
    )
else:
    from minimax_h3_a6000.env import DEFAULT_ENV_SWITCHES  # noqa: E402

    H3_HOOK_METADATA_SOURCE = None
    PackedH3Metadata = SolAttnPolicy = SolAttnTelemetry = None
    dense_attention_packed_reference = dense_attention_reference = derive_h3_sol_attn_hook_metadata = None
    derive_sink_range = estimate_sparse_density = None
    sm86_capability_guard = sol_attn_h3_reference_or_decline = triton_candidate_enabled = None


def _qkv(tokens=6, heads=2, dim=128):
    torch.manual_seed(13)
    q = torch.randn(1, tokens, heads, dim).to(torch.bfloat16)
    k = torch.randn(1, tokens, heads, dim).to(torch.bfloat16)
    v = torch.randn(1, tokens, heads, dim).to(torch.bfloat16)
    return q, k, v


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


if torch is None:
    def test_sol_attn_backend_requires_pytorch_dependency():
        assert "1" not in DEFAULT_ENV_SWITCHES.values()
        assert DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_EXACT_INDEXED_STRATEGY") == "auto"

    for _name, _obj in list(globals().items()):
        if _name.startswith("test_") and _name != "test_sol_attn_backend_requires_pytorch_dependency" and callable(_obj):
            globals()[_name] = test_sol_attn_backend_requires_pytorch_dependency
