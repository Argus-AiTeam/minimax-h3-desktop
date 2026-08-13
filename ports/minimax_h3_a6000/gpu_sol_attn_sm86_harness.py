#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""External single-A6000 correctness/bench harness for H3 Sol-Attn.

This script is for an outer single-GPU experiment only. It loads no MiniMax-H3
model weights (``model_load=False``), requires exactly one visible RTX
A6000/SM86 CUDA device, and writes JSON evidence. Correctness uses the observed
fused-QKV V view, forces all Sol-Attn blocks exact, verifies no dense fallback or
input materialization occurred, and compares against a materialized reference.

The ``large-bench`` mode is synthetic/model-free kernel evidence for the observed
r8 H3 large-shape lane.  It constructs B=1, T_total=38272, T_valid=38247, H=56,
D=128 by default, with a source-backed fused-QKV V view whose layout matches the
r8 telemetry (stride [823001088, 21504, 128, 1], storage_offset 14336).  Its
primary ablation isolates the V-materialization variable: the stride-aware
no-copy candidate and the pre-materialized contiguous-V reference run the same
SM86 kernel policy, while explicit ``v.contiguous()`` copy timing/bytes are
reported separately and as a copy-inclusive reference sum.  It bisects to the
largest legal scaled shape if the full shape is infeasible.  It is not H3 E2E,
long-video, BF16-fidelity, product quality, normal-PC, SOTA, or product speedup
evidence.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

import torch  # noqa: E402

from minimax_h3_a6000.sol_attn_backend import (  # noqa: E402
    PackedH3Metadata,
    SolAttnPolicy,
    SolAttnTelemetry,
    decline_reason,
    dense_attention_packed_reference,
    dense_attention_reference,
    sol_attn_h3_reference_or_decline,
)
from minimax_h3_a6000 import sol_attn_triton_sm86 as triton_sm86  # noqa: E402

TARGET_H3_TOTAL = 38_272
TARGET_H3_VALID = 38_247
TARGET_H3_PREFIX = 951
TARGET_H3_HEADS = 56
TARGET_H3_D = 128
TARGET_H3_LATENT_GRID = (6, 56, 111)  # product = 37,296 = 38,247 - 951
TARGET_H3_R8_MATERIALIZE_CALLS = 192
TARGET_H3_R8_MATERIALIZE_BYTES_TOTAL = 105_344_139_264


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="must resolve to the only visible A6000, normally cuda:0")
    parser.add_argument("--output", required=True, type=Path, help="JSON result path")
    parser.add_argument(
        "--mode",
        choices=(
            "correctness",
            "bench",
            "large-bench",
            "phase-bench",
            "prefix-skip-bench",
            "static-prefix-sink-bench",
            "bitmask-scheduler-bench",
            "both",
        ),
        default="both",
    )
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--large-warmup", type=int, default=1)
    parser.add_argument("--large-repeats", type=int, default=5)
    parser.add_argument("--large-target-total", type=int, default=TARGET_H3_TOTAL)
    parser.add_argument("--large-target-valid", type=int, default=TARGET_H3_VALID)
    parser.add_argument("--large-prefix", type=int, default=TARGET_H3_PREFIX)
    parser.add_argument("--large-heads", type=int, default=TARGET_H3_HEADS)
    parser.add_argument("--large-min-total", type=int, default=1024)
    parser.add_argument("--large-bisect-granularity", type=int, default=64)
    parser.add_argument("--large-tau", type=float, default=1.0)
    parser.add_argument("--large-thresh-type", choices=("diag", "exact"), default="diag")
    parser.add_argument("--large-step-index", type=int, default=10)
    parser.add_argument("--large-layer-index", type=int, default=2)
    parser.add_argument(
        "--phase-candidate-forward-config",
        default="bv64_w8_s1",
        choices=tuple(sorted(triton_sm86.FORWARD_CONFIGS)),
        help="fixed forward-kernel launch config for the same-semantics phase-bench candidate",
    )
    parser.add_argument(
        "--phase-min-gain-pct",
        type=float,
        default=3.0,
        help="minimum candidate median improvement required to retain a launch-config candidate",
    )
    parser.add_argument(
        "--prefix-skip-min-gain-pct",
        type=float,
        default=1.0,
        help="minimum whole-lane median improvement required to retain the full-prefix-block skip candidate",
    )
    parser.add_argument(
        "--static-prefix-sink-min-gain-pct",
        type=float,
        default=0.5,
        help="minimum whole-lane or forward-subphase median improvement required to retain the static prefix-sink scheduler",
    )
    parser.add_argument(
        "--bitmask-scheduler-min-gain-pct",
        type=float,
        default=1.0,
        help="minimum whole-lane or forward-subphase median improvement required to retain the bitmask exact-block scheduler",
    )
    return parser.parse_args()


def _validate_single_a6000(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type != "cuda" or (device.index not in (None, 0)):
        raise RuntimeError(f"GPU harness requires --device cuda:0 with one visible GPU, got {device_arg!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    visible = torch.cuda.device_count()
    if visible != 1:
        raise RuntimeError(f"expected exactly one visible GPU, saw {visible}")
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    if cap != (8, 6) or "a6000" not in props.name.lower():
        raise RuntimeError(f"expected one A6000 SM86, saw name={props.name!r} capability={cap}")
    return torch.device("cuda:0")


def _case_tensors(
    device: torch.device,
    *,
    total: int = 96,
    heads: int = 2,
    prefix: int = 6,
    latent_grid: tuple[int, int, int] = (1, 8, 8),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, PackedH3Metadata]:
    q = torch.randn((1, total, heads, 128), device=device, dtype=torch.float32).to(torch.bfloat16).contiguous()
    k = torch.randn_like(q).contiguous()
    plane = heads * 128
    fused_qkv = torch.randn((1, total, 3 * plane), device=device, dtype=torch.float32).to(torch.bfloat16)
    v = fused_qkv[..., 2 * plane :].view(1, total, heads, 128)
    if v.is_contiguous() or tuple(v.stride()) != (total * 3 * plane, 3 * plane, 128, 1):
        raise RuntimeError(f"failed to construct source-backed fused-QKV V view: {v.stride()}")
    valid = prefix + int(latent_grid[0]) * int(latent_grid[1]) * int(latent_grid[2])
    metadata = PackedH3Metadata(prefix_len=prefix, latent_grid=latent_grid, valid_length=valid, total_length=total)
    return q, k, v, metadata


def _metrics(candidate: torch.Tensor, reference: torch.Tensor, prefix: int, valid: int) -> dict[str, Any]:
    torch.cuda.synchronize()
    cand = candidate.detach().to(torch.float32)
    ref = reference.detach().to(torch.float32)
    valid_diff = (cand[:, :valid] - ref[:, :valid]).abs()
    prefix_equal = bool(torch.equal(candidate[:, :prefix], reference[:, :prefix]))
    padding_zero = bool(torch.count_nonzero(candidate[:, valid:]).item() == 0)
    denom = ref[:, :valid].abs().clamp_min(1.0e-7)
    return {
        "max_abs_valid": float(valid_diff.max().item()) if valid_diff.numel() else 0.0,
        "max_rel_valid": float((valid_diff / denom).max().item()) if valid_diff.numel() else 0.0,
        "prefix_rows_equal_dense": prefix_equal,
        "padding_rows_zero": padding_zero,
        "numel_valid": int(candidate[:, :valid].numel()),
    }


def run_correctness(device: torch.device) -> dict[str, Any]:
    q, k, v, metadata = _case_tensors(device)
    telemetry = SolAttnTelemetry()
    policy = SolAttnPolicy(
        allow_sparse=True,
        strict=True,
        stride_aware_value=True,
        tau=-1.0e6,
        thresh_type="diag",
    )
    started = time.time()
    out = sol_attn_h3_reference_or_decline(
        q,
        k,
        v,
        metadata=metadata,
        step_index=10,
        layer_index=2,
        policy=policy,
        telemetry=telemetry,
        device_capability=(8, 6),
    )
    reference = dense_attention_packed_reference(q, k, v.contiguous(), metadata=metadata)
    metrics = _metrics(out, reference, metadata.prefix_len, metadata.valid_length)
    if telemetry.materialize_copy_count or telemetry.materialize_copy_bytes:
        raise RuntimeError(f"stride-aware correctness unexpectedly materialized inputs: {telemetry}")
    if telemetry.stride_aware_value_calls != 1 or telemetry.sparse_calls != 1 or telemetry.fallback_calls:
        raise RuntimeError(f"stride-aware correctness path was not exercised exactly once: {telemetry}")
    return {
        "mode": "correctness",
        "elapsed_s": time.time() - started,
        "compile_status": "compiled_and_launched",
        "telemetry": _jsonable(telemetry.__dict__),
        "tolerance_note": "tau=-1e6 forces all route blocks exact; prefix query rows must equal dense exactly",
        "value_layout": _tensor_layout(v),
        "comparison_reference": "materialized_same_values",
        **metrics,
    }


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "samples_ms": []}
    ordered = sorted(float(x) for x in values)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    mean = sum(ordered) / n
    return {
        "n": n,
        "median_ms": median,
        "mean_ms": mean,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "samples_ms": values,
    }


def _time_cuda(fn: Callable[[], Any], *, warmup: int, repeats: int) -> dict[str, Any]:
    if warmup < 20 or repeats < 100:
        raise RuntimeError("warmup < 20 or repeats < 100 is not accepted for Sol-Attn bench evidence")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
    return {"warmup": warmup, "repeats": repeats, **{k: v for k, v in _stats(times_ms).items() if k != "n"}}


def run_bench(device: torch.device, *, warmup: int, repeats: int) -> dict[str, Any]:
    q, k, v, metadata = _case_tensors(device, total=512, heads=8, prefix=6, latent_grid=(1, 17, 26))
    sparse_policy = SolAttnPolicy(
        allow_sparse=True,
        strict=True,
        stride_aware_value=True,
        tau=1.0,
        thresh_type="diag",
    )

    def sparse_call() -> torch.Tensor:
        telemetry = SolAttnTelemetry()
        return sol_attn_h3_reference_or_decline(
            q,
            k,
            v,
            metadata=metadata,
            step_index=10,
            layer_index=2,
            policy=sparse_policy,
            telemetry=telemetry,
            device_capability=(8, 6),
        )

    def dense_call() -> torch.Tensor:
        return dense_attention_packed_reference(q, k, v, metadata=metadata)

    # Force first compile outside the timed region and confirm no fallback.
    telemetry = SolAttnTelemetry()
    sol_attn_h3_reference_or_decline(
        q,
        k,
        v,
        metadata=metadata,
        step_index=10,
        layer_index=2,
        policy=sparse_policy,
        telemetry=telemetry,
        device_capability=(8, 6),
    )
    if telemetry.fallback_calls or telemetry.sparse_calls != 1:
        raise RuntimeError(f"Sol-Attn compile preflight did not produce one sparse call: {telemetry}")

    sparse = _time_cuda(sparse_call, warmup=warmup, repeats=repeats)
    dense = _time_cuda(dense_call, warmup=warmup, repeats=repeats)
    return {
        "mode": "bench",
        "kernel_candidates_only_not_h3_e2e": True,
        "model_load": False,
        "shape": {"B": 1, "T_total": 512, "T_valid": 448, "H": 8, "D": 128},
        "sparse_ms": sparse,
        "dense_ms": dense,
        "speedup_dense_over_sparse_median": dense["median_ms"] / sparse["median_ms"] if sparse["median_ms"] else None,
        "preflight_telemetry": _jsonable(telemetry.__dict__),
    }


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _tensor_layout(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": [int(x) for x in tensor.shape],
        "stride": [int(x) for x in tensor.stride()],
        "storage_offset": int(tensor.storage_offset()),
        "is_contiguous": bool(tensor.is_contiguous()),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device_type": tensor.device.type,
    }


def _copy_bytes_for(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _latent_grid_for(prefix: int, valid: int, target_total: int, target_valid: int) -> tuple[int, int, int]:
    video_rows = int(valid) - int(prefix)
    if video_rows <= 0:
        raise ValueError(f"valid length must exceed prefix, got valid={valid}, prefix={prefix}")
    if int(target_total) == TARGET_H3_TOTAL and int(target_valid) == TARGET_H3_VALID and video_rows == math.prod(TARGET_H3_LATENT_GRID):
        return TARGET_H3_LATENT_GRID
    return (1, 1, video_rows)


def _shape_for_total(total: int, *, target_total: int, target_valid: int, prefix: int) -> dict[str, int]:
    padding = max(0, int(target_total) - int(target_valid))
    total = int(total)
    valid = total - padding if total - padding > prefix else total
    if valid <= prefix:
        raise ValueError(f"scaled shape too small for prefix={prefix}: total={total}, valid={valid}")
    return {"B": 1, "T_total": total, "T_valid": valid, "H": TARGET_H3_HEADS, "D": TARGET_H3_D, "prefix": int(prefix)}


def _large_case_tensors(
    device: torch.device,
    *,
    total: int,
    valid: int,
    heads: int,
    prefix: int,
    seed: int,
    target_total: int,
    target_valid: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, PackedH3Metadata]:
    if total <= 0 or valid <= 0 or valid > total:
        raise ValueError(f"invalid total/valid: total={total}, valid={valid}")
    if valid <= prefix:
        raise ValueError(f"valid length must exceed prefix, valid={valid}, prefix={prefix}")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    q = torch.randn((1, total, heads, TARGET_H3_D), device=device, dtype=torch.bfloat16, generator=generator).contiguous()
    k = torch.randn(q.shape, device=device, dtype=torch.bfloat16, generator=generator).contiguous()
    plane = heads * TARGET_H3_D
    fused_qkv = torch.randn((1, total, 3 * plane), device=device, dtype=torch.bfloat16, generator=generator).contiguous()
    v = fused_qkv[..., 2 * plane :].view(1, total, heads, TARGET_H3_D)
    expected_stride = (total * 3 * plane, 3 * plane, TARGET_H3_D, 1)
    expected_offset = 2 * plane
    if v.is_contiguous() or tuple(v.stride()) != expected_stride or int(v.storage_offset()) != expected_offset:
        raise RuntimeError(
            "failed to construct source-backed fused-QKV V view: "
            f"shape={tuple(v.shape)} stride={tuple(v.stride())} storage_offset={int(v.storage_offset())}"
        )
    metadata = PackedH3Metadata(
        prefix_len=int(prefix),
        latent_grid=_latent_grid_for(prefix, valid, target_total, target_valid),
        valid_length=int(valid),
        total_length=int(total),
    )
    return q, k, fused_qkv, v, metadata


def _new_large_policy(
    *,
    stride_aware_value: bool,
    tau: float,
    thresh_type: str,
    prefix_query_dense: bool = True,
    exact_prefix_query: bool = False,
    skip_full_prefix_blocks: bool = False,
    static_prefix_sink: bool = False,
    bitmask_exact_scheduler: bool = False,
    forward_config: str | None = None,
) -> SolAttnPolicy:
    return SolAttnPolicy(
        allow_sparse=True,
        strict=True,
        stride_aware_value=bool(stride_aware_value),
        diagnostic_materialize_noncontiguous=False,
        dense_first_steps=10,
        dense_first_layers=2,
        prefix_query_dense=bool(prefix_query_dense),
        exact_prefix_query=bool(exact_prefix_query),
        skip_full_prefix_blocks=bool(skip_full_prefix_blocks),
        static_prefix_sink=bool(static_prefix_sink),
        bitmask_exact_scheduler=bool(bitmask_exact_scheduler),
        forward_config=forward_config,
        tau=float(tau),
        thresh_type=str(thresh_type),
    )


def _checked_sparse_call(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    metadata: PackedH3Metadata,
    *,
    policy: SolAttnPolicy,
    step_index: int,
    layer_index: int,
) -> tuple[torch.Tensor, SolAttnTelemetry]:
    reason = decline_reason(
        query=q,
        key=k,
        value=v,
        metadata=metadata,
        step_index=step_index,
        layer_index=layer_index,
        policy=policy,
        device_capability=(8, 6),
    )
    if reason is not None:
        raise RuntimeError(f"fail_closed_decline_before_sparse:{reason}")
    telemetry = SolAttnTelemetry()
    out = sol_attn_h3_reference_or_decline(
        q,
        k,
        v,
        metadata=metadata,
        step_index=step_index,
        layer_index=layer_index,
        policy=policy,
        telemetry=telemetry,
        device_capability=(8, 6),
    )
    telemetry.finalize_materialize_gpu_timing()
    if telemetry.fallback_calls:
        raise RuntimeError(f"sparse call fell back: {telemetry.fallback_reasons}")
    return out, telemetry


def _time_cuda_distribution(
    fn: Callable[[], tuple[torch.Tensor, SolAttnTelemetry]],
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    if warmup < 0 or repeats < 1:
        raise RuntimeError("large-bench requires repeats >= 1 and warmup >= 0")
    warmup_telemetry: list[dict[str, Any]] = []
    timed_telemetry: list[dict[str, Any]] = []
    for _ in range(warmup):
        out, telemetry = fn()
        torch.cuda.synchronize()
        warmup_telemetry.append(_jsonable(telemetry.__dict__))
        del out
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out, telemetry = fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
        timed_telemetry.append(_jsonable(telemetry.__dict__))
        del out
    peak = _cuda_memory_snapshot()
    return {
        "warmup": int(warmup),
        "repeats": int(repeats),
        "latency_ms": _stats(times_ms),
        "timed_telemetry_summary": _summarize_telemetry(timed_telemetry),
        "warmup_telemetry_summary": _summarize_telemetry(warmup_telemetry),
        "peak_cuda_memory": peak,
    }


def _time_cuda_tensor_distribution(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    if warmup < 0 or repeats < 1:
        raise RuntimeError("large-bench requires repeats >= 1 and warmup >= 0")
    for _ in range(warmup):
        out = fn()
        torch.cuda.synchronize()
        del out
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
        del out
    return {
        "warmup": int(warmup),
        "repeats": int(repeats),
        "latency_ms": _stats(times_ms),
        "peak_cuda_memory": _cuda_memory_snapshot(),
    }


def _cuda_event_call(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    value = fn()
    end.record()
    torch.cuda.synchronize()
    return value, float(start.elapsed_time(end))


def _forward_autotune_cache_snapshot() -> list[dict[str, Any]]:
    cache = getattr(triton_sm86._forward_ptr_kernel, "cache", {})
    rows: list[dict[str, Any]] = []
    for key, config in getattr(cache, "items", lambda: [])():
        rows.append(
            {
                "key": _jsonable(key),
                "kwargs": _jsonable(getattr(config, "kwargs", {})),
                "num_warps": int(getattr(config, "num_warps", 0) or 0),
                "num_stages": int(getattr(config, "num_stages", 0) or 0),
            }
        )
    return rows


def _profiled_current_semantics_call(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    metadata: PackedH3Metadata,
    *,
    tau: float,
    thresh_type: str,
    forward_config: str | None,
    skip_full_prefix_blocks: bool = False,
    static_prefix_sink: bool = False,
    bitmask_exact_scheduler: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid = int(metadata.valid_length)
    prefix = int(metadata.prefix_len)
    arch, value_strides, active_tokens = triton_sm86._validate_inputs(
        q,
        k,
        v,
        thresh_type,
        prefix,
        0,
        allow_strided_value=not v.is_contiguous(),
        tokens=valid,
    )
    if arch != (8, 6):
        raise RuntimeError(f"phase profile requires SM86; got SM{arch[0]}{arch[1]}")
    scale = q.shape[-1] ** -0.5
    blocks = triton_sm86.triton.cdiv(active_tokens, triton_sm86.BLOCK_SIZE)
    sink_start_block, sink_end_block = triton_sm86._sink_block_range(active_tokens, 0, prefix)

    kc_vc, reduce_ms = _cuda_event_call(lambda: triton_sm86._reduce_kv(k, v, tokens=active_tokens))
    kc, vc = kc_vc
    if thresh_type == "exact":
        threshold, threshold_ms = _cuda_event_call(
            lambda: triton_sm86._compute_exact_threshold(q, kc, tau=tau, scale=scale, tokens=active_tokens)
        )
    else:
        threshold, threshold_ms = _cuda_event_call(
            lambda: triton_sm86._compute_diag_threshold(q, kc, tau=tau, scale=scale, tokens=active_tokens)
        )

    def allocate_and_zero() -> torch.Tensor:
        output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        if active_tokens < int(q.shape[1]):
            output[:, active_tokens:].zero_()
        return output

    output, allocation_padding_zero_ms = _cuda_event_call(allocate_and_zero)
    _none, forward_ms = _cuda_event_call(
        lambda: triton_sm86._launch_forward_ptr(
            q,
            k,
            v,
            kc,
            vc,
            threshold,
            output,
            scale=scale,
            active_tokens=active_tokens,
            sink_start_block=sink_start_block,
            sink_end_block=sink_end_block,
            prefix_exact_tokens=prefix if skip_full_prefix_blocks else 0,
            value_strides=value_strides,
            exact_prefix_query=False,
            skip_full_prefix_blocks=skip_full_prefix_blocks,
            static_prefix_sink=static_prefix_sink,
            forward_config=forward_config,
            bitmask_exact_scheduler=bitmask_exact_scheduler,
        )
    )

    if prefix > 0:
        def prefix_overwrite() -> None:
            output[:, :prefix] = dense_attention_reference(
                q[:, :prefix],
                k[:, :valid],
                v[:, :valid],
                softmax_scale=scale,
            )

        _none, prefix_ms = _cuda_event_call(prefix_overwrite)
    else:
        prefix_ms = 0.0
    phase_ms = {
        "reduce_kv": reduce_ms,
        "diag_threshold_preparation" if thresh_type == "diag" else "exact_threshold_preparation": threshold_ms,
        "output_allocation_padding_zero": allocation_padding_zero_ms,
        "forward_pointer_kernel": forward_ms,
        "prefix_dense_overwrite": prefix_ms,
    }
    phase_ms["total_profiled_sum"] = float(sum(phase_ms.values()))
    return output, phase_ms


def _phase_profile_distribution(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    metadata: PackedH3Metadata,
    *,
    warmup: int,
    repeats: int,
    tau: float,
    thresh_type: str,
    forward_config: str | None,
    skip_full_prefix_blocks: bool = False,
    static_prefix_sink: bool = False,
    bitmask_exact_scheduler: bool = False,
) -> dict[str, Any]:
    if warmup < 0 or repeats < 1:
        raise RuntimeError("phase-bench requires repeats >= 1 and warmup >= 0")
    for _ in range(warmup):
        out, _phase_ms = _profiled_current_semantics_call(
            q,
            k,
            v,
            metadata,
            tau=tau,
            thresh_type=thresh_type,
            forward_config=forward_config,
            skip_full_prefix_blocks=skip_full_prefix_blocks,
            static_prefix_sink=static_prefix_sink,
            bitmask_exact_scheduler=bitmask_exact_scheduler,
        )
        del out
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples_by_phase: dict[str, list[float]] = {}
    for _ in range(repeats):
        out, phase_ms = _profiled_current_semantics_call(
            q,
            k,
            v,
            metadata,
            tau=tau,
            thresh_type=thresh_type,
            forward_config=forward_config,
            skip_full_prefix_blocks=skip_full_prefix_blocks,
            static_prefix_sink=static_prefix_sink,
            bitmask_exact_scheduler=bitmask_exact_scheduler,
        )
        for key, value in phase_ms.items():
            samples_by_phase.setdefault(key, []).append(float(value))
        del out
    phase_stats = {key: _stats(values) for key, values in samples_by_phase.items()}
    medians = {key: float(value.get("median_ms", 0.0)) for key, value in phase_stats.items()}
    dominant = max((key for key in medians if key != "total_profiled_sum"), key=lambda key: medians[key])
    return {
        "warmup": int(warmup),
        "repeats": int(repeats),
        "forward_config": forward_config or "autotune_current",
        "skip_full_prefix_blocks": bool(skip_full_prefix_blocks),
        "static_prefix_sink": bool(static_prefix_sink),
        "bitmask_exact_scheduler": bool(bitmask_exact_scheduler),
        "phase_latency_ms": phase_stats,
        "dominant_phase_by_median": dominant,
        "dominant_phase_median_ms": medians[dominant],
        "peak_cuda_memory": _cuda_memory_snapshot(),
        "forward_autotune_cache": _forward_autotune_cache_snapshot(),
    }


def _time_materialize_distribution(v: torch.Tensor, *, warmup: int, repeats: int) -> dict[str, Any]:
    if warmup < 0 or repeats < 1:
        raise RuntimeError("large-bench materialization timing requires repeats >= 1 and warmup >= 0")
    bytes_per_copy = _copy_bytes_for(v)
    for _ in range(warmup):
        tmp = v.contiguous()
        torch.cuda.synchronize()
        del tmp
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        tmp = v.contiguous()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
        del tmp
    return {
        "warmup": int(warmup),
        "repeats": int(repeats),
        "bytes_per_copy": bytes_per_copy,
        "total_timed_bytes": bytes_per_copy * int(repeats),
        "latency_ms": _stats(times_ms),
        "peak_cuda_memory": _cuda_memory_snapshot(),
        "copy_kind": "explicit_v_contiguous_materialization_before_reference_kernel",
    }


def _cuda_memory_snapshot() -> dict[str, int]:
    return {
        "memory_allocated_bytes": int(torch.cuda.memory_allocated()),
        "memory_reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _summarize_telemetry(items: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "dense_calls",
        "sparse_candidate_calls",
        "sparse_calls",
        "fallback_calls",
        "prefix_query_dense_calls",
        "exact_prefix_query_calls",
        "materialize_copy_count",
        "materialize_copy_bytes",
        "materialize_gpu_copy_latency_ms",
        "materialize_gpu_timing_failures",
        "stride_aware_value_calls",
        "stride_aware_value_bytes",
    )
    totals: dict[str, Any] = {key: 0 for key in keys}
    decline_reasons: dict[str, int] = {}
    fallback_reasons: dict[str, int] = {}
    materialize_copy_by_tensor: dict[str, int] = {}
    layout_samples: list[dict[str, Any]] = []
    density_samples: list[dict[str, Any]] = []
    for item in items:
        for key in keys:
            totals[key] += item.get(key, 0) or 0
        for key, value in (item.get("decline_reasons") or {}).items():
            decline_reasons[key] = decline_reasons.get(key, 0) + int(value)
        for key, value in (item.get("fallback_reasons") or {}).items():
            fallback_reasons[key] = fallback_reasons.get(key, 0) + int(value)
        for key, value in (item.get("materialize_copy_by_tensor") or {}).items():
            materialize_copy_by_tensor[key] = materialize_copy_by_tensor.get(key, 0) + int(value)
        if not layout_samples and item.get("layout_samples"):
            layout_samples = item["layout_samples"]
        if not density_samples and item.get("density_samples"):
            density_samples = item["density_samples"]
    return {
        "n_calls_recorded": len(items),
        **totals,
        "decline_reasons": decline_reasons,
        "fallback_reasons": fallback_reasons,
        "materialize_copy_by_tensor": materialize_copy_by_tensor,
        "first_layout_samples": layout_samples,
        "first_density_samples": density_samples,
    }


def _prefix_dense_overwrite_call(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    prefix: int,
    valid: int,
) -> torch.Tensor:
    return dense_attention_packed_reference(
        q[:, : int(prefix)],
        k[:, : int(valid)],
        v[:, : int(valid)],
        metadata=None,
    )


def _large_output_sanity(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    prefix: int,
    valid: int,
    chunk_tokens: int = 1024,
) -> dict[str, Any]:
    torch.cuda.synchronize()
    max_abs = 0.0
    max_rel = 0.0
    candidate_finite = True
    reference_finite = True
    for start in range(0, int(valid), int(chunk_tokens)):
        end = min(int(valid), start + int(chunk_tokens))
        cand = candidate[:, start:end]
        ref = reference[:, start:end]
        candidate_finite = candidate_finite and bool(torch.isfinite(cand).all().item())
        reference_finite = reference_finite and bool(torch.isfinite(ref).all().item())
        diff = (cand.float() - ref.float()).abs()
        if diff.numel():
            max_abs = max(max_abs, float(diff.max().item()))
            denom = ref.float().abs().clamp_min(1.0e-7)
            max_rel = max(max_rel, float((diff / denom).max().item()))
        del cand, ref
    padding_candidate_zero = bool(torch.count_nonzero(candidate[:, int(valid) :]).item() == 0)
    padding_reference_zero = bool(torch.count_nonzero(reference[:, int(valid) :]).item() == 0)
    prefix_i = int(prefix)
    valid_i = int(valid)
    block = int(triton_sm86.BLOCK_SIZE)
    full_prefix_end = min(valid_i, (prefix_i // block) * block)
    mixed_start = full_prefix_end
    mixed_end = min(valid_i, mixed_start + block)
    prefix_diff = (candidate[:, :prefix_i].float() - reference[:, :prefix_i].float()).abs()
    prefix_equal = bool(torch.equal(candidate[:, :prefix_i], reference[:, :prefix_i]))
    full_prefix_equal = bool(torch.equal(candidate[:, :full_prefix_end], reference[:, :full_prefix_end]))
    mixed_boundary_equal = bool(torch.equal(candidate[:, mixed_start:mixed_end], reference[:, mixed_start:mixed_end]))
    mixed_tail_equal = bool(torch.equal(candidate[:, prefix_i:mixed_end], reference[:, prefix_i:mixed_end]))
    tail_equal = bool(torch.equal(candidate[:, prefix_i:valid_i], reference[:, prefix_i:valid_i]))
    full_equal = bool(max_abs == 0.0)
    return {
        "candidate_all_finite_valid": candidate_finite,
        "reference_all_finite_valid": reference_finite,
        "padding_rows_zero_candidate": padding_candidate_zero,
        "padding_rows_zero_reference": padding_reference_zero,
        "prefix_rows_equal_reference": prefix_equal,
        "full_prefix_block_rows_equal_reference": full_prefix_equal,
        "mixed_boundary_block_rows_equal_reference": mixed_boundary_equal,
        "mixed_boundary_tail_rows_equal_reference": mixed_tail_equal,
        "tail_rows_equal_reference": tail_equal,
        "mixed_boundary_has_prefix_and_tail": bool(prefix_i % block and prefix_i < mixed_end),
        "mixed_boundary_token_range": [int(mixed_start), int(mixed_end)],
        "full_prefix_token_range": [0, int(full_prefix_end)],
        "tail_token_range": [int(prefix_i), int(valid_i)],
        "candidate_reference_exact_equal_valid": full_equal,
        "max_abs_valid": max_abs,
        "max_abs_prefix": float(prefix_diff.max().item()) if prefix_diff.numel() else 0.0,
        "max_rel_valid": max_rel,
        "numel_valid": int(candidate[:, : int(valid)].numel()),
        "comparison_reference": "same_sm86_kernel_with_pre_materialized_contiguous_v",
    }


def _attempt_large_shape(
    device: torch.device,
    *,
    shape: dict[str, int],
    seed: int,
    warmup: int,
    repeats: int,
    tau: float,
    thresh_type: str,
    step_index: int,
    layer_index: int,
    target_total: int,
    target_valid: int,
) -> dict[str, Any]:
    total = int(shape["T_total"])
    valid = int(shape["T_valid"])
    heads = int(shape["H"])
    prefix = int(shape["prefix"])
    started = time.time()
    q = k = fused_qkv = v = v_ref = None
    current_out = exact_prefix_out = materialized_out = kernel_only_out = prefix_dense_out = None
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        q, k, fused_qkv, v, metadata = _large_case_tensors(
            device,
            total=total,
            valid=valid,
            heads=heads,
            prefix=prefix,
            seed=seed,
            target_total=target_total,
            target_valid=target_valid,
        )
        allocation_memory = _cuda_memory_snapshot()
        value_layout = _tensor_layout(v)
        fused_layout = _tensor_layout(fused_qkv)
        expected_materialize_bytes = _copy_bytes_for(v)

        materialize_timing = _time_materialize_distribution(v, warmup=warmup, repeats=repeats)
        v_ref = v.contiguous()
        torch.cuda.synchronize()

        pre_materialized_policy = _new_large_policy(
            stride_aware_value=False,
            tau=tau,
            thresh_type=thresh_type,
            prefix_query_dense=True,
            exact_prefix_query=False,
        )
        kernel_only_policy = _new_large_policy(
            stride_aware_value=True,
            tau=tau,
            thresh_type=thresh_type,
            prefix_query_dense=False,
            exact_prefix_query=False,
        )
        current_policy = _new_large_policy(
            stride_aware_value=True,
            tau=tau,
            thresh_type=thresh_type,
            prefix_query_dense=True,
            exact_prefix_query=False,
        )
        exact_prefix_policy = _new_large_policy(
            stride_aware_value=True,
            tau=tau,
            thresh_type=thresh_type,
            prefix_query_dense=True,
            exact_prefix_query=True,
        )

        materialized_out, materialized_preflight_telemetry = _checked_sparse_call(
            q,
            k,
            v_ref,
            metadata,
            policy=pre_materialized_policy,
            step_index=step_index,
            layer_index=layer_index,
        )
        kernel_only_out, kernel_only_preflight_telemetry = _checked_sparse_call(
            q,
            k,
            v,
            metadata,
            policy=kernel_only_policy,
            step_index=step_index,
            layer_index=layer_index,
        )
        current_out, current_preflight_telemetry = _checked_sparse_call(
            q,
            k,
            v,
            metadata,
            policy=current_policy,
            step_index=step_index,
            layer_index=layer_index,
        )
        exact_prefix_out, exact_prefix_preflight_telemetry = _checked_sparse_call(
            q,
            k,
            v,
            metadata,
            policy=exact_prefix_policy,
            step_index=step_index,
            layer_index=layer_index,
        )
        prefix_dense_out = _prefix_dense_overwrite_call(q, k, v, prefix=prefix, valid=valid)
        torch.cuda.synchronize()

        current_vs_materialized_sanity = _large_output_sanity(current_out, materialized_out, prefix=prefix, valid=valid)
        current_vs_materialized_sanity["comparison_reference"] = "same_policy_current_prefix_dense_overwrite_with_pre_materialized_contiguous_v"
        exact_prefix_sanity = _large_output_sanity(exact_prefix_out, current_out, prefix=prefix, valid=valid)
        exact_prefix_sanity["comparison_reference"] = "current_stride_aware_v_prefix_dense_overwrite_lane"
        kernel_only_sanity = _large_output_sanity(kernel_only_out, current_out, prefix=prefix, valid=valid)
        kernel_only_sanity["comparison_reference"] = "current_lane_expected_to_differ_on_prefix_queries_only"
        prefix_dense_diff_current = (current_out[:, :prefix].float() - prefix_dense_out.float()).abs()
        prefix_dense_diff_exact = (exact_prefix_out[:, :prefix].float() - prefix_dense_out.float()).abs()
        prefix_dense_checks = {
            "current_prefix_rows_equal_direct_dense": bool(torch.equal(current_out[:, :prefix], prefix_dense_out)),
            "exact_prefix_rows_equal_direct_dense": bool(torch.equal(exact_prefix_out[:, :prefix], prefix_dense_out)),
            "current_prefix_max_abs_vs_direct_dense": float(prefix_dense_diff_current.max().item()) if prefix_dense_diff_current.numel() else 0.0,
            "exact_prefix_max_abs_vs_direct_dense": float(prefix_dense_diff_exact.max().item()) if prefix_dense_diff_exact.numel() else 0.0,
            "prefix_rows": int(prefix),
            "valid_kv_rows": int(valid),
        }
        preflight_peak_memory = _cuda_memory_snapshot()

        del materialized_out, kernel_only_out, current_out, exact_prefix_out, prefix_dense_out
        materialized_out = kernel_only_out = current_out = exact_prefix_out = prefix_dense_out = None
        gc.collect()
        torch.cuda.synchronize()

        prefix_dense_timing = _time_cuda_tensor_distribution(
            lambda: _prefix_dense_overwrite_call(
                q,  # type: ignore[arg-type]
                k,  # type: ignore[arg-type]
                v,  # type: ignore[arg-type]
                prefix=prefix,
                valid=valid,
            ),
            warmup=warmup,
            repeats=repeats,
        )
        pre_materialized_timing = _time_cuda_distribution(
            lambda: _checked_sparse_call(
                q,
                k,
                v_ref,  # type: ignore[arg-type]
                metadata,
                policy=pre_materialized_policy,
                step_index=step_index,
                layer_index=layer_index,
            ),
            warmup=warmup,
            repeats=repeats,
        )
        kernel_only_timing = _time_cuda_distribution(
            lambda: _checked_sparse_call(
                q,
                k,
                v,  # type: ignore[arg-type]
                metadata,
                policy=kernel_only_policy,
                step_index=step_index,
                layer_index=layer_index,
            ),
            warmup=warmup,
            repeats=repeats,
        )
        current_timing = _time_cuda_distribution(
            lambda: _checked_sparse_call(
                q,
                k,
                v,  # type: ignore[arg-type]
                metadata,
                policy=current_policy,
                step_index=step_index,
                layer_index=layer_index,
            ),
            warmup=warmup,
            repeats=repeats,
        )
        exact_prefix_timing = _time_cuda_distribution(
            lambda: _checked_sparse_call(
                q,
                k,
                v,  # type: ignore[arg-type]
                metadata,
                policy=exact_prefix_policy,
                step_index=step_index,
                layer_index=layer_index,
            ),
            warmup=warmup,
            repeats=repeats,
        )
        final_peak_memory = _cuda_memory_snapshot()
        record: dict[str, Any] = {
            "status": "success",
            "shape": {"B": 1, "T_total": total, "T_valid": valid, "H": heads, "D": TARGET_H3_D, "prefix": prefix},
            "metadata": {
                "prefix_len": metadata.prefix_len,
                "latent_grid": list(metadata.latent_grid),
                "valid_length": metadata.valid_length,
                "total_length": metadata.total_length,
            },
            "upstream_semantics_grounding": {
                "pinned_sources": [
                    "upstreams/Sana-sol-engine/techniques/sparse_backends/sol_attn_backend.py",
                    "upstreams/Sana-sol-engine/models/minimax_h3/gb200/optimized/sol_attn_h3.py",
                ],
                "grounded_semantics": "Sol-Attn exact KV sink does not make prefix/text query rows dense; MMDiT/H3 integrations must compute valid prefix/text query rows with dense attention. This harness treats the existing prefix-dense-overwrite lane as the reference semantics.",
            },
            "value_layout": value_layout,
            "fused_qkv_layout": fused_layout,
            "expected_materialize_bytes_per_call": expected_materialize_bytes,
            "materialize_timing_ms": materialize_timing,
            "prefix_dense_overwrite_timing_ms": prefix_dense_timing,
            "pre_materialized_preflight_telemetry": _jsonable(materialized_preflight_telemetry.__dict__),
            "kernel_only_preflight_telemetry": _jsonable(kernel_only_preflight_telemetry.__dict__),
            "current_prefix_dense_preflight_telemetry": _jsonable(current_preflight_telemetry.__dict__),
            "exact_prefix_preflight_telemetry": _jsonable(exact_prefix_preflight_telemetry.__dict__),
            "pre_materialized_timing_ms": pre_materialized_timing,
            "kernel_only_no_prefix_overwrite_timing_ms": kernel_only_timing,
            "current_prefix_dense_overwrite_timing_ms": current_timing,
            "exact_prefix_query_timing_ms": exact_prefix_timing,
            "current_vs_materialized_sanity": current_vs_materialized_sanity,
            "exact_prefix_vs_current_sanity": exact_prefix_sanity,
            "kernel_only_vs_current_sanity": kernel_only_sanity,
            "prefix_dense_checks": prefix_dense_checks,
            "cuda_memory": {
                "after_input_allocation": allocation_memory,
                "after_preflight_sanity": preflight_peak_memory,
                "after_all_lanes": final_peak_memory,
            },
            "elapsed_s": time.time() - started,
        }
        record["route_decision"] = _large_route_decision(record, target_total=target_total, target_valid=target_valid)
        return record
    except Exception as exc:  # noqa: BLE001 - failures are the boundary evidence
        failure = {
            "status": "failed",
            "shape": {"B": 1, "T_total": total, "T_valid": valid, "H": heads, "D": TARGET_H3_D, "prefix": prefix},
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-12:],
            "cuda_memory_at_failure": _cuda_memory_snapshot() if torch.cuda.is_available() else None,
            "elapsed_s": time.time() - started,
        }
        return failure
    finally:
        del q, k, fused_qkv, v, v_ref, current_out, exact_prefix_out, materialized_out, kernel_only_out, prefix_dense_out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def _attempt_large_v_materialization_shape(
    device: torch.device,
    *,
    shape: dict[str, int],
    seed: int,
    warmup: int,
    repeats: int,
    tau: float,
    thresh_type: str,
    step_index: int,
    layer_index: int,
    target_total: int,
    target_valid: int,
) -> dict[str, Any]:
    """Primary large-shape ablation: no-copy stride-aware V vs pre-materialized V."""

    total = int(shape["T_total"])
    valid = int(shape["T_valid"])
    heads = int(shape["H"])
    prefix = int(shape["prefix"])
    started = time.time()
    q = k = fused_qkv = v = v_ref = None
    candidate_out = materialized_out = prefix_dense_out = None
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        q, k, fused_qkv, v, metadata = _large_case_tensors(
            device,
            total=total,
            valid=valid,
            heads=heads,
            prefix=prefix,
            seed=seed,
            target_total=target_total,
            target_valid=target_valid,
        )
        allocation_memory = _cuda_memory_snapshot()
        value_layout = _tensor_layout(v)
        fused_layout = _tensor_layout(fused_qkv)
        expected_materialize_bytes = _copy_bytes_for(v)

        materialize_timing = _time_materialize_distribution(v, warmup=warmup, repeats=repeats)
        v_ref = v.contiguous()
        torch.cuda.synchronize()

        candidate_policy = _new_large_policy(
            stride_aware_value=True,
            tau=tau,
            thresh_type=thresh_type,
            prefix_query_dense=True,
            exact_prefix_query=False,
            skip_full_prefix_blocks=False,
        )
        pre_materialized_policy = _new_large_policy(
            stride_aware_value=False,
            tau=tau,
            thresh_type=thresh_type,
            prefix_query_dense=True,
            exact_prefix_query=False,
            skip_full_prefix_blocks=False,
        )

        materialized_out, materialized_preflight_telemetry = _checked_sparse_call(
            q,
            k,
            v_ref,
            metadata,
            policy=pre_materialized_policy,
            step_index=step_index,
            layer_index=layer_index,
        )
        candidate_out, candidate_preflight_telemetry = _checked_sparse_call(
            q,
            k,
            v,
            metadata,
            policy=candidate_policy,
            step_index=step_index,
            layer_index=layer_index,
        )
        prefix_dense_out = _prefix_dense_overwrite_call(q, k, v, prefix=prefix, valid=valid)
        torch.cuda.synchronize()

        candidate_vs_materialized_sanity = _large_output_sanity(
            candidate_out,
            materialized_out,
            prefix=prefix,
            valid=valid,
        )
        candidate_vs_materialized_sanity[
            "comparison_reference"
        ] = "same_sm86_kernel_policy_with_pre_materialized_contiguous_v"
        prefix_dense_diff_candidate = (candidate_out[:, :prefix].float() - prefix_dense_out.float()).abs()
        materialized_prefix_diff = (materialized_out[:, :prefix].float() - prefix_dense_out.float()).abs()
        prefix_dense_checks = {
            "candidate_prefix_rows_equal_direct_dense": bool(torch.equal(candidate_out[:, :prefix], prefix_dense_out)),
            "pre_materialized_prefix_rows_equal_direct_dense": bool(
                torch.equal(materialized_out[:, :prefix], prefix_dense_out)
            ),
            "candidate_prefix_max_abs_vs_direct_dense": float(prefix_dense_diff_candidate.max().item())
            if prefix_dense_diff_candidate.numel()
            else 0.0,
            "pre_materialized_prefix_max_abs_vs_direct_dense": float(materialized_prefix_diff.max().item())
            if materialized_prefix_diff.numel()
            else 0.0,
            "prefix_rows": int(prefix),
            "valid_kv_rows": int(valid),
        }
        preflight_peak_memory = _cuda_memory_snapshot()

        del materialized_out, candidate_out, prefix_dense_out
        materialized_out = candidate_out = prefix_dense_out = None
        gc.collect()
        torch.cuda.synchronize()

        candidate_phase = _phase_profile_distribution(
            q,
            k,
            v,
            metadata,
            warmup=warmup,
            repeats=repeats,
            tau=tau,
            thresh_type=thresh_type,
            forward_config=None,
            skip_full_prefix_blocks=False,
        )
        pre_materialized_phase = _phase_profile_distribution(
            q,
            k,
            v_ref,
            metadata,
            warmup=warmup,
            repeats=repeats,
            tau=tau,
            thresh_type=thresh_type,
            forward_config=None,
            skip_full_prefix_blocks=False,
        )
        candidate_timing = _time_cuda_distribution(
            lambda: _checked_sparse_call(
                q,
                k,
                v,
                metadata,
                policy=candidate_policy,
                step_index=step_index,
                layer_index=layer_index,
            ),
            warmup=warmup,
            repeats=repeats,
        )
        pre_materialized_timing = _time_cuda_distribution(
            lambda: _checked_sparse_call(
                q,
                k,
                v_ref,
                metadata,
                policy=pre_materialized_policy,
                step_index=step_index,
                layer_index=layer_index,
            ),
            warmup=warmup,
            repeats=repeats,
        )
        final_peak_memory = _cuda_memory_snapshot()
        record: dict[str, Any] = {
            "status": "success",
            "shape": {"B": 1, "T_total": total, "T_valid": valid, "H": heads, "D": TARGET_H3_D, "prefix": prefix},
            "metadata": {
                "prefix_len": metadata.prefix_len,
                "latent_grid": list(metadata.latent_grid),
                "valid_length": metadata.valid_length,
                "total_length": metadata.total_length,
            },
            "upstream_semantics_grounding": {
                "pinned_sources": [
                    "upstreams/Sana-sol-engine/techniques/sparse_backends/sol_attn_backend.py",
                    "upstreams/Sana-sol-engine/models/minimax_h3/gb200/optimized/sol_attn_h3.py",
                    "ports/minimax_h3_a6000/src/minimax_h3_a6000/sol_attn_backend.py",
                    "ports/minimax_h3_a6000/src/minimax_h3_a6000/sol_attn_triton_sm86.py",
                ],
                "grounded_semantics": (
                    "Primary ablation fixes tau/routing, exact prefix KV sink, dense prefix-query overwrite, "
                    "valid-length padding, dense-first policy, and cache-off policy; only the value tensor layout "
                    "differs between the source-backed fused-QKV V view and the pre-materialized contiguous V reference."
                ),
            },
            "value_layout": value_layout,
            "fused_qkv_layout": fused_layout,
            "expected_materialize_bytes_per_call": expected_materialize_bytes,
            "r8_materialization_context": {
                "observed_sparse_calls_per_5step_run": TARGET_H3_R8_MATERIALIZE_CALLS,
                "observed_materialize_bytes_per_5step_run": TARGET_H3_R8_MATERIALIZE_BYTES_TOTAL,
                "observed_materialize_bytes_per_call": TARGET_H3_R8_MATERIALIZE_BYTES_TOTAL
                // TARGET_H3_R8_MATERIALIZE_CALLS,
            },
            "materialize_timing_ms": materialize_timing,
            "pre_materialized_preflight_telemetry": _jsonable(materialized_preflight_telemetry.__dict__),
            "candidate_preflight_telemetry": _jsonable(candidate_preflight_telemetry.__dict__),
            "candidate_phase_profile_ms": candidate_phase,
            "pre_materialized_phase_profile_ms": pre_materialized_phase,
            "candidate_kernel_timing_ms": candidate_timing,
            "pre_materialized_timing_ms": pre_materialized_timing,
            "candidate_vs_materialized_sanity": candidate_vs_materialized_sanity,
            "prefix_dense_checks": prefix_dense_checks,
            "cuda_memory": {
                "after_input_allocation": allocation_memory,
                "after_preflight_sanity": preflight_peak_memory,
                "after_all_lanes": final_peak_memory,
            },
            "elapsed_s": time.time() - started,
        }
        record["route_decision"] = _v_materialization_route_decision(
            record,
            target_total=target_total,
            target_valid=target_valid,
        )
        return record
    except Exception as exc:  # noqa: BLE001 - failures are the boundary evidence
        return {
            "status": "failed",
            "shape": {"B": 1, "T_total": total, "T_valid": valid, "H": heads, "D": TARGET_H3_D, "prefix": prefix},
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-12:],
            "cuda_memory_at_failure": _cuda_memory_snapshot() if torch.cuda.is_available() else None,
            "elapsed_s": time.time() - started,
        }
    finally:
        del q, k, fused_qkv, v, v_ref, candidate_out, materialized_out, prefix_dense_out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def _phase_medians(record: dict[str, Any], key: str) -> dict[str, float]:
    phases = record.get(key, {}).get("phase_latency_ms", {})
    return {name: float(stats.get("median_ms", 0.0)) for name, stats in phases.items() if isinstance(stats, dict)}


def _v_materialization_route_decision(
    record: dict[str, Any],
    *,
    target_total: int,
    target_valid: int,
    min_gain_pct: float = 0.5,
) -> dict[str, Any]:
    shape = record["shape"]
    full_shape = int(shape["T_total"]) == int(target_total) and int(shape["T_valid"]) == int(target_valid)
    candidate_pre = record["candidate_preflight_telemetry"]
    materialized_pre = record["pre_materialized_preflight_telemetry"]
    candidate_summary = record["candidate_kernel_timing_ms"]["timed_telemetry_summary"]
    materialized_summary = record["pre_materialized_timing_ms"]["timed_telemetry_summary"]
    repeats = int(record["candidate_kernel_timing_ms"]["repeats"])
    candidate_latency = record["candidate_kernel_timing_ms"]["latency_ms"]
    materialized_latency = record["pre_materialized_timing_ms"]["latency_ms"]
    materialize_latency = record["materialize_timing_ms"]["latency_ms"]
    candidate_median = float(candidate_latency["median_ms"])
    materialized_median = float(materialized_latency["median_ms"])
    materialize_median = float(materialize_latency["median_ms"])
    reference_copy_inclusive_median = materialized_median + materialize_median
    saved_ms = reference_copy_inclusive_median - candidate_median
    copy_inclusive_gain_pct = (
        100.0 * saved_ms / reference_copy_inclusive_median if reference_copy_inclusive_median else 0.0
    )
    candidate_phase_medians = _phase_medians(record, "candidate_phase_profile_ms")
    materialized_phase_medians = _phase_medians(record, "pre_materialized_phase_profile_ms")
    dominant_phase = record.get("candidate_phase_profile_ms", {}).get("dominant_phase_by_median")
    dominant_phase_median = float(record.get("candidate_phase_profile_ms", {}).get("dominant_phase_median_ms", 0.0))
    total_profiled = float(candidate_phase_medians.get("total_profiled_sum", 0.0))
    dominant_phase_share = dominant_phase_median / total_profiled if total_profiled else 0.0
    sanity = record["candidate_vs_materialized_sanity"]
    expected_per_call = int(record["expected_materialize_bytes_per_call"])
    r8_context = record["r8_materialization_context"]
    observed_r8_per_call = int(r8_context["observed_materialize_bytes_per_call"])
    value_layout = record["value_layout"]
    expected_stride = [TARGET_H3_TOTAL * 3 * TARGET_H3_HEADS * TARGET_H3_D, 3 * TARGET_H3_HEADS * TARGET_H3_D, TARGET_H3_D, 1]
    gates = {
        "full_target_shape": full_shape,
        "value_layout_matches_observed_r8_shape_stride_offset": bool(
            value_layout.get("shape") == [1, TARGET_H3_TOTAL, TARGET_H3_HEADS, TARGET_H3_D]
            and value_layout.get("stride") == expected_stride
            and int(value_layout.get("storage_offset", -1)) == 2 * TARGET_H3_HEADS * TARGET_H3_D
            and value_layout.get("is_contiguous") is False
        ),
        "materialization_bytes_match_r8_per_call_shape": expected_per_call == observed_r8_per_call,
        "materialization_time_measured": materialize_median > 0.0,
        "candidate_preflight_sparse_once": int(candidate_pre.get("sparse_calls", 0)) == 1,
        "materialized_preflight_sparse_once": int(materialized_pre.get("sparse_calls", 0)) == 1,
        "candidate_preflight_zero_fallback": int(candidate_pre.get("fallback_calls", -1)) == 0,
        "materialized_preflight_zero_fallback": int(materialized_pre.get("fallback_calls", -1)) == 0,
        "candidate_preflight_zero_materialization": int(candidate_pre.get("materialize_copy_count", -1)) == 0
        and int(candidate_pre.get("materialize_copy_bytes", -1)) == 0,
        "materialized_preflight_zero_materialization": int(materialized_pre.get("materialize_copy_count", -1)) == 0
        and int(materialized_pre.get("materialize_copy_bytes", -1)) == 0,
        "candidate_timed_zero_fallback": int(candidate_summary.get("fallback_calls", -1)) == 0,
        "materialized_timed_zero_fallback": int(materialized_summary.get("fallback_calls", -1)) == 0,
        "candidate_timed_zero_materialization": int(candidate_summary.get("materialize_copy_count", -1)) == 0
        and int(candidate_summary.get("materialize_copy_bytes", -1)) == 0,
        "materialized_timed_zero_materialization": int(materialized_summary.get("materialize_copy_count", -1)) == 0
        and int(materialized_summary.get("materialize_copy_bytes", -1)) == 0,
        "candidate_stride_aware_calls_match_repeats": int(candidate_summary.get("stride_aware_value_calls", 0)) == repeats,
        "candidate_stride_aware_bytes_match_repeats": int(candidate_summary.get("stride_aware_value_bytes", 0))
        == expected_per_call * repeats,
        "materialized_stride_aware_calls_zero": int(materialized_summary.get("stride_aware_value_calls", -1)) == 0,
        "candidate_prefix_dense_calls_match_repeats": int(candidate_summary.get("prefix_query_dense_calls", 0)) == repeats,
        "materialized_prefix_dense_calls_match_repeats": int(materialized_summary.get("prefix_query_dense_calls", 0))
        == repeats,
        "candidate_exact_prefix_query_calls_zero": int(candidate_summary.get("exact_prefix_query_calls", -1)) == 0,
        "materialized_exact_prefix_query_calls_zero": int(materialized_summary.get("exact_prefix_query_calls", -1)) == 0,
        "candidate_all_finite_valid": bool(sanity.get("candidate_all_finite_valid")),
        "reference_all_finite_valid": bool(sanity.get("reference_all_finite_valid")),
        "candidate_matches_pre_materialized_valid_exact": bool(sanity.get("candidate_reference_exact_equal_valid")),
        "candidate_prefix_rows_equal_pre_materialized": bool(sanity.get("prefix_rows_equal_reference")),
        "candidate_tail_rows_equal_pre_materialized": bool(sanity.get("tail_rows_equal_reference")),
        "candidate_padding_zero": bool(sanity.get("padding_rows_zero_candidate")),
        "reference_padding_zero": bool(sanity.get("padding_rows_zero_reference")),
        "copy_inclusive_benefit_or_clear_next_bottleneck": bool(
            copy_inclusive_gain_pct >= float(min_gain_pct) or dominant_phase_share >= 0.75
        ),
    }
    failed = [key for key, value in gates.items() if not value]
    correctness_gate_names = {
        "full_target_shape",
        "value_layout_matches_observed_r8_shape_stride_offset",
        "materialization_bytes_match_r8_per_call_shape",
        "materialization_time_measured",
        "candidate_preflight_sparse_once",
        "materialized_preflight_sparse_once",
        "candidate_preflight_zero_fallback",
        "materialized_preflight_zero_fallback",
        "candidate_preflight_zero_materialization",
        "materialized_preflight_zero_materialization",
        "candidate_timed_zero_fallback",
        "materialized_timed_zero_fallback",
        "candidate_timed_zero_materialization",
        "materialized_timed_zero_materialization",
        "candidate_stride_aware_calls_match_repeats",
        "candidate_stride_aware_bytes_match_repeats",
        "materialized_stride_aware_calls_zero",
        "candidate_prefix_dense_calls_match_repeats",
        "materialized_prefix_dense_calls_match_repeats",
        "candidate_exact_prefix_query_calls_zero",
        "materialized_exact_prefix_query_calls_zero",
        "candidate_all_finite_valid",
        "reference_all_finite_valid",
        "candidate_matches_pre_materialized_valid_exact",
        "candidate_prefix_rows_equal_pre_materialized",
        "candidate_tail_rows_equal_pre_materialized",
        "candidate_padding_zero",
        "reference_padding_zero",
    }
    failed_correctness = [key for key in failed if key in correctness_gate_names]
    if failed_correctness:
        decision = "reject_stride_aware_v_route_gate_failure"
    elif copy_inclusive_gain_pct >= float(min_gain_pct):
        decision = "retain_stride_aware_v_route_for_next_real_chain_gate"
    elif dominant_phase_share >= 0.75:
        decision = "retain_stride_aware_v_route_but_next_bottleneck_is_forward_kernel"
    else:
        decision = "reject_stride_aware_v_route_no_meaningful_copy_inclusive_benefit"
    projected_materialize_ms_192 = materialize_median * TARGET_H3_R8_MATERIALIZE_CALLS
    projected_saved_ms_192 = saved_ms * TARGET_H3_R8_MATERIALIZE_CALLS
    return {
        "decision": decision,
        "not_h3_e2e": True,
        "not_long_video": True,
        "not_bf16_fidelity": True,
        "not_quality_or_product_speedup": True,
        "gates": gates,
        "failed_gates": failed,
        "median_ms": {
            "candidate_stride_aware_no_copy_kernel": candidate_median,
            "pre_materialized_contiguous_v_reference_kernel": materialized_median,
            "explicit_v_materialization": materialize_median,
            "reference_kernel_plus_materialization_sum": reference_copy_inclusive_median,
            "copy_inclusive_saved_ms": saved_ms,
            "copy_inclusive_gain_pct": copy_inclusive_gain_pct,
            "kernel_only_delta_candidate_minus_reference": candidate_median - materialized_median,
        },
        "phase_medians_ms": {
            "candidate_stride_aware_no_copy": candidate_phase_medians,
            "pre_materialized_contiguous_v_reference": materialized_phase_medians,
            "candidate_dominant_phase": dominant_phase,
            "candidate_dominant_phase_share": dominant_phase_share,
        },
        "projected_r8_192_call_materialization_context": {
            "observed_materialization_calls_per_5step_run": TARGET_H3_R8_MATERIALIZE_CALLS,
            "observed_materialization_bytes_per_5step_run": TARGET_H3_R8_MATERIALIZE_BYTES_TOTAL,
            "materialization_ms_if_repeated_192x_at_measured_median": projected_materialize_ms_192,
            "copy_inclusive_saved_ms_if_repeated_192x_at_measured_median": projected_saved_ms_192,
            "projection_boundary": "linear context only from per-call synthetic kernel timing; not an H3 E2E speedup claim",
        },
        "principal_variable": (
            "source-backed fused-QKV stride-aware V view without v.contiguous() versus the same SM86 "
            "policy fed by pre-materialized contiguous V; tau/routing/prefix overwrite/padding/cache semantics fixed"
        ),
        "claim_boundary": "Synthetic/model-free Sol-Attn kernel evidence only; no H3 E2E, long-video, BF16-fidelity, quality, normal-PC, product, or SOTA claim.",
    }


def _large_route_decision(record: dict[str, Any], *, target_total: int, target_valid: int) -> dict[str, Any]:
    shape = record["shape"]
    full_shape = int(shape["T_total"]) == int(target_total) and int(shape["T_valid"]) == int(target_valid)
    current_pre = record["current_prefix_dense_preflight_telemetry"]
    exact_pre = record["exact_prefix_preflight_telemetry"]
    materialized_pre = record["pre_materialized_preflight_telemetry"]
    kernel_only_pre = record["kernel_only_preflight_telemetry"]
    current_summary = record["current_prefix_dense_overwrite_timing_ms"]["timed_telemetry_summary"]
    exact_summary = record["exact_prefix_query_timing_ms"]["timed_telemetry_summary"]
    kernel_only_summary = record["kernel_only_no_prefix_overwrite_timing_ms"]["timed_telemetry_summary"]
    materialized_summary = record["pre_materialized_timing_ms"]["timed_telemetry_summary"]
    repeats = int(record["exact_prefix_query_timing_ms"]["repeats"])
    current_latency = record["current_prefix_dense_overwrite_timing_ms"]["latency_ms"]
    exact_latency = record["exact_prefix_query_timing_ms"]["latency_ms"]
    kernel_only_latency = record["kernel_only_no_prefix_overwrite_timing_ms"]["latency_ms"]
    prefix_dense_latency = record["prefix_dense_overwrite_timing_ms"]["latency_ms"]
    materialize_latency = record["materialize_timing_ms"]["latency_ms"]
    materialized_latency = record["pre_materialized_timing_ms"]["latency_ms"]
    current_median = float(current_latency["median_ms"])
    exact_median = float(exact_latency["median_ms"])
    kernel_only_median = float(kernel_only_latency["median_ms"])
    prefix_dense_median = float(prefix_dense_latency["median_ms"])
    observed_prefix_delta = max(0.0, current_median - kernel_only_median)
    material_overhead_threshold_ms = max(1.0, current_median * 0.01)
    prefix_material_overhead = prefix_dense_median >= material_overhead_threshold_ms or observed_prefix_delta >= material_overhead_threshold_ms

    gates = {
        "full_target_shape": full_shape,
        "current_preflight_sparse_once": int(current_pre.get("sparse_calls", 0)) == 1,
        "exact_preflight_sparse_once": int(exact_pre.get("sparse_calls", 0)) == 1,
        "kernel_only_preflight_sparse_once": int(kernel_only_pre.get("sparse_calls", 0)) == 1,
        "materialized_preflight_sparse_once": int(materialized_pre.get("sparse_calls", 0)) == 1,
        "current_preflight_zero_fallback": int(current_pre.get("fallback_calls", -1)) == 0,
        "exact_preflight_zero_fallback": int(exact_pre.get("fallback_calls", -1)) == 0,
        "kernel_only_preflight_zero_fallback": int(kernel_only_pre.get("fallback_calls", -1)) == 0,
        "current_preflight_zero_materialization": int(current_pre.get("materialize_copy_count", -1)) == 0
        and int(current_pre.get("materialize_copy_bytes", -1)) == 0,
        "exact_preflight_zero_materialization": int(exact_pre.get("materialize_copy_count", -1)) == 0
        and int(exact_pre.get("materialize_copy_bytes", -1)) == 0,
        "current_prefix_dense_path_exercised": int(current_pre.get("prefix_query_dense_calls", 0)) == 1,
        "exact_prefix_path_exercised": int(exact_pre.get("exact_prefix_query_calls", 0)) == 1
        and int(exact_pre.get("prefix_query_dense_calls", -1)) == 0,
        "current_timed_zero_fallback": int(current_summary.get("fallback_calls", -1)) == 0,
        "exact_timed_zero_fallback": int(exact_summary.get("fallback_calls", -1)) == 0,
        "kernel_only_timed_zero_fallback": int(kernel_only_summary.get("fallback_calls", -1)) == 0,
        "materialized_timed_zero_fallback": int(materialized_summary.get("fallback_calls", -1)) == 0,
        "current_timed_zero_materialization": int(current_summary.get("materialize_copy_count", -1)) == 0
        and int(current_summary.get("materialize_copy_bytes", -1)) == 0,
        "exact_timed_zero_materialization": int(exact_summary.get("materialize_copy_count", -1)) == 0
        and int(exact_summary.get("materialize_copy_bytes", -1)) == 0,
        "exact_timed_exact_prefix_calls_match_repeats": int(exact_summary.get("exact_prefix_query_calls", 0)) == repeats,
        "exact_timed_prefix_dense_overwrite_skipped": int(exact_summary.get("prefix_query_dense_calls", -1)) == 0,
        "current_timed_prefix_dense_calls_match_repeats": int(current_summary.get("prefix_query_dense_calls", 0)) == repeats,
        "candidate_stride_aware_calls_match_repeats": int(exact_summary.get("stride_aware_value_calls", 0)) == repeats,
        "materialized_stride_aware_calls_zero": int(materialized_summary.get("stride_aware_value_calls", -1)) == 0,
        "materialization_bytes_match_r8_per_call": int(record["materialize_timing_ms"].get("bytes_per_copy", -1))
        == int(record["expected_materialize_bytes_per_call"]),
        "materialization_time_measured": float(materialize_latency.get("median_ms", 0.0)) > 0.0,
        "current_matches_pre_materialized_valid": bool(record["current_vs_materialized_sanity"].get("candidate_reference_exact_equal_valid")),
        "current_prefix_rows_equal_direct_dense": bool(record["prefix_dense_checks"].get("current_prefix_rows_equal_direct_dense")),
        "exact_candidate_matches_current_valid": bool(record["exact_prefix_vs_current_sanity"].get("candidate_reference_exact_equal_valid")),
        "exact_candidate_prefix_rows_equal_current": bool(record["exact_prefix_vs_current_sanity"].get("prefix_rows_equal_reference")),
        "exact_candidate_padding_zero": bool(record["exact_prefix_vs_current_sanity"].get("padding_rows_zero_candidate")),
        "current_padding_zero": bool(record["exact_prefix_vs_current_sanity"].get("padding_rows_zero_reference")),
        "prefix_path_material_overhead": prefix_material_overhead,
        "exact_prefix_median_beats_current_prefix_dense_median": exact_median < current_median,
    }
    failed = [key for key, value in gates.items() if not value]
    if failed:
        if any(key in failed for key in ("exact_candidate_matches_current_valid", "exact_candidate_prefix_rows_equal_current")):
            decision = "reject_exact_prefix_query_route_semantic_mismatch"
        elif "exact_prefix_median_beats_current_prefix_dense_median" in failed:
            decision = "reject_exact_prefix_query_route_slower_or_no_median_gain"
        elif "prefix_path_material_overhead" in failed:
            decision = "reject_exact_prefix_query_route_prefix_overwrite_not_material"
        elif failed == ["full_target_shape"]:
            decision = "continue_scaled_only_full_shape_infeasible"
        else:
            decision = "reject_exact_prefix_query_route_gate_failure"
    else:
        decision = "retain_default_off_exact_prefix_query_candidate_for_next_real_chain_gate"
    return {
        "decision": decision,
        "not_h3_e2e": True,
        "not_long_video": True,
        "not_bf16_fidelity": True,
        "not_quality_or_product_speedup": True,
        "gates": gates,
        "failed_gates": failed,
        "median_ms": {
            "current_prefix_dense_overwrite_total": current_median,
            "kernel_only_no_prefix_overwrite": kernel_only_median,
            "direct_prefix_dense_overwrite": prefix_dense_median,
            "observed_current_minus_kernel_only_delta": observed_prefix_delta,
            "exact_prefix_query_total": exact_median,
            "pre_materialized_contiguous_v_total": float(materialized_latency["median_ms"]),
            "explicit_v_materialization": float(materialize_latency["median_ms"]),
        },
        "exact_prefix_over_current_ratio": (exact_median / current_median) if current_median else None,
        "current_over_exact_prefix_ratio": (current_median / exact_median) if exact_median else None,
        "prefix_material_overhead_threshold_ms": material_overhead_threshold_ms,
        "principal_variable": "default-off exact-prefix-query Sol-Attn kernel routing versus the current prefix-dense-overwrite wrapper lane on the same synthetic r8 shape",
        "claim_boundary": "Synthetic/model-free Sol-Attn kernel evidence only; no H3 E2E, long-video, BF16-fidelity, quality, normal-PC, product, or SOTA claim.",
    }

def _phase_route_decision(record: dict[str, Any], *, min_gain_pct: float) -> dict[str, Any]:
    current_pre = record["current_preflight_telemetry"]
    candidate_pre = record["candidate_preflight_telemetry"]
    materialized_pre = record["pre_materialized_preflight_telemetry"]
    current_summary = record["current_timing_ms"]["timed_telemetry_summary"]
    candidate_summary = record["candidate_timing_ms"]["timed_telemetry_summary"]
    materialized_summary = record["pre_materialized_timing_ms"]["timed_telemetry_summary"]
    current_latency = record["current_timing_ms"]["latency_ms"]
    candidate_latency = record["candidate_timing_ms"]["latency_ms"]
    current_median = float(current_latency["median_ms"])
    candidate_median = float(candidate_latency["median_ms"])
    improvement_pct = 100.0 * (current_median - candidate_median) / current_median if current_median else 0.0
    candidate_vs_current = record["candidate_vs_current_sanity"]
    current_vs_materialized = record["current_vs_materialized_sanity"]
    current_phase = record["current_phase_profile_ms"]
    gates = {
        "full_target_shape": int(record["shape"]["T_total"]) == TARGET_H3_TOTAL
        and int(record["shape"]["T_valid"]) == TARGET_H3_VALID,
        "current_preflight_sparse_once": int(current_pre.get("sparse_calls", 0)) == 1,
        "candidate_preflight_sparse_once": int(candidate_pre.get("sparse_calls", 0)) == 1,
        "materialized_preflight_sparse_once": int(materialized_pre.get("sparse_calls", 0)) == 1,
        "current_preflight_zero_fallback": int(current_pre.get("fallback_calls", -1)) == 0,
        "candidate_preflight_zero_fallback": int(candidate_pre.get("fallback_calls", -1)) == 0,
        "materialized_preflight_zero_fallback": int(materialized_pre.get("fallback_calls", -1)) == 0,
        "current_preflight_zero_unintended_materialization": int(current_pre.get("materialize_copy_count", -1)) == 0
        and int(current_pre.get("materialize_copy_bytes", -1)) == 0,
        "candidate_preflight_zero_unintended_materialization": int(candidate_pre.get("materialize_copy_count", -1)) == 0
        and int(candidate_pre.get("materialize_copy_bytes", -1)) == 0,
        "current_timed_zero_fallback": int(current_summary.get("fallback_calls", -1)) == 0,
        "candidate_timed_zero_fallback": int(candidate_summary.get("fallback_calls", -1)) == 0,
        "materialized_timed_zero_fallback": int(materialized_summary.get("fallback_calls", -1)) == 0,
        "current_timed_zero_unintended_materialization": int(current_summary.get("materialize_copy_count", -1)) == 0
        and int(current_summary.get("materialize_copy_bytes", -1)) == 0,
        "candidate_timed_zero_unintended_materialization": int(candidate_summary.get("materialize_copy_count", -1)) == 0
        and int(candidate_summary.get("materialize_copy_bytes", -1)) == 0,
        "current_matches_pre_materialized_valid": bool(current_vs_materialized.get("candidate_reference_exact_equal_valid")),
        "current_prefix_rows_equal_direct_dense": bool(record["prefix_dense_checks"].get("current_prefix_rows_equal_direct_dense")),
        "current_padding_zero": bool(current_vs_materialized.get("padding_rows_zero_candidate")),
        "candidate_matches_current_valid": bool(candidate_vs_current.get("candidate_reference_exact_equal_valid")),
        "candidate_prefix_rows_equal_current": bool(candidate_vs_current.get("prefix_rows_equal_reference")),
        "candidate_padding_zero": bool(candidate_vs_current.get("padding_rows_zero_candidate")),
        "phase_profile_supports_forward_launch_config_test": current_phase.get("dominant_phase_by_median") == "forward_pointer_kernel",
        "candidate_median_improves_by_min_gain": improvement_pct >= float(min_gain_pct),
    }
    failed = [key for key, value in gates.items() if not value]
    if failed:
        if any(key in failed for key in ("candidate_matches_current_valid", "candidate_prefix_rows_equal_current", "candidate_padding_zero")):
            decision = "reject_forward_config_candidate_semantic_mismatch"
        elif "phase_profile_supports_forward_launch_config_test" in failed:
            decision = "reject_forward_config_candidate_not_supported_by_phase_profile"
        elif "candidate_median_improves_by_min_gain" in failed:
            decision = "reject_forward_config_candidate_no_meaningful_median_gain"
        else:
            decision = "reject_forward_config_candidate_gate_failure"
    else:
        decision = "retain_forward_config_candidate_for_next_real_chain_gate"
    return {
        "decision": decision,
        "not_h3_e2e": True,
        "not_long_video": True,
        "not_bf16_fidelity": True,
        "not_quality_or_product_speedup": True,
        "gates": gates,
        "failed_gates": failed,
        "median_ms": {
            "current_prefix_dense_overwrite_total": current_median,
            "candidate_prefix_dense_overwrite_total": candidate_median,
            "pre_materialized_contiguous_v_total": float(record["pre_materialized_timing_ms"]["latency_ms"]["median_ms"]),
            "explicit_v_materialization": float(record["materialize_timing_ms"]["latency_ms"]["median_ms"]),
            "current_profiled_forward_pointer_kernel": float(
                current_phase["phase_latency_ms"]["forward_pointer_kernel"].get("median_ms", 0.0)
            ),
            "current_profiled_reduce_kv": float(current_phase["phase_latency_ms"]["reduce_kv"].get("median_ms", 0.0)),
            "current_profiled_diag_threshold_preparation": float(
                current_phase["phase_latency_ms"].get("diag_threshold_preparation", {}).get("median_ms", 0.0)
            ),
            "current_profiled_output_allocation_padding_zero": float(
                current_phase["phase_latency_ms"]["output_allocation_padding_zero"].get("median_ms", 0.0)
            ),
            "current_profiled_prefix_dense_overwrite": float(
                current_phase["phase_latency_ms"]["prefix_dense_overwrite"].get("median_ms", 0.0)
            ),
        },
        "candidate_over_current_ratio": (candidate_median / current_median) if current_median else None,
        "candidate_improvement_pct": improvement_pct,
        "min_required_gain_pct": float(min_gain_pct),
        "principal_variable": (
            "fixed Triton forward pointer launch config "
            f"{record['candidate_forward_config']!r} versus the current autotuned forward launch, with identical tau, routing, sinks, "
            "stride-aware V, valid length, padding, and dense-prefix-overwrite semantics"
        ),
        "claim_boundary": "Synthetic/model-free Sol-Attn kernel evidence only; no H3 E2E, long-video, BF16-fidelity, quality, normal-PC, product, or SOTA claim.",
    }


def _density_sample_flag(summary_or_telemetry: dict[str, Any], key: str) -> bool:
    samples = summary_or_telemetry.get("first_density_samples") or summary_or_telemetry.get("density_samples") or []
    return any(bool(sample.get(key)) for sample in samples if isinstance(sample, dict))


def _density_sample_int(summary_or_telemetry: dict[str, Any], key: str) -> int | None:
    samples = summary_or_telemetry.get("first_density_samples") or summary_or_telemetry.get("density_samples") or []
    for sample in samples:
        if isinstance(sample, dict) and key in sample:
            try:
                return int(sample[key])
            except (TypeError, ValueError):
                return None
    return None


def _prefix_skip_route_decision(record: dict[str, Any], *, min_gain_pct: float) -> dict[str, Any]:
    current_pre = record["current_preflight_telemetry"]
    candidate_pre = record["candidate_preflight_telemetry"]
    materialized_pre = record["pre_materialized_preflight_telemetry"]
    current_summary = record["current_timing_ms"]["timed_telemetry_summary"]
    candidate_summary = record["candidate_timing_ms"]["timed_telemetry_summary"]
    materialized_summary = record["pre_materialized_timing_ms"]["timed_telemetry_summary"]
    current_latency = record["current_timing_ms"]["latency_ms"]
    candidate_latency = record["candidate_timing_ms"]["latency_ms"]
    current_median = float(current_latency["median_ms"])
    candidate_median = float(candidate_latency["median_ms"])
    improvement_pct = 100.0 * (current_median - candidate_median) / current_median if current_median else 0.0
    current_vs_materialized = record["current_vs_materialized_sanity"]
    candidate_vs_current = record["candidate_vs_current_sanity"]
    current_phase = record["current_phase_profile_ms"]
    candidate_phase = record["candidate_phase_profile_ms"]
    expected_skipped_blocks = int(record["shape"]["prefix"]) // int(triton_sm86.BLOCK_SIZE)
    preflight_skipped_blocks = _density_sample_int(candidate_pre, "skipped_full_prefix_query_blocks_estimate")
    timed_skipped_blocks = _density_sample_int(candidate_summary, "skipped_full_prefix_query_blocks_estimate")
    gates = {
        "full_target_shape": int(record["shape"]["T_total"]) == TARGET_H3_TOTAL
        and int(record["shape"]["T_valid"]) == TARGET_H3_VALID,
        "current_preflight_sparse_once": int(current_pre.get("sparse_calls", 0)) == 1,
        "candidate_preflight_sparse_once": int(candidate_pre.get("sparse_calls", 0)) == 1,
        "materialized_preflight_sparse_once": int(materialized_pre.get("sparse_calls", 0)) == 1,
        "current_preflight_zero_fallback": int(current_pre.get("fallback_calls", -1)) == 0,
        "candidate_preflight_zero_fallback": int(candidate_pre.get("fallback_calls", -1)) == 0,
        "materialized_preflight_zero_fallback": int(materialized_pre.get("fallback_calls", -1)) == 0,
        "current_preflight_zero_materialization": int(current_pre.get("materialize_copy_count", -1)) == 0
        and int(current_pre.get("materialize_copy_bytes", -1)) == 0,
        "candidate_preflight_zero_materialization": int(candidate_pre.get("materialize_copy_count", -1)) == 0
        and int(candidate_pre.get("materialize_copy_bytes", -1)) == 0,
        "current_timed_zero_fallback": int(current_summary.get("fallback_calls", -1)) == 0,
        "candidate_timed_zero_fallback": int(candidate_summary.get("fallback_calls", -1)) == 0,
        "materialized_timed_zero_fallback": int(materialized_summary.get("fallback_calls", -1)) == 0,
        "current_timed_zero_materialization": int(current_summary.get("materialize_copy_count", -1)) == 0
        and int(current_summary.get("materialize_copy_bytes", -1)) == 0,
        "candidate_timed_zero_materialization": int(candidate_summary.get("materialize_copy_count", -1)) == 0
        and int(candidate_summary.get("materialize_copy_bytes", -1)) == 0,
        "candidate_preflight_prefix_dense_path_exercised": int(candidate_pre.get("prefix_query_dense_calls", 0)) == 1,
        "candidate_preflight_exact_prefix_not_used": int(candidate_pre.get("exact_prefix_query_calls", -1)) == 0,
        "candidate_timed_prefix_dense_calls_match_repeats": int(candidate_summary.get("prefix_query_dense_calls", 0))
        == int(record["candidate_timing_ms"]["repeats"]),
        "candidate_timed_exact_prefix_calls_zero": int(candidate_summary.get("exact_prefix_query_calls", -1)) == 0,
        "candidate_stride_aware_calls_match_repeats": int(candidate_summary.get("stride_aware_value_calls", 0))
        == int(record["candidate_timing_ms"]["repeats"]),
        "materialized_stride_aware_calls_zero": int(materialized_summary.get("stride_aware_value_calls", -1)) == 0,
        "candidate_preflight_skip_marker_present": _density_sample_flag(candidate_pre, "skip_full_prefix_blocks"),
        "candidate_timed_skip_marker_present": _density_sample_flag(candidate_summary, "skip_full_prefix_blocks"),
        "candidate_preflight_skipped_block_count_matches_prefix_floor": preflight_skipped_blocks == expected_skipped_blocks,
        "candidate_timed_skipped_block_count_matches_prefix_floor": timed_skipped_blocks == expected_skipped_blocks,
        "current_skip_marker_absent": not _density_sample_flag(current_pre, "skip_full_prefix_blocks")
        and not _density_sample_flag(current_summary, "skip_full_prefix_blocks"),
        "current_matches_pre_materialized_valid": bool(current_vs_materialized.get("candidate_reference_exact_equal_valid")),
        "current_prefix_rows_equal_direct_dense": bool(record["prefix_dense_checks"].get("current_prefix_rows_equal_direct_dense")),
        "candidate_matches_current_valid": bool(candidate_vs_current.get("candidate_reference_exact_equal_valid")),
        "candidate_prefix_rows_equal_current": bool(candidate_vs_current.get("prefix_rows_equal_reference")),
        "candidate_full_prefix_block_rows_equal_current": bool(candidate_vs_current.get("full_prefix_block_rows_equal_reference")),
        "candidate_mixed_boundary_block_rows_equal_current": bool(candidate_vs_current.get("mixed_boundary_block_rows_equal_reference")),
        "candidate_mixed_boundary_tail_rows_equal_current": bool(candidate_vs_current.get("mixed_boundary_tail_rows_equal_reference")),
        "candidate_tail_rows_equal_current": bool(candidate_vs_current.get("tail_rows_equal_reference")),
        "candidate_padding_zero": bool(candidate_vs_current.get("padding_rows_zero_candidate")),
        "current_padding_zero": bool(candidate_vs_current.get("padding_rows_zero_reference")),
        "phase_profile_supports_forward_kernel_candidate": current_phase.get("dominant_phase_by_median") == "forward_pointer_kernel"
        and candidate_phase.get("dominant_phase_by_median") == "forward_pointer_kernel",
        "candidate_median_improves_by_min_gain": improvement_pct >= float(min_gain_pct),
    }
    failed = [key for key, value in gates.items() if not value]
    if failed:
        semantic_keys = (
            "candidate_matches_current_valid",
            "candidate_prefix_rows_equal_current",
            "candidate_full_prefix_block_rows_equal_current",
            "candidate_mixed_boundary_block_rows_equal_current",
            "candidate_mixed_boundary_tail_rows_equal_current",
            "candidate_tail_rows_equal_current",
            "candidate_padding_zero",
        )
        if any(key in failed for key in semantic_keys):
            decision = "reject_prefix_skip_full_blocks_semantic_mismatch"
        elif "candidate_median_improves_by_min_gain" in failed:
            decision = "reject_prefix_skip_full_blocks_no_meaningful_median_gain"
        else:
            decision = "reject_prefix_skip_full_blocks_gate_failure"
    else:
        decision = "retain_default_off_prefix_skip_full_blocks_candidate_for_next_real_chain_gate"
    return {
        "decision": decision,
        "not_h3_e2e": True,
        "not_long_video": True,
        "not_bf16_fidelity": True,
        "not_quality_or_product_speedup": True,
        "gates": gates,
        "failed_gates": failed,
        "median_ms": {
            "current_prefix_dense_overwrite_total": current_median,
            "candidate_prefix_skip_full_blocks_total": candidate_median,
            "pre_materialized_contiguous_v_total": float(record["pre_materialized_timing_ms"]["latency_ms"]["median_ms"]),
            "explicit_v_materialization": float(record["materialize_timing_ms"]["latency_ms"]["median_ms"]),
            "current_profiled_forward_pointer_kernel": float(
                current_phase["phase_latency_ms"]["forward_pointer_kernel"].get("median_ms", 0.0)
            ),
            "candidate_profiled_forward_pointer_kernel": float(
                candidate_phase["phase_latency_ms"]["forward_pointer_kernel"].get("median_ms", 0.0)
            ),
            "current_profiled_prefix_dense_overwrite": float(
                current_phase["phase_latency_ms"]["prefix_dense_overwrite"].get("median_ms", 0.0)
            ),
            "candidate_profiled_prefix_dense_overwrite": float(
                candidate_phase["phase_latency_ms"]["prefix_dense_overwrite"].get("median_ms", 0.0)
            ),
        },
        "candidate_over_current_ratio": (candidate_median / current_median) if current_median else None,
        "candidate_improvement_pct": improvement_pct,
        "min_required_gain_pct": float(min_gain_pct),
        "expected_skipped_full_prefix_query_blocks": expected_skipped_blocks,
        "principal_variable": (
            "default-off SM86 forward pointer launch that skips only full query blocks wholly inside prefix_exact_tokens, "
            "while keeping tau, routing, exact KV sink, mixed prefix/tail blocks, stride-aware V, valid padding, and dense-prefix-overwrite semantics fixed"
        ),
        "claim_boundary": "Synthetic/model-free Sol-Attn kernel evidence only; no H3 E2E, long-video, BF16-fidelity, quality, normal-PC, product, or SOTA claim.",
    }


def _routing_attribution_profile(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    metadata: PackedH3Metadata,
    *,
    tau: float,
    thresh_type: str,
    skip_full_prefix_blocks: bool,
    bv: int = 64,
) -> dict[str, Any]:
    """Model the exact work classes inside the current pointer forward kernel.

    This is a count/work model, not an in-kernel timer: Triton exposes the
    forward pointer as one CUDA kernel.  The counts are computed from the same
    reduced K summaries and routing threshold that the kernel uses, then scaled
    by the two BV=64 value tiles selected by autotune on the observed r8 shape.
    """

    valid = int(metadata.valid_length)
    prefix = int(metadata.prefix_len)
    scale = q.shape[-1] ** -0.5
    block = int(triton_sm86.BLOCK_SIZE)
    group = int(triton_sm86.GROUP_SIZE)
    blocks = int(triton_sm86.triton.cdiv(valid, block))
    heads = int(q.shape[2])
    head_dim = int(q.shape[3])
    v_tiles = head_dim // int(bv)
    prefix_skip_blocks = int(prefix) // block if bool(skip_full_prefix_blocks) else 0
    sink_start_block, sink_end_block = triton_sm86._sink_block_range(valid, 0, prefix)
    kc, _vc = triton_sm86._reduce_kv(k, v, tokens=valid)
    if thresh_type == "exact":
        threshold = triton_sm86._compute_exact_threshold(q, kc, tau=tau, scale=scale, tokens=valid)
    else:
        threshold = triton_sm86._compute_diag_threshold(q, kc, tau=tau, scale=scale, tokens=valid)
    pad_tokens = blocks * block - valid
    q_valid = q[:, :valid]
    if pad_tokens:
        q_valid = torch.nn.functional.pad(q_valid, (0, 0, 0, 0, 0, pad_tokens))
    counts = torch.full((blocks,), float(block), device=q.device, dtype=torch.float32)
    counts[-1] = float(valid - (blocks - 1) * block)
    q_bar = q_valid.view(q.shape[0], blocks, block, heads, head_dim).float().sum(dim=2)
    q_bar = q_bar / counts.view(1, blocks, 1, 1)
    scores = torch.einsum("bqhd,bkhd->bqkh", q_bar, kc[:, :blocks].float()).mul_(scale * math.log2(math.e))
    routed = scores > threshold[:, :, None, :]
    q_ids = torch.arange(blocks, device=q.device)
    k_ids = torch.arange(blocks, device=q.device)
    local = (q_ids[:, None] - k_ids[None, :]).abs() <= 1
    sink = ((k_ids[None, :] >= int(sink_start_block)) & (k_ids[None, :] < int(sink_end_block))).expand(blocks, blocks)
    launched = q_ids >= int(prefix_skip_blocks)
    routed_l = routed[:, launched]
    local_l = local[launched][None, :, :, None]
    sink_l = sink[launched][None, :, :, None]
    exact = routed_l | local_l | sink_l
    dynamic_route = routed_l & ~local_l & ~sink_l
    local_only = local_l & ~sink_l
    sink_b = sink_l.expand_as(routed_l)
    local_b = local_only.expand_as(routed_l)
    approx = ~exact
    groups = int(math.ceil(blocks / group))
    pad_blocks = groups * group - blocks
    approx_padded = torch.nn.functional.pad(approx, (0, 0, 0, pad_blocks))
    approx_group_updates = approx_padded.view(1, int(launched.sum().item()), groups, group, heads).any(dim=3).sum()
    sink_block_heads = int(sink_b.sum().item())
    local_block_heads = int(local_b.sum().item())
    dynamic_block_heads = int(dynamic_route.sum().item())
    exact_block_heads = sink_block_heads + local_block_heads + dynamic_block_heads
    approx_block_heads = int(approx.sum().item())
    q_blocks_launched = int(launched.sum().item())
    program_multiplier = int(v_tiles)
    exact_program_visits = exact_block_heads * program_multiplier
    prefix_program_visits = sink_block_heads * program_multiplier
    local_program_visits = local_block_heads * program_multiplier
    dynamic_program_visits = dynamic_block_heads * program_multiplier
    approx_group_program_updates = int(approx_group_updates.item()) * program_multiplier
    summary_route_dot_fma = q_blocks_launched * heads * program_multiplier * groups * block * group * head_dim
    approximate_value_dot_fma = q_blocks_launched * heads * program_multiplier * groups * block * group * int(bv)
    exact_score_dot_fma = exact_program_visits * block * block * head_dim
    exact_value_dot_fma = exact_program_visits * block * block * int(bv)
    strided_v_gather_bytes = exact_program_visits * block * int(bv) * int(v.element_size())
    total_block_heads = q_blocks_launched * blocks * heads
    return {
        "kind": "forward_pointer_count_and_work_model_not_in_kernel_timer",
        "shape": {"blocks": blocks, "q_blocks_launched": q_blocks_launched, "heads": heads, "head_dim": head_dim, "bv": int(bv), "v_tiles": v_tiles, "group": group},
        "sink": {"sink_start_block": int(sink_start_block), "sink_end_block": int(sink_end_block), "prefix_skip_blocks": int(prefix_skip_blocks)},
        "block_head_counts_before_value_tile_multiplier": {
            "prefix_sink_exact": sink_block_heads,
            "local_exact_outside_prefix_sink": local_block_heads,
            "dynamic_threshold_exact_outside_prefix_and_local": dynamic_block_heads,
            "approximate_summary_blocks": approx_block_heads,
            "all_possible_blocks": int(total_block_heads),
        },
        "program_visit_counts_after_value_tile_multiplier": {
            "prefix_sink_exact": prefix_program_visits,
            "local_exact_outside_prefix_sink": local_program_visits,
            "dynamic_threshold_exact_outside_prefix_and_local": dynamic_program_visits,
            "total_exact_block_visits": exact_program_visits,
            "approximate_online_updates": approx_group_program_updates,
            "online_softmax_updates_total": approx_group_program_updates + exact_program_visits,
            "dynamic_exact_offset_scheduler_iterations_current": exact_program_visits,
            "scheduler_iterations_static_prefix_sink_can_remove": prefix_program_visits,
        },
        "shares": {
            "prefix_sink_share_of_exact_block_visits": (prefix_program_visits / exact_program_visits) if exact_program_visits else 0.0,
            "dynamic_threshold_share_of_exact_block_visits": (dynamic_program_visits / exact_program_visits) if exact_program_visits else 0.0,
            "effective_exact_block_density_launched": (exact_block_heads / total_block_heads) if total_block_heads else 0.0,
            "dynamic_threshold_density_launched": (dynamic_block_heads / total_block_heads) if total_block_heads else 0.0,
        },
        "work_estimate_fma_or_bytes": {
            "approximate_summary_routing_dot_fma": int(summary_route_dot_fma),
            "approximate_value_summary_dot_fma": int(approximate_value_dot_fma),
            "exact_qk_dot_fma": int(exact_score_dot_fma),
            "exact_probability_value_dot_fma": int(exact_value_dot_fma),
            "strided_exact_v_gather_bytes": int(strided_v_gather_bytes),
        },
        "attribution_boundary": "Counts use actual generated synthetic q/k routing for this seed and shape, but they are not a replacement for CUDA timing distributions.",
    }


def _static_prefix_sink_route_decision(record: dict[str, Any], *, min_gain_pct: float) -> dict[str, Any]:
    current_pre = record["current_preflight_telemetry"]
    candidate_pre = record["candidate_preflight_telemetry"]
    current_summary = record["current_timing_ms"]["timed_telemetry_summary"]
    candidate_summary = record["candidate_timing_ms"]["timed_telemetry_summary"]
    current_median = float(record["current_timing_ms"]["latency_ms"]["median_ms"])
    candidate_median = float(record["candidate_timing_ms"]["latency_ms"]["median_ms"])
    whole_gain_pct = 100.0 * (current_median - candidate_median) / current_median if current_median else 0.0
    current_forward = float(record["current_phase_profile_ms"]["phase_latency_ms"]["forward_pointer_kernel"].get("median_ms", 0.0))
    candidate_forward = float(record["candidate_phase_profile_ms"]["phase_latency_ms"]["forward_pointer_kernel"].get("median_ms", 0.0))
    forward_gain_pct = 100.0 * (current_forward - candidate_forward) / current_forward if current_forward else 0.0
    candidate_vs_current = record["candidate_vs_current_sanity"]
    expected_static_blocks = int(record["shape"]["prefix"]) // int(triton_sm86.BLOCK_SIZE) + (1 if int(record["shape"]["prefix"]) % int(triton_sm86.BLOCK_SIZE) else 0)
    gates = {
        "full_target_shape": int(record["shape"]["T_total"]) == TARGET_H3_TOTAL and int(record["shape"]["T_valid"]) == TARGET_H3_VALID,
        "current_preflight_sparse_once": int(current_pre.get("sparse_calls", 0)) == 1,
        "candidate_preflight_sparse_once": int(candidate_pre.get("sparse_calls", 0)) == 1,
        "current_preflight_zero_fallback": int(current_pre.get("fallback_calls", -1)) == 0,
        "candidate_preflight_zero_fallback": int(candidate_pre.get("fallback_calls", -1)) == 0,
        "current_preflight_zero_materialization": int(current_pre.get("materialize_copy_count", -1)) == 0 and int(current_pre.get("materialize_copy_bytes", -1)) == 0,
        "candidate_preflight_zero_materialization": int(candidate_pre.get("materialize_copy_count", -1)) == 0 and int(candidate_pre.get("materialize_copy_bytes", -1)) == 0,
        "current_timed_zero_fallback": int(current_summary.get("fallback_calls", -1)) == 0,
        "candidate_timed_zero_fallback": int(candidate_summary.get("fallback_calls", -1)) == 0,
        "current_timed_zero_materialization": int(current_summary.get("materialize_copy_count", -1)) == 0 and int(current_summary.get("materialize_copy_bytes", -1)) == 0,
        "candidate_timed_zero_materialization": int(candidate_summary.get("materialize_copy_count", -1)) == 0 and int(candidate_summary.get("materialize_copy_bytes", -1)) == 0,
        "current_skip_marker_present": _density_sample_flag(current_pre, "skip_full_prefix_blocks") and _density_sample_flag(current_summary, "skip_full_prefix_blocks"),
        "candidate_skip_marker_present": _density_sample_flag(candidate_pre, "skip_full_prefix_blocks") and _density_sample_flag(candidate_summary, "skip_full_prefix_blocks"),
        "candidate_static_marker_present": _density_sample_flag(candidate_pre, "static_prefix_sink") and _density_sample_flag(candidate_summary, "static_prefix_sink"),
        "candidate_static_block_count_matches_sink": _density_sample_int(candidate_pre, "static_prefix_sink_blocks_estimate") == expected_static_blocks,
        "candidate_matches_current_valid": bool(candidate_vs_current.get("candidate_reference_exact_equal_valid")),
        "candidate_prefix_rows_equal_current": bool(candidate_vs_current.get("prefix_rows_equal_reference")),
        "candidate_tail_rows_equal_current": bool(candidate_vs_current.get("tail_rows_equal_reference")),
        "candidate_padding_zero": bool(candidate_vs_current.get("padding_rows_zero_candidate")),
        "current_padding_zero": bool(candidate_vs_current.get("padding_rows_zero_reference")),
        "profile_identifies_prefix_sink_scheduler_opportunity": float(record["forward_attribution_model"]["shares"].get("prefix_sink_share_of_exact_block_visits", 0.0)) >= 0.5,
        "candidate_median_improves_by_min_gain": max(whole_gain_pct, forward_gain_pct) >= float(min_gain_pct),
    }
    failed = [key for key, value in gates.items() if not value]
    if failed:
        semantic_keys = ("candidate_matches_current_valid", "candidate_prefix_rows_equal_current", "candidate_tail_rows_equal_current", "candidate_padding_zero")
        if any(key in failed for key in semantic_keys):
            decision = "reject_static_prefix_sink_scheduler_semantic_mismatch"
        elif "candidate_median_improves_by_min_gain" in failed:
            decision = "reject_static_prefix_sink_scheduler_no_meaningful_median_gain"
        else:
            decision = "reject_static_prefix_sink_scheduler_gate_failure"
    else:
        decision = "retain_default_off_static_prefix_sink_scheduler_for_next_real_chain_gate"
    return {
        "decision": decision,
        "not_h3_e2e": True,
        "not_long_video": True,
        "not_bf16_fidelity": True,
        "not_quality_or_product_speedup": True,
        "gates": gates,
        "failed_gates": failed,
        "median_ms": {
            "current_prefix_skip_total": current_median,
            "candidate_static_prefix_sink_total": candidate_median,
            "current_forward_pointer_kernel": current_forward,
            "candidate_forward_pointer_kernel": candidate_forward,
            "whole_lane_gain_pct": whole_gain_pct,
            "forward_subphase_gain_pct": forward_gain_pct,
        },
        "min_required_gain_pct": float(min_gain_pct),
        "principal_variable": "default-off static scheduling of group-0 prefix-sink exact blocks versus the current dynamic exact-offset scheduler, with prefix-skip, tau/routing, local/dynamic exact blocks, stride-aware V, dense-prefix overwrite, cache-off, and padding semantics fixed",
        "claim_boundary": "Synthetic/model-free Sol-Attn kernel evidence only; no H3 E2E, long-video, BF16-fidelity, quality, product speedup, normal-PC, or SOTA claim.",
    }


def run_static_prefix_sink_bench(
    device: torch.device,
    *,
    seed: int,
    warmup: int,
    repeats: int,
    target_total: int,
    target_valid: int,
    prefix: int,
    heads: int,
    tau: float,
    thresh_type: str,
    step_index: int,
    layer_index: int,
    min_gain_pct: float,
) -> dict[str, Any]:
    if heads != TARGET_H3_HEADS:
        raise RuntimeError(f"static-prefix-sink-bench requires H={TARGET_H3_HEADS}, got {heads}")
    if int(target_total) != TARGET_H3_TOTAL or int(target_valid) != TARGET_H3_VALID or int(prefix) != TARGET_H3_PREFIX:
        raise RuntimeError("static-prefix-sink-bench is intentionally fixed to the full observed r8 shape")
    started = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    q, k, fused_qkv, v, metadata = _large_case_tensors(
        device,
        total=int(target_total),
        valid=int(target_valid),
        heads=int(heads),
        prefix=int(prefix),
        seed=seed,
        target_total=target_total,
        target_valid=target_valid,
    )
    allocation_memory = _cuda_memory_snapshot()
    value_layout = _tensor_layout(v)
    fused_layout = _tensor_layout(fused_qkv)
    expected_materialize_bytes = _copy_bytes_for(v)
    current_policy = _new_large_policy(
        stride_aware_value=True,
        tau=tau,
        thresh_type=thresh_type,
        prefix_query_dense=True,
        exact_prefix_query=False,
        skip_full_prefix_blocks=True,
        static_prefix_sink=False,
    )
    candidate_policy = _new_large_policy(
        stride_aware_value=True,
        tau=tau,
        thresh_type=thresh_type,
        prefix_query_dense=True,
        exact_prefix_query=False,
        skip_full_prefix_blocks=True,
        static_prefix_sink=True,
    )
    try:
        forward_attribution = _routing_attribution_profile(
            q,
            k,
            v,
            metadata,
            tau=tau,
            thresh_type=thresh_type,
            skip_full_prefix_blocks=True,
        )
        current_out, current_preflight_telemetry = _checked_sparse_call(
            q, k, v, metadata, policy=current_policy, step_index=step_index, layer_index=layer_index
        )
        candidate_out, candidate_preflight_telemetry = _checked_sparse_call(
            q, k, v, metadata, policy=candidate_policy, step_index=step_index, layer_index=layer_index
        )
        torch.cuda.synchronize()
        candidate_vs_current_sanity = _large_output_sanity(candidate_out, current_out, prefix=prefix, valid=target_valid)
        candidate_vs_current_sanity["comparison_reference"] = "current_prefix_skip_stride_aware_v_dense_prefix_overwrite_lane"
        preflight_peak_memory = _cuda_memory_snapshot()
        del current_out, candidate_out
        gc.collect()
        torch.cuda.synchronize()
        current_phase = _phase_profile_distribution(
            q, k, v, metadata, warmup=warmup, repeats=repeats, tau=tau, thresh_type=thresh_type,
            forward_config=None, skip_full_prefix_blocks=True, static_prefix_sink=False,
        )
        candidate_phase = _phase_profile_distribution(
            q, k, v, metadata, warmup=warmup, repeats=repeats, tau=tau, thresh_type=thresh_type,
            forward_config=None, skip_full_prefix_blocks=True, static_prefix_sink=True,
        )
        current_timing = _time_cuda_distribution(
            lambda: _checked_sparse_call(q, k, v, metadata, policy=current_policy, step_index=step_index, layer_index=layer_index),
            warmup=warmup,
            repeats=repeats,
        )
        candidate_timing = _time_cuda_distribution(
            lambda: _checked_sparse_call(q, k, v, metadata, policy=candidate_policy, step_index=step_index, layer_index=layer_index),
            warmup=warmup,
            repeats=repeats,
        )
        final_peak_memory = _cuda_memory_snapshot()
        record: dict[str, Any] = {
            "mode": "static-prefix-sink-bench",
            "schema_version": "minimax_h3_a6000_sol_attn_sm86_static_prefix_sink_bench_v1",
            "kernel_candidates_only_not_h3_e2e": True,
            "synthetic_model_free": True,
            "model_load": False,
            "not_h3_e2e": True,
            "not_long_video": True,
            "not_bf16_fidelity": True,
            "not_quality_or_product_speedup": True,
            "shape": {"B": 1, "T_total": int(target_total), "T_valid": int(target_valid), "H": int(heads), "D": TARGET_H3_D, "prefix": int(prefix)},
            "metadata": {"prefix_len": metadata.prefix_len, "latent_grid": list(metadata.latent_grid), "valid_length": metadata.valid_length, "total_length": metadata.total_length},
            "upstream_semantics_grounding": {
                "pinned_sources": [
                    "upstreams/Sana-sol-engine/techniques/sparse_backends/sol_attn/triton_ref/fwd.py",
                    "upstreams/Sana-sol-engine/models/minimax_h3/gb200/optimized/sol_attn_h3.py",
                    "ports/minimax_h3_a6000/src/minimax_h3_a6000/sol_attn_backend.py",
                    "ports/minimax_h3_a6000/src/minimax_h3_a6000/sol_attn_triton_sm86.py",
                    ".venv/lib/python3.12/site-packages/triton/language/core.py",
                ],
                "grounded_semantics": "Current lane keeps stride-aware fused-QKV V, full-prefix-block skip, exact prefix KV sink, local exact blocks, threshold routing, online softmax order, dense prefix-query overwrite, valid padding, dense-first policy, and cache-off policy. Candidate only replaces group-0 prefix-sink dynamic exact-offset selection with a static loop in the same exact-block order.",
            },
            "candidate": {"name": "default_off_static_prefix_sink_scheduler", "env_switch": "MINIMAX_H3_A6000_SOL_ATTN_STATIC_PREFIX_SINK", "default_off": True},
            "timing_policy": {"warmup": int(warmup), "repeats": int(repeats), "cuda_event_distributions": True, "min_required_gain_pct": float(min_gain_pct)},
            "value_layout": value_layout,
            "fused_qkv_layout": fused_layout,
            "expected_materialize_bytes_per_call": expected_materialize_bytes,
            "forward_attribution_model": forward_attribution,
            "current_preflight_telemetry": _jsonable(current_preflight_telemetry.__dict__),
            "candidate_preflight_telemetry": _jsonable(candidate_preflight_telemetry.__dict__),
            "current_phase_profile_ms": current_phase,
            "candidate_phase_profile_ms": candidate_phase,
            "current_timing_ms": current_timing,
            "candidate_timing_ms": candidate_timing,
            "candidate_vs_current_sanity": candidate_vs_current_sanity,
            "cuda_memory": {"after_input_allocation": allocation_memory, "after_preflight_sanity": preflight_peak_memory, "after_all_lanes": final_peak_memory},
            "elapsed_s": time.time() - started,
            "claim_boundary": "Synthetic/model-free Sol-Attn kernel evidence only; no H3 E2E, long-video, BF16-fidelity, quality, product speedup, normal-PC, or SOTA claim.",
        }
        record["route_decision"] = _static_prefix_sink_route_decision(record, min_gain_pct=min_gain_pct)
        return record
    finally:
        del q, k, fused_qkv, v
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _bitmask_scheduler_route_decision(record: dict[str, Any], *, min_gain_pct: float) -> dict[str, Any]:
    current_pre = record["current_preflight_telemetry"]
    candidate_pre = record["candidate_preflight_telemetry"]
    current_summary = record["current_timing_ms"]["timed_telemetry_summary"]
    candidate_summary = record["candidate_timing_ms"]["timed_telemetry_summary"]
    current_median = float(record["current_timing_ms"]["latency_ms"]["median_ms"])
    candidate_median = float(record["candidate_timing_ms"]["latency_ms"]["median_ms"])
    whole_gain_pct = 100.0 * (current_median - candidate_median) / current_median if current_median else 0.0
    current_forward = float(record["current_phase_profile_ms"]["phase_latency_ms"]["forward_pointer_kernel"].get("median_ms", 0.0))
    candidate_forward = float(record["candidate_phase_profile_ms"]["phase_latency_ms"]["forward_pointer_kernel"].get("median_ms", 0.0))
    forward_gain_pct = 100.0 * (current_forward - candidate_forward) / current_forward if current_forward else 0.0
    candidate_vs_current = record["candidate_vs_current_sanity"]
    gates = {
        "full_target_shape": int(record["shape"]["T_total"]) == TARGET_H3_TOTAL and int(record["shape"]["T_valid"]) == TARGET_H3_VALID,
        "current_preflight_sparse_once": int(current_pre.get("sparse_calls", 0)) == 1,
        "candidate_preflight_sparse_once": int(candidate_pre.get("sparse_calls", 0)) == 1,
        "current_preflight_zero_fallback": int(current_pre.get("fallback_calls", -1)) == 0,
        "candidate_preflight_zero_fallback": int(candidate_pre.get("fallback_calls", -1)) == 0,
        "current_preflight_zero_materialization": int(current_pre.get("materialize_copy_count", -1)) == 0 and int(current_pre.get("materialize_copy_bytes", -1)) == 0,
        "candidate_preflight_zero_materialization": int(candidate_pre.get("materialize_copy_count", -1)) == 0 and int(candidate_pre.get("materialize_copy_bytes", -1)) == 0,
        "current_timed_zero_fallback": int(current_summary.get("fallback_calls", -1)) == 0,
        "candidate_timed_zero_fallback": int(candidate_summary.get("fallback_calls", -1)) == 0,
        "current_timed_zero_materialization": int(current_summary.get("materialize_copy_count", -1)) == 0 and int(current_summary.get("materialize_copy_bytes", -1)) == 0,
        "candidate_timed_zero_materialization": int(candidate_summary.get("materialize_copy_count", -1)) == 0 and int(candidate_summary.get("materialize_copy_bytes", -1)) == 0,
        "current_skip_marker_present": _density_sample_flag(current_pre, "skip_full_prefix_blocks") and _density_sample_flag(current_summary, "skip_full_prefix_blocks"),
        "candidate_skip_marker_present": _density_sample_flag(candidate_pre, "skip_full_prefix_blocks") and _density_sample_flag(candidate_summary, "skip_full_prefix_blocks"),
        "candidate_bitmask_marker_present": _density_sample_flag(candidate_pre, "bitmask_exact_scheduler") and _density_sample_flag(candidate_summary, "bitmask_exact_scheduler"),
        "candidate_matches_current_valid": bool(candidate_vs_current.get("candidate_reference_exact_equal_valid")),
        "candidate_prefix_rows_equal_current": bool(candidate_vs_current.get("prefix_rows_equal_reference")),
        "candidate_tail_rows_equal_current": bool(candidate_vs_current.get("tail_rows_equal_reference")),
        "candidate_padding_zero": bool(candidate_vs_current.get("padding_rows_zero_candidate")),
        "current_padding_zero": bool(candidate_vs_current.get("padding_rows_zero_reference")),
        "profile_identifies_dynamic_scheduler_opportunity": float(record["forward_attribution_model"]["shares"].get("dynamic_threshold_share_of_exact_block_visits", 0.0)) >= 0.5,
        "candidate_median_improves_by_min_gain": max(whole_gain_pct, forward_gain_pct) >= float(min_gain_pct),
    }
    failed = [key for key, value in gates.items() if not value]
    if failed:
        semantic_keys = (
            "candidate_matches_current_valid",
            "candidate_prefix_rows_equal_current",
            "candidate_tail_rows_equal_current",
            "candidate_padding_zero",
        )
        if any(key in failed for key in semantic_keys):
            decision = "reject_bitmask_exact_scheduler_semantic_mismatch"
        elif "candidate_median_improves_by_min_gain" in failed:
            decision = "reject_bitmask_exact_scheduler_no_meaningful_median_gain"
        else:
            decision = "reject_bitmask_exact_scheduler_gate_failure"
    else:
        decision = "retain_default_off_bitmask_exact_scheduler_for_next_real_chain_gate"
    return {
        "decision": decision,
        "not_h3_e2e": True,
        "not_long_video": True,
        "not_bf16_fidelity": True,
        "not_quality_or_product_speedup": True,
        "gates": gates,
        "failed_gates": failed,
        "median_ms": {
            "current_prefix_skip_total": current_median,
            "candidate_bitmask_scheduler_total": candidate_median,
            "current_forward_pointer_kernel": current_forward,
            "candidate_forward_pointer_kernel": candidate_forward,
            "whole_lane_gain_pct": whole_gain_pct,
            "forward_subphase_gain_pct": forward_gain_pct,
        },
        "min_required_gain_pct": float(min_gain_pct),
        "principal_variable": "default-off GROUP=32 bitmask exact-block scheduler versus the current dynamic vector min/update exact-offset scheduler, with prefix-skip, tau/routing, local/dynamic exact blocks, stride-aware V, dense-prefix overwrite, cache-off, and padding semantics fixed",
        "claim_boundary": "Synthetic/model-free Sol-Attn kernel evidence only; no H3 E2E, long-video, BF16-fidelity, quality, product speedup, normal-PC, or SOTA claim.",
    }


def run_bitmask_scheduler_bench(
    device: torch.device,
    *,
    seed: int,
    warmup: int,
    repeats: int,
    target_total: int,
    target_valid: int,
    prefix: int,
    heads: int,
    tau: float,
    thresh_type: str,
    step_index: int,
    layer_index: int,
    min_gain_pct: float,
) -> dict[str, Any]:
    if heads != TARGET_H3_HEADS:
        raise RuntimeError(f"bitmask-scheduler-bench requires H={TARGET_H3_HEADS}, got {heads}")
    if int(target_total) != TARGET_H3_TOTAL or int(target_valid) != TARGET_H3_VALID or int(prefix) != TARGET_H3_PREFIX:
        raise RuntimeError("bitmask-scheduler-bench is intentionally fixed to the full observed r8 shape")
    started = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    q, k, fused_qkv, v, metadata = _large_case_tensors(
        device,
        total=int(target_total),
        valid=int(target_valid),
        heads=int(heads),
        prefix=int(prefix),
        seed=seed,
        target_total=target_total,
        target_valid=target_valid,
    )
    allocation_memory = _cuda_memory_snapshot()
    value_layout = _tensor_layout(v)
    fused_layout = _tensor_layout(fused_qkv)
    expected_materialize_bytes = _copy_bytes_for(v)
    current_policy = _new_large_policy(
        stride_aware_value=True,
        tau=tau,
        thresh_type=thresh_type,
        prefix_query_dense=True,
        exact_prefix_query=False,
        skip_full_prefix_blocks=True,
        static_prefix_sink=False,
        bitmask_exact_scheduler=False,
    )
    candidate_policy = _new_large_policy(
        stride_aware_value=True,
        tau=tau,
        thresh_type=thresh_type,
        prefix_query_dense=True,
        exact_prefix_query=False,
        skip_full_prefix_blocks=True,
        static_prefix_sink=False,
        bitmask_exact_scheduler=True,
    )
    try:
        forward_attribution = _routing_attribution_profile(
            q,
            k,
            v,
            metadata,
            tau=tau,
            thresh_type=thresh_type,
            skip_full_prefix_blocks=True,
            bv=64,
        )
        current_out, current_preflight_telemetry = _checked_sparse_call(
            q, k, v, metadata, policy=current_policy, step_index=step_index, layer_index=layer_index
        )
        candidate_out, candidate_preflight_telemetry = _checked_sparse_call(
            q, k, v, metadata, policy=candidate_policy, step_index=step_index, layer_index=layer_index
        )
        torch.cuda.synchronize()
        candidate_vs_current_sanity = _large_output_sanity(candidate_out, current_out, prefix=prefix, valid=target_valid)
        candidate_vs_current_sanity["comparison_reference"] = "current_prefix_skip_stride_aware_v_dense_prefix_overwrite_lane"
        preflight_peak_memory = _cuda_memory_snapshot()
        del current_out, candidate_out
        gc.collect()
        torch.cuda.synchronize()
        current_phase = _phase_profile_distribution(
            q, k, v, metadata, warmup=warmup, repeats=repeats, tau=tau, thresh_type=thresh_type,
            forward_config=None, skip_full_prefix_blocks=True, static_prefix_sink=False, bitmask_exact_scheduler=False,
        )
        candidate_phase = _phase_profile_distribution(
            q, k, v, metadata, warmup=warmup, repeats=repeats, tau=tau, thresh_type=thresh_type,
            forward_config=None, skip_full_prefix_blocks=True, static_prefix_sink=False, bitmask_exact_scheduler=True,
        )
        current_timing = _time_cuda_distribution(
            lambda: _checked_sparse_call(q, k, v, metadata, policy=current_policy, step_index=step_index, layer_index=layer_index),
            warmup=warmup,
            repeats=repeats,
        )
        candidate_timing = _time_cuda_distribution(
            lambda: _checked_sparse_call(q, k, v, metadata, policy=candidate_policy, step_index=step_index, layer_index=layer_index),
            warmup=warmup,
            repeats=repeats,
        )
        final_peak_memory = _cuda_memory_snapshot()
        record: dict[str, Any] = {
            "mode": "bitmask-scheduler-bench",
            "schema_version": "minimax_h3_a6000_sol_attn_sm86_bitmask_scheduler_bench_v1",
            "kernel_candidates_only_not_h3_e2e": True,
            "synthetic_model_free": True,
            "model_load": False,
            "not_h3_e2e": True,
            "not_long_video": True,
            "not_bf16_fidelity": True,
            "not_quality_or_product_speedup": True,
            "shape": {"B": 1, "T_total": int(target_total), "T_valid": int(target_valid), "H": int(heads), "D": TARGET_H3_D, "prefix": int(prefix)},
            "metadata": {"prefix_len": metadata.prefix_len, "latent_grid": list(metadata.latent_grid), "valid_length": metadata.valid_length, "total_length": metadata.total_length},
            "upstream_semantics_grounding": {
                "pinned_sources": [
                    "upstreams/Sana-sol-engine/techniques/sparse_backends/sol_attn/triton_ref/fwd.py",
                    "upstreams/Sana-sol-engine/techniques/sparse_backends/sol_attn/sm90/mainloop.py",
                    "upstreams/Sana-sol-engine/techniques/sparse_backends/sol_attn/sm90/exact.py",
                    "upstreams/Sana-sol-engine/techniques/sparse_backends/sol_attn/sm120/kernel.py",
                    "ports/minimax_h3_a6000/src/minimax_h3_a6000/sol_attn_backend.py",
                    "ports/minimax_h3_a6000/src/minimax_h3_a6000/sol_attn_triton_sm86.py",
                ],
                "grounded_semantics": "Current lane keeps stride-aware fused-QKV V, full-prefix-block skip, exact prefix KV sink, local exact blocks, threshold routing, online softmax order, dense prefix-query overwrite, valid padding, dense-first policy, and cache-off policy. Candidate only replaces dynamic exact-offset vector min/update selection with a GROUP=32 bitmask that visits set bits in ascending order, following the SM90 route-mask/lowbit consumption pattern at pointer-kernel scale.",
            },
            "candidate": {"name": "default_off_bitmask_exact_scheduler", "env_switch": "MINIMAX_H3_A6000_SOL_ATTN_BITMASK_SCHEDULER", "default_off": True},
            "timing_policy": {"warmup": int(warmup), "repeats": int(repeats), "cuda_event_distributions": True, "min_required_gain_pct": float(min_gain_pct)},
            "value_layout": value_layout,
            "fused_qkv_layout": fused_layout,
            "expected_materialize_bytes_per_call": expected_materialize_bytes,
            "forward_attribution_model": forward_attribution,
            "current_preflight_telemetry": _jsonable(current_preflight_telemetry.__dict__),
            "candidate_preflight_telemetry": _jsonable(candidate_preflight_telemetry.__dict__),
            "current_phase_profile_ms": current_phase,
            "candidate_phase_profile_ms": candidate_phase,
            "current_timing_ms": current_timing,
            "candidate_timing_ms": candidate_timing,
            "candidate_vs_current_sanity": candidate_vs_current_sanity,
            "cuda_memory": {"after_input_allocation": allocation_memory, "after_preflight_sanity": preflight_peak_memory, "after_all_lanes": final_peak_memory},
            "elapsed_s": time.time() - started,
            "claim_boundary": "Synthetic/model-free Sol-Attn kernel evidence only; no H3 E2E, long-video, BF16-fidelity, quality, product speedup, normal-PC, or SOTA claim.",
        }
        record["route_decision"] = _bitmask_scheduler_route_decision(record, min_gain_pct=min_gain_pct)
        return record
    finally:
        del q, k, fused_qkv, v
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_prefix_skip_bench(
    device: torch.device,
    *,
    seed: int,
    warmup: int,
    repeats: int,
    target_total: int,
    target_valid: int,
    prefix: int,
    heads: int,
    tau: float,
    thresh_type: str,
    step_index: int,
    layer_index: int,
    min_gain_pct: float,
) -> dict[str, Any]:
    if heads != TARGET_H3_HEADS:
        raise RuntimeError(f"prefix-skip-bench requires H={TARGET_H3_HEADS}, got {heads}")
    if int(target_total) != TARGET_H3_TOTAL or int(target_valid) != TARGET_H3_VALID or int(prefix) != TARGET_H3_PREFIX:
        raise RuntimeError("prefix-skip-bench is intentionally fixed to the full observed r8 shape")
    if repeats < 1 or warmup < 0:
        raise RuntimeError("prefix-skip-bench requires repeats >= 1 and warmup >= 0")

    started = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    q, k, fused_qkv, v, metadata = _large_case_tensors(
        device,
        total=int(target_total),
        valid=int(target_valid),
        heads=int(heads),
        prefix=int(prefix),
        seed=seed,
        target_total=target_total,
        target_valid=target_valid,
    )
    allocation_memory = _cuda_memory_snapshot()
    value_layout = _tensor_layout(v)
    fused_layout = _tensor_layout(fused_qkv)
    expected_materialize_bytes = _copy_bytes_for(v)

    current_policy = _new_large_policy(
        stride_aware_value=True,
        tau=tau,
        thresh_type=thresh_type,
        prefix_query_dense=True,
        exact_prefix_query=False,
        skip_full_prefix_blocks=False,
    )
    candidate_policy = _new_large_policy(
        stride_aware_value=True,
        tau=tau,
        thresh_type=thresh_type,
        prefix_query_dense=True,
        exact_prefix_query=False,
        skip_full_prefix_blocks=True,
    )
    pre_materialized_policy = _new_large_policy(
        stride_aware_value=False,
        tau=tau,
        thresh_type=thresh_type,
        prefix_query_dense=True,
        exact_prefix_query=False,
        skip_full_prefix_blocks=False,
    )

    try:
        materialize_timing = _time_materialize_distribution(v, warmup=warmup, repeats=repeats)
        v_ref = v.contiguous()
        torch.cuda.synchronize()
        materialized_out, materialized_preflight_telemetry = _checked_sparse_call(
            q,
            k,
            v_ref,
            metadata,
            policy=pre_materialized_policy,
            step_index=step_index,
            layer_index=layer_index,
        )
        current_out, current_preflight_telemetry = _checked_sparse_call(
            q,
            k,
            v,
            metadata,
            policy=current_policy,
            step_index=step_index,
            layer_index=layer_index,
        )
        candidate_out, candidate_preflight_telemetry = _checked_sparse_call(
            q,
            k,
            v,
            metadata,
            policy=candidate_policy,
            step_index=step_index,
            layer_index=layer_index,
        )
        prefix_dense_out = _prefix_dense_overwrite_call(q, k, v, prefix=prefix, valid=target_valid)
        torch.cuda.synchronize()
        current_vs_materialized_sanity = _large_output_sanity(current_out, materialized_out, prefix=prefix, valid=target_valid)
        current_vs_materialized_sanity["comparison_reference"] = "same_policy_current_prefix_dense_overwrite_with_pre_materialized_contiguous_v"
        candidate_vs_current_sanity = _large_output_sanity(candidate_out, current_out, prefix=prefix, valid=target_valid)
        candidate_vs_current_sanity["comparison_reference"] = "current_full_forward_prefix_dense_overwrite_lane"
        prefix_dense_diff_current = (current_out[:, :prefix].float() - prefix_dense_out.float()).abs()
        prefix_dense_diff_candidate = (candidate_out[:, :prefix].float() - prefix_dense_out.float()).abs()
        prefix_dense_checks = {
            "current_prefix_rows_equal_direct_dense": bool(torch.equal(current_out[:, :prefix], prefix_dense_out)),
            "candidate_prefix_rows_equal_direct_dense": bool(torch.equal(candidate_out[:, :prefix], prefix_dense_out)),
            "candidate_prefix_rows_equal_current": bool(torch.equal(candidate_out[:, :prefix], current_out[:, :prefix])),
            "current_prefix_max_abs_vs_direct_dense": float(prefix_dense_diff_current.max().item()) if prefix_dense_diff_current.numel() else 0.0,
            "candidate_prefix_max_abs_vs_direct_dense": float(prefix_dense_diff_candidate.max().item()) if prefix_dense_diff_candidate.numel() else 0.0,
            "prefix_rows": int(prefix),
            "valid_kv_rows": int(target_valid),
            "full_prefix_blocks": int(prefix) // int(triton_sm86.BLOCK_SIZE),
            "mixed_boundary_block_index": int(prefix) // int(triton_sm86.BLOCK_SIZE),
        }
        preflight_peak_memory = _cuda_memory_snapshot()
        del materialized_out, current_out, candidate_out, prefix_dense_out
        gc.collect()
        torch.cuda.synchronize()

        current_phase = _phase_profile_distribution(
            q,
            k,
            v,
            metadata,
            warmup=warmup,
            repeats=repeats,
            tau=tau,
            thresh_type=thresh_type,
            forward_config=None,
            skip_full_prefix_blocks=False,
        )
        candidate_phase = _phase_profile_distribution(
            q,
            k,
            v,
            metadata,
            warmup=warmup,
            repeats=repeats,
            tau=tau,
            thresh_type=thresh_type,
            forward_config=None,
            skip_full_prefix_blocks=True,
        )
        current_timing = _time_cuda_distribution(
            lambda: _checked_sparse_call(
                q,
                k,
                v,
                metadata,
                policy=current_policy,
                step_index=step_index,
                layer_index=layer_index,
            ),
            warmup=warmup,
            repeats=repeats,
        )
        candidate_timing = _time_cuda_distribution(
            lambda: _checked_sparse_call(
                q,
                k,
                v,
                metadata,
                policy=candidate_policy,
                step_index=step_index,
                layer_index=layer_index,
            ),
            warmup=warmup,
            repeats=repeats,
        )
        pre_materialized_timing = _time_cuda_distribution(
            lambda: _checked_sparse_call(
                q,
                k,
                v_ref,
                metadata,
                policy=pre_materialized_policy,
                step_index=step_index,
                layer_index=layer_index,
            ),
            warmup=warmup,
            repeats=repeats,
        )
        final_peak_memory = _cuda_memory_snapshot()
        record: dict[str, Any] = {
            "mode": "prefix-skip-bench",
            "schema_version": "minimax_h3_a6000_sol_attn_sm86_prefix_skip_bench_v1",
            "kernel_candidates_only_not_h3_e2e": True,
            "synthetic_model_free": True,
            "model_load": False,
            "not_h3_e2e": True,
            "not_long_video": True,
            "not_bf16_fidelity": True,
            "not_quality_or_product_speedup": True,
            "shape": {"B": 1, "T_total": int(target_total), "T_valid": int(target_valid), "H": int(heads), "D": TARGET_H3_D, "prefix": int(prefix)},
            "metadata": {
                "prefix_len": metadata.prefix_len,
                "latent_grid": list(metadata.latent_grid),
                "valid_length": metadata.valid_length,
                "total_length": metadata.total_length,
            },
            "upstream_semantics_grounding": {
                "pinned_sources": [
                    "upstreams/Sana-sol-engine/models/minimax_h3/gb200/optimized/sol_attn_h3.py",
                    "upstreams/Sana-sol-engine/techniques/sparse_backends/sol_attn/triton_ref/fwd.py",
                    "ports/minimax_h3_a6000/src/minimax_h3_a6000/sol_attn_backend.py",
                    "ports/minimax_h3_a6000/src/minimax_h3_a6000/sol_attn_triton_sm86.py",
                ],
                "grounded_semantics": "Candidate skips only full query blocks wholly inside prefix_exact_tokens in the SM86 forward launch. It keeps exact prefix KV sink, tau/routing, mixed prefix/tail block execution, valid-length padding, and the wrapper dense-prefix query overwrite unchanged.",
            },
            "candidate": {
                "name": "default_off_skip_full_prefix_blocks",
                "env_switch": "MINIMAX_H3_A6000_SOL_ATTN_SKIP_FULL_PREFIX_BLOCKS",
                "default_off": True,
                "prefix_exact_tokens": int(prefix),
                "expected_skipped_full_prefix_query_blocks": int(prefix) // int(triton_sm86.BLOCK_SIZE),
                "mixed_prefix_tail_block_executed": True,
            },
            "timing_policy": {
                "warmup": int(warmup),
                "repeats": int(repeats),
                "same_policy_for_current_candidate_and_reference": True,
                "cuda_event_distributions": True,
                "decision_sized_not_formal_speedup_distribution": True,
                "min_required_gain_pct": float(min_gain_pct),
            },
            "sparse_policy": {
                "tau": float(tau),
                "thresh_type": str(thresh_type),
                "step_index": int(step_index),
                "layer_index": int(layer_index),
                "dense_first_steps": 10,
                "dense_first_layers": 2,
                "prefix_query_dense": True,
                "exact_prefix_query": False,
                "cache_enabled": False,
            },
            "value_layout": value_layout,
            "fused_qkv_layout": fused_layout,
            "expected_materialize_bytes_per_call": expected_materialize_bytes,
            "materialize_timing_ms": materialize_timing,
            "pre_materialized_preflight_telemetry": _jsonable(materialized_preflight_telemetry.__dict__),
            "current_preflight_telemetry": _jsonable(current_preflight_telemetry.__dict__),
            "candidate_preflight_telemetry": _jsonable(candidate_preflight_telemetry.__dict__),
            "current_phase_profile_ms": current_phase,
            "candidate_phase_profile_ms": candidate_phase,
            "current_timing_ms": current_timing,
            "candidate_timing_ms": candidate_timing,
            "pre_materialized_timing_ms": pre_materialized_timing,
            "current_vs_materialized_sanity": current_vs_materialized_sanity,
            "candidate_vs_current_sanity": candidate_vs_current_sanity,
            "prefix_dense_checks": prefix_dense_checks,
            "cuda_memory": {
                "after_input_allocation": allocation_memory,
                "after_preflight_sanity": preflight_peak_memory,
                "after_all_lanes": final_peak_memory,
            },
            "elapsed_s": time.time() - started,
            "claim_boundary": "Synthetic/model-free Sol-Attn kernel evidence only; no H3 E2E, long-video, BF16-fidelity, quality, product speedup, normal-PC, or SOTA claim.",
        }
        record["route_decision"] = _prefix_skip_route_decision(record, min_gain_pct=min_gain_pct)
        return record
    finally:
        del q, k, fused_qkv, v
        if "v_ref" in locals():
            del v_ref
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_phase_bench(
    device: torch.device,
    *,
    seed: int,
    warmup: int,
    repeats: int,
    target_total: int,
    target_valid: int,
    prefix: int,
    heads: int,
    tau: float,
    thresh_type: str,
    step_index: int,
    layer_index: int,
    candidate_forward_config: str,
    min_gain_pct: float,
) -> dict[str, Any]:
    if heads != TARGET_H3_HEADS:
        raise RuntimeError(f"phase-bench requires H={TARGET_H3_HEADS}, got {heads}")
    if int(target_total) != TARGET_H3_TOTAL or int(target_valid) != TARGET_H3_VALID or int(prefix) != TARGET_H3_PREFIX:
        raise RuntimeError("phase-bench is intentionally fixed to the full observed r8 shape")
    if repeats < 1 or warmup < 0:
        raise RuntimeError("phase-bench requires repeats >= 1 and warmup >= 0")

    started = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    q, k, fused_qkv, v, metadata = _large_case_tensors(
        device,
        total=int(target_total),
        valid=int(target_valid),
        heads=int(heads),
        prefix=int(prefix),
        seed=seed,
        target_total=target_total,
        target_valid=target_valid,
    )
    allocation_memory = _cuda_memory_snapshot()
    value_layout = _tensor_layout(v)
    fused_layout = _tensor_layout(fused_qkv)
    expected_materialize_bytes = _copy_bytes_for(v)

    current_policy = _new_large_policy(
        stride_aware_value=True,
        tau=tau,
        thresh_type=thresh_type,
        prefix_query_dense=True,
        exact_prefix_query=False,
    )
    candidate_policy = _new_large_policy(
        stride_aware_value=True,
        tau=tau,
        thresh_type=thresh_type,
        prefix_query_dense=True,
        exact_prefix_query=False,
        forward_config=candidate_forward_config,
    )
    pre_materialized_policy = _new_large_policy(
        stride_aware_value=False,
        tau=tau,
        thresh_type=thresh_type,
        prefix_query_dense=True,
        exact_prefix_query=False,
    )

    try:
        materialize_timing = _time_materialize_distribution(v, warmup=warmup, repeats=repeats)
        v_ref = v.contiguous()
        torch.cuda.synchronize()
        materialized_out, materialized_preflight_telemetry = _checked_sparse_call(
            q,
            k,
            v_ref,
            metadata,
            policy=pre_materialized_policy,
            step_index=step_index,
            layer_index=layer_index,
        )
        current_out, current_preflight_telemetry = _checked_sparse_call(
            q,
            k,
            v,
            metadata,
            policy=current_policy,
            step_index=step_index,
            layer_index=layer_index,
        )
        candidate_out, candidate_preflight_telemetry = _checked_sparse_call(
            q,
            k,
            v,
            metadata,
            policy=candidate_policy,
            step_index=step_index,
            layer_index=layer_index,
        )
        prefix_dense_out = _prefix_dense_overwrite_call(q, k, v, prefix=prefix, valid=target_valid)
        torch.cuda.synchronize()
        current_vs_materialized_sanity = _large_output_sanity(current_out, materialized_out, prefix=prefix, valid=target_valid)
        current_vs_materialized_sanity["comparison_reference"] = "same_policy_current_prefix_dense_overwrite_with_pre_materialized_contiguous_v"
        candidate_vs_current_sanity = _large_output_sanity(candidate_out, current_out, prefix=prefix, valid=target_valid)
        candidate_vs_current_sanity["comparison_reference"] = "current_autotuned_forward_config_prefix_dense_overwrite_lane"
        prefix_dense_diff_current = (current_out[:, :prefix].float() - prefix_dense_out.float()).abs()
        prefix_dense_checks = {
            "current_prefix_rows_equal_direct_dense": bool(torch.equal(current_out[:, :prefix], prefix_dense_out)),
            "candidate_prefix_rows_equal_current": bool(torch.equal(candidate_out[:, :prefix], current_out[:, :prefix])),
            "current_prefix_max_abs_vs_direct_dense": float(prefix_dense_diff_current.max().item()) if prefix_dense_diff_current.numel() else 0.0,
            "prefix_rows": int(prefix),
            "valid_kv_rows": int(target_valid),
        }
        preflight_peak_memory = _cuda_memory_snapshot()
        del materialized_out, current_out, candidate_out, prefix_dense_out
        gc.collect()
        torch.cuda.synchronize()

        current_phase = _phase_profile_distribution(
            q,
            k,
            v,
            metadata,
            warmup=warmup,
            repeats=repeats,
            tau=tau,
            thresh_type=thresh_type,
            forward_config=None,
        )
        candidate_phase = _phase_profile_distribution(
            q,
            k,
            v,
            metadata,
            warmup=warmup,
            repeats=repeats,
            tau=tau,
            thresh_type=thresh_type,
            forward_config=candidate_forward_config,
        )
        current_timing = _time_cuda_distribution(
            lambda: _checked_sparse_call(
                q,
                k,
                v,
                metadata,
                policy=current_policy,
                step_index=step_index,
                layer_index=layer_index,
            ),
            warmup=warmup,
            repeats=repeats,
        )
        candidate_timing = _time_cuda_distribution(
            lambda: _checked_sparse_call(
                q,
                k,
                v,
                metadata,
                policy=candidate_policy,
                step_index=step_index,
                layer_index=layer_index,
            ),
            warmup=warmup,
            repeats=repeats,
        )
        pre_materialized_timing = _time_cuda_distribution(
            lambda: _checked_sparse_call(
                q,
                k,
                v_ref,
                metadata,
                policy=pre_materialized_policy,
                step_index=step_index,
                layer_index=layer_index,
            ),
            warmup=warmup,
            repeats=repeats,
        )
        final_peak_memory = _cuda_memory_snapshot()
        record: dict[str, Any] = {
            "mode": "phase-bench",
            "schema_version": "minimax_h3_a6000_sol_attn_sm86_phase_bench_v1",
            "kernel_candidates_only_not_h3_e2e": True,
            "synthetic_model_free": True,
            "model_load": False,
            "not_h3_e2e": True,
            "not_long_video": True,
            "not_bf16_fidelity": True,
            "not_quality_or_product_speedup": True,
            "shape": {"B": 1, "T_total": int(target_total), "T_valid": int(target_valid), "H": int(heads), "D": TARGET_H3_D, "prefix": int(prefix)},
            "metadata": {
                "prefix_len": metadata.prefix_len,
                "latent_grid": list(metadata.latent_grid),
                "valid_length": metadata.valid_length,
                "total_length": metadata.total_length,
            },
            "upstream_semantics_grounding": {
                "pinned_sources": [
                    "upstreams/Sana-sol-engine/techniques/sparse_backends/sol_attn_backend.py",
                    "upstreams/Sana-sol-engine/techniques/sparse_backends/sol_attn/interface.py",
                    "ports/minimax_h3_a6000/src/minimax_h3_a6000/sol_attn_backend.py",
                ],
                "grounded_semantics": "Same current lane semantics: stride-aware fused-QKV V view, exact prefix KV sink, valid-length padded output, and dense prefix query overwrite outside the sparse forward kernel.",
            },
            "candidate_forward_config": str(candidate_forward_config),
            "timing_policy": {
                "warmup": int(warmup),
                "repeats": int(repeats),
                "same_policy_for_current_candidate_and_reference": True,
                "cuda_event_distributions": True,
                "decision_sized_not_formal_speedup_distribution": True,
            },
            "sparse_policy": {
                "tau": float(tau),
                "thresh_type": str(thresh_type),
                "step_index": int(step_index),
                "layer_index": int(layer_index),
                "dense_first_steps": 10,
                "dense_first_layers": 2,
                "prefix_query_dense": True,
                "exact_prefix_query": False,
                "cache_enabled": False,
            },
            "value_layout": value_layout,
            "fused_qkv_layout": fused_layout,
            "expected_materialize_bytes_per_call": expected_materialize_bytes,
            "materialize_timing_ms": materialize_timing,
            "pre_materialized_preflight_telemetry": _jsonable(materialized_preflight_telemetry.__dict__),
            "current_preflight_telemetry": _jsonable(current_preflight_telemetry.__dict__),
            "candidate_preflight_telemetry": _jsonable(candidate_preflight_telemetry.__dict__),
            "current_phase_profile_ms": current_phase,
            "candidate_phase_profile_ms": candidate_phase,
            "current_timing_ms": current_timing,
            "candidate_timing_ms": candidate_timing,
            "pre_materialized_timing_ms": pre_materialized_timing,
            "current_vs_materialized_sanity": current_vs_materialized_sanity,
            "candidate_vs_current_sanity": candidate_vs_current_sanity,
            "prefix_dense_checks": prefix_dense_checks,
            "cuda_memory": {
                "after_input_allocation": allocation_memory,
                "after_preflight_sanity": preflight_peak_memory,
                "after_all_lanes": final_peak_memory,
            },
            "elapsed_s": time.time() - started,
            "claim_boundary": "Synthetic/model-free Sol-Attn kernel evidence only; no H3 E2E, long-video, BF16-fidelity, quality, product speedup, normal-PC, or SOTA claim.",
        }
        record["route_decision"] = _phase_route_decision(record, min_gain_pct=min_gain_pct)
        return record
    finally:
        del q, k, fused_qkv, v
        if "v_ref" in locals():
            del v_ref
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_large_bench(
    device: torch.device,
    *,
    seed: int,
    warmup: int,
    repeats: int,
    target_total: int,
    target_valid: int,
    prefix: int,
    heads: int,
    min_total: int,
    granularity: int,
    tau: float,
    thresh_type: str,
    step_index: int,
    layer_index: int,
) -> dict[str, Any]:
    if heads != TARGET_H3_HEADS:
        # The r8 materialization byte accounting is tied to H=56.  Accepting a
        # different value would make the output no longer represent the observed lane.
        raise RuntimeError(f"large-bench requires H={TARGET_H3_HEADS}, got {heads}")
    if repeats < 1 or warmup < 0:
        raise RuntimeError("large-bench requires repeats >= 1 and warmup >= 0")
    if granularity < 1:
        raise RuntimeError("large-bisect-granularity must be >= 1")

    attempts: list[dict[str, Any]] = []
    target_shape = {"B": 1, "T_total": int(target_total), "T_valid": int(target_valid), "H": int(heads), "D": TARGET_H3_D, "prefix": int(prefix)}
    first = _attempt_large_v_materialization_shape(
        device,
        shape=target_shape,
        seed=seed,
        warmup=warmup,
        repeats=repeats,
        tau=tau,
        thresh_type=thresh_type,
        step_index=step_index,
        layer_index=layer_index,
        target_total=target_total,
        target_valid=target_valid,
    )
    attempts.append(first)
    if first.get("status") == "success":
        return _finalize_large_result(
            attempts=attempts,
            selected=first,
            target_shape=target_shape,
            failure_boundary=None,
            warmup=warmup,
            repeats=repeats,
            tau=tau,
            thresh_type=thresh_type,
            step_index=step_index,
            layer_index=layer_index,
        )

    lower_success: dict[str, Any] | None = None
    upper_failure: dict[str, Any] = first
    probe_total = max(int(min_total), int(target_total) // 2)
    while probe_total >= int(min_total):
        shape = _shape_for_total(probe_total, target_total=target_total, target_valid=target_valid, prefix=prefix)
        shape["H"] = int(heads)
        probe = _attempt_large_v_materialization_shape(
            device,
            shape=shape,
            seed=seed,
            warmup=warmup,
            repeats=repeats,
            tau=tau,
            thresh_type=thresh_type,
            step_index=step_index,
            layer_index=layer_index,
            target_total=target_total,
            target_valid=target_valid,
        )
        attempts.append(probe)
        if probe.get("status") == "success":
            lower_success = probe
            break
        upper_failure = probe
        probe_total //= 2

    if lower_success is None:
        return _finalize_large_result(
            attempts=attempts,
            selected=None,
            target_shape=target_shape,
            failure_boundary={"largest_success": None, "lowest_observed_failure": upper_failure.get("shape")},
            warmup=warmup,
            repeats=repeats,
            tau=tau,
            thresh_type=thresh_type,
            step_index=step_index,
            layer_index=layer_index,
        )

    low = int(lower_success["shape"]["T_total"])
    high = int(upper_failure["shape"]["T_total"])
    best = lower_success
    while high - low > int(granularity):
        mid = low + (high - low) // 2
        mid = max(int(min_total), (mid // int(granularity)) * int(granularity))
        if mid <= low:
            break
        shape = _shape_for_total(mid, target_total=target_total, target_valid=target_valid, prefix=prefix)
        shape["H"] = int(heads)
        probe = _attempt_large_v_materialization_shape(
            device,
            shape=shape,
            seed=seed,
            warmup=warmup,
            repeats=repeats,
            tau=tau,
            thresh_type=thresh_type,
            step_index=step_index,
            layer_index=layer_index,
            target_total=target_total,
            target_valid=target_valid,
        )
        attempts.append(probe)
        if probe.get("status") == "success":
            best = probe
            low = int(probe["shape"]["T_total"])
        else:
            upper_failure = probe
            high = int(probe["shape"]["T_total"])

    return _finalize_large_result(
        attempts=attempts,
        selected=best,
        target_shape=target_shape,
        failure_boundary={"largest_success": best.get("shape"), "lowest_observed_failure": upper_failure.get("shape")},
        warmup=warmup,
        repeats=repeats,
        tau=tau,
        thresh_type=thresh_type,
        step_index=step_index,
        layer_index=layer_index,
    )


def _finalize_large_result(
    *,
    attempts: list[dict[str, Any]],
    selected: dict[str, Any] | None,
    target_shape: dict[str, int],
    failure_boundary: dict[str, Any] | None,
    warmup: int,
    repeats: int,
    tau: float,
    thresh_type: str,
    step_index: int,
    layer_index: int,
) -> dict[str, Any]:
    if selected is None:
        decision = {
            "decision": "reject_stride_aware_v_route_no_legal_scaled_shape_succeeded",
            "failed_gates": ["no_successful_scaled_shape"],
            "claim_boundary": "Synthetic/model-free kernel evidence only; no H3 E2E, long-video, BF16-fidelity, quality, product, or SOTA claim.",
        }
    else:
        decision = selected.get("route_decision")
    return {
        "mode": "large-bench",
        "schema_version": "minimax_h3_a6000_sol_attn_sm86_large_shape_harness_v3_stride_aware_v_materialization",
        "kernel_candidates_only_not_h3_e2e": True,
        "synthetic_model_free": True,
        "model_load": False,
        "not_h3_e2e": True,
        "not_long_video": True,
        "not_bf16_fidelity": True,
        "not_quality_or_product_speedup": True,
        "target_observed_h3_shape": target_shape,
        "timing_policy": {
            "warmup": int(warmup),
            "repeats": int(repeats),
            "same_policy_for_candidate_and_reference": True,
            "primary_variable": "explicit V materialization only",
            "large_shape_decision_sized_not_formal_speedup_distribution": True,
        },
        "sparse_policy": {
            "tau": float(tau),
            "thresh_type": str(thresh_type),
            "step_index": int(step_index),
            "layer_index": int(layer_index),
            "dense_first_steps": 10,
            "dense_first_layers": 2,
            "prefix_query_dense": True,
            "exact_prefix_query": False,
            "skip_full_prefix_blocks": False,
            "cache_enabled": False,
        },
        "attempts": attempts,
        "selected_attempt_index": attempts.index(selected) if selected in attempts else None,
        "failure_boundary": failure_boundary,
        "route_decision": decision,
        "claim_boundary": "Synthetic/model-free Sol-Attn kernel evidence only; no H3 E2E, long-video, BF16-fidelity, quality, product speedup, normal-PC, or SOTA claim.",
    }


def main() -> int:
    args = _parse_args()
    torch.manual_seed(args.seed)
    device = _validate_single_a6000(args.device)
    props = torch.cuda.get_device_properties(0)
    results: dict[str, Any] = {
        "schema_version": "minimax_h3_a6000_sol_attn_sm86_harness_v1",
        "model_load": False,
        "seed": args.seed,
        "device": args.device,
        "capability": [8, 6],
        "visible_cuda_device_count": int(torch.cuda.device_count()),
        "visible_device_name": props.name,
        "visible_device_total_memory_bytes": int(props.total_memory),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if args.mode in ("correctness", "both"):
        results["correctness"] = run_correctness(device)
    if args.mode in ("bench", "both"):
        results["bench"] = run_bench(device, warmup=args.warmup, repeats=args.repeats)
    if args.mode == "large-bench":
        results["large_bench"] = run_large_bench(
            device,
            seed=args.seed,
            warmup=args.large_warmup,
            repeats=args.large_repeats,
            target_total=args.large_target_total,
            target_valid=args.large_target_valid,
            prefix=args.large_prefix,
            heads=args.large_heads,
            min_total=args.large_min_total,
            granularity=args.large_bisect_granularity,
            tau=args.large_tau,
            thresh_type=args.large_thresh_type,
            step_index=args.large_step_index,
            layer_index=args.large_layer_index,
        )
    if args.mode == "phase-bench":
        results["phase_bench"] = run_phase_bench(
            device,
            seed=args.seed,
            warmup=args.large_warmup,
            repeats=args.large_repeats,
            target_total=args.large_target_total,
            target_valid=args.large_target_valid,
            prefix=args.large_prefix,
            heads=args.large_heads,
            tau=args.large_tau,
            thresh_type=args.large_thresh_type,
            step_index=args.large_step_index,
            layer_index=args.large_layer_index,
            candidate_forward_config=args.phase_candidate_forward_config,
            min_gain_pct=args.phase_min_gain_pct,
        )
    if args.mode == "prefix-skip-bench":
        results["prefix_skip_bench"] = run_prefix_skip_bench(
            device,
            seed=args.seed,
            warmup=args.large_warmup,
            repeats=args.large_repeats,
            target_total=args.large_target_total,
            target_valid=args.large_target_valid,
            prefix=args.large_prefix,
            heads=args.large_heads,
            tau=args.large_tau,
            thresh_type=args.large_thresh_type,
            step_index=args.large_step_index,
            layer_index=args.large_layer_index,
            min_gain_pct=args.prefix_skip_min_gain_pct,
        )
    if args.mode == "static-prefix-sink-bench":
        results["static_prefix_sink_bench"] = run_static_prefix_sink_bench(
            device,
            seed=args.seed,
            warmup=args.large_warmup,
            repeats=args.large_repeats,
            target_total=args.large_target_total,
            target_valid=args.large_target_valid,
            prefix=args.large_prefix,
            heads=args.large_heads,
            tau=args.large_tau,
            thresh_type=args.large_thresh_type,
            step_index=args.large_step_index,
            layer_index=args.large_layer_index,
            min_gain_pct=args.static_prefix_sink_min_gain_pct,
        )
    if args.mode == "bitmask-scheduler-bench":
        results["bitmask_scheduler_bench"] = run_bitmask_scheduler_bench(
            device,
            seed=args.seed,
            warmup=args.large_warmup,
            repeats=args.large_repeats,
            target_total=args.large_target_total,
            target_valid=args.large_target_valid,
            prefix=args.large_prefix,
            heads=args.large_heads,
            tau=args.large_tau,
            thresh_type=args.large_thresh_type,
            step_index=args.large_step_index,
            layer_index=args.large_layer_index,
            min_gain_pct=args.bitmask_scheduler_min_gain_pct,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
