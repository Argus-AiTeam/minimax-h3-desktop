#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""External single-A6000 correctness/bench harness for H3 Sol-Attn.

This script is for the outer GPU2 experiment only. It loads no MiniMax-H3 model
weights (``model_load=False``), requires exactly one visible RTX A6000/SM86 CUDA
device, and writes JSON evidence. Correctness uses a very low ``tau`` to force
all Sol-Attn blocks exact, verifies no dense fallback occurred, and separately
checks that prefix query rows are replaced by dense SDPA while prefix KV rows are
kept as an exact sink.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

import torch  # noqa: E402

from minimax_h3_a6000.sol_attn_backend import (  # noqa: E402
    PackedH3Metadata,
    SolAttnPolicy,
    SolAttnTelemetry,
    dense_attention_packed_reference,
    sol_attn_h3_reference_or_decline,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="must resolve to the only visible A6000, normally cuda:0")
    parser.add_argument("--output", required=True, type=Path, help="JSON result path")
    parser.add_argument("--mode", choices=("correctness", "bench", "both"), default="both")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
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
    v = torch.randn_like(q).contiguous()
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
    policy = SolAttnPolicy(allow_sparse=True, strict=True, tau=-1.0e6, thresh_type="diag")
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
    reference = dense_attention_packed_reference(q, k, v, metadata=metadata)
    metrics = _metrics(out, reference, metadata.prefix_len, metadata.valid_length)
    return {
        "mode": "correctness",
        "elapsed_s": time.time() - started,
        "compile_status": "compiled_and_launched",
        "telemetry": telemetry.__dict__,
        "tolerance_note": "tau=-1e6 forces all route blocks exact; prefix query rows must equal dense exactly",
        **metrics,
    }


def _time_cuda(fn, *, warmup: int, repeats: int) -> dict[str, Any]:
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
    t = torch.tensor(times_ms, dtype=torch.float32)
    return {
        "warmup": warmup,
        "repeats": repeats,
        "median_ms": float(t.median().item()),
        "mean_ms": float(t.mean().item()),
        "min_ms": float(t.min().item()),
        "max_ms": float(t.max().item()),
    }


def run_bench(device: torch.device, *, warmup: int, repeats: int) -> dict[str, Any]:
    q, k, v, metadata = _case_tensors(device, total=512, heads=8, prefix=6, latent_grid=(1, 17, 26))
    sparse_policy = SolAttnPolicy(allow_sparse=True, strict=True, tau=1.0, thresh_type="diag")

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
        "preflight_telemetry": telemetry.__dict__,
    }


def main() -> int:
    args = _parse_args()
    torch.manual_seed(args.seed)
    device = _validate_single_a6000(args.device)
    results: dict[str, Any] = {
        "schema_version": "minimax_h3_a6000_sol_attn_sm86_harness_v1",
        "model_load": False,
        "seed": args.seed,
        "device": args.device,
        "capability": [8, 6],
    }
    if args.mode in ("correctness", "both"):
        results["correctness"] = run_correctness(device)
    if args.mode in ("bench", "both"):
        results["bench"] = run_bench(device, warmup=args.warmup, repeats=args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
