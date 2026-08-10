#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""External single-A6000 microbenchmark for MiniMax-H3 exact kernel candidates.

Compares PyTorch eager references with Triton candidate launchers and writes raw
latency JSON.  It is not an H3 end-to-end benchmark and loads no model weights.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

import torch  # noqa: E402

from minimax_h3_a6000.exact_kernels import (  # noqa: E402
    apply_rope_bf16,
    get_exact_kernel_telemetry,
    indexed_gate_bf16,
    indexed_modulate_bf16,
    reset_exact_kernel_telemetry,
    swiglu_bf16,
)
from minimax_h3_a6000.reference_ops import (  # noqa: E402
    apply_rope_bf16_reference,
    indexed_gate_bf16_reference,
    indexed_modulate_bf16_reference,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    return parser.parse_args()


def _validate_single_a6000(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type != "cuda" or (device.index not in (None, 0)):
        raise RuntimeError(f"benchmark requires --device cuda:0 with one visible GPU, got {device_arg!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one visible GPU, saw {torch.cuda.device_count()}")
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    if cap != (8, 6) or "a6000" not in props.name.lower():
        raise RuntimeError(f"expected one A6000 SM86, saw name={props.name!r} capability={cap}")
    return torch.device("cuda:0")


def _bf(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.randn(shape, device=device, dtype=torch.float32).to(torch.bfloat16).contiguous()


def _time_ms(fn: Callable[[], torch.Tensor], warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    latencies: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        # Keep the result live until after timing has completed.
        _ = out
        latencies.append(float(start.elapsed_time(end)))
    return latencies


def _swiglu_ref(x: torch.Tensor, order: str) -> torch.Tensor:
    half = x.shape[-1] // 2
    if order == "value_gate":
        value, gate = x[..., :half].float(), x[..., half:].float()
    else:
        gate, value = x[..., :half].float(), x[..., half:].float()
    activated = torch.nn.functional.silu(gate).to(torch.bfloat16).float()
    return (value * activated).to(torch.bfloat16)


def _benchmarks(device: torch.device) -> dict[str, tuple[Callable[[], torch.Tensor], Callable[[], torch.Tensor]]]:
    cap = (8, 6)
    rows, hidden, tags = 38247, 5376, 9
    x = _bf((rows, hidden), device)
    scale = _bf((tags, hidden), device)
    shift = _bf((tags, hidden), device)
    residual = _bf((rows, hidden), device)
    branch = _bf((rows, hidden), device)
    gate = _bf((tags, hidden), device)
    indices = (torch.arange(rows, device=device, dtype=torch.int64) % tags).contiguous()

    packed_tables = _bf((tags, hidden * 3), device)
    scale_sliced = packed_tables[:, hidden : 2 * hidden]
    shift_sliced = packed_tables[:, 2 * hidden :]
    gate_sliced = packed_tables[:, :hidden]
    branch_expanded = _bf((1, hidden), device).expand(rows, hidden)

    rope_hidden = _bf((1024, 56, 128), device)
    freqs = torch.randn((1024, 96), device=device, dtype=torch.float32).contiguous()
    swiglu_value_gate = _bf((8192, 2 * 14336), device)
    swiglu_gate_up = _bf((8192, 2 * 14336), device)

    return {
        "indexed_modulate_38247x5376": (
            lambda: indexed_modulate_bf16_reference(x, scale, shift, indices),
            lambda: indexed_modulate_bf16(x, scale, shift, indices, enable=True, strict=True, device_capability=cap),
        ),
        "indexed_gate_38247x5376": (
            lambda: indexed_gate_bf16_reference(residual, gate, branch, indices),
            lambda: indexed_gate_bf16(residual, gate, branch, indices, enable=True, strict=True, device_capability=cap),
        ),
        "indexed_modulate_sliced_table_stride_aware_38247x5376_copy_free": (
            lambda: indexed_modulate_bf16_reference(x, scale_sliced, shift_sliced, indices),
            lambda: indexed_modulate_bf16(x, scale_sliced, shift_sliced, indices, enable=True, strict=True, device_capability=cap),
        ),
        "indexed_modulate_sliced_table_materialize_38247x5376_copy_inclusive": (
            lambda: indexed_modulate_bf16_reference(x, scale_sliced, shift_sliced, indices),
            lambda: indexed_modulate_bf16(
                x,
                scale_sliced,
                shift_sliced,
                indices,
                enable=True,
                strict=True,
                device_capability=cap,
                strategy="materialize",
            ),
        ),
        "indexed_gate_sliced_gate_expanded_branch_stride_aware_38247x5376_copy_free": (
            lambda: indexed_gate_bf16_reference(residual, gate_sliced, branch_expanded, indices),
            lambda: indexed_gate_bf16(residual, gate_sliced, branch_expanded, indices, enable=True, strict=True, device_capability=cap),
        ),
        "indexed_gate_sliced_gate_expanded_branch_materialize_38247x5376_copy_inclusive": (
            lambda: indexed_gate_bf16_reference(residual, gate_sliced, branch_expanded, indices),
            lambda: indexed_gate_bf16(
                residual,
                gate_sliced,
                branch_expanded,
                indices,
                enable=True,
                strict=True,
                device_capability=cap,
                strategy="materialize",
            ),
        ),
        "rope_1024x56x128_rot96": (
            lambda: apply_rope_bf16_reference(rope_hidden, freqs),
            lambda: apply_rope_bf16(rope_hidden, freqs, enable=True, strict=True, device_capability=cap),
        ),
        "swiglu_value_gate_8192x28672": (
            lambda: _swiglu_ref(swiglu_value_gate, "value_gate"),
            lambda: swiglu_bf16(swiglu_value_gate, order="value_gate", enable=True, strict=True, device_capability=cap),
        ),
        "swiglu_gate_up_8192x28672": (
            lambda: _swiglu_ref(swiglu_gate_up, "gate_up"),
            lambda: swiglu_bf16(swiglu_gate_up, order="gate_up", enable=True, strict=True, device_capability=cap),
        ),
    }


def main() -> int:
    args = _parse_args()
    if args.warmup < 20 or args.repeats < 100:
        raise RuntimeError("benchmark contract requires warmup >= 20 and repeats >= 100")
    os.environ.setdefault("MINIMAX_H3_A6000_ENABLE_TELEMETRY", "1")
    torch.manual_seed(args.seed)
    device = _validate_single_a6000(args.device)
    results = {
        "seed": args.seed,
        "device": args.device,
        "validated_single_a6000_sm86": True,
        "model_load": False,
        "scope": "kernel_candidates_only_not_h3_e2e",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "benchmarks": {},
    }
    for name, (eager, triton_candidate) in _benchmarks(device).items():
        eager_ms = _time_ms(eager, args.warmup, args.repeats)
        reset_exact_kernel_telemetry()
        candidate_ms = _time_ms(triton_candidate, args.warmup, args.repeats)
        results["benchmarks"][name] = {
            "pytorch_eager_ms": eager_ms,
            "triton_candidate_ms": candidate_ms,
            "candidate_telemetry_including_warmup": get_exact_kernel_telemetry(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
