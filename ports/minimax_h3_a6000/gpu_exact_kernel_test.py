#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""External single-A6000 correctness harness for MiniMax-H3 exact kernels.

This script intentionally loads no model weights.  It is an external GPU gate:
run only when the operator has granted one visible RTX A6000/SM86 device, e.g.
``--device cuda:0``.  Results are written as JSON with per-case max_abs,
max_rel, mismatch count, and Triton compile/launch status.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

import torch  # noqa: E402

from minimax_h3_a6000.exact_kernels import (  # noqa: E402
    apply_rope_bf16,
    indexed_gate_bf16,
    indexed_modulate_bf16,
    swiglu_bf16,
)
from minimax_h3_a6000.reference_ops import (  # noqa: E402
    apply_rope_bf16_reference,
    indexed_gate_bf16_reference,
    indexed_modulate_bf16_reference,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="must resolve to the only visible A6000, normally cuda:0")
    parser.add_argument("--output", required=True, type=Path, help="JSON result path")
    parser.add_argument("--seed", type=int, default=20260809)
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
    name = props.name.lower()
    if cap != (8, 6) or "a6000" not in name:
        raise RuntimeError(f"expected one A6000 SM86, saw name={props.name!r} capability={cap}")
    return torch.device("cuda:0")


_EXTREME_VALUES = (0.0, 1.0, -1.0, 448.0, -448.0, 1.0e-7, -1.0e-7, 3.14159)


def _bf_rand(shape: tuple[int, ...], device: torch.device, scale: float = 1.0) -> torch.Tensor:
    return (torch.randn(shape, device=device, dtype=torch.float32) * scale).to(torch.bfloat16).contiguous()


def _inject_extremes(tensor: torch.Tensor) -> torch.Tensor:
    """Place explicit edge values in every op input without changing its shape."""

    if tensor.numel() == 0:
        return tensor
    values = torch.tensor(_EXTREME_VALUES, device=tensor.device, dtype=torch.float32).to(tensor.dtype)
    n = min(tensor.numel(), values.numel())
    tensor.view(-1)[:n] = values[:n]
    return tensor


def _metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    torch.cuda.synchronize()
    cand = candidate.detach().to(torch.float32)
    ref = reference.detach().to(torch.float32)
    diff = (cand - ref).abs()
    denom = ref.abs().clamp_min(1.0e-7)
    mismatches = (candidate != reference) & ~(torch.isnan(cand) & torch.isnan(ref))
    return {
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "max_rel": float((diff / denom).max().item()) if diff.numel() else 0.0,
        "mismatch": int(mismatches.sum().item()),
        "numel": int(candidate.numel()),
    }


def _run_case(name: str, make_and_run: Callable[[], tuple[torch.Tensor, torch.Tensor]]) -> dict[str, Any]:
    started = time.time()
    try:
        candidate, reference = make_and_run()
        status = "compiled_and_launched"
        metrics = _metrics(candidate, reference)
    except Exception as exc:  # noqa: BLE001 - preserve failure in JSON gate output
        status = "failed"
        metrics = {"error": f"{type(exc).__name__}: {exc}"}
    return {"case": name, "compile_status": status, "elapsed_s": time.time() - started, **metrics}


def _swiglu_ref(x: torch.Tensor, order: str) -> torch.Tensor:
    half = x.shape[-1] // 2
    if order == "value_gate":
        value, gate = x[..., :half].float(), x[..., half:].float()
    else:
        gate, value = x[..., :half].float(), x[..., half:].float()
    activated = torch.nn.functional.silu(gate).to(torch.bfloat16).float()
    return (value * activated).to(torch.bfloat16).to(torch.float32).to(torch.bfloat16)


def _cases(device: torch.device) -> list[tuple[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]]]:
    cap = (8, 6)
    cases: list[tuple[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]]] = []

    for rows, hidden, tags in ((7, 129, 5), (33, 5376, 9)):
        def mod_case(rows: int = rows, hidden: int = hidden, tags: int = tags) -> tuple[torch.Tensor, torch.Tensor]:
            x = _inject_extremes(_bf_rand((rows, hidden), device))
            scale = _inject_extremes(_bf_rand((tags, hidden), device, scale=0.25))
            shift = _inject_extremes(_bf_rand((tags, hidden), device, scale=0.25))
            indices = (torch.arange(rows, device=device, dtype=torch.int64) % tags).contiguous()
            # Hit edge tags explicitly.
            if rows >= 2:
                indices[0] = 0
                indices[-1] = tags - 1
            ref = indexed_modulate_bf16_reference(x, scale, shift, indices)
            out = indexed_modulate_bf16(x, scale, shift, indices, enable=True, strict=True, device_capability=cap)
            return out, ref
        cases.append((f"indexed_modulate_rows{rows}_hidden{hidden}_tags{tags}", mod_case))

        def gate_case(rows: int = rows, hidden: int = hidden, tags: int = tags) -> tuple[torch.Tensor, torch.Tensor]:
            residual = _inject_extremes(_bf_rand((rows, hidden), device))
            branch = _inject_extremes(_bf_rand((rows, hidden), device))
            gate = _inject_extremes(_bf_rand((tags, hidden), device, scale=0.5))
            indices = torch.tensor([(i * 3) % tags for i in range(rows)], device=device, dtype=torch.int64).contiguous()
            ref = indexed_gate_bf16_reference(residual, gate, branch, indices)
            out = indexed_gate_bf16(residual, gate, branch, indices, enable=True, strict=True, device_capability=cap)
            return out, ref
        cases.append((f"indexed_gate_rows{rows}_hidden{hidden}_tags{tags}", gate_case))

    def mod_sliced_table_stride_case() -> tuple[torch.Tensor, torch.Tensor]:
        rows, hidden, tags = 13, 129, 5
        x = _inject_extremes(_bf_rand((rows, hidden), device))
        packed = _inject_extremes(_bf_rand((tags, hidden * 3), device, scale=0.25))
        scale = packed[:, hidden : 2 * hidden]
        shift = packed[:, 2 * hidden :]
        assert not scale.is_contiguous() and not shift.is_contiguous()
        indices = (torch.arange(rows, device=device, dtype=torch.int64) % tags).contiguous()
        ref = indexed_modulate_bf16_reference(x, scale, shift, indices)
        out = indexed_modulate_bf16(x, scale, shift, indices, enable=True, strict=True, device_capability=cap)
        return out, ref
    cases.append(("indexed_modulate_sliced_table_stride_aware", mod_sliced_table_stride_case))

    def gate_expanded_broadcast_case() -> tuple[torch.Tensor, torch.Tensor]:
        rows, hidden, tags = 17, 129, 7
        residual = _inject_extremes(_bf_rand((rows, hidden), device))
        branch_base = _inject_extremes(_bf_rand((1, hidden), device))
        branch = branch_base.expand(rows, hidden)
        gate_base = _inject_extremes(_bf_rand((1, hidden), device, scale=0.5))
        gate = gate_base.expand(tags, hidden)
        assert not branch.is_contiguous() and not gate.is_contiguous()
        indices = (torch.arange(rows, device=device, dtype=torch.int64) % tags).contiguous()
        ref = indexed_gate_bf16_reference(residual, gate, branch, indices)
        out = indexed_gate_bf16(residual, gate, branch, indices, enable=True, strict=True, device_capability=cap)
        return out, ref
    cases.append(("indexed_gate_expanded_broadcast_stride_aware", gate_expanded_broadcast_case))

    def mod_sliced_table_materialize_case() -> tuple[torch.Tensor, torch.Tensor]:
        rows, hidden, tags = 11, 257, 5
        x = _inject_extremes(_bf_rand((rows, hidden), device))
        packed = _inject_extremes(_bf_rand((tags, hidden * 2), device, scale=0.25))
        scale = packed[:, :hidden]
        shift = packed[:, hidden:]
        indices = (torch.arange(rows, device=device, dtype=torch.int64) % tags).contiguous()
        ref = indexed_modulate_bf16_reference(x, scale, shift, indices)
        out = indexed_modulate_bf16(
            x,
            scale,
            shift,
            indices,
            enable=True,
            strict=True,
            device_capability=cap,
            strategy="materialize",
        )
        return out, ref
    cases.append(("indexed_modulate_sliced_table_materialize", mod_sliced_table_materialize_case))

    for shape, rotary_dim in (((17, 3, 130), 96), ((2, 19, 56, 128), 96)):
        def rope_case(shape: tuple[int, ...] = shape, rotary_dim: int = rotary_dim) -> tuple[torch.Tensor, torch.Tensor]:
            hidden = _inject_extremes(_bf_rand(shape, device))
            seq = shape[0] if len(shape) == 3 else shape[1]
            freqs = torch.randn((seq, rotary_dim), device=device, dtype=torch.float32).contiguous()
            freqs[0, : min(4, rotary_dim)] = 0.0
            ref = apply_rope_bf16_reference(hidden, freqs)
            out = apply_rope_bf16(hidden, freqs, enable=True, strict=True, device_capability=cap)
            return out, ref
        cases.append((f"rope_shape{'x'.join(map(str, shape))}_rot{rotary_dim}", rope_case))

    for rows, width, order in ((11, 258, "value_gate"), (29, 28672, "gate_up")):
        def swiglu_case(rows: int = rows, width: int = width, order: str = order) -> tuple[torch.Tensor, torch.Tensor]:
            x = _inject_extremes(_bf_rand((rows, width), device))
            ref = _swiglu_ref(x, order)
            out = swiglu_bf16(x, order=order, enable=True, strict=True, device_capability=cap)
            return out, ref
        cases.append((f"swiglu_rows{rows}_width{width}_{order}", swiglu_case))

    return cases


def main() -> int:
    args = _parse_args()
    torch.manual_seed(args.seed)
    device = _validate_single_a6000(args.device)
    results = {
        "seed": args.seed,
        "device": args.device,
        "validated_single_a6000_sm86": True,
        "model_load": False,
        "coverage_tags": [
            "fixed_seed",
            "random_inputs",
            "explicit_extreme_values_per_op",
            "tag_index_edges",
            "non_aligned_tail_shapes",
            "representative_T_H_D_shapes",
            "sliced_table_strides",
            "expanded_broadcast_views",
            "explicit_materialize_strategy",
        ],
        "cases": [],
    }
    for name, fn in _cases(device):
        results["cases"].append(_run_case(name, fn))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    failures = [case for case in results["cases"] if case.get("compile_status") != "compiled_and_launched" or case.get("mismatch") != 0]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
