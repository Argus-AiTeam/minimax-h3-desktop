# SPDX-License-Identifier: Apache-2.0
#
# Adapted from NVlabs/Sana sol-engine MiniMax-H3 Triton fusion work
# (Apache-2.0).  This A6000/SM86 candidate keeps only BF16 exact
# elementwise kernels: indexed AdaLN modulation, indexed gated residual, H3
# leading-channel RoPE, and SwiGLU.  Checkpoint layout conversion paths are not
# implemented here.
"""Default-off Triton exact-kernel candidates for MiniMax-H3 on A6000 SM86.

The public launchers never probe CUDA at import time and always fall back to the
PyTorch references when the opt-in environment, tensor contract, Triton import,
or SM86 guard is not satisfied.  Pass ``enable=True`` from an authorized GPU
harness to force the candidate path; pass ``strict=True`` to turn an unsupported
candidate into an exception instead of a reference fallback.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
from pathlib import Path
from typing import Any, Literal, NamedTuple

import torch

from .env import DEFAULT_ENV_SWITCHES
from .reference_ops import (
    apply_rope_bf16_reference,
    indexed_gate_bf16_reference,
    indexed_modulate_bf16_reference,
    swiglu_bf16_reference,
)

_BF16 = torch.bfloat16
_MAX_ROW_BLOCK = 8192
_TILE = 1024
_MAX_LAYOUT_SAMPLES = 64
_MAX_SHADOW_SAMPLES = 32
_TRITON_CACHE: tuple[Any, dict[str, Any]] | None = None
_TELEMETRY_OPS = (
    "indexed_modulate_bf16",
    "indexed_gate_bf16",
    "apply_rope_bf16",
    "swiglu_bf16",
)
_ABLATION_DISABLE_ENV_BY_OP = {
    "indexed_modulate_bf16": "MINIMAX_H3_A6000_ABLATION_DISABLE_INDEXED_MODULATE",
    "indexed_gate_bf16": "MINIMAX_H3_A6000_ABLATION_DISABLE_INDEXED_GATE",
    "apply_rope_bf16": "MINIMAX_H3_A6000_ABLATION_DISABLE_ROPE",
    "swiglu_bf16": "MINIMAX_H3_A6000_ABLATION_DISABLE_SWIGLU",
}
_TELEMETRY_LOCK = threading.Lock()
_TELEMETRY_ATEXIT_REGISTERED = False
_TELEMETRY: dict[str, Any] = {
    "schema_version": "minimax_h3_a6000_exact_telemetry_v1",
    "pid": os.getpid(),
    "ops": {
        op: {
            "calls": 0,
            "candidate": 0,
            "fallback": 0,
            "decline": 0,
            "strict_error": 0,
            "reasons": {},
            "strategies": {"stride_aware": 0, "materialize": 0},
            "materialize_copy_calls": 0,
            "materialize_copy_bytes": 0,
            "materialize_copy_by_tensor": {},
            "materialized_tensors": {},
            "tensor_layouts": {},
            "tensor_layout_summary": {},
            "tensor_layout_samples": [],
            "shadow": {
                "comparisons": 0,
                "mismatches": 0,
                "strict_error": 0,
                "reference_error": 0,
                "max_abs": 0.0,
                "max_rel": 0.0,
                "samples": [],
            },
        }
        for op in _TELEMETRY_OPS
    },
}


class KernelSupport(NamedTuple):
    supported: bool
    reason: str


class ShadowMismatchError(RuntimeError):
    """Raised when opt-in shadow comparison is strict and detects drift."""


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, DEFAULT_ENV_SWITCHES.get(name, "0")) == "1"


def _op_enabled(op_env: str | tuple[str, ...], override: bool | None) -> bool:
    if override is not None:
        return bool(override)
    op_specific_enabled = _env_enabled(op_env) if isinstance(op_env, str) else any(_env_enabled(name) for name in op_env)
    return (
        _env_enabled("MINIMAX_H3_A6000_ENABLE_OVERLAY")
        and _env_enabled("MINIMAX_H3_A6000_ENABLE_TRITON_CANDIDATES")
        and op_specific_enabled
    )


def _ablation_disable_reason(op_name: str) -> str | None:
    env_name = _ABLATION_DISABLE_ENV_BY_OP.get(op_name)
    if env_name and _env_enabled(env_name):
        return f"{op_name}: disabled by per-kernel ablation env {env_name}"
    return None


def exact_kernel_telemetry_enabled() -> bool:
    """Return whether process-local exact-kernel telemetry is recording."""

    return _env_enabled("MINIMAX_H3_A6000_ENABLE_TELEMETRY") or bool(os.environ.get("MINIMAX_H3_A6000_TELEMETRY_JSON"))


def exact_kernel_shadow_enabled() -> bool:
    """Return whether opt-in vLLM eager shadow comparisons are enabled."""

    return _env_enabled("MINIMAX_H3_A6000_ENABLE_SHADOW")


def exact_kernel_shadow_strict() -> bool:
    """Return whether shadow mismatches should abort the wrapper call."""

    return _env_enabled("MINIMAX_H3_A6000_SHADOW_STRICT")


def _shadow_call_limit() -> int:
    raw = os.environ.get(
        "MINIMAX_H3_A6000_SHADOW_CALLS",
        DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_SHADOW_CALLS", "3"),
    )
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _telemetry_path() -> str | None:
    path = os.environ.get("MINIMAX_H3_A6000_TELEMETRY_JSON", DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_TELEMETRY_JSON", ""))
    return path or None


def _ensure_telemetry_atexit() -> None:
    global _TELEMETRY_ATEXIT_REGISTERED
    if _TELEMETRY_ATEXIT_REGISTERED or not exact_kernel_telemetry_enabled():
        return
    if not (_env_enabled("MINIMAX_H3_A6000_TELEMETRY_ATEXIT") and _telemetry_path()):
        return
    atexit.register(write_exact_kernel_telemetry_json)
    _TELEMETRY_ATEXIT_REGISTERED = True


def _record_call(op_name: str) -> None:
    if not exact_kernel_telemetry_enabled():
        return
    _ensure_telemetry_atexit()
    with _TELEMETRY_LOCK:
        _TELEMETRY["ops"][op_name]["calls"] += 1


def _record_outcome(op_name: str, outcome: Literal["candidate", "fallback", "decline", "strict_error"], reason: str | None = None) -> None:
    if not exact_kernel_telemetry_enabled():
        return
    _ensure_telemetry_atexit()
    with _TELEMETRY_LOCK:
        op = _TELEMETRY["ops"][op_name]
        op[outcome] += 1
        if reason:
            reasons = op["reasons"]
            reasons[reason] = reasons.get(reason, 0) + 1


def _record_strategy(op_name: str, strategy: Literal["stride_aware", "materialize"]) -> None:
    if not exact_kernel_telemetry_enabled():
        return
    _ensure_telemetry_atexit()
    with _TELEMETRY_LOCK:
        strategies = _TELEMETRY["ops"][op_name]["strategies"]
        strategies[strategy] = strategies.get(strategy, 0) + 1


def _layout_key(name: str, tensor: torch.Tensor) -> str:
    return (
        f"{name}: shape={tuple(tensor.shape)} stride={tuple(tensor.stride())} "
        f"storage_offset={tensor.storage_offset()} contiguous={tensor.is_contiguous()}"
    )


def _tensor_layout_record(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    strides = [int(stride) for stride in tensor.stride()]
    zero_stride = any(stride == 0 for stride in strides)
    return {
        "name": name,
        "shape": [int(dim) for dim in tensor.shape],
        "stride": strides,
        "storage_offset": int(tensor.storage_offset()),
        "numel": int(tensor.numel()),
        "element_size": int(tensor.element_size()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "ndim": int(tensor.ndim),
        "is_contiguous": bool(tensor.is_contiguous()),
        "has_zero_stride": bool(zero_stride),
    }


def _record_tensor_layouts(op_name: str, named_tensors: tuple[tuple[str, torch.Tensor], ...]) -> None:
    if not exact_kernel_telemetry_enabled():
        return
    _ensure_telemetry_atexit()
    with _TELEMETRY_LOCK:
        op = _TELEMETRY["ops"][op_name]
        layouts = op["tensor_layouts"]
        summary = op["tensor_layout_summary"]
        samples = op["tensor_layout_samples"]
        for name, tensor in named_tensors:
            record = _tensor_layout_record(name, tensor)
            bucket = summary.setdefault(
                name,
                {
                    "seen": 0,
                    "contiguous": 0,
                    "noncontiguous": 0,
                    "zero_stride": 0,
                    "last": None,
                },
            )
            bucket["seen"] += 1
            if record["is_contiguous"]:
                bucket["contiguous"] += 1
            else:
                bucket["noncontiguous"] += 1
                key = _layout_key(name, tensor)
                layouts[key] = layouts.get(key, 0) + 1
            if record["has_zero_stride"]:
                bucket["zero_stride"] += 1
            bucket["last"] = record
            if len(samples) < _MAX_LAYOUT_SAMPLES:
                samples.append(record)


def _record_materialized_tensor(op_name: str, name: str, tensor: torch.Tensor) -> None:
    if not exact_kernel_telemetry_enabled() or tensor.is_contiguous():
        return
    _ensure_telemetry_atexit()
    bytes_copied = int(tensor.numel() * tensor.element_size())
    with _TELEMETRY_LOCK:
        op = _TELEMETRY["ops"][op_name]
        op["materialize_copy_calls"] += 1
        op["materialize_copy_bytes"] += bytes_copied
        by_tensor = op["materialize_copy_by_tensor"].setdefault(name, {"calls": 0, "bytes": 0})
        by_tensor["calls"] += 1
        by_tensor["bytes"] += bytes_copied
        key = _layout_key(name, tensor)
        materialized = op["materialized_tensors"]
        materialized[key] = materialized.get(key, 0) + 1


def _contiguous_for_materialize(op_name: str, named_tensors: tuple[tuple[str, torch.Tensor], ...]) -> tuple[torch.Tensor, ...]:
    out: list[torch.Tensor] = []
    for name, tensor in named_tensors:
        _record_materialized_tensor(op_name, name, tensor)
        out.append(tensor if tensor.is_contiguous() else tensor.contiguous())
    return tuple(out)


def reset_exact_kernel_telemetry() -> None:
    """Reset process-local exact-kernel counters; useful for isolated tests."""

    with _TELEMETRY_LOCK:
        for op in _TELEMETRY["ops"].values():
            op["calls"] = op["candidate"] = op["fallback"] = op["decline"] = op["strict_error"] = 0
            op["reasons"].clear()
            op["strategies"]["stride_aware"] = 0
            op["strategies"]["materialize"] = 0
            op["materialize_copy_calls"] = 0
            op["materialize_copy_bytes"] = 0
            op["materialize_copy_by_tensor"].clear()
            op["materialized_tensors"].clear()
            op["tensor_layouts"].clear()
            op["tensor_layout_summary"].clear()
            op["tensor_layout_samples"].clear()
            shadow = op["shadow"]
            shadow["comparisons"] = shadow["mismatches"] = shadow["strict_error"] = shadow["reference_error"] = 0
            shadow["max_abs"] = 0.0
            shadow["max_rel"] = 0.0
            shadow["samples"].clear()


def get_exact_kernel_telemetry() -> dict[str, Any]:
    """Return a JSON-serializable snapshot of process-local exact telemetry."""

    with _TELEMETRY_LOCK:
        snapshot = json.loads(json.dumps(_TELEMETRY, sort_keys=True))
    snapshot["enabled"] = exact_kernel_telemetry_enabled()
    snapshot["telemetry_json"] = _telemetry_path()
    snapshot["shadow_enabled"] = exact_kernel_shadow_enabled()
    snapshot["shadow_call_limit"] = _shadow_call_limit()
    snapshot["shadow_strict"] = exact_kernel_shadow_strict()
    return snapshot


def write_exact_kernel_telemetry_json(path: str | os.PathLike[str] | None = None) -> Path:
    """Export telemetry JSON to ``path`` or ``MINIMAX_H3_A6000_TELEMETRY_JSON``."""

    raw_path = path or _telemetry_path()
    if not raw_path:
        raise RuntimeError("MINIMAX_H3_A6000_TELEMETRY_JSON is not set")
    target = Path(raw_path)
    if target.parent != Path("") and str(target.parent) != ".":
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(get_exact_kernel_telemetry(), indent=2, sort_keys=True), encoding="utf-8")
    return target


def _raise_or_false(reason: str, strict: bool) -> KernelSupport:
    if strict:
        raise RuntimeError(reason)
    return KernelSupport(False, reason)


def _device_capability(tensor: torch.Tensor, explicit: tuple[int, int] | None) -> tuple[int, int] | None:
    if explicit is not None:
        return explicit
    if not tensor.is_cuda:
        return None
    # Deliberately launcher-only: importing this module must not initialize CUDA.
    return tuple(int(x) for x in torch.cuda.get_device_capability(tensor.device))  # type: ignore[return-value]


def _same_device(*tensors: torch.Tensor) -> bool:
    if not tensors:
        return True
    device = tensors[0].device
    return all(t.device == device for t in tensors)


def _strides_non_negative(*tensors: torch.Tensor) -> bool:
    return all(all(int(stride) >= 0 for stride in tensor.stride()) for tensor in tensors)


def _indexed_strided_layout_supported(payloads: tuple[torch.Tensor, ...], indices: torch.Tensor) -> bool:
    # The direct Triton indexed kernels flatten rows with one row stride and one
    # column stride.  This intentionally covers the r3 integration failures:
    # sliced AdaLN tables (larger row stride/storage offset) and expanded views
    # (zero stride).  Higher-rank payloads remain exact candidates via the
    # explicit materialize strategy so their copy cost is visible.
    return all(tensor.ndim == 2 for tensor in payloads) and indices.ndim == 1 and _strides_non_negative(*payloads, indices)


def _indexed_materialize_layout_supported(payloads: tuple[torch.Tensor, ...], indices: torch.Tensor) -> bool:
    return all(tensor.ndim >= 2 for tensor in payloads) and indices.ndim == 1


def _indices_contract(rows: int, table_rows: int, indices: torch.Tensor, *, require_contiguous: bool = True) -> str | None:
    if indices.ndim != 1 or indices.numel() != rows:
        return f"indices must be one per flattened row: rows={rows}, indices={tuple(indices.shape)}"
    if indices.dtype not in (torch.int32, torch.int64, torch.long):
        return f"indices must be int32/int64, got {indices.dtype}"
    if require_contiguous and not indices.is_contiguous():
        return "indices must be contiguous for Triton candidate"
    if indices.numel():
        lo = int(indices.min().item())
        hi = int(indices.max().item())
        if lo < 0 or hi >= table_rows:
            return f"index range [{lo}, {hi}] outside table rows {table_rows}"
    return None


def _support_common(
    op_name: str,
    op_env: str | tuple[str, ...],
    payload: torch.Tensor,
    tensors: tuple[torch.Tensor, ...],
    *,
    enable: bool | None,
    strict: bool,
    device_capability: tuple[int, int] | None,
    require_contiguous: bool = True,
) -> KernelSupport:
    if not _op_enabled(op_env, enable):
        return _raise_or_false(f"{op_name}: disabled by default-off environment", strict)
    if not payload.is_cuda:
        return _raise_or_false(f"{op_name}: tensor device must be CUDA", strict)
    cap = _device_capability(payload, device_capability)
    if cap != (8, 6):
        return _raise_or_false(f"{op_name}: requires SM86/A6000, got capability {cap}", strict)
    if not _same_device(payload, *tensors):
        return _raise_or_false(f"{op_name}: all tensors must be on the same device", strict)
    if require_contiguous:
        for tensor in (payload, *tensors):
            if not tensor.is_contiguous():
                return _raise_or_false(f"{op_name}: all tensors must be contiguous", strict)
    return KernelSupport(True, "supported")


def explain_indexed_modulate_support(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
    *,
    enable: bool | None = None,
    device_capability: tuple[int, int] | None = None,
) -> KernelSupport:
    common = _support_common(
        "indexed_modulate_bf16",
        ("MINIMAX_H3_A6000_ENABLE_FUSED_ADALN", "MINIMAX_H3_A6000_ENABLE_FUSED_ADALN_MODULATE"),
        x,
        (scale, shift, indices),
        enable=enable,
        strict=False,
        device_capability=device_capability,
        require_contiguous=False,
    )
    if not common.supported:
        return common
    if x.dtype != _BF16 or scale.dtype != _BF16 or shift.dtype != _BF16:
        return KernelSupport(False, f"indexed_modulate_bf16: payload tensors must be BF16")
    if x.ndim < 2:
        return KernelSupport(False, f"indexed_modulate_bf16: x must have at least 2 dims, got {tuple(x.shape)}")
    rows = x.numel() // x.shape[-1]
    cols = x.shape[-1]
    if scale.ndim != 2 or shift.shape != scale.shape or scale.shape[-1] != cols:
        return KernelSupport(False, f"indexed_modulate_bf16: table shapes must be [tags,{cols}]")
    if cols <= 0 or cols > _MAX_ROW_BLOCK:
        return KernelSupport(False, f"indexed_modulate_bf16: hidden width {cols} exceeds candidate block {_MAX_ROW_BLOCK}")
    bad = _indices_contract(rows, scale.shape[0], indices, require_contiguous=False)
    if bad:
        return KernelSupport(False, f"indexed_modulate_bf16: {bad}")
    payloads = (x, scale, shift)
    if not (_indexed_strided_layout_supported(payloads, indices) or _indexed_materialize_layout_supported(payloads, indices)):
        return KernelSupport(False, "indexed_modulate_bf16: layout is neither 2D stride-aware nor materializable")
    return KernelSupport(True, "supported")


def explain_indexed_gate_support(
    residual: torch.Tensor,
    gate: torch.Tensor,
    branch: torch.Tensor,
    indices: torch.Tensor,
    *,
    enable: bool | None = None,
    device_capability: tuple[int, int] | None = None,
) -> KernelSupport:
    common = _support_common(
        "indexed_gate_bf16",
        ("MINIMAX_H3_A6000_ENABLE_FUSED_ADALN", "MINIMAX_H3_A6000_ENABLE_FUSED_ADALN_GATE"),
        residual,
        (gate, branch, indices),
        enable=enable,
        strict=False,
        device_capability=device_capability,
        require_contiguous=False,
    )
    if not common.supported:
        return common
    if residual.dtype != _BF16 or gate.dtype != _BF16 or branch.dtype != _BF16:
        return KernelSupport(False, "indexed_gate_bf16: payload tensors must be BF16")
    if residual.shape != branch.shape or residual.ndim < 2:
        return KernelSupport(False, f"indexed_gate_bf16: residual/branch shape mismatch {tuple(residual.shape)} vs {tuple(branch.shape)}")
    rows = residual.numel() // residual.shape[-1]
    cols = residual.shape[-1]
    if gate.ndim != 2 or gate.shape[-1] != cols:
        return KernelSupport(False, f"indexed_gate_bf16: gate shape must be [tags,{cols}]")
    if cols <= 0 or cols > _MAX_ROW_BLOCK:
        return KernelSupport(False, f"indexed_gate_bf16: hidden width {cols} exceeds candidate block {_MAX_ROW_BLOCK}")
    bad = _indices_contract(rows, gate.shape[0], indices, require_contiguous=False)
    if bad:
        return KernelSupport(False, f"indexed_gate_bf16: {bad}")
    payloads = (residual, gate, branch)
    if not (_indexed_strided_layout_supported(payloads, indices) or _indexed_materialize_layout_supported(payloads, indices)):
        return KernelSupport(False, "indexed_gate_bf16: layout is neither 2D stride-aware nor materializable")
    return KernelSupport(True, "supported")


def explain_rope_support(
    hidden_states: torch.Tensor,
    freqs: torch.Tensor,
    *,
    enable: bool | None = None,
    device_capability: tuple[int, int] | None = None,
) -> KernelSupport:
    common = _support_common(
        "apply_rope_bf16",
        "MINIMAX_H3_A6000_ENABLE_FUSED_ROPE",
        hidden_states,
        (freqs,),
        enable=enable,
        strict=False,
        device_capability=device_capability,
    )
    if not common.supported:
        return common
    if hidden_states.dtype != _BF16:
        return KernelSupport(False, "apply_rope_bf16: hidden_states must be BF16")
    if freqs.dtype != torch.float32:
        return KernelSupport(False, f"apply_rope_bf16: freqs must be FP32, got {freqs.dtype}")
    if hidden_states.ndim not in (3, 4):
        return KernelSupport(False, f"apply_rope_bf16: hidden_states must be [T,H,D] or [B,T,H,D], got {tuple(hidden_states.shape)}")
    seq_len = hidden_states.shape[0] if hidden_states.ndim == 3 else hidden_states.shape[1]
    if freqs.ndim != 2 or freqs.shape[0] != seq_len:
        return KernelSupport(False, f"apply_rope_bf16: freqs must be [T,R] with T={seq_len}, got {tuple(freqs.shape)}")
    head_dim = hidden_states.shape[-1]
    rotary_dim = freqs.shape[-1]
    if rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > head_dim:
        return KernelSupport(False, f"apply_rope_bf16: invalid rotary_dim={rotary_dim} for head_dim={head_dim}")
    return KernelSupport(True, "supported")


def explain_swiglu_support(
    x: torch.Tensor,
    *,
    enable: bool | None = None,
    device_capability: tuple[int, int] | None = None,
) -> KernelSupport:
    common = _support_common(
        "swiglu_bf16",
        "MINIMAX_H3_A6000_ENABLE_FUSED_SWIGLU",
        x,
        (),
        enable=enable,
        strict=False,
        device_capability=device_capability,
    )
    if not common.supported:
        return common
    if x.dtype != _BF16:
        return KernelSupport(False, "swiglu_bf16: x must be BF16")
    if x.ndim < 2 or x.shape[-1] % 2:
        return KernelSupport(False, f"swiglu_bf16: last dimension must be even, got {tuple(x.shape)}")
    return KernelSupport(True, "supported")


def _ensure_triton_kernels() -> tuple[Any, dict[str, Any]]:
    global _TRITON_CACHE
    if _TRITON_CACHE is not None:
        return _TRITON_CACHE

    import triton
    import triton.language as tl

    # Triton's JIT source parser resolves ``tl`` annotations through the
    # function globals on current torch/triton wheels, not this local import
    # frame.  Keep the import lazy (no CUDA probe at module import) but publish
    # the language module before defining nested @triton.jit kernels.
    globals()["tl"] = tl

    # From Sana sol-engine's exact BF16 fusions: Triton may fold
    # x.to(tl.bfloat16).to(tl.float32) away, so intermediate BF16 rounding is
    # forced in the integer domain with round-to-nearest-even semantics.
    @triton.jit
    def _round_bf16(x):
        bits = x.to(tl.int32, bitcast=True)
        bits = bits + 0x7FFF + ((bits >> 16) & 1)
        return (bits & -65536).to(tl.float32, bitcast=True)

    globals()["_round_bf16"] = _round_bf16

    @triton.jit
    def _indexed_modulate_kernel(x_ptr, scale_ptr, shift_ptr, idx_ptr, out_ptr, n_cols: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        table_row = tl.load(idx_ptr + row)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + table_row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + table_row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        out = _round_bf16(_round_bf16(x * _round_bf16(1.0 + scale)) + shift)
        tl.store(out_ptr + row * n_cols + cols, out.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _indexed_gate_kernel(residual_ptr, gate_ptr, branch_ptr, idx_ptr, out_ptr, n_cols: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        table_row = tl.load(idx_ptr + row)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        residual = tl.load(residual_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        branch = tl.load(branch_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + table_row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        out = _round_bf16(residual + _round_bf16(gate * branch))
        tl.store(out_ptr + row * n_cols + cols, out.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _indexed_modulate_strided_kernel(
        x_ptr,
        scale_ptr,
        shift_ptr,
        idx_ptr,
        out_ptr,
        n_cols: tl.constexpr,
        x_row_stride: tl.constexpr,
        x_col_stride: tl.constexpr,
        scale_row_stride: tl.constexpr,
        scale_col_stride: tl.constexpr,
        shift_row_stride: tl.constexpr,
        shift_col_stride: tl.constexpr,
        idx_stride: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        table_row = tl.load(idx_ptr + row * idx_stride)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * x_row_stride + cols * x_col_stride, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + table_row * scale_row_stride + cols * scale_col_stride, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + table_row * shift_row_stride + cols * shift_col_stride, mask=mask, other=0.0).to(tl.float32)
        out = _round_bf16(_round_bf16(x * _round_bf16(1.0 + scale)) + shift)
        tl.store(out_ptr + row * n_cols + cols, out.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _indexed_gate_strided_kernel(
        residual_ptr,
        gate_ptr,
        branch_ptr,
        idx_ptr,
        out_ptr,
        n_cols: tl.constexpr,
        residual_row_stride: tl.constexpr,
        residual_col_stride: tl.constexpr,
        gate_row_stride: tl.constexpr,
        gate_col_stride: tl.constexpr,
        branch_row_stride: tl.constexpr,
        branch_col_stride: tl.constexpr,
        idx_stride: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        table_row = tl.load(idx_ptr + row * idx_stride)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        residual = tl.load(residual_ptr + row * residual_row_stride + cols * residual_col_stride, mask=mask, other=0.0).to(tl.float32)
        branch = tl.load(branch_ptr + row * branch_row_stride + cols * branch_col_stride, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + table_row * gate_row_stride + cols * gate_col_stride, mask=mask, other=0.0).to(tl.float32)
        out = _round_bf16(residual + _round_bf16(gate * branch))
        tl.store(out_ptr + row * n_cols + cols, out.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, seq_len: tl.constexpr, width: tl.constexpr, head_dim: tl.constexpr, rotary_dim: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < width
        channel = cols % head_dim
        head_start = cols - channel
        half = rotary_dim // 2
        rotating = channel < rotary_dim
        low = channel < half

        x = tl.load(x_ptr + row * width + cols, mask=mask, other=0.0).to(tl.float32)
        partner = head_start + tl.where(low, channel + half, channel - half)
        paired = tl.load(x_ptr + row * width + partner, mask=mask & rotating, other=0.0).to(tl.float32)
        rotated = tl.where(low, -paired, paired)
        seq = row % seq_len
        freq_channel = channel % half
        cos = tl.load(cos_ptr + seq * half + freq_channel, mask=mask & rotating, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + seq * half + freq_channel, mask=mask & rotating, other=0.0).to(tl.float32)
        y = _round_bf16(_round_bf16(x * cos) + _round_bf16(rotated * sin))
        tl.store(out_ptr + row * width + cols, tl.where(rotating, y, x).to(tl.bfloat16), mask=mask)

    @triton.jit
    def _swiglu_kernel(x_ptr, out_ptr, half: tl.constexpr, value_first: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < half
        a = tl.load(x_ptr + row * 2 * half + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(x_ptr + row * 2 * half + half + cols, mask=mask, other=0.0).to(tl.float32)
        if value_first:
            value = a
            gate = b
        else:
            gate = a
            value = b
        activated = _round_bf16(gate * tl.sigmoid(gate))
        tl.store(out_ptr + row * half + cols, _round_bf16(value * activated).to(tl.bfloat16), mask=mask)

    _TRITON_CACHE = (
        triton,
        {
            "indexed_modulate": _indexed_modulate_kernel,
            "indexed_gate": _indexed_gate_kernel,
            "indexed_modulate_strided": _indexed_modulate_strided_kernel,
            "indexed_gate_strided": _indexed_gate_strided_kernel,
            "rope": _rope_kernel,
            "swiglu": _swiglu_kernel,
        },
    )
    return _TRITON_CACHE


def _block_for_cols(cols: int) -> int:
    return 1 << (int(cols) - 1).bit_length()


def _indexed_strategy(requested: Literal["auto", "stride_aware", "materialize"] | None) -> Literal["auto", "stride_aware", "materialize"]:
    raw = requested or os.environ.get(
        "MINIMAX_H3_A6000_EXACT_INDEXED_STRATEGY",
        DEFAULT_ENV_SWITCHES.get("MINIMAX_H3_A6000_EXACT_INDEXED_STRATEGY", "auto"),
    )
    if raw not in ("auto", "stride_aware", "materialize"):
        raise ValueError(f"unsupported indexed exact strategy {raw!r}")
    return raw  # type: ignore[return-value]


def _select_indexed_strategy(
    requested: Literal["auto", "stride_aware", "materialize"] | None,
    payloads: tuple[torch.Tensor, ...],
    indices: torch.Tensor,
) -> KernelSupport:
    strategy = _indexed_strategy(requested)
    stride_ok = _indexed_strided_layout_supported(payloads, indices)
    if strategy == "stride_aware" and not stride_ok:
        return KernelSupport(False, "requested stride_aware strategy does not support this layout")
    if strategy == "stride_aware" or (strategy == "auto" and stride_ok):
        return KernelSupport(True, "stride_aware")
    if _indexed_materialize_layout_supported(payloads, indices):
        return KernelSupport(True, "materialize")
    return KernelSupport(False, "layout is not materializable")


def _fallback_or_raise(
    op_name: str,
    reason: str,
    strict: bool,
    fallback: Any,
    *,
    declined: bool = False,
) -> Any:
    if declined:
        _record_outcome(op_name, "decline", reason)
    if strict:
        _record_outcome(op_name, "strict_error", reason)
        raise RuntimeError(reason)
    _record_outcome(op_name, "fallback", reason)
    return fallback()


def _shape_list(tensor: torch.Tensor) -> list[int]:
    return [int(dim) for dim in tensor.shape]


def _reserve_shadow_comparison(op_name: str) -> int | None:
    if not exact_kernel_shadow_enabled():
        return None
    limit = _shadow_call_limit()
    if limit <= 0:
        return None
    _ensure_telemetry_atexit()
    with _TELEMETRY_LOCK:
        shadow = _TELEMETRY["ops"][op_name]["shadow"]
        if shadow["comparisons"] >= limit:
            return None
        shadow["comparisons"] += 1
        return int(shadow["comparisons"])


def _record_shadow_sample(op_name: str, sample: dict[str, Any]) -> None:
    with _TELEMETRY_LOCK:
        shadow = _TELEMETRY["ops"][op_name]["shadow"]
        if sample.get("bitwise_mismatch"):
            shadow["mismatches"] += 1
        if sample.get("reference_error"):
            shadow["reference_error"] += 1
        if sample.get("strict_error"):
            shadow["strict_error"] += 1
        max_abs = sample.get("max_abs")
        max_rel = sample.get("max_rel")
        if isinstance(max_abs, (int, float)):
            shadow["max_abs"] = max(float(shadow["max_abs"]), float(max_abs))
        if isinstance(max_rel, (int, float)):
            shadow["max_rel"] = max(float(shadow["max_rel"]), float(max_rel))
        samples = shadow["samples"]
        if len(samples) < _MAX_SHADOW_SAMPLES:
            samples.append(sample)


def _shadow_compare_or_raise(op_name: str, candidate: torch.Tensor, reference: Any) -> None:
    comparison_index = _reserve_shadow_comparison(op_name)
    if comparison_index is None:
        return
    try:
        eager = reference()
    except Exception as exc:  # pragma: no cover - reference failures are data-contract bugs
        sample = {
            "comparison": comparison_index,
            "reference_error": repr(exc),
            "candidate_shape": _shape_list(candidate),
            "candidate_dtype": str(candidate.dtype),
            "candidate_device": str(candidate.device),
        }
        if exact_kernel_shadow_strict():
            sample["strict_error"] = True
            _record_shadow_sample(op_name, sample)
            raise ShadowMismatchError(f"{op_name}: shadow reference failed: {exc}") from exc
        _record_shadow_sample(op_name, sample)
        return

    same_shape = tuple(candidate.shape) == tuple(eager.shape)
    same_dtype = candidate.dtype == eager.dtype
    bitwise_equal = bool(same_shape and same_dtype and torch.equal(candidate, eager))
    max_abs: float | None = None
    max_rel: float | None = None
    if same_shape and candidate.numel() and eager.numel() and candidate.is_floating_point() and eager.is_floating_point():
        diff = (candidate.detach().to(torch.float32) - eager.detach().to(torch.float32)).abs()
        max_abs = float(diff.max().item())
        denom = eager.detach().to(torch.float32).abs().clamp_min(1.0e-30)
        max_rel = float((diff / denom).max().item())
    elif same_shape:
        max_abs = 0.0 if bitwise_equal else None
        max_rel = 0.0 if bitwise_equal else None

    sample = {
        "comparison": comparison_index,
        "bitwise_mismatch": not bitwise_equal,
        "candidate_shape": _shape_list(candidate),
        "reference_shape": _shape_list(eager),
        "candidate_dtype": str(candidate.dtype),
        "reference_dtype": str(eager.dtype),
        "candidate_device": str(candidate.device),
        "reference_device": str(eager.device),
        "max_abs": max_abs,
        "max_rel": max_rel,
    }
    if not bitwise_equal and exact_kernel_shadow_strict():
        sample["strict_error"] = True
        _record_shadow_sample(op_name, sample)
        raise ShadowMismatchError(
            f"{op_name}: shadow mismatch candidate_shape={tuple(candidate.shape)} "
            f"reference_shape={tuple(eager.shape)} max_abs={max_abs} max_rel={max_rel}"
        )
    _record_shadow_sample(op_name, sample)


def indexed_modulate_bf16(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
    *,
    enable: bool | None = None,
    strict: bool = False,
    device_capability: tuple[int, int] | None = None,
    strategy: Literal["auto", "stride_aware", "materialize"] | None = None,
) -> torch.Tensor:
    """Return ``x * (1 + scale[indices]) + shift[indices]`` with BF16 eager rounding."""

    op_name = "indexed_modulate_bf16"
    _record_call(op_name)
    _record_tensor_layouts(op_name, (("x", x), ("scale", scale), ("shift", shift), ("indices", indices)))
    ablation_reason = _ablation_disable_reason(op_name)
    if ablation_reason is not None:
        return _fallback_or_raise(op_name, ablation_reason, strict, lambda: indexed_modulate_bf16_reference(x, scale, shift, indices), declined=True)
    support = explain_indexed_modulate_support(x, scale, shift, indices, enable=enable, device_capability=device_capability)
    if not support.supported:
        return _fallback_or_raise(op_name, support.reason, strict, lambda: indexed_modulate_bf16_reference(x, scale, shift, indices), declined=True)
    selected = _select_indexed_strategy(strategy, (x, scale, shift), indices)
    if not selected.supported:
        return _fallback_or_raise(op_name, f"indexed_modulate_bf16: {selected.reason}", strict, lambda: indexed_modulate_bf16_reference(x, scale, shift, indices), declined=True)
    try:
        _triton, kernels = _ensure_triton_kernels()
        rows = x.numel() // x.shape[-1]
        cols = x.shape[-1]
        if selected.reason == "materialize":
            x_c, scale_c, shift_c, indices_c = _contiguous_for_materialize(
                op_name,
                (("x", x), ("scale", scale), ("shift", shift), ("indices", indices)),
            )
            flat = x_c.reshape(rows, cols)
            out = torch.empty_like(flat)
            kernels["indexed_modulate"][(rows,)](
                flat,
                scale_c,
                shift_c,
                indices_c,
                out,
                cols,
                BLOCK=_block_for_cols(cols),
                num_warps=8,
            )
            _record_strategy(op_name, "materialize")
        else:
            out = torch.empty((rows, cols), dtype=x.dtype, device=x.device)
            kernels["indexed_modulate_strided"][(rows,)](
                x,
                scale,
                shift,
                indices,
                out,
                cols,
                int(x.stride(0)),
                int(x.stride(1)),
                int(scale.stride(0)),
                int(scale.stride(1)),
                int(shift.stride(0)),
                int(shift.stride(1)),
                int(indices.stride(0)),
                BLOCK=_block_for_cols(cols),
                num_warps=8,
            )
            _record_strategy(op_name, "stride_aware")
        candidate = out.reshape_as(x)
        _shadow_compare_or_raise(op_name, candidate, lambda: indexed_modulate_bf16_reference(x, scale, shift, indices))
        _record_outcome(op_name, "candidate")
        return candidate
    except ShadowMismatchError:
        raise
    except Exception as exc:  # pragma: no cover - exercised by external GPU gate
        return _fallback_or_raise(op_name, f"indexed_modulate_bf16: Triton launch failed: {exc}", strict, lambda: indexed_modulate_bf16_reference(x, scale, shift, indices))


def indexed_gate_bf16(
    residual: torch.Tensor,
    gate: torch.Tensor,
    branch: torch.Tensor,
    indices: torch.Tensor,
    *,
    enable: bool | None = None,
    strict: bool = False,
    device_capability: tuple[int, int] | None = None,
    strategy: Literal["auto", "stride_aware", "materialize"] | None = None,
) -> torch.Tensor:
    """Return ``residual + gate[indices] * branch`` with BF16 eager rounding."""

    op_name = "indexed_gate_bf16"
    _record_call(op_name)
    _record_tensor_layouts(op_name, (("residual", residual), ("gate", gate), ("branch", branch), ("indices", indices)))
    ablation_reason = _ablation_disable_reason(op_name)
    if ablation_reason is not None:
        return _fallback_or_raise(op_name, ablation_reason, strict, lambda: indexed_gate_bf16_reference(residual, gate, branch, indices), declined=True)
    support = explain_indexed_gate_support(residual, gate, branch, indices, enable=enable, device_capability=device_capability)
    if not support.supported:
        return _fallback_or_raise(op_name, support.reason, strict, lambda: indexed_gate_bf16_reference(residual, gate, branch, indices), declined=True)
    selected = _select_indexed_strategy(strategy, (residual, gate, branch), indices)
    if not selected.supported:
        return _fallback_or_raise(op_name, f"indexed_gate_bf16: {selected.reason}", strict, lambda: indexed_gate_bf16_reference(residual, gate, branch, indices), declined=True)
    try:
        _triton, kernels = _ensure_triton_kernels()
        rows = residual.numel() // residual.shape[-1]
        cols = residual.shape[-1]
        if selected.reason == "materialize":
            residual_c, gate_c, branch_c, indices_c = _contiguous_for_materialize(
                op_name,
                (("residual", residual), ("gate", gate), ("branch", branch), ("indices", indices)),
            )
            residual_flat = residual_c.reshape(rows, cols)
            branch_flat = branch_c.reshape(rows, cols)
            out = torch.empty_like(residual_flat)
            kernels["indexed_gate"][(rows,)](
                residual_flat,
                gate_c,
                branch_flat,
                indices_c,
                out,
                cols,
                BLOCK=_block_for_cols(cols),
                num_warps=8,
            )
            _record_strategy(op_name, "materialize")
        else:
            out = torch.empty((rows, cols), dtype=residual.dtype, device=residual.device)
            kernels["indexed_gate_strided"][(rows,)](
                residual,
                gate,
                branch,
                indices,
                out,
                cols,
                int(residual.stride(0)),
                int(residual.stride(1)),
                int(gate.stride(0)),
                int(gate.stride(1)),
                int(branch.stride(0)),
                int(branch.stride(1)),
                int(indices.stride(0)),
                BLOCK=_block_for_cols(cols),
                num_warps=8,
            )
            _record_strategy(op_name, "stride_aware")
        candidate = out.reshape_as(residual)
        _shadow_compare_or_raise(op_name, candidate, lambda: indexed_gate_bf16_reference(residual, gate, branch, indices))
        _record_outcome(op_name, "candidate")
        return candidate
    except ShadowMismatchError:
        raise
    except Exception as exc:  # pragma: no cover
        return _fallback_or_raise(op_name, f"indexed_gate_bf16: Triton launch failed: {exc}", strict, lambda: indexed_gate_bf16_reference(residual, gate, branch, indices))


def apply_rope_bf16(
    hidden_states: torch.Tensor,
    freqs: torch.Tensor,
    *,
    enable: bool | None = None,
    strict: bool = False,
    device_capability: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Rotate H3 leading rotary channels and preserve the tail channels."""

    op_name = "apply_rope_bf16"
    _record_call(op_name)
    _record_tensor_layouts(op_name, (("hidden_states", hidden_states), ("freqs", freqs)))
    ablation_reason = _ablation_disable_reason(op_name)
    if ablation_reason is not None:
        return _fallback_or_raise(op_name, ablation_reason, strict, lambda: apply_rope_bf16_reference(hidden_states, freqs), declined=True)
    support = explain_rope_support(hidden_states, freqs, enable=enable, device_capability=device_capability)
    if not support.supported:
        return _fallback_or_raise(op_name, support.reason, strict, lambda: apply_rope_bf16_reference(hidden_states, freqs), declined=True)
    try:
        triton, kernels = _ensure_triton_kernels()
        seq_len = hidden_states.shape[0] if hidden_states.ndim == 3 else hidden_states.shape[1]
        heads = hidden_states.shape[-2]
        head_dim = hidden_states.shape[-1]
        rotary_dim = freqs.shape[-1]
        flat = hidden_states.reshape(-1, heads * head_dim)
        out = torch.empty_like(flat)
        # vLLM RotaryEmbedding(half_head_dim=False, is_neox_style=True)
        # consumes only the first rotary half, then broadcasts it over both
        # NeoX halves.  MiniMax-H3's own RoPE generator duplicates this half,
        # but the exact kernel must also match the locked eager expression when
        # a harness supplies non-duplicated freqs.
        half = rotary_dim // 2
        freqs_half = freqs[:, :half].to(torch.float32)
        cos = torch.cos(freqs_half).to(torch.bfloat16).contiguous()
        sin = torch.sin(freqs_half).to(torch.bfloat16).contiguous()
        width = heads * head_dim
        kernels["rope"][(flat.shape[0], triton.cdiv(width, _TILE))](
            flat,
            cos,
            sin,
            out,
            seq_len,
            width,
            head_dim,
            rotary_dim,
            BLOCK=_TILE,
            num_warps=4,
        )
        candidate = out.reshape_as(hidden_states)
        _shadow_compare_or_raise(op_name, candidate, lambda: apply_rope_bf16_reference(hidden_states, freqs))
        _record_outcome(op_name, "candidate")
        return candidate
    except ShadowMismatchError:
        raise
    except Exception as exc:  # pragma: no cover
        return _fallback_or_raise(op_name, f"apply_rope_bf16: Triton launch failed: {exc}", strict, lambda: apply_rope_bf16_reference(hidden_states, freqs))


def _swiglu_reference_ordered(x: torch.Tensor, order: Literal["value_gate", "gate_up"]) -> torch.Tensor:
    return swiglu_bf16_reference(x, order=order)


def swiglu_bf16(
    x: torch.Tensor,
    *,
    order: Literal["value_gate", "gate_up"] = "value_gate",
    enable: bool | None = None,
    strict: bool = False,
    device_capability: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Fused SwiGLU for ``[value; gate]`` (Sana) or ``[gate; up]`` (vLLM) rows."""

    if order not in ("value_gate", "gate_up"):
        raise ValueError(f"unsupported SwiGLU order {order!r}")
    op_name = "swiglu_bf16"
    _record_call(op_name)
    _record_tensor_layouts(op_name, (("x", x),))
    ablation_reason = _ablation_disable_reason(op_name)
    if ablation_reason is not None:
        return _fallback_or_raise(op_name, ablation_reason, strict, lambda: _swiglu_reference_ordered(x, order), declined=True)
    support = explain_swiglu_support(x, enable=enable, device_capability=device_capability)
    if not support.supported:
        return _fallback_or_raise(op_name, support.reason, strict, lambda: _swiglu_reference_ordered(x, order), declined=True)
    try:
        triton, kernels = _ensure_triton_kernels()
        half = x.shape[-1] // 2
        flat = x.reshape(-1, x.shape[-1])
        out = torch.empty((flat.shape[0], half), dtype=x.dtype, device=x.device)
        kernels["swiglu"][(flat.shape[0], triton.cdiv(half, _TILE))](
            flat,
            out,
            half,
            order == "value_gate",
            BLOCK=_TILE,
            num_warps=4,
        )
        candidate = out.reshape(*x.shape[:-1], half)
        _shadow_compare_or_raise(op_name, candidate, lambda: _swiglu_reference_ordered(x, order))
        _record_outcome(op_name, "candidate")
        return candidate
    except ShadowMismatchError:
        raise
    except Exception as exc:  # pragma: no cover
        return _fallback_or_raise(op_name, f"swiglu_bf16: Triton launch failed: {exc}", strict, lambda: _swiglu_reference_ordered(x, order))


__all__ = [
    "KernelSupport",
    "ShadowMismatchError",
    "apply_rope_bf16",
    "explain_indexed_gate_support",
    "explain_indexed_modulate_support",
    "explain_rope_support",
    "exact_kernel_telemetry_enabled",
    "exact_kernel_shadow_enabled",
    "exact_kernel_shadow_strict",
    "explain_swiglu_support",
    "get_exact_kernel_telemetry",
    "indexed_gate_bf16",
    "indexed_modulate_bf16",
    "reset_exact_kernel_telemetry",
    "swiglu_bf16",
    "write_exact_kernel_telemetry_json",
]
