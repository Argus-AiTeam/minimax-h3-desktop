# SPDX-License-Identifier: Apache-2.0
"""Locked PyTorch eager references for MiniMax-H3 vLLM-Omni wrappers.

These functions intentionally mirror the original vLLM-Omni eager expressions
used by ``minimax_h3_transformer.py``.  They are CPU-friendly references for
shadow comparison and tests; they do not bake in the older Sana-specific
intermediate BF16 rounding contract.  If PyTorch eager changes promotion or
store semantics, these references should change only after re-auditing the
locked upstream expressions.
"""

from __future__ import annotations

from typing import Literal

import torch


def _require_bf16(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"{name} must be torch.bfloat16, got {tensor.dtype}")


def bf16_round_to_fp32(x: torch.Tensor) -> torch.Tensor:
    """Round ``x`` to BF16 precision and return an FP32 tensor.

    This helper remains available for legacy differential tests, but the H3
    vLLM eager references below deliberately do not call it.
    """

    return x.to(torch.bfloat16).to(torch.float32)


def _flatten_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    if x.ndim < 2:
        raise ValueError(f"expected at least 2 dimensions, got {tuple(x.shape)}")
    return x.reshape(-1, x.shape[-1]), x.shape


def _restore_rows(x: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    return x.reshape(*original_shape)


def _validate_indexed_table(
    op_name: str,
    rows: torch.Tensor,
    table: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    if table.ndim != 2:
        raise ValueError(f"{op_name}: table must be [num_tags, hidden], got {tuple(table.shape)}")
    if rows.shape[-1] != table.shape[-1]:
        raise ValueError(
            f"{op_name}: hidden mismatch rows={rows.shape[-1]} table={table.shape[-1]}"
        )
    if indices.ndim != 1 or indices.numel() != rows.shape[0]:
        raise ValueError(
            f"{op_name}: indices must be one per flattened row, got {tuple(indices.shape)} for {rows.shape[0]} rows"
        )
    if indices.dtype not in (torch.int32, torch.int64, torch.long):
        raise TypeError(f"{op_name}: indices must be an integer tensor, got {indices.dtype}")
    if indices.device != rows.device:
        indices = indices.to(rows.device)
    if indices.numel():
        lo = int(indices.min().item())
        hi = int(indices.max().item())
        if lo < 0 or hi >= table.shape[0]:
            raise IndexError(f"{op_name}: index range [{lo}, {hi}] outside table rows {table.shape[0]}")
    return indices.to(torch.long)


def indexed_modulate_bf16_reference(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """vLLM eager ``(x * (1 + scale[idx]) + shift[idx]).to(x.dtype)``.

    This mirrors ``_modulate_scale_shift`` in the locked H3 transformer; no
    extra Sana-style rounding is inserted between the PyTorch ops.
    """

    for name, tensor in (("x", x), ("scale", scale), ("shift", shift)):
        _require_bf16(name, tensor)
    rows, original_shape = _flatten_rows(x)
    if scale.shape != shift.shape:
        raise ValueError(f"scale and shift must share shape, got {tuple(scale.shape)} and {tuple(shift.shape)}")
    indices = _validate_indexed_table("indexed_modulate_bf16_reference", rows, scale, indices)

    out = rows * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)
    return _restore_rows(out.to(x.dtype), original_shape)


def indexed_gate_bf16_reference(
    residual: torch.Tensor,
    gate: torch.Tensor,
    branch: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """vLLM eager ``(residual + gate[idx] * branch).to(residual.dtype)``."""

    for name, tensor in (("residual", residual), ("gate", gate), ("branch", branch)):
        _require_bf16(name, tensor)
    residual_rows, original_shape = _flatten_rows(residual)
    branch_rows, branch_shape = _flatten_rows(branch)
    if branch_shape != original_shape:
        raise ValueError(f"residual and branch must share shape, got {tuple(original_shape)} and {tuple(branch_shape)}")
    indices = _validate_indexed_table("indexed_gate_bf16_reference", residual_rows, gate, indices)

    out = residual_rows + gate.index_select(0, indices) * branch_rows
    return _restore_rows(out.to(residual.dtype), original_shape)


def _rotate_half_neox(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _broadcast_rope_cos_sin(cos_half: torch.Tensor, sin_half: torch.Tensor, x_rot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Original RotaryEmbedding(half_head_dim=False, is_neox_style=True) slices
    # the first half of full-width cos/sin, then repeats each half over the two
    # NeoX rotary halves with a singleton head dimension.
    cos = torch.cat((cos_half, cos_half), dim=-1)
    sin = torch.cat((sin_half, sin_half), dim=-1)
    if x_rot.ndim == 3:
        return cos[:, None, :], sin[:, None, :]
    if x_rot.ndim == 4:
        return cos[None, :, None, :], sin[None, :, None, :]
    raise ValueError(f"hidden_states must be [T,H,D] or [B,T,H,D], got {tuple(x_rot.shape)}")


def apply_rope_bf16_reference(hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """vLLM MiniMax-H3 RoPE eager expression over leading rotary channels.

    Locked expression: ``cos = torch.cos(freqs).to(x.dtype)``, same for sin,
    then ``RotaryEmbedding(is_neox_style=True, half_head_dim=False)`` on the
    rotary prefix and ``torch.cat((x_rot, x_pass), dim=-1)``.
    """

    _require_bf16("hidden_states", hidden_states)
    if freqs.ndim != 2:
        raise ValueError(f"freqs must be [T, rotary_dim], got {tuple(freqs.shape)}")
    seq_axis = 0 if hidden_states.ndim == 3 else 1 if hidden_states.ndim == 4 else None
    if seq_axis is None:
        raise ValueError(f"hidden_states must be [T,H,D] or [B,T,H,D], got {tuple(hidden_states.shape)}")
    if hidden_states.shape[seq_axis] != freqs.shape[0]:
        raise ValueError(f"sequence length mismatch: hidden={hidden_states.shape[seq_axis]} freqs={freqs.shape[0]}")
    rotary_dim = freqs.shape[-1]
    if rotary_dim <= 0 or rotary_dim % 2:
        raise ValueError(f"rotary_dim must be positive and even, got {rotary_dim}")
    if rotary_dim > hidden_states.shape[-1]:
        raise ValueError(f"rotary_dim {rotary_dim} exceeds head_dim {hidden_states.shape[-1]}")

    x_rot = hidden_states[..., :rotary_dim]
    x_pass = hidden_states[..., rotary_dim:]
    cos = torch.cos(freqs).to(hidden_states.dtype)
    sin = torch.sin(freqs).to(hidden_states.dtype)
    half = rotary_dim // 2
    cos_b, sin_b = _broadcast_rope_cos_sin(cos[..., :half], sin[..., :half], x_rot)
    out_rot = x_rot * cos_b + _rotate_half_neox(x_rot) * sin_b
    return torch.cat((out_rot, x_pass), dim=-1)


def swiglu_bf16_reference(
    x: torch.Tensor,
    *,
    order: Literal["value_gate", "gate_up"] = "value_gate",
) -> torch.Tensor:
    """vLLM/Sana eager SwiGLU expression without extra rounding.

    ``order="gate_up"`` mirrors H3 vLLM's ``gate, up = hidden.chunk(2)`` then
    ``silu(gate) * up``.  ``order="value_gate"`` is retained for legacy Sana
    standalone checks.
    """

    _require_bf16("x", x)
    if order not in ("value_gate", "gate_up"):
        raise ValueError(f"unsupported SwiGLU order {order!r}")
    if x.shape[-1] % 2:
        raise ValueError(f"last dimension must be even, got {x.shape[-1]}")
    first, second = x.chunk(2, dim=-1)
    if order == "gate_up":
        gate, value = first, second
    else:
        value, gate = first, second
    return torch.nn.functional.silu(gate) * value
