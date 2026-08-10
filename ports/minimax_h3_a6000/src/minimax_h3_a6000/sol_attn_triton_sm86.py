# SPDX-License-Identifier: Apache-2.0
#
# Adapted from NVlabs/Sana Sol-Engine
# techniques/sparse_backends/sol_attn/triton_ref/{preprocess.py,fwd.py}
# at d00eef311670a58deb2c323fe072738fcb945600.  This local overlay keeps
# only the pointer-backed Triton path needed by RTX A6000 / SM86; the upstream
# SM>=8 guard is preserved and the caller adds the stricter SM86 policy gate.
"""Pointer-backed Triton Sol-Attn candidate for MiniMax-H3 A6000/SM86.

The module is intentionally not imported by :mod:`minimax_h3_a6000` package
initialization.  Importing this file requires PyTorch+Triton, but no CUDA work is
performed until ``sol_attn_sm86`` is called by an already-authorized GPU harness
or disposable vLLM-Omni integration tree.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


BLOCK_SIZE = 64
HEAD_DIM = 128
THRESHOLD_GROUP_SIZE = 64
SUMMARY_PAD = 64
GROUP_SIZE = 32


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=warps, num_stages=stages)
        for warps in (4, 8)
        for stages in (1, 2)
    ],
    key=["T"],
)
@triton.jit
def _reduce_kv_kernel(
    k,
    v,
    kc,
    vc,
    T,
    TP,
    NPAD,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    tokens = block * BLOCK + tl.arange(0, BLOCK)
    dims = tl.arange(0, D)
    valid = tokens < T
    offsets = (
        ((batch * TP + tokens[:, None]).to(tl.int64) * H + head) * D
        + dims[None, :]
    )
    k_values = tl.load(k + offsets, mask=valid[:, None], other=0.0)
    v_values = tl.load(v + offsets, mask=valid[:, None], other=0.0)
    block_len = tl.minimum(BLOCK, T - block * BLOCK).to(tl.float32)
    summary_offsets = ((batch * NPAD + block) * H + head) * D + dims
    tl.store(kc + summary_offsets, tl.sum(k_values, axis=0) / block_len)
    tl.store(vc + summary_offsets, tl.sum(v_values, axis=0))


@triton.jit
def _reduce_kc_stats_kernel(
    kc,
    kc_mean,
    kc_var_diag,
    NPAD,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
):
    batch_head = tl.program_id(0)
    batch, head = batch_head // H, batch_head % H
    blocks = tl.max_contiguous(tl.arange(0, GROUP), GROUP)
    dims = tl.arange(0, D)
    total = tl.zeros((D,), dtype=tl.float32)
    total_sq = tl.zeros((D,), dtype=tl.float32)
    count = tl.full((), 0.0, dtype=tl.float32)
    for start in range(0, N, GROUP):
        block_indices = start + blocks
        valid = block_indices < N
        offsets = (
            ((batch * NPAD + block_indices[:, None]) * H + head) * D
            + dims[None, :]
        )
        values = tl.load(kc + offsets, mask=valid[:, None], other=0.0).to(tl.float32)
        total += tl.sum(values, axis=0)
        total_sq += tl.sum(values * values, axis=0)
        count += tl.sum(valid.to(tl.float32), axis=0)
    mean = total / count
    variance = tl.maximum(total_sq / count - mean * mean, 0.0)
    tl.store(kc_mean + batch_head * D + dims, mean)
    tl.store(kc_var_diag + batch_head * D + dims, variance)


@triton.jit
def _diag_threshold_kernel(
    q,
    kc_mean,
    kc_var_diag,
    threshold,
    scale,
    T,
    TP,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TAU: tl.constexpr,
):
    q_block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    tokens = q_block * BLOCK + tl.arange(0, BLOCK)
    dims = tl.arange(0, D)
    valid = tokens < T
    offsets = (
        ((batch * TP + tokens[:, None]).to(tl.int64) * H + head) * D
        + dims[None, :]
    )
    q_values = tl.load(q + offsets, mask=valid[:, None], other=0.0)
    q_len = tl.minimum(BLOCK, T - q_block * BLOCK).to(tl.float32)
    q_centroid = tl.sum(q_values.to(tl.float32), axis=0) / q_len
    mean_kc = tl.load(kc_mean + batch_head * D + dims)
    var_kc = tl.load(kc_var_diag + batch_head * D + dims)
    log2_scale = scale * 1.4426950408889634
    mean = tl.sum(q_centroid * mean_kc, axis=0) * log2_scale
    variance = tl.sum(q_centroid * q_centroid * var_kc, axis=0) * (log2_scale * log2_scale)
    std = tl.sqrt(tl.maximum(variance, 0.0) + 1.0e-6)
    tl.store(threshold + (batch * N + q_block) * H + head, mean + TAU * std)


@triton.jit
def _pool_query_kernel(
    q,
    q_bar,
    T,
    TP,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    q_block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    tokens = q_block * BLOCK + tl.arange(0, BLOCK)
    dims = tl.arange(0, D)
    valid = tokens < T
    offsets = (
        ((batch * TP + tokens[:, None]).to(tl.int64) * H + head) * D
        + dims[None, :]
    )
    values = tl.load(q + offsets, mask=valid[:, None], other=0.0)
    q_len = tl.minimum(BLOCK, T - q_block * BLOCK).to(tl.float32)
    centroid = tl.sum(values.to(tl.float32), axis=0) / q_len
    tl.store(q_bar + (batch_head * N + q_block) * D + dims, centroid)


@triton.jit
def _exact_fused_threshold_kernel(
    q_bar,
    kc_mean,
    kc_second_moment,
    threshold,
    scale,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TAU: tl.constexpr,
):
    row_tile, batch_head = tl.program_id(0), tl.program_id(1)
    rows = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, D)
    valid_rows = rows < N
    q_centroid = tl.load(
        q_bar + (batch_head * N + rows[:, None]) * D + dims[None, :],
        mask=valid_rows[:, None],
        other=0.0,
    )
    mean_kc = tl.load(kc_mean + batch_head * D + dims)
    second_moment = tl.load(
        kc_second_moment + batch_head * D * D + dims[:, None] * D + dims[None, :]
    )
    raw_mean = tl.sum(q_centroid.to(tl.float32) * mean_kc[None, :], axis=1)
    projected = tl.dot(q_centroid, second_moment, out_dtype=tl.float32)
    raw_second_moment = tl.sum(projected * q_centroid.to(tl.float32), axis=1)
    log2_scale = scale * 1.4426950408889634
    mean = raw_mean * log2_scale
    variance = tl.maximum(raw_second_moment - raw_mean * raw_mean, 0.0) * (log2_scale * log2_scale)
    result = mean + TAU * tl.sqrt(variance + 1.0e-6)
    batch, head = batch_head // H, batch_head % H
    tl.store(threshold + (batch * N + rows) * H + head, result, mask=valid_rows)


@triton.autotune(
    configs=[
        triton.Config({"BV": 128}, num_warps=4, num_stages=1),
        triton.Config({"BV": 128}, num_warps=8, num_stages=1),
        triton.Config({"BV": 128}, num_warps=4, num_stages=2),
        triton.Config({"BV": 64}, num_warps=4, num_stages=1),
    ],
    key=["T"],
)
@triton.jit
def _forward_ptr_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    kc_ptr,
    vc_ptr,
    threshold_ptr,
    o_ptr,
    scale,
    T,
    NPAD,
    sink_start_block,
    sink_end_block,
    HAS_SINK: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    NT: tl.constexpr,
    BV: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUP: tl.constexpr,
):
    v_tile, q_block, batch_head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    batch, head = batch_head // H, batch_head % H
    group_offsets = tl.max_contiguous(tl.arange(0, GROUP), GROUP)
    token_offsets = tl.max_contiguous(tl.arange(0, BLOCK), BLOCK)
    dims = tl.arange(0, D)
    value_dims = v_tile * BV + tl.arange(0, BV)
    q_tokens = q_block * BLOCK + token_offsets
    q_valid = q_tokens < T
    q_offsets = ((batch * T + q_tokens[:, None]).to(tl.int64) * H + head) * D + dims[None, :]
    q = tl.load(q_ptr + q_offsets, mask=q_valid[:, None], other=0.0)
    q_len = tl.minimum(BLOCK, T - q_block * BLOCK).to(tl.float32)

    output = tl.zeros([BLOCK, BV], dtype=tl.float32)
    row_sum = tl.zeros((BLOCK,), dtype=tl.float32)
    row_max = tl.full((BLOCK,), -float("inf"), tl.float32)
    scale_log2 = scale * 1.4426950408889634
    route_threshold = tl.load(threshold_ptr + (batch * NT + q_block) * H + head)

    for group_start in range(0, NT, GROUP):
        block_indices = group_start + group_offsets
        valid = block_indices < NT
        kc_offsets = ((batch * NPAD + block_indices[:, None]) * H + head) * D + dims[None, :]
        vc_offsets = ((batch * NPAD + block_indices[:, None]) * H + head) * D + value_dims[None, :]
        kc = tl.load(kc_ptr + kc_offsets)
        vc = tl.load(vc_ptr + vc_offsets)
        scores = tl.dot(q, kc.T).to(tl.float32) * scale_log2
        exact = (tl.sum(scores, axis=0) / q_len > route_threshold) | (tl.abs(q_block - block_indices) <= 1)
        if HAS_SINK:
            exact = exact | ((block_indices >= sink_start_block) & (block_indices < sink_end_block))
        exact = exact & valid

        approximate = valid & ~exact
        has_approximate = tl.sum(approximate.to(tl.int32), axis=0) > 0
        approximate_scores = tl.where(approximate[None, :], scores, -float("inf"))
        safe_scores = tl.where(has_approximate, approximate_scores, 0.0)
        candidate_max = tl.maximum(row_max, tl.max(safe_scores, axis=1))
        new_max = tl.where(has_approximate, candidate_max, row_max)
        alpha = tl.math.exp2(tl.where(has_approximate, row_max - new_max, 0.0))
        probability = tl.math.exp2(safe_scores - tl.where(has_approximate, new_max, 0.0)[:, None])
        probability = tl.where(has_approximate & approximate[None, :], probability, 0.0)
        output = output * alpha[:, None] + tl.dot(probability.to(vc.dtype), vc)
        lengths = tl.minimum(BLOCK, tl.maximum(0, T - block_indices * BLOCK)).to(tl.float32)
        row_sum = row_sum * alpha + tl.sum(probability * lengths[None, :], axis=1)
        row_max = new_max

        exact_offsets = tl.where(exact, group_offsets, GROUP)
        num_exact = tl.sum(exact.to(tl.int32), axis=0)
        for _ in range(num_exact):
            offset = tl.min(exact_offsets)
            block = group_start + offset
            exact_offsets = tl.where(group_offsets == offset, GROUP, exact_offsets)
            kv_tokens = block * BLOCK + token_offsets
            kv_valid = kv_tokens < T
            k_offsets = ((batch * T + kv_tokens[:, None]).to(tl.int64) * H + head) * D + dims[None, :]
            k = tl.load(k_ptr + k_offsets, mask=kv_valid[:, None], other=0.0)
            exact_scores = tl.dot(q, k.T).to(tl.float32) * scale_log2
            exact_scores += tl.where(kv_valid[None, :], 0.0, -float("inf"))
            new_max = tl.maximum(row_max, tl.max(exact_scores, axis=1))
            alpha = tl.math.exp2(row_max - new_max)
            exact_probability = tl.math.exp2(exact_scores - new_max[:, None])
            row_sum = row_sum * alpha + tl.sum(exact_probability, axis=1)
            v_offsets = ((batch * T + kv_tokens[:, None]).to(tl.int64) * H + head) * D + value_dims[None, :]
            v = tl.load(v_ptr + v_offsets, mask=kv_valid[:, None], other=0.0)
            output = output * alpha[:, None] + tl.dot(exact_probability.to(v.dtype), v)
            row_max = new_max

    output_offsets = ((batch * T + q_tokens[:, None]).to(tl.int64) * H + head) * D + value_dims[None, :]
    tl.store(
        o_ptr + output_offsets,
        (output / row_sum[:, None]).to(tl.bfloat16),
        mask=q_valid[:, None],
    )


def _sink_block_range(tokens: int, sink_start: int | None, sink_tokens: int) -> tuple[int, int]:
    blocks = (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    if not sink_tokens:
        return blocks, blocks
    start = tokens - sink_tokens if sink_start is None else int(sink_start)
    return start // BLOCK_SIZE, (start + int(sink_tokens) + BLOCK_SIZE - 1) // BLOCK_SIZE


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    thresh_type: str,
    sink_tokens: int = 0,
    sink_start: int | None = None,
) -> tuple[int, int]:
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must share shape [B, T, H, 128]")
    if q.shape[1] == 0 or q.shape[3] != HEAD_DIM:
        raise ValueError("Sol-Attn requires T > 0 and head dimension 128")
    if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
        raise TypeError("q, k, and v must use torch.bfloat16")
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("q, k, and v must be contiguous BTHD tensors")
    if thresh_type not in ("diag", "exact"):
        raise ValueError("thresh_type must be 'diag' or 'exact'")
    if not isinstance(sink_tokens, int):
        raise TypeError("sink_tokens must be an integer")
    if not 0 <= sink_tokens <= q.shape[1]:
        raise ValueError("sink_tokens must be in [0, T]")
    if sink_start is not None:
        if not isinstance(sink_start, int):
            raise TypeError("sink_start must be an integer or None")
        if not 0 <= sink_start <= q.shape[1]:
            raise ValueError("sink_start must be in [0, T]")
        if sink_start + sink_tokens > q.shape[1]:
            raise ValueError("sink_start + sink_tokens must be <= T")

    arch = tuple(torch.cuda.get_device_capability(q.device))
    if arch[0] < 8:
        raise RuntimeError(
            "Triton Sol-Attn requires an NVIDIA GPU with compute capability >= 8.0; "
            f"got SM{arch[0]}{arch[1]}"
        )
    return arch


def _reduce_kv(k: torch.Tensor, v: torch.Tensor, *, tokens: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    batch, padded_tokens, heads, head_dim = k.shape
    tokens = padded_tokens if tokens is None else int(tokens)
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    padded_blocks = triton.cdiv(blocks, SUMMARY_PAD) * SUMMARY_PAD
    kc = torch.zeros((batch, padded_blocks, heads, head_dim), device=k.device, dtype=torch.bfloat16)
    vc = torch.zeros_like(kc)
    _reduce_kv_kernel[(blocks, batch * heads)](
        k,
        v,
        kc,
        vc,
        tokens,
        padded_tokens,
        padded_blocks,
        heads,
        head_dim,
        BLOCK_SIZE,
    )
    return kc, vc


def _compute_diag_threshold(
    q: torch.Tensor,
    kc: torch.Tensor,
    *,
    tau: float,
    scale: float,
    tokens: int | None = None,
) -> torch.Tensor:
    batch, padded_tokens, heads, head_dim = q.shape
    tokens = padded_tokens if tokens is None else int(tokens)
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    batch_heads = batch * heads
    kc_mean = torch.empty((batch_heads, head_dim), device=q.device, dtype=torch.float32)
    kc_var_diag = torch.empty_like(kc_mean)
    threshold = torch.empty((batch, blocks, heads), device=q.device, dtype=torch.float32)
    _reduce_kc_stats_kernel[(batch_heads,)](
        kc,
        kc_mean,
        kc_var_diag,
        kc.shape[1],
        heads,
        blocks,
        head_dim,
        THRESHOLD_GROUP_SIZE,
        num_warps=4,
        num_stages=2,
    )
    _diag_threshold_kernel[(blocks, batch_heads)](
        q,
        kc_mean,
        kc_var_diag,
        threshold,
        scale,
        tokens,
        padded_tokens,
        heads,
        blocks,
        head_dim,
        BLOCK_SIZE,
        tau,
        num_warps=4,
        num_stages=2,
    )
    return threshold


def _compute_exact_threshold(
    q: torch.Tensor,
    kc: torch.Tensor,
    *,
    tau: float,
    scale: float,
    tokens: int | None = None,
) -> torch.Tensor:
    batch, padded_tokens, heads, head_dim = q.shape
    tokens = padded_tokens if tokens is None else int(tokens)
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    batch_heads = batch * heads
    kc_bh = kc[:, :blocks].permute(0, 2, 1, 3)
    kc_mean = kc_bh.mean(dim=2, dtype=torch.float32)
    kc_second_moment = torch.matmul(kc_bh.transpose(-1, -2), kc_bh)
    kc_second_moment.div_(blocks)
    q_bar = torch.empty((batch_heads, blocks, head_dim), device=q.device, dtype=torch.bfloat16)
    threshold = torch.empty((batch, blocks, heads), device=q.device, dtype=torch.float32)
    _pool_query_kernel[(blocks, batch_heads)](
        q,
        q_bar,
        tokens,
        padded_tokens,
        heads,
        blocks,
        head_dim,
        BLOCK_SIZE,
        num_warps=4,
        num_stages=1,
    )
    block_m = 64
    _exact_fused_threshold_kernel[(triton.cdiv(blocks, block_m), batch_heads)](
        q_bar,
        kc_mean,
        kc_second_moment,
        threshold,
        scale,
        heads,
        blocks,
        head_dim,
        block_m,
        tau,
        num_warps=4,
        num_stages=1,
    )
    return threshold


def _prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float,
    scale: float,
    thresh_type: str = "diag",
    tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kc, vc = _reduce_kv(k, v, tokens=tokens)
    if thresh_type == "exact":
        threshold = _compute_exact_threshold(q, kc, tau=tau, scale=scale, tokens=tokens)
    else:
        threshold = _compute_diag_threshold(q, kc, tau=tau, scale=scale, tokens=tokens)
    return kc, vc, threshold


def sol_attn_sm86(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    tau: float = 1.0,
    thresh_type: str = "diag",
    sink_tokens: int = 0,
    sink_start: int | None = None,
) -> torch.Tensor:
    """Run the Triton Sol-Attn pointer path on contiguous BF16 BTHD SM86 inputs."""

    arch = _validate_inputs(q, k, v, thresh_type, int(sink_tokens), sink_start)
    if arch != (8, 6):
        raise RuntimeError(f"MiniMax-H3 A6000 Sol-Attn overlay requires SM86; got SM{arch[0]}{arch[1]}")
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    tau = float(tau)
    batch, tokens, heads, head_dim = q.shape
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    sink_start_block, sink_end_block = _sink_block_range(tokens, sink_start, int(sink_tokens))

    kc, vc, threshold = _prepare(q, k, v, scale=scale, tau=tau, thresh_type=thresh_type, tokens=tokens)
    output = torch.empty_like(v)
    grid = lambda meta: (head_dim // meta["BV"], blocks, batch * heads)
    _forward_ptr_kernel[grid](
        q,
        k,
        v,
        kc,
        vc,
        threshold,
        output,
        scale,
        tokens,
        kc.shape[1],
        sink_start_block,
        sink_end_block,
        HAS_SINK=int(sink_tokens) > 0,
        H=heads,
        D=head_dim,
        NT=blocks,
        BLOCK=BLOCK_SIZE,
        GROUP=GROUP_SIZE,
    )
    return output


__all__ = ["BLOCK_SIZE", "HEAD_DIM", "sol_attn_sm86"]
