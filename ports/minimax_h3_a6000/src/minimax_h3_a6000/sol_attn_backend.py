# SPDX-License-Identifier: Apache-2.0
"""Default-off MiniMax-H3 Sol-Attn policy and SM86 candidate wrapper.

This module owns the CPU-testable contract around the real Triton Sol-Attn
candidate in :mod:`minimax_h3_a6000.sol_attn_triton_sm86`:

* packed MiniMax-H3 BTHD tensors with ``head_dim == 128``;
* ``[prefix | video tail | padding]`` metadata and contiguous valid length;
* first-10 denoise steps and first-2 layers stay dense;
* every prefix KV block is an exact sink, while prefix query rows are replaced
  by dense SDPA output;
* cache stays disabled by contract;
* unsupported inputs strictly fall back to dense reference attention unless the
  caller opts into ``strict``;
* importing this module never imports Triton or probes CUDA.

The Triton candidate is loaded only after the env/policy, tensor, metadata, and
explicit SM86 guards pass.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import torch

from .env import DEFAULT_ENV_SWITCHES

SOL_ATTN_BLOCK_SIZE = 64
H3_HOOK_METADATA_SOURCE = "vllm_omni_attention_metadata_v1"
MISSING_H3_HOOK_METADATA_PREFIX = "missing_h3_hook_metadata"


def env_enabled(name: str, env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(name, DEFAULT_ENV_SWITCHES.get(name, "0"))).strip() == "1"


def sm86_capability_guard(device_capability: tuple[int, int] | None) -> bool:
    """A6000 candidate guard: accept only an explicit SM86 capability tuple."""

    return tuple(device_capability) == (8, 6) if device_capability is not None else False


def triton_candidate_enabled(
    *,
    device_capability: tuple[int, int] | None,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Return whether optional Triton candidates may be considered.

    The guard never probes CUDA. Callers must pass a capability observed by a
    separately authorized GPU gate. CPU tests pass ``None`` or synthetic tuples.
    """

    return (
        env_enabled("MINIMAX_H3_A6000_ENABLE_OVERLAY", env)
        and env_enabled("MINIMAX_H3_A6000_ENABLE_TRITON_CANDIDATES", env)
        and sm86_capability_guard(device_capability)
    )


@functools.lru_cache(maxsize=1)
def _load_sol_attn_sm86() -> Callable[..., torch.Tensor]:
    """Lazy-load the real Triton implementation only for eligible GPU calls."""

    from .sol_attn_triton_sm86 import sol_attn_sm86

    return sol_attn_sm86


@dataclass(frozen=True, slots=True)
class PackedH3Metadata:
    """Packed MiniMax-H3 prefix/video layout contract.

    ``prefix_len`` is the non-video prefix length. ``latent_grid`` is
    ``(latent_t, latent_h_patches, latent_w_patches)``. The valid sequence must
    be exactly ``prefix_len + product(latent_grid)`` so the video segment is the
    tail of the semantic sequence; ``total_length`` includes alignment padding
    and must match the BTHD tensor length for the sparse candidate.
    """

    prefix_len: int
    latent_grid: tuple[int, int, int]
    valid_length: int
    total_length: int

    def __post_init__(self) -> None:
        if self.prefix_len < 0:
            raise ValueError("prefix_len must be non-negative")
        if len(self.latent_grid) != 3 or any(int(x) <= 0 for x in self.latent_grid):
            raise ValueError(f"latent_grid must contain three positive ints, got {self.latent_grid!r}")
        video_rows = self.video_rows
        expected_valid = self.prefix_len + video_rows
        if self.valid_length != expected_valid:
            raise ValueError(
                "valid_length must equal prefix plus video rows for [prefix | video tail | padding]: "
                f"valid={self.valid_length}, expected={expected_valid}, prefix={self.prefix_len}, video={video_rows}"
            )
        if self.total_length < self.valid_length:
            raise ValueError("total_length must be >= valid_length")

    @property
    def video_rows(self) -> int:
        t, h, w = (int(x) for x in self.latent_grid)
        return t * h * w

    @property
    def video_start(self) -> int:
        return self.prefix_len

    @property
    def video_end(self) -> int:
        return self.prefix_len + self.video_rows


@dataclass(frozen=True, slots=True)
class H3SolAttnHookMetadata:
    """Source-backed metadata extracted from a vLLM-Omni H3 attention hook."""

    packed: PackedH3Metadata
    step_index: int
    layer_index: int
    source: str = H3_HOOK_METADATA_SOURCE


def _field(source: Any, name: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def _metadata_extra(attn_metadata: Any) -> Mapping[str, Any]:
    extra = _field(attn_metadata, "extra")
    return extra if isinstance(extra, Mapping) else {}


def _read_int(source: Mapping[str, Any] | Any, *names: str) -> int | None:
    for name in names:
        value = _field(source, name)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _missing_hook_reason(detail: str) -> str:
    return f"{MISSING_H3_HOOK_METADATA_PREFIX}:{detail}"


def derive_h3_sol_attn_hook_metadata(
    attn_metadata: Any,
    *,
    total_length: int,
) -> tuple[H3SolAttnHookMetadata | None, str | None]:
    """Derive MiniMax-H3 Sol-Attn metadata from a vLLM-Omni hook object.

    The only accepted source is the real H3 packed-sequence metadata published
    by vLLM-Omni: ``AttentionMetadata.video_layout`` for ``[prefix | video |
    padding]`` and ``AttentionMetadata.extra`` for contiguous valid length,
    denoise step, and DiT layer index. Missing or inconsistent fields return a
    stable fail-closed blocker reason; this helper never guesses layout from
    tensor sizes.
    """

    if int(total_length) <= 0:
        return None, _missing_hook_reason("invalid_total_length")
    if attn_metadata is None:
        return None, _missing_hook_reason("missing_attention_metadata")

    layout = _field(attn_metadata, "video_layout")
    if layout is None:
        return None, _missing_hook_reason("missing_packed_video_layout")
    prefix_len = _read_int(layout, "prefix_len")
    latent_grid_raw = _field(layout, "latent_grid")
    if prefix_len is None or latent_grid_raw is None:
        return None, _missing_hook_reason("invalid_packed_video_layout")
    try:
        latent_grid = tuple(int(x) for x in latent_grid_raw)
    except (TypeError, ValueError):
        return None, _missing_hook_reason("invalid_packed_video_layout")
    if len(latent_grid) != 3:
        return None, _missing_hook_reason("invalid_packed_video_layout")

    extra = _metadata_extra(attn_metadata)
    valid_length = _read_int(extra, "h3_valid_kv_length", "valid_kv_length")
    if valid_length is None:
        return None, _missing_hook_reason("missing_valid_kv_length_metadata")
    step_index = _read_int(extra, "h3_denoise_step_index", "denoise_step_index", "step_index")
    layer_index = _read_int(extra, "h3_layer_index", "layer_index")
    if step_index is None or layer_index is None:
        return None, _missing_hook_reason("missing_step_layer_metadata")
    if step_index < 0 or layer_index < 0:
        return None, _missing_hook_reason("invalid_step_layer_metadata")

    try:
        packed = PackedH3Metadata(
            prefix_len=prefix_len,
            latent_grid=latent_grid,
            valid_length=valid_length,
            total_length=int(total_length),
        )
    except ValueError:
        return None, _missing_hook_reason("invalid_packed_video_layout")
    return H3SolAttnHookMetadata(packed=packed, step_index=step_index, layer_index=layer_index), None


@dataclass(frozen=True, slots=True)
class SolAttnPolicy:
    """Practical Sol-Attn policy for the A6000 sparse candidate."""

    dense_first_steps: int = 10
    dense_first_layers: int = 2
    tau: float = 1.0
    thresh_type: str = "diag"
    allow_sparse: bool = False
    sink_start: int = 0
    sink_tokens: int | None = None
    prefix_query_dense: bool = True
    cache_enabled: bool = False
    strict: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "SolAttnPolicy":
        env = os.environ if env is None else env
        return cls(
            allow_sparse=(
                env_enabled("MINIMAX_H3_A6000_ENABLE_OVERLAY", env)
                and env_enabled("MINIMAX_H3_A6000_ENABLE_TRITON_CANDIDATES", env)
                and env_enabled("MINIMAX_H3_A6000_ENABLE_SOL_ATTN", env)
            ),
            cache_enabled=env_enabled("MINIMAX_H3_A6000_SOL_ATTN_CACHE", env),
            strict=env_enabled("MINIMAX_H3_A6000_SOL_ATTN_STRICT", env),
        )


@dataclass
class SolAttnTelemetry:
    """Per-run Sol-Attn counters for logs and external GPU gates."""

    dense_calls: int = 0
    sparse_candidate_calls: int = 0
    sparse_calls: int = 0
    fallback_calls: int = 0
    prefix_query_dense_calls: int = 0
    decline_reasons: dict[str, int] = field(default_factory=dict)
    fallback_reasons: dict[str, int] = field(default_factory=dict)
    sink_ranges: list[tuple[int, int]] = field(default_factory=list)
    density_samples: list[dict[str, Any]] = field(default_factory=list)

    def record_decline(self, reason: str) -> None:
        self.dense_calls += 1
        self.decline_reasons[reason] = self.decline_reasons.get(reason, 0) + 1

    def record_fallback(self, reason: str) -> None:
        self.fallback_calls += 1
        self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1
        self.record_decline(reason)

    def record_sparse_candidate(self, sink_range: tuple[int, int], density: dict[str, Any]) -> None:
        self.sparse_candidate_calls += 1
        self.sink_ranges.append(sink_range)
        self.density_samples.append(density)

    def record_sparse_success(self, *, prefix_query_dense: bool) -> None:
        self.sparse_calls += 1
        if prefix_query_dense:
            self.prefix_query_dense_calls += 1


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def derive_sink_range(metadata: PackedH3Metadata, policy: SolAttnPolicy) -> tuple[int, int]:
    """Derive the exact-KV sink range from packed prefix metadata."""

    start = int(policy.sink_start)
    tokens = metadata.prefix_len if policy.sink_tokens is None else int(policy.sink_tokens)
    end = start + tokens
    if start < 0 or tokens < 0 or end > metadata.valid_length:
        raise ValueError(f"invalid sink range [{start}, {end}) for valid length {metadata.valid_length}")
    return start, end


def estimate_sparse_density(metadata: PackedH3Metadata, policy: SolAttnPolicy) -> dict[str, Any]:
    """Estimate the static exact-block lower bound for Sol-Attn telemetry.

    Triton routing can promote additional blocks from approximate to exact based
    on thresholds. This estimate records the always-exact local window plus the
    configured sink blocks, so it is a stable lower bound rather than a fidelity
    claim.
    """

    valid = int(metadata.valid_length)
    blocks = _ceil_div(valid, SOL_ATTN_BLOCK_SIZE)
    sink_start, sink_end = derive_sink_range(metadata, policy)
    sink_start_block = sink_start // SOL_ATTN_BLOCK_SIZE if sink_end > sink_start else blocks
    sink_end_block = _ceil_div(sink_end, SOL_ATTN_BLOCK_SIZE) if sink_end > sink_start else blocks
    exact_pairs = 0
    for query_block in range(blocks):
        exact = set(range(max(0, query_block - 1), min(blocks, query_block + 2)))
        exact.update(range(sink_start_block, sink_end_block))
        exact_pairs += len(exact)
    total_pairs = blocks * blocks
    return {
        "kind": "static_exact_block_lower_bound",
        "block_size": SOL_ATTN_BLOCK_SIZE,
        "valid_tokens": valid,
        "blocks": blocks,
        "prefix_tokens": int(metadata.prefix_len),
        "video_tokens": int(metadata.video_rows),
        "sink_start": sink_start,
        "sink_end": sink_end,
        "sink_start_block": sink_start_block,
        "sink_end_block": sink_end_block,
        "exact_block_pairs_lower_bound": exact_pairs,
        "total_block_pairs": total_pairs,
        "exact_density_lower_bound": (exact_pairs / total_pairs) if total_pairs else 1.0,
    }


def decline_reason(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    metadata: PackedH3Metadata | None,
    step_index: int,
    layer_index: int,
    policy: SolAttnPolicy,
    device_capability: tuple[int, int] | None = None,
) -> str | None:
    """Return why the sparse candidate must decline, or ``None`` if eligible."""

    if not policy.allow_sparse:
        return "env_disabled"
    if policy.cache_enabled:
        return "cache_disabled_by_contract"
    if step_index < policy.dense_first_steps:
        return "dense_first_steps"
    if layer_index < policy.dense_first_layers:
        return "dense_first_layers"
    if metadata is None:
        return "missing_packed_metadata"
    if policy.thresh_type not in ("diag", "exact"):
        return "unsupported_threshold_type"
    if query.shape != key.shape or query.shape != value.shape:
        return "qkv_shape_mismatch"
    if query.ndim != 4 or query.shape[-1] != 128:
        return "unsupported_qkv_layout"
    if int(query.shape[1]) != int(metadata.total_length):
        return "metadata_total_length_mismatch"
    if metadata.valid_length <= 0 or metadata.valid_length > query.shape[1]:
        return "invalid_valid_length"
    if query.dtype != torch.bfloat16 or key.dtype != query.dtype or value.dtype != query.dtype:
        return "unsupported_dtype"
    if query.device.type != "cuda" or key.device != query.device or value.device != query.device:
        return "unsupported_device"
    if not (query.is_contiguous() and key.is_contiguous() and value.is_contiguous()):
        return "unsupported_contiguity"
    if not sm86_capability_guard(device_capability):
        return "unsupported_or_unprobed_sm"
    return None


def dense_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense BTHD reference attention."""

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value must be [B, T, H, D]")
    if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:] or key.shape != value.shape:
        raise ValueError("batch/head/dim and key/value shapes must match")
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    out = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=softmax_scale,
    )
    return out.permute(0, 2, 1, 3).contiguous()


def dense_attention_packed_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    metadata: PackedH3Metadata | None,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Dense reference respecting H3 valid-length padding semantics."""

    if metadata is None:
        return dense_attention_reference(query, key, value, softmax_scale=softmax_scale)
    if int(query.shape[1]) != int(metadata.total_length):
        return dense_attention_reference(query, key, value, softmax_scale=softmax_scale)
    valid = int(metadata.valid_length)
    output = torch.zeros_like(value)
    output[:, :valid] = dense_attention_reference(
        query[:, :valid],
        key[:, :valid],
        value[:, :valid],
        softmax_scale=softmax_scale,
    )
    return output


def sol_attn_h3_sparse_candidate(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    metadata: PackedH3Metadata | None,
    step_index: int,
    layer_index: int,
    policy: SolAttnPolicy | None = None,
    telemetry: SolAttnTelemetry | None = None,
    device_capability: tuple[int, int] | None = None,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run the real SM86 Sol-Attn candidate or a strict dense fallback."""

    policy = SolAttnPolicy() if policy is None else policy
    telemetry = SolAttnTelemetry() if telemetry is None else telemetry
    reason = decline_reason(
        query=query,
        key=key,
        value=value,
        metadata=metadata,
        step_index=step_index,
        layer_index=layer_index,
        policy=policy,
        device_capability=device_capability,
    )
    if reason is not None:
        telemetry.record_decline(reason)
        return dense_attention_packed_reference(query, key, value, metadata=metadata, softmax_scale=softmax_scale)

    assert metadata is not None  # narrowed by decline_reason
    sink_range = derive_sink_range(metadata, policy)
    density = estimate_sparse_density(metadata, policy)
    telemetry.record_sparse_candidate(sink_range, density)
    valid = int(metadata.valid_length)
    prefix = int(metadata.prefix_len)

    try:
        kernel = _load_sol_attn_sm86()
        q_valid = query[:, :valid].contiguous()
        k_valid = key[:, :valid].contiguous()
        v_valid = value[:, :valid].contiguous()
        sparse_valid = kernel(
            q_valid,
            k_valid,
            v_valid,
            scale=softmax_scale,
            tau=float(policy.tau),
            thresh_type=str(policy.thresh_type),
            sink_start=sink_range[0],
            sink_tokens=sink_range[1] - sink_range[0],
        )
        output = torch.zeros_like(value)
        output[:, :valid] = sparse_valid
        prefix_dense = bool(policy.prefix_query_dense and prefix > 0)
        if prefix_dense:
            output[:, :prefix] = dense_attention_reference(
                q_valid[:, :prefix],
                k_valid,
                v_valid,
                softmax_scale=softmax_scale,
            )
        telemetry.record_sparse_success(prefix_query_dense=prefix_dense)
        return output
    except Exception as exc:  # noqa: BLE001 - fallback must preserve runtime availability
        reason = f"kernel_error:{type(exc).__name__}"
        telemetry.record_fallback(reason)
        if policy.strict:
            raise RuntimeError(f"Sol-Attn sparse candidate failed: {type(exc).__name__}: {exc}") from exc
        return dense_attention_packed_reference(query, key, value, metadata=metadata, softmax_scale=softmax_scale)


def sol_attn_h3_reference_or_decline(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    metadata: PackedH3Metadata | None,
    step_index: int,
    layer_index: int,
    policy: SolAttnPolicy | None = None,
    telemetry: SolAttnTelemetry | None = None,
    device_capability: tuple[int, int] | None = None,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Backward-compatible entry point for the Sol-Attn sparse candidate."""

    return sol_attn_h3_sparse_candidate(
        query,
        key,
        value,
        metadata=metadata,
        step_index=step_index,
        layer_index=layer_index,
        policy=policy,
        telemetry=telemetry,
        device_capability=device_capability,
        softmax_scale=softmax_scale,
    )
