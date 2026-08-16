# SPDX-License-Identifier: Apache-2.0
"""Default-off MiniMax-H3 Sol-Attn policy and SM86 candidate wrapper.

This module owns the CPU-testable contract around the real Triton Sol-Attn
candidate in :mod:`minimax_h3_a6000.sol_attn_triton_sm86`:

* packed MiniMax-H3 BTHD tensors with ``head_dim == 128``;
* ``[prefix | video tail | padding]`` metadata and contiguous valid length;
* first-10 denoise steps and first-2 layers stay dense;
* default-off adaptive routing may be guarded by denoise-step/layer ranges;
  outside the guard it keeps the retained tau=1.0/diag sparse route rather
  than silently falling back to an aggressive adaptive tau;
* every prefix KV block is an exact sink, while prefix query rows are replaced
  by dense SDPA output unless the default-off exact-prefix-query experiment is
  explicitly selected; separate default-off scheduler probes may skip only query
  blocks wholly inside that overwritten prefix range, statically schedule the
  prefix-sink exact blocks, consume GROUP=32 exact-route masks via a bitmask,
  or pair the two BV64 value halves behind one shared score/probability stream;
  each probe must preserve exact-block order and dense-prefix overwrite;
* cache stays disabled by contract;
* unsupported inputs strictly fall back to dense reference attention unless the
  caller opts into ``strict``;
* non-contiguous H3 Q/K/V decline by default; the separate stride-aware-V
  switch accepts only the observed non-overlapping fused-QKV value view while
  the legacy diagnostic materialization switch remains an explicit reference;
* copy telemetry separates host enqueue latency from deferred CUDA-event copy
  duration, so asynchronous enqueue time is never reported as GPU copy time;
* importing this module never imports Triton or probes CUDA.

The Triton candidate is loaded only after the env/policy, tensor, metadata, and
explicit SM86 guards pass.
"""

from __future__ import annotations

import functools
import hashlib
import math
import os
import time
from dataclasses import dataclass, field, replace
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
    adaptive_routing: bool = False
    adaptive_routing_policy_error: str | None = None
    adaptive_profile: str | None = None
    adaptive_step_min: int | None = None
    adaptive_step_max: int | None = None
    adaptive_layer_min: int | None = None
    adaptive_layer_max: int | None = None
    allow_sparse: bool = False
    sink_start: int = 0
    sink_tokens: int | None = None
    prefix_query_dense: bool = True
    exact_prefix_query: bool = False
    skip_full_prefix_blocks: bool = False
    static_prefix_sink: bool = False
    bitmask_exact_scheduler: bool = False
    pair_value_halves: bool = False
    shadow_pair_value_halves: bool = False
    shadow_row_state_probe: bool = False
    shadow_max_mismatches: int = 8
    forward_config: str | None = None
    cache_enabled: bool = False
    strict: bool = False
    stride_aware_value: bool = False
    diagnostic_materialize_noncontiguous: bool = False
    diagnostic_materialize_max_bytes: int = 67_108_864

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "SolAttnPolicy":
        env = os.environ if env is None else env
        adaptive_routing = env_enabled("MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_ROUTING", env)
        tau = 1.0
        thresh_type = "diag"
        adaptive_profile = None
        adaptive_step_min = None
        adaptive_step_max = None
        adaptive_layer_min = None
        adaptive_layer_max = None
        routing_errors: list[str] = []
        if adaptive_routing:
            tau, tau_error = _read_env_float_with_error(
                env,
                "MINIMAX_H3_A6000_SOL_ATTN_TAU",
                default=1.0,
            )
            if tau_error is not None:
                routing_errors.append(tau_error)
            elif not (0.0 <= tau <= 8.0):
                routing_errors.append("tau_out_of_range_0_to_8")
            thresh_type = _read_env_str_or_none(env, "MINIMAX_H3_A6000_SOL_ATTN_THRESH_TYPE") or "diag"
            if thresh_type not in ("diag", "exact"):
                routing_errors.append("unsupported_threshold_type")
            adaptive_profile = _read_env_str_or_none(env, "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_PROFILE")
            adaptive_step_min, error = _read_env_optional_nonnegative_int_with_error(
                env, "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_STEP_MIN", error_prefix="adaptive_step_min"
            )
            if error is not None:
                routing_errors.append(error)
            adaptive_step_max, error = _read_env_optional_nonnegative_int_with_error(
                env, "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_STEP_MAX", error_prefix="adaptive_step_max"
            )
            if error is not None:
                routing_errors.append(error)
            adaptive_layer_min, error = _read_env_optional_nonnegative_int_with_error(
                env, "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_LAYER_MIN", error_prefix="adaptive_layer_min"
            )
            if error is not None:
                routing_errors.append(error)
            adaptive_layer_max, error = _read_env_optional_nonnegative_int_with_error(
                env, "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_LAYER_MAX", error_prefix="adaptive_layer_max"
            )
            if error is not None:
                routing_errors.append(error)
            if adaptive_step_min is not None and adaptive_step_max is not None and adaptive_step_min > adaptive_step_max:
                routing_errors.append("adaptive_step_range_invalid")
            if adaptive_layer_min is not None and adaptive_layer_max is not None and adaptive_layer_min > adaptive_layer_max:
                routing_errors.append("adaptive_layer_range_invalid")
        return cls(
            tau=tau,
            thresh_type=thresh_type,
            adaptive_routing=adaptive_routing,
            adaptive_routing_policy_error=";".join(routing_errors) or None,
            adaptive_profile=adaptive_profile,
            adaptive_step_min=adaptive_step_min,
            adaptive_step_max=adaptive_step_max,
            adaptive_layer_min=adaptive_layer_min,
            adaptive_layer_max=adaptive_layer_max,
            dense_first_steps=_read_env_int(
                env,
                "MINIMAX_H3_A6000_SOL_ATTN_DENSE_FIRST_STEPS",
                default=10,
            ),
            dense_first_layers=_read_env_int(
                env,
                "MINIMAX_H3_A6000_SOL_ATTN_DENSE_FIRST_LAYERS",
                default=2,
            ),
            allow_sparse=(
                env_enabled("MINIMAX_H3_A6000_ENABLE_OVERLAY", env)
                and env_enabled("MINIMAX_H3_A6000_ENABLE_TRITON_CANDIDATES", env)
                and env_enabled("MINIMAX_H3_A6000_ENABLE_SOL_ATTN", env)
            ),
            cache_enabled=env_enabled("MINIMAX_H3_A6000_SOL_ATTN_CACHE", env),
            strict=env_enabled("MINIMAX_H3_A6000_SOL_ATTN_STRICT", env),
            stride_aware_value=env_enabled("MINIMAX_H3_A6000_SOL_ATTN_STRIDE_AWARE_V", env),
            exact_prefix_query=env_enabled("MINIMAX_H3_A6000_SOL_ATTN_EXACT_PREFIX_QUERY", env),
            skip_full_prefix_blocks=env_enabled("MINIMAX_H3_A6000_SOL_ATTN_SKIP_FULL_PREFIX_BLOCKS", env),
            static_prefix_sink=env_enabled("MINIMAX_H3_A6000_SOL_ATTN_STATIC_PREFIX_SINK", env),
            bitmask_exact_scheduler=env_enabled("MINIMAX_H3_A6000_SOL_ATTN_BITMASK_SCHEDULER", env),
            pair_value_halves=env_enabled("MINIMAX_H3_A6000_SOL_ATTN_PAIR_VALUE_HALVES", env),
            shadow_pair_value_halves=env_enabled("MINIMAX_H3_A6000_SOL_ATTN_SHADOW_PAIR_VALUE_HALVES", env),
            shadow_row_state_probe=env_enabled("MINIMAX_H3_A6000_SOL_ATTN_SHADOW_ROW_STATE_PROBE", env),
            shadow_max_mismatches=_read_env_int(
                env,
                "MINIMAX_H3_A6000_SOL_ATTN_SHADOW_MAX_MISMATCHES",
                default=8,
            ),
            forward_config=_read_env_str_or_none(env, "MINIMAX_H3_A6000_SOL_ATTN_FORWARD_CONFIG"),
            diagnostic_materialize_noncontiguous=env_enabled(
                "MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_MATERIALIZE", env
            ),
            diagnostic_materialize_max_bytes=_read_env_int(
                env,
                "MINIMAX_H3_A6000_SOL_ATTN_MATERIALIZE_MAX_BYTES",
                default=67_108_864,
            ),
        )


@dataclass
class SolAttnTelemetry:
    """Per-run Sol-Attn counters for logs and external GPU gates."""

    dense_calls: int = 0
    sparse_candidate_calls: int = 0
    sparse_calls: int = 0
    fallback_calls: int = 0
    prefix_query_dense_calls: int = 0
    exact_prefix_query_calls: int = 0
    decline_reasons: dict[str, int] = field(default_factory=dict)
    fallback_reasons: dict[str, int] = field(default_factory=dict)
    sink_ranges: list[tuple[int, int]] = field(default_factory=list)
    density_samples: list[dict[str, Any]] = field(default_factory=list)
    layout_samples: list[dict[str, Any]] = field(default_factory=list)
    diagnostic_output_records: list[dict[str, Any]] = field(default_factory=list)
    diagnostic_output_digest_failures: int = 0
    materialize_copy_count: int = 0
    materialize_copy_bytes: int = 0
    materialize_copy_by_tensor: dict[str, int] = field(default_factory=dict)
    # Compatibility field: host-side enqueue latency, never GPU copy duration.
    materialize_latency_ms: float = 0.0
    materialize_gpu_copy_latency_ms: float = 0.0
    materialize_gpu_timing_failures: int = 0
    stride_aware_value_calls: int = 0
    stride_aware_value_bytes: int = 0
    shadow_pair_value_halves_calls: int = 0
    shadow_pair_value_halves_equal_calls: int = 0
    shadow_pair_value_halves_mismatch_count: int = 0
    shadow_pair_value_halves_record_failures: int = 0
    shadow_pair_value_halves_records: list[dict[str, Any]] = field(default_factory=list)
    _materialize_gpu_event_pairs: list[tuple[Any, Any]] = field(default_factory=list, repr=False)

    @property
    def materialize_host_enqueue_latency_ms(self) -> float:
        return self.materialize_latency_ms

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

    def record_sparse_success(self, *, prefix_query_dense: bool, exact_prefix_query: bool = False) -> None:
        self.sparse_calls += 1
        if prefix_query_dense:
            self.prefix_query_dense_calls += 1
        if exact_prefix_query:
            self.exact_prefix_query_calls += 1

    def record_layout(self, *, stage: str, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
        if len(self.layout_samples) >= 32:
            return
        self.layout_samples.append(
            {
                "stage": str(stage),
                "tensors": [
                    _tensor_layout("query", query),
                    _tensor_layout("key", key),
                    _tensor_layout("value", value),
                ],
            }
        )

    def record_diagnostic_output(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        metadata: PackedH3Metadata,
        step_index: int,
        layer_index: int,
        policy: SolAttnPolicy,
        sink_range: tuple[int, int],
        stage: str,
    ) -> None:
        """Record bounded no-raw-tensor call metadata plus output digest.

        This diagnostic is deliberately default-off because hashing the returned
        BF16 attention tensor copies it to host memory and synchronizes CUDA.
        The JSON record contains only layouts, scalar H3 metadata, policy flags,
        and a SHA-256 digest of the contiguous output bytes; tensor values are
        never serialized.
        """

        if not sol_attn_diagnostic_output_digest_enabled():
            return
        max_calls = _diagnostic_output_max_calls()
        if max_calls <= 0 or len(self.diagnostic_output_records) >= max_calls:
            return
        record: dict[str, Any] = {
            "call_index": len(self.diagnostic_output_records),
            "stage": str(stage),
            "raw_tensor_exported": False,
            "input_layouts": [
                _tensor_layout("query", query),
                _tensor_layout("key", key),
                _tensor_layout("value", value),
            ],
            "output_layout": _tensor_layout("output", output),
            "metadata": {
                "step_index": int(step_index),
                "layer_index": int(layer_index),
                "prefix_len": int(metadata.prefix_len),
                "latent_grid": [int(x) for x in metadata.latent_grid],
                "valid_length": int(metadata.valid_length),
                "total_length": int(metadata.total_length),
                "sink_start": int(sink_range[0]),
                "sink_end": int(sink_range[1]),
            },
            "policy": {
                "tau": float(policy.tau),
                "thresh_type": str(policy.thresh_type),
                "adaptive_routing": bool(policy.adaptive_routing),
                "dense_first_steps": int(policy.dense_first_steps),
                "dense_first_layers": int(policy.dense_first_layers),
                "stride_aware_value": bool(policy.stride_aware_value),
                "skip_full_prefix_blocks": bool(policy.skip_full_prefix_blocks),
                "dense_prefix_overwrite": bool(policy.prefix_query_dense),
                "exact_prefix_query": bool(policy.exact_prefix_query),
                "pair_value_halves": bool(policy.pair_value_halves),
                "cache_enabled": bool(policy.cache_enabled),
                "forward_config": policy.forward_config,
            },
            "comparison_tolerance_policy": {
                "output_sha256_mismatches_allowed": 0,
                "metadata_mismatches_allowed": 0,
                "layout_mismatches_allowed": 0,
                "raw_tensor_values_available": False,
            },
        }
        try:
            record["output_digest"] = {
                "algorithm": "sha256",
                "scope": "contiguous_output_tensor_bytes_with_shape_and_dtype_prefix",
                "sha256": _tensor_sha256_digest(output),
            }
        except Exception as exc:  # noqa: BLE001 - diagnostics must not perturb inference outcome
            self.diagnostic_output_digest_failures += 1
            record["output_digest_error"] = f"{type(exc).__name__}: {exc}"
        self.diagnostic_output_records.append(record)

    def record_shadow_pair_value_halves(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        retained_output: torch.Tensor,
        candidate_output: torch.Tensor,
        metadata: PackedH3Metadata,
        step_index: int,
        layer_index: int,
        policy: SolAttnPolicy,
        sink_range: tuple[int, int],
        sparse_call_index: int,
        softmax_scale: float | None = None,
    ) -> None:
        """Record bounded same-input current-vs-pair scalar divergence metadata.

        The diagnostic is default-off and always returns ``retained_output`` to
        the caller.  It may synchronize/copy scalar reductions to host, but it
        never serializes raw tensor values or filesystem paths.
        """

        self.shadow_pair_value_halves_calls += 1
        try:
            if torch.equal(retained_output, candidate_output):
                self.shadow_pair_value_halves_equal_calls += 1
                return
            self.shadow_pair_value_halves_mismatch_count += 1
            max_records = max(0, int(policy.shadow_max_mismatches))
            if len(self.shadow_pair_value_halves_records) >= max_records:
                return
            self.shadow_pair_value_halves_records.append(
                _shadow_pair_value_halves_record(
                    call_index=self.shadow_pair_value_halves_calls - 1,
                    sparse_call_index=sparse_call_index,
                    query=query,
                    key=key,
                    value=value,
                    retained_output=retained_output,
                    candidate_output=candidate_output,
                    metadata=metadata,
                    step_index=step_index,
                    layer_index=layer_index,
                    policy=policy,
                    sink_range=sink_range,
                    softmax_scale=softmax_scale,
                )
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics must not perturb inference outcome
            self.shadow_pair_value_halves_record_failures += 1
            if len(self.shadow_pair_value_halves_records) < max(0, int(policy.shadow_max_mismatches)):
                self.shadow_pair_value_halves_records.append(
                    {
                        "call_index": self.shadow_pair_value_halves_calls - 1,
                        "sparse_call_index": int(sparse_call_index),
                        "raw_tensor_exported": False,
                        "record_error": f"{type(exc).__name__}: {exc}",
                    }
                )

    def record_materialize(
        self,
        *,
        by_tensor: dict[str, int],
        host_enqueue_latency_ms: float,
        gpu_event_pair: tuple[Any, Any] | None = None,
    ) -> None:
        self.materialize_copy_count += len(by_tensor)
        self.materialize_copy_bytes += sum(int(v) for v in by_tensor.values())
        self.materialize_latency_ms += float(host_enqueue_latency_ms)
        if gpu_event_pair is not None:
            self._materialize_gpu_event_pairs.append(gpu_event_pair)
        for name, value in by_tensor.items():
            self.materialize_copy_by_tensor[name] = self.materialize_copy_by_tensor.get(name, 0) + int(value)

    def record_stride_aware_value(self, value: torch.Tensor) -> None:
        self.stride_aware_value_calls += 1
        copy_bytes = _copy_bytes_for(value)
        if copy_bytes is not None:
            self.stride_aware_value_bytes += copy_bytes

    def finalize_materialize_gpu_timing(self) -> float:
        """Resolve deferred CUDA events after the measured request has completed."""

        pairs, self._materialize_gpu_event_pairs = self._materialize_gpu_event_pairs, []
        for start, end in pairs:
            try:
                end.synchronize()
                self.materialize_gpu_copy_latency_ms += float(start.elapsed_time(end))
            except Exception:  # noqa: BLE001 - telemetry must not break inference
                self.materialize_gpu_timing_failures += 1
        return self.materialize_gpu_copy_latency_ms


def _read_env_int(env: Mapping[str, str], name: str, *, default: int) -> int:
    try:
        return int(str(env.get(name, DEFAULT_ENV_SWITCHES.get(name, str(default)))).strip())
    except (TypeError, ValueError):
        return int(default)


def _read_env_optional_nonnegative_int_with_error(
    env: Mapping[str, str], name: str, *, error_prefix: str
) -> tuple[int | None, str | None]:
    raw = str(env.get(name, DEFAULT_ENV_SWITCHES.get(name, ""))).strip()
    if raw == "":
        return None, None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None, f"{error_prefix}_parse_error"
    if value < 0:
        return None, f"{error_prefix}_negative"
    return value, None


def _read_env_float_with_error(
    env: Mapping[str, str],
    name: str,
    *,
    default: float,
) -> tuple[float, str | None]:
    raw = str(env.get(name, DEFAULT_ENV_SWITCHES.get(name, str(default)))).strip()
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(default), "tau_parse_error"
    if not math.isfinite(value):
        return float(default), "tau_not_finite"
    return value, None


def _read_env_str_or_none(env: Mapping[str, str], name: str) -> str | None:
    value = str(env.get(name, DEFAULT_ENV_SWITCHES.get(name, ""))).strip()
    return value or None


def _tensor_layout(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "name": name,
        "shape": [int(x) for x in tensor.shape],
        "stride": [int(x) for x in tensor.stride()],
        "storage_offset": int(tensor.storage_offset()),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device_type": tensor.device.type,
        "is_contiguous": bool(tensor.is_contiguous()),
    }


def sol_attn_diagnostic_output_digest_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Whether to hash Sol-Attn outputs for diagnostic localization.

    The switch is default-off because it copies attention outputs to host memory
    and synchronizes CUDA. It is intended only for bounded current-vs-current and
    current-vs-candidate correctness diagnostics.
    """

    return env_enabled("MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_OUTPUT_DIGEST", env)


def _diagnostic_output_max_calls(env: Mapping[str, str] | None = None) -> int:
    env = os.environ if env is None else env
    return max(
        0,
        _read_env_int(
            env,
            "MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_MAX_CALLS",
            default=256,
        ),
    )


def _tensor_sha256_digest(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().contiguous()
    byte_tensor = contiguous.view(torch.uint8).cpu()
    data = byte_tensor.numpy().tobytes(order="C")
    h = hashlib.sha256()
    h.update(str([int(x) for x in contiguous.shape]).encode("ascii"))
    h.update(str(contiguous.dtype).encode("ascii"))
    h.update(data)
    return h.hexdigest()


@functools.lru_cache(maxsize=1)
def _source_code_hashes() -> dict[str, str]:
    """Return hashes for the installed Sol-Attn code without recording paths."""

    base = os.path.dirname(__file__)
    inputs = {
        "sol_attn_backend_py_sha256": __file__,
        "sol_attn_triton_sm86_py_sha256": os.path.join(base, "sol_attn_triton_sm86.py"),
    }
    hashes: dict[str, str] = {}
    for key, path in inputs.items():
        try:
            with open(path, "rb") as f:
                hashes[key] = hashlib.sha256(f.read()).hexdigest()
        except OSError as exc:
            hashes[key] = f"unavailable:{type(exc).__name__}"
    return hashes


def _shadow_argmax_bucket(flat_index: int, shape: tuple[int, ...], *, prefix: int, valid: int) -> dict[str, Any]:
    if len(shape) != 4 or flat_index < 0:
        return {"region": "unknown", "flat_index_available": False}
    _batch, tokens, heads, head_dim = (int(x) for x in shape)
    dim = flat_index % head_dim
    tmp = flat_index // head_dim
    head = tmp % heads
    tmp //= heads
    token = tmp % tokens
    batch = tmp // tokens
    if token < int(prefix):
        region = "prefix"
    elif token < int(valid):
        region = "tail"
    else:
        region = "padding"
    return {
        "flat_index_available": True,
        "region": region,
        "batch": int(batch),
        "token_index": int(token),
        "token_block64": int(token) // SOL_ATTN_BLOCK_SIZE,
        "head": int(head),
        "dim": int(dim),
        "value_half": "lo" if int(dim) < 64 else "hi",
    }


def _region_equal(a: torch.Tensor, b: torch.Tensor, start: int, end: int) -> bool:
    start = max(0, int(start))
    end = max(start, int(end))
    if end <= start:
        return True
    return bool(torch.equal(a[:, start:end], b[:, start:end]))


def _digest_int_sequence(values: list[int], *, domain: str) -> str:
    h = hashlib.sha256()
    h.update(domain.encode("ascii"))
    h.update(str(len(values)).encode("ascii"))
    for value in values:
        h.update(int(value).to_bytes(8, "little", signed=False))
    return h.hexdigest()


def _scalar_float(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu().item())


def _row_vector_summary(vector: torch.Tensor) -> dict[str, dict[str, float]]:
    vec = vector.detach().float()
    lo = vec[:64]
    hi = vec[64:]
    return {
        "lo": {
            "signed_sum": _scalar_float(lo.sum()),
            "abs_sum": _scalar_float(lo.abs().sum()),
            "max_abs": _scalar_float(lo.abs().max()) if lo.numel() else 0.0,
        },
        "hi": {
            "signed_sum": _scalar_float(hi.sum()),
            "abs_sum": _scalar_float(hi.abs().sum()),
            "max_abs": _scalar_float(hi.abs().max()) if hi.numel() else 0.0,
        },
    }


def _shadow_pair_value_halves_row_state_probe(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    retained_output: torch.Tensor,
    candidate_output: torch.Tensor,
    metadata: PackedH3Metadata,
    policy: SolAttnPolicy,
    sink_range: tuple[int, int],
    target_bucket: dict[str, Any],
    softmax_scale: float | None,
) -> dict[str, Any]:
    """Record scalar live-QKV route, row-state, and PV summaries.

    The probe recomputes the source-level Sol-Attn row state for the single
    mismatching row/head from the same live Q/K/V tensors.  It records only
    counts, digests, equality booleans, and scalar summaries; no tensor values,
    routes, media, paths, or raw rows are serialized.
    """

    record: dict[str, Any] = {
        "schema_version": "minimax_h3_a6000_sol_attn_pair_value_halves_row_state_probe_v1",
        "status": "not_run",
        "raw_tensor_exported": False,
        "raw_tensor_payload_available": False,
        "same_live_qkv_and_metadata": True,
        "probe_kind": "source_level_row_state_reference_from_live_qkv",
    }
    if not bool(target_bucket.get("flat_index_available")):
        return {**record, "status": "skipped_no_argmax_index"}
    if str(target_bucket.get("region")) == "padding":
        return {**record, "status": "skipped_padding_target"}
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return {**record, "status": "skipped_non_bthd"}
    batch = int(target_bucket.get("batch", 0))
    token = int(target_bucket.get("token_index", -1))
    head = int(target_bucket.get("head", -1))
    dim = int(target_bucket.get("dim", -1))
    active = int(metadata.valid_length)
    total = int(metadata.total_length)
    if batch != 0 or token < 0 or token >= active or head < 0 or head >= int(query.shape[2]) or dim < 0 or dim >= 128:
        return {**record, "status": "skipped_invalid_target"}
    if int(query.shape[1]) != total or int(key.shape[1]) != total or int(value.shape[1]) != total:
        return {**record, "status": "skipped_shape_metadata_mismatch"}

    try:
        with torch.no_grad():
            block = token // SOL_ATTN_BLOCK_SIZE
            block_start = block * SOL_ATTN_BLOCK_SIZE
            block_end = min(active, block_start + SOL_ATTN_BLOCK_SIZE)
            q_len = max(1, block_end - block_start)
            blocks = _ceil_div(active, SOL_ATTN_BLOCK_SIZE)
            padded_active = blocks * SOL_ATTN_BLOCK_SIZE
            pad = padded_active - active
            q_block = query[0, block_start:block_end, head, :].detach()
            q_row = query[0, token, head, :].detach().float()
            k_head = key[0, :active, head, :].detach()
            v_head = value[0, :active, head, :].detach()
            if pad:
                k_pad = torch.zeros((pad, 128), dtype=k_head.dtype, device=k_head.device)
                v_pad = torch.zeros((pad, 128), dtype=v_head.dtype, device=v_head.device)
                k_head_padded = torch.cat((k_head, k_pad), dim=0)
                v_head_padded = torch.cat((v_head, v_pad), dim=0)
            else:
                k_head_padded = k_head
                v_head_padded = v_head
            k_blocks = k_head_padded.reshape(blocks, SOL_ATTN_BLOCK_SIZE, 128)
            v_blocks = v_head_padded.reshape(blocks, SOL_ATTN_BLOCK_SIZE, 128)
            lengths = torch.full((blocks,), float(SOL_ATTN_BLOCK_SIZE), dtype=torch.float32, device=query.device)
            if pad:
                lengths[-1] = float(SOL_ATTN_BLOCK_SIZE - pad)
            kc = (k_blocks.float().sum(dim=1) / lengths[:, None]).to(torch.bfloat16).float()
            vc = v_blocks.float().sum(dim=1).to(torch.bfloat16).float()
            q_centroid = q_block.float().sum(dim=0) / float(q_len)
            scale = float(query.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale)
            scale_log2 = scale * 1.4426950408889634
            mean_kc = kc.mean(dim=0)
            var_kc = torch.clamp((kc * kc).mean(dim=0) - mean_kc * mean_kc, min=0.0)
            route_threshold = (q_centroid * mean_kc).sum() * scale_log2 + float(policy.tau) * torch.sqrt(
                (q_centroid * q_centroid * var_kc).sum() * (scale_log2 * scale_log2) + 1.0e-6
            )
            route_scores = torch.matmul(q_block.float(), kc.T) * scale_log2
            dynamic = route_scores.sum(dim=0) / float(q_len) > route_threshold
            block_indices = torch.arange(blocks, device=query.device)
            local = (block_indices - int(block)).abs() <= 1
            sink_start_block, sink_end_block = int(sink_range[0]), int(sink_range[1])
            sink = (block_indices >= sink_start_block) & (block_indices < sink_end_block)
            exact = dynamic | local | sink
            approx = ~exact
            exact_order = [int(x) for x in torch.nonzero(exact, as_tuple=False).flatten().detach().cpu().tolist()]
            route_digest = _digest_int_sequence(
                [idx for idx, is_exact in enumerate(exact.detach().cpu().tolist()) if bool(is_exact)],
                domain="exact-block-mask-v1",
            )
            exact_order_digest = _digest_int_sequence(exact_order, domain="exact-block-order-v1")

            approx_scores = torch.matmul(kc[approx], q_row) * scale_log2 if bool(approx.any().item()) else kc.new_empty((0,))
            token_ids = torch.arange(active, device=query.device)
            exact_token_mask = exact[token_ids // SOL_ATTN_BLOCK_SIZE]
            exact_k = k_head[exact_token_mask].float()
            exact_v = v_head[exact_token_mask].float()
            exact_scores = torch.matmul(exact_k, q_row) * scale_log2 if exact_k.numel() else kc.new_empty((0,))
            score_parts = []
            if approx_scores.numel():
                score_parts.append(approx_scores)
            if exact_scores.numel():
                score_parts.append(exact_scores)
            if not score_parts:
                return {**record, "status": "probe_error", "error": "empty_route"}
            row_max = torch.cat(score_parts).max()
            approx_prob = torch.exp2(approx_scores - row_max) if approx_scores.numel() else approx_scores
            exact_prob = torch.exp2(exact_scores - row_max) if exact_scores.numel() else exact_scores
            approx_row_sum = (approx_prob * lengths[approx]).sum() if approx_prob.numel() else row_max.new_tensor(0.0)
            exact_row_sum = exact_prob.sum() if exact_prob.numel() else row_max.new_tensor(0.0)
            row_sum = approx_row_sum + exact_row_sum
            if not bool(torch.isfinite(row_sum).item()) or _scalar_float(row_sum) <= 0.0:
                return {**record, "status": "probe_error", "error": "nonfinite_or_zero_row_sum"}
            approx_num = (
                (approx_prob.to(torch.bfloat16).float()[:, None] * vc[approx]).sum(dim=0)
                if approx_prob.numel()
                else torch.zeros((128,), dtype=torch.float32, device=query.device)
            )
            exact_num = (
                (exact_prob.to(torch.bfloat16).float()[:, None] * exact_v).sum(dim=0)
                if exact_prob.numel()
                else torch.zeros((128,), dtype=torch.float32, device=query.device)
            )
            numerator = approx_num + exact_num
            reference_row = (numerator / row_sum).to(torch.bfloat16)
            retained_row = retained_output[0, token, head, :].detach()
            candidate_row = candidate_output[0, token, head, :].detach()
            current_diff = (retained_row.float() - reference_row.float()).abs()
            pair_diff = (candidate_row.float() - reference_row.float()).abs()
            route_counts = {
                "total_blocks": int(blocks),
                "approx_block_count": int(approx.sum().item()),
                "exact_block_count": int(exact.sum().item()),
                "dynamic_threshold_block_count": int(dynamic.sum().item()),
                "local_block_count": int(local.sum().item()),
                "sink_block_count": int(sink.sum().item()),
                "exact_token_count": int(exact_token_mask.sum().item()),
            }
            return {
                **record,
                "status": "pass",
                "target": {
                    "token_index": int(token),
                    "token_block64": int(block),
                    "head": int(head),
                    "dim": int(dim),
                    "value_half": "lo" if int(dim) < 64 else "hi",
                    "region": str(target_bucket.get("region")),
                },
                "route": {
                    **route_counts,
                    "group_size": GROUP_SIZE if "GROUP_SIZE" in globals() else 32,
                    "current_route_digest": route_digest,
                    "pair_value_halves_route_digest": route_digest,
                    "route_digest_equal": True,
                    "exact_order_digest": exact_order_digest,
                    "pair_value_halves_exact_order_digest": exact_order_digest,
                    "exact_order_digest_equal": True,
                    "route_values_exported": False,
                    "exact_block_indices_exported": False,
                    "threshold_scalar": _scalar_float(route_threshold),
                },
                "row_state": {
                    "current_reference_row_max": _scalar_float(row_max),
                    "pair_value_halves_reference_row_max": _scalar_float(row_max),
                    "row_max_abs_delta": 0.0,
                    "current_reference_row_sum": _scalar_float(row_sum),
                    "pair_value_halves_reference_row_sum": _scalar_float(row_sum),
                    "row_sum_abs_delta": 0.0,
                    "approx_row_sum_component": _scalar_float(approx_row_sum),
                    "exact_row_sum_component": _scalar_float(exact_row_sum),
                    "finite": bool(torch.isfinite(row_max).item() and torch.isfinite(row_sum).item()),
                },
                "target_row_current_reference": {
                    "retained_current_row_equal_reference_bf16": bool(torch.equal(retained_row, reference_row)),
                    "retained_current_row_max_abs_vs_reference": _scalar_float(current_diff.max()),
                    "pair_value_halves_row_equal_reference_bf16": bool(torch.equal(candidate_row, reference_row)),
                    "pair_value_halves_row_max_abs_vs_reference": _scalar_float(pair_diff.max()),
                    "target_dim_retained_current": _scalar_float(retained_row[dim]),
                    "target_dim_pair_value_halves": _scalar_float(candidate_row[dim]),
                    "target_dim_reference_bf16": _scalar_float(reference_row[dim]),
                    "target_dim_retained_abs_delta_reference": _scalar_float((retained_row[dim].float() - reference_row[dim].float()).abs()),
                    "target_dim_pair_abs_delta_reference": _scalar_float((candidate_row[dim].float() - reference_row[dim].float()).abs()),
                    "target_dim_retained_pair_abs_delta": _scalar_float((retained_row[dim].float() - candidate_row[dim].float()).abs()),
                },
                "pv_contribution_summaries": {
                    "approximate_blocks": _row_vector_summary(approx_num),
                    "exact_tokens": _row_vector_summary(exact_num),
                    "total_numerator": _row_vector_summary(numerator),
                    "lo_hi_raw_vectors_exported": False,
                },
            }
    except Exception as exc:  # noqa: BLE001 - diagnostics must not perturb inference outcome
        return {**record, "status": "probe_error", "error": f"{type(exc).__name__}: {exc}"}


def _shadow_pair_value_halves_record(
    *,
    call_index: int,
    sparse_call_index: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    retained_output: torch.Tensor,
    candidate_output: torch.Tensor,
    metadata: PackedH3Metadata,
    step_index: int,
    layer_index: int,
    policy: SolAttnPolicy,
    sink_range: tuple[int, int],
    softmax_scale: float | None = None,
) -> dict[str, Any]:
    diff = (retained_output - candidate_output).abs()
    if diff.numel() > 0:
        flat = diff.reshape(-1)
        max_abs_tensor, flat_index_tensor = torch.max(flat, dim=0)
        max_abs = float(max_abs_tensor.float().item())
        mean_abs = float(diff.mean(dtype=torch.float32).item())
        flat_index = int(flat_index_tensor.item())
    else:
        max_abs = 0.0
        mean_abs = 0.0
        flat_index = -1
    retained_finite = torch.isfinite(retained_output)
    candidate_finite = torch.isfinite(candidate_output)
    prefix = int(metadata.prefix_len)
    valid = int(metadata.valid_length)
    total = int(metadata.total_length)
    bucket = _shadow_argmax_bucket(
        flat_index,
        tuple(int(x) for x in retained_output.shape),
        prefix=prefix,
        valid=valid,
    )
    record = {
        "schema_version": "minimax_h3_a6000_sol_attn_shadow_pair_value_halves_record_v1",
        "call_index": int(call_index),
        "sparse_call_index": int(sparse_call_index),
        "same_live_qkv_and_metadata": True,
        "raw_tensor_exported": False,
        "raw_tensor_values_available": False,
        "candidate_marker": "pair_value_halves_shadow_candidate",
        "candidate_pair_value_halves": True,
        "returned_output": "retained_current",
        "input_layouts": [_tensor_layout("query", query), _tensor_layout("key", key), _tensor_layout("value", value)],
        "output_layouts": {
            "retained_current": _tensor_layout("retained_current_output", retained_output),
            "pair_value_halves_candidate": _tensor_layout("pair_value_halves_output", candidate_output),
        },
        "metadata": {
            "step_index": int(step_index),
            "layer_index": int(layer_index),
            "prefix_len": prefix,
            "latent_grid": [int(x) for x in metadata.latent_grid],
            "valid_length": valid,
            "total_length": total,
            "sink_start": int(sink_range[0]),
            "sink_end": int(sink_range[1]),
        },
        "policy": {
            "tau": float(policy.tau),
            "thresh_type": str(policy.thresh_type),
            "adaptive_routing": bool(policy.adaptive_routing),
            "dense_first_steps": int(policy.dense_first_steps),
            "dense_first_layers": int(policy.dense_first_layers),
            "stride_aware_value": bool(policy.stride_aware_value),
            "skip_full_prefix_blocks": bool(policy.skip_full_prefix_blocks),
            "dense_prefix_overwrite": bool(policy.prefix_query_dense),
            "exact_prefix_query": bool(policy.exact_prefix_query),
            "retained_pair_value_halves": False,
            "candidate_pair_value_halves": True,
            "cache_enabled": bool(policy.cache_enabled),
            "forward_config": policy.forward_config,
            "shadow_max_mismatches": int(policy.shadow_max_mismatches),
            "shadow_row_state_probe": bool(policy.shadow_row_state_probe),
        },
        "error": {
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "argmax_region_bucket": bucket,
        },
        "finite_equality": {
            "retained_all_finite": bool(retained_finite.all().item()),
            "candidate_all_finite": bool(candidate_finite.all().item()),
            "finite_mask_equal": bool(torch.equal(retained_finite, candidate_finite)),
        },
        "region_equality": {
            "prefix_equal": _region_equal(retained_output, candidate_output, 0, prefix),
            "tail_equal": _region_equal(retained_output, candidate_output, prefix, valid),
            "padding_equal": _region_equal(retained_output, candidate_output, valid, total),
        },
        "code_hashes": _source_code_hashes(),
    }
    if policy.shadow_row_state_probe:
        record["row_state_probe"] = _shadow_pair_value_halves_row_state_probe(
            query=query,
            key=key,
            value=value,
            retained_output=retained_output,
            candidate_output=candidate_output,
            metadata=metadata,
            policy=policy,
            sink_range=sink_range,
            target_bucket=bucket,
            softmax_scale=softmax_scale,
        )
    return record


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def stride_aware_value_layout_reason(value: torch.Tensor) -> str | None:
    """Validate the one source-backed non-contiguous H3 value layout.

    PyTorch strides are element jumps and ``storage_offset`` is expressed in
    storage elements.  The accepted view is exactly the V third of a fused
    ``[B, T, 3 * H * D]`` BF16 projection, reshaped to BTHD: D and H are packed,
    token stride is three head planes, and V starts at the third plane.  The
    batch stride may retain tail-padding rows after slicing to valid tokens but
    must remain a whole, non-overlapping number of fused token rows.
    """

    if value.ndim != 4:
        return "rank"
    batch, tokens, heads, head_dim = (int(x) for x in value.shape)
    if batch != 1:
        return "batch_not_one"
    if tokens <= 0 or heads <= 0 or head_dim != 128:
        return "shape"
    strides = tuple(int(x) for x in value.stride())
    if len(strides) != 4 or any(x <= 0 for x in strides):
        return "non_positive_stride"
    stride_b, stride_t, stride_h, stride_d = strides
    if stride_d != 1:
        return "inner_stride"
    if stride_h != head_dim:
        return "head_stride"
    head_plane = heads * head_dim
    if stride_t != 3 * head_plane:
        return "token_stride"
    if stride_b < tokens * stride_t or stride_b % stride_t:
        return "batch_stride"
    storage_offset = int(value.storage_offset())
    if storage_offset < 0 or storage_offset % stride_t != 2 * head_plane:
        return "storage_offset"
    max_storage_index = storage_offset + (tokens - 1) * stride_t + (heads - 1) * stride_h + head_dim - 1
    try:
        storage_elements = int(value.untyped_storage().nbytes()) // int(value.element_size())
    except Exception:  # noqa: BLE001 - an uninspectable storage is unsupported
        return "storage_uninspectable"
    if max_storage_index >= storage_elements:
        return "storage_bounds"
    return None


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
    if policy.adaptive_routing_policy_error:
        return f"unsupported_adaptive_routing_policy:{policy.adaptive_routing_policy_error}"
    if not math.isfinite(float(policy.tau)):
        return "unsupported_tau"
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
    if not (query.is_contiguous() and key.is_contiguous()):
        return "unsupported_contiguity"
    if not value.is_contiguous():
        if not policy.stride_aware_value:
            return "unsupported_contiguity"
        value_layout_reason = stride_aware_value_layout_reason(value)
        if value_layout_reason is not None:
            return f"unsupported_stride_aware_v_layout:{value_layout_reason}"
    if query.device.type != "cuda" or key.device != query.device or value.device != query.device:
        return "unsupported_device"
    if not sm86_capability_guard(device_capability):
        return "unsupported_or_unprobed_sm"
    return None


def adaptive_routing_guard_reason(policy: SolAttnPolicy, *, step_index: int, layer_index: int) -> str | None:
    """Return why a valid adaptive policy should use retained tau=1.0/diag for this call.

    The guard is not a dense fallback.  It lets a default-off adaptive profile
    protect identity/background/motion-critical denoise regions by routing them
    through the retained sparse tau=1.0/diag policy while applying the adaptive
    tau only inside an explicitly configured lower-risk step/layer range.
    """

    if not policy.adaptive_routing:
        return "adaptive_routing_disabled"
    if policy.adaptive_step_min is not None and step_index < policy.adaptive_step_min:
        return "step_before_adaptive_min"
    if policy.adaptive_step_max is not None and step_index > policy.adaptive_step_max:
        return "step_after_adaptive_max"
    if policy.adaptive_layer_min is not None and layer_index < policy.adaptive_layer_min:
        return "layer_before_adaptive_min"
    if policy.adaptive_layer_max is not None and layer_index > policy.adaptive_layer_max:
        return "layer_after_adaptive_max"
    return None


def effective_sol_attn_policy_for_call(
    policy: SolAttnPolicy, *, step_index: int, layer_index: int
) -> tuple[SolAttnPolicy, str | None]:
    """Return the actual per-call policy and optional guard reason.

    Invalid adaptive policies are left unchanged so ``decline_reason`` can fail
    closed with the recorded parse/range error.  Valid guarded inactive calls
    use the retained sparse tau=1.0/diag route instead of the adaptive tau.
    """

    if not policy.adaptive_routing or policy.adaptive_routing_policy_error:
        return policy, None
    reason = adaptive_routing_guard_reason(policy, step_index=step_index, layer_index=layer_index)
    if reason is None:
        return policy, None
    return replace(policy, adaptive_routing=False, tau=1.0, thresh_type="diag"), reason


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


def _copy_bytes_for(tensor: torch.Tensor) -> int | None:
    numel = int(tensor.numel())
    element_size = int(tensor.element_size())
    if numel < 0 or element_size <= 0:
        return None
    if numel > ((1 << 63) - 1) // element_size:
        return None
    return numel * element_size


def _diagnostic_materialize_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    policy: SolAttnPolicy,
    telemetry: SolAttnTelemetry,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str | None]:
    if query.is_contiguous() and key.is_contiguous() and value.is_contiguous():
        return query, key, value, None
    if not policy.diagnostic_materialize_noncontiguous:
        return query, key, value, "unsupported_contiguity"
    max_bytes = int(policy.diagnostic_materialize_max_bytes)
    if max_bytes <= 0:
        return query, key, value, "diagnostic_materialize_invalid_cap"

    tensors = {"query": query, "key": key, "value": value}
    by_tensor: dict[str, int] = {}
    total = 0
    for name, tensor in tensors.items():
        if tensor.is_contiguous():
            continue
        copy_bytes = _copy_bytes_for(tensor)
        if copy_bytes is None:
            return query, key, value, "diagnostic_materialize_allocation_overflow"
        total += copy_bytes
        if total > max_bytes:
            return query, key, value, "diagnostic_materialize_cap_exceeded"
        by_tensor[name] = copy_bytes

    gpu_event_pair: tuple[Any, Any] | None = None
    try:
        if value.device.type == "cuda":
            gpu_event_pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            gpu_event_pair[0].record()
        start = time.perf_counter_ns()
        out = tuple(t.contiguous() if not t.is_contiguous() else t for t in (query, key, value))
        host_enqueue_latency_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        if gpu_event_pair is not None:
            gpu_event_pair[1].record()
    except Exception as exc:  # noqa: BLE001 - diagnostic path must fail closed
        return query, key, value, f"diagnostic_materialize_error:{type(exc).__name__}"
    telemetry.record_materialize(
        by_tensor=by_tensor,
        host_enqueue_latency_ms=host_enqueue_latency_ms,
        gpu_event_pair=gpu_event_pair,
    )
    return out[0], out[1], out[2], None


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

    requested_policy = SolAttnPolicy() if policy is None else policy
    policy, adaptive_guard_reason_for_call = effective_sol_attn_policy_for_call(
        requested_policy, step_index=step_index, layer_index=layer_index
    )
    telemetry = SolAttnTelemetry() if telemetry is None else telemetry
    telemetry.record_layout(stage="pre_decline", query=query, key=key, value=value)
    candidate_query, candidate_key, candidate_value = query, key, value
    reason = decline_reason(
        query=candidate_query,
        key=candidate_key,
        value=candidate_value,
        metadata=metadata,
        step_index=step_index,
        layer_index=layer_index,
        policy=policy,
        device_capability=device_capability,
    )
    if reason == "unsupported_contiguity" and not policy.stride_aware_value:
        candidate_query, candidate_key, candidate_value, materialize_reason = _diagnostic_materialize_qkv(
            candidate_query,
            candidate_key,
            candidate_value,
            policy=policy,
            telemetry=telemetry,
        )
        if materialize_reason is None:
            telemetry.record_layout(
                stage="post_diagnostic_materialize",
                query=candidate_query,
                key=candidate_key,
                value=candidate_value,
            )
            reason = decline_reason(
                query=candidate_query,
                key=candidate_key,
                value=candidate_value,
                metadata=metadata,
                step_index=step_index,
                layer_index=layer_index,
                policy=policy,
                device_capability=device_capability,
            )
        else:
            reason = materialize_reason
    if reason is not None:
        telemetry.record_decline(reason)
        return dense_attention_packed_reference(query, key, value, metadata=metadata, softmax_scale=softmax_scale)

    assert metadata is not None  # narrowed by decline_reason
    sink_range = derive_sink_range(metadata, policy)
    density = estimate_sparse_density(metadata, policy)
    density = {
        **density,
        "step_index": int(step_index),
        "layer_index": int(layer_index),
        "tau": float(policy.tau),
        "thresh_type": str(policy.thresh_type),
        "adaptive_routing": bool(policy.adaptive_routing),
    }
    if requested_policy.adaptive_routing:
        density = {
            **density,
            "adaptive_routing_requested": True,
            "adaptive_profile": requested_policy.adaptive_profile,
            "adaptive_candidate_tau": float(requested_policy.tau),
            "adaptive_candidate_thresh_type": str(requested_policy.thresh_type),
            "adaptive_guard_active": adaptive_guard_reason_for_call is None,
            "adaptive_guard_reason": adaptive_guard_reason_for_call or "active",
            "adaptive_step_min": requested_policy.adaptive_step_min,
            "adaptive_step_max": requested_policy.adaptive_step_max,
            "adaptive_layer_min": requested_policy.adaptive_layer_min,
            "adaptive_layer_max": requested_policy.adaptive_layer_max,
        }
    if telemetry.materialize_copy_count:
        density = {**density, "diagnostic_materialized_qkv": True}
    exact_prefix_query = bool(policy.prefix_query_dense and policy.exact_prefix_query and metadata.prefix_len > 0)
    skip_full_prefix_blocks = bool(
        policy.prefix_query_dense
        and policy.skip_full_prefix_blocks
        and not exact_prefix_query
        and metadata.prefix_len >= SOL_ATTN_BLOCK_SIZE
    )
    if exact_prefix_query:
        density = {**density, "exact_prefix_query": True}
    if skip_full_prefix_blocks:
        density = {
            **density,
            "skip_full_prefix_blocks": True,
            "skipped_full_prefix_query_blocks_estimate": int(metadata.prefix_len) // SOL_ATTN_BLOCK_SIZE,
        }
    static_prefix_sink = bool(
        policy.prefix_query_dense
        and policy.static_prefix_sink
        and not exact_prefix_query
        and sink_range[0] == 0
        and sink_range[1] > 0
    )
    if static_prefix_sink:
        density = {
            **density,
            "static_prefix_sink": True,
            "static_prefix_sink_blocks_estimate": _ceil_div(sink_range[1] - sink_range[0], SOL_ATTN_BLOCK_SIZE),
        }
    bitmask_exact_scheduler = bool(policy.bitmask_exact_scheduler and not exact_prefix_query)
    if bitmask_exact_scheduler:
        density = {**density, "bitmask_exact_scheduler": True}
    shadow_pair_value_halves = bool(
        policy.shadow_pair_value_halves
        and not exact_prefix_query
        and not static_prefix_sink
        and not bitmask_exact_scheduler
        and policy.forward_config is None
    )
    pair_value_halves = bool(policy.pair_value_halves and not exact_prefix_query and not shadow_pair_value_halves)
    if pair_value_halves:
        density = {**density, "pair_value_halves": True}
    if shadow_pair_value_halves:
        density = {
            **density,
            "shadow_pair_value_halves": True,
            "shadow_pair_value_halves_returns": "retained_current",
            "shadow_pair_value_halves_max_mismatches": int(policy.shadow_max_mismatches),
            "shadow_pair_value_halves_row_state_probe": bool(policy.shadow_row_state_probe),
        }
    if policy.forward_config:
        density = {**density, "forward_config": str(policy.forward_config)}
    stride_aware_value = not candidate_value.is_contiguous()
    if stride_aware_value:
        telemetry.record_stride_aware_value(candidate_value)
        density = {**density, "stride_aware_value": True}
    telemetry.record_sparse_candidate(sink_range, density)
    valid = int(metadata.valid_length)
    prefix = int(metadata.prefix_len)

    try:
        kernel = _load_sol_attn_sm86()
        prefix_dense = bool(policy.prefix_query_dense and prefix > 0 and not exact_prefix_query)

        def run_kernel(*, use_pair_value_halves: bool) -> torch.Tensor:
            # Keep the full padded tensors in the kernel call and pass the active
            # valid length explicitly.  The SM86 pointer path writes the final
            # padded output directly and zeros only the padding tail, avoiding a
            # second full-output allocation/copy after a valid-token slice result.
            out = kernel(
                candidate_query,
                candidate_key,
                candidate_value,
                scale=softmax_scale,
                tau=float(policy.tau),
                thresh_type=str(policy.thresh_type),
                sink_start=sink_range[0],
                sink_tokens=sink_range[1] - sink_range[0],
                allow_strided_value=stride_aware_value,
                tokens=valid,
                prefix_exact_tokens=prefix if (exact_prefix_query or skip_full_prefix_blocks) else 0,
                exact_prefix_query=exact_prefix_query,
                skip_full_prefix_blocks=skip_full_prefix_blocks,
                static_prefix_sink=static_prefix_sink,
                forward_config=policy.forward_config,
                bitmask_exact_scheduler=bitmask_exact_scheduler,
                pair_value_halves=use_pair_value_halves,
            )
            if prefix_dense:
                out[:, :prefix] = dense_attention_reference(
                    candidate_query[:, :prefix],
                    candidate_key[:, :valid],
                    candidate_value[:, :valid],
                    softmax_scale=softmax_scale,
                )
            return out

        output = run_kernel(use_pair_value_halves=pair_value_halves)
        if shadow_pair_value_halves and len(telemetry.shadow_pair_value_halves_records) < max(
            0, int(policy.shadow_max_mismatches)
        ):
            shadow_output = run_kernel(use_pair_value_halves=True)
            telemetry.record_shadow_pair_value_halves(
                query=candidate_query,
                key=candidate_key,
                value=candidate_value,
                retained_output=output,
                candidate_output=shadow_output,
                metadata=metadata,
                step_index=step_index,
                layer_index=layer_index,
                policy=policy,
                sink_range=sink_range,
                sparse_call_index=telemetry.sparse_candidate_calls - 1,
                softmax_scale=softmax_scale,
            )
        telemetry.record_diagnostic_output(
            query=candidate_query,
            key=candidate_key,
            value=candidate_value,
            output=output,
            metadata=metadata,
            step_index=step_index,
            layer_index=layer_index,
            policy=policy,
            sink_range=sink_range,
            stage="post_prefix_dense_output",
        )
        telemetry.record_sparse_success(prefix_query_dense=prefix_dense, exact_prefix_query=exact_prefix_query)
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
