#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Finalize the r9 pair-value-halves diagnostic real-chain gate.

The finalizer compares two retained-current runs first, then reports the
current-vs-pair-value-halves diagnostic. It records only decoded media metrics,
telemetry counters, bounded attention-output digests, and scalar/layout metadata;
it never exports raw tensors or private filesystem paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import av
    import numpy as np
    MEDIA_IMPORT_ERROR: str | None = None
except Exception as exc:  # fail closed in the decision below
    av = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    MEDIA_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

CURRENT_A = "current_retained_a"
CURRENT_B = "current_retained_b"
CANDIDATE = "pair_value_halves"
EXPECTED_SPARSE_CALLS = 192


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_http(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in {"http_code", "size_download"}:
            out[key] = int(float(value))
        elif key == "time_total_s":
            out[key] = float(value)
    return out


def _floatish(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        cleaned = "".join(ch for ch in str(value) if ch.isdigit() or ch in ".-")
        try:
            return float(cleaned) if cleaned else None
        except ValueError:
            return None


def peak_csv_metric(path: Path, needle: str) -> float | None:
    values: list[float] = []
    with path.open(newline="", errors="replace") as f:
        for row in csv.DictReader(f):
            for key, value in row.items():
                if key and needle in key.lower():
                    parsed = _floatish(value)
                    if parsed is not None:
                        values.append(parsed)
    return max(values) if values else None


def host_mem_available_kib(path: Path) -> int | None:
    try:
        record = read_json(path)
        meminfo = record.get("host_meminfo", {})
        raw = str(meminfo.get("MemAvailable", "")).split()[0]
        return int(raw) if raw else None
    except Exception:
        return None


def structural_av(record: dict[str, Any]) -> bool:
    return (
        record.get("video_present") is True
        and record.get("audio_present") is True
        and int(record.get("width", -1)) == 1344
        and int(record.get("height", -1)) == 768
        and int(record.get("decoded_video_frames", -1)) == 124
        and int(record.get("audio_sample_rate", -1)) == 32000
        and int(record.get("audio_channels", -1)) == 2
        and int(record.get("decoded_audio_samples", 0)) > 0
    )


def zero_copy_contract(tel: dict[str, Any]) -> bool:
    return (
        int(tel.get("materialize_copy_count", -1)) == 0
        and int(tel.get("materialize_copy_bytes", -1)) == 0
        and int(tel.get("input_copy_events", -1)) == 0
        and int(tel.get("input_copy_bytes", -1)) == 0
        and tel.get("materialize_copy_by_tensor") == {}
        and tel.get("input_copy_by_tensor") == {}
    )


def density_has(tel: dict[str, Any], key: str) -> bool:
    samples = tel.get("density_samples", [])
    return any(bool(sample.get(key)) for sample in samples if isinstance(sample, dict))


def file_sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _hash_arrays(arrays: list[object]) -> str:
    h = hashlib.sha256()
    h.update(str(len(arrays)).encode("ascii"))
    for index, array in enumerate(arrays):
        h.update(str(index).encode("ascii"))
        h.update(str(getattr(array, "shape", None)).encode("ascii"))
        h.update(str(getattr(array, "dtype", None)).encode("ascii"))
        h.update(array.tobytes())
    return h.hexdigest()


def _decode_video(path: Path) -> tuple[list[dict[str, Any]], list[object]]:
    assert av is not None
    streams_info: list[dict[str, Any]] = []
    frames: list[object] = []
    with av.open(str(path)) as container:
        for stream in container.streams:
            if stream.type == "video":
                streams_info.append(
                    {
                        "index": int(stream.index),
                        "codec": stream.codec_context.name,
                        "width": int(stream.codec_context.width),
                        "height": int(stream.codec_context.height),
                        "average_rate": str(stream.average_rate),
                    }
                )
        if not streams_info:
            return streams_info, frames
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24").copy())
    return streams_info, frames


def _decode_audio(path: Path) -> tuple[list[dict[str, Any]], list[object]]:
    assert av is not None
    streams_info: list[dict[str, Any]] = []
    frames: list[object] = []
    with av.open(str(path)) as container:
        for stream in container.streams:
            if stream.type == "audio":
                streams_info.append(
                    {
                        "index": int(stream.index),
                        "codec": stream.codec_context.name,
                        "sample_rate": int(stream.codec_context.sample_rate),
                        "channels": int(stream.codec_context.channels),
                        "format": str(stream.codec_context.format),
                        "layout": str(stream.codec_context.layout),
                    }
                )
        if not streams_info:
            return streams_info, frames
        for frame in container.decode(audio=0):
            frames.append(frame.to_ndarray().copy())
    return streams_info, frames


def decoded_av_compare(reference_path: Path, candidate_path: Path, root: Path, comparison: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": "minimax_h3_a6000_decoded_av_comparison_v2",
        "comparison": comparison,
        "reference_path": str(reference_path.relative_to(root)),
        "candidate_path": str(candidate_path.relative_to(root)),
        "mp4_file_sha256_equal_recorded_not_gate": file_sha256(reference_path) == file_sha256(candidate_path),
        "reference_mp4_sha256": file_sha256(reference_path),
        "candidate_mp4_sha256": file_sha256(candidate_path),
        "reference_bytes": reference_path.stat().st_size if reference_path.exists() else None,
        "candidate_bytes": candidate_path.stat().st_size if candidate_path.exists() else None,
        "mp4_sha256_is_opaque_identifier": True,
        "hash_equality_used_for_decision": False,
        "decoded_content_used_for_decision": True,
    }
    if MEDIA_IMPORT_ERROR is not None:
        record.update(
            {
                "status": "blocked_comparator_unavailable",
                "media_integrity_ok": False,
                "decoded_content_equal": False,
                "error": MEDIA_IMPORT_ERROR,
            }
        )
        return record
    try:
        ref_video_info, ref_video = _decode_video(reference_path)
        cand_video_info, cand_video = _decode_video(candidate_path)
        ref_audio_info, ref_audio = _decode_audio(reference_path)
        cand_audio_info, cand_audio = _decode_audio(candidate_path)
    except Exception as exc:
        record.update(
            {
                "status": "blocked_decode_error",
                "media_integrity_ok": False,
                "decoded_content_equal": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return record

    def video_metrics() -> dict[str, Any]:
        compared = min(len(ref_video), len(cand_video))
        different = 0
        first_differences: list[dict[str, Any]] = []
        max_abs = 0
        sum_sq = 0.0
        sum_abs = 0.0
        count = 0
        for index, (ref_frame, cand_frame) in enumerate(zip(ref_video, cand_video)):
            if ref_frame.shape != cand_frame.shape:
                different += 1
                if len(first_differences) < 8:
                    first_differences.append(
                        {
                            "frame_index": index,
                            "reason": "shape_mismatch",
                            "reference_shape": list(ref_frame.shape),
                            "candidate_shape": list(cand_frame.shape),
                        }
                    )
                continue
            diff = ref_frame.astype(np.int16) - cand_frame.astype(np.int16)
            frame_max = int(np.max(np.abs(diff))) if diff.size else 0
            frame_mse = float(np.mean(diff.astype(np.float32) ** 2)) if diff.size else 0.0
            frame_mae = float(np.mean(np.abs(diff))) if diff.size else 0.0
            if frame_max != 0:
                different += 1
                if len(first_differences) < 8:
                    first_differences.append(
                        {"frame_index": index, "max_abs": frame_max, "mse": frame_mse, "mae": frame_mae}
                    )
            max_abs = max(max_abs, frame_max)
            diff64 = diff.astype(np.float64)
            sum_sq += float(np.sum(diff64 * diff64))
            sum_abs += float(np.sum(np.abs(diff64)))
            count += int(diff.size)
        mean_mse = (sum_sq / count) if count else None
        mean_mae = (sum_abs / count) if count else None
        psnr = (
            99.0
            if mean_mse is not None and mean_mse < 1.0e-12
            else (10.0 * math.log10((255.0 * 255.0) / mean_mse) if mean_mse else None)
        )
        hashes_equal = _hash_arrays(ref_video) == _hash_arrays(cand_video)
        exact_equal = len(ref_video) == len(cand_video) and different == 0 and hashes_equal
        return {
            "reference_streams": ref_video_info,
            "candidate_streams": cand_video_info,
            "reference_decoded_frames": len(ref_video),
            "candidate_decoded_frames": len(cand_video),
            "frames_compared": compared,
            "decoded_rgb24_sha256_equal": hashes_equal,
            "exact_equal": exact_equal,
            "different_frames": different + abs(len(ref_video) - len(cand_video)),
            "first_differences": first_differences,
            "max_abs": max_abs,
            "mean_mse": mean_mse,
            "mean_mae": mean_mae,
            "psnr_db": psnr,
        }

    def audio_metrics() -> dict[str, Any]:
        compared = min(len(ref_audio), len(cand_audio))
        different = 0
        first_differences: list[dict[str, Any]] = []
        max_abs = 0.0
        sum_sq = 0.0
        sum_abs = 0.0
        count = 0
        dot = 0.0
        ref_norm_sq = 0.0
        cand_norm_sq = 0.0
        for index, (ref_frame, cand_frame) in enumerate(zip(ref_audio, cand_audio)):
            shape_mismatch = ref_frame.shape != cand_frame.shape or str(ref_frame.dtype) != str(cand_frame.dtype)
            if shape_mismatch:
                different += 1
                if len(first_differences) < 8:
                    first_differences.append(
                        {
                            "frame_index": index,
                            "reason": "shape_or_dtype_mismatch",
                            "reference_shape": list(ref_frame.shape),
                            "candidate_shape": list(cand_frame.shape),
                            "reference_dtype": str(ref_frame.dtype),
                            "candidate_dtype": str(cand_frame.dtype),
                        }
                    )
                continue
            diff = ref_frame.astype(np.float64) - cand_frame.astype(np.float64)
            frame_max = float(np.max(np.abs(diff))) if diff.size else 0.0
            frame_mse = float(np.mean(diff * diff)) if diff.size else 0.0
            frame_mae = float(np.mean(np.abs(diff))) if diff.size else 0.0
            if frame_max != 0.0:
                different += 1
                if len(first_differences) < 8:
                    first_differences.append(
                        {"frame_index": index, "max_abs": frame_max, "mse": frame_mse, "mae": frame_mae}
                    )
            max_abs = max(max_abs, frame_max)
            ref_flat = ref_frame.astype(np.float64).reshape(-1)
            cand_flat = cand_frame.astype(np.float64).reshape(-1)
            dot += float(np.dot(ref_flat, cand_flat))
            ref_norm_sq += float(np.dot(ref_flat, ref_flat))
            cand_norm_sq += float(np.dot(cand_flat, cand_flat))
            sum_sq += float(np.sum(diff * diff))
            sum_abs += float(np.sum(np.abs(diff)))
            count += int(diff.size)
        mean_mse = (sum_sq / count) if count else None
        mean_mae = (sum_abs / count) if count else None
        cosine = (dot / math.sqrt(ref_norm_sq * cand_norm_sq)) if ref_norm_sq > 0.0 and cand_norm_sq > 0.0 else None
        hashes_equal = _hash_arrays(ref_audio) == _hash_arrays(cand_audio)
        exact_equal = len(ref_audio) == len(cand_audio) and different == 0 and hashes_equal
        return {
            "reference_streams": ref_audio_info,
            "candidate_streams": cand_audio_info,
            "reference_decoded_frames": len(ref_audio),
            "candidate_decoded_frames": len(cand_audio),
            "frames_compared": compared,
            "decoded_samples_sha256_equal": hashes_equal,
            "exact_equal": exact_equal,
            "different_frames": different + abs(len(ref_audio) - len(cand_audio)),
            "first_differences": first_differences,
            "max_abs": max_abs,
            "mean_mse": mean_mse,
            "mean_mae": mean_mae,
            "waveform_cosine": cosine,
        }

    video = video_metrics()
    audio = audio_metrics()
    media_integrity_ok = bool(
        ref_video_info
        and cand_video_info
        and ref_audio_info
        and cand_audio_info
        and video["reference_decoded_frames"] > 0
        and video["candidate_decoded_frames"] > 0
        and audio["reference_decoded_frames"] > 0
        and audio["candidate_decoded_frames"] > 0
    )
    decoded_content_equal = bool(media_integrity_ok and video["exact_equal"] and audio["exact_equal"])
    record.update(
        {
            "status": "pass",
            "media_integrity_ok": media_integrity_ok,
            "decoded_content_equal": decoded_content_equal,
            "video": video,
            "audio": audio,
        }
    )
    return record


def _rel(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def _digest(record: dict[str, Any]) -> str | None:
    digest = record.get("output_digest")
    return digest.get("sha256") if isinstance(digest, dict) else None


def _layout(record: dict[str, Any], key: str) -> Any:
    return record.get(key)


def _metadata(record: dict[str, Any]) -> Any:
    return record.get("metadata")


def _fixed_policy(policy: Any) -> Any:
    if not isinstance(policy, dict):
        return policy
    return {key: value for key, value in policy.items() if key != "pair_value_halves"}


def attention_output_compare(
    reference_tel: dict[str, Any],
    candidate_tel: dict[str, Any],
    *,
    comparison: str,
    compare_policy_pair_flag: bool,
) -> dict[str, Any]:
    ref_records = [r for r in reference_tel.get("diagnostic_output_records", []) if isinstance(r, dict)]
    cand_records = [r for r in candidate_tel.get("diagnostic_output_records", []) if isinstance(r, dict)]
    ref_failures = int(reference_tel.get("diagnostic_output_digest_failures", 0) or 0)
    cand_failures = int(candidate_tel.get("diagnostic_output_digest_failures", 0) or 0)
    first_mismatches: list[dict[str, Any]] = []
    digest_mismatches = 0
    metadata_mismatches = 0
    input_layout_mismatches = 0
    output_layout_mismatches = 0
    policy_mismatches = 0
    raw_tensor_exported = False
    for index, (ref, cand) in enumerate(zip(ref_records, cand_records)):
        raw_tensor_exported = raw_tensor_exported or bool(ref.get("raw_tensor_exported")) or bool(
            cand.get("raw_tensor_exported")
        )
        mismatch: dict[str, Any] = {"call_index": index}
        if _digest(ref) != _digest(cand):
            digest_mismatches += 1
            mismatch["reference_output_sha256"] = _digest(ref)
            mismatch["candidate_output_sha256"] = _digest(cand)
        if _metadata(ref) != _metadata(cand):
            metadata_mismatches += 1
            mismatch["reference_metadata"] = _metadata(ref)
            mismatch["candidate_metadata"] = _metadata(cand)
        if _layout(ref, "input_layouts") != _layout(cand, "input_layouts"):
            input_layout_mismatches += 1
            mismatch["input_layout_mismatch"] = True
        if _layout(ref, "output_layout") != _layout(cand, "output_layout"):
            output_layout_mismatches += 1
            mismatch["output_layout_mismatch"] = True
        ref_policy = ref.get("policy")
        cand_policy = cand.get("policy")
        policy_equal = ref_policy == cand_policy if compare_policy_pair_flag else _fixed_policy(ref_policy) == _fixed_policy(cand_policy)
        if not policy_equal:
            policy_mismatches += 1
            mismatch["reference_policy"] = ref_policy
            mismatch["candidate_policy"] = cand_policy
        if len(mismatch) > 1 and len(first_mismatches) < 8:
            first_mismatches.append(mismatch)
    count_equal = len(ref_records) == len(cand_records) == EXPECTED_SPARSE_CALLS
    failures_zero = ref_failures == 0 and cand_failures == 0
    raw_tensor_clean = not raw_tensor_exported and all(
        rec.get("raw_tensor_exported") is False for rec in ref_records[:EXPECTED_SPARSE_CALLS] + cand_records[:EXPECTED_SPARSE_CALLS]
    )
    status = "pass" if count_equal and failures_zero and raw_tensor_clean else "blocked"
    output_digest_exact_equal = status == "pass" and digest_mismatches == 0
    metadata_equal = status == "pass" and metadata_mismatches == 0
    input_layouts_equal = status == "pass" and input_layout_mismatches == 0
    output_layouts_equal = status == "pass" and output_layout_mismatches == 0
    policy_equal = status == "pass" and policy_mismatches == 0
    return {
        "schema_version": "minimax_h3_a6000_attention_output_digest_comparison_v1",
        "comparison": comparison,
        "status": status,
        "expected_sparse_records_per_mode": EXPECTED_SPARSE_CALLS,
        "reference_records": len(ref_records),
        "candidate_records": len(cand_records),
        "reference_digest_failures": ref_failures,
        "candidate_digest_failures": cand_failures,
        "raw_tensor_exported": raw_tensor_exported,
        "raw_tensor_values_available": False,
        "tolerances": {
            "output_sha256_mismatches_allowed": 0,
            "metadata_mismatches_allowed": 0,
            "input_layout_mismatches_allowed": 0,
            "output_layout_mismatches_allowed": 0,
            "policy_mismatches_allowed": 0,
        },
        "output_digest_exact_equal": output_digest_exact_equal,
        "metadata_equal": metadata_equal,
        "input_layouts_equal": input_layouts_equal,
        "output_layouts_equal": output_layouts_equal,
        "fixed_policy_equal": policy_equal,
        "digest_mismatch_count": digest_mismatches + abs(len(ref_records) - len(cand_records)),
        "metadata_mismatch_count": metadata_mismatches,
        "input_layout_mismatch_count": input_layout_mismatches,
        "output_layout_mismatch_count": output_layout_mismatches,
        "policy_mismatch_count": policy_mismatches,
        "first_mismatches": first_mismatches,
        "all_equal_for_gate": bool(
            status == "pass"
            and output_digest_exact_equal
            and metadata_equal
            and input_layouts_equal
            and output_layouts_equal
            and policy_equal
        ),
    }


def summarize_av_comparison(cmp: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": cmp.get("status"),
        "media_integrity_ok": cmp.get("media_integrity_ok"),
        "decoded_content_equal": cmp.get("decoded_content_equal"),
        "video_exact_equal": bool((cmp.get("video") or {}).get("exact_equal")),
        "video_different_frames": (cmp.get("video") or {}).get("different_frames"),
        "video_max_abs": (cmp.get("video") or {}).get("max_abs"),
        "video_mean_mse": (cmp.get("video") or {}).get("mean_mse"),
        "audio_exact_equal": bool((cmp.get("audio") or {}).get("exact_equal")),
        "audio_different_frames": (cmp.get("audio") or {}).get("different_frames"),
        "audio_max_abs": (cmp.get("audio") or {}).get("max_abs"),
        "audio_mean_mse": (cmp.get("audio") or {}).get("mean_mse"),
    }


def shadow_pair_value_summary(tel: dict[str, Any]) -> dict[str, Any]:
    records = [record for record in tel.get("shadow_pair_value_halves_records", []) if isinstance(record, dict)]
    mismatch_count = int(tel.get("shadow_pair_value_halves_mismatch_count", 0) or 0)
    calls = int(tel.get("shadow_pair_value_halves_calls", 0) or 0)
    equal_calls = int(tel.get("shadow_pair_value_halves_equal_calls", 0) or 0)
    record_failures = int(tel.get("shadow_pair_value_halves_record_failures", 0) or 0)
    raw_tensor_exported = any(bool(record.get("raw_tensor_exported")) for record in records)
    candidate_marker_ok = all(record.get("candidate_marker") == "pair_value_halves_shadow_candidate" for record in records)
    returns_current_ok = all(record.get("returned_output") == "retained_current" for record in records)
    finite_masks_equal = all(bool((record.get("finite_equality") or {}).get("finite_mask_equal")) for record in records)
    first_mismatches = records[:8]
    return {
        "schema_version": "minimax_h3_a6000_same_input_shadow_pair_value_halves_summary_v1",
        "enabled": calls > 0,
        "calls": calls,
        "equal_calls": equal_calls,
        "mismatch_count": mismatch_count,
        "record_failures": record_failures,
        "records_written": len(records),
        "records_bounded": len(records) <= 8,
        "raw_tensor_exported": raw_tensor_exported,
        "raw_tensor_values_available": False,
        "candidate_marker_ok": candidate_marker_ok,
        "returns_retained_current_ok": returns_current_ok,
        "finite_masks_equal_in_records": finite_masks_equal,
        "all_shadowed_calls_equal": calls > 0 and mismatch_count == 0 and record_failures == 0,
        "first_mismatches": first_mismatches,
    }


def row_state_probe_summary(shadow_summary: dict[str, Any]) -> dict[str, Any]:
    records = [record for record in shadow_summary.get("first_mismatches", []) if isinstance(record, dict)]
    # The telemetry itself is bounded to at most eight mismatch records; all are
    # embedded in first_mismatches by shadow_pair_value_summary.
    probes = [record.get("row_state_probe") for record in records if isinstance(record.get("row_state_probe"), dict)]
    pass_probes = [probe for probe in probes if probe.get("status") == "pass"]
    raw_tensor_exported = any(bool(probe.get("raw_tensor_exported")) for probe in probes)
    route_equal = all(bool((probe.get("route") or {}).get("route_digest_equal")) for probe in pass_probes)
    order_equal = all(bool((probe.get("route") or {}).get("exact_order_digest_equal")) for probe in pass_probes)
    row_deltas = [float((probe.get("row_state") or {}).get("row_sum_abs_delta", 0.0) or 0.0) for probe in pass_probes]
    max_row_sum_delta = max(row_deltas) if row_deltas else None
    row_max_deltas = [float((probe.get("row_state") or {}).get("row_max_abs_delta", 0.0) or 0.0) for probe in pass_probes]
    max_row_max_delta = max(row_max_deltas) if row_max_deltas else None
    current_reference_equal = all(
        bool((probe.get("target_row_current_reference") or {}).get("retained_current_row_equal_reference_bf16"))
        for probe in pass_probes
    )
    pair_reference_equal = all(
        bool((probe.get("target_row_current_reference") or {}).get("pair_value_halves_row_equal_reference_bf16"))
        for probe in pass_probes
    )
    first_probe = pass_probes[0] if pass_probes else (probes[0] if probes else None)
    return {
        "schema_version": "minimax_h3_a6000_pair_value_halves_row_state_probe_summary_v1",
        "enabled": bool(probes),
        "records_with_probe": len(probes),
        "pass_records": len(pass_probes),
        "probe_status_counts": {
            str(status): sum(1 for probe in probes if probe.get("status") == status)
            for status in sorted({probe.get("status") for probe in probes})
        },
        "raw_tensor_exported": raw_tensor_exported,
        "route_digest_equal_all_pass_records": bool(pass_probes) and route_equal,
        "exact_order_digest_equal_all_pass_records": bool(pass_probes) and order_equal,
        "max_row_sum_abs_delta": max_row_sum_delta,
        "max_row_max_abs_delta": max_row_max_delta,
        "retained_current_reference_equal_all_pass_records": bool(pass_probes) and current_reference_equal,
        "pair_value_halves_reference_equal_all_pass_records": bool(pass_probes) and pair_reference_equal,
        "first_probe": first_probe,
    }


def source_runtime_grounding() -> dict[str, Any]:
    rel_paths = [
        "ports/minimax_h3_a6000/src/minimax_h3_a6000/env.py",
        "ports/minimax_h3_a6000/src/minimax_h3_a6000/sol_attn_backend.py",
        "ports/minimax_h3_a6000/src/minimax_h3_a6000/sol_attn_triton_sm86.py",
        "ports/minimax_h3_a6000/integration/run_sol_attn_h3_pair_value_halves_n1.sh",
        "ports/minimax_h3_a6000/integration/finalize_sol_attn_pair_value_halves_diagnostic.py",
        "ports/minimax_h3_a6000/patches/vllm_omni_h3_a6000_opt_in.patch",
        "upstreams/Sana-sol-engine/models/minimax_h3/gb200/optimized/sol_attn_h3.py",
        "upstreams/Sana-sol-engine/techniques/sparse_backends/sol_attn/interface.py",
        "upstreams/Sana-sol-engine/techniques/sparse_backends/sol_attn/triton_ref/fwd.py",
    ]
    hashes = {}
    for rel in rel_paths:
        path = Path(rel)
        if path.exists() and path.is_file():
            hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            hashes[rel] = "missing"
    try:
        upstream_head = subprocess.run(
            ["git", "-C", "upstreams/Sana-sol-engine", "rev-parse", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        upstream_head = f"unavailable:{type(exc).__name__}"
    runtime_versions: dict[str, Any] = {"python": sys.version.split()[0]}
    try:
        import torch  # type: ignore

        runtime_versions["torch"] = getattr(torch, "__version__", "unknown")
        runtime_versions["torch_cuda"] = getattr(getattr(torch, "version", None), "cuda", None)
    except Exception as exc:  # noqa: BLE001
        runtime_versions["torch"] = f"unavailable:{type(exc).__name__}"
    try:
        import triton  # type: ignore

        runtime_versions["triton"] = getattr(triton, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        runtime_versions["triton"] = f"unavailable:{type(exc).__name__}"
    return {
        "schema_version": "minimax_h3_a6000_source_runtime_grounding_v1",
        "pinned_upstream_revision": upstream_head,
        "source_hashes": hashes,
        "runtime_versions": runtime_versions,
    }


def build_decision(root: Path, timeout_values: dict[str, int]) -> dict[str, Any]:
    dirs = {mode: root / mode for mode in (CURRENT_A, CURRENT_B, CANDIDATE)}
    required = [
        root / "gpu_hygiene_preflight.json",
        root / "nvidia_smi_full.txt",
        root / "nvidia_smi_compute_apps.csv",
        root / "gpu_lease_status.txt",
        root / "disk_preflight.txt",
        root / "resource_monitor.csv",
        root / "overall_wall_time.json",
        root / "r9_image_identity.env",
        root / "workload.env",
        root / "docker_info_summary.txt",
        root / "docker_ps_before.jsonl",
        root / "docker_ps_after.jsonl",
        root / "r9_image_inspect.json",
    ]
    for mode_dir in dirs.values():
        required.extend(
            [
                mode_dir / "startup_timeout_config.env",
                mode_dir / "warmup_http_metrics.txt",
                mode_dir / "http_metrics.txt",
                mode_dir / "av_validation.json",
                mode_dir / "sol_attn_telemetry.sol_attn.json",
                mode_dir / "host_resource_before.json",
                mode_dir / "host_resource_after.json",
                mode_dir / "gpu_resource_samples.csv",
                mode_dir / "wall_time.json",
            ]
        )
    missing = [_rel(root, path) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        return {
            "schema_version": "minimax_h3_a6000_sol_attn_pair_value_halves_diagnostic_n1_v1",
            "classification": "blocked",
            "reason": "incomplete_artifacts",
            "missing_paths": missing,
            "promote_to_matched_n3": False,
            "promote_to_n3": False,
            "not_speedup_claim": True,
            "no_product_speedup_claim": True,
            "reviewer_acceptance_required_before_promotion": True,
            "reviewer_acceptance_status": "pending_external_reviewer_not_authored_by_runner",
        }

    av = {mode: read_json(dirs[mode] / "av_validation.json") for mode in dirs}
    tel = {mode: read_json(dirs[mode] / "sol_attn_telemetry.sol_attn.json") for mode in dirs}
    http = {mode: parse_http(dirs[mode] / "http_metrics.txt") for mode in dirs}
    warm = {mode: parse_http(dirs[mode] / "warmup_http_metrics.txt") for mode in dirs}
    memories = {mode: peak_csv_metric(dirs[mode] / "gpu_resource_samples.csv", "memory.used") for mode in dirs}
    powers = {mode: peak_csv_metric(dirs[mode] / "gpu_resource_samples.csv", "power.draw") for mode in dirs}
    temps = {mode: peak_csv_metric(dirs[mode] / "gpu_resource_samples.csv", "temperature.gpu") for mode in dirs}
    host_before = {mode: host_mem_available_kib(dirs[mode] / "host_resource_before.json") for mode in dirs}
    host_after = {mode: host_mem_available_kib(dirs[mode] / "host_resource_after.json") for mode in dirs}
    shadow_summary = shadow_pair_value_summary(tel[CANDIDATE])
    shadow_mode = bool(shadow_summary["enabled"])
    row_state_summary = row_state_probe_summary(shadow_summary)
    (root / "same_input_shadow_comparison_pair_value_halves.json").write_text(
        json.dumps(shadow_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "row_state_shadow_probe_summary.json").write_text(
        json.dumps(row_state_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    current_cmp = decoded_av_compare(
        dirs[CURRENT_A] / "output.mp4",
        dirs[CURRENT_B] / "output.mp4",
        root,
        "current_retained_a_vs_current_retained_b_measured_outputs",
    )
    pair_cmp = decoded_av_compare(
        dirs[CURRENT_A] / "output.mp4",
        dirs[CANDIDATE] / "output.mp4",
        root,
        "current_retained_a_vs_pair_value_halves_measured_outputs",
    )
    (root / "decoded_av_comparison_current_vs_current.json").write_text(
        json.dumps(current_cmp, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "decoded_av_comparison_current_vs_pair_value_halves.json").write_text(
        json.dumps(pair_cmp, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # Preserve the historical file name for tooling that expects the pair comparison there.
    (root / "decoded_av_comparison.json").write_text(
        json.dumps(pair_cmp, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    current_attention = attention_output_compare(
        tel[CURRENT_A],
        tel[CURRENT_B],
        comparison="current_retained_a_vs_current_retained_b_attention_outputs",
        compare_policy_pair_flag=True,
    )
    pair_attention = attention_output_compare(
        tel[CURRENT_A],
        tel[CANDIDATE],
        comparison="current_retained_a_vs_pair_value_halves_attention_outputs",
        compare_policy_pair_flag=False,
    )
    (root / "attention_output_comparison_current_vs_current.json").write_text(
        json.dumps(current_attention, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "attention_output_comparison_current_vs_pair_value_halves.json").write_text(
        json.dumps(pair_attention, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    observed_r8_cv_pct = 0.5072177175606011
    promotion_threshold_pct = max(1.5, 2.0 * observed_r8_cv_pct)
    improvement_pct = (http[CURRENT_A]["time_total_s"] - http[CANDIDATE]["time_total_s"]) / http[CURRENT_A][
        "time_total_s"
    ] * 100.0

    def real_h3_v_layout_seen(mode: str) -> bool:
        layout_samples = tel[mode].get("layout_samples", [])
        return any(
            any(
                t.get("name") == "value"
                and t.get("shape") == [1, 38272, 56, 128]
                and t.get("stride") == [823001088, 21504, 128, 1]
                and t.get("storage_offset") == 14336
                and t.get("is_contiguous") is False
                for t in sample.get("tensors", [])
            )
            for sample in layout_samples
            if isinstance(sample, dict)
        )

    av_fields = (
        "width",
        "height",
        "average_rate",
        "decoded_video_frames",
        "audio_sample_rate",
        "audio_channels",
        "decoded_audio_frames",
        "decoded_audio_samples",
    )
    av_metadata_equal_current = all(av[CURRENT_A].get(key) == av[CURRENT_B].get(key) for key in av_fields)
    av_metadata_equal_pair = all(av[CURRENT_A].get(key) == av[CANDIDATE].get(key) for key in av_fields)

    current_media_integrity_ok = current_cmp.get("status") == "pass" and bool(current_cmp.get("media_integrity_ok"))
    pair_media_integrity_ok = pair_cmp.get("status") == "pass" and bool(pair_cmp.get("media_integrity_ok"))
    current_video_equal = bool((current_cmp.get("video") or {}).get("exact_equal"))
    current_audio_equal = bool((current_cmp.get("audio") or {}).get("exact_equal"))
    current_av_equal = bool(current_cmp.get("decoded_content_equal"))
    pair_video_equal = bool((pair_cmp.get("video") or {}).get("exact_equal"))
    pair_audio_equal = bool((pair_cmp.get("audio") or {}).get("exact_equal"))
    pair_av_equal = bool(pair_cmp.get("decoded_content_equal"))

    gates: dict[str, bool] = {
        "all_http_200": all(http[mode].get("http_code") == 200 for mode in dirs),
        "all_warmups_http_200": all(warm[mode].get("http_code") == 200 for mode in dirs),
        "all_structural_av_valid": all(structural_av(av[mode]) for mode in dirs),
        "current_current_av_metadata_equal": av_metadata_equal_current,
        "current_current_decoded_media_integrity_ok": current_media_integrity_ok,
        "current_current_decoded_video_content_equal": current_video_equal,
        "current_current_decoded_audio_content_equal": current_audio_equal,
        "current_current_decoded_av_content_equal": current_av_equal,
        "current_current_attention_output_digests_equal": bool(current_attention.get("all_equal_for_gate")),
        "pair_av_metadata_equal": av_metadata_equal_pair,
        "pair_decoded_media_integrity_ok": pair_media_integrity_ok,
        "pair_decoded_video_content_equal": pair_video_equal,
        "pair_decoded_audio_content_equal": pair_audio_equal,
        "pair_decoded_av_content_equal": pair_av_equal,
        "pair_attention_output_digests_equal": bool(pair_attention.get("all_equal_for_gate")),
        "all_sparse_calls_192": all(int(tel[mode].get("sparse_calls", 0)) == EXPECTED_SPARSE_CALLS for mode in dirs),
        "all_sparse_candidates_192": all(
            int(tel[mode].get("sparse_candidate_calls", 0)) == EXPECTED_SPARSE_CALLS for mode in dirs
        ),
        "all_fallback_calls_zero": all(int(tel[mode].get("fallback_calls", -1)) == 0 for mode in dirs),
        "all_zero_materialization_and_input_copies": all(zero_copy_contract(tel[mode]) for mode in dirs),
        "all_stride_aware_value_calls_192": all(
            int(tel[mode].get("stride_aware_value_calls", 0)) == EXPECTED_SPARSE_CALLS for mode in dirs
        ),
        "all_skip_full_prefix_blocks_seen": all(density_has(tel[mode], "skip_full_prefix_blocks") for mode in dirs),
        "candidate_pair_value_halves_seen": density_has(tel[CANDIDATE], "pair_value_halves")
        or bool(shadow_summary["enabled"]),
        "current_pair_value_halves_absent": not density_has(tel[CURRENT_A], "pair_value_halves")
        and not density_has(tel[CURRENT_B], "pair_value_halves"),
        "same_input_shadow_enabled": bool(shadow_summary["enabled"]),
        "same_input_shadow_returns_current": bool(shadow_summary["returns_retained_current_ok"]),
        "same_input_shadow_raw_tensor_export_zero": not bool(shadow_summary["raw_tensor_exported"]),
        "same_input_shadow_record_failures_zero": int(shadow_summary["record_failures"]) == 0,
        "same_input_shadow_pair_value_halves_exact": bool(shadow_summary["all_shadowed_calls_equal"]),
        "same_input_shadow_row_state_probe_present": (not shadow_mode) or bool(row_state_summary["enabled"]),
        "same_input_shadow_row_state_probe_raw_tensor_export_zero": not bool(row_state_summary["raw_tensor_exported"]),
        "same_input_shadow_row_state_route_digest_equal": (not shadow_mode) or bool(row_state_summary["route_digest_equal_all_pass_records"]),
        "same_input_shadow_row_state_row_sum_delta_zero": (not shadow_mode) or row_state_summary.get("max_row_sum_abs_delta") == 0.0,
        "same_input_shadow_row_state_row_max_delta_zero": (not shadow_mode) or row_state_summary.get("max_row_max_abs_delta") == 0.0,
        "same_input_shadow_row_state_current_reference_measured": (not shadow_mode) or bool(row_state_summary["pass_records"]),
        "real_h3_fused_value_layout_seen_all_modes": all(real_h3_v_layout_seen(mode) for mode in dirs),
        "all_gpu_copy_time_zero": all(float(tel[mode].get("materialize_gpu_copy_latency_ms", -1.0)) == 0.0 for mode in dirs),
        "attention_gpu_timing_complete": all(
            int(tel[mode].get("sparse_attention_timed_calls", 0)) == EXPECTED_SPARSE_CALLS
            and float(tel[mode].get("sparse_attention_gpu_latency_ms", 0.0)) > 0.0
            for mode in dirs
        ),
        "denoise_gpu_timing_complete": all(
            int(tel[mode].get("denoise_timed_calls", 0)) > 0
            and float(tel[mode].get("denoise_gpu_latency_ms", 0.0)) > 0.0
            for mode in dirs
        ),
        "no_gpu_timing_failures": all(
            int(tel[mode].get(key, -1)) == 0
            for mode in dirs
            for key in (
                "materialize_gpu_timing_failures",
                "sparse_attention_gpu_timing_failures",
                "denoise_gpu_timing_failures",
            )
        ),
        "diagnostic_raw_tensor_export_zero": not bool(current_attention.get("raw_tensor_exported"))
        and not bool(pair_attention.get("raw_tensor_exported")),
        "resource_samples_present": all(
            value is not None
            for metric in (memories, powers, temps)
            for value in metric.values()
        ),
        "candidate_peak_memory_not_higher": memories[CANDIDATE] is not None
        and memories[CURRENT_A] is not None
        and memories[CANDIDATE] <= memories[CURRENT_A],
        "e2e_signal_exceeds_predeclared_threshold": improvement_pct > promotion_threshold_pct,
    }

    media_compare_blocked = current_cmp.get("status") != "pass" or pair_cmp.get("status") != "pass"
    if media_compare_blocked:
        classification = "blocked"
        reason = "decoded_av_comparator_failed"
    elif not current_media_integrity_ok:
        classification = "blocked"
        reason = "current_current_decoded_av_media_integrity_failed"
    elif current_attention.get("status") != "pass" or not gates["current_current_attention_output_digests_equal"]:
        classification = "blocked"
        reason = "current_current_attention_output_nondeterminism"
    elif not current_av_equal:
        classification = "blocked"
        reason = "current_current_decoded_av_unstable_separate_full_chain_from_kernel"
    elif not pair_media_integrity_ok:
        classification = "blocked"
        reason = "pair_decoded_av_media_integrity_failed"
    else:
        correctness_gate_names = [key for key in gates if key != "e2e_signal_exceeds_predeclared_threshold"]
        correctness_ok = all(gates[key] for key in correctness_gate_names)
        if shadow_mode and int(shadow_summary["mismatch_count"]) > 0:
            classification = "reject_no_promotion"
            reason = (
                "same_input_shadow_pair_value_halves_row_state_probe_no_exact_repair"
                if row_state_summary.get("enabled")
                else "same_input_shadow_pair_value_halves_divergence"
            )
        elif not correctness_ok:
            classification = "reject"
            if not gates["pair_attention_output_digests_equal"]:
                reason = "attention_output_divergence"
            elif shadow_mode and not gates["same_input_shadow_pair_value_halves_exact"]:
                reason = "same_input_shadow_pair_value_halves_not_exact"
            else:
                reason = "correctness_or_contract"
        elif shadow_mode:
            classification = "reject"
            reason = "shadow_only_diagnostic_no_promotion_without_fixed_pair_gate"
        elif not gates["e2e_signal_exceeds_predeclared_threshold"]:
            classification = "reject"
            reason = "no_above_noise_n1_signal"
        else:
            classification = "promote_to_matched_n3"
            reason = "n1_gate_passed_requires_independent_reviewer_before_n3"

    decisive_boundary = (
        "N=1 diagnostic pair-value-halves real-chain gate only; no product speedup, BF16-fidelity, "
        "long-video, quality-equivalence, public-comparison, or SOTA claim."
    )
    principal_variable = (
        "MINIMAX_H3_A6000_SOL_ATTN_SHADOW_PAIR_VALUE_HALVES=1 same-input diagnostic; "
        "retained-current output returned"
        if shadow_mode
        else "MINIMAX_H3_A6000_SOL_ATTN_PAIR_VALUE_HALVES=1 vs 0 after same-run current-vs-current diagnostic"
    )
    pair_decision_scope = (
        "not_decisive_due_current_current_instability"
        if classification == "blocked" and str(reason).startswith("current_current")
        else classification
    )
    return {
        "schema_version": "minimax_h3_a6000_sol_attn_pair_value_halves_diagnostic_n1_v1",
        "classification": classification,
        "reason": reason,
        "exact_fix_supported": False if classification == "reject_no_promotion" else None,
        "exact_fix_implemented": False,
        "primary_localized_mechanism": "pv_dot_or_bf16_probability_codegen_delta_after_route_row_state_probe" if classification == "reject_no_promotion" else None,
        "promote_to_matched_n3": classification == "promote_to_matched_n3",
        "promote_to_n3": classification == "promote_to_matched_n3",
        "reviewer_acceptance_required_before_promotion": True,
        "reviewer_acceptance_status": "pending_external_reviewer_not_authored_by_runner",
        "not_speedup_claim": True,
        "no_product_speedup_claim": True,
        "lane": "matched_n1_5step_sol_attn_opt_in_not_bf16_fidelity_diagnostic_digest",
        "workload": {"width": 1344, "height": 768, "frames": 124, "fps": 24, "duration_s": 5.166667, "steps": 5, "seed": 0},
        "principal_variable": principal_variable,
        "fixed_variables": {
            "cache": "off",
            "stride_aware_v": "on",
            "skip_full_prefix_blocks": "on",
            "dense_prefix_overwrite": "preserved",
            "diagnostic_materialization": "off_for_all_modes",
            "diagnostic_output_digest": "on_for_bounded_gate_only",
            "diagnostic_output_max_calls": 256,
            "same_input_shadow_pair_value_halves": "on_returns_retained_current" if shadow_mode else "off",
            "shadow_max_mismatches": 8 if shadow_mode else 0,
            "dense_first_steps": 0,
            "dense_first_layers": 2,
        },
        "timeout_values": timeout_values,
        "observed_r8_cv_pct": observed_r8_cv_pct,
        "promotion_threshold_pct": promotion_threshold_pct,
        "http_e2e_seconds": {mode: http[mode]["time_total_s"] for mode in dirs},
        "excluded_warmup_http_seconds": {mode: warm[mode]["time_total_s"] for mode in dirs},
        "n1_http_e2e_improvement_pct_current_a_vs_pair": improvement_pct,
        "gpu_component_ms": {
            mode: {
                key: tel[mode].get(key)
                for key in (
                    "materialize_gpu_copy_latency_ms",
                    "materialize_host_enqueue_latency_ms",
                    "sparse_attention_gpu_latency_ms",
                    "denoise_gpu_latency_ms",
                )
            }
            for mode in dirs
        },
        "resource_summary": {
            "peak_gpu_memory_mib": memories,
            "peak_gpu_power_w": powers,
            "peak_gpu_temperature_c": temps,
            "host_mem_available_kib_before": host_before,
            "host_mem_available_kib_after": host_after,
            "resource_sample_files": {mode: _rel(root, dirs[mode] / "gpu_resource_samples.csv") for mode in dirs}
            | {"overall": "resource_monitor.csv"},
        },
        "telemetry_counts": {
            mode: {
                key: tel[mode].get(key)
                for key in (
                    "sparse_candidate_calls",
                    "sparse_calls",
                    "fallback_calls",
                    "materialize_copy_count",
                    "materialize_copy_bytes",
                    "input_copy_events",
                    "input_copy_bytes",
                    "stride_aware_value_calls",
                    "diagnostic_output_digest_failures",
                    "shadow_pair_value_halves_calls",
                    "shadow_pair_value_halves_equal_calls",
                    "shadow_pair_value_halves_mismatch_count",
                    "shadow_pair_value_halves_record_failures",
                )
            }
            | {
                "diagnostic_output_records": len(tel[mode].get("diagnostic_output_records", [])),
                "shadow_pair_value_halves_records": len(tel[mode].get("shadow_pair_value_halves_records", [])),
            }
            for mode in dirs
        },
        "current_vs_current_decision": {
            "decoded_av_comparison_path": "decoded_av_comparison_current_vs_current.json",
            "attention_output_comparison_path": "attention_output_comparison_current_vs_current.json",
            "decoded_av_stable": current_av_equal,
            "decoded_video_stable": current_video_equal,
            "decoded_audio_stable": current_audio_equal,
            "attention_output_deterministic": bool(current_attention.get("all_equal_for_gate")),
            "decision": "stable" if current_av_equal and current_attention.get("all_equal_for_gate") else "fail_closed_full_chain_or_attention_nondeterminism",
            "summary": summarize_av_comparison(current_cmp),
        },
        "current_vs_pair_decision": {
            "decoded_av_comparison_path": "decoded_av_comparison_current_vs_pair_value_halves.json",
            "attention_output_comparison_path": "attention_output_comparison_current_vs_pair_value_halves.json",
            "decoded_av_equal": pair_av_equal,
            "decoded_video_equal": pair_video_equal,
            "decoded_audio_equal": pair_audio_equal,
            "attention_output_equal": bool(pair_attention.get("all_equal_for_gate")),
            "decision": pair_decision_scope,
            "summary": summarize_av_comparison(pair_cmp),
        },
        "output_checks": {
            "structural_av_all_modes": gates["all_structural_av_valid"],
            "current_current_av_metadata_equal": av_metadata_equal_current,
            "pair_av_metadata_equal": av_metadata_equal_pair,
            "mp4_sha256_recorded_not_gate_current_current": current_cmp.get("mp4_file_sha256_equal_recorded_not_gate"),
            "mp4_sha256_recorded_not_gate_pair": pair_cmp.get("mp4_file_sha256_equal_recorded_not_gate"),
            "hash_equality_used_for_decision": False,
            "decoded_av_content_used_for_decision": True,
            "attention_output_digest_used_for_kernel_localization": True,
            "same_input_shadow_diff_used_for_kernel_localization": shadow_mode,
            "same_input_shadow_records_path": "same_input_shadow_comparison_pair_value_halves.json" if shadow_mode else None,
            "row_state_shadow_probe_summary_path": "row_state_shadow_probe_summary.json" if shadow_mode else None,
            "row_state_raw_tensor_export_zero": not bool(row_state_summary.get("raw_tensor_exported")),
            "raw_tensor_exported": False,
        },
        "output_identifier_policy": {
            "mp4_sha256_is_opaque_identifier": True,
            "hash_equality_used_for_decision": False,
            "decoded_av_content_used_for_decision": True,
            "attention_output_sha256_is_diagnostic_not_raw_tensor": True,
            "justification": "MP4 container/file bytes can change independently of decoded media; this gate records file SHA only as an opaque identifier and gates on decoded RGB/audio content plus Sol-Attn per-call output digests/telemetry.",
        },
        "decoded_av_comparison_summary": {
            "current_vs_current": summarize_av_comparison(current_cmp),
            "current_vs_pair_value_halves": summarize_av_comparison(pair_cmp),
        },
        "same_input_shadow_comparison_summary": shadow_summary,
        "row_state_shadow_probe_summary": row_state_summary,
        "source_runtime_grounding": source_runtime_grounding(),
        "live_hygiene_records": {
            "gpu_hygiene_preflight": "gpu_hygiene_preflight.json",
            "nvidia_smi_full": "nvidia_smi_full.txt",
            "nvidia_smi_compute_apps": "nvidia_smi_compute_apps.csv",
            "gpu_lease_status_before": "gpu_lease_status.txt",
            "gpu_lease_status_after": "gpu_lease_status_after.json",
            "docker_info_summary": "docker_info_summary.txt",
            "docker_ps_before": "docker_ps_before.jsonl",
            "docker_ps_after": "docker_ps_after.jsonl",
            "r9_image_inspect": "r9_image_inspect.json",
        },
        "attention_output_comparison_summary": {
            "current_vs_current": {
                key: current_attention.get(key)
                for key in (
                    "status",
                    "all_equal_for_gate",
                    "output_digest_exact_equal",
                    "digest_mismatch_count",
                    "metadata_equal",
                    "input_layouts_equal",
                    "output_layouts_equal",
                    "reference_records",
                    "candidate_records",
                    "raw_tensor_exported",
                    "first_mismatches",
                )
            },
            "current_vs_pair_value_halves": {
                key: pair_attention.get(key)
                for key in (
                    "status",
                    "all_equal_for_gate",
                    "output_digest_exact_equal",
                    "digest_mismatch_count",
                    "metadata_equal",
                    "input_layouts_equal",
                    "output_layouts_equal",
                    "reference_records",
                    "candidate_records",
                    "raw_tensor_exported",
                    "first_mismatches",
                )
            },
        },
        "gates": gates,
        "failed_gates": [key for key, passed in gates.items() if not passed],
        "claim_boundary": decisive_boundary,
    }


def write_run_report(root: Path, decision: dict[str, Any]) -> None:
    shadow = decision.get("same_input_shadow_comparison_summary") or {}
    lines = [
        "# r9 pair-value-halves same-input shadow diagnostic RUN_REPORT",
        "",
        f"classification: {decision.get('classification')}",
        f"reason: {decision.get('reason')}",
        f"current_vs_current: {(decision.get('current_vs_current_decision') or {}).get('decision')}",
        f"shadow_enabled: {shadow.get('enabled')}",
        f"shadow_calls: {shadow.get('calls')}",
        f"shadow_mismatch_count: {shadow.get('mismatch_count')}",
        f"shadow_records_written: {shadow.get('records_written')}",
        "raw_tensor_exported: false",
        f"promote_to_n3: {decision.get('promote_to_n3')}",
        "boundary: diagnostic-only; no product speedup, long-video, BF16-fidelity, or quality-equivalence claim.",
        "",
    ]
    first = shadow.get("first_mismatches") if isinstance(shadow, dict) else None
    if first:
        rec = first[0]
        err = rec.get("error") or {}
        bucket = err.get("argmax_region_bucket") or {}
        lines.extend(
            [
                "first_mismatch:",
                f"  call_index: {rec.get('call_index')}",
                f"  step_layer: {(rec.get('metadata') or {}).get('step_index')}/{(rec.get('metadata') or {}).get('layer_index')}",
                f"  max_abs: {err.get('max_abs')}",
                f"  mean_abs: {err.get('mean_abs')}",
                f"  argmax_region: {bucket.get('region')}",
                f"  argmax_value_half: {bucket.get('value_half')}",
                "",
            ]
        )
    (root / "RUN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--vllm-omni-init-timeout-s", type=int, required=True)
    parser.add_argument("--vllm-omni-stage-init-timeout-s", type=int, required=True)
    parser.add_argument("--server-ready-timeout-s", type=int, required=True)
    parser.add_argument("--server-ready-poll-interval-s", type=int, required=True)
    parser.add_argument("--video-sync-timeout-s", type=int, required=True)
    parser.add_argument("--request-timeout-s", type=int, required=True)
    args = parser.parse_args()
    root = args.out_dir.resolve()
    timeout_values = {
        "vllm_omni_init_timeout_s": args.vllm_omni_init_timeout_s,
        "vllm_omni_stage_init_timeout_s": args.vllm_omni_stage_init_timeout_s,
        "server_ready_timeout_s": args.server_ready_timeout_s,
        "server_ready_poll_interval_s": args.server_ready_poll_interval_s,
        "video_sync_timeout_s": args.video_sync_timeout_s,
        "request_timeout_s": args.request_timeout_s,
    }
    decision = build_decision(root, timeout_values)
    (root / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_run_report(root, decision)
    print(
        json.dumps(
            {
                "classification": decision.get("classification"),
                "reason": decision.get("reason"),
                "promote_to_matched_n3": decision.get("promote_to_matched_n3", False),
                "decision": str(root / "decision.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
