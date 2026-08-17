#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Finalize the r9 full-prefix-block-skip real-chain gate.

The gate compares the retained Sol-Attn lane with
MINIMAX_H3_A6000_SOL_ATTN_SKIP_FULL_PREFIX_BLOCKS=0 versus =1 as the only
principal sparse-policy variable.  A second skip-off run is used as a same-policy
current-vs-current stability control before interpreting skip-on drift or timing.
Only decoded media metrics, bounded attention-output digests, telemetry counters,
and relative source/runtime facts are recorded; raw tensors are never exported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# Reuse the already-reviewed media comparator helpers from the pair-value gate.
# The policy comparator is local because this gate intentionally ignores only the
# skip_full_prefix_blocks flag when comparing candidate output digests.
try:
    from finalize_sol_attn_pair_value_halves_diagnostic import (  # type: ignore
        decoded_av_compare,
        host_mem_available_kib,
        parse_http,
        peak_csv_metric,
        read_json,
        structural_av,
        summarize_av_comparison,
        zero_copy_contract,
    )
except ModuleNotFoundError:  # pragma: no cover - direct import fallback in unusual launch contexts
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from finalize_sol_attn_pair_value_halves_diagnostic import (  # type: ignore
        decoded_av_compare,
        host_mem_available_kib,
        parse_http,
        peak_csv_metric,
        read_json,
        structural_av,
        summarize_av_comparison,
        zero_copy_contract,
    )

SKIP_OFF_A = "skip_off_a"
SKIP_OFF_B = "skip_off_b"
SKIP_ON = "skip_on"
MODES = (SKIP_OFF_A, SKIP_OFF_B, SKIP_ON)
EXPECTED_SPARSE_CALLS = 192


def _rel(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def density_has(tel: dict[str, Any], key: str) -> bool:
    samples = tel.get("density_samples", [])
    return any(bool(sample.get(key)) for sample in samples if isinstance(sample, dict))


def density_all(tel: dict[str, Any], key: str, expected: bool) -> bool:
    samples = [sample for sample in tel.get("density_samples", []) if isinstance(sample, dict)]
    if len(samples) != EXPECTED_SPARSE_CALLS:
        return False
    return all(bool(sample.get(key)) is expected for sample in samples)


def skipped_prefix_blocks_estimate(tel: dict[str, Any]) -> int | None:
    values: list[int] = []
    for sample in tel.get("density_samples", []):
        if not isinstance(sample, dict):
            continue
        value = sample.get("skipped_full_prefix_query_blocks_estimate")
        try:
            values.append(int(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return max(values)


def real_h3_v_layout_seen(tel: dict[str, Any]) -> bool:
    for sample in tel.get("layout_samples", []):
        if not isinstance(sample, dict):
            continue
        for tensor in sample.get("tensors", []):
            if tensor.get("name") != "value":
                continue
            if (
                tensor.get("shape") == [1, 38272, 56, 128]
                and tensor.get("stride") == [823001088, 21504, 128, 1]
                and tensor.get("storage_offset") == 14336
                and tensor.get("is_contiguous") is False
            ):
                return True
    return False


def _digest(record: dict[str, Any]) -> str | None:
    digest = record.get("output_digest")
    return digest.get("sha256") if isinstance(digest, dict) else None


def _policy_without(policy: Any, ignored: set[str]) -> Any:
    if not isinstance(policy, dict):
        return policy
    return {key: value for key, value in policy.items() if key not in ignored}


def attention_output_compare(
    reference_tel: dict[str, Any],
    candidate_tel: dict[str, Any],
    *,
    comparison: str,
    ignored_policy_keys: set[str],
) -> dict[str, Any]:
    ref_records = [r for r in reference_tel.get("diagnostic_output_records", []) if isinstance(r, dict)]
    cand_records = [r for r in candidate_tel.get("diagnostic_output_records", []) if isinstance(r, dict)]
    ref_failures = int(reference_tel.get("diagnostic_output_digest_failures", 0) or 0)
    cand_failures = int(candidate_tel.get("diagnostic_output_digest_failures", 0) or 0)
    first_mismatches: list[dict[str, Any]] = []
    counts = {
        "digest": 0,
        "metadata": 0,
        "input_layout": 0,
        "output_layout": 0,
        "policy": 0,
    }
    raw_tensor_exported = False
    for index, (ref, cand) in enumerate(zip(ref_records, cand_records)):
        raw_tensor_exported = raw_tensor_exported or bool(ref.get("raw_tensor_exported")) or bool(
            cand.get("raw_tensor_exported")
        )
        mismatch: dict[str, Any] = {"call_index": index}
        if _digest(ref) != _digest(cand):
            counts["digest"] += 1
            mismatch["reference_output_sha256"] = _digest(ref)
            mismatch["candidate_output_sha256"] = _digest(cand)
        if ref.get("metadata") != cand.get("metadata"):
            counts["metadata"] += 1
            mismatch["reference_metadata"] = ref.get("metadata")
            mismatch["candidate_metadata"] = cand.get("metadata")
        if ref.get("input_layouts") != cand.get("input_layouts"):
            counts["input_layout"] += 1
            mismatch["input_layout_mismatch"] = True
        if ref.get("output_layout") != cand.get("output_layout"):
            counts["output_layout"] += 1
            mismatch["output_layout_mismatch"] = True
        if _policy_without(ref.get("policy"), ignored_policy_keys) != _policy_without(
            cand.get("policy"), ignored_policy_keys
        ):
            counts["policy"] += 1
            mismatch["reference_policy"] = ref.get("policy")
            mismatch["candidate_policy"] = cand.get("policy")
        if len(mismatch) > 1 and len(first_mismatches) < 8:
            first_mismatches.append(mismatch)

    count_equal = len(ref_records) == len(cand_records) == EXPECTED_SPARSE_CALLS
    failures_zero = ref_failures == 0 and cand_failures == 0
    raw_tensor_clean = not raw_tensor_exported and all(
        rec.get("raw_tensor_exported") is False
        for rec in ref_records[:EXPECTED_SPARSE_CALLS] + cand_records[:EXPECTED_SPARSE_CALLS]
    )
    status = "pass" if count_equal and failures_zero and raw_tensor_clean else "blocked"
    digest_mismatches = counts["digest"] + abs(len(ref_records) - len(cand_records))
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
        "ignored_policy_keys": sorted(ignored_policy_keys),
        "output_digest_exact_equal": status == "pass" and digest_mismatches == 0,
        "metadata_equal": status == "pass" and counts["metadata"] == 0,
        "input_layouts_equal": status == "pass" and counts["input_layout"] == 0,
        "output_layouts_equal": status == "pass" and counts["output_layout"] == 0,
        "fixed_policy_equal": status == "pass" and counts["policy"] == 0,
        "digest_mismatch_count": digest_mismatches,
        "metadata_mismatch_count": counts["metadata"],
        "input_layout_mismatch_count": counts["input_layout"],
        "output_layout_mismatch_count": counts["output_layout"],
        "policy_mismatch_count": counts["policy"],
        "first_mismatches": first_mismatches,
        "all_equal_for_gate": bool(
            status == "pass"
            and digest_mismatches == 0
            and counts["metadata"] == 0
            and counts["input_layout"] == 0
            and counts["output_layout"] == 0
            and counts["policy"] == 0
        ),
    }


def source_runtime_grounding() -> dict[str, Any]:
    rel_paths = [
        "ports/minimax_h3_a6000/src/minimax_h3_a6000/env.py",
        "ports/minimax_h3_a6000/src/minimax_h3_a6000/sol_attn_backend.py",
        "ports/minimax_h3_a6000/src/minimax_h3_a6000/sol_attn_triton_sm86.py",
        "ports/minimax_h3_a6000/integration/run_sol_attn_h3_prefix_skip_n1.sh",
        "ports/minimax_h3_a6000/integration/finalize_sol_attn_prefix_skip_diagnostic.py",
        "ports/minimax_h3_a6000/patches/vllm_omni_h3_a6000_opt_in.patch",
        "upstreams/Sana-sol-engine/models/minimax_h3/gb200/optimized/sol_attn_h3.py",
        "upstreams/Sana-sol-engine/techniques/sparse_backends/sol_attn/interface.py",
        "upstreams/Sana-sol-engine/techniques/sparse_backends/sol_attn/triton_ref/fwd.py",
    ]
    hashes: dict[str, str] = {}
    for rel in rel_paths:
        path = Path(rel)
        hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() and path.is_file() else "missing"
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
    for module_name in ("torch", "triton"):
        try:
            module = __import__(module_name)
            runtime_versions[module_name] = getattr(module, "__version__", "unknown")
            if module_name == "torch":
                runtime_versions["torch_cuda"] = getattr(getattr(module, "version", None), "cuda", None)
        except Exception as exc:  # noqa: BLE001
            runtime_versions[module_name] = f"unavailable:{type(exc).__name__}"
    return {
        "schema_version": "minimax_h3_a6000_source_runtime_grounding_v1",
        "pinned_upstream_revision": upstream_head,
        "source_hashes": hashes,
        "runtime_versions": runtime_versions,
    }


def build_decision(root: Path, timeout_values: dict[str, int]) -> dict[str, Any]:
    dirs = {mode: root / mode for mode in MODES}
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
            "schema_version": "minimax_h3_a6000_sol_attn_prefix_skip_n1_v1",
            "classification": "blocked",
            "reason": "incomplete_artifacts",
            "missing_paths": missing,
            "promote_to_matched_n3": False,
            "promote_to_n3": False,
            "not_speedup_claim": True,
            "no_product_speedup_claim": True,
            "timeout_values": timeout_values,
            "claim_boundary": "Incomplete artifact blocker only; no product speedup, BF16-fidelity, long-video, quality-equivalence, or promotion claim.",
        }

    av = {mode: read_json(dirs[mode] / "av_validation.json") for mode in MODES}
    tel = {mode: read_json(dirs[mode] / "sol_attn_telemetry.sol_attn.json") for mode in MODES}
    http = {mode: parse_http(dirs[mode] / "http_metrics.txt") for mode in MODES}
    warm = {mode: parse_http(dirs[mode] / "warmup_http_metrics.txt") for mode in MODES}
    memories = {mode: peak_csv_metric(dirs[mode] / "gpu_resource_samples.csv", "memory.used") for mode in MODES}
    powers = {mode: peak_csv_metric(dirs[mode] / "gpu_resource_samples.csv", "power.draw") for mode in MODES}
    temps = {mode: peak_csv_metric(dirs[mode] / "gpu_resource_samples.csv", "temperature.gpu") for mode in MODES}
    host_before = {mode: host_mem_available_kib(dirs[mode] / "host_resource_before.json") for mode in MODES}
    host_after = {mode: host_mem_available_kib(dirs[mode] / "host_resource_after.json") for mode in MODES}

    current_cmp = decoded_av_compare(
        dirs[SKIP_OFF_A] / "output.mp4",
        dirs[SKIP_OFF_B] / "output.mp4",
        root,
        "skip_off_a_vs_skip_off_b_measured_outputs",
    )
    candidate_cmp = decoded_av_compare(
        dirs[SKIP_OFF_A] / "output.mp4",
        dirs[SKIP_ON] / "output.mp4",
        root,
        "skip_off_a_vs_skip_on_measured_outputs",
    )
    (root / "decoded_av_comparison_current_vs_current.json").write_text(
        json.dumps(current_cmp, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "decoded_av_comparison_skip_off_vs_skip_on.json").write_text(
        json.dumps(candidate_cmp, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "decoded_av_comparison.json").write_text(
        json.dumps(candidate_cmp, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    current_attention = attention_output_compare(
        tel[SKIP_OFF_A],
        tel[SKIP_OFF_B],
        comparison="skip_off_a_vs_skip_off_b_attention_outputs",
        ignored_policy_keys=set(),
    )
    candidate_attention = attention_output_compare(
        tel[SKIP_OFF_A],
        tel[SKIP_ON],
        comparison="skip_off_a_vs_skip_on_attention_outputs",
        ignored_policy_keys={"skip_full_prefix_blocks"},
    )
    (root / "attention_output_comparison_current_vs_current.json").write_text(
        json.dumps(current_attention, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "attention_output_comparison_skip_off_vs_skip_on.json").write_text(
        json.dumps(candidate_attention, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    observed_r8_cv_pct = 0.5072177175606011
    promotion_threshold_pct = max(1.5, 2.0 * observed_r8_cv_pct)
    improvement_pct = (http[SKIP_OFF_A]["time_total_s"] - http[SKIP_ON]["time_total_s"]) / http[SKIP_OFF_A][
        "time_total_s"
    ] * 100.0

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
    current_av_metadata_equal = all(av[SKIP_OFF_A].get(key) == av[SKIP_OFF_B].get(key) for key in av_fields)
    candidate_av_metadata_equal = all(av[SKIP_OFF_A].get(key) == av[SKIP_ON].get(key) for key in av_fields)
    current_media_integrity_ok = current_cmp.get("status") == "pass" and bool(current_cmp.get("media_integrity_ok"))
    candidate_media_integrity_ok = candidate_cmp.get("status") == "pass" and bool(candidate_cmp.get("media_integrity_ok"))
    current_av_equal = bool(current_cmp.get("decoded_content_equal"))
    candidate_av_equal = bool(candidate_cmp.get("decoded_content_equal"))

    gates: dict[str, bool] = {
        "all_http_200": all(http[mode].get("http_code") == 200 for mode in MODES),
        "all_warmups_http_200": all(warm[mode].get("http_code") == 200 for mode in MODES),
        "all_structural_av_valid": all(structural_av(av[mode]) for mode in MODES),
        "current_current_av_metadata_equal": current_av_metadata_equal,
        "current_current_decoded_media_integrity_ok": current_media_integrity_ok,
        "current_current_decoded_av_content_equal": current_av_equal,
        "current_current_attention_output_digests_equal": bool(current_attention.get("all_equal_for_gate")),
        "candidate_av_metadata_equal": candidate_av_metadata_equal,
        "candidate_decoded_media_integrity_ok": candidate_media_integrity_ok,
        "candidate_decoded_av_content_equal": candidate_av_equal,
        "candidate_attention_output_digests_equal": bool(candidate_attention.get("all_equal_for_gate")),
        "all_sparse_calls_192": all(int(tel[mode].get("sparse_calls", 0)) == EXPECTED_SPARSE_CALLS for mode in MODES),
        "all_sparse_candidates_192": all(
            int(tel[mode].get("sparse_candidate_calls", 0)) == EXPECTED_SPARSE_CALLS for mode in MODES
        ),
        "all_fallback_calls_zero": all(int(tel[mode].get("fallback_calls", -1)) == 0 for mode in MODES),
        "all_zero_materialization_and_input_copies": all(zero_copy_contract(tel[mode]) for mode in MODES),
        "all_stride_aware_value_calls_192": all(
            int(tel[mode].get("stride_aware_value_calls", 0)) == EXPECTED_SPARSE_CALLS for mode in MODES
        ),
        "all_prefix_dense_overwrite_calls_192": all(
            int(tel[mode].get("prefix_query_dense_calls", 0)) == EXPECTED_SPARSE_CALLS for mode in MODES
        ),
        "all_exact_prefix_query_absent": all(int(tel[mode].get("exact_prefix_query_calls", 0) or 0) == 0 for mode in MODES),
        "skip_off_modes_have_no_full_prefix_skip_marker": all(
            density_all(tel[mode], "skip_full_prefix_blocks", False) for mode in (SKIP_OFF_A, SKIP_OFF_B)
        ),
        "skip_on_mode_has_full_prefix_skip_marker": density_all(tel[SKIP_ON], "skip_full_prefix_blocks", True),
        "skip_on_skipped_prefix_blocks_positive": (skipped_prefix_blocks_estimate(tel[SKIP_ON]) or 0) > 0,
        "all_pair_value_halves_absent": all(not density_has(tel[mode], "pair_value_halves") for mode in MODES),
        "real_h3_fused_value_layout_seen_all_modes": all(real_h3_v_layout_seen(tel[mode]) for mode in MODES),
        "all_gpu_copy_time_zero": all(float(tel[mode].get("materialize_gpu_copy_latency_ms", -1.0)) == 0.0 for mode in MODES),
        "attention_gpu_timing_complete": all(
            int(tel[mode].get("sparse_attention_timed_calls", 0)) == EXPECTED_SPARSE_CALLS
            and float(tel[mode].get("sparse_attention_gpu_latency_ms", 0.0)) > 0.0
            for mode in MODES
        ),
        "denoise_gpu_timing_complete": all(
            int(tel[mode].get("denoise_timed_calls", 0)) > 0
            and float(tel[mode].get("denoise_gpu_latency_ms", 0.0)) > 0.0
            for mode in MODES
        ),
        "no_gpu_timing_failures": all(
            int(tel[mode].get(key, -1)) == 0
            for mode in MODES
            for key in (
                "materialize_gpu_timing_failures",
                "sparse_attention_gpu_timing_failures",
                "denoise_gpu_timing_failures",
            )
        ),
        "diagnostic_raw_tensor_export_zero": not bool(current_attention.get("raw_tensor_exported"))
        and not bool(candidate_attention.get("raw_tensor_exported")),
        "resource_samples_present": all(value is not None for metric in (memories, powers, temps) for value in metric.values()),
        "candidate_peak_memory_not_higher": memories[SKIP_ON] is not None
        and memories[SKIP_OFF_A] is not None
        and memories[SKIP_ON] <= memories[SKIP_OFF_A],
        "e2e_signal_exceeds_predeclared_threshold": improvement_pct > promotion_threshold_pct,
    }

    media_compare_blocked = current_cmp.get("status") != "pass" or candidate_cmp.get("status") != "pass"
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
    elif not candidate_media_integrity_ok:
        classification = "blocked"
        reason = "candidate_decoded_av_media_integrity_failed"
    else:
        correctness_gate_names = [key for key in gates if key != "e2e_signal_exceeds_predeclared_threshold"]
        correctness_ok = all(gates[key] for key in correctness_gate_names)
        if not correctness_ok:
            classification = "reject"
            if not gates["candidate_attention_output_digests_equal"]:
                reason = "attention_output_divergence"
            elif not gates["candidate_decoded_av_content_equal"]:
                reason = "decoded_av_divergence"
            else:
                reason = "correctness_or_contract"
        elif not gates["e2e_signal_exceeds_predeclared_threshold"]:
            classification = "reject"
            reason = "no_above_noise_n1_signal"
        else:
            classification = "promote_to_matched_n3"
            reason = "n1_gate_passed_requires_independent_reviewer_before_n3"

    return {
        "schema_version": "minimax_h3_a6000_sol_attn_prefix_skip_n1_v1",
        "classification": classification,
        "reason": reason,
        "promote_to_matched_n3": classification == "promote_to_matched_n3",
        "promote_to_n3": classification == "promote_to_matched_n3",
        "reviewer_acceptance_required_before_promotion": True,
        "reviewer_acceptance_status": "pending_external_reviewer_not_authored_by_runner",
        "not_speedup_claim": True,
        "no_product_speedup_claim": True,
        "lane": "matched_n1_5step_sol_attn_opt_in_not_bf16_fidelity_prefix_skip",
        "workload": {"width": 1344, "height": 768, "frames": 124, "fps": 24, "duration_s": 5.166667, "steps": 5, "seed": 0},
        "principal_variable": "MINIMAX_H3_A6000_SOL_ATTN_SKIP_FULL_PREFIX_BLOCKS=0_vs_1",
        "fixed_variables": {
            "cache": "off",
            "stride_aware_v": "on",
            "dense_prefix_overwrite": "preserved",
            "tau": 1.0,
            "routing_and_exact_block_order": "unchanged_retained_sol_attn_policy",
            "pair_value_halves": "off",
            "exact_prefix_query": "off",
            "static_prefix_sink": "off",
            "bitmask_scheduler": "off",
            "diagnostic_materialization": "off_for_all_modes",
            "diagnostic_output_digest": "on_for_bounded_gate_only_all_modes",
            "diagnostic_output_max_calls": 256,
            "dense_first_steps": 0,
            "dense_first_layers": 2,
        },
        "timeout_values": timeout_values,
        "observed_r8_cv_pct": observed_r8_cv_pct,
        "promotion_threshold_pct": promotion_threshold_pct,
        "http_e2e_seconds": {mode: http[mode]["time_total_s"] for mode in MODES},
        "excluded_warmup_http_seconds": {mode: warm[mode]["time_total_s"] for mode in MODES},
        "n1_http_e2e_improvement_pct_skip_off_a_vs_skip_on": improvement_pct,
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
            for mode in MODES
        },
        "resource_summary": {
            "peak_gpu_memory_mib": memories,
            "peak_gpu_power_w": powers,
            "peak_gpu_temperature_c": temps,
            "host_mem_available_kib_before": host_before,
            "host_mem_available_kib_after": host_after,
            "resource_sample_files": {mode: _rel(root, dirs[mode] / "gpu_resource_samples.csv") for mode in MODES}
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
                    "prefix_query_dense_calls",
                    "exact_prefix_query_calls",
                    "diagnostic_output_digest_failures",
                )
            }
            | {
                "diagnostic_output_records": len(tel[mode].get("diagnostic_output_records", [])),
                "skip_full_prefix_blocks_seen": density_has(tel[mode], "skip_full_prefix_blocks"),
                "skipped_full_prefix_query_blocks_estimate_max": skipped_prefix_blocks_estimate(tel[mode]),
            }
            for mode in MODES
        },
        "current_vs_current_decision": {
            "decoded_av_comparison_path": "decoded_av_comparison_current_vs_current.json",
            "attention_output_comparison_path": "attention_output_comparison_current_vs_current.json",
            "decoded_av_stable": current_av_equal,
            "attention_output_deterministic": bool(current_attention.get("all_equal_for_gate")),
            "decision": "stable" if current_av_equal and current_attention.get("all_equal_for_gate") else "fail_closed_full_chain_or_attention_nondeterminism",
            "summary": summarize_av_comparison(current_cmp),
        },
        "skip_off_vs_skip_on_decision": {
            "decoded_av_comparison_path": "decoded_av_comparison_skip_off_vs_skip_on.json",
            "attention_output_comparison_path": "attention_output_comparison_skip_off_vs_skip_on.json",
            "decoded_av_equal": candidate_av_equal,
            "attention_output_equal": bool(candidate_attention.get("all_equal_for_gate")),
            "decision": classification,
            "summary": summarize_av_comparison(candidate_cmp),
        },
        "output_checks": {
            "structural_av_all_modes": gates["all_structural_av_valid"],
            "current_current_av_metadata_equal": current_av_metadata_equal,
            "candidate_av_metadata_equal": candidate_av_metadata_equal,
            "mp4_sha256_recorded_not_gate_current_current": current_cmp.get("mp4_file_sha256_equal_recorded_not_gate"),
            "mp4_sha256_recorded_not_gate_candidate": candidate_cmp.get("mp4_file_sha256_equal_recorded_not_gate"),
            "hash_equality_used_for_decision": False,
            "decoded_av_content_used_for_decision": True,
            "attention_output_digest_used_for_kernel_localization": True,
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
            "skip_off_vs_skip_on": summarize_av_comparison(candidate_cmp),
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
                    "fixed_policy_equal",
                    "reference_records",
                    "candidate_records",
                    "raw_tensor_exported",
                    "first_mismatches",
                )
            },
            "skip_off_vs_skip_on": {
                key: candidate_attention.get(key)
                for key in (
                    "status",
                    "all_equal_for_gate",
                    "output_digest_exact_equal",
                    "digest_mismatch_count",
                    "metadata_equal",
                    "input_layouts_equal",
                    "output_layouts_equal",
                    "fixed_policy_equal",
                    "ignored_policy_keys",
                    "reference_records",
                    "candidate_records",
                    "raw_tensor_exported",
                    "first_mismatches",
                )
            },
        },
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
        "gates": gates,
        "failed_gates": [key for key, passed in gates.items() if not passed],
        "claim_boundary": "Matched N=1 5-step prefix-skip Sol-Attn diagnostic only; no product/BF16/long-video/formal/public/SOTA speedup claim.",
    }


def write_run_report(root: Path, decision: dict[str, Any]) -> None:
    timing = decision.get("http_e2e_seconds") or {}
    components = decision.get("gpu_component_ms") or {}
    current = decision.get("current_vs_current_decision") or {}
    candidate = decision.get("skip_off_vs_skip_on_decision") or {}
    lines = [
        "# r9 Sol-Attn full-prefix-block skip N=1 RUN_REPORT",
        "",
        f"classification: {decision.get('classification')}",
        f"reason: {decision.get('reason')}",
        f"principal_variable: {decision.get('principal_variable')}",
        f"current_vs_current: {current.get('decision')}",
        f"skip_off_vs_skip_on: {candidate.get('decision')}",
        f"http_e2e_seconds: {timing}",
        f"gpu_component_ms: {components}",
        f"failed_gates: {decision.get('failed_gates')}",
        f"promote_to_n3: {decision.get('promote_to_n3')}",
        "raw_tensor_exported: false",
        "boundary: diagnostic-only; no product speedup, long-video, BF16-fidelity, formal, public-comparison, or SOTA claim.",
        "",
    ]
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
