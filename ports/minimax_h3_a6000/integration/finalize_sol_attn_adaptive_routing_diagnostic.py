#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Finalize the r9 adaptive-routing matched N=1 real-chain gate.

Compares retained current Sol-Attn routing (tau=1.0/diag, adaptive off) with a
single default-off practical/approximate routing candidate.  The candidate is
not required to be attention-output- or decoded-media-identical; this finalizer
records the objective deltas and keeps the lane default-off pending later
multi-prompt/multi-seed quality review.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from finalize_sol_attn_pair_value_halves_diagnostic import (  # noqa: E402
    EXPECTED_SPARSE_CALLS,
    decoded_av_compare,
    density_has,
    host_mem_available_kib,
    parse_http,
    peak_csv_metric,
    read_json,
    structural_av,
    summarize_av_comparison,
    zero_copy_contract,
)

CURRENT_A = "current_retained_a"
CURRENT_B = "current_retained_b"
CANDIDATE = "adaptive_tau2_diag"


def _digest(record: dict[str, Any]) -> str | None:
    digest = record.get("output_digest")
    return digest.get("sha256") if isinstance(digest, dict) else None


def _attention_digest_compare(
    reference_tel: dict[str, Any],
    candidate_tel: dict[str, Any],
    *,
    comparison: str,
    require_equal: bool,
) -> dict[str, Any]:
    ref_records = [r for r in reference_tel.get("diagnostic_output_records", []) if isinstance(r, dict)]
    cand_records = [r for r in candidate_tel.get("diagnostic_output_records", []) if isinstance(r, dict)]
    digest_mismatches = 0
    metadata_mismatches = 0
    layout_mismatches = 0
    policy_mismatches = 0
    raw_tensor_exported = False
    first_mismatches: list[dict[str, Any]] = []
    for index, (ref, cand) in enumerate(zip(ref_records, cand_records)):
        mismatch: dict[str, Any] = {"call_index": index}
        raw_tensor_exported = raw_tensor_exported or bool(ref.get("raw_tensor_exported")) or bool(cand.get("raw_tensor_exported"))
        if _digest(ref) != _digest(cand):
            digest_mismatches += 1
            mismatch["reference_output_sha256"] = _digest(ref)
            mismatch["candidate_output_sha256"] = _digest(cand)
        if ref.get("metadata") != cand.get("metadata"):
            metadata_mismatches += 1
            mismatch["metadata_mismatch"] = True
        if ref.get("input_layouts") != cand.get("input_layouts") or ref.get("output_layout") != cand.get("output_layout"):
            layout_mismatches += 1
            mismatch["layout_mismatch"] = True
        if ref.get("policy") != cand.get("policy"):
            policy_mismatches += 1
            mismatch["reference_policy"] = ref.get("policy")
            mismatch["candidate_policy"] = cand.get("policy")
        if len(mismatch) > 1 and len(first_mismatches) < 8:
            first_mismatches.append(mismatch)
    ref_failures = int(reference_tel.get("diagnostic_output_digest_failures", 0) or 0)
    cand_failures = int(candidate_tel.get("diagnostic_output_digest_failures", 0) or 0)
    count_equal = len(ref_records) == len(cand_records) == EXPECTED_SPARSE_CALLS
    raw_clean = not raw_tensor_exported and all(
        r.get("raw_tensor_exported") is False for r in ref_records[:EXPECTED_SPARSE_CALLS] + cand_records[:EXPECTED_SPARSE_CALLS]
    )
    status = "pass" if count_equal and ref_failures == 0 and cand_failures == 0 and raw_clean else "blocked"
    exact_equal = bool(status == "pass" and digest_mismatches == 0 and metadata_mismatches == 0 and layout_mismatches == 0 and (policy_mismatches == 0 or not require_equal))
    return {
        "schema_version": "minimax_h3_a6000_attention_output_digest_comparison_v1_adaptive",
        "comparison": comparison,
        "status": status,
        "expected_sparse_records_per_mode": EXPECTED_SPARSE_CALLS,
        "reference_records": len(ref_records),
        "candidate_records": len(cand_records),
        "reference_digest_failures": ref_failures,
        "candidate_digest_failures": cand_failures,
        "raw_tensor_exported": raw_tensor_exported,
        "raw_tensor_values_available": False,
        "require_equal": bool(require_equal),
        "output_digest_exact_equal": digest_mismatches == 0,
        "metadata_equal": metadata_mismatches == 0,
        "layouts_equal": layout_mismatches == 0,
        "policy_equal": policy_mismatches == 0,
        "digest_mismatch_count": digest_mismatches + abs(len(ref_records) - len(cand_records)),
        "metadata_mismatch_count": metadata_mismatches,
        "layout_mismatch_count": layout_mismatches,
        "policy_mismatch_count": policy_mismatches,
        "first_mismatches": first_mismatches,
        "all_equal_for_gate": exact_equal if require_equal else status == "pass",
    }


def _density_value(tel: dict[str, Any], key: str) -> Any:
    for sample in tel.get("density_samples", []):
        if isinstance(sample, dict) and key in sample:
            return sample.get(key)
    return None


def _mode_summary(mode: str, mode_dir: Path, tel: dict[str, Any], http: dict[str, Any], warm: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": mode,
        "http": http,
        "warmup_http": warm,
        "telemetry": {
            "sparse_candidate_calls": int(tel.get("sparse_candidate_calls", 0) or 0),
            "sparse_calls": int(tel.get("sparse_calls", 0) or 0),
            "fallback_calls": int(tel.get("fallback_calls", 0) or 0),
            "decline_reasons": tel.get("decline_reasons", {}),
            "fallback_reasons": tel.get("fallback_reasons", {}),
            "materialize_copy_count": int(tel.get("materialize_copy_count", 0) or 0),
            "materialize_copy_bytes": int(tel.get("materialize_copy_bytes", 0) or 0),
            "input_copy_events": int(tel.get("input_copy_events", 0) or 0),
            "input_copy_bytes": int(tel.get("input_copy_bytes", 0) or 0),
            "stride_aware_value_calls": int(tel.get("stride_aware_value_calls", 0) or 0),
            "sparse_attention_gpu_latency_ms": float(tel.get("sparse_attention_gpu_latency_ms", 0.0) or 0.0),
            "sparse_attention_timed_calls": int(tel.get("sparse_attention_timed_calls", 0) or 0),
            "denoise_gpu_latency_ms": float(tel.get("denoise_gpu_latency_ms", 0.0) or 0.0),
            "denoise_timed_calls": int(tel.get("denoise_timed_calls", 0) or 0),
            "density_first": tel.get("density_samples", [])[:1],
        },
        "resources": {
            "peak_gpu_memory_mib": peak_csv_metric(mode_dir / "gpu_resource_samples.csv", "memory.used"),
            "peak_power_w": peak_csv_metric(mode_dir / "gpu_resource_samples.csv", "power.draw"),
            "peak_temperature_c": peak_csv_metric(mode_dir / "gpu_resource_samples.csv", "temperature.gpu"),
            "host_mem_available_before_kib": host_mem_available_kib(mode_dir / "host_resource_before.json"),
            "host_mem_available_after_kib": host_mem_available_kib(mode_dir / "host_resource_after.json"),
        },
    }


def finalize(root: Path) -> dict[str, Any]:
    modes = [CURRENT_A, CURRENT_B, CANDIDATE]
    dirs = {mode: root / mode for mode in modes}
    missing = [str(path.relative_to(root)) for path in dirs.values() if not path.exists()]
    if missing:
        return {
            "schema_version": "minimax_h3_a6000_sol_attn_adaptive_routing_n1_v1",
            "classification": "blocked",
            "decision": "blocked_missing_mode_dirs",
            "missing": missing,
            "promote_to_matched_n3": False,
        }

    av = {mode: read_json(dirs[mode] / "av_validation.json") for mode in modes}
    tel = {mode: read_json(dirs[mode] / "sol_attn_telemetry.sol_attn.json") for mode in modes}
    http = {mode: parse_http(dirs[mode] / "http_metrics.txt") for mode in modes}
    warm = {mode: parse_http(dirs[mode] / "warmup_http_metrics.txt") for mode in modes}

    current_cmp = decoded_av_compare(
        dirs[CURRENT_A] / "output.mp4",
        dirs[CURRENT_B] / "output.mp4",
        root,
        "current_retained_a_vs_current_retained_b_measured_outputs",
    )
    candidate_cmp = decoded_av_compare(
        dirs[CURRENT_A] / "output.mp4",
        dirs[CANDIDATE] / "output.mp4",
        root,
        "current_retained_a_vs_adaptive_tau2_diag_measured_outputs",
    )
    (root / "decoded_av_comparison_current_vs_current.json").write_text(json.dumps(current_cmp, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "decoded_av_comparison_current_vs_adaptive_tau2_diag.json").write_text(json.dumps(candidate_cmp, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    current_attention = _attention_digest_compare(
        tel[CURRENT_A], tel[CURRENT_B], comparison="current_retained_a_vs_current_retained_b_attention_outputs", require_equal=True
    )
    candidate_attention = _attention_digest_compare(
        tel[CURRENT_A], tel[CANDIDATE], comparison="current_retained_a_vs_adaptive_tau2_diag_attention_outputs", require_equal=False
    )
    (root / "attention_output_comparison_current_vs_current.json").write_text(json.dumps(current_attention, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "attention_output_comparison_current_vs_adaptive_tau2_diag.json").write_text(json.dumps(candidate_attention, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    observed_r8_cv_pct = 0.5072177175606011
    promotion_threshold_pct = max(1.5, 2.0 * observed_r8_cv_pct)
    current_http_s = float(http[CURRENT_A].get("time_total_s", 0.0) or 0.0)
    candidate_http_s = float(http[CANDIDATE].get("time_total_s", 0.0) or 0.0)
    improvement_pct = 100.0 * (current_http_s - candidate_http_s) / current_http_s if current_http_s else 0.0

    current_av_equal = bool(current_cmp.get("decoded_content_equal"))
    current_media_integrity = current_cmp.get("status") == "pass" and bool(current_cmp.get("media_integrity_ok"))
    candidate_media_integrity = candidate_cmp.get("status") == "pass" and bool(candidate_cmp.get("media_integrity_ok"))
    candidate_video = candidate_cmp.get("video") or {}
    candidate_audio = candidate_cmp.get("audio") or {}

    gates = {
        "all_http_200": all(http[mode].get("http_code") == 200 for mode in modes),
        "all_warmups_http_200": all(warm[mode].get("http_code") == 200 for mode in modes),
        "all_structural_av_valid": all(structural_av(av[mode]) for mode in modes),
        "current_current_decoded_media_integrity_ok": current_media_integrity,
        "current_current_decoded_av_content_equal": current_av_equal,
        "current_current_attention_output_digests_equal": bool(current_attention.get("all_equal_for_gate")),
        "candidate_decoded_media_integrity_ok": candidate_media_integrity,
        "candidate_decoded_video_frames_match": candidate_video.get("reference_decoded_frames") == candidate_video.get("candidate_decoded_frames") == 124,
        "candidate_decoded_audio_present": int((candidate_audio.get("candidate_decoded_frames") or 0)) > 0,
        "all_sparse_calls_192": all(int(tel[mode].get("sparse_calls", 0) or 0) == EXPECTED_SPARSE_CALLS for mode in modes),
        "all_sparse_candidates_192": all(int(tel[mode].get("sparse_candidate_calls", 0) or 0) == EXPECTED_SPARSE_CALLS for mode in modes),
        "all_fallback_calls_zero": all(int(tel[mode].get("fallback_calls", -1)) == 0 for mode in modes),
        "all_zero_materialization_and_input_copies": all(zero_copy_contract(tel[mode]) for mode in modes),
        "all_stride_aware_value_calls_192": all(int(tel[mode].get("stride_aware_value_calls", 0) or 0) == EXPECTED_SPARSE_CALLS for mode in modes),
        "all_skip_full_prefix_blocks_seen": all(density_has(tel[mode], "skip_full_prefix_blocks") for mode in modes),
        "candidate_adaptive_routing_seen": density_has(tel[CANDIDATE], "adaptive_routing"),
        "candidate_tau2_diag_seen": _density_value(tel[CANDIDATE], "tau") == 2.0 and _density_value(tel[CANDIDATE], "thresh_type") == "diag",
        "current_adaptive_routing_absent": not density_has(tel[CURRENT_A], "adaptive_routing") and not density_has(tel[CURRENT_B], "adaptive_routing"),
        "candidate_attention_digest_records_present": candidate_attention.get("status") == "pass",
        "resource_samples_present": all((root / mode / "gpu_resource_samples.csv").exists() for mode in modes),
        "e2e_signal_exceeds_predeclared_threshold": improvement_pct > promotion_threshold_pct,
    }
    failed = [key for key, value in gates.items() if not value]
    if current_cmp.get("status") != "pass" or candidate_cmp.get("status") != "pass":
        classification = "blocked"
        decision = "blocked_decoded_av_comparator_failed"
    elif not gates["current_current_attention_output_digests_equal"] or not gates["current_current_decoded_av_content_equal"]:
        classification = "blocked"
        decision = "blocked_current_retained_nondeterminism"
    elif any(key in failed for key in ("candidate_decoded_media_integrity_ok", "all_structural_av_valid")):
        classification = "reject"
        decision = "reject_adaptive_routing_media_integrity_failed"
    elif any(key in failed for key in ("all_sparse_calls_192", "all_fallback_calls_zero", "all_zero_materialization_and_input_copies", "candidate_adaptive_routing_seen", "candidate_tau2_diag_seen")):
        classification = "reject"
        decision = "reject_adaptive_routing_contract_or_telemetry_failed"
    elif not gates["e2e_signal_exceeds_predeclared_threshold"]:
        classification = "reject"
        decision = "reject_adaptive_routing_no_n1_e2e_signal"
    else:
        classification = "survived_n1_practical_approx_route_gate"
        decision = "retain_default_off_adaptive_routing_for_reviewer_and_quality_gate"

    summaries = {mode: _mode_summary(mode, dirs[mode], tel[mode], http[mode], warm[mode]) for mode in modes}
    decision_record = {
        "schema_version": "minimax_h3_a6000_sol_attn_adaptive_routing_n1_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "classification": classification,
        "decision": decision,
        "failed_gates": failed,
        "gates": gates,
        "promote_to_matched_n3": False,
        "promote_to_formal_speedup": False,
        "reviewer_required_before_any_promotion": True,
        "principal_variable": "MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_ROUTING=1 with tau=2.0 and thresh_type=diag versus retained tau=1.0 diag Sol-Attn lane",
        "workload": {"width": 1344, "height": 768, "frames": 124, "fps": 24, "duration_s": 5.166667, "steps": 5, "seed": 0},
        "mode_summaries": summaries,
        "timing_summary": {
            "current_a_http_s": current_http_s,
            "current_b_http_s": float(http[CURRENT_B].get("time_total_s", 0.0) or 0.0),
            "candidate_http_s": candidate_http_s,
            "n1_http_improvement_pct_candidate_vs_current_a": improvement_pct,
            "promotion_threshold_pct_for_route_gate_only": promotion_threshold_pct,
            "not_formal_speedup": True,
        },
        "decoded_av_comparison_summary": {
            "current_vs_current": summarize_av_comparison(current_cmp),
            "current_vs_candidate": summarize_av_comparison(candidate_cmp),
            "candidate_exact_decoded_equality_required": False,
            "candidate_is_practical_approximate": True,
        },
        "attention_output_comparison_summary": {
            "current_vs_current": current_attention,
            "current_vs_candidate": candidate_attention,
            "candidate_attention_digest_equality_required": False,
            "raw_tensor_exported": bool(current_attention.get("raw_tensor_exported")) or bool(candidate_attention.get("raw_tensor_exported")),
        },
        "claim_boundaries": [
            "N=1 short 5-step Sol-Attn opt-in route gate only.",
            "No BF16-fidelity, long-video/native-context, formal speedup, product-quality, public-comparison, or SOTA claim.",
            "Adaptive routing is approximate and remains default-off pending independent Reviewer and later multi-prompt/multi-seed quality gates.",
        ],
    }
    (root / "decision.json").write_text(json.dumps(decision_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "RUN_REPORT.md").write_text(
        "# Adaptive-routing Sol-Attn N=1 gate\n\n"
        f"- classification: `{classification}`\n"
        f"- decision: `{decision}`\n"
        f"- candidate HTTP: `{candidate_http_s:.6f}` s; current A HTTP: `{current_http_s:.6f}` s; N=1 route signal: `{improvement_pct:.3f}%`\n"
        "- boundary: default-off practical/approximate short 5-step gate; no formal speedup or quality claim.\n",
        encoding="utf-8",
    )
    return decision_record


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: finalize_sol_attn_adaptive_routing_diagnostic.py <run_dir>", file=sys.stderr)
        return 2
    root = Path(args[0]).resolve()
    record = finalize(root)
    print(json.dumps({"classification": record.get("classification"), "decision": record.get("decision"), "decision_path": str(root / "decision.json")}, indent=2, sort_keys=True))
    return 0 if record.get("classification") != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
