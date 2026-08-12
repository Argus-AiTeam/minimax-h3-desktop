#!/usr/bin/env python3
"""Strict ARGUS-IR-04 evidence aggregation and delivery manifest tooling.

This tool reads existing local evidence only. It does not benchmark, download,
load models, run Docker, or turn practical Turbo smoke metrics into fidelity
claims. Its strict mode fails on missing or internally inconsistent evidence so
claim boundaries remain explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

BASELINE_REL = "baseline_a6000/baseline_certification.json"
GPU2_BRINGUP_RUN_REL = "turbo_merged/runs/gpu2_turbo_20260809T195558Z"
TURBO_TIMING_LATEST_REL = "turbo_merged/timing_repeats/LATEST_RUN_ID"
TURBO_TIMING_REPEATS_REL = "turbo_merged/timing_repeats"
TURBO_QUALITY_LATEST_REL = "turbo_merged/LATEST_QUALITY_SUITE_RUN_ID"
TURBO_QUALITY_RUNS_REL = "turbo_merged/quality_suite_runs"
DMD_REL = "dmd_primary_source_note.md"
ADALN_QUALITY_REL = "sol_engine_port/r5_ablation_20260809T181515Z/adaln/quality_vs_dense.json"
ADALN_POSTHOC_REL = "sol_engine_port/a6000_adaln_candidate_50step_20260809T190402Z/av_validation_posthoc.json"
ADALN_VERDICT_REL = "sol_engine_port/a6000_adaln_candidate_50step_20260809T190402Z/candidate_verdict.json"
ROPE_QUALITY_REL = "sol_engine_port/r5_ablation_20260809T181515Z/rope/quality_vs_dense.json"
ALL_EXACT_QUALITY_REL = "sol_engine_port/r5_ablation_20260809T181515Z/all_exact/quality_vs_dense.json"
SWIGLU_QUALITY_REL = "sol_engine_port/r5_ablation_20260809T181515Z/swiglu/quality_vs_dense.json"
SWIGLU_HTTP_REL = "sol_engine_port/r5_ablation_20260809T181515Z/swiglu/http_metrics.txt"
SOL_ATTN_RESULT_REL = "sol_engine_port/sol_attn_gpu_20260809T173323Z/result.json"
QUALITY_CONFIG_REL = "turbo_merged/quality_suite_config.json"
QUALITY_DRY_RUN_REL = "turbo_merged/quality_suite_dry_run"
DELIVERY_REL = "delivery"
LOCAL_LIFECYCLE_PREFIX = "local_lifecycle_clean_room_"
R8_SOL_ATTN_CPU_INGEST_PREFIX = "r8_sol_attn_cpu_ingest_"
R8_SOL_ATTN_CLASSIFICATION_FILE = "r8_terminal_classification.json"
R8_MATCHED_RETEST_INSPECTION_PREFIX = "r8_matched_retest_nonterminal_inspection_"
R8_MATCHED_RETEST_INSPECTION_FILE = "r8_matched_retest_nonterminal_inspection.json"
R8_MATCHED_RETEST_TERMINAL_RUN_PREFIX = "sol_attn_h3_matched_retest_r8_n3_"
R8_MATCHED_RETEST_DECISION_FILE = "decision.json"
R8_MATCHED_RETEST_TERMINAL_FILES = (
    "decision.json",
    "RUN_REPORT.md",
    "timing_summary.json",
    "quality_proxy_comparison.json",
    "resource_summary.json",
)
R8_MATCHED_RETEST_TERMINAL_RECHECK_PREFIX = "r8_matched_retest_terminal_recheck_"
R8_MATCHED_RETEST_TERMINAL_RECHECK_FILE = "summary.json"
R8_FORMAL_N10_PREFIX = "sol_attn_h3_formal_n10_r8_n"
R8_FORMAL_N10_DECISION_FILE = "formal_n10_decision.json"
R8_FORMAL_N10_REPORT_FILE = "RUN_REPORT.md"
R8_FORMAL_N10_TERMINAL_FILES = (
    "formal_n10_decision.json",
    "RUN_REPORT.md",
    "formal_n10_summary.json",
    "timing_summary.json",
    "quality_proxy_comparison.json",
    "resource_summary.json",
)
FINAL_CPU_STATIC_GATE_PREFIX = "final_cpu_static_gate_"
FINAL_DECISIVE_EXPORT_AUDIT_PREFIX = "final_decisive_export_audit_"
FORMAL_N10_CPU_SYNC_EXPORT_AUDIT_PREFIX = "formal_n10_cpu_sync_export_audit_"
ACTIVE_HOLD_SYNC_EXPORT_AUDIT_PREFIX = "active_hold_sync_export_audit_"
TURBO_OPERATOR_GATE_LATEST_REL = "delivery/LATEST_TURBO_OPERATOR_GATE_REVIEWER_PACKET"
TURBO_OPERATOR_GATE_PACKET_FILES = (
    "summary.json",
    "media_listening_manifest.json",
    "reviewer_packet.md",
    "reviewer_verdict.json",
    "reviewer_verdict_request.json",
    "manager_reviewer_handoff_crosswalk.json",
    "manager_stage_closeout_crosswalk.json",
    "manager_visibility_resolution.json",
)
DELIVERY_REVIEWER_RECOGNITION_REPAIR_LATEST_REL = "delivery/LATEST_DELIVERY_REVIEWER_EVIDENCE_RECOGNITION_REPAIR_PACKET"
DELIVERY_REVIEWER_RECOGNITION_REPAIR_PACKET_FILES = (
    "INDEX.json",
    "summary.json",
    "schema_gap_analysis.json",
    "manager_stage_authority_probe_*.json",
    "manager_recognition_probe_*.json",
    "manager_recognition_check_*.json",
    "legacy_canonical_chain_recognition_check*.json",
    "manager_reviewer_handoff_crosswalk.json",
    "reviewer_verdict.json",
    "reviewer_verdict_request.json",
)
DELIVERY_REVIEWER_ACTIVE_HOLD_RECONCILIATION_LATEST_REL = "delivery/LATEST_DELIVERY_REVIEWER_EVIDENCE_ACTIVE_HOLD_RECONCILIATION_PACKET"
DELIVERY_REVIEWER_ACTIVE_HOLD_RECONCILIATION_PACKET_FILES = (
    "INDEX.json",
    "summary.json",
    "active_hold_reconciliation_probe.json",
    "reviewer_verdict.json",
)
CURRENT_MANAGER_HOLD_NO_GAP_LATEST_REL = "delivery/LATEST_CURRENT_MANAGER_HOLD_NO_GAP_PROBE"
CURRENT_MANAGER_HOLD_NO_GAP_PACKET_FILES = (
    "INDEX.json",
    "summary.json",
    "current_manager_hold_no_gap_probe.json",
    "reviewer_verdict_request.json",
    "reviewer_verdict.json",
    "manager_reviewer_handoff_crosswalk.json",
)

Fidelity = "fidelity_bf16_exact"
Practical = "practical_disclosed_approx"


class AggregationError(Exception):
    """Raised when strict evidence aggregation cannot accept an artifact."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise AggregationError(f"required evidence file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AggregationError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise AggregationError(f"{path} must contain a JSON object")
    return data


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise AggregationError(f"required evidence file not found: {path}") from exc


def formal_pair_completed(pair_dir: Path) -> bool:
    """Return True when a formal-N pair has terminal per-pair success evidence."""
    if (pair_dir / "decision.json").is_file():
        return True
    exit_path = pair_dir.parent / f"{pair_dir.name}.exit_code"
    try:
        return exit_path.read_text(encoding="utf-8").strip() == "0"
    except FileNotFoundError:
        return False


def _formal_same_expected_gpu(decision: dict[str, Any]) -> bool | None:
    if isinstance(decision.get("same_expected_gpu"), bool):
        return bool(decision["same_expected_gpu"])
    gpu = decision.get("same_baseline_physical_gpu_evidence")
    if isinstance(gpu, dict) and isinstance(gpu.get("same_expected_gpu"), bool):
        return bool(gpu["same_expected_gpu"])
    return None


def _formal_raw_classification(decision: dict[str, Any]) -> str | None:
    value = decision.get("raw_matched_classification") or decision.get("raw_matched_retest_classification")
    return value if isinstance(value, str) else None


def _formal_accepted_gate_failures(decision: dict[str, Any], requested_pairs: int, completed_pairs: int) -> list[str]:
    """Validate both legacy and current formal decision schemas.

    Older report readers expected normalized gate names.  The actual r8 formal
    supervisor preserved the delegated matched-retest gate names in
    ``decision['gates']``.  Treat only derivable facts as accepted gates.
    """
    gates = decision.get("gates") if isinstance(decision.get("gates"), dict) else {}
    checks = {
        "completed_pairs_ge_10": bool(decision.get("formal_pair_count_ok")) or (requested_pairs >= 10 and completed_pairs >= 10),
        "same_expected_gpu": _formal_same_expected_gpu(decision) is True,
        "raw_route_gate_passed": _formal_raw_classification(decision) == "proceed_to_formal_n10_candidate" and not decision.get("failed_gates"),
        "timing_gate_passed": gates.get("timing_gate_passed") is True or (gates.get("median_improvement_exceeds_threshold") is True and gates.get("no_pair_slower") is True),
    }
    return [name for name, ok in checks.items() if not ok]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _assert_close(name: str, actual: float, expected: float, *, tol: float = 1e-6) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=tol, abs_tol=tol):
        raise AggregationError(f"{name} mismatch: actual={actual!r} expected={expected!r}")


def _sample_cv_percent(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / statistics.mean(values) * 100.0


def _require_physical_gpu3(device: dict[str, Any], *, label: str) -> None:
    if device.get("host_gpu_index") != 3:
        raise AggregationError(f"{label} must be from host GPU3")
    if str(device.get("compute_capability")) != "8.6":
        raise AggregationError(f"{label} must be SM86/A6000 evidence")
    if not device.get("uuid"):
        raise AggregationError(f"{label} must include physical GPU UUID")


def _read_latest_run_id(path: Path) -> str:
    run_id = read_text(path).strip()
    if not run_id or "/" in run_id or ".." in run_id:
        raise AggregationError(f"invalid latest-run pointer in {path}: {run_id!r}")
    return run_id


def _latest_prefixed_dir(parent: Path, prefix: str) -> Path | None:
    if not parent.is_dir():
        return None
    candidates = sorted(path for path in parent.glob(f"{prefix}*") if path.is_dir())
    return candidates[-1] if candidates else None


def summarize_baseline(root: Path) -> dict[str, Any]:
    path = root / BASELINE_REL
    data = load_json(path)
    if data.get("schema") != "argus-h3-a6000-fidelity-baseline-certification-v2":
        raise AggregationError(f"baseline schema must be v2: {data.get('schema')!r}")
    if data.get("status") != "certified_internal_same_physical_device_baseline":
        raise AggregationError(f"baseline status is not certified v2: {data.get('status')!r}")
    if data.get("track") != Fidelity:
        raise AggregationError(f"baseline track must be {Fidelity}: {data.get('track')!r}")
    if data.get("platform") != "single_a6000_48gb_workstation":
        raise AggregationError(f"unexpected baseline platform: {data.get('platform')!r}")
    physical_device = data.get("physical_device")
    if not isinstance(physical_device, dict):
        raise AggregationError("baseline must include physical_device")
    _require_physical_gpu3(physical_device, label="baseline")

    runs = data.get("runs")
    if not isinstance(runs, list) or len(runs) != 13:
        raise AggregationError("baseline v2 certification must contain exactly 13 total requests")
    all_latencies = [float(run["latency_s"]) for run in runs]
    warm_latencies = [float(run["latency_s"]) for run in runs if run.get("kind") == "warm"]
    session_first_latencies = [float(run["latency_s"]) for run in runs if run.get("kind") == "session_first"]
    if len(warm_latencies) != 10:
        raise AggregationError(f"baseline warm denominator must be N=10 true warm requests, got {len(warm_latencies)}")
    if len(session_first_latencies) != 3:
        raise AggregationError(f"baseline must preserve three session-first requests, got {len(session_first_latencies)}")
    if not all(run.get("structural_av_pass") is True for run in runs):
        raise AggregationError("baseline v2 requires structural AV pass for every request")

    sessions = data.get("service_sessions", {})
    if not isinstance(sessions, dict) or sessions.get("count") != 3:
        raise AggregationError("baseline v2 must record three service sessions")

    all_stats = data.get("all_requests", {})
    warm_stats = data.get("warm_requests_primary_denominator", {})
    first_stats = data.get("session_first_requests", {})
    if all_stats.get("n") != 13 or warm_stats.get("n") != 10 or first_stats.get("n") != 3:
        raise AggregationError("baseline v2 summary n fields must be all=13, warm=10, session_first=3")
    _assert_close("baseline all mean", all_stats.get("mean_s"), statistics.mean(all_latencies))
    _assert_close("baseline all median", all_stats.get("median_s"), statistics.median(all_latencies))
    _assert_close("baseline warm mean", warm_stats.get("mean_s"), statistics.mean(warm_latencies))
    _assert_close("baseline warm median", warm_stats.get("median_s"), statistics.median(warm_latencies))
    _assert_close("baseline session-first mean", first_stats.get("mean_s"), statistics.mean(session_first_latencies))
    _assert_close("baseline session-first median", first_stats.get("median_s"), statistics.median(session_first_latencies))

    return {
        "schema": data["schema"],
        "status": data["status"],
        "track": data["track"],
        "platform": data["platform"],
        "physical_device": physical_device,
        "evidence_path": rel(path, root),
        "all_requests": {
            "n": all_stats["n"],
            "mean_s": all_stats["mean_s"],
            "median_s": all_stats["median_s"],
            "cv_percent": all_stats.get("cv_percent"),
        },
        "warm_primary_denominator": {
            "n": warm_stats["n"],
            "mean_s": warm_stats["mean_s"],
            "median_s": warm_stats["median_s"],
            "cv_percent": warm_stats.get("cv_percent"),
        },
        "session_first_requests": {
            "n": first_stats["n"],
            "mean_s": first_stats["mean_s"],
            "median_s": first_stats["median_s"],
        },
        "service_session_count": sessions["count"],
        "recomputed_from_run_latencies": True,
        "claim_boundary": data.get("claim_boundary"),
        "correction_note": data.get("correction_note"),
    }


def summarize_turbo_quality(root: Path) -> dict[str, Any]:
    latest_path = root / TURBO_QUALITY_LATEST_REL
    run_id = _read_latest_run_id(latest_path)
    run_dir = root / TURBO_QUALITY_RUNS_REL / run_id
    analysis_path = run_dir / "quality_suite_analysis.json"
    baseline_comparison_path = run_dir / "baseline_seed0_quality_comparison.json"
    audio_energy_path = run_dir / "audio_energy_envelopes.json"
    human_review_path = run_dir / "human_review.md"
    operator_acceptance_path = run_dir / "operator_acceptance.json"
    analysis = load_json(analysis_path)
    baseline_comparison = load_json(baseline_comparison_path)
    audio_energy = json.loads(read_text(audio_energy_path))
    if not isinstance(audio_energy, list):
        raise AggregationError("quality-suite audio energy envelope must be a JSON list")
    human_review = read_text(human_review_path)

    if analysis.get("status") != "structural_av_suite_pass_semantic_quality_not_certified":
        raise AggregationError(f"unexpected quality-suite status: {analysis.get('status')!r}")
    if analysis.get("track") != Practical:
        raise AggregationError("Turbo quality suite must remain in the practical lane")
    if analysis.get("case_count") != 24 or analysis.get("pair_count") != 12:
        raise AggregationError("Turbo quality suite must contain 24 cases / 12 paired cases")
    if analysis.get("all_cases_structural_av_pass") is not True:
        raise AggregationError("Turbo quality suite must pass structural AV for every case")
    cases = analysis.get("cases")
    if not isinstance(cases, list) or len(cases) != 24:
        raise AggregationError("Turbo quality suite cases list must contain 24 entries")
    if not all(case.get("structural_av_contract_pass") is True for case in cases):
        raise AggregationError("Turbo quality suite includes a non-passing structural AV case")
    if len(audio_energy) != 24:
        raise AggregationError("Turbo quality suite audio energy envelope must contain 24 entries")

    latency_by_step = analysis.get("latency_by_step", {})
    steps: dict[str, Any] = {}
    for step in ("4", "8"):
        stats = latency_by_step.get(step)
        if not isinstance(stats, dict) or stats.get("n") != 12:
            raise AggregationError(f"quality-suite step {step} must have N=12")
        comp = baseline_comparison.get(step)
        if not isinstance(comp, dict):
            raise AggregationError(f"baseline seed0 comparison missing step {step}")
        if comp.get("comparison") != "same_physical_gpu3_same_prompt_seed_turbo_vs_50step_bf16":
            raise AggregationError(f"baseline seed0 comparison for step {step} must be same-GPU3 vs 50-step BF16")
        steps[step] = {
            "n": stats["n"],
            "median_s": stats["median_s"],
            "cv_percent": stats.get("cv_percent"),
            "baseline_seed0_audio_cosine": comp.get("audio_cosine"),
            "baseline_seed0_video_mse": comp.get("video_mse"),
            "baseline_seed0_video_psnr_db": comp.get("video_psnr_db"),
            "baseline_seed0_audio_rms_turbo": comp.get("audio_rms_turbo"),
            "baseline_seed0_audio_rms_baseline": comp.get("audio_rms_baseline"),
        }

    contact_sheets = sorted((run_dir / "contacts").glob("REVIEW_*.jpg"))
    if len(contact_sheets) != 6:
        raise AggregationError(f"expected six review contact sheets, got {len(contact_sheets)}")
    operator_acceptance = load_json(operator_acceptance_path) if operator_acceptance_path.is_file() else {}
    operator_accepted = bool(
        operator_acceptance.get("status") == "accepted_overall_by_operator"
        and operator_acceptance.get("scope", {}).get("human_playback_and_listening_review_completed") is True
        and operator_acceptance.get("scope", {}).get("overall_practical_quality_accepted") is True
        and operator_acceptance.get("preserved_limits", {}).get("turbo_is_bf16_exact_fidelity") is False
        and operator_acceptance.get("preserved_limits", {}).get("four_step_promoted_to_default") is False
        and operator_acceptance.get("preserved_limits", {}).get("agent_subjective_listening_performed") is False
    )
    pending_human_review = not operator_accepted
    return {
        "run_id": run_id,
        "run_dir": rel(run_dir, root),
        "latest_pointer": rel(latest_path, root),
        "analysis_path": rel(analysis_path, root),
        "baseline_seed0_comparison_path": rel(baseline_comparison_path, root),
        "audio_energy_envelopes_path": rel(audio_energy_path, root),
        "human_review_path": rel(human_review_path, root),
        "operator_acceptance_path": rel(operator_acceptance_path, root) if operator_acceptance_path.is_file() else None,
        "operator_acceptance": operator_acceptance or None,
        "operator_overall_acceptance_recorded": operator_accepted,
        "review_contact_sheet_count": len(contact_sheets),
        "case_count": analysis["case_count"],
        "pair_count": analysis["pair_count"],
        "steps": steps,
        "all_cases_structural_av_pass": True,
        "quality_certification": (
            "operator_accepted_practical_8step_with_known_4step_visual_failure_preserved"
            if operator_accepted
            else analysis.get("quality_certification")
        ),
        "status": (
            "structural_av_suite_pass_operator_overall_acceptance_recorded"
            if operator_accepted
            else analysis["status"]
        ),
        "pending_human_review": pending_human_review,
        "human_auditory_listening": "operator_overall_playback_listening_accepted" if operator_accepted else "pending",
        "metric_limits": analysis.get("metric_limits", []),
    }


def summarize_turbo(root: Path, baseline: dict[str, Any]) -> dict[str, Any]:
    latest_path = root / TURBO_TIMING_LATEST_REL
    run_id = _read_latest_run_id(latest_path)
    run_dir = root / TURBO_TIMING_REPEATS_REL / run_id
    timing_path = run_dir / "timing_summary.json"
    manifest_path = run_dir / "merge_manifest.json"
    timing = load_json(timing_path)
    manifest = load_json(manifest_path)
    if manifest.get("status") != "completed":
        raise AggregationError(f"Turbo merge manifest is not completed: {manifest.get('status')!r}")
    merge = manifest.get("merge")
    if not isinstance(merge, dict) or merge.get("strength") != 1.0:
        raise AggregationError("Turbo merge manifest must have merge.strength=1.0")
    completed = manifest.get("completed_shards")
    if not isinstance(completed, dict) or len(completed) <= 0:
        raise AggregationError("Turbo merge manifest must list completed shards")

    if timing.get("schema") != "argus-ir04-turbo-paired-timing-v1":
        raise AggregationError(f"unexpected Turbo timing schema: {timing.get('schema')!r}")
    if timing.get("status") != "pass_same_physical_device_paired_n10":
        raise AggregationError(f"unexpected Turbo timing status: {timing.get('status')!r}")
    if timing.get("track") != Practical:
        raise AggregationError("Turbo timing must remain in the practical lane")
    physical_device = timing.get("physical_device")
    if not isinstance(physical_device, dict):
        raise AggregationError("Turbo timing must include physical_device")
    _require_physical_gpu3(physical_device, label="Turbo timing")
    if physical_device.get("uuid") != baseline["physical_device"].get("uuid"):
        raise AggregationError("Turbo timing and baseline must use the same physical GPU UUID")

    denominator = timing.get("baseline_denominator", {})
    if denominator.get("kind") != "warm_requests_primary_denominator" or denominator.get("n") != 10:
        raise AggregationError("Turbo timing denominator must be baseline warm-primary N=10")
    if denominator.get("same_physical_device") is not True:
        raise AggregationError("Turbo timing denominator must be same physical device")
    warm_median = float(baseline["warm_primary_denominator"]["median_s"])
    _assert_close("Turbo timing denominator median", denominator.get("median_s"), warm_median)

    excluded = timing.get("excluded_warmups", {})
    if sorted(excluded) != ["4step", "8step"]:
        raise AggregationError("Turbo timing must disclose one excluded warmup for each schedule")
    pairs = timing.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 10:
        raise AggregationError("Turbo timing must have exactly ten formal paired samples")

    steps: dict[str, Any] = {}
    structural_av_count = 0
    schedules = timing.get("schedules", {})
    for step in ("4", "8"):
        sched = schedules.get(step)
        if not isinstance(sched, dict):
            raise AggregationError(f"Turbo timing missing schedule {step}")
        runs = sched.get("runs")
        if not isinstance(runs, list) or len(runs) != 10 or sched.get("n") != 10:
            raise AggregationError(f"Turbo timing schedule {step} must have paired N=10")
        latencies = [float(run["latency_s"]) for run in runs]
        _assert_close(f"Turbo {step}-step mean", sched.get("mean_s"), statistics.mean(latencies))
        _assert_close(f"Turbo {step}-step median", sched.get("median_s"), statistics.median(latencies))
        _assert_close(f"Turbo {step}-step CV", sched.get("cv_percent"), _sample_cv_percent(latencies))
        _assert_close(
            f"Turbo {step}-step speedup",
            sched.get("speedup_vs_same_gpu3_bf16_warm_n10_median"),
            warm_median / float(sched["median_s"]),
            tol=1e-6,
        )
        for run in runs:
            av_path = root.parent.parent.parent / run["av_validation"]
            av = load_json(av_path)
            if av.get("structural_av_contract_pass") is not True:
                raise AggregationError(f"Turbo timing structural AV failed: {run['av_validation']}")
            structural_av_count += 1
        steps[step] = {
            "n": sched["n"],
            "median_s": sched["median_s"],
            "mean_s": sched["mean_s"],
            "cv_percent": sched["cv_percent"],
            "speedup_vs_same_gpu3_bf16_warm_n10_median": sched["speedup_vs_same_gpu3_bf16_warm_n10_median"],
            "speedup_denominator_s": sched["speedup_denominator_s"],
            "strict_av_pass": True,
        }

    quality_suite = summarize_turbo_quality(root)
    return {
        "track": Practical,
        "run_id": run_id,
        "run_dir": rel(run_dir, root),
        "latest_pointer": rel(latest_path, root),
        "timing_summary_path": rel(timing_path, root),
        "merge_manifest_path": rel(manifest_path, root),
        "merge_status": manifest["status"],
        "merge_strength": merge["strength"],
        "completed_shard_count": len(completed),
        "physical_device": physical_device,
        "baseline_denominator": denominator,
        "excluded_warmups": excluded,
        "paired_formal_n_per_schedule": 10,
        "structural_av_pass_count": structural_av_count,
        "steps": steps,
        "quality_suite": quality_suite,
        "practical_default_candidate": "8-step",
        "ultra_fast_quality_cost_experimental": "4-step",
        "quality_certification": (
            quality_suite["quality_certification"]
            if quality_suite.get("operator_overall_acceptance_recorded")
            else "structural_av_pass_semantic_quality_not_certified_human_listening_pending"
        ),
        "fidelity_claim": "rejected_not_fidelity_lane",
        "gpu2_bringup_scope": "earlier GPU2 smoke is bring-up only and is not used as a speedup denominator/result",
    }


def summarize_dmd(root: Path) -> dict[str, Any]:
    path = root / DMD_REL
    text = read_text(path)
    required = ["BLOCKED / RESEARCH ONLY", "practical track only", "no first-source basis"]
    missing = [needle for needle in required if needle not in text]
    if missing:
        raise AggregationError(f"DMD note is missing required no-go language: {missing}")
    return {
        "status": "blocked_research_only_no_go_after_turbo_unless_feasibility_changes",
        "track_limit": Practical,
        "evidence_path": rel(path, root),
        "rationale": "No official reproducible MiniMax-H3 DMD/DMD2 recipe/checkpoint is present in the evidence note.",
    }


def summarize_r8_formal_n10(root: Path) -> dict[str, Any]:
    formal_dir = _latest_prefixed_dir(root / "sol_engine_port", R8_FORMAL_N10_PREFIX)
    if formal_dir is None:
        return {
            "status": "not_available",
            "reason": "no r8 formal N>=10 Sol-Attn run directory found",
        }

    status_path = formal_dir / "formal_n10_supervisor_status.json"
    stdout_path = formal_dir / "formal_n10_supervisor_stdout.log"
    decision_path = formal_dir / R8_FORMAL_N10_DECISION_FILE
    report_path = formal_dir / R8_FORMAL_N10_REPORT_FILE
    summary_path = formal_dir / "formal_n10_summary.json"
    timing_path = formal_dir / "timing_summary.json"
    quality_path = formal_dir / "quality_proxy_comparison.json"
    resource_path = formal_dir / "resource_summary.json"
    terminal_artifacts = {name: (formal_dir / name).is_file() for name in R8_FORMAL_N10_TERMINAL_FILES}
    pair_dirs = sorted(path for path in formal_dir.glob("pair[0-9][0-9]") if path.is_dir())
    completed_pair_dirs = [path for path in pair_dirs if formal_pair_completed(path)]
    supervisor = load_json(status_path) if status_path.is_file() else {}
    requested_pairs = supervisor.get("n_pairs") or supervisor.get("requested_pairs") or 10

    if decision_path.is_file():
        decision = load_json(decision_path)
        classification = decision.get("formal_classification")
        allowed = {
            "accepted_formal_n10_same_gpu_sol_attn_speed_candidate",
            "rejected_formal_n10_incomplete_pair_count",
            "rejected_formal_n10_correctness_sparse_or_quality_gate_failed",
            "rejected_formal_n10_resource_gate_failed",
            "rejected_formal_n10_timing_gate_failed",
            "rejected_formal_n10_unclassified_gate_failure",
            "inconclusive_incomplete_formal_n10_run",
            "blocked_requested_pairs_below_formal_n10",
        }
        if classification not in allowed:
            raise AggregationError(f"unexpected formal N10 classification: {classification!r}")
        gates = decision.get("gates") if isinstance(decision.get("gates"), dict) else {}
        requested = int(gates.get("requested_pairs") or requested_pairs or decision.get("requested_pairs") or 0)
        completed = int(gates.get("completed_pairs") or decision.get("completed_pairs") or 0)
        same_expected_gpu = _formal_same_expected_gpu(decision)
        raw_classification = _formal_raw_classification(decision)
        if classification == "accepted_formal_n10_same_gpu_sol_attn_speed_candidate":
            missing_terminal = [name for name, present in terminal_artifacts.items() if not present]
            if missing_terminal:
                raise AggregationError(f"accepted formal N10 decision is missing terminal artifacts: {missing_terminal}")
            failed = _formal_accepted_gate_failures(decision, requested, completed)
            if failed:
                raise AggregationError(f"accepted formal N10 decision has failed gates: {failed}")
        return {
            "status": classification,
            "source_run_dir": rel(formal_dir, root),
            "decision_path": rel(decision_path, root),
            "report_path": rel(report_path, root) if report_path.is_file() else None,
            "summary_path": rel(summary_path, root) if summary_path.is_file() else None,
            "timing_summary_path": rel(timing_path, root) if timing_path.is_file() else None,
            "quality_proxy_comparison_path": rel(quality_path, root) if quality_path.is_file() else None,
            "resource_summary_path": rel(resource_path, root) if resource_path.is_file() else None,
            "reason": decision.get("reason"),
            "requested_pairs": requested,
            "completed_pairs": completed,
            "started_pairs": len(pair_dirs),
            "supervisor_status": supervisor.get("status"),
            "supervisor_return_code": supervisor.get("return_code"),
            "same_expected_gpu": same_expected_gpu,
            "raw_matched_classification": raw_classification,
            "median_http_time_improvement_pct": decision.get("median_http_time_improvement_pct"),
            "timing_threshold_pct": decision.get("timing_threshold_pct"),
            "failed_gates": decision.get("failed_gates"),
            "lane": decision.get("lane"),
            "terminal_artifacts_present": terminal_artifacts,
        }

    status = "incomplete_formal_n10_no_terminal_decision"
    if supervisor.get("status") == "running":
        status = "incomplete_formal_n10_running_no_terminal_decision"
    return {
        "status": status,
        "source_run_dir": rel(formal_dir, root),
        "status_path": rel(status_path, root) if status_path.is_file() else None,
        "stdout_path": rel(stdout_path, root) if stdout_path.is_file() else None,
        "reason": (
            "formal N>=10 supervisor is marked running and has no formal_n10_decision.json/RUN_REPORT terminal artifacts; "
            "the run is not accepted, rejected, or a speedup claim until terminal per-pair evidence and summary are present"
            if supervisor.get("status") == "running"
            else "formal N>=10 run directory lacks a terminal formal_n10_decision.json/RUN_REPORT; do not promote or claim speedup"
        ),
        "requested_pairs": requested_pairs,
        "started_pairs": len(pair_dirs),
        "completed_pairs": len(completed_pair_dirs),
        "supervisor_status": supervisor.get("status"),
        "supervisor_pid": supervisor.get("pid"),
        "gpu_index": supervisor.get("gpu_index"),
        "expected_uuid": supervisor.get("expected_uuid"),
        "terminal_artifacts_present": terminal_artifacts,
        "lane": "formal_n10_matched_5step_sol_attn_opt_in_not_bf16_fidelity",
    }


def summarize_r8_h3_sol_attn(root: Path) -> dict[str, Any]:
    delivery_dir = root / DELIVERY_REL
    ingest_dir = _latest_prefixed_dir(delivery_dir, R8_SOL_ATTN_CPU_INGEST_PREFIX)
    if ingest_dir is None:
        return {
            "status": "not_available",
            "claim_boundary": "no r8 H3 Sol-Attn CPU terminal ingest evidence found in this evidence root",
        }

    classification_path = ingest_dir / R8_SOL_ATTN_CLASSIFICATION_FILE
    data = load_json(classification_path)
    classification = data.get("classification")
    if classification != "sparse_runtime_valid_5step_diagnostic":
        raise AggregationError(f"unexpected r8 H3 Sol-Attn classification: {classification!r}")
    if data.get("accepted_metadata") is not True or data.get("accepted_runtime_evidence") is not True:
        raise AggregationError("r8 H3 Sol-Attn ingest must accept both metadata and runtime evidence")
    if data.get("release_manifest_eligible") is not False:
        raise AggregationError("r8 H3 Sol-Attn 5-step diagnostic must not be release-manifest eligible")
    if data.get("not_fidelity_or_performance_claim") is not True:
        raise AggregationError("r8 H3 Sol-Attn diagnostic must reject fidelity/performance-claim status")

    telemetry = data.get("telemetry")
    if not isinstance(telemetry, dict):
        raise AggregationError("r8 H3 Sol-Attn ingest missing telemetry object")
    sparse_candidates = int(telemetry.get("sparse_candidate_calls") or 0)
    sparse_calls = int(telemetry.get("sparse_calls") or 0)
    fallback_calls = int(telemetry.get("fallback_calls") or 0)
    density_samples = int(telemetry.get("density_sample_count") or 0)
    materialized_calls = int(telemetry.get("materialized_copy_calls") or 0)
    materialized_bytes = int(telemetry.get("materialized_copy_bytes") or 0)
    if sparse_candidates <= 0 or sparse_calls <= 0:
        raise AggregationError("r8 H3 Sol-Attn diagnostic must have sparse_candidate_calls>0 and sparse_calls>0")
    if fallback_calls != 0:
        raise AggregationError("r8 H3 Sol-Attn accepted diagnostic must not rely on fallback calls")
    if density_samples <= 0:
        raise AggregationError("r8 H3 Sol-Attn diagnostic must include density samples")
    if materialized_calls <= 0 or materialized_bytes <= 0:
        raise AggregationError("r8 H3 Sol-Attn diagnostic must record materialized-copy calls and bytes")

    for label in ("dense_h3_backend_reference", "sol_attn_opt_in"):
        branch = data.get(label)
        if not isinstance(branch, dict):
            raise AggregationError(f"r8 H3 Sol-Attn ingest missing {label}")
        av = branch.get("av")
        http = branch.get("http")
        if not isinstance(av, dict) or av.get("structural_av_contract_pass") is not True:
            raise AggregationError(f"r8 H3 Sol-Attn {label} structural AV did not pass")
        if not isinstance(http, dict) or http.get("status") != "present":
            raise AggregationError(f"r8 H3 Sol-Attn {label} HTTP metrics missing")

    readable = data.get("readable_provenance")
    if not isinstance(readable, dict):
        raise AggregationError("r8 H3 Sol-Attn ingest missing readable provenance")
    if readable.get("image_version_label") != "r8" or readable.get("required_image_version_label") != "r8":
        raise AggregationError("r8 H3 Sol-Attn ingest must be attributed by readable r8 version labels")
    if readable.get("workload_attention_backend") != "H3_A6000_SOL_ATTN":
        raise AggregationError("r8 H3 Sol-Attn workload must use H3_A6000_SOL_ATTN")

    resources = data.get("resource_summary")
    if not isinstance(resources, dict) or resources.get("status") != "present":
        raise AggregationError("r8 H3 Sol-Attn ingest missing resource summary")

    dense_http = data.get("dense_h3_backend_reference", {}).get("http", {})
    opt_http = data.get("sol_attn_opt_in", {}).get("http", {})
    dense_av = data.get("dense_h3_backend_reference", {}).get("av", {})
    opt_av = data.get("sol_attn_opt_in", {}).get("av", {})

    result = {
        "status": classification,
        "classification": classification,
        "evidence_path": rel(classification_path, root),
        "ingest_dir": rel(ingest_dir, root),
        "selected_run_dir": data.get("selected_run_dir"),
        "final_pass_fail": data.get("final_pass_fail"),
        "reason": data.get("reason"),
        "release_manifest_eligible": data.get("release_manifest_eligible"),
        "claim_boundary": "accepted 5-step sparse-execution metadata-plumbing diagnostic only; not a speedup, N10, BF16 fidelity, release, or quality-equivalence claim",
        "readable_provenance": {
            "image_tag": readable.get("workload_image"),
            "image_version_label": readable.get("image_version_label"),
            "runtime_label": readable.get("runtime_label"),
            "workload_attention_backend": readable.get("workload_attention_backend"),
        },
        "http_time_total_s": {
            "dense_h3_backend_reference": dense_http.get("time_total_s"),
            "sol_attn_opt_in": opt_http.get("time_total_s"),
        },
        "paired_http_ratio_dense_over_opt_in_not_speedup": data.get("paired_http_ratio_dense_over_opt_in_not_speedup"),
        "telemetry": {
            "sparse_candidate_calls": sparse_candidates,
            "sparse_calls": sparse_calls,
            "dense_calls": int(telemetry.get("dense_calls") or 0),
            "fallback_calls": fallback_calls,
            "decline_reasons": telemetry.get("decline_reasons") or {},
            "density_sample_count": density_samples,
            "materialized_copy_calls": materialized_calls,
            "materialized_copy_bytes": materialized_bytes,
        },
        "resource_summary": {
            "peak_gpu_memory_mib": resources.get("peak_gpu_memory_mib"),
            "peak_temperature_c": resources.get("peak_temperature_c"),
            "peak_power_w": resources.get("peak_power_w"),
        },
        "av": {
            "dense_structural_pass": dense_av.get("structural_av_contract_pass"),
            "opt_in_structural_pass": opt_av.get("structural_av_contract_pass"),
            "dense_video_frames": dense_av.get("decoded_video_frames"),
            "opt_in_video_frames": opt_av.get("decoded_video_frames"),
            "dense_audio_sample_rate_hz": dense_av.get("audio_sample_rate_hz"),
            "opt_in_audio_sample_rate_hz": opt_av.get("audio_sample_rate_hz"),
        },
        "matched_retest": {
            "status": "not_available",
            "reason": "no r8 matched-workload nonterminal/terminal route-decision evidence was found by the aggregator",
        },
        "formal_n10": summarize_r8_formal_n10(root),
    }

    terminal_run_dir = _latest_prefixed_dir(root / "sol_engine_port", R8_MATCHED_RETEST_TERMINAL_RUN_PREFIX)
    terminal_decision_path = terminal_run_dir / R8_MATCHED_RETEST_DECISION_FILE if terminal_run_dir is not None else None
    if terminal_decision_path is not None and terminal_decision_path.is_file():
        decision = load_json(terminal_decision_path)
        classification = decision.get("classification")
        allowed_terminal = {
            "proceed_to_formal_n10_candidate",
            "diagnostic_only_rejected_no_n10_timing_gate",
            "diagnostic_only_rejected_resource_envelope",
            "needs_fix_incomplete_matched_retest",
            "needs_fix_invalid_http_or_structural_av",
            "needs_fix_sparse_runtime_or_telemetry_gate_failed",
            "needs_fix_quality_proxy_red_flags",
        }
        if classification not in allowed_terminal:
            raise AggregationError(f"unexpected r8 matched retest terminal classification: {classification!r}")
        if decision.get("not_fidelity_or_performance_claim") is not True or decision.get("not_formal_n10") is not True:
            raise AggregationError("r8 matched retest terminal decision must reject fidelity/performance/formal-N10 claim status")
        if decision.get("lane") != "diagnostic_practical_opt_in_sol_attn_not_bf16_fidelity":
            raise AggregationError("r8 matched retest terminal decision must remain in the diagnostic practical opt-in lane")
        gates = decision.get("gates")
        if not isinstance(gates, dict):
            raise AggregationError("r8 matched retest terminal decision missing gates object")
        if classification == "proceed_to_formal_n10_candidate":
            required_true = [
                "all_pairs_completed",
                "all_http_200",
                "all_structural_av_valid",
                "all_sparse_calls_positive",
                "all_fallback_calls_zero",
                "complete_density_and_materialization_telemetry",
                "resource_envelope_comparable_to_prior_r8",
                "no_quality_proxy_red_flags",
                "no_pair_slower",
                "median_improvement_exceeds_threshold",
            ]
            missing_true = [name for name in required_true if gates.get(name) is not True]
            if missing_true:
                raise AggregationError(f"r8 matched retest proceed classification has failed gates: {missing_true}")
            if decision.get("proceed_to_n10_recommended") is not True or decision.get("failed_gates") not in ([], None):
                raise AggregationError("r8 matched retest proceed classification must recommend N10 with no failed gates")
        terminal_artifacts = {name: (terminal_run_dir / name).is_file() for name in R8_MATCHED_RETEST_TERMINAL_FILES}
        if not all(terminal_artifacts.values()):
            raise AggregationError(f"r8 matched retest terminal artifact set incomplete: {terminal_artifacts}")
        supervisor_status_path = terminal_run_dir / "supervisor_status.json"
        supervisor_status = load_json(supervisor_status_path) if supervisor_status_path.is_file() else {}
        posthoc_note_path = terminal_run_dir / "posthoc_finalization_note.json"
        terminal_recheck_dir = _latest_prefixed_dir(delivery_dir, R8_MATCHED_RETEST_TERMINAL_RECHECK_PREFIX)
        terminal_recheck_path = terminal_recheck_dir / R8_MATCHED_RETEST_TERMINAL_RECHECK_FILE if terminal_recheck_dir is not None else None
        if terminal_recheck_path is not None and terminal_recheck_path.is_file():
            terminal_recheck = load_json(terminal_recheck_path)
            if terminal_recheck.get("classification") != classification:
                raise AggregationError("latest r8 matched terminal recheck classification disagrees with decision.json")
        result["matched_retest"] = {
            "status": classification,
            "evidence_path": rel(terminal_decision_path, root),
            "source_run_dir": rel(terminal_run_dir, root),
            "supervisor_status": supervisor_status.get("status"),
            "supervisor_return_code": supervisor_status.get("return_code"),
            "supervisor_pid_alive": False,
            "posthoc_finalization_note": rel(posthoc_note_path, root) if posthoc_note_path.is_file() else None,
            "terminal_recheck_evidence_path": rel(terminal_recheck_path, root) if terminal_recheck_path is not None and terminal_recheck_path.is_file() else None,
            "terminal_artifacts_present": terminal_artifacts,
            "requested_pairs": gates.get("requested_pairs"),
            "completed_pairs": gates.get("completed_pairs"),
            "median_http_time_improvement_pct": decision.get("median_http_time_improvement_pct"),
            "timing_threshold_pct": decision.get("timing_threshold_pct"),
            "n10_recommendation": "proceed_to_formal_n10_candidate" if decision.get("proceed_to_n10_recommended") is True else "do_not_promote_without_new_evidence",
            "reason": decision.get("reason"),
        }
    else:
        matched_dir = _latest_prefixed_dir(delivery_dir, R8_MATCHED_RETEST_INSPECTION_PREFIX)
        if matched_dir is not None:
            matched_path = matched_dir / R8_MATCHED_RETEST_INSPECTION_FILE
            matched = load_json(matched_path)
            if matched.get("not_fidelity_or_speedup_claim") is not True:
                raise AggregationError("r8 matched retest inspection must reject fidelity/speedup-claim status")
            if matched.get("classification") != "pending_nonterminal_do_not_promote_or_stop":
                raise AggregationError(f"unexpected r8 matched retest nonterminal classification: {matched.get('classification')!r}")
            result["matched_retest"] = {
                "status": matched.get("classification"),
                "evidence_path": rel(matched_path, root),
                "source_run_dir": matched.get("source_run_dir"),
                "supervisor_status": matched.get("supervisor_status", {}).get("status"),
                "supervisor_pid_alive": matched.get("supervisor_pid_liveness", {}).get("alive"),
                "terminal_artifacts_present": matched.get("terminal_artifacts_present"),
                "n10_recommendation": matched.get("n10_recommendation"),
                "reason": matched.get("reason"),
            }
    return result


def summarize_sol(root: Path, baseline: dict[str, Any]) -> dict[str, Any]:
    adaln_quality_path = root / ADALN_QUALITY_REL
    adaln_posthoc_path = root / ADALN_POSTHOC_REL
    adaln_verdict_path = root / ADALN_VERDICT_REL
    rope_quality_path = root / ROPE_QUALITY_REL
    all_exact_quality_path = root / ALL_EXACT_QUALITY_REL
    swiglu_quality_path = root / SWIGLU_QUALITY_REL
    swiglu_http_path = root / SWIGLU_HTTP_REL
    sol_attn_result_path = root / SOL_ATTN_RESULT_REL

    adaln_quality = load_json(adaln_quality_path)
    adaln_posthoc = load_json(adaln_posthoc_path)
    adaln_verdict = load_json(adaln_verdict_path)
    rope_quality = load_json(rope_quality_path)
    all_exact_quality = load_json(all_exact_quality_path)
    swiglu_quality = load_json(swiglu_quality_path)
    swiglu_http = read_text(swiglu_http_path)
    sol_attn = load_json(sol_attn_result_path)

    if adaln_quality.get("mode") != "adaln" or adaln_quality.get("claim_scope") != "diagnostic_ablation_only_not_fidelity_acceptance":
        raise AggregationError("AdaLN quality evidence must remain diagnostic-only")
    _assert_close("AdaLN video MSE", adaln_quality.get("video_mean_mse"), 0.0)
    _assert_close("AdaLN audio cosine", adaln_quality.get("audio_waveform_cosine"), 1.0, tol=1e-12)
    if adaln_posthoc.get("claim") != "adaln_fidelity_candidate_single_run_not_N10":
        raise AggregationError("AdaLN posthoc evidence must disclose single-run/not-N10 scope")
    if "harness_failure" not in adaln_posthoc:
        raise AggregationError("AdaLN posthoc evidence must preserve harness-tail-failure disclosure")
    if adaln_verdict.get("bitwise_container_equal_to_dense_run1") is not True or adaln_verdict.get("status") != "retained_single_candidate_pending_repeats":
        raise AggregationError("AdaLN candidate verdict must remain N=1/pending repeats")
    adaln_single_run_speedup = float(adaln_verdict.get("single_run_speedup"))
    baseline_warm_cv = float(baseline["warm_primary_denominator"].get("cv_percent") or 0.0)
    if (adaln_single_run_speedup - 1.0) * 100.0 >= baseline_warm_cv:
        raise AggregationError("AdaLN single-run benefit is no longer below baseline warm CV; revisit N=10 acceptance")

    for label, quality in (("rope", rope_quality), ("all_exact", all_exact_quality)):
        if float(quality.get("video_mean_mse", 0.0)) <= 0.0:
            raise AggregationError(f"{label} rejection evidence expected non-zero video drift")
        if float(quality.get("audio_waveform_cosine", 1.0)) >= 0.999:
            raise AggregationError(f"{label} rejection evidence expected audio drift")

    if swiglu_quality.get("mode") != "swiglu" or swiglu_quality.get("claim_scope") != "diagnostic_ablation_only_not_fidelity_acceptance":
        raise AggregationError("SwiGLU quality evidence must remain diagnostic-only")
    _assert_close("SwiGLU video MSE", swiglu_quality.get("video_mean_mse"), 0.0)
    _assert_close("SwiGLU audio cosine", swiglu_quality.get("audio_waveform_cosine"), 1.0, tol=1e-12)
    if "time_total_s=" not in swiglu_http:
        raise AggregationError("SwiGLU HTTP timing evidence is missing")

    bench = sol_attn.get("bench")
    if not isinstance(bench, dict):
        raise AggregationError("Sol-Attn toy result missing bench object")
    speedup = float(bench.get("speedup_dense_over_sparse_median"))
    if speedup >= 1.0:
        raise AggregationError("toy Sol-Attn must not be marked faster/deployed from current evidence")

    return {
        "adaln": {
            "status": "exact_output_candidate_n1_rejected_for_speedup_acceptance_not_deployed",
            "quality_path": rel(adaln_quality_path, root),
            "posthoc_path": rel(adaln_posthoc_path, root),
            "verdict_path": rel(adaln_verdict_path, root),
            "video_mean_mse": adaln_quality["video_mean_mse"],
            "audio_waveform_cosine": adaln_quality["audio_waveform_cosine"],
            "harness_failure_disclosure": adaln_posthoc["harness_failure"],
            "original_harness_exit_code": adaln_posthoc.get("original_harness_exit_code"),
            "single_run_speedup": adaln_single_run_speedup,
            "baseline_warm_cv_percent": baseline_warm_cv,
            "n": 1,
        },
        "rope": {
            "status": "rejected_for_output_drift_not_accepted_exact_kernel",
            "quality_path": rel(rope_quality_path, root),
            "video_mean_mse": rope_quality["video_mean_mse"],
            "audio_waveform_cosine": rope_quality["audio_waveform_cosine"],
        },
        "all_exact": {
            "status": "rejected_for_output_drift_not_accepted_exact_kernel",
            "quality_path": rel(all_exact_quality_path, root),
            "video_mean_mse": all_exact_quality["video_mean_mse"],
            "audio_waveform_cosine": all_exact_quality["audio_waveform_cosine"],
        },
        "swiglu": {
            "status": "rejected_no_retained_speedup_gain_not_deployed",
            "quality_path": rel(swiglu_quality_path, root),
            "http_metrics_path": rel(swiglu_http_path, root),
            "video_mean_mse": swiglu_quality["video_mean_mse"],
            "audio_waveform_cosine": swiglu_quality["audio_waveform_cosine"],
        },
        "toy_sol_attn": {
            "status": "rejected_slower_not_deployed_kernel_candidate_only",
            "result_path": rel(sol_attn_result_path, root),
            "model_load": sol_attn.get("model_load"),
            "kernel_candidates_only_not_h3_e2e": bench.get("kernel_candidates_only_not_h3_e2e"),
            "speedup_dense_over_sparse_median": speedup,
            "dense_median_ms": bench.get("dense_ms", {}).get("median_ms"),
            "sparse_median_ms": bench.get("sparse_ms", {}).get("median_ms"),
        },
        "h3_sol_attn_r8": summarize_r8_h3_sol_attn(root),
    }


def summarize_final_gates(root: Path) -> dict[str, Any]:
    delivery_dir = root / DELIVERY_REL
    cpu_dir = _latest_prefixed_dir(delivery_dir, FINAL_CPU_STATIC_GATE_PREFIX)
    decisive_dir = _latest_prefixed_dir(delivery_dir, FINAL_DECISIVE_EXPORT_AUDIT_PREFIX)
    sync_dir = _latest_prefixed_dir(delivery_dir, FORMAL_N10_CPU_SYNC_EXPORT_AUDIT_PREFIX)
    active_hold_sync_dir = _latest_prefixed_dir(delivery_dir, ACTIVE_HOLD_SYNC_EXPORT_AUDIT_PREFIX)

    cpu_status = "not_available"
    cpu_summary = ""
    if cpu_dir is not None:
        summary_path = cpu_dir / "summary.txt"
        cpu_summary = read_text(summary_path) if summary_path.is_file() else ""
        cpu_status = "pass" if "status=pass" in cpu_summary else "present_but_not_pass"

    decisive_status = "not_available"
    decisive_summary: dict[str, Any] = {}
    if decisive_dir is not None:
        summary_path = decisive_dir / "summary.json"
        decisive_summary = load_json(summary_path) if summary_path.is_file() else {}
        decisive_status = decisive_summary.get("status") or "present_but_not_pass"

    sync_status = "not_available"
    sync_summary: dict[str, Any] = {}
    if sync_dir is not None:
        summary_path = sync_dir / "summary.json"
        sync_summary = load_json(summary_path) if summary_path.is_file() else {}
        sync_status = sync_summary.get("status") or "present_but_not_pass"

    active_hold_sync_status = "not_available"
    active_hold_sync_summary: dict[str, Any] = {}
    if active_hold_sync_dir is not None:
        summary_path = active_hold_sync_dir / "summary.json"
        active_hold_sync_summary = load_json(summary_path) if summary_path.is_file() else {}
        active_hold_sync_status = active_hold_sync_summary.get("status") or "present_but_not_pass"

    overall = (
        "pass"
        if cpu_status == "pass"
        and decisive_status == "pass"
        and sync_status in {"not_available", "pass"}
        and active_hold_sync_status in {"not_available", "pass"}
        else "pending_or_failed"
    )
    return {
        "status": overall,
        "cpu_static_gate": {
            "status": cpu_status,
            "dir": rel(cpu_dir, root) if cpu_dir is not None else None,
            "summary_path": rel(cpu_dir / "summary.txt", root) if cpu_dir is not None else None,
            "summary_tail": cpu_summary.strip().splitlines()[-4:] if cpu_summary else [],
        },
        "decisive_export_audit_gate": {
            "status": decisive_status,
            "dir": rel(decisive_dir, root) if decisive_dir is not None else None,
            "summary_path": rel(decisive_dir / "summary.json", root) if decisive_dir is not None else None,
            "export_file_count": decisive_summary.get("export_file_count"),
            "publication_audit_status": decisive_summary.get("publication_audit_status"),
            "publication_issue_count": decisive_summary.get("publication_issue_count"),
        },
        "formal_n10_cpu_sync_export_audit_gate": {
            "status": sync_status,
            "dir": rel(sync_dir, root) if sync_dir is not None else None,
            "summary_path": rel(sync_dir / "summary.json", root) if sync_dir is not None else None,
            "export_file_count": sync_summary.get("export_file_count"),
            "publication_audit_status": sync_summary.get("publication_audit_status"),
            "publication_issue_count": sync_summary.get("publication_issue_count"),
            "push_performed": sync_summary.get("push_performed"),
            "reviewer_status": sync_summary.get("reviewer_status"),
        },
        "active_hold_sync_export_audit_gate": {
            "status": active_hold_sync_status,
            "dir": rel(active_hold_sync_dir, root) if active_hold_sync_dir is not None else None,
            "summary_path": rel(active_hold_sync_dir / "summary.json", root) if active_hold_sync_dir is not None else None,
            "strict_aggregation_status": active_hold_sync_summary.get("strict_aggregation_status"),
            "export_status": active_hold_sync_summary.get("export_status"),
            "export_file_count": active_hold_sync_summary.get("export_file_count"),
            "publication_audit_status": active_hold_sync_summary.get("publication_audit_status"),
            "publication_issue_count": active_hold_sync_summary.get("publication_issue_count"),
            "active_hold_reconciliation_packet": active_hold_sync_summary.get("active_hold_reconciliation_packet"),
            "active_hold_reconciliation_reviewer_verdict": active_hold_sync_summary.get("active_hold_reconciliation_reviewer_verdict"),
        },
        "claim_boundary": "CPU/static/export/audit gates only; no GPU, Docker-run, model-load, speed, fidelity, or quality claim is created by these gates.",
    }


def _resolve_latest_packet_dir(root: Path, repo_root: Path, latest_rel: str) -> Path | None:
    latest_path = root / latest_rel
    if not latest_path.is_file():
        return None
    pointer = read_text(latest_path).strip()
    if not pointer:
        raise AggregationError(f"latest packet selector is empty: {latest_path}")
    raw = Path(pointer)
    candidates = [raw] if raw.is_absolute() else [repo_root / raw, root / raw, latest_path.parent / raw]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise AggregationError(f"latest packet selector {latest_path} points to missing directory: {pointer}")


def _indexed_packet_path(packet_dir: Path, repo_root: Path, index: dict[str, Any], key: str) -> Path | None:
    value = index.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [repo_root / raw, packet_dir / raw.name, packet_dir / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _recursive_key_count(value: Any, key: str) -> int:
    if isinstance(value, dict):
        return sum((1 if item_key == key else 0) + _recursive_key_count(item_value, key) for item_key, item_value in value.items())
    if isinstance(value, list):
        return sum(_recursive_key_count(item, key) for item in value)
    return 0


def summarize_delivery_reviewer_evidence_repair(root: Path, *, repo_root: Path) -> dict[str, Any]:
    """Summarize the current-stage delivery Reviewer-recognition repair packet.

    This is separate from the older Turbo operator-gate packet: it exposes the
    fresh current_stage=delivery Reviewer verdict and sealed Reviewer handoff
    that a Manager checker may require without reusing the ambiguous Turbo
    closeout chain as the current repair verdict.
    """

    latest_path = root / DELIVERY_REVIEWER_RECOGNITION_REPAIR_LATEST_REL
    packet_dir = _resolve_latest_packet_dir(root, repo_root, DELIVERY_REVIEWER_RECOGNITION_REPAIR_LATEST_REL)
    if packet_dir is None:
        return {
            "status": "not_available",
            "latest_pointer": repo_rel(latest_path, repo_root),
            "claim_boundary": "No current-stage delivery Reviewer-evidence recognition repair packet selector is present in this evidence root.",
        }

    index_path = packet_dir / "INDEX.json"
    index = load_json(index_path)
    unresolved_reviewer_blocker = index.get("exact_blocker_for_reviewer")
    if isinstance(unresolved_reviewer_blocker, str) and unresolved_reviewer_blocker.strip():
        raise AggregationError(
            "delivery Reviewer repair INDEX still advertises unresolved exact_blocker_for_reviewer: "
            f"{repo_rel(index_path, repo_root)}"
        )
    summary_path = _indexed_packet_path(packet_dir, repo_root, index, "summary") or packet_dir / "summary.json"
    manager_stage_probe_path = _indexed_packet_path(packet_dir, repo_root, index, "manager_stage_authority_probe")
    if manager_stage_probe_path is None:
        manager_stage_probe_paths = sorted(packet_dir.glob("manager_stage_authority_probe_*.json"))
        manager_stage_probe_path = manager_stage_probe_paths[-1] if manager_stage_probe_paths else None
    manager_stage_probe = load_json(manager_stage_probe_path) if manager_stage_probe_path is not None else {}
    schema_gap_path = _indexed_packet_path(packet_dir, repo_root, index, "schema_gap_analysis") or packet_dir / "schema_gap_analysis.json"
    schema_gap = load_json(schema_gap_path) if schema_gap_path.is_file() else {}
    verdict_path = _indexed_packet_path(packet_dir, repo_root, index, "reviewer_verdict") or packet_dir / "reviewer_verdict.json"
    verdict = load_json(verdict_path) if verdict_path.is_file() else {}
    request_path = _indexed_packet_path(packet_dir, repo_root, index, "reviewer_verdict_request") or packet_dir / "reviewer_verdict_request.json"
    crosswalk_path = _indexed_packet_path(packet_dir, repo_root, index, "manager_reviewer_handoff_crosswalk") or packet_dir / "manager_reviewer_handoff_crosswalk.json"
    crosswalk = load_json(crosswalk_path) if crosswalk_path.is_file() else {}
    recognition_check_path = _indexed_packet_path(packet_dir, repo_root, index, "manager_recognition_check")
    if recognition_check_path is None:
        recognition_check_paths = sorted(packet_dir.glob("manager_recognition_check*.json"))
        recognition_check_path = recognition_check_paths[-1] if recognition_check_paths else None
    recognition_check = load_json(recognition_check_path) if recognition_check_path is not None else {}
    legacy_check_path = _indexed_packet_path(packet_dir, repo_root, index, "legacy_canonical_chain_recognition_check")
    if legacy_check_path is None:
        legacy_check_paths = sorted(packet_dir.glob("legacy_canonical_chain_recognition_check*.json"))
        legacy_check_path = legacy_check_paths[-1] if legacy_check_paths else None
    legacy_check = load_json(legacy_check_path) if legacy_check_path is not None else {}

    boundary = verdict.get("boundary_covered") if isinstance(verdict.get("boundary_covered"), dict) else {}
    index_boundary = index.get("operator_only_boundary") if isinstance(index.get("operator_only_boundary"), dict) else {}
    schema_boundary = schema_gap.get("operator_only_boundary_preserved") if isinstance(schema_gap.get("operator_only_boundary_preserved"), dict) else {}
    operator_only_gate = (
        index_boundary.get("only_remaining_operator_gate")
        or schema_boundary.get("only_remaining_operator_gate")
        or "operator_human_auditory_listening_and_semantic_av_sync_review_for_Turbo_quality_certification"
    )
    agent_subjective_listening = bool(boundary.get("agent_subjective_listening_performed") or index_boundary.get("agent_subjective_listening_performed") or schema_boundary.get("agent_subjective_listening_performed"))
    semantic_certified = bool(boundary.get("semantic_av_quality_certified") or index_boundary.get("semantic_av_quality_certified") or schema_boundary.get("semantic_av_quality_certified"))
    av_sync_certified = bool(boundary.get("av_sync_certified") or index_boundary.get("av_sync_certified") or schema_boundary.get("av_sync_certified"))
    if agent_subjective_listening:
        raise AggregationError(f"delivery Reviewer repair packet claims agent subjective listening: {verdict_path}")
    if semantic_certified or av_sync_certified:
        raise AggregationError(f"delivery Reviewer repair packet certifies subjective semantic AV/audio quality: {verdict_path}")

    handoff_source: dict[str, Any] = {}
    for source in (
        verdict.get("sealed_reviewer_handoff_source"),
        verdict.get("sealed_reviewer_handoff_reference"),
        crosswalk.get("fresh_reviewer_handoff_source"),
        crosswalk.get("expected_fresh_reviewer_handoff_source"),
        index.get("sealed_reviewer_handoff_reference"),
    ):
        if isinstance(source, dict) and source.get("path"):
            handoff_source = source
            break
    handoff_valid = bool(
        handoff_source.get("path")
        and handoff_source.get("kind") == "round_reviewed_handoff"
        and handoff_source.get("producer_role") == "reviewer"
        and handoff_source.get("review_status") == "done"
        and handoff_source.get("current_stage") == "delivery"
    )
    verdict_status = verdict.get("status") or index.get("status") or "not_available"
    verdict_independent = bool(verdict.get("verdict_is_independent") is True or handoff_valid)
    verdict_accepted = bool(
        verdict_path.is_file()
        and verdict.get("current_stage") == "delivery"
        and verdict.get("decision") == "done"
        and verdict_status == "accepted_current_stage_delivery_reviewer_passed"
        and verdict_independent
    )
    recognition_ready = bool(
        recognition_check.get("status") == "pass"
        and recognition_check.get("ready_for_manager_recognition") is True
        and recognition_check.get("mismatch_count") == 0
    )
    crosswalk_ready = bool(
        crosswalk.get("status") == "pass"
        and (crosswalk.get("manager_recognition_fields_after_reviewer") or {}).get("current_repair_packet_has_fresh_independent_reviewer_verdict") is True
    )
    schema_gap_ready = schema_gap.get("status") in {
        "gap_identified_and_reviewer_evidence_repaired",
        "pass",
    }
    legacy_ready = bool(
        legacy_check.get("status") == "pass"
        and legacy_check.get("ready_for_manager_recognition") is True
        and legacy_check.get("mismatch_count") == 0
    )
    ready = bool(verdict_accepted and handoff_valid and recognition_ready and crosswalk_ready and schema_gap_ready and not agent_subjective_listening and not semantic_certified and not av_sync_certified)

    return {
        "status": index.get("status", "present"),
        "packet_dir": repo_rel(packet_dir, repo_root),
        "latest_pointer": repo_rel(latest_path, repo_root),
        "index_path": repo_rel(index_path, repo_root),
        "packet_summary_path": repo_rel(summary_path, repo_root) if summary_path.is_file() else None,
        "manager_stage_authority_probe_path": repo_rel(manager_stage_probe_path, repo_root) if manager_stage_probe_path is not None else None,
        "manager_stage_authority_probe_status": manager_stage_probe.get("status"),
        "manager_stage_reports_reviewer_evidence_complete": manager_stage_probe.get("manager_stage_reports_reviewer_evidence_complete"),
        "manager_stage_transition_status": manager_stage_probe.get("stage_transition_status"),
        "current_stage": verdict.get("current_stage") or index.get("current_stage"),
        "current_mission_id": verdict.get("current_mission_id") or index.get("current_mission_id"),
        "schema_gap_analysis_path": repo_rel(schema_gap_path, repo_root) if schema_gap_path.is_file() else None,
        "schema_gap_status": schema_gap.get("status"),
        "reviewer_verdict_path": repo_rel(verdict_path, repo_root) if verdict_path.is_file() else None,
        "reviewer_verdict_request_path": repo_rel(request_path, repo_root) if request_path.is_file() else None,
        "reviewer_status": verdict_status,
        "reviewer_decision": verdict.get("decision"),
        "reviewer_verdict_independent": verdict_independent,
        "sealed_reviewer_handoff_source": handoff_source or None,
        "sealed_reviewer_handoff_source_valid": handoff_valid,
        "manager_reviewer_handoff_crosswalk_path": repo_rel(crosswalk_path, repo_root) if crosswalk_path.is_file() else None,
        "manager_reviewer_handoff_crosswalk_status": crosswalk.get("status"),
        "manager_recognition_check_path": repo_rel(recognition_check_path, repo_root) if recognition_check_path is not None else None,
        "manager_recognition_check_status": recognition_check.get("status"),
        "manager_recognition_check_ready": recognition_check.get("ready_for_manager_recognition"),
        "manager_recognition_check_mismatch_count": recognition_check.get("mismatch_count"),
        "legacy_canonical_chain_recognition_check_path": repo_rel(legacy_check_path, repo_root) if legacy_check_path is not None else None,
        "legacy_canonical_chain_status": legacy_check.get("status"),
        "legacy_canonical_chain_ready": legacy_check.get("ready_for_manager_recognition"),
        "ready_for_manager_recognition": ready,
        "operator_only_residual": {
            "only_remaining_gate": operator_only_gate,
            "agent_subjective_listening_performed": agent_subjective_listening,
            "semantic_av_quality_certified": semantic_certified,
            "av_sync_certified": av_sync_certified,
            "status": "operator_action_required",
        },
        "non_claims": verdict.get("non_claims") or schema_gap.get("non_claims") or [],
        "claim_boundary": "Current-stage delivery Reviewer-evidence recognition repair only; it preserves prior automatable delivery evidence and leaves Turbo human listening / semantic AV-sync operator-only.",
        "diagnosis": recognition_check.get("diagnosis") or schema_gap.get("exact_prior_missing_reviewer_evidence_requirement"),
        "legacy_canonical_chain_still_recognized": legacy_ready,
    }


def summarize_delivery_reviewer_active_hold_reconciliation(root: Path, *, repo_root: Path) -> dict[str, Any]:
    """Summarize the accepted active-hold reconciliation packet.

    This packet is not new model evidence. It records the current Manager/Planner
    hold surfaces, preserves the Manager-stage complete counter-evidence, and
    requires an independent Reviewer verdict before delivery report/manifest
    surfaces may claim the stale-hold diagnosis is accepted.
    """

    latest_path = root / DELIVERY_REVIEWER_ACTIVE_HOLD_RECONCILIATION_LATEST_REL
    packet_dir = _resolve_latest_packet_dir(root, repo_root, DELIVERY_REVIEWER_ACTIVE_HOLD_RECONCILIATION_LATEST_REL)
    if packet_dir is None:
        return {
            "status": "not_available",
            "latest_pointer": repo_rel(latest_path, repo_root),
            "claim_boundary": "No delivery Reviewer active-hold reconciliation packet selector is present in this evidence root.",
        }

    index_path = packet_dir / "INDEX.json"
    index = load_json(index_path)
    summary_path = _indexed_packet_path(packet_dir, repo_root, index, "summary") or packet_dir / "summary.json"
    probe_path = _indexed_packet_path(packet_dir, repo_root, index, "active_hold_reconciliation_probe") or packet_dir / "active_hold_reconciliation_probe.json"
    verdict_path = _indexed_packet_path(packet_dir, repo_root, index, "reviewer_verdict") or packet_dir / "reviewer_verdict.json"
    summary = load_json(summary_path)
    probe = load_json(probe_path)
    verdict = load_json(verdict_path)

    pending = summary.get("decision_pending") or index.get("decision_pending") or probe.get("decision_pending")
    if isinstance(pending, str) and pending.strip():
        raise AggregationError(
            "active-hold reconciliation packet still advertises pending Reviewer verdict: "
            f"{repo_rel(packet_dir, repo_root)}"
        )
    raw_event_sha256_count = sum(_recursive_key_count(item, "raw_event_sha256") for item in (index, summary, probe, verdict))
    if raw_event_sha256_count:
        raise AggregationError(
            "active-hold reconciliation packet still exposes opaque raw_event_sha256 fields: "
            f"count={raw_event_sha256_count} packet={repo_rel(packet_dir, repo_root)}"
        )

    boundary = verdict.get("boundary_covered") if isinstance(verdict.get("boundary_covered"), dict) else {}
    summary_boundary = summary.get("operator_only_boundary_preserved") if isinstance(summary.get("operator_only_boundary_preserved"), dict) else {}
    index_boundary = index.get("operator_only_boundary") if isinstance(index.get("operator_only_boundary"), dict) else {}
    agent_subjective_listening = bool(
        boundary.get("agent_subjective_listening_performed")
        or summary_boundary.get("agent_subjective_listening_performed")
        or index_boundary.get("agent_subjective_listening_performed")
    )
    semantic_certified = bool(
        boundary.get("semantic_av_quality_certified")
        or summary_boundary.get("semantic_av_quality_certified")
        or index_boundary.get("semantic_av_quality_certified")
    )
    av_sync_certified = bool(
        boundary.get("av_sync_certified")
        or summary_boundary.get("av_sync_certified")
        or index_boundary.get("av_sync_certified")
    )
    if agent_subjective_listening:
        raise AggregationError(f"active-hold reconciliation claims agent subjective listening: {repo_rel(verdict_path, repo_root)}")
    if semantic_certified or av_sync_certified:
        raise AggregationError(f"active-hold reconciliation certifies semantic AV/audio quality or AV-sync: {repo_rel(verdict_path, repo_root)}")

    verdict_status = verdict.get("status") or index.get("status") or "not_available"
    verdict_independent = bool(verdict.get("verdict_is_independent") is True or index.get("reviewer_verdict_independent") is True)
    verdict_accepted = bool(
        verdict_path.is_file()
        and verdict.get("current_stage") == "delivery"
        and verdict.get("decision") == "done"
        and verdict_status == "accepted_active_hold_reconciliation_stale_hold_diagnosis"
        and verdict_independent
    )
    if not verdict_accepted:
        raise AggregationError(f"active-hold reconciliation Reviewer verdict is not accepted/done: {repo_rel(verdict_path, repo_root)}")

    surfaces = summary.get("exact_active_hold_surfaces")
    if not isinstance(surfaces, list):
        surfaces = probe.get("exact_consumed_active_hold_surfaces_identified_this_turn")
    surfaces = surfaces if isinstance(surfaces, list) else []
    manager_feedback_surface_present = any(
        isinstance(item, dict)
        and item.get("line_number") == 12012
        and item.get("event_type") == "life.manager.feedback.persisted"
        and item.get("field") == "diagnostic"
        for item in surfaces
    )
    planner_surface_present = any(
        isinstance(item, dict)
        and item.get("line_number") == 12041
        and item.get("event_type") == "life.planner.verdict"
        and "reason" in (item.get("fields") or [])
        and "summary" in (item.get("fields") or [])
        for item in surfaces
    )
    manager_stage_complete = summary.get("manager_stage_complete_line") if isinstance(summary.get("manager_stage_complete_line"), dict) else {}
    if not manager_stage_complete:
        manager_stage_complete = probe.get("manager_stage_complete_observation") if isinstance(probe.get("manager_stage_complete_observation"), dict) else {}
    manager_stage_complete_present = bool(
        manager_stage_complete.get("line_number") == 12011
        and manager_stage_complete.get("event_type") == "life.manager.stage_decision"
    )
    if not (manager_feedback_surface_present and planner_surface_present and manager_stage_complete_present):
        raise AggregationError(
            "active-hold reconciliation packet is missing required Manager/Planner hold surfaces or Manager-stage counter-evidence"
        )

    active_hold_lines = []
    for item in surfaces:
        if isinstance(item, dict) and item.get("line_number") in {12012, 12041}:
            fields = item.get("fields") if isinstance(item.get("fields"), list) else [item.get("field")]
            active_hold_lines.append(
                {
                    "path": item.get("path"),
                    "line_number": item.get("line_number"),
                    "event_type": item.get("event_type"),
                    "fields": [field for field in fields if field],
                    "token": item.get("token"),
                }
            )

    return {
        "status": verdict_status,
        "packet_dir": repo_rel(packet_dir, repo_root),
        "latest_pointer": repo_rel(latest_path, repo_root),
        "index_path": repo_rel(index_path, repo_root),
        "summary_path": repo_rel(summary_path, repo_root),
        "active_hold_probe_path": repo_rel(probe_path, repo_root),
        "reviewer_verdict_path": repo_rel(verdict_path, repo_root),
        "reviewer_decision": verdict.get("decision"),
        "reviewer_verdict_independent": verdict_independent,
        "accepted_for_manager_visible_delivery_sync": True,
        "current_stage": verdict.get("current_stage") or index.get("current_stage"),
        "current_mission_id": verdict.get("current_mission_id") or index.get("current_mission_id"),
        "mismatch_classification": (probe.get("mismatch_classification") or {}).get("conclusion") if isinstance(probe.get("mismatch_classification"), dict) else None,
        "manager_stage_complete_counter_evidence_line": {
            "path": manager_stage_complete.get("source_path"),
            "line_number": manager_stage_complete.get("line_number"),
            "event_type": manager_stage_complete.get("event_type"),
            "diagnostic": manager_stage_complete.get("diagnostic"),
        },
        "active_hold_surfaces": active_hold_lines,
        "raw_event_sha256_count": raw_event_sha256_count,
        "operator_only_residual": {
            "only_remaining_gate": summary_boundary.get("only_remaining_operator_gate") or index_boundary.get("only_remaining_operator_gate") or "operator_human_auditory_listening_and_semantic_av_sync_review_for_Turbo_quality_certification",
            "agent_subjective_listening_performed": agent_subjective_listening,
            "semantic_av_quality_certified": semantic_certified,
            "av_sync_certified": av_sync_certified,
            "status": "operator_action_required",
        },
        "claim_boundary": "Accepted active-hold reconciliation/stale-hold diagnosis only; no GPU, Docker, model, benchmark, listening, semantic AV-sync certification, or Manager-owned state edit is claimed.",
    }


def summarize_current_manager_hold_no_gap_probe(root: Path, *, repo_root: Path) -> dict[str, Any]:
    """Summarize the latest live Manager-hold no-gap/repair probe.

    This packet is a Manager-visible bridge for the recurring generic
    `manager_hold_requires_stage_repair` token. It may be pending Reviewer
    acceptance; strict aggregation should still surface it so the Reviewer can
    inspect the exact current event lines and delivery-summary/package surfaces.
    """

    latest_path = root / CURRENT_MANAGER_HOLD_NO_GAP_LATEST_REL
    packet_dir = _resolve_latest_packet_dir(root, repo_root, CURRENT_MANAGER_HOLD_NO_GAP_LATEST_REL)
    if packet_dir is None:
        return {
            "status": "not_available",
            "latest_pointer": repo_rel(latest_path, repo_root),
            "claim_boundary": "No current Manager-hold no-gap packet selector is present in this evidence root.",
        }

    index_path = packet_dir / "INDEX.json"
    index = load_json(index_path)
    summary_path = _indexed_packet_path(packet_dir, repo_root, index, "summary") or packet_dir / "summary.json"
    probe_path = _indexed_packet_path(packet_dir, repo_root, index, "current_manager_hold_no_gap_probe") or packet_dir / "current_manager_hold_no_gap_probe.json"
    request_path = _indexed_packet_path(packet_dir, repo_root, index, "reviewer_verdict_request") or packet_dir / "reviewer_verdict_request.json"
    verdict_path = _indexed_packet_path(packet_dir, repo_root, index, "reviewer_verdict") or packet_dir / "reviewer_verdict.json"
    summary = load_json(summary_path)
    probe = load_json(probe_path)
    request = load_json(request_path) if request_path.is_file() else {}
    verdict = load_json(verdict_path) if verdict_path.is_file() else {}
    crosswalk_path = _indexed_packet_path(packet_dir, repo_root, index, "manager_reviewer_handoff_crosswalk") or packet_dir / "manager_reviewer_handoff_crosswalk.json"
    crosswalk = load_json(crosswalk_path) if crosswalk_path.is_file() else {}

    raw_event_sha256_count = sum(_recursive_key_count(item, "raw_event_sha256") for item in (index, summary, probe, request, verdict, crosswalk))
    if raw_event_sha256_count:
        raise AggregationError(
            "current Manager-hold no-gap packet exposes opaque raw_event_sha256 fields: "
            f"count={raw_event_sha256_count} packet={repo_rel(packet_dir, repo_root)}"
        )

    boundary_sources = [
        item
        for item in (
            index.get("operator_only_residual"),
            summary.get("operator_only_residual"),
            probe.get("operator_only_residual"),
            verdict.get("boundary_covered"),
        )
        if isinstance(item, dict)
    ]
    if any(bool(item.get("agent_subjective_listening_performed")) for item in boundary_sources):
        raise AggregationError(f"current Manager-hold no-gap packet claims agent subjective listening: {repo_rel(packet_dir, repo_root)}")
    if any(bool(item.get("semantic_av_quality_certified") or item.get("av_sync_certified")) for item in boundary_sources):
        raise AggregationError(f"current Manager-hold no-gap packet certifies semantic AV/audio quality or AV-sync: {repo_rel(packet_dir, repo_root)}")

    live = probe.get("live_authority_observation") if isinstance(probe.get("live_authority_observation"), dict) else {}
    manager_stage = live.get("manager_stage_decision") if isinstance(live.get("manager_stage_decision"), dict) else {}
    manager_feedback = live.get("manager_feedback") if isinstance(live.get("manager_feedback"), dict) else {}
    planner_surface = live.get("lower_authority_planner_hold_surface") if isinstance(live.get("lower_authority_planner_hold_surface"), dict) else {}
    if not (
        manager_stage.get("event_type") == "life.manager.stage_decision"
        and manager_stage.get("diagnostic") == "intentional_hold"
        and manager_feedback.get("event_type") == "life.manager.feedback.persisted"
        and manager_feedback.get("diagnostic") == "manager_hold_requires_stage_repair"
    ):
        raise AggregationError(f"current Manager-hold no-gap packet lacks current Manager stage/feedback surfaces: {repo_rel(packet_dir, repo_root)}")

    classification = probe.get("classification") if isinstance(probe.get("classification"), dict) else {}
    mismatch = bool(
        summary.get("project_local_reviewer_evidence_locator_schema_mismatch")
        or classification.get("project_local_reviewer_evidence_locator_schema_mismatch") is True
    )
    decisive_passed = bool(summary.get("decisive_check_passed") is True or probe.get("decisive_check_passed") is True)
    delivery_surfaces = probe.get("project_local_delivery_surfaces_checked") if isinstance(probe.get("project_local_delivery_surfaces_checked"), dict) else {}
    missing_manifest_paths = delivery_surfaces.get("required_manifest_paths_missing")
    missing_manifest_paths = missing_manifest_paths if isinstance(missing_manifest_paths, list) else []

    handoff_source: dict[str, Any] = {}
    for source in (
        verdict.get("sealed_reviewer_handoff_source"),
        verdict.get("sealed_reviewer_handoff_reference"),
        crosswalk.get("fresh_reviewer_handoff_source"),
        crosswalk.get("expected_fresh_reviewer_handoff_source"),
        index.get("sealed_reviewer_handoff_reference"),
    ):
        if isinstance(source, dict) and source.get("path"):
            handoff_source = source
            break
    handoff_valid = bool(
        handoff_source.get("path")
        and handoff_source.get("kind") == "round_reviewed_handoff"
        and handoff_source.get("producer_role") == "reviewer"
        and handoff_source.get("review_status") == "done"
        and handoff_source.get("current_stage") == "delivery"
    )
    crosswalk_ready = bool(
        crosswalk.get("status") == "pass"
        and (
            (crosswalk.get("manager_recognition_fields_after_reviewer") or {}).get("current_no_gap_packet_has_fresh_independent_reviewer_verdict") is True
            or (crosswalk.get("manager_recognition_fields_after_reviewer") or {}).get("current_manager_hold_no_gap_packet_has_fresh_independent_reviewer_verdict") is True
            or bool(handoff_source.get("path"))
        )
    )
    verdict_status = verdict.get("status") if verdict else None
    reviewer_decision = verdict.get("decision") if verdict else None
    reviewer_independent = bool(verdict.get("verdict_is_independent") is True or handoff_valid) if verdict else False
    explicit_gap = (
        summary.get("exact_current_stage_reviewer_evidence_gap")
        or probe.get("exact_current_stage_reviewer_evidence_gap")
        or request.get("exact_current_stage_reviewer_evidence_gap")
    )
    if not isinstance(explicit_gap, dict):
        missing_paths = []
        missing_or_unaccepted_fields = []
        if not verdict_path.is_file():
            missing_paths.append(repo_rel(verdict_path, repo_root))
            missing_or_unaccepted_fields.extend([
                "reviewer_verdict_path",
                "reviewer_status",
                "reviewer_decision",
                "reviewer_verdict_independent",
            ])
        if not crosswalk_path.is_file():
            missing_paths.append(repo_rel(crosswalk_path, repo_root))
            missing_or_unaccepted_fields.append("manager_reviewer_handoff_crosswalk_path")
        if not handoff_valid:
            missing_or_unaccepted_fields.append("sealed_reviewer_handoff_source_valid")
        if missing_paths or missing_or_unaccepted_fields:
            explicit_gap = {
                "status": "open_until_fresh_reviewer_verdict_and_crosswalk_are_present_and_accepted",
                "missing_paths": missing_paths,
                "missing_or_unaccepted_manager_visible_fields": sorted(set(missing_or_unaccepted_fields)),
            }
        else:
            explicit_gap = {}
    accepted_statuses = {
        "accepted_current_stage_delivery_reviewer_passed",
        "accepted_current_manager_hold_no_gap_reviewer_evidence_completion",
        "accepted_current_manager_hold_no_gap_blocker",
        "accepted_current_manager_hold_no_gap_probe",
    }
    reviewer_accepted = bool(
        verdict
        and verdict.get("current_stage") == "delivery"
        and reviewer_decision == "done"
        and reviewer_independent
        and handoff_valid
        and crosswalk_ready
        and verdict_status in accepted_statuses
        and not mismatch
        and decisive_passed
        and not missing_manifest_paths
    )
    reviewer_status = (
        "accepted_current_stage_delivery_reviewer_passed"
        if reviewer_accepted
        else (verdict_status or summary.get("reviewer_status") or index.get("reviewer_status") or "pending_fresh_independent_reviewer")
    )

    residual = summary.get("operator_only_residual") if isinstance(summary.get("operator_only_residual"), dict) else {}
    if not residual:
        residual = probe.get("operator_only_residual") if isinstance(probe.get("operator_only_residual"), dict) else {}

    return {
        "status": verdict_status or summary.get("status") or index.get("status") or "present",
        "packet_dir": repo_rel(packet_dir, repo_root),
        "latest_pointer": repo_rel(latest_path, repo_root),
        "index_path": repo_rel(index_path, repo_root),
        "summary_path": repo_rel(summary_path, repo_root),
        "probe_path": repo_rel(probe_path, repo_root),
        "reviewer_verdict_request_path": repo_rel(request_path, repo_root) if request_path.is_file() else None,
        "reviewer_verdict_path": repo_rel(verdict_path, repo_root) if verdict_path.is_file() else None,
        "reviewer_status": reviewer_status,
        "reviewer_verdict_status": verdict_status,
        "reviewer_decision": reviewer_decision,
        "reviewer_verdict_independent": reviewer_independent,
        "sealed_reviewer_handoff_source": handoff_source or None,
        "sealed_reviewer_handoff_source_valid": handoff_valid,
        "manager_reviewer_handoff_crosswalk_path": repo_rel(crosswalk_path, repo_root) if crosswalk_path.is_file() else None,
        "manager_reviewer_handoff_crosswalk_status": crosswalk.get("status"),
        "accepted_for_manager_visible_delivery_sync": reviewer_accepted,
        "manager_visible_delivery_sync_ready": reviewer_accepted,
        "exact_current_stage_reviewer_evidence_gap": {} if reviewer_accepted else explicit_gap,
        "current_stage": summary.get("current_stage") or index.get("current_stage"),
        "current_mission_id": summary.get("current_mission_id") or index.get("current_mission_id"),
        "decisive_check_passed": decisive_passed,
        "project_local_reviewer_evidence_locator_schema_mismatch": mismatch,
        "missing_manifest_paths_count": len(missing_manifest_paths),
        "manager_stage_decision": {
            "path": manager_stage.get("path"),
            "line_number": manager_stage.get("line_number"),
            "reason": manager_stage.get("reason"),
            "diagnostic": manager_stage.get("diagnostic"),
        },
        "manager_feedback": {
            "path": manager_feedback.get("path"),
            "line_number": manager_feedback.get("line_number"),
            "reason": manager_feedback.get("reason"),
            "diagnostic": manager_feedback.get("diagnostic"),
        },
        "planner_hold_surface": {
            "path": planner_surface.get("path"),
            "line_number": planner_surface.get("line_number"),
            "reason": planner_surface.get("reason"),
            "status": planner_surface.get("status"),
        },
        "operator_only_residual": {
            "only_remaining_gate": residual.get("only_remaining_gate") or "operator_human_auditory_listening_and_semantic_av_sync_review_for_Turbo_quality_certification",
            "agent_subjective_listening_performed": bool(residual.get("agent_subjective_listening_performed")),
            "semantic_av_quality_certified": bool(residual.get("semantic_av_quality_certified")),
            "av_sync_certified": bool(residual.get("av_sync_certified")),
            "status": residual.get("status") or "operator_action_required",
        },
        "raw_event_sha256_count": raw_event_sha256_count,
        "claim_boundary": "Current Manager-hold no-gap/visibility packet only; no GPU, Docker, model, benchmark, subjective listening, semantic AV-sync certification, or Manager-owned state edit is claimed.",
    }


def summarize_turbo_operator_gate_packet(root: Path, *, repo_root: Path) -> dict[str, Any]:
    latest_path = root / TURBO_OPERATOR_GATE_LATEST_REL
    packet_dir = _resolve_latest_packet_dir(root, repo_root, TURBO_OPERATOR_GATE_LATEST_REL)
    if packet_dir is None:
        return {
            "status": "not_available",
            "latest_pointer": rel(latest_path, root),
            "claim_boundary": "No Turbo operator-gate Reviewer packet selector is present in this evidence root.",
        }

    summary_path = packet_dir / "summary.json"
    summary = load_json(summary_path)
    manifest_path = packet_dir / "media_listening_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.is_file() else {}
    verdict_path = packet_dir / "reviewer_verdict.json"
    verdict = load_json(verdict_path) if verdict_path.is_file() else {}
    crosswalk_path = packet_dir / "manager_reviewer_handoff_crosswalk.json"
    crosswalk = load_json(crosswalk_path) if crosswalk_path.is_file() else {}
    stage_closeout_crosswalk_path = packet_dir / "manager_stage_closeout_crosswalk.json"
    manager_visibility_path = packet_dir / "manager_visibility_resolution.json"
    reviewer_request_path = packet_dir / "reviewer_verdict_request.json"
    recognition_check_paths = sorted(packet_dir.glob("*recognition_check*.json"))
    recognition_check_path = recognition_check_paths[-1] if recognition_check_paths else None
    recognition_check = load_json(recognition_check_path) if recognition_check_path is not None else {}

    if summary.get("agent_subjective_listening_performed") is True:
        raise AggregationError(f"Turbo operator-gate packet claims agent subjective listening: {summary_path}")
    if summary.get("semantic_av_quality_certified") is True or summary.get("audio_semantics_certified") is True or summary.get("av_sync_certified") is True:
        raise AggregationError(f"Turbo operator-gate packet certifies subjective semantic AV/audio quality: {summary_path}")

    reviewer_status = (
        summary.get("reviewer_status")
        or verdict.get("status")
        or verdict.get("mapped_reviewer_status_for_this_packet")
        or summary.get("independent_reviewer_verdict", {}).get("mapped_status")
        or "not_available"
    )
    reviewer_accepted = reviewer_status in {
        "accepted_independent_reviewer_passed",
        "accepted_current_stage_delivery_boundary",
        "accepted_current_stage_delivery_reviewer_passed",
    }
    manager_repair = summary.get("manager_recognition_repair") if isinstance(summary.get("manager_recognition_repair"), dict) else {}
    reviewer_handoff_source = {}
    for source in (
        verdict.get("sealed_reviewer_handoff_source"),
        verdict.get("reviewer_handoff_source"),
        manager_repair.get("sealed_reviewer_handoff_source"),
        crosswalk.get("reviewer_handoff_source"),
    ):
        if isinstance(source, dict) and source.get("path"):
            reviewer_handoff_source = source
            break
    reviewer_handoff_source_valid = bool(
        reviewer_handoff_source.get("path")
        and reviewer_handoff_source.get("kind") == "round_reviewed_handoff"
        and reviewer_handoff_source.get("producer_role") == "reviewer"
        and reviewer_handoff_source.get("review_status") == "done"
    )
    verdict_independent = (
        reviewer_handoff_source_valid
        or verdict.get("source_producer_role") == "reviewer"
        or summary.get("independent_reviewer_verdict", {}).get("source_producer_role") == "reviewer"
        or verdict.get("verdict_is_independent") is True
    )
    current_stage_reviewer_requires_handoff_source = reviewer_status in {
        "accepted_current_stage_delivery_boundary",
        "accepted_current_stage_delivery_reviewer_passed",
    } or summary.get("status") == "accepted_current_stage_delivery_boundary"
    automatable_complete = summary.get("automatable_delivery_gates_complete") is True
    human_gate = summary.get("human_auditory_or_semantic_gate") or summary.get("remaining_gate") or "operator_action_required"
    semantic_certified = bool(summary.get("semantic_av_quality_certified") or summary.get("audio_semantics_certified"))
    av_sync_certified = bool(summary.get("av_sync_certified"))
    media_case_count = manifest.get("case_count") or summary.get("turbo_quality_automatable_evidence", {}).get("case_count")

    return {
        "status": summary.get("status", "present"),
        "packet_dir": repo_rel(packet_dir, repo_root),
        "latest_pointer": repo_rel(latest_path, repo_root),
        "summary_path": repo_rel(summary_path, repo_root),
        "media_listening_manifest_path": repo_rel(manifest_path, repo_root) if manifest_path.is_file() else None,
        "reviewer_packet_path": repo_rel(packet_dir / "reviewer_packet.md", repo_root) if (packet_dir / "reviewer_packet.md").is_file() else None,
        "reviewer_verdict_path": repo_rel(verdict_path, repo_root) if verdict_path.is_file() else None,
        "reviewer_verdict_request_path": repo_rel(reviewer_request_path, repo_root) if reviewer_request_path.is_file() else None,
        "automatable_delivery_gates_complete": automatable_complete,
        "operator_listening_manifest_case_count": media_case_count,
        "agent_subjective_listening_performed": summary.get("agent_subjective_listening_performed", False),
        "human_auditory_semantic_av_sync_gate": human_gate,
        "semantic_av_quality_certified": semantic_certified,
        "av_sync_certified": av_sync_certified,
        "reviewer_status": reviewer_status,
        "reviewer_verdict_independent": verdict_independent,
        "accepted_for_current_stage_closing": bool(
            reviewer_accepted
            and verdict_independent
            and (reviewer_handoff_source_valid or not current_stage_reviewer_requires_handoff_source)
            and automatable_complete
            and human_gate == "operator_action_required"
            and not semantic_certified
            and not av_sync_certified
            and summary.get("agent_subjective_listening_performed") is False
        ),
        "manager_recognition_repair_path": repo_rel(crosswalk_path, repo_root) if crosswalk_path.is_file() else None,
        "manager_stage_closeout_crosswalk_path": repo_rel(stage_closeout_crosswalk_path, repo_root) if stage_closeout_crosswalk_path.is_file() else None,
        "manager_visibility_resolution_path": repo_rel(manager_visibility_path, repo_root) if manager_visibility_path.is_file() else None,
        "manager_recognition_check_path": repo_rel(recognition_check_path, repo_root) if recognition_check_path is not None else None,
        "manager_recognition_check_status": recognition_check.get("status"),
        "manager_recognition_check_ready": recognition_check.get("ready_for_manager_recognition"),
        "manager_recognition_check_mismatch_count": recognition_check.get("mismatch_count"),
        "manager_hold_diagnosis": crosswalk.get("manager_hold_diagnosis") or manager_repair.get("diagnosis"),
        "reviewer_handoff_source": reviewer_handoff_source or None,
        "reviewer_handoff_source_valid": reviewer_handoff_source_valid,
        "exact_manager_hold_repair": summary.get("exact_manager_hold_repair"),
        "claim_boundary": summary.get(
            "automatable_delivery_gates_scope",
            "Turbo operator-gate packet only; human auditory listening and semantic AV-sync remain operator-only.",
        ),
    }


def summarize_local_lifecycle(root: Path) -> dict[str, Any]:
    delivery_dir = root / DELIVERY_REL
    candidates = sorted(p for p in delivery_dir.glob(f"{LOCAL_LIFECYCLE_PREFIX}*") if p.is_dir())
    if not candidates:
        raise AggregationError("missing clean-room local lifecycle evidence")
    selected = candidates[-1]
    summary_rel = selected.relative_to(root).as_posix() + "/lifecycle/stages/05_lifecycle_summary.json"
    model_rel = selected.relative_to(root).as_posix() + "/lifecycle/stages/02_model_prepare.json"
    deploy_rel = selected.relative_to(root).as_posix() + "/lifecycle/stages/03_deploy.json"
    summary = load_json(root / summary_rel)
    model = load_json(root / model_rel)
    deploy = load_json(root / deploy_rel)
    if summary.get("status") != "pass" or summary.get("publication_audit_status") != "pass":
        raise AggregationError(f"local lifecycle did not pass: {summary_rel}")
    if model.get("status") != "pass" or int(model.get("local_non_symlink_file_count", 0)) <= 0:
        raise AggregationError(f"local lifecycle model-prepare did not pass: {model_rel}")
    if deploy.get("status") != "pass":
        raise AggregationError(f"local lifecycle deploy did not pass: {deploy_rel}")
    return {
        "status": "pass_packaging_lifecycle_only",
        "run_dir": selected.relative_to(root).as_posix(),
        "summary_path": summary_rel,
        "model_prepare_path": model_rel,
        "deploy_path": deploy_rel,
        "publication_audit_status": summary["publication_audit_status"],
        "model_file_count": model["local_non_symlink_file_count"],
        "model_total_bytes": model["local_total_bytes"],
        "deploy_status": deploy["status"],
        "claim_boundary": summary["claim_boundary"],
    }


def aggregate_evidence(root: Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    repo_root = repo_root or root.parent.parent.parent
    baseline = summarize_baseline(root)
    turbo = summarize_turbo(root, baseline)
    dmd = summarize_dmd(root)
    sol = summarize_sol(root, baseline)
    lifecycle = summarize_local_lifecycle(root)
    final_gates = summarize_final_gates(root)
    operator_gate = summarize_turbo_operator_gate_packet(root, repo_root=repo_root)
    delivery_reviewer_repair = summarize_delivery_reviewer_evidence_repair(root, repo_root=repo_root)
    active_hold_reconciliation = summarize_delivery_reviewer_active_hold_reconciliation(root, repo_root=repo_root)
    current_manager_hold_no_gap = summarize_current_manager_hold_no_gap_probe(root, repo_root=repo_root)
    quality_config_path = root / QUALITY_CONFIG_REL
    quality_dry_run_dir = root / QUALITY_DRY_RUN_REL

    summary = {
        "schema_version": "argus-ir04-delivery-aggregation-v1",
        "status": "pass_strict_evidence_grounded",
        "input_root": rel(root, repo_root),
        "lanes": {
            Fidelity: {
                "baseline": baseline,
                "accepted_scope": "internal same-physical-GPU3 baseline only; warm-primary denominator is N=10 true warm requests, total all-request set is N=13 with three session-first requests",
                "must_not_include": ["Turbo merged LoRA", "DMD/DMD2", "Sol-Attn approximations"],
            },
            Practical: {
                "turbo_merged": turbo,
                "dmd": dmd,
                "sol_exact_and_sol_attn": sol,
                "accepted_scope": "disclosed approximation/diagnostic lane only unless future evidence changes",
            },
        },
        "claims": {
            "accepted": [
                {
                    "claim": "A single-A6000 internal BF16-exact baseline v2 is certified on physical GPU3 with N=13 total requests, N=10 true warm-primary requests, and three session-first requests.",
                    "track": Fidelity,
                    "evidence": [BASELINE_REL],
                    "limits": "Internal same-device denominator only; not an optimized or external reproduction.",
                },
                {
                    "claim": "The latest GPU3 paired Turbo timing run is accepted as the only practical speed result: two excluded warmups, paired N=10 per schedule, strict structural AV pass, and baseline-v2 warm N=10 denominator.",
                    "track": Practical,
                    "evidence": [TURBO_TIMING_LATEST_REL, f"{TURBO_TIMING_REPEATS_REL}/{turbo['run_id']}/timing_summary.json", f"{TURBO_TIMING_REPEATS_REL}/{turbo['run_id']}/merge_manifest.json"],
                    "limits": (
                        "Practical approximation only; operator overall playback/listening acceptance is recorded; 8-step remains the default and the known 4-step visual failure remains disclosed."
                        if turbo["quality_suite"].get("operator_overall_acceptance_recorded")
                        else "Practical approximation only; semantic AV quality and human auditory listening remain pending; 8-step is the default candidate and 4-step is ultra-fast quality-cost experimental."
                    ),
                },
                {
                    "claim": "AdaLN is an N=1 exact-output candidate only; the original harness-tail failure is preserved and the single-run benefit is below baseline warm-run noise.",
                    "track": Practical,
                    "evidence": [ADALN_QUALITY_REL, ADALN_POSTHOC_REL, ADALN_VERDICT_REL],
                    "limits": "Not an accepted N=10 speedup and not deployed as a certified fidelity path.",
                },
                {
                    "claim": "The gated one-command local lifecycle passed in a clean-room export/work directory using existing local locked resources only.",
                    "track": "packaging_deployment_only",
                    "evidence": [lifecycle["summary_path"], lifecycle["model_prepare_path"], lifecycle["deploy_path"]],
                    "limits": "Packaging/deployment evidence only: no container start, model load, GPU inference, media generation, speed, fidelity, or quality claim.",
                },
            ],
            "rejected": [
                {
                    "claim": "Turbo merged LoRA is a BF16-exact/fidelity result.",
                    "reason": "Turbo is practical_disclosed_approx and uses a statically merged LoRA approximation.",
                    "evidence": [f"{TURBO_TIMING_REPEATS_REL}/{turbo['run_id']}/timing_summary.json"],
                },
                {
                    "claim": "Treating the earlier GPU2 Turbo smoke run as an accepted speedup denominator/result.",
                    "reason": "GPU2 smoke is retained only as bring-up evidence; the only speedup result uses same physical GPU3 paired timing against baseline v2 warm N=10.",
                    "evidence": [f"{GPU2_BRINGUP_RUN_REL}/turbo_summary.json", f"{TURBO_TIMING_REPEATS_REL}/{turbo['run_id']}/timing_summary.json"],
                },
                {
                    "claim": "RoPE/all-exact kernels are accepted exact replacements.",
                    "reason": "Same-prompt diagnostic evidence shows non-zero video/audio drift.",
                    "evidence": [ROPE_QUALITY_REL, ALL_EXACT_QUALITY_REL],
                },
                {
                    "claim": "SwiGLU is a retained practical speed gain.",
                    "reason": "Current evidence is exact diagnostic output only; no retained speedup gain is accepted or deployed.",
                    "evidence": [SWIGLU_QUALITY_REL, SWIGLU_HTTP_REL],
                },
                {
                    "claim": "Toy Sol-Attn is deployed or faster.",
                    "reason": "Current toy harness is kernel-candidate-only, model_load=false, and sparse median is slower than dense.",
                    "evidence": [SOL_ATTN_RESULT_REL],
                },
            ],
            "blocked": [
                {
                    "claim": "MiniMax-H3 DMD/DMD2 is a no-go after Turbo unless feasibility evidence changes.",
                    "status": dmd["status"],
                    "evidence": [DMD_REL],
                },
                {
                    "claim": "Turbo human-auditory quality, semantic AV quality, and semantic AV-sync remain operator-only and uncertified.",
                    "status": "blocked_human_auditory_listening_pending_semantic_quality_not_certified",
                    "evidence": [TURBO_QUALITY_LATEST_REL, f"{TURBO_QUALITY_RUNS_REL}/{turbo['quality_suite']['run_id']}/quality_suite_analysis.json", f"{TURBO_QUALITY_RUNS_REL}/{turbo['quality_suite']['run_id']}/human_review.md"],
                },
            ],
            "pending": [
                {
                    "claim": "Formal DLO N10 performance promotion is complete.",
                    "status": "pending_no_formal_n10_because_current_candidate_is_below_baseline_noise",
                    "evidence": ["dlo_autotune/detached_continuation/status.txt", "dlo_autotune/runs/a6000_dlo_candidate_50_rl16_20260810T141257Z/candidate50_summary.json"],
                },
            ],
        },
        "delivery_lifecycle": lifecycle,
        "final_gates": final_gates,
        "turbo_operator_gate_reviewer_packet": operator_gate,
        "delivery_reviewer_evidence_recognition_repair": delivery_reviewer_repair,
        "delivery_reviewer_evidence_active_hold_reconciliation": active_hold_reconciliation,
        "current_manager_hold_no_gap_probe": current_manager_hold_no_gap,
        "quality_suite_runner": {
            "config_path": rel(quality_config_path, root) if quality_config_path.exists() else None,
            "dry_run_plan_dir": rel(quality_dry_run_dir, root) if quality_dry_run_dir.exists() else None,
            "default_mode": "dry_run_fail_closed",
            "execution_status": "not_run_by_aggregator",
        },
        "test_policy": {
            "gpu_docker_inference_tests_counted_as_passed": False,
            "skipped_gpu_placeholders_counted_as_coverage": False,
            "cpu_static_tests_are_the_only_tests_counted_in_this_delivery": True,
        },
        "operator_commands": [
            "PYTHONPATH=code:.:ports/minimax_h3_a6000/src python3 -m pytest -q tests ports/minimax_h3_a6000/tests",
            "python3 tools/verify_run.py tests/fixtures/minimal_av_case/run_record.json",
            f"python3 tools/turbo_quality_suite_runner.py --dry-run --config {repo_rel(quality_config_path, repo_root)} --out {repo_rel(quality_dry_run_dir, repo_root)}",
            f"python3 tools/argus_ir04_aggregate.py --strict --input {repo_rel(root, repo_root)} --out technical_report/evidence/minimax_h3_desktop/delivery/argus_ir04_delivery_summary.json --report-out technical_report/final_technical_report.md --manifest-out technical_report/evidence/minimax_h3_desktop/delivery/package_manifest.json",
        ],
    }

    if turbo["quality_suite"].get("operator_overall_acceptance_recorded"):
        summary["claims"]["blocked"] = [
            item
            for item in summary["claims"]["blocked"]
            if not str(item.get("claim", "")).startswith("Turbo human-auditory quality")
        ]
        summary["claims"]["accepted"].append(
            {
                "claim": "The operator completed overall playback/listening review and accepted the practical Turbo release quality, with 8-step retained as default and the known 4-step visual failure preserved.",
                "track": Practical,
                "evidence": [
                    f"{TURBO_QUALITY_RUNS_REL}/{turbo['quality_suite']['run_id']}/operator_acceptance.json",
                    f"{TURBO_QUALITY_RUNS_REL}/{turbo['quality_suite']['run_id']}/human_review.md",
                ],
                "limits": "Overall operator acceptance without a fabricated per-case rubric transcript; Turbo remains approximate and the known 4-step failure remains rejected for default promotion.",
            }
        )

    h3_sol = sol.get("h3_sol_attn_r8", {})
    if h3_sol.get("status") == "sparse_runtime_valid_5step_diagnostic":
        summary["claims"]["accepted"].append(
            {
                "claim": "H3 Sol-Attn r8 metadata plumbing reached the real 5-step H3 attention boundary and executed the sparse path with valid structural AV/resource telemetry.",
                "track": "diagnostic_metadata_plumbing_only",
                "evidence": [h3_sol["evidence_path"]],
                "limits": h3_sol["claim_boundary"],
            }
        )
        matched = h3_sol.get("matched_retest", {})
        formal = h3_sol.get("formal_n10", {})
        promotion_status = "pending_matched_workload_gate_required_before_speedup_n10_or_quality_claim"
        promotion_evidence = [h3_sol["evidence_path"]]
        promotion_recorded = False
        if matched.get("status") == "proceed_to_formal_n10_candidate":
            matched_claim = (
                "The r8 N=3 matched-workload route gate is terminal and led to the later accepted formal N>=10 Sol-Attn gate."
                if formal.get("status") == "accepted_formal_n10_same_gpu_sol_attn_speed_candidate"
                else "The r8 N=3 matched-workload route gate is terminal and recommends formal N>=10 Sol-Attn testing."
            )
            summary["claims"]["accepted"].append(
                {
                    "claim": matched_claim,
                    "track": "diagnostic_metadata_plumbing_only",
                    "evidence": [matched["evidence_path"]],
                    "limits": "Bounded N=3 5-step route decision only; not formal N10, not a speedup, not BF16 fidelity, and not quality-equivalence certification.",
                }
            )
            promotion_status = "pending_formal_n10_required_after_r8_n3_candidate_before_speedup_or_quality_claim"
            promotion_evidence = [matched["evidence_path"]]
        formal_status = formal.get("status")
        if formal_status not in {None, "not_available"}:
            formal_evidence = [formal.get("decision_path") or formal.get("status_path") or formal.get("source_run_dir")]
            if formal_status == "accepted_formal_n10_same_gpu_sol_attn_speed_candidate":
                summary["claims"]["accepted"].append(
                    {
                        "claim": "Formal r8 Sol-Attn N>=10 matched-workload promotion is complete.",
                        "track": "formal_n10_matched_5step_sol_attn_opt_in_not_bf16_fidelity",
                        "evidence": formal_evidence,
                        "limits": "Formal matched 5-step Sol-Attn lane only; not BF16 fidelity, Turbo, DLO, DMD, release, or human-auditory quality certification.",
                    }
                )
                promotion_recorded = True
            elif str(formal_status).startswith("rejected_") or str(formal_status).startswith("blocked_"):
                summary["claims"]["rejected"].append(
                    {
                        "claim": "Formal r8 Sol-Attn N>=10 matched-workload promotion is accepted.",
                        "reason": f"{formal_status}: {formal.get('reason')}",
                        "evidence": formal_evidence,
                    }
                )
                promotion_recorded = True
            elif str(formal_status).startswith("incomplete_") or formal_status == "inconclusive_incomplete_formal_n10_run":
                promotion_status = formal_status
                promotion_evidence = formal_evidence
        if not promotion_recorded:
            summary["claims"]["pending"].append(
                {
                    "claim": "Sol-Attn matched-workload correctness/quality and performance promotion is complete.",
                    "status": promotion_status,
                    "evidence": promotion_evidence,
                }
            )
        if matched.get("status") == "pending_nonterminal_do_not_promote_or_stop":
            summary["claims"]["pending"].append(
                {
                    "claim": "The started r8 matched-workload retest has a terminal route decision.",
                    "status": matched["status"],
                    "evidence": [matched["evidence_path"]],
                }
            )
    return summary


def report_markdown(summary: dict[str, Any]) -> str:
    baseline = summary["lanes"][Fidelity]["baseline"]
    turbo = summary["lanes"][Practical]["turbo_merged"]
    sol = summary["lanes"][Practical]["sol_exact_and_sol_attn"]
    h3_sol = sol.get("h3_sol_attn_r8", {})
    dmd = summary["lanes"][Practical]["dmd"]
    lifecycle = summary["delivery_lifecycle"]
    final_gates = summary.get("final_gates", {})
    operator_gate = summary.get("turbo_operator_gate_reviewer_packet", {})
    delivery_reviewer_repair = summary.get("delivery_reviewer_evidence_recognition_repair", {})
    active_hold_reconciliation = summary.get("delivery_reviewer_evidence_active_hold_reconciliation", {})
    current_manager_hold_no_gap = summary.get("current_manager_hold_no_gap_probe", {})
    operator_quality_accepted = bool(turbo.get("quality_suite", {}).get("operator_overall_acceptance_recorded"))
    lines = [
        "# ARGUS-IR-04 Final Technical Report",
        "",
        (
            "Status: **final evidence integration; operator overall practical-quality gate accepted**. This report is evidence-grounded and CPU/static-generated; it does not add measurements, run inference, or publish results."
            if operator_quality_accepted
            else "Status: **final evidence integration, not quality-complete**. This report is evidence-grounded and CPU/static-generated; it does not add measurements, run inference, or publish results."
        ),
        "",
        "## Scope and lane separation",
        "",
        f"- Fidelity lane: `{Fidelity}`. Only the certified same-physical-GPU3 baseline is accepted here.",
        f"- Practical lane: `{Practical}`. GPU3 paired Turbo timing, DMD feasibility notes, and exact-kernel diagnostics live here unless future evidence changes.",
        "- Turbo practical results must not be relabeled as fidelity/BF16-exact results.",
        "",
        "## Baseline certification",
        "",
        f"- Evidence: `{baseline['evidence_path']}`.",
        f"- Status: `{baseline['status']}` on `{baseline['platform']}`; schema `{baseline['schema']}`.",
        f"- Physical device: host GPU{baseline['physical_device']['host_gpu_index']}, SM{baseline['physical_device']['compute_capability']}, UUID `{baseline['physical_device']['uuid']}`.",
        f"- All requests: N={baseline['all_requests']['n']}, mean={baseline['all_requests']['mean_s']}s, median={baseline['all_requests']['median_s']}s.",
        f"- Warm-primary denominator: N={baseline['warm_primary_denominator']['n']}, mean={baseline['warm_primary_denominator']['mean_s']}s, median={baseline['warm_primary_denominator']['median_s']}s, CV={baseline['warm_primary_denominator']['cv_percent']}%.",
        f"- Session-first requests: N={baseline['session_first_requests']['n']} across {baseline['service_session_count']} service sessions.",
        "- Supersedes the prior v1 warm-count interpretation; speedups below use only this warm N=10 GPU3 denominator.",
        "",
        "## Turbo merged practical timing and quality evidence",
        "",
        f"- Timing evidence: `{turbo['latest_pointer']}` -> `{turbo['timing_summary_path']}`; merge manifest `{turbo['merge_manifest_path']}`.",
        f"- Same physical device: host GPU{turbo['physical_device']['host_gpu_index']}, SM{turbo['physical_device']['compute_capability']}, UUID `{turbo['physical_device']['uuid']}`.",
        f"- Merge: status `{turbo['merge_status']}`, strength {turbo['merge_strength']}, completed shards {turbo['completed_shard_count']}.",
        f"- Paired timing design: two excluded warmups, then N={turbo['paired_formal_n_per_schedule']} formal paired samples per schedule; strict AV pass count {turbo['structural_av_pass_count']}.",
        f"- 4-step paired median: {turbo['steps']['4']['median_s']}s; speedup vs same-GPU3 BF16 warm N=10 median: {turbo['steps']['4']['speedup_vs_same_gpu3_bf16_warm_n10_median']}x; CV={turbo['steps']['4']['cv_percent']}%.",
        f"- 8-step paired median: {turbo['steps']['8']['median_s']}s; speedup vs same-GPU3 BF16 warm N=10 median: {turbo['steps']['8']['speedup_vs_same_gpu3_bf16_warm_n10_median']}x; CV={turbo['steps']['8']['cv_percent']}%.",
        "- The 8-step schedule is the practical default candidate. The 4-step schedule is ultra-fast/quality-cost experimental because the visual suite exposed a teapot-geometry failure and lower audio fidelity.",
        f"- Quality-suite evidence: `{turbo['quality_suite']['latest_pointer']}` -> `{turbo['quality_suite']['analysis_path']}`; baseline seed0 comparison `{turbo['quality_suite']['baseline_seed0_comparison_path']}`; audio envelopes `{turbo['quality_suite']['audio_energy_envelopes_path']}`; human review `{turbo['quality_suite']['human_review_path']}`; review contact sheets={turbo['quality_suite']['review_contact_sheet_count']}.",
        f"- Quality-suite coverage: {turbo['quality_suite']['case_count']} outputs (3 prompts x 4 seeds x 4/8 steps), N={turbo['quality_suite']['steps']['4']['n']} per schedule; structural AV pass={turbo['quality_suite']['all_cases_structural_av_pass']}.",
        (
            f"- Operator overall playback/listening acceptance is recorded at `{turbo['quality_suite'].get('operator_acceptance_path')}`; 8-step remains the practical default and the known 4-step visual failure remains preserved."
            if operator_quality_accepted
            else "- Human auditory listening remains pending; semantic AV quality is not certified."
        ),
        f"- GPU2 smoke scope: {turbo['gpu2_bringup_scope']}.",
        (
            "- The delivery hold/Reviewer packets below are preserved as historical pre-acceptance evidence. Their operator-action-required fields were superseded by the later operator acceptance record; they are not the current release gate."
            if operator_quality_accepted
            else ""
        ),
        "",
    ]
    if operator_gate.get("status") != "not_available":
        lines.extend(
            [
                "## Turbo operator-only listening gate packet" + (" (historical pre-acceptance packet)" if operator_quality_accepted else ""),
                "",
                f"- Latest packet: `{operator_gate.get('packet_dir')}`; packet_status=`{operator_gate.get('status')}`; reviewer_status=`{operator_gate.get('reviewer_status')}`; accepted_for_current_stage_closing={operator_gate.get('accepted_for_current_stage_closing')}",
                f"- Automatable delivery gates complete={operator_gate.get('automatable_delivery_gates_complete')}; operator listening manifest cases={operator_gate.get('operator_listening_manifest_case_count')}; manifest `{operator_gate.get('media_listening_manifest_path')}`.",
                f"- Operator-only residual: human auditory listening / semantic AV-sync gate=`{operator_gate.get('human_auditory_semantic_av_sync_gate')}`; agent_subjective_listening_performed={operator_gate.get('agent_subjective_listening_performed')}; semantic_av_quality_certified={operator_gate.get('semantic_av_quality_certified')}; av_sync_certified={operator_gate.get('av_sync_certified')}.",
                f"- Reviewer handoff source: `{(operator_gate.get('reviewer_handoff_source') or {}).get('path')}`; source_valid={operator_gate.get('reviewer_handoff_source_valid')}; manager_recognition_repair=`{operator_gate.get('manager_recognition_repair_path')}`; manager_stage_closeout_crosswalk=`{operator_gate.get('manager_stage_closeout_crosswalk_path')}`; manager_visibility_resolution=`{operator_gate.get('manager_visibility_resolution_path')}`.",
                f"- Manager recognition check: `{operator_gate.get('manager_recognition_check_path')}`; status=`{operator_gate.get('manager_recognition_check_status')}`; ready={operator_gate.get('manager_recognition_check_ready')}; mismatch_count={operator_gate.get('manager_recognition_check_mismatch_count')}.",
                f"- Boundary: {operator_gate.get('claim_boundary')}.",
                "",
            ]
        )
    if delivery_reviewer_repair.get("status") != "not_available":
        handoff_source = delivery_reviewer_repair.get("sealed_reviewer_handoff_source") or {}
        residual = delivery_reviewer_repair.get("operator_only_residual") or {}
        lines.extend(
            [
                "## Current-stage delivery Reviewer evidence recognition repair",
                "",
                f"- Repair packet: `{delivery_reviewer_repair.get('packet_dir')}`; status=`{delivery_reviewer_repair.get('status')}`; ready_for_manager_recognition={delivery_reviewer_repair.get('ready_for_manager_recognition')}; current_stage=`{delivery_reviewer_repair.get('current_stage')}`; current_mission_id=`{delivery_reviewer_repair.get('current_mission_id')}`.",
                f"- Fresh Reviewer verdict: `{delivery_reviewer_repair.get('reviewer_verdict_path')}`; reviewer_status=`{delivery_reviewer_repair.get('reviewer_status')}`; decision=`{delivery_reviewer_repair.get('reviewer_decision')}`; independent={delivery_reviewer_repair.get('reviewer_verdict_independent')}; sealed_handoff=`{handoff_source.get('path')}`; sealed_handoff_valid={delivery_reviewer_repair.get('sealed_reviewer_handoff_source_valid')}.",
                f"- Manager recognition check: `{delivery_reviewer_repair.get('manager_recognition_check_path')}`; status=`{delivery_reviewer_repair.get('manager_recognition_check_status')}`; ready={delivery_reviewer_repair.get('manager_recognition_check_ready')}; mismatch_count={delivery_reviewer_repair.get('manager_recognition_check_mismatch_count')}.",
                f"- Manager-stage authority probe: `{delivery_reviewer_repair.get('manager_stage_authority_probe_path')}`; status=`{delivery_reviewer_repair.get('manager_stage_authority_probe_status')}`; reviewer_evidence_complete={delivery_reviewer_repair.get('manager_stage_reports_reviewer_evidence_complete')}; transition_status=`{delivery_reviewer_repair.get('manager_stage_transition_status')}`.",
                f"- Schema/crosswalk evidence: schema_gap=`{delivery_reviewer_repair.get('schema_gap_analysis_path')}` status=`{delivery_reviewer_repair.get('schema_gap_status')}`; handoff_crosswalk=`{delivery_reviewer_repair.get('manager_reviewer_handoff_crosswalk_path')}` status=`{delivery_reviewer_repair.get('manager_reviewer_handoff_crosswalk_status')}`; legacy_chain_ready={delivery_reviewer_repair.get('legacy_canonical_chain_still_recognized')}.",
                f"- Operator-only residual preserved: gate=`{residual.get('only_remaining_gate')}`; agent_subjective_listening_performed={residual.get('agent_subjective_listening_performed')}; semantic_av_quality_certified={residual.get('semantic_av_quality_certified')}; av_sync_certified={residual.get('av_sync_certified')}.",
                f"- Boundary: {delivery_reviewer_repair.get('claim_boundary')}",
                "",
            ]
        )
    if active_hold_reconciliation.get("status") != "not_available":
        residual = active_hold_reconciliation.get("operator_only_residual") or {}
        manager_line = active_hold_reconciliation.get("manager_stage_complete_counter_evidence_line") or {}
        surface_bits = []
        for surface in active_hold_reconciliation.get("active_hold_surfaces") or []:
            surface_bits.append(
                f"{surface.get('path')}:{surface.get('line_number')} {surface.get('event_type')} fields={surface.get('fields')} token={surface.get('token')}"
            )
        lines.extend(
            [
                "## Delivery Reviewer active-hold reconciliation",
                "",
                f"- Active packet: `{active_hold_reconciliation.get('packet_dir')}`; status=`{active_hold_reconciliation.get('status')}`; accepted_for_manager_visible_delivery_sync={active_hold_reconciliation.get('accepted_for_manager_visible_delivery_sync')}; current_stage=`{active_hold_reconciliation.get('current_stage')}`; current_mission_id=`{active_hold_reconciliation.get('current_mission_id')}`.",
                f"- Active Reviewer verdict: `{active_hold_reconciliation.get('reviewer_verdict_path')}`; decision=`{active_hold_reconciliation.get('reviewer_decision')}`; independent={active_hold_reconciliation.get('reviewer_verdict_independent')}; active_hold_probe=`{active_hold_reconciliation.get('active_hold_probe_path')}`; INDEX=`{active_hold_reconciliation.get('index_path')}`.",
                f"- Active hold surfaces captured: {surface_bits}.",
                f"- Manager-stage complete counter-evidence: `{manager_line.get('path')}`:{manager_line.get('line_number')} `{manager_line.get('event_type')}` diagnostic=`{manager_line.get('diagnostic')}`; classification=`{active_hold_reconciliation.get('mismatch_classification')}`; raw_event_sha256_count={active_hold_reconciliation.get('raw_event_sha256_count')}.",
                f"- Operator-only residual preserved: gate=`{residual.get('only_remaining_gate')}`; agent_subjective_listening_performed={residual.get('agent_subjective_listening_performed')}; semantic_av_quality_certified={residual.get('semantic_av_quality_certified')}; av_sync_certified={residual.get('av_sync_certified')}.",
                f"- Boundary: {active_hold_reconciliation.get('claim_boundary')}",
                "",
            ]
        )
    if current_manager_hold_no_gap.get("status") != "not_available":
        residual = current_manager_hold_no_gap.get("operator_only_residual") or {}
        manager_stage = current_manager_hold_no_gap.get("manager_stage_decision") or {}
        manager_feedback = current_manager_hold_no_gap.get("manager_feedback") or {}
        planner_surface = current_manager_hold_no_gap.get("planner_hold_surface") or {}
        exact_gap = current_manager_hold_no_gap.get("exact_current_stage_reviewer_evidence_gap") or {}
        exact_gap_status = exact_gap.get("status") if isinstance(exact_gap, dict) else None
        exact_gap_paths = exact_gap.get("missing_paths") if isinstance(exact_gap, dict) else None
        exact_gap_fields = exact_gap.get("missing_or_unaccepted_manager_visible_fields") if isinstance(exact_gap, dict) else None
        lines.extend(
            [
                "## Current Manager-hold no-gap probe",
                "",
                f"- Packet: `{current_manager_hold_no_gap.get('packet_dir')}`; status=`{current_manager_hold_no_gap.get('status')}`; decisive_check_passed={current_manager_hold_no_gap.get('decisive_check_passed')}; project_local_reviewer_evidence_locator_schema_mismatch={current_manager_hold_no_gap.get('project_local_reviewer_evidence_locator_schema_mismatch')}; current_stage=`{current_manager_hold_no_gap.get('current_stage')}`; current_mission_id=`{current_manager_hold_no_gap.get('current_mission_id')}`.",
                f"- Reviewer boundary: reviewer_status=`{current_manager_hold_no_gap.get('reviewer_status')}`; decision=`{current_manager_hold_no_gap.get('reviewer_decision')}`; independent={current_manager_hold_no_gap.get('reviewer_verdict_independent')}; sealed_handoff_valid={current_manager_hold_no_gap.get('sealed_reviewer_handoff_source_valid')}; manager_visible_delivery_sync_ready={current_manager_hold_no_gap.get('manager_visible_delivery_sync_ready')}; verdict=`{current_manager_hold_no_gap.get('reviewer_verdict_path')}`; request=`{current_manager_hold_no_gap.get('reviewer_verdict_request_path')}`; crosswalk=`{current_manager_hold_no_gap.get('manager_reviewer_handoff_crosswalk_path')}`.",
                f"- Exact current-stage Reviewer-evidence gap: status=`{exact_gap_status}`; missing_paths={exact_gap_paths}; missing_or_unaccepted_fields={exact_gap_fields}.",
                f"- Live Manager stage: `{manager_stage.get('path')}`:{manager_stage.get('line_number')} diagnostic=`{manager_stage.get('diagnostic')}`; reason=`{manager_stage.get('reason')}`.",
                f"- Persisted hold token: `{manager_feedback.get('path')}`:{manager_feedback.get('line_number')} diagnostic=`{manager_feedback.get('diagnostic')}`; reason=`{manager_feedback.get('reason')}`.",
                f"- Lower-authority planner surface: `{planner_surface.get('path')}`:{planner_surface.get('line_number')} status=`{planner_surface.get('status')}`; reason=`{planner_surface.get('reason')}`.",
                f"- Operator-only residual preserved: gate=`{residual.get('only_remaining_gate')}`; agent_subjective_listening_performed={residual.get('agent_subjective_listening_performed')}; semantic_av_quality_certified={residual.get('semantic_av_quality_certified')}; av_sync_certified={residual.get('av_sync_certified')}. Manifests still missing {current_manager_hold_no_gap.get('missing_manifest_paths_count')} required current-hold paths.",
                f"- Boundary: {current_manager_hold_no_gap.get('claim_boundary')}",
                "",
            ]
        )
    lines.extend([
        "## Clean-room one-command local lifecycle",
        "",
        f"- Evidence: `{lifecycle['run_dir']}`.",
        f"- Status: `{lifecycle['status']}`; publication audit `{lifecycle['publication_audit_status']}`; deploy `{lifecycle['deploy_status']}`.",
        f"- Existing local FL2VA resource inspection: {lifecycle['model_file_count']} files, {lifecycle['model_total_bytes']} bytes.",
        f"- Boundary: {lifecycle['claim_boundary']}.",
        "",
        "## Final CPU/static/export/audit gates",
        "",
        f"- Overall gate status: `{final_gates.get('status', 'not_available')}`.",
        f"- CPU/static gate: `{final_gates.get('cpu_static_gate', {}).get('status', 'not_available')}`; evidence `{final_gates.get('cpu_static_gate', {}).get('summary_path')}`; summary tail={final_gates.get('cpu_static_gate', {}).get('summary_tail', [])}.",
        f"- Strict aggregation/export/publication audit gate: `{final_gates.get('decisive_export_audit_gate', {}).get('status', 'not_available')}`; evidence `{final_gates.get('decisive_export_audit_gate', {}).get('summary_path')}`; export_file_count={final_gates.get('decisive_export_audit_gate', {}).get('export_file_count')}; publication_audit={final_gates.get('decisive_export_audit_gate', {}).get('publication_audit_status')}; issues={final_gates.get('decisive_export_audit_gate', {}).get('publication_issue_count')}.",
        f"- Formal-N10 report-sync export/publication audit gate: `{final_gates.get('formal_n10_cpu_sync_export_audit_gate', {}).get('status', 'not_available')}`; evidence `{final_gates.get('formal_n10_cpu_sync_export_audit_gate', {}).get('summary_path')}`; export_file_count={final_gates.get('formal_n10_cpu_sync_export_audit_gate', {}).get('export_file_count')}; publication_audit={final_gates.get('formal_n10_cpu_sync_export_audit_gate', {}).get('publication_audit_status')}; issues={final_gates.get('formal_n10_cpu_sync_export_audit_gate', {}).get('publication_issue_count')}; reviewer_status={final_gates.get('formal_n10_cpu_sync_export_audit_gate', {}).get('reviewer_status')}; push_performed={final_gates.get('formal_n10_cpu_sync_export_audit_gate', {}).get('push_performed')}.",
        f"- Active-hold report-sync export/publication audit gate: `{final_gates.get('active_hold_sync_export_audit_gate', {}).get('status', 'not_available')}`; evidence `{final_gates.get('active_hold_sync_export_audit_gate', {}).get('summary_path')}`; strict_aggregation={final_gates.get('active_hold_sync_export_audit_gate', {}).get('strict_aggregation_status')}; export_status={final_gates.get('active_hold_sync_export_audit_gate', {}).get('export_status')}; export_file_count={final_gates.get('active_hold_sync_export_audit_gate', {}).get('export_file_count')}; publication_audit={final_gates.get('active_hold_sync_export_audit_gate', {}).get('publication_audit_status')}; issues={final_gates.get('active_hold_sync_export_audit_gate', {}).get('publication_issue_count')}; active_hold_reviewer=`{final_gates.get('active_hold_sync_export_audit_gate', {}).get('active_hold_reconciliation_reviewer_verdict')}`.",
        f"- Boundary: {final_gates.get('claim_boundary', 'gate evidence unavailable')}.",
        "",
        "## Reproducible Turbo quality-suite runner",
        "",
        f"- Config: `{summary['quality_suite_runner']['config_path']}`.",
        f"- Dry-run plan directory: `{summary['quality_suite_runner']['dry_run_plan_dir']}`.",
        "- Default behavior is fail-closed dry-run: it writes the prompt/seed/step matrix and operator commands without launching Docker/GPU/model inference.",
        "- Non-dry execution requires a fresh operator authorization, the `ARGUS_ALLOW_TURBO_QUALITY_SUITE=1` environment gate, and the explicit acknowledgement flag printed by the runner.",
        "",
        "## Sol / exact-kernel diagnostics",
        "",
        f"- AdaLN: `{sol['adaln']['status']}`; video MSE {sol['adaln']['video_mean_mse']}; audio cosine {sol['adaln']['audio_waveform_cosine']}; N={sol['adaln']['n']}; single-run gain {(sol['adaln']['single_run_speedup'] - 1.0) * 100.0}% below baseline warm CV {sol['adaln']['baseline_warm_cv_percent']}%.",
        f"- AdaLN disclosure preserved: `{sol['adaln']['harness_failure_disclosure']}`; original harness exit code {sol['adaln']['original_harness_exit_code']}; not an accepted N=10 speedup.",
        f"- RoPE: `{sol['rope']['status']}`; video MSE {sol['rope']['video_mean_mse']}; audio cosine {sol['rope']['audio_waveform_cosine']}.",
        f"- All-exact: `{sol['all_exact']['status']}`; video MSE {sol['all_exact']['video_mean_mse']}; audio cosine {sol['all_exact']['audio_waveform_cosine']}.",
        f"- SwiGLU: `{sol['swiglu']['status']}`; exact diagnostic output retained no accepted speedup gain.",
        f"- Toy Sol-Attn: `{sol['toy_sol_attn']['status']}`; dense median {sol['toy_sol_attn']['dense_median_ms']} ms, sparse median {sol['toy_sol_attn']['sparse_median_ms']} ms, dense/sparse median speedup {sol['toy_sol_attn']['speedup_dense_over_sparse_median']}.",
    ])
    if h3_sol.get("status") != "not_available":
        tele = h3_sol["telemetry"]
        resources = h3_sol["resource_summary"]
        lines.extend(
            [
                f"- H3 Sol-Attn r8: `{h3_sol['classification']}`; evidence `{h3_sol['evidence_path']}`; sparse candidates={tele['sparse_candidate_calls']}, sparse calls={tele['sparse_calls']}, dense calls={tele['dense_calls']}, fallback calls={tele['fallback_calls']}, density samples={tele['density_sample_count']}, materialized copies={tele['materialized_copy_calls']} / {tele['materialized_copy_bytes']} bytes.",
                f"- H3 Sol-Attn r8 HTTP/resource boundary: dense={h3_sol['http_time_total_s']['dense_h3_backend_reference']}s, opt-in={h3_sol['http_time_total_s']['sol_attn_opt_in']}s, dense/opt-in ratio={h3_sol['paired_http_ratio_dense_over_opt_in_not_speedup']} (diagnostic only, not a speedup); peak GPU memory={resources['peak_gpu_memory_mib']} MiB, peak temperature={resources['peak_temperature_c']} C, peak power={resources['peak_power_w']} W.",
                f"- H3 Sol-Attn r8 claim boundary: {h3_sol['claim_boundary']}.",
            ]
        )
        matched = h3_sol.get("matched_retest", {})
        formal_for_matched_boundary = h3_sol.get("formal_n10", {})
        if matched.get("status") == "proceed_to_formal_n10_candidate":
            matched_boundary = (
                "This N=3 route gate led to the later accepted formal N>=10 gate; the N=3 gate itself remains bounded route evidence, not formal N10 or a speedup/quality/fidelity claim."
                if formal_for_matched_boundary.get("status") == "accepted_formal_n10_same_gpu_sol_attn_speed_candidate"
                else "This is still only a bounded route gate, not formal N10 or a speedup/quality/fidelity claim."
            )
            lines.append(
                f"- R8 matched-workload retest route decision: `{matched['status']}`; evidence `{matched['evidence_path']}`; terminal_recheck=`{matched.get('terminal_recheck_evidence_path')}`; completed_pairs={matched.get('completed_pairs')}/{matched.get('requested_pairs')}; median_http_time_improvement={matched.get('median_http_time_improvement_pct')}%; threshold={matched.get('timing_threshold_pct')}%; supervisor_status={matched.get('supervisor_status')}; supervisor_return_code={matched.get('supervisor_return_code')}; posthoc_finalization_note=`{matched.get('posthoc_finalization_note')}`; n10_recommendation={matched.get('n10_recommendation')}. Reason: {matched.get('reason')}. {matched_boundary}"
            )
        elif matched.get("status") != "not_available":
            lines.append(
                f"- R8 matched-workload retest route decision: `{matched['status']}`; evidence `{matched['evidence_path']}`; supervisor_status={matched['supervisor_status']}; pid_alive={matched['supervisor_pid_alive']}; n10_recommendation={matched['n10_recommendation']}. Reason: {matched['reason']}"
            )
        formal = h3_sol.get("formal_n10", {})
        if formal.get("status") not in {None, "not_available"}:
            formal_evidence = formal.get("decision_path") or formal.get("status_path") or formal.get("source_run_dir")
            formal_boundary = (
                "Accepted only within the formal matched 5-step Sol-Attn opt-in lane; this is not BF16 fidelity, Turbo/DLO/DMD evidence, release approval, or human-auditory/semantic quality certification."
                if formal.get("status") == "accepted_formal_n10_same_gpu_sol_attn_speed_candidate"
                else "Nonterminal/incomplete formal evidence is not a Sol-Attn speedup, BF16 fidelity, release, or quality-equivalence claim."
            )
            lines.append(
                f"- R8 formal N>=10 matched-workload gate: `{formal.get('status')}`; evidence `{formal_evidence}`; requested_pairs={formal.get('requested_pairs')}; started_pairs={formal.get('started_pairs')}; completed_pairs={formal.get('completed_pairs')}; supervisor_status={formal.get('supervisor_status')}. Reason: {formal.get('reason')}. {formal_boundary}"
            )
    lines.extend([
        "",
        "## DMD / DMD2",
        "",
        f"- Evidence: `{dmd['evidence_path']}`.",
        f"- Status: **{dmd['status']}**.",
        "- DMD remains a no-go after Turbo unless a legal H3 DMD/DMD2 recipe/checkpoint, resource profile, and AV quality bar appear in future evidence.",
        "",
        "## Accepted / rejected / blocked matrix",
        "",
        "### Accepted",
    ])
    for item in summary["claims"]["accepted"]:
        lines.append(f"- {item['claim']} Evidence: {', '.join('`' + e + '`' for e in item['evidence'])}. Limit: {item['limits']}")
    lines.extend(["", "### Rejected"])
    for item in summary["claims"]["rejected"]:
        lines.append(f"- {item['claim']} Reason: {item['reason']} Evidence: {', '.join('`' + e + '`' for e in item['evidence'])}.")
    lines.extend(["", "### Blocked"])
    for item in summary["claims"]["blocked"]:
        lines.append(f"- {item['claim']} Status: {item['status']} Evidence: {', '.join('`' + e + '`' for e in item['evidence'])}.")
    lines.extend(["", "### Pending"])
    for item in summary["claims"].get("pending", []):
        lines.append(f"- {item['claim']} Status: {item['status']} Evidence: {', '.join('`' + e + '`' for e in item['evidence'])}.")
    lines.extend(
        [
            "",
            "## Reproduction / final gate commands",
            "",
        ]
    )
    for command in summary["operator_commands"]:
        lines.append(f"```bash\n{command}\n```")
    lines.extend(
        [
            "",
            "## Test accounting",
            "",
            "Skipped GPU placeholders are not counted as passed coverage. This delivery counts only CPU/static tests and the metadata verifier command reported by the engineer.",
            "",
        ]
    )
    return "\n".join(lines)


def default_manifest_paths(repo_root: Path, evidence_root: Path, out_path: Path | None, report_path: Path | None) -> list[Path]:
    timing_run_id = _read_latest_run_id(evidence_root / TURBO_TIMING_LATEST_REL)
    quality_run_id = _read_latest_run_id(evidence_root / TURBO_QUALITY_LATEST_REL)
    quality_run_dir = evidence_root / TURBO_QUALITY_RUNS_REL / quality_run_id
    lifecycle = summarize_local_lifecycle(evidence_root)
    lifecycle_paths = [evidence_root / lifecycle[key] for key in ("summary_path", "model_prepare_path", "deploy_path")]
    r8_paths: list[Path] = []
    delivery_dir = evidence_root / DELIVERY_REL
    r8_ingest_dir = _latest_prefixed_dir(delivery_dir, R8_SOL_ATTN_CPU_INGEST_PREFIX)
    if r8_ingest_dir is not None:
        r8_paths.extend(
            [
                r8_ingest_dir / R8_SOL_ATTN_CLASSIFICATION_FILE,
                r8_ingest_dir / "README.md",
                r8_ingest_dir / "performance_summary.json",
            ]
        )
    r8_matched_terminal_dir = _latest_prefixed_dir(evidence_root / "sol_engine_port", R8_MATCHED_RETEST_TERMINAL_RUN_PREFIX)
    if r8_matched_terminal_dir is not None and (r8_matched_terminal_dir / R8_MATCHED_RETEST_DECISION_FILE).is_file():
        r8_paths.extend(r8_matched_terminal_dir / name for name in R8_MATCHED_RETEST_TERMINAL_FILES)
        posthoc_note = r8_matched_terminal_dir / "posthoc_finalization_note.json"
        if posthoc_note.is_file():
            r8_paths.append(posthoc_note)
        terminal_recheck_dir = _latest_prefixed_dir(delivery_dir, R8_MATCHED_RETEST_TERMINAL_RECHECK_PREFIX)
        if terminal_recheck_dir is not None:
            r8_paths.append(terminal_recheck_dir / R8_MATCHED_RETEST_TERMINAL_RECHECK_FILE)
    else:
        r8_matched_dir = _latest_prefixed_dir(delivery_dir, R8_MATCHED_RETEST_INSPECTION_PREFIX)
        if r8_matched_dir is not None:
            r8_paths.append(r8_matched_dir / R8_MATCHED_RETEST_INSPECTION_FILE)
    r8_formal_dir = _latest_prefixed_dir(evidence_root / "sol_engine_port", R8_FORMAL_N10_PREFIX)
    if r8_formal_dir is not None:
        for name in (
            "formal_n10_supervisor_status.json",
            "formal_n10_supervisor_stdout.log",
            "FORMAL_N10_RUN_REPORT.md",
            *R8_FORMAL_N10_TERMINAL_FILES,
        ):
            candidate = r8_formal_dir / name
            if candidate.is_file():
                r8_paths.append(candidate)
    final_gate_paths: list[Path] = []
    cpu_gate_dir = _latest_prefixed_dir(delivery_dir, FINAL_CPU_STATIC_GATE_PREFIX)
    if cpu_gate_dir is not None:
        final_gate_paths.extend(
            [
                cpu_gate_dir / "summary.txt",
                cpu_gate_dir / "commands.txt",
                cpu_gate_dir / "full_pytest.log",
                cpu_gate_dir / "verify_run.log",
                cpu_gate_dir / "turbo_quality_dry_run.log",
            ]
        )
    decisive_gate_dir = _latest_prefixed_dir(delivery_dir, FINAL_DECISIVE_EXPORT_AUDIT_PREFIX)
    if decisive_gate_dir is not None:
        final_gate_paths.extend(
            [
                decisive_gate_dir / "summary.json",
                decisive_gate_dir / "commands.txt",
                decisive_gate_dir / "strict_aggregation.log",
                decisive_gate_dir / "publication_audit.json",
                decisive_gate_dir / "export_build.json",
            ]
        )
    formal_sync_gate_dir = _latest_prefixed_dir(delivery_dir, FORMAL_N10_CPU_SYNC_EXPORT_AUDIT_PREFIX)
    if formal_sync_gate_dir is not None:
        final_gate_paths.extend(
            [
                formal_sync_gate_dir / "summary.json",
                formal_sync_gate_dir / "commands.txt",
                formal_sync_gate_dir / "export_publication_tests.log",
                formal_sync_gate_dir / "publication_audit.json",
                formal_sync_gate_dir / "export_build.json",
            ]
        )
    active_hold_sync_gate_dir = _latest_prefixed_dir(delivery_dir, ACTIVE_HOLD_SYNC_EXPORT_AUDIT_PREFIX)
    if active_hold_sync_gate_dir is not None:
        final_gate_paths.extend(
            [
                active_hold_sync_gate_dir / "summary.json",
                active_hold_sync_gate_dir / "commands.txt",
                active_hold_sync_gate_dir / "strict_aggregation.log",
                active_hold_sync_gate_dir / "strict_aggregation.stderr",
                active_hold_sync_gate_dir / "publication_audit.json",
                active_hold_sync_gate_dir / "publication_audit.stderr",
                active_hold_sync_gate_dir / "export_build.json",
                active_hold_sync_gate_dir / "export_build.stderr",
            ]
        )
    operator_gate_paths: list[Path] = []
    operator_gate_latest = evidence_root / TURBO_OPERATOR_GATE_LATEST_REL
    operator_gate_dir = _resolve_latest_packet_dir(evidence_root, repo_root, TURBO_OPERATOR_GATE_LATEST_REL)
    if operator_gate_latest.is_file():
        operator_gate_paths.append(operator_gate_latest)
    if operator_gate_dir is not None:
        operator_gate_paths.extend(operator_gate_dir / name for name in TURBO_OPERATOR_GATE_PACKET_FILES)
        operator_gate_paths.extend(sorted(operator_gate_dir.glob("*consistency_check*.json")))
        operator_gate_paths.extend(sorted(operator_gate_dir.glob("*recognition_check*.json")))
        stage_addendum = operator_gate_dir / "stage_closing_addendum.md"
        if stage_addendum.is_file():
            operator_gate_paths.append(stage_addendum)

    delivery_reviewer_repair_paths: list[Path] = []
    delivery_reviewer_repair_latest = evidence_root / DELIVERY_REVIEWER_RECOGNITION_REPAIR_LATEST_REL
    delivery_reviewer_repair_dir = _resolve_latest_packet_dir(
        evidence_root,
        repo_root,
        DELIVERY_REVIEWER_RECOGNITION_REPAIR_LATEST_REL,
    )
    if delivery_reviewer_repair_latest.is_file():
        delivery_reviewer_repair_paths.append(delivery_reviewer_repair_latest)
    if delivery_reviewer_repair_dir is not None:
        for pattern in DELIVERY_REVIEWER_RECOGNITION_REPAIR_PACKET_FILES:
            delivery_reviewer_repair_paths.extend(sorted(delivery_reviewer_repair_dir.glob(pattern)))

    active_hold_reconciliation_paths: list[Path] = []
    active_hold_reconciliation_latest = evidence_root / DELIVERY_REVIEWER_ACTIVE_HOLD_RECONCILIATION_LATEST_REL
    active_hold_reconciliation_dir = _resolve_latest_packet_dir(
        evidence_root,
        repo_root,
        DELIVERY_REVIEWER_ACTIVE_HOLD_RECONCILIATION_LATEST_REL,
    )
    if active_hold_reconciliation_latest.is_file():
        active_hold_reconciliation_paths.append(active_hold_reconciliation_latest)
    if active_hold_reconciliation_dir is not None:
        for pattern in DELIVERY_REVIEWER_ACTIVE_HOLD_RECONCILIATION_PACKET_FILES:
            active_hold_reconciliation_paths.extend(sorted(active_hold_reconciliation_dir.glob(pattern)))

    current_manager_hold_no_gap_paths: list[Path] = []
    current_manager_hold_no_gap_latest = evidence_root / CURRENT_MANAGER_HOLD_NO_GAP_LATEST_REL
    current_manager_hold_no_gap_dir = _resolve_latest_packet_dir(
        evidence_root,
        repo_root,
        CURRENT_MANAGER_HOLD_NO_GAP_LATEST_REL,
    )
    if current_manager_hold_no_gap_latest.is_file():
        current_manager_hold_no_gap_paths.append(current_manager_hold_no_gap_latest)
    if current_manager_hold_no_gap_dir is not None:
        for pattern in CURRENT_MANAGER_HOLD_NO_GAP_PACKET_FILES:
            current_manager_hold_no_gap_paths.extend(sorted(current_manager_hold_no_gap_dir.glob(pattern)))

    candidates = [
        repo_root / "release/github_release_manifest.json",
        repo_root / "tools/build_github_release_tree.py",
        repo_root / "tools/publication_audit.py",
        repo_root / "tools/turbo_quality_suite_runner.py",
        repo_root / "tools/turbo_quality_suite_analyze.py",
        repo_root / "tools/minimax_h3_a6000_performance_report.py",
        repo_root / "tools/build_periodic_progress.py",
        repo_root / "tools/argus_ir04_aggregate.py",
        repo_root / "tools/check_turbo_operator_gate_recognition.py",
        repo_root / "tools/argus_h3_verifier.py",
        repo_root / "tools/verify_run.py",
        repo_root / "scripts/a6000_one_command.sh",
        repo_root / "scripts/run_a6000_fidelity_baseline_repeats.sh",
        repo_root / "scripts/run_a6000_adaln_candidate_50step.sh",
        repo_root / "scripts/run_a6000_turbo_timing_repeats.sh",
        repo_root / "ports/minimax_h3_a6000/README.md",
        repo_root / "ports/minimax_h3_a6000/NOTICE",
        repo_root / "ports/minimax_h3_a6000/UPSTREAM.md",
        repo_root / "ports/minimax_h3_a6000/gpu_exact_kernel_test.py",
        repo_root / "ports/minimax_h3_a6000/gpu_exact_kernel_bench.py",
        repo_root / "ports/minimax_h3_a6000/gpu_sol_attn_sm86_harness.py",
        repo_root / "ports/minimax_h3_a6000/patches/vllm_omni_h3_a6000_opt_in.patch",
        repo_root / "ports/minimax_h3_a6000/src/minimax_h3_a6000/exact_kernels.py",
        repo_root / "ports/minimax_h3_a6000/src/minimax_h3_a6000/reference_ops.py",
        repo_root / "ports/minimax_h3_a6000/src/minimax_h3_a6000/sol_attn_backend.py",
        repo_root / "ports/minimax_h3_a6000/src/minimax_h3_a6000/sol_attn_triton_sm86.py",
        repo_root / "code/pytest.py",
        repo_root / "tests/test_github_release_tree_builder.py",
        repo_root / "tests/test_publication_audit.py",
        repo_root / "tests/test_minimax_h3_a6000_performance_report.py",
        repo_root / "tests/test_turbo_quality_suite_runner.py",
        repo_root / "tests/test_argus_ir04_aggregate.py",
        repo_root / "tests/test_turbo_operator_gate_recognition_check.py",
        repo_root / "tests/test_verify_run.py",
        evidence_root / QUALITY_CONFIG_REL,
        evidence_root / QUALITY_DRY_RUN_REL / "quality_suite_plan.json",
        evidence_root / QUALITY_DRY_RUN_REL / "quality_suite_requests.jsonl",
        evidence_root / QUALITY_DRY_RUN_REL / "operator_commands.sh",
        repo_root / "technical_report/progress_update.md",
        repo_root / "technical_report/minimax_h3_a6000_performance.md",
        evidence_root / BASELINE_REL,
        evidence_root / "baseline_a6000/baseline_certification_v1_superseded_notice.md",
        evidence_root / TURBO_TIMING_LATEST_REL,
        evidence_root / TURBO_TIMING_REPEATS_REL / timing_run_id / "timing_summary.json",
        evidence_root / TURBO_TIMING_REPEATS_REL / timing_run_id / "merge_manifest.json",
        evidence_root / TURBO_TIMING_REPEATS_REL / timing_run_id / "host_orphan_recovery_note.md",
        evidence_root / TURBO_TIMING_REPEATS_REL / timing_run_id / "status.txt",
        evidence_root / TURBO_QUALITY_LATEST_REL,
        quality_run_dir / "quality_suite_plan.json",
        quality_run_dir / "quality_suite_analysis.json",
        quality_run_dir / "baseline_seed0_quality_comparison.json",
        quality_run_dir / "audio_energy_envelopes.json",
        quality_run_dir / "human_review.md",
        *(sorted((quality_run_dir / "contacts").glob("REVIEW_*.jpg"))),
        evidence_root / f"{GPU2_BRINGUP_RUN_REL}/turbo_summary.json",
        evidence_root / f"{GPU2_BRINGUP_RUN_REL}/merge_manifest.json",
        evidence_root / DMD_REL,
        evidence_root / ADALN_QUALITY_REL,
        evidence_root / ADALN_POSTHOC_REL,
        evidence_root / ADALN_VERDICT_REL,
        evidence_root / ROPE_QUALITY_REL,
        evidence_root / ALL_EXACT_QUALITY_REL,
        evidence_root / SWIGLU_QUALITY_REL,
        evidence_root / SWIGLU_HTTP_REL,
        evidence_root / SOL_ATTN_RESULT_REL,
        *r8_paths,
        *lifecycle_paths,
        *final_gate_paths,
        *operator_gate_paths,
        *delivery_reviewer_repair_paths,
        *active_hold_reconciliation_paths,
        *current_manager_hold_no_gap_paths,
    ]
    if out_path is not None:
        candidates.append(out_path)
    if report_path is not None:
        candidates.append(report_path)
    return candidates


def preview_manifest(path: Path, *, paths: list[Path], repo_root: Path, assume_existing: set[Path] | None = None) -> dict[str, Any]:
    assume_resolved = {item.resolve() for item in (assume_existing or set())}
    existing: list[str] = []
    missing: list[str] = []
    seen: set[Path] = set()
    for candidate in paths:
        resolved = candidate.resolve()
        if resolved == path.resolve() or resolved in seen:
            continue
        seen.add(resolved)
        if candidate.is_file() or resolved in assume_resolved:
            existing.append(repo_rel(candidate, repo_root))
        else:
            missing.append(repo_rel(candidate, repo_root))
    return {"entry_count": len(existing), "missing_optional_paths": sorted(missing)}


def write_manifest(path: Path, *, paths: list[Path], repo_root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    seen: set[Path] = set()
    for candidate in paths:
        resolved = candidate.resolve()
        if resolved == path.resolve() or resolved in seen:
            continue
        seen.add(resolved)
        if not candidate.is_file():
            missing.append(repo_rel(candidate, repo_root))
            continue
        entries.append(
            {
                "path": repo_rel(candidate, repo_root),
                "size_bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
        )
    payload = {
        "schema_version": "argus-ir04-delivery-package-manifest-v1",
        "hash_algorithm": "sha256",
        "self_included": False,
        "scope_note": "Generated tooling, CPU/static tests, report/summary artifacts, small evidence JSON/Markdown, and review contact sheets only; model weights and generated MP4 outputs are intentionally excluded.",
        "entries": sorted(entries, key=lambda item: item["path"]),
        "missing_optional_paths": sorted(missing),
    }
    write_json(path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate ARGUS-IR-04 evidence without running model inference.")
    parser.add_argument("--input", type=Path, required=True, help="technical_report/evidence/minimax_h3_desktop root")
    parser.add_argument("--out", type=Path, required=True, help="Summary JSON output path")
    parser.add_argument("--strict", action="store_true", help="Fail on missing/inconsistent evidence (default behavior)")
    parser.add_argument("--report-out", type=Path, help="Optional draft technical report Markdown path")
    parser.add_argument("--manifest-out", type=Path, help="Optional scoped sha256 package manifest path")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root for relative paths")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    evidence_root = args.input if args.input.is_absolute() else repo_root / args.input
    out_path = args.out if args.out.is_absolute() else repo_root / args.out
    report_path = args.report_out if args.report_out is None or args.report_out.is_absolute() else repo_root / args.report_out
    manifest_path = args.manifest_out if args.manifest_out is None or args.manifest_out.is_absolute() else repo_root / args.manifest_out
    try:
        summary = aggregate_evidence(evidence_root, repo_root=repo_root)
        manifest_paths: list[Path] | None = None
        if manifest_path is not None:
            manifest_paths = default_manifest_paths(repo_root, evidence_root, out_path, report_path)
            assumed = {out_path}
            if report_path is not None:
                assumed.add(report_path)
            preview = preview_manifest(manifest_path, paths=manifest_paths, repo_root=repo_root, assume_existing=assumed)
            summary["package_manifest"] = {
                "path": repo_rel(manifest_path, repo_root),
                "entry_count": preview["entry_count"],
                "missing_optional_paths": preview["missing_optional_paths"],
            }
        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(report_markdown(summary), encoding="utf-8")
        write_json(out_path, summary)
        if manifest_path is not None and manifest_paths is not None:
            manifest = write_manifest(manifest_path, paths=manifest_paths, repo_root=repo_root)
            summary["package_manifest"]["entry_count"] = len(manifest["entries"])
            summary["package_manifest"]["missing_optional_paths"] = manifest["missing_optional_paths"]
        print(f"PASS aggregation summary={repo_rel(out_path, repo_root)}")
        if report_path is not None:
            print(f"report={repo_rel(report_path, repo_root)}")
        if manifest_path is not None:
            print(f"manifest={repo_rel(manifest_path, repo_root)}")
        return 0
    except AggregationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
