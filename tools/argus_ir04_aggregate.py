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
    pending_human_review = "PENDING" in human_review
    return {
        "run_id": run_id,
        "run_dir": rel(run_dir, root),
        "latest_pointer": rel(latest_path, root),
        "analysis_path": rel(analysis_path, root),
        "baseline_seed0_comparison_path": rel(baseline_comparison_path, root),
        "audio_energy_envelopes_path": rel(audio_energy_path, root),
        "human_review_path": rel(human_review_path, root),
        "review_contact_sheet_count": len(contact_sheets),
        "case_count": analysis["case_count"],
        "pair_count": analysis["pair_count"],
        "steps": steps,
        "all_cases_structural_av_pass": True,
        "quality_certification": analysis.get("quality_certification"),
        "status": analysis["status"],
        "pending_human_review": pending_human_review,
        "human_auditory_listening": "pending",
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
        "quality_certification": "structural_av_pass_semantic_quality_not_certified_human_listening_pending",
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
    }


def aggregate_evidence(root: Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    repo_root = repo_root or root.parent.parent.parent
    baseline = summarize_baseline(root)
    turbo = summarize_turbo(root, baseline)
    dmd = summarize_dmd(root)
    sol = summarize_sol(root, baseline)
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
                    "limits": "Practical approximation only; semantic AV quality and human auditory listening remain pending; 8-step is the default candidate and 4-step is ultra-fast quality-cost experimental.",
                },
                {
                    "claim": "AdaLN is an N=1 exact-output candidate only; the original harness-tail failure is preserved and the single-run benefit is below baseline warm-run noise.",
                    "track": Practical,
                    "evidence": [ADALN_QUALITY_REL, ADALN_POSTHOC_REL, ADALN_VERDICT_REL],
                    "limits": "Not an accepted N=10 speedup and not deployed as a certified fidelity path.",
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
                    "claim": "Turbo semantic AV quality and human-auditory quality are fully certified.",
                    "status": "blocked_human_auditory_listening_pending_semantic_quality_not_certified",
                    "evidence": [TURBO_QUALITY_LATEST_REL, f"{TURBO_QUALITY_RUNS_REL}/{turbo['quality_suite']['run_id']}/quality_suite_analysis.json", f"{TURBO_QUALITY_RUNS_REL}/{turbo['quality_suite']['run_id']}/human_review.md"],
                },
            ],
        },
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
            "PYTHONPATH=code:. python3 -m pytest -q tests ports/minimax_h3_a6000/tests",
            "python3 tools/verify_run.py tests/fixtures/minimal_av_case/run_record.json",
            f"python3 tools/turbo_quality_suite_runner.py --dry-run --config {repo_rel(quality_config_path, repo_root)} --out {repo_rel(quality_dry_run_dir, repo_root)}",
            f"python3 tools/argus_ir04_aggregate.py --strict --input {repo_rel(root, repo_root)} --out technical_report/evidence/minimax_h3_desktop/delivery/argus_ir04_delivery_summary.json --report-out technical_report/final_technical_report.md --manifest-out technical_report/evidence/minimax_h3_desktop/delivery/package_manifest.json",
        ],
    }
    return summary


def report_markdown(summary: dict[str, Any]) -> str:
    baseline = summary["lanes"][Fidelity]["baseline"]
    turbo = summary["lanes"][Practical]["turbo_merged"]
    sol = summary["lanes"][Practical]["sol_exact_and_sol_attn"]
    dmd = summary["lanes"][Practical]["dmd"]
    lines = [
        "# ARGUS-IR-04 Final Technical Report",
        "",
        "Status: **final evidence integration, not quality-complete**. This report is evidence-grounded and CPU/static-generated; it does not add measurements, run inference, or publish results.",
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
        "- Human auditory listening remains pending; semantic AV quality is not certified.",
        f"- GPU2 smoke scope: {turbo['gpu2_bringup_scope']}.",
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
    ]
    for item in summary["claims"]["accepted"]:
        lines.append(f"- {item['claim']} Evidence: {', '.join('`' + e + '`' for e in item['evidence'])}. Limit: {item['limits']}")
    lines.extend(["", "### Rejected"])
    for item in summary["claims"]["rejected"]:
        lines.append(f"- {item['claim']} Reason: {item['reason']} Evidence: {', '.join('`' + e + '`' for e in item['evidence'])}.")
    lines.extend(["", "### Blocked"])
    for item in summary["claims"]["blocked"]:
        lines.append(f"- {item['claim']} Status: {item['status']} Evidence: {', '.join('`' + e + '`' for e in item['evidence'])}.")
    lines.extend(
        [
            "",
            "## Exact operator commands left for follow-up",
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
    candidates = [
        repo_root / "tools/turbo_quality_suite_runner.py",
        repo_root / "tools/turbo_quality_suite_analyze.py",
        repo_root / "tools/argus_ir04_aggregate.py",
        repo_root / "tools/argus_h3_verifier.py",
        repo_root / "tools/verify_run.py",
        repo_root / "scripts/run_a6000_fidelity_baseline_repeats.sh",
        repo_root / "scripts/run_a6000_adaln_candidate_50step.sh",
        repo_root / "scripts/run_a6000_turbo_timing_repeats.sh",
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
        repo_root / "tests/test_turbo_quality_suite_runner.py",
        repo_root / "tests/test_argus_ir04_aggregate.py",
        repo_root / "tests/test_verify_run.py",
        evidence_root / QUALITY_CONFIG_REL,
        evidence_root / QUALITY_DRY_RUN_REL / "quality_suite_plan.json",
        evidence_root / QUALITY_DRY_RUN_REL / "quality_suite_requests.jsonl",
        evidence_root / QUALITY_DRY_RUN_REL / "operator_commands.sh",
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
