#!/usr/bin/env python3
"""CPU-only MiniMax-H3 A6000 performance report generator.

This tool reads already-written local evidence and emits a Markdown report. It
must not run Docker, touch GPUs, load models, download data, or mutate live
experiment evidence. Missing optional evidence is represented explicitly as
``pending`` instead of estimated.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "argus-minimax-h3-a6000-performance-report-v1"
DEFAULT_SCHEMA_REL = Path("schemas/minimax_h3_a6000_performance_report.schema.json")

FIDELITY = "fidelity_bf16_exact"
PRACTICAL = "practical_disclosed_approx"

BASELINE_CERT_REL = Path("baseline_a6000/baseline_certification.json")
BASELINE_CONTRACT_REL = Path("baseline_a6000/baseline_contract.json")
TURBO_TIMING_LATEST_REL = Path("turbo_merged/timing_repeats/LATEST_RUN_ID")
TURBO_TIMING_REPEATS_REL = Path("turbo_merged/timing_repeats")
TURBO_QUALITY_LATEST_REL = Path("turbo_merged/LATEST_QUALITY_SUITE_RUN_ID")
TURBO_QUALITY_RUNS_REL = Path("turbo_merged/quality_suite_runs")
DMD_REL = Path("dmd_primary_source_note.md")
DLO_PLAN_REL = Path("dlo_autotune/resident_layer_candidates.json")
DLO_RUNS_REL = Path("dlo_autotune/runs")
DLO_STATE_REL = Path("dlo_autotune/detached_continuation")
EXACT_LATEST_REL = Path("sol_engine_port/LATEST_GPU_EXACT_DIR")
R5_ABLATION_LATEST_REL = Path("sol_engine_port/LATEST_R5_ABLATION_DIR")
SOL_ATTN_LATEST_REL = Path("sol_engine_port/LATEST_SOL_ATTN_GPU_DIR")
SOL_ATTN_SUPERVISOR_REL = Path("sol_engine_port/sol_attn_gpu2_supervisor")
R8_MATCHED_RETEST_TERMINAL_GLOB = "sol_attn_h3_matched_retest_r8_n3_*"
R8_MATCHED_RETEST_TERMINAL_RECHECK_GLOB = "r8_matched_retest_terminal_recheck_*"
R8_MATCHED_RETEST_NONTERMINAL_GLOBS = ("r8_matched_retest_inspection_*", "r8_matched_retest_nonterminal_inspection_*")
R8_FORMAL_N10_GLOB = "sol_attn_h3_formal_n10_r8_n*_*"
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

JsonDict = dict[str, Any]


class ReportError(Exception):
    """Raised when the report payload cannot be validated against the schema."""


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _evidence_rel(path: Path, evidence_root: Path) -> str:
    try:
        return path.resolve().relative_to(evidence_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_artifact_path(path_text: str | None, *, evidence_root: Path, repo_root: Path) -> Path | None:
    if not path_text:
        return None
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate
    repo_candidate = repo_root / candidate
    if repo_candidate.exists():
        return repo_candidate
    evidence_candidate = evidence_root / candidate
    return evidence_candidate


def _section(status: str, *, evidence: list[str] | None = None, data: JsonDict | None = None, notes: list[str] | None = None, reason: str | None = None) -> JsonDict:
    payload: JsonDict = {
        "status": status,
        "evidence": sorted(set(evidence or [])),
        "data": data or {},
        "notes": notes or [],
    }
    if reason:
        payload["reason"] = reason
    return payload


def _load_json(path: Path) -> tuple[JsonDict | None, str | None]:
    if not path.is_file():
        return None, f"missing evidence: {path.as_posix()}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON in {path.as_posix()}: {exc}"
    if not isinstance(data, dict):
        return None, f"expected JSON object in {path.as_posix()}"
    return data, None


def _read_text(path: Path) -> tuple[str | None, str | None]:
    if not path.is_file():
        return None, f"missing evidence: {path.as_posix()}"
    return path.read_text(encoding="utf-8"), None


def _formal_pair_completed(pair_dir: Path) -> bool:
    """Return True when a formal-N pair has terminal per-pair success evidence."""
    if (pair_dir / "decision.json").is_file():
        return True
    exit_path = pair_dir.parent / f"{pair_dir.name}.exit_code"
    try:
        return exit_path.read_text(encoding="utf-8").strip() == "0"
    except FileNotFoundError:
        return False


def _formal_same_expected_gpu(decision: JsonDict) -> bool | None:
    if isinstance(decision.get("same_expected_gpu"), bool):
        return bool(decision["same_expected_gpu"])
    gpu = decision.get("same_baseline_physical_gpu_evidence")
    if isinstance(gpu, dict) and isinstance(gpu.get("same_expected_gpu"), bool):
        return bool(gpu["same_expected_gpu"])
    return None


def _formal_raw_classification(decision: JsonDict) -> str | None:
    value = decision.get("raw_matched_classification") or decision.get("raw_matched_retest_classification")
    return value if isinstance(value, str) else None


def _read_pointer(path: Path) -> tuple[str | None, str | None]:
    text, err = _read_text(path)
    if err:
        return None, err
    assert text is not None
    value = text.strip()
    if not value or "/" in value or ".." in value:
        return None, f"invalid run-id pointer in {path.as_posix()}: {value!r}"
    return value, None


def _maybe_number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _stats_subset(stats: Any) -> JsonDict:
    if not isinstance(stats, dict):
        return {}
    keys = ("n", "mean_s", "median_s", "cv_percent", "sample_std_s", "min_s", "max_s")
    return {key: stats[key] for key in keys if key in stats}


def _resource_subset(resource: Any) -> JsonDict:
    if not isinstance(resource, dict):
        return {}
    aliases = {
        "peak_gpu_memory_mib": ("max_gpu_mem_mib", "max_gpu_memory_used_mib", "peak_gpu_mem_mib"),
        "peak_host_memory_gib": ("max_host_used_gib", "max_host_memory_used_gib", "peak_host_memory_gib"),
        "peak_temperature_c": ("max_temp_c", "max_temperature_c", "peak_temperature_c"),
        "peak_power_w": ("max_power_w", "peak_power_w"),
        "samples": ("samples",),
    }
    out: JsonDict = {}
    for canonical, candidates in aliases.items():
        for key in candidates:
            if key in resource:
                out[canonical] = resource[key]
                break
    return out


def _median_speedups(benchmarks: Any) -> list[JsonDict]:
    rows: list[JsonDict] = []
    if not isinstance(benchmarks, dict):
        return rows
    for name, bench in sorted(benchmarks.items()):
        if not isinstance(bench, dict):
            continue
        eager = bench.get("pytorch_eager_ms")
        triton = bench.get("triton_candidate_ms")
        if not isinstance(eager, list) or not isinstance(triton, list) or not eager or not triton:
            continue
        eager_values = [float(x) for x in eager]
        triton_values = [float(x) for x in triton]
        eager_median = statistics.median(eager_values)
        triton_median = statistics.median(triton_values)
        rows.append(
            {
                "kernel": name,
                "pytorch_eager_median_ms": eager_median,
                "triton_candidate_median_ms": triton_median,
                "speedup_eager_over_triton_median": eager_median / triton_median if triton_median else None,
                "n": min(len(eager_values), len(triton_values)),
            }
        )
    return rows


def _parse_http_time_total(text: str | None) -> float | None:
    if text is None:
        return None
    match = re.search(r"time_total_s=([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        return None
    return float(match.group(1))


def _section_status_from_parts(*sections: JsonDict) -> str:
    statuses = {section.get("status") for section in sections if section}
    if "invalid" in statuses:
        return "invalid"
    if "present" in statuses and "pending" in statuses:
        return "partial"
    if "partial" in statuses:
        return "partial"
    if "present" in statuses:
        return "present"
    return "pending"


def collect_workload(evidence_root: Path, repo_root: Path) -> JsonDict:
    evidence: list[str] = []
    notes: list[str] = []
    contract_path = evidence_root / BASELINE_CONTRACT_REL
    contract, err = _load_json(contract_path)
    if err is None and contract is not None:
        evidence.append(BASELINE_CONTRACT_REL.as_posix())
        workload = contract.get("workload") if isinstance(contract.get("workload"), dict) else {}
        data: JsonDict = {
            "task": workload.get("task"),
            "width": workload.get("width"),
            "height": workload.get("height"),
            "duration_seconds": workload.get("duration_seconds"),
            "expected_frames": workload.get("expected_frames"),
            "fps": workload.get("fps"),
            "audio_sample_rate_hz": workload.get("audio_sample_rate_hz"),
            "audio_channels": workload.get("audio_channels"),
            "baseline_num_inference_steps": workload.get("num_inference_steps"),
            "seed": workload.get("seed"),
            "prompt_sha256": workload.get("prompt_sha256"),
            "prompt_source": workload.get("prompt_source"),
            "partition": contract.get("partition"),
            "model_repo": contract.get("model_repo"),
            "checkpoint_revision": contract.get("checkpoint_revision"),
            "runtime_source_commit": contract.get("runtime_source_commit"),
        }
        if data.get("prompt_source") and "sol-engine" in str(data["prompt_source"]).lower():
            data["attribution"] = "Workload/prompt source preserves NVLabs/Sana Sol-Engine team attribution."
        else:
            notes.append("Prompt source attribution was not explicit in the contract.")
        return _section("present", evidence=evidence, data=data, notes=notes)

    # Fallback to the first AV validation referenced by timing evidence. This is
    # still evidence-derived; duration is marked as derived when computed.
    timing_run_id, ptr_err = _read_pointer(evidence_root / TURBO_TIMING_LATEST_REL)
    if ptr_err is None and timing_run_id:
        timing_path = evidence_root / TURBO_TIMING_REPEATS_REL / timing_run_id / "timing_summary.json"
        timing, timing_err = _load_json(timing_path)
        if timing_err is None and timing is not None:
            evidence.append(_evidence_rel(timing_path, evidence_root))
            for step in ("8", "4"):
                runs = ((timing.get("schedules") or {}).get(step) or {}).get("runs")
                if isinstance(runs, list) and runs:
                    av_path = _resolve_artifact_path(runs[0].get("av_validation"), evidence_root=evidence_root, repo_root=repo_root)
                    if av_path is not None:
                        av, av_err = _load_json(av_path)
                        if av_err is None and av is not None:
                            evidence.append(_evidence_rel(av_path, evidence_root))
                            fps = av.get("fps") or av.get("average_rate")
                            fps_num = _maybe_number(fps)
                            frames = av.get("decoded_video_frames")
                            duration = None
                            if fps_num and frames:
                                duration = float(frames) / float(fps_num)
                                notes.append("Duration is derived from decoded_video_frames / FPS because baseline contract is absent.")
                            data = {
                                "width": av.get("width"),
                                "height": av.get("height"),
                                "duration_seconds": duration,
                                "expected_frames": frames,
                                "fps": fps_num,
                                "audio_sample_rate_hz": av.get("audio_sample_rate_hz") or av.get("audio_sample_rate"),
                                "audio_channels": av.get("audio_channels"),
                                "baseline_num_inference_steps": None,
                                "attribution": "pending: baseline contract with NVLabs/Sana Sol-Engine prompt source is absent",
                            }
                            return _section("partial", evidence=evidence, data=data, notes=notes)
    return _section("pending", reason=err or ptr_err or "missing baseline contract and AV fallback evidence", notes=notes)


def collect_baseline(evidence_root: Path) -> JsonDict:
    path = evidence_root / BASELINE_CERT_REL
    data, err = _load_json(path)
    if err:
        return _section("pending", reason=err)
    assert data is not None
    evidence = [BASELINE_CERT_REL.as_posix()]
    notes: list[str] = []
    status = "present"
    if data.get("schema") != "argus-h3-a6000-fidelity-baseline-certification-v2":
        status = "invalid"
        notes.append(f"Expected baseline v2 schema, found {data.get('schema')!r}.")
    if data.get("track") != FIDELITY:
        status = "invalid"
        notes.append(f"Expected fidelity lane {FIDELITY}, found {data.get('track')!r}.")
    runs = data.get("runs") if isinstance(data.get("runs"), list) else []
    av_pass_count = sum(1 for run in runs if isinstance(run, dict) and run.get("structural_av_pass") is True)
    out = {
        "schema": data.get("schema"),
        "status": data.get("status"),
        "track": data.get("track"),
        "platform": data.get("platform"),
        "physical_device": data.get("physical_device") if isinstance(data.get("physical_device"), dict) else {},
        "all_requests": _stats_subset(data.get("all_requests")),
        "warm_primary_denominator": _stats_subset(data.get("warm_requests_primary_denominator")),
        "session_first_requests": _stats_subset(data.get("session_first_requests")),
        "service_session_count": (data.get("service_sessions") or {}).get("count") if isinstance(data.get("service_sessions"), dict) else None,
        "resource": _resource_subset(data.get("resource")),
        "structural_av_pass_count": av_pass_count,
        "run_count": len(runs),
        "claim_boundary": data.get("claim_boundary"),
        "correction_note": data.get("correction_note"),
    }
    return _section(status, evidence=evidence, data=out, notes=notes)


def _count_turbo_av_passes(timing: JsonDict, *, evidence_root: Path, repo_root: Path, evidence: list[str], notes: list[str]) -> JsonDict:
    total = 0
    passed = 0
    missing = 0
    for sched in (timing.get("schedules") or {}).values():
        if not isinstance(sched, dict):
            continue
        runs = sched.get("runs")
        if not isinstance(runs, list):
            continue
        for run in runs:
            if not isinstance(run, dict):
                continue
            av_path = _resolve_artifact_path(run.get("av_validation"), evidence_root=evidence_root, repo_root=repo_root)
            if av_path is None:
                missing += 1
                continue
            total += 1
            av, err = _load_json(av_path)
            if err:
                missing += 1
                notes.append(err)
                continue
            evidence.append(_evidence_rel(av_path, evidence_root))
            if av and av.get("structural_av_contract_pass") is True:
                passed += 1
    return {"referenced_av_count": total, "structural_av_pass_count": passed, "missing_or_unreadable_av_count": missing}


def collect_turbo(evidence_root: Path, repo_root: Path) -> JsonDict:
    evidence: list[str] = []
    notes: list[str] = []
    latest_path = evidence_root / TURBO_TIMING_LATEST_REL
    run_id, err = _read_pointer(latest_path)
    if err:
        return _section("pending", reason=err)
    assert run_id is not None
    evidence.append(TURBO_TIMING_LATEST_REL.as_posix())
    run_dir = evidence_root / TURBO_TIMING_REPEATS_REL / run_id
    timing_path = run_dir / "timing_summary.json"
    timing, timing_err = _load_json(timing_path)
    if timing_err:
        return _section("pending", evidence=evidence, reason=timing_err)
    assert timing is not None
    evidence.append(_evidence_rel(timing_path, evidence_root))
    merge_path = run_dir / "merge_manifest.json"
    merge, merge_err = _load_json(merge_path)
    if merge_err is None:
        evidence.append(_evidence_rel(merge_path, evidence_root))
    else:
        notes.append(merge_err)

    status = "present"
    if timing.get("track") != PRACTICAL:
        status = "invalid"
        notes.append(f"Turbo timing must remain {PRACTICAL}, found {timing.get('track')!r}.")
    if timing.get("status") != "pass_same_physical_device_paired_n10":
        status = "invalid"
        notes.append(f"Unexpected Turbo timing status: {timing.get('status')!r}.")

    schedules: JsonDict = {}
    for step, sched in sorted((timing.get("schedules") or {}).items(), key=lambda item: str(item[0])):
        if not isinstance(sched, dict):
            continue
        schedules[str(step)] = {
            "steps": int(step) if str(step).isdigit() else step,
            "n": sched.get("n"),
            "median_s": sched.get("median_s"),
            "mean_s": sched.get("mean_s"),
            "cv_percent": sched.get("cv_percent"),
            "speedup_vs_same_gpu3_bf16_warm_n10_median": sched.get("speedup_vs_same_gpu3_bf16_warm_n10_median"),
            "speedup_denominator_s": sched.get("speedup_denominator_s"),
        }
    av_summary = _count_turbo_av_passes(timing, evidence_root=evidence_root, repo_root=repo_root, evidence=evidence, notes=notes)
    quality = collect_turbo_quality(evidence_root)
    evidence.extend(quality.get("evidence", []))
    data = {
        "run_id": run_id,
        "track": timing.get("track"),
        "status": timing.get("status"),
        "physical_device": timing.get("physical_device") if isinstance(timing.get("physical_device"), dict) else {},
        "baseline_denominator": timing.get("baseline_denominator") if isinstance(timing.get("baseline_denominator"), dict) else {},
        "excluded_warmups": timing.get("excluded_warmups") if isinstance(timing.get("excluded_warmups"), dict) else {},
        "paired_formal_n_per_schedule": min([int(row.get("n", 0)) for row in schedules.values()] or [0]),
        "schedules": schedules,
        "resource": _resource_subset(timing.get("resource")),
        "av": av_summary,
        "quality_scope": timing.get("quality_scope"),
        "quality_suite": quality,
        "merge_status": merge.get("status") if isinstance(merge, dict) else None,
        "merge_strength": (merge.get("merge") or {}).get("strength") if isinstance(merge, dict) and isinstance(merge.get("merge"), dict) else None,
        "completed_shard_count": len(merge.get("completed_shards", {})) if isinstance(merge, dict) and isinstance(merge.get("completed_shards"), dict) else None,
        "claim_boundary": "Turbo is practical_disclosed_approx only and must not be relabeled as BF16-exact/fidelity.",
    }
    if quality.get("status") == "pending":
        status = "partial" if status == "present" else status
    return _section(status, evidence=evidence, data=data, notes=notes)


def collect_turbo_quality(evidence_root: Path) -> JsonDict:
    evidence: list[str] = []
    latest_path = evidence_root / TURBO_QUALITY_LATEST_REL
    run_id, err = _read_pointer(latest_path)
    if err:
        return _section("pending", reason=err)
    evidence.append(TURBO_QUALITY_LATEST_REL.as_posix())
    run_dir = evidence_root / TURBO_QUALITY_RUNS_REL / run_id
    analysis_path = run_dir / "quality_suite_analysis.json"
    analysis, analysis_err = _load_json(analysis_path)
    if analysis_err:
        return _section("pending", evidence=evidence, reason=analysis_err)
    assert analysis is not None
    evidence.append(_evidence_rel(analysis_path, evidence_root))
    comparison_path = run_dir / "baseline_seed0_quality_comparison.json"
    comparison, comp_err = _load_json(comparison_path)
    if comp_err is None:
        evidence.append(_evidence_rel(comparison_path, evidence_root))
    human_path = run_dir / "human_review.md"
    human_text, human_err = _read_text(human_path)
    if human_err is None:
        evidence.append(_evidence_rel(human_path, evidence_root))
    steps: JsonDict = {}
    for step, stats in sorted((analysis.get("latency_by_step") or {}).items(), key=lambda item: str(item[0])):
        if not isinstance(stats, dict):
            continue
        step_row = _stats_subset(stats)
        if isinstance(comparison, dict) and isinstance(comparison.get(str(step)), dict):
            step_row.update(
                {
                    "baseline_seed0_audio_cosine": comparison[str(step)].get("audio_cosine"),
                    "baseline_seed0_video_mse": comparison[str(step)].get("video_mse"),
                    "baseline_seed0_video_psnr_db": comparison[str(step)].get("video_psnr_db"),
                }
            )
        steps[str(step)] = step_row
    pending_human = True if human_text is None else ("PENDING" in human_text.upper())
    data = {
        "run_id": run_id,
        "status": analysis.get("status"),
        "track": analysis.get("track"),
        "case_count": analysis.get("case_count"),
        "pair_count": analysis.get("pair_count"),
        "steps": steps,
        "all_cases_structural_av_pass": analysis.get("all_cases_structural_av_pass"),
        "quality_certification": analysis.get("quality_certification"),
        "pending_human_review": pending_human,
        "human_auditory_listening": "pending" if pending_human else "review_text_present",
        "semantic_quality_certified": False,
        "metric_limits": analysis.get("metric_limits") if isinstance(analysis.get("metric_limits"), list) else [],
    }
    return _section("present", evidence=evidence, data=data)


def _parse_resident_layers_from_name(name: str) -> int | None:
    match = re.search(r"_rl(\d+)_", name)
    return int(match.group(1)) if match else None


def _summarize_dlo_attempt(run_dir: Path, evidence_root: Path) -> JsonDict:
    rel_dir = _evidence_rel(run_dir, evidence_root)
    verdict_path = run_dir / "capacity_gate_verdict.json"
    verdict, verdict_err = _load_json(verdict_path)
    resident_layers = _parse_resident_layers_from_name(run_dir.name)
    evidence = [rel_dir]
    if verdict_err is None and verdict is not None:
        evidence.append(_evidence_rel(verdict_path, evidence_root))
        baseline = verdict.get("baseline_5step") if isinstance(verdict.get("baseline_5step"), dict) else {}
        candidate = verdict.get("candidate_5step") if isinstance(verdict.get("candidate_5step"), dict) else {}
        baseline_latency = _maybe_number(baseline.get("latency_s"))
        candidate_latency = _maybe_number(candidate.get("latency_s"))
        speedup = None
        if baseline_latency and candidate_latency:
            speedup = float(baseline_latency) / float(candidate_latency)
        return {
            "resident_layers": verdict.get("resident_layers", resident_layers),
            "stage": verdict.get("stage"),
            "status": verdict.get("status"),
            "result_status": "present" if verdict.get("status") == "pass" else "terminal_or_failed",
            "baseline_5step_latency_s": baseline.get("latency_s"),
            "candidate_5step_latency_s": candidate.get("latency_s"),
            "capacity_5step_speedup_candidate_vs_baseline": speedup,
            "exact_hash_match": bool(baseline.get("sha256") and baseline.get("sha256") == candidate.get("sha256")),
            "resource": _resource_subset(verdict.get("resource")),
            "evidence": evidence,
        }
    status_text, _ = _read_text(run_dir / "status.txt")
    infra_text, _ = _read_text(run_dir / "INFRASTRUCTURE_ABORT.md")
    exit_code, _ = _read_text(run_dir / "exit_code")
    result_status = "pending_no_capacity_verdict"
    status_value = status_text.strip() if status_text else None
    if status_value and "infrastructure" in status_value:
        result_status = "infrastructure_interrupted_no_capacity_result"
    elif exit_code and exit_code.strip() not in {"0", ""}:
        result_status = "pending_or_failed_no_capacity_verdict"
    return {
        "resident_layers": resident_layers,
        "stage": "capacity-5step",
        "status": status_value,
        "result_status": result_status,
        "infrastructure_abort_note_present": infra_text is not None,
        "exit_code": exit_code.strip() if exit_code else None,
        "evidence": evidence,
    }


def collect_dlo(evidence_root: Path) -> JsonDict:
    evidence: list[str] = []
    notes: list[str] = []
    plan, plan_err = _load_json(evidence_root / DLO_PLAN_REL)
    data: JsonDict = {
        "plan": {},
        "detached_continuation": {},
        "capacity_attempts": [],
        "capacity_passes": [],
        "candidate50": {"status": "pending", "reason": "no candidate50 summary or pointer evidence found"},
        "formal_n10": {"status": "pending", "reason": "no DLO formal N10 timing evidence found"},
    }
    status = "pending"
    if plan_err is None and plan is not None:
        evidence.append(DLO_PLAN_REL.as_posix())
        status = "present"
        data["plan"] = {
            "status": plan.get("status"),
            "baseline_denominator": plan.get("baseline_denominator") if isinstance(plan.get("baseline_denominator"), dict) else {},
            "candidate_resident_layers": [item.get("resident_layers") for item in plan.get("candidate_resident_layers", []) if isinstance(item, dict)],
            "safety_model": plan.get("safety_model") if isinstance(plan.get("safety_model"), dict) else {},
        }
    else:
        notes.append(plan_err or "missing DLO plan")

    state_dir = evidence_root / DLO_STATE_REL
    state: JsonDict = {}
    state_status, _ = _read_text(state_dir / "status.txt")
    if state_status is not None:
        state["status"] = state_status.strip()
        evidence.append((DLO_STATE_REL / "status.txt").as_posix())
    context, _ = _read_text(state_dir / "context.env")
    if context is not None:
        state["context_env"] = dict(line.split("=", 1) for line in context.splitlines() if "=" in line)
        evidence.append((DLO_STATE_REL / "context.env").as_posix())
    for pointer_name in ("rl13_run_id.txt", "rl16_run_id.txt", "rl18_run_id.txt", "candidate50_run_id.txt"):
        pointer, _ = _read_text(state_dir / pointer_name)
        if pointer is not None:
            state[pointer_name[:-4]] = pointer.strip()
            evidence.append((DLO_STATE_REL / pointer_name).as_posix())
    latest_run, _ = _read_text(evidence_root / DLO_RUNS_REL / "LATEST_RUN_ID")
    if latest_run is not None:
        state["latest_run_id"] = latest_run.strip()
        evidence.append((DLO_RUNS_REL / "LATEST_RUN_ID").as_posix())
    data["detached_continuation"] = state

    runs_dir = evidence_root / DLO_RUNS_REL
    if runs_dir.is_dir():
        for run_dir in sorted(runs_dir.glob("a6000_dlo_capacity_5step_rl*")):
            if not run_dir.is_dir():
                continue
            attempt = _summarize_dlo_attempt(run_dir, evidence_root)
            data["capacity_attempts"].append(attempt)
            evidence.extend(attempt.get("evidence", []))
            if attempt.get("result_status") == "present" and attempt.get("status") == "pass":
                data["capacity_passes"].append(attempt)
    if data["capacity_attempts"] and status == "pending":
        status = "partial"

    candidate_summary_paths = [
        state_dir / "candidate50_summary.json",
        evidence_root / DLO_RUNS_REL / "candidate50_summary.json",
    ]
    for path in sorted(runs_dir.glob("*/candidate50_summary.json")) if runs_dir.is_dir() else []:
        candidate_summary_paths.append(path)
    for path in candidate_summary_paths:
        candidate, err = _load_json(path)
        if err is None and candidate is not None:
            data["candidate50"] = {"status": "present", "summary": candidate, "evidence": [_evidence_rel(path, evidence_root)]}
            evidence.append(_evidence_rel(path, evidence_root))
            break
    if data["candidate50"].get("status") == "pending" and state.get("candidate50_run_id"):
        data["candidate50"]["reason"] = f"candidate50_run_id {state['candidate50_run_id']} is present, but no candidate50 summary evidence was found"

    formal_paths = [evidence_root / "dlo_autotune/formal_n10_summary.json"]
    for path in sorted(runs_dir.glob("*/formal_n10_summary.json")) if runs_dir.is_dir() else []:
        formal_paths.append(path)
    for path in formal_paths:
        formal, err = _load_json(path)
        if err is None and formal is not None:
            data["formal_n10"] = {"status": "present", "summary": formal, "evidence": [_evidence_rel(path, evidence_root)]}
            evidence.append(_evidence_rel(path, evidence_root))
            break

    if data["candidate50"].get("status") == "pending" or data["formal_n10"].get("status") == "pending":
        status = "partial" if status == "present" else status
    reason = None if status != "pending" else "missing DLO plan and run evidence"
    return _section(status, evidence=evidence, data=data, notes=notes, reason=reason)


def _read_latest_dir(root: Path, pointer_rel: Path, fallback_glob: str) -> Path | None:
    pointer, err = _read_pointer(root / pointer_rel)
    if err is None and pointer:
        path = root / pointer_rel.parent / pointer
        if path.is_dir():
            return path
    candidates = sorted((root / pointer_rel.parent).glob(fallback_glob))
    dirs = [path for path in candidates if path.is_dir()]
    return dirs[-1] if dirs else None


def collect_exact_kernels(evidence_root: Path) -> JsonDict:
    evidence: list[str] = []
    notes: list[str] = []
    data: JsonDict = {
        "lane": "diagnostic_exact_kernel_candidates_not_deployed_as_certified_speedups",
        "microbenchmarks": [],
        "correctness": {},
        "e2e_ablation": {},
    }
    status = "pending"

    exact_dir = _read_latest_dir(evidence_root, EXACT_LATEST_REL, "gpu_exact_*")
    if exact_dir is not None:
        micro_path = exact_dir / "microbenchmark.json"
        micro, micro_err = _load_json(micro_path)
        if micro_err is None and micro is not None:
            evidence.append(_evidence_rel(micro_path, evidence_root))
            data["microbenchmarks"] = _median_speedups(micro.get("benchmarks"))
            data["microbenchmark_scope"] = micro.get("scope")
            data["microbenchmark_model_load"] = micro.get("model_load")
            data["microbenchmark_repeats"] = micro.get("repeats")
            status = "present"
        else:
            notes.append(micro_err or "missing exact microbenchmark")
        correctness_path = exact_dir / "correctness.json"
        correctness, correctness_err = _load_json(correctness_path)
        if correctness_err is None and correctness is not None:
            evidence.append(_evidence_rel(correctness_path, evidence_root))
            cases = correctness.get("cases") if isinstance(correctness.get("cases"), list) else []
            data["correctness"] = {
                "model_load": correctness.get("model_load"),
                "validated_single_a6000_sm86": correctness.get("validated_single_a6000_sm86"),
                "case_count": len(cases),
            }
            status = "present"
    else:
        notes.append("missing exact-kernel microbenchmark/correctness directory")

    r5_dir = _read_latest_dir(evidence_root, R5_ABLATION_LATEST_REL, "r5_ablation_*")
    if r5_dir is not None:
        modes: JsonDict = {}
        for mode in ("adaln", "rope", "all_exact", "swiglu"):
            mode_dir = r5_dir / mode
            if not mode_dir.is_dir():
                continue
            row: JsonDict = {}
            quality, quality_err = _load_json(mode_dir / "quality_vs_dense.json")
            if quality_err is None and quality is not None:
                evidence.append(_evidence_rel(mode_dir / "quality_vs_dense.json", evidence_root))
                row["quality_vs_dense"] = {
                    "claim_scope": quality.get("claim_scope"),
                    "video_mean_mse": quality.get("video_mean_mse"),
                    "audio_waveform_cosine": quality.get("audio_waveform_cosine"),
                }
            telemetry, telemetry_err = _load_json(mode_dir / "exact_telemetry.json")
            if telemetry_err is None and telemetry is not None:
                evidence.append(_evidence_rel(mode_dir / "exact_telemetry.json", evidence_root))
                ops: JsonDict = {}
                for op_name, op_stats in (telemetry.get("ops") or {}).items():
                    if not isinstance(op_stats, dict):
                        continue
                    ops[op_name] = {key: op_stats.get(key) for key in ("calls", "candidate", "fallback", "decline", "strict_error")}
                row["telemetry_ops"] = ops
            av, av_err = _load_json(mode_dir / "av_validation.json")
            if av_err is None and av is not None:
                evidence.append(_evidence_rel(mode_dir / "av_validation.json", evidence_root))
                row["av"] = {
                    "width": av.get("width"),
                    "height": av.get("height"),
                    "decoded_video_frames": av.get("decoded_video_frames"),
                    "average_rate": av.get("average_rate"),
                    "audio_channels": av.get("audio_channels"),
                    "audio_sample_rate_hz": av.get("audio_sample_rate") or av.get("audio_sample_rate_hz"),
                    "video_present": av.get("video_present"),
                    "audio_present": av.get("audio_present"),
                    "steps": av.get("steps"),
                }
            http_text, _ = _read_text(mode_dir / "http_metrics.txt")
            time_total = _parse_http_time_total(http_text)
            if time_total is not None:
                evidence.append(_evidence_rel(mode_dir / "http_metrics.txt", evidence_root))
                row["latency_s"] = time_total
            if row:
                modes[mode] = row
        if modes:
            data["e2e_ablation"] = {"run_dir": _evidence_rel(r5_dir, evidence_root), "modes": modes}
            status = "present" if status == "present" else "partial"
    else:
        notes.append("missing r5 exact-kernel ablation directory")

    if not data["microbenchmarks"] and not data["e2e_ablation"]:
        status = "pending"
    return _section(status, evidence=evidence, data=data, notes=notes, reason="missing exact-kernel evidence" if status == "pending" else None)


def _parse_env_lines(text: str | None) -> JsonDict:
    data: JsonDict = {}
    if text is None:
        return data
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip().strip('"')
    return data


def _resource_csv_subset(path: Path, evidence_root: Path, evidence: list[str], notes: list[str]) -> JsonDict:
    if not path.is_file():
        return {"status": "missing", "reason": f"missing evidence: {path.as_posix()}"}
    evidence.append(_evidence_rel(path, evidence_root))
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except csv.Error as exc:
        notes.append(f"invalid resource CSV in {path.as_posix()}: {exc}")
        return {"status": "invalid", "reason": str(exc)}
    out: JsonDict = {"status": "present", "samples": len(rows)}
    numeric_aliases = {
        "peak_gpu_memory_mib": ("gpu_memory_used_mib", "memory.used [MiB]", "memory.used"),
        "peak_gpu_util_percent": ("gpu_util_percent", "utilization.gpu [%]", "utilization.gpu"),
        "peak_power_w": ("power_w", "power.draw [W]", "power.draw"),
        "peak_temperature_c": ("temperature_c", "temperature.gpu", "temperature.gpu [C]"),
        "peak_host_memory_used_bytes": ("host_memory_used_bytes",),
        "peak_host_swap_used_bytes": ("host_swap_used_bytes",),
    }
    for canonical, columns in numeric_aliases.items():
        values: list[float] = []
        for row in rows:
            normalized_row = {str(key).strip(): str(value).strip() for key, value in row.items() if key is not None}
            for column in columns:
                number = _maybe_number(normalized_row.get(column))
                if number is not None:
                    values.append(float(number))
                    break
        if values:
            out[canonical] = max(values)
    return out


def _av_contract_subset(av: JsonDict | None) -> JsonDict:
    av = av or {}
    data = {
        "mode": av.get("mode"),
        "opaque_output_identifier_policy": "sha256_omitted_not_evidence",
        "bytes": av.get("bytes"),
        "video_present": av.get("video_present"),
        "audio_present": av.get("audio_present"),
        "width": av.get("width"),
        "height": av.get("height"),
        "average_rate": av.get("average_rate"),
        "decoded_video_frames": av.get("decoded_video_frames"),
        "audio_sample_rate_hz": av.get("audio_sample_rate_hz") or av.get("audio_sample_rate"),
        "audio_channels": av.get("audio_channels"),
        "decoded_audio_frames": av.get("decoded_audio_frames"),
        "decoded_audio_samples": av.get("decoded_audio_samples"),
    }
    required_ok = (
        data["video_present"] is True
        and data["audio_present"] is True
        and data["width"] == 1344
        and data["height"] == 768
        and _maybe_number(data["decoded_video_frames"]) is not None
        and float(_maybe_number(data["decoded_video_frames"]) or 0) > 0
        and data["audio_sample_rate_hz"] == 32000
        and data["audio_channels"] == 2
    )
    data["structural_av_contract_pass"] = required_ok
    return data


def _load_av_contract(path: Path, evidence_root: Path, evidence: list[str], notes: list[str]) -> JsonDict:
    data, err = _load_json(path)
    if err:
        notes.append(err)
        return {"structural_av_contract_pass": False, "reason": err}
    evidence.append(_evidence_rel(path, evidence_root))
    return _av_contract_subset(data)


def _load_http_metrics(path: Path, evidence_root: Path, evidence: list[str], notes: list[str]) -> JsonDict:
    text, err = _read_text(path)
    if err:
        notes.append(err)
        return {"status": "missing", "reason": err}
    evidence.append(_evidence_rel(path, evidence_root))
    return {"status": "present", "time_total_s": _parse_http_time_total(text)}


def _legacy_sol_attn_kernel_diagnostic(evidence_root: Path, evidence: list[str], notes: list[str]) -> JsonDict:
    legacy: JsonDict = {"status": "pending"}
    pointer_text, pointer_err = _read_text(evidence_root / SOL_ATTN_LATEST_REL)
    sol_dir: Path | None = None
    if pointer_err is None and pointer_text is not None:
        evidence.append(SOL_ATTN_LATEST_REL.as_posix())
        raw_pointer = pointer_text.strip()
        legacy["latest_pointer_raw"] = raw_pointer
        pointer_path = Path(raw_pointer)
        if pointer_path.is_absolute() and pointer_path.is_dir():
            sol_dir = pointer_path
        elif raw_pointer and "/" not in raw_pointer and ".." not in raw_pointer:
            candidate = evidence_root / SOL_ATTN_LATEST_REL.parent / raw_pointer
            if candidate.is_dir():
                sol_dir = candidate
        else:
            legacy["pointer_reason"] = f"stale or non-run-id Sol-Attn pointer is not accepted as strict r6 runtime evidence: {raw_pointer!r}"
    elif pointer_err:
        legacy["pointer_reason"] = pointer_err
        notes.append(pointer_err)
    if sol_dir is None:
        sol_dir = _read_latest_dir(evidence_root, SOL_ATTN_LATEST_REL, "sol_attn_gpu_*")
    if sol_dir is None:
        legacy["reason"] = "missing legacy Sol-Attn kernel diagnostic"
        return legacy
    result_path = sol_dir / "result.json"
    result, err = _load_json(result_path)
    if err:
        legacy["reason"] = err
        notes.append(err)
        return legacy
    assert result is not None
    evidence.append(_evidence_rel(result_path, evidence_root))
    bench = result.get("bench") if isinstance(result.get("bench"), dict) else {}
    correctness = result.get("correctness") if isinstance(result.get("correctness"), dict) else {}
    speedup = bench.get("speedup_dense_over_sparse_median")
    if _maybe_number(speedup) is not None and float(speedup) < 1.0:
        notes.append("Legacy Sol-Attn kernel diagnostic is slower than dense and is not deployed as a speed result.")
    return {
        **{key: value for key, value in legacy.items() if key in {"latest_pointer_raw", "pointer_reason"}},
        "status": "present",
        "run_dir": _evidence_rel(sol_dir, evidence_root),
        "schema_version": result.get("schema_version"),
        "model_load": result.get("model_load"),
        "device": result.get("device"),
        "capability": result.get("capability"),
        "correctness": {
            "compile_status": correctness.get("compile_status"),
            "elapsed_s": correctness.get("elapsed_s"),
            "max_abs_valid": correctness.get("max_abs_valid"),
            "prefix_rows_equal_dense": correctness.get("prefix_rows_equal_dense"),
            "padding_rows_zero": correctness.get("padding_rows_zero"),
        },
        "bench": {
            "kernel_candidates_only_not_h3_e2e": bench.get("kernel_candidates_only_not_h3_e2e"),
            "dense_median_ms": (bench.get("dense_ms") or {}).get("median_ms") if isinstance(bench.get("dense_ms"), dict) else None,
            "dense_mean_ms": (bench.get("dense_ms") or {}).get("mean_ms") if isinstance(bench.get("dense_ms"), dict) else None,
            "sparse_median_ms": (bench.get("sparse_ms") or {}).get("median_ms") if isinstance(bench.get("sparse_ms"), dict) else None,
            "sparse_mean_ms": (bench.get("sparse_ms") or {}).get("mean_ms") if isinstance(bench.get("sparse_ms"), dict) else None,
            "speedup_dense_over_sparse_median": speedup,
            "shape": bench.get("shape") if isinstance(bench.get("shape"), dict) else {},
        },
    }


def _runtime_finish(
    classification: str,
    reason: str,
    data: JsonDict,
    *,
    accepted_metadata: bool = False,
    accepted_runtime: bool = False,
    release_manifest_eligible: bool | None = None,
) -> JsonDict:
    release_eligible = accepted_runtime if release_manifest_eligible is None else release_manifest_eligible
    data["classification"] = classification
    data["reason"] = reason
    data["accepted_metadata"] = accepted_metadata
    data["accepted_runtime_evidence"] = accepted_runtime
    data["release_manifest_eligible"] = release_eligible
    h3_status = "pending" if classification == "pending_non_terminal_supervisor_status" else classification
    data["h3_e2e"] = {
        "status": h3_status,
        "reason": reason,
        "accepted_runtime_evidence": accepted_runtime,
        "release_manifest_eligible": release_eligible,
    }
    return data


def _classify_sol_attn_runtime(evidence_root: Path, evidence: list[str], notes: list[str]) -> JsonDict:
    data: JsonDict = {"status": "pending"}
    supervisor_dir = evidence_root / SOL_ATTN_SUPERVISOR_REL
    status_text, status_err = _read_text(supervisor_dir / "status.txt")
    latest_run_id, latest_err = _read_pointer(supervisor_dir / "latest_run_id")
    if status_err is None:
        evidence.append((SOL_ATTN_SUPERVISOR_REL / "status.txt").as_posix())
    if latest_err is None and latest_run_id is not None:
        evidence.append((SOL_ATTN_SUPERVISOR_REL / "latest_run_id").as_posix())
    exit_text, _ = _read_text(supervisor_dir / "exit_code")
    if exit_text is not None:
        evidence.append((SOL_ATTN_SUPERVISOR_REL / "exit_code").as_posix())
    supervisor_status = status_text.strip() if status_text else None
    data["supervisor"] = {"status": supervisor_status, "latest_run_id": latest_run_id, "exit_code": exit_text.strip() if exit_text is not None else None}
    if status_err or latest_err or not latest_run_id:
        return _runtime_finish("pending", status_err or latest_err or "missing Sol-Attn supervisor latest_run_id", data)
    if latest_run_id.startswith("sol_attn_gpu_"):
        return _runtime_finish("stale_or_dry_run_rejected", "legacy Sol-Attn toy/kernel run is not accepted as H3 diagnostic runtime", data)

    run_dir = evidence_root / "sol_engine_port" / latest_run_id
    data["run_dir"] = _evidence_rel(run_dir, evidence_root)
    if not run_dir.is_dir():
        return _runtime_finish("pending", f"supervisor latest_run_id {latest_run_id!r} has no evidence directory", data)

    resource = _resource_csv_subset(run_dir / "resource_monitor.csv", evidence_root, evidence, notes)
    if resource.get("status") == "present":
        data["resource"] = resource

    workload_text, workload_err = _read_text(run_dir / "workload.env")
    if workload_err is None:
        evidence.append(_evidence_rel(run_dir / "workload.env", evidence_root))
    workload = _parse_env_lines(workload_text)
    image_text = str(workload.get("image") or "")

    r8_identity_text, r8_identity_err = _read_text(run_dir / "r8_image_identity.env")
    r7_identity_text, r7_identity_err = _read_text(run_dir / "r7_image_identity.env")
    r6_identity_text, r6_identity_err = _read_text(run_dir / "r6_image_identity.env")
    if r8_identity_err is None:
        evidence.append(_evidence_rel(run_dir / "r8_image_identity.env", evidence_root))
    if r7_identity_err is None:
        evidence.append(_evidence_rel(run_dir / "r7_image_identity.env", evidence_root))
    if r6_identity_err is None:
        evidence.append(_evidence_rel(run_dir / "r6_image_identity.env", evidence_root))
    r8_identity = _parse_env_lines(r8_identity_text)
    r7_identity = _parse_env_lines(r7_identity_text)
    r6_identity = _parse_env_lines(r6_identity_text)
    if r8_identity or "r8" in image_text:
        runtime_label = "r8"
        identity = r8_identity
        identity_err = r8_identity_err
    elif r7_identity or "r7" in image_text:
        runtime_label = "r7"
        identity = r7_identity
        identity_err = r7_identity_err
    elif r6_identity or "r6" in image_text:
        runtime_label = "r6"
        identity = r6_identity
        identity_err = r6_identity_err
    else:
        runtime_label = "unknown"
        identity = {}
        identity_err = "missing r6/r7/r8 readable image provenance evidence"
    data["runtime_label"] = runtime_label

    data["opaque_integrity_policy"] = {
        "image_identifiers": "omitted_not_evidence",
        "output_identifiers": "omitted_not_evidence",
        "opaque_identifier_equality": "not_used_for_classification",
    }
    data["image_identity"] = {
        "runtime_label": runtime_label,
        "version_label": identity.get("actual_image_version_label"),
        "required_version_label": identity.get("required_image_version_label"),
        "base_label": identity.get("actual_image_base_label"),
        "title_label": identity.get("actual_image_title_label"),
        "image_tag": image_text or None,
        "opaque_identifier_policy": "omitted_not_evidence",
    }
    data["readable_provenance"] = {
        "selected_run_dir": _evidence_rel(run_dir, evidence_root),
        "runtime_label": runtime_label,
        "workload_image": image_text or None,
        "workload_attention_backend": workload.get("attention_backend"),
        "image_version_label": identity.get("actual_image_version_label"),
        "required_image_version_label": identity.get("required_image_version_label"),
        "image_base_label": identity.get("actual_image_base_label"),
        "image_title_label": identity.get("actual_image_title_label"),
        "supervisor_status": supervisor_status,
    }
    data["workload"] = {key: workload.get(key) for key in ("image", "steps", "seed", "width", "height", "fps", "duration", "attention_backend", "sol_attn_opt_in", "sol_attn_cache", "sol_attn_diagnostic_materialize", "sol_attn_materialize_max_bytes", "network")}
    data["workload"]["run_id_text_prefix_note"] = "latest_run_id may retain r6 text prefix; runtime label is decided from readable workload/version-label provenance"

    active_statuses = {"running", "active", "starting", "started"}
    if str(supervisor_status).lower() in active_statuses:
        return _runtime_finish("pending_non_terminal_supervisor_status", f"Sol-Attn supervisor status is still {supervisor_status!r}; dense/opt-in runtime evidence is not terminal and is not ingested as success", data)

    lowered_status = str(supervisor_status or "").lower()
    if lowered_status.startswith("failed") or (exit_text is not None and exit_text.strip() not in {"", "0"}):
        return _runtime_finish("runtime_failure", f"supervisor ended with status={supervisor_status!r} exit_code={exit_text.strip() if exit_text else 'missing'}", data)
    if lowered_status not in {"complete", "completed", "success", "succeeded", "done"}:
        return _runtime_finish("pending", f"supervisor status {supervisor_status!r} is not a completed runtime marker", data)

    if runtime_label not in {"r6", "r7", "r8"} or identity_err or workload_err:
        missing = [msg for msg in (identity_err, workload_err) if msg]
        return _runtime_finish("fail_closed_missing_metadata", "; ".join(missing) or "missing readable runtime/workload provenance", data)
    version_label = data["image_identity"].get("version_label")
    required_version_label = data["image_identity"].get("required_version_label") or runtime_label
    if version_label != runtime_label or required_version_label != runtime_label:
        return _runtime_finish("identity_mismatch", f"{runtime_label} readable image version-label provenance is absent or mismatched", data)
    workload_mismatches = []
    expected_workload = {
        "steps": "5",
        "seed": "0",
        "width": "1344",
        "height": "768",
        "attention_backend": "H3_A6000_SOL_ATTN",
        "sol_attn_cache": "off",
        "network": "none",
    }
    for key, expected_value in expected_workload.items():
        if str(workload.get(key)) != expected_value:
            workload_mismatches.append(f"{key}={workload.get(key)!r} expected {expected_value!r}")
    if runtime_label in {"r7", "r8"} and workload.get("sol_attn_diagnostic_materialize") != f"on_for_{runtime_label}_only":
        workload_mismatches.append(f"sol_attn_diagnostic_materialize is not on_for_{runtime_label}_only")
    if workload_mismatches:
        return _runtime_finish("identity_mismatch", "workload identity mismatch: " + "; ".join(workload_mismatches), data)

    dense_http = _load_http_metrics(run_dir / "dense_h3_backend_reference" / "http_metrics.txt", evidence_root, evidence, notes)
    sol_http = _load_http_metrics(run_dir / "sol_attn" / "http_metrics.txt", evidence_root, evidence, notes)
    dense_av = _load_av_contract(run_dir / "dense_h3_backend_reference" / "av_validation.json", evidence_root, evidence, notes)
    sol_av = _load_av_contract(run_dir / "sol_attn" / "av_validation.json", evidence_root, evidence, notes)
    status_json, status_json_err = _load_json(run_dir / "sol_attn_diagnostic_status.json")
    if status_json_err is None and status_json is not None:
        evidence.append(_evidence_rel(run_dir / "sol_attn_diagnostic_status.json", evidence_root))
    telemetry, telemetry_err = _load_json(run_dir / "sol_attn" / "sol_attn_telemetry.sol_attn.json")
    if telemetry_err is None and telemetry is not None:
        evidence.append(_evidence_rel(run_dir / "sol_attn" / "sol_attn_telemetry.sol_attn.json", evidence_root))
    data["dense_h3_backend_reference"] = {"http": dense_http, "av": dense_av}
    data["sol_attn_opt_in"] = {"http": sol_http, "av": sol_av}
    dense_time = _maybe_number(dense_http.get("time_total_s"))
    sol_time = _maybe_number(sol_http.get("time_total_s"))
    data["paired_http_time_total_s"] = {"dense_h3_backend_reference": dense_time, "sol_attn_opt_in": sol_time}
    if dense_time and sol_time:
        data["paired_http_ratio_dense_over_opt_in_not_speedup"] = float(dense_time) / float(sol_time)

    telemetry_subset: JsonDict = {}
    if isinstance(telemetry, dict):
        telemetry_subset = {
            "dense_calls": telemetry.get("dense_calls"),
            "sparse_candidate_calls": telemetry.get("sparse_candidate_calls"),
            "sparse_calls": telemetry.get("sparse_calls"),
            "fallback_calls": telemetry.get("fallback_calls"),
            "prefix_query_dense_calls": telemetry.get("prefix_query_dense_calls"),
            "materialized_copy_calls": telemetry.get("materialized_copy_calls") or telemetry.get("materialize_copy_count") or telemetry.get("copy_calls"),
            "materialized_copy_bytes": telemetry.get("materialized_copy_bytes") or telemetry.get("materialize_copy_bytes") or telemetry.get("copy_bytes"),
            "materialization_failures": telemetry.get("materialization_failures"),
            "decline_reasons": telemetry.get("decline_reasons") if isinstance(telemetry.get("decline_reasons"), dict) else {},
            "fallback_reasons": telemetry.get("fallback_reasons") if isinstance(telemetry.get("fallback_reasons"), dict) else {},
            "density_samples": telemetry.get("density_samples") if isinstance(telemetry.get("density_samples"), list) else [],
        }
    data["telemetry"] = telemetry_subset

    missing_runtime = [err for err in (status_json_err, telemetry_err) if err]
    if dense_http.get("status") != "present" or dense_http.get("time_total_s") is None or sol_http.get("status") != "present" or sol_http.get("time_total_s") is None:
        missing_runtime.append("missing or invalid dense/opt-in HTTP timing")
    if not dense_av.get("structural_av_contract_pass") or not sol_av.get("structural_av_contract_pass"):
        missing_runtime.append("missing or invalid dense/opt-in structural AV evidence")
    if missing_runtime:
        return _runtime_finish("runtime_failure", "; ".join(missing_runtime), data)

    status_marker = status_json.get("status") if isinstance(status_json, dict) else None
    sparse_calls = _maybe_number(telemetry_subset.get("sparse_calls")) or 0
    sparse_candidates = _maybe_number(telemetry_subset.get("sparse_candidate_calls")) or 0
    fallback_calls = _maybe_number(telemetry_subset.get("fallback_calls")) or 0
    decline_reasons = telemetry_subset.get("decline_reasons") if isinstance(telemetry_subset.get("decline_reasons"), dict) else {}
    density_samples = telemetry_subset.get("density_samples") if isinstance(telemetry_subset.get("density_samples"), list) else []
    if status_marker == "fail_closed_dense_fallback" or (not sparse_candidates and decline_reasons):
        return _runtime_finish("fail_closed_missing_metadata", "Sol-Attn runtime failed closed to dense fallback with decline reasons", data)
    if sparse_calls <= 0:
        return _runtime_finish("fail_closed_missing_metadata", "Sol-Attn telemetry sparse_calls==0; opt-in result is not accepted as a sparse runtime", data)
    if fallback_calls > 0:
        return _runtime_finish("runtime_failure", "Sol-Attn telemetry reported fallback_calls>0", data)
    if telemetry_subset.get("materialization_failures"):
        return _runtime_finish("runtime_failure", "Sol-Attn telemetry reported materialization failures", data)
    if status_marker not in {"metadata_path_accepted_sparse_candidate_attempted", "sparse_runtime_valid"}:
        return _runtime_finish("fail_closed_missing_metadata", f"Sol-Attn metadata was not accepted: diagnostic status={status_marker!r}", data)
    if not density_samples:
        return _runtime_finish("fail_closed_missing_metadata", "Sol-Attn sparse candidate was attempted but density telemetry is absent", data)

    if dense_time and sol_time:
        data["paired_http_ratio_dense_over_opt_in_not_speedup"] = float(dense_time) / float(sol_time)
    return _runtime_finish(
        "sparse_runtime_valid_5step_diagnostic",
        "Sol-Attn sparse_calls>0 with HTTP 200, structural AV, resource, density, and materialization telemetry; this is only a 5-step sparse-execution diagnostic candidate, not a speedup, N10, BF16 fidelity, release, or quality-equivalence claim",
        data,
        accepted_metadata=True,
        accepted_runtime=True,
        release_manifest_eligible=False,
    )


def _latest_glob_dir(parent: Path, pattern: str) -> Path | None:
    if not parent.is_dir():
        return None
    dirs = sorted(path for path in parent.glob(pattern) if path.is_dir())
    return dirs[-1] if dirs else None


def _collect_r8_matched_retest(evidence_root: Path, evidence: list[str], notes: list[str]) -> JsonDict:
    sol_root = evidence_root / "sol_engine_port"
    terminal_dir = _latest_glob_dir(sol_root, R8_MATCHED_RETEST_TERMINAL_GLOB)
    if terminal_dir is not None and (terminal_dir / "decision.json").is_file():
        decision_path = terminal_dir / "decision.json"
        decision, err = _load_json(decision_path)
        if err or decision is None:
            notes.append(err or "invalid r8 matched decision")
            return {"status": "invalid", "reason": err or "invalid r8 matched decision"}
        terminal_files = ["decision.json", "RUN_REPORT.md", "timing_summary.json", "quality_proxy_comparison.json", "resource_summary.json"]
        terminal_artifacts = {name: (terminal_dir / name).is_file() for name in terminal_files}
        for name, present in terminal_artifacts.items():
            if present:
                evidence.append(_evidence_rel(terminal_dir / name, evidence_root))
        posthoc = terminal_dir / "posthoc_finalization_note.json"
        posthoc_rel = None
        if posthoc.is_file():
            posthoc_rel = _evidence_rel(posthoc, evidence_root)
            evidence.append(posthoc_rel)
        recheck_rel = None
        recheck_dir = _latest_glob_dir(evidence_root / "delivery", R8_MATCHED_RETEST_TERMINAL_RECHECK_GLOB)
        if recheck_dir is not None and (recheck_dir / "summary.json").is_file():
            recheck_rel = _evidence_rel(recheck_dir / "summary.json", evidence_root)
            evidence.append(recheck_rel)
        gates = decision.get("gates") if isinstance(decision.get("gates"), dict) else {}
        return {
            "status": decision.get("classification"),
            "source_run_dir": _evidence_rel(terminal_dir, evidence_root),
            "decision_path": _evidence_rel(decision_path, evidence_root),
            "terminal_recheck_path": recheck_rel,
            "posthoc_finalization_note": posthoc_rel,
            "reason": decision.get("reason"),
            "requested_pairs": gates.get("requested_pairs"),
            "completed_pairs": gates.get("completed_pairs"),
            "median_http_time_improvement_pct": decision.get("median_http_time_improvement_pct"),
            "timing_threshold_pct": decision.get("timing_threshold_pct"),
            "failed_gates": decision.get("failed_gates"),
            "proceed_to_n10_recommended": decision.get("proceed_to_n10_recommended"),
            "not_formal_n10": decision.get("not_formal_n10"),
            "not_fidelity_or_performance_claim": decision.get("not_fidelity_or_performance_claim"),
            "terminal_artifacts_present": terminal_artifacts,
        }

    latest_nonterminal: Path | None = None
    delivery = evidence_root / "delivery"
    for pattern in R8_MATCHED_RETEST_NONTERMINAL_GLOBS:
        candidate = _latest_glob_dir(delivery, pattern)
        if candidate is not None and (latest_nonterminal is None or candidate.name > latest_nonterminal.name):
            latest_nonterminal = candidate
    if latest_nonterminal is None:
        return {"status": "not_available", "reason": "no r8 matched-workload route evidence found"}
    json_files = sorted(latest_nonterminal.glob("*.json"))
    if not json_files:
        return {"status": "pending", "reason": f"no JSON file in {latest_nonterminal}"}
    path = json_files[0]
    data, err = _load_json(path)
    if err or data is None:
        return {"status": "pending", "reason": err or "invalid nonterminal inspection"}
    evidence.append(_evidence_rel(path, evidence_root))
    return {
        "status": data.get("classification", "pending"),
        "evidence_path": _evidence_rel(path, evidence_root),
        "source_run_dir": data.get("source_run_dir"),
        "reason": data.get("reason"),
        "n10_recommendation": data.get("n10_recommendation"),
    }


def _collect_r8_formal_n10(evidence_root: Path, evidence: list[str], notes: list[str]) -> JsonDict:
    sol_root = evidence_root / "sol_engine_port"
    formal_dir = _latest_glob_dir(sol_root, R8_FORMAL_N10_GLOB)
    if formal_dir is None:
        return {"status": "not_available", "reason": "no r8 formal N>=10 Sol-Attn run directory found"}

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
    completed_pair_dirs = [path for path in pair_dirs if _formal_pair_completed(path)]
    for path in (
        status_path,
        stdout_path,
        decision_path,
        report_path,
        formal_dir / "FORMAL_N10_RUN_REPORT.md",
        summary_path,
        timing_path,
        quality_path,
        resource_path,
    ):
        if path.is_file():
            evidence.append(_evidence_rel(path, evidence_root))

    supervisor, status_err = _load_json(status_path)
    if status_err:
        notes.append(status_err)
        supervisor = {}
    requested_pairs = supervisor.get("n_pairs") or supervisor.get("requested_pairs") or 10

    if decision_path.is_file():
        decision, err = _load_json(decision_path)
        if err or decision is None:
            notes.append(err or "invalid formal N10 decision")
            return {
                "status": "invalid_formal_n10_decision",
                "source_run_dir": _evidence_rel(formal_dir, evidence_root),
                "reason": err or "invalid formal N10 decision",
                "terminal_artifacts_present": terminal_artifacts,
            }
        gates = decision.get("gates") if isinstance(decision.get("gates"), dict) else {}
        return {
            "status": decision.get("formal_classification"),
            "source_run_dir": _evidence_rel(formal_dir, evidence_root),
            "decision_path": _evidence_rel(decision_path, evidence_root),
            "report_path": _evidence_rel(report_path, evidence_root) if report_path.is_file() else None,
            "summary_path": _evidence_rel(summary_path, evidence_root) if summary_path.is_file() else None,
            "timing_summary_path": _evidence_rel(timing_path, evidence_root) if timing_path.is_file() else None,
            "quality_proxy_comparison_path": _evidence_rel(quality_path, evidence_root) if quality_path.is_file() else None,
            "resource_summary_path": _evidence_rel(resource_path, evidence_root) if resource_path.is_file() else None,
            "reason": decision.get("reason"),
            "requested_pairs": gates.get("requested_pairs", requested_pairs),
            "completed_pairs": gates.get("completed_pairs", decision.get("completed_pairs")),
            "started_pairs": len(pair_dirs),
            "supervisor_status": supervisor.get("status"),
            "supervisor_return_code": supervisor.get("return_code"),
            "same_expected_gpu": _formal_same_expected_gpu(decision),
            "raw_matched_classification": _formal_raw_classification(decision),
            "median_http_time_improvement_pct": decision.get("median_http_time_improvement_pct"),
            "timing_threshold_pct": decision.get("timing_threshold_pct"),
            "failed_gates": decision.get("failed_gates"),
            "lane": decision.get("lane"),
            "terminal_artifacts_present": terminal_artifacts,
            "not_bf16_fidelity": decision.get("not_bf16_fidelity"),
            "not_public_release": decision.get("not_public_release"),
        }

    status = "incomplete_formal_n10_no_terminal_decision"
    if supervisor.get("status") == "running":
        status = "incomplete_formal_n10_running_no_terminal_decision"
    reason = (
        "formal N>=10 supervisor is marked running and has no formal_n10_decision.json/RUN_REPORT terminal artifacts; "
        "the run is not accepted, rejected, or a speedup claim until terminal per-pair evidence and summary are present"
        if supervisor.get("status") == "running"
        else "formal N>=10 run directory lacks a terminal formal_n10_decision.json/RUN_REPORT; do not promote or claim speedup"
    )
    return {
        "status": status,
        "source_run_dir": _evidence_rel(formal_dir, evidence_root),
        "status_path": _evidence_rel(status_path, evidence_root) if status_path.is_file() else None,
        "stdout_path": _evidence_rel(stdout_path, evidence_root) if stdout_path.is_file() else None,
        "reason": reason,
        "requested_pairs": requested_pairs,
        "started_pairs": len(pair_dirs),
        "completed_pairs": len(completed_pair_dirs),
        "supervisor_status": supervisor.get("status"),
        "supervisor_pid": supervisor.get("pid"),
        "gpu_index": supervisor.get("gpu_index"),
        "expected_uuid": supervisor.get("expected_uuid"),
        "terminal_artifacts_present": terminal_artifacts,
        "not_fidelity_or_performance_claim": True,
        "lane": "formal_n10_matched_5step_sol_attn_opt_in_not_bf16_fidelity",
    }


def collect_sol_attn(evidence_root: Path) -> JsonDict:
    evidence: list[str] = []
    notes: list[str] = []
    legacy = _legacy_sol_attn_kernel_diagnostic(evidence_root, evidence, notes)
    runtime = _classify_sol_attn_runtime(evidence_root, evidence, notes)
    matched_retest = _collect_r8_matched_retest(evidence_root, evidence, notes)
    formal_n10 = _collect_r8_formal_n10(evidence_root, evidence, notes)
    h3_e2e = runtime.get("h3_e2e") if isinstance(runtime.get("h3_e2e"), dict) else {"status": "pending", "reason": runtime.get("reason", "missing Sol-Attn runtime evidence")}
    data = {
        "legacy_kernel_diagnostic": legacy,
        "strict_runtime": runtime,
        "strict_r6_runtime": runtime,
        "h3_e2e": h3_e2e,
        "matched_retest": matched_retest,
        "formal_n10": formal_n10,
        "deployment_status": "accepted_5step_diagnostic_not_release_manifest_eligible" if runtime.get("accepted_runtime_evidence") and not runtime.get("release_manifest_eligible") else ("release_manifest_eligible_runtime_evidence" if runtime.get("release_manifest_eligible") else "not_deployed_release_manifest_blocked"),
    }
    status = "partial" if legacy.get("status") == "present" or runtime.get("accepted_metadata") else "pending"
    if runtime.get("classification") in {"pending_non_terminal_supervisor_status", "identity_mismatch", "runtime_failure", "quality_drift", "fail_closed_missing_metadata", "speed_only_no_quality", "stale_or_dry_run_rejected"}:
        status = "partial"
    reason = None if status != "pending" else runtime.get("reason") or legacy.get("reason") or "missing Sol-Attn diagnostic evidence"
    return _section(status, evidence=evidence, data=data, notes=notes, reason=reason)


def collect_dmd(evidence_root: Path) -> JsonDict:
    path = evidence_root / DMD_REL
    text, err = _read_text(path)
    if err:
        return _section("pending", reason=err)
    assert text is not None
    lowered = text.lower()
    blocked = "blocked" in lowered and "research only" in lowered and "no first-source basis" in lowered
    status = "present" if blocked else "partial"
    data = {
        "status": "blocked_research_only_no_go_after_turbo_unless_feasibility_changes" if blocked else "note_present_status_unclassified",
        "track_limit": PRACTICAL,
        "evidence_excerpt": " ".join(text.strip().split())[:500],
        "claim_boundary": "No DMD/DMD2 speed or quality value is reported unless a first-source H3 recipe/checkpoint appears.",
    }
    notes = [] if blocked else ["DMD note did not contain all expected blocked/no-go language."]
    return _section(status, evidence=[DMD_REL.as_posix()], data=data, notes=notes)


def _collect_pending(section_name: str, section: JsonDict) -> list[JsonDict]:
    pending: list[JsonDict] = []
    status = section.get("status")
    if status in {"pending", "invalid"}:
        pending.append({"section": section_name, "status": status, "reason": section.get("reason") or "; ".join(section.get("notes", [])) or "not present"})
    data = section.get("data") if isinstance(section.get("data"), dict) else {}
    if section_name == "turbo":
        quality = data.get("quality_suite")
        if isinstance(quality, dict) and quality.get("status") == "pending":
            pending.append({"section": "turbo.quality_suite", "status": "pending", "reason": quality.get("reason", "missing quality evidence")})
    if section_name == "dlo":
        for key in ("candidate50", "formal_n10"):
            item = data.get(key)
            if isinstance(item, dict) and item.get("status") == "pending":
                pending.append({"section": f"dlo.{key}", "status": "pending", "reason": item.get("reason", "missing evidence")})
        for attempt in data.get("capacity_attempts", []) if isinstance(data.get("capacity_attempts"), list) else []:
            if isinstance(attempt, dict) and str(attempt.get("result_status", "")).startswith("pending"):
                pending.append({"section": f"dlo.capacity.rl{attempt.get('resident_layers')}", "status": "pending", "reason": attempt.get("result_status")})
    if section_name == "sol_attn":
        h3_e2e = data.get("h3_e2e")
        if isinstance(h3_e2e, dict) and h3_e2e.get("status") in {"pending", "identity_mismatch", "stale_or_dry_run_rejected", "fail_closed_missing_metadata", "runtime_failure", "quality_drift", "speed_only_no_quality", "metadata_accepted"}:
            pending.append({"section": "sol_attn.h3_e2e", "status": h3_e2e.get("status", "pending"), "reason": h3_e2e.get("reason", "missing evidence")})
        formal = data.get("formal_n10")
        formal_status = formal.get("status") if isinstance(formal, dict) else None
        matched = data.get("matched_retest")
        if isinstance(matched, dict):
            if matched.get("status") == "proceed_to_formal_n10_candidate":
                if formal_status in {None, "not_available"}:
                    pending.append({"section": "sol_attn.formal_n10", "status": "pending_formal_n10_required_after_r8_n3_candidate_before_speedup_or_quality_claim", "reason": "r8 N=3 route gate recommends formal N>=10, but no formal N>=10 promotion result is accepted"})
                elif str(formal_status).startswith("incomplete_") or formal_status == "invalid_formal_n10_decision":
                    pending.append({"section": "sol_attn.formal_n10", "status": formal_status, "reason": formal.get("reason", "formal N10 run is nonterminal")})
            elif matched.get("status") not in {None, "not_available"}:
                pending.append({"section": "sol_attn.matched_workload", "status": matched.get("status", "pending"), "reason": matched.get("reason", "matched-workload route evidence is not terminal/pass")})
    return pending


def _collect_evidence(sections: dict[str, JsonDict]) -> list[str]:
    evidence: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            ev = value.get("evidence")
            if isinstance(ev, list):
                for item in ev:
                    if isinstance(item, str):
                        evidence.add(item)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for section in sections.values():
        visit(section)
    return sorted(evidence)


def build_payload(evidence_root: Path, *, repo_root: Path | None = None) -> JsonDict:
    repo_root = (repo_root or Path.cwd()).resolve()
    evidence_root = evidence_root.resolve()
    sections = {
        "workload": collect_workload(evidence_root, repo_root),
        "baseline": collect_baseline(evidence_root),
        "turbo": collect_turbo(evidence_root, repo_root),
        "dlo": collect_dlo(evidence_root),
        "exact_kernels": collect_exact_kernels(evidence_root),
        "sol_attn": collect_sol_attn(evidence_root),
        "dmd": collect_dmd(evidence_root),
    }
    pending_items: list[JsonDict] = []
    for name, section in sections.items():
        pending_items.extend(_collect_pending(name, section))

    blockers: list[JsonDict] = []
    turbo_quality = ((sections["turbo"].get("data") or {}).get("quality_suite") or {}) if isinstance(sections["turbo"].get("data"), dict) else {}
    if isinstance(turbo_quality, dict):
        qdata = turbo_quality.get("data") if isinstance(turbo_quality.get("data"), dict) else {}
        if qdata.get("pending_human_review") or qdata.get("semantic_quality_certified") is False:
            blockers.append({"scope": "Turbo quality", "status": "pending", "reason": "semantic quality is not certified and human auditory listening remains pending"})
    dlo_data = sections["dlo"].get("data") if isinstance(sections["dlo"].get("data"), dict) else {}
    if isinstance(dlo_data.get("formal_n10"), dict) and dlo_data["formal_n10"].get("status") == "pending":
        blockers.append({"scope": "DLO", "status": "pending", "reason": dlo_data["formal_n10"].get("reason")})
    sol_data = sections["sol_attn"].get("data") if isinstance(sections["sol_attn"].get("data"), dict) else {}
    h3_e2e = sol_data.get("h3_e2e") if isinstance(sol_data, dict) else None
    if isinstance(h3_e2e, dict) and h3_e2e.get("accepted_runtime_evidence") is not True:
        blockers.append({"scope": "Sol-Attn", "status": h3_e2e.get("status", "pending"), "reason": h3_e2e.get("reason") or "Sol-Attn runtime evidence is not accepted"})
    dmd_data = sections["dmd"].get("data") if isinstance(sections["dmd"].get("data"), dict) else {}
    if dmd_data.get("status", "").startswith("blocked"):
        blockers.append({"scope": "DMD/DMD2", "status": "blocked", "reason": dmd_data.get("claim_boundary")})

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "generated_cpu_evidence_reader",
        "input_root": _repo_rel(evidence_root, repo_root),
        "non_execution_notice": "CPU-only evidence reader: no GPU, Docker, model loading, network, downloads, publication, benchmarks, or live evidence mutation.",
        "lanes": {"fidelity": FIDELITY, "practical": PRACTICAL},
        "sections": sections,
        "pending_items": pending_items,
        "blockers": blockers,
        "evidence_index": _collect_evidence(sections),
    }
    return payload


def _json_type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if value is None:
        return "null"
    return type(value).__name__


def _schema_validate(value: Any, schema: JsonDict, *, root_schema: JsonDict, path: str = "$", errors: list[str] | None = None) -> list[str]:
    errors = errors if errors is not None else []
    if "$ref" in schema:
        ref = schema["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
            errors.append(f"{path}: unsupported $ref {ref!r}")
            return errors
        target = root_schema.get("$defs", {}).get(ref.split("/")[-1])
        if not isinstance(target, dict):
            errors.append(f"{path}: unresolved $ref {ref!r}")
            return errors
        return _schema_validate(value, target, root_schema=root_schema, path=path, errors=errors)
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected const {schema['const']!r}, got {value!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: expected one of {schema['enum']!r}, got {value!r}")
    expected_type = schema.get("type")
    if expected_type:
        valid = False
        if expected_type == "object":
            valid = isinstance(value, dict)
        elif expected_type == "array":
            valid = isinstance(value, list)
        elif expected_type == "string":
            valid = isinstance(value, str)
        elif expected_type == "boolean":
            valid = isinstance(value, bool)
        elif expected_type == "integer":
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif expected_type == "number":
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not valid:
            errors.append(f"{path}: expected {expected_type}, got {_json_type_name(value)}")
            return errors
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required key {key!r}")
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        additional = schema.get("additionalProperties", True)
        if additional is False:
            for key in value:
                if key not in properties:
                    errors.append(f"{path}: unexpected key {key!r}")
        for key, child_schema in properties.items():
            if key in value and isinstance(child_schema, dict):
                _schema_validate(value[key], child_schema, root_schema=root_schema, path=f"{path}.{key}", errors=errors)
    if isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(value):
                _schema_validate(item, item_schema, root_schema=root_schema, path=f"{path}[{idx}]", errors=errors)
    if isinstance(value, str) and "minLength" in schema and len(value) < int(schema["minLength"]):
        errors.append(f"{path}: string shorter than minLength={schema['minLength']}")
    return errors


def validate_payload(payload: JsonDict, schema_path: Path) -> None:
    schema, err = _load_json(schema_path)
    if err:
        raise ReportError(err)
    assert schema is not None
    errors = _schema_validate(payload, schema, root_schema=schema)
    if errors:
        raise ReportError("payload schema validation failed: " + "; ".join(errors[:20]))


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None or value == {} or value == []:
        return "pending"
    if isinstance(value, float):
        return f"{value}{suffix}"
    return f"{value}{suffix}"


def _fmt_speedup(value: Any) -> str:
    if value is None or value == {} or value == []:
        return "pending"
    return f"{value}x"


def _fmt_resource(resource: JsonDict) -> str:
    if not resource:
        return "pending"
    parts = []
    if "peak_gpu_memory_mib" in resource:
        parts.append(f"peak GPU memory {resource['peak_gpu_memory_mib']} MiB")
    if "peak_host_memory_gib" in resource:
        parts.append(f"peak host memory {resource['peak_host_memory_gib']} GiB")
    if "peak_temperature_c" in resource:
        parts.append(f"peak temperature {resource['peak_temperature_c']} C")
    if "peak_power_w" in resource:
        parts.append(f"peak power {resource['peak_power_w']} W")
    return "; ".join(parts) if parts else "pending"


def render_markdown(payload: JsonDict) -> str:
    sections = payload["sections"]
    workload = sections["workload"].get("data", {})
    baseline = sections["baseline"].get("data", {})
    turbo = sections["turbo"].get("data", {})
    dlo = sections["dlo"].get("data", {})
    exact = sections["exact_kernels"].get("data", {})
    sol = sections["sol_attn"].get("data", {})
    dmd = sections["dmd"].get("data", {})

    workload_line = "pending"
    if workload:
        width = workload.get("width")
        height = workload.get("height")
        duration = workload.get("duration_seconds")
        frames = workload.get("expected_frames")
        fps = workload.get("fps")
        if width and height and duration and frames and fps:
            workload_line = f"{width}x{height}, {duration}s, {frames} frames, {fps} FPS"
        else:
            workload_line = f"width={_fmt(width)}, height={_fmt(height)}, duration={_fmt(duration, 's')}, frames={_fmt(frames)}, FPS={_fmt(fps)}"

    lines: list[str] = [
        "# MiniMax-H3 A6000 Performance Report",
        "",
        f"Status: **{payload['status']}**. {payload['non_execution_notice']}",
        f"Schema: `{payload['schema_version']}`.",
        f"Evidence root: `{payload['input_root']}`.",
        "",
        "## Scope and lane boundaries",
        "",
        f"- Fidelity lane: `{payload['lanes']['fidelity']}`. Only evidence explicitly retained in this lane is treated as BF16-exact.",
        f"- Practical lane: `{payload['lanes']['practical']}`. Turbo merged LoRA and diagnostic acceleration work stay here unless later evidence changes the boundary.",
        "- Turbo practical results must not be relabeled as BF16-exact/fidelity results.",
        "- Missing DLO/Sol-Attn/DMD/AV evidence is reported as **pending** or **blocked**; no missing value is estimated.",
        "",
        "## Platform and workload",
        "",
        f"- Platform: `{_fmt(baseline.get('platform'))}`; physical device `{baseline.get('physical_device') or 'pending'}`.",
        f"- Workload: {workload_line}.",
        f"- Task/model: task `{_fmt(workload.get('task'))}`, partition `{_fmt(workload.get('partition'))}`, baseline dense steps `{_fmt(workload.get('baseline_num_inference_steps'))}`.",
        f"- Audio: {_fmt(workload.get('audio_channels'))} channels at {_fmt(workload.get('audio_sample_rate_hz'))} Hz.",
        f"- Prompt/workload attribution: {_fmt(workload.get('prompt_source'))}. {workload.get('attribution', 'pending attribution evidence')}",
        "",
        "## Baseline v2 BF16-exact denominator",
        "",
    ]
    if sections["baseline"]["status"] == "pending":
        lines.append(f"- **pending**: {sections['baseline'].get('reason', 'missing baseline evidence')}.")
    else:
        all_req = baseline.get("all_requests", {})
        warm = baseline.get("warm_primary_denominator", {})
        session = baseline.get("session_first_requests", {})
        lines.extend(
            [
                f"- Evidence: {', '.join('`' + e + '`' for e in sections['baseline']['evidence'])}.",
                f"- Status: `{baseline.get('status')}`; track `{baseline.get('track')}`; schema `{baseline.get('schema')}`.",
                f"- All requests: N={_fmt(all_req.get('n'))}, median={_fmt(all_req.get('median_s'), 's')}, mean={_fmt(all_req.get('mean_s'), 's')}, CV={_fmt(all_req.get('cv_percent'), '%')}.",
                f"- Warm-primary denominator: N={_fmt(warm.get('n'))}, median={_fmt(warm.get('median_s'), 's')}, mean={_fmt(warm.get('mean_s'), 's')}, CV={_fmt(warm.get('cv_percent'), '%')}.",
                f"- Session-first requests: N={_fmt(session.get('n'))}; service sessions={_fmt(baseline.get('service_session_count'))}.",
                f"- Resource envelope: {_fmt_resource(baseline.get('resource', {}))}.",
                f"- Structural AV pass count: {_fmt(baseline.get('structural_av_pass_count'))}/{_fmt(baseline.get('run_count'))}.",
                f"- Claim boundary: {str(baseline.get('claim_boundary', 'pending')).rstrip('.')}.",
            ]
        )
    lines.extend(["", "## Turbo GPU3 paired practical timing and quality", ""])
    if sections["turbo"]["status"] == "pending":
        lines.append(f"- **pending**: {sections['turbo'].get('reason', 'missing Turbo timing evidence')}.")
    else:
        lines.extend(
            [
                f"- Run: `{turbo.get('run_id')}`; status `{turbo.get('status')}`; track `{turbo.get('track')}`.",
                f"- Baseline denominator: {turbo.get('baseline_denominator') or 'pending'}.",
                f"- Merge: status `{_fmt(turbo.get('merge_status'))}`, strength {_fmt(turbo.get('merge_strength'))}, completed shards {_fmt(turbo.get('completed_shard_count'))}.",
                f"- Formal paired N per schedule: {_fmt(turbo.get('paired_formal_n_per_schedule'))}; excluded warmups: {turbo.get('excluded_warmups') or 'pending'}.",
                f"- Resource envelope: {_fmt_resource(turbo.get('resource', {}))}.",
                f"- AV timing outputs: structural pass {_fmt((turbo.get('av') or {}).get('structural_av_pass_count'))}/{_fmt((turbo.get('av') or {}).get('referenced_av_count'))}; missing/unreadable AV {_fmt((turbo.get('av') or {}).get('missing_or_unreadable_av_count'))}.",
                "",
                "| Schedule | Steps | N | Median (s) | Mean (s) | CV (%) | Speedup vs BF16 warm N10 | Lane |",
                "|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for step, row in sorted((turbo.get("schedules") or {}).items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0])):
            lines.append(
                f"| {step}-step | {_fmt(row.get('steps'))} | {_fmt(row.get('n'))} | {_fmt(row.get('median_s'))} | {_fmt(row.get('mean_s'))} | {_fmt(row.get('cv_percent'))} | {_fmt_speedup(row.get('speedup_vs_same_gpu3_bf16_warm_n10_median'))} | `{PRACTICAL}` |"
            )
        quality = turbo.get("quality_suite") if isinstance(turbo.get("quality_suite"), dict) else {}
        qdata = quality.get("data") if isinstance(quality.get("data"), dict) else {}
        lines.extend(
            [
                "",
                f"- Quality suite: status `{_fmt(qdata.get('status'))}`, cases={_fmt(qdata.get('case_count'))}, pairs={_fmt(qdata.get('pair_count'))}, structural AV pass={_fmt(qdata.get('all_cases_structural_av_pass'))}.",
                f"- AV/semantic boundary: human auditory listening `{_fmt(qdata.get('human_auditory_listening'))}`; semantic_quality_certified={_fmt(qdata.get('semantic_quality_certified'))}; quality certification `{_fmt(qdata.get('quality_certification'))}`.",
                "- Practical recommendation boundary: 8-step remains the default practical candidate; 4-step is ultra-fast/quality-cost experimental, not fidelity evidence.",
            ]
        )
    lines.extend(["", "## DLO resident-layer optimization", ""])
    if sections["dlo"]["status"] == "pending":
        lines.append(f"- **pending**: {sections['dlo'].get('reason', 'missing DLO evidence')}.")
    else:
        plan = dlo.get("plan", {})
        baseline_denominator = plan.get("baseline_denominator", {}) if isinstance(plan, dict) else {}
        lines.extend(
            [
                f"- Plan status: `{_fmt(plan.get('status'))}`; baseline resident_layers={_fmt(baseline_denominator.get('resident_layers'))}; candidates={_fmt(plan.get('candidate_resident_layers'))}.",
                f"- Detached continuation: {dlo.get('detached_continuation') or 'pending'}.",
                "- Capacity gates are not formal 50-step/N10 performance unless separately marked present.",
                "",
                "| Resident layers | Stage | Result status | Gate status | Baseline 5-step (s) | Candidate 5-step (s) | Speedup | Hash match | Resource |",
                "|---:|---|---|---|---:|---:|---:|---|---|",
            ]
        )
        for attempt in dlo.get("capacity_attempts", []):
            if not isinstance(attempt, dict):
                continue
            lines.append(
                f"| {_fmt(attempt.get('resident_layers'))} | {_fmt(attempt.get('stage'))} | `{_fmt(attempt.get('result_status'))}` | `{_fmt(attempt.get('status'))}` | {_fmt(attempt.get('baseline_5step_latency_s'))} | {_fmt(attempt.get('candidate_5step_latency_s'))} | {_fmt_speedup(attempt.get('capacity_5step_speedup_candidate_vs_baseline'))} | {_fmt(attempt.get('exact_hash_match'))} | {_fmt_resource(attempt.get('resource', {}))} |"
            )
        lines.extend(
            [
                "",
                f"- Candidate-50: **{_fmt((dlo.get('candidate50') or {}).get('status'))}** — {(dlo.get('candidate50') or {}).get('reason', 'evidence present')}.",
                f"- Formal DLO N10: **{_fmt((dlo.get('formal_n10') or {}).get('status'))}** — {(dlo.get('formal_n10') or {}).get('reason', 'evidence present')}.",
            ]
        )
    lines.extend(["", "## Exact-kernel diagnostics and Sol-Attn", ""])
    if sections["exact_kernels"]["status"] == "pending":
        lines.append(f"- Exact-kernel diagnostics: **pending** ({sections['exact_kernels'].get('reason', 'missing evidence')}).")
    else:
        lines.extend(
            [
                f"- Exact-kernel lane: `{exact.get('lane')}`; microbenchmark scope `{_fmt(exact.get('microbenchmark_scope'))}`; model_load={_fmt(exact.get('microbenchmark_model_load'))}.",
                "",
                "| Kernel | N | PyTorch median (ms) | Triton candidate median (ms) | Median speedup |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in exact.get("microbenchmarks", []):
            lines.append(
                f"| `{row.get('kernel')}` | {_fmt(row.get('n'))} | {_fmt(row.get('pytorch_eager_median_ms'))} | {_fmt(row.get('triton_candidate_median_ms'))} | {_fmt_speedup(row.get('speedup_eager_over_triton_median'))} |"
            )
        modes = ((exact.get("e2e_ablation") or {}).get("modes") or {}) if isinstance(exact.get("e2e_ablation"), dict) else {}
        if modes:
            lines.extend(["", "E2E diagnostic ablation quality/telemetry:"])
            for mode, row in modes.items():
                quality = row.get("quality_vs_dense", {}) if isinstance(row, dict) else {}
                telemetry = row.get("telemetry_ops", {}) if isinstance(row, dict) else {}
                lines.append(
                    f"- `{mode}`: video_mean_mse={_fmt(quality.get('video_mean_mse'))}; audio_waveform_cosine={_fmt(quality.get('audio_waveform_cosine'))}; latency_s={_fmt(row.get('latency_s') if isinstance(row, dict) else None)}; telemetry_ops={list(telemetry.keys())}."
                )
        lines.append("- Exact-kernel diagnostics are retained as diagnostic/candidate evidence only unless a separate accepted N10 speed result exists.")
    if sections["sol_attn"]["status"] == "pending":
        lines.append(f"- Sol-Attn: **pending** ({sections['sol_attn'].get('reason', 'missing evidence')}).")
    else:
        legacy = sol.get("legacy_kernel_diagnostic", {}) if isinstance(sol.get("legacy_kernel_diagnostic"), dict) else {}
        runtime = sol.get("strict_runtime", {}) if isinstance(sol.get("strict_runtime"), dict) else (sol.get("strict_r6_runtime", {}) if isinstance(sol.get("strict_r6_runtime"), dict) else {})
        bench = legacy.get("bench", {}) if isinstance(legacy.get("bench"), dict) else {}
        h3_e2e = sol.get("h3_e2e", {}) if isinstance(sol.get("h3_e2e"), dict) else {}
        telemetry = runtime.get("telemetry", {}) if isinstance(runtime.get("telemetry"), dict) else {}
        resource = runtime.get("resource", {}) if isinstance(runtime.get("resource"), dict) else {}
        times = runtime.get("paired_http_time_total_s", {}) if isinstance(runtime.get("paired_http_time_total_s"), dict) else {}
        image = runtime.get("image_identity", {}) if isinstance(runtime.get("image_identity"), dict) else {}
        matched = sol.get("matched_retest", {}) if isinstance(sol.get("matched_retest"), dict) else {}
        formal = sol.get("formal_n10", {}) if isinstance(sol.get("formal_n10"), dict) else {}
        runtime_label = runtime.get("runtime_label") or image.get("runtime_label") or "unknown"
        timing_ratio_label = "dense/opt-in timing ratio (diagnostic only, not a speedup claim)"
        timing_ratio_value = runtime.get('paired_http_ratio_dense_over_opt_in_not_speedup')
        lines.extend(
            [
                f"- Sol-Attn legacy toy/kernel diagnostic: model_load={_fmt(legacy.get('model_load'))}; run_dir=`{_fmt(legacy.get('run_dir'))}`.",
                f"- Sol-Attn H3 diagnostic deployment boundary: `{_fmt(sol.get('deployment_status'))}`.",
                f"- Sol-Attn toy/kernel bench: dense median={_fmt(bench.get('dense_median_ms'))} ms; sparse median={_fmt(bench.get('sparse_median_ms'))} ms; dense/sparse median speedup={_fmt(bench.get('speedup_dense_over_sparse_median'))}.",
                f"- Sol-Attn {runtime_label} supervisor (current selected run by readable workload/version-label provenance, not run-id text prefix): status `{_fmt((runtime.get('supervisor') or {}).get('status') if isinstance(runtime.get('supervisor'), dict) else None)}`, latest_run_id `{_fmt((runtime.get('supervisor') or {}).get('latest_run_id') if isinstance(runtime.get('supervisor'), dict) else None)}`, classified `{_fmt(runtime.get('classification'))}`.",
                f"- Sol-Attn {runtime_label} readable provenance: image_tag=`{_fmt(image.get('image_tag'))}`, version=`{_fmt(image.get('version_label'))}`, required_version=`{_fmt(image.get('required_version_label'))}`, title=`{_fmt(image.get('title_label'))}`; opaque image/output identifiers are omitted and are not classification evidence.",
                f"- Sol-Attn {runtime_label} HTTP timing: dense={_fmt(times.get('dense_h3_backend_reference'), 's')}, opt-in={_fmt(times.get('sol_attn_opt_in'), 's')}, {timing_ratio_label}={_fmt_speedup(timing_ratio_value)}.",
                f"- Sol-Attn {runtime_label} telemetry: sparse_candidates={_fmt(telemetry.get('sparse_candidate_calls'))}, sparse_calls={_fmt(telemetry.get('sparse_calls'))}, fallback_calls={_fmt(telemetry.get('fallback_calls'))}, materialized_copy_calls={_fmt(telemetry.get('materialized_copy_calls'))}, materialized_copy_bytes={_fmt(telemetry.get('materialized_copy_bytes'))}, declines={telemetry.get('decline_reasons') or 'pending'}, density_samples={_fmt(len(telemetry.get('density_samples', [])) if isinstance(telemetry.get('density_samples'), list) else None)}.",
                f"- Sol-Attn {runtime_label} resource envelope: {_fmt_resource(resource)}.",
                f"- Sol-Attn H3 end-to-end: **{_fmt(h3_e2e.get('status'))}** — {h3_e2e.get('reason', 'pending')}.",
                "- Sol-Attn diagnostic boundary: the selected 5-step run may be used only as sparse-execution metadata plumbing evidence; matched-workload quality/correctness and formal performance promotion remain separate follow-up gates.",
            ]
        )
        if matched.get("status") == "proceed_to_formal_n10_candidate":
            matched_boundary = (
                "This N=3 route gate led to the later accepted formal N>=10 gate; the N=3 gate itself is not formal N10, not a speedup claim, not BF16 fidelity, and not quality-equivalence certification."
                if formal.get("status") == "accepted_formal_n10_same_gpu_sol_attn_speed_candidate"
                else "This recommends a future formal N>=10 Sol-Attn run, but is not formal N10, not a speedup claim, not BF16 fidelity, and not quality-equivalence certification."
            )
            lines.append(
                f"- Latest r8 matched-workload route decision: `{matched.get('status')}`; evidence `{matched.get('decision_path')}`, terminal recheck `{matched.get('terminal_recheck_path')}`. Completed pairs={matched.get('completed_pairs')}/{matched.get('requested_pairs')}; median HTTP-time improvement={_fmt(matched.get('median_http_time_improvement_pct'))}%; route threshold>{_fmt(matched.get('timing_threshold_pct'))}%; failed_gates={matched.get('failed_gates')}. {matched_boundary}"
            )
        elif matched.get("status") not in {None, "not_available"}:
            lines.append(f"- Latest r8 matched-workload retest CPU inspection: `{matched.get('status')}`; evidence `{matched.get('evidence_path')}`. {matched.get('reason', 'pending')}.")
        if formal.get("status") not in {None, "not_available"}:
            formal_evidence = formal.get("decision_path") or formal.get("status_path") or formal.get("source_run_dir")
            formal_boundary = (
                "Accepted only inside the formal matched 5-step Sol-Attn opt-in lane; not BF16 fidelity, not Turbo/DLO/DMD evidence, not release approval, and not human-auditory/semantic quality certification."
                if formal.get("status") == "accepted_formal_n10_same_gpu_sol_attn_speed_candidate"
                else "No formal Sol-Attn speedup, BF16 fidelity, release, or quality-equivalence claim is created by a nonterminal/incomplete gate."
            )
            lines.append(
                f"- Latest r8 formal N>=10 gate CPU inspection: `{formal.get('status')}`; evidence `{formal_evidence}`. "
                f"Requested pairs={_fmt(formal.get('requested_pairs'))}; started pairs={_fmt(formal.get('started_pairs'))}; completed pairs={_fmt(formal.get('completed_pairs'))}; "
                f"supervisor_status={_fmt(formal.get('supervisor_status'))}; same_expected_gpu={_fmt(formal.get('same_expected_gpu'))}. {formal.get('reason', 'pending')}. "
                f"{formal_boundary}"
            )
    lines.extend(["", "## DMD / DMD2 status", ""])
    if sections["dmd"]["status"] == "pending":
        lines.append(f"- **pending**: {sections['dmd'].get('reason', 'missing DMD evidence')}.")
    else:
        lines.extend(
            [
                f"- Status: **{_fmt(dmd.get('status'))}**; track limit `{_fmt(dmd.get('track_limit'))}`.",
                f"- Boundary: {str(dmd.get('claim_boundary', 'pending')).rstrip('.')}.",
            ]
        )
    lines.extend(["", "## Pending stages and blockers", ""])
    if payload["pending_items"]:
        for item in payload["pending_items"]:
            lines.append(f"- `{item['section']}`: **{item['status']}** — {item['reason']}.")
    else:
        lines.append("- No pending schema-tracked stages.")
    if payload["blockers"]:
        lines.extend(["", "Blockers:"])
        for item in payload["blockers"]:
            lines.append(f"- `{item['scope']}`: **{item['status']}** — {item['reason']}.")
    lines.extend(
        [
            "",
            "## Evidence index",
            "",
        ]
    )
    for item in payload["evidence_index"]:
        lines.append(f"- `{item}`")
    lines.extend(
        [
            "",
            "## Reproduction command",
            "",
            "```bash",
            "python3 tools/minimax_h3_a6000_performance_report.py --evidence-root technical_report/evidence/minimax_h3_desktop --out technical_report/minimax_h3_a6000_performance.md",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the CPU-only MiniMax-H3 A6000 performance report from local evidence.")
    parser.add_argument("--evidence-root", type=Path, required=True, help="technical_report/evidence/minimax_h3_desktop root")
    parser.add_argument("--out", type=Path, required=True, help="Markdown report output path")
    parser.add_argument("--summary-out", type=Path, help="Optional structured JSON payload output")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_REL, help="JSON schema path used to validate the structured payload")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root for relative paths")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    evidence_root = args.evidence_root if args.evidence_root.is_absolute() else repo_root / args.evidence_root
    out_path = args.out if args.out.is_absolute() else repo_root / args.out
    summary_path = args.summary_out if args.summary_out is None or args.summary_out.is_absolute() else repo_root / args.summary_out
    schema_path = args.schema if args.schema.is_absolute() else repo_root / args.schema
    try:
        payload = build_payload(evidence_root, repo_root=repo_root)
        validate_payload(payload, schema_path)
        markdown = render_markdown(payload)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
        if summary_path is not None:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"PASS performance report={_repo_rel(out_path, repo_root)} pending_items={len(payload['pending_items'])}")
        if summary_path is not None:
            print(f"summary={_repo_rel(summary_path, repo_root)}")
        return 0
    except ReportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
