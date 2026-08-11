from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from minimax_h3_a6000_performance_report import (  # noqa: E402
    build_payload,
    main,
    render_markdown,
    validate_payload,
)

SCHEMA = ROOT / "schemas" / "minimax_h3_a6000_performance_report.schema.json"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stats(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean_s": statistics.mean(values),
        "median_s": statistics.median(values),
        "cv_percent": statistics.stdev(values) / statistics.mean(values) * 100 if len(values) > 1 else 0.0,
        "min_s": min(values),
        "max_s": max(values),
    }


def _minimal_report_evidence(root: Path) -> None:
    repo = root.parents[2]
    physical = {"host_gpu_index": 3, "compute_capability": "8.6", "uuid": "GPU-test"}
    _write_json(
        root / "baseline_a6000" / "baseline_contract.json",
        {
            "schema": "argus-minimax-h3-a6000-fidelity-baseline-v1",
            "platform": "single_a6000_48gb_workstation",
            "partition": "FL2VA",
            "model_repo": "MiniMaxAI/MiniMax-H3",
            "checkpoint_revision": "6818f6c32d12b210915e44ad56a4228c2608f160",
            "runtime_source_commit": "8e2e9b6b53e86e6a479ed2c0a53782f655f60e04",
            "workload": {
                "task": "t2va",
                "width": 1344,
                "height": 768,
                "duration_seconds": 5.166667,
                "expected_frames": 124,
                "fps": 24,
                "audio_sample_rate_hz": 32000,
                "audio_channels": 2,
                "num_inference_steps": 50,
                "seed": 0,
                "prompt_sha256": "prompt-sha",
                "prompt_source": "NVlabs/Sana sol-engine t2va_example_1 at test-pin",
            },
        },
    )
    latencies = [100.0, 90.0, 101.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 102.0, 98.0, 99.0]
    first_indices = {1, 3, 11}
    runs = []
    for idx, latency in enumerate(latencies, start=1):
        kind = "session_first" if idx in first_indices else "warm"
        runs.append({"run": f"run{idx:02d}", "run_index": idx, "kind": kind, "latency_s": latency, "structural_av_pass": True})
    warm = [run["latency_s"] for run in runs if run["kind"] == "warm"]
    first = [run["latency_s"] for run in runs if run["kind"] == "session_first"]
    _write_json(
        root / "baseline_a6000" / "baseline_certification.json",
        {
            "schema": "argus-h3-a6000-fidelity-baseline-certification-v2",
            "status": "certified_internal_same_physical_device_baseline",
            "track": "fidelity_bf16_exact",
            "platform": "single_a6000_48gb_workstation",
            "physical_device": physical,
            "service_sessions": {"count": 3},
            "all_requests": _stats(latencies),
            "warm_requests_primary_denominator": _stats(warm),
            "session_first_requests": _stats(first),
            "resource": {"max_gpu_mem_mib": 26836.0, "max_host_used_gib": 204.8, "max_temp_c": 84.0},
            "runs": runs,
            "claim_boundary": "Internal same-physical-device A6000 BF16 fidelity baseline.",
        },
    )

    timing_id = "gpu3_turbo_paired_n10_test"
    (root / "turbo_merged" / "timing_repeats").mkdir(parents=True, exist_ok=True)
    (root / "turbo_merged" / "timing_repeats" / "LATEST_RUN_ID").write_text(timing_id + "\n", encoding="utf-8")
    timing_dir = root / "turbo_merged" / "timing_repeats" / timing_id
    schedules = {}
    pairs = []
    for step, base in (("4", 10.0), ("8", 20.0)):
        values = [base + idx * 0.01 for idx in range(10)]
        step_runs = []
        for idx, latency in enumerate(values, start=1):
            av_rel = f"technical_report/evidence/minimax_h3_desktop/turbo_merged/timing_repeats/{timing_id}/outputs/pair{idx:02d}_{step}step_av_validation.json"
            _write_json(
                repo / av_rel,
                {
                    "width": 1344,
                    "height": 768,
                    "average_rate": "24",
                    "decoded_video_frames": 124,
                    "audio_channels": 2,
                    "audio_sample_rate_hz": 32000,
                    "steps": int(step),
                    "structural_av_contract_pass": True,
                },
            )
            step_runs.append({"pair_index": idx, "latency_s": latency, "av_validation": av_rel})
        stat = _stats(values)
        schedules[step] = {
            **stat,
            "runs": step_runs,
            "speedup_denominator_s": statistics.median(warm),
            "speedup_vs_same_gpu3_bf16_warm_n10_median": statistics.median(warm) / stat["median_s"],
        }
    for idx in range(1, 11):
        pairs.append({"pair_index": idx, "latency_4step_s": schedules["4"]["runs"][idx - 1]["latency_s"], "latency_8step_s": schedules["8"]["runs"][idx - 1]["latency_s"]})
    _write_json(timing_dir / "merge_manifest.json", {"status": "completed", "merge": {"strength": 1.0}, "completed_shards": {"a": {}, "b": {}}})
    _write_json(
        timing_dir / "timing_summary.json",
        {
            "schema": "argus-ir04-turbo-paired-timing-v1",
            "status": "pass_same_physical_device_paired_n10",
            "track": "practical_disclosed_approx",
            "physical_device": physical,
            "baseline_denominator": {"kind": "warm_requests_primary_denominator", "n": 10, "median_s": statistics.median(warm), "same_physical_device": True},
            "excluded_warmups": {"4step": {"latency_s": 10.5}, "8step": {"latency_s": 20.5}},
            "pairs": pairs,
            "schedules": schedules,
            "resource": {"max_gpu_memory_used_mib": 26836.0, "max_host_memory_used_gib": 195.1, "max_temperature_c": 83.0},
            "quality_scope": "All formal timing outputs passed strict structural AV validation; semantic quality is separate.",
        },
    )

    quality_id = "gpu3_turbo_quality_test"
    (root / "turbo_merged" / "LATEST_QUALITY_SUITE_RUN_ID").write_text(quality_id + "\n", encoding="utf-8")
    quality_dir = root / "turbo_merged" / "quality_suite_runs" / quality_id
    _write_json(
        quality_dir / "quality_suite_analysis.json",
        {
            "status": "structural_av_suite_pass_semantic_quality_not_certified",
            "track": "practical_disclosed_approx",
            "case_count": 24,
            "pair_count": 12,
            "all_cases_structural_av_pass": True,
            "latency_by_step": {"4": {"n": 12, "median_s": 10.1, "cv_percent": 1.0}, "8": {"n": 12, "median_s": 20.1, "cv_percent": 1.0}},
            "quality_certification": "pending_human_review_and_prompt_matched_fidelity_references",
            "metric_limits": ["Do not relabel this practical Turbo suite as fidelity evidence."],
        },
    )
    _write_json(quality_dir / "baseline_seed0_quality_comparison.json", {"4": {"audio_cosine": 0.4, "video_mse": 1.0}, "8": {"audio_cosine": 0.8, "video_mse": 0.5}})
    (quality_dir / "human_review.md").parent.mkdir(parents=True, exist_ok=True)
    (quality_dir / "human_review.md").write_text("PENDING\n", encoding="utf-8")

    _write_json(
        root / "dlo_autotune" / "resident_layer_candidates.json",
        {
            "status": "candidate_plan_derived_from_local_weight_headers",
            "baseline_denominator": {"resident_layers": 12, "warm_n": 10, "warm_median_s": statistics.median(warm)},
            "candidate_resident_layers": [{"resident_layers": 13}, {"resident_layers": 16}],
            "safety_model": {"thermal_stop_c": 88, "safety_cap_gpu_mem_mib": 45500.0},
        },
    )
    state = root / "dlo_autotune" / "detached_continuation"
    state.mkdir(parents=True, exist_ok=True)
    (state / "status.txt").write_text("running\n", encoding="utf-8")
    (state / "rl13_run_id.txt").write_text("a6000_dlo_capacity_5step_rl13_test\n", encoding="utf-8")
    dlo_run = root / "dlo_autotune" / "runs" / "a6000_dlo_capacity_5step_rl13_test"
    (root / "dlo_autotune" / "runs").mkdir(parents=True, exist_ok=True)
    (root / "dlo_autotune" / "runs" / "LATEST_RUN_ID").write_text("a6000_dlo_capacity_5step_rl13_test\n", encoding="utf-8")
    _write_json(
        dlo_run / "capacity_gate_verdict.json",
        {
            "schema": "argus-minimax-h3-a6000-dlo-capacity-gate-v1",
            "stage": "capacity-5step",
            "status": "pass",
            "resident_layers": 13,
            "baseline_5step": {"resident_layers": 12, "latency_s": 188.0, "sha256": "same"},
            "candidate_5step": {"resident_layers": 13, "latency_s": 186.0, "sha256": "same"},
            "resource": {"max_gpu_memory_used_mib": 28060.0, "max_temperature_c": 82.0},
        },
    )

    exact_dir = root / "sol_engine_port" / "gpu_exact_test"
    (root / "sol_engine_port").mkdir(parents=True, exist_ok=True)
    (root / "sol_engine_port" / "LATEST_GPU_EXACT_DIR").write_text("gpu_exact_test\n", encoding="utf-8")
    _write_json(exact_dir / "correctness.json", {"model_load": False, "validated_single_a6000_sm86": True, "cases": [{}, {}]})
    _write_json(
        exact_dir / "microbenchmark.json",
        {
            "scope": "kernel_candidates_only_not_h3_e2e",
            "model_load": False,
            "repeats": 3,
            "benchmarks": {"indexed_gate": {"pytorch_eager_ms": [10.0, 11.0, 12.0], "triton_candidate_ms": [2.0, 2.2, 2.4]}},
        },
    )
    ablation_dir = root / "sol_engine_port" / "r5_ablation_test"
    (root / "sol_engine_port" / "LATEST_R5_ABLATION_DIR").write_text("r5_ablation_test\n", encoding="utf-8")
    _write_json(ablation_dir / "adaln" / "quality_vs_dense.json", {"claim_scope": "diagnostic_ablation_only_not_fidelity_acceptance", "video_mean_mse": 0.0, "audio_waveform_cosine": 1.0})
    _write_json(ablation_dir / "adaln" / "exact_telemetry.json", {"ops": {"indexed_gate_bf16": {"calls": 4, "candidate": 4, "fallback": 0, "decline": 0, "strict_error": 0}}})
    _write_json(ablation_dir / "adaln" / "av_validation.json", {"width": 1344, "height": 768, "decoded_video_frames": 124, "average_rate": "24", "audio_channels": 2, "audio_sample_rate": 32000, "video_present": True, "audio_present": True, "steps": 50})
    (ablation_dir / "adaln" / "http_metrics.txt").write_text("time_total_s=187.0\n", encoding="utf-8")

    sol_dir = root / "sol_engine_port" / "sol_attn_gpu_test"
    (root / "sol_engine_port" / "LATEST_SOL_ATTN_GPU_DIR").write_text("sol_attn_gpu_test\n", encoding="utf-8")
    _write_json(
        sol_dir / "result.json",
        {
            "schema_version": "minimax_h3_a6000_sol_attn_sm86_harness_v1",
            "model_load": False,
            "device": "cuda:0",
            "capability": [8, 6],
            "correctness": {"compile_status": "compiled_and_launched", "elapsed_s": 1.0, "max_abs_valid": 0.00390625, "prefix_rows_equal_dense": True, "padding_rows_zero": True},
            "bench": {"kernel_candidates_only_not_h3_e2e": True, "dense_ms": {"median_ms": 0.12, "mean_ms": 0.13}, "sparse_ms": {"median_ms": 0.40, "mean_ms": 0.41}, "speedup_dense_over_sparse_median": 0.3},
        },
    )
    (root / "dmd_primary_source_note.md").write_text("Status: **P5 BLOCKED / RESEARCH ONLY / practical track only**. There is no first-source basis for an H3 DMD claim.\n", encoding="utf-8")


def _write_r6_sol_attn_runtime(
    root: Path,
    *,
    run_id: str = "sol_attn_h3_gpu2_5step_r6_20260811T000000Z",
    supervisor_status: str = "complete",
    exit_code: str = "0",
    diagnostic_status: str = "metadata_path_accepted_sparse_candidate_attempted",
    sparse_candidates: int = 3,
    decline_reasons: dict | None = None,
    dense_sha: str = "same-sha",
    sol_sha: str = "same-sha",
    write_telemetry: bool = True,
    write_runtime_files: bool = True,
) -> Path:
    sol_root = root / "sol_engine_port"
    supervisor = sol_root / "sol_attn_gpu2_supervisor"
    supervisor.mkdir(parents=True, exist_ok=True)
    (supervisor / "status.txt").write_text(supervisor_status + "\n", encoding="utf-8")
    (supervisor / "latest_run_id").write_text(run_id + "\n", encoding="utf-8")
    (supervisor / "exit_code").write_text(exit_code + "\n", encoding="utf-8")

    iid = "sha256:" + "a" * 64
    image_dir = sol_root / "r6_overlay_image"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "r6_image_iid.txt").write_text(iid + "\n", encoding="utf-8")

    run = sol_root / run_id
    run.mkdir(parents=True, exist_ok=True)
    (run / "resource_monitor.csv").write_text(
        "timestamp,gpu_memory_used_mib,gpu_util_percent,power_w,temperature_c,host_memory_used_bytes,host_memory_available_bytes,host_swap_used_bytes\n"
        "2026-08-11T00:00:00Z,24000,80,250.5,75,1000,2000,0\n",
        encoding="utf-8",
    )
    if not write_runtime_files:
        return run
    (run / "r6_image_identity.env").write_text(
        f"expected_r6_image_iid={iid}\n"
        f"actual_r6_image_iid={iid}\n"
        "actual_image_version_label=r6\n"
        "actual_image_base_label=argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r2\n"
        "actual_image_title_label=MiniMax-H3 A6000 r6 Sol-Attn integration overlay\n",
        encoding="utf-8",
    )
    (run / "workload.env").write_text(
        "steps=5\nseed=0\nwidth=1344\nheight=768\nfps=24\nduration=5.166667\nattention_backend=H3_A6000_SOL_ATTN\nsol_attn_cache=off\nnetwork=none\n",
        encoding="utf-8",
    )
    for mode, sha, latency in (("dense_h3_backend_reference", dense_sha, 120.0), ("sol_attn", sol_sha, 100.0)):
        mode_dir = run / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        (mode_dir / "http_metrics.txt").write_text(f"http_code=200\ntime_total_s={latency}\n", encoding="utf-8")
        _write_json(
            mode_dir / "av_validation.json",
            {
                "mode": mode,
                "steps": 5,
                "seed": 0,
                "sha256": sha,
                "bytes": 2048,
                "video_present": True,
                "audio_present": True,
                "width": 1344,
                "height": 768,
                "average_rate": "24",
                "decoded_video_frames": 124,
                "audio_sample_rate": 32000,
                "audio_channels": 2,
                "decoded_audio_frames": 10,
                "decoded_audio_samples": 160000,
            },
        )
    if write_telemetry:
        _write_json(
            run / "sol_attn" / "sol_attn_telemetry.sol_attn.json",
            {
                "dense_calls": 4,
                "sparse_candidate_calls": sparse_candidates,
                "sparse_calls": sparse_candidates,
                "fallback_calls": 0,
                "prefix_query_dense_calls": 0,
                "decline_reasons": decline_reasons or {},
                "fallback_reasons": {},
                "density_samples": [{"kind": "static_exact_block_lower_bound", "exact_density_lower_bound": 0.5}] if sparse_candidates else [],
            },
        )
    _write_json(
        run / "sol_attn_diagnostic_status.json",
        {
            "status": diagnostic_status,
            "scope": "diagnostic_5_step_sol_attn_metadata_gate_not_fidelity_or_performance_claim",
            "dense_sha256": dense_sha,
            "sol_attn_sha256": sol_sha,
        },
    )
    return run


def test_payload_and_markdown_preserve_lane_boundaries_and_pending(tmp_path: Path) -> None:
    evidence = tmp_path / "technical_report" / "evidence" / "minimax_h3_desktop"
    _minimal_report_evidence(evidence)

    payload = build_payload(evidence, repo_root=tmp_path)
    validate_payload(payload, SCHEMA)
    report = render_markdown(payload)

    assert payload["sections"]["baseline"]["data"]["warm_primary_denominator"]["n"] == 10
    assert payload["sections"]["turbo"]["data"]["track"] == "practical_disclosed_approx"
    assert payload["sections"]["turbo"]["data"]["av"]["structural_av_pass_count"] == 20
    assert payload["sections"]["dlo"]["data"]["capacity_passes"][0]["resident_layers"] == 13
    assert any(item["section"] == "dlo.formal_n10" for item in payload["pending_items"])
    assert payload["sections"]["sol_attn"]["data"]["h3_e2e"]["status"] == "pending"
    assert "1344x768, 5.166667s, 124 frames, 24 FPS" in report
    assert "NVlabs/Sana sol-engine" in report
    assert "Turbo practical results must not be relabeled as BF16-exact/fidelity" in report
    assert "Formal DLO N10: **pending**" in report
    assert "Sol-Attn H3 end-to-end: **pending**" in report


def test_sol_attn_r6_completed_metadata_is_speed_only_until_quality_gate(tmp_path: Path) -> None:
    evidence = tmp_path / "technical_report" / "evidence" / "minimax_h3_desktop"
    _minimal_report_evidence(evidence)
    _write_r6_sol_attn_runtime(evidence)

    payload = build_payload(evidence, repo_root=tmp_path)
    validate_payload(payload, SCHEMA)
    strict = payload["sections"]["sol_attn"]["data"]["strict_r6_runtime"]
    report = render_markdown(payload)

    assert strict["accepted_metadata"] is True
    assert strict["classification"] == "speed_only_no_quality"
    assert strict["paired_http_speedup_dense_over_sol_attn"] == 1.2
    assert strict["telemetry"]["sparse_candidate_calls"] == 3
    assert strict["telemetry"]["density_samples"]
    assert strict["release_manifest_eligible"] is False
    assert any(item["section"] == "sol_attn.h3_e2e" and item["status"] == "speed_only_no_quality" for item in payload["pending_items"])
    assert "Sol-Attn r6 supervisor" in report


def test_sol_attn_r6_active_supervisor_remains_pending(tmp_path: Path) -> None:
    evidence = tmp_path / "technical_report" / "evidence" / "minimax_h3_desktop"
    _minimal_report_evidence(evidence)
    _write_r6_sol_attn_runtime(evidence, supervisor_status="running")

    strict = build_payload(evidence, repo_root=tmp_path)["sections"]["sol_attn"]["data"]["strict_r6_runtime"]

    assert strict["classification"] == "pending_non_terminal_supervisor_status"
    assert "still 'running'" in strict["reason"]
    assert strict["accepted_metadata"] is False


def test_sol_attn_r6_fail_closed_missing_metadata_decline(tmp_path: Path) -> None:
    evidence = tmp_path / "technical_report" / "evidence" / "minimax_h3_desktop"
    _minimal_report_evidence(evidence)
    _write_r6_sol_attn_runtime(
        evidence,
        diagnostic_status="fail_closed_dense_fallback",
        sparse_candidates=0,
        decline_reasons={"missing_h3_hook_metadata": 12},
    )

    payload = build_payload(evidence, repo_root=tmp_path)
    strict = payload["sections"]["sol_attn"]["data"]["strict_r6_runtime"]
    report = render_markdown(payload)

    assert strict["classification"] == "fail_closed_missing_metadata"
    assert strict["accepted_metadata"] is False
    assert "paired_http_speedup_dense_over_sol_attn" not in strict
    assert strict["paired_http_ratio_dense_over_opt_in_not_speedup"] == 1.2
    assert strict["telemetry"]["decline_reasons"] == {"missing_h3_hook_metadata": 12}
    assert "dense/opt-in timing ratio (not a speedup claim)=1.2x" in report


def test_sol_attn_r6_runtime_failure_quality_drift_and_stale_rejection(tmp_path: Path) -> None:
    evidence = tmp_path / "technical_report" / "evidence" / "minimax_h3_desktop"
    _minimal_report_evidence(evidence)
    _write_r6_sol_attn_runtime(evidence, supervisor_status="failed_rc_61", exit_code="61", write_runtime_files=False)
    runtime_failure = build_payload(evidence, repo_root=tmp_path)["sections"]["sol_attn"]["data"]["strict_r6_runtime"]
    assert runtime_failure["classification"] == "runtime_failure"

    drift_root = tmp_path / "drift" / "technical_report" / "evidence" / "minimax_h3_desktop"
    _minimal_report_evidence(drift_root)
    _write_r6_sol_attn_runtime(drift_root, dense_sha="dense-sha", sol_sha="sol-sha")
    no_hash_drift = build_payload(drift_root, repo_root=tmp_path / "drift")["sections"]["sol_attn"]["data"]["strict_r6_runtime"]
    assert no_hash_drift["classification"] == "speed_only_no_quality"
    assert no_hash_drift["accepted_metadata"] is True
    assert no_hash_drift["opaque_integrity_policy"]["output_identifiers"] == "omitted_not_evidence"

    stale_root = tmp_path / "stale" / "technical_report" / "evidence" / "minimax_h3_desktop"
    _minimal_report_evidence(stale_root)
    _write_r6_sol_attn_runtime(stale_root, run_id="sol_attn_gpu_20260809T173323Z")
    stale = build_payload(stale_root, repo_root=tmp_path / "stale")["sections"]["sol_attn"]["data"]["strict_r6_runtime"]
    assert stale["classification"] == "stale_or_dry_run_rejected"


def test_sparse_evidence_root_succeeds_with_explicit_pending(tmp_path: Path) -> None:
    evidence = tmp_path / "technical_report" / "evidence" / "minimax_h3_desktop"
    evidence.mkdir(parents=True)

    payload = build_payload(evidence, repo_root=tmp_path)
    validate_payload(payload, SCHEMA)
    report = render_markdown(payload)

    assert payload["sections"]["baseline"]["status"] == "pending"
    assert payload["sections"]["turbo"]["status"] == "pending"
    assert payload["sections"]["dmd"]["status"] == "pending"
    assert len(payload["pending_items"]) >= 5
    assert "**pending**" in report
    assert "no missing value is estimated" in report


def test_cli_writes_markdown_and_optional_summary(tmp_path: Path, capsys) -> None:
    evidence = tmp_path / "technical_report" / "evidence" / "minimax_h3_desktop"
    _minimal_report_evidence(evidence)
    out = tmp_path / "technical_report" / "minimax_h3_a6000_performance.md"
    summary = tmp_path / "technical_report" / "minimax_h3_a6000_performance.json"

    rc = main([
        "--repo-root",
        str(tmp_path),
        "--schema",
        str(SCHEMA),
        "--evidence-root",
        str(evidence),
        "--out",
        str(out),
        "--summary-out",
        str(summary),
    ])

    assert rc == 0
    assert "PASS performance report" in capsys.readouterr().out
    assert out.is_file()
    assert summary.is_file()
    text = out.read_text(encoding="utf-8")
    assert "practical_disclosed_approx" in text
    assert "DMD / DMD2" in text
    loaded = json.loads(summary.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "argus-minimax-h3-a6000-performance-report-v1"
