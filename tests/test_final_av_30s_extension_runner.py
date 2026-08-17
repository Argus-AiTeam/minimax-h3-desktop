from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from final_av_30s_extension_runner import (  # noqa: E402
    _attention_mechanisms,
    _summarize_cache_dit_telemetry,
    build_final_av_ffmpeg_command,
    build_final_video_ffmpeg_command,
    finalize_r10_cache_dit_matched,
    finalize_r9_adaptive_matched,
    finalize_r9_adaptive_matched_formal,
    finalize_r9_adaptive_matched_n3,
    request_timing_breakdown,
    start_server,
    summarize_stage_durations,
    wait_for_process_after_stdin_close,
)


def test_start_server_records_requested_attention_backend(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        evidence = Path(tmp)
        captured = {}

        class DummyProc:
            pass

        def fake_popen(cmd, stdout=None, stderr=None, text=None):  # type: ignore[no-untyped-def]
            captured["cmd"] = cmd
            return DummyProc()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        start_server(evidence, "/models/Turbo/FL2VA", 11, 12, 13, attention_backend="H3_A6000_SOL_ATTN")
        cmd = captured["cmd"]
        assert cmd[cmd.index("--diffusion-attention-backend") + 1] == "H3_A6000_SOL_ATTN"
        assert cmd[cmd.index("--dlo-resident-layers") + 1] == "13"


def test_start_server_defaults_to_enforce_eager(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        evidence = Path(tmp)
        captured = {}

        class DummyProc:
            pass

        def fake_popen(cmd, stdout=None, stderr=None, text=None):  # type: ignore[no-untyped-def]
            captured["cmd"] = cmd
            return DummyProc()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        start_server(evidence, "/models/Turbo/FL2VA", 11, 12, 13)
        cmd = captured["cmd"]
        assert "--enforce-eager" in cmd
        assert "--diffusion-compile-granularity" not in cmd
        assert "--no-diffusion-compile-dynamic" not in cmd


def test_start_server_regional_compile_omits_enforce_eager_and_uses_fixed_shape(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        evidence = Path(tmp)
        captured = {}

        class DummyProc:
            pass

        def fake_popen(cmd, stdout=None, stderr=None, text=None):  # type: ignore[no-untyped-def]
            captured["cmd"] = cmd
            return DummyProc()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        start_server(
            evidence,
            "/models/Turbo/FL2VA",
            11,
            12,
            13,
            regional_compile=True,
            diffusion_compile_dynamic=False,
        )
        cmd = captured["cmd"]
        assert "--enforce-eager" not in cmd
        assert cmd[cmd.index("--diffusion-compile-granularity") + 1] == "regional"
        assert "--no-diffusion-compile-dynamic" in cmd


def test_start_server_can_enable_cache_dit_summary_for_request_scoped_h3(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        evidence = Path(tmp)
        captured = {}

        class DummyProc:
            pass

        def fake_popen(cmd, stdout=None, stderr=None, text=None):  # type: ignore[no-untyped-def]
            captured["cmd"] = cmd
            return DummyProc()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        start_server(
            evidence,
            "/models/Turbo/FL2VA",
            11,
            12,
            13,
            attention_backend="H3_A6000_SOL_ATTN",
            server_cache_backend="cache_dit",
            enable_cache_dit_summary=True,
        )
        cmd = captured["cmd"]
        assert cmd[cmd.index("--cache-backend") + 1] == "cache_dit"
        assert "--enable-cache-dit-summary" in cmd


def test_attention_mechanisms_disclose_vae_spatial_tile_batching() -> None:
    mechanisms = _attention_mechanisms(
        "H3_A6000_SOL_ATTN",
        "r10_adaptive_tau1_5_step3_diag_vae_spatial_tile_batching",
        "lossless",
        True,
    )

    assert "guarded_adaptive_routing_tau1_5_diag_step_min_3" in mechanisms
    assert "video_vae_spatial_tile_batching_stack_tiling" in mechanisms
    assert "practical_approximate_vae_decode" in mechanisms


def test_attention_mechanisms_disclose_bounded_vae_tile_batching() -> None:
    mechanisms = _attention_mechanisms(
        "H3_A6000_SOL_ATTN",
        "r10_adaptive_tau1_5_step3_diag_vae_tile_batch_cap_7",
        "lossless",
        False,
        7,
    )

    assert "guarded_adaptive_routing_tau1_5_diag_step_min_3" in mechanisms
    assert "video_vae_bounded_spatial_tile_batching_cap_7" in mechanisms
    assert "video_vae_spatial_tile_batching_stack_tiling" not in mechanisms
    assert "practical_approximate_vae_decode" in mechanisms


def test_cache_dit_summary_parser_accepts_machine_readable_counters(tmp_path: Path) -> None:
    (tmp_path / "server.log").write_text(
        "\n".join(
            [
                "INFO Enabling cache-dit on transformer: Fn=1, Bn=0, W=4",
                "INFO Refreshing cache context for transformer with num_inference_steps: 8",
                'INFO Cache-DiT request summary json: {"cached_steps": 2, "executed_steps": 6, "transformer_executed_steps": 8, "stats_with_cache_fields": 1}',
                "INFO Cache-DiT request summary: cached_steps=2 executed_steps=6 transformer_executed_steps=8 pruned_steps=0 pruned_blocks=0 actual_blocks=0 stats_with_cache_fields=1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = _summarize_cache_dit_telemetry(
        tmp_path,
        request_quality="high",
        server_cache_backend="cache_dit",
        enable_cache_dit_summary=True,
    )["summary"]

    assert summary["parsed_reuse_or_skip_count"] == 2
    assert summary["parsed_compute_count"] == 6
    assert summary["refresh_log_count"] == 1


def test_cache_dit_summary_parser_records_warmup2_profile(tmp_path: Path) -> None:
    (tmp_path / "server.log").write_text(
        'INFO Cache-DiT request summary json: {"cached_steps": 3, "executed_steps": 5, "transformer_executed_steps": 8, "stats_with_cache_fields": 1}\n',
        encoding="utf-8",
    )

    summary = _summarize_cache_dit_telemetry(
        tmp_path,
        request_quality="high_warmup2",
        server_cache_backend="cache_dit",
        enable_cache_dit_summary=True,
    )

    assert summary["expected_config"]["max_warmup_steps"] == 2
    assert summary["expected_config"]["Fn_compute_blocks"] == 1
    assert summary["expected_config"]["Bn_compute_blocks"] == 0
    assert summary["summary"]["parsed_reuse_or_skip_count"] == 3
    assert summary["summary"]["parsed_compute_count"] == 5


def test_cache_dit_summary_parser_accepts_upstream_human_counter_lines(tmp_path: Path) -> None:
    (tmp_path / "server.log").write_text(
        "\n".join(
            [
                "[Cache-DiT] ⚡️Cache Steps and Residual Diffs Statistics: MiniMaxH3DiTBlock, Executed Steps: 6, Transformer Executed Steps: 8",
                "[Cache-DiT] Cache Steps: 2, [4, 6]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = _summarize_cache_dit_telemetry(
        tmp_path,
        request_quality="high",
        server_cache_backend="cache_dit",
        enable_cache_dit_summary=True,
    )["summary"]

    assert summary["parsed_reuse_or_skip_count"] == 2
    assert summary["parsed_compute_count"] == 6


def test_cache_dit_summary_parser_rejects_context_only_logs(tmp_path: Path) -> None:
    (tmp_path / "server.log").write_text(
        "\n".join(
            [
                "[Cache-DiT] Collected Context Config: DBCache_F1B0_W4I1M0MC1_R0.04_N8_CFG0, Calibrator Config: None",
                "[Cache-DiT] {'cache_config': DBCacheConfig(cache_type=<CacheType.DBCache: 'DBCache'>, Fn_compute_blocks=1, Bn_compute_blocks=0)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = _summarize_cache_dit_telemetry(
        tmp_path,
        request_quality="high",
        server_cache_backend="cache_dit",
        enable_cache_dit_summary=True,
    )["summary"]

    assert summary["parsed_reuse_or_skip_count"] is None
    assert summary["parsed_compute_count"] is None


def test_request_timing_breakdown_uses_server_header_for_true_transfer() -> None:
    timing = request_timing_breakdown(
        curl_wall_s=201.790,
        http_metrics={"time_total_s": 201.778, "time_starttransfer_s": 0.000189, "size_download": 9555899.0},
        headers={"x-inference-time-s": "201.751"},
    )

    assert timing["true_client_wait_seconds"] == 201.790
    assert timing["server_request_wall_seconds"] == 201.751
    assert abs(timing["response_transfer_seconds"] - 0.027) < 1e-9
    assert timing["curl_starttransfer_based_transfer_seconds"] > 200.0
    assert timing["response_transfer_source"] == "http_total_minus_x_inference_time_s"


def test_summarize_stage_durations_extracts_split_profile_components() -> None:
    chunks = [
        {
            "chunk_index": 1,
            "seed": 4200,
            "request_task": "t2va",
            "request": {
                "curl_wall_seconds": 10.0,
                "response_download_seconds": 0.25,
                "http_metrics": {"time_total_s": 9.5},
            },
            "stage_durations": {
                "MiniMaxH3Pipeline.encode_prompt": 1.0,
                "MiniMaxH3Pipeline.diffuse": 6.0,
                "MiniMaxH3Pipeline.decode": 2.0,
                "MiniMaxH3Pipeline.decode.video_vae.wall": 1.2,
                "MiniMaxH3Pipeline.decode.video_vae.device": 1.1,
                "MiniMaxH3Pipeline.decode.audio_vae.wall": 0.4,
                "MiniMaxH3Pipeline.decode.audio_vae.device": 0.35,
                "MiniMaxH3Pipeline.decode.video_crop_contiguous.wall": 0.1,
                "MiniMaxH3Pipeline.postprocess.video_device_to_host_copy.wall": 0.2,
                "MiniMaxH3Pipeline.postprocess.video_device_to_host_copy.bytes": 1234.0,
                "MiniMaxH3Pipeline.postprocess.audio_device_to_host_copy.wall": 0.03,
                "MiniMaxH3Pipeline.postprocess.audio_device_to_host_copy.bytes": 56.0,
                "OmniOpenAIServingVideo.response_encoding_mp4_bytes.wall": 0.7,
            },
        },
        {
            "chunk_index": 2,
            "seed": 4201,
            "request_task": "fl2va",
            "request": {
                "curl_wall_seconds": 11.0,
                "response_download_seconds": 0.30,
                "http_metrics": {"time_total_s": 10.5},
            },
            "stage_durations": {
                "MiniMaxH3Pipeline.encode_prompt": 1.5,
                "MiniMaxH3Pipeline._encode_video_conditions": 0.5,
                "MiniMaxH3Pipeline.diffuse": 6.5,
                "MiniMaxH3Pipeline.decode.video_vae.wall": 1.3,
                "MiniMaxH3Pipeline.decode.video_vae.device": 1.2,
                "MiniMaxH3Pipeline.decode.audio_vae.wall": 0.5,
                "MiniMaxH3Pipeline.decode.audio_vae.device": 0.45,
                "MiniMaxH3Pipeline.postprocess.video_device_to_host_copy.wall": 0.25,
                "MiniMaxH3Pipeline.postprocess.video_device_to_host_copy.bytes": 1234.0,
                "MiniMaxH3Pipeline.postprocess.audio_device_to_host_copy.wall": 0.04,
                "MiniMaxH3Pipeline.postprocess.audio_device_to_host_copy.bytes": 56.0,
                "OmniOpenAIServingVideo.response_encoding_mp4_bytes.wall": 0.8,
            },
        },
    ]

    summary = summarize_stage_durations(chunks)

    assert summary["split_profile_status"] == "present"
    assert summary["text_conditioning_seconds"] == 3.0
    assert summary["denoise_seconds"] == 12.5
    assert summary["video_vae_decode_wall_seconds"] == 2.5
    assert summary["audio_vae_decode_device_seconds"] == 0.8
    assert summary["server_encoding_mux_wall_seconds"] == 1.5
    assert summary["true_client_wait_seconds"] == 21.0
    assert summary["server_request_wall_seconds"] == 20.0
    assert summary["response_download_seconds"] == 0.55
    assert summary["python_cpu_sync_copy_bytes"] == 2580.0
    assert summary["per_chunk_variance"]["curl_wall_seconds"]["n"] == 2


def test_final_av_mux_commands_avoid_rawvideo_audio_single_process_truncation() -> None:
    video_cmd = build_final_video_ffmpeg_command(Path("/tmp/final_video.mov"))
    mux_cmd = build_final_av_ffmpeg_command(
        Path("/tmp/final_video.mov"),
        Path("/tmp/final_audio.raw"),
        Path("/tmp/final.mov"),
    )

    assert "-shortest" not in video_cmd
    assert "-shortest" not in mux_cmd
    assert video_cmd[video_cmd.index("-frames:v") + 1] == "720"
    assert "-an" in video_cmd
    assert "pipe:0" in video_cmd
    assert "pipe:0" not in mux_cmd
    assert mux_cmd[mux_cmd.index("-i") + 1] == "/tmp/final_video.mov"
    assert mux_cmd[mux_cmd.index("-i", mux_cmd.index("-ac")) + 1] == "/tmp/final_audio.raw"
    assert mux_cmd[mux_cmd.index("-map") + 1] == "0:v:0"
    assert mux_cmd[mux_cmd.index("-map", mux_cmd.index("-map") + 1) + 1] == "1:a:0"
    assert mux_cmd[mux_cmd.index("-c:v") + 1] == "copy"
    assert mux_cmd[-1] == "/tmp/final.mov"


def test_wait_for_process_after_manual_stdin_close_does_not_call_communicate() -> None:
    """Regression for Python 3.12 ValueError: flush of closed file."""
    code = "import sys; data=sys.stdin.buffer.read(); sys.stdout.buffer.write(data.upper()); sys.stderr.write('ok')"
    with tempfile.TemporaryFile() as stdout_tmp, tempfile.TemporaryFile() as stderr_tmp:
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdin=subprocess.PIPE,
            stdout=stdout_tmp,
            stderr=stderr_tmp,
        )
        assert proc.stdin is not None
        proc.stdin.write(b"final-av")
        proc.stdin.close()

        wait_for_process_after_stdin_close(proc, timeout=10)

        stdout_tmp.seek(0)
        stderr_tmp.seek(0)
        assert proc.returncode == 0
        assert stdout_tmp.read() == b"FINAL-AV"
        assert stderr_tmp.read() == b"ok"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fake_objective(value: float) -> dict:
    return {
        "subject_identity_consistency": {"status": "measured", "value": value},
        "background_consistency": {"status": "measured", "value": value},
        "camera_consistency": {"status": "measured", "value": value},
        "motion": {"status": "measured", "value": value},
        "automatic_red_flags": ["duplicate_window_fraction"] if value == 1.0 else [],
    }


def _fake_record(record_id: str, warm: float, cold: float, objective_value: float) -> dict:
    return {
        "deployment": {"physical_gpu_uuids": ["GPU-test"], "gpu_count_visible": 1, "gpu_count_used": 1},
        "workload_fingerprint": "wf-test",
        "production": {"generation_mode": "extension", "chunk_count": 6},
        "output_av": {"final_accounting_complete": True, "video": {"frames": 720}, "audio": {"effective_samples_per_channel": 960000}},
        "quality": {"objective": _fake_objective(objective_value)},
        "resources": {
            "peak_gpu_memory_mib": {"value": 27946.0},
            "peak_host_memory_gib": {"value": 140.0},
            "peak_power_w": {"value": 300.0},
            "failures": {"value": 0},
        },
        "timing": {
            "boundary_id": "final_av_30s_extension_warm_after_one_excluded_warmup_v1",
            "warm_e2e": {"seconds": warm},
            "cold_e2e": {"seconds": cold},
            "seconds_per_generated_second": {"seconds": warm / 30.0},
        },
        "record_id": record_id,
    }


def _fake_sol_summary(candidate: bool, *, candidate_profile: str = "r9_adaptive_tau1_5_late_steps_diag", step_min: int = 4) -> dict:
    base = {
        "sparse_calls": 10,
        "sparse_candidate_calls": 10,
        "stride_aware_value_calls": 10,
        "dense_calls": 2,
        "density_sample_count": 10,
        "fallback_calls": 0,
        "fallback_reasons": {},
        "materialize_copy_count": 0,
        "materialize_copy_bytes": 0,
        "input_copy_events": 0,
        "input_copy_bytes": 0,
        "diagnostic_raw_tensor_exported": False,
        "thresh_type_values": ["diag"],
        "denoise_gpu_latency_ms": 1000.0,
        "sparse_attention_gpu_latency_ms": 100.0,
    }
    if candidate:
        inactive_by_step: dict[str, dict] = {}
        remaining_inactive = 6
        for idx, step in enumerate(range(step_min)):
            remaining_slots = step_min - idx
            count = remaining_inactive // remaining_slots
            if remaining_inactive % remaining_slots:
                count += 1
            remaining_inactive -= count
            inactive_by_step[str(step)] = {"active": 0, "inactive": count, "reasons": {"step_before_adaptive_min": count}}
        guard_counts_by_step = {**inactive_by_step, str(step_min): {"active": 4, "inactive": 0, "reasons": {"active": 4}}}
        base.update(
            {
                "tau_values": [1.0, 1.5],
                "tau_counts": {"1.0": 6, "1.5": 4},
                "adaptive_routing_values": [False, True],
                "adaptive_profiles": [candidate_profile],
                "adaptive_candidate_tau_values": [1.5],
                "adaptive_step_min_values": [step_min],
                "adaptive_guard_requested_count": 10,
                "adaptive_guard_active_count": 4,
                "adaptive_guard_inactive_count": 6,
                "adaptive_guard_reason_counts": {"active": 4, "step_before_adaptive_min": 6},
                "adaptive_guard_counts_by_step": guard_counts_by_step,
                "step_index_values": list(range(step_min + 1)),
            }
        )
    else:
        base.update({"tau_values": [1.0], "tau_counts": {"1.0": 10}, "adaptive_routing_values": [False]})
    return base


def _populate_pair(
    root: Path,
    pair: str,
    ref_warm: float,
    cand_warm: float,
    *,
    candidate_profile: str = "r9_adaptive_tau1_5_late_steps_diag",
    step_min: int = 4,
    reference_profile: str = "r9_current_sol_attn",
    reference_step_min: int | None = None,
) -> None:
    pair_dir = root / pair
    ref_dir = pair_dir / reference_profile
    cand_dir = pair_dir / candidate_profile
    _write_json(pair_dir / "decision.json", {"status": "pass", "failed_gates": [], "classification": "pair-pass"})
    _write_json(pair_dir / "reference_benchmark_record_validation.json", {"status": "pass", "failures": []})
    _write_json(pair_dir / "candidate_benchmark_record_validation.json", {"status": "pass", "failures": []})
    for lane_dir, record_id, warm, objective_value, candidate, lane_step_min in (
        (ref_dir, reference_profile, ref_warm, 1.0, reference_profile != "r9_current_sol_attn", reference_step_min or 3),
        (cand_dir, candidate_profile, cand_warm, 0.98, True, step_min),
    ):
        _write_json(lane_dir / "objective_metrics.json", _fake_objective(objective_value))
        _write_json(lane_dir / "benchmark_record.json", _fake_record(record_id, warm, warm + 100.0, objective_value))
        _write_json(
            lane_dir / "container_summary.json",
            {
                "final_av_accounting": {"objective_metrics_path": "/evidence/objective_metrics.json"},
                "sol_attn": {"status": "present", "summary": _fake_sol_summary(candidate, candidate_profile=record_id, step_min=lane_step_min)},
            },
        )
        (lane_dir / "sol_attn_telemetry.sol_attn.json").write_text("{}\n", encoding="utf-8")


def test_finalize_r9_guarded_adaptive_n3_accepts_three_valid_pairs(tmp_path: Path) -> None:
    _populate_pair(tmp_path, "pair01", 1000.0, 980.0)
    _populate_pair(tmp_path, "pair02", 1000.0, 970.0)
    _populate_pair(tmp_path, "pair03", 1000.0, 990.0)

    rc = finalize_r9_adaptive_matched_n3(
        Namespace(
            out_dir=str(tmp_path),
            candidate_profile="r9_adaptive_tau1_5_late_steps_diag",
            candidate_tau=1.5,
            pairs=3,
            min_median_delta_pct=1.0,
            max_slower_pair_pct=0.0,
        )
    )

    assert rc == 0
    decision = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    assert decision["status"] == "pass"
    assert decision["formal_validation_recommended_pending_reviewer"] is True
    assert decision["timing_summary"]["warm_e2e_delta_percent_n3_gate_not_speedup"]["median"] == 2.0
    assert decision["gates"]["all_candidate_guarded_adaptive_exercised"] is True
    assert (tmp_path / "reviewer_verdict_request.json").exists()


def test_finalize_r10_step3_guarded_adaptive_n1_uses_r10_labels(tmp_path: Path) -> None:
    _populate_pair(
        tmp_path,
        "pair01",
        1000.0,
        970.0,
        candidate_profile="r10_adaptive_tau1_5_step3_diag",
        step_min=3,
    )
    out_dir = tmp_path / "n1_final"

    rc = finalize_r9_adaptive_matched(
        Namespace(
            out_dir=str(out_dir),
            reference_evidence=str(tmp_path / "pair01" / "r9_current_sol_attn"),
            candidate_evidence=str(tmp_path / "pair01" / "r10_adaptive_tau1_5_step3_diag"),
            candidate_profile="r10_adaptive_tau1_5_step3_diag",
            candidate_tau=1.5,
            candidate_step_min=3,
        )
    )

    assert rc == 0
    decision = json.loads((out_dir / "decision.json").read_text(encoding="utf-8"))
    report = (out_dir / "RUN_REPORT.md").read_text(encoding="utf-8")
    assert decision["status"] == "pass"
    assert decision["schema_version"] == "minimax-h3-final-av-30s-r9-current-vs-r10-guarded-adaptive-step3-sol-attn-n1-decision-v1"
    assert decision["classification"] == "keep_default_off_r10_adaptive_tau1_5_step3_diag_long_lane_n1_pending_reviewer_quality_gate"
    assert "r10_adaptive_tau1_5_step3_diag" in decision["principal_variable"]
    assert "# r9 Current vs r10 Guarded Adaptive Step-Min=3" in report


def test_finalize_r10_step3_guarded_adaptive_n3_accepts_expected_step_min_and_uses_r10_labels(tmp_path: Path) -> None:
    for idx, warm in enumerate((980.0, 970.0, 990.0), start=1):
        _populate_pair(
            tmp_path,
            f"pair{idx:02d}",
            1000.0,
            warm,
            candidate_profile="r10_adaptive_tau1_5_step3_diag",
            step_min=3,
        )

    rc = finalize_r9_adaptive_matched_n3(
        Namespace(
            out_dir=str(tmp_path),
            candidate_profile="r10_adaptive_tau1_5_step3_diag",
            candidate_tau=1.5,
            candidate_step_min=3,
            pairs=3,
            min_median_delta_pct=1.0,
            max_slower_pair_pct=0.0,
        )
    )

    assert rc == 0
    decision = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    report = (tmp_path / "RUN_REPORT.md").read_text(encoding="utf-8")
    assert decision["status"] == "pass"
    assert decision["schema_version"] == "minimax-h3-final-av-30s-r9-current-vs-r10-guarded-adaptive-step3-sol-attn-n3-decision-v1"
    assert decision["classification"] == "recommend_followon_formal_validation_default_off_r10_adaptive_tau1_5_step3_diag_n3_pass"
    assert decision["candidate_expected_adaptive_step_min"] == 3
    assert decision["gates"]["all_candidate_adaptive_step_min_expected_seen"] is True
    assert decision["gates"]["all_candidate_guard_counts_match_step_min"] is True
    assert "# r9 Current vs r10 Guarded Adaptive Step-Min=3" in report


def test_finalize_r11_step2_against_retained_r10_n1_uses_reference_profile(tmp_path: Path) -> None:
    _populate_pair(
        tmp_path,
        "pair01",
        1000.0,
        985.0,
        reference_profile="r10_adaptive_tau1_5_step3_diag",
        reference_step_min=3,
        candidate_profile="r11_adaptive_tau1_5_step2_diag",
        step_min=2,
    )
    out_dir = tmp_path / "n1_r11_final"

    rc = finalize_r9_adaptive_matched(
        Namespace(
            out_dir=str(out_dir),
            reference_evidence=str(tmp_path / "pair01" / "r10_adaptive_tau1_5_step3_diag"),
            reference_profile="r10_adaptive_tau1_5_step3_diag",
            reference_tau=1.5,
            reference_step_min=3,
            candidate_evidence=str(tmp_path / "pair01" / "r11_adaptive_tau1_5_step2_diag"),
            candidate_profile="r11_adaptive_tau1_5_step2_diag",
            candidate_tau=1.5,
            candidate_step_min=2,
        )
    )

    assert rc == 0
    decision = json.loads((out_dir / "decision.json").read_text(encoding="utf-8"))
    report = (out_dir / "RUN_REPORT.md").read_text(encoding="utf-8")
    assert decision["status"] == "pass"
    assert decision["schema_version"] == "minimax-h3-final-av-30s-r10-guarded-adaptive-step3-vs-r11-guarded-adaptive-step2-sol-attn-n1-decision-v1"
    assert decision["reference_profile"] == "r10_adaptive_tau1_5_step3_diag"
    assert decision["candidate_expected_adaptive_step_min"] == 2
    assert decision["gates"]["reference_guard_counts_match_step_min"] is True
    assert decision["gates"]["candidate_guard_counts_match_step_min"] is True
    assert "# r10 Guarded Adaptive Step-Min=3 vs r11 Guarded Adaptive Step-Min=2" in report


def _fake_r12_step2_layer_summary() -> dict:
    return {
        "sparse_calls": 10,
        "sparse_candidate_calls": 10,
        "stride_aware_value_calls": 10,
        "dense_calls": 2,
        "density_sample_count": 10,
        "fallback_calls": 0,
        "fallback_reasons": {},
        "materialize_copy_count": 0,
        "materialize_copy_bytes": 0,
        "input_copy_events": 0,
        "input_copy_bytes": 0,
        "diagnostic_raw_tensor_exported": False,
        "thresh_type_values": ["diag"],
        "denoise_gpu_latency_ms": 1000.0,
        "sparse_attention_gpu_latency_ms": 100.0,
        "tau_values": [1.0, 1.5],
        "tau_counts": {"1.0": 6, "1.5": 4},
        "adaptive_routing_values": [False, True],
        "adaptive_profiles": ["r12_adaptive_tau1_5_step2_layers34_49_diag"],
        "adaptive_candidate_tau_values": [1.5],
        "adaptive_step_min_values": [2],
        "adaptive_layer_min_values": [4],
        "adaptive_layer_max_values": [5],
        "adaptive_layer_range_scope_values": ["step_min_only"],
        "adaptive_guard_requested_count": 10,
        "adaptive_guard_active_count": 4,
        "adaptive_guard_inactive_count": 6,
        "adaptive_guard_reason_counts": {"active": 4, "step_before_adaptive_min": 4, "layer_before_adaptive_min": 2},
        "adaptive_guard_counts_by_step": {
            "0": {"active": 0, "inactive": 2, "reasons": {"step_before_adaptive_min": 2}},
            "1": {"active": 0, "inactive": 2, "reasons": {"step_before_adaptive_min": 2}},
            "2": {"active": 2, "inactive": 2, "reasons": {"active": 2, "layer_before_adaptive_min": 2}},
            "3": {"active": 2, "inactive": 0, "reasons": {"active": 2}},
        },
        "step_index_values": [0, 1, 2, 3],
        "layer_index_values": [2, 3, 4, 5],
    }


def test_finalize_r12_step2_layer_subset_against_retained_r10_n1_checks_layer_scope(tmp_path: Path) -> None:
    _populate_pair(
        tmp_path,
        "pair01",
        1000.0,
        996.0,
        reference_profile="r10_adaptive_tau1_5_step3_diag",
        reference_step_min=3,
        candidate_profile="r12_adaptive_tau1_5_step2_layers34_49_diag",
        step_min=2,
    )
    cand_summary_path = tmp_path / "pair01" / "r12_adaptive_tau1_5_step2_layers34_49_diag" / "container_summary.json"
    cand_summary = json.loads(cand_summary_path.read_text(encoding="utf-8"))
    cand_summary["sol_attn"]["summary"] = _fake_r12_step2_layer_summary()
    cand_summary_path.write_text(json.dumps(cand_summary), encoding="utf-8")
    out_dir = tmp_path / "n1_r12_final"

    rc = finalize_r9_adaptive_matched(
        Namespace(
            out_dir=str(out_dir),
            reference_evidence=str(tmp_path / "pair01" / "r10_adaptive_tau1_5_step3_diag"),
            reference_profile="r10_adaptive_tau1_5_step3_diag",
            reference_tau=1.5,
            reference_step_min=3,
            candidate_evidence=str(tmp_path / "pair01" / "r12_adaptive_tau1_5_step2_layers34_49_diag"),
            candidate_profile="r12_adaptive_tau1_5_step2_layers34_49_diag",
            candidate_tau=1.5,
            candidate_step_min=2,
            candidate_layer_min=4,
            candidate_layer_max=5,
            candidate_layer_range_scope="step_min_only",
        )
    )

    assert rc == 0
    decision = json.loads((out_dir / "decision.json").read_text(encoding="utf-8"))
    assert decision["status"] == "pass"
    assert decision["schema_version"] == "minimax-h3-final-av-30s-r10-guarded-adaptive-step3-vs-r12-guarded-adaptive-step2-layers34-49-sol-attn-n1-decision-v1"
    assert decision["candidate_expected_adaptive_layer_min"] == 4
    assert decision["candidate_expected_adaptive_layer_max"] == 5
    assert decision["candidate_expected_adaptive_layer_range_scope"] == "step_min_only"
    assert decision["gates"]["candidate_adaptive_layer_range_expected_seen"] is True
    assert decision["gates"]["candidate_guard_counts_match_step_min"] is True


def test_finalize_r10_cache_dit_high_n1_accepts_cache_telemetry_and_fixed_r10_sol_attn(tmp_path: Path) -> None:
    ref_dir = tmp_path / "r10_adaptive_tau1_5_step3_diag_cache_off_lossless"
    cand_dir = tmp_path / "r10_adaptive_tau1_5_step3_diag_cache_dit_high"
    for lane_dir, quality, warm, objective_value, cache_summary in (
        (
            ref_dir,
            "lossless",
            1000.0,
            1.0,
            {
                "status": "present",
                "summary": {
                    "request_quality": "lossless",
                    "server_cache_backend": "cache_dit",
                    "refresh_log_count": 1,
                    "parsed_reuse_or_skip_count": 0,
                    "parsed_compute_count": 8,
                },
            },
        ),
        (
            cand_dir,
            "high",
            970.0,
            0.98,
            {
                "status": "present",
                "summary": {
                    "request_quality": "high",
                    "server_cache_backend": "cache_dit",
                    "refresh_log_count": 7,
                    "parsed_reuse_or_skip_count": 2,
                    "parsed_compute_count": 6,
                    "high_request_seen": True,
                },
            },
        ),
    ):
        _write_json(lane_dir / "objective_metrics.json", _fake_objective(objective_value))
        _write_json(lane_dir / "benchmark_record.json", _fake_record(lane_dir.name, warm, warm + 100.0, objective_value))
        _write_json(
            lane_dir / "container_summary.json",
            {
                "request_quality": quality,
                "final_av_accounting": {"objective_metrics_path": "/evidence/objective_metrics.json"},
                "cache_dit": cache_summary,
                "sol_attn": {
                    "status": "present",
                    "summary": _fake_sol_summary(
                        True,
                        candidate_profile="r10_adaptive_tau1_5_step3_diag",
                        step_min=3,
                    ),
                },
            },
        )

    out_dir = tmp_path / "cache_final"
    rc = finalize_r10_cache_dit_matched(
        Namespace(
            out_dir=str(out_dir),
            reference_evidence=str(ref_dir),
            candidate_evidence=str(cand_dir),
            min_delta_pct=1.0,
        )
    )

    assert rc == 0
    decision = json.loads((out_dir / "decision.json").read_text(encoding="utf-8"))
    report = (out_dir / "RUN_REPORT.md").read_text(encoding="utf-8")
    assert decision["status"] == "pass"
    assert decision["classification"] == "keep_r10_cache_dit_high_n1_route_gate_pass_pending_independent_review"
    assert decision["gates"]["candidate_cache_reuse_or_skip_positive"] is True
    assert decision["gates"]["r10_step_min_fixed_when_reported"] is True
    assert "quality=high" in decision["principal_variable"] or "quality=high" in report
    assert (out_dir / "cache_dit_telemetry_summary.json").exists()


def test_finalize_r10_cache_dit_warmup2_n1_accepts_only_expected_request_quality(tmp_path: Path) -> None:
    ref_dir = tmp_path / "r10_adaptive_tau1_5_step3_diag_cache_off_lossless"
    cand_dir = tmp_path / "r10_adaptive_tau1_5_step3_diag_cache_dit_high_warmup2"
    for lane_dir, quality, warm, objective_value, parsed_reuse, parsed_compute in (
        (ref_dir, "lossless", 1000.0, 1.0, 0, 8),
        (cand_dir, "high_warmup2", 960.0, 0.98, 3, 5),
    ):
        _write_json(lane_dir / "objective_metrics.json", _fake_objective(objective_value))
        _write_json(lane_dir / "benchmark_record.json", _fake_record(lane_dir.name, warm, warm + 100.0, objective_value))
        _write_json(
            lane_dir / "container_summary.json",
            {
                "request_quality": quality,
                "final_av_accounting": {"objective_metrics_path": "/evidence/objective_metrics.json"},
                "cache_dit": {
                    "status": "present",
                    "summary": {
                        "request_quality": quality,
                        "server_cache_backend": "cache_dit",
                        "refresh_log_count": 7,
                        "parsed_reuse_or_skip_count": parsed_reuse,
                        "parsed_compute_count": parsed_compute,
                    },
                },
                "sol_attn": {
                    "status": "present",
                    "summary": _fake_sol_summary(True, candidate_profile="r10_adaptive_tau1_5_step3_diag", step_min=3),
                },
            },
        )

    out_dir = tmp_path / "cache_warmup2_final"
    rc = finalize_r10_cache_dit_matched(
        Namespace(
            out_dir=str(out_dir),
            reference_evidence=str(ref_dir),
            candidate_evidence=str(cand_dir),
            candidate_quality="high_warmup2",
            min_delta_pct=1.0,
        )
    )

    assert rc == 0
    decision = json.loads((out_dir / "decision.json").read_text(encoding="utf-8"))
    assert decision["status"] == "pass"
    assert decision["classification"] == "keep_r10_cache_dit_high_warmup2_n1_route_gate_pass_pending_independent_review"
    assert decision["candidate_profile"] == "r10_adaptive_tau1_5_step3_diag_cache_dit_high_warmup2"
    assert decision["cache_telemetry_summary"]["candidate_expected_profile"]["max_warmup_steps"] == 2
    assert decision["gates"]["candidate_request_quality_expected"] is True


def test_finalize_r11_step2_against_retained_r10_n3_accepts_reference_profile(tmp_path: Path) -> None:
    for idx, warm in enumerate((985.0, 982.0, 986.0), start=1):
        _populate_pair(
            tmp_path,
            f"pair{idx:02d}",
            1000.0,
            warm,
            reference_profile="r10_adaptive_tau1_5_step3_diag",
            reference_step_min=3,
            candidate_profile="r11_adaptive_tau1_5_step2_diag",
            step_min=2,
        )

    rc = finalize_r9_adaptive_matched_n3(
        Namespace(
            out_dir=str(tmp_path),
            reference_profile="r10_adaptive_tau1_5_step3_diag",
            reference_tau=1.5,
            reference_step_min=3,
            candidate_profile="r11_adaptive_tau1_5_step2_diag",
            candidate_tau=1.5,
            candidate_step_min=2,
            pairs=3,
            min_median_delta_pct=1.0,
            max_slower_pair_pct=0.0,
        )
    )

    assert rc == 0
    decision = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    report = (tmp_path / "RUN_REPORT.md").read_text(encoding="utf-8")
    assert decision["status"] == "pass"
    assert decision["schema_version"] == "minimax-h3-final-av-30s-r10-guarded-adaptive-step3-vs-r11-guarded-adaptive-step2-sol-attn-n3-decision-v1"
    assert decision["reference_profile"] == "r10_adaptive_tau1_5_step3_diag"
    assert decision["candidate_profile"] == "r11_adaptive_tau1_5_step2_diag"
    assert decision["formal_validation_recommended_pending_reviewer"] is True
    assert "# r10 Guarded Adaptive Step-Min=3 vs r11 Guarded Adaptive Step-Min=2" in report


def test_finalize_r9_guarded_adaptive_formal_accepts_ten_valid_pairs(tmp_path: Path) -> None:
    for idx in range(1, 11):
        _populate_pair(tmp_path, f"pair{idx:02d}", 1000.0, 970.0)

    rc = finalize_r9_adaptive_matched_formal(
        Namespace(
            out_dir=str(tmp_path),
            candidate_profile="r9_adaptive_tau1_5_late_steps_diag",
            candidate_tau=1.5,
            pairs=10,
            min_required_pairs=10,
            min_median_delta_pct=1.0,
            max_slower_pair_pct=0.0,
        )
    )

    assert rc == 0
    decision = json.loads((tmp_path / "formal_n10_decision.json").read_text(encoding="utf-8"))
    assert decision["status"] == "pass"
    assert decision["completed_pairs"] == 10
    assert decision["formal_speedup_claimed"] is False
    assert decision["accepted_formal_candidate_pending_reviewer"] is True
    assert decision["timing_summary"]["warm_e2e_delta_percent_formal_candidate_pending_reviewer"]["median"] == 3.0
    assert decision["timing_summary"]["component_timing_seconds"]["reference"]["denoise"]["seconds"]["status"] == "missing"
    assert (tmp_path / "component_timing_summary.json").exists()
    assert (tmp_path / "final_av_accounting_summary.json").exists()
    assert (tmp_path / "reviewer_verdict_request.json").exists()


def test_finalize_r10_step3_guarded_adaptive_formal_uses_r10_labels(tmp_path: Path) -> None:
    for idx in range(1, 11):
        _populate_pair(
            tmp_path,
            f"pair{idx:02d}",
            1000.0,
            970.0,
            candidate_profile="r10_adaptive_tau1_5_step3_diag",
            step_min=3,
        )

    rc = finalize_r9_adaptive_matched_formal(
        Namespace(
            out_dir=str(tmp_path),
            candidate_profile="r10_adaptive_tau1_5_step3_diag",
            candidate_tau=1.5,
            candidate_step_min=3,
            pairs=10,
            min_required_pairs=10,
            min_median_delta_pct=1.0,
            max_slower_pair_pct=0.0,
        )
    )

    assert rc == 0
    decision = json.loads((tmp_path / "formal_n10_decision.json").read_text(encoding="utf-8"))
    report = (tmp_path / "FORMAL_N10_RUN_REPORT.md").read_text(encoding="utf-8")
    request = json.loads((tmp_path / "reviewer_verdict_request.json").read_text(encoding="utf-8"))
    assert decision["status"] == "pass"
    assert decision["schema_version"] == "minimax-h3-final-av-30s-r9-current-vs-r10-guarded-adaptive-step3-sol-attn-formal-n10-decision-v1"
    assert decision["classification"] == "accepted_formal_n10_r10_adaptive_tau1_5_step3_diag_candidate_pending_independent_reviewer"
    assert decision["candidate_profile"] == "r10_adaptive_tau1_5_step3_diag"
    assert decision["candidate_expected_adaptive_step_min"] == 3
    assert "r10_adaptive_tau1_5_step3_diag" in decision["principal_variable"]
    assert "# r9 Current vs r10 Guarded Adaptive Step-Min=3" in report
    assert "r9-current-vs-r10-guarded-adaptive-step3" in request["schema_version"]
