#!/usr/bin/env python3
"""MiniMax-H3 video VAE bounded spatial tile-batch characterization probe."""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import time
import traceback
from typing import Any

import torch

LATENT_SHAPE = (1, 24, 37, 48, 84)
HEIGHT = 768
WIDTH = 1344
WARM_E2E_SECONDS = 1373.34316037799
ONE_PERCENT_SECONDS = WARM_E2E_SECONDS * 0.01
PER_CHUNK_SIGNAL_SECONDS = ONE_PERCENT_SECONDS / 6.0
SPATIAL_ENV = "MINIMAX_H3_A6000_VIDEO_VAE_SPATIAL_TILE_BATCHING"
CAP_ENV = "MINIMAX_H3_A6000_VIDEO_VAE_TILE_BATCH_SIZE"
DEFAULT_CAPS = (4, 7, 14)
PRIOR_ALL_STACK = {
    "source": "technical_report/evidence/minimax_h3_desktop/long_video/r15_vae_spatial_tile_batching_probe_20260817T033336Z/vae_spatial_tile_batching_probe.json",
    "cap_equivalent": "unbounded_all_stack_per_temporal_clip_28_tiles",
    "device_seconds": 27.060970703125,
    "peak_memory_allocated_bytes": 18907423744,
    "max_abs_delta": 0.005423754453659058,
    "mean_abs_delta": 9.858288103714585e-05,
    "rmse_delta": 0.00014504284815114847,
    "psnr_db_assuming_unit_range": 76.77007361507077,
    "seam_to_interior_mean_ratio": 1.0752652540318102,
    "single_chunk_saving_seconds": 5.855095703125002,
    "projected_six_chunk_saving_seconds": 35.13057421875001,
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def tensor_summary(t: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "numel": int(t.numel()),
        "bytes": int(t.numel() * t.element_size()),
        "stride": list(t.stride()),
        "is_contiguous": bool(t.is_contiguous()),
    }


def load_vae(model_path: str):
    from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

    device = torch.device("cuda:0")
    vae = MiniMaxH3VideoVAE(model_path, device=device)
    vae.eval()
    vae.load_to_device()
    return vae


def configure_cap(vae, cap: int) -> int:
    from vllm_omni.diffusion.models.minimax_h3 import vae as vae_module

    os.environ[SPATIAL_ENV] = "0"
    os.environ[CAP_ENV] = str(cap)
    return int(vae_module._configure_video_vae_bounded_tile_batching(vae.model))


def decode_timed(vae, latent: torch.Tensor, label: str) -> tuple[torch.Tensor, dict[str, Any]]:
    torch.cuda.reset_peak_memory_stats()
    sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall0 = time.perf_counter()
    start.record()
    with torch.autocast("cuda", dtype=torch.float16, enabled=True):
        out = vae.decode_latent(latent)
    out = out[..., :HEIGHT, :WIDTH].contiguous()
    end.record()
    sync()
    return out, {
        "label": label,
        "wall_seconds": time.perf_counter() - wall0,
        "device_seconds": float(start.elapsed_time(end)) / 1000.0,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "output": tensor_summary(out),
        "stack_tiling": bool(getattr(vae.model, "stack_tiling", False)),
        "tile_batch_size": int(getattr(vae.model, "_a6000_video_vae_tile_batch_size", 0) or 0),
    }


def split_tiles(input_len: int, tile_size: int, overlap_min: int, vae_ratio: int) -> tuple[list[int], list[int], list[int]]:
    if tile_size >= input_len:
        return [0], [input_len], []
    n = math.ceil(input_len / tile_size)
    while True:
        overlaps = [overlap_min] * (n - 1)
        remaining = tile_size * n - sum(overlaps) - input_len
        if remaining < 0:
            n += 1
        else:
            break
    remaining_units = remaining // vae_ratio
    for i in range(remaining_units):
        overlaps[i % (n - 1)] += vae_ratio
    starts = [0]
    for i in range(n - 1):
        starts.append(starts[-1] + tile_size - overlaps[i])
    return starts, [tile_size] * n, overlaps


def seam_delta_summary(diff_video: torch.Tensor, *, tile_size: int, overlap_min: int, vae_ratio: int) -> dict[str, Any]:
    diff = diff_video.float().mean(dim=(0, 1, 2))
    y_idx, _y_len, _y_overlap = split_tiles(HEIGHT, tile_size, overlap_min, vae_ratio)
    x_idx, _x_len, _x_overlap = split_tiles(WIDTH, tile_size, overlap_min, vae_ratio)
    y_seams = [y for y in y_idx[1:]]
    x_seams = [x for x in x_idx[1:]]
    mask = torch.zeros((HEIGHT, WIDTH), device=diff.device, dtype=torch.bool)
    band = 4
    for y in y_seams:
        mask[max(0, y - band): min(HEIGHT, y + band + 1), :] = True
    for x in x_seams:
        mask[:, max(0, x - band): min(WIDTH, x + band + 1)] = True
    seam = diff[mask]
    interior = diff[~mask]
    seam_mean = float(seam.mean().item()) if seam.numel() else None
    interior_mean = float(interior.mean().item()) if interior.numel() else None
    ratio = seam_mean / max(interior_mean or 0.0, 1e-12) if seam_mean is not None and interior_mean is not None else None
    return {
        "tile_seams_y": y_seams,
        "tile_seams_x": x_seams,
        "band_pixels": band,
        "seam_mean_abs_delta": seam_mean,
        "interior_mean_abs_delta": interior_mean,
        "seam_to_interior_mean_ratio": ratio,
    }


def compare_to_prior_all_stack(metrics: dict[str, Any], rec: dict[str, Any]) -> dict[str, Any]:
    seam_ratio = metrics["seam_delta"].get("seam_to_interior_mean_ratio")
    improvements = {
        "max_abs_lower": metrics["max_abs_delta"] < PRIOR_ALL_STACK["max_abs_delta"],
        "mean_abs_lower": metrics["mean_abs_delta"] < PRIOR_ALL_STACK["mean_abs_delta"],
        "rmse_lower": metrics["rmse_delta"] < PRIOR_ALL_STACK["rmse_delta"],
        "psnr_higher": metrics["psnr_db_assuming_unit_range"] > PRIOR_ALL_STACK["psnr_db_assuming_unit_range"],
        "seam_ratio_not_higher": seam_ratio is not None and seam_ratio <= PRIOR_ALL_STACK["seam_to_interior_mean_ratio"],
        "peak_memory_lower": int(rec["peak_memory_allocated_bytes"]) < int(PRIOR_ALL_STACK["peak_memory_allocated_bytes"]),
    }
    material = (
        metrics["rmse_delta"] <= 0.9 * PRIOR_ALL_STACK["rmse_delta"]
        or metrics["mean_abs_delta"] <= 0.9 * PRIOR_ALL_STACK["mean_abs_delta"]
        or metrics["max_abs_delta"] <= 0.9 * PRIOR_ALL_STACK["max_abs_delta"]
        or metrics["psnr_db_assuming_unit_range"] >= PRIOR_ALL_STACK["psnr_db_assuming_unit_range"] + 1.0
        or int(rec["peak_memory_allocated_bytes"]) <= int(PRIOR_ALL_STACK["peak_memory_allocated_bytes"]) - (1 << 30)
    )
    return {
        "prior_all_stack_source": PRIOR_ALL_STACK["source"],
        "prior_all_stack_cap_equivalent": PRIOR_ALL_STACK["cap_equivalent"],
        "improvements": improvements,
        "materially_reduces_approximation_or_memory_risk": bool(material),
        "risk_score_lower_is_better": (
            metrics["rmse_delta"] / PRIOR_ALL_STACK["rmse_delta"]
            + metrics["mean_abs_delta"] / PRIOR_ALL_STACK["mean_abs_delta"]
            + metrics["max_abs_delta"] / PRIOR_ALL_STACK["max_abs_delta"]
            + (seam_ratio or 99.0) / PRIOR_ALL_STACK["seam_to_interior_mean_ratio"]
        ),
    }


def cap_metrics(baseline: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    diff = (candidate - baseline).abs()
    mse = float(diff.float().pow(2).mean().item())
    rmse = math.sqrt(mse)
    psnr = 20.0 * math.log10(1.0 / max(rmse, 1e-12))
    return {
        "max_abs_delta": float(diff.max().item()),
        "mean_abs_delta": float(diff.float().mean().item()),
        "rmse_delta": rmse,
        "psnr_db_assuming_unit_range": psnr,
        "bit_exact": bool(torch.equal(candidate, baseline)),
        "seam_delta": None,
    }


def parse_caps(raw: str) -> list[int]:
    caps = []
    for part in raw.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        value = int(stripped)
        if value <= 0:
            raise ValueError(f"cap must be positive, got {stripped!r}")
        if value not in caps:
            caps.append(value)
    if not caps:
        raise ValueError("at least one positive cap is required")
    return caps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=pathlib.Path)
    parser.add_argument("--model-path", default="/models/MiniMax-H3/FL2VA/video_vae")
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--caps", default=",".join(str(x) for x in DEFAULT_CAPS))
    args = parser.parse_args()
    evidence = args.evidence
    profile_path = evidence / "vae_bounded_tile_batching_probe.json"
    try:
        caps = parse_caps(args.caps)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(f"expected exactly one visible CUDA device, got cuda_available={torch.cuda.is_available()} count={torch.cuda.device_count()}")
        props = torch.cuda.get_device_properties(0)
        if "a6000" not in props.name.lower() or torch.cuda.get_device_capability(0) != (8, 6):
            raise RuntimeError(f"expected one RTX A6000/SM86, got name={props.name!r} capability={torch.cuda.get_device_capability(0)!r}")
        os.environ[SPATIAL_ENV] = "0"
        os.environ[CAP_ENV] = "0"
        torch.manual_seed(args.seed)
        vae = load_vae(args.model_path)
        latent = torch.randn(LATENT_SHAPE, device="cuda", dtype=torch.float32)
        baseline, baseline_rec = decode_timed(vae, latent, "baseline_current_serial_spatial_tiles_cap0")

        fail_closed_checks: dict[str, Any] = {}
        from vllm_omni.diffusion.models.minimax_h3 import vae as vae_module
        os.environ[CAP_ENV] = "not-an-int"
        try:
            vae_module._configure_video_vae_bounded_tile_batching(vae.model)
            fail_closed_checks["invalid_cap_rejected"] = False
        except RuntimeError as exc:
            fail_closed_checks["invalid_cap_rejected"] = True
            fail_closed_checks["invalid_cap_error"] = str(exc)
        finally:
            configure_cap(vae, 0)
        probe_model = torch.nn.Module()
        probe_model.stack_tiling = False
        os.environ[CAP_ENV] = "2"
        try:
            vae_module._configure_video_vae_bounded_tile_batching(probe_model)
            fail_closed_checks["missing_run_tile_tasks_rejected"] = False
        except RuntimeError as exc:
            fail_closed_checks["missing_run_tile_tasks_rejected"] = True
            fail_closed_checks["missing_run_tile_tasks_error"] = str(exc)
        finally:
            configure_cap(vae, 0)

        tile_size = int(getattr(vae.model, "decoder_tile_size", 256))
        overlap_min = int(getattr(vae.model, "decoder_tile_overlap_min", 64))
        vae_ratio = int(getattr(vae.model, "vae_ratio", 16))
        thresholds = {
            "per_chunk_signal_seconds_min": PER_CHUNK_SIGNAL_SECONDS,
            "six_chunk_signal_seconds_min": ONE_PERCENT_SECONDS,
            "max_abs_delta_max": 0.01,
            "mean_abs_delta_max": 5e-4,
            "psnr_db_min": 60.0,
            "seam_to_interior_mean_ratio_max": 4.0,
            "candidate_peak_memory_allocated_bytes_max": 44 * (2**30),
        }
        cap_results = []
        for cap in caps:
            configure_cap(vae, cap)
            candidate, rec = decode_timed(vae, latent, f"candidate_bounded_tile_batch_cap_{cap}")
            metrics = cap_metrics(baseline, candidate)
            diff = (candidate - baseline).abs()
            metrics["seam_delta"] = seam_delta_summary(diff, tile_size=tile_size, overlap_min=overlap_min, vae_ratio=vae_ratio)
            saving = float(baseline_rec["device_seconds"]) - float(rec["device_seconds"])
            metrics["single_chunk_saving_seconds"] = saving
            metrics["projected_six_chunk_saving_seconds"] = saving * 6.0
            prior_cmp = compare_to_prior_all_stack(metrics, rec)
            gates = {
                "cap_applied": rec.get("tile_batch_size") == cap and rec.get("stack_tiling") is True,
                "timing_signal_ge_one_percent_warm_e2e": saving * 6.0 >= ONE_PERCENT_SECONDS,
                "max_abs_delta_within_practical_screen": metrics["max_abs_delta"] <= thresholds["max_abs_delta_max"],
                "mean_abs_delta_within_practical_screen": metrics["mean_abs_delta"] <= thresholds["mean_abs_delta_max"],
                "psnr_within_practical_screen": metrics["psnr_db_assuming_unit_range"] >= thresholds["psnr_db_min"],
                "seam_delta_not_concentrated": metrics["seam_delta"].get("seam_to_interior_mean_ratio") is not None and metrics["seam_delta"]["seam_to_interior_mean_ratio"] <= thresholds["seam_to_interior_mean_ratio_max"],
                "candidate_memory_within_a6000_screen": int(rec["peak_memory_allocated_bytes"]) <= thresholds["candidate_peak_memory_allocated_bytes_max"],
                "risk_reduced_vs_prior_all_stack": prior_cmp["materially_reduces_approximation_or_memory_risk"],
            }
            failed = [name for name, ok in gates.items() if not ok]
            cap_results.append({
                "cap": cap,
                "record": rec,
                "metrics": metrics,
                "prior_all_stack_comparison": prior_cmp,
                "gates": gates,
                "failed_gates": failed,
                "status": "pass" if not failed else "reject",
            })
            del candidate, diff
            torch.cuda.empty_cache()
        configure_cap(vae, 0)

        viable = [item for item in cap_results if item["status"] == "pass"]
        selected = None
        if viable:
            selected = sorted(
                viable,
                key=lambda item: (
                    item["prior_all_stack_comparison"]["risk_score_lower_is_better"],
                    -item["metrics"]["projected_six_chunk_saving_seconds"],
                    item["cap"],
                ),
            )[0]
        status = "pass" if selected is not None else "reject"
        classification = "select_single_bounded_cap_pending_final_av_n1" if selected else "no_viable_bounded_cap_no_final_av_n1"
        payload = {
            "schema_version": "minimax_h3_a6000_video_vae_bounded_tile_batching_probe_v1",
            "created_utc": utc_now(),
            "status": status,
            "classification": classification,
            "scope": "VAE-only same-input representative latent characterization across bounded spatial tile-batch caps; no denoise/text/audio/final-AV claim.",
            "model_path": args.model_path,
            "caps": caps,
            "selected_cap": selected["cap"] if selected else None,
            "selected_result": selected,
            "baseline": baseline_rec,
            "latent": tensor_summary(latent),
            "gpu": {"name": props.name, "capability": list(torch.cuda.get_device_capability(0)), "total_memory_bytes": int(props.total_memory)},
            "env_toggles": {"all_stack_spatial_tile_batching": SPATIAL_ENV, "bounded_tile_batch_size": CAP_ENV},
            "prior_all_stack_reference": PRIOR_ALL_STACK,
            "thresholds": thresholds,
            "fail_closed_checks": fail_closed_checks,
            "cap_results": cap_results,
            "source_revision_license_grounding": {
                "minimax_h3_model": {"revision": "6818f6c32d12b210915e44ad56a4228c2608f160", "license": "MiniMax H3 Community License Agreement", "local_video_vae_bundle": "models/MiniMax-H3/FL2VA/video_vae"},
                "vllm_omni_runtime": {"revision": "8e2e9b6b53e86e6a479ed2c0a53782f655f60e04", "license": "Apache-2.0", "patched_file": "vllm_omni/diffusion/models/minimax_h3/vae.py"},
                "sana_sol_engine": {"revision": "d00eef311670a58deb2c323fe072738fcb945600", "license": "Apache-2.0 per README; bundled FlashAttention-derived notice BSD-3-Clause", "relevant_files": ["upstreams/Sana-sol-engine/models/minimax_h3/gb200/optimized/vae_shard.py", "upstreams/Sana-sol-engine/models/minimax_h3/rtx5090/pipeline.py"]},
                "benchmark_contract": "benchmark_contract/v1/contract.json; final-av-30s extension lane; n1 requires no automatic red flags before promotion",
            },
            "claim_boundary": "Practical-approximate VAE-only bounded-cap screen. Not exact/lossless, not BF16 fidelity, not final-AV, not native long context, not a speedup claim, and not human/product quality.",
            "next_decision": "build_overlay_and_run_matched_retained_r10_final_av_n1" if selected else "do_not_build_overlay_or_run_final_av_n1_from_this_candidate",
        }
        write_json(profile_path, payload)
        write_json(evidence / "decision.json", payload)
        lines = [
            "# VAE bounded tile-batch cap characterization",
            "",
            f"- Status: `{status}`",
            f"- Classification: `{classification}`",
            f"- Caps tested: `{caps}`",
            f"- Selected cap: `{selected['cap'] if selected else None}`",
            f"- Baseline device seconds: {baseline_rec['device_seconds']:.3f}",
            f"- Failed closed checks: `{fail_closed_checks}`",
            f"- Claim boundary: {payload['claim_boundary']}",
        ]
        for item in cap_results:
            lines.append(
                f"- cap={item['cap']}: status={item['status']}, device={item['record']['device_seconds']:.3f}s, "
                f"six_chunk_saving={item['metrics']['projected_six_chunk_saving_seconds']:.3f}s, "
                f"max_abs={item['metrics']['max_abs_delta']:.6g}, mean_abs={item['metrics']['mean_abs_delta']:.6g}, "
                f"psnr={item['metrics']['psnr_db_assuming_unit_range']:.2f}dB, "
                f"seam_ratio={item['metrics']['seam_delta'].get('seam_to_interior_mean_ratio')}, "
                f"failed={item['failed_gates']}"
            )
        (evidence / "RUN_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 0
    except Exception as exc:  # noqa: BLE001
        payload = {
            "schema_version": "minimax_h3_a6000_video_vae_bounded_tile_batching_probe_blocker_v1",
            "created_utc": utc_now(),
            "status": "failed",
            "classification": "bounded_cap_characterization_runtime_blocker_no_final_av_n1",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback_tail": traceback.format_exc()[-8000:],
            "claim_boundary": "No qualifying VAE bounded-cap characterization or final-AV result was produced.",
        }
        write_json(profile_path, payload)
        write_json(evidence / "decision.json", payload)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
