#!/usr/bin/env python3
"""Semantic validation for the MiniMax-H3 A6000 benchmark contract v1.

The JSON Schema checks record shape. This module enforces cross-field claim
boundaries that JSON Schema cannot express clearly: timing hierarchy, exact AV
accounting, long-production labels, and legal speedup denominators.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

CONTRACT_SCHEMA_VERSION = "minimax-h3-a6000-benchmark-contract-v1"
MANIFEST_SCHEMA_VERSION = "minimax-h3-a6000-lane-manifest-v1"
RECORD_SCHEMA_VERSION = "minimax-h3-a6000-benchmark-record-v1"
CONTRACT_VERSION = "1.0.0"
LONG_MODES = {
    "native_long_context",
    "chunked_overlap",
    "extension",
    "montage_stitching",
}
TRACKS = {"fidelity_bf16_exact", "practical_disclosed_approx"}
DEPLOYMENT_SCOPES = {"single_a6000", "multi_gpu_production"}
REQUIRED_COMPONENTS = (
    "text_conditioning",
    "denoise",
    "attention",
    "video_vae",
    "audio_vae",
    "encoding_mux",
    "io",
)
ADDITIVE_COMPONENTS = (
    "text_conditioning",
    "denoise",
    "video_vae",
    "audio_vae",
    "encoding_mux",
    "io",
)
REQUIRED_QUALITY_METRICS = (
    "subject_identity_consistency",
    "background_consistency",
    "camera_consistency",
    "motion",
    "repetition",
    "freezing",
    "visual_seams",
    "loudness",
    "silence",
    "audio_continuity",
    "av_sync_proxy",
)
METRIC_STATUSES = {
    "measured",
    "pass",
    "fail",
    "not_available_historical_evidence",
    "not_applicable_single_clip",
    "not_applicable_no_detectable_events",
    "not_run_dry_run",
}
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

LANE_SPECS: dict[str, dict[str, Any]] = {
    "native-short-1344x768-124f-24fps-v1": {
        "is_long": False,
        "width": 1344,
        "height": 768,
        "fps": 24,
        "frames": 124,
        "duration": 124 / 24,
        "effective_audio_samples": None,
        "chunk_count": 1,
    },
    "final-av-30s-1344x768-24fps-v1": {
        "is_long": True,
        "width": 1344,
        "height": 768,
        "fps": 24,
        "frames": 720,
        "duration": 30.0,
        "effective_audio_samples": 960_000,
        "chunk_count": 6,
    },
    "final-av-60s-1344x768-24fps-v1": {
        "is_long": True,
        "width": 1344,
        "height": 768,
        "fps": 24,
        "frames": 1440,
        "duration": 60.0,
        "effective_audio_samples": 1_920_000,
        "chunk_count": 12,
    },
}


class ContractValidationError(Exception):
    """Raised when an artifact violates the benchmark contract."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractValidationError([f"artifact not found: {path}"]) from exc
    except json.JSONDecodeError as exc:
        raise ContractValidationError([f"invalid JSON in {path}: {exc}"]) from exc
    if not isinstance(value, dict):
        raise ContractValidationError([f"{path} must contain a JSON object"])
    return value


def _mapping(parent: Mapping[str, Any], key: str, errors: list[str], path: str = "") -> dict[str, Any]:
    value = parent.get(key)
    name = f"{path}.{key}" if path else key
    if not isinstance(value, dict):
        errors.append(f"{name} must be an object")
        return {}
    return value


def _list(parent: Mapping[str, Any], key: str, errors: list[str], path: str = "") -> list[Any]:
    value = parent.get(key)
    name = f"{path}.{key}" if path else key
    if not isinstance(value, list):
        errors.append(f"{name} must be an array")
        return []
    return value


def _close(actual: Any, expected: float, *, abs_tol: float = 1e-6) -> bool:
    return isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=abs_tol
    )


def _metric(parent: Mapping[str, Any], key: str, errors: list[str], path: str) -> dict[str, Any]:
    metric = _mapping(parent, key, errors, path)
    status = metric.get("status")
    if status not in METRIC_STATUSES:
        errors.append(f"{path}.{key}.status must be a recognized explicit metric status")
    value = metric.get("value")
    if status in {"measured", "pass", "fail"} and value is None:
        errors.append(f"{path}.{key}.value is required when status={status}")
    if status not in {"measured", "pass", "fail"} and "reason" not in metric:
        errors.append(f"{path}.{key}.reason is required when the metric is not measured")
    return metric


def _timing_measure(
    parent: Mapping[str, Any],
    key: str,
    errors: list[str],
    path: str = "timing",
) -> dict[str, Any]:
    metric = _mapping(parent, key, errors, path)
    status = metric.get("status")
    if status not in METRIC_STATUSES:
        errors.append(f"{path}.{key}.status must be explicit")
    if status == "measured":
        if not isinstance(metric.get("seconds"), (int, float)) or metric.get("seconds", 0) <= 0:
            errors.append(f"{path}.{key}.seconds must be positive when measured")
        if not isinstance(metric.get("n"), int) or metric.get("n", 0) <= 0:
            errors.append(f"{path}.{key}.n must be positive when measured")
    elif "reason" not in metric:
        errors.append(f"{path}.{key}.reason is required when not measured")
    return metric


def _identity_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    workload = record.get("workload") if isinstance(record.get("workload"), dict) else {}
    production = record.get("production") if isinstance(record.get("production"), dict) else {}
    prompt = workload.get("prompt") if isinstance(workload.get("prompt"), dict) else {}
    seed_plan = workload.get("seed_plan") if isinstance(workload.get("seed_plan"), dict) else {}
    return {
        "lane_id": record.get("lane_id"),
        "task": workload.get("task"),
        "partition": workload.get("partition"),
        "width": workload.get("width"),
        "height": workload.get("height"),
        "fps": workload.get("fps"),
        "final_frame_count": workload.get("final_frame_count"),
        "nominal_duration_seconds": workload.get("nominal_duration_seconds"),
        "audio_sample_rate_hz": workload.get("audio_sample_rate_hz"),
        "audio_channels": workload.get("audio_channels"),
        "num_inference_steps": workload.get("num_inference_steps"),
        "prompt_sha256": prompt.get("sha256"),
        "seeds": seed_plan.get("seeds"),
        "generation_mode": production.get("generation_mode"),
        "chunk_count": production.get("chunk_count"),
    }


def workload_fingerprint(record: Mapping[str, Any]) -> str:
    encoded = json.dumps(_identity_payload(record), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_lane_geometry(lane_id: Any, workload: Mapping[str, Any], errors: list[str]) -> dict[str, Any] | None:
    spec = LANE_SPECS.get(str(lane_id))
    if spec is None:
        errors.append(f"unknown lane_id: {lane_id!r}")
        return None
    expected_fields = {
        "width": spec["width"],
        "height": spec["height"],
        "fps": spec["fps"],
        "final_frame_count": spec["frames"],
    }
    for key, expected in expected_fields.items():
        if workload.get(key) != expected:
            errors.append(f"workload.{key} must be {expected} for {lane_id}")
    if not _close(workload.get("nominal_duration_seconds"), spec["duration"]):
        errors.append(
            f"workload.nominal_duration_seconds must equal {spec['frames']}/{spec['fps']} for {lane_id}"
        )
    if workload.get("audio_sample_rate_hz") != 32000 or workload.get("audio_channels") != 2:
        errors.append("workload audio must be 32 kHz stereo")
    return spec


def validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        errors.append(f"schema_version must be {CONTRACT_SCHEMA_VERSION}")
    if contract.get("contract_version") != CONTRACT_VERSION:
        errors.append(f"contract_version must be {CONTRACT_VERSION}")
    capabilities = _mapping(contract, "capabilities", errors)
    native = _mapping(capabilities, "native_output_context", errors, "capabilities")
    if native.get("max_seconds") != 15:
        errors.append("capabilities.native_output_context.max_seconds must be 15 for the pinned sources")
    if native.get("native_30_seconds_supported") is not False or native.get("native_60_seconds_supported") is not False:
        errors.append("the pinned open-source path must not claim native 30/60-second support")
    sources = _list(contract, "source_grounding", errors)
    if len(sources) < 3:
        errors.append("source_grounding must include MiniMax-H3, vLLM-Omni, and Sana/Sol-Engine")
    lane_ids = set(contract.get("lane_ids", []))
    if lane_ids != set(LANE_SPECS):
        errors.append("lane_ids must contain exactly the three canonical v1 lanes")
    if errors:
        raise ContractValidationError(errors)
    return {"status": "pass", "kind": "contract", "contract_version": CONTRACT_VERSION}


def validate_lane_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append(f"schema_version must be {MANIFEST_SCHEMA_VERSION}")
    if manifest.get("contract_version") != CONTRACT_VERSION:
        errors.append(f"contract_version must be {CONTRACT_VERSION}")
    if manifest.get("dry_run") is not True:
        errors.append("lane manifest must set dry_run=true")
    if manifest.get("measurement_status") not in {"unmeasured", "dry_run_no_new_measurement"}:
        errors.append("lane manifest measurement_status must remain explicitly unmeasured")
    workload = _mapping(manifest, "workload", errors)
    spec = _validate_lane_geometry(manifest.get("lane_id"), workload, errors)
    prompt = _mapping(workload, "prompt", errors, "workload")
    if not HEX64_RE.fullmatch(str(prompt.get("sha256", ""))):
        errors.append("workload.prompt.sha256 must be a lowercase SHA-256")
    seed_plan = _mapping(workload, "seed_plan", errors, "workload")
    seeds = _list(seed_plan, "seeds", errors, "workload.seed_plan")
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        errors.append("workload.seed_plan.seeds must contain integers")
    production = _mapping(manifest, "production", errors)
    if spec is not None:
        if production.get("is_long") is not spec["is_long"]:
            errors.append("production.is_long does not match lane")
        if production.get("chunk_count") != spec["chunk_count"]:
            errors.append("production.chunk_count does not match the canonical dry-run lane")
        mode = production.get("generation_mode")
        if spec["is_long"]:
            if mode not in LONG_MODES:
                errors.append("every long lane must declare one of the four long-production modes")
            if mode == "native_long_context" and production.get("native_context_supported") is not True:
                errors.append("unsupported native long context cannot be selected")
            if manifest.get("measurement_status") != "unmeasured":
                errors.append("30/60-second manifests must remain unmeasured until real final AV exists")
        elif mode != "native_short_clip":
            errors.append("the short lane must use generation_mode=native_short_clip")
    required_metrics = set(_list(manifest, "required_metrics", errors))
    expected_metrics = {
        "cold_e2e",
        "warm_e2e",
        "denoise",
        "attention_nested_in_denoise",
        "video_vae",
        "audio_vae",
        "encoding_mux",
        "io",
        "seconds_per_generated_second",
        "peak_gpu_memory",
        "peak_host_memory",
        "power",
        "failures",
        *REQUIRED_QUALITY_METRICS,
        "human_gate_separate",
    }
    missing = sorted(expected_metrics - required_metrics)
    if missing:
        errors.append(f"required_metrics missing: {missing}")
    accounting = _mapping(manifest, "final_av_accounting", errors)
    if spec is not None and spec["is_long"]:
        if accounting.get("effective_audio_samples_per_channel") != spec["effective_audio_samples"]:
            errors.append("long-lane final effective audio sample count is not exact")
        if accounting.get("video_frames") != spec["frames"]:
            errors.append("long-lane final video frame count is not exact")
    if errors:
        raise ContractValidationError(errors)
    return {
        "status": "pass",
        "kind": "lane_manifest",
        "lane_id": manifest.get("lane_id"),
        "measurement_status": manifest.get("measurement_status"),
    }


def validate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if record.get("schema_version") != RECORD_SCHEMA_VERSION:
        errors.append(f"schema_version must be {RECORD_SCHEMA_VERSION}")
    if record.get("contract_version") != CONTRACT_VERSION:
        errors.append(f"contract_version must be {CONTRACT_VERSION}")
    record_status = record.get("record_status")
    if record_status not in {"accepted_historical", "candidate", "accepted", "failed"}:
        errors.append("record_status must be accepted_historical, candidate, accepted, or failed")
    workload = _mapping(record, "workload", errors)
    spec = _validate_lane_geometry(record.get("lane_id"), workload, errors)
    prompt = _mapping(workload, "prompt", errors, "workload")
    if not HEX64_RE.fullmatch(str(prompt.get("sha256", ""))):
        errors.append("workload.prompt.sha256 must be a lowercase SHA-256")
    seeds = _list(_mapping(workload, "seed_plan", errors, "workload"), "seeds", errors, "workload.seed_plan")
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        errors.append("workload.seed_plan.seeds must contain integers")
    if not isinstance(workload.get("num_inference_steps"), int) or workload.get("num_inference_steps", 0) <= 0:
        errors.append("workload.num_inference_steps must be positive")

    production = _mapping(record, "production", errors)
    mode = production.get("generation_mode")
    if spec is not None:
        if production.get("is_long") is not spec["is_long"]:
            errors.append("production.is_long does not match lane")
        if spec["is_long"]:
            if mode not in LONG_MODES:
                errors.append("every long result must declare a valid long-production mode")
            if mode == "native_long_context" and (
                production.get("chunk_count") != 1 or production.get("assembly_method") != "none"
            ):
                errors.append("stitched/chunked output cannot be claimed as native_long_context")
        elif mode != "native_short_clip":
            errors.append("short result must use generation_mode=native_short_clip")

    track = _mapping(record, "track", errors)
    track_id = track.get("id")
    if track_id not in TRACKS:
        errors.append(f"track.id must be one of {sorted(TRACKS)}")
    mechanisms = _list(track, "mechanisms", errors, "track")
    positive_mechanisms = [
        str(item).lower()
        for item in mechanisms
        if not str(item).lower().startswith(("no_", "disabled_"))
    ]
    if track_id == "fidelity_bf16_exact" and any(
        token in " ".join(positive_mechanisms)
        for token in ("turbo", "cache", "quant", "sparse")
    ):
        errors.append("approximate/Turbo/cache/quant/sparse mechanisms cannot be labeled BF16 fidelity")

    deployment = _mapping(record, "deployment", errors)
    scope = deployment.get("scope")
    if scope not in DEPLOYMENT_SCOPES:
        errors.append(f"deployment.scope must be one of {sorted(DEPLOYMENT_SCOPES)}")
    uuids = _list(deployment, "physical_gpu_uuids", errors, "deployment")
    if scope == "single_a6000":
        if deployment.get("gpu_count_visible") != 1 or deployment.get("gpu_count_used") != 1 or len(uuids) != 1:
            errors.append("single_a6000 results must prove exactly one visible and used physical GPU")
        if deployment.get("gpu_model") != "NVIDIA RTX A6000":
            errors.append("single_a6000 result gpu_model must be NVIDIA RTX A6000")
    if scope == "multi_gpu_production" and deployment.get("gpu_count_used", 0) < 2:
        errors.append("multi_gpu_production must use at least two GPUs")

    expected_fingerprint = workload_fingerprint(record)
    if record.get("workload_fingerprint") != expected_fingerprint:
        errors.append("workload_fingerprint does not match the canonical workload identity fields")

    timing = _mapping(record, "timing", errors)
    cold = _timing_measure(timing, "cold_e2e", errors)
    warm = _timing_measure(timing, "warm_e2e", errors)
    components = _mapping(timing, "components", errors, "timing")
    component_metrics: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_COMPONENTS:
        component_metrics[name] = _timing_measure(components, name, errors, "timing.components")
    attention = component_metrics["attention"]
    if attention.get("parent") != "denoise" or attention.get("additive_to_e2e") is not False:
        errors.append("timing.components.attention must be nested in denoise and non-additive")
    additive = timing.get("additive_component_order")
    if additive != list(ADDITIVE_COMPONENTS):
        errors.append("timing.additive_component_order must exclude attention to prevent double-counting")
    denoise = component_metrics["denoise"]
    if attention.get("status") == "measured" and denoise.get("status") == "measured":
        if float(attention["seconds"]) > float(denoise["seconds"]) + 1e-9:
            errors.append("nested attention seconds cannot exceed denoise seconds on the same timing basis")
    spgs = _timing_measure(timing, "seconds_per_generated_second", errors)
    if spgs.get("status") == "measured" and warm.get("status") == "measured" and spec is not None:
        expected = float(warm["seconds"]) / float(spec["duration"])
        if not _close(spgs.get("seconds"), expected, abs_tol=1e-5):
            errors.append("seconds_per_generated_second must equal warm_e2e/final generated seconds")

    resources = _mapping(record, "resources", errors)
    for name in ("peak_gpu_memory_mib", "peak_host_memory_gib", "peak_power_w", "failures"):
        _metric(resources, name, errors, "resources")

    output = _mapping(record, "output_av", errors)
    video = _mapping(output, "video", errors, "output_av")
    audio = _mapping(output, "audio", errors, "output_av")
    accepted_like = record_status in {"accepted_historical", "accepted"}
    if accepted_like:
        if output.get("status") != "complete" or output.get("final_accounting_complete") is not True:
            errors.append("accepted records require complete final AV accounting")
        if video.get("present") is not True or video.get("full_decode") is not True:
            errors.append("accepted records require a present, fully decoded final video stream")
        if audio.get("present") is not True or audio.get("full_decode") is not True:
            errors.append("accepted records require a present, fully decoded final audio stream")
    if spec is not None:
        if video.get("width") != spec["width"] or video.get("height") != spec["height"]:
            errors.append("final video geometry does not match lane")
        if video.get("fps") != spec["fps"] or video.get("frames") != spec["frames"]:
            errors.append("final video frame accounting does not match lane")
        if audio.get("sample_rate_hz") != 32000 or audio.get("channels") != 2:
            errors.append("final audio must be 32 kHz stereo")
        if not isinstance(audio.get("decoded_samples_per_channel"), int) or audio.get(
            "decoded_samples_per_channel", 0
        ) <= 0:
            errors.append("decoded final audio samples per channel must be reported")
        if spec["is_long"] and audio.get("effective_samples_per_channel") != spec["effective_audio_samples"]:
            errors.append("long final AV effective audio sample accounting is incomplete or inexact")

    quality = _mapping(record, "quality", errors)
    objective = _mapping(quality, "objective", errors, "quality")
    quality_metrics: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_QUALITY_METRICS:
        quality_metrics[name] = _metric(objective, name, errors, "quality.objective")
    human_gate = _mapping(quality, "human_gate", errors, "quality")
    if human_gate.get("status") not in {
        "pass",
        "fail",
        "not_performed_no_semantic_claim",
        "not_required_historical_timing_baseline",
    }:
        errors.append("quality.human_gate.status must remain separate and explicit")
    if spec is not None and spec["is_long"] and record_status == "accepted":
        unavailable = [
            name
            for name, value in quality_metrics.items()
            if value.get("status") in {"not_available_historical_evidence", "not_run_dry_run"}
        ]
        if unavailable:
            errors.append(f"accepted long result has unmeasured mandatory quality metrics: {unavailable}")
        if human_gate.get("status") != "pass":
            errors.append("accepted long-production result requires a separate passing human gate")

    promotion = _mapping(record, "promotion", errors)
    sample_count = promotion.get("sample_count")
    if not isinstance(sample_count, int) or sample_count <= 0:
        errors.append("promotion.sample_count must be positive")
    if promotion.get("level") == "formal_n_ge_10" and isinstance(sample_count, int) and sample_count < 10:
        errors.append("formal_n_ge_10 requires at least 10 samples")
    threshold_id = promotion.get("quality_threshold_id")
    if not isinstance(threshold_id, str) or not threshold_id:
        errors.append("promotion.quality_threshold_id is required")

    comparisons = _list(record, "comparisons", errors)
    for index, comparison in enumerate(comparisons):
        if not isinstance(comparison, dict):
            errors.append(f"comparisons[{index}] must be an object")
            continue
        candidate = _mapping(comparison, "candidate", errors, f"comparisons[{index}]")
        denominator = _mapping(comparison, "denominator", errors, f"comparisons[{index}]")
        expected_candidate = {
            "track": track_id,
            "workload_fingerprint": record.get("workload_fingerprint"),
            "deployment_scope": scope,
            "physical_gpu_uuid": uuids[0] if len(uuids) == 1 else None,
            "timing_boundary_id": timing.get("boundary_id"),
            "quality_threshold_id": threshold_id,
            "generation_mode": mode,
        }
        for key, expected in expected_candidate.items():
            if candidate.get(key) != expected:
                errors.append(f"comparisons[{index}].candidate.{key} does not match this record")
        for key in (
            "track",
            "workload_fingerprint",
            "deployment_scope",
            "physical_gpu_uuid",
            "timing_boundary_id",
            "quality_threshold_id",
            "generation_mode",
        ):
            if candidate.get(key) != denominator.get(key):
                errors.append(f"comparisons[{index}] mismatched denominator field: {key}")
        if comparison.get("principal_variable") in (None, ""):
            errors.append(f"comparisons[{index}].principal_variable is required")
        if not isinstance(comparison.get("sample_count"), int) or comparison.get("sample_count", 0) < 10:
            errors.append(f"comparisons[{index}] formal speedup/improvement requires N>=10")
        if comparison.get("result_kind") not in {
            "median_speedup_ratio",
            "median_time_improvement_percent",
        }:
            errors.append(f"comparisons[{index}].result_kind is invalid")
        if not isinstance(comparison.get("value"), (int, float)) or comparison.get("value", 0) <= 0:
            errors.append(f"comparisons[{index}].value must be positive")

    if errors:
        raise ContractValidationError(errors)
    return {
        "status": "pass",
        "kind": "benchmark_record",
        "record_id": record.get("record_id"),
        "lane_id": record.get("lane_id"),
        "record_status": record_status,
        "comparison_count": len(comparisons),
    }


def validate_artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    version = value.get("schema_version")
    if version == CONTRACT_SCHEMA_VERSION:
        return validate_contract(value)
    if version == MANIFEST_SCHEMA_VERSION:
        return validate_lane_manifest(value)
    if version == RECORD_SCHEMA_VERSION:
        return validate_record(value)
    raise ContractValidationError([f"unsupported schema_version: {version!r}"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate MiniMax-H3 benchmark contract v1 artifacts.")
    parser.add_argument("paths", nargs="+", type=Path, help="contract, lane manifest, or benchmark record JSON")
    parser.add_argument("--json", action="store_true", help="emit a machine-readable result")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for path in args.paths:
        try:
            summary = validate_artifact(load_json(path))
            results.append({"path": path.as_posix(), **summary})
        except ContractValidationError as exc:
            failures.append({"path": path.as_posix(), "errors": exc.errors})
    payload = {
        "status": "pass" if not failures else "fail",
        "validated": results,
        "failures": failures,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif failures:
        for failure in failures:
            for error in failure["errors"]:
                print(f"ERROR {failure['path']}: {error}", file=sys.stderr)
    else:
        for result in results:
            print(f"PASS {result['path']} kind={result['kind']}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
