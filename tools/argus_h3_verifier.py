#!/usr/bin/env python3
"""Pre-gate MiniMax-H3 run verifier helpers.

This module intentionally supports a metadata-only mock AV fixture so the
pre-gate infrastructure can be tested without MiniMax-H3 weights, generated
outputs, ffmpeg/ffprobe, GPU inference, or large downloads.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PINNED_SOURCE_COMMIT = "b7227fa6a6206e9fb30562383d39e53cf3866a48"
PINNED_CHECKPOINT_REVISION = "6818f6c32d12b210915e44ad56a4228c2608f160"
SCHEMA_VERSION = "argus-minimax-h3-run-v1"
ALLOWED_TRACKS = {"fidelity_bf16_exact", "practical_disclosed_approx"}
ALLOWED_PARTITIONS = {"FL2VA", "Ref2VA"}
ALLOWED_TASKS = {"t2va", "fl2va", "ref2va"}
REFERENCE_PLATFORM_IDS = {"current_a6000_reference"}
TARGET_PLATFORM_IDS = {"single_a6000_48gb_workstation"}
BLOCKED_PLATFORM_IDS = {"dgx_spark", "rtx_5090_single"}


class VerificationError(Exception):
    """Raised when a run record cannot be accepted by the pre-gate verifier."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise VerificationError([f"{path} must contain a JSON object"])
    return data


def require_mapping(parent: dict[str, Any], key: str, errors: list[str]) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        errors.append(f"missing or non-object field: {key}")
        return {}
    return value


def require_list(parent: dict[str, Any], key: str, errors: list[str]) -> list[Any]:
    value = parent.get(key)
    if not isinstance(value, list):
        errors.append(f"missing or non-list field: {key}")
        return []
    return value


def _positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and value > 0


def verify_record(record: dict[str, Any], *, root: Path | None = None, require_real_output: bool = False) -> dict[str, Any]:
    """Validate a MiniMax-H3 run/fixture record and return a concise summary."""

    errors: list[str] = []
    warnings: list[str] = []

    for key in (
        "schema_version",
        "run_id",
        "artifact_kind",
        "mock_fixture",
        "real_h3_output",
        "platform",
        "track",
        "model_identity",
        "workload",
        "environment",
        "timing",
        "resources",
        "output_av",
        "hidden_cache_controls",
        "gates",
        "claims",
    ):
        if key not in record:
            errors.append(f"missing required top-level field: {key}")

    if errors:
        raise VerificationError(errors)

    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")

    mock_fixture = record.get("mock_fixture") is True
    real_h3_output = record.get("real_h3_output") is True
    if mock_fixture and real_h3_output:
        errors.append("mock_fixture cannot also claim real_h3_output")
    if require_real_output and not real_h3_output:
        errors.append("--require-real-output was set, but record is not a real H3 output")

    artifact_kind = record.get("artifact_kind")
    if mock_fixture and artifact_kind != "metadata_only_av_fixture":
        errors.append("mock fixtures must use artifact_kind=metadata_only_av_fixture")

    platform = require_mapping(record, "platform", errors)
    platform_id = platform.get("id")
    platform_role = platform.get("role")
    if platform_id in BLOCKED_PLATFORM_IDS and platform.get("status") != "BLOCKED":
        errors.append(f"blocked target platform {platform_id} must be marked BLOCKED")
    if platform_id in REFERENCE_PLATFORM_IDS and platform_role != "reference":
        errors.append("current_a6000_reference must be labeled role=reference")
    if platform_id in TARGET_PLATFORM_IDS and platform_role != "target":
        errors.append("single_a6000_48gb_workstation must be labeled role=target")
    if platform_id not in REFERENCE_PLATFORM_IDS | TARGET_PLATFORM_IDS | BLOCKED_PLATFORM_IDS:
        warnings.append(f"unrecognized platform id: {platform_id}")

    track = record.get("track")
    if track not in ALLOWED_TRACKS:
        errors.append(f"track must be one of {sorted(ALLOWED_TRACKS)}")

    model = require_mapping(record, "model_identity", errors)
    if model.get("source_commit") != PINNED_SOURCE_COMMIT:
        errors.append("model_identity.source_commit does not match pinned MiniMax-H3 source commit")
    if model.get("checkpoint_revision") != PINNED_CHECKPOINT_REVISION:
        errors.append("model_identity.checkpoint_revision does not match pinned MiniMax-H3 checkpoint revision")
    if model.get("task_partition") not in ALLOWED_PARTITIONS:
        errors.append("model_identity.task_partition must be FL2VA or Ref2VA")
    if model.get("weights_downloaded") is True and not real_h3_output:
        errors.append("weights_downloaded=true is not allowed for metadata-only/pre-gate fixture records")

    workload = require_mapping(record, "workload", errors)
    if workload.get("task") not in ALLOWED_TASKS:
        errors.append("workload.task must be t2va, fl2va, or ref2va")
    seeds = require_list(workload, "seeds", errors)
    if not seeds or not all(isinstance(seed, int) for seed in seeds):
        errors.append("workload.seeds must contain at least one integer seed")
    if not _positive_number(workload.get("duration_seconds")):
        errors.append("workload.duration_seconds must be positive")
    if workload.get("fps") != 24:
        errors.append("workload.fps must be the pinned official 24 FPS unless the baseline lock is updated")
    audio = require_mapping(workload, "audio", errors)
    if audio.get("sample_rate_hz") != 32000 or audio.get("channels") != 2:
        errors.append("workload.audio must be pinned to 32 kHz stereo")
    steps = workload.get("steps")
    if isinstance(steps, dict) and steps.get("status") == "unresolved_pre_gate":
        if not mock_fixture:
            errors.append("real runs must freeze denoising/inference steps before verification")
        warnings.append("steps are unresolved because this is a pre-gate metadata-only fixture")
    elif not isinstance(steps, int) or steps <= 0:
        errors.append("workload.steps must be a positive integer for real runs, or unresolved_pre_gate for mock fixtures")

    environment = require_mapping(record, "environment", errors)
    container = require_mapping(environment, "container", errors)
    if not container.get("image_digest"):
        errors.append("environment.container.image_digest is required")
    if environment.get("global_system_modified") is not False:
        errors.append("environment.global_system_modified must be false for pre-gate records")

    timing = require_mapping(record, "timing", errors)
    for boundary in ("cold_load", "first_compile", "warm_e2e", "hot_path", "fps"):
        if boundary not in timing:
            errors.append(f"timing.{boundary} boundary is required")
    if timing.get("speedup_claimed") not in (False, None):
        errors.append("pre-gate verifier records must not claim speedup")

    resources = require_mapping(record, "resources", errors)
    if resources.get("gpu_compute_performed") is not False and mock_fixture:
        errors.append("mock fixtures must not claim GPU compute")
    if resources.get("gpu_count_used") not in (0, None) and mock_fixture:
        errors.append("mock fixtures must use zero GPUs")

    output_av = require_mapping(record, "output_av", errors)
    video = require_mapping(output_av, "video", errors)
    audio_out = require_mapping(output_av, "audio", errors)
    output_files = output_av.get("output_files", [])
    if mock_fixture:
        if output_av.get("metadata_only") is not True:
            errors.append("mock AV fixture must set output_av.metadata_only=true")
        if output_files:
            errors.append("mock AV fixture must not list real output files")
        if video.get("present") is not True or audio_out.get("present") is not True:
            errors.append("mock AV metadata must include both video and audio presence")
        if audio_out.get("channels") != 2 or audio_out.get("sample_rate_hz") != 32000:
            errors.append("mock AV metadata must describe 32 kHz stereo audio")
    else:
        if output_av.get("metadata_only") is True:
            errors.append("real runs cannot be metadata-only")
        if not output_files:
            errors.append("real runs must list output files for checksum/decode verification")
        root = root or Path.cwd()
        for item in output_files:
            rel = item.get("path") if isinstance(item, dict) else None
            if not rel or not (root / rel).exists():
                errors.append(f"listed output file does not exist under verifier root: {rel}")
        if video.get("full_decode") is not True or audio_out.get("full_decode") is not True:
            errors.append("real runs must prove full video and stereo-audio decode")

    cache = require_mapping(record, "hidden_cache_controls", errors)
    declared_roots = require_list(cache, "declared_cache_roots", errors)
    forbidden_roots = require_list(cache, "forbidden_implicit_cache_roots", errors)
    pre_snapshot = require_mapping(cache, "pre_run_snapshot", errors)
    post_snapshot = require_mapping(cache, "post_run_snapshot", errors)
    if not declared_roots:
        errors.append("hidden_cache_controls.declared_cache_roots must not be empty")
    if not forbidden_roots:
        errors.append("hidden_cache_controls.forbidden_implicit_cache_roots must not be empty")
    if cache.get("clean_room_required_for_claim") is not True:
        errors.append("hidden_cache_controls.clean_room_required_for_claim must be true")
    if not cache.get("no_hidden_cache_assertion"):
        errors.append("hidden_cache_controls.no_hidden_cache_assertion is required")
    if mock_fixture:
        if pre_snapshot.get("total_bytes", 0) != 0 or post_snapshot.get("total_bytes", 0) != 0:
            errors.append("mock fixture cache snapshots must remain empty")

    gates = require_mapping(record, "gates", errors)
    if mock_fixture:
        for gate_name in ("license_territory_approved", "large_download_approved", "expensive_resource_approved"):
            if gates.get(gate_name) is not False:
                errors.append(f"mock fixture must leave {gate_name}=false")
        if gates.get("model_weights_accessed") is not False:
            errors.append("mock fixture must set model_weights_accessed=false")

    claims = require_mapping(record, "claims", errors)
    if claims.get("speedup") is not None:
        errors.append("no speedup claim is allowed in pre-gate/mock verification")
    if claims.get("argus_result") is not False:
        errors.append("pre-gate fixture must not be marked as an Argus result")

    if errors:
        raise VerificationError(errors)

    return {
        "status": "pass",
        "schema_version": record.get("schema_version"),
        "run_id": record.get("run_id"),
        "artifact_kind": artifact_kind,
        "mock_fixture": mock_fixture,
        "real_h3_output": real_h3_output,
        "platform_id": platform_id,
        "platform_role": platform_role,
        "track": track,
        "task_partition": model.get("task_partition"),
        "task": workload.get("task"),
        "av": {
            "metadata_only": output_av.get("metadata_only"),
            "video_present": video.get("present"),
            "audio_present": audio_out.get("present"),
            "audio_channels": audio_out.get("channels"),
            "audio_sample_rate_hz": audio_out.get("sample_rate_hz"),
        },
        "hidden_cache": {
            "declared_cache_roots": declared_roots,
            "forbidden_implicit_cache_roots": forbidden_roots,
            "pre_run_total_bytes": pre_snapshot.get("total_bytes", 0),
            "post_run_total_bytes": post_snapshot.get("total_bytes", 0),
            "clean_room_required_for_claim": cache.get("clean_room_required_for_claim"),
        },
        "warnings": warnings,
    }


def verify_fixture(fixture_dir: Path, *, require_real_output: bool = False) -> dict[str, Any]:
    fixture_dir = fixture_dir.resolve()
    record = load_json(fixture_dir / "run_record.json")
    return verify_record(record, root=fixture_dir, require_real_output=require_real_output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify MiniMax-H3 run metadata or pre-gate AV fixtures.")
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--fixture", type=Path, help="Fixture directory containing run_record.json")
    source.add_argument("--run-record", type=Path, help="Path to a run_record.json file")
    parser.add_argument("positional_run_record", nargs="?", type=Path, help="Compatibility positional path to run_record.json")
    parser.add_argument("--schema", type=Path, default=Path("schemas/minimax_h3_run.schema.json"), help="Schema path to check for presence/readability")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable summary")
    parser.add_argument("--require-real-output", action="store_true", help="Reject metadata-only/mock records")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        schema_path = args.schema
        if schema_path.exists():
            load_json(schema_path)
        else:
            raise VerificationError([f"schema file not found: {schema_path}"])

        source_count = sum(value is not None for value in (args.fixture, args.run_record, args.positional_run_record))
        if source_count != 1:
            parser.error("provide exactly one of --fixture, --run-record, or positional run_record.json")
        if args.fixture:
            summary = verify_fixture(args.fixture, require_real_output=args.require_real_output)
        else:
            record_arg = args.run_record or args.positional_run_record
            record_path = record_arg.resolve()
            summary = verify_record(load_json(record_path), root=record_path.parent, require_real_output=args.require_real_output)
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print(f"PASS {summary['run_id']} track={summary['track']} platform={summary['platform_id']} mock={summary['mock_fixture']}")
            for warning in summary["warnings"]:
                print(f"WARNING {warning}", file=sys.stderr)
        return 0
    except VerificationError as exc:
        if args.json:
            print(json.dumps({"status": "fail", "errors": exc.errors}, indent=2, sort_keys=True))
        else:
            for error in exc.errors:
                print(f"ERROR {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
