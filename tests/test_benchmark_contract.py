from __future__ import annotations

import copy
import glob
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from minimax_h3_benchmark_contract import (  # noqa: E402
    ContractValidationError,
    REQUIRED_COMPONENTS,
    REQUIRED_QUALITY_METRICS,
    validate_artifact,
    validate_record,
)

CONTRACT_ROOT = ROOT / "benchmark_contract" / "v1"
RECORDS = sorted((CONTRACT_ROOT / "normalized-records").glob("*.json"))
MANIFESTS = sorted((CONTRACT_ROOT / "lane-manifests").glob("*.json"))
REJECTED = sorted((ROOT / "tests" / "fixtures" / "benchmark_contract" / "rejected").glob("*.json"))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mutate(base: dict, mutation: dict) -> None:
    parts = mutation["path"].split(".")
    parent = base
    for part in parts[:-1]:
        parent = parent[int(part)] if isinstance(parent, list) else parent[part]
    leaf = parts[-1]
    if mutation["op"] == "set":
        if isinstance(parent, list):
            parent[int(leaf)] = mutation["value"]
        else:
            parent[leaf] = mutation["value"]
    elif mutation["op"] == "delete":
        if isinstance(parent, list):
            del parent[int(leaf)]
        else:
            del parent[leaf]
    else:  # pragma: no cover - fixture format guard
        raise AssertionError(f"unsupported mutation: {mutation}")


def test_contract_and_three_dry_run_lane_manifests_validate() -> None:
    contract = _load(CONTRACT_ROOT / "contract.json")
    assert validate_artifact(contract)["status"] == "pass"
    assert len(MANIFESTS) == 3

    summaries = [validate_artifact(_load(path)) for path in MANIFESTS]
    assert {item["lane_id"] for item in summaries} == set(contract["lane_ids"])
    long_manifests = [
        _load(path)
        for path in MANIFESTS
        if _load(path)["production"]["is_long"]
    ]
    assert {item["measurement_status"] for item in long_manifests} == {"unmeasured"}
    assert {item["production"]["generation_mode"] for item in long_manifests} == {"extension"}
    assert {item["production"]["native_context_supported"] for item in long_manifests} == {False}


def test_source_grounding_establishes_real_temporal_and_conditioning_boundaries() -> None:
    contract = _load(CONTRACT_ROOT / "contract.json")
    native = contract["capabilities"]["native_output_context"]
    local = contract["capabilities"]["local_model"]

    assert native["max_seconds"] == 15
    assert native["native_30_seconds_supported"] is False
    assert native["native_60_seconds_supported"] is False
    assert local["prepared_partitions"] == ["FL2VA"]
    assert local["available_tasks"] == ["t2va", "fl2va"]
    assert local["unavailable_local_tasks"] == ["ref2va"]
    assert {item["revision"] for item in contract["source_grounding"]} >= {
        "6818f6c32d12b210915e44ad56a4228c2608f160",
        "8e2e9b6b53e86e6a479ed2c0a53782f655f60e04",
        "d00eef311670a58deb2c323fe072738fcb945600",
    }
    assert all(item["license"] and item["revision_date_utc"] for item in contract["source_grounding"])


def test_schema_and_semantic_validator_accept_normalized_historical_records() -> None:
    schema = _load(ROOT / "schemas" / "minimax_h3_benchmark_record_v1.schema.json")
    jsonschema.Draft202012Validator.check_schema(schema)
    static_validator = jsonschema.Draft202012Validator(schema)

    assert len(RECORDS) == 4
    for path in RECORDS:
        record = _load(path)
        static_validator.validate(record)
        summary = validate_record(record)
        assert summary["status"] == "pass"
        assert set(record["timing"]["components"]) == set(REQUIRED_COMPONENTS)
        assert set(record["quality"]["objective"]) == set(REQUIRED_QUALITY_METRICS)
        assert record["timing"]["components"]["attention"]["parent"] == "denoise"
        assert record["timing"]["components"]["attention"]["additive_to_e2e"] is False
        assert "attention" not in record["timing"]["additive_component_order"]


def test_normalization_retains_exact_short_claim_boundaries_without_new_cross_track_speedup() -> None:
    by_id = {_load(path)["record_id"]: _load(path) for path in RECORDS}
    bf16 = by_id["historical-bf16-warm-n10-v1"]
    turbo8 = by_id["historical-turbo-8step-warm-n10-v1"]
    turbo4 = by_id["historical-turbo-4step-warm-n10-v1"]
    sol = by_id["historical-sol-attn-r8-formal-n10-v1"]

    assert bf16["timing"]["warm_e2e"]["seconds"] == 1792.2021025
    assert turbo8["timing"]["warm_e2e"]["seconds"] == 290.9976015
    assert turbo4["timing"]["warm_e2e"]["seconds"] == 149.6191865
    assert turbo8["track"]["id"] == turbo4["track"]["id"] == "practical_disclosed_approx"
    assert turbo8["comparisons"] == turbo4["comparisons"] == []
    assert sol["comparisons"][0]["value"] == 15.203295894081867
    assert sol["comparisons"][0]["candidate"]["track"] == sol["comparisons"][0]["denominator"]["track"]
    assert "5-step" in sol["claim_boundary"]
    assert all(record["production"]["is_long"] is False for record in by_id.values())


@pytest.mark.parametrize("fixture_path", REJECTED, ids=lambda path: path.stem)
def test_negative_claim_boundary_fixtures_fail_closed(fixture_path: Path) -> None:
    fixture = _load(fixture_path)
    record = copy.deepcopy(_load(ROOT / fixture["base_record"]))
    for mutation in fixture["mutations"]:
        _mutate(record, mutation)

    with pytest.raises(ContractValidationError) as excinfo:
        validate_record(record)

    assert any(fixture["expected_error"] in error for error in excinfo.value.errors)


def test_cli_validates_contract_manifests_and_records() -> None:
    paths = [CONTRACT_ROOT / "contract.json", *MANIFESTS, *RECORDS]
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "validate_benchmark_record.py"), "--json", *map(str, paths)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["status"] == "pass"
    assert len(payload["validated"]) == 8
    assert payload["failures"] == []
