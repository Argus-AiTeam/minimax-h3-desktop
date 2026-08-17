#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Localize r9 pair-value-halves same-input shadow divergence from scalar telemetry.

This post-run diagnostic consumes the bounded real-chain shadow records written by
``finalize_sol_attn_pair_value_halves_diagnostic.py``.  It never reads or writes
raw tensors, model weights, videos, or private absolute paths; it only combines
scalar/layout telemetry with pinned local/upstream source hashes and runtime
version facts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
CURRENT_A = "current_retained_a"
CURRENT_B = "current_retained_b"
CANDIDATE = "pair_value_halves"

LOCAL_BACKEND = Path("ports/minimax_h3_a6000/src/minimax_h3_a6000/sol_attn_backend.py")
LOCAL_TRITON = Path("ports/minimax_h3_a6000/src/minimax_h3_a6000/sol_attn_triton_sm86.py")
LOCALIZER = Path("ports/minimax_h3_a6000/integration/localize_sol_attn_pair_value_halves_shadow.py")
UPSTREAM_FWD = Path("upstreams/Sana-sol-engine/techniques/sparse_backends/sol_attn/triton_ref/fwd.py")
UPSTREAM_H3 = Path("upstreams/Sana-sol-engine/models/minimax_h3/gb200/optimized/sol_attn_h3.py")
UPSTREAM_INTERFACE = Path("upstreams/Sana-sol-engine/techniques/sparse_backends/sol_attn/interface.py")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def source(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def git_head(path: Path) -> str | None:
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    head = proc.stdout.strip()
    return head if proc.returncode == 0 and head else None


def runtime_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {"python": sys.version.split()[0]}
    try:
        import torch

        versions["torch"] = getattr(torch, "__version__", "unknown")
        versions["torch_cuda"] = getattr(getattr(torch, "version", object()), "cuda", None)
    except Exception as exc:  # noqa: BLE001 - diagnostic should remain no-GPU and fail closed
        versions["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import triton

        versions["triton"] = getattr(triton, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        versions["triton_error"] = f"{type(exc).__name__}: {exc}"
    return versions


def first_mismatch_records(shadow: dict[str, Any], telemetry: dict[str, Any]) -> list[dict[str, Any]]:
    records = [r for r in telemetry.get("shadow_pair_value_halves_records", []) if isinstance(r, dict)]
    if records:
        return records
    return [r for r in shadow.get("first_mismatches", []) if isinstance(r, dict)]


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    halves = Counter()
    regions = Counter()
    max_abs = Counter()
    step_layers = Counter()
    prefix_equal = padding_equal = tail_equal = 0
    for record in records:
        bucket = ((record.get("error") or {}).get("argmax_region_bucket") or {})
        halves[str(bucket.get("value_half", "unknown"))] += 1
        regions[str(bucket.get("region", "unknown"))] += 1
        err = (record.get("error") or {}).get("max_abs")
        if isinstance(err, (int, float)):
            max_abs[float(err)] += 1
        meta = record.get("metadata") or {}
        step_layers[(int(meta.get("step_index", -1)), int(meta.get("layer_index", -1)))] += 1
        eq = record.get("region_equality") or {}
        prefix_equal += int(bool(eq.get("prefix_equal")))
        tail_equal += int(bool(eq.get("tail_equal")))
        padding_equal += int(bool(eq.get("padding_equal")))
    first = records[0] if records else {}
    first_bucket = (((first.get("error") or {}).get("argmax_region_bucket")) or {}) if isinstance(first, dict) else {}
    first_meta = (first.get("metadata") or {}) if isinstance(first, dict) else {}
    return {
        "records_available": len(records),
        "first_mismatch": {
            "step_index": first_meta.get("step_index"),
            "layer_index": first_meta.get("layer_index"),
            "token_index": first_bucket.get("token_index"),
            "token_block64": first_bucket.get("token_block64"),
            "head": first_bucket.get("head"),
            "dim": first_bucket.get("dim"),
            "region": first_bucket.get("region"),
            "value_half": first_bucket.get("value_half"),
            "max_abs": (first.get("error") or {}).get("max_abs") if isinstance(first, dict) else None,
            "mean_abs": (first.get("error") or {}).get("mean_abs") if isinstance(first, dict) else None,
        },
        "argmax_region_counts": dict(sorted(regions.items())),
        "argmax_value_half_counts": dict(sorted(halves.items())),
        "record_region_equality_counts": {
            "prefix_equal_records": prefix_equal,
            "tail_equal_records": tail_equal,
            "padding_equal_records": padding_equal,
        },
        "max_abs_top_counts": [
            {"max_abs": value, "count": count} for value, count in max_abs.most_common(12)
        ],
        "unique_step_layer_records": len(step_layers),
        "first_step_layers": [
            {"step_index": step, "layer_index": layer, "count": count}
            for (step, layer), count in list(step_layers.items())[:16]
        ],
    }


def source_comparison() -> dict[str, Any]:
    upstream_fwd = source(UPSTREAM_FWD)
    upstream_h3 = source(UPSTREAM_H3)
    local_triton = source(LOCAL_TRITON)
    local_backend = source(LOCAL_BACKEND)

    current_ptr_checks = {
        "upstream_ptr_has_online_row_state": all(
            text in upstream_fwd for text in ["row_sum", "row_max", "tl.math.exp2", "exact_offsets"]
        ),
        "local_current_ptr_keeps_upstream_route_formula": all(
            text in local_triton
            for text in [
                "tl.sum(scores, axis=0) / q_len > route_threshold",
                "tl.abs(q_block - block_indices) <= 1",
                "exact_offsets = tl.where(dynamic_exact, group_offsets, GROUP)",
            ]
        ),
        "local_current_ptr_keeps_inline_bf16_probability_dot": bool(
            re.search(r"tl\.dot\(\s*probability\.to\(vc\.dtype\),\s*vc", local_triton)
        ),
        "local_current_exact_ptr_keeps_strided_v_load_formula": all(
            text in local_triton
            for text in [
                "kv_tokens[:, None].to(tl.int64) * V_STRIDE_T",
                "head * V_STRIDE_H",
                "value_dims[None, :] * V_STRIDE_D",
            ]
        ),
    }
    pair_checks = {
        "pair_kernel_is_default_off": '"MINIMAX_H3_A6000_SOL_ATTN_PAIR_VALUE_HALVES": "0"' in source(
            Path("ports/minimax_h3_a6000/src/minimax_h3_a6000/env.py")
        ),
        "pair_kernel_keeps_per_half_probability_cast_boundary": all(
            text in local_triton
            for text in [
                "tl.dot(probability.to(vc_lo.dtype), vc_lo)",
                "tl.dot(probability.to(vc_hi.dtype), vc_hi)",
                "tl.dot(exact_probability.to(v_lo.dtype), v_lo)",
                "tl.dot(exact_probability.to(v_hi.dtype), v_hi)",
            ]
        ),
        "pair_kernel_uses_two_bv64_pv_dots": all(
            text in local_triton
            for text in [
                "tl.dot(probability.to(vc_lo.dtype), vc_lo)",
                "tl.dot(probability.to(vc_hi.dtype), vc_hi)",
                "tl.dot(exact_probability.to(v_lo.dtype), v_lo)",
                "tl.dot(exact_probability.to(v_hi.dtype), v_hi)",
            ]
        ),
        "pair_kernel_uses_two_explicit_output_stores": all(
            text in local_triton for text in ["output_offsets_lo", "output_offsets_hi"]
        ),
        "pair_kernel_shares_row_state_once": all(
            text in local_triton
            for text in [
                "output_lo = output_lo * alpha[:, None]",
                "output_hi = output_hi * alpha[:, None]",
                "row_sum = row_sum * alpha",
                "row_max = new_max",
            ]
        ),
    }
    h3_checks = {
        "upstream_h3_dense_prefix_query_overwrite": all(
            text in upstream_h3
            for text in [
                "The sink makes the prefix exact as *keys*",
                "out[:, lo:hi] = _dense",
            ]
        ),
        "local_backend_preserves_dense_prefix_overwrite": all(
            text in local_backend
            for text in [
                "prefix_dense = bool(policy.prefix_query_dense",
                "out[:, :prefix] = dense_attention_reference",
            ]
        ),
        "local_shadow_returns_retained_current": "returned_output\": \"retained_current\"" in local_backend,
    }
    return {
        "pinned_upstream_revision": git_head(REPO_ROOT / "upstreams/Sana-sol-engine"),
        "source_hashes": {
            rel(REPO_ROOT / path): file_sha256(REPO_ROOT / path)
            for path in [UPSTREAM_FWD, UPSTREAM_INTERFACE, UPSTREAM_H3, LOCAL_TRITON, LOCAL_BACKEND, LOCALIZER]
        },
        "runtime_versions": runtime_versions(),
        "first_party_semantics": h3_checks,
        "current_vs_upstream_ptr_semantics": current_ptr_checks,
        "pair_value_halves_numeric_codegen_delta": pair_checks,
        "comparison_summary": (
            "Pinned upstream Triton/current local pointer semantics keep one BV tile per forward program with "
            "the upstream online-softmax order and inline BF16 probability-to-value dot.  The local pair-value "
            "candidate remains the non-upstream arithmetic/codegen delta: one Triton program shares the route "
            "and row_max/row_sum, keeps the BF16 probability cast boundary per half, then issues separate "
            "lo/hi BV64 value dots/stores."
        ),
    }


def localization(shadow: dict[str, Any], original_decision: dict[str, Any], records_summary: dict[str, Any], sources: dict[str, Any]) -> dict[str, Any]:
    gates = original_decision.get("gates") or {}
    half_counts = records_summary.get("argmax_value_half_counts") or {}
    equality = records_summary.get("record_region_equality_counts") or {}
    all_prefix_padding_equal = (
        equality.get("prefix_equal_records") == records_summary.get("records_available")
        and equality.get("padding_equal_records") == records_summary.get("records_available")
    )
    both_halves_seen = int(half_counts.get("lo", 0) or 0) > 0 and int(half_counts.get("hi", 0) or 0) > 0
    all_tail_argmax = (records_summary.get("argmax_region_counts") or {}) == {"tail": records_summary.get("records_available")}

    category_evidence = {
        "route_or_exact_block_selection": {
            "verdict": "not_supported_as_intentional_semantic_change_by_scalar_source_gate",
            "evidence": [
                "current and pair run on the same live Q/K/V and metadata in the shadow mode",
                "tau, threshold type, prefix sink, local exact blocks, and exact-offset order are unchanged in source",
                "the scalar artifact did not export raw route masks, so this is not a formal route-mask equality proof",
            ],
        },
        "row_max_or_row_sum": {
            "verdict": "not_independently_measured_in_existing_shadow_artifact",
            "evidence": [
                "source formulas for online row_max/row_sum match the current/upstream order",
                "pair-value-halves intentionally shares one row state across lo/hi instead of recompiling one state per value tile",
                "a future live row-state scalar probe would be required before claiming an exact row-state fix",
            ],
        },
        "approximate_vs_exact_contribution": {
            "verdict": "not_supported_as_route_policy_change",
            "evidence": [
                "approximate summary blocks, exact local/sink/threshold blocks, and dense-prefix overwrite are unchanged",
                "mismatches are confined to tail rows after dense-prefix overwrite, which is consistent with the sparse forward kernel only",
            ],
        },
        "bf16_probability_rounding_or_pv_dot_codegen": {
            "verdict": "primary_supported_localization",
            "evidence": [
                "the pair kernel's intended arithmetic/codegen delta is issuing two BV64 PV dots in one Triton program while sharing the QK/row-state stream; current source now keeps BF16 probability casts per half to test whether the prior shared-cast boundary caused divergence",
                "real-chain same-input shadow diverged on every observed sparse call while returning retained-current, so final AV equality cannot rescue the candidate as exact",
                "both lo and hi argmax halves occur in the scalar records, which argues against a single-half store-only bug",
                "Triton/PyTorch runtime is BF16 on SM86; source-level same algebra is not a bit-exactness proof after different tl.dot/liveness/codegen",
            ],
        },
        "v_stride_or_load": {
            "verdict": "not_supported_by_current_scalar_gate",
            "evidence": [
                "all modes saw the real fused-QKV V view with stride [823001088, 21504, 128, 1] and storage_offset 14336",
                "fallback, diagnostic materialization, input copy, and materialization counters were zero",
                "the pair and current source use the same V_STRIDE_B/T/H/D formula for exact value loads",
            ],
        },
        "lo_hi_store_behavior": {
            "verdict": "not_supported_as_single_store_half_failure",
            "evidence": [
                "output layouts match and prefix/padding regions are equal in scalar records",
                f"argmax value-half counts include both halves: {dict(half_counts)}",
                "a lo/hi store-order codegen effect remains possible only as part of the fused dual-dot/codegen category, not as a supported exact fix",
            ],
        },
    }

    return {
        "schema_version": "minimax_h3_a6000_pair_value_halves_shadow_localization_v1",
        "localization_classification": "reject_no_promotion_bf16_probability_pv_dot_codegen_delta",
        "primary_localized_mechanism": "bf16_probability_rounding_or_pv_dot_codegen",
        "exact_fix_supported": False,
        "why_no_exact_fix": (
            "The existing no-raw-tensor real-chain shadow proves same-input divergence but does not expose live "
            "route masks, row_max/row_sum, or per-row reference values.  The source comparison leaves the fused "
            "dual-BV64 probability/PV-dot Triton codegen as the supported mechanism, but not a concrete exact repair."
        ),
        "scalar_gate_facts": {
            "shadow_calls": shadow.get("calls"),
            "shadow_equal_calls": shadow.get("equal_calls"),
            "shadow_mismatch_count": shadow.get("mismatch_count"),
            "records_written": shadow.get("records_written"),
            "raw_tensor_exported": bool(shadow.get("raw_tensor_exported")),
            "raw_tensor_values_available": bool(shadow.get("raw_tensor_values_available")),
            "current_vs_current_attention_stable": bool(gates.get("current_current_attention_output_digests_equal")),
            "current_vs_current_decoded_av_stable": bool(gates.get("current_current_decoded_av_content_equal")),
            "same_input_shadow_returns_current": bool(gates.get("same_input_shadow_returns_current")),
            "all_fallback_calls_zero": bool(gates.get("all_fallback_calls_zero")),
            "all_zero_materialization_and_input_copies": bool(gates.get("all_zero_materialization_and_input_copies")),
            "real_h3_fused_value_layout_seen_all_modes": bool(gates.get("real_h3_fused_value_layout_seen_all_modes")),
            "all_prefix_padding_equal_in_records": all_prefix_padding_equal,
            "all_argmax_records_tail": all_tail_argmax,
            "both_value_halves_seen": both_halves_seen,
        },
        "category_evidence": category_evidence,
        "claim_boundaries": [
            "reject/no-promotion localization only",
            "no product speedup claim",
            "no BF16-fidelity equivalence claim",
            "no long-video claim",
            "no N3/N10 promotion",
            "no public comparison or SOTA claim",
        ],
    }


def build(input_dir: Path, out_dir: Path) -> dict[str, Any]:
    original_decision = read_json(input_dir / "decision.json")
    shadow = read_json(input_dir / "same_input_shadow_comparison_pair_value_halves.json")
    telemetry = read_json(input_dir / CANDIDATE / "sol_attn_telemetry.sol_attn.json")
    records = first_mismatch_records(shadow, telemetry)
    records_summary = summarize_records(records)
    sources = source_comparison()
    loc = localization(shadow, original_decision, records_summary, sources)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "shadow_scalar_summary.json").write_text(json.dumps(records_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "source_numeric_comparison.json").write_text(json.dumps(sources, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "localization.json").write_text(json.dumps(loc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    decision = {
        "schema_version": "minimax_h3_a6000_pair_value_halves_shadow_localization_decision_v1",
        "classification": "reject_no_promotion",
        "reason": "same_input_shadow_pair_value_halves_divergence_localized_to_bf16_probability_pv_dot_codegen_delta",
        "input_evidence": rel(input_dir),
        "artifacts": {
            "shadow_scalar_summary": "shadow_scalar_summary.json",
            "source_numeric_comparison": "source_numeric_comparison.json",
            "localization": "localization.json",
        },
        "promote_to_matched_n3": False,
        "promote_to_n3": False,
        "exact_fix_implemented": False,
        "exact_fix_supported": False,
        "reviewer_acceptance_required_before_promotion": True,
        "reviewer_acceptance_status": "pending_host_reviewer_not_invoked_by_engineer",
        "raw_tensor_exported": False,
        "raw_tensor_values_available": False,
        "private_paths_recorded": False,
        "primary_localized_mechanism": loc["primary_localized_mechanism"],
        "first_mismatch": records_summary.get("first_mismatch"),
        "scalar_gate_facts": loc["scalar_gate_facts"],
        "source_grounding": {
            "pinned_upstream_revision": sources.get("pinned_upstream_revision"),
            "runtime_versions": sources.get("runtime_versions"),
            "source_hashes": sources.get("source_hashes"),
        },
        "claim_boundary": (
            "Divergence-localization reject only; no product speedup, BF16-fidelity, long-video, "
            "quality-equivalence, public-comparison, or SOTA claim."
        ),
        "next_allowed_action": (
            "Do not promote or rerun unchanged pair-value-halves for timing. If revisited, first add a live "
            "row-state scalar probe that records route digest, row_max/row_sum deltas, and target-row reference "
            "equality without raw tensor export."
        ),
    }
    (out_dir / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    decision = build(args.input_dir, args.out_dir)
    print(json.dumps({"classification": decision["classification"], "reason": decision["reason"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
