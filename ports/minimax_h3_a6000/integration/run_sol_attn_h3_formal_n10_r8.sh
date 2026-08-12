#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Formal r8 Sol-Attn same-GPU N>=10 dense-vs-opt-in matched workload supervisor.
# Default is dry-run. Non-dry execution must be launched through a durable GPU
# lease/supervisor so the terminal evidence outlives the agent turn.
set -euo pipefail

ROOT=${ROOT:-${PWD}}
GPU_INDEX=${GPU_INDEX:-2}
EXPECTED_UUID=${EXPECTED_UUID:-GPU-5a6b7f13-4a03-c3c5-bb17-ea86b46d8aed}
N_PAIRS=${N_PAIRS:-10}
IMAGE=${IMAGE:-argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r8-sol-attn-overlay}
REQUIRED_IMAGE_VERSION_LABEL=${REQUIRED_IMAGE_VERSION_LABEL:-r8}
MODEL_ROOT=${MODEL_ROOT:-$ROOT/models/MiniMax-H3}
MATCHED_RETEST_SCRIPT=${MATCHED_RETEST_SCRIPT:-$ROOT/ports/minimax_h3_a6000/integration/run_sol_attn_h3_matched_retest_r8.sh}
OUT_DIR=${OUT_DIR:-$ROOT/technical_report/evidence/minimax_h3_desktop/sol_engine_port/sol_attn_h3_formal_n10_r8_n${N_PAIRS}_$(date -u +%Y%m%dT%H%M%SZ)}
if [[ "$OUT_DIR" != /* ]]; then
  OUT_DIR="$ROOT/$OUT_DIR"
fi
DRY_RUN=1

usage() {
  cat <<'EOF_USAGE'
Usage: run_sol_attn_h3_formal_n10_r8.sh [--dry-run|--execute]

Runs a formal same-physical-GPU N>=10 dense-vs-opt-in matched workload for the
r8 H3 Sol-Attn route by delegating each pair to the accepted r8 5-step H3 gate
and then writing a formal_n10_decision.json classification.

This is a formal N>=10 Sol-Attn matched-workload promotion gate only. It is not
BF16 fidelity certification, not a Turbo/DMD/DLO result, not a public release,
and not a human auditory/semantic quality judgment.

Non-dry execution requires ARGUS_ALLOW_GPU2_SOL_ATTN_H3_FORMAL_N10=1 and should
be launched with a durable single-GPU supervisor, for example:
  ARGUS_SKILL_PYTHON=<private-path> \
  ARGUS_ALLOW_GPU2_SOL_ATTN_H3_FORMAL_N10=1 \
  <ARGUS_SKILL_PYTHON> -m argus_skill.tools.gpu_lease run --detach \
    --owner argus-ir04-sol-attn-r8-formal-n10 --gpus <gpu-index> -- \
    bash ports/minimax_h3_a6000/integration/run_sol_attn_h3_formal_n10_r8.sh --execute
EOF_USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --execute) DRY_RUN=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

print_plan() {
  cat <<EOF_PLAN
DRY_RUN=1: no GPU, Docker, model load, network, or inference work performed.
root=$ROOT
gpu_index=$GPU_INDEX
expected_uuid=$EXPECTED_UUID
n_pairs=$N_PAIRS
image=$IMAGE
required_image_version_label=$REQUIRED_IMAGE_VERSION_LABEL
model_root=$MODEL_ROOT
matched_retest_script=$MATCHED_RETEST_SCRIPT
out_dir=$OUT_DIR
workload=1344x768,5.166667s,124frames,24FPS,32kHz stereo,prompt=t2va_example_1,steps=5,seed=0
comparison=dense_h3_backend_reference_vs_sol_attn_opt_in
formal_gate=N>=10_same_physical_gpu_matched_workload
accept_gate=completed_pairs>=10+same_expected_uuid+HTTP200+structural_AV+sparse_calls>0+fallback_calls=0+density/materialization telemetry+resource envelope+no quality proxy red flags+no slower pair+median improvement > max(3%,2*baseline_cv)
lane=formal_n10_matched_5step_sol_attn_opt_in_not_bf16_fidelity
operator_gate=ARGUS_ALLOW_GPU2_SOL_ATTN_H3_FORMAL_N10
opaque_identifier_policy=image/output hashes or digests are not used as classification evidence
EOF_PLAN
}

if [[ "$DRY_RUN" == "1" ]]; then
  print_plan
  exit 0
fi

if [[ "${ARGUS_ALLOW_GPU2_SOL_ATTN_H3_FORMAL_N10:-0}" != "1" ]]; then
  echo "ERROR: refusing non-dry formal N10 Sol-Attn run without ARGUS_ALLOW_GPU2_SOL_ATTN_H3_FORMAL_N10=1" >&2
  exit 11
fi

if [[ "$N_PAIRS" -lt 10 ]]; then
  mkdir -p "$OUT_DIR"
  python3 - "$OUT_DIR" "$N_PAIRS" <<'PY'
import json, sys, time
from pathlib import Path
root = Path(sys.argv[1]); n = int(sys.argv[2])
decision = {
    "schema_version": "minimax_h3_a6000_sol_attn_formal_n10_decision_v1",
    "formal_classification": "blocked_requested_pairs_below_formal_n10",
    "reason": "formal Sol-Attn promotion requires N_PAIRS>=10",
    "requested_pairs": n,
    "accepted": False,
    "rejected": False,
    "blocked": True,
    "not_bf16_fidelity": True,
    "lane": "formal_n10_matched_5step_sol_attn_opt_in_not_bf16_fidelity",
    "generated_at_unix": time.time(),
}
(root / "formal_n10_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
PY
  exit 12
fi

mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/formal_n10_supervisor_stdout.log") 2>&1
formal_status_json="$OUT_DIR/formal_n10_supervisor_status.json"
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
cat > "$formal_status_json" <<EOF_STATUS
{
  "schema_version": "minimax_h3_a6000_sol_attn_formal_n10_supervisor_status_v1",
  "status": "running",
  "started_utc": "$started_utc",
  "pid": $$,
  "gpu_index": "$GPU_INDEX",
  "expected_uuid": "$EXPECTED_UUID",
  "n_pairs": $N_PAIRS,
  "out_dir": "$OUT_DIR"
}
EOF_STATUS

finish_formal_status() {
  local rc=$?
  local finished_utc
  finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  python3 - "$formal_status_json" "$rc" "$finished_utc" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1]); rc = int(sys.argv[2]); finished = sys.argv[3]
try:
    data = json.loads(path.read_text())
except Exception:
    data = {"schema_version": "minimax_h3_a6000_sol_attn_formal_n10_supervisor_status_v1"}
data["status"] = "complete" if rc == 0 else "failed"
data["return_code"] = rc
data["finished_utc"] = finished
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
}
trap finish_formal_status EXIT

cat > "$OUT_DIR/formal_n10_commands_env.txt" <<EOF_ENV
schema=minimax_h3_a6000_sol_attn_formal_n10_commands_env_v1
started_utc=$started_utc
root=$ROOT
gpu_index=$GPU_INDEX
expected_uuid=$EXPECTED_UUID
n_pairs=$N_PAIRS
image=$IMAGE
required_image_version_label=$REQUIRED_IMAGE_VERSION_LABEL
model_root=$MODEL_ROOT
matched_retest_script=$MATCHED_RETEST_SCRIPT
out_dir=$OUT_DIR
operator_gate=ARGUS_ALLOW_GPU2_SOL_ATTN_H3_FORMAL_N10
delegated_operator_gate=ARGUS_ALLOW_GPU2_SOL_ATTN_H3_MATCHED_RETEST
pair_operator_gate=ARGUS_ALLOW_GPU2_SOL_ATTN_H3_5STEP
formal_n10=yes
lane=formal_n10_matched_5step_sol_attn_opt_in_not_bf16_fidelity
opaque_identifier_policy=opaque image/output identifiers are omitted from classification evidence
EOF_ENV

set +e
ARGUS_ALLOW_GPU2_SOL_ATTN_H3_MATCHED_RETEST=1 \
ROOT="$ROOT" GPU_INDEX="$GPU_INDEX" EXPECTED_UUID="$EXPECTED_UUID" N_PAIRS="$N_PAIRS" \
IMAGE="$IMAGE" REQUIRED_IMAGE_VERSION_LABEL="$REQUIRED_IMAGE_VERSION_LABEL" MODEL_ROOT="$MODEL_ROOT" OUT_DIR="$OUT_DIR" \
bash "$MATCHED_RETEST_SCRIPT" --execute
matched_rc=$?
set -e

echo "$matched_rc" > "$OUT_DIR/matched_retest_exit_code.txt"

python3 - "$OUT_DIR" "$N_PAIRS" "$EXPECTED_UUID" "$matched_rc" <<'PY'
from __future__ import annotations
import json, sys, time
from pathlib import Path

root = Path(sys.argv[1])
requested_pairs = int(sys.argv[2])
expected_uuid = sys.argv[3]
matched_rc = int(sys.argv[4])


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def read_kv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out

raw = load_json(root / "decision.json") or {}
gpu_hygiene = load_json(root / "gpu_hygiene_preflight.json") or {}
commands_env = read_kv(root / "commands_env.txt")
selected_uuid = None
selected_gpu = gpu_hygiene.get("selected_gpu") if isinstance(gpu_hygiene, dict) else None
if isinstance(selected_gpu, dict):
    selected_uuid = selected_gpu.get("uuid")
raw_gates = raw.get("gates") if isinstance(raw.get("gates"), dict) else {}
completed_pairs = int(raw_gates.get("completed_pairs") or 0)
failed_gates = list(raw.get("failed_gates") or []) if isinstance(raw.get("failed_gates"), list) else []
raw_classification = raw.get("classification")
same_expected_gpu = bool(expected_uuid and selected_uuid == expected_uuid)
formal_pair_count_ok = requested_pairs >= 10 and completed_pairs >= 10
raw_pass = raw_classification == "proceed_to_formal_n10_candidate" and not failed_gates

if not raw:
    formal_classification = "inconclusive_missing_matched_retest_decision"
    reason = "delegated matched-retest supervisor did not produce decision.json"
    accepted = rejected = False
    blocked = True
elif not same_expected_gpu:
    formal_classification = "blocked_not_same_baseline_physical_gpu"
    reason = "formal speed classification requires the baseline-certified physical GPU UUID"
    accepted = rejected = False
    blocked = True
elif matched_rc != 0 and not formal_pair_count_ok:
    formal_classification = "inconclusive_incomplete_formal_n10_run"
    reason = "delegated matched-retest command exited nonzero before N>=10 complete pairs"
    accepted = rejected = False
    blocked = False
elif not formal_pair_count_ok:
    formal_classification = "rejected_formal_n10_incomplete_pair_count"
    reason = "terminal artifacts do not contain at least 10 completed matched pairs"
    accepted = False
    rejected = True
    blocked = False
elif raw_pass:
    formal_classification = "accepted_formal_n10_same_gpu_sol_attn_speed_candidate"
    reason = "N>=10 same-GPU matched workload passed sparse-runtime, structural AV, resource, quality-proxy, and timing gates"
    accepted = True
    rejected = False
    blocked = False
elif any(name in failed_gates for name in ("all_http_200", "all_structural_av_valid", "all_sparse_calls_positive", "all_fallback_calls_zero", "complete_density_and_materialization_telemetry", "no_quality_proxy_red_flags")):
    formal_classification = "rejected_formal_n10_correctness_sparse_or_quality_gate_failed"
    reason = "N>=10 run reached terminal artifacts but a correctness, sparse-runtime, telemetry, or quality-proxy gate failed"
    accepted = False
    rejected = True
    blocked = False
elif any(name in failed_gates for name in ("resource_envelope_comparable_to_prior_r8",)):
    formal_classification = "rejected_formal_n10_resource_gate_failed"
    reason = "N>=10 run reached terminal artifacts but resource envelope was outside the accepted prior-r8 bounds"
    accepted = False
    rejected = True
    blocked = False
elif any(name in failed_gates for name in ("no_pair_slower", "median_improvement_exceeds_threshold")):
    formal_classification = "rejected_formal_n10_timing_gate_failed"
    reason = "N>=10 run preserved terminal artifacts but the paired runtime benefit was not above the declared noise/threshold gate"
    accepted = False
    rejected = True
    blocked = False
else:
    formal_classification = "rejected_formal_n10_unclassified_gate_failure"
    reason = "N>=10 run reached terminal artifacts but did not satisfy the formal acceptance gates"
    accepted = False
    rejected = True
    blocked = False

decision = {
    "schema_version": "minimax_h3_a6000_sol_attn_formal_n10_decision_v1",
    "formal_classification": formal_classification,
    "reason": reason,
    "accepted": accepted,
    "rejected": rejected,
    "blocked": blocked,
    "review_required": True,
    "matched_retest_return_code": matched_rc,
    "raw_matched_retest_classification": raw_classification,
    "raw_matched_retest_decision": "decision.json" if raw else None,
    "requested_pairs": requested_pairs,
    "completed_pairs": completed_pairs,
    "formal_pair_count_ok": formal_pair_count_ok,
    "same_baseline_physical_gpu_required": True,
    "same_baseline_physical_gpu_evidence": {
        "gpu_index": commands_env.get("gpu_index"),
        "expected_uuid": expected_uuid,
        "selected_uuid": selected_uuid,
        "same_expected_gpu": same_expected_gpu,
    },
    "failed_gates": failed_gates,
    "gates": raw_gates,
    "median_http_time_improvement_pct": raw.get("median_http_time_improvement_pct"),
    "timing_threshold_pct": raw.get("timing_threshold_pct"),
    "baseline_warm_cv_pct": raw.get("baseline_warm_cv_pct"),
    "resource_peak_summary": raw.get("resource_peak_summary"),
    "resource_thresholds": raw.get("resource_thresholds"),
    "quality_proxy_red_flags": raw.get("quality_proxy_red_flags"),
    "pairs": raw.get("pairs"),
    "lane": "formal_n10_matched_5step_sol_attn_opt_in_not_bf16_fidelity",
    "not_bf16_fidelity": True,
    "not_turbo_dlo_dmd": True,
    "not_human_audio_or_semantic_quality_certification": True,
    "not_public_release": True,
    "opaque_integrity_policy": {
        "image_identifiers": "readable labels/tags only; digests not used as classification evidence",
        "output_identifiers": "sha256/output identity not used as classification evidence",
    },
    "generated_at_unix": time.time(),
}
(root / "formal_n10_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
report = f"""# Sol-Attn r8 formal N>=10 matched workload report\n\nStatus: `{formal_classification}`.\n\nThis is a formal N={requested_pairs} same-physical-GPU 5-step dense-vs-opt-in Sol-Attn matched-workload gate. It is not BF16 fidelity certification, not Turbo/DLO/DMD evidence, not a human auditory/semantic quality judgment, and not a public release.\n\n- Decision reason: {reason}\n- Completed pairs: {completed_pairs}/{requested_pairs}\n- Same baseline physical GPU: {same_expected_gpu} (expected `{expected_uuid}`, observed `{selected_uuid}`)\n- Raw matched-retest classification: `{raw_classification}`\n- Median HTTP-time improvement: {raw.get('median_http_time_improvement_pct') if raw else 'pending'}%\n- Timing threshold: > {raw.get('timing_threshold_pct') if raw else 'pending'}%\n- Failed gates: {', '.join(failed_gates) if failed_gates else 'none'}\n- Review required: true\n\nRaw artifacts are in this directory and its per-pair subdirectories. Opaque image/output identifiers are not used as classification evidence.\n"""
(root / "FORMAL_N10_RUN_REPORT.md").write_text(report)
print(json.dumps({"formal_classification": formal_classification, "decision": str(root / "formal_n10_decision.json"), "matched_rc": matched_rc}, sort_keys=True))
PY

exit "$matched_rc"
