#!/usr/bin/env bash
# Matched N=1 long-lane gate: retained r10 VAE serial spatial tiles vs bounded VAE tile-batch cap.
# Default is dry-run. Non-dry execution must be launched under a fresh single-A6000 Argus lease.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd -P)}
cd "$ROOT"

MANIFEST=${MANIFEST:-$ROOT/benchmark_contract/v1/lane-manifests/final-av-30s-1344x768-24fps-v1.json}
CONTRACT=${CONTRACT:-$ROOT/benchmark_contract/v1/contract.json}
PROMPT_FILE=${PROMPT_FILE:-$ROOT/benchmark_contract/v1/prompts/orbital-continuity-long-v1.txt}
MODEL_DIR=${MODEL_DIR:-$ROOT/models/MiniMax-H3-Turbo-Merged/FL2VA}
R20_IMAGE=${R20_IMAGE:-argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r20-vae-bounded-tile-batching}
DOCKER_HOST=${DOCKER_HOST:-unix:///tmp/minih3-20260814T135313Z.sock}
GPU_INDEX=${GPU_INDEX:-0}
EXPECTED_UUID=${EXPECTED_UUID:-}
SOL_PROFILE=${SOL_PROFILE:-r10_adaptive_tau1_5_step3_diag}
VAE_TILE_BATCH_SIZE=${VAE_TILE_BATCH_SIZE:-0}
REFERENCE_MODE=${REFERENCE_MODE:-r10_adaptive_tau1_5_step3_diag_vae_serial}
CANDIDATE_MODE=${CANDIDATE_MODE:-r10_adaptive_tau1_5_step3_diag_vae_tile_batch_cap_${VAE_TILE_BATCH_SIZE}}
RUN_ID=${RUN_ID:-final-av-30s-r10-vae-tile-batch-cap-${VAE_TILE_BATCH_SIZE}-n1-$(date -u +%Y%m%dT%H%M%SZ)}
OUT_DIR=${OUT_DIR:-$ROOT/technical_report/evidence/minimax_h3_desktop/long_video/$RUN_ID}
MIN_DELTA_PCT=${MIN_DELTA_PCT:-1.0}
DRY_RUN=1

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_final_av_30s_r10_vae_bounded_tile_batching_n1.sh --dry-run
  ARGUS_ALLOW_FINAL_AV_30S_R10_VAE_BOUNDED_TILE_BATCHING_N1=1 I_ACCEPT_MINIMAX_H3_LICENSE=YES \
    DOCKER_HOST=unix:///tmp/minih3-20260814T135313Z.sock GPU_INDEX=<idle-a6000> VAE_TILE_BATCH_SIZE=<selected-cap> \
    bash scripts/run_final_av_30s_r10_vae_bounded_tile_batching_n1.sh --execute

Runs a matched 30-second extension N=1 gate with exactly one principal variable:
MINIMAX_H3_A6000_VIDEO_VAE_TILE_BATCH_SIZE=0 versus the selected positive cap.
Both modes keep retained r10 guarded-adaptive Sol-Attn variables, Turbo 8-step,
same prompt/contract/timing boundary, split component profiler, same R20 runtime,
one visible A6000, no Cache-DiT, and no BF16/exact/lossless claim.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --execute) DRY_RUN=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 64 ;;
  esac
done

if [[ "$SOL_PROFILE" != "r10_adaptive_tau1_5_step3_diag" ]]; then
  echo "ERROR: this bounded VAE gate fixes SOL_PROFILE=r10_adaptive_tau1_5_step3_diag" >&2
  exit 2
fi
if ! [[ "$VAE_TILE_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: VAE_TILE_BATCH_SIZE must be a positive selected cap for the bounded candidate" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
OUT_DIR=$(cd "$OUT_DIR" && pwd -P)

if (( DRY_RUN )); then
  python3 tools/validate_benchmark_record.py --json "$CONTRACT" "$MANIFEST" benchmark_contract/v1/normalized-records/*.json > "$OUT_DIR/contract_validation.json"
  ATTENTION_BACKEND=H3_A6000_SOL_ATTN MODE_LABEL="$REFERENCE_MODE" SOL_ATTN_PROFILE="$SOL_PROFILE" RUNTIME_IMAGE="$R20_IMAGE" PROFILE_SPLIT=1 VAE_SPATIAL_TILE_BATCHING=0 VAE_TILE_BATCH_SIZE=0 \
    bash scripts/run_final_av_30s_extension_n1.sh --dry-run > "$OUT_DIR/${REFERENCE_MODE}_dry_run.txt"
  ATTENTION_BACKEND=H3_A6000_SOL_ATTN MODE_LABEL="$CANDIDATE_MODE" SOL_ATTN_PROFILE="$SOL_PROFILE" RUNTIME_IMAGE="$R20_IMAGE" PROFILE_SPLIT=1 VAE_SPATIAL_TILE_BATCHING=0 VAE_TILE_BATCH_SIZE="$VAE_TILE_BATCH_SIZE" \
    bash scripts/run_final_av_30s_extension_n1.sh --dry-run > "$OUT_DIR/${CANDIDATE_MODE}_dry_run.txt"
  cat <<EOF
[DRY-RUN] retained r10 video VAE bounded tile-batch final-AV 30s N=1 gate
Out dir: $OUT_DIR
Image: $R20_IMAGE
Docker host: $DOCKER_HOST
GPU index: $GPU_INDEX
Reference: $REFERENCE_MODE with VAE_TILE_BATCH_SIZE=0
Candidate: $CANDIDATE_MODE with VAE_TILE_BATCH_SIZE=$VAE_TILE_BATCH_SIZE
Fixed Sol-Attn profile: $SOL_PROFILE
Split profiler: enabled for both lanes
Promotion threshold: N=1 warm E2E delta >= ${MIN_DELTA_PCT}% plus final-AV/proxy/resource/Sol-Attn invariant gates; no formal speedup claim.
EOF
  exit 0
fi

if [[ "${ARGUS_ALLOW_FINAL_AV_30S_R10_VAE_BOUNDED_TILE_BATCHING_N1:-}" != "1" ]]; then
  echo "ERROR: set ARGUS_ALLOW_FINAL_AV_30S_R10_VAE_BOUNDED_TILE_BATCHING_N1=1 for this matched VAE long-lane gate" >&2
  exit 2
fi
if [[ "${I_ACCEPT_MINIMAX_H3_LICENSE:-}" != "YES" ]]; then
  echo "ERROR: read/accept the MiniMax-H3 license for local inference, then set I_ACCEPT_MINIMAX_H3_LICENSE=YES" >&2
  exit 2
fi

python3 tools/validate_benchmark_record.py --json "$CONTRACT" "$MANIFEST" benchmark_contract/v1/normalized-records/*.json > "$OUT_DIR/contract_validation.json"
cp "$MANIFEST" "$OUT_DIR/lane_manifest.json"
cp "$CONTRACT" "$OUT_DIR/contract.json"
cp "$PROMPT_FILE" "$OUT_DIR/base_prompt.txt"
cat > "$OUT_DIR/route_gate_scope.json" <<JSON
{
  "schema_version": "minimax-h3-final-av-30s-r10-vae-bounded-tile-batching-n1-scope-v1",
  "created_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "principal_variable": "MINIMAX_H3_A6000_VIDEO_VAE_TILE_BATCH_SIZE=0 versus =$VAE_TILE_BATCH_SIZE",
  "reference_mode": "$REFERENCE_MODE",
  "candidate_mode": "$CANDIDATE_MODE",
  "fixed_variables": {
    "runtime_image": "$R20_IMAGE",
    "attention_backend": "H3_A6000_SOL_ATTN",
    "sol_attn_profile": "$SOL_PROFILE",
    "tau": 1.5,
    "adaptive_step_min": 3,
    "cache": false,
    "pair_value_halves": false,
    "request_quality": "lossless",
    "profile_split": true,
    "generation_mode": "extension",
    "native_context_supported": false,
    "steps": 8,
    "chunks": 6,
    "same_prompt_workload_contract": true
  },
  "claim_boundary": "N=1 matched 30-second final-AV extension-lane VAE practical-approximate route gate only; no formal speedup, BF16 fidelity, exact/lossless, native long context, product quality, public comparison, SOTA, or human-quality claim."
}
JSON

run_one_mode() {
  local mode=$1
  local cap=$2
  local evidence=$3
  mkdir -p "$evidence"
  ARGUS_ALLOW_FINAL_AV_30S_EXTENSION_N1=1 \
  I_ACCEPT_MINIMAX_H3_LICENSE=YES \
  ROOT="$ROOT" MANIFEST="$MANIFEST" CONTRACT="$CONTRACT" PROMPT_FILE="$PROMPT_FILE" MODEL_DIR="$MODEL_DIR" \
  RUNTIME_IMAGE="$R20_IMAGE" DOCKER_HOST="$DOCKER_HOST" GPU_INDEX="$GPU_INDEX" EXPECTED_UUID="$EXPECTED_UUID" \
  EVIDENCE_DIR="$evidence" ATTENTION_BACKEND=H3_A6000_SOL_ATTN MODE_LABEL="$mode" SOL_ATTN_PROFILE="$SOL_PROFILE" SOL_ATTN_R8_OPT_IN=0 \
  REQUEST_QUALITY=lossless SERVER_CACHE_BACKEND=none ENABLE_CACHE_DIT_SUMMARY=0 PROFILE_SPLIT=1 VAE_SPATIAL_TILE_BATCHING=0 VAE_TILE_BATCH_SIZE="$cap" \
  bash scripts/run_final_av_30s_extension_n1.sh --execute
}

run_one_mode "$REFERENCE_MODE" 0 "$OUT_DIR/$REFERENCE_MODE"
run_one_mode "$CANDIDATE_MODE" "$VAE_TILE_BATCH_SIZE" "$OUT_DIR/$CANDIDATE_MODE"

python3 tools/final_av_30s_extension_runner.py finalize-r10-vae-spatial-tile-batching-matched \
  --out-dir "$OUT_DIR" \
  --reference-evidence "$OUT_DIR/$REFERENCE_MODE" \
  --candidate-evidence "$OUT_DIR/$CANDIDATE_MODE" \
  --min-delta-pct "$MIN_DELTA_PCT" \
  --candidate-vae-tile-batch-size "$VAE_TILE_BATCH_SIZE" > "$OUT_DIR/finalize_matched_stdout.log" 2> "$OUT_DIR/finalize_matched_stderr.log"

python3 tools/validate_benchmark_record.py --json "$OUT_DIR/$REFERENCE_MODE/benchmark_record.json" > "$OUT_DIR/reference_benchmark_record_validation.json"
python3 tools/validate_benchmark_record.py --json "$OUT_DIR/$CANDIDATE_MODE/benchmark_record.json" > "$OUT_DIR/candidate_benchmark_record_validation.json"

echo "final-av-30s-r10-vae-bounded-tile-batching-n1=COMPLETE evidence=$OUT_DIR"
