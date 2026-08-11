#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# External GPU2 Sol-Attn H3 diagnostic gate only.
# Default is --dry-run: no GPU execution, Docker execution, nvidia-smi, CUDA,
# network access, downloads, model loading, inference, cache enablement, or publication.
set -euo pipefail

ROOT=${ROOT:-${PWD}}
GPU_INDEX=${GPU_INDEX:-2}
EXPECTED_UUID=${EXPECTED_UUID:-}
IMAGE=${IMAGE:-argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r7-sol-attn-overlay}
REQUIRED_IMAGE_VERSION_LABEL=${REQUIRED_IMAGE_VERSION_LABEL:-r7}
MODEL_ROOT=${MODEL_ROOT:-$ROOT/models/MiniMax-H3}
PROMPT_FILE=${PROMPT_FILE:-$ROOT/technical_report/evidence/minimax_h3_desktop/baseline_a6000/t2va_example_1.prompt.txt}
OUT_DIR=${OUT_DIR:-$ROOT/technical_report/evidence/minimax_h3_desktop/sol_engine_port/sol_attn_h3_gpu2_5step_r7_$(date -u +%Y%m%dT%H%M%SZ)}
PORT=${PORT:-8000}
DRY_RUN=1

usage() {
  cat <<'EOF_USAGE'
Usage: run_gpu2_sol_attn_h3_5step_diagnostic.sh [--dry-run|--execute]

Dry-run is the default and prints the exact external GPU2 plan without GPU
execution, Docker execution, CUDA, nvidia-smi, network access, downloads, model
loading, inference, cache enablement, or publication. Non-dry execution requires
ARGUS_ALLOW_GPU2_SOL_ATTN_H3_5STEP=1 plus readable fresh r7 image
version/base/title label checks, and is outside CPU/static stages.

The diagnostic compares two 5-step runs through the opt-in H3_A6000_SOL_ATTN
backend only:
  1. dense_h3_backend_reference: Sol-Attn env off -> dense fallback;
  2. sol_attn_opt_in: overlay/triton/Sol-Attn env on, Sol-Attn cache off,
     r7 diagnostic materialization on with explicit copy telemetry.
It is a diagnostic_5_step_sol_attn_metadata_gate_not_fidelity_or_performance_claim.
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
DRY_RUN=1 (default): no GPU execution, Docker execution, network access, downloads, model loading, inference, cache enablement, or publication performed.
root=$ROOT
gpu_index=$GPU_INDEX
expected_uuid=${EXPECTED_UUID:-<not-set>}
image=$IMAGE
required_image_label=org.opencontainers.image.version=$REQUIRED_IMAGE_VERSION_LABEL
identity_guard=non-dry verifies readable image tag plus version/base/title labels for the fresh r7 overlay; opaque image identifiers are omitted and are not proof
external_r7_build_command=EVIDENCE_DIR=technical_report/evidence/minimax_h3_desktop/sol_engine_port/r7_overlay_image bash ports/minimax_h3_a6000/integration/r7/build_r7_overlay_image.sh
model_root=$MODEL_ROOT
prompt_file=$PROMPT_FILE
out_dir=$OUT_DIR
steps=5
seed=0
backend=H3_A6000_SOL_ATTN
network=none
one_visible_gpu_guard=container asserts torch.cuda.device_count()==1 and SM86 A6000
sol_attn_cache=off
sol_attn_diagnostic_materialize=on_for_r7_only
sol_attn_materialize_max_bytes=67108864
exact_wrappers=off
telemetry=/evidence/sol_attn/sol_attn_telemetry.sol_attn.json
blocker_if_metadata_missing=missing_h3_hook_metadata:<missing_attention_metadata|missing_packed_video_layout|missing_valid_kv_length_metadata|missing_step_layer_metadata|invalid_packed_video_layout>
classification=diagnostic_5_step_sol_attn_metadata_gate_not_fidelity_or_performance_claim

Would execute only with:
  ARGUS_ALLOW_GPU2_SOL_ATTN_H3_5STEP=1 bash ports/minimax_h3_a6000/integration/run_gpu2_sol_attn_h3_5step_diagnostic.sh --execute
EOF_PLAN
}

if [[ "$DRY_RUN" == "1" ]]; then
  print_plan
  exit 0
fi

if [[ "${ARGUS_ALLOW_GPU2_SOL_ATTN_H3_5STEP:-0}" != "1" ]]; then
  echo "ERROR: refusing non-dry Sol-Attn GPU2 diagnostic without ARGUS_ALLOW_GPU2_SOL_ATTN_H3_5STEP=1" >&2
  exit 11
fi

verify_r7_readable_image_provenance() {
  local version_label base_label title_label
  version_label=$(docker image inspect --format '{{ index .Config.Labels "org.opencontainers.image.version" }}' "$IMAGE")
  base_label=$(docker image inspect --format '{{ index .Config.Labels "org.opencontainers.image.base.name" }}' "$IMAGE")
  title_label=$(docker image inspect --format '{{ index .Config.Labels "org.opencontainers.image.title" }}' "$IMAGE")
  if [[ "$version_label" != "$REQUIRED_IMAGE_VERSION_LABEL" ]]; then
    echo "ERROR: image version label mismatch for $IMAGE: got $version_label expected $REQUIRED_IMAGE_VERSION_LABEL" >&2
    exit 15
  fi
  if [[ "$base_label" != "argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r2" ]]; then
    echo "ERROR: image base label mismatch for $IMAGE: got $base_label" >&2
    exit 15
  fi
  if [[ "$title_label" != *"r7 Sol-Attn"* ]]; then
    echo "ERROR: image title label does not identify the fresh Sol-Attn overlay: $title_label" >&2
    exit 15
  fi
  cat <<EOF_IDENTITY
image=$IMAGE
required_image_version_label=$REQUIRED_IMAGE_VERSION_LABEL
actual_image_version_label=$version_label
actual_image_base_label=$base_label
actual_image_title_label=$title_label
opaque_image_identifier_policy=omitted_not_evidence
EOF_IDENTITY
}

mkdir -p "$OUT_DIR"
verify_r7_readable_image_provenance > "$OUT_DIR/r7_image_identity.env"
cp "$PROMPT_FILE" "$OUT_DIR/prompt.txt"
cat > "$OUT_DIR/workload.env" <<EOF_WORKLOAD
image=$IMAGE
gpu_index=$GPU_INDEX
model_root=$MODEL_ROOT
prompt_file=$PROMPT_FILE
steps=5
seed=0
width=1344
height=768
fps=24
duration=5.166667
attention_backend=H3_A6000_SOL_ATTN
sol_attn_opt_in=diagnostic_only_not_fidelity
sol_attn_cache=off
sol_attn_diagnostic_materialize=on_for_r7_only
sol_attn_materialize_max_bytes=67108864
network=none
EOF_WORKLOAD

if [[ -n "$EXPECTED_UUID" ]]; then
  actual_uuid=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid --format=csv,noheader)
  if [[ "$actual_uuid" != "$EXPECTED_UUID" ]]; then
    echo "ERROR: GPU $GPU_INDEX uuid mismatch: got $actual_uuid expected $EXPECTED_UUID" >&2
    exit 12
  fi
fi

run_one() {
  local mode=$1
  shift
  local evidence=/evidence/$mode
  mkdir -p "$OUT_DIR/$mode"
  chmod 0777 "$OUT_DIR" "$OUT_DIR/$mode"
  docker run --rm \
    --gpus "device=$GPU_INDEX" \
    --ipc=host \
    --ulimit memlock=-1 \
    --network none \
    --cap-drop=ALL \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_OMNI_VIDEO_SYNC_TIMEOUT=3600 \
    -e VLLM_DISABLE_COMPILE_CACHE=1 \
    -e HF_HOME=/tmp/minimax_h3_sol_attn_no_cache/hf \
    -e TRANSFORMERS_CACHE=/tmp/minimax_h3_sol_attn_no_cache/hf \
    -e TORCHINDUCTOR_CACHE_DIR=/tmp/minimax_h3_sol_attn_no_cache/torchinductor \
    -e TRITON_CACHE_DIR=/tmp/minimax_h3_sol_attn_no_cache/triton \
    "$@" \
    -v "$MODEL_ROOT":/models/MiniMax-H3:ro \
    -v "$OUT_DIR":/evidence:rw \
    "$IMAGE" bash -lc "
set -euo pipefail
mkdir -p '$evidence'
python3 - <<'PY' > '$evidence/gpu_preflight.json'
import json, torch
assert torch.cuda.is_available(), 'cuda unavailable'
assert torch.cuda.device_count() == 1, f'expected one visible GPU, got {torch.cuda.device_count()}'
props = torch.cuda.get_device_properties(0)
cap = torch.cuda.get_device_capability(0)
assert cap == (8, 6) and 'a6000' in props.name.lower(), (props.name, cap)
print(json.dumps({'name': props.name, 'capability': cap, 'visible_count': 1}, sort_keys=True))
PY
vllm-omni serve /models/MiniMax-H3/FL2VA \
  --omni --trust-remote-code --host 127.0.0.1 --port '$PORT' --task-type fl2va \
  --num-gpus 1 --tensor-parallel-size 1 --text-encoder-tp-size 1 --usp 1 --ring 1 \
  --vae-patch-parallel-size 1 --vae-parallel-mode tile --vae-use-tiling \
  --enable-distributed-layerwise-offload --dlo-no-use-allgather --dlo-resident-layers 12 \
  --enforce-eager --diffusion-attention-backend H3_A6000_SOL_ATTN \
  > '$evidence/server.log' 2>&1 &
server_pid=\$!
trap 'kill \$server_pid 2>/dev/null || true' EXIT
for i in \$(seq 1 540); do
  if grep -q 'Application startup complete\.' '$evidence/server.log' && curl --fail --silent http://127.0.0.1:'$PORT'/health >/dev/null; then
    break
  fi
  kill -0 \$server_pid 2>/dev/null || { tail -300 '$evidence/server.log'; exit 61; }
  test \$i -lt 540 || exit 62
  sleep 5
done
curl --fail-with-body --silent --show-error --max-time 3000 \
  --dump-header '$evidence/request_headers.txt' \
  --write-out 'http_code=%{http_code}\ntime_total_s=%{time_total}\nsize_download=%{size_download}\n' \
  -X POST http://127.0.0.1:'$PORT'/v1/videos/sync \
  -F 'prompt=</evidence/prompt.txt' \
  -F 'width=1344' -F 'height=768' -F 'aspect_ratio=16:9' -F 'fps=24' \
  -F 'num_inference_steps=5' -F 'flow_shift=12' -F 'seed=0' -F 'quality=lossless' \
  -F 'extra_params={\"task\":\"t2va\",\"duration\":5.166667,\"audio_flow_shift\":3.0}' \
  -o '$evidence/output.mp4' > '$evidence/http_metrics.txt'
python3 - <<'PY'
import hashlib, json, pathlib
import av
mode = '$mode'
p = pathlib.Path('$evidence/output.mp4')
assert p.stat().st_size > 1024
c = av.open(str(p))
video = [s for s in c.streams if s.type == 'video']
audio = [s for s in c.streams if s.type == 'audio']
assert video and audio
vf = af = samples = 0
for frame in c.decode():
    if isinstance(frame, av.VideoFrame):
        vf += 1
    elif isinstance(frame, av.AudioFrame):
        af += 1
        samples += frame.samples
v = video[0]; a = audio[0]
record = {
    'mode': mode,
    'steps': 5,
    'seed': 0,
    'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
    'bytes': p.stat().st_size,
    'video_present': True,
    'audio_present': True,
    'width': v.codec_context.width,
    'height': v.codec_context.height,
    'average_rate': str(v.average_rate),
    'decoded_video_frames': vf,
    'audio_sample_rate': a.codec_context.sample_rate,
    'audio_channels': a.codec_context.channels,
    'decoded_audio_frames': af,
    'decoded_audio_samples': samples,
}
assert record['width'] == 1344 and record['height'] == 768
assert record['audio_sample_rate'] == 32000 and record['audio_channels'] == 2
assert vf > 0 and af > 0
pathlib.Path('$evidence/av_validation.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
pathlib.Path('$evidence/output.sha256').write_text(record['sha256'] + '  output.mp4\n')
PY
kill \$server_pid || true
wait \$server_pid || true
"
}

# Dense reference through the same opt-in backend with Sol-Attn disabled. This
# isolates Sol-Attn routing/fallback behavior; it is not the formal CUDNN DLO denominator.
run_one dense_h3_backend_reference \
  -e MINIMAX_H3_A6000_ENABLE_OVERLAY=0 \
  -e MINIMAX_H3_A6000_ENABLE_TRITON_CANDIDATES=0 \
  -e MINIMAX_H3_A6000_ENABLE_SOL_ATTN=0 \
  -e MINIMAX_H3_A6000_SOL_ATTN_CACHE=0 \
  -e MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_MATERIALIZE=0 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ADALN=0 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ROPE=0 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_SWIGLU=0 \
  -e MINIMAX_H3_A6000_ENABLE_TELEMETRY=0

# Sol-Attn opt-in diagnostic: default-off exact wrappers remain off, cache is off,
# telemetry must say whether H3 metadata was accepted or why it failed closed.
run_one sol_attn \
  -e MINIMAX_H3_A6000_ENABLE_OVERLAY=1 \
  -e MINIMAX_H3_A6000_ENABLE_TRITON_CANDIDATES=1 \
  -e MINIMAX_H3_A6000_ENABLE_SOL_ATTN=1 \
  -e MINIMAX_H3_A6000_SOL_ATTN_CACHE=0 \
  -e MINIMAX_H3_A6000_SOL_ATTN_STRICT=0 \
  -e MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_MATERIALIZE=1 \
  -e MINIMAX_H3_A6000_SOL_ATTN_MATERIALIZE_MAX_BYTES=67108864 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ADALN=0 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ROPE=0 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_SWIGLU=0 \
  -e MINIMAX_H3_A6000_ENABLE_TELEMETRY=1 \
  -e MINIMAX_H3_A6000_TELEMETRY_ATEXIT=1 \
  -e MINIMAX_H3_A6000_SOL_ATTN_TELEMETRY_JSON=/evidence/sol_attn/sol_attn_telemetry

python3 - "$OUT_DIR" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
dense = json.loads((root / 'dense_h3_backend_reference' / 'av_validation.json').read_text())
sol = json.loads((root / 'sol_attn' / 'av_validation.json').read_text())
telemetry_path = root / 'sol_attn' / 'sol_attn_telemetry.sol_attn.json'
assert dense['video_present'] and dense['audio_present']
assert sol['video_present'] and sol['audio_present']
if not telemetry_path.exists():
    status = {'status': 'blocked', 'reason': 'missing_sol_attn_telemetry_file', 'telemetry_path': str(telemetry_path)}
else:
    telemetry = json.loads(telemetry_path.read_text())
    sparse_candidates = int(telemetry.get('sparse_candidate_calls', 0))
    decline_reasons = telemetry.get('decline_reasons', {})
    if sparse_candidates > 0:
        status = {
            'status': 'metadata_path_accepted_sparse_candidate_attempted',
            'not_fidelity_or_performance_claim': True,
            'telemetry': telemetry,
        }
    elif decline_reasons:
        status = {
            'status': 'fail_closed_dense_fallback',
            'not_fidelity_or_performance_claim': True,
            'decline_reasons': decline_reasons,
            'telemetry': telemetry,
        }
    else:
        status = {'status': 'blocked', 'reason': 'no_sparse_attempt_and_no_decline_reason', 'telemetry': telemetry}
status['scope'] = 'diagnostic_5_step_sol_attn_metadata_gate_not_fidelity_or_performance_claim'
status['opaque_integrity_policy'] = {
    'image_identifiers': 'omitted_not_evidence',
    'output_identifiers': 'omitted_not_evidence',
    'opaque_identifier_equality': 'not_used_for_classification',
}
(root / 'sol_attn_diagnostic_status.json').write_text(json.dumps(status, indent=2, sort_keys=True) + '\n')
print(json.dumps(status, sort_keys=True))
PY
