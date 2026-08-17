#!/usr/bin/env bash
# VAE-only gate for default-off bounded spatial tile-batch-size decode.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd -P)}
cd "$ROOT"

IMAGE=${IMAGE:-argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r13-split-profiler}
DOCKER_HOST=${DOCKER_HOST:-unix:///tmp/minih3-20260814T135313Z.sock}
GPU_INDEX=${GPU_INDEX:-0}
EXPECTED_UUID=${EXPECTED_UUID:-}
CAPS=${CAPS:-4,7,14}
RUN_ID=${RUN_ID:-r20_vae_bounded_tile_batching_probe_$(date -u +%Y%m%dT%H%M%SZ)}
EVIDENCE_DIR=${EVIDENCE_DIR:-$ROOT/technical_report/evidence/minimax_h3_desktop/long_video/$RUN_ID}
OWNER=${OWNER:-minimax-h3-r20-vae-bounded-tile-batching-probe}
LOCAL_VAE=${LOCAL_VAE:-$ROOT/runtime/single_a6000_bf16/src/vllm-omni/vllm_omni/diffusion/models/minimax_h3/vae.py}
DRY_RUN=1

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_vae_bounded_tile_batching_probe.sh --dry-run
  ARGUS_ALLOW_VAE_BOUNDED_TILE_BATCHING_PROBE=1 I_ACCEPT_MINIMAX_H3_LICENSE=YES \
    DOCKER_HOST=unix:///tmp/minih3-20260814T135313Z.sock GPU_INDEX=<idle-a6000> CAPS=4,7,14 \
    bash scripts/run_vae_bounded_tile_batching_probe.sh --execute

Runs a VAE-only same-input representative-latent characterization on one RTX
A6000 for default-off MINIMAX_H3_A6000_VIDEO_VAE_TILE_BATCH_SIZE. It does not
run denoise/text/audio/final-AV and does not make speedup, BF16-fidelity,
exact/lossless, native-long-context, or human-quality claims.
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

mkdir -p "$EVIDENCE_DIR/prelaunch" "$EVIDENCE_DIR/after_lease"
EVIDENCE_DIR=$(cd "$EVIDENCE_DIR" && pwd -P)
chmod 0777 "$EVIDENCE_DIR" "$EVIDENCE_DIR/prelaunch" "$EVIDENCE_DIR/after_lease" 2>/dev/null || true

if (( DRY_RUN )); then
  cat <<EOF
[DRY-RUN] VAE bounded tile-batch-size characterization probe
Evidence: $EVIDENCE_DIR
Image: $IMAGE
Docker host: $DOCKER_HOST
GPU index: $GPU_INDEX
Caps: $CAPS
Adapter bind mount: $LOCAL_VAE
Plan: baseline cap=0 and bounded caps for latent [1,24,37,48,84], no denoise/text/audio/final-AV.
No Docker container, GPU inference, model load, or media generation was performed.
EOF
  exit 0
fi

if [[ "${ARGUS_ALLOW_VAE_BOUNDED_TILE_BATCHING_PROBE:-}" != "1" ]]; then
  echo "ERROR: set ARGUS_ALLOW_VAE_BOUNDED_TILE_BATCHING_PROBE=1 for this GPU VAE probe" >&2
  exit 2
fi
if [[ "${I_ACCEPT_MINIMAX_H3_LICENSE:-}" != "YES" ]]; then
  echo "ERROR: read/accept the MiniMax-H3 license for local inference, then set I_ACCEPT_MINIMAX_H3_LICENSE=YES" >&2
  exit 2
fi
if [[ ! -f "$LOCAL_VAE" ]]; then
  echo "ERROR: local VAE adapter source not found: $LOCAL_VAE" >&2
  exit 2
fi

cp tools/vae_bounded_tile_batching_probe.py "$EVIDENCE_DIR/probe.py"
chmod a+r "$EVIDENCE_DIR/probe.py"
export DOCKER_HOST
{
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "root=$ROOT"
  echo "evidence_dir=$EVIDENCE_DIR"
  echo "docker_host=$DOCKER_HOST"
  echo "image=$IMAGE"
  echo "gpu_index=$GPU_INDEX"
  echo "expected_uuid=$EXPECTED_UUID"
  echo "caps=$CAPS"
  echo "local_vae=$LOCAL_VAE"
} > "$EVIDENCE_DIR/launch_env.env"

git -C runtime/single_a6000_bf16/src/vllm-omni diff -- vllm_omni/diffusion/models/minimax_h3/vae.py > "$EVIDENCE_DIR/local_vae_adapter.diff" || true
git -C upstreams/Sana-sol-engine rev-parse HEAD > "$EVIDENCE_DIR/upstream_sana_sol_engine_head.txt" 2> "$EVIDENCE_DIR/upstream_sana_sol_engine_head.stderr" || true
git -C runtime/single_a6000_bf16/src/vllm-omni rev-parse HEAD > "$EVIDENCE_DIR/runtime_vllm_omni_head.txt" 2> "$EVIDENCE_DIR/runtime_vllm_omni_head.stderr" || true

nvidia-smi -L > "$EVIDENCE_DIR/prelaunch/nvidia_smi_L.txt" 2> "$EVIDENCE_DIR/prelaunch/nvidia_smi_L.stderr"
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw,power.limit,temperature.gpu --format=csv,noheader,nounits > "$EVIDENCE_DIR/prelaunch/nvidia_smi.csv" 2> "$EVIDENCE_DIR/prelaunch/nvidia_smi.stderr"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits > "$EVIDENCE_DIR/prelaunch/compute_apps.csv" 2> "$EVIDENCE_DIR/prelaunch/compute_apps.stderr" || true
python3 -m argus_skill.tools.gpu_lease status > "$EVIDENCE_DIR/prelaunch/gpu_lease_status.json" 2> "$EVIDENCE_DIR/prelaunch/gpu_lease_status.stderr" || true
docker info --format '{{json .}}' > "$EVIDENCE_DIR/prelaunch/docker_info.json" 2> "$EVIDENCE_DIR/prelaunch/docker_info.stderr"
docker ps --format '{{json .}}' > "$EVIDENCE_DIR/prelaunch/docker_ps.jsonl" 2> "$EVIDENCE_DIR/prelaunch/docker_ps.stderr"
docker image inspect "$IMAGE" > "$EVIDENCE_DIR/prelaunch/image.inspect.json" 2> "$EVIDENCE_DIR/prelaunch/image.inspect.stderr"
python3 - "$EVIDENCE_DIR/prelaunch/docker_info.json" <<'PY'
import json, pathlib, sys
root = str(json.loads(pathlib.Path(sys.argv[1]).read_text()).get('DockerRootDir',''))
if root == '<private-path>' or root.startswith('<private-path>'):
    raise SystemExit(f'refusing shared Docker root: {root}')
if '.isolated_docker' not in root:
    raise SystemExit(f'Docker root does not look project-isolated: {root}')
PY
GPU_UUID=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
printf '%s\n' "$GPU_UUID" > "$EVIDENCE_DIR/prelaunch/selected_gpu_uuid.txt"
if [[ -n "$EXPECTED_UUID" && "$GPU_UUID" != "$EXPECTED_UUID" ]]; then
  echo "ERROR: GPU UUID mismatch for index $GPU_INDEX: expected $EXPECTED_UUID got $GPU_UUID" >&2
  exit 4
fi
if grep -F "$GPU_UUID" "$EVIDENCE_DIR/prelaunch/compute_apps.csv" >/dev/null 2>&1; then
  echo "ERROR: selected GPU $GPU_INDEX ($GPU_UUID) already has a compute process" >&2
  exit 4
fi
GPU_MEM_USED=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
if [[ "$GPU_MEM_USED" =~ ^[0-9]+$ ]] && (( GPU_MEM_USED > 1000 )); then
  echo "ERROR: selected GPU $GPU_INDEX ($GPU_UUID) memory.used=${GPU_MEM_USED}MiB is not idle enough" >&2
  exit 4
fi

cat > "$EVIDENCE_DIR/container_command.json" <<JSON
{"image":"$IMAGE","gpu_index":"$GPU_INDEX","gpu_uuid":"$GPU_UUID","model_mount":"$ROOT/models/MiniMax-H3:/models/MiniMax-H3:ro","adapter_mount":"$LOCAL_VAE","network":"none","cap_drop":"ALL","script":"/evidence/probe.py","mechanism":"bounded_video_vae_tile_batch_size","caps":"$CAPS"}
JSON

set +e
python3 -m argus_skill.tools.gpu_lease run --owner "$OWNER" --gpus "$GPU_INDEX" --ttl 3600 -- \
  bash -lc "set -euo pipefail; export DOCKER_HOST='$DOCKER_HOST'; docker run --rm --gpus 'device=$GPU_INDEX' --ipc=host --ulimit memlock=-1 --network none --cap-drop=ALL -e CUDA_VISIBLE_DEVICES=0 -e NVIDIA_VISIBLE_DEVICES=$GPU_INDEX -e HF_HOME=/tmp/minimax_h3_vae_bounded_tile_probe/hf -e TRANSFORMERS_CACHE=/tmp/minimax_h3_vae_bounded_tile_probe/hf -e TORCHINDUCTOR_CACHE_DIR=/tmp/minimax_h3_vae_bounded_tile_probe/torchinductor -e TRITON_CACHE_DIR=/tmp/minimax_h3_vae_bounded_tile_probe/triton -e MINIMAX_H3_VAE_DECODER_STREAM_TEMPORAL_CAT=1 -e MINIMAX_H3_A6000_VIDEO_VAE_SPATIAL_TILE_BATCHING=0 -e MINIMAX_H3_A6000_VIDEO_VAE_TILE_BATCH_SIZE=0 -v '$ROOT/models/MiniMax-H3:/models/MiniMax-H3:ro' -v '$EVIDENCE_DIR:/evidence:rw' -v '$LOCAL_VAE:/usr/local/lib/python3.12/dist-packages/vllm_omni/diffusion/models/minimax_h3/vae.py:ro' -v '$LOCAL_VAE:/app/vllm-omni/vllm_omni/diffusion/models/minimax_h3/vae.py:ro' '$IMAGE' python3 /evidence/probe.py --evidence /evidence --model-path /models/MiniMax-H3/FL2VA/video_vae --caps '$CAPS'" \
  > "$EVIDENCE_DIR/lease_stdout.raw" 2> "$EVIDENCE_DIR/lease_stderr.log"
lease_rc=$?
set -e
inner_rc=$(python3 - "$EVIDENCE_DIR/lease_stdout.raw" "$lease_rc" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
fallback = int(sys.argv[2])
try:
    text = path.read_text(encoding='utf-8', errors='replace').strip()
    data = json.loads(text[text.find('{'):]) if '{' in text else {}
    value = data.get('returncode', fallback)
    print(int(value) if value is not None else fallback)
except Exception:
    print(fallback)
PY
)
if [[ "$inner_rc" != "0" ]]; then
  lease_rc="$inner_rc"
fi
if [[ -f "$EVIDENCE_DIR/decision.json" ]]; then
  decision_status=$(python3 - "$EVIDENCE_DIR/decision.json" <<'PY'
import json, pathlib, sys
try:
    print(str(json.loads(pathlib.Path(sys.argv[1]).read_text()).get('status', 'missing')))
except Exception:
    print('unreadable')
PY
)
  case "$decision_status" in
    pass|reject) lease_rc=0 ;;
    *) lease_rc=1 ;;
  esac
fi
printf '%s\n' "$lease_rc" > "$EVIDENCE_DIR/lease_exit_code"
docker ps --format '{{json .}}' > "$EVIDENCE_DIR/after_lease/docker_ps.jsonl" 2> "$EVIDENCE_DIR/after_lease/docker_ps.stderr" || true
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw,power.limit,temperature.gpu --format=csv,noheader,nounits > "$EVIDENCE_DIR/after_lease/nvidia_smi.csv" 2> "$EVIDENCE_DIR/after_lease/nvidia_smi.stderr" || true
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits > "$EVIDENCE_DIR/after_lease/compute_apps.csv" 2> "$EVIDENCE_DIR/after_lease/compute_apps.stderr" || true
python3 -m argus_skill.tools.gpu_lease status > "$EVIDENCE_DIR/after_lease/gpu_lease_status.json" 2> "$EVIDENCE_DIR/after_lease/gpu_lease_status.stderr" || true
exit "$lease_rc"
