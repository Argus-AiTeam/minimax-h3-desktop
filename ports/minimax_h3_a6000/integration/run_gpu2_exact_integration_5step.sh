#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# External GPU gate only. Do not run from CPU/static tasks.
# The operator should map the approved physical GPU2 as the only visible device
# inside the container (CUDA_VISIBLE_DEVICES=0 in-container) via --gpus device=2.
set -euo pipefail

ROOT=${ROOT:-${PWD}}
GPU_INDEX=${GPU_INDEX:-2}
EXPECTED_UUID=${EXPECTED_UUID:-}
IMAGE=${IMAGE:-argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r3-exact-overlay}
MODEL_ROOT=${MODEL_ROOT:-$ROOT/models/MiniMax-H3}
PROMPT_FILE=${PROMPT_FILE:-$ROOT/technical_report/evidence/minimax_h3_desktop/baseline_a6000/t2va_example_1.prompt.txt}
OUT_DIR=${OUT_DIR:-$ROOT/technical_report/evidence/minimax_h3_desktop/sol_engine_port/r3_integration_$(date -u +%Y%m%dT%H%M%SZ)}
PORT=${PORT:-8000}

mkdir -p "$OUT_DIR" "$OUT_DIR/cache/dense" "$OUT_DIR/cache/exact"
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
attention_backend=CUDNN_ATTN
sol_attn=off
sol_attn_cache=off
exact_indexed_strategy=auto
EOF_WORKLOAD

if [[ -n "$EXPECTED_UUID" ]]; then
  actual_uuid=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid --format=csv,noheader)
  if [[ "$actual_uuid" != "$EXPECTED_UUID" ]]; then
    echo "ERROR: GPU $GPU_INDEX uuid mismatch: got $actual_uuid expected $EXPECTED_UUID" >&2
    exit 10
  fi
fi

run_one() {
  local mode=$1
  local evidence=/evidence/$mode
  local cache=/workspace/cache/$mode
  shift
  mkdir -p "$OUT_DIR/$mode" "$OUT_DIR/cache/$mode"
  chmod 0777 "$OUT_DIR" "$OUT_DIR/$mode" "$OUT_DIR/cache" "$OUT_DIR/cache/$mode"
  docker run --rm \
    --gpus "device=$GPU_INDEX" \
    --ipc=host \
    --ulimit memlock=-1 \
    --network none \
    --cap-drop=ALL \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_OMNI_VIDEO_SYNC_TIMEOUT=3600 \
    -e HF_HOME="$cache/hf" \
    -e TRANSFORMERS_CACHE="$cache/hf" \
    -e TORCHINDUCTOR_CACHE_DIR="$cache/torchinductor" \
    -e TRITON_CACHE_DIR="$cache/triton" \
    "$@" \
    -v "$MODEL_ROOT":/models/MiniMax-H3:ro \
    -v "$OUT_DIR":/evidence:rw \
    -v "$OUT_DIR/cache":/workspace/cache:rw \
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
  --enforce-eager --diffusion-attention-backend CUDNN_ATTN \
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

# Dense reference: exact wrappers and Sol-Attn/cache explicitly off.
run_one dense \
  -e MINIMAX_H3_A6000_ENABLE_OVERLAY=0 \
  -e MINIMAX_H3_A6000_ENABLE_TRITON_CANDIDATES=0 \
  -e MINIMAX_H3_A6000_EXACT_INDEXED_STRATEGY=auto \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ADALN=0 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ROPE=0 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_SWIGLU=0 \
  -e MINIMAX_H3_A6000_ENABLE_SOL_ATTN=0 \
  -e MINIMAX_H3_A6000_SOL_ATTN_CACHE=0 \
  -e MINIMAX_H3_A6000_ENABLE_TELEMETRY=0

# Exact candidate: enable the three exact wrapper families only; keep Sol-Attn/cache off.
run_one exact \
  -e MINIMAX_H3_A6000_ENABLE_OVERLAY=1 \
  -e MINIMAX_H3_A6000_ENABLE_TRITON_CANDIDATES=1 \
  -e MINIMAX_H3_A6000_EXACT_INDEXED_STRATEGY=auto \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ADALN=1 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ROPE=1 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_SWIGLU=1 \
  -e MINIMAX_H3_A6000_ENABLE_SOL_ATTN=0 \
  -e MINIMAX_H3_A6000_SOL_ATTN_CACHE=0 \
  -e MINIMAX_H3_A6000_ENABLE_TELEMETRY=1 \
  -e MINIMAX_H3_A6000_TELEMETRY_ATEXIT=1 \
  -e MINIMAX_H3_A6000_TELEMETRY_JSON=/evidence/exact/exact_telemetry.json

python3 - "$OUT_DIR" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for mode in ('dense', 'exact'):
    av = json.loads((root / mode / 'av_validation.json').read_text())
    assert av['video_present'] and av['audio_present'], av
telemetry = json.loads((root / 'exact' / 'exact_telemetry.json').read_text())
ops = telemetry['ops']
required = {
    'indexed_modulate_bf16': 'AdaLN modulation',
    'indexed_gate_bf16': 'AdaLN gate',
    'apply_rope_bf16': 'RoPE',
    'swiglu_bf16': 'SwiGLU',
}
def http_metrics(mode):
    data = {}
    for line in (root / mode / 'http_metrics.txt').read_text().splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            data[k] = v
    return data
summary = {}
strategy_summary = {}
copy_summary = {}
for op, label in required.items():
    item = ops[op]
    summary[op] = item
    assert item['calls'] > 0, f'{label} wrapper call count is zero'
    assert item['candidate'] > 0, f'{label} candidate launch count is zero: {item}'
    strategy_summary[op] = item.get('strategies', {})
    copy_summary[op] = {
        'materialize_copy_calls': item.get('materialize_copy_calls', 0),
        'materialize_copy_bytes': item.get('materialize_copy_bytes', 0),
    }
for op in ('indexed_modulate_bf16', 'indexed_gate_bf16'):
    strategies = strategy_summary[op]
    assert 'stride_aware' in strategies and 'materialize' in strategies, f'{op} missing strategy telemetry: {strategies}'
status = {
    'status': 'pass',
    'scope': '5_step_same_workload_integration_not_e2e_benchmark',
    'http_metrics': {'dense': http_metrics('dense'), 'exact': http_metrics('exact')},
    'telemetry_summary': summary,
    'strategy_summary': strategy_summary,
    'materialize_copy_summary': copy_summary,
}
(root / 'integration_status.json').write_text(json.dumps(status, indent=2, sort_keys=True) + '\n')
print(json.dumps(status, sort_keys=True))
PY
