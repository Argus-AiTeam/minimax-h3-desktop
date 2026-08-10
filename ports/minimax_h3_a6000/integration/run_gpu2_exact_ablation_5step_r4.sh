#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# External GPU2 gate only. Do not run from CPU/static tasks.
# Runs dense plus one exact-kernel-family at a time to localize the r4
# dense-vs-exact AV divergence. It loads the MiniMax-H3 model inside the
# authorized container only and keeps Sol-Attn/cache disabled throughout.
set -euo pipefail

if [[ ${MINIMAX_H3_A6000_RUN_ABLATION:-0} != 1 ]]; then
  echo "MiniMax-H3 exact ablation runner is default-off; set MINIMAX_H3_A6000_RUN_ABLATION=1 to launch the external GPU gate." >&2
  exit 0
fi

ROOT=${ROOT:-${PWD}}
GPU_INDEX=${GPU_INDEX:-2}
EXPECTED_UUID=${EXPECTED_UUID:-}
IMAGE=${IMAGE:-argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r4-exact-overlay}
MODEL_ROOT=${MODEL_ROOT:-$ROOT/models/MiniMax-H3}
PROMPT_FILE=${PROMPT_FILE:-$ROOT/technical_report/evidence/minimax_h3_desktop/baseline_a6000/t2va_example_1.prompt.txt}
OUT_DIR=${OUT_DIR:-$ROOT/technical_report/evidence/minimax_h3_desktop/sol_engine_port/r4_ablation_$(date -u +%Y%m%dT%H%M%SZ)}
PORT=${PORT:-8000}
SHADOW_COMPARE=${SHADOW_COMPARE:-1}
SHADOW_CALLS=${SHADOW_CALLS:-3}
SHADOW_STRICT=${SHADOW_STRICT:-0}

mkdir -p "$OUT_DIR" "$OUT_DIR/cache"
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
ablation_modes=dense,adaln,rope,swiglu,all_exact
shadow_compare=$SHADOW_COMPARE
shadow_calls=$SHADOW_CALLS
shadow_strict=$SHADOW_STRICT
note=r4_needs_revision_diagnostic_only_no_fidelity_claim
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
python3 - <<'PY' > '$evidence/overlay_preflight.json'
import json
from minimax_h3_a6000.env import DEFAULT_ENV_SWITCHES
required = [
    'MINIMAX_H3_A6000_ENABLE_FUSED_ADALN_MODULATE',
    'MINIMAX_H3_A6000_ENABLE_FUSED_ADALN_GATE',
    'MINIMAX_H3_A6000_ABLATION_DISABLE_INDEXED_MODULATE',
    'MINIMAX_H3_A6000_ABLATION_DISABLE_INDEXED_GATE',
    'MINIMAX_H3_A6000_ABLATION_DISABLE_ROPE',
    'MINIMAX_H3_A6000_ABLATION_DISABLE_SWIGLU',
    'MINIMAX_H3_A6000_ENABLE_SHADOW',
    'MINIMAX_H3_A6000_SHADOW_CALLS',
    'MINIMAX_H3_A6000_SHADOW_STRICT',
]
missing = [key for key in required if key not in DEFAULT_ENV_SWITCHES]
assert not missing, f'rebuild the r4 overlay image from the current port; missing ablation switches: {missing}'
print(json.dumps({'ablation_switches_present': required, 'defaults': {key: DEFAULT_ENV_SWITCHES[key] for key in required}}, sort_keys=True))
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
if [[ '$mode' != dense && -f /evidence/dense/output.mp4 ]]; then
python3 - <<'PY'
import json, math, pathlib
import av
import numpy as np

baseline = pathlib.Path('/evidence/dense/output.mp4')
candidate = pathlib.Path('$evidence/output.mp4')

def video_frames(path):
    out = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            out.append(frame.to_ndarray(format='rgb24').astype(np.float32))
    return out

def audio_values(path):
    chunks = []
    with av.open(str(path)) as container:
        streams = [s for s in container.streams if s.type == 'audio']
        if not streams:
            return np.zeros((0,), dtype=np.float32)
        for frame in container.decode(streams[0]):
            arr = frame.to_ndarray().astype(np.float32)
            if arr.ndim == 2:
                arr = arr.T.reshape(-1)
            else:
                arr = arr.reshape(-1)
            chunks.append(arr)
    return np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.float32)

base_frames = video_frames(baseline)
cand_frames = video_frames(candidate)
n_frames = min(len(base_frames), len(cand_frames))
if n_frames:
    mse_values = [float(np.mean((base_frames[i] - cand_frames[i]) ** 2)) for i in range(n_frames)]
    mae_values = [float(np.mean(np.abs(base_frames[i] - cand_frames[i]))) for i in range(n_frames)]
    mean_mse = float(np.mean(mse_values))
    mean_mae = float(np.mean(mae_values))
    psnr = 99.0 if mean_mse < 1.0e-9 else float(10.0 * math.log10((255.0 * 255.0) / mean_mse))
else:
    mean_mse = mean_mae = psnr = None
base_audio = audio_values(baseline)
cand_audio = audio_values(candidate)
n_audio = min(int(base_audio.size), int(cand_audio.size))
if n_audio:
    a = base_audio[:n_audio].astype(np.float64)
    b = cand_audio[:n_audio].astype(np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine = float(np.dot(a, b) / denom) if denom else None
    rms_a = float(np.sqrt(np.mean(a * a)))
    rms_b = float(np.sqrt(np.mean(b * b)))
else:
    cosine = rms_a = rms_b = None
record = {
    'mode': '$mode',
    'comparison': 'candidate_vs_dense_same_prompt_seed_5step',
    'claim_scope': 'diagnostic_ablation_only_not_fidelity_acceptance',
    'video_frames_compared': int(n_frames),
    'video_mean_mse': mean_mse,
    'video_mean_mae': mean_mae,
    'video_psnr_db': psnr,
    'audio_values_compared': int(n_audio),
    'audio_waveform_cosine': cosine,
    'audio_rms_dense': rms_a,
    'audio_rms_candidate': rms_b,
}
pathlib.Path('$evidence/quality_vs_dense.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
PY
fi
kill \$server_pid || true
wait \$server_pid || true
"
}

# Dense reference: all exact wrappers and Sol-Attn/cache explicitly off.
run_one dense \
  -e MINIMAX_H3_A6000_ENABLE_OVERLAY=0 \
  -e MINIMAX_H3_A6000_ENABLE_TRITON_CANDIDATES=0 \
  -e MINIMAX_H3_A6000_EXACT_INDEXED_STRATEGY=auto \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ADALN=0 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ADALN_MODULATE=0 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ADALN_GATE=0 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ROPE=0 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_SWIGLU=0 \
  -e MINIMAX_H3_A6000_ABLATION_DISABLE_INDEXED_MODULATE=0 \
  -e MINIMAX_H3_A6000_ABLATION_DISABLE_INDEXED_GATE=0 \
  -e MINIMAX_H3_A6000_ABLATION_DISABLE_ROPE=0 \
  -e MINIMAX_H3_A6000_ABLATION_DISABLE_SWIGLU=0 \
  -e MINIMAX_H3_A6000_ENABLE_SOL_ATTN=0 \
  -e MINIMAX_H3_A6000_SOL_ATTN_CACHE=0 \
  -e MINIMAX_H3_A6000_ENABLE_TELEMETRY=0 \
  -e MINIMAX_H3_A6000_ENABLE_SHADOW=0 \
  -e MINIMAX_H3_A6000_SHADOW_CALLS=0 \
  -e MINIMAX_H3_A6000_SHADOW_STRICT=0

# AdaLN only: both indexed modulation and gated residual wrappers on; RoPE/SwiGLU remain dense.
run_one adaln \
  -e MINIMAX_H3_A6000_ENABLE_OVERLAY=1 \
  -e MINIMAX_H3_A6000_ENABLE_TRITON_CANDIDATES=1 \
  -e MINIMAX_H3_A6000_EXACT_INDEXED_STRATEGY=auto \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ADALN=1 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ADALN_MODULATE=0 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ADALN_GATE=0 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ROPE=0 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_SWIGLU=0 \
  -e MINIMAX_H3_A6000_ABLATION_DISABLE_INDEXED_MODULATE=0 \
  -e MINIMAX_H3_A6000_ABLATION_DISABLE_INDEXED_GATE=0 \
  -e MINIMAX_H3_A6000_ABLATION_DISABLE_ROPE=0 \
  -e MINIMAX_H3_A6000_ABLATION_DISABLE_SWIGLU=0 \
  -e MINIMAX_H3_A6000_ENABLE_SOL_ATTN=0 \
  -e MINIMAX_H3_A6000_SOL_ATTN_CACHE=0 \
  -e MINIMAX_H3_A6000_ENABLE_TELEMETRY=1 \
  -e MINIMAX_H3_A6000_TELEMETRY_ATEXIT=1 \
  -e MINIMAX_H3_A6000_TELEMETRY_JSON=/evidence/adaln/exact_telemetry.json \
  -e MINIMAX_H3_A6000_ENABLE_SHADOW="$SHADOW_COMPARE" \
  -e MINIMAX_H3_A6000_SHADOW_CALLS="$SHADOW_CALLS" \
  -e MINIMAX_H3_A6000_SHADOW_STRICT="$SHADOW_STRICT"

# RoPE only.
run_one rope \
  -e MINIMAX_H3_A6000_ENABLE_OVERLAY=1 \
  -e MINIMAX_H3_A6000_ENABLE_TRITON_CANDIDATES=1 \
  -e MINIMAX_H3_A6000_EXACT_INDEXED_STRATEGY=auto \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ADALN=0 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ADALN_MODULATE=0 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ADALN_GATE=0 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ROPE=1 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_SWIGLU=0 \
  -e MINIMAX_H3_A6000_ABLATION_DISABLE_INDEXED_MODULATE=0 \
  -e MINIMAX_H3_A6000_ABLATION_DISABLE_INDEXED_GATE=0 \
  -e MINIMAX_H3_A6000_ABLATION_DISABLE_ROPE=0 \
  -e MINIMAX_H3_A6000_ABLATION_DISABLE_SWIGLU=0 \
  -e MINIMAX_H3_A6000_ENABLE_SOL_ATTN=0 \
  -e MINIMAX_H3_A6000_SOL_ATTN_CACHE=0 \
  -e MINIMAX_H3_A6000_ENABLE_TELEMETRY=1 \
  -e MINIMAX_H3_A6000_TELEMETRY_ATEXIT=1 \
  -e MINIMAX_H3_A6000_TELEMETRY_JSON=/evidence/rope/exact_telemetry.json \
  -e MINIMAX_H3_A6000_ENABLE_SHADOW="$SHADOW_COMPARE" \
  -e MINIMAX_H3_A6000_SHADOW_CALLS="$SHADOW_CALLS" \
  -e MINIMAX_H3_A6000_SHADOW_STRICT="$SHADOW_STRICT"

# SwiGLU only.
run_one swiglu \
  -e MINIMAX_H3_A6000_ENABLE_OVERLAY=1 \
  -e MINIMAX_H3_A6000_ENABLE_TRITON_CANDIDATES=1 \
  -e MINIMAX_H3_A6000_EXACT_INDEXED_STRATEGY=auto \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ADALN=0 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ADALN_MODULATE=0 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ADALN_GATE=0 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ROPE=0 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_SWIGLU=1 \
  -e MINIMAX_H3_A6000_ABLATION_DISABLE_INDEXED_MODULATE=0 \
  -e MINIMAX_H3_A6000_ABLATION_DISABLE_INDEXED_GATE=0 \
  -e MINIMAX_H3_A6000_ABLATION_DISABLE_ROPE=0 \
  -e MINIMAX_H3_A6000_ABLATION_DISABLE_SWIGLU=0 \
  -e MINIMAX_H3_A6000_ENABLE_SOL_ATTN=0 \
  -e MINIMAX_H3_A6000_SOL_ATTN_CACHE=0 \
  -e MINIMAX_H3_A6000_ENABLE_TELEMETRY=1 \
  -e MINIMAX_H3_A6000_TELEMETRY_ATEXIT=1 \
  -e MINIMAX_H3_A6000_TELEMETRY_JSON=/evidence/swiglu/exact_telemetry.json \
  -e MINIMAX_H3_A6000_ENABLE_SHADOW="$SHADOW_COMPARE" \
  -e MINIMAX_H3_A6000_SHADOW_CALLS="$SHADOW_CALLS" \
  -e MINIMAX_H3_A6000_SHADOW_STRICT="$SHADOW_STRICT"

# All exact kernels together: r4 reproduction mode, still diagnostic only.
run_one all_exact \
  -e MINIMAX_H3_A6000_ENABLE_OVERLAY=1 \
  -e MINIMAX_H3_A6000_ENABLE_TRITON_CANDIDATES=1 \
  -e MINIMAX_H3_A6000_EXACT_INDEXED_STRATEGY=auto \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ADALN=1 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ADALN_MODULATE=0 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ADALN_GATE=0 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ROPE=1 \
  -e MINIMAX_H3_A6000_ENABLE_FUSED_SWIGLU=1 \
  -e MINIMAX_H3_A6000_ABLATION_DISABLE_INDEXED_MODULATE=0 \
  -e MINIMAX_H3_A6000_ABLATION_DISABLE_INDEXED_GATE=0 \
  -e MINIMAX_H3_A6000_ABLATION_DISABLE_ROPE=0 \
  -e MINIMAX_H3_A6000_ABLATION_DISABLE_SWIGLU=0 \
  -e MINIMAX_H3_A6000_ENABLE_SOL_ATTN=0 \
  -e MINIMAX_H3_A6000_SOL_ATTN_CACHE=0 \
  -e MINIMAX_H3_A6000_ENABLE_TELEMETRY=1 \
  -e MINIMAX_H3_A6000_TELEMETRY_ATEXIT=1 \
  -e MINIMAX_H3_A6000_TELEMETRY_JSON=/evidence/all_exact/exact_telemetry.json \
  -e MINIMAX_H3_A6000_ENABLE_SHADOW="$SHADOW_COMPARE" \
  -e MINIMAX_H3_A6000_SHADOW_CALLS="$SHADOW_CALLS" \
  -e MINIMAX_H3_A6000_SHADOW_STRICT="$SHADOW_STRICT"

python3 - "$OUT_DIR" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
contracts = {
    'adaln': {
        'expected_candidates': ['indexed_modulate_bf16', 'indexed_gate_bf16'],
        'ablation_disabled': [],
        'expected_zero_calls': ['apply_rope_bf16', 'swiglu_bf16'],
    },
    'rope': {
        'expected_candidates': ['apply_rope_bf16'],
        'ablation_disabled': [],
        'expected_zero_calls': ['indexed_modulate_bf16', 'indexed_gate_bf16', 'swiglu_bf16'],
    },
    'swiglu': {
        'expected_candidates': ['swiglu_bf16'],
        'ablation_disabled': [],
        'expected_zero_calls': ['indexed_modulate_bf16', 'indexed_gate_bf16', 'apply_rope_bf16'],
    },
    'all_exact': {
        'expected_candidates': ['indexed_modulate_bf16', 'indexed_gate_bf16', 'apply_rope_bf16', 'swiglu_bf16'],
        'ablation_disabled': [],
        'expected_zero_calls': [],
    },
}

def http_metrics(mode):
    data = {}
    for line in (root / mode / 'http_metrics.txt').read_text().splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            data[k] = v
    return data

status = {
    'status': 'pass',
    'scope': 'diagnostic_5_step_per_kernel_ablation_not_fidelity_or_performance_claim',
    'r4_needs_revision_context': {
        'apparent_improvement_percent': 1.79,
        'video_psnr_db': 24.63,
        'audio_waveform_cosine': 0.9776,
        'verdict': 'needs-revision; do not retain as fidelity evidence',
    },
    'modes': {},
}
for mode in ['dense', *contracts.keys()]:
    av = json.loads((root / mode / 'av_validation.json').read_text())
    assert av['video_present'] and av['audio_present'], av
    entry = {'av': av, 'http_metrics': http_metrics(mode)}
    if mode != 'dense':
        telemetry = json.loads((root / mode / 'exact_telemetry.json').read_text())
        ops = telemetry['ops']
        contract = contracts[mode]
        for op in contract['expected_candidates']:
            item = ops[op]
            assert item['calls'] > 0, f'{mode}:{op} wrapper call count is zero'
            assert item['candidate'] > 0, f'{mode}:{op} candidate count is zero: {item}'
            assert item['fallback'] == 0, f'{mode}:{op} fallback is nonzero: {item}'
        for op in contract['ablation_disabled']:
            item = ops[op]
            assert item['calls'] > 0, f'{mode}:{op} disabled ablation op was never called'
            assert item['candidate'] == 0, f'{mode}:{op} candidate launched despite ablation disable: {item}'
            assert item['fallback'] > 0 and item['decline'] > 0, f'{mode}:{op} did not record disabled fallback: {item}'
            assert any('per-kernel ablation env' in reason for reason in item['reasons']), item
        for op in contract['expected_zero_calls']:
            item = ops[op]
            assert item['calls'] == 0 and item['candidate'] == 0, f'{mode}:{op} unexpectedly ran: {item}'
        quality_path = root / mode / 'quality_vs_dense.json'
        assert quality_path.exists(), f'{mode} missing quality_vs_dense.json'
        entry['quality_vs_dense'] = json.loads(quality_path.read_text())
        entry['telemetry_summary'] = {op: ops[op] for op in sorted(ops)}
    status['modes'][mode] = entry
(root / 'ablation_status.json').write_text(json.dumps(status, indent=2, sort_keys=True) + '\n')
print(json.dumps(status, sort_keys=True))
PY
