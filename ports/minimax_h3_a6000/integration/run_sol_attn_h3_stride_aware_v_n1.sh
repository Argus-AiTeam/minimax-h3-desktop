#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# External single-A6000 r9 materialized-vs-stride-aware V matched N=1 gate.
# Default is --dry-run: no GPU execution, Docker execution, nvidia-smi, CUDA,
# network access, downloads, model loading, inference, cache enablement, or publication.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd -P)}
GPU_INDEX=${GPU_INDEX:-2}
EXPECTED_UUID=${EXPECTED_UUID:-}
IMAGE=${IMAGE:-argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r9-sol-attn-overlay}
REQUIRED_IMAGE_VERSION_LABEL=${REQUIRED_IMAGE_VERSION_LABEL:-r9}
MODEL_ROOT=${MODEL_ROOT:-$ROOT/models/MiniMax-H3}
PROMPT_FILE=${PROMPT_FILE:-$ROOT/technical_report/evidence/minimax_h3_desktop/baseline_a6000/t2va_example_1.prompt.txt}
OUT_DIR=${OUT_DIR:-$ROOT/technical_report/evidence/minimax_h3_desktop/sol_engine_port/sol_attn_h3_stride_aware_v_r9_n1_$(date -u +%Y%m%dT%H%M%SZ)}
PORT=${PORT:-8000}
GPU_IDLE_MAX_MEMORY_MIB=${GPU_IDLE_MAX_MEMORY_MIB:-512}
GPU_IDLE_MAX_UTIL_PCT=${GPU_IDLE_MAX_UTIL_PCT:-5}
GPU_RESOURCE_SAMPLE_INTERVAL_S=${GPU_RESOURCE_SAMPLE_INTERVAL_S:-5}
if [[ "$OUT_DIR" != /* ]]; then
  OUT_DIR="$ROOT/$OUT_DIR"
fi
DRY_RUN=1

usage() {
  cat <<'EOF_USAGE'
Usage: run_sol_attn_h3_stride_aware_v_n1.sh [--dry-run|--execute]

Dry-run is the default and prints the exact external A6000 plan without GPU
execution, Docker execution, CUDA, nvidia-smi, network access, downloads, model
loading, inference, cache enablement, or publication. Non-dry execution requires
ARGUS_ALLOW_A6000_SOL_ATTN_STRIDE_AWARE_V_N1=1 plus readable fresh r9 image
version/base/title label checks, a fresh GPU-hygiene preflight showing a truly
idle selected A6000, and is outside CPU/static stages.

The gate compares two 5-step runs through the same opt-in H3_A6000_SOL_ATTN
backend and unchanged r8 sparse policy:
  1. r8_materialized_reference: source-backed V is materialized exactly as in r8;
  2. stride_aware_v: source-backed V strides are consumed directly by Triton.
Each mode receives one excluded warmup request and then arms per-call CUDA-event
telemetry for the measured request. The N=1 gate is promoted only if the E2E
signal exceeds max(2*0.5072177176%, 1.5%) and all correctness, copy,
sparse/fallback, resource, and stability gates pass.
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
identity_guard=non-dry verifies readable image tag plus version/base/title labels for the fresh r9 overlay; opaque image identifiers are omitted and are not proof
external_r9_build_command=EVIDENCE_DIR=technical_report/evidence/minimax_h3_desktop/sol_engine_port/r9_overlay_image bash ports/minimax_h3_a6000/integration/r9/build_r9_overlay_image.sh
model_root=$MODEL_ROOT
prompt_file=$PROMPT_FILE
out_dir=$OUT_DIR
steps=5
seed=0
backend=H3_A6000_SOL_ATTN
network=none
one_visible_gpu_guard=container asserts torch.cuda.device_count()==1 and SM86 A6000
gpu_hygiene_preflight=non-dry records nvidia_smi_full.txt,nvidia_smi_compute_apps.csv,gpu_lease_status.txt,disk_preflight.txt,gpu_hygiene_preflight.json and writes gpu_hygiene_blocker.json instead of running if the selected A6000 is not idle
resource_telemetry=non-dry records per-mode host_resource_before.json,host_resource_after.json,gpu_resource_samples.csv,wall_time.json plus root resource_monitor.csv and overall_wall_time.json
warm_lifecycle=one excluded warmup per mode in the same server process as its measured request
principal_variable=MINIMAX_H3_A6000_SOL_ATTN_STRIDE_AWARE_V with diagnostic materialization disabled only for candidate
policy=r8 unchanged; cache off; dense_first_steps=0; dense_first_layers=2
materialized_reference=diagnostic materialization on, stride-aware V off
stride_aware_candidate=diagnostic materialization off, stride-aware V on
telemetry_arm_file=/evidence/<mode>/measure.arm
actual_gpu_copy_time=CUDA events resolved after measured request; host enqueue is reported separately
promotion_threshold_pct=max(1.5,2*0.5072177175606011)=1.5
classification=matched_N1_gate_not_speedup_claim_until_promoted

Would execute only with:
  ARGUS_ALLOW_A6000_SOL_ATTN_STRIDE_AWARE_V_N1=1 bash ports/minimax_h3_a6000/integration/run_sol_attn_h3_stride_aware_v_n1.sh --execute
EOF_PLAN
}

if [[ "$DRY_RUN" == "1" ]]; then
  print_plan
  exit 0
fi

if [[ "${ARGUS_ALLOW_A6000_SOL_ATTN_STRIDE_AWARE_V_N1:-0}" != "1" ]]; then
  echo "ERROR: refusing non-dry Sol-Attn A6000 diagnostic without ARGUS_ALLOW_A6000_SOL_ATTN_STRIDE_AWARE_V_N1=1" >&2
  exit 11
fi

verify_r9_readable_image_provenance() {
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
  if [[ "$title_label" != *"r9 Sol-Attn"* ]]; then
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

record_gpu_hygiene_preflight() {
  nvidia-smi > "$OUT_DIR/nvidia_smi_full.txt"
  nvidia-smi -L > "$OUT_DIR/nvidia_smi_L.txt"
  nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_memory --format=csv > "$OUT_DIR/nvidia_smi_compute_apps.csv" 2> "$OUT_DIR/nvidia_smi_compute_apps.stderr" || true
  df -h "$ROOT" "$MODEL_ROOT" "$OUT_DIR" > "$OUT_DIR/disk_preflight.txt" 2>&1 || true
  if [[ -n "${ARGUS_SKILL_PYTHON:-}" ]]; then
    "$ARGUS_SKILL_PYTHON" -m argus_skill.tools.gpu_lease status > "$OUT_DIR/gpu_lease_status.txt" 2>&1 || true
  else
    python3 -m argus_skill.tools.gpu_lease status > "$OUT_DIR/gpu_lease_status.txt" 2>&1 || true
  fi
  python3 - "$OUT_DIR/gpu_hygiene_preflight.json" "$OUT_DIR/gpu_hygiene_blocker.json" "$GPU_INDEX" "$EXPECTED_UUID" "$GPU_IDLE_MAX_MEMORY_MIB" "$GPU_IDLE_MAX_UTIL_PCT" <<'PY'
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

out_path = Path(sys.argv[1])
blocker_path = Path(sys.argv[2])
gpu_index = sys.argv[3]
expected_uuid = sys.argv[4]
max_mem_mib = float(sys.argv[5])
max_util_pct = float(sys.argv[6])

def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(cmd)}\nstderr={proc.stderr.strip()}")
    return proc

def parse_csv_line(line: str) -> list[str]:
    return [part.strip() for part in line.split(',')]

def parse_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        cleaned = ''.join(ch for ch in value if ch.isdigit() or ch in '.-')
        return float(cleaned) if cleaned else 0.0

gpu_fields = [
    'index', 'uuid', 'name', 'memory.total', 'memory.used', 'memory.free',
    'utilization.gpu', 'utilization.memory', 'temperature.gpu', 'power.draw', 'power.limit', 'pstate',
]
gpu_proc = run(['nvidia-smi', '-i', gpu_index, '--query-gpu=' + ','.join(gpu_fields), '--format=csv,noheader,nounits'])
gpu_values = parse_csv_line(gpu_proc.stdout.strip().splitlines()[0])
gpu = dict(zip(gpu_fields, gpu_values))
compute_proc = run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name,used_memory', '--format=csv,noheader,nounits'], check=False)
apps = []
if compute_proc.returncode == 0:
    for line in compute_proc.stdout.splitlines():
        line = line.strip()
        if not line or 'No running' in line:
            continue
        parts = parse_csv_line(line)
        if len(parts) >= 4:
            apps.append({'pid': parts[0], 'gpu_uuid': parts[1], 'process_name': parts[2], 'used_memory_mib': parts[3]})
selected_apps = [app for app in apps if app.get('gpu_uuid') == gpu.get('uuid')]
meminfo = {}
try:
    for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines():
        key, rest = line.split(':', 1)
        meminfo[key] = rest.strip()
except OSError as exc:
    meminfo['error'] = str(exc)
paths = []
for raw in (os.environ.get('ROOT'), os.environ.get('MODEL_ROOT'), os.environ.get('OUT_DIR')):
    if raw:
        try:
            usage = shutil.disk_usage(raw)
            paths.append({'path': raw, 'total_bytes': usage.total, 'used_bytes': usage.used, 'free_bytes': usage.free})
        except OSError as exc:
            paths.append({'path': raw, 'error': str(exc)})
blockers = []
if expected_uuid and gpu.get('uuid') != expected_uuid:
    blockers.append({'kind': 'uuid_mismatch', 'actual_uuid': gpu.get('uuid'), 'expected_uuid': expected_uuid})
if 'a6000' not in str(gpu.get('name', '')).lower():
    blockers.append({'kind': 'not_a6000', 'name': gpu.get('name')})
mem_used = parse_float(gpu.get('memory.used', '0'))
util_gpu = parse_float(gpu.get('utilization.gpu', '0'))
if selected_apps:
    blockers.append({'kind': 'compute_apps_present_on_selected_gpu', 'selected_compute_apps': selected_apps})
if mem_used > max_mem_mib:
    blockers.append({'kind': 'memory_used_above_idle_threshold', 'memory_used_mib': mem_used, 'threshold_mib': max_mem_mib})
if util_gpu > max_util_pct:
    blockers.append({'kind': 'utilization_above_idle_threshold', 'utilization_gpu_pct': util_gpu, 'threshold_pct': max_util_pct})
record = {
    'schema_version': 'minimax_h3_a6000_gpu_hygiene_preflight_v1',
    'timestamp_unix': time.time(),
    'gpu_index': gpu_index,
    'expected_uuid': expected_uuid or None,
    'idle_thresholds': {'max_memory_used_mib': max_mem_mib, 'max_utilization_gpu_pct': max_util_pct, 'requires_no_compute_apps': True},
    'selected_gpu': gpu,
    'selected_compute_apps': selected_apps,
    'all_compute_apps': apps,
    'host_meminfo': meminfo,
    'disk_usage': paths,
    'status': 'blocked' if blockers else 'idle_ok',
    'blockers': blockers,
    'recheck_condition': 'rerun after nvidia-smi shows the selected A6000 has no compute apps, low idle memory/utilization, matching UUID if set, and enough disk space',
}
out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
if blockers:
    blocker_path.write_text(json.dumps({
        'schema_version': 'minimax_h3_a6000_gpu_hygiene_blocker_v1',
        'status': 'blocked',
        'reason': 'selected_gpu_not_legally_idle',
        'blockers': blockers,
        'selected_gpu': gpu,
        'selected_compute_apps': selected_apps,
        'recheck_condition': record['recheck_condition'],
    }, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    raise SystemExit(13)
PY
}

record_resource_snapshot() {
  local output=$1
  local mode=$2
  local phase=$3
  python3 - "$output" "$mode" "$phase" "$GPU_INDEX" <<'PY'
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

out = Path(sys.argv[1])
mode = sys.argv[2]
phase = sys.argv[3]
gpu_index = sys.argv[4]
fields = [
    'timestamp', 'index', 'uuid', 'name', 'utilization.gpu', 'utilization.memory',
    'memory.total', 'memory.used', 'memory.free', 'temperature.gpu', 'power.draw', 'power.limit', 'pstate',
]
record: dict[str, object] = {
    'schema_version': 'minimax_h3_a6000_resource_snapshot_v1',
    'mode': mode,
    'phase': phase,
    'timestamp_unix': time.time(),
}
try:
    proc = subprocess.run(
        ['nvidia-smi', '-i', gpu_index, '--query-gpu=' + ','.join(fields), '--format=csv,noheader,nounits'],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=True,
    )
    values = [part.strip() for part in proc.stdout.strip().splitlines()[0].split(',')]
    record['selected_gpu'] = dict(zip(fields, values))
except Exception as exc:  # evidence should capture telemetry failure without hiding the run result
    record['selected_gpu_error'] = str(exc)
try:
    meminfo = {}
    for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines():
        key, rest = line.split(':', 1)
        meminfo[key] = rest.strip()
    record['host_meminfo'] = meminfo
except OSError as exc:
    record['host_meminfo_error'] = str(exc)
out.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY
}

start_mode_resource_sampler() {
  local mode_dir=$1
  nvidia-smi -i "$GPU_INDEX" \
    --query-gpu=timestamp,index,uuid,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu,power.draw,power.limit,pstate \
    --format=csv,nounits -l "$GPU_RESOURCE_SAMPLE_INTERVAL_S" \
    > "$mode_dir/gpu_resource_samples.csv" 2> "$mode_dir/gpu_resource_samples.stderr" &
  echo $!
}

start_overall_resource_monitor() {
  nvidia-smi -i "$GPU_INDEX" \
    --query-gpu=timestamp,index,name,memory.used,memory.total,power.draw,temperature.gpu,utilization.gpu,utilization.memory \
    --format=csv,nounits -l "$GPU_RESOURCE_SAMPLE_INTERVAL_S" \
    > "$OUT_DIR/resource_monitor.csv" 2> "$OUT_DIR/resource_monitor.stderr" &
  echo $!
}

stop_resource_sampler() {
  local pid=${1:-}
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

write_wall_time_json() {
  local output=$1
  local label=$2
  local rc=$3
  local start_ns=$4
  local end_ns=$5
  local start_iso=$6
  local end_iso=$7
  python3 - "$output" "$label" "$rc" "$start_ns" "$end_ns" "$start_iso" "$end_iso" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
label = sys.argv[2]
rc = int(sys.argv[3])
start_ns = int(sys.argv[4])
end_ns = int(sys.argv[5])
record = {
    'schema_version': 'minimax_h3_a6000_wall_time_v1',
    'label': label,
    'return_code': rc,
    'start_epoch_ns': start_ns,
    'end_epoch_ns': end_ns,
    'start_utc': sys.argv[6],
    'end_utc': sys.argv[7],
    'duration_s': (end_ns - start_ns) / 1_000_000_000,
}
out.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY
}

mkdir -p "$OUT_DIR"
verify_r9_readable_image_provenance > "$OUT_DIR/r9_image_identity.env"
install -m 0644 "$PROMPT_FILE" "$OUT_DIR/prompt.txt"
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
comparison=r8_materialized_reference_vs_stride_aware_v
lane=matched_n1_5step_sol_attn_opt_in_not_bf16_fidelity
sol_attn_cache=off
sol_attn_dense_first_steps=0
sol_attn_dense_first_layers=2
warm_lifecycle=one_excluded_warmup_then_one_measured_request_per_mode
promotion_threshold_pct=1.5
network=none
resource_telemetry=cuda_event_copy_attention_denoise+http_e2e+host_gpu_memory_temperature_power_wall_time
EOF_WORKLOAD
record_gpu_hygiene_preflight

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
  local mode_dir="$OUT_DIR/$mode"
  local sampler_pid=""
  local start_ns end_ns start_iso end_iso rc
  mkdir -p "$mode_dir"
  chmod 0777 "$OUT_DIR" "$mode_dir"
  record_resource_snapshot "$mode_dir/host_resource_before.json" "$mode" before
  start_ns=$(python3 - <<'PY'
import time
print(time.time_ns())
PY
)
  start_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  sampler_pid=$(start_mode_resource_sampler "$mode_dir")
  set +e
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
request() {
  local stem=\$1
  curl --fail-with-body --silent --show-error --max-time 3000 \
    --dump-header \"$evidence/\${stem}_headers.txt\" \
    --write-out 'http_code=%{http_code}\ntime_total_s=%{time_total}\nsize_download=%{size_download}\n' \
    -X POST http://127.0.0.1:'$PORT'/v1/videos/sync \
    -F 'prompt=</evidence/prompt.txt' \
    -F 'width=1344' -F 'height=768' -F 'aspect_ratio=16:9' -F 'fps=24' \
    -F 'num_inference_steps=5' -F 'flow_shift=12' -F 'seed=0' -F 'quality=lossless' \
    -F 'extra_params={\"task\":\"t2va\",\"duration\":5.166667,\"audio_flow_shift\":3.0}' \
    -o \"$evidence/\${stem}.mp4\" > \"$evidence/\${stem}_http_metrics.txt\"
}
request warmup
printf '%s\n' armed > '$evidence/measure.arm'
request output
cp '$evidence/output_http_metrics.txt' '$evidence/http_metrics.txt'
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
  rc=$?
  set -e
  stop_resource_sampler "$sampler_pid"
  end_ns=$(python3 - <<'PY'
import time
print(time.time_ns())
PY
)
  end_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  record_resource_snapshot "$mode_dir/host_resource_after.json" "$mode" after
  write_wall_time_json "$mode_dir/wall_time.json" "$mode" "$rc" "$start_ns" "$end_ns" "$start_iso" "$end_iso"
  return "$rc"
}

OVERALL_SAMPLER_PID=""
cleanup_overall_sampler() {
  stop_resource_sampler "${OVERALL_SAMPLER_PID:-}"
}
trap cleanup_overall_sampler EXIT

overall_start_ns=$(python3 - <<'PY'
import time
print(time.time_ns())
PY
)
overall_start_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
OVERALL_SAMPLER_PID=$(start_overall_resource_monitor)

set +e
common_sol_env=(
  -e MINIMAX_H3_A6000_ENABLE_OVERLAY=1
  -e MINIMAX_H3_A6000_ENABLE_TRITON_CANDIDATES=1
  -e MINIMAX_H3_A6000_ENABLE_SOL_ATTN=1
  -e MINIMAX_H3_A6000_SOL_ATTN_CACHE=0
  -e MINIMAX_H3_A6000_SOL_ATTN_STRICT=0
  -e MINIMAX_H3_A6000_SOL_ATTN_DENSE_FIRST_STEPS=0
  -e MINIMAX_H3_A6000_SOL_ATTN_DENSE_FIRST_LAYERS=2
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ADALN=0
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ROPE=0
  -e MINIMAX_H3_A6000_ENABLE_FUSED_SWIGLU=0
  -e MINIMAX_H3_A6000_ENABLE_TELEMETRY=1
  -e MINIMAX_H3_A6000_TELEMETRY_ATEXIT=1
)
run_one r8_materialized_reference \
  "${common_sol_env[@]}" \
  -e MINIMAX_H3_A6000_SOL_ATTN_STRIDE_AWARE_V=0 \
  -e MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_MATERIALIZE=1 \
  -e MINIMAX_H3_A6000_SOL_ATTN_MATERIALIZE_MAX_BYTES=1073741824 \
  -e MINIMAX_H3_A6000_SOL_ATTN_TELEMETRY_ARM_FILE=/evidence/r8_materialized_reference/measure.arm \
  -e MINIMAX_H3_A6000_SOL_ATTN_TELEMETRY_JSON=/evidence/r8_materialized_reference/sol_attn_telemetry
materialized_rc=$?
stride_rc=0
if [[ "$materialized_rc" -eq 0 ]]; then
  run_one stride_aware_v \
    "${common_sol_env[@]}" \
    -e MINIMAX_H3_A6000_SOL_ATTN_STRIDE_AWARE_V=1 \
    -e MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_MATERIALIZE=0 \
    -e MINIMAX_H3_A6000_SOL_ATTN_MATERIALIZE_MAX_BYTES=1073741824 \
    -e MINIMAX_H3_A6000_SOL_ATTN_TELEMETRY_ARM_FILE=/evidence/stride_aware_v/measure.arm \
    -e MINIMAX_H3_A6000_SOL_ATTN_TELEMETRY_JSON=/evidence/stride_aware_v/sol_attn_telemetry
  stride_rc=$?
fi
set -e
stop_resource_sampler "$OVERALL_SAMPLER_PID"
OVERALL_SAMPLER_PID=""
overall_end_ns=$(python3 - <<'PY'
import time
print(time.time_ns())
PY
)
overall_end_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
combined_rc=$materialized_rc
if [[ "$combined_rc" -eq 0 ]]; then
  combined_rc=$stride_rc
fi
write_wall_time_json "$OUT_DIR/overall_wall_time.json" overall "$combined_rc" "$overall_start_ns" "$overall_end_ns" "$overall_start_iso" "$overall_end_iso"

if [[ "$materialized_rc" -ne 0 ]]; then
  echo "ERROR: r8 materialized reference failed with rc=$materialized_rc; evidence in $OUT_DIR" >&2
  exit "$materialized_rc"
fi
if [[ "$stride_rc" -ne 0 ]]; then
  echo "ERROR: stride-aware V candidate failed with rc=$stride_rc; evidence in $OUT_DIR" >&2
  exit "$stride_rc"
fi

python3 - "$OUT_DIR" <<'PY'
from __future__ import annotations
import csv, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
reference_name = 'r8_materialized_reference'
candidate_name = 'stride_aware_v'
reference_dir = root / reference_name
candidate_dir = root / candidate_name


def read_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def parse_http(path: pathlib.Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if '=' not in line:
            continue
        key, value = line.split('=', 1)
        if key == 'http_code' or key == 'size_download':
            out[key] = int(float(value))
        elif key == 'time_total_s':
            out[key] = float(value)
    return out


def peak_memory(path: pathlib.Path) -> float | None:
    values = []
    with path.open(newline='', errors='replace') as f:
        for row in csv.DictReader(f):
            for key, value in row.items():
                if key and 'memory.used' in key.lower():
                    try:
                        values.append(float(value))
                    except (TypeError, ValueError):
                        pass
    return max(values) if values else None


def structural_av(av: dict) -> bool:
    return (
        av.get('video_present') is True and av.get('audio_present') is True
        and int(av.get('width', -1)) == 1344 and int(av.get('height', -1)) == 768
        and int(av.get('decoded_video_frames', -1)) == 124
        and int(av.get('audio_sample_rate', -1)) == 32000
        and int(av.get('audio_channels', -1)) == 2
        and int(av.get('decoded_audio_samples', 0)) > 0
    )


required = [
    root / 'gpu_hygiene_preflight.json', root / 'nvidia_smi_full.txt',
    root / 'nvidia_smi_compute_apps.csv', root / 'gpu_lease_status.txt',
    root / 'disk_preflight.txt', root / 'resource_monitor.csv', root / 'overall_wall_time.json',
]
for mode_dir in (reference_dir, candidate_dir):
    required.extend([
        mode_dir / 'warmup_http_metrics.txt', mode_dir / 'http_metrics.txt',
        mode_dir / 'av_validation.json', mode_dir / 'sol_attn_telemetry.sol_attn.json',
        mode_dir / 'host_resource_before.json', mode_dir / 'host_resource_after.json',
        mode_dir / 'gpu_resource_samples.csv', mode_dir / 'wall_time.json',
    ])
missing = [str(path.relative_to(root)) for path in required if not path.exists() or path.stat().st_size == 0]
if missing:
    decision = {
        'schema_version': 'minimax_h3_a6000_sol_attn_stride_aware_v_n1_v1',
        'classification': 'blocked_incomplete_artifacts',
        'missing_paths': missing,
        'promote_to_n3': False,
        'not_speedup_claim': True,
    }
else:
    ref_av, cand_av = read_json(reference_dir / 'av_validation.json'), read_json(candidate_dir / 'av_validation.json')
    ref_tel = read_json(reference_dir / 'sol_attn_telemetry.sol_attn.json')
    cand_tel = read_json(candidate_dir / 'sol_attn_telemetry.sol_attn.json')
    ref_http, cand_http = parse_http(reference_dir / 'http_metrics.txt'), parse_http(candidate_dir / 'http_metrics.txt')
    ref_warm, cand_warm = parse_http(reference_dir / 'warmup_http_metrics.txt'), parse_http(candidate_dir / 'warmup_http_metrics.txt')
    ref_memory = peak_memory(reference_dir / 'gpu_resource_samples.csv')
    cand_memory = peak_memory(candidate_dir / 'gpu_resource_samples.csv')
    improvement_pct = (ref_http['time_total_s'] - cand_http['time_total_s']) / ref_http['time_total_s'] * 100.0
    observed_r8_cv_pct = 0.5072177175606011
    promotion_threshold_pct = max(1.5, 2.0 * observed_r8_cv_pct)
    layout_samples = cand_tel.get('layout_samples', [])
    real_h3_v_layout_seen = any(
        any(
            t.get('name') == 'value' and t.get('shape') == [1, 38272, 56, 128]
            and t.get('stride') == [823001088, 21504, 128, 1]
            and t.get('storage_offset') == 14336 and t.get('is_contiguous') is False
            for t in sample.get('tensors', [])
        )
        for sample in layout_samples
    )
    av_fields = ('width', 'height', 'average_rate', 'decoded_video_frames', 'audio_sample_rate',
                 'audio_channels', 'decoded_audio_frames', 'decoded_audio_samples')
    av_metadata_equal = all(ref_av.get(key) == cand_av.get(key) for key in av_fields)
    output_exact = ref_av.get('sha256') == cand_av.get('sha256')
    gates = {
        'both_http_200': ref_http.get('http_code') == 200 and cand_http.get('http_code') == 200,
        'both_warmups_http_200': ref_warm.get('http_code') == 200 and cand_warm.get('http_code') == 200,
        'both_structural_av_valid': structural_av(ref_av) and structural_av(cand_av),
        'measured_output_hash_equal': output_exact,
        'measured_av_metadata_equal': av_metadata_equal,
        'both_sparse_calls_192': int(ref_tel.get('sparse_calls', 0)) == 192 and int(cand_tel.get('sparse_calls', 0)) == 192,
        'both_sparse_candidates_192': int(ref_tel.get('sparse_candidate_calls', 0)) == 192 and int(cand_tel.get('sparse_candidate_calls', 0)) == 192,
        'both_fallback_calls_zero': int(ref_tel.get('fallback_calls', -1)) == 0 and int(cand_tel.get('fallback_calls', -1)) == 0,
        'reference_only_materialized_v': (
            int(ref_tel.get('materialize_copy_count', 0)) == 192
            and int(ref_tel.get('materialize_copy_bytes', 0)) == 105344139264
            and ref_tel.get('materialize_copy_by_tensor') == {'value': 105344139264}
        ),
        'candidate_zero_input_copy_events_bytes': (
            int(cand_tel.get('materialize_copy_count', -1)) == 0
            and int(cand_tel.get('materialize_copy_bytes', -1)) == 0
            and int(cand_tel.get('input_copy_events', -1)) == 0
            and int(cand_tel.get('input_copy_bytes', -1)) == 0
            and cand_tel.get('input_copy_by_tensor') == {}
        ),
        'candidate_stride_aware_value_calls_192': int(cand_tel.get('stride_aware_value_calls', 0)) == 192,
        'real_h3_fused_value_layout_seen': real_h3_v_layout_seen,
        'reference_actual_gpu_copy_time_present': (
            float(ref_tel.get('materialize_gpu_copy_latency_ms', 0.0)) > 0.0
            and int(ref_tel.get('materialize_gpu_timing_failures', -1)) == 0
        ),
        'candidate_gpu_copy_time_zero': float(cand_tel.get('materialize_gpu_copy_latency_ms', -1.0)) == 0.0,
        'attention_gpu_timing_complete': (
            int(ref_tel.get('sparse_attention_timed_calls', 0)) == 192
            and int(cand_tel.get('sparse_attention_timed_calls', 0)) == 192
            and float(ref_tel.get('sparse_attention_gpu_latency_ms', 0.0)) > 0.0
            and float(cand_tel.get('sparse_attention_gpu_latency_ms', 0.0)) > 0.0
        ),
        'denoise_gpu_timing_complete': (
            int(ref_tel.get('denoise_timed_calls', 0)) > 0 and int(cand_tel.get('denoise_timed_calls', 0)) > 0
            and float(ref_tel.get('denoise_gpu_latency_ms', 0.0)) > 0.0
            and float(cand_tel.get('denoise_gpu_latency_ms', 0.0)) > 0.0
        ),
        'no_gpu_timing_failures': all(int(t.get(key, -1)) == 0 for t in (ref_tel, cand_tel) for key in (
            'materialize_gpu_timing_failures', 'sparse_attention_gpu_timing_failures', 'denoise_gpu_timing_failures')),
        'candidate_peak_memory_not_higher': (
            ref_memory is not None and cand_memory is not None and cand_memory <= ref_memory
        ),
        'e2e_signal_exceeds_predeclared_threshold': improvement_pct > promotion_threshold_pct,
    }
    correctness_gate_names = [key for key in gates if key != 'e2e_signal_exceeds_predeclared_threshold']
    correctness_ok = all(gates[key] for key in correctness_gate_names)
    if not correctness_ok:
        classification = 'rejected_stride_aware_correctness_or_contract'
    elif not gates['e2e_signal_exceeds_predeclared_threshold']:
        classification = 'rejected_no_above_noise_product_signal'
    else:
        classification = 'promote_to_matched_n3'
    decision = {
        'schema_version': 'minimax_h3_a6000_sol_attn_stride_aware_v_n1_v1',
        'classification': classification,
        'promote_to_n3': classification == 'promote_to_matched_n3',
        'not_speedup_claim': True,
        'lane': 'matched_n1_5step_sol_attn_opt_in_not_bf16_fidelity',
        'principal_variable': 'stride-aware direct V loads instead of r8 V materialization',
        'observed_r8_cv_pct': observed_r8_cv_pct,
        'promotion_threshold_pct': promotion_threshold_pct,
        'http_e2e_seconds': {'r8_materialized_reference': ref_http['time_total_s'], 'stride_aware_v': cand_http['time_total_s']},
        'excluded_warmup_http_seconds': {'r8_materialized_reference': ref_warm['time_total_s'], 'stride_aware_v': cand_warm['time_total_s']},
        'n1_http_e2e_improvement_pct': improvement_pct,
        'gpu_component_ms': {
            mode: {key: tel.get(key) for key in ('materialize_gpu_copy_latency_ms', 'materialize_host_enqueue_latency_ms',
                    'sparse_attention_gpu_latency_ms', 'denoise_gpu_latency_ms')}
            for mode, tel in ((reference_name, ref_tel), (candidate_name, cand_tel))
        },
        'peak_gpu_memory_mib': {reference_name: ref_memory, candidate_name: cand_memory},
        'telemetry_counts': {
            mode: {key: tel.get(key) for key in ('sparse_candidate_calls', 'sparse_calls', 'fallback_calls',
                    'materialize_copy_count', 'materialize_copy_bytes', 'stride_aware_value_calls')}
            for mode, tel in ((reference_name, ref_tel), (candidate_name, cand_tel))
        },
        'output_checks': {'structural_av': True, 'av_metadata_equal': av_metadata_equal, 'sha256_equal': output_exact},
        'gates': gates,
        'failed_gates': [key for key, passed in gates.items() if not passed],
        'claim_boundary': 'N=1 promotion gate only; no acceleration, BF16-fidelity, long-video, or quality-equivalence claim.',
    }
(root / 'decision.json').write_text(json.dumps(decision, indent=2, sort_keys=True) + '\n')
print(json.dumps({'classification': decision['classification'], 'promote_to_n3': decision.get('promote_to_n3', False), 'decision': str(root / 'decision.json')}, sort_keys=True))
PY
