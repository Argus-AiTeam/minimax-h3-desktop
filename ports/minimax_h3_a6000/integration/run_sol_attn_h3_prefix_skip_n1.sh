#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# External single-A6000 r9 full-prefix-block-skip matched N=1 gate.
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
OUT_DIR=${OUT_DIR:-$ROOT/technical_report/evidence/minimax_h3_desktop/sol_engine_port/sol_attn_h3_prefix_skip_r9_n1_$(date -u +%Y%m%dT%H%M%SZ)}
PORT=${PORT:-8000}
GPU_IDLE_MAX_MEMORY_MIB=${GPU_IDLE_MAX_MEMORY_MIB:-512}
GPU_IDLE_MAX_UTIL_PCT=${GPU_IDLE_MAX_UTIL_PCT:-5}
GPU_RESOURCE_SAMPLE_INTERVAL_S=${GPU_RESOURCE_SAMPLE_INTERVAL_S:-5}
VLLM_OMNI_INIT_TIMEOUT_S=${VLLM_OMNI_INIT_TIMEOUT_S:-2400}
VLLM_OMNI_STAGE_INIT_TIMEOUT_S=${VLLM_OMNI_STAGE_INIT_TIMEOUT_S:-1800}
SERVER_READY_POLL_INTERVAL_S=${SERVER_READY_POLL_INTERVAL_S:-5}
SERVER_READY_TIMEOUT_S=${SERVER_READY_TIMEOUT_S:-$((VLLM_OMNI_INIT_TIMEOUT_S + 600))}
VIDEO_SYNC_TIMEOUT_S=${VIDEO_SYNC_TIMEOUT_S:-3600}
REQUEST_TIMEOUT_S=${REQUEST_TIMEOUT_S:-3000}
HOST_PYTHON=${HOST_PYTHON:-$ROOT/.venv/bin/python}
if [[ ! -x "$HOST_PYTHON" ]]; then
  HOST_PYTHON=python3
fi
if [[ "$OUT_DIR" != /* ]]; then
  OUT_DIR="$ROOT/$OUT_DIR"
fi
DRY_RUN=1

usage() {
  cat <<'EOF_USAGE'
Usage: run_sol_attn_h3_prefix_skip_n1.sh [--dry-run|--execute]

Dry-run is the default and prints the exact external A6000 plan without GPU
execution, Docker execution, CUDA, nvidia-smi, network access, downloads, model
loading, inference, cache enablement, or publication. Non-dry execution requires
ARGUS_ALLOW_A6000_SOL_ATTN_PREFIX_SKIP_N1=1 plus readable fresh r9 image
version/base/title label checks, a fresh GPU-hygiene preflight showing a truly
idle selected A6000, and is outside CPU/static stages.

The gate compares the retained Sol-Attn lane with
MINIMAX_H3_A6000_SOL_ATTN_SKIP_FULL_PREFIX_BLOCKS=0 vs =1 as the only principal
variable:
  1. skip_off_a: retained sparse policy except full-prefix-block skip disabled;
  2. skip_off_b: same skip-off lane for decoded-AV and attention-output stability;
  3. skip_on: same lane with full-prefix-block skip enabled.
All modes keep cache off, stride-aware V on, pair-value-halves off, diagnostic
materialization off, dense-prefix overwrite preserved, exact-prefix-query off,
static/bitmask scheduler off, prompt/seed/workload fixed, one excluded warmup,
and one measured request with bounded output-digest diagnostics.  This is a
matched N=1 diagnostic only; promotion requires exact current-vs-current control,
exact skip-off vs skip-on decoded media and attention-output digests, sparse and
copy telemetry gates, resource samples, and an above-threshold N=1 signal.
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
vllm_omni_init_timeout_s=$VLLM_OMNI_INIT_TIMEOUT_S
vllm_omni_stage_init_timeout_s=$VLLM_OMNI_STAGE_INIT_TIMEOUT_S
server_ready_timeout_s=$SERVER_READY_TIMEOUT_S
server_ready_poll_interval_s=$SERVER_READY_POLL_INTERVAL_S
video_sync_timeout_s=$VIDEO_SYNC_TIMEOUT_S
request_timeout_s=$REQUEST_TIMEOUT_S
host_python_for_finalizer=$HOST_PYTHON
warm_lifecycle=one excluded warmup per mode in the same server process as its measured request
principal_variable=MINIMAX_H3_A6000_SOL_ATTN_SKIP_FULL_PREFIX_BLOCKS=0_vs_1
policy=retained r9 sparse lane; cache off; stride-aware V on; pair-value-halves off; diagnostic materialization off; dense-prefix overwrite preserved; dense_first_steps=0; dense_first_layers=2; exact-prefix-query/static-prefix-sink/bitmask off
skip_off_a=skip_full_prefix_blocks off; output digest diagnostic on
skip_off_b=skip_full_prefix_blocks off; output digest diagnostic on; current-vs-current control
skip_on=skip_full_prefix_blocks on; output digest diagnostic on
telemetry_arm_file=/evidence/<mode>/measure.arm
copy_contract=no diagnostic materialization in any mode; materialize/input copy counters must remain zero
attention_output_digest_contract=bounded per-call output SHA/tolerance metadata only; no raw tensor export
promotion_threshold_pct=max(1.5,2*0.5072177175606011)=1.5
classification=matched_N1_gate_not_speedup_claim_until_promoted

Would execute only with:
  ARGUS_ALLOW_A6000_SOL_ATTN_PREFIX_SKIP_N1=1 bash ports/minimax_h3_a6000/integration/run_sol_attn_h3_prefix_skip_n1.sh --execute
EOF_PLAN
}

if [[ "$DRY_RUN" == "1" ]]; then
  print_plan
  exit 0
fi

if [[ "${ARGUS_ALLOW_A6000_SOL_ATTN_PREFIX_SKIP_N1:-0}" != "1" ]]; then
  echo "ERROR: refusing non-dry Sol-Attn A6000 prefix-skip gate without ARGUS_ALLOW_A6000_SOL_ATTN_PREFIX_SKIP_N1=1" >&2
  exit 11
fi

write_empty_docker_ps_sentinel() {
  local path=$1
  local phase=$2
  if [[ ! -s "$path" ]]; then
    printf '{"argus_no_running_containers":true,"phase":"%s","timestamp_utc":"%s","command":"docker ps --format {{json .}}"}\n' \
      "$phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$path"
  fi
}

record_docker_hygiene_preflight() {
  {
    echo "docker_host=${DOCKER_HOST:-default}"
    docker info --format 'DockerRootDir={{.DockerRootDir}} Driver={{.Driver}} ServerVersion={{.ServerVersion}} CgroupDriver={{.CgroupDriver}}'
  } > "$OUT_DIR/docker_info_summary.txt"
  docker ps --format '{{json .}}' > "$OUT_DIR/docker_ps_before.jsonl"
  write_empty_docker_ps_sentinel "$OUT_DIR/docker_ps_before.jsonl" before
  docker image inspect "$IMAGE" > "$OUT_DIR/r9_image_inspect.json"
}

record_docker_hygiene_after() {
  local docker_ps_rc=0
  docker ps --format '{{json .}}' > "$OUT_DIR/docker_ps_after.jsonl" 2> "$OUT_DIR/docker_ps_after.stderr" || docker_ps_rc=$?
  if [[ "$docker_ps_rc" -eq 0 ]]; then
    write_empty_docker_ps_sentinel "$OUT_DIR/docker_ps_after.jsonl" after
  else
    : > "$OUT_DIR/docker_ps_after.jsonl"
  fi
}

verify_r9_readable_image_provenance() {
  local version_label base_label title_label license_label
  version_label=$(docker image inspect --format '{{ index .Config.Labels "org.opencontainers.image.version" }}' "$IMAGE")
  base_label=$(docker image inspect --format '{{ index .Config.Labels "org.opencontainers.image.base.name" }}' "$IMAGE")
  title_label=$(docker image inspect --format '{{ index .Config.Labels "org.opencontainers.image.title" }}' "$IMAGE")
  license_label=$(docker image inspect --format '{{ index .Config.Labels "org.opencontainers.image.licenses" }}' "$IMAGE")
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
actual_image_license_label=$license_label
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
import json, os, shutil, subprocess, sys, time
from pathlib import Path
out_path = Path(sys.argv[1]); blocker_path = Path(sys.argv[2]); gpu_index = sys.argv[3]; expected_uuid = sys.argv[4]
max_mem_mib = float(sys.argv[5]); max_util_pct = float(sys.argv[6])
def run(cmd, check=True):
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(cmd)}\nstderr={proc.stderr.strip()}")
    return proc
def parse_csv_line(line): return [part.strip() for part in line.split(',')]
def parse_float(value):
    try: return float(value)
    except ValueError:
        cleaned = ''.join(ch for ch in value if ch.isdigit() or ch in '.-')
        return float(cleaned) if cleaned else 0.0
gpu_fields = ['index','uuid','name','memory.total','memory.used','memory.free','utilization.gpu','utilization.memory','temperature.gpu','power.draw','power.limit','pstate']
gpu_proc = run(['nvidia-smi','-i',gpu_index,'--query-gpu=' + ','.join(gpu_fields),'--format=csv,noheader,nounits'])
gpu = dict(zip(gpu_fields, parse_csv_line(gpu_proc.stdout.strip().splitlines()[0])))
compute_proc = run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory','--format=csv,noheader,nounits'], check=False)
apps = []
if compute_proc.returncode == 0:
    for line in compute_proc.stdout.splitlines():
        line = line.strip()
        if not line or 'No running' in line: continue
        parts = parse_csv_line(line)
        if len(parts) >= 4:
            apps.append({'pid': parts[0], 'gpu_uuid': parts[1], 'process_name': parts[2], 'used_memory_mib': parts[3]})
selected_apps = [app for app in apps if app.get('gpu_uuid') == gpu.get('uuid')]
meminfo = {}
try:
    for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines():
        key, rest = line.split(':', 1); meminfo[key] = rest.strip()
except OSError as exc:
    meminfo['error'] = str(exc)
paths = []
for raw in (os.environ.get('ROOT'), os.environ.get('MODEL_ROOT'), os.environ.get('OUT_DIR')):
    if raw:
        try:
            usage = shutil.disk_usage(raw); paths.append({'path': raw, 'total_bytes': usage.total, 'used_bytes': usage.used, 'free_bytes': usage.free})
        except OSError as exc:
            paths.append({'path': raw, 'error': str(exc)})
blockers = []
if expected_uuid and gpu.get('uuid') != expected_uuid: blockers.append({'kind': 'uuid_mismatch', 'actual_uuid': gpu.get('uuid'), 'expected_uuid': expected_uuid})
if 'a6000' not in str(gpu.get('name', '')).lower(): blockers.append({'kind': 'not_a6000', 'name': gpu.get('name')})
mem_used = parse_float(gpu.get('memory.used', '0')); util_gpu = parse_float(gpu.get('utilization.gpu', '0'))
if selected_apps: blockers.append({'kind': 'compute_apps_present_on_selected_gpu', 'selected_compute_apps': selected_apps})
if mem_used > max_mem_mib: blockers.append({'kind': 'memory_used_above_idle_threshold', 'memory_used_mib': mem_used, 'threshold_mib': max_mem_mib})
if util_gpu > max_util_pct: blockers.append({'kind': 'utilization_above_idle_threshold', 'utilization_gpu_pct': util_gpu, 'threshold_pct': max_util_pct})
record = {'schema_version': 'minimax_h3_a6000_gpu_hygiene_preflight_v1', 'timestamp_unix': time.time(), 'gpu_index': gpu_index, 'expected_uuid': expected_uuid or None, 'idle_thresholds': {'max_memory_used_mib': max_mem_mib, 'max_utilization_gpu_pct': max_util_pct, 'requires_no_compute_apps': True}, 'selected_gpu': gpu, 'selected_compute_apps': selected_apps, 'all_compute_apps': apps, 'host_meminfo': meminfo, 'disk_usage': paths, 'status': 'blocked' if blockers else 'idle_ok', 'blockers': blockers, 'recheck_condition': 'rerun after nvidia-smi shows the selected A6000 has no compute apps, low idle memory/utilization, matching UUID if set, and enough disk space'}
out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
if blockers:
    blocker_path.write_text(json.dumps({'schema_version': 'minimax_h3_a6000_gpu_hygiene_blocker_v1', 'status': 'blocked', 'reason': 'selected_gpu_not_legally_idle', 'blockers': blockers, 'selected_gpu': gpu, 'selected_compute_apps': selected_apps, 'recheck_condition': record['recheck_condition']}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    raise SystemExit(13)
PY
}

record_resource_snapshot() {
  local output=$1 mode=$2 phase=$3
  python3 - "$output" "$mode" "$phase" "$GPU_INDEX" <<'PY'
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path
out = Path(sys.argv[1]); mode = sys.argv[2]; phase = sys.argv[3]; gpu_index = sys.argv[4]
fields = ['timestamp','index','uuid','name','utilization.gpu','utilization.memory','memory.total','memory.used','memory.free','temperature.gpu','power.draw','power.limit','pstate']
record = {'schema_version': 'minimax_h3_a6000_resource_snapshot_v1', 'mode': mode, 'phase': phase, 'timestamp_unix': time.time()}
try:
    proc = subprocess.run(['nvidia-smi','-i',gpu_index,'--query-gpu=' + ','.join(fields),'--format=csv,noheader,nounits'], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=True)
    record['selected_gpu'] = dict(zip(fields, [part.strip() for part in proc.stdout.strip().splitlines()[0].split(',')]))
except Exception as exc:
    record['selected_gpu_error'] = str(exc)
try:
    meminfo = {}
    for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines():
        key, rest = line.split(':', 1); meminfo[key] = rest.strip()
    record['host_meminfo'] = meminfo
except OSError as exc:
    record['host_meminfo_error'] = str(exc)
out.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY
}

start_mode_resource_sampler() {
  local mode_dir=$1
  nvidia-smi -i "$GPU_INDEX" --query-gpu=timestamp,index,uuid,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu,power.draw,power.limit,pstate --format=csv,nounits -l "$GPU_RESOURCE_SAMPLE_INTERVAL_S" > "$mode_dir/gpu_resource_samples.csv" 2> "$mode_dir/gpu_resource_samples.stderr" &
  echo $!
}

start_overall_resource_monitor() {
  nvidia-smi -i "$GPU_INDEX" --query-gpu=timestamp,index,name,memory.used,memory.total,power.draw,temperature.gpu,utilization.gpu,utilization.memory --format=csv,nounits -l "$GPU_RESOURCE_SAMPLE_INTERVAL_S" > "$OUT_DIR/resource_monitor.csv" 2> "$OUT_DIR/resource_monitor.stderr" &
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
  local output=$1 label=$2 rc=$3 start_ns=$4 end_ns=$5 start_iso=$6 end_iso=$7
  python3 - "$output" "$label" "$rc" "$start_ns" "$end_ns" "$start_iso" "$end_iso" <<'PY'
from __future__ import annotations
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); label = sys.argv[2]; rc = int(sys.argv[3]); start_ns = int(sys.argv[4]); end_ns = int(sys.argv[5])
out.write_text(json.dumps({'schema_version': 'minimax_h3_a6000_wall_time_v1', 'label': label, 'return_code': rc, 'start_epoch_ns': start_ns, 'end_epoch_ns': end_ns, 'start_utc': sys.argv[6], 'end_utc': sys.argv[7], 'duration_s': (end_ns - start_ns) / 1_000_000_000}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY
}

write_runtime_failure_decision() {
  local off_a_rc=$1 off_b_rc=$2 on_rc=$3
  python3 - "$OUT_DIR" "$off_a_rc" "$off_b_rc" "$on_rc" "$VLLM_OMNI_INIT_TIMEOUT_S" "$VLLM_OMNI_STAGE_INIT_TIMEOUT_S" "$SERVER_READY_TIMEOUT_S" "$SERVER_READY_POLL_INTERVAL_S" "$VIDEO_SYNC_TIMEOUT_S" "$REQUEST_TIMEOUT_S" <<'PY'
from __future__ import annotations
import json, pathlib, sys, time
root = pathlib.Path(sys.argv[1])
return_codes = {'skip_off_a': int(sys.argv[2]), 'skip_off_b': int(sys.argv[3]), 'skip_on': int(sys.argv[4])}
timeout_values = {'vllm_omni_init_timeout_s': int(sys.argv[5]), 'vllm_omni_stage_init_timeout_s': int(sys.argv[6]), 'server_ready_timeout_s': int(sys.argv[7]), 'server_ready_poll_interval_s': int(sys.argv[8]), 'video_sync_timeout_s': int(sys.argv[9]), 'request_timeout_s': int(sys.argv[10])}
def read_text(path, limit_bytes=120_000):
    try: data = path.read_bytes()
    except FileNotFoundError: return ''
    return data[-limit_bytes:].decode('utf-8', errors='replace')
def read_json_if_present(path):
    try: return json.loads(path.read_text(encoding='utf-8'))
    except Exception: return None
first_failed_mode = next((m for m, rc in return_codes.items() if rc != 0), None)
server_logs = {}; startup_failure_seen = False
for mode, rc in return_codes.items():
    log_path = root / mode / 'server.log'; text = read_text(log_path); ready = 'Application startup complete.' in text
    timeout_seen = 'Orchestrator startup timed out' in text or 'did not become ready within' in text or rc == 62 or (rc != 0 and not ready)
    if rc != 0 and timeout_seen: startup_failure_seen = True
    server_logs[mode] = {'path': f'{mode}/server.log', 'exists': log_path.exists(), 'bytes': log_path.stat().st_size if log_path.exists() else 0, 'application_startup_complete': ready, 'orchestrator_startup_timeout_seen': 'Orchestrator startup timed out' in text, 'did_not_become_ready_seen': 'did not become ready within' in text, 'tail_last_200_lines': '\n'.join(text.splitlines()[-200:])}
wall_time = {'overall': read_json_if_present(root / 'overall_wall_time.json')} | {m: read_json_if_present(root / m / 'wall_time.json') for m in return_codes}
resource_paths = ['gpu_hygiene_preflight.json','nvidia_smi_full.txt','nvidia_smi_L.txt','nvidia_smi_compute_apps.csv','gpu_lease_status.txt','disk_preflight.txt','resource_monitor.csv','overall_wall_time.json','r9_image_identity.env','workload.env','docker_info_summary.txt','docker_ps_before.jsonl','docker_ps_after.jsonl','r9_image_inspect.json']
for mode in return_codes:
    resource_paths.extend([f'{mode}/startup_timeout_config.env', f'{mode}/gpu_preflight.json', f'{mode}/host_resource_before.json', f'{mode}/host_resource_after.json', f'{mode}/gpu_resource_samples.csv', f'{mode}/wall_time.json', f'{mode}/server.log'])
resource_evidence = {item: {'exists': (root / item).exists(), 'bytes': (root / item).stat().st_size if (root / item).exists() else 0} for item in resource_paths}
reason = 'extended_startup_failure' if startup_failure_seen else 'runtime_failure_no_promotion'
decision = {'schema_version': 'minimax_h3_a6000_sol_attn_prefix_skip_n1_runtime_failure_v1', 'classification': 'blocked', 'reason': reason, 'failure_type': reason, 'promote_to_matched_n3': False, 'promote_to_n3': False, 'not_speedup_claim': True, 'no_product_speedup_claim': True, 'timestamp_unix': time.time(), 'lane': 'matched_n1_5step_sol_attn_opt_in_not_bf16_fidelity_prefix_skip', 'workload': {'width': 1344, 'height': 768, 'frames': 124, 'fps': 24, 'duration_s': 5.166667, 'steps': 5, 'seed': 0}, 'principal_variable': 'MINIMAX_H3_A6000_SOL_ATTN_SKIP_FULL_PREFIX_BLOCKS=0_vs_1', 'fixed_variables': {'cache': 'off', 'stride_aware_v': 'on', 'dense_prefix_overwrite': 'preserved', 'pair_value_halves': 'off', 'diagnostic_materialization': 'off_for_all_modes', 'diagnostic_output_digest': 'on_for_bounded_gate_only', 'dense_first_steps': 0, 'dense_first_layers': 2}, 'timeout_values': timeout_values, 'return_codes': return_codes, 'first_failed_mode': first_failed_mode, 'server_logs': server_logs, 'wall_time': wall_time, 'resource_evidence': resource_evidence, 'claim_boundary': 'Startup/runtime failure artifact only; no product speedup, BF16-fidelity, long-video, quality-equivalence, or promotion claim.'}
(root / 'decision.json').write_text(json.dumps(decision, indent=2, sort_keys=True) + '\n', encoding='utf-8')
(root / 'RUN_REPORT.md').write_text(f"# r9 Sol-Attn full-prefix-block skip N=1 RUN_REPORT\n\nclassification: blocked\nreason: {reason}\nfirst_failed_mode: {first_failed_mode}\nboundary: no product/BF16/long-video/formal/public/SOTA claim.\n", encoding='utf-8')
print(json.dumps({'classification': 'blocked', 'reason': reason, 'decision': str(root / 'decision.json'), 'promote_to_matched_n3': False}, sort_keys=True))
PY
}

mkdir -p "$OUT_DIR"
record_docker_hygiene_preflight
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
comparison=skip_off_a_vs_skip_off_b_then_skip_on
lane=matched_n1_5step_sol_attn_opt_in_not_bf16_fidelity_prefix_skip
principal_variable=MINIMAX_H3_A6000_SOL_ATTN_SKIP_FULL_PREFIX_BLOCKS_0_vs_1
sol_attn_cache=off
sol_attn_stride_aware_v=on
sol_attn_pair_value_halves=off
sol_attn_dense_prefix_overwrite=preserved
sol_attn_diagnostic_materialize=off_for_all_modes
sol_attn_diagnostic_output_digest=on_for_bounded_gate_only
sol_attn_diagnostic_output_max_calls=256
sol_attn_exact_prefix_query=off
sol_attn_static_prefix_sink=off
sol_attn_bitmask_scheduler=off
sol_attn_dense_first_steps=0
sol_attn_dense_first_layers=2
warm_lifecycle=one_excluded_warmup_then_one_measured_request_per_mode
promotion_threshold_pct=1.5
network=none
resource_telemetry=cuda_event_attention_denoise+http_e2e+host_gpu_memory_temperature_power_wall_time+attention_output_digest
vllm_omni_init_timeout_s=$VLLM_OMNI_INIT_TIMEOUT_S
vllm_omni_stage_init_timeout_s=$VLLM_OMNI_STAGE_INIT_TIMEOUT_S
server_ready_timeout_s=$SERVER_READY_TIMEOUT_S
server_ready_poll_interval_s=$SERVER_READY_POLL_INTERVAL_S
video_sync_timeout_s=$VIDEO_SYNC_TIMEOUT_S
request_timeout_s=$REQUEST_TIMEOUT_S
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
  local sampler_pid="" start_ns end_ns start_iso end_iso rc
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
    -e VLLM_OMNI_VIDEO_SYNC_TIMEOUT="$VIDEO_SYNC_TIMEOUT_S" \
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
cat > '$evidence/startup_timeout_config.env' <<'EOF_STARTUP_TIMEOUTS'
vllm_omni_init_timeout_s=$VLLM_OMNI_INIT_TIMEOUT_S
vllm_omni_stage_init_timeout_s=$VLLM_OMNI_STAGE_INIT_TIMEOUT_S
server_ready_timeout_s=$SERVER_READY_TIMEOUT_S
server_ready_poll_interval_s=$SERVER_READY_POLL_INTERVAL_S
video_sync_timeout_s=$VIDEO_SYNC_TIMEOUT_S
request_timeout_s=$REQUEST_TIMEOUT_S
EOF_STARTUP_TIMEOUTS
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
  --init-timeout '$VLLM_OMNI_INIT_TIMEOUT_S' --stage-init-timeout '$VLLM_OMNI_STAGE_INIT_TIMEOUT_S' \
  > '$evidence/server.log' 2>&1 &
server_pid=\$!
trap 'kill \$server_pid 2>/dev/null || true' EXIT
ready_deadline=\$((SECONDS + $SERVER_READY_TIMEOUT_S))
while true; do
  if grep -q 'Application startup complete\.' '$evidence/server.log' && curl --fail --silent http://127.0.0.1:'$PORT'/health >/dev/null; then
    break
  fi
  kill -0 \$server_pid 2>/dev/null || { tail -300 '$evidence/server.log'; exit 61; }
  if (( SECONDS >= ready_deadline )); then
    tail -300 '$evidence/server.log'
    exit 62
  fi
  sleep '$SERVER_READY_POLL_INTERVAL_S'
done
request() {
  local stem=\$1
  curl --fail-with-body --silent --show-error --max-time '$REQUEST_TIMEOUT_S' \
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
record = {'mode': mode, 'steps': 5, 'seed': 0, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size, 'video_present': True, 'audio_present': True, 'width': v.codec_context.width, 'height': v.codec_context.height, 'average_rate': str(v.average_rate), 'decoded_video_frames': vf, 'audio_sample_rate': a.codec_context.sample_rate, 'audio_channels': a.codec_context.channels, 'decoded_audio_frames': af, 'decoded_audio_samples': samples}
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
cleanup_overall_sampler() { stop_resource_sampler "${OVERALL_SAMPLER_PID:-}"; }
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
  -e MINIMAX_H3_A6000_SOL_ATTN_STRIDE_AWARE_V=1
  -e MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_MATERIALIZE=0
  -e MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_OUTPUT_DIGEST=1
  -e MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_MAX_CALLS=256
  -e MINIMAX_H3_A6000_SOL_ATTN_PAIR_VALUE_HALVES=0
  -e MINIMAX_H3_A6000_SOL_ATTN_SHADOW_PAIR_VALUE_HALVES=0
  -e MINIMAX_H3_A6000_SOL_ATTN_SHADOW_ROW_STATE_PROBE=0
  -e MINIMAX_H3_A6000_SOL_ATTN_EXACT_PREFIX_QUERY=0
  -e MINIMAX_H3_A6000_SOL_ATTN_STATIC_PREFIX_SINK=0
  -e MINIMAX_H3_A6000_SOL_ATTN_BITMASK_SCHEDULER=0
  -e MINIMAX_H3_A6000_SOL_ATTN_FORWARD_CONFIG=
  -e MINIMAX_H3_A6000_SOL_ATTN_MATERIALIZE_MAX_BYTES=1073741824
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ADALN=0
  -e MINIMAX_H3_A6000_ENABLE_FUSED_ROPE=0
  -e MINIMAX_H3_A6000_ENABLE_FUSED_SWIGLU=0
  -e MINIMAX_H3_A6000_ENABLE_TELEMETRY=1
  -e MINIMAX_H3_A6000_TELEMETRY_ATEXIT=1
)
run_one skip_off_a \
  "${common_sol_env[@]}" \
  -e MINIMAX_H3_A6000_SOL_ATTN_SKIP_FULL_PREFIX_BLOCKS=0 \
  -e MINIMAX_H3_A6000_SOL_ATTN_TELEMETRY_ARM_FILE=/evidence/skip_off_a/measure.arm \
  -e MINIMAX_H3_A6000_SOL_ATTN_TELEMETRY_JSON=/evidence/skip_off_a/sol_attn_telemetry
off_a_rc=$?
off_b_rc=0
on_rc=0
if [[ "$off_a_rc" -eq 0 ]]; then
  run_one skip_off_b \
    "${common_sol_env[@]}" \
    -e MINIMAX_H3_A6000_SOL_ATTN_SKIP_FULL_PREFIX_BLOCKS=0 \
    -e MINIMAX_H3_A6000_SOL_ATTN_TELEMETRY_ARM_FILE=/evidence/skip_off_b/measure.arm \
    -e MINIMAX_H3_A6000_SOL_ATTN_TELEMETRY_JSON=/evidence/skip_off_b/sol_attn_telemetry
  off_b_rc=$?
fi
if [[ "$off_a_rc" -eq 0 && "$off_b_rc" -eq 0 ]]; then
  run_one skip_on \
    "${common_sol_env[@]}" \
    -e MINIMAX_H3_A6000_SOL_ATTN_SKIP_FULL_PREFIX_BLOCKS=1 \
    -e MINIMAX_H3_A6000_SOL_ATTN_TELEMETRY_ARM_FILE=/evidence/skip_on/measure.arm \
    -e MINIMAX_H3_A6000_SOL_ATTN_TELEMETRY_JSON=/evidence/skip_on/sol_attn_telemetry
  on_rc=$?
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
combined_rc=$off_a_rc
if [[ "$combined_rc" -eq 0 ]]; then combined_rc=$off_b_rc; fi
if [[ "$combined_rc" -eq 0 ]]; then combined_rc=$on_rc; fi
write_wall_time_json "$OUT_DIR/overall_wall_time.json" overall "$combined_rc" "$overall_start_ns" "$overall_end_ns" "$overall_start_iso" "$overall_end_iso"
record_docker_hygiene_after
if [[ -n "${ARGUS_SKILL_PYTHON:-}" ]]; then
  "$ARGUS_SKILL_PYTHON" -m argus_skill.tools.gpu_lease status > "$OUT_DIR/gpu_lease_status_after.json" 2> "$OUT_DIR/gpu_lease_status_after.stderr" || true
else
  python3 -m argus_skill.tools.gpu_lease status > "$OUT_DIR/gpu_lease_status_after.json" 2> "$OUT_DIR/gpu_lease_status_after.stderr" || true
fi

if [[ "$off_a_rc" -ne 0 ]]; then
  write_runtime_failure_decision "$off_a_rc" "$off_b_rc" "$on_rc"
  echo "ERROR: first skip-off lane failed with rc=$off_a_rc; decision in $OUT_DIR/decision.json" >&2
  exit "$off_a_rc"
fi
if [[ "$off_b_rc" -ne 0 ]]; then
  write_runtime_failure_decision "$off_a_rc" "$off_b_rc" "$on_rc"
  echo "ERROR: second skip-off lane failed with rc=$off_b_rc; decision in $OUT_DIR/decision.json" >&2
  exit "$off_b_rc"
fi
if [[ "$on_rc" -ne 0 ]]; then
  write_runtime_failure_decision "$off_a_rc" "$off_b_rc" "$on_rc"
  echo "ERROR: skip-on lane failed with rc=$on_rc; decision in $OUT_DIR/decision.json" >&2
  exit "$on_rc"
fi

"$HOST_PYTHON" "$SCRIPT_DIR/finalize_sol_attn_prefix_skip_diagnostic.py" "$OUT_DIR" \
  --vllm-omni-init-timeout-s "$VLLM_OMNI_INIT_TIMEOUT_S" \
  --vllm-omni-stage-init-timeout-s "$VLLM_OMNI_STAGE_INIT_TIMEOUT_S" \
  --server-ready-timeout-s "$SERVER_READY_TIMEOUT_S" \
  --server-ready-poll-interval-s "$SERVER_READY_POLL_INTERVAL_S" \
  --video-sync-timeout-s "$VIDEO_SYNC_TIMEOUT_S" \
  --request-timeout-s "$REQUEST_TIMEOUT_S"
