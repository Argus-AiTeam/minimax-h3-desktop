#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Bounded r8 Sol-Attn same-GPU matched dense-vs-opt-in retest supervisor.
# Default is dry-run. Non-dry execution is intended to be launched through
# argus_skill.tools.gpu_lease run --detach so the GPU lease outlives the agent turn.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd -P)}
GPU_INDEX=${GPU_INDEX:-2}
EXPECTED_UUID=${EXPECTED_UUID:-}
N_PAIRS=${N_PAIRS:-3}
IMAGE=${IMAGE:-argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r8-sol-attn-overlay}
REQUIRED_IMAGE_VERSION_LABEL=${REQUIRED_IMAGE_VERSION_LABEL:-r8}
MODEL_ROOT=${MODEL_ROOT:-$ROOT/models/MiniMax-H3}
PAIR_SCRIPT=${PAIR_SCRIPT:-$ROOT/ports/minimax_h3_a6000/integration/run_gpu2_sol_attn_h3_5step_diagnostic.sh}
BASELINE_WARM_CV_PCT=${BASELINE_WARM_CV_PCT:-0.8371622556580874}
TIMING_MIN_IMPROVEMENT_PCT=${TIMING_MIN_IMPROVEMENT_PCT:-3.0}
PRIOR_R8_PEAK_GPU_MEMORY_MIB=${PRIOR_R8_PEAK_GPU_MEMORY_MIB:-27354}
PRIOR_R8_PEAK_TEMPERATURE_C=${PRIOR_R8_PEAK_TEMPERATURE_C:-84}
PRIOR_R8_PEAK_POWER_W=${PRIOR_R8_PEAK_POWER_W:-299.88}
GPU_IDLE_MAX_MEMORY_MIB=${GPU_IDLE_MAX_MEMORY_MIB:-512}
GPU_IDLE_MAX_UTIL_PCT=${GPU_IDLE_MAX_UTIL_PCT:-5}
OUT_DIR=${OUT_DIR:-$ROOT/technical_report/evidence/minimax_h3_desktop/sol_engine_port/sol_attn_h3_matched_retest_r8_n${N_PAIRS}_$(date -u +%Y%m%dT%H%M%SZ)}
if [[ "$OUT_DIR" != /* ]]; then
  OUT_DIR="$ROOT/$OUT_DIR"
fi
DRY_RUN=1

usage() {
  cat <<'EOF_USAGE'
Usage: run_sol_attn_h3_matched_retest_r8.sh [--dry-run|--execute]

Runs a bounded same-GPU matched retest by invoking the accepted r8 5-step
H3 diagnostic gate N_PAIRS times, each pair containing:
  1. dense_h3_backend_reference (Sol-Attn disabled through the same backend)
  2. sol_attn opt-in (Sol-Attn enabled, cache off, diagnostic materialization on)

This is not formal N>=10 and not a BF16 fidelity speedup claim. It only writes
machine-readable evidence and a decision about whether the candidate is eligible
for a separate formal N>=10 delegation.

Non-dry execution requires ARGUS_ALLOW_GPU2_SOL_ATTN_H3_MATCHED_RETEST=1 and
should be launched with:
  ARGUS_SKILL_PYTHON=<private-path> \
  ARGUS_ALLOW_GPU2_SOL_ATTN_H3_MATCHED_RETEST=1 \
  <ARGUS_SKILL_PYTHON> -m argus_skill.tools.gpu_lease run --detach --owner argus-ir04-sol-attn-r8-matched --gpus <gpu-index> -- \
    bash ports/minimax_h3_a6000/integration/run_sol_attn_h3_matched_retest_r8.sh --execute
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
expected_uuid=${EXPECTED_UUID:-<not-set>}
n_pairs=$N_PAIRS
image=$IMAGE
required_image_version_label=$REQUIRED_IMAGE_VERSION_LABEL
model_root=$MODEL_ROOT
pair_script=$PAIR_SCRIPT
out_dir=$OUT_DIR
workload=1344x768,5.166667s,124frames,24FPS,32kHz stereo,prompt=t2va_example_1,steps=5,seed=0
comparison=dense_h3_backend_reference_vs_sol_attn_opt_in
formal_n10=not_run_by_this_script
proceed_gate=HTTP200+structural_AV+sparse_calls>0+fallback_calls=0+density/materialization telemetry+resource envelope+no quality proxy red flags+no slower pair+median improvement > max(3%,2*baseline_cv)
baseline_warm_cv_pct=$BASELINE_WARM_CV_PCT
timing_threshold_pct=max($TIMING_MIN_IMPROVEMENT_PCT, 2*$BASELINE_WARM_CV_PCT)
opaque_identifier_policy=image/output hashes or digests are not used as classification evidence
EOF_PLAN
}

if [[ "$DRY_RUN" == "1" ]]; then
  print_plan
  exit 0
fi

if [[ "${ARGUS_ALLOW_GPU2_SOL_ATTN_H3_MATCHED_RETEST:-0}" != "1" ]]; then
  echo "ERROR: refusing non-dry matched retest without ARGUS_ALLOW_GPU2_SOL_ATTN_H3_MATCHED_RETEST=1" >&2
  exit 11
fi

mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/supervisor_stdout.log") 2>&1
status_json="$OUT_DIR/supervisor_status.json"
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
expected_uuid_json=$(python3 - "$EXPECTED_UUID" <<'PY'
import json, sys
value = sys.argv[1]
print(json.dumps(value if value else None))
PY
)
cat > "$status_json" <<EOF_STATUS
{
  "schema_version": "minimax_h3_a6000_sol_attn_matched_retest_supervisor_status_v1",
  "status": "running",
  "started_utc": "$started_utc",
  "pid": $$,
  "gpu_index": "$GPU_INDEX",
  "expected_uuid": $expected_uuid_json,
  "out_dir": "$OUT_DIR"
}
EOF_STATUS

finish_status() {
  local rc=$?
  local finished_utc
  finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  python3 - "$status_json" "$rc" "$finished_utc" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
rc = int(sys.argv[2])
finished = sys.argv[3]
try:
    data = json.loads(path.read_text())
except Exception:
    data = {"schema_version": "minimax_h3_a6000_sol_attn_matched_retest_supervisor_status_v1"}
data["status"] = "complete" if rc == 0 else "failed"
data["return_code"] = rc
data["finished_utc"] = finished
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
}
trap finish_status EXIT

cat > "$OUT_DIR/commands_env.txt" <<EOF_ENV
schema=minimax_h3_a6000_sol_attn_matched_retest_commands_env_v1
started_utc=$started_utc
root=$ROOT
gpu_index=$GPU_INDEX
expected_uuid=${EXPECTED_UUID:-}
n_pairs=$N_PAIRS
image=$IMAGE
required_image_version_label=$REQUIRED_IMAGE_VERSION_LABEL
model_root=$MODEL_ROOT
pair_script=$PAIR_SCRIPT
baseline_warm_cv_pct=$BASELINE_WARM_CV_PCT
timing_min_improvement_pct=$TIMING_MIN_IMPROVEMENT_PCT
prior_r8_peak_gpu_memory_mib=$PRIOR_R8_PEAK_GPU_MEMORY_MIB
prior_r8_peak_temperature_c=$PRIOR_R8_PEAK_TEMPERATURE_C
prior_r8_peak_power_w=$PRIOR_R8_PEAK_POWER_W
operator_gate=ARGUS_ALLOW_GPU2_SOL_ATTN_H3_MATCHED_RETEST
pair_operator_gate=ARGUS_ALLOW_GPU2_SOL_ATTN_H3_5STEP
formal_n10=not_run
opaque_identifier_policy=opaque image/output identifiers are omitted from decision evidence
EOF_ENV

# Readable image provenance only: labels/tags, not digests.
set +e
{
  echo "image=$IMAGE"
  echo "required_image_version_label=$REQUIRED_IMAGE_VERSION_LABEL"
  echo -n "actual_image_version_label="
  docker image inspect --format '{{ index .Config.Labels "org.opencontainers.image.version" }}' "$IMAGE"
  echo -n "actual_image_base_label="
  docker image inspect --format '{{ index .Config.Labels "org.opencontainers.image.base.name" }}' "$IMAGE"
  echo -n "actual_image_title_label="
  docker image inspect --format '{{ index .Config.Labels "org.opencontainers.image.title" }}' "$IMAGE"
  echo "opaque_image_identifier_policy=omitted_not_evidence"
} > "$OUT_DIR/r8_image_identity.env" 2> "$OUT_DIR/r8_image_identity.stderr"
image_rc=$?
set -e
if [[ "$image_rc" -ne 0 ]]; then
  python3 - "$OUT_DIR" "$image_rc" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rc = int(sys.argv[2])
decision = {
    "schema_version": "minimax_h3_a6000_sol_attn_matched_retest_decision_v1",
    "classification": "blocked_image_provenance_unavailable",
    "reason": "docker image readable r8 label inspection failed before GPU execution",
    "image_inspect_return_code": rc,
    "proceed_to_n10_recommended": False,
    "not_fidelity_or_performance_claim": True,
}
(root / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
PY
  exit 15
fi
if ! grep -q "actual_image_version_label=$REQUIRED_IMAGE_VERSION_LABEL" "$OUT_DIR/r8_image_identity.env"; then
  python3 - "$OUT_DIR" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
decision = {
    "schema_version": "minimax_h3_a6000_sol_attn_matched_retest_decision_v1",
    "classification": "blocked_image_version_label_mismatch",
    "reason": "readable image version label is not the required r8 overlay",
    "proceed_to_n10_recommended": False,
    "not_fidelity_or_performance_claim": True,
}
(root / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
PY
  exit 15
fi

# Fresh selected-GPU/disk/process evidence immediately before Docker/model work.
nvidia-smi > "$OUT_DIR/nvidia_smi_full.txt"
nvidia-smi -L > "$OUT_DIR/nvidia_smi_L.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits > "$OUT_DIR/nvidia_smi_gpu_query.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits > "$OUT_DIR/nvidia_smi_compute_apps.csv" 2> "$OUT_DIR/nvidia_smi_compute_apps.stderr" || true
df -h "$ROOT" "$MODEL_ROOT" "$OUT_DIR" > "$OUT_DIR/disk_preflight.txt" 2>&1 || true
LEASE_PY=${ARGUS_SKILL_PYTHON:-python3}
"$LEASE_PY" -m argus_skill.tools.gpu_lease status > "$OUT_DIR/gpu_lease_status.json" 2> "$OUT_DIR/gpu_lease_status.stderr" || true

set +e
python3 - "$OUT_DIR" "$GPU_INDEX" "$EXPECTED_UUID" "$GPU_IDLE_MAX_MEMORY_MIB" "$GPU_IDLE_MAX_UTIL_PCT" <<'PY'
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path

root = Path(sys.argv[1])
gpu_index = sys.argv[2]
expected_uuid = sys.argv[3]
max_mem = float(sys.argv[4])
max_util = float(sys.argv[5])

def run(cmd, check=False):
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=check)

def parts(line):
    return [p.strip() for p in line.split(',')]

def number(value):
    try:
        return float(value)
    except Exception:
        s = ''.join(ch for ch in str(value) if ch.isdigit() or ch in '.-')
        return float(s) if s else 0.0

fields = ['index','uuid','name','memory.used','utilization.gpu','power.draw','temperature.gpu']
gpu_proc = run(['nvidia-smi','-i',gpu_index,'--query-gpu=' + ','.join(fields),'--format=csv,noheader,nounits'])
record = {
    'schema_version': 'minimax_h3_a6000_matched_retest_gpu_hygiene_preflight_v1',
    'timestamp_unix': time.time(),
    'gpu_index': gpu_index,
    'expected_uuid': expected_uuid or None,
    'idle_thresholds': {'max_memory_used_mib': max_mem, 'max_utilization_gpu_pct': max_util, 'requires_no_compute_apps': True},
    'status': 'blocked',
    'blockers': [],
}
if gpu_proc.returncode != 0:
    record['blockers'].append({'kind':'nvidia_smi_selected_gpu_query_failed','stderr':gpu_proc.stderr.strip()})
    selected = {}
else:
    selected = dict(zip(fields, parts(gpu_proc.stdout.strip().splitlines()[0])))
    record['selected_gpu'] = selected
    if expected_uuid and selected.get('uuid') != expected_uuid:
        record['blockers'].append({'kind':'uuid_mismatch','actual_uuid':selected.get('uuid'),'expected_uuid':expected_uuid})
    if 'a6000' not in str(selected.get('name','')).lower():
        record['blockers'].append({'kind':'not_a6000','name':selected.get('name')})
    if number(selected.get('memory.used', 0)) > max_mem:
        record['blockers'].append({'kind':'memory_used_above_idle_threshold','memory_used_mib':number(selected.get('memory.used',0)),'threshold_mib':max_mem})
    if number(selected.get('utilization.gpu', 0)) > max_util:
        record['blockers'].append({'kind':'utilization_above_idle_threshold','utilization_gpu_pct':number(selected.get('utilization.gpu',0)),'threshold_pct':max_util})

apps_proc = run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'])
apps=[]
if apps_proc.returncode == 0:
    for line in apps_proc.stdout.splitlines():
        line=line.strip()
        if not line or 'No running' in line:
            continue
        p=parts(line)
        if len(p) >= 4:
            apps.append({'gpu_uuid':p[0], 'pid':p[1], 'process_name':p[2], 'used_memory_mib':p[3]})
record['all_compute_apps'] = apps
selected_apps = [a for a in apps if a.get('gpu_uuid') == selected.get('uuid')]
record['selected_compute_apps'] = selected_apps
if selected_apps:
    record['blockers'].append({'kind':'compute_apps_present_on_selected_gpu','selected_compute_apps':selected_apps})
record['status'] = 'blocked' if record['blockers'] else 'idle_ok'
record['recheck_condition'] = 'rerun after selected A6000 has no compute apps, low idle memory/utilization, matching UUID if set, and enough disk space'
(root/'gpu_hygiene_preflight.json').write_text(json.dumps(record, indent=2, sort_keys=True)+'\n')
if record['blockers']:
    (root/'gpu_hygiene_blocker.json').write_text(json.dumps(record, indent=2, sort_keys=True)+'\n')
    decision = {
        'schema_version': 'minimax_h3_a6000_sol_attn_matched_retest_decision_v1',
        'classification': 'blocked_no_truly_idle_selected_a6000',
        'reason': 'fresh preflight found selected GPU unavailable before Docker/model execution',
        'blockers': record['blockers'],
        'proceed_to_n10_recommended': False,
        'not_fidelity_or_performance_claim': True,
    }
    (root/'decision.json').write_text(json.dumps(decision, indent=2, sort_keys=True)+'\n')
    raise SystemExit(13)
PY
preflight_rc=$?
set -e
if [[ "$preflight_rc" -ne 0 ]]; then
  exit "$preflight_rc"
fi

pair_rc=0
: > "$OUT_DIR/pair_runs.tsv"
for pair_num in $(seq 1 "$N_PAIRS"); do
  pair_id=$(printf 'pair%02d' "$pair_num")
  pair_dir="$OUT_DIR/$pair_id"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start $pair_id $pair_dir" | tee -a "$OUT_DIR/pair_runs.tsv"
  set +e
  ARGUS_ALLOW_GPU2_SOL_ATTN_H3_5STEP=1 \
  ROOT="$ROOT" GPU_INDEX="$GPU_INDEX" EXPECTED_UUID="$EXPECTED_UUID" IMAGE="$IMAGE" \
  REQUIRED_IMAGE_VERSION_LABEL="$REQUIRED_IMAGE_VERSION_LABEL" MODEL_ROOT="$MODEL_ROOT" OUT_DIR="$pair_dir" \
  bash "$PAIR_SCRIPT" --execute > "$OUT_DIR/${pair_id}.log" 2>&1
  rc=$?
  set -e
  echo "$rc" > "$pair_dir.exit_code"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end $pair_id rc=$rc" | tee -a "$OUT_DIR/pair_runs.tsv"
  if [[ "$rc" -ne 0 ]]; then
    pair_rc="$rc"
    break
  fi
done

python3 - "$OUT_DIR" "$N_PAIRS" "$BASELINE_WARM_CV_PCT" "$TIMING_MIN_IMPROVEMENT_PCT" "$PRIOR_R8_PEAK_GPU_MEMORY_MIB" "$PRIOR_R8_PEAK_TEMPERATURE_C" "$PRIOR_R8_PEAK_POWER_W" <<'PY'
from __future__ import annotations
import csv, json, math, statistics, sys, time
from pathlib import Path

root = Path(sys.argv[1])
requested_pairs = int(sys.argv[2])
baseline_cv_pct = float(sys.argv[3])
configured_min_pct = float(sys.argv[4])
threshold_pct = max(configured_min_pct, 2.0 * baseline_cv_pct)
prior_mem = float(sys.argv[5])
prior_temp = float(sys.argv[6])
prior_power = float(sys.argv[7])
resource_thresholds = {
    'peak_gpu_memory_mib_max': prior_mem + max(2048.0, prior_mem * 0.10),
    'peak_temperature_c_max': prior_temp + 5.0,
    'peak_power_w_max': max(310.0, prior_power + 10.0),
}

def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

def parse_http(path: Path) -> dict:
    out = {'status': 'missing'}
    if not path.exists():
        return out
    out = {'status': 'present'}
    for line in path.read_text(errors='replace').splitlines():
        if '=' not in line:
            continue
        k, v = line.split('=', 1)
        k = k.strip(); v = v.strip()
        if k in {'http_code', 'size_download'}:
            try: out[k] = int(float(v))
            except Exception: out[k] = v
        elif k == 'time_total_s':
            try: out[k] = float(v)
            except Exception: out[k] = v
        else:
            out[k] = v
    return out

def av_value(av: dict, *names, default=None):
    for n in names:
        if isinstance(av, dict) and n in av:
            return av[n]
    return default

def structural_av_ok(av: dict | None) -> bool:
    if not isinstance(av, dict):
        return False
    return (
        bool(av.get('video_present')) and bool(av.get('audio_present'))
        and int(av_value(av, 'width', default=-1)) == 1344
        and int(av_value(av, 'height', default=-1)) == 768
        and int(av_value(av, 'decoded_video_frames', default=-1)) == 124
        and int(av_value(av, 'audio_sample_rate', 'audio_sample_rate_hz', default=-1)) == 32000
        and int(av_value(av, 'audio_channels', default=-1)) == 2
        and int(av_value(av, 'decoded_audio_samples', default=0)) > 0
    )

def materialize_calls(tel: dict | None) -> int:
    if not isinstance(tel, dict): return 0
    for key in ('materialize_copy_count','materialized_copy_calls','copy_calls'):
        if tel.get(key) is not None:
            try: return int(tel[key])
            except Exception: return 0
    return 0

def materialize_bytes(tel: dict | None) -> int:
    if not isinstance(tel, dict): return 0
    for key in ('materialize_copy_bytes','materialized_copy_bytes','copy_bytes'):
        if tel.get(key) is not None:
            try: return int(tel[key])
            except Exception: return 0
    return 0

def density_count(tel: dict | None) -> int:
    if not isinstance(tel, dict): return 0
    ds = tel.get('density_samples')
    if isinstance(ds, list): return len(ds)
    try: return int(tel.get('density_sample_count', 0))
    except Exception: return 0

def parse_float(value):
    try:
        return float(value)
    except Exception:
        s = ''.join(ch for ch in str(value) if ch.isdigit() or ch in '.-')
        return float(s) if s else None

def normalize_header(name: str) -> str:
    return ''.join(ch for ch in name.lower() if ch.isalnum() or ch == '.')

def resource_from_csv(path: Path) -> dict:
    summary = {'path': str(path.relative_to(root)) if path.exists() else str(path), 'samples': 0}
    if not path.exists() or path.stat().st_size == 0:
        summary['status'] = 'missing'
        return summary
    peaks = {'peak_gpu_memory_mib': None, 'peak_power_w': None, 'peak_temperature_c': None, 'peak_gpu_util_percent': None}
    try:
        with path.open(newline='', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                summary['samples'] += 1
                norm = {normalize_header(k): v for k, v in row.items() if k is not None}
                candidates = {
                    'peak_gpu_memory_mib': ['memory.usedmib','memory.used'],
                    'peak_power_w': ['power.draww','power.draw'],
                    'peak_temperature_c': ['temperature.gpu'],
                    'peak_gpu_util_percent': ['utilization.gpu','utilization.gpupct'],
                }
                for out_key, keys in candidates.items():
                    for key in keys:
                        if key in norm:
                            val = parse_float(norm[key])
                            if val is not None:
                                peaks[out_key] = val if peaks[out_key] is None else max(peaks[out_key], val)
                            break
        summary.update({k: v for k, v in peaks.items() if v is not None})
        summary['status'] = 'present'
    except Exception as exc:
        summary['status'] = 'parse_error'
        summary['error'] = str(exc)
    return summary

pairs=[]
quality_red_flags=[]
timing_improvements=[]
resource_summaries=[]
for idx in range(1, requested_pairs + 1):
    pair_id = f'pair{idx:02d}'
    pair_dir = root / pair_id
    dense_dir = pair_dir / 'dense_h3_backend_reference'
    sol_dir = pair_dir / 'sol_attn'
    tel = read_json(sol_dir / 'sol_attn_telemetry.sol_attn.json')
    dense_av = read_json(dense_dir / 'av_validation.json')
    sol_av = read_json(sol_dir / 'av_validation.json')
    dense_http = parse_http(dense_dir / 'http_metrics.txt')
    sol_http = parse_http(sol_dir / 'http_metrics.txt')
    exit_code_path = root / f'{pair_id}.exit_code'
    exit_code = None
    if exit_code_path.exists():
        try: exit_code = int(exit_code_path.read_text().strip())
        except Exception: exit_code = exit_code_path.read_text().strip()
    dense_t = dense_http.get('time_total_s') if isinstance(dense_http.get('time_total_s'), (int,float)) else None
    sol_t = sol_http.get('time_total_s') if isinstance(sol_http.get('time_total_s'), (int,float)) else None
    improvement = None
    if dense_t and sol_t is not None and dense_t > 0:
        improvement = (dense_t - sol_t) / dense_t * 100.0
        timing_improvements.append(improvement)
    av_keys = ['width','height','decoded_video_frames','audio_channels','decoded_audio_samples']
    for key in av_keys:
        if isinstance(dense_av, dict) and isinstance(sol_av, dict) and av_value(dense_av, key) != av_value(sol_av, key):
            quality_red_flags.append({'pair': pair_id, 'kind': 'av_metadata_mismatch', 'field': key, 'dense': av_value(dense_av, key), 'sol_attn': av_value(sol_av, key)})
    dense_sr = av_value(dense_av or {}, 'audio_sample_rate', 'audio_sample_rate_hz')
    sol_sr = av_value(sol_av or {}, 'audio_sample_rate', 'audio_sample_rate_hz')
    if dense_sr != sol_sr:
        quality_red_flags.append({'pair': pair_id, 'kind': 'av_metadata_mismatch', 'field': 'audio_sample_rate', 'dense': dense_sr, 'sol_attn': sol_sr})
    resource = resource_from_csv(pair_dir / 'resource_monitor.csv')
    resource_summaries.append({'pair': pair_id, **resource})
    pairs.append({
        'pair': pair_id,
        'exit_code': exit_code,
        'dense_http': dense_http,
        'sol_attn_http': sol_http,
        'dense_structural_av_ok': structural_av_ok(dense_av),
        'sol_attn_structural_av_ok': structural_av_ok(sol_av),
        'http_time_total_s': {'dense': dense_t, 'sol_attn': sol_t},
        'http_time_improvement_pct': improvement,
        'telemetry': {
            'sparse_candidate_calls': (tel or {}).get('sparse_candidate_calls'),
            'sparse_calls': (tel or {}).get('sparse_calls'),
            'fallback_calls': (tel or {}).get('fallback_calls'),
            'dense_calls': (tel or {}).get('dense_calls'),
            'decline_reasons': (tel or {}).get('decline_reasons', {}),
            'fallback_reasons': (tel or {}).get('fallback_reasons', {}),
            'density_sample_count': density_count(tel),
            'materialized_copy_calls': materialize_calls(tel),
            'materialized_copy_bytes': materialize_bytes(tel),
        },
        'resource_summary': resource,
        'opaque_output_identifier_policy': 'sha256/output byte identity not used as decision evidence',
    })

completed_pairs = [p for p in pairs if p['exit_code'] == 0]
all_pairs_completed = len(completed_pairs) == requested_pairs
all_http_200 = all(p['dense_http'].get('http_code') == 200 and p['sol_attn_http'].get('http_code') == 200 for p in completed_pairs) and all_pairs_completed
all_structural_av = all(p['dense_structural_av_ok'] and p['sol_attn_structural_av_ok'] for p in completed_pairs) and all_pairs_completed
all_sparse_positive = all(int(p['telemetry'].get('sparse_calls') or 0) > 0 for p in completed_pairs) and all_pairs_completed
all_fallback_zero = all(int(p['telemetry'].get('fallback_calls') or 0) == 0 for p in completed_pairs) and all_pairs_completed
complete_sparse_telemetry = all(
    int(p['telemetry'].get('density_sample_count') or 0) > 0
    and int(p['telemetry'].get('materialized_copy_calls') or 0) > 0
    and int(p['telemetry'].get('materialized_copy_bytes') or 0) > 0
    for p in completed_pairs
) and all_pairs_completed
no_pair_slower = all((p['http_time_improvement_pct'] is not None and p['http_time_improvement_pct'] >= 0.0) for p in completed_pairs) and all_pairs_completed
median_improvement_pct = statistics.median(timing_improvements) if len(timing_improvements) == requested_pairs else None
median_timing_gate = bool(median_improvement_pct is not None and median_improvement_pct > threshold_pct)

peak_gpu_memory = max([r.get('peak_gpu_memory_mib') for r in resource_summaries if isinstance(r.get('peak_gpu_memory_mib'), (int,float))] or [None])
peak_temperature = max([r.get('peak_temperature_c') for r in resource_summaries if isinstance(r.get('peak_temperature_c'), (int,float))] or [None])
peak_power = max([r.get('peak_power_w') for r in resource_summaries if isinstance(r.get('peak_power_w'), (int,float))] or [None])
resource_envelope_comparable = (
    peak_gpu_memory is not None and peak_gpu_memory <= resource_thresholds['peak_gpu_memory_mib_max']
    and peak_temperature is not None and peak_temperature <= resource_thresholds['peak_temperature_c_max']
    and peak_power is not None and peak_power <= resource_thresholds['peak_power_w_max']
)
no_quality_red_flags = len(quality_red_flags) == 0

gates = {
    'requested_pairs': requested_pairs,
    'completed_pairs': len(completed_pairs),
    'all_pairs_completed': all_pairs_completed,
    'all_http_200': all_http_200,
    'all_structural_av_valid': all_structural_av,
    'all_sparse_calls_positive': all_sparse_positive,
    'all_fallback_calls_zero': all_fallback_zero,
    'complete_density_and_materialization_telemetry': complete_sparse_telemetry,
    'resource_envelope_comparable_to_prior_r8': resource_envelope_comparable,
    'no_quality_proxy_red_flags': no_quality_red_flags,
    'no_pair_slower': no_pair_slower,
    'median_improvement_exceeds_threshold': median_timing_gate,
}
failed_gates = [k for k, v in gates.items() if isinstance(v, bool) and not v]
if not all_pairs_completed:
    classification = 'needs_fix_incomplete_matched_retest'
    reason = 'one or more dense/opt-in pair invocations failed or did not produce complete artifacts'
elif not (all_http_200 and all_structural_av):
    classification = 'needs_fix_invalid_http_or_structural_av'
    reason = 'matched retest did not preserve HTTP 200 and structural AV validity for every pair'
elif not (all_sparse_positive and all_fallback_zero and complete_sparse_telemetry):
    classification = 'needs_fix_sparse_runtime_or_telemetry_gate_failed'
    reason = 'opt-in path did not provide sparse_calls>0, zero fallbacks, and complete density/materialization telemetry for every pair'
elif not no_quality_red_flags:
    classification = 'needs_fix_quality_proxy_red_flags'
    reason = 'automatic AV quality proxy found dense-vs-opt-in metadata mismatches'
elif not resource_envelope_comparable:
    classification = 'diagnostic_only_rejected_resource_envelope'
    reason = 'resource envelope exceeded prior-r8-comparable thresholds for this bounded gate'
elif not (no_pair_slower and median_timing_gate):
    classification = 'diagnostic_only_rejected_no_n10_timing_gate'
    reason = 'correctness/sparse execution may hold, but timing gate for formal N10 eligibility did not pass'
else:
    classification = 'proceed_to_formal_n10_candidate'
    reason = 'bounded matched retest passed correctness, sparse-runtime, resource, quality-proxy, and timing gates'

decision = {
    'schema_version': 'minimax_h3_a6000_sol_attn_matched_retest_decision_v1',
    'classification': classification,
    'reason': reason,
    'proceed_to_n10_recommended': classification == 'proceed_to_formal_n10_candidate',
    'not_formal_n10': True,
    'not_fidelity_or_performance_claim': True,
    'lane': 'diagnostic_practical_opt_in_sol_attn_not_bf16_fidelity',
    'baseline_warm_cv_pct': baseline_cv_pct,
    'timing_threshold_pct': threshold_pct,
    'median_http_time_improvement_pct': median_improvement_pct,
    'failed_gates': failed_gates,
    'gates': gates,
    'quality_proxy_red_flags': quality_red_flags,
    'resource_thresholds': resource_thresholds,
    'resource_peak_summary': {
        'peak_gpu_memory_mib': peak_gpu_memory,
        'peak_temperature_c': peak_temperature,
        'peak_power_w': peak_power,
    },
    'opaque_integrity_policy': {
        'image_identifiers': 'readable labels/tags only; digests not used as classification evidence',
        'output_identifiers': 'sha256/output identity not used as classification evidence',
    },
    'pairs': pairs,
    'generated_at_unix': time.time(),
}
(root / 'decision.json').write_text(json.dumps(decision, indent=2, sort_keys=True) + '\n')
(root / 'timing_summary.json').write_text(json.dumps({
    'schema_version': 'minimax_h3_a6000_sol_attn_matched_retest_timing_summary_v1',
    'requested_pairs': requested_pairs,
    'completed_pairs': len(completed_pairs),
    'threshold_pct': threshold_pct,
    'median_http_time_improvement_pct': median_improvement_pct,
    'per_pair_http_time_improvement_pct': [p['http_time_improvement_pct'] for p in pairs],
    'per_pair_http_time_total_s': [{'pair': p['pair'], **p['http_time_total_s']} for p in pairs],
    'not_speedup_claim': True,
}, indent=2, sort_keys=True) + '\n')
(root / 'quality_proxy_comparison.json').write_text(json.dumps({
    'schema_version': 'minimax_h3_a6000_sol_attn_matched_retest_quality_proxy_v1',
    'proxy_scope': 'automatic structural AV/timing-sync metadata only; no human auditory or semantic quality certification',
    'no_quality_proxy_red_flags': no_quality_red_flags,
    'red_flags': quality_red_flags,
    'per_pair_structural_av': [{
        'pair': p['pair'],
        'dense_structural_av_ok': p['dense_structural_av_ok'],
        'sol_attn_structural_av_ok': p['sol_attn_structural_av_ok'],
    } for p in pairs],
}, indent=2, sort_keys=True) + '\n')
(root / 'resource_summary.json').write_text(json.dumps({
    'schema_version': 'minimax_h3_a6000_sol_attn_matched_retest_resource_summary_v1',
    'resource_thresholds': resource_thresholds,
    'resource_peak_summary': decision['resource_peak_summary'],
    'resource_envelope_comparable_to_prior_r8': resource_envelope_comparable,
    'per_pair': resource_summaries,
}, indent=2, sort_keys=True) + '\n')
report = f"""# Sol-Attn r8 matched retest run report\n\nStatus: `{classification}`.\n\nThis is a bounded N={requested_pairs} same-GPU 5-step matched retest, not formal N>=10, not BF16 fidelity, and not a speedup claim.\n\n- Decision reason: {reason}\n- Completed pairs: {len(completed_pairs)}/{requested_pairs}\n- Median HTTP-time improvement: {median_improvement_pct if median_improvement_pct is not None else 'pending'}%\n- Timing eligibility threshold: > {threshold_pct}%\n- Proceed to N>=10 recommended: {classification == 'proceed_to_formal_n10_candidate'}\n- Failed gates: {', '.join(failed_gates) if failed_gates else 'none'}\n\nRaw artifacts are the per-pair directories under this run directory. Opaque image/output identifiers are not used as classification evidence.\n"""
(root / 'RUN_REPORT.md').write_text(report)
print(json.dumps({'classification': classification, 'failed_gates': failed_gates, 'decision': str(root / 'decision.json')}, sort_keys=True))
PY

exit "$pair_rc"
