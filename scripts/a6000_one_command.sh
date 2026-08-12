#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="dry-run"
STAGE="all"
AUTHORIZATION_ID=""
TRACK="fidelity_bf16_exact"
PLATFORM="single_a6000_48gb_workstation"
TASK="FL2VA"
MODEL_DIR="${REPO_ROOT}/models/MiniMax-H3"
WORK_DIR=""
CONTAINER_IMAGE=""
ALLOW_ENV_NAME="ARGUS_ALLOW_MINIMAX_H3_RUN"

usage() {
  cat <<'EOF'
Usage: bash scripts/a6000_one_command.sh --dry-run [options]
       ARGUS_ALLOW_MINIMAX_H3_RUN=1 bash scripts/a6000_one_command.sh --execute --authorization-id ID --work-dir PATH [options]

Options:
  --stage preflight|model-prepare|deploy|run|verify|all
  --track fidelity_bf16_exact|practical_disclosed_approx
  --platform single_a6000_48gb_workstation|current_a6000_reference
  --task FL2VA
  --model-dir PATH             Existing local MiniMax-H3 root; weights are never modified.
  --container-image IMAGE      Existing local locked runtime tag; default comes from runtime lock metadata.
  --work-dir PATH              Clean isolated lifecycle output directory.

Non-dry mode is a local prepare/deploy/run/verify lifecycle verifier. It uses
only disclosed local locked resources: an existing FL2VA model directory, the
locked runtime image metadata, the local Docker image inspect path, and the
checked-in verifier fixture. It does not download, publish, pull containers,
start containers, load model weights, run GPU inference, or create media.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      MODE="dry-run"
      shift
      ;;
    --execute|--non-dry)
      MODE="execute"
      shift
      ;;
    --stage)
      STAGE="${2:?missing stage}"
      shift 2
      ;;
    --authorization-id)
      AUTHORIZATION_ID="${2:?missing authorization id}"
      shift 2
      ;;
    --track)
      TRACK="${2:?missing track}"
      shift 2
      ;;
    --platform)
      PLATFORM="${2:?missing platform}"
      shift 2
      ;;
    --task)
      TASK="${2:?missing task}"
      shift 2
      ;;
    --model-dir)
      MODEL_DIR="${2:?missing model dir}"
      shift 2
      ;;
    --container-image)
      CONTAINER_IMAGE="${2:?missing image}"
      shift 2
      ;;
    --work-dir)
      WORK_DIR="${2:?missing work dir}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 64
      ;;
  esac
done

case "$TRACK" in
  fidelity_bf16_exact|practical_disclosed_approx) ;;
  *) echo "ERROR: unsupported --track: $TRACK" >&2; exit 64 ;;
esac

case "$TASK" in
  FL2VA) ;;
  Ref2VA) echo "ERROR: Ref2VA is not authorized for this bounded local lifecycle; use FL2VA." >&2; exit 64 ;;
  *) echo "ERROR: --task must be FL2VA" >&2; exit 64 ;;
esac

case "$PLATFORM" in
  single_a6000_48gb_workstation)
    PLATFORM_GATE_NOTE="target single RTX A6000 48GB lane; this lifecycle verifier is CPU/local-resource only and is not a new benchmark."
    ;;
  current_a6000_reference)
    PLATFORM_GATE_NOTE="current host reference lane; this lifecycle verifier hard-limits itself to CPU/local-resource checks and is not a multi-GPU result."
    ;;
  *)
    echo "ERROR: unsupported --platform: $PLATFORM" >&2
    echo "Allowed platforms: single_a6000_48gb_workstation, current_a6000_reference" >&2
    exit 64
    ;;
esac

LOCK_JSON="$REPO_ROOT/runtime/single_a6000_bf16/source_commit.json"
LOCK_DIGEST="$REPO_ROOT/runtime/single_a6000_bf16/container_image.digest"
if [[ -z "$CONTAINER_IMAGE" && -f "$LOCK_JSON" ]]; then
  CONTAINER_IMAGE="$(python3 - "$LOCK_JSON" <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf-8') as f:
    data = json.load(f)
print(data.get('derived_image_local_tag') or '')
PY
)"
fi
if [[ -z "$CONTAINER_IMAGE" ]]; then
  CONTAINER_IMAGE="argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r2"
fi

if [[ "$MODE" == "dry-run" ]]; then
  cat <<EOF
[DRY-RUN] MiniMax-H3 A6000 one-command local lifecycle
- stages: ${STAGE}
- platform: ${PLATFORM}
- platform gate: ${PLATFORM_GATE_NOTE}
- lane: ${TRACK}
- task: ${TASK}
- model dir: ${MODEL_DIR}
- locked runtime image: ${CONTAINER_IMAGE}

Dry-run only: no download, no Docker pull/run, no model load, no GPU use, no media generation, no publication.
Non-dry local verifier command shape:
  ARGUS_ALLOW_MINIMAX_H3_RUN=1 bash scripts/a6000_one_command.sh --execute --authorization-id <AUTHZ-ID> --work-dir <clean-dir> --track ${TRACK} --platform ${PLATFORM} --task ${TASK} --model-dir '${MODEL_DIR}' --container-image '${CONTAINER_IMAGE}'
EOF
  exit 0
fi

if [[ -z "$AUTHORIZATION_ID" || "${ARGUS_ALLOW_MINIMAX_H3_RUN:-0}" != "1" ]]; then
  cat >&2 <<EOF
ERROR: blocked by fail-closed non-dry lifecycle guard.
Provide both:
  ARGUS_ALLOW_MINIMAX_H3_RUN=1
  --authorization-id <operator-approved-id>
EOF
  exit 2
fi

if [[ -z "$WORK_DIR" ]]; then
  WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/minimax-h3-a6000-lifecycle.XXXXXX")"
else
  mkdir -p "$WORK_DIR"
fi
WORK_DIR="$(cd "$WORK_DIR" && pwd)"
mkdir -p "$WORK_DIR/stages"

write_json() {
  local path="$1"
  shift
  python3 - "$path" "$@" <<'PY'
import json, os, sys
path = sys.argv[1]
pairs = sys.argv[2:]
data = {}
for pair in pairs:
    key, value = pair.split('=', 1)
    if value in {'true', 'false'}:
        data[key] = (value == 'true')
    else:
        try:
            data[key] = int(value)
        except ValueError:
            data[key] = value
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, sort_keys=True)
    f.write('\n')
PY
}

stage_preflight() {
  local out="$WORK_DIR/stages/01_preflight.txt"
  {
    echo "schema=argus-minimax-h3-a6000-local-lifecycle-preflight-v1"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repo_root=$REPO_ROOT"
    echo "work_dir=$WORK_DIR"
    echo "authorization_id=$AUTHORIZATION_ID"
    echo "track=$TRACK"
    echo "platform=$PLATFORM"
    echo "platform_gate=$PLATFORM_GATE_NOTE"
    echo "task=$TASK"
    echo "model_dir=$MODEL_DIR"
    echo "container_image=$CONTAINER_IMAGE"
    echo "python=$(command -v python3 || true)"
    python3 -V 2>&1 || true
    echo "docker=$(command -v docker || true)"
    echo "df_work_dir:"
    df -h "$WORK_DIR" || true
    echo "nvidia_smi_snapshot_not_used_for_compute:"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader || true
      nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
    else
      echo "nvidia-smi not found"
    fi
    echo "compute_backend=CPU/local-resource verifier; GPU inference NOT_APPLICABLE"
  } > "$out"
  echo "$out"
}

stage_model_prepare() {
  local task_dir="$MODEL_DIR/$TASK"
  if [[ ! -d "$task_dir" ]]; then
    echo "ERROR: local locked model task directory is missing: $task_dir" >&2
    return 20
  fi
  python3 - "$MODEL_DIR" "$TASK" "$WORK_DIR/stages/02_model_prepare.json" <<'PY'
import json, os, sys
from pathlib import Path
model_dir = Path(sys.argv[1]).resolve()
task = sys.argv[2]
out = Path(sys.argv[3])
task_dir = model_dir / task
required = ['model_index.json', 'text_encoder', 'transformer', 'video_vae', 'audio_vae', 'processor']
missing = [name for name in required if not (task_dir / name).exists()]
file_count = 0
total_bytes = 0
for path in task_dir.rglob('*'):
    if path.is_file() and not path.is_symlink():
        file_count += 1
        total_bytes += path.stat().st_size
payload = {
    'schema_version': 'argus-minimax-h3-a6000-local-model-prepare-v1',
    'status': 'pass' if not missing else 'fail',
    'task': task,
    'model_dir': str(model_dir),
    'task_dir': str(task_dir),
    'required_entries': required,
    'missing_required_entries': missing,
    'local_non_symlink_file_count': file_count,
    'local_total_bytes': total_bytes,
    'mutation_policy': 'read_only_existing_local_weights_no_download_no_delete_no_rewrite',
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
if missing:
    print('ERROR missing required model entries: ' + ', '.join(missing), file=sys.stderr)
    raise SystemExit(21)
PY
  echo "$WORK_DIR/stages/02_model_prepare.json"
}

stage_deploy() {
  if [[ ! -f "$LOCK_JSON" || ! -f "$LOCK_DIGEST" ]]; then
    echo "ERROR: locked runtime metadata missing under runtime/single_a6000_bf16" >&2
    return 30
  fi
  local inspect_json="$WORK_DIR/stages/03_docker_image_inspect.json"
  local image_id=""
  if command -v docker >/dev/null 2>&1; then
    if ! docker image inspect "$CONTAINER_IMAGE" > "$inspect_json" 2>"$WORK_DIR/stages/03_docker_image_inspect.stderr"; then
      echo "ERROR: locked local Docker image is not inspectable: $CONTAINER_IMAGE" >&2
      return 31
    fi
    image_id="$(python3 - "$inspect_json" <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf-8') as f:
    data = json.load(f)
print(data[0].get('Id', ''))
PY
)"
  else
    echo "ERROR: docker is required for deploy-stage local image inspection" >&2
    return 32
  fi
  python3 - "$LOCK_JSON" "$LOCK_DIGEST" "$CONTAINER_IMAGE" "$image_id" "$WORK_DIR/stages/03_deploy.json" <<'PY'
import json, sys
from pathlib import Path
lock_json = Path(sys.argv[1])
lock_digest = Path(sys.argv[2])
image = sys.argv[3]
image_id = sys.argv[4]
out = Path(sys.argv[5])
source = json.loads(lock_json.read_text(encoding='utf-8'))
expected = lock_digest.read_text(encoding='utf-8').strip()
status = 'pass' if image_id == expected and image == source.get('derived_image_local_tag') else 'fail'
payload = {
    'schema_version': 'argus-minimax-h3-a6000-local-deploy-v1',
    'status': status,
    'container_image': image,
    'expected_image_id': expected,
    'actual_image_id': image_id,
    'source_commit': source.get('source_commit'),
    'source_describe': source.get('source_describe'),
    'runtime_versions': source.get('versions', {}),
    'deploy_action': 'local_image_inspected_and_locked_no_container_started_no_gpu_used',
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
if status != 'pass':
    print('ERROR local image id/tag does not match locked runtime metadata', file=sys.stderr)
    raise SystemExit(33)
PY
  echo "$WORK_DIR/stages/03_deploy.json"
}

stage_run() {
  local out="$WORK_DIR/stages/04_verify_run_fixture.json"
  python3 "$REPO_ROOT/tools/verify_run.py" \
    --fixture "$REPO_ROOT/tests/fixtures/minimal_av_case" \
    --schema "$REPO_ROOT/schemas/minimax_h3_run.schema.json" \
    --json > "$out"
  write_json "$WORK_DIR/stages/04_run.json" \
    schema_version=argus-minimax-h3-a6000-local-run-v1 \
    status=pass \
    action=cpu_fixture_verifier_executed_no_h3_model_load_no_media_generation \
    verifier_output="$out" \
    track="$TRACK"
  echo "$WORK_DIR/stages/04_run.json"
}

stage_verify() {
  local audit_status="not_run"
  local audit_note="publication audit is run only when the command root is a sanitized release tree"
  if [[ -f "$REPO_ROOT/.argus_release_tree" ]]; then
    if python3 "$REPO_ROOT/tools/publication_audit.py" --root "$REPO_ROOT" --max-bytes 1000000 --json > "$WORK_DIR/stages/05_publication_audit.json"; then
      audit_status="pass"
      audit_note="sanitized release-root publication audit passed"
    else
      audit_status="fail"
      audit_note="sanitized release-root publication audit failed"
    fi
  fi
  python3 - "$WORK_DIR" "$audit_status" "$audit_note" "$WORK_DIR/stages/05_lifecycle_summary.json" <<'PY'
import json, sys
from pathlib import Path
work = Path(sys.argv[1])
audit_status = sys.argv[2]
audit_note = sys.argv[3]
out = Path(sys.argv[4])
stage_files = sorted(str(p.relative_to(work)) for p in (work / 'stages').glob('*'))
required = [
    'stages/01_preflight.txt',
    'stages/02_model_prepare.json',
    'stages/03_deploy.json',
    'stages/04_run.json',
]
missing = [p for p in required if not (work / p).exists()]
status = 'pass' if not missing and audit_status in {'pass', 'not_run'} else 'fail'
payload = {
    'schema_version': 'argus-minimax-h3-a6000-local-lifecycle-summary-v1',
    'status': status,
    'work_dir': str(work),
    'stage_files': stage_files,
    'missing_required_stage_files': missing,
    'publication_audit_status': audit_status,
    'publication_audit_note': audit_note,
    'claim_boundary': 'local clean-room lifecycle verifier only; no new speedup, quality, BF16 fidelity, Sol-Attn, Turbo, DLO, DMD, GPU, Docker-run, model-load, or publication claim',
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
if status != 'pass':
    raise SystemExit(40)
PY
  echo "$WORK_DIR/stages/05_lifecycle_summary.json"
}

run_stage() {
  case "$1" in
    preflight) stage_preflight ;;
    model-prepare) stage_model_prepare ;;
    deploy) stage_deploy ;;
    run) stage_run ;;
    verify) stage_verify ;;
    *) echo "ERROR: unsupported stage: $1" >&2; exit 64 ;;
  esac
}

case "$STAGE" in
  all)
    for s in preflight model-prepare deploy run verify; do
      run_stage "$s" >/dev/null
    done
    ;;
  preflight|model-prepare|deploy|run|verify)
    run_stage "$STAGE" >/dev/null
    ;;
  *) echo "ERROR: unsupported --stage: $STAGE" >&2; exit 64 ;;
esac

echo "local lifecycle ${STAGE} completed: $WORK_DIR"
if [[ -f "$WORK_DIR/stages/05_lifecycle_summary.json" ]]; then
  python3 - "$WORK_DIR/stages/05_lifecycle_summary.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf-8') as f:
    data = json.load(f)
print('summary_status=' + data['status'])
print('publication_audit_status=' + data['publication_audit_status'])
PY
fi
