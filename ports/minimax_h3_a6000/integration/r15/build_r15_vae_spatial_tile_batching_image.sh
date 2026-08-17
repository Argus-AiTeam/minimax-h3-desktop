#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build-only VAE spatial tile batching overlay image generator.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../../../.." && pwd -P)}
BASE_IMAGE=${BASE_IMAGE:-argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r13-split-profiler}
TAG=${TAG:-argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r15-vae-spatial-batching}
EVIDENCE_DIR=${EVIDENCE_DIR:-technical_report/evidence/minimax_h3_desktop/long_video/r15_vae_spatial_tile_batching_overlay_$(date -u +%Y%m%dT%H%M%SZ)}
DOCKERFILE=ports/minimax_h3_a6000/integration/r15/Dockerfile
PATCH=ports/minimax_h3_a6000/patches/vllm_omni_h3_vae_spatial_tile_batching.patch
HELPER=ports/minimax_h3_a6000/integration/r15/install_vae_spatial_tile_batching_patch.py
DRY_RUN=0

usage() {
  cat <<'EOF_USAGE'
Usage: build_r15_vae_spatial_tile_batching_image.sh [--dry-run|--execute]

Build a local r15 overlay image from an already-present r13 split-profiler base.
The VAE spatial tile batching switch is default-off and requires
MINIMAX_H3_A6000_VIDEO_VAE_SPATIAL_TILE_BATCHING=1 at runtime. Non-dry execution
records a fail-closed blocker if the base image is not inspectable locally.
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

cd "$ROOT"
for path in "$DOCKERFILE" "$PATCH" "$HELPER"; do
  [[ -f "$path" ]] || { echo "ERROR: missing required file: $path" >&2; exit 2; }
done

if [[ "$DRY_RUN" == "1" ]]; then
  cat <<EOF_DRY
DRY_RUN=1: no Docker build, Docker inspect, GPU access, container run, network access, model loading, or evidence mutation.
root=$ROOT
base_image=$BASE_IMAGE
tag=$TAG
dockerfile=$DOCKERFILE
patch=$PATCH
evidence_dir=$EVIDENCE_DIR
required_preflight=non-dry verifies the r13 split-profiler base image with docker image inspect and records r15_base_image_blocker.json on absence
EOF_DRY
  exit 0
fi

mkdir -p "$EVIDENCE_DIR"
rm -f \
  "$EVIDENCE_DIR/r15_base_image_blocker.json" \
  "$EVIDENCE_DIR/r15_base_image_inspect.json" \
  "$EVIDENCE_DIR/r15_base_image_inspect.stderr" \
  "$EVIDENCE_DIR/r15_build_params.env" \
  "$EVIDENCE_DIR/r15_image_iid.txt" \
  "$EVIDENCE_DIR/r15_image_inspect.json" \
  "$EVIDENCE_DIR/r15_image_identity_summary.txt" \
  "$EVIDENCE_DIR/r15_source_hashes.json" \
  "$EVIDENCE_DIR/r15_source_hashes.sha256"

if ! docker image inspect "$BASE_IMAGE" > "$EVIDENCE_DIR/r15_base_image_inspect.json" 2> "$EVIDENCE_DIR/r15_base_image_inspect.stderr"; then
  python3 - "$EVIDENCE_DIR/r15_base_image_blocker.json" "$BASE_IMAGE" "$EVIDENCE_DIR/r15_base_image_inspect.stderr" <<'PY'
from __future__ import annotations
import json, sys, time
from pathlib import Path
out = Path(sys.argv[1])
base_image = sys.argv[2]
stderr_path = Path(sys.argv[3])
stderr = stderr_path.read_text(encoding='utf-8', errors='replace') if stderr_path.exists() else ''
out.write_text(json.dumps({
    'schema_version': 'minimax_h3_a6000_r15_vae_spatial_tile_batching_base_image_blocker_v1',
    'status': 'blocked',
    'overlay': 'r15_vae_spatial_tile_batching',
    'base_image': base_image,
    'reason': 'r13_split_profiler_base_image_not_inspectable_locally',
    'docker_image_inspect_stderr': stderr.strip(),
    'recheck_condition': 'restore or rebuild the r13 split-profiler image in the isolated Docker daemon, then rerun this build script',
    'timestamp_unix': time.time(),
}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY
  echo "ERROR: r13 base image is not inspectable locally: $BASE_IMAGE; wrote $EVIDENCE_DIR/r15_base_image_blocker.json" >&2
  exit 24
fi

python3 - "$EVIDENCE_DIR/r15_source_hashes.sha256" "$EVIDENCE_DIR/r15_source_hashes.json" <<'PY'
from __future__ import annotations
import hashlib, json, pathlib, sys
out_sha = pathlib.Path(sys.argv[1])
out_json = pathlib.Path(sys.argv[2])
inputs = [
    pathlib.Path('ports/minimax_h3_a6000/integration/r15/Dockerfile'),
    pathlib.Path('ports/minimax_h3_a6000/integration/r15/build_r15_vae_spatial_tile_batching_image.sh'),
    pathlib.Path('ports/minimax_h3_a6000/integration/r15/install_vae_spatial_tile_batching_patch.py'),
    pathlib.Path('ports/minimax_h3_a6000/patches/vllm_omni_h3_vae_spatial_tile_batching.patch'),
]
records = []
lines = []
for path in inputs:
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    rel = path.as_posix()
    records.append({'path': rel, 'sha256': digest, 'bytes': len(data)})
    lines.append(f'{digest}  {rel}')
out_sha.write_text('\n'.join(lines) + '\n', encoding='utf-8')
out_json.write_text(json.dumps({'schema_version': 'minimax_h3_a6000_r15_vae_spatial_tile_batching_source_hashes_v1', 'records': records}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY
patch_sha=$(python3 - <<'PY'
import hashlib, pathlib
print(hashlib.sha256(pathlib.Path('ports/minimax_h3_a6000/patches/vllm_omni_h3_vae_spatial_tile_batching.patch').read_bytes()).hexdigest())
PY
)
printf 'base_image=%s\ntag=%s\ndockerfile=%s\npatch=%s\npatch_sha256=%s\nnetwork=none\ngpu_flags=none\npull=false\n' \
  "$BASE_IMAGE" "$TAG" "$DOCKERFILE" "$PATCH" "$patch_sha" > "$EVIDENCE_DIR/r15_build_params.env"

DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-1} docker build --pull=false --network=none \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  -f "$DOCKERFILE" \
  -t "$TAG" \
  --iidfile "$EVIDENCE_DIR/r15_image_iid.txt" \
  .
docker image inspect "$TAG" > "$EVIDENCE_DIR/r15_image_inspect.json"
printf 'VAE spatial tile batching image identity: image=%s iid_file=%s inspect=%s\n' \
  "$TAG" "$EVIDENCE_DIR/r15_image_iid.txt" "$EVIDENCE_DIR/r15_image_inspect.json" \
  > "$EVIDENCE_DIR/r15_image_identity_summary.txt"
