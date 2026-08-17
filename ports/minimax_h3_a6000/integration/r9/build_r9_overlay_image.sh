#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# External build-only r9 overlay image generator. It records source, patch-file,
# IID, and inspect evidence, then stops after the image build/inspect steps.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../../../.." && pwd -P)}
BASE_IMAGE=${BASE_IMAGE:-argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r2}
TAG=${TAG:-argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r9-sol-attn-overlay}
EVIDENCE_DIR=${EVIDENCE_DIR:-technical_report/evidence/minimax_h3_desktop/sol_engine_port/r9_overlay_image}
DOCKERFILE=ports/minimax_h3_a6000/integration/r9/Dockerfile
PATCH=ports/minimax_h3_a6000/patches/vllm_omni_h3_a6000_opt_in.patch
CACHE_PATCH=ports/minimax_h3_a6000/patches/vllm_omni_h3_cachedit_telemetry.patch
HELPER=ports/minimax_h3_a6000/integration/r9/dual_install_patch_files.py
CACHE_HELPER=ports/minimax_h3_a6000/integration/r9/install_cachedit_telemetry_patch.py
DRY_RUN=0

usage() {
  cat <<'EOF_USAGE'
Usage: build_r9_overlay_image.sh [--dry-run|--execute]

Build the local r9 Sol-Attn stride-aware overlay image from an already-present
pinned r2 base image. --dry-run prints the intended local build without Docker
access. Non-dry execution records a fail-closed base-image blocker instead of
asking BuildKit to resolve or pull the private pinned base image when it is absent.
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

if [[ ! -f "$DOCKERFILE" ]]; then
  echo "missing r9 Dockerfile; run from repository root or set ROOT" >&2
  exit 2
fi
if [[ ! -f "$PATCH" ]]; then
  echo "missing Sol-Attn patch artifact: $PATCH" >&2
  exit 2
fi
if [[ ! -f "$CACHE_PATCH" ]]; then
  echo "missing Cache-DiT telemetry patch artifact: $CACHE_PATCH" >&2
  exit 2
fi
if [[ ! -f "$HELPER" ]]; then
  echo "missing r9 patch install helper: $HELPER" >&2
  exit 2
fi
if [[ ! -f "$CACHE_HELPER" ]]; then
  echo "missing r9 Cache-DiT telemetry install helper: $CACHE_HELPER" >&2
  exit 2
fi

if [[ "$DRY_RUN" == "1" ]]; then
  cat <<EOF_DRY_RUN
DRY_RUN=1: no Docker build, Docker inspect, GPU access, container run, network access, model loading, or evidence mutation.
root=$ROOT
base_image=$BASE_IMAGE
tag=$TAG
dockerfile=$DOCKERFILE
patch=$PATCH
cache_patch=$CACHE_PATCH
evidence_dir=$EVIDENCE_DIR
required_preflight=non-dry first verifies the pinned base image with docker image inspect and records r9_base_image_blocker.json on absence
EOF_DRY_RUN
  exit 0
fi

mkdir -p "$EVIDENCE_DIR"
rm -f \
  "$EVIDENCE_DIR/r9_patch_changed_files.txt" \
  "$EVIDENCE_DIR/r9_source_hashes.sha256" \
  "$EVIDENCE_DIR/r9_source_hashes.json" \
  "$EVIDENCE_DIR/r9_build_params.env" \
  "$EVIDENCE_DIR/r9_image_iid.txt" \
  "$EVIDENCE_DIR/r9_image_inspect.json" \
  "$EVIDENCE_DIR/r9_image_identity_summary.txt" \
  "$EVIDENCE_DIR/r9_base_image_blocker.json" \
  "$EVIDENCE_DIR/r9_cachedit_telemetry_changed_files.txt" \
  "$EVIDENCE_DIR/r9_cachedit_telemetry_source_hashes.json" \
  "$EVIDENCE_DIR/r9_cachedit_telemetry_source_hashes.sha256"
if ! docker image inspect "$BASE_IMAGE" > "$EVIDENCE_DIR/r9_base_image_inspect.json" 2> "$EVIDENCE_DIR/r9_base_image_inspect.stderr"; then
  python3 - "$EVIDENCE_DIR/r9_base_image_blocker.json" "$BASE_IMAGE" "$EVIDENCE_DIR/r9_base_image_inspect.stderr" <<'PY'
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

out = Path(sys.argv[1])
base_image = sys.argv[2]
stderr_path = Path(sys.argv[3])
stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
out.write_text(
    json.dumps(
        {
            "schema_version": "minimax_h3_a6000_overlay_base_image_blocker_v1",
            "status": "blocked",
            "overlay": "r9",
            "base_image": base_image,
            "reason": "pinned_base_image_not_inspectable_locally",
            "docker_image_inspect_stderr": stderr.strip(),
            "recheck_condition": "restore or rebuild the pinned r2 base image in the local Docker daemon, then rerun this build script",
            "timestamp_unix": time.time(),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
  echo "ERROR: pinned base image is not inspectable locally: $BASE_IMAGE; wrote $EVIDENCE_DIR/r9_base_image_blocker.json" >&2
  exit 24
fi
python3 "$HELPER" --patch "$PATCH" --list-patch-files > "$EVIDENCE_DIR/r9_patch_changed_files.txt"
python3 - "$EVIDENCE_DIR/r9_source_hashes.sha256" "$EVIDENCE_DIR/r9_source_hashes.json" <<'PY'
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

out_sha = pathlib.Path(sys.argv[1])
out_json = pathlib.Path(sys.argv[2])
inputs = [
    pathlib.Path("ports/minimax_h3_a6000/integration/r9/Dockerfile"),
    pathlib.Path("ports/minimax_h3_a6000/integration/r9/build_r9_overlay_image.sh"),
    pathlib.Path("ports/minimax_h3_a6000/integration/r9/dual_install_patch_files.py"),
    pathlib.Path("ports/minimax_h3_a6000/integration/r9/install_cachedit_telemetry_patch.py"),
    pathlib.Path("ports/minimax_h3_a6000/patches/vllm_omni_h3_a6000_opt_in.patch"),
    pathlib.Path("ports/minimax_h3_a6000/patches/vllm_omni_h3_cachedit_telemetry.patch"),
    pathlib.Path("ports/minimax_h3_a6000/NOTICE"),
    pathlib.Path("ports/minimax_h3_a6000/UPSTREAM.md"),
]
src_root = pathlib.Path("ports/minimax_h3_a6000/src/minimax_h3_a6000")
inputs.extend(sorted(p for p in src_root.glob("*.py") if p.is_file()))
records = []
lines = []
for path in inputs:
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    rel = path.as_posix()
    records.append({"path": rel, "sha256": digest, "bytes": len(data)})
    lines.append(f"{digest}  {rel}")
out_sha.write_text("\n".join(lines) + "\n", encoding="utf-8")
out_json.write_text(
    json.dumps(
        {
            "schema_version": "minimax_h3_a6000_r9_source_hashes_v1",
            "records": records,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
patch_sha=$(python3 - <<'PY'
import hashlib, pathlib
print(hashlib.sha256(pathlib.Path('ports/minimax_h3_a6000/patches/vllm_omni_h3_a6000_opt_in.patch').read_bytes()).hexdigest())
PY
)
cache_patch_sha=$(python3 - <<'PY'
import hashlib, pathlib
print(hashlib.sha256(pathlib.Path('ports/minimax_h3_a6000/patches/vllm_omni_h3_cachedit_telemetry.patch').read_bytes()).hexdigest())
PY
)
printf 'base_image=%s\ntag=%s\ndockerfile=%s\npatch=%s\npatch_sha256=%s\ncache_patch=%s\ncache_patch_sha256=%s\nnetwork=none\ngpu_flags=none\npull=false\n' \
  "$BASE_IMAGE" "$TAG" "$DOCKERFILE" "$PATCH" "$patch_sha" "$CACHE_PATCH" "$cache_patch_sha" > "$EVIDENCE_DIR/r9_build_params.env"

# Build only; deliberately no container execution and no GPU flag in this script.
DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-1} docker build --pull=false --network=none \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  -f ports/minimax_h3_a6000/integration/r9/Dockerfile \
  -t "$TAG" \
  --iidfile "$EVIDENCE_DIR/r9_image_iid.txt" \
  .
docker image inspect "$TAG" > "$EVIDENCE_DIR/r9_image_inspect.json"
printf 'External diagnostic image identity: image=%s iid_file=%s inspect=%s\n' \
  "$TAG" "$EVIDENCE_DIR/r9_image_iid.txt" "$EVIDENCE_DIR/r9_image_inspect.json" \
  > "$EVIDENCE_DIR/r9_image_identity_summary.txt"
