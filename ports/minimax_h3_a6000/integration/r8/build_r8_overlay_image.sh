#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# External build-only r8 overlay image generator. It records source, patch-file,
# IID, and inspect evidence, then stops after the image build/inspect steps.
set -euo pipefail

ROOT=${ROOT:-${PWD}}
BASE_IMAGE=${BASE_IMAGE:-argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r2}
TAG=${TAG:-argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r8-sol-attn-overlay}
EVIDENCE_DIR=${EVIDENCE_DIR:-technical_report/evidence/minimax_h3_desktop/sol_engine_port/r8_overlay_image}
DOCKERFILE=ports/minimax_h3_a6000/integration/r8/Dockerfile
PATCH=ports/minimax_h3_a6000/patches/vllm_omni_h3_a6000_opt_in.patch
HELPER=ports/minimax_h3_a6000/integration/r8/dual_install_patch_files.py

cd "$ROOT"

if [[ ! -f "$DOCKERFILE" ]]; then
  echo "missing r8 Dockerfile; run from repository root or set ROOT" >&2
  exit 2
fi
if [[ ! -f "$PATCH" ]]; then
  echo "missing Sol-Attn patch artifact: $PATCH" >&2
  exit 2
fi
if [[ ! -f "$HELPER" ]]; then
  echo "missing r8 patch install helper: $HELPER" >&2
  exit 2
fi

mkdir -p "$EVIDENCE_DIR"
python3 "$HELPER" --patch "$PATCH" --list-patch-files > "$EVIDENCE_DIR/r8_patch_changed_files.txt"
python3 - "$EVIDENCE_DIR/r8_source_hashes.sha256" "$EVIDENCE_DIR/r8_source_hashes.json" <<'PY'
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

out_sha = pathlib.Path(sys.argv[1])
out_json = pathlib.Path(sys.argv[2])
inputs = [
    pathlib.Path("ports/minimax_h3_a6000/integration/r8/Dockerfile"),
    pathlib.Path("ports/minimax_h3_a6000/integration/r8/build_r8_overlay_image.sh"),
    pathlib.Path("ports/minimax_h3_a6000/integration/r8/dual_install_patch_files.py"),
    pathlib.Path("ports/minimax_h3_a6000/patches/vllm_omni_h3_a6000_opt_in.patch"),
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
            "schema_version": "minimax_h3_a6000_r8_source_hashes_v1",
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
printf 'base_image=%s\ntag=%s\ndockerfile=%s\npatch=%s\npatch_sha256=%s\nnetwork=none\ngpu_flags=none\npull=false\n' \
  "$BASE_IMAGE" "$TAG" "$DOCKERFILE" "$PATCH" "$patch_sha" > "$EVIDENCE_DIR/r8_build_params.env"

# Build only; deliberately no container execution and no GPU flag in this script.
DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-1} docker build --pull=false --network=none \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  -f ports/minimax_h3_a6000/integration/r8/Dockerfile \
  -t "$TAG" \
  --iidfile "$EVIDENCE_DIR/r8_image_iid.txt" \
  .
docker image inspect "$TAG" > "$EVIDENCE_DIR/r8_image_inspect.json"
printf 'External diagnostic image identity: image=%s iid_file=%s inspect=%s\n' \
  "$TAG" "$EVIDENCE_DIR/r8_image_iid.txt" "$EVIDENCE_DIR/r8_image_inspect.json" \
  > "$EVIDENCE_DIR/r8_image_identity_summary.txt"
