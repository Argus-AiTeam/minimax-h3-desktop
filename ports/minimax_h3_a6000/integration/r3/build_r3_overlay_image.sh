#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ROOT=${ROOT:-$(cd -- "$SCRIPT_DIR/../../../.." && pwd -P)}
BASE_IMAGE=${BASE_IMAGE:-argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r2}
TAG=${TAG:-argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r3-exact-overlay}
EVIDENCE_DIR=${EVIDENCE_DIR:-}

cd "$ROOT"

if [[ ! -f ports/minimax_h3_a6000/integration/r3/Dockerfile ]]; then
  echo "missing r3 Dockerfile; run from repository root or set ROOT" >&2
  exit 2
fi

args=(
  docker build
  --pull=false
  --network=none
  --build-arg "BASE_IMAGE=$BASE_IMAGE"
  -f ports/minimax_h3_a6000/integration/r3/Dockerfile
  -t "$TAG"
)

if [[ -n "$EVIDENCE_DIR" ]]; then
  mkdir -p "$EVIDENCE_DIR"
  args+=(--iidfile "$EVIDENCE_DIR/r3_image_iid.txt")
  printf 'base_image=%s\ntag=%s\nnetwork=none\ngpu_flags=none\n' "$BASE_IMAGE" "$TAG" > "$EVIDENCE_DIR/r3_build_params.env"
fi

# Build only; deliberately no container execution and no GPU flag in this script.
DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-1} "${args[@]}" .

if [[ -n "$EVIDENCE_DIR" ]]; then
  docker image inspect "$TAG" > "$EVIDENCE_DIR/r3_image_inspect.json"
fi
