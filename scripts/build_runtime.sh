#!/usr/bin/env bash
# Build the pinned vLLM-Omni CUDA runtime used by the A6000 measurements.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_REPO="${SOURCE_REPO:-https://github.com/vllm-project/vllm-omni.git}"
SOURCE_COMMIT="${SOURCE_COMMIT:-8e2e9b6b53e86e6a479ed2c0a53782f655f60e04}"
SOURCE_DIR="${SOURCE_DIR:-$ROOT/.cache/vllm-omni-$SOURCE_COMMIT}"
BASE_IMAGE="${BASE_IMAGE:-docker.io/vllm/vllm-openai@sha256:770fe65b2c73ee74a5c42165cf3433de4048cc2cd9c57a937ca4e35aba5aa87b}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r2}"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; shift; fi
if [[ $# -gt 0 ]]; then echo "ERROR: unknown arguments: $*" >&2; exit 64; fi

cat <<EOF
MiniMax-H3 A6000 runtime build
  source: $SOURCE_REPO
  commit: $SOURCE_COMMIT
  base:   $BASE_IMAGE
  tag:    $RUNTIME_IMAGE
  work:   $SOURCE_DIR
EOF
if (( DRY_RUN )); then
  echo "DRY-RUN: no network, Git, Docker pull, or image build was performed."
  exit 0
fi

command -v git >/dev/null || { echo "ERROR: git is required" >&2; exit 2; }
command -v docker >/dev/null || { echo "ERROR: Docker with the NVIDIA runtime is required" >&2; exit 2; }

if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  mkdir -p "$(dirname "$SOURCE_DIR")"
  git init -q "$SOURCE_DIR"
  git -C "$SOURCE_DIR" remote add origin "$SOURCE_REPO"
fi
actual_origin="$(git -C "$SOURCE_DIR" remote get-url origin)"
[[ "$actual_origin" == "$SOURCE_REPO" ]] || {
  echo "ERROR: existing source origin mismatch: $actual_origin" >&2; exit 2;
}
git -C "$SOURCE_DIR" fetch --depth 1 origin "$SOURCE_COMMIT"
git -C "$SOURCE_DIR" checkout -q --detach FETCH_HEAD
[[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || {
  echo "ERROR: source commit verification failed" >&2; exit 2;
}

echo "Building the pinned CUDA image; this can take a long time and requires substantial disk space."
DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}" docker build \
  -f "$SOURCE_DIR/docker/Dockerfile.cuda" \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --label "org.opencontainers.image.revision=$SOURCE_COMMIT" \
  --label "org.opencontainers.image.source=$SOURCE_REPO" \
  -t "$RUNTIME_IMAGE" \
  "$SOURCE_DIR"

docker image inspect "$RUNTIME_IMAGE" --format 'built image={{.Id}} size={{.Size}} bytes'
echo "Runtime ready: $RUNTIME_IMAGE"
