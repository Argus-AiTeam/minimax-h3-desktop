#!/usr/bin/env bash
# Download pinned licensed assets and build the disclosed merged Turbo checkpoint.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_REPO="${BASE_REPO:-MiniMaxAI/MiniMax-H3}"
BASE_REVISION="${BASE_REVISION:-6818f6c32d12b210915e44ad56a4228c2608f160}"
BASE_DIR="${BASE_DIR:-$ROOT/models/MiniMax-H3}"
TURBO_REPO="${TURBO_REPO:-larryvrh/MiniMax-H3-Turbo-Lora}"
TURBO_REVISION="${TURBO_REVISION:-43a74557ac3f6539db8e0f2a959d03feb7a81480}"
TURBO_FILE="${TURBO_FILE:-minimax_h3_turbo_v4_step600_ema.safetensors}"
TURBO_SHA256="${TURBO_SHA256:-5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3}"
TURBO_DIR="${TURBO_DIR:-$ROOT/models/MiniMax-H3-Turbo-Lora}"
MERGED_DIR="${MERGED_DIR:-$ROOT/models/MiniMax-H3-Turbo-Merged/FL2VA}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r2}"
GPU_INDEX="${GPU_INDEX:-0}"
DRY_RUN=0
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_MERGE="${SKIP_MERGE:-0}"

usage() {
  cat <<'EOF'
Usage: I_ACCEPT_MINIMAX_H3_LICENSE=YES bash scripts/prepare_models.sh
       bash scripts/prepare_models.sh --dry-run

Environment options:
  GPU_INDEX=0       GPU used only for the offline LoRA merge.
  SKIP_DOWNLOAD=1   Validate existing files and continue to merge.
  SKIP_MERGE=1      Download/validate only.
  BASE_DIR=...      Official MiniMax-H3 local directory.
  TURBO_DIR=...     Turbo LoRA local directory.
  MERGED_DIR=...    Separate merged practical-model output directory.
  RUNTIME_IMAGE=... Pinned image built by scripts/build_runtime.sh.

Expected local storage is roughly 230 GiB including the official FL2VA assets,
the separate merged transformer, the adapter, caches, and the runtime image.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 64 ;;
  esac
done

cat <<EOF
Pinned model preparation
  base:   $BASE_REPO@$BASE_REVISION (FL2VA/*, about 134.16 GiB)
  turbo:  $TURBO_REPO@$TURBO_REVISION/$TURBO_FILE
  output: $MERGED_DIR
EOF
if (( DRY_RUN )); then
  echo "DRY-RUN: no download, model read, Docker, or GPU action was performed."
  exit 0
fi

if [[ "${I_ACCEPT_MINIMAX_H3_LICENSE:-}" != "YES" ]]; then
  echo "ERROR: review the MiniMax-H3 license and set I_ACCEPT_MINIMAX_H3_LICENSE=YES" >&2
  exit 2
fi
command -v hf >/dev/null || {
  echo "ERROR: install and authenticate the Hugging Face CLI first: pip install -U huggingface_hub && hf auth login" >&2
  exit 2
}
command -v docker >/dev/null || { echo "ERROR: Docker is required for the offline merge" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi is required" >&2; exit 2; }

mkdir -p "$BASE_DIR" "$TURBO_DIR" "$(dirname "$MERGED_DIR")"
if [[ "$SKIP_DOWNLOAD" != 1 ]]; then
  hf download "$BASE_REPO" \
    --revision "$BASE_REVISION" \
    --include 'FL2VA/*' \
    --local-dir "$BASE_DIR"
  hf download "$TURBO_REPO" "$TURBO_FILE" \
    --revision "$TURBO_REVISION" \
    --local-dir "$TURBO_DIR"
fi

required=(model_index.json processor tokenizer text_encoder transformer video_vae audio_vae)
for name in "${required[@]}"; do
  [[ -e "$BASE_DIR/FL2VA/$name" ]] || { echo "ERROR: missing $BASE_DIR/FL2VA/$name" >&2; exit 3; }
done
actual_turbo_sha="$(sha256sum "$TURBO_DIR/$TURBO_FILE" | awk '{print $1}')"
[[ "$actual_turbo_sha" == "$TURBO_SHA256" ]] || {
  echo "ERROR: Turbo SHA256 mismatch: $actual_turbo_sha" >&2; exit 3;
}
echo "Pinned downloads validated."

if [[ "$SKIP_MERGE" == 1 ]]; then
  echo "SKIP_MERGE=1: download/validation complete."
  exit 0
fi

docker image inspect "$RUNTIME_IMAGE" >/dev/null 2>&1 || {
  echo "ERROR: runtime image is missing; run scripts/build_runtime.sh first" >&2; exit 4;
}
GPU_UUID="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -F "$GPU_UUID" >/dev/null; then
  echo "ERROR: selected merge GPU $GPU_INDEX ($GPU_UUID) is busy" >&2
  exit 4
fi

base_rel="$(python3 - "$BASE_DIR/FL2VA" "$ROOT" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve().relative_to(Path(sys.argv[2]).resolve()))
PY
)"
turbo_rel="$(python3 - "$TURBO_DIR/$TURBO_FILE" "$ROOT" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve().relative_to(Path(sys.argv[2]).resolve()))
PY
)"
merged_rel="$(python3 - "$MERGED_DIR" "$ROOT" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve().relative_to(Path(sys.argv[2]).resolve()))
PY
)"

uid="$(id -u)"; gid="$(id -g)"
docker run --rm --gpus "device=$GPU_INDEX" \
  --user "$uid:$gid" --network none --ipc=host \
  -e CUDA_VISIBLE_DEVICES=0 -e HOME=/tmp \
  -v "$ROOT":/workspace/project:rw -w /workspace/project \
  "$RUNTIME_IMAGE" \
  python3 tools/turbo_lora_offline_merge.py merge \
    --base-root "$base_rel" \
    --lora "$turbo_rel" \
    --output-root "$merged_rel" \
    --strength 1.0 --engine torch --device cuda:0 \
    --link-mode hardlink --resume --sha256-lora

python3 - "$MERGED_DIR/merge_manifest.json" <<'PY'
import json, sys
m=json.load(open(sys.argv[1], encoding='utf-8'))
assert m.get('status') == 'completed'
assert m.get('merge', {}).get('strength') == 1.0
assert len(m.get('completed_shards', {})) == 13
assert m.get('validation', {}).get('pair_count') == 259
print('Merged Turbo FL2VA validated: 13 shards, 259 LoRA pairs, strength=1.0')
PY
