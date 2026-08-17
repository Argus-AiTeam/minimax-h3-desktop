#!/usr/bin/env bash
# Generate one or more MiniMax-H3 Turbo clips on a single RTX A6000.
# Model weights stay read-only and are never copied into the output directory.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_INDEX="${GPU_INDEX:-0}"
EXPECTED_UUID="${EXPECTED_UUID:-}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r2}"
MODEL_DIR="${MODEL_DIR:-$ROOT/models/MiniMax-H3-Turbo-Merged/FL2VA}"
PROMPT_FILE="${PROMPT_FILE:-$ROOT/examples/a6000-turbo-8step-sci-fi/prompt.txt}"
INPUT_REFERENCE="${INPUT_REFERENCE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/out/a6000-turbo-demo-$(date -u +%Y%m%dT%H%M%SZ)}"
STEPS="${STEPS:-8}"
SEEDS="${SEEDS:-42}"
WIDTH="${WIDTH:-1344}"
HEIGHT="${HEIGHT:-768}"
FPS="${FPS:-24}"
DURATION="${DURATION:-5.166667}"
FLOW_SHIFT="${FLOW_SHIFT:-12}"
AUDIO_FLOW_SHIFT="${AUDIO_FLOW_SHIFT:-3.0}"
DLO_RESIDENT_LAYERS="${DLO_RESIDENT_LAYERS:-12}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: I_ACCEPT_MINIMAX_H3_LICENSE=YES [environment options] bash scripts/run_turbo_demo.sh
       bash scripts/run_turbo_demo.sh --dry-run

Important environment options:
  GPU_INDEX=0                    Host GPU index; exactly one GPU is exposed.
  EXPECTED_UUID=GPU-...          Optional UUID guard for the selected GPU.
  RUNTIME_IMAGE=...              Local vLLM-Omni image.
  MODEL_DIR=...                  Prepared merged Turbo FL2VA directory.
  PROMPT_FILE=...                UTF-8 prompt text file.
  INPUT_REFERENCE=...            Optional PNG/JPEG/WebP first-frame reference; enables FL2VA.
  OUTPUT_DIR=...                 New result directory.
  STEPS=8                        Practical Turbo schedule (4 or 8).
  SEEDS=42                       One seed or comma-separated seeds, e.g. 42,137.

The script starts an isolated local API server in Docker with network disabled,
submits the requested prompt, validates H.264/AAC video and stereo audio, writes
per-clip timing/metadata, then removes the container. It refuses a busy GPU.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 64 ;;
  esac
done

case "$STEPS" in 4|8) ;; *) echo "ERROR: STEPS must be 4 or 8" >&2; exit 64 ;; esac
[[ "$WIDTH" =~ ^[0-9]+$ && "$HEIGHT" =~ ^[0-9]+$ && "$FPS" =~ ^[0-9]+$ ]] || {
  echo "ERROR: WIDTH, HEIGHT and FPS must be integers" >&2; exit 64;
}

if (( DRY_RUN )); then
  cat <<EOF
[DRY-RUN] MiniMax-H3 single-A6000 Turbo demo
GPU_INDEX=$GPU_INDEX
EXPECTED_UUID=${EXPECTED_UUID:-<not-set>}
RUNTIME_IMAGE=$RUNTIME_IMAGE
MODEL_DIR=$MODEL_DIR
PROMPT_FILE=$PROMPT_FILE
INPUT_REFERENCE=${INPUT_REFERENCE:-<text-only>}
OUTPUT_DIR=$OUTPUT_DIR
STEPS=$STEPS
SEEDS=$SEEDS
WORKLOAD=${WIDTH}x${HEIGHT}, ${DURATION}s, ${FPS} FPS, stereo audio
No Docker, GPU, model, network, or output action was performed.
EOF
  exit 0
fi

if [[ "${I_ACCEPT_MINIMAX_H3_LICENSE:-}" != "YES" ]]; then
  echo "ERROR: read the upstream MiniMax-H3 license, then set I_ACCEPT_MINIMAX_H3_LICENSE=YES" >&2
  exit 2
fi
command -v docker >/dev/null || { echo "ERROR: docker is required" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi is required" >&2; exit 2; }
test -r "$PROMPT_FILE" || { echo "ERROR: prompt file is not readable: $PROMPT_FILE" >&2; exit 2; }
if [[ -n "$INPUT_REFERENCE" ]]; then
  test -r "$INPUT_REFERENCE" || { echo "ERROR: input reference is not readable: $INPUT_REFERENCE" >&2; exit 2; }
fi
test -r "$MODEL_DIR/merge_manifest.json" || { echo "ERROR: prepared Turbo merge manifest missing: $MODEL_DIR/merge_manifest.json" >&2; exit 2; }
docker image inspect "$RUNTIME_IMAGE" >/dev/null 2>&1 || {
  echo "ERROR: runtime image not found: $RUNTIME_IMAGE (see README runtime build step)" >&2; exit 2;
}

python3 - "$MODEL_DIR/merge_manifest.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
assert m.get("status") == "completed", m.get("status")
assert m.get("merge", {}).get("strength") == 1.0
completed = m.get("completed_shards", {})
assert len(completed) == 13, len(completed)
PY

GPU_UUID="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
if [[ -n "$EXPECTED_UUID" && "$GPU_UUID" != "$EXPECTED_UUID" ]]; then
  echo "ERROR: GPU UUID mismatch: expected $EXPECTED_UUID, got $GPU_UUID" >&2
  exit 2
fi
if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -F "$GPU_UUID" >/dev/null; then
  echo "ERROR: selected GPU $GPU_INDEX ($GPU_UUID) already has a compute process" >&2
  exit 3
fi

mkdir -p "$OUTPUT_DIR/cache/hf" "$OUTPUT_DIR/cache/torchinductor" "$OUTPUT_DIR/cache/triton" "$OUTPUT_DIR/clips"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
cp "$PROMPT_FILE" "$OUTPUT_DIR/prompt.txt"
if [[ -n "$INPUT_REFERENCE" ]]; then
  # Keep the operator-supplied source private. The pinned runtime already ships
  # Pillow for MiniMax-H3 preprocessing and creates the normalized condition in
  # the isolated container without modifying the source image.
  cp "$INPUT_REFERENCE" "$OUTPUT_DIR/input_reference_source"
  sha256sum "$INPUT_REFERENCE" > "$OUTPUT_DIR/input_reference_source.sha256"
fi
printf '%s\n' "$SEEDS" | tr ',' '\n' | sed '/^[[:space:]]*$/d' > "$OUTPUT_DIR/seeds.txt"
python3 - "$OUTPUT_DIR/seeds.txt" <<'PY'
from pathlib import Path
import sys
seeds=[]
for raw in Path(sys.argv[1]).read_text().splitlines():
    value=int(raw.strip())
    if value < 0: raise SystemExit("seeds must be non-negative")
    seeds.append(value)
if not seeds or len(set(seeds)) != len(seeds): raise SystemExit("SEEDS must contain unique integer values")
PY
chmod -R a+rwX "$OUTPUT_DIR"

nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw,power.limit,temperature.gpu --format=csv,noheader > "$OUTPUT_DIR/host_gpu_preflight.csv"
cat > "$OUTPUT_DIR/container_run.sh" <<'CONTAINER'
#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY' > /evidence/container_gpu_preflight.json
import hashlib, json, os, pathlib, torch
from PIL import Image, ImageOps
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
p = torch.cuda.get_device_properties(0)
cap = list(torch.cuda.get_device_capability(0))
assert cap == [8, 6] and "a6000" in p.name.lower(), (p.name, cap)
reference = pathlib.Path("/evidence/input_reference_source")
reference_meta = {"present": False}
if reference.is_file():
    width, height = int(os.environ["WIDTH"]), int(os.environ["HEIGHT"])
    with Image.open(reference) as source:
        source.load()
        if source.width < 256 or source.height < 256:
            raise ValueError(f"input reference is too small: {source.width}x{source.height}")
        normalized = ImageOps.fit(
            source.convert("RGB"),
            (width, height),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        normalized.save("/evidence/input_reference.png", format="PNG", optimize=True)
        reference_meta = {
            "present": True,
            "source_width": source.width,
            "source_height": source.height,
            "normalized_width": width,
            "normalized_height": height,
            "source_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
            "normalized_sha256": hashlib.sha256(pathlib.Path("/evidence/input_reference.png").read_bytes()).hexdigest(),
        }
print(json.dumps({"visible_gpu_count": 1, "name": p.name, "compute_capability": cap, "input_reference": reference_meta}, indent=2))
PY

vllm-omni serve /models/Turbo/FL2VA \
  --omni --trust-remote-code --host 127.0.0.1 --port 8000 --task-type fl2va \
  --num-gpus 1 --tensor-parallel-size 1 --text-encoder-tp-size 1 --usp 1 --ring 1 \
  --vae-patch-parallel-size 1 --vae-parallel-mode tile --vae-use-tiling \
  --enable-distributed-layerwise-offload --dlo-no-use-allgather \
  --dlo-resident-layers "$DLO_RESIDENT_LAYERS" \
  --enforce-eager --diffusion-attention-backend CUDNN_ATTN > /evidence/server.log 2>&1 &
server_pid=$!
monitor_pid=""
cleanup() {
  kill "$server_pid" >/dev/null 2>&1 || true
  [[ -z "$monitor_pid" ]] || kill "$monitor_pid" >/dev/null 2>&1 || true
}
trap cleanup EXIT

printf 'timestamp,gpu_memory_used_mib,gpu_util_percent,power_w,temperature_c,host_memory_used_bytes,host_memory_available_bytes,host_swap_used_bytes\n' > /evidence/resource_monitor.csv
(
  while kill -0 "$server_pid" 2>/dev/null; do
    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits) || exit
    IFS=',' read -r mem util power temp <<< "$gpu"
    mem=${mem// /}; util=${util// /}; power=${power// /}; temp=${temp// /}
    read -r hu ha su < <(free -b | awk '/^Mem:/{u=$3;a=$7}/^Swap:/{s=$3}END{print u,a,s}')
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$ts" "$mem" "$util" "$power" "$temp" "$hu" "$ha" "$su" >> /evidence/resource_monitor.csv
    if (( mem > 45500 || temp > 88 )); then
      echo "$ts safety threshold memory_mib=$mem temperature_c=$temp" > /evidence/SAFETY_STOP_REASON.txt
      kill "$server_pid" >/dev/null 2>&1 || true
      exit
    fi
    sleep 5
  done
) &
monitor_pid=$!

ready=0
for _ in $(seq 1 720); do
  if grep -q 'Application startup complete\.' /evidence/server.log && curl --fail --silent http://127.0.0.1:8000/health >/dev/null; then
    ready=1
    break
  fi
  kill -0 "$server_pid" 2>/dev/null || { tail -300 /evidence/server.log; exit 61; }
  [[ ! -s /evidence/SAFETY_STOP_REASON.txt ]] || exit 70
  sleep 5
done
[[ "$ready" == 1 ]] || { echo "ERROR: server readiness timeout" >&2; exit 62; }

while IFS= read -r seed; do
  name="turbo_${STEPS}step_seed${seed}"
  task=t2va
  extra_params="{\"task\":\"t2va\",\"duration\":${DURATION},\"audio_flow_shift\":${AUDIO_FLOW_SHIFT}}"
  reference_args=()
  if [[ -s /evidence/input_reference.png ]]; then
    task=fl2va
    extra_params="{\"task\":\"fl2va\",\"duration\":${DURATION},\"audio_flow_shift\":${AUDIO_FLOW_SHIFT},\"frame_indices\":[0]}"
    reference_args=(-F "input_reference=@/evidence/input_reference.png;type=image/png")
  fi
  curl --fail-with-body --silent --show-error --max-time 3600 \
    --dump-header "/evidence/clips/${name}_headers.txt" \
    --write-out 'http_code=%{http_code}\ntime_total_s=%{time_total}\nsize_download=%{size_download}\n' \
    -X POST http://127.0.0.1:8000/v1/videos/sync \
    -F 'prompt=</evidence/prompt.txt' \
    -F "width=${WIDTH}" -F "height=${HEIGHT}" -F "aspect_ratio=16:9" -F "fps=${FPS}" \
    -F "num_inference_steps=${STEPS}" -F "flow_shift=${FLOW_SHIFT}" -F "seed=${seed}" -F 'quality=lossless' \
    "${reference_args[@]}" \
    -F "extra_params=${extra_params}" \
    -o "/evidence/clips/${name}.mp4" > "/evidence/clips/${name}_http_metrics.txt"

  python3 - "$name" "$seed" <<'PY'
import av, hashlib, json, pathlib, sys
name, seed = sys.argv[1], int(sys.argv[2])
p = pathlib.Path("/evidence/clips") / f"{name}.mp4"
assert p.stat().st_size > 1024
with av.open(str(p)) as container:
    videos = [s for s in container.streams if s.type == "video"]
    audios = [s for s in container.streams if s.type == "audio"]
    assert len(videos) == len(audios) == 1
    video_frames = audio_frames = audio_samples = 0
    for frame in container.decode():
        if isinstance(frame, av.VideoFrame): video_frames += 1
        elif isinstance(frame, av.AudioFrame):
            audio_frames += 1
            audio_samples += frame.samples
    v, a = videos[0], audios[0]
    record = {
        "schema_version": "minimax-h3-a6000-public-demo-v1",
        "track": "practical_disclosed_approx",
        "schedule_steps": int(__import__("os").environ["STEPS"]),
        "seed": seed,
        "task": "fl2va" if pathlib.Path("/evidence/input_reference.png").is_file() else "t2va",
        "input_reference_present": pathlib.Path("/evidence/input_reference.png").is_file(),
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "bytes": p.stat().st_size,
        "width": v.codec_context.width,
        "height": v.codec_context.height,
        "average_rate": str(v.average_rate),
        "decoded_video_frames": video_frames,
        "audio_sample_rate_hz": a.codec_context.sample_rate,
        "audio_channels": a.codec_context.channels,
        "decoded_audio_frames": audio_frames,
        "decoded_audio_samples": audio_samples,
        "structural_av_contract_pass": False,
    }
    assert record["width"] == int(__import__("os").environ["WIDTH"])
    assert record["height"] == int(__import__("os").environ["HEIGHT"])
    assert video_frames == 124
    assert record["audio_sample_rate_hz"] == 32000 and record["audio_channels"] == 2
    assert audio_frames > 0 and audio_samples > 0
    record["structural_av_contract_pass"] = True
(pathlib.Path("/evidence/clips") / f"{name}_av_validation.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY
  [[ ! -s /evidence/SAFETY_STOP_REASON.txt ]] || exit 70
done < /evidence/seeds.txt

kill "$server_pid" >/dev/null 2>&1 || true
wait "$server_pid" >/dev/null 2>&1 || true
kill "$monitor_pid" >/dev/null 2>&1 || true
wait "$monitor_pid" >/dev/null 2>&1 || true
CONTAINER
chmod +x "$OUTPUT_DIR/container_run.sh"

set +e
timeout --kill-after=60s 90m docker run --rm \
  --gpus "device=$GPU_INDEX" --ipc=host --ulimit memlock=-1 \
  --network none --cap-drop=ALL \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200 \
  -e HF_HOME=/workspace/cache/hf \
  -e TRANSFORMERS_CACHE=/workspace/cache/hf \
  -e TORCHINDUCTOR_CACHE_DIR=/workspace/cache/torchinductor \
  -e TRITON_CACHE_DIR=/workspace/cache/triton \
  -e STEPS="$STEPS" -e WIDTH="$WIDTH" -e HEIGHT="$HEIGHT" -e FPS="$FPS" \
  -e DURATION="$DURATION" -e FLOW_SHIFT="$FLOW_SHIFT" -e AUDIO_FLOW_SHIFT="$AUDIO_FLOW_SHIFT" \
  -e DLO_RESIDENT_LAYERS="$DLO_RESIDENT_LAYERS" \
  -v "$MODEL_DIR":/models/Turbo/FL2VA:ro \
  -v "$OUTPUT_DIR":/evidence:rw \
  -v "$OUTPUT_DIR/cache":/workspace/cache:rw \
  "$RUNTIME_IMAGE" bash /evidence/container_run.sh > "$OUTPUT_DIR/console.log" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > "$OUTPUT_DIR/exit_code"
if (( rc != 0 )); then
  tail -300 "$OUTPUT_DIR/console.log" >&2
  exit "$rc"
fi

python3 - "$OUTPUT_DIR" "$GPU_INDEX" "$GPU_UUID" "$RUNTIME_IMAGE" "$STEPS" <<'PY'
import csv, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = list(csv.DictReader((root / "resource_monitor.csv").open()))
clips=[]
for path in sorted((root / "clips").glob("*_av_validation.json")):
    record=json.loads(path.read_text())
    metrics={}
    metric_path=path.with_name(path.name.replace("_av_validation.json", "_http_metrics.txt"))
    for line in metric_path.read_text().splitlines():
        if "=" in line:
            k,v=line.split("=",1); metrics[k]=v
    record["http_time_total_s"]=float(metrics["time_total_s"])
    clips.append(record)
reference = root / "input_reference.png"
summary={
    "schema_version":"minimax-h3-a6000-public-demo-run-v1",
    "status":"pass",
    "track":"practical_disclosed_approx",
    "task":"fl2va" if reference.is_file() else "t2va",
    "input_reference": ({
        "present": True,
        "normalized_sha256": __import__("hashlib").sha256(reference.read_bytes()).hexdigest(),
        "redistribution": "not included in the public example; operator-supplied local reference",
    } if reference.is_file() else {"present": False}),
    "host_gpu_index":int(sys.argv[2]),
    "gpu_uuid":sys.argv[3],
    "runtime_image":sys.argv[4],
    "steps":int(sys.argv[5]),
    "clips":clips,
    "resource":{
        "sample_count":len(rows),
        "peak_gpu_memory_mib":max(float(r["gpu_memory_used_mib"]) for r in rows),
        "peak_power_w":max(float(r["power_w"]) for r in rows),
        "peak_temperature_c":max(float(r["temperature_c"]) for r in rows),
        "peak_host_memory_used_gib":max(float(r["host_memory_used_bytes"]) for r in rows)/(2**30),
    },
    "claim_boundary":"Public demo artifacts only; Turbo is a disclosed practical approximation, not BF16-exact fidelity.",
}
(root / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
# Container-created cache files may be owned by root on some Docker setups.
# Result files are already world-readable; cache chmod failures are non-fatal.
chmod -R a+rX "$OUTPUT_DIR" 2>/dev/null || true
echo "Turbo demo complete: $OUTPUT_DIR"
