#!/usr/bin/env python3
"""Private MiniMax-H3 final-AV 30s extension N=1 runner/finalizer.

This tool is intentionally evidence-local: the shell wrapper copies it into the
run directory and executes the copied file inside the isolated Docker container.
It does not download assets or mutate model weights.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any

WIDTH = 1344
HEIGHT = 768
FPS = 24
SOURCE_FRAMES_PER_CHUNK = 124
FINAL_FRAMES = 720
CHUNK_COUNT = 6
RETAINED_FRAMES_PER_CHUNK = 120
AUDIO_RATE = 32000
AUDIO_CHANNELS = 2
AUDIO_SAMPLES_PER_CHUNK = 160_000
FINAL_AUDIO_SAMPLES_PER_CHANNEL = 960_000
SOURCE_DURATION_SECONDS = 5.166667
FINAL_DURATION_SECONDS = 30.0
LANE_ID = "final-av-30s-1344x768-24fps-v1"
DURATION_LABEL = "30s"
DURATION_READABLE = "30-second"
TIMING_BOUNDARY_ID = "final_av_30s_extension_warm_after_one_excluded_warmup_v1"

CACHE_DIT_REQUEST_PROFILES: dict[str, dict[str, Any]] = {
    "high": {
        "mechanism": "MiniMax-H3 request-scoped Cache-DiT high profile",
        "cache_mechanism": "request_scoped_cache_dit_high",
        "mechanism_tokens": [
            "request_scoped_cache_dit_high",
            "dbcache_fn1_threshold_0_04_warmup4_max_continuous_1",
            "practical_approximate_cache",
        ],
        "Fn_compute_blocks": 1,
        "Bn_compute_blocks": 0,
        "max_warmup_steps": 4,
        "residual_diff_threshold": 0.04,
        "max_continuous_cached_steps": 1,
        "enable_taylorseer": False,
        "scm_steps_mask_policy": None,
    },
    "high_warmup2": {
        "mechanism": "MiniMax-H3 request-scoped Cache-DiT high profile with max_warmup_steps=2",
        "cache_mechanism": "request_scoped_cache_dit_high_warmup2",
        "mechanism_tokens": [
            "request_scoped_cache_dit_high_warmup2",
            "dbcache_fn1_threshold_0_04_warmup2_max_continuous_1",
            "practical_approximate_cache",
        ],
        "Fn_compute_blocks": 1,
        "Bn_compute_blocks": 0,
        "max_warmup_steps": 2,
        "residual_diff_threshold": 0.04,
        "max_continuous_cached_steps": 1,
        "enable_taylorseer": False,
        "scm_steps_mask_policy": None,
    },
}
CACHE_DIT_REQUEST_QUALITIES = tuple(["lossless", *CACHE_DIT_REQUEST_PROFILES.keys()])


def cache_dit_profile_label(request_quality: str) -> str:
    if request_quality == "high":
        return "cache_dit_high"
    if request_quality == "high_warmup2":
        return "cache_dit_high_warmup2"
    if request_quality == "lossless":
        return "cache_off_lossless"
    return request_quality.replace("-", "_")


def cache_dit_expected_config(request_quality: str) -> dict[str, Any]:
    expected = {
        "quality": request_quality,
    }
    profile = CACHE_DIT_REQUEST_PROFILES.get(request_quality)
    if profile:
        expected.update({key: value for key, value in profile.items() if key != "mechanism_tokens"})
    else:
        expected["mechanism"] = "lossless request path; Cache-DiT disabled for the request"
    return expected


def _format_seconds(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}"


def _duration_label(seconds: float) -> str:
    if abs(seconds - round(seconds)) < 1e-9:
        return f"{int(round(seconds))}s"
    return f"{seconds:g}s".replace(".", "p")


def _duration_readable(seconds: float) -> str:
    if abs(seconds - round(seconds)) < 1e-9:
        return f"{int(round(seconds))}-second"
    return f"{seconds:g}-second"


def configure_lane_from_manifest(manifest: dict[str, Any]) -> None:
    """Apply the frozen final-AV lane manifest to the evidence-local runner.

    The original tool was written for the 30-second lane.  The v1 contract now
    also freezes a 60-second lane with the same source chunk cell, so the runner
    derives exact final frames, audio samples, and chunk count from the manifest
    before any media generation or record finalization.
    """
    global WIDTH, HEIGHT, FPS, SOURCE_FRAMES_PER_CHUNK, FINAL_FRAMES, CHUNK_COUNT
    global RETAINED_FRAMES_PER_CHUNK, AUDIO_RATE, AUDIO_CHANNELS, AUDIO_SAMPLES_PER_CHUNK
    global FINAL_AUDIO_SAMPLES_PER_CHANNEL, SOURCE_DURATION_SECONDS, FINAL_DURATION_SECONDS
    global LANE_ID, DURATION_LABEL, DURATION_READABLE, TIMING_BOUNDARY_ID

    workload = manifest.get("workload", {})
    production = manifest.get("production", {})
    accounting = manifest.get("final_av_accounting", {})
    audio_retention = production.get("audio_retention", {}) if isinstance(production.get("audio_retention"), dict) else {}

    LANE_ID = str(manifest.get("lane_id") or LANE_ID)
    WIDTH = int(workload.get("width", WIDTH))
    HEIGHT = int(workload.get("height", HEIGHT))
    FPS = int(workload.get("fps", FPS))
    SOURCE_FRAMES_PER_CHUNK = int(production.get("source_frames_per_chunk", SOURCE_FRAMES_PER_CHUNK))
    FINAL_FRAMES = int(workload.get("final_frame_count", accounting.get("video_frames", FINAL_FRAMES)))
    CHUNK_COUNT = int(production.get("chunk_count", CHUNK_COUNT))
    if CHUNK_COUNT <= 0:
        raise RuntimeError(f"invalid chunk_count in manifest: {CHUNK_COUNT}")
    if FINAL_FRAMES % CHUNK_COUNT != 0:
        raise RuntimeError(f"final frames {FINAL_FRAMES} are not divisible by chunks {CHUNK_COUNT}")
    RETAINED_FRAMES_PER_CHUNK = FINAL_FRAMES // CHUNK_COUNT
    AUDIO_RATE = int(workload.get("audio_sample_rate_hz", AUDIO_RATE))
    AUDIO_CHANNELS = int(workload.get("audio_channels", AUDIO_CHANNELS))
    AUDIO_SAMPLES_PER_CHUNK = int(audio_retention.get("samples_per_chunk_per_channel", AUDIO_SAMPLES_PER_CHUNK))
    FINAL_AUDIO_SAMPLES_PER_CHANNEL = int(accounting.get("effective_audio_samples_per_channel", AUDIO_SAMPLES_PER_CHUNK * CHUNK_COUNT))
    if FINAL_AUDIO_SAMPLES_PER_CHANNEL != AUDIO_SAMPLES_PER_CHUNK * CHUNK_COUNT:
        raise RuntimeError(
            f"final audio samples {FINAL_AUDIO_SAMPLES_PER_CHANNEL} do not equal per-chunk {AUDIO_SAMPLES_PER_CHUNK} * chunks {CHUNK_COUNT}"
        )
    FINAL_DURATION_SECONDS = float(workload.get("nominal_duration_seconds", FINAL_FRAMES / FPS))
    SOURCE_DURATION_SECONDS = SOURCE_FRAMES_PER_CHUNK / float(FPS)
    DURATION_LABEL = _duration_label(FINAL_DURATION_SECONDS)
    DURATION_READABLE = _duration_readable(FINAL_DURATION_SECONDS)
    TIMING_BOUNDARY_ID = f"final_av_{DURATION_LABEL}_extension_warm_after_one_excluded_warmup_v1"


def final_audio_raw_name() -> str:
    return f"final_audio_{FINAL_AUDIO_SAMPLES_PER_CHANNEL}x{AUDIO_CHANNELS}_s16le.raw"


def final_video_name() -> str:
    return f"final_video_{FINAL_FRAMES}f_h264.mov"


def final_av_name() -> str:
    return f"final_av_{DURATION_LABEL}_extension_n1.mov"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(args: list[str], *, stdout: Path | None = None, stderr: Path | None = None, check: bool = False, text: bool = True, timeout: float | None = None) -> subprocess.CompletedProcess:
    out_handle = stdout.open("w", encoding="utf-8") if stdout is not None and text else (stdout.open("wb") if stdout is not None else subprocess.PIPE)
    err_handle = stderr.open("w", encoding="utf-8") if stderr is not None and text else (stderr.open("wb") if stderr is not None else subprocess.PIPE)
    try:
        return subprocess.run(args, stdout=out_handle, stderr=err_handle, check=check, text=text, timeout=timeout)
    finally:
        if stdout is not None:
            out_handle.close()
        if stderr is not None:
            err_handle.close()


def wait_for_process_after_stdin_close(proc: subprocess.Popen[Any], *, timeout: float) -> None:
    """Wait for a process whose stdin pipe was manually closed by the caller.

    subprocess.communicate() flushes stdin internally; calling it after an
    explicit proc.stdin.close() raises ValueError on Python 3.12.  The final-AV
    encoder streams raw frames to stdin, closes the pipe to signal EOF, and must
    wait without communicate().
    """
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive guard
            pass
        raise


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_headers(path: Path) -> dict[str, str]:
    headers: dict[str, str] = {}
    # curl -D may include multiple HTTP header blocks; the last value for a key wins.
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        headers[key.strip().lower()] = value.strip()
    return headers


def parse_http_metrics(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key in {"http_code"}:
            try:
                result[key] = int(value)
            except ValueError:
                result[key] = value
        elif key.startswith("time_") or key in {"size_download"}:
            try:
                result[key] = float(value)
            except ValueError:
                result[key] = value
        else:
            result[key] = value
    return result


def finite_seconds(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) and out >= 0.0 else None
    if isinstance(value, str):
        try:
            out = float(value.strip())
        except ValueError:
            return None
        return out if math.isfinite(out) and out >= 0.0 else None
    return None


def request_timing_breakdown(*, curl_wall_s: float, http_metrics: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    """Separate client wait, server wall, and transfer for sync video requests.

    The sync endpoint returns ``X-Inference-Time-S`` after the server has encoded
    the MP4 bytes.  On localhost curl may report ``time_starttransfer`` near
    zero for this endpoint, so the reliable response-transfer upper bound is
    ``time_total - X-Inference-Time-S`` rather than ``time_total -
    time_starttransfer``.  The latter is retained only as a diagnostic.
    """
    http_total = finite_seconds(http_metrics.get("time_total_s"))
    starttransfer = finite_seconds(http_metrics.get("time_starttransfer_s"))
    server_wall = finite_seconds(headers.get("x-inference-time-s"))
    client_wait = finite_seconds(curl_wall_s)
    client_minus_http = None
    if client_wait is not None and http_total is not None:
        client_minus_http = max(0.0, client_wait - http_total)
    starttransfer_transfer = None
    if http_total is not None and starttransfer is not None:
        starttransfer_transfer = max(0.0, http_total - starttransfer)
    transfer = None
    source = "unavailable"
    if http_total is not None and server_wall is not None:
        transfer = max(0.0, http_total - server_wall)
        source = "http_total_minus_x_inference_time_s"
    elif starttransfer_transfer is not None:
        transfer = starttransfer_transfer
        source = "curl_time_total_minus_time_starttransfer_fallback"
    return {
        "schema_version": "minimax_h3_a6000_request_timing_breakdown_v1",
        "true_client_wait_seconds": client_wait,
        "curl_http_total_seconds": http_total,
        "server_request_wall_seconds": server_wall,
        "response_transfer_seconds": transfer,
        "response_transfer_source": source,
        "curl_reported_starttransfer_seconds": starttransfer,
        "curl_starttransfer_based_transfer_seconds": starttransfer_transfer,
        "curl_process_overhead_seconds": client_minus_http,
        "size_download_bytes": finite_seconds(http_metrics.get("size_download")),
        "caveat": "Do not add true_client_wait/server_request_wall/response_transfer together: server_request_wall is nested inside client wait; response_transfer is the post-server byte-transfer upper bound.",
    }


def ensure_av_numpy() -> tuple[Any, Any]:
    import av  # type: ignore
    import numpy as np  # type: ignore

    return av, np


def ensure_imports() -> tuple[Any, Any, Any]:
    av, np = ensure_av_numpy()
    from PIL import Image  # type: ignore

    return av, np, Image


class ResourceMonitor:
    def __init__(self, path: Path, server_proc: subprocess.Popen[Any], interval_s: float = 5.0) -> None:
        self.path = path
        self.server_proc = server_proc
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self.failures: list[str] = []

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            "timestamp,gpu_memory_used_mib,gpu_util_percent,power_w,temperature_c,host_memory_used_bytes,host_memory_available_bytes,host_swap_used_bytes\n",
            encoding="utf-8",
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)

    def _run(self) -> None:
        while not self.stop_event.is_set() and self.server_proc.poll() is None:
            ts = utc_now()
            try:
                gpu = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip().splitlines()[0]
                mem, util, power, temp = [item.strip().replace(" ", "") for item in gpu.split(",")]
                free_out = subprocess.check_output(["free", "-b"], text=True, stderr=subprocess.DEVNULL)
                host_used = host_avail = swap_used = "0"
                for line in free_out.splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    if parts[0] == "Mem:":
                        host_used = parts[2]
                        host_avail = parts[6] if len(parts) > 6 else parts[-1]
                    elif parts[0] == "Swap:":
                        swap_used = parts[2]
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(f"{ts},{mem},{util},{power},{temp},{host_used},{host_avail},{swap_used}\n")
                if float(mem) > 45500 or float(temp) > 88:
                    self.failures.append(f"safety_stop memory_mib={mem} temperature_c={temp}")
                    try:
                        self.server_proc.terminate()
                    except Exception:
                        pass
                    return
            except Exception as exc:  # noqa: BLE001
                self.failures.append(f"resource_monitor_error:{type(exc).__name__}:{exc}")
            self.stop_event.wait(self.interval_s)


def wait_ready(server_proc: subprocess.Popen[Any], server_log: Path, timeout_s: int) -> float:
    start = time.perf_counter()
    deadline = start + timeout_s
    last_error = ""
    while time.perf_counter() < deadline:
        if server_proc.poll() is not None:
            tail = server_log.read_text(encoding="utf-8", errors="replace")[-6000:] if server_log.exists() else ""
            raise RuntimeError(f"server exited before readiness rc={server_proc.returncode}\n{tail}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as response:
                if 200 <= response.status < 300:
                    return time.perf_counter() - start
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}:{exc}"
        time.sleep(5)
    tail = server_log.read_text(encoding="utf-8", errors="replace")[-6000:] if server_log.exists() else ""
    raise TimeoutError(f"server readiness timeout after {timeout_s}s last_error={last_error}\n{tail}")


def container_gpu_preflight(evidence: Path) -> dict[str, Any]:
    code = """
import json, torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
p = torch.cuda.get_device_properties(0)
cap = list(torch.cuda.get_device_capability(0))
assert cap == [8, 6] and 'a6000' in p.name.lower(), (p.name, cap)
print(json.dumps({'visible_gpu_count': 1, 'name': p.name, 'compute_capability': cap}, indent=2, sort_keys=True))
""".strip()
    out = subprocess.check_output([sys.executable, "-c", code], text=True)
    payload = json.loads(out)
    write_json(evidence / "container_gpu_preflight.json", payload)
    return payload


def start_server(
    evidence: Path,
    model_dir: str,
    init_timeout_s: int,
    stage_init_timeout_s: int,
    dlo_resident_layers: int,
    attention_backend: str = "CUDNN_ATTN",
    server_cache_backend: str = "none",
    enable_cache_dit_summary: bool = False,
    regional_compile: bool = False,
    diffusion_compile_dynamic: bool = True,
) -> subprocess.Popen[Any]:
    server_log = (evidence / "server.log").open("w", encoding="utf-8")
    cmd = [
        "vllm-omni",
        "serve",
        model_dir,
        "--omni",
        "--trust-remote-code",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--task-type",
        "fl2va",
        "--num-gpus",
        "1",
        "--tensor-parallel-size",
        "1",
        "--text-encoder-tp-size",
        "1",
        "--usp",
        "1",
        "--ring",
        "1",
        "--vae-patch-parallel-size",
        "1",
        "--vae-parallel-mode",
        "tile",
        "--vae-use-tiling",
        "--enable-distributed-layerwise-offload",
        "--dlo-no-use-allgather",
        "--dlo-resident-layers",
        str(dlo_resident_layers),
        "--diffusion-attention-backend",
        str(attention_backend),
        "--enable-diffusion-pipeline-profiler",
        "--init-timeout",
        str(init_timeout_s),
        "--stage-init-timeout",
        str(stage_init_timeout_s),
    ]
    if regional_compile:
        cmd.extend(["--diffusion-compile-granularity", "regional"])
        if not diffusion_compile_dynamic:
            cmd.append("--no-diffusion-compile-dynamic")
    else:
        cmd.append("--enforce-eager")
    cache_backend = str(server_cache_backend or "none").lower()
    if cache_backend not in {"none", "cache_dit"}:
        raise ValueError(f"unsupported server_cache_backend={server_cache_backend!r}")
    if cache_backend == "cache_dit":
        cmd.extend(["--cache-backend", "cache_dit"])
        if enable_cache_dit_summary:
            cmd.append("--enable-cache-dit-summary")
    elif enable_cache_dit_summary:
        raise ValueError("enable_cache_dit_summary requires server_cache_backend='cache_dit'")
    (evidence / "server_command.json").write_text(json.dumps(cmd, indent=2) + "\n", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=server_log, stderr=subprocess.STDOUT, text=True)


def chunk_prompt(base_prompt: str, index: int) -> str:
    segment_seconds = FINAL_DURATION_SECONDS / float(CHUNK_COUNT)
    start = (index - 1) * segment_seconds
    end = index * segment_seconds
    return (
        base_prompt.strip()
        + "\n\n"
        + f"Continuation segment {index} of {CHUNK_COUNT}; cover final-video seconds {_format_seconds(start)} through {_format_seconds(end)}; "
        + "preserve the prior retained terminal frame exactly as the first-frame condition."
    )


def request_video(
    *,
    chunk_dir: Path,
    prompt_file: Path,
    output_file: Path,
    seed: int,
    task: str,
    conditioner: Path | None,
    duration_s: float,
    steps: int,
    flow_shift: float,
    audio_flow_shift: float,
    timeout_s: int,
    request_quality: str = "lossless",
) -> dict[str, Any]:
    headers_path = chunk_dir / f"{output_file.stem}_headers.txt"
    metrics_path = chunk_dir / f"{output_file.stem}_http_metrics.txt"
    if request_quality not in CACHE_DIT_REQUEST_QUALITIES:
        raise ValueError(f"unsupported MiniMax-H3 request quality: {request_quality!r}")
    extra: dict[str, Any] = {"task": task, "duration": duration_s, "audio_flow_shift": audio_flow_shift}
    if task == "fl2va":
        extra["frame_indices"] = [0]
    cmd = [
        "curl",
        "--fail-with-body",
        "--silent",
        "--show-error",
        "--max-time",
        str(timeout_s),
        "--dump-header",
        str(headers_path),
        "--write-out",
        "http_code=%{http_code}\ntime_namelookup_s=%{time_namelookup}\ntime_connect_s=%{time_connect}\ntime_pretransfer_s=%{time_pretransfer}\ntime_starttransfer_s=%{time_starttransfer}\ntime_total_s=%{time_total}\nsize_download=%{size_download}\n",
        "-X",
        "POST",
        "http://127.0.0.1:8000/v1/videos/sync",
        "-F",
        f"prompt=<{prompt_file}",
        "-F",
        f"width={WIDTH}",
        "-F",
        f"height={HEIGHT}",
        "-F",
        "aspect_ratio=16:9",
        "-F",
        f"fps={FPS}",
        "-F",
        f"num_inference_steps={steps}",
        "-F",
        f"flow_shift={flow_shift:g}",
        "-F",
        f"seed={seed}",
        "-F",
        f"quality={request_quality}",
        "-F",
        "extra_params=" + json.dumps(extra, separators=(",", ":")),
        "-o",
        str(output_file),
    ]
    if conditioner is not None:
        cmd.extend(["-F", f"input_reference=@{conditioner};type=image/png"])
    (chunk_dir / f"{output_file.stem}_request_command.json").write_text(json.dumps(cmd, indent=2) + "\n", encoding="utf-8")
    t0 = time.perf_counter()
    with metrics_path.open("w", encoding="utf-8") as metrics:
        proc = subprocess.run(cmd, stdout=metrics, stderr=subprocess.PIPE, text=True)
    wall_s = time.perf_counter() - t0
    if proc.returncode != 0:
        (chunk_dir / f"{output_file.stem}_curl_stderr.txt").write_text(proc.stderr, encoding="utf-8")
        raise RuntimeError(f"curl failed for {output_file.name} rc={proc.returncode}: {proc.stderr[-2000:]}")
    headers = parse_headers(headers_path)
    metrics_obj = parse_http_metrics(metrics_path)
    stage_durations: dict[str, float] = {}
    if headers.get("x-stage-durations"):
        try:
            raw = json.loads(headers["x-stage-durations"])
            if isinstance(raw, dict):
                stage_durations = {str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float))}
        except Exception:
            stage_durations = {}
    timing_breakdown = request_timing_breakdown(curl_wall_s=wall_s, http_metrics=metrics_obj, headers=headers)
    response_download_s = timing_breakdown.get("response_transfer_seconds")
    record = {
        "created_utc": utc_now(),
        "task": task,
        "seed": seed,
        "duration_seconds": duration_s,
        "num_inference_steps": steps,
        "request_quality": request_quality,
        "cache_mechanism": CACHE_DIT_REQUEST_PROFILES.get(request_quality, {}).get("cache_mechanism", "cache_disabled_lossless_request"),
        "conditioner": str(conditioner) if conditioner is not None else None,
        "output": str(output_file),
        "output_bytes": output_file.stat().st_size,
        "output_sha256_opaque_id": sha256_file(output_file),
        "curl_wall_seconds": wall_s,
        "http_metrics": metrics_obj,
        "response_headers_subset": {
            "x-request-id": headers.get("x-request-id"),
            "x-inference-time-s": headers.get("x-inference-time-s"),
            "x-peak-memory-mb": headers.get("x-peak-memory-mb"),
        },
        "stage_durations": stage_durations,
        "request_timing_breakdown": timing_breakdown,
        "server_request_wall_seconds": timing_breakdown.get("server_request_wall_seconds"),
        "true_client_wait_seconds": timing_breakdown.get("true_client_wait_seconds"),
        "response_download_seconds": response_download_s,
    }
    write_json(chunk_dir / f"{output_file.stem}_request_record.json", record)
    return record


def decode_chunk_validation(chunk_dir: Path, output_file: Path, chunk_index: int, retain_start: int, retain_end: int) -> dict[str, Any]:
    av, np, Image = ensure_imports()
    video_frames = []
    audio_samples = 0
    width = height = None
    fps_text = None
    with av.open(str(output_file)) as container:
        video_streams = [s for s in container.streams if s.type == "video"]
        audio_streams = [s for s in container.streams if s.type == "audio"]
        if len(video_streams) != 1 or len(audio_streams) != 1:
            raise RuntimeError(f"{output_file}: expected exactly one video and one audio stream")
        v = video_streams[0]
        a = audio_streams[0]
        width = int(v.codec_context.width)
        height = int(v.codec_context.height)
        fps_text = str(v.average_rate)
        for frame in container.decode(video=0):
            video_frames.append(frame.to_ndarray(format="rgb24"))
    with av.open(str(output_file)) as container:
        for frame in container.decode(audio=0):
            audio_samples += int(frame.samples)
    if width != WIDTH or height != HEIGHT:
        raise RuntimeError(f"{output_file}: wrong geometry {width}x{height}")
    if len(video_frames) != SOURCE_FRAMES_PER_CHUNK:
        raise RuntimeError(f"{output_file}: expected {SOURCE_FRAMES_PER_CHUNK} decoded frames, got {len(video_frames)}")
    if audio_samples <= 0:
        raise RuntimeError(f"{output_file}: no decoded audio samples")
    retained = video_frames[retain_start:retain_end]
    if len(retained) != RETAINED_FRAMES_PER_CHUNK:
        raise RuntimeError(f"{output_file}: retained {len(retained)} frames, expected {RETAINED_FRAMES_PER_CHUNK}")
    terminal_frame_index = retain_end - 1
    terminal = retained[-1]
    terminal_rgb_sha = sha256_bytes(terminal.tobytes())
    terminal_png = chunk_dir / "retained_terminal_frame.png"
    Image.fromarray(terminal).save(terminal_png)
    validation = {
        "schema_version": "minimax-h3-final-av-extension-chunk-validation-v1",
        "chunk_index": chunk_index,
        "source_mp4": str(output_file),
        "source_video_frames": len(video_frames),
        "source_audio_decoded_samples_per_channel": int(audio_samples),
        "width": width,
        "height": height,
        "fps": fps_text,
        "retention_zero_based_half_open": [retain_start, retain_end],
        "retained_frame_count": len(retained),
        "retained_terminal_frame_source_index": terminal_frame_index,
        "retained_terminal_frame_png": str(terminal_png),
        "retained_terminal_frame_rgb_sha256": terminal_rgb_sha,
        "retained_terminal_frame_png_sha256": sha256_file(terminal_png),
        "structural_source_av_pass": True,
    }
    write_json(chunk_dir / "media_validation.json", validation)
    return validation


def pcm_from_mp4(path: Path) -> bytes:
    return subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            str(AUDIO_CHANNELS),
            "-ar",
            str(AUDIO_RATE),
            "-f",
            "s16le",
            "pipe:1",
        ]
    )


def build_final_video_ffmpeg_command(final_video: Path) -> list[str]:
    """Encode exactly the manifest final RGB frame count to a video-only H.264 MOV.

    Keep the rawvideo pipe in a video-only subprocess.  FFmpeg 4.4 can still
    produce a short PCM track when rawvideo stdin and exact raw audio are muxed
    in the same invocation, even without ``-shortest``.  The audio is therefore
    muxed only after the video encoder has exited cleanly.
    """
    return [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "pipe:0",
        "-frames:v",
        str(FINAL_FRAMES),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        str(final_video),
    ]


def build_final_av_ffmpeg_command(final_video: Path, final_audio: Path, final_mov: Path) -> list[str]:
    """Build the exact-size final AV remux command.

    Inputs are contract-bounded before muxing: the video-only file must decode
    to the manifest frame count and ``final_audio`` must contain exactly the
    manifest stereo s16 samples/channel.  Do not combine rawvideo stdin with the
    PCM input here and do not add ``-shortest``; both patterns previously yielded
    apparently complete video with truncated decoded PCM.
    """
    return [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(final_video),
        "-f",
        "s16le",
        "-ar",
        str(AUDIO_RATE),
        "-ac",
        str(AUDIO_CHANNELS),
        "-i",
        str(final_audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "pcm_s16le",
        str(final_mov),
    ]


def assemble_final_av(evidence: Path, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    av, np, Image = ensure_imports()
    assembly_dir = evidence / "assembly"
    assembly_dir.mkdir(parents=True, exist_ok=True)
    final_audio = assembly_dir / final_audio_raw_name()
    final_video = assembly_dir / final_video_name()
    final_mov = assembly_dir / final_av_name()
    ffmpeg_log = assembly_dir / "ffmpeg_final_encode.log"
    ffmpeg_video_log = assembly_dir / "ffmpeg_final_video_encode.log"
    ffmpeg_mux_log = assembly_dir / "ffmpeg_final_mux.log"

    t_io_start = time.perf_counter()
    audio_arrays: list[Any] = []
    with final_audio.open("wb") as out_audio:
        for chunk in chunks:
            raw = pcm_from_mp4(Path(chunk["output"]))
            need = AUDIO_SAMPLES_PER_CHUNK * AUDIO_CHANNELS * 2
            if len(raw) < need:
                raise RuntimeError(f"chunk {chunk['chunk_index']} audio too short: {len(raw)} < {need}")
            retained = raw[:need]
            out_audio.write(retained)
            arr = np.frombuffer(retained, dtype=np.int16).reshape(-1, AUDIO_CHANNELS)
            audio_arrays.append(arr.astype(np.float32) / 32768.0)
    audio_write_s = time.perf_counter() - t_io_start
    audio = np.concatenate(audio_arrays, axis=0)
    if audio.shape != (FINAL_AUDIO_SAMPLES_PER_CHANNEL, AUDIO_CHANNELS):
        raise RuntimeError(f"final audio shape mismatch: {audio.shape}")

    video_cmd = build_final_video_ffmpeg_command(final_video)
    mux_cmd = build_final_av_ffmpeg_command(final_video, final_audio, final_mov)
    command_record = {
        "schema_version": "minimax-h3-final-av-ffmpeg-two-stage-v1",
        "note": "Encode rawvideo stdin to a video-only file first, then remux exact raw PCM; no -shortest and no same-process rawvideo+audio mux.",
        "video_encode": video_cmd,
        "final_mux": mux_cmd,
    }
    (assembly_dir / "ffmpeg_final_encode_command.json").write_text(json.dumps(command_record, indent=2) + "\n", encoding="utf-8")
    (assembly_dir / "ffmpeg_final_video_encode_command.json").write_text(json.dumps(video_cmd, indent=2) + "\n", encoding="utf-8")
    (assembly_dir / "ffmpeg_final_mux_command.json").write_text(json.dumps(mux_cmd, indent=2) + "\n", encoding="utf-8")
    stdout_tmp = tempfile.TemporaryFile()
    stderr_tmp = tempfile.TemporaryFile()
    proc = subprocess.Popen(video_cmd, stdin=subprocess.PIPE, stdout=stdout_tmp, stderr=stderr_tmp)
    small_frames: list[Any] = []
    histograms: list[Any] = []
    transition_diffs: list[float] = []
    seam_transition_flags: list[bool] = []
    written = 0
    previous_small = None
    encode_start = time.perf_counter()
    try:
        assert proc.stdin is not None
        for chunk in chunks:
            chunk_index = int(chunk["chunk_index"])
            retain_start, retain_end = chunk["retention"]
            with av.open(str(chunk["output"])) as container:
                for frame_index, frame in enumerate(container.decode(video=0)):
                    if retain_start <= frame_index < retain_end:
                        arr = frame.to_ndarray(format="rgb24")
                        proc.stdin.write(arr.tobytes())
                        written += 1
                        img = Image.fromarray(arr).resize((32, 18), Image.Resampling.BILINEAR)
                        small = np.asarray(img).astype(np.float32) / 255.0
                        gray = small.mean(axis=2)
                        small_frames.append(gray)
                        hist, _ = np.histogramdd(
                            small.reshape(-1, 3), bins=(8, 8, 8), range=((0, 1), (0, 1), (0, 1)), density=False
                        )
                        hist = hist.astype(np.float64).reshape(-1)
                        histograms.append(hist / max(hist.sum(), 1.0))
                        if previous_small is not None:
                            transition_diffs.append(float(np.mean(np.abs(gray - previous_small))))
                            seam_transition_flags.append(frame_index == retain_start and chunk_index > 1)
                        previous_small = gray
        proc.stdin.close()
        wait_for_process_after_stdin_close(proc, timeout=600)
        stdout_tmp.seek(0)
        stderr_tmp.seek(0)
        video_stdout = stdout_tmp.read()
        video_stderr = stderr_tmp.read()
    except Exception:
        proc.kill()
        try:
            proc.wait(timeout=30)
        except Exception:
            pass
        raise
    finally:
        stdout_tmp.close()
        stderr_tmp.close()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg final video encode failed rc={proc.returncode}: {video_stderr[-2000:]!r}")
    if written != FINAL_FRAMES:
        raise RuntimeError(f"wrote {written} frames, expected {FINAL_FRAMES}")
    ffmpeg_video_log.write_bytes((video_stdout or b"") + b"\n--- STDERR ---\n" + (video_stderr or b""))

    mux_proc = subprocess.run(mux_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    ffmpeg_mux_log.write_bytes((mux_proc.stdout or b"") + b"\n--- STDERR ---\n" + (mux_proc.stderr or b""))
    encode_s = time.perf_counter() - encode_start
    ffmpeg_log.write_bytes(
        b"--- VIDEO ENCODE STDOUT ---\n"
        + (video_stdout or b"")
        + b"\n--- VIDEO ENCODE STDERR ---\n"
        + (video_stderr or b"")
        + b"\n--- FINAL MUX STDOUT ---\n"
        + (mux_proc.stdout or b"")
        + b"\n--- FINAL MUX STDERR ---\n"
        + (mux_proc.stderr or b"")
    )
    if mux_proc.returncode != 0:
        raise RuntimeError(f"ffmpeg final mux failed rc={mux_proc.returncode}: {mux_proc.stderr[-2000:]!r}")

    validation = validate_final_av(final_mov)
    objective = compute_objective_metrics(
        np=np,
        histograms=histograms,
        transition_diffs=transition_diffs,
        seam_transition_flags=seam_transition_flags,
        audio=audio,
    )
    write_json(assembly_dir / "objective_metrics.json", objective)
    accounting = {
        "schema_version": f"minimax-h3-final-av-{DURATION_LABEL}-accounting-v1",
        "created_utc": utc_now(),
        "final_av_path": str(final_mov),
        "final_av_sha256_opaque_id": sha256_file(final_mov),
        "final_av_bytes": final_mov.stat().st_size,
        "final_audio_raw_path": str(final_audio),
        "final_audio_raw_bytes": final_audio.stat().st_size,
        "video_frames_effective": FINAL_FRAMES,
        "audio_samples_effective_per_channel": FINAL_AUDIO_SAMPLES_PER_CHANNEL,
        "audio_encoder_priming_samples": 0,
        "audio_end_padding_samples": 0,
        "source_chunk_count": len(chunks),
        "assembly_timing_seconds": {
            "audio_extract_write_io": audio_write_s,
            "encoding_mux_wall": encode_s,
        },
        "final_decode_validation": validation,
        "objective_metrics_path": str(assembly_dir / "objective_metrics.json"),
    }
    write_json(assembly_dir / "final_av_accounting.json", accounting)
    return accounting


def validate_final_av(path: Path) -> dict[str, Any]:
    av, np = ensure_av_numpy()
    video_frames = 0
    audio_samples = 0
    width = height = sample_rate = channels = None
    fps_text = None
    with av.open(str(path)) as container:
        videos = [s for s in container.streams if s.type == "video"]
        audios = [s for s in container.streams if s.type == "audio"]
        if len(videos) != 1 or len(audios) != 1:
            raise RuntimeError(f"final AV expected one video/audio stream, got {len(videos)}/{len(audios)}")
        v = videos[0]
        a = audios[0]
        width = int(v.codec_context.width)
        height = int(v.codec_context.height)
        fps_text = str(v.average_rate)
        sample_rate = int(a.codec_context.sample_rate)
        channels = int(a.codec_context.channels)
        for frame in container.decode(video=0):
            video_frames += 1
    with av.open(str(path)) as container:
        for frame in container.decode(audio=0):
            audio_samples += int(frame.samples)
    complete = (
        width == WIDTH
        and height == HEIGHT
        and video_frames == FINAL_FRAMES
        and sample_rate == AUDIO_RATE
        and channels == AUDIO_CHANNELS
        and audio_samples == FINAL_AUDIO_SAMPLES_PER_CHANNEL
    )
    payload = {
        "status": "complete" if complete else "incomplete",
        "path": str(path),
        "video": {"present": True, "full_decode": True, "width": width, "height": height, "fps": fps_text, "frames": video_frames},
        "audio": {
            "present": True,
            "full_decode": True,
            "sample_rate_hz": sample_rate,
            "channels": channels,
            "decoded_samples_per_channel": audio_samples,
            "effective_samples_per_channel": FINAL_AUDIO_SAMPLES_PER_CHANNEL,
            "encoder_priming_samples": 0,
            "end_padding_samples": 0,
        },
        "final_accounting_complete": complete,
    }
    if not complete:
        raise RuntimeError(f"final AV accounting incomplete: {payload}")
    return payload


def compute_objective_metrics(*, np: Any, histograms: list[Any], transition_diffs: list[float], seam_transition_flags: list[bool], audio: Any) -> dict[str, Any]:
    h0 = histograms[0]
    h_last = histograms[-1]
    hist_intersection = float(np.minimum(h0, h_last).sum())
    chunk_terminal_intersections = []
    for left, right in zip(range(RETAINED_FRAMES_PER_CHUNK - 1, FINAL_FRAMES, RETAINED_FRAMES_PER_CHUNK), range(RETAINED_FRAMES_PER_CHUNK * 2 - 1, FINAL_FRAMES, RETAINED_FRAMES_PER_CHUNK)):
        if left < len(histograms) and right < len(histograms):
            chunk_terminal_intersections.append(float(np.minimum(histograms[left], histograms[right]).sum()))
    diffs = np.asarray(transition_diffs, dtype=np.float64)
    seam_mask = np.asarray(seam_transition_flags, dtype=bool)
    internal = diffs[~seam_mask] if diffs.size else np.asarray([], dtype=np.float64)
    seams = diffs[seam_mask] if diffs.size else np.asarray([], dtype=np.float64)
    internal_p95 = float(np.percentile(internal, 95)) if internal.size else 0.0
    seam_p95 = float(np.percentile(seams, 95)) if seams.size else 0.0
    denom = internal_p95 if internal_p95 > 1e-12 else 1e-12
    seam_ratio = seam_p95 / denom
    near_frozen_fraction = float(np.mean(diffs < 0.002)) if diffs.size else 0.0
    duplicate_window_fraction = float(np.mean(diffs < 0.001)) if diffs.size else 0.0
    motion_mean = float(diffs.mean()) if diffs.size else 0.0

    mono = audio.mean(axis=1)
    rms = float(np.sqrt(np.mean(np.square(mono))) + 1e-12)
    loudness_dbfs = 20.0 * math.log10(rms)
    window = int(AUDIO_RATE * 0.1)
    if len(mono) >= window:
        trimmed = mono[: (len(mono) // window) * window].reshape(-1, window)
        win_rms = np.sqrt(np.mean(np.square(trimmed), axis=1)) + 1e-12
        silence_fraction = float(np.mean(20.0 * np.log10(win_rms) < -50.0))
        near_clip_fraction = float(np.mean(np.abs(mono) > 0.999))
    else:
        win_rms = np.asarray([], dtype=np.float64)
        silence_fraction = 0.0
        near_clip_fraction = 0.0
    seam_ratios = []
    for boundary in range(AUDIO_SAMPLES_PER_CHUNK, FINAL_AUDIO_SAMPLES_PER_CHANNEL, AUDIO_SAMPLES_PER_CHUNK):
        lo = max(0, boundary - 1024)
        hi = min(len(mono), boundary + 1024)
        before = mono[lo:boundary]
        after = mono[boundary:hi]
        if len(before) and len(after):
            rb = float(np.sqrt(np.mean(np.square(before))) + 1e-12)
            ra = float(np.sqrt(np.mean(np.square(after))) + 1e-12)
            seam_ratios.append(max(rb, ra) / max(min(rb, ra), 1e-6))
    audio_continuity_ratio = float(max(seam_ratios)) if seam_ratios else 1.0

    # AV-sync proxy: correlate frame-to-frame visual motion with per-frame audio energy.
    frame_edges = np.linspace(0, len(mono), FINAL_FRAMES + 1).astype(int)
    frame_audio = []
    for i in range(FINAL_FRAMES):
        segment = mono[frame_edges[i] : frame_edges[i + 1]]
        frame_audio.append(float(np.sqrt(np.mean(np.square(segment))) + 1e-12) if len(segment) else 0.0)
    audio_env = np.asarray(frame_audio[1:], dtype=np.float64)
    motion_env = diffs.astype(np.float64)
    av_sync_status = "measured"
    av_sync_offset_ms: float | None = 0.0
    if len(audio_env) != len(motion_env) or float(audio_env.std()) < 1e-6 or float(motion_env.std()) < 1e-6:
        av_sync_status = "not_applicable_no_detectable_events"
        av_sync_offset_ms = None
    else:
        audio_norm = (audio_env - audio_env.mean()) / (audio_env.std() + 1e-12)
        motion_norm = (motion_env - motion_env.mean()) / (motion_env.std() + 1e-12)
        max_lag = int(round(0.5 * FPS))
        best_lag = 0
        best_corr = -1e9
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                a = audio_norm[-lag:]
                m = motion_norm[: len(a)]
            elif lag > 0:
                m = motion_norm[lag:]
                a = audio_norm[: len(m)]
            else:
                a = audio_norm
                m = motion_norm
            if len(a) < 8:
                continue
            corr = float(np.mean(a * m))
            if corr > best_corr:
                best_corr = corr
                best_lag = lag
        av_sync_offset_ms = float(best_lag * 1000.0 / FPS)

    metrics = {
        "schema_version": f"minimax-h3-final-av-{DURATION_LABEL}-objective-proxies-v1",
        "created_utc": utc_now(),
        "proxy_scope": "No-reference objective proxies only; they do not certify semantic identity, prompt faithfulness, human audio quality, or perceived AV sync.",
        "subject_identity_consistency": {"status": "measured", "value": hist_intersection, "unit": "rgb_histogram_intersection_first_last"},
        "background_consistency": {"status": "measured", "value": hist_intersection, "unit": "rgb_histogram_intersection_first_last"},
        "camera_consistency": {"status": "measured", "value": float(statistics.mean(chunk_terminal_intersections)) if chunk_terminal_intersections else hist_intersection, "unit": "chunk_terminal_histogram_intersection_mean"},
        "motion": {"status": "measured", "value": motion_mean, "unit": "mean_abs_downsampled_frame_delta"},
        "repetition": {"status": "measured", "value": duplicate_window_fraction, "unit": "fraction_near_duplicate_transitions"},
        "freezing": {"status": "measured", "value": near_frozen_fraction, "unit": "fraction_near_frozen_transitions"},
        "visual_seams": {"status": "measured", "value": seam_ratio, "unit": "seam_transition_p95_over_internal_p95"},
        "loudness": {"status": "measured", "value": loudness_dbfs, "unit": "dBFS_rms_proxy"},
        "silence": {"status": "measured", "value": silence_fraction, "unit": "fraction_100ms_windows_below_minus_50dbfs"},
        "audio_continuity": {"status": "measured", "value": audio_continuity_ratio, "unit": "max_boundary_rms_ratio_1024_sample_windows"},
        "av_sync_proxy": {"status": av_sync_status, "value": av_sync_offset_ms, "unit": "ms_motion_audio_envelope_xcorr"},
        "near_clip_fraction": {"status": "measured", "value": near_clip_fraction, "unit": "fraction_samples_abs_gt_0.999"},
        "automatic_red_flags_v1": {
            "near_frozen_transition_fraction_max": 0.05,
            "duplicate_window_fraction_max": 0.05,
            "near_clip_fraction_max": 0.0001,
            "near_silence_fraction_max": 0.25,
            "seam_to_internal_transition_p95_ratio_max": 2.0,
            "event_av_sync_absolute_offset_ms_max": 250,
        },
    }
    flags = []
    if near_frozen_fraction > 0.05:
        flags.append("near_frozen_transition_fraction")
    if duplicate_window_fraction > 0.05:
        flags.append("duplicate_window_fraction")
    if near_clip_fraction > 0.0001:
        flags.append("near_clip_fraction")
    if silence_fraction > 0.25:
        flags.append("near_silence_fraction")
    if seam_ratio > 2.0:
        flags.append("visual_seam_ratio")
    if av_sync_offset_ms is not None and abs(av_sync_offset_ms) > 250:
        flags.append("av_sync_proxy_offset")
    metrics["automatic_red_flags"] = flags
    metrics["automatic_proxy_gate"] = "pass" if not flags else "fail"
    return metrics


def summarize_stage_durations(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, float] = {}
    per_chunk: list[dict[str, Any]] = []

    def _as_float(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return None

    def _duration_item(key: str, value: Any) -> tuple[str, float] | None:
        numeric = _as_float(value)
        if numeric is None:
            return None
        return key, numeric

    def _sum_from(stage: dict[str, Any], predicate) -> float:
        total = 0.0
        for key, value in stage.items():
            item = _duration_item(str(key), value)
            if item is None:
                continue
            item_key, numeric = item
            if predicate(item_key):
                total += numeric
        return total

    def _stats_local(values: list[float]) -> dict[str, Any]:
        finite = [float(v) for v in values if math.isfinite(float(v))]
        if not finite:
            return {"n": 0, "status": "missing"}
        out: dict[str, Any] = {
            "n": len(finite),
            "status": "measured",
            "min": min(finite),
            "max": max(finite),
            "mean": statistics.mean(finite),
            "median": statistics.median(finite),
        }
        if len(finite) > 1:
            out["sample_std"] = statistics.stdev(finite)
            out["cv_percent"] = out["sample_std"] / out["mean"] * 100.0 if abs(out["mean"]) > 1e-12 else None
        else:
            out["sample_std"] = 0.0
            out["cv_percent"] = 0.0
        return out

    def _components_for_chunk(chunk: dict[str, Any]) -> dict[str, float]:
        stage = chunk.get("stage_durations", {}) if isinstance(chunk.get("stage_durations"), dict) else {}
        request = chunk.get("request", {}) if isinstance(chunk.get("request"), dict) else {}
        http = request.get("http_metrics", {}) if isinstance(request.get("http_metrics"), dict) else {}
        timing = request.get("request_timing_breakdown", {}) if isinstance(request.get("request_timing_breakdown"), dict) else {}
        http_total = finite_seconds(http.get("time_total_s")) or 0.0
        client_wait = finite_seconds(timing.get("true_client_wait_seconds"))
        server_wall = finite_seconds(timing.get("server_request_wall_seconds"))
        response_transfer = finite_seconds(timing.get("response_transfer_seconds"))
        client_overhead = finite_seconds(timing.get("curl_process_overhead_seconds"))
        return {
            "curl_wall_seconds": float(request.get("curl_wall_seconds") or 0.0),
            "true_client_wait_seconds": client_wait if client_wait is not None else float(request.get("curl_wall_seconds") or 0.0),
            "http_total_seconds": http_total,
            "server_request_wall_seconds": server_wall if server_wall is not None else http_total,
            "response_download_seconds": response_transfer if response_transfer is not None else float(request.get("response_download_seconds") or 0.0),
            "curl_process_overhead_seconds": client_overhead if client_overhead is not None else 0.0,
            "text_conditioning_seconds": _sum_from(stage, lambda key: "encode_prompt" in key or "_encode_video_conditions" in key or "_encode_video_audio_conditions" in key),
            "denoise_seconds": _sum_from(stage, lambda key: key.endswith(".diffuse") or key == "MiniMaxH3Pipeline.diffuse" or key.endswith("Pipeline.diffuse")),
            "combined_decode_seconds": _sum_from(stage, lambda key: (key == "MiniMaxH3Pipeline.decode" or key.endswith("Pipeline.decode")) and not any(token in key for token in ("video_vae", "audio_vae", "crop"))),
            "video_vae_decode_wall_seconds": _sum_from(stage, lambda key: key.endswith("decode.video_vae.wall")),
            "video_vae_decode_device_seconds": _sum_from(stage, lambda key: key.endswith("decode.video_vae.device")),
            "audio_vae_decode_wall_seconds": _sum_from(stage, lambda key: key.endswith("decode.audio_vae.wall")),
            "audio_vae_decode_device_seconds": _sum_from(stage, lambda key: key.endswith("decode.audio_vae.device")),
            "video_crop_contiguous_wall_seconds": _sum_from(stage, lambda key: key.endswith("decode.video_crop_contiguous.wall")),
            "video_crop_contiguous_device_seconds": _sum_from(stage, lambda key: key.endswith("decode.video_crop_contiguous.device")),
            "postprocess_video_cpu_copy_wall_seconds": _sum_from(stage, lambda key: key.endswith("postprocess.video_device_to_host_copy.wall")),
            "postprocess_audio_cpu_copy_wall_seconds": _sum_from(stage, lambda key: key.endswith("postprocess.audio_device_to_host_copy.wall")),
            "postprocess_video_numpy_wall_seconds": _sum_from(stage, lambda key: key.endswith("postprocess.video_numpy_finalize.wall")),
            "postprocess_audio_numpy_wall_seconds": _sum_from(stage, lambda key: key.endswith("postprocess.audio_numpy_finalize.wall")),
            "postprocess_func_wall_seconds": _sum_from(stage, lambda key: key.endswith("postprocess_func.wall")),
            "server_encoding_mux_wall_seconds": _sum_from(stage, lambda key: "response_encoding" in key and key.endswith(".wall")),
            "postprocess_video_cpu_copy_bytes": _sum_from(stage, lambda key: key.endswith("postprocess.video_device_to_host_copy.bytes")),
            "postprocess_audio_cpu_copy_bytes": _sum_from(stage, lambda key: key.endswith("postprocess.audio_device_to_host_copy.bytes")),
        }

    for chunk in chunks:
        stage = chunk.get("stage_durations", {}) if isinstance(chunk.get("stage_durations"), dict) else {}
        for key, value in stage.items():
            item = _duration_item(str(key), value)
            if item is not None:
                item_key, numeric = item
                totals[item_key] = totals.get(item_key, 0.0) + numeric
        components = _components_for_chunk(chunk)
        per_chunk.append(
            {
                "chunk_index": int(chunk.get("chunk_index", len(per_chunk) + 1)),
                "seed": chunk.get("seed"),
                "request_task": chunk.get("request_task"),
                "components": components,
                "stage_duration_keys": sorted(stage),
            }
        )

    def sum_components(name: str) -> float:
        return sum(float(item["components"].get(name, 0.0)) for item in per_chunk)

    per_chunk_variance = {
        name: _stats_local([float(item["components"].get(name, 0.0)) for item in per_chunk])
        for name in (
            "curl_wall_seconds",
            "true_client_wait_seconds",
            "http_total_seconds",
            "server_request_wall_seconds",
            "response_download_seconds",
            "curl_process_overhead_seconds",
            "text_conditioning_seconds",
            "denoise_seconds",
            "combined_decode_seconds",
            "video_vae_decode_wall_seconds",
            "video_vae_decode_device_seconds",
            "audio_vae_decode_wall_seconds",
            "audio_vae_decode_device_seconds",
            "video_crop_contiguous_wall_seconds",
            "postprocess_video_cpu_copy_wall_seconds",
            "postprocess_audio_cpu_copy_wall_seconds",
            "server_encoding_mux_wall_seconds",
        )
    }
    split_profile_keys = [key for key in totals if key.startswith("MiniMaxH3Pipeline.decode.") or key.startswith("MiniMaxH3Pipeline.postprocess.") or key.startswith("DiffusionEngine.postprocess")]
    python_cpu_sync_copy_seconds = (
        sum_components("postprocess_video_cpu_copy_wall_seconds")
        + sum_components("postprocess_audio_cpu_copy_wall_seconds")
        + sum_components("postprocess_video_numpy_wall_seconds")
        + sum_components("postprocess_audio_numpy_wall_seconds")
        + sum_components("video_crop_contiguous_wall_seconds")
    )
    return {
        "raw_totals": totals,
        "per_chunk": per_chunk,
        "per_chunk_variance": per_chunk_variance,
        "true_client_wait_seconds": sum_components("true_client_wait_seconds"),
        "server_request_wall_seconds": sum_components("server_request_wall_seconds"),
        "curl_http_total_seconds": sum_components("http_total_seconds"),
        "curl_process_overhead_seconds": sum_components("curl_process_overhead_seconds"),
        "text_conditioning_seconds": sum_components("text_conditioning_seconds"),
        "denoise_seconds": sum_components("denoise_seconds"),
        "combined_decode_seconds": sum_components("combined_decode_seconds"),
        "video_vae_decode_wall_seconds": sum_components("video_vae_decode_wall_seconds"),
        "video_vae_decode_device_seconds": sum_components("video_vae_decode_device_seconds"),
        "audio_vae_decode_wall_seconds": sum_components("audio_vae_decode_wall_seconds"),
        "audio_vae_decode_device_seconds": sum_components("audio_vae_decode_device_seconds"),
        "video_crop_contiguous_wall_seconds": sum_components("video_crop_contiguous_wall_seconds"),
        "video_crop_contiguous_device_seconds": sum_components("video_crop_contiguous_device_seconds"),
        "postprocess_video_cpu_copy_wall_seconds": sum_components("postprocess_video_cpu_copy_wall_seconds"),
        "postprocess_audio_cpu_copy_wall_seconds": sum_components("postprocess_audio_cpu_copy_wall_seconds"),
        "postprocess_video_numpy_wall_seconds": sum_components("postprocess_video_numpy_wall_seconds"),
        "postprocess_audio_numpy_wall_seconds": sum_components("postprocess_audio_numpy_wall_seconds"),
        "postprocess_func_wall_seconds": sum_components("postprocess_func_wall_seconds"),
        "server_encoding_mux_wall_seconds": sum_components("server_encoding_mux_wall_seconds"),
        "response_download_seconds": sum_components("response_download_seconds"),
        "postprocess_video_cpu_copy_bytes": sum_components("postprocess_video_cpu_copy_bytes"),
        "postprocess_audio_cpu_copy_bytes": sum_components("postprocess_audio_cpu_copy_bytes"),
        "python_cpu_sync_copy_seconds": python_cpu_sync_copy_seconds,
        "python_cpu_sync_copy_bytes": sum_components("postprocess_video_cpu_copy_bytes") + sum_components("postprocess_audio_cpu_copy_bytes"),
        "split_profile_status": "present" if split_profile_keys else "not_present",
        "attention_seconds_status": "from_sol_attn_telemetry_when_present_else_not_exposed_by_dense_backend",
        "component_scope": "X-Stage-Durations from vLLM-Omni sync headers with --enable-diffusion-pipeline-profiler. true_client_wait/server_request_wall/response_transfer are separated from curl metrics; server_request_wall is nested inside client wait and response_transfer is a post-server byte-transfer upper bound, so they are not additively summed. Split VAE/postprocess/response-encoding keys require MINIMAX_H3_A6000_PROFILE_SPLIT=1; attention remains nested in denoise and uses Sol-Attn telemetry when available.",
    }


def peak_resource_summary(path: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(path.open(encoding="utf-8"))) if path.exists() else []
    def maximum(key: str) -> float | None:
        vals = []
        for row in rows:
            try:
                vals.append(float(row[key]))
            except Exception:
                pass
        return max(vals) if vals else None
    host_bytes = maximum("host_memory_used_bytes")
    return {
        "sample_count": len(rows),
        "peak_gpu_memory_mib": maximum("gpu_memory_used_mib"),
        "peak_gpu_util_percent": maximum("gpu_util_percent"),
        "peak_power_w": maximum("power_w"),
        "peak_temperature_c": maximum("temperature_c"),
        "peak_host_memory_used_gib": (host_bytes / (2**30) if host_bytes is not None else None),
    }


def _base_mechanisms(request_quality: str = "lossless") -> list[str]:
    base = ["statically_merged_turbo_lora", "8_step", "extension"]
    profile = CACHE_DIT_REQUEST_PROFILES.get(request_quality)
    if profile:
        return base + list(profile["mechanism_tokens"])
    return base + ["no_cache"]


def _with_vae_spatial_tile_batching_mechanism(
    mechanisms: list[str],
    enabled: bool,
    tile_batch_size: int = 0,
) -> list[str]:
    if tile_batch_size > 0:
        return mechanisms + [
            f"video_vae_bounded_spatial_tile_batching_cap_{tile_batch_size}",
            "practical_approximate_vae_decode",
        ]
    if enabled:
        return mechanisms + [
            "video_vae_spatial_tile_batching_stack_tiling",
            "practical_approximate_vae_decode",
        ]
    return mechanisms + ["video_vae_spatial_tile_batching_off"]


def _attention_mechanisms(
    attention_backend: str,
    mode_label: str,
    request_quality: str = "lossless",
    vae_spatial_tile_batching: bool = False,
    vae_tile_batch_size: int = 0,
) -> list[str]:
    base = _base_mechanisms(request_quality)
    backend = attention_backend.strip().upper()
    label = mode_label.lower()
    if backend == "H3_A6000_SOL_ATTN":
        mechanisms = base + ["sol_attn_sparse_opt_in"]
        if "r8" in label:
            return _with_vae_spatial_tile_batching_mechanism(
                mechanisms + ["sol_attn_r8_sparse_opt_in", "diagnostic_qkv_materialization"],
                vae_spatial_tile_batching,
                vae_tile_batch_size,
            )
        if "adaptive" in label:
            if "step2" in label and ("tau1_5" in label or "tau1.5" in label):
                return _with_vae_spatial_tile_batching_mechanism(
                    mechanisms + ["sol_attn_r9_stride_aware_zero_copy", "guarded_adaptive_routing_tau1_5_diag_step_min_2"],
                    vae_spatial_tile_batching,
                    vae_tile_batch_size,
                )
            if "step3" in label and ("tau1_5" in label or "tau1.5" in label):
                return _with_vae_spatial_tile_batching_mechanism(
                    mechanisms + ["sol_attn_r9_stride_aware_zero_copy", "guarded_adaptive_routing_tau1_5_diag_step_min_3"],
                    vae_spatial_tile_batching,
                    vae_tile_batch_size,
                )
            if "late_steps" in label and ("tau1_5" in label or "tau1.5" in label):
                return _with_vae_spatial_tile_batching_mechanism(
                    mechanisms + ["sol_attn_r9_stride_aware_zero_copy", "guarded_adaptive_routing_tau1_5_diag_late_steps"],
                    vae_spatial_tile_batching,
                    vae_tile_batch_size,
                )
            if "tau1_5" in label or "tau1.5" in label:
                return _with_vae_spatial_tile_batching_mechanism(
                    mechanisms + ["sol_attn_r9_stride_aware_zero_copy", "adaptive_routing_tau1_5_diag"],
                    vae_spatial_tile_batching,
                    vae_tile_batch_size,
                )
            if "tau2" in label or "tau2_0" in label or "tau2.0" in label:
                return _with_vae_spatial_tile_batching_mechanism(
                    mechanisms + ["sol_attn_r9_stride_aware_zero_copy", "adaptive_routing_tau2_diag"],
                    vae_spatial_tile_batching,
                    vae_tile_batch_size,
                )
            return _with_vae_spatial_tile_batching_mechanism(
                mechanisms + ["sol_attn_r9_stride_aware_zero_copy", "adaptive_routing_tau_diag"],
                vae_spatial_tile_batching,
                vae_tile_batch_size,
            )
        return _with_vae_spatial_tile_batching_mechanism(
            mechanisms + ["sol_attn_r9_stride_aware_zero_copy", "retained_tau1_diag"],
            vae_spatial_tile_batching,
            vae_tile_batch_size,
        )
    if "sol" in label:
        return _with_vae_spatial_tile_batching_mechanism(base + ["sol_attn_disabled_fail_closed"], vae_spatial_tile_batching, vae_tile_batch_size)
    return _with_vae_spatial_tile_batching_mechanism(base + ["dense_cudnn_attention"], vae_spatial_tile_batching, vae_tile_batch_size)


def _find_sol_attn_telemetry(evidence: Path) -> Path | None:
    candidates = sorted(evidence.glob("**/*sol_attn*.json"))
    for path in candidates:
        if path.name.endswith(".sol_attn.json") or path.name == "sol_attn_telemetry.sol_attn.json":
            return path
    return None


def _summarize_sol_attn_telemetry(evidence: Path) -> dict[str, Any]:
    path = _find_sol_attn_telemetry(evidence)
    if path is None:
        return {"status": "not_present", "reason": "Sol-Attn telemetry file was not emitted for this mode."}
    try:
        telemetry = read_json(path)
    except Exception as exc:  # noqa: BLE001
        return {"status": "unreadable", "path": str(path), "reason": f"{type(exc).__name__}: {exc}"}
    keys = (
        "dense_calls",
        "sparse_candidate_calls",
        "sparse_calls",
        "fallback_calls",
        "materialize_copy_count",
        "materialize_copy_bytes",
        "input_copy_events",
        "input_copy_bytes",
        "sparse_attention_gpu_latency_ms",
        "sparse_attention_timed_calls",
        "denoise_gpu_latency_ms",
        "denoise_timed_calls",
        "stride_aware_value_calls",
    )
    summary = {key: telemetry.get(key) for key in keys if key in telemetry}
    summary["decline_reasons"] = telemetry.get("decline_reasons", {})
    summary["fallback_reasons"] = telemetry.get("fallback_reasons", {})
    density_samples = telemetry.get("density_samples", []) or []
    summary["density_sample_count"] = len(density_samples)
    if density_samples:
        summary["density_first"] = density_samples[:3]
        summary["adaptive_routing_values"] = sorted({bool(item.get("adaptive_routing", False)) for item in density_samples if isinstance(item, dict)})
        summary["tau_values"] = sorted({float(item["tau"]) for item in density_samples if isinstance(item, dict) and isinstance(item.get("tau"), (int, float))})
        summary["thresh_type_values"] = sorted({str(item.get("thresh_type")) for item in density_samples if isinstance(item, dict) and item.get("thresh_type") is not None})
        summary["stride_aware_value_density_count"] = sum(1 for item in density_samples if isinstance(item, dict) and item.get("stride_aware_value") is True)
        guarded = [item for item in density_samples if isinstance(item, dict) and item.get("adaptive_routing_requested") is True]
        if guarded:
            reason_counts: dict[str, int] = {}
            counts_by_step: dict[str, dict[str, Any]] = {}
            for item in guarded:
                reason = str(item.get("adaptive_guard_reason") or "missing")
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                step_value = item.get("step_index")
                step_key = str(int(step_value)) if isinstance(step_value, int) else "missing"
                step_bucket = counts_by_step.setdefault(step_key, {"active": 0, "inactive": 0, "reasons": {}})
                active = item.get("adaptive_guard_active") is True
                step_bucket["active" if active else "inactive"] += 1
                step_bucket["reasons"][reason] = step_bucket["reasons"].get(reason, 0) + 1
            summary["adaptive_guard_requested_count"] = len(guarded)
            summary["adaptive_guard_active_count"] = sum(1 for item in guarded if item.get("adaptive_guard_active") is True)
            summary["adaptive_guard_inactive_count"] = sum(1 for item in guarded if item.get("adaptive_guard_active") is False)
            summary["adaptive_guard_reason_counts"] = reason_counts
            summary["adaptive_guard_counts_by_step"] = counts_by_step
            summary["adaptive_profiles"] = sorted({str(item.get("adaptive_profile")) for item in guarded if item.get("adaptive_profile")})
            summary["adaptive_step_min_values"] = sorted({int(item["adaptive_step_min"]) for item in guarded if isinstance(item.get("adaptive_step_min"), int)})
            summary["adaptive_step_max_values"] = sorted({int(item["adaptive_step_max"]) for item in guarded if isinstance(item.get("adaptive_step_max"), int)})
            summary["adaptive_layer_min_values"] = sorted({int(item["adaptive_layer_min"]) for item in guarded if isinstance(item.get("adaptive_layer_min"), int)})
            summary["adaptive_layer_max_values"] = sorted({int(item["adaptive_layer_max"]) for item in guarded if isinstance(item.get("adaptive_layer_max"), int)})
            summary["adaptive_layer_range_scope_values"] = sorted({str(item.get("adaptive_layer_range_scope")) for item in guarded if item.get("adaptive_layer_range_scope") is not None})
            summary["step_index_values"] = sorted({int(item["step_index"]) for item in density_samples if isinstance(item, dict) and isinstance(item.get("step_index"), int)})
            layer_values = sorted({int(item["layer_index"]) for item in density_samples if isinstance(item, dict) and isinstance(item.get("layer_index"), int)})
            summary["layer_index_values"] = layer_values
            summary["layer_index_values_sample"] = layer_values[:16]
    summary["diagnostic_raw_tensor_exported"] = bool(telemetry.get("diagnostic_raw_tensor_exported", False))
    return {"status": "present", "path": str(path), "summary": summary}


def _summarize_cache_dit_telemetry(
    evidence: Path,
    *,
    request_quality: str,
    server_cache_backend: str,
    enable_cache_dit_summary: bool,
) -> dict[str, Any]:
    """Extract bounded Cache-DiT evidence from the pinned runtime logs.

    The pinned vLLM-Omni H3 path owns request-scoped Cache-DiT.  Its public
    interface is the request ``quality`` field; with ``--enable-cache-dit-summary``
    it also emits Cache-DiT summary/log lines.  This parser keeps the raw log
    excerpts in the private evidence tree and never treats log text as a human
    quality certificate.
    """

    expected_config = cache_dit_expected_config(request_quality)
    expected_config.update(
        {
            "server_cache_backend": server_cache_backend,
            "enabled_summary_requested": bool(enable_cache_dit_summary),
        }
    )

    server_log = evidence / "server.log"
    if not server_log.exists():
        return {"status": "not_present", "expected_config": expected_config, "reason": "server.log missing"}
    text = server_log.read_text(encoding="utf-8", errors="replace")
    interesting_lines = [
        line
        for line in text.splitlines()
        if any(
            token.lower() in line.lower()
            for token in (
                "cache-dit",
                "cachedit",
                "cache_dit",
                "dbcache",
                "cached steps",
                "pruned",
                "reuse",
                "refreshing cache context",
                "disabling cache-dit",
                "minimax_h3.high",
                "minimax_h3.high_warmup2",
            )
        )
    ]
    excerpt = "\n".join(interesting_lines[-240:])
    excerpt_path = evidence / "cache_dit_telemetry_excerpt.log"
    excerpt_path.write_text(excerpt + ("\n" if excerpt else ""), encoding="utf-8")

    def _sum_int_patterns(patterns: tuple[str, ...]) -> int | None:
        total = 0
        matched = False
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                matched = True
                total += int(match.group(1))
        return total if matched else None

    # Cache-DiT log format can change across releases; keep several conservative
    # aliases and retain the raw excerpt for reviewer inspection.  Prefer the
    # local machine-readable line emitted from returned CacheStats, but also
    # accept upstream human summary phrasings when residual-diff details are
    # populated.
    machine_summary_totals: list[dict[str, Any]] = []
    for match in re.finditer(r"Cache-DiT request summary json:\s*(\{.*?\})", text, flags=re.IGNORECASE):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            machine_summary_totals.append(payload)

    def _sum_machine_summary(*keys: str) -> int | None:
        if not machine_summary_totals:
            return None
        total = 0
        matched = False
        for payload in machine_summary_totals:
            for key in keys:
                value = payload.get(key)
                if isinstance(value, bool):
                    total += int(value)
                    matched = True
                elif isinstance(value, (int, float)):
                    total += int(value)
                    matched = True
        return total if matched else None

    reuse_count = _sum_machine_summary("cached_steps")
    if reuse_count in (None, 0):
        pruned_count = _sum_machine_summary("pruned_steps", "pruned_blocks")
        if pruned_count not in (None, 0):
            reuse_count = pruned_count
    if reuse_count is None:
        reuse_count = _sum_int_patterns(
            (
                r"Cache-DiT request summary:.*(?:cached_steps|pruned_steps|pruned_blocks)\s*=\s*(\d+)",
                r"Cache Steps\s*:\s*(\d+)",
            )
        )
    if reuse_count is None:
        reuse_count = _sum_int_patterns(
            (
                r"(?:cached|cache|pruned|skip(?:ped)?|reuse(?:d)?)\s*(?:steps?|blocks?|count)?\s*[:=]\s*(\d+)",
                r"(\d+)\s*(?:cached|pruned|skipped|reused)\s*(?:steps?|blocks?)",
            )
        )
    compute_count = _sum_machine_summary("executed_steps")
    if compute_count in (None, 0):
        transformer_compute_count = _sum_machine_summary("transformer_executed_steps")
        if transformer_compute_count not in (None, 0):
            compute_count = transformer_compute_count
    if compute_count is None:
        compute_count = _sum_int_patterns(
            (
                r"Cache-DiT request summary:.*(?:executed_steps|transformer_executed_steps|computed_steps)\s*=\s*(\d+)",
                r"(?<!Transformer\s)Executed Steps\s*:\s*(\d+)",
            )
        )
    if compute_count is None:
        compute_count = _sum_int_patterns(
            (
                r"Transformer\s+Executed Steps\s*:\s*(\d+)",
                r"(?:computed|compute|executed)\s*(?:steps?|blocks?|count)?\s*[:=]\s*(\d+)",
                r"(\d+)\s*(?:computed|executed)\s*(?:steps?|blocks?)",
            )
        )
    refresh_count = len(re.findall(r"Refreshing cache context", text, flags=re.IGNORECASE))
    enable_count = len(re.findall(r"Cache-dit enabled successfully|Enabling cache-dit", text, flags=re.IGNORECASE))
    disable_count = len(re.findall(r"Disabling cache-dit", text, flags=re.IGNORECASE))
    high_request_seen = (
        "quality=high" in text
        or "quality=high_warmup2" in text
        or "minimax_h3.high" in text
        or "minimax_h3.high_warmup2" in text
        or request_quality in CACHE_DIT_REQUEST_PROFILES
    )
    lossless_request_seen = "quality=lossless" in text or request_quality == "lossless"
    return {
        "status": "present" if interesting_lines else "not_present",
        "path": str(excerpt_path),
        "expected_config": expected_config,
        "summary": {
            "request_quality": request_quality,
            "server_cache_backend": server_cache_backend,
            "cache_summary_requested": bool(enable_cache_dit_summary),
            "cache_log_line_count": len(interesting_lines),
            "enable_log_count": enable_count,
            "disable_log_count": disable_count,
            "refresh_log_count": refresh_count,
            "parsed_reuse_or_skip_count": reuse_count,
            "parsed_compute_count": compute_count,
            "high_request_seen": high_request_seen,
            "lossless_request_seen": lossless_request_seen,
            "raw_excerpt_path": str(excerpt_path),
        },
        "claim_boundary": "Cache-DiT log telemetry only; cache-on output remains practical approximate and needs matched objective/structural gates.",
    }


def _shutdown_server(evidence: Path, server: subprocess.Popen[Any], monitor: ResourceMonitor, *, already_stopped: bool) -> bool:
    if already_stopped:
        return True
    monitor.stop()
    if server.poll() is None:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=30)
    write_json(evidence / "server_exit.json", {"returncode": server.returncode, "created_utc": utc_now()})
    return True


def run_regional_compile_probe_mode(args: argparse.Namespace) -> int:
    evidence = Path(args.evidence).resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    base_prompt = Path(args.prompt).read_text(encoding="utf-8")
    manifest = read_json(Path(args.manifest))
    configure_lane_from_manifest(manifest)
    ensure_imports()
    for exe in ("ffmpeg", "ffprobe", "curl", "nvidia-smi", "vllm-omni"):
        if subprocess.run(["bash", "-lc", f"command -v {exe}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
            raise RuntimeError(f"missing required executable in container: {exe}")
    container_gpu_preflight(evidence)
    server = start_server(
        evidence,
        args.model_dir,
        init_timeout_s=args.init_timeout_s,
        stage_init_timeout_s=args.stage_init_timeout_s,
        dlo_resident_layers=args.dlo_resident_layers,
        attention_backend=args.attention_backend,
        server_cache_backend="none",
        enable_cache_dit_summary=False,
        regional_compile=bool(args.regional_compile),
        diffusion_compile_dynamic=bool(args.diffusion_compile_dynamic),
    )
    monitor = ResourceMonitor(evidence / "resource_monitor.csv", server)
    monitor.start()
    server_stopped = False
    try:
        readiness_s = wait_ready(server, evidence / "server.log", args.readiness_timeout_s)
        write_json(evidence / "readiness.json", {"status": "ready", "readiness_seconds": readiness_s, "created_utc": utc_now()})
        prompt_text = chunk_prompt(base_prompt, 1)
        records: dict[str, Any] = {}
        for phase, seed in (("compile_warmup_excluded", args.seed), ("measured_same_input", args.seed)):
            phase_dir = evidence / phase
            phase_dir.mkdir(parents=True, exist_ok=True)
            prompt_file = phase_dir / "prompt.txt"
            prompt_file.write_text(prompt_text, encoding="utf-8")
            output = phase_dir / f"{phase}.mp4"
            record = request_video(
                chunk_dir=phase_dir,
                prompt_file=prompt_file,
                output_file=output,
                seed=int(seed),
                task="t2va",
                conditioner=None,
                duration_s=SOURCE_DURATION_SECONDS,
                steps=args.steps,
                flow_shift=args.flow_shift,
                audio_flow_shift=args.audio_flow_shift,
                timeout_s=args.request_timeout_s,
                request_quality="lossless",
            )
            validation = decode_chunk_validation(
                phase_dir,
                output,
                1,
                0,
                RETAINED_FRAMES_PER_CHUNK,
            )
            records[phase] = {"request": record, "validation": validation, "output": str(output)}
        write_json(
            evidence / "probe_mode_summary.json",
            {
                "schema_version": "minimax_h3_a6000_regional_compile_probe_mode_v1",
                "status": "pass",
                "created_utc": utc_now(),
                "mode_label": args.mode_label,
                "regional_compile": bool(args.regional_compile),
                "diffusion_compile_dynamic": bool(args.diffusion_compile_dynamic),
                "attention_backend": args.attention_backend,
                "sol_attn_profile": os.environ.get("MINIMAX_H3_A6000_SOL_ATTN_ADAPTIVE_PROFILE"),
                "dlo_resident_layers": int(args.dlo_resident_layers),
                "readiness_seconds": readiness_s,
                "records": records,
                "claim_boundary": "One warmup plus one same-input representative 5.1667s T2VA request only; not final-AV, not native long context, not a speedup claim.",
            },
        )
    except Exception as exc:  # noqa: BLE001 - probe artifact should fail closed.
        write_json(
            evidence / "blocker.json",
            {
                "schema_version": "minimax_h3_a6000_regional_compile_probe_blocker_v1",
                "status": "failed",
                "created_utc": utc_now(),
                "mode_label": args.mode_label,
                "regional_compile": bool(args.regional_compile),
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "claim_boundary": "Probe failed before any promotion decision; no final-AV or speedup claim.",
            },
        )
        raise
    finally:
        try:
            server_stopped = stop_server_and_monitor(evidence, server, monitor, server_stopped)
        finally:
            del server_stopped
    return 0


def run_container(args: argparse.Namespace) -> int:
    evidence = Path(args.evidence).resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    base_prompt = Path(args.prompt).read_text(encoding="utf-8")
    manifest = read_json(Path(args.manifest))
    configure_lane_from_manifest(manifest)
    seeds = manifest["workload"]["seed_plan"]["seeds"]
    if len(seeds) != CHUNK_COUNT or any(not isinstance(seed, int) for seed in seeds):
        raise RuntimeError(f"unexpected seed plan for {DURATION_LABEL}/{CHUNK_COUNT} chunks: {seeds}")
    if int(getattr(args, "vae_tile_batch_size", 0) or 0) < 0:
        raise RuntimeError("--vae-tile-batch-size must be non-negative")
    ensure_imports()
    for exe in ("ffmpeg", "ffprobe", "curl", "nvidia-smi", "vllm-omni"):
        if subprocess.run(["bash", "-lc", f"command -v {exe}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
            raise RuntimeError(f"missing required executable in container: {exe}")
    container_gpu_preflight(evidence)
    server = start_server(
        evidence,
        args.model_dir,
        init_timeout_s=args.init_timeout_s,
        stage_init_timeout_s=args.stage_init_timeout_s,
        dlo_resident_layers=args.dlo_resident_layers,
        attention_backend=args.attention_backend,
        server_cache_backend=args.server_cache_backend,
        enable_cache_dit_summary=bool(args.enable_cache_dit_summary),
        regional_compile=bool(args.regional_compile),
        diffusion_compile_dynamic=bool(args.diffusion_compile_dynamic),
    )
    monitor = ResourceMonitor(evidence / "resource_monitor.csv", server)
    monitor.start()
    chunks: list[dict[str, Any]] = []
    warm_start_perf: float | None = None
    final_done_wall_ns: int | None = None
    server_stopped = False
    try:
        readiness_s = wait_ready(server, evidence / "server.log", args.readiness_timeout_s)
        write_json(evidence / "readiness.json", {"status": "ready", "readiness_seconds": readiness_s, "created_utc": utc_now()})
        warmup_dir = evidence / "warmup_excluded"
        warmup_dir.mkdir(exist_ok=True)
        warmup_prompt = warmup_dir / "prompt.txt"
        warmup_prompt.write_text(chunk_prompt(base_prompt, 1), encoding="utf-8")
        warmup_record = request_video(
            chunk_dir=warmup_dir,
            prompt_file=warmup_prompt,
            output_file=warmup_dir / "warmup_excluded.mp4",
            seed=4199,
            task="t2va",
            conditioner=None,
            duration_s=SOURCE_DURATION_SECONDS,
            steps=args.steps,
            flow_shift=args.flow_shift,
            audio_flow_shift=args.audio_flow_shift,
            timeout_s=args.request_timeout_s,
            request_quality=args.request_quality,
        )
        warmup_validation = decode_chunk_validation(warmup_dir, warmup_dir / "warmup_excluded.mp4", 0, 0, RETAINED_FRAMES_PER_CHUNK)
        write_json(warmup_dir / "warmup_record.json", {"request": warmup_record, "validation": warmup_validation, "excluded_from_production": True})
        warm_start_perf = time.perf_counter()

        conditioner: Path | None = None
        conditioner_provenance: dict[str, Any] | None = None
        for idx, seed in enumerate(seeds, start=1):
            chunk_dir = evidence / "chunks" / f"chunk_{idx:02d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            prompt_file = chunk_dir / "prompt.txt"
            prompt_file.write_text(chunk_prompt(base_prompt, idx), encoding="utf-8")
            task = "t2va" if idx == 1 else "fl2va"
            if conditioner_provenance is not None:
                write_json(chunk_dir / "conditioner_input_provenance.json", conditioner_provenance)
            record = request_video(
                chunk_dir=chunk_dir,
                prompt_file=prompt_file,
                output_file=chunk_dir / f"chunk_{idx:02d}.mp4",
                seed=int(seed),
                task=task,
                conditioner=conditioner,
                duration_s=SOURCE_DURATION_SECONDS,
                steps=args.steps,
                flow_shift=args.flow_shift,
                audio_flow_shift=args.audio_flow_shift,
                timeout_s=args.request_timeout_s,
                request_quality=args.request_quality,
            )
            retain_start, retain_end = (0, 120) if idx == 1 else (1, 121)
            validation = decode_chunk_validation(chunk_dir, chunk_dir / f"chunk_{idx:02d}.mp4", idx, retain_start, retain_end)
            chunk_summary = {
                "chunk_index": idx,
                "seed": int(seed),
                "request_task": task,
                "prompt_file": str(prompt_file),
                "output": str(chunk_dir / f"chunk_{idx:02d}.mp4"),
                "retention": [retain_start, retain_end],
                "conditioner_input_provenance": conditioner_provenance,
                "request": record,
                "media_validation": validation,
                "stage_durations": record.get("stage_durations", {}),
            }
            write_json(chunk_dir / "chunk_summary.json", chunk_summary)
            chunks.append(chunk_summary)
            conditioner = chunk_dir / "retained_terminal_frame.png"
            conditioner_provenance = {
                "kind": "generated_prior_terminal_rgb_frame",
                "for_next_chunk_index": idx + 1,
                "source_chunk_index": idx,
                "source_retained_terminal_frame_index": validation["retained_terminal_frame_source_index"],
                "path": str(conditioner),
                "rgb_sha256": validation["retained_terminal_frame_rgb_sha256"],
                "png_sha256": validation["retained_terminal_frame_png_sha256"],
                "provenance_rule": "previous chunk retained terminal RGB frame saved as lossless PNG before the next FL2VA request",
            }

        accounting = assemble_final_av(evidence, chunks)
        final_done_wall_ns = time.time_ns()
        warm_e2e = time.perf_counter() - warm_start_perf if warm_start_perf is not None else None
        cold_start_ns_raw = os.environ.get("HOST_COLD_START_EPOCH_NS")
        cold_e2e = None
        if cold_start_ns_raw:
            try:
                cold_e2e = (final_done_wall_ns - int(cold_start_ns_raw)) / 1e9
            except Exception:
                cold_e2e = None
        server_stopped = _shutdown_server(evidence, server, monitor, already_stopped=server_stopped)
        stage_summary = summarize_stage_durations(chunks)
        resources = peak_resource_summary(evidence / "resource_monitor.csv")
        sol_attn = _summarize_sol_attn_telemetry(evidence)
        cache_dit = _summarize_cache_dit_telemetry(
            evidence,
            request_quality=args.request_quality,
            server_cache_backend=args.server_cache_backend,
            enable_cache_dit_summary=bool(args.enable_cache_dit_summary),
        )
        vae_spatial_tile_batching = bool(args.vae_spatial_tile_batching)
        vae_tile_batch_size = int(getattr(args, "vae_tile_batch_size", 0) or 0)
        mechanisms = _attention_mechanisms(
            args.attention_backend,
            args.mode_label,
            args.request_quality,
            vae_spatial_tile_batching,
            vae_tile_batch_size,
        )
        summary = {
            "schema_version": f"minimax-h3-final-av-{DURATION_LABEL}-extension-n1-container-summary-v1",
            "created_utc": utc_now(),
            "status": "pass",
            "classification": f"descriptive_n1_final_av_{DURATION_LABEL}_extension_complete",
            "mode_label": args.mode_label,
            "generation_mode": "extension",
            "native_context_supported": False,
            "track": "practical_disclosed_approx",
            "attention_backend": args.attention_backend,
            "request_quality": args.request_quality,
            "server_cache_backend": args.server_cache_backend,
            "enable_cache_dit_summary": bool(args.enable_cache_dit_summary),
            "video_vae": {
                "spatial_tile_batching": vae_spatial_tile_batching,
                "tile_batch_size": vae_tile_batch_size,
                "env": {
                    "MINIMAX_H3_A6000_VIDEO_VAE_SPATIAL_TILE_BATCHING": "1" if vae_spatial_tile_batching else "0",
                    "MINIMAX_H3_A6000_VIDEO_VAE_TILE_BATCH_SIZE": str(vae_tile_batch_size),
                },
                "mechanism": (
                    f"bounded_spatial_tile_batching_cap_{vae_tile_batch_size}"
                    if vae_tile_batch_size > 0
                    else ("remote_stack_tiling_spatial_tile_batching" if vae_spatial_tile_batching else "current_serial_spatial_tile_decode")
                ),
                "claim_boundary": "Practical approximate VAE decode path only when enabled; not BF16 fidelity or exact/lossless.",
            },
            "mechanisms": mechanisms,
            "chunks": chunks,
            "warmup_excluded": str(warmup_dir),
            "final_av_accounting": accounting,
            "stage_duration_summary": stage_summary,
            "timing": {
                "boundary_id": TIMING_BOUNDARY_ID,
                "cold_e2e_seconds": cold_e2e,
                "warm_e2e_seconds": warm_e2e,
                "seconds_per_generated_second": (warm_e2e / FINAL_DURATION_SECONDS if warm_e2e else None),
                "cold_boundary_note": "HOST_COLD_START_EPOCH_NS before docker run to final AV validation durable; includes service startup and one excluded warmup.",
                "warm_boundary_note": "After one excluded warmup in the same service lifecycle through final AV validation durable.",
            },
            "resources": resources,
            "sol_attn": sol_attn,
            "cache_dit": cache_dit,
            "hidden_filesystem_cache": {
                "status": "separated_from_model_weights_and_algorithmic_cache",
                "container_cache_root": "/workspace/cache",
                "host_cache_root": str(evidence / "cache"),
                "algorithmic_cache_scope": "Cache-DiT request-scoped tensor residual state is in-process only and not stored in this filesystem cache.",
            },
            "monitor_failures": monitor.failures,
            "claim_boundary": f"First descriptive N=1 {DURATION_READABLE} extension baseline only; request quality {args.request_quality!r}; video VAE spatial tile batching={vae_spatial_tile_batching}; video VAE tile batch size={vae_tile_batch_size}; no native-long-context, speedup, BF16-fidelity, public-comparison, SOTA, or human-quality claim.",
        }
        write_json(evidence / "container_summary.json", summary)
        return 0
    except Exception as exc:  # noqa: BLE001
        blocker = {
            "schema_version": f"minimax-h3-final-av-{DURATION_LABEL}-extension-runtime-blocker-v1",
            "created_utc": utc_now(),
            "status": "failed",
            "classification": "fail_closed_runtime_or_contract_blocker",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback_tail": traceback.format_exc()[-8000:],
            "claim_boundary": f"No qualifying {DURATION_READABLE} final-AV baseline result was produced from this failed run.",
        }
        write_json(evidence / "blocker.json", blocker)
        return 1
    finally:
        _shutdown_server(evidence, server, monitor, already_stopped=server_stopped)


def metric(status: str, value: Any, unit: str, reason: str | None = None) -> dict[str, Any]:
    out = {"status": status, "value": value, "unit": unit}
    if reason is not None:
        out["reason"] = reason
    return out


def timing_metric(status: str, seconds: float | None, n: int, reason: str | None = None, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"status": status, "seconds": seconds, "n": n}
    if reason is not None:
        out["reason"] = reason
    out.update(extra)
    return out


def component_metric(status: str, seconds: float | None, n: int, timing_basis: str, additive_to_e2e: bool, reason: str | None = None, parent: str | None = None) -> dict[str, Any]:
    out = timing_metric(status, seconds, n, reason)
    out["timing_basis"] = timing_basis
    out["additive_to_e2e"] = additive_to_e2e
    if parent is not None:
        out["parent"] = parent
    return out


def workload_fingerprint(record: dict[str, Any]) -> str:
    workload = record["workload"]
    production = record["production"]
    prompt = workload["prompt"]
    seed_plan = workload["seed_plan"]
    payload = {
        "lane_id": record.get("lane_id"),
        "task": workload.get("task"),
        "partition": workload.get("partition"),
        "width": workload.get("width"),
        "height": workload.get("height"),
        "fps": workload.get("fps"),
        "final_frame_count": workload.get("final_frame_count"),
        "nominal_duration_seconds": workload.get("nominal_duration_seconds"),
        "audio_sample_rate_hz": workload.get("audio_sample_rate_hz"),
        "audio_channels": workload.get("audio_channels"),
        "num_inference_steps": workload.get("num_inference_steps"),
        "prompt_sha256": prompt.get("sha256"),
        "seeds": seed_plan.get("seeds"),
        "generation_mode": production.get("generation_mode"),
        "chunk_count": production.get("chunk_count"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _host_evidence_path(evidence: Path, raw_path: str) -> Path:
    """Map a path recorded inside the container (/evidence/...) to host evidence."""
    path = Path(raw_path)
    if path.is_absolute() and path.parts[:2] == ("/", "evidence"):
        return evidence.joinpath(*path.parts[2:])
    return path


def make_benchmark_record(evidence: Path, summary: dict[str, Any], manifest: dict[str, Any], gpu_uuid: str) -> dict[str, Any]:
    timing = summary["timing"]
    stage = summary["stage_duration_summary"]
    accounting = summary["final_av_accounting"]
    accounting_path = Path(summary.get("final_av_accounting_path", evidence / "assembly" / "final_av_accounting.json"))
    validation = accounting["final_decode_validation"]
    objective = read_json(_host_evidence_path(evidence, accounting["objective_metrics_path"]))
    resources = summary["resources"]
    request_quality = str(summary.get("request_quality", "lossless"))
    video_vae = summary.get("video_vae") if isinstance(summary.get("video_vae"), dict) else {}
    vae_spatial_tile_batching = bool(video_vae.get("spatial_tile_batching"))
    vae_tile_batch_size = int(video_vae.get("tile_batch_size") or 0)
    mechanisms = list(
        summary.get("mechanisms")
        or _attention_mechanisms(
            str(summary.get("attention_backend", "CUDNN_ATTN")),
            str(summary.get("mode_label", "dense_cudnn")),
            request_quality,
            vae_spatial_tile_batching,
            vae_tile_batch_size,
        )
    )
    sol_attn = summary.get("sol_attn") if isinstance(summary.get("sol_attn"), dict) else {}
    cache_dit = summary.get("cache_dit") if isinstance(summary.get("cache_dit"), dict) else {}
    sol_summary = sol_attn.get("summary") if isinstance(sol_attn.get("summary"), dict) else {}
    unavailable = "Split profiler was not enabled or the pinned runtime did not expose this component; container_summary.stage_duration_summary preserves available raw timings."
    attention_seconds: float | None = None
    if sol_attn.get("status") == "present" and isinstance(sol_summary.get("sparse_attention_gpu_latency_ms"), (int, float)):
        attention_seconds = float(sol_summary["sparse_attention_gpu_latency_ms"]) / 1000.0

    def positive_stage_seconds(name: str) -> float | None:
        value = stage.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value = float(value)
        return value if value > 0.0 and math.isfinite(value) else None

    video_vae_wall = positive_stage_seconds("video_vae_decode_wall_seconds")
    audio_vae_wall = positive_stage_seconds("audio_vae_decode_wall_seconds")
    server_encoding_mux_wall = float(stage.get("server_encoding_mux_wall_seconds") or 0.0)
    response_download_wall = float(stage.get("response_download_seconds") or 0.0)
    final_encoding_mux_wall = float(accounting["assembly_timing_seconds"]["encoding_mux_wall"])
    final_audio_io_wall = float(accounting["assembly_timing_seconds"]["audio_extract_write_io"])
    objective_record = {}
    for name in (
        "subject_identity_consistency",
        "background_consistency",
        "camera_consistency",
        "motion",
        "repetition",
        "freezing",
        "visual_seams",
        "loudness",
        "silence",
        "audio_continuity",
        "av_sync_proxy",
    ):
        item = dict(objective[name])
        if item.get("status") not in {"measured", "pass", "fail"} and "reason" not in item:
            item["reason"] = "Proxy was not applicable under the automatic detector for this first descriptive baseline."
        objective_record[name] = item

    default_claim_boundary = f"First descriptive N=1 {DURATION_READABLE} 1344x768/24FPS final-AV extension baseline on one A6000. Classified strictly as extension, not native long context. No speedup, human-quality, BF16-fidelity, public-comparison, or SOTA claim."
    record: dict[str, Any] = {
        "schema_version": "minimax-h3-a6000-benchmark-record-v1",
        "contract_version": "1.0.0",
        "record_id": evidence.name,
        "record_status": "candidate",
        "lane_id": manifest["lane_id"],
        "workload_fingerprint": "",
        "source_evidence": [
            {"path": str(evidence / "container_summary.json"), "scope": f"Real single-A6000 {CHUNK_COUNT}-chunk {DURATION_READABLE} extension run summary."},
            {"path": str(accounting_path), "scope": "Final AV frame/audio accounting and objective proxy paths."},
            {"path": str(evidence / "contract_validation.json"), "scope": "Frozen benchmark contract validation before run."},
            {"path": str(evidence / "cache_dit_telemetry_excerpt.log"), "scope": "Cache-DiT telemetry excerpt when request-scoped cache was enabled or requested; absent/empty is explicit in container_summary.cache_dit."},
        ],
        "workload": {
            "task": manifest["workload"]["task"],
            "partition": manifest["workload"]["partition"],
            "width": WIDTH,
            "height": HEIGHT,
            "fps": FPS,
            "final_frame_count": FINAL_FRAMES,
            "nominal_duration_seconds": FINAL_DURATION_SECONDS,
            "audio_sample_rate_hz": AUDIO_RATE,
            "audio_channels": AUDIO_CHANNELS,
            "num_inference_steps": manifest["workload"]["num_inference_steps"],
            "prompt": manifest["workload"]["prompt"],
            "seed_plan": manifest["workload"]["seed_plan"],
            "conditioning_assets": manifest["workload"]["conditioning_assets"],
        },
        "production": {
            "is_long": True,
            "generation_mode": "extension",
            "chunk_count": CHUNK_COUNT,
            "assembly_method": "sample_exact_extension_concat",
        },
        "track": {
            "id": "practical_disclosed_approx",
            "mechanisms": mechanisms,
            "disclosure": f"Turbo 8-step practical approximation assembled as FL2VA extension chunks; first chunk is text-only bootstrap, chunks 2-{CHUNK_COUNT} condition on the prior retained terminal RGB frame. Attention mode is explicitly recorded in container_summary.attention_backend and track.mechanisms. Request quality is {request_quality!r}; Cache-DiT is a practical approximate mechanism when present and never a BF16-fidelity claim. Video VAE spatial tile batching and bounded tile batch size are explicitly disclosed in track.mechanisms and container_summary.video_vae, and are practical approximate, not BF16 fidelity, when enabled.",
        },
        "deployment": {
            "scope": "single_a6000",
            "gpu_model": "NVIDIA RTX A6000",
            "gpu_count_visible": 1,
            "gpu_count_used": 1,
            "physical_gpu_uuids": [gpu_uuid],
        },
        "timing": {
            "boundary_id": timing.get("boundary_id", TIMING_BOUNDARY_ID),
            "cold_e2e": timing_metric("measured", float(timing["cold_e2e_seconds"]), 1),
            "warm_e2e": timing_metric("measured", float(timing["warm_e2e_seconds"]), 1),
            "components": {
                "text_conditioning": component_metric("measured", float(stage.get("text_conditioning_seconds") or 0.0), CHUNK_COUNT, "wall_seconds", True),
                "denoise": component_metric("measured", float(stage.get("denoise_seconds") or 0.0), CHUNK_COUNT, "wall_seconds", True),
                "attention": (
                    component_metric("measured", attention_seconds, CHUNK_COUNT, "cuda_event_device_seconds_nested_in_denoise", False, parent="denoise")
                    if attention_seconds is not None and attention_seconds > 0
                    else component_metric("not_available_historical_evidence", None, 0, "not_available", False, unavailable, parent="denoise")
                ),
                "video_vae": (
                    component_metric("measured", video_vae_wall, CHUNK_COUNT, "wall_seconds", True)
                    if video_vae_wall is not None
                    else component_metric("not_available_historical_evidence", None, 0, "not_available", True, unavailable)
                ),
                "audio_vae": (
                    component_metric("measured", audio_vae_wall, CHUNK_COUNT, "wall_seconds", True)
                    if audio_vae_wall is not None
                    else component_metric("not_available_historical_evidence", None, 0, "not_available", True, unavailable)
                ),
                "encoding_mux": component_metric("measured", server_encoding_mux_wall + final_encoding_mux_wall, CHUNK_COUNT + 1, "wall_seconds", True),
                "io": component_metric("measured", response_download_wall + final_audio_io_wall, CHUNK_COUNT + 1, "wall_seconds", True),
            },
            "additive_component_order": ["text_conditioning", "denoise", "video_vae", "audio_vae", "encoding_mux", "io"],
            "seconds_per_generated_second": timing_metric("measured", float(timing["seconds_per_generated_second"]), 1),
            "profile_counters": {
                "split_profile_status": stage.get("split_profile_status"),
                "video_vae_decode_device_seconds": stage.get("video_vae_decode_device_seconds"),
                "audio_vae_decode_device_seconds": stage.get("audio_vae_decode_device_seconds"),
                "video_crop_contiguous_wall_seconds": stage.get("video_crop_contiguous_wall_seconds"),
                "video_crop_contiguous_device_seconds": stage.get("video_crop_contiguous_device_seconds"),
                "python_cpu_sync_copy_seconds": stage.get("python_cpu_sync_copy_seconds"),
                "python_cpu_sync_copy_bytes": stage.get("python_cpu_sync_copy_bytes"),
                "postprocess_video_cpu_copy_wall_seconds": stage.get("postprocess_video_cpu_copy_wall_seconds"),
                "postprocess_audio_cpu_copy_wall_seconds": stage.get("postprocess_audio_cpu_copy_wall_seconds"),
                "true_client_wait_seconds": stage.get("true_client_wait_seconds"),
                "server_request_wall_seconds": stage.get("server_request_wall_seconds"),
                "curl_http_total_seconds": stage.get("curl_http_total_seconds"),
                "curl_process_overhead_seconds": stage.get("curl_process_overhead_seconds"),
                "server_encoding_mux_wall_seconds": server_encoding_mux_wall,
                "response_transfer_seconds": response_download_wall,
                "response_download_seconds": response_download_wall,
                "per_chunk_variance": stage.get("per_chunk_variance"),
            },
        },
        "resources": {
            "peak_gpu_memory_mib": metric("measured", resources.get("peak_gpu_memory_mib"), "MiB"),
            "peak_host_memory_gib": metric("measured", resources.get("peak_host_memory_used_gib"), "GiB"),
            "peak_power_w": metric("measured", resources.get("peak_power_w"), "W"),
            "failures": metric("measured", 0, "count"),
        },
        "output_av": {
            "status": "complete",
            "final_accounting_complete": True,
            "video": {"present": True, "full_decode": True, "width": WIDTH, "height": HEIGHT, "fps": FPS, "frames": validation["video"]["frames"]},
            "audio": {
                "present": True,
                "full_decode": True,
                "sample_rate_hz": AUDIO_RATE,
                "channels": AUDIO_CHANNELS,
                "decoded_samples_per_channel": validation["audio"]["decoded_samples_per_channel"],
                "effective_samples_per_channel": FINAL_AUDIO_SAMPLES_PER_CHANNEL,
                "encoder_priming_samples": 0,
                "end_padding_samples": 0,
            },
        },
        "quality": {
            "objective": objective_record,
            "human_gate": {"status": "not_performed_no_semantic_claim", "scope": "Human visual/audio/AV-sync review was not performed; objective proxies are descriptive only."},
        },
        "promotion": {"level": "n1_gate", "sample_count": 1, "quality_threshold_id": "objective-proxy-no-human-descriptive-v1"},
        "comparisons": [],
        "claim_boundary": summary.get("claim_boundary", default_claim_boundary),
    }
    record["workload_fingerprint"] = workload_fingerprint(record)
    return record


def write_json_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite existing recovery artifact: {path}")
    write_json(path, payload)


def epoch_to_utc(epoch_s: float) -> str:
    return dt.datetime.fromtimestamp(epoch_s, tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc_epoch(value: str) -> float:
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    return dt.datetime.fromisoformat(raw).timestamp()


def preserve_failed_provenance(evidence: Path) -> dict[str, Any]:
    preserve_dir = evidence / "failed_provenance_preserved"
    index = preserve_dir / "index.json"
    if index.exists():
        return read_json(index)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    preserved: list[dict[str, str]] = []
    for name in ("blocker.json", "decision.json", "RUN_REPORT.md", "container_exit_code", "finalize_exit_code"):
        src = evidence / name
        if not src.exists():
            continue
        preserve_dir.mkdir(parents=True, exist_ok=True)
        dst = preserve_dir / f"{name}.pre_recovery_{stamp}"
        shutil.copy2(src, dst)
        preserved.append({"source": str(src), "preserved_copy": str(dst)})
    payload = {
        "schema_version": "minimax-h3-failed-provenance-preservation-v1",
        "created_utc": utc_now(),
        "reason": "CPU-only recovery may update root decision/RUN_REPORT; failed blocker and pre-recovery terminal artifacts remain inspectable here.",
        "preserved": preserved,
    }
    write_json(index, payload)
    return payload


def load_recoverable_chunks(evidence: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for idx in range(1, CHUNK_COUNT + 1):
        chunk_dir = evidence / "chunks" / f"chunk_{idx:02d}"
        summary_path = chunk_dir / "chunk_summary.json"
        if not summary_path.exists():
            raise RuntimeError(f"missing chunk summary: {summary_path}")
        summary = read_json(summary_path)
        if int(summary.get("chunk_index", -1)) != idx:
            raise RuntimeError(f"chunk index mismatch in {summary_path}: {summary.get('chunk_index')}")
        output = _host_evidence_path(evidence, str(summary.get("output", "")))
        if not output.exists():
            raise RuntimeError(f"missing chunk output: {output}")
        request = summary.get("request", {})
        expected_sha = request.get("output_sha256_opaque_id")
        if expected_sha and sha256_file(output) != expected_sha:
            raise RuntimeError(f"chunk output sha256 mismatch: {output}")
        retention = summary.get("retention")
        expected_retention = [0, 120] if idx == 1 else [1, 121]
        if retention != expected_retention:
            raise RuntimeError(f"chunk {idx} retention mismatch: {retention} != {expected_retention}")
        validation = summary.get("media_validation", {})
        if validation.get("structural_source_av_pass") is not True:
            raise RuntimeError(f"chunk {idx} structural validation was not pass")
        if validation.get("source_video_frames") != SOURCE_FRAMES_PER_CHUNK or validation.get("retained_frame_count") != RETAINED_FRAMES_PER_CHUNK:
            raise RuntimeError(f"chunk {idx} frame accounting mismatch in media_validation")
        chunks.append(summary)
    return chunks


def small_rgb_proxy(np: Any, arr: Any) -> Any:
    try:
        from PIL import Image  # type: ignore

        resampling = getattr(Image, "Resampling", Image).BILINEAR
        return np.asarray(Image.fromarray(arr).resize((32, 18), resampling)).astype(np.float32) / 255.0
    except ImportError:
        ys = np.linspace(0, arr.shape[0] - 1, 18).astype(np.int64)
        xs = np.linspace(0, arr.shape[1] - 1, 32).astype(np.int64)
        return arr[ys][:, xs].astype(np.float32) / 255.0


def read_final_audio_raw(path: Path) -> Any:
    _av, np = ensure_av_numpy()
    raw = path.read_bytes()
    expected = FINAL_AUDIO_SAMPLES_PER_CHANNEL * AUDIO_CHANNELS * 2
    if len(raw) != expected:
        raise RuntimeError(f"final raw audio size mismatch: {len(raw)} != {expected}")
    return np.frombuffer(raw, dtype=np.int16).reshape(-1, AUDIO_CHANNELS).astype(np.float32) / 32768.0


def compute_objective_metrics_from_chunks(evidence: Path, chunks: list[dict[str, Any]], audio: Any) -> dict[str, Any]:
    av, np = ensure_av_numpy()
    small_frames: list[Any] = []
    histograms: list[Any] = []
    transition_diffs: list[float] = []
    seam_transition_flags: list[bool] = []
    previous_small = None
    retained_frames = 0
    for chunk in chunks:
        chunk_index = int(chunk["chunk_index"])
        retain_start, retain_end = chunk["retention"]
        output = _host_evidence_path(evidence, str(chunk["output"]))
        with av.open(str(output)) as container:
            for frame_index, frame in enumerate(container.decode(video=0)):
                if retain_start <= frame_index < retain_end:
                    arr = frame.to_ndarray(format="rgb24")
                    small = small_rgb_proxy(np, arr)
                    gray = small.mean(axis=2)
                    small_frames.append(gray)
                    hist, _ = np.histogramdd(
                        small.reshape(-1, 3), bins=(8, 8, 8), range=((0, 1), (0, 1), (0, 1)), density=False
                    )
                    hist = hist.astype(np.float64).reshape(-1)
                    histograms.append(hist / max(hist.sum(), 1.0))
                    if previous_small is not None:
                        transition_diffs.append(float(np.mean(np.abs(gray - previous_small))))
                        seam_transition_flags.append(frame_index == retain_start and chunk_index > 1)
                    previous_small = gray
                    retained_frames += 1
    if retained_frames != FINAL_FRAMES:
        raise RuntimeError(f"recovered objective frame count mismatch: {retained_frames} != {FINAL_FRAMES}")
    metrics = compute_objective_metrics(
        np=np,
        histograms=histograms,
        transition_diffs=transition_diffs,
        seam_transition_flags=seam_transition_flags,
        audio=audio,
    )
    metrics["recovery_basis"] = "recomputed_cpu_only_from_existing_retained_chunk_frames_and_existing_final_raw_audio"
    return metrics


def recover_cold_start_epoch(evidence: Path) -> tuple[float, str]:
    pattern = re.compile(r"HOST_COLD_START_EPOCH_NS=(\d+)")
    for probe in sorted(evidence.glob("supervisor_status_probe_*.txt")):
        match = pattern.search(probe.read_text(encoding="utf-8", errors="replace"))
        if match:
            return int(match.group(1)) / 1e9, f"{probe}:HOST_COLD_START_EPOCH_NS"
    lease = evidence / "lease_request.json"
    if lease.exists():
        payload = read_json(lease)
        if payload.get("created_utc"):
            return parse_utc_epoch(str(payload["created_utc"])), f"{lease}:created_utc"
    return evidence.stat().st_mtime, "evidence_directory_mtime_fallback"


def recovered_timing(evidence: Path) -> dict[str, Any]:
    final_mov = evidence / "assembly" / final_av_name()
    final_audio = evidence / "assembly" / final_audio_raw_name()
    ffmpeg_cmd = evidence / "assembly" / "ffmpeg_final_encode_command.json"
    warm_marker = evidence / "warmup_excluded" / "warmup_record.json"
    if not warm_marker.exists():
        warm_marker = evidence / "warmup_excluded" / "media_validation.json"
    last_chunk_marker = evidence / "chunks" / f"chunk_{CHUNK_COUNT:02d}" / "media_validation.json"
    cold_start_s, cold_basis = recover_cold_start_epoch(evidence)
    final_av_s = final_mov.stat().st_mtime
    warm_start_s = warm_marker.stat().st_mtime
    audio_done_s = final_audio.stat().st_mtime
    encode_start_s = ffmpeg_cmd.stat().st_mtime if ffmpeg_cmd.exists() else audio_done_s
    chunk_done_s = last_chunk_marker.stat().st_mtime if last_chunk_marker.exists() else warm_start_s
    cold_e2e = max(final_av_s - cold_start_s, 1e-6)
    warm_e2e = max(final_av_s - warm_start_s, 1e-6)
    audio_io = max(audio_done_s - chunk_done_s, 1e-6)
    encode_mux = max(final_av_s - max(encode_start_s, audio_done_s), 1e-6)
    return {
        "boundary_id": f"final_av_{DURATION_LABEL}_extension_recovered_existing_final_av_mtime_v1",
        "cold_e2e_seconds": cold_e2e,
        "warm_e2e_seconds": warm_e2e,
        "seconds_per_generated_second": warm_e2e / FINAL_DURATION_SECONDS,
        "cold_boundary_note": "Recovered from original HOST_COLD_START_EPOCH_NS (or lease fallback) to existing final AV file mtime; no GPU rerun.",
        "warm_boundary_note": "Recovered from original warmup_record mtime to existing final AV file mtime; no GPU rerun.",
        "recovery_timing_basis": {
            "cold_start": {"epoch_seconds": cold_start_s, "utc": epoch_to_utc(cold_start_s), "source": cold_basis},
            "warm_start": {"epoch_seconds": warm_start_s, "utc": epoch_to_utc(warm_start_s), "source": str(warm_marker)},
            "final_av_complete": {"epoch_seconds": final_av_s, "utc": epoch_to_utc(final_av_s), "source": str(final_mov)},
            "audio_extract_write_io_seconds_from_mtime": audio_io,
            "encoding_mux_seconds_from_mtime": encode_mux,
        },
    }


def encode_recovered_final_av_pyav(evidence: Path, chunks: list[dict[str, Any]], final_audio: Path, recovered_mov: Path) -> dict[str, Any]:
    if recovered_mov.exists():
        raise RuntimeError(f"refusing to overwrite existing recovered final AV: {recovered_mov}")
    av, np = ensure_av_numpy()
    raw = np.frombuffer(final_audio.read_bytes(), dtype=np.int16).copy().reshape(-1, AUDIO_CHANNELS)
    if raw.shape != (FINAL_AUDIO_SAMPLES_PER_CHANNEL, AUDIO_CHANNELS):
        raise RuntimeError(f"recovered encode audio shape mismatch: {raw.shape}")
    t0 = time.perf_counter()
    written = 0
    container = av.open(str(recovered_mov), "w", format="mov")
    try:
        video_stream = container.add_stream("libx264", rate=FPS)
        video_stream.width = WIDTH
        video_stream.height = HEIGHT
        video_stream.pix_fmt = "yuv420p"
        video_stream.options = {"preset": "ultrafast"}
        audio_stream = container.add_stream("pcm_s16le", rate=AUDIO_RATE)
        audio_stream.layout = "stereo"
        for chunk in chunks:
            retain_start, retain_end = chunk["retention"]
            output = _host_evidence_path(evidence, str(chunk["output"]))
            with av.open(str(output)) as source:
                for frame_index, frame in enumerate(source.decode(video=0)):
                    if retain_start <= frame_index < retain_end:
                        arr = frame.to_ndarray(format="rgb24")
                        out_frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
                        out_frame.pts = written
                        for packet in video_stream.encode(out_frame):
                            container.mux(packet)
                        written += 1
        for packet in video_stream.encode():
            container.mux(packet)
        audio_frame = av.AudioFrame.from_ndarray(raw.reshape(1, -1), format="s16", layout="stereo")
        audio_frame.sample_rate = AUDIO_RATE
        for packet in audio_stream.encode(audio_frame):
            container.mux(packet)
        for packet in audio_stream.encode():
            container.mux(packet)
    finally:
        container.close()
    if written != FINAL_FRAMES:
        raise RuntimeError(f"recovered encode wrote {written} frames, expected {FINAL_FRAMES}")
    return {
        "path": str(recovered_mov),
        "frames_written": written,
        "cpu_recovery_encode_wall_seconds": time.perf_counter() - t0,
        "encoder": "pyav-libx264-ultrafast-pcm_s16le-mov",
    }


def recover_final_av(args: argparse.Namespace) -> int:
    evidence = Path(args.evidence).resolve()
    manifest = read_json(Path(args.manifest))
    configure_lane_from_manifest(manifest)
    gpu_uuid = args.gpu_uuid
    if not gpu_uuid:
        lease = evidence / "lease_request.json"
        if not lease.exists():
            raise RuntimeError("--gpu-uuid not provided and lease_request.json is missing")
        gpu_uuid = str(read_json(lease).get("gpu_uuid", ""))
    if not gpu_uuid:
        raise RuntimeError("GPU UUID is required for recovered benchmark record")
    chunks = load_recoverable_chunks(evidence)
    assembly_dir = evidence / "assembly"
    recovery_dir = evidence / "cpu_recovery"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    final_audio = assembly_dir / final_audio_raw_name()
    original_final_mov = assembly_dir / final_av_name()
    final_mov = original_final_mov
    if not final_audio.exists() or not original_final_mov.exists():
        raise RuntimeError("existing final AV/raw audio artifacts are missing; CPU-only recovery is not valid")
    recovery_encode: dict[str, Any] | None = None
    original_final_validation_error: str | None = None
    try:
        validation = validate_final_av(original_final_mov)
    except Exception as exc:  # noqa: BLE001
        original_final_validation_error = f"{type(exc).__name__}: {exc}"
        final_mov = recovery_dir / f"recovered_{final_av_name()}"
        recovery_encode = encode_recovered_final_av_pyav(evidence, chunks, final_audio, final_mov)
        validation = validate_final_av(final_mov)
    audio = read_final_audio_raw(final_audio)
    objective = compute_objective_metrics_from_chunks(evidence, chunks, audio)
    objective_path = recovery_dir / "objective_metrics.json"
    write_json_new(objective_path, objective)
    timing = recovered_timing(evidence)
    accounting = {
        "schema_version": f"minimax-h3-final-av-{DURATION_LABEL}-accounting-v1",
        "created_utc": utc_now(),
        "recovery_status": "cpu_only_recovered_from_existing_final_av_after_subprocess_stdin_bug",
        "original_final_av_path": str(original_final_mov),
        "original_final_av_validation_error": original_final_validation_error,
        "cpu_recovery_encode": recovery_encode,
        "final_av_path": str(final_mov),
        "final_av_sha256_opaque_id": sha256_file(final_mov),
        "final_av_bytes": final_mov.stat().st_size,
        "final_audio_raw_path": str(final_audio),
        "final_audio_raw_bytes": final_audio.stat().st_size,
        "video_frames_effective": FINAL_FRAMES,
        "audio_samples_effective_per_channel": FINAL_AUDIO_SAMPLES_PER_CHANNEL,
        "audio_encoder_priming_samples": 0,
        "audio_end_padding_samples": 0,
        "source_chunk_count": len(chunks),
        "assembly_timing_seconds": {
            "audio_extract_write_io": timing["recovery_timing_basis"]["audio_extract_write_io_seconds_from_mtime"],
            "encoding_mux_wall": timing["recovery_timing_basis"]["encoding_mux_seconds_from_mtime"],
            "timing_basis": "recovered_original_filesystem_mtime_no_gpu_rerun",
        },
        "final_decode_validation": validation,
        "objective_metrics_path": str(objective_path),
    }
    accounting_path = recovery_dir / "final_av_accounting.json"
    write_json_new(accounting_path, accounting)
    stage_summary = summarize_stage_durations(chunks)
    resources = peak_resource_summary(evidence / "resource_monitor.csv")
    summary = {
        "schema_version": f"minimax-h3-final-av-{DURATION_LABEL}-extension-n1-container-summary-v1",
        "created_utc": utc_now(),
        "status": "pass",
        "classification": f"descriptive_n1_final_av_{DURATION_LABEL}_extension_complete_recovered_cpu_only",
        "recovery_status": "original_gpu_generation_failed_in_finalizer_subprocess_stdin_lifecycle; existing chunks/raw audio were validated without GPU rerun and final MOV was reused if valid or recovered to a new path if the original MOV was unindexed",
        "generation_mode": "extension",
        "native_context_supported": False,
        "track": "practical_disclosed_approx",
        "mechanisms": ["statically_merged_turbo_lora", "8_step", "extension", "no_cache", "dense_attention"],
        "chunks": chunks,
        "warmup_excluded": str(evidence / "warmup_excluded"),
        "final_av_accounting": accounting,
        "final_av_accounting_path": str(accounting_path),
        "stage_duration_summary": stage_summary,
        "timing": timing,
        "resources": resources,
        "monitor_failures": [],
        "failed_provenance": str(evidence / "blocker.json") if (evidence / "blocker.json").exists() else None,
        "claim_boundary": f"Recovered descriptive N=1 {DURATION_READABLE} 1344x768/24FPS final-AV extension baseline only; no native-long-context, speedup, BF16-fidelity, public-comparison, SOTA, or human-quality claim.",
    }
    write_json_new(evidence / "container_summary.json", summary)
    record = make_benchmark_record(evidence, summary, manifest, gpu_uuid)
    record["source_evidence"].extend([
        {"path": str(evidence / "blocker.json"), "scope": "Preserved failed finalizer provenance: subprocess stdin lifecycle bug after media generation."},
        {"path": str(evidence / "failed_provenance_preserved" / "index.json"), "scope": "Pre-recovery failed terminal artifacts preserved before root decision/RUN_REPORT update."},
    ])
    write_json_new(evidence / "benchmark_record.json", record)
    preserved = preserve_failed_provenance(evidence)
    decision = {
        "schema_version": f"minimax-h3-final-av-{DURATION_LABEL}-extension-n1-decision-v1",
        "created_utc": utc_now(),
        "status": "pass",
        "classification": f"descriptive_n1_final_av_{DURATION_LABEL}_extension_complete_recovered_pending_independent_review",
        "benchmark_record": str(evidence / "benchmark_record.json"),
        "generation_mode": "extension",
        "native_context_supported": False,
        "promote_to_n3": False,
        "speedup_claimed": False,
        "human_quality_claimed": False,
        "reviewer_acceptance_status": "pending_host_reviewer_not_invoked_by_engineer",
        "failed_provenance_preserved": preserved,
        "claim_boundary": record["claim_boundary"],
    }
    write_json(evidence / "decision.json", decision)
    write_run_report(evidence, decision, record)
    return 0


def write_run_report(evidence: Path, decision: dict[str, Any], record: dict[str, Any] | None = None) -> None:
    lines = [
        f"# Final-AV {DURATION_LABEL} Extension N=1 RUN_REPORT",
        "",
        f"- Status: `{decision.get('status')}`",
        f"- Classification: `{decision.get('classification')}`",
        f"- Evidence directory: `{evidence}`",
        f"- Generation mode: `extension` (not native long context)",
        f"- Claim boundary: {decision.get('claim_boundary')}",
    ]
    if record is not None:
        lines.extend([
            f"- Benchmark record: `{evidence / 'benchmark_record.json'}`",
            f"- Warm E2E: {record['timing']['warm_e2e']['seconds']:.3f} s",
            f"- Cold E2E: {record['timing']['cold_e2e']['seconds']:.3f} s",
            f"- Seconds per generated second: {record['timing']['seconds_per_generated_second']['seconds']:.3f}",
            f"- Final video frames: {record['output_av']['video']['frames']}",
            f"- Effective audio samples/channel: {record['output_av']['audio']['effective_samples_per_channel']}",
            "- Human gate: not performed; no semantic/human-quality claim.",
            "- Component caveat: attention remains nested under denoise; split video/audio VAE fields are measured only when the default-off split profiler was enabled, otherwise unavailable fields are explicit in the candidate record.",
        ])
    if decision.get("reviewer_acceptance_status"):
        lines.append(f"- Reviewer acceptance: `{decision.get('reviewer_acceptance_status')}`")
    (evidence / "RUN_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_record(args: argparse.Namespace) -> int:
    evidence = Path(args.evidence).resolve()
    manifest = read_json(Path(args.manifest))
    configure_lane_from_manifest(manifest)
    if (evidence / "container_summary.json").exists():
        summary = read_json(evidence / "container_summary.json")
        record = make_benchmark_record(evidence, summary, manifest, args.gpu_uuid)
        write_json(evidence / "benchmark_record.json", record)
        decision = {
            "schema_version": f"minimax-h3-final-av-{DURATION_LABEL}-extension-n1-decision-v1",
            "created_utc": utc_now(),
            "status": "pass",
            "classification": f"descriptive_n1_final_av_{DURATION_LABEL}_extension_complete_pending_independent_review",
            "benchmark_record": str(evidence / "benchmark_record.json"),
            "generation_mode": "extension",
            "native_context_supported": False,
            "promote_to_n3": False,
            "speedup_claimed": False,
            "human_quality_claimed": False,
            "reviewer_acceptance_status": "pending_host_reviewer_not_invoked_by_engineer",
            "claim_boundary": record["claim_boundary"],
        }
        write_json(evidence / "decision.json", decision)
        write_run_report(evidence, decision, record)
        return 0
    blocker_path = evidence / "blocker.json"
    blocker = read_json(blocker_path) if blocker_path.exists() else {
        "status": "failed",
        "classification": "fail_closed_no_container_summary",
        "claim_boundary": f"No qualifying {DURATION_READABLE} final-AV baseline result was produced.",
    }
    decision = {
        "schema_version": f"minimax-h3-final-av-{DURATION_LABEL}-extension-n1-decision-v1",
        "created_utc": utc_now(),
        "status": "failed",
        "classification": blocker.get("classification", "fail_closed_runtime_or_contract_blocker"),
        "blocker": str(blocker_path) if blocker_path.exists() else None,
        "promote_to_n3": False,
        "speedup_claimed": False,
        "human_quality_claimed": False,
        "reviewer_acceptance_status": "not_applicable_failed_before_review",
        "claim_boundary": blocker.get("claim_boundary", "No qualifying result."),
    }
    write_json(evidence / "decision.json", decision)
    write_run_report(evidence, decision, None)
    return 1


def _load_objective_from_summary(evidence: Path, summary: dict[str, Any]) -> dict[str, Any]:
    accounting = summary.get("final_av_accounting", {})
    path = accounting.get("objective_metrics_path")
    if not path and summary.get("final_av_accounting_path"):
        accounting = read_json(_host_evidence_path(evidence, str(summary["final_av_accounting_path"])))
        path = accounting.get("objective_metrics_path")
    if not path:
        return {}
    return read_json(_host_evidence_path(evidence, str(path)))


def _objective_noninferiority(dense: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    higher_better = ("subject_identity_consistency", "background_consistency", "camera_consistency", "motion")
    out: dict[str, Any] = {"relative_margin_percent": 5.0, "metrics": {}, "failed_metrics": []}
    for name in higher_better:
        d_item = dense.get(name, {}) if isinstance(dense.get(name), dict) else {}
        c_item = candidate.get(name, {}) if isinstance(candidate.get(name), dict) else {}
        d_val = d_item.get("value")
        c_val = c_item.get("value")
        passed = False
        rel = None
        if isinstance(d_val, (int, float)) and isinstance(c_val, (int, float)):
            threshold = float(d_val) * 0.95
            passed = float(c_val) >= threshold
            rel = (float(c_val) / float(d_val)) if abs(float(d_val)) > 1e-12 else None
        out["metrics"][name] = {
            "dense_value": d_val,
            "candidate_value": c_val,
            "candidate_over_dense": rel,
            "pass_5pct_noninferiority": passed,
            "higher_is_better": True,
        }
        if not passed:
            out["failed_metrics"].append(name)
    out["status"] = "pass" if not out["failed_metrics"] else "fail"
    return out


def finalize_matched(args: argparse.Namespace) -> int:
    root = Path(args.out_dir).resolve()
    dense_dir = Path(args.dense_evidence).resolve()
    candidate_dir = Path(args.candidate_evidence).resolve()
    root.mkdir(parents=True, exist_ok=True)
    dense_record = read_json(dense_dir / "benchmark_record.json")
    candidate_record = read_json(candidate_dir / "benchmark_record.json")
    dense_summary = read_json(dense_dir / "container_summary.json")
    candidate_summary = read_json(candidate_dir / "container_summary.json")
    dense_objective = _load_objective_from_summary(dense_dir, dense_summary)
    candidate_objective = _load_objective_from_summary(candidate_dir, candidate_summary)
    sol_attn = candidate_summary.get("sol_attn") if isinstance(candidate_summary.get("sol_attn"), dict) else {}
    sol_summary = sol_attn.get("summary") if isinstance(sol_attn.get("summary"), dict) else {}
    dense_warm = float(dense_record["timing"]["warm_e2e"]["seconds"])
    candidate_warm = float(candidate_record["timing"]["warm_e2e"]["seconds"])
    n1_delta_pct = (dense_warm - candidate_warm) / dense_warm * 100.0
    dense_cold = float(dense_record["timing"]["cold_e2e"]["seconds"])
    candidate_cold = float(candidate_record["timing"]["cold_e2e"]["seconds"])
    cold_delta_pct = (dense_cold - candidate_cold) / dense_cold * 100.0
    same_gpu = dense_record["deployment"].get("physical_gpu_uuids") == candidate_record["deployment"].get("physical_gpu_uuids")
    same_workload = dense_record.get("workload_fingerprint") == candidate_record.get("workload_fingerprint")
    same_timing = dense_record["timing"].get("boundary_id") == candidate_record["timing"].get("boundary_id")
    same_generation = dense_record["production"].get("generation_mode") == candidate_record["production"].get("generation_mode") == "extension"
    candidate_flags = list(candidate_objective.get("automatic_red_flags", []) or [])
    dense_flags = list(dense_objective.get("automatic_red_flags", []) or [])
    objective_cmp = _objective_noninferiority(dense_record["quality"]["objective"], candidate_record["quality"]["objective"])
    sparse_calls = int(sol_summary.get("sparse_calls") or 0)
    sparse_candidates = int(sol_summary.get("sparse_candidate_calls") or 0)
    fallback_calls = int(sol_summary.get("fallback_calls") or 0)
    materialize_count = sol_summary.get("materialize_copy_count")
    materialize_bytes = sol_summary.get("materialize_copy_bytes")
    gates = {
        "same_physical_gpu": same_gpu,
        "same_workload_fingerprint": same_workload,
        "same_timing_boundary": same_timing,
        "same_extension_generation_mode": same_generation,
        "both_final_av_complete": dense_record["output_av"].get("final_accounting_complete") is True and candidate_record["output_av"].get("final_accounting_complete") is True,
        "candidate_sol_attn_telemetry_present": sol_attn.get("status") == "present",
        "candidate_sparse_calls_positive": sparse_calls > 0,
        "candidate_sparse_candidates_positive": sparse_candidates > 0,
        "candidate_fallback_calls_zero": fallback_calls == 0,
        "candidate_materialization_telemetry_present": isinstance(materialize_count, int) and isinstance(materialize_bytes, int),
        "candidate_no_raw_tensor_export": sol_summary.get("diagnostic_raw_tensor_exported") is False,
        "candidate_no_automatic_proxy_flags": len(candidate_flags) == 0,
        "dense_no_automatic_proxy_flags": len(dense_flags) == 0,
        "objective_5pct_noninferiority_core_metrics": objective_cmp.get("status") == "pass",
        "candidate_warm_e2e_not_slower": candidate_warm <= dense_warm,
        "n1_delta_meets_route_threshold": n1_delta_pct >= float(args.min_delta_pct),
    }
    failed = [name for name, ok in gates.items() if not ok]
    correctness_failed = [name for name in failed if name not in {"candidate_warm_e2e_not_slower", "n1_delta_meets_route_threshold"}]
    if correctness_failed:
        status = "reject"
        classification = "reject_r8_sol_attn_long_lane_correctness_telemetry_or_quality_gate_failed"
        promote = False
    elif failed:
        status = "reject"
        classification = "reject_r8_sol_attn_long_lane_no_n3_small_or_negative_n1_signal"
        promote = False
    else:
        status = "pass"
        classification = "keep_r8_sol_attn_long_lane_n1_route_gate_pass_pending_independent_review"
        promote = True
    decision = {
        "schema_version": "minimax-h3-final-av-30s-dense-vs-r8-sol-attn-n1-decision-v1",
        "created_utc": utc_now(),
        "status": status,
        "classification": classification,
        "dense_evidence": str(dense_dir),
        "candidate_evidence": str(candidate_dir),
        "principal_variable": "attention backend/config: dense CUDNN_ATTN versus accepted r8 H3_A6000_SOL_ATTN opt-in",
        "generation_mode": "extension",
        "native_context_supported": False,
        "sample_count": 1,
        "speedup_claimed": False,
        "n1_route_gate_language_only": True,
        "promote_to_n3_recommended_pending_reviewer": promote,
        "reviewer_acceptance_status": "pending_host_reviewer_not_invoked_by_engineer",
        "timing": {
            "dense_warm_e2e_seconds": dense_warm,
            "candidate_warm_e2e_seconds": candidate_warm,
            "warm_e2e_delta_percent_n1_not_speedup": n1_delta_pct,
            "dense_cold_e2e_seconds": dense_cold,
            "candidate_cold_e2e_seconds": candidate_cold,
            "cold_e2e_delta_percent_n1_not_speedup": cold_delta_pct,
            "route_threshold_percent": float(args.min_delta_pct),
        },
        "sol_attn_telemetry_summary": sol_summary,
        "resources": {
            "dense": dense_record.get("resources"),
            "candidate": candidate_record.get("resources"),
        },
        "objective_proxy_comparison": objective_cmp,
        "automatic_red_flags": {"dense": dense_flags, "candidate": candidate_flags},
        "gates": gates,
        "failed_gates": failed,
        "claim_boundary": "Matched N=1 long-lane route gate only. It compares dense CUDNN_ATTN with accepted r8 H3_A6000_SOL_ATTN opt-in on the 30-second extension lane; no formal speedup, BF16-fidelity, native-long-context, human-quality, public-comparison, or SOTA claim.",
    }
    write_json(root / "decision.json", decision)
    lines = [
        "# Dense CUDNN vs r8 Sol-Attn Final-AV 30s N=1 RUN_REPORT",
        "",
        f"- Status: `{status}`",
        f"- Classification: `{classification}`",
        f"- Dense evidence: `{dense_dir}`",
        f"- Candidate evidence: `{candidate_dir}`",
        f"- Warm E2E dense/candidate: {dense_warm:.3f}s / {candidate_warm:.3f}s",
        f"- N=1 warm delta (route-gate only, not speedup): {n1_delta_pct:.3f}%",
        f"- Sol-Attn sparse/fallback/materialization: sparse={sparse_calls}, fallback={fallback_calls}, materialize={materialize_count}/{materialize_bytes} bytes",
        f"- Failed gates: {failed}",
        "- Claim boundary: extension output, N=1 only, no formal speedup or human-quality claim.",
        "- Reviewer acceptance: `pending_host_reviewer_not_invoked_by_engineer`",
    ]
    (root / "RUN_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def _numeric_summary_value(summary: dict[str, Any], key: str, default: int = 0) -> int:
    value = summary.get(key, default)
    try:
        return int(value)
    except Exception:
        return default


def _sol_attn_zero_copy_ok(summary: dict[str, Any]) -> bool:
    return (
        _numeric_summary_value(summary, "fallback_calls", -1) == 0
        and _numeric_summary_value(summary, "materialize_copy_count", -1) == 0
        and _numeric_summary_value(summary, "materialize_copy_bytes", -1) == 0
        and _numeric_summary_value(summary, "input_copy_events", -1) == 0
        and _numeric_summary_value(summary, "input_copy_bytes", -1) == 0
    )


def _sol_attn_sparse_stride_ok(summary: dict[str, Any]) -> bool:
    sparse_calls = _numeric_summary_value(summary, "sparse_calls", 0)
    sparse_candidates = _numeric_summary_value(summary, "sparse_candidate_calls", 0)
    stride_calls = _numeric_summary_value(summary, "stride_aware_value_calls", 0)
    return sparse_calls > 0 and sparse_candidates > 0 and stride_calls == sparse_calls


def _tau_gate_label(tau: float) -> str:
    text = f"{tau:g}".replace(".", "_").replace("-", "minus_")
    return text


def _candidate_step_min_from_profile(candidate_profile: str, explicit_step_min: int | None = None) -> int | None:
    if explicit_step_min is not None:
        return int(explicit_step_min)
    profile = candidate_profile.lower()
    if profile == "r11_adaptive_tau1_5_step2_diag" or profile == "r12_adaptive_tau1_5_step2_layers34_49_diag" or "step_min_2" in profile or "step2" in profile:
        return 2
    if profile == "r10_adaptive_tau1_5_step3_diag" or "step_min_3" in profile or "step3" in profile:
        return 3
    if profile == "r9_adaptive_tau1_5_late_steps_diag" or "late_steps" in profile or "step_min_4" in profile:
        return 4
    return None


def _candidate_layer_range_from_profile(
    candidate_profile: str,
    explicit_layer_min: int | None = None,
    explicit_layer_max: int | None = None,
    explicit_scope: str | None = None,
) -> tuple[int | None, int | None, str | None]:
    profile = str(candidate_profile or "").lower()
    layer_min = explicit_layer_min
    layer_max = explicit_layer_max
    scope = explicit_scope
    if profile == "r12_adaptive_tau1_5_step2_layers34_49_diag":
        layer_min = 34 if layer_min is None else layer_min
        layer_max = 49 if layer_max is None else layer_max
        scope = "step_min_only" if scope is None else scope
    return layer_min, layer_max, scope


def _candidate_label_bundle(candidate_profile: str, expected_step_min: int | None = None) -> dict[str, str]:
    """Return artifact labels that keep guarded adaptive profiles distinct."""
    profile = str(candidate_profile or "candidate")
    if profile == "r11_adaptive_tau1_5_step2_diag":
        return {
            "schema_token": "r11-guarded-adaptive-step2",
            "classification_token": "r11_adaptive_tau1_5_step2_diag",
            "candidate_readable": "r11 Guarded Adaptive Step-Min=2",
            "comparison_readable": "r10 Retained vs r11 Guarded Adaptive Step-Min=2",
        }
    if profile == "r12_adaptive_tau1_5_step2_layers34_49_diag":
        return {
            "schema_token": "r12-guarded-adaptive-step2-layers34-49",
            "classification_token": "r12_adaptive_tau1_5_step2_layers34_49_diag",
            "candidate_readable": "r12 Guarded Adaptive Step-2 Layers 34-49",
            "comparison_readable": "r10 Retained vs r12 Guarded Adaptive Step-2 Layers 34-49",
        }
    if profile == "r10_adaptive_tau1_5_step3_diag":
        return {
            "schema_token": "r10-guarded-adaptive-step3",
            "classification_token": "r10_adaptive_tau1_5_step3_diag",
            "candidate_readable": "r10 Guarded Adaptive Step-Min=3",
            "comparison_readable": "r9 Current vs r10 Guarded Adaptive Step-Min=3",
        }
    if profile == "r9_adaptive_tau1_5_late_steps_diag":
        return {
            "schema_token": "r9-guarded-adaptive-step4",
            "classification_token": "r9_guarded_adaptive_step4",
            "candidate_readable": "r9 Guarded Adaptive Step-Min=4",
            "comparison_readable": "r9 Current vs r9 Guarded Adaptive Step-Min=4",
        }
    suffix = f"step{expected_step_min}" if expected_step_min is not None else "unguarded"
    safe = re.sub(r"[^a-z0-9]+", "_", profile.lower()).strip("_") or "candidate"
    return {
        "schema_token": safe.replace("_", "-"),
        "classification_token": f"{safe}_{suffix}",
        "candidate_readable": f"{profile} ({suffix})",
        "comparison_readable": f"r9 Current vs {profile} ({suffix})",
    }


def _lane_context_from_records(reference_record: dict[str, Any], candidate_record: dict[str, Any]) -> dict[str, Any]:
    workload = candidate_record.get("workload") if isinstance(candidate_record.get("workload"), dict) else {}
    production = candidate_record.get("production") if isinstance(candidate_record.get("production"), dict) else {}
    output = candidate_record.get("output_av") if isinstance(candidate_record.get("output_av"), dict) else {}
    video = output.get("video") if isinstance(output.get("video"), dict) else {}
    audio = output.get("audio") if isinstance(output.get("audio"), dict) else {}
    duration = workload.get("nominal_duration_seconds", 30.0)
    try:
        duration_f = float(duration)
    except Exception:
        duration_f = 30.0
    duration_label = _duration_label(duration_f)
    chunk_count = production.get("chunk_count", 6)
    try:
        chunk_count_i = int(chunk_count)
    except Exception:
        chunk_count_i = 6
    return {
        "lane_id": candidate_record.get("lane_id") or reference_record.get("lane_id") or "final-av-30s-1344x768-24fps-v1",
        "duration_seconds": duration_f,
        "duration_label": duration_label,
        "duration_readable": _duration_readable(duration_f),
        "chunk_count": chunk_count_i,
        "final_frames": video.get("frames") or workload.get("final_frame_count", 720),
        "effective_audio_samples_per_channel": audio.get("effective_samples_per_channel", 960000),
    }


def _adaptive_guard_consistency(
    summary: dict[str, Any],
    *,
    expected_step_min: int | None,
    sparse_calls: int,
    expected_layer_min: int | None = None,
    expected_layer_max: int | None = None,
    expected_layer_range_scope: str | None = None,
) -> dict[str, bool]:
    if expected_step_min is None:
        return {
            "candidate_guarded_profile_exercised": True,
            "candidate_adaptive_step_min_expected_seen": True,
            "candidate_adaptive_layer_range_expected_seen": expected_layer_min is None and expected_layer_max is None,
            "candidate_guard_counts_match_step_min": True,
        }
    requested = _numeric_summary_value(summary, "adaptive_guard_requested_count", 0)
    active = _numeric_summary_value(summary, "adaptive_guard_active_count", 0)
    inactive = _numeric_summary_value(summary, "adaptive_guard_inactive_count", 0)
    reason_counts = summary.get("adaptive_guard_reason_counts", {}) if isinstance(summary.get("adaptive_guard_reason_counts"), dict) else {}
    step_min_values = summary.get("adaptive_step_min_values", []) if isinstance(summary.get("adaptive_step_min_values"), list) else []
    layer_min_values = summary.get("adaptive_layer_min_values", []) if isinstance(summary.get("adaptive_layer_min_values"), list) else []
    layer_max_values = summary.get("adaptive_layer_max_values", []) if isinstance(summary.get("adaptive_layer_max_values"), list) else []
    layer_scope_values = summary.get("adaptive_layer_range_scope_values", []) if isinstance(summary.get("adaptive_layer_range_scope_values"), list) else []
    step_values = summary.get("step_index_values", []) if isinstance(summary.get("step_index_values"), list) else []
    layer_values = summary.get("layer_index_values", []) if isinstance(summary.get("layer_index_values"), list) else []
    basic = active > 0 and inactive > 0 and requested == sparse_calls and sparse_calls > 0
    step_min_seen = step_min_values == [expected_step_min]
    layer_min_seen = expected_layer_min is None or layer_min_values == [expected_layer_min]
    layer_max_seen = expected_layer_max is None or layer_max_values == [expected_layer_max]
    layer_scope = expected_layer_range_scope or "all_adaptive_steps"
    layer_scope_seen = expected_layer_range_scope is None or layer_scope_values == [expected_layer_range_scope]
    layer_range_seen = layer_min_seen and layer_max_seen and layer_scope_seen
    expected_reason_counts: dict[str, int] | None = None
    if (
        layer_scope == "step_min_only"
        and step_values
        and layer_values
        and sparse_calls % len(step_values) == 0
        and expected_layer_min is not None
        and expected_layer_max is not None
    ):
        calls_per_step = sparse_calls // len(step_values)
        layer_active_count = sum(
            1
            for layer in layer_values
            if isinstance(layer, int) and expected_layer_min <= layer <= expected_layer_max
        )
        layer_total = len([layer for layer in layer_values if isinstance(layer, int)])
        repeats_per_layer_step = calls_per_step // layer_total if layer_total and calls_per_step % layer_total == 0 else None
        if repeats_per_layer_step is not None:
            expected_reason_counts = {"active": 0, "step_before_adaptive_min": 0}
            for step in step_values:
                if not isinstance(step, int):
                    expected_reason_counts = None
                    break
                if step < expected_step_min:
                    expected_reason_counts["step_before_adaptive_min"] += calls_per_step
                elif step == expected_step_min:
                    expected_reason_counts["active"] += layer_active_count * repeats_per_layer_step
                    before = sum(1 for layer in layer_values if isinstance(layer, int) and layer < expected_layer_min) * repeats_per_layer_step
                    after = sum(1 for layer in layer_values if isinstance(layer, int) and layer > expected_layer_max) * repeats_per_layer_step
                    if before:
                        expected_reason_counts["layer_before_adaptive_min"] = expected_reason_counts.get("layer_before_adaptive_min", 0) + before
                    if after:
                        expected_reason_counts["layer_after_adaptive_max"] = expected_reason_counts.get("layer_after_adaptive_max", 0) + after
                else:
                    expected_reason_counts["active"] += calls_per_step
    if expected_reason_counts is None:
        reason_counts_match = int(reason_counts.get("active", -1)) == active and int(reason_counts.get("step_before_adaptive_min", -1)) <= inactive
    else:
        expected_reason_counts = {k: v for k, v in expected_reason_counts.items() if v}
        reason_counts_match = reason_counts == expected_reason_counts
    counts_match_step_min = False
    by_step = summary.get("adaptive_guard_counts_by_step")
    if isinstance(by_step, dict) and by_step:
        per_step_ok = True
        active_total = 0
        inactive_total = 0
        for step_key, bucket in by_step.items():
            if not isinstance(bucket, dict):
                per_step_ok = False
                continue
            try:
                step = int(step_key)
            except Exception:
                per_step_ok = False
                continue
            bucket_active = int(bucket.get("active", 0) or 0)
            bucket_inactive = int(bucket.get("inactive", 0) or 0)
            active_total += bucket_active
            inactive_total += bucket_inactive
            if step < expected_step_min:
                per_step_ok = per_step_ok and bucket_active == 0 and bucket_inactive > 0
            elif layer_scope == "step_min_only" and step == expected_step_min and expected_layer_min is not None and expected_layer_max is not None:
                per_step_ok = per_step_ok and bucket_active > 0 and bucket_inactive > 0
            else:
                per_step_ok = per_step_ok and bucket_active > 0 and bucket_inactive == 0
        counts_match_step_min = per_step_ok and active_total == active and inactive_total == inactive
    elif expected_reason_counts is not None:
        counts_match_step_min = reason_counts_match
    else:
        counts_match_step_min = False
    return {
        "candidate_guarded_profile_exercised": basic,
        "candidate_adaptive_step_min_expected_seen": step_min_seen,
        "candidate_adaptive_layer_range_expected_seen": layer_range_seen,
        "candidate_guard_counts_match_step_min": reason_counts_match and counts_match_step_min,
    }


def _is_retained_current_profile(profile: str) -> bool:
    normalized = str(profile or "").strip()
    return normalized in {"r9_current", "r9_current_sol_attn", "retained_r9_current_sol_attn"}


def _expected_tau_from_arg(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 6)
    except Exception:
        return None


def _profile_policy_gates(
    *,
    role: str,
    profile: str,
    summary: dict[str, Any],
    expected_tau: float | None,
    expected_step_min: int | None,
    sparse_calls: int,
    tau_gate_name: str | None = None,
    expected_layer_min: int | None = None,
    expected_layer_max: int | None = None,
    expected_layer_range_scope: str | None = None,
) -> dict[str, bool]:
    adaptive_values = set(summary.get("adaptive_routing_values", []) or [])
    tau_values = {round(float(x), 6) for x in summary.get("tau_values", []) or []}
    thresh = set(summary.get("thresh_type_values", []) or [])
    if _is_retained_current_profile(profile):
        return {f"{role}_retained_tau1_diag_seen": True not in adaptive_values and 1.0 in tau_values and "diag" in thresh}
    profile_values = {str(x) for x in summary.get("adaptive_profiles", []) or []}
    tau_ok = expected_tau is not None and True in adaptive_values and expected_tau in tau_values and "diag" in thresh
    gate_name = tau_gate_name or f"{role}_adaptive_tau_diag_seen"
    guard = _adaptive_guard_consistency(
        summary,
        expected_step_min=expected_step_min,
        sparse_calls=sparse_calls,
        expected_layer_min=expected_layer_min,
        expected_layer_max=expected_layer_max,
        expected_layer_range_scope=expected_layer_range_scope,
    )
    renamed_guard = {key.replace("candidate_", f"{role}_", 1): value for key, value in guard.items()}
    return {
        f"{role}_adaptive_profile_seen": bool(profile) and profile in profile_values and tau_ok,
        gate_name: tau_ok,
        **renamed_guard,
    }


def _profile_principal(
    profile: str,
    *,
    expected_tau: float | None,
    expected_step_min: int | None,
    expected_layer_min: int | None = None,
    expected_layer_max: int | None = None,
    expected_layer_range_scope: str | None = None,
) -> str:
    if _is_retained_current_profile(profile):
        return "retained r9_current_sol_attn adaptive_routing=0 tau=1.0 diag"
    tau = f"{float(expected_tau):g}" if expected_tau is not None else "<unresolved>"
    if expected_step_min is None:
        return f"{profile} adaptive_routing=1 tau={tau} diag"
    if expected_layer_min is not None or expected_layer_max is not None:
        scope = expected_layer_range_scope or "all_adaptive_steps"
        lo = "*" if expected_layer_min is None else str(expected_layer_min)
        hi = "*" if expected_layer_max is None else str(expected_layer_max)
        return f"{profile} adaptive_routing=1 tau={tau} diag active from adaptive_step_min={expected_step_min} with layer_range={lo}-{hi} scope={scope}"
    return f"{profile} adaptive_routing=1 tau={tau} diag active from adaptive_step_min={expected_step_min}"


def finalize_r9_adaptive_matched(args: argparse.Namespace) -> int:
    root = Path(args.out_dir).resolve()
    reference_dir = Path(args.reference_evidence).resolve()
    candidate_dir = Path(args.candidate_evidence).resolve()
    root.mkdir(parents=True, exist_ok=True)
    reference_record = read_json(reference_dir / "benchmark_record.json")
    candidate_record = read_json(candidate_dir / "benchmark_record.json")
    reference_summary = read_json(reference_dir / "container_summary.json")
    candidate_summary = read_json(candidate_dir / "container_summary.json")
    reference_objective = _load_objective_from_summary(reference_dir, reference_summary)
    candidate_objective = _load_objective_from_summary(candidate_dir, candidate_summary)
    reference_sol = reference_summary.get("sol_attn") if isinstance(reference_summary.get("sol_attn"), dict) else {}
    candidate_sol = candidate_summary.get("sol_attn") if isinstance(candidate_summary.get("sol_attn"), dict) else {}
    reference_sol_summary = reference_sol.get("summary") if isinstance(reference_sol.get("summary"), dict) else {}
    candidate_sol_summary = candidate_sol.get("summary") if isinstance(candidate_sol.get("summary"), dict) else {}
    reference_warm = float(reference_record["timing"]["warm_e2e"]["seconds"])
    candidate_warm = float(candidate_record["timing"]["warm_e2e"]["seconds"])
    n1_delta_pct = (reference_warm - candidate_warm) / reference_warm * 100.0
    reference_cold = float(reference_record["timing"]["cold_e2e"]["seconds"])
    candidate_cold = float(candidate_record["timing"]["cold_e2e"]["seconds"])
    cold_delta_pct = (reference_cold - candidate_cold) / reference_cold * 100.0
    same_gpu = reference_record["deployment"].get("physical_gpu_uuids") == candidate_record["deployment"].get("physical_gpu_uuids")
    same_workload = reference_record.get("workload_fingerprint") == candidate_record.get("workload_fingerprint")
    same_timing = reference_record["timing"].get("boundary_id") == candidate_record["timing"].get("boundary_id")
    same_generation = reference_record["production"].get("generation_mode") == candidate_record["production"].get("generation_mode") == "extension"
    candidate_flags = list(candidate_objective.get("automatic_red_flags", []) or [])
    reference_flags = list(reference_objective.get("automatic_red_flags", []) or [])
    proxy_not_worse = set(candidate_flags).issubset(set(reference_flags))
    objective_cmp = _objective_noninferiority(reference_record["quality"]["objective"], candidate_record["quality"]["objective"])
    candidate_tau_values = {round(float(x), 6) for x in candidate_sol_summary.get("tau_values", []) or []}
    reference_sparse = _numeric_summary_value(reference_sol_summary, "sparse_calls", 0)
    candidate_sparse = _numeric_summary_value(candidate_sol_summary, "sparse_calls", 0)
    expected_candidate_tau = _expected_tau_from_arg(args.candidate_tau) if args.candidate_tau is not None else (next(iter(candidate_tau_values)) if len(candidate_tau_values) == 1 else None)
    candidate_profile = str(args.candidate_profile or candidate_dir.name)
    reference_profile = str(getattr(args, "reference_profile", None) or reference_dir.name)
    expected_step_min = _candidate_step_min_from_profile(candidate_profile, getattr(args, "candidate_step_min", None))
    expected_reference_step_min = _candidate_step_min_from_profile(reference_profile, getattr(args, "reference_step_min", None))
    expected_reference_tau = _expected_tau_from_arg(getattr(args, "reference_tau", None))
    expected_layer_min, expected_layer_max, expected_layer_scope = _candidate_layer_range_from_profile(
        candidate_profile,
        getattr(args, "candidate_layer_min", None),
        getattr(args, "candidate_layer_max", None),
        getattr(args, "candidate_layer_range_scope", None),
    )
    expected_reference_layer_min, expected_reference_layer_max, expected_reference_layer_scope = _candidate_layer_range_from_profile(
        reference_profile,
        getattr(args, "reference_layer_min", None),
        getattr(args, "reference_layer_max", None),
        getattr(args, "reference_layer_range_scope", None),
    )
    label = _candidate_label_bundle(candidate_profile, expected_step_min)
    lane_ctx = _lane_context_from_records(reference_record, candidate_record)
    strict_no_red_flags = bool(getattr(args, "strict_no_automatic_red_flags", False))
    force_no_promotion = bool(getattr(args, "force_no_promotion", False))
    candidate_tau_gate_name = f"candidate_adaptive_tau{_tau_gate_label(float(expected_candidate_tau))}_diag_seen" if expected_candidate_tau is not None else "candidate_adaptive_tau_diag_seen"
    reference_tau_gate_name = f"reference_adaptive_tau{_tau_gate_label(float(expected_reference_tau))}_diag_seen" if expected_reference_tau is not None else "reference_adaptive_tau_diag_seen"
    reference_profile_gates = _profile_policy_gates(
        role="reference",
        profile=reference_profile,
        summary=reference_sol_summary,
        expected_tau=expected_reference_tau,
        expected_step_min=expected_reference_step_min,
        sparse_calls=reference_sparse,
        tau_gate_name=reference_tau_gate_name,
        expected_layer_min=expected_reference_layer_min,
        expected_layer_max=expected_reference_layer_max,
        expected_layer_range_scope=expected_reference_layer_scope,
    )
    candidate_profile_gates = _profile_policy_gates(
        role="candidate",
        profile=candidate_profile,
        summary=candidate_sol_summary,
        expected_tau=expected_candidate_tau,
        expected_step_min=expected_step_min,
        sparse_calls=candidate_sparse,
        tau_gate_name=candidate_tau_gate_name,
        expected_layer_min=expected_layer_min,
        expected_layer_max=expected_layer_max,
        expected_layer_range_scope=expected_layer_scope,
    )
    gates = {
        "same_physical_gpu": same_gpu,
        "same_workload_fingerprint": same_workload,
        "same_timing_boundary": same_timing,
        "same_extension_generation_mode": same_generation,
        "both_final_av_complete": reference_record["output_av"].get("final_accounting_complete") is True and candidate_record["output_av"].get("final_accounting_complete") is True,
        "reference_sol_attn_telemetry_present": reference_sol.get("status") == "present",
        "candidate_sol_attn_telemetry_present": candidate_sol.get("status") == "present",
        "reference_sparse_stride_calls_valid": _sol_attn_sparse_stride_ok(reference_sol_summary),
        "candidate_sparse_stride_calls_valid": _sol_attn_sparse_stride_ok(candidate_sol_summary),
        "same_sparse_call_count": reference_sparse == candidate_sparse and reference_sparse > 0,
        "reference_zero_fallback_materialization_input_copy": _sol_attn_zero_copy_ok(reference_sol_summary),
        "candidate_zero_fallback_materialization_input_copy": _sol_attn_zero_copy_ok(candidate_sol_summary),
        "reference_no_raw_tensor_export": reference_sol_summary.get("diagnostic_raw_tensor_exported") is False,
        "candidate_no_raw_tensor_export": candidate_sol_summary.get("diagnostic_raw_tensor_exported") is False,
        **reference_profile_gates,
        **candidate_profile_gates,
        "candidate_no_worse_automatic_proxy_flags": proxy_not_worse,
        "objective_5pct_noninferiority_core_metrics": objective_cmp.get("status") == "pass",
        "candidate_warm_e2e_not_slower": candidate_warm <= reference_warm,
    }
    if strict_no_red_flags:
        gates["reference_no_automatic_proxy_flags"] = len(reference_flags) == 0
        gates["candidate_no_automatic_proxy_flags"] = len(candidate_flags) == 0
    failed = [name for name, ok in gates.items() if not ok]
    contract_gate_names = {
        "same_physical_gpu",
        "same_workload_fingerprint",
        "same_timing_boundary",
        "same_extension_generation_mode",
        "both_final_av_complete",
        "reference_sol_attn_telemetry_present",
        "candidate_sol_attn_telemetry_present",
        "reference_sparse_stride_calls_valid",
        "candidate_sparse_stride_calls_valid",
        "same_sparse_call_count",
        "reference_zero_fallback_materialization_input_copy",
        "candidate_zero_fallback_materialization_input_copy",
        "reference_no_raw_tensor_export",
        "candidate_no_raw_tensor_export",
        *reference_profile_gates.keys(),
        *candidate_profile_gates.keys(),
    }
    contract_failed = [name for name in failed if name in contract_gate_names]
    strict_flag_failed = strict_no_red_flags and ("reference_no_automatic_proxy_flags" in failed or "candidate_no_automatic_proxy_flags" in failed)
    if contract_failed:
        status = "reject"
        classification = f"reject_{label['classification_token']}_{lane_ctx['duration_label']}_long_lane_contract_or_telemetry_failed"
        promote = False
    elif strict_flag_failed:
        status = "descriptive"
        classification = f"descriptive_no_promotion_{label['classification_token']}_{lane_ctx['duration_label']}_automatic_proxy_red_flags"
        promote = False
    elif failed:
        status = "reject"
        classification = f"reject_{label['classification_token']}_{lane_ctx['duration_label']}_long_lane_slower_or_proxy_regression"
        promote = False
    else:
        status = "pass"
        if force_no_promotion:
            classification = f"pass_bounded_n1_no_promotion_{label['classification_token']}_{lane_ctx['duration_label']}_parse_clean"
            promote = False
        else:
            classification = f"keep_default_off_{label['classification_token']}_long_lane_n1_pending_reviewer_quality_gate"
            promote = True
    reference_label = _candidate_label_bundle(reference_profile, expected_reference_step_min)
    reference_readable = "r9 Current" if _is_retained_current_profile(reference_profile) else reference_label["candidate_readable"]
    comparison_readable = f"{reference_readable} vs {label['candidate_readable']}"
    reference_schema_token = "r9-current" if _is_retained_current_profile(reference_profile) else reference_label["schema_token"]
    reference_principal = _profile_principal(
        reference_profile,
        expected_tau=expected_reference_tau,
        expected_step_min=expected_reference_step_min,
        expected_layer_min=expected_reference_layer_min,
        expected_layer_max=expected_reference_layer_max,
        expected_layer_range_scope=expected_reference_layer_scope,
    )
    candidate_principal = _profile_principal(
        candidate_profile,
        expected_tau=expected_candidate_tau,
        expected_step_min=expected_step_min,
        expected_layer_min=expected_layer_min,
        expected_layer_max=expected_layer_max,
        expected_layer_range_scope=expected_layer_scope,
    )
    same_gpu_proof = {
        "lane_id": lane_ctx["lane_id"],
        "duration_label": lane_ctx["duration_label"],
        "final_frames": lane_ctx["final_frames"],
        "effective_audio_samples_per_channel": lane_ctx["effective_audio_samples_per_channel"],
        "chunk_count": lane_ctx["chunk_count"],
        "reference_physical_gpu_uuids": reference_record["deployment"].get("physical_gpu_uuids"),
        "candidate_physical_gpu_uuids": candidate_record["deployment"].get("physical_gpu_uuids"),
        "reference_workload_fingerprint": reference_record.get("workload_fingerprint"),
        "candidate_workload_fingerprint": candidate_record.get("workload_fingerprint"),
        "reference_timing_boundary": reference_record["timing"].get("boundary_id"),
        "candidate_timing_boundary": candidate_record["timing"].get("boundary_id"),
    }
    decision = {
        "schema_version": f"minimax-h3-final-av-{lane_ctx['duration_label']}-{reference_schema_token}-vs-{label['schema_token']}-sol-attn-n1-decision-v1",
        "created_utc": utc_now(),
        "status": status,
        "classification": classification,
        "reference_evidence": str(reference_dir),
        "candidate_evidence": str(candidate_dir),
        "principal_variable": f"{reference_principal} versus {candidate_principal}",
        "reference_profile": reference_profile,
        "reference_expected_tau": expected_reference_tau,
        "reference_expected_adaptive_step_min": expected_reference_step_min,
        "reference_expected_adaptive_layer_min": expected_reference_layer_min,
        "reference_expected_adaptive_layer_max": expected_reference_layer_max,
        "reference_expected_adaptive_layer_range_scope": expected_reference_layer_scope,
        "candidate_profile": candidate_profile,
        "candidate_expected_tau": expected_candidate_tau,
        "candidate_expected_adaptive_step_min": expected_step_min,
        "candidate_expected_adaptive_layer_min": expected_layer_min,
        "candidate_expected_adaptive_layer_max": expected_layer_max,
        "candidate_expected_adaptive_layer_range_scope": expected_layer_scope,
        "generation_mode": "extension",
        "native_context_supported": False,
        "sample_count": 1,
        "speedup_claimed": False,
        "n1_route_gate_language_only": True,
        "promote_to_n3_recommended_pending_reviewer": promote,
        "no_promotion_reason": f"bounded_{lane_ctx['duration_label']}_n1_scope_or_failed_gate" if not promote else None,
        "reviewer_acceptance_status": "pending_host_reviewer_not_invoked_by_engineer",
        "timing": {
            "reference_warm_e2e_seconds": reference_warm,
            "candidate_warm_e2e_seconds": candidate_warm,
            "warm_e2e_delta_percent_n1_not_speedup": n1_delta_pct,
            "reference_cold_e2e_seconds": reference_cold,
            "candidate_cold_e2e_seconds": candidate_cold,
            "cold_e2e_delta_percent_n1_not_speedup": cold_delta_pct,
        },
        "sol_attn_telemetry_summary": {"reference": reference_sol_summary, "candidate": candidate_sol_summary},
        "resources": {"reference": reference_record.get("resources"), "candidate": candidate_record.get("resources")},
        "objective_proxy_comparison": objective_cmp,
        "automatic_red_flags": {"reference": reference_flags, "candidate": candidate_flags},
        "same_gpu_workload_timing_boundary_proof": same_gpu_proof,
        "gates": gates,
        "failed_gates": failed,
        "claim_boundary": f"Matched N=1 {lane_ctx['duration_readable']} final-AV extension-lane {candidate_profile} versus retained {reference_profile} route/config gate only. Output is extension/chunked, not native long context; Turbo 8-step is practical/disclosed approximate, not BF16 fidelity; no formal speedup, human-quality, public-comparison, or SOTA claim.",
    }
    write_json(root / "decision.json", decision)
    reviewer_request = {
        "schema_version": f"minimax-h3-final-av-{lane_ctx['duration_label']}-{reference_schema_token}-vs-{label['schema_token']}-n1-reviewer-request-v1",
        "created_utc": utc_now(),
        "evidence_root": str(root),
        "requested_review": "artifact-only audit; do not rerun GPU/model/Docker",
        "engineer_classification": classification,
        "required_scope": f"N=1 matched {lane_ctx['duration_readable']} extension-lane pass/reject/descriptive classification only; no formal speedup/human-quality/native-context/BF16/SOTA claim",
        "decision": str(root / "decision.json"),
        "run_report": str(root / "RUN_REPORT.md"),
    }
    write_json(root / "reviewer_verdict_request.json", reviewer_request)
    lines = [
        f"# {comparison_readable} Sol-Attn Final-AV {lane_ctx['duration_label']} N=1 RUN_REPORT",
        "",
        f"- Status: `{status}`",
        f"- Classification: `{classification}`",
        f"- Reference evidence: `{reference_dir}`",
        f"- Candidate evidence: `{candidate_dir}`",
        f"- Reference profile: `{reference_profile}` (expected tau={expected_reference_tau}, diag, adaptive_step_min={expected_reference_step_min}, layer_range={expected_reference_layer_min}-{expected_reference_layer_max}, layer_scope={expected_reference_layer_scope})",
        f"- Candidate profile: `{candidate_profile}` (expected tau={expected_candidate_tau}, diag, adaptive_step_min={expected_step_min}, layer_range={expected_layer_min}-{expected_layer_max}, layer_scope={expected_layer_scope})",
        f"- Warm E2E reference/candidate: {reference_warm:.3f}s / {candidate_warm:.3f}s",
        f"- N=1 warm delta (route-gate only, not speedup): {n1_delta_pct:.3f}%",
        f"- Reference sparse/fallback/materialization/input-copy: sparse={reference_sparse}, fallback={reference_sol_summary.get('fallback_calls')}, materialize={reference_sol_summary.get('materialize_copy_count')}/{reference_sol_summary.get('materialize_copy_bytes')} bytes, input_copy={reference_sol_summary.get('input_copy_events')}/{reference_sol_summary.get('input_copy_bytes')} bytes",
        f"- Candidate sparse/fallback/materialization/input-copy: sparse={candidate_sparse}, fallback={candidate_sol_summary.get('fallback_calls')}, materialize={candidate_sol_summary.get('materialize_copy_count')}/{candidate_sol_summary.get('materialize_copy_bytes')} bytes, input_copy={candidate_sol_summary.get('input_copy_events')}/{candidate_sol_summary.get('input_copy_bytes')} bytes",
        f"- Automatic proxy flags reference/candidate: {reference_flags} / {candidate_flags}",
        f"- Failed gates: {failed}",
        f"- Same GPU/workload/timing-boundary proof: {same_gpu_proof}",
        f"- Promotion: promote_to_n3_recommended_pending_reviewer={promote}; force_no_promotion={force_no_promotion}; strict_no_automatic_red_flags={strict_no_red_flags}",
        f"- Claim boundary: extension output, not native long context; practical Turbo 8-step approximation; N=1 only for `{candidate_profile}` vs retained `{reference_profile}`; no formal speedup or human-quality claim.",
        "- Reviewer acceptance: `pending_host_reviewer_not_invoked_by_engineer`; request written to `reviewer_verdict_request.json`.",
    ]
    (root / "RUN_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    packet_lines = [
        f"# Artifact-only Reviewer Packet: {label['candidate_readable']} {lane_ctx['duration_label']} N=1",
        "",
        f"Evidence root: `{root}`",
        f"Engineer status/classification: `{status}` / `{classification}`",
        "",
        "Review only the root decision/report, per-lane benchmark records and validation outputs, final AV accounting/objective metrics, Sol-Attn telemetry, isolated Docker/runtime provenance, and terminal hygiene artifacts. Do not rerun GPU/model/Docker for review.",
        "",
        "Non-acceptances requested: no formal speedup, no BF16 fidelity, no native long context, no human semantic/audio quality, no product readiness, no public comparison/SOTA.",
    ]
    (root / "reviewer_packet.md").write_text("\n".join(packet_lines) + "\n", encoding="utf-8")
    return 0


def _stats(values: list[float]) -> dict[str, Any]:
    finite = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not finite:
        return {"n": 0, "status": "missing"}
    out: dict[str, Any] = {
        "n": len(finite),
        "status": "measured",
        "values": finite,
        "min": min(finite),
        "max": max(finite),
        "mean": statistics.mean(finite),
        "median": statistics.median(finite),
    }
    if len(finite) > 1:
        out["stdev"] = statistics.stdev(finite)
        out["cv_percent"] = (out["stdev"] / out["mean"] * 100.0) if abs(out["mean"]) > 1e-12 else None
    else:
        out["stdev"] = 0.0
        out["cv_percent"] = 0.0
    return out


def _metric_value(record: dict[str, Any], metric: str) -> float | None:
    item = record.get("resources", {}).get(metric, {}) if isinstance(record.get("resources"), dict) else {}
    value = item.get("value") if isinstance(item, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def _validation_status(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        data = read_json(path)
    except Exception:
        return "unparseable"
    status = data.get("status")
    return str(status) if status is not None else None


def _compact_sol_attn_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "sparse_calls",
        "sparse_candidate_calls",
        "dense_calls",
        "density_sample_count",
        "fallback_calls",
        "fallback_reasons",
        "materialize_copy_count",
        "materialize_copy_bytes",
        "input_copy_events",
        "input_copy_bytes",
        "stride_aware_value_calls",
        "diagnostic_raw_tensor_exported",
        "tau_values",
        "tau_counts",
        "thresh_type_values",
        "adaptive_routing_values",
        "adaptive_profiles",
        "adaptive_candidate_tau_values",
        "adaptive_step_min_values",
        "adaptive_step_max_values",
        "adaptive_layer_min_values",
        "adaptive_layer_max_values",
        "adaptive_layer_range_scope_values",
        "layer_index_values",
        "adaptive_guard_requested_count",
        "adaptive_guard_active_count",
        "adaptive_guard_inactive_count",
        "adaptive_guard_reason_counts",
        "adaptive_guard_counts_by_step",
        "denoise_gpu_latency_ms",
        "denoise_timed_calls",
        "sparse_attention_gpu_latency_ms",
        "sparse_attention_timed_calls",
    ]
    return {key: summary.get(key) for key in keys if key in summary}


def _r9_pair_analysis(
    pair_dir: Path,
    pair_name: str,
    candidate_profile: str,
    expected_tau: float,
    expected_step_min: int | None = None,
    *,
    reference_profile: str = "r9_current_sol_attn",
    expected_reference_tau: float | None = None,
    expected_reference_step_min: int | None = None,
    expected_layer_min: int | None = None,
    expected_layer_max: int | None = None,
    expected_layer_range_scope: str | None = None,
    expected_reference_layer_min: int | None = None,
    expected_reference_layer_max: int | None = None,
    expected_reference_layer_range_scope: str | None = None,
) -> dict[str, Any]:
    reference_dir = pair_dir / reference_profile
    candidate_dir = pair_dir / candidate_profile
    out: dict[str, Any] = {
        "pair": pair_name,
        "pair_dir": str(pair_dir),
        "reference_evidence": str(reference_dir),
        "candidate_evidence": str(candidate_dir),
        "complete": False,
        "failed_gates": [],
        "gates": {},
    }
    required = {
        "reference_record": reference_dir / "benchmark_record.json",
        "candidate_record": candidate_dir / "benchmark_record.json",
        "reference_summary": reference_dir / "container_summary.json",
        "candidate_summary": candidate_dir / "container_summary.json",
        "pair_decision": pair_dir / "decision.json",
        "reference_validation": pair_dir / "reference_benchmark_record_validation.json",
        "candidate_validation": pair_dir / "candidate_benchmark_record_validation.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    out["required_artifacts"] = {name: str(path) for name, path in required.items()}
    out["missing_artifacts"] = missing
    if missing:
        out["failed_gates"] = ["pair_required_artifacts_complete"]
        out["gates"] = {"pair_required_artifacts_complete": False}
        return out
    try:
        reference_record = read_json(required["reference_record"])
        candidate_record = read_json(required["candidate_record"])
        reference_summary = read_json(required["reference_summary"])
        candidate_summary = read_json(required["candidate_summary"])
        pair_decision = read_json(required["pair_decision"])
    except Exception as exc:  # noqa: BLE001
        out["failed_gates"] = ["pair_artifacts_parse"]
        out["gates"] = {"pair_artifacts_parse": False}
        out["parse_error"] = f"{type(exc).__name__}: {exc}"
        return out
    reference_objective = _load_objective_from_summary(reference_dir, reference_summary)
    candidate_objective = _load_objective_from_summary(candidate_dir, candidate_summary)
    reference_sol = reference_summary.get("sol_attn") if isinstance(reference_summary.get("sol_attn"), dict) else {}
    candidate_sol = candidate_summary.get("sol_attn") if isinstance(candidate_summary.get("sol_attn"), dict) else {}
    reference_sol_summary = reference_sol.get("summary") if isinstance(reference_sol.get("summary"), dict) else {}
    candidate_sol_summary = candidate_sol.get("summary") if isinstance(candidate_sol.get("summary"), dict) else {}
    reference_warm = float(reference_record["timing"]["warm_e2e"]["seconds"])
    candidate_warm = float(candidate_record["timing"]["warm_e2e"]["seconds"])
    reference_cold = float(reference_record["timing"]["cold_e2e"]["seconds"])
    candidate_cold = float(candidate_record["timing"]["cold_e2e"]["seconds"])
    warm_delta_pct = (reference_warm - candidate_warm) / reference_warm * 100.0
    cold_delta_pct = (reference_cold - candidate_cold) / reference_cold * 100.0
    reference_flags = list(reference_objective.get("automatic_red_flags", []) or [])
    candidate_flags = list(candidate_objective.get("automatic_red_flags", []) or [])
    objective_cmp = _objective_noninferiority(reference_record["quality"]["objective"], candidate_record["quality"]["objective"])
    reference_sparse = _numeric_summary_value(reference_sol_summary, "sparse_calls", 0)
    candidate_sparse = _numeric_summary_value(candidate_sol_summary, "sparse_calls", 0)
    expected_tau_rounded = round(float(expected_tau), 6)
    expected_step_min = _candidate_step_min_from_profile(candidate_profile, expected_step_min)
    expected_reference_step_min = _candidate_step_min_from_profile(reference_profile, expected_reference_step_min)
    expected_reference_tau = _expected_tau_from_arg(expected_reference_tau)
    expected_layer_min, expected_layer_max, expected_layer_range_scope = _candidate_layer_range_from_profile(
        candidate_profile,
        expected_layer_min,
        expected_layer_max,
        expected_layer_range_scope,
    )
    expected_reference_layer_min, expected_reference_layer_max, expected_reference_layer_range_scope = _candidate_layer_range_from_profile(
        reference_profile,
        expected_reference_layer_min,
        expected_reference_layer_max,
        expected_reference_layer_range_scope,
    )
    reference_profile_gates = _profile_policy_gates(
        role="reference",
        profile=reference_profile,
        summary=reference_sol_summary,
        expected_tau=expected_reference_tau,
        expected_step_min=expected_reference_step_min,
        sparse_calls=reference_sparse,
        expected_layer_min=expected_reference_layer_min,
        expected_layer_max=expected_reference_layer_max,
        expected_layer_range_scope=expected_reference_layer_range_scope,
    )
    candidate_profile_gates = _profile_policy_gates(
        role="candidate",
        profile=candidate_profile,
        summary=candidate_sol_summary,
        expected_tau=expected_tau_rounded,
        expected_step_min=expected_step_min,
        sparse_calls=candidate_sparse,
        expected_layer_min=expected_layer_min,
        expected_layer_max=expected_layer_max,
        expected_layer_range_scope=expected_layer_range_scope,
    )
    gates = {
        "pair_decision_pass_no_failed_gates": pair_decision.get("status") == "pass" and not pair_decision.get("failed_gates"),
        "reference_benchmark_validation_pass": _validation_status(required["reference_validation"]) == "pass",
        "candidate_benchmark_validation_pass": _validation_status(required["candidate_validation"]) == "pass",
        "same_physical_gpu": reference_record["deployment"].get("physical_gpu_uuids") == candidate_record["deployment"].get("physical_gpu_uuids"),
        "same_workload_fingerprint": reference_record.get("workload_fingerprint") == candidate_record.get("workload_fingerprint"),
        "same_timing_boundary": reference_record["timing"].get("boundary_id") == candidate_record["timing"].get("boundary_id"),
        "same_extension_generation_mode": reference_record["production"].get("generation_mode") == candidate_record["production"].get("generation_mode") == "extension",
        "both_final_av_complete": reference_record["output_av"].get("final_accounting_complete") is True and candidate_record["output_av"].get("final_accounting_complete") is True,
        "reference_sol_attn_telemetry_present": reference_sol.get("status") == "present",
        "candidate_sol_attn_telemetry_present": candidate_sol.get("status") == "present",
        "reference_sparse_stride_calls_valid": _sol_attn_sparse_stride_ok(reference_sol_summary),
        "candidate_sparse_stride_calls_valid": _sol_attn_sparse_stride_ok(candidate_sol_summary),
        "same_sparse_call_count": reference_sparse == candidate_sparse and reference_sparse > 0,
        "reference_zero_fallback_materialization_input_copy": _sol_attn_zero_copy_ok(reference_sol_summary),
        "candidate_zero_fallback_materialization_input_copy": _sol_attn_zero_copy_ok(candidate_sol_summary),
        "reference_no_raw_tensor_export": reference_sol_summary.get("diagnostic_raw_tensor_exported") is False,
        "candidate_no_raw_tensor_export": candidate_sol_summary.get("diagnostic_raw_tensor_exported") is False,
        **reference_profile_gates,
        **candidate_profile_gates,
        "candidate_no_worse_automatic_proxy_flags": set(candidate_flags).issubset(set(reference_flags)),
        "objective_5pct_noninferiority_core_metrics": objective_cmp.get("status") == "pass",
        "candidate_warm_e2e_not_slower": candidate_warm <= reference_warm,
    }
    failed = [name for name, ok in gates.items() if not ok]
    out.update({
        "complete": True,
        "gates": gates,
        "failed_gates": failed,
        "pair_decision_status": pair_decision.get("status"),
        "pair_decision_classification": pair_decision.get("classification"),
        "workload_fingerprint": reference_record.get("workload_fingerprint"),
        "timing_boundary": reference_record.get("timing", {}).get("boundary_id"),
        "physical_gpu_uuids": reference_record.get("deployment", {}).get("physical_gpu_uuids"),
        "timing": {
            "reference_warm_e2e_seconds": reference_warm,
            "candidate_warm_e2e_seconds": candidate_warm,
            "warm_e2e_delta_percent_n3_gate_not_speedup": warm_delta_pct,
            "reference_cold_e2e_seconds": reference_cold,
            "candidate_cold_e2e_seconds": candidate_cold,
            "cold_e2e_delta_percent_n3_gate_not_speedup": cold_delta_pct,
            "reference_seconds_per_generated_second": reference_record.get("timing", {}).get("seconds_per_generated_second", {}).get("seconds"),
            "candidate_seconds_per_generated_second": candidate_record.get("timing", {}).get("seconds_per_generated_second", {}).get("seconds"),
        },
        "resources": {
            "reference": reference_record.get("resources"),
            "candidate": candidate_record.get("resources"),
        },
        "container_resources": {
            "reference": reference_summary.get("resources"),
            "candidate": candidate_summary.get("resources"),
        },
        "component_timing": {
            "reference": reference_record.get("timing", {}).get("components", {}),
            "candidate": candidate_record.get("timing", {}).get("components", {}),
        },
        "sol_attn_telemetry_summary": {
            "reference": _compact_sol_attn_summary(reference_sol_summary),
            "candidate": _compact_sol_attn_summary(candidate_sol_summary),
        },
        "reference_profile": reference_profile,
        "reference_expected_tau": expected_reference_tau,
        "reference_expected_adaptive_step_min": expected_reference_step_min,
        "reference_expected_adaptive_layer_min": expected_reference_layer_min,
        "reference_expected_adaptive_layer_max": expected_reference_layer_max,
        "reference_expected_adaptive_layer_range_scope": expected_reference_layer_range_scope,
        "candidate_expected_adaptive_step_min": expected_step_min,
        "candidate_expected_adaptive_layer_min": expected_layer_min,
        "candidate_expected_adaptive_layer_max": expected_layer_max,
        "candidate_expected_adaptive_layer_range_scope": expected_layer_range_scope,
        "telemetry_paths": {
            "reference": str(reference_dir / "sol_attn_telemetry.sol_attn.json"),
            "candidate": str(candidate_dir / "sol_attn_telemetry.sol_attn.json"),
        },
        "objective_proxy_comparison": objective_cmp,
        "automatic_red_flags": {"reference": reference_flags, "candidate": candidate_flags},
        "final_av_accounting": {
            "reference": reference_record.get("output_av"),
            "candidate": candidate_record.get("output_av"),
        },
        "benchmark_record_paths": {
            "reference": str(required["reference_record"]),
            "candidate": str(required["candidate_record"]),
        },
        "benchmark_validation_paths": {
            "reference": str(required["reference_validation"]),
            "candidate": str(required["candidate_validation"]),
        },
    })
    return out


def finalize_r9_adaptive_matched_n3(args: argparse.Namespace) -> int:
    root = Path(args.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidate_profile = str(args.candidate_profile)
    reference_profile = str(getattr(args, "reference_profile", None) or "r9_current_sol_attn")
    expected_tau = float(args.candidate_tau)
    expected_reference_tau = _expected_tau_from_arg(getattr(args, "reference_tau", None))
    expected_step_min = _candidate_step_min_from_profile(candidate_profile, getattr(args, "candidate_step_min", None))
    expected_reference_step_min = _candidate_step_min_from_profile(reference_profile, getattr(args, "reference_step_min", None))
    expected_layer_min, expected_layer_max, expected_layer_scope = _candidate_layer_range_from_profile(
        candidate_profile,
        getattr(args, "candidate_layer_min", None),
        getattr(args, "candidate_layer_max", None),
        getattr(args, "candidate_layer_range_scope", None),
    )
    expected_reference_layer_min, expected_reference_layer_max, expected_reference_layer_scope = _candidate_layer_range_from_profile(
        reference_profile,
        getattr(args, "reference_layer_min", None),
        getattr(args, "reference_layer_max", None),
        getattr(args, "reference_layer_range_scope", None),
    )
    label = _candidate_label_bundle(candidate_profile, expected_step_min)
    reference_label = _candidate_label_bundle(reference_profile, expected_reference_step_min)
    reference_readable = "r9 Current" if _is_retained_current_profile(reference_profile) else reference_label["candidate_readable"]
    comparison_readable = f"{reference_readable} vs {label['candidate_readable']}"
    reference_schema_token = "r9-current" if _is_retained_current_profile(reference_profile) else reference_label["schema_token"]
    reference_principal = _profile_principal(
        reference_profile,
        expected_tau=expected_reference_tau,
        expected_step_min=expected_reference_step_min,
        expected_layer_min=expected_reference_layer_min,
        expected_layer_max=expected_reference_layer_max,
        expected_layer_range_scope=expected_reference_layer_scope,
    )
    candidate_principal = _profile_principal(
        candidate_profile,
        expected_tau=expected_tau,
        expected_step_min=expected_step_min,
        expected_layer_min=expected_layer_min,
        expected_layer_max=expected_layer_max,
        expected_layer_range_scope=expected_layer_scope,
    )
    requested_pairs = int(args.pairs)
    pairs = [
        _r9_pair_analysis(
            root / f"pair{i:02d}",
            f"pair{i:02d}",
            candidate_profile,
            expected_tau,
            expected_step_min,
            reference_profile=reference_profile,
            expected_reference_tau=expected_reference_tau,
            expected_reference_step_min=expected_reference_step_min,
            expected_layer_min=expected_layer_min,
            expected_layer_max=expected_layer_max,
            expected_layer_range_scope=expected_layer_scope,
            expected_reference_layer_min=expected_reference_layer_min,
            expected_reference_layer_max=expected_reference_layer_max,
            expected_reference_layer_range_scope=expected_reference_layer_scope,
        )
        for i in range(1, requested_pairs + 1)
    ]
    completed_pairs = [p for p in pairs if p.get("complete")]
    pair_gate_failures = {p["pair"]: p.get("failed_gates", []) for p in pairs if p.get("failed_gates")}
    warm_deltas = [float(p["timing"]["warm_e2e_delta_percent_n3_gate_not_speedup"]) for p in completed_pairs if "timing" in p]
    cold_deltas = [float(p["timing"]["cold_e2e_delta_percent_n3_gate_not_speedup"]) for p in completed_pairs if "timing" in p]
    reference_warm = [float(p["timing"]["reference_warm_e2e_seconds"]) for p in completed_pairs if "timing" in p]
    candidate_warm = [float(p["timing"]["candidate_warm_e2e_seconds"]) for p in completed_pairs if "timing" in p]
    reference_cold = [float(p["timing"]["reference_cold_e2e_seconds"]) for p in completed_pairs if "timing" in p]
    candidate_cold = [float(p["timing"]["candidate_cold_e2e_seconds"]) for p in completed_pairs if "timing" in p]
    all_gpu_sets = [tuple(p.get("physical_gpu_uuids") or []) for p in completed_pairs]
    all_workloads = [p.get("workload_fingerprint") for p in completed_pairs]
    all_boundaries = [p.get("timing_boundary") for p in completed_pairs]
    median_delta = statistics.median(warm_deltas) if warm_deltas else None
    min_pair_delta = min(warm_deltas) if warm_deltas else None
    timing_summary = {
        "requested_pairs": requested_pairs,
        "completed_pairs": len(completed_pairs),
        "reference_warm_e2e_seconds": _stats(reference_warm),
        "candidate_warm_e2e_seconds": _stats(candidate_warm),
        "warm_e2e_delta_percent_n3_gate_not_speedup": _stats(warm_deltas),
        "reference_cold_e2e_seconds": _stats(reference_cold),
        "candidate_cold_e2e_seconds": _stats(candidate_cold),
        "cold_e2e_delta_percent_n3_gate_not_speedup": _stats(cold_deltas),
        "min_median_delta_threshold_percent": float(args.min_median_delta_pct),
        "max_slower_pair_tolerance_percent": float(args.max_slower_pair_pct),
    }
    resource_summary: dict[str, Any] = {"reference": {}, "candidate": {}}
    for lane in ("reference", "candidate"):
        for metric in ("peak_gpu_memory_mib", "peak_host_memory_gib", "peak_power_w", "failures"):
            values = []
            for p in completed_pairs:
                data = p.get("resources", {}).get(lane, {}) if isinstance(p.get("resources"), dict) else {}
                item = data.get(metric, {}) if isinstance(data, dict) else {}
                value = item.get("value") if isinstance(item, dict) else None
                if isinstance(value, (int, float)):
                    values.append(float(value))
            resource_summary[lane][metric] = _stats(values)
    quality_proxy_summary = {
        "pairs": [
            {
                "pair": p.get("pair"),
                "objective_proxy_comparison": p.get("objective_proxy_comparison"),
                "automatic_red_flags": p.get("automatic_red_flags"),
                "candidate_no_worse_automatic_proxy_flags": p.get("gates", {}).get("candidate_no_worse_automatic_proxy_flags"),
                "objective_5pct_noninferiority_core_metrics": p.get("gates", {}).get("objective_5pct_noninferiority_core_metrics"),
            }
            for p in pairs
        ],
        "scope": "No-reference objective proxies only; no semantic, human visual/audio, prompt-faithfulness, or perceived AV-sync certification.",
    }
    telemetry_summary = {
        "pairs": [
            {
                "pair": p.get("pair"),
                "telemetry_paths": p.get("telemetry_paths"),
                "sol_attn_telemetry_summary": p.get("sol_attn_telemetry_summary"),
            }
            for p in pairs
        ]
    }
    gates = {
        "all_pairs_completed": len(completed_pairs) == requested_pairs,
        "all_pair_gates_pass": not pair_gate_failures,
        "same_physical_gpu_all_pairs": bool(all_gpu_sets) and len(set(all_gpu_sets)) == 1 and all(len(x) == 1 for x in all_gpu_sets),
        "same_workload_fingerprint_all_pairs": bool(all_workloads) and len(set(all_workloads)) == 1,
        "same_timing_boundary_all_pairs": bool(all_boundaries) and len(set(all_boundaries)) == 1,
        "median_warm_e2e_delta_meets_threshold": median_delta is not None and median_delta >= float(args.min_median_delta_pct),
        "no_pair_slower_beyond_tolerance": min_pair_delta is not None and min_pair_delta >= -float(args.max_slower_pair_pct),
        "all_objective_proxy_core_noninferiority_pass": all(p.get("gates", {}).get("objective_5pct_noninferiority_core_metrics") is True for p in completed_pairs) and len(completed_pairs) == requested_pairs,
        "all_candidate_proxy_flags_no_worse": all(p.get("gates", {}).get("candidate_no_worse_automatic_proxy_flags") is True for p in completed_pairs) and len(completed_pairs) == requested_pairs,
        "all_final_av_accounting_complete": all(p.get("gates", {}).get("both_final_av_complete") is True for p in completed_pairs) and len(completed_pairs) == requested_pairs,
        "all_sol_attn_zero_fallback_materialization_input_copy": all(
            p.get("gates", {}).get("reference_zero_fallback_materialization_input_copy") is True
            and p.get("gates", {}).get("candidate_zero_fallback_materialization_input_copy") is True
            for p in completed_pairs
        ) and len(completed_pairs) == requested_pairs,
        "all_sol_attn_sparse_stride_calls_valid": all(
            p.get("gates", {}).get("reference_sparse_stride_calls_valid") is True
            and p.get("gates", {}).get("candidate_sparse_stride_calls_valid") is True
            and p.get("gates", {}).get("same_sparse_call_count") is True
            for p in completed_pairs
        ) and len(completed_pairs) == requested_pairs,
        "all_candidate_guarded_adaptive_exercised": all(p.get("gates", {}).get("candidate_guarded_profile_exercised") is True for p in completed_pairs) and len(completed_pairs) == requested_pairs,
        "all_candidate_adaptive_step_min_expected_seen": all(p.get("gates", {}).get("candidate_adaptive_step_min_expected_seen") is True for p in completed_pairs) and len(completed_pairs) == requested_pairs,
        "all_candidate_adaptive_layer_range_expected_seen": all(p.get("gates", {}).get("candidate_adaptive_layer_range_expected_seen") is True for p in completed_pairs) and len(completed_pairs) == requested_pairs,
        "all_candidate_guard_counts_match_step_min": all(p.get("gates", {}).get("candidate_guard_counts_match_step_min") is True for p in completed_pairs) and len(completed_pairs) == requested_pairs,
        "all_no_raw_tensor_export": all(
            p.get("gates", {}).get("reference_no_raw_tensor_export") is True
            and p.get("gates", {}).get("candidate_no_raw_tensor_export") is True
            for p in completed_pairs
        ) and len(completed_pairs) == requested_pairs,
    }
    failed = [name for name, ok in gates.items() if not ok]
    contract_like = {
        "all_pairs_completed",
        "all_pair_gates_pass",
        "same_physical_gpu_all_pairs",
        "same_workload_fingerprint_all_pairs",
        "same_timing_boundary_all_pairs",
        "all_final_av_accounting_complete",
        "all_sol_attn_zero_fallback_materialization_input_copy",
        "all_sol_attn_sparse_stride_calls_valid",
        "all_candidate_guarded_adaptive_exercised",
        "all_candidate_adaptive_step_min_expected_seen",
        "all_candidate_adaptive_layer_range_expected_seen",
        "all_candidate_guard_counts_match_step_min",
        "all_no_raw_tensor_export",
    }
    if any(name in failed for name in contract_like):
        status = "reject"
        classification = f"reject_no_promotion_{label['classification_token']}_n3_contract_telemetry_or_media_failed"
        promote = False
    elif failed:
        status = "reject"
        classification = f"reject_no_promotion_{label['classification_token']}_n3_timing_or_proxy_failed"
        promote = False
    else:
        status = "pass"
        classification = f"recommend_followon_formal_validation_default_off_{label['classification_token']}_n3_pass"
        promote = True
    decision = {
        "schema_version": f"minimax-h3-final-av-30s-{reference_schema_token}-vs-{label['schema_token']}-sol-attn-n3-decision-v1",
        "created_utc": utc_now(),
        "status": status,
        "classification": classification,
        "requested_pairs": requested_pairs,
        "completed_pairs": len(completed_pairs),
        "sample_count": len(completed_pairs),
        "principal_variable": f"{reference_principal} versus {candidate_principal}",
        "reference_profile": reference_profile,
        "reference_expected_tau": expected_reference_tau,
        "reference_expected_adaptive_step_min": expected_reference_step_min,
        "reference_expected_adaptive_layer_min": expected_reference_layer_min,
        "reference_expected_adaptive_layer_max": expected_reference_layer_max,
        "reference_expected_adaptive_layer_range_scope": expected_reference_layer_scope,
        "candidate_profile": candidate_profile,
        "candidate_expected_tau": expected_tau,
        "candidate_expected_adaptive_step_min": expected_step_min,
        "candidate_expected_adaptive_layer_min": expected_layer_min,
        "candidate_expected_adaptive_layer_max": expected_layer_max,
        "candidate_expected_adaptive_layer_range_scope": expected_layer_scope,
        "generation_mode": "extension",
        "native_context_supported": False,
        "speedup_claimed": False,
        "n3_route_gate_language_only": True,
        "formal_validation_recommended_pending_reviewer": promote,
        "reviewer_acceptance_status": "pending_host_reviewer_not_invoked_by_engineer",
        "timing_summary": timing_summary,
        "resource_summary": resource_summary,
        "quality_proxy_summary": quality_proxy_summary,
        "telemetry_summary": telemetry_summary,
        "pairs": pairs,
        "gates": gates,
        "failed_gates": failed,
        "pair_gate_failures": pair_gate_failures,
        "claim_exclusions": {
            "formal_speedup": True,
            "bf16_fidelity": True,
            "native_long_context": True,
            "human_semantic_or_audio_quality": True,
            "product_readiness": True,
            "public_comparison_or_sota": True,
        },
        "claim_boundary": f"Matched N=3 30-second final-AV extension-lane {candidate_profile} versus retained {reference_profile} route/config gate only. Output is six-chunk extension/chunked, not native long context; Turbo 8-step is practical/disclosed approximate, not BF16 fidelity; no formal speedup, human-quality, product-readiness, public-comparison, or SOTA claim.",
    }
    write_json(root / "timing_summary.json", timing_summary)
    write_json(root / "resource_summary.json", resource_summary)
    write_json(root / "quality_proxy_comparison.json", quality_proxy_summary)
    write_json(root / "sol_attn_telemetry_summary.json", telemetry_summary)
    write_json(root / "decision.json", decision)
    reviewer_request = {
        "schema_version": f"minimax-h3-final-av-30s-{reference_schema_token}-vs-{label['schema_token']}-n3-reviewer-request-v1",
        "created_utc": utc_now(),
        "evidence_root": str(root),
        "requested_review": "artifact-only audit; do not rerun GPU/model/Docker",
        "engineer_classification": classification,
        "required_scope": "N=3 route-gate acceptance or rejection only; no formal speedup/human-quality/native-context/BF16/SOTA claim",
        "decision": str(root / "decision.json"),
        "run_report": str(root / "RUN_REPORT.md"),
    }
    write_json(root / "reviewer_verdict_request.json", reviewer_request)
    report_lines = [
        f"# {comparison_readable} Sol-Attn Final-AV 30s N=3 RUN_REPORT",
        "",
        f"- Status: `{status}`",
        f"- Classification: `{classification}`",
        f"- Requested/completed pairs: {requested_pairs}/{len(completed_pairs)}",
        f"- Reference profile: `{reference_profile}` (tau={expected_reference_tau}, diag, adaptive_step_min={expected_reference_step_min}, layer_range={expected_reference_layer_min}-{expected_reference_layer_max}, layer_scope={expected_reference_layer_scope})",
        f"- Candidate profile: `{candidate_profile}` (tau={expected_tau:g}, diag, adaptive_step_min={expected_step_min}, layer_range={expected_layer_min}-{expected_layer_max}, layer_scope={expected_layer_scope})",
        f"- Median warm E2E delta (N=3 route gate, not speedup): {median_delta:.3f}%" if median_delta is not None else "- Median warm E2E delta: unavailable",
        f"- Warm delta distribution (%): {warm_deltas}",
        f"- Timing threshold: median >= {float(args.min_median_delta_pct):.3f}% and no pair slower beyond {float(args.max_slower_pair_pct):.3f}%.",
        f"- Failed gates: {failed}",
        "- Sol-Attn gate: each completed lane must have sparse calls, stride-aware V, zero fallback/materialization/input-copy, no raw tensor export, and candidate guarded-active plus guarded-inactive calls.",
        "- Media/proxy gate: each completed lane must have complete final-AV accounting; objective proxies are no-reference only and do not certify human semantic/audio quality.",
        f"- Claim boundary: extension/chunked output, practical Turbo 8-step approximation, N=3 gate only for `{candidate_profile}` vs retained `{reference_profile}`; no formal speedup, BF16 fidelity, native long context, product-readiness, public comparison, SOTA, or human-quality claim.",
        "- Reviewer acceptance: `pending_host_reviewer_not_invoked_by_engineer`; request written to `reviewer_verdict_request.json`.",
    ]
    (root / "RUN_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    packet_lines = [
        f"# Artifact-only Reviewer Packet: {label['candidate_readable']} N=3",
        "",
        f"Evidence root: `{root}`",
        f"Engineer status/classification: `{status}` / `{classification}`",
        "",
        "Review only the root decision/report, per-pair lane records and validation outputs, final AV accounting, Sol-Attn telemetry summaries/full telemetry files, isolated Docker/runtime provenance, and terminal hygiene artifacts. Do not rerun GPU/model/Docker for review.",
        "",
        "Non-acceptances requested: no formal speedup, no BF16 fidelity, no native long context, no human semantic/audio quality, no product readiness, no public comparison/SOTA.",
    ]
    (root / "reviewer_packet.md").write_text("\n".join(packet_lines) + "\n", encoding="utf-8")
    return 0


def _component_distribution_summary(completed_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    components = ["text_conditioning", "denoise", "attention", "video_vae", "audio_vae", "encoding_mux", "io"]
    out: dict[str, Any] = {"scope": "Per-lane benchmark_record.timing.components distributions across completed matched pairs."}
    for lane in ("reference", "candidate"):
        lane_out: dict[str, Any] = {}
        for component in components:
            values: list[float] = []
            status_counts: dict[str, int] = {}
            units: set[str] = set()
            timing_bases: set[str] = set()
            reasons: set[str] = set()
            additive_values: set[bool] = set()
            parents: set[str] = set()
            n_values: list[float] = []
            for pair in completed_pairs:
                item = pair.get("component_timing", {}).get(lane, {}).get(component, {})
                if not isinstance(item, dict):
                    item = {}
                status = str(item.get("status") or "missing")
                status_counts[status] = status_counts.get(status, 0) + 1
                if item.get("unit") is not None:
                    units.add(str(item.get("unit")))
                if item.get("timing_basis") is not None:
                    timing_bases.add(str(item.get("timing_basis")))
                if item.get("reason") is not None:
                    reasons.add(str(item.get("reason")))
                if item.get("additive_to_e2e") is not None:
                    additive_values.add(bool(item.get("additive_to_e2e")))
                if item.get("parent") is not None:
                    parents.add(str(item.get("parent")))
                seconds = item.get("seconds")
                if isinstance(seconds, (int, float)) and math.isfinite(float(seconds)):
                    values.append(float(seconds))
                n = item.get("n")
                if isinstance(n, (int, float)) and math.isfinite(float(n)):
                    n_values.append(float(n))
            lane_out[component] = {
                "seconds": _stats(values),
                "status_counts": status_counts,
                "units": sorted(units),
                "timing_bases": sorted(timing_bases),
                "reasons": sorted(reasons),
                "additive_to_e2e_values": sorted(additive_values),
                "parents": sorted(parents),
                "component_n_distribution": _stats(n_values),
            }
        out[lane] = lane_out
    return out


def _formal_resource_summary(completed_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "scope": "Resource distributions across completed matched pairs. benchmark_record resources are contract fields; container_resources include the finer local monitor fields when present.",
        "reference": {},
        "candidate": {},
    }
    contract_metrics = ("peak_gpu_memory_mib", "peak_host_memory_gib", "peak_power_w", "failures")
    container_metrics = ("peak_gpu_memory_mib", "peak_gpu_util_percent", "peak_host_memory_used_gib", "peak_power_w", "peak_temperature_c", "sample_count")
    for lane in ("reference", "candidate"):
        for metric in contract_metrics:
            values: list[float] = []
            for pair in completed_pairs:
                data = pair.get("resources", {}).get(lane, {}) if isinstance(pair.get("resources"), dict) else {}
                item = data.get(metric, {}) if isinstance(data, dict) else {}
                value = item.get("value") if isinstance(item, dict) else None
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    values.append(float(value))
            out[lane][metric] = {"source": "benchmark_record.resources", "distribution": _stats(values)}
        for metric in container_metrics:
            values = []
            for pair in completed_pairs:
                data = pair.get("container_resources", {}).get(lane, {}) if isinstance(pair.get("container_resources"), dict) else {}
                value = data.get(metric) if isinstance(data, dict) else None
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    values.append(float(value))
            out[lane][f"container_{metric}"] = {"source": "container_summary.resources", "distribution": _stats(values)}
    return out


def _final_av_accounting_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "scope": "Final AV decode/accounting copied from each lane benchmark record; complete means 720 video frames and 960000 effective audio samples/channel for this 30-second lane.",
        "pairs": [
            {
                "pair": pair.get("pair"),
                "reference": pair.get("final_av_accounting", {}).get("reference"),
                "candidate": pair.get("final_av_accounting", {}).get("candidate"),
                "benchmark_record_paths": pair.get("benchmark_record_paths"),
                "benchmark_validation_paths": pair.get("benchmark_validation_paths"),
            }
            for pair in pairs
        ],
    }


def finalize_r9_adaptive_matched_formal(args: argparse.Namespace) -> int:
    root = Path(args.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidate_profile = str(args.candidate_profile)
    expected_tau = float(args.candidate_tau)
    expected_step_min = _candidate_step_min_from_profile(candidate_profile, getattr(args, "candidate_step_min", None))
    label = _candidate_label_bundle(candidate_profile, expected_step_min)
    requested_pairs = int(args.pairs)
    min_required_pairs = int(args.min_required_pairs)
    if min_required_pairs < 10:
        raise ValueError(f"formal validation requires min_required_pairs >= 10, got {min_required_pairs}")
    pairs = [_r9_pair_analysis(root / f"pair{i:02d}", f"pair{i:02d}", candidate_profile, expected_tau, expected_step_min) for i in range(1, requested_pairs + 1)]
    completed_pairs = [p for p in pairs if p.get("complete")]
    warm_deltas = [float(p["timing"]["warm_e2e_delta_percent_n3_gate_not_speedup"]) for p in completed_pairs if "timing" in p]
    cold_deltas = [float(p["timing"]["cold_e2e_delta_percent_n3_gate_not_speedup"]) for p in completed_pairs if "timing" in p]
    reference_warm = [float(p["timing"]["reference_warm_e2e_seconds"]) for p in completed_pairs if "timing" in p]
    candidate_warm = [float(p["timing"]["candidate_warm_e2e_seconds"]) for p in completed_pairs if "timing" in p]
    reference_cold = [float(p["timing"]["reference_cold_e2e_seconds"]) for p in completed_pairs if "timing" in p]
    candidate_cold = [float(p["timing"]["candidate_cold_e2e_seconds"]) for p in completed_pairs if "timing" in p]
    reference_spgs = [float(p["timing"]["reference_seconds_per_generated_second"]) for p in completed_pairs if p.get("timing", {}).get("reference_seconds_per_generated_second") is not None]
    candidate_spgs = [float(p["timing"]["candidate_seconds_per_generated_second"]) for p in completed_pairs if p.get("timing", {}).get("candidate_seconds_per_generated_second") is not None]
    all_gpu_sets = [tuple(p.get("physical_gpu_uuids") or []) for p in completed_pairs]
    all_workloads = [p.get("workload_fingerprint") for p in completed_pairs]
    all_boundaries = [p.get("timing_boundary") for p in completed_pairs]
    median_delta = statistics.median(warm_deltas) if warm_deltas else None
    min_pair_delta = min(warm_deltas) if warm_deltas else None
    timing_summary = {
        "schema_version": f"minimax-h3-final-av-30s-r9-current-vs-{label['schema_token']}-sol-attn-formal-timing-summary-v1",
        "requested_pairs": requested_pairs,
        "min_required_pairs": min_required_pairs,
        "completed_pairs": len(completed_pairs),
        "reference_warm_e2e_seconds": _stats(reference_warm),
        "candidate_warm_e2e_seconds": _stats(candidate_warm),
        "warm_e2e_delta_percent_formal_candidate_pending_reviewer": _stats(warm_deltas),
        "reference_cold_e2e_seconds": _stats(reference_cold),
        "candidate_cold_e2e_seconds": _stats(candidate_cold),
        "cold_e2e_delta_percent_formal_candidate_pending_reviewer": _stats(cold_deltas),
        "reference_seconds_per_generated_second": _stats(reference_spgs),
        "candidate_seconds_per_generated_second": _stats(candidate_spgs),
        "component_timing_seconds": _component_distribution_summary(completed_pairs),
        "min_median_delta_threshold_percent": float(args.min_median_delta_pct),
        "max_slower_pair_tolerance_percent": float(args.max_slower_pair_pct),
        "claim_boundary": "Formal N>=10 candidate timing distributions only until independent Reviewer acceptance; do not report a formal speedup claim before Reviewer acceptance.",
    }
    resource_summary = _formal_resource_summary(completed_pairs)
    quality_proxy_summary = {
        "schema_version": f"minimax-h3-final-av-30s-r9-current-vs-{label['schema_token']}-sol-attn-formal-quality-proxy-summary-v1",
        "pairs": [
            {
                "pair": p.get("pair"),
                "objective_proxy_comparison": p.get("objective_proxy_comparison"),
                "automatic_red_flags": p.get("automatic_red_flags"),
                "candidate_no_worse_automatic_proxy_flags": p.get("gates", {}).get("candidate_no_worse_automatic_proxy_flags"),
                "objective_5pct_noninferiority_core_metrics": p.get("gates", {}).get("objective_5pct_noninferiority_core_metrics"),
            }
            for p in pairs
        ],
        "scope": "No-reference objective proxies only; no semantic, human visual/audio, prompt-faithfulness, or perceived AV-sync certification.",
    }
    telemetry_summary = {
        "schema_version": f"minimax-h3-final-av-30s-r9-current-vs-{label['schema_token']}-sol-attn-formal-telemetry-summary-v1",
        "pairs": [
            {
                "pair": p.get("pair"),
                "telemetry_paths": p.get("telemetry_paths"),
                "sol_attn_telemetry_summary": p.get("sol_attn_telemetry_summary"),
            }
            for p in pairs
        ],
    }
    final_av_summary = _final_av_accounting_summary(pairs)
    gates = {
        "completed_pair_count_at_least_min_required": len(completed_pairs) >= min_required_pairs,
        "all_requested_pairs_completed": len(completed_pairs) == requested_pairs,
        "all_pair_n1_decisions_pass": all(p.get("gates", {}).get("pair_decision_pass_no_failed_gates") is True for p in completed_pairs) and len(completed_pairs) == requested_pairs,
        "all_benchmark_validations_pass": all(
            p.get("gates", {}).get("reference_benchmark_validation_pass") is True
            and p.get("gates", {}).get("candidate_benchmark_validation_pass") is True
            for p in completed_pairs
        ) and len(completed_pairs) == requested_pairs,
        "same_physical_gpu_all_pairs": bool(all_gpu_sets) and len(set(all_gpu_sets)) == 1 and all(len(x) == 1 for x in all_gpu_sets),
        "same_workload_fingerprint_all_pairs": bool(all_workloads) and len(set(all_workloads)) == 1,
        "same_timing_boundary_all_pairs": bool(all_boundaries) and len(set(all_boundaries)) == 1,
        "median_warm_e2e_delta_meets_formal_threshold": median_delta is not None and median_delta >= float(args.min_median_delta_pct),
        "no_pair_slower_beyond_tolerance": min_pair_delta is not None and min_pair_delta >= -float(args.max_slower_pair_pct),
        "all_objective_proxy_core_noninferiority_pass": all(p.get("gates", {}).get("objective_5pct_noninferiority_core_metrics") is True for p in completed_pairs) and len(completed_pairs) == requested_pairs,
        "all_candidate_proxy_flags_no_worse": all(p.get("gates", {}).get("candidate_no_worse_automatic_proxy_flags") is True for p in completed_pairs) and len(completed_pairs) == requested_pairs,
        "all_final_av_accounting_complete": all(p.get("gates", {}).get("both_final_av_complete") is True for p in completed_pairs) and len(completed_pairs) == requested_pairs,
        "all_sol_attn_zero_fallback_materialization_input_copy": all(
            p.get("gates", {}).get("reference_zero_fallback_materialization_input_copy") is True
            and p.get("gates", {}).get("candidate_zero_fallback_materialization_input_copy") is True
            for p in completed_pairs
        ) and len(completed_pairs) == requested_pairs,
        "all_sol_attn_sparse_stride_calls_valid": all(
            p.get("gates", {}).get("reference_sparse_stride_calls_valid") is True
            and p.get("gates", {}).get("candidate_sparse_stride_calls_valid") is True
            and p.get("gates", {}).get("same_sparse_call_count") is True
            for p in completed_pairs
        ) and len(completed_pairs) == requested_pairs,
        "all_candidate_guarded_adaptive_exercised": all(p.get("gates", {}).get("candidate_guarded_profile_exercised") is True for p in completed_pairs) and len(completed_pairs) == requested_pairs,
        "all_candidate_adaptive_step_min_expected_seen": all(p.get("gates", {}).get("candidate_adaptive_step_min_expected_seen") is True for p in completed_pairs) and len(completed_pairs) == requested_pairs,
        "all_candidate_guard_counts_match_step_min": all(p.get("gates", {}).get("candidate_guard_counts_match_step_min") is True for p in completed_pairs) and len(completed_pairs) == requested_pairs,
        "all_no_raw_tensor_export": all(
            p.get("gates", {}).get("reference_no_raw_tensor_export") is True
            and p.get("gates", {}).get("candidate_no_raw_tensor_export") is True
            for p in completed_pairs
        ) and len(completed_pairs) == requested_pairs,
    }
    failed = [name for name, ok in gates.items() if not ok]
    contract_like = {
        "completed_pair_count_at_least_min_required",
        "all_requested_pairs_completed",
        "all_benchmark_validations_pass",
        "same_physical_gpu_all_pairs",
        "same_workload_fingerprint_all_pairs",
        "same_timing_boundary_all_pairs",
        "all_final_av_accounting_complete",
        "all_sol_attn_zero_fallback_materialization_input_copy",
        "all_sol_attn_sparse_stride_calls_valid",
        "all_candidate_guarded_adaptive_exercised",
        "all_candidate_adaptive_step_min_expected_seen",
        "all_candidate_guard_counts_match_step_min",
        "all_no_raw_tensor_export",
    }
    if any(name in failed for name in contract_like):
        status = "reject"
        classification = f"reject_formal_n10_{label['classification_token']}_contract_telemetry_or_media_failed"
        accepted_candidate_pending_reviewer = False
    elif failed:
        status = "reject"
        classification = f"reject_formal_n10_{label['classification_token']}_timing_or_proxy_failed"
        accepted_candidate_pending_reviewer = False
    else:
        status = "pass"
        classification = f"accepted_formal_n10_{label['classification_token']}_candidate_pending_independent_reviewer"
        accepted_candidate_pending_reviewer = True
    same_gpu_proof = {
        "physical_gpu_uuid_sets": [list(x) for x in all_gpu_sets],
        "unique_physical_gpu_uuid_sets": [list(x) for x in sorted(set(all_gpu_sets))],
        "workload_fingerprints": all_workloads,
        "timing_boundaries": all_boundaries,
    }
    decision = {
        "schema_version": f"minimax-h3-final-av-30s-r9-current-vs-{label['schema_token']}-sol-attn-formal-n10-decision-v1",
        "created_utc": utc_now(),
        "status": status,
        "classification": classification,
        "requested_pairs": requested_pairs,
        "min_required_pairs": min_required_pairs,
        "completed_pairs": len(completed_pairs),
        "sample_count": len(completed_pairs),
        "principal_variable": f"retained r9_current_sol_attn adaptive_routing=0 tau=1.0 diag versus {candidate_profile} tau={expected_tau:g} diag active from adaptive_step_min={expected_step_min}",
        "candidate_profile": candidate_profile,
        "candidate_expected_tau": expected_tau,
        "candidate_expected_adaptive_step_min": expected_step_min,
        "generation_mode": "extension",
        "native_context_supported": False,
        "track": "practical_disclosed_approx",
        "turbo_steps": 8,
        "formal_n_ge_10": True,
        "formal_speedup_claimed": False,
        "accepted_formal_candidate_pending_reviewer": accepted_candidate_pending_reviewer,
        "reviewer_acceptance_status": "pending_host_reviewer_not_invoked_by_engineer",
        "timing_summary": timing_summary,
        "resource_summary": resource_summary,
        "quality_proxy_summary": quality_proxy_summary,
        "telemetry_summary": telemetry_summary,
        "final_av_accounting_summary": final_av_summary,
        "same_gpu_workload_timing_boundary_proof": same_gpu_proof,
        "pairs": pairs,
        "gates": gates,
        "failed_gates": failed,
        "claim_exclusions": {
            "bf16_fidelity": True,
            "native_long_context": True,
            "human_semantic_or_audio_quality": True,
            "product_readiness": True,
            "public_comparison_or_sota": True,
        },
        "claim_boundary": f"Matched formal N>=10 30-second final-AV extension-lane {candidate_profile} versus retained r9_current_sol_attn candidate decision. Output is six-chunk extension/chunked, not native long context; Turbo 8-step is practical/disclosed approximate, not BF16 fidelity; no human-quality, product-readiness, public-comparison, or SOTA claim. Do not report a formal speedup until independent Reviewer acceptance is present.",
    }
    formal_summary = {
        "schema_version": f"minimax-h3-final-av-30s-r9-current-vs-{label['schema_token']}-sol-attn-formal-n10-summary-v1",
        "created_utc": decision["created_utc"],
        "status": status,
        "classification": classification,
        "requested_pairs": requested_pairs,
        "completed_pairs": len(completed_pairs),
        "candidate_profile": candidate_profile,
        "median_warm_delta_percent_pending_reviewer": median_delta,
        "failed_gates": failed,
        "same_gpu_workload_timing_boundary_proof": same_gpu_proof,
        "formal_speedup_claimed": False,
        "reviewer_acceptance_status": "pending_host_reviewer_not_invoked_by_engineer",
    }
    write_json(root / "timing_summary.json", timing_summary)
    write_json(root / "component_timing_summary.json", timing_summary["component_timing_seconds"])
    write_json(root / "resource_summary.json", resource_summary)
    write_json(root / "quality_proxy_comparison.json", quality_proxy_summary)
    write_json(root / "sol_attn_telemetry_summary.json", telemetry_summary)
    write_json(root / "final_av_accounting_summary.json", final_av_summary)
    write_json(root / "formal_n10_summary.json", formal_summary)
    write_json(root / "formal_n10_decision.json", decision)
    write_json(root / "decision.json", decision)
    reviewer_request = {
        "schema_version": f"minimax-h3-final-av-30s-r9-current-vs-{label['schema_token']}-formal-n10-reviewer-request-v1",
        "created_utc": utc_now(),
        "evidence_root": str(root),
        "requested_review": "artifact-only audit; do not rerun GPU/model/Docker",
        "engineer_classification": classification,
        "required_scope": "Formal N>=10 matched 30-second extension-lane candidate acceptance or rejection only; no BF16/human-quality/native-context/product/SOTA/public-comparison claim",
        "decision": str(root / "formal_n10_decision.json"),
        "run_report": str(root / "FORMAL_N10_RUN_REPORT.md"),
    }
    write_json(root / "reviewer_verdict_request.json", reviewer_request)
    report_lines = [
        f"# {label['comparison_readable']} Sol-Attn Final-AV 30s Formal N>=10 RUN_REPORT",
        "",
        f"- Status: `{status}`",
        f"- Classification: `{classification}`",
        f"- Requested/completed pairs: {requested_pairs}/{len(completed_pairs)} (minimum required {min_required_pairs})",
        f"- Candidate profile: `{candidate_profile}` (tau={expected_tau:g}, diag, adaptive_step_min={expected_step_min})",
        f"- Median warm E2E delta (formal candidate, pending Reviewer): {median_delta:.3f}%" if median_delta is not None else "- Median warm E2E delta: unavailable",
        f"- Warm delta distribution (%): {warm_deltas}",
        f"- Timing threshold: median >= {float(args.min_median_delta_pct):.3f}% and no pair slower beyond {float(args.max_slower_pair_pct):.3f}%.",
        f"- Same GPU/workload/timing-boundary proof: {same_gpu_proof}",
        f"- Failed gates: {failed}",
        "- Summary files: `timing_summary.json`, `component_timing_summary.json`, `resource_summary.json`, `quality_proxy_comparison.json`, `sol_attn_telemetry_summary.json`, `final_av_accounting_summary.json`, `formal_n10_summary.json`.",
        "- Sol-Attn gate: each completed lane must have sparse calls, stride-aware V, zero fallback/materialization/input-copy, no raw tensor export, and candidate guarded-active plus guarded-inactive calls.",
        "- Media/proxy gate: each completed lane must have complete final-AV accounting; objective proxies are no-reference only and do not certify human semantic/audio quality.",
        f"- Claim boundary: extension/chunked output, practical Turbo 8-step approximation, formal N>=10 candidate only for `{candidate_profile}` vs retained `r9_current_sol_attn` until Reviewer; no BF16 fidelity, native long context, product-readiness, public comparison, SOTA, or human-quality claim.",
        "- Reviewer acceptance: `pending_host_reviewer_not_invoked_by_engineer`; request written to `reviewer_verdict_request.json`.",
    ]
    report = "\n".join(report_lines) + "\n"
    (root / "FORMAL_N10_RUN_REPORT.md").write_text(report, encoding="utf-8")
    (root / "RUN_REPORT.md").write_text(report, encoding="utf-8")
    packet_lines = [
        f"# Artifact-only Reviewer Packet: {label['candidate_readable']} formal N>=10",
        "",
        f"Evidence root: `{root}`",
        f"Engineer status/classification: `{status}` / `{classification}`",
        "",
        "Review only the root formal decision/report, per-pair lane records and validation outputs, final AV accounting, timing/component/resource/proxy/Sol-Attn summaries/full telemetry files, isolated Docker/runtime provenance, and terminal hygiene artifacts. Do not rerun GPU/model/Docker for review.",
        "",
        "Non-acceptances requested: no BF16 fidelity, no native long context, no human semantic/audio quality, no product readiness, no public comparison/SOTA. Formal speedup may be accepted only if this artifact set passes Reviewer checks.",
    ]
    (root / "reviewer_packet.md").write_text("\n".join(packet_lines) + "\n", encoding="utf-8")
    return 0


def _cache_summary_from_container(summary: dict[str, Any]) -> dict[str, Any]:
    cache = summary.get("cache_dit") if isinstance(summary.get("cache_dit"), dict) else {}
    return cache.get("summary") if isinstance(cache.get("summary"), dict) else {}


def _sol_summary_from_container(summary: dict[str, Any]) -> dict[str, Any]:
    sol = summary.get("sol_attn") if isinstance(summary.get("sol_attn"), dict) else {}
    return sol.get("summary") if isinstance(sol.get("summary"), dict) else {}


def _zero_copy_sol_attn_ok(sol_summary: dict[str, Any]) -> bool:
    return (
        _numeric_summary_value(sol_summary, "fallback_calls", -1) == 0
        and _numeric_summary_value(sol_summary, "materialize_copy_count", -1) == 0
        and _numeric_summary_value(sol_summary, "materialize_copy_bytes", -1) == 0
        and _numeric_summary_value(sol_summary, "input_copy_events", 0) == 0
        and _numeric_summary_value(sol_summary, "input_copy_bytes", 0) == 0
        and sol_summary.get("diagnostic_raw_tensor_exported") is False
    )


def finalize_r10_vae_spatial_tile_batching_matched(args: argparse.Namespace) -> int:
    """Finalize retained r10 VAE serial-tiles vs default-off spatial tile batching N=1."""
    root = Path(args.out_dir).resolve()
    reference_dir = Path(args.reference_evidence).resolve()
    candidate_dir = Path(args.candidate_evidence).resolve()
    root.mkdir(parents=True, exist_ok=True)
    reference_record = read_json(reference_dir / "benchmark_record.json")
    candidate_record = read_json(candidate_dir / "benchmark_record.json")
    reference_summary = read_json(reference_dir / "container_summary.json")
    candidate_summary = read_json(candidate_dir / "container_summary.json")
    reference_objective = _load_objective_from_summary(reference_dir, reference_summary)
    candidate_objective = _load_objective_from_summary(candidate_dir, candidate_summary)
    lane_ctx = _lane_context_from_records(reference_record, candidate_record)

    reference_warm = float(reference_record["timing"]["warm_e2e"]["seconds"])
    candidate_warm = float(candidate_record["timing"]["warm_e2e"]["seconds"])
    warm_delta_pct = (reference_warm - candidate_warm) / reference_warm * 100.0
    reference_cold = float(reference_record["timing"]["cold_e2e"]["seconds"])
    candidate_cold = float(candidate_record["timing"]["cold_e2e"]["seconds"])
    cold_delta_pct = (reference_cold - candidate_cold) / reference_cold * 100.0
    min_delta_pct = float(getattr(args, "min_delta_pct", 1.0))

    same_gpu = reference_record["deployment"].get("physical_gpu_uuids") == candidate_record["deployment"].get("physical_gpu_uuids")
    same_workload = reference_record.get("workload_fingerprint") == candidate_record.get("workload_fingerprint")
    same_timing = reference_record["timing"].get("boundary_id") == candidate_record["timing"].get("boundary_id")
    same_generation = reference_record["production"].get("generation_mode") == candidate_record["production"].get("generation_mode") == "extension"
    objective_cmp = _objective_noninferiority(reference_record["quality"]["objective"], candidate_record["quality"]["objective"])
    reference_flags = list(reference_objective.get("automatic_red_flags", []) or [])
    candidate_flags = list(candidate_objective.get("automatic_red_flags", []) or [])
    proxy_not_worse = set(candidate_flags).issubset(set(reference_flags))
    # Frozen v1 long-video route gates treat automatic proxy red flags as an
    # absolute N=1 promotion blocker.  A candidate being "no worse" or having
    # fewer flags than the reference is diagnostic only; do not recommend N=3 or
    # reviewer promotion while either matched lane has any automatic red flag.
    reference_flags_clear = len(reference_flags) == 0
    candidate_flags_clear = len(candidate_flags) == 0

    reference_sol = reference_summary.get("sol_attn") if isinstance(reference_summary.get("sol_attn"), dict) else {}
    candidate_sol = candidate_summary.get("sol_attn") if isinstance(candidate_summary.get("sol_attn"), dict) else {}
    reference_sol_summary = reference_sol.get("summary") if isinstance(reference_sol.get("summary"), dict) else {}
    candidate_sol_summary = candidate_sol.get("summary") if isinstance(candidate_sol.get("summary"), dict) else {}
    reference_sparse = _numeric_summary_value(reference_sol_summary, "sparse_calls", 0)
    candidate_sparse = _numeric_summary_value(candidate_sol_summary, "sparse_calls", 0)
    reference_profile_gates = _profile_policy_gates(
        role="reference",
        profile="r10_adaptive_tau1_5_step3_diag",
        summary=reference_sol_summary,
        expected_tau=1.5,
        expected_step_min=3,
        sparse_calls=reference_sparse,
        tau_gate_name="reference_adaptive_tau1_5_diag_seen",
    )
    candidate_profile_gates = _profile_policy_gates(
        role="candidate",
        profile="r10_adaptive_tau1_5_step3_diag",
        summary=candidate_sol_summary,
        expected_tau=1.5,
        expected_step_min=3,
        sparse_calls=candidate_sparse,
        tau_gate_name="candidate_adaptive_tau1_5_diag_seen",
    )

    reference_vae = reference_summary.get("video_vae") if isinstance(reference_summary.get("video_vae"), dict) else {}
    candidate_vae = candidate_summary.get("video_vae") if isinstance(candidate_summary.get("video_vae"), dict) else {}
    reference_tile_batch_size = int(reference_vae.get("tile_batch_size") or 0)
    candidate_tile_batch_size = int(candidate_vae.get("tile_batch_size") or 0)
    expected_candidate_tile_batch_size = getattr(args, "candidate_vae_tile_batch_size", None)
    if expected_candidate_tile_batch_size is not None:
        expected_candidate_tile_batch_size = int(expected_candidate_tile_batch_size)
    reference_mechanisms = list(reference_record.get("track", {}).get("mechanisms") or [])
    candidate_mechanisms = list(candidate_record.get("track", {}).get("mechanisms") or [])
    reference_stage = reference_summary.get("stage_duration_summary") if isinstance(reference_summary.get("stage_duration_summary"), dict) else {}
    candidate_stage = candidate_summary.get("stage_duration_summary") if isinstance(candidate_summary.get("stage_duration_summary"), dict) else {}
    reference_video_vae_wall = reference_stage.get("video_vae_decode_wall_seconds")
    candidate_video_vae_wall = candidate_stage.get("video_vae_decode_wall_seconds")
    video_vae_wall_improved = (
        isinstance(reference_video_vae_wall, (int, float))
        and isinstance(candidate_video_vae_wall, (int, float))
        and float(candidate_video_vae_wall) < float(reference_video_vae_wall)
    )

    def resource_value(record: dict[str, Any], name: str) -> float | None:
        value = record.get("resources", {}).get(name, {}).get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    candidate_bounded_mechanism = f"video_vae_bounded_spatial_tile_batching_cap_{candidate_tile_batch_size}"
    candidate_vae_batching_active = candidate_vae.get("spatial_tile_batching") is True or candidate_tile_batch_size > 0
    candidate_mechanism_disclosed = (
        "practical_approximate_vae_decode" in candidate_mechanisms
        and (
            "video_vae_spatial_tile_batching_stack_tiling" in candidate_mechanisms
            or (candidate_tile_batch_size > 0 and candidate_bounded_mechanism in candidate_mechanisms)
        )
    )

    gates = {
        "same_physical_gpu": same_gpu,
        "same_workload_fingerprint": same_workload,
        "same_timing_boundary": same_timing,
        "same_extension_generation_mode": same_generation,
        "both_final_av_complete": reference_record["output_av"].get("final_accounting_complete") is True and candidate_record["output_av"].get("final_accounting_complete") is True,
        "reference_request_quality_lossless": reference_summary.get("request_quality") == "lossless",
        "candidate_request_quality_lossless": candidate_summary.get("request_quality") == "lossless",
        "reference_vae_batching_off": reference_vae.get("spatial_tile_batching") is False and reference_tile_batch_size == 0,
        "candidate_vae_batching_active": candidate_vae_batching_active,
        "candidate_vae_tile_batch_size_matches_expected": expected_candidate_tile_batch_size is None or candidate_tile_batch_size == expected_candidate_tile_batch_size,
        "reference_mechanism_discloses_off": "video_vae_spatial_tile_batching_off" in reference_mechanisms,
        "candidate_mechanism_discloses_approx_vae": candidate_mechanism_disclosed,
        "reference_sol_attn_telemetry_present": reference_sol.get("status") == "present",
        "candidate_sol_attn_telemetry_present": candidate_sol.get("status") == "present",
        "reference_sparse_stride_calls_valid": _sol_attn_sparse_stride_ok(reference_sol_summary),
        "candidate_sparse_stride_calls_valid": _sol_attn_sparse_stride_ok(candidate_sol_summary),
        "same_sparse_call_count": reference_sparse == candidate_sparse and reference_sparse > 0,
        "reference_zero_fallback_materialization_input_copy": _sol_attn_zero_copy_ok(reference_sol_summary),
        "candidate_zero_fallback_materialization_input_copy": _sol_attn_zero_copy_ok(candidate_sol_summary),
        **reference_profile_gates,
        **candidate_profile_gates,
        "reference_no_automatic_proxy_red_flags": reference_flags_clear,
        "candidate_no_automatic_proxy_red_flags": candidate_flags_clear,
        "candidate_no_worse_automatic_proxy_flags": proxy_not_worse,
        "candidate_no_worse_automatic_proxy_flags_diagnostic": proxy_not_worse,
        "objective_5pct_noninferiority_core_metrics": objective_cmp.get("status") == "pass",
        "candidate_warm_e2e_delta_meets_n1_signal": warm_delta_pct >= min_delta_pct,
        "candidate_video_vae_wall_improved_when_split_profile_present": video_vae_wall_improved,
        "candidate_failures_zero": resource_value(candidate_record, "failures") == 0.0,
        "candidate_peak_gpu_memory_within_a6000": (resource_value(candidate_record, "peak_gpu_memory_mib") or math.inf) < 47000.0,
    }
    failed = [name for name, ok in gates.items() if not ok]
    promote = len(failed) == 0
    status = "pass" if promote else "reject"
    classification = (
        "promote_to_n3_default_off_r10_video_vae_spatial_tile_batching_n1_pending_reviewer"
        if promote
        else "reject_no_promotion_r10_video_vae_spatial_tile_batching_n1_failed_timing_or_proxy_gate"
    )
    timing_summary = {
        "schema_version": "minimax-h3-final-av-30s-r10-vae-spatial-tile-batching-n1-timing-v1",
        "reference_warm_e2e_seconds": reference_warm,
        "candidate_warm_e2e_seconds": candidate_warm,
        "warm_e2e_delta_percent_n1_not_speedup": warm_delta_pct,
        "min_delta_percent_for_n3_promotion": min_delta_pct,
        "reference_cold_e2e_seconds": reference_cold,
        "candidate_cold_e2e_seconds": candidate_cold,
        "cold_e2e_delta_percent_n1_not_speedup": cold_delta_pct,
        "reference_video_vae_decode_wall_seconds": reference_video_vae_wall,
        "candidate_video_vae_decode_wall_seconds": candidate_video_vae_wall,
    }
    quality_summary = {
        "schema_version": "minimax-h3-final-av-30s-r10-vae-spatial-tile-batching-n1-quality-proxy-v1",
        "status": objective_cmp.get("status"),
        "objective_noninferiority": objective_cmp,
        "automatic_red_flags": {"reference": reference_flags, "candidate": candidate_flags},
        "strict_automatic_red_flag_gate": {
            "status": "pass" if reference_flags_clear and candidate_flags_clear else "fail",
            "requirement": "both matched lanes must have zero automatic red flags before any N=3/reviewer promotion",
        },
        "candidate_no_worse_flags_diagnostic": proxy_not_worse,
        "scope": "No-reference objective proxies only; no human semantic/audio quality certification.",
    }
    sol_summary = {
        "schema_version": "minimax-h3-final-av-30s-r10-vae-spatial-tile-batching-n1-sol-attn-v1",
        "scope": "Sol-Attn routing/tau/cache variables are fixed to retained r10; VAE tile batching is the only intended principal variable.",
        "reference": reference_sol_summary,
        "candidate": candidate_sol_summary,
    }
    decision = {
        "schema_version": "minimax-h3-final-av-30s-r10-video-vae-spatial-tile-batching-n1-decision-v1",
        "created_utc": utc_now(),
        "status": status,
        "classification": classification,
        "reference_evidence": str(reference_dir),
        "candidate_evidence": str(candidate_dir),
        "principal_variable": (
            f"MINIMAX_H3_A6000_VIDEO_VAE_TILE_BATCH_SIZE=0 versus ={candidate_tile_batch_size}; retained r10 Sol-Attn variables unchanged"
            if candidate_tile_batch_size > 0
            else "MINIMAX_H3_A6000_VIDEO_VAE_SPATIAL_TILE_BATCHING=0 versus =1; retained r10 Sol-Attn variables unchanged"
        ),
        "generation_mode": "extension",
        "native_context_supported": False,
        "track": "practical_disclosed_approx",
        "sample_count": 1,
        "speedup_claimed": False,
        "n1_route_gate_language_only": True,
        "promote_to_n3_recommended_pending_reviewer": promote,
        "reviewer_acceptance_status": "pending_host_reviewer_not_invoked_by_engineer",
        "timing_summary": timing_summary,
        "quality_proxy_comparison": quality_summary,
        "sol_attn_telemetry_summary": sol_summary,
        "resources": {"reference": reference_record.get("resources"), "candidate": candidate_record.get("resources")},
        "gates": gates,
        "failed_gates": failed,
        "claim_boundary": "Matched N=1 30-second final-AV extension route gate only. Video VAE spatial tile batching or bounded tile-batch-size is practical approximate VAE decode, not exact/lossless or BF16 fidelity; output is extension/chunked, not native long context; no formal speedup, human-quality, product, public-comparison, or SOTA claim.",
    }
    write_json(root / "timing_summary.json", timing_summary)
    write_json(root / "quality_proxy_comparison.json", quality_summary)
    write_json(root / "sol_attn_telemetry_summary.json", sol_summary)
    write_json(root / "decision.json", decision)
    lines = [
        "# r10 Video VAE Spatial Tile Batching Final-AV 30s N=1 RUN_REPORT",
        "",
        f"- Status: `{status}`",
        f"- Classification: `{classification}`",
        f"- Reference evidence: `{reference_dir}`",
        f"- Candidate evidence: `{candidate_dir}`",
        f"- Warm E2E reference/candidate: {reference_warm:.3f}s / {candidate_warm:.3f}s",
        f"- N=1 warm delta (route-gate only, not speedup): {warm_delta_pct:.3f}% (threshold {min_delta_pct:.3f}%)",
        f"- Video VAE wall reference/candidate: {reference_video_vae_wall} / {candidate_video_vae_wall}",
        f"- Sol-Attn sparse calls reference/candidate: {reference_sparse} / {candidate_sparse}; fallback/materialization/input-copy fixed by gates.",
        f"- Automatic proxy flags reference/candidate: {reference_flags} / {candidate_flags}",
        f"- Strict automatic-red-flag gate: {'pass' if reference_flags_clear and candidate_flags_clear else 'fail'} (promotion requires zero flags in both lanes)",
        f"- Failed gates: {failed}",
        f"- Claim boundary: {decision['claim_boundary']}",
        "- Reviewer acceptance: `pending_host_reviewer_not_invoked_by_engineer`.",
    ]
    (root / "RUN_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def finalize_r10_cache_dit_matched(args: argparse.Namespace) -> int:
    """Finalize retained r10-cache-off vs one request-scoped Cache-DiT N=1 candidate."""

    candidate_quality = str(getattr(args, "candidate_quality", "high"))
    if candidate_quality not in CACHE_DIT_REQUEST_PROFILES:
        raise ValueError(f"unsupported Cache-DiT candidate quality: {candidate_quality!r}")
    candidate_label = cache_dit_profile_label(candidate_quality)
    candidate_profile = f"r10_adaptive_tau1_5_step3_diag_{candidate_label}"
    classification_token = "r10_cache_dit_high" if candidate_quality == "high" else f"r10_{candidate_label}"
    schema_token = "r10-cache-dit-high" if candidate_quality == "high" else f"r10-{candidate_label.replace('_', '-')}"
    expected_profile = cache_dit_expected_config(candidate_quality)

    root = Path(args.out_dir).resolve()
    reference_dir = Path(args.reference_evidence).resolve()
    candidate_dir = Path(args.candidate_evidence).resolve()
    root.mkdir(parents=True, exist_ok=True)
    reference_record = read_json(reference_dir / "benchmark_record.json")
    candidate_record = read_json(candidate_dir / "benchmark_record.json")
    reference_summary = read_json(reference_dir / "container_summary.json")
    candidate_summary = read_json(candidate_dir / "container_summary.json")
    reference_objective = _load_objective_from_summary(reference_dir, reference_summary)
    candidate_objective = _load_objective_from_summary(candidate_dir, candidate_summary)
    objective_cmp = _objective_noninferiority(reference_record["quality"]["objective"], candidate_record["quality"]["objective"])

    reference_warm = float(reference_record["timing"]["warm_e2e"]["seconds"])
    candidate_warm = float(candidate_record["timing"]["warm_e2e"]["seconds"])
    reference_cold = float(reference_record["timing"]["cold_e2e"]["seconds"])
    candidate_cold = float(candidate_record["timing"]["cold_e2e"]["seconds"])
    warm_delta_pct = (reference_warm - candidate_warm) / reference_warm * 100.0
    cold_delta_pct = (reference_cold - candidate_cold) / reference_cold * 100.0

    ref_sol = _sol_summary_from_container(reference_summary)
    cand_sol = _sol_summary_from_container(candidate_summary)
    ref_cache = _cache_summary_from_container(reference_summary)
    cand_cache = _cache_summary_from_container(candidate_summary)
    ref_flags = list(reference_objective.get("automatic_red_flags", []) or [])
    cand_flags = list(candidate_objective.get("automatic_red_flags", []) or [])
    parsed_reuse = cand_cache.get("parsed_reuse_or_skip_count")
    parsed_compute = cand_cache.get("parsed_compute_count")
    same_gpu = reference_record["deployment"].get("physical_gpu_uuids") == candidate_record["deployment"].get("physical_gpu_uuids")
    same_workload = reference_record.get("workload_fingerprint") == candidate_record.get("workload_fingerprint")
    same_timing = reference_record["timing"].get("boundary_id") == candidate_record["timing"].get("boundary_id")
    same_generation = reference_record["production"].get("generation_mode") == candidate_record["production"].get("generation_mode") == "extension"

    gates = {
        "same_physical_gpu": same_gpu,
        "same_workload_fingerprint": same_workload,
        "same_timing_boundary": same_timing,
        "same_extension_generation_mode": same_generation,
        "both_final_av_complete": reference_record["output_av"].get("final_accounting_complete") is True and candidate_record["output_av"].get("final_accounting_complete") is True,
        "reference_request_quality_lossless": reference_summary.get("request_quality") == "lossless",
        "candidate_request_quality_expected": candidate_summary.get("request_quality") == candidate_quality,
        "candidate_cache_dit_telemetry_present": (candidate_summary.get("cache_dit") or {}).get("status") == "present",
        "candidate_cache_refresh_seen": int(cand_cache.get("refresh_log_count") or 0) > 0,
        "candidate_cache_reuse_or_skip_count_parsed": isinstance(parsed_reuse, int),
        "candidate_cache_compute_count_parsed": isinstance(parsed_compute, int),
        "candidate_cache_reuse_or_skip_positive": isinstance(parsed_reuse, int) and parsed_reuse > 0,
        "reference_sol_attn_zero_copy_no_raw": _zero_copy_sol_attn_ok(ref_sol),
        "candidate_sol_attn_zero_copy_no_raw": _zero_copy_sol_attn_ok(cand_sol),
        "reference_sparse_calls_positive": int(ref_sol.get("sparse_calls") or 0) > 0,
        "candidate_sparse_calls_positive": int(cand_sol.get("sparse_calls") or 0) > 0,
        "r10_step_min_fixed_when_reported": (
            not ref_sol.get("adaptive_step_min_values") or ref_sol.get("adaptive_step_min_values") == [3]
        ) and (
            not cand_sol.get("adaptive_step_min_values") or cand_sol.get("adaptive_step_min_values") == [3]
        ),
        "candidate_no_worse_automatic_proxy_flags": len(cand_flags) <= len(ref_flags),
        "objective_5pct_noninferiority_core_metrics": objective_cmp.get("status") == "pass",
        "candidate_warm_e2e_not_slower": candidate_warm <= reference_warm,
        "n1_delta_meets_route_threshold": warm_delta_pct >= float(args.min_delta_pct),
    }
    failed = [name for name, ok in gates.items() if not ok]
    contract_like = {
        "same_physical_gpu",
        "same_workload_fingerprint",
        "same_timing_boundary",
        "same_extension_generation_mode",
        "both_final_av_complete",
        "reference_request_quality_lossless",
        "candidate_request_quality_expected",
        "candidate_cache_dit_telemetry_present",
        "candidate_cache_refresh_seen",
        "candidate_cache_reuse_or_skip_count_parsed",
        "candidate_cache_compute_count_parsed",
        "reference_sol_attn_zero_copy_no_raw",
        "candidate_sol_attn_zero_copy_no_raw",
        "reference_sparse_calls_positive",
        "candidate_sparse_calls_positive",
        "r10_step_min_fixed_when_reported",
    }
    if any(name in failed for name in contract_like):
        status = "reject"
        classification = f"reject_{classification_token}_contract_cache_or_sol_attn_telemetry_failed"
        promote = False
    elif failed:
        status = "reject"
        classification = f"reject_{classification_token}_timing_or_objective_proxy_failed"
        promote = False
    else:
        status = "pass"
        classification = f"keep_{classification_token}_n1_route_gate_pass_pending_independent_review"
        promote = True

    timing_summary = {
        "schema_version": f"minimax-h3-final-av-30s-{schema_token}-n1-timing-summary-v1",
        "reference_warm_e2e_seconds": reference_warm,
        "candidate_warm_e2e_seconds": candidate_warm,
        "warm_e2e_delta_percent_n1_not_speedup": warm_delta_pct,
        "reference_cold_e2e_seconds": reference_cold,
        "candidate_cold_e2e_seconds": candidate_cold,
        "cold_e2e_delta_percent_n1_not_speedup": cold_delta_pct,
        "route_threshold_percent": float(args.min_delta_pct),
    }
    cache_summary = {
        "schema_version": f"minimax-h3-final-av-30s-{schema_token}-n1-cache-telemetry-v1",
        "reference": reference_summary.get("cache_dit"),
        "candidate": candidate_summary.get("cache_dit"),
        "candidate_expected_profile": expected_profile,
    }
    quality_summary = {
        "schema_version": f"minimax-h3-final-av-30s-{schema_token}-n1-quality-proxy-comparison-v1",
        "objective_proxy_comparison": objective_cmp,
        "automatic_red_flags": {"reference": ref_flags, "candidate": cand_flags},
        "scope": "No-reference objective proxies only; no human semantic/audio quality certification.",
    }
    sol_summary = {
        "schema_version": f"minimax-h3-final-av-30s-{schema_token}-n1-sol-attn-telemetry-v1",
        "reference": ref_sol,
        "candidate": cand_sol,
        "scope": "Sol-Attn policy/tau/step-min remain retained r10; call counts may differ only because Cache-DiT skips full block-stack work.",
    }
    decision = {
        "schema_version": f"minimax-h3-final-av-30s-{schema_token}-n1-decision-v1",
        "created_utc": utc_now(),
        "status": status,
        "classification": classification,
        "reference_evidence": str(reference_dir),
        "candidate_evidence": str(candidate_dir),
        "principal_variable": f"request-scoped Cache-DiT quality={candidate_quality} versus quality=lossless while retaining r10_adaptive_tau1_5_step3_diag Sol-Attn variables",
        "reference_profile": "r10_adaptive_tau1_5_step3_diag_cache_off_lossless",
        "candidate_profile": candidate_profile,
        "generation_mode": "extension",
        "native_context_supported": False,
        "track": "practical_disclosed_approx",
        "sample_count": 1,
        "speedup_claimed": False,
        "n1_route_gate_language_only": True,
        "promote_to_n3_recommended_pending_reviewer": promote,
        "reviewer_acceptance_status": "pending_host_reviewer_not_invoked_by_engineer",
        "timing_summary": timing_summary,
        "cache_telemetry_summary": cache_summary,
        "sol_attn_telemetry_summary": sol_summary,
        "quality_proxy_comparison": quality_summary,
        "resources": {"reference": reference_record.get("resources"), "candidate": candidate_record.get("resources")},
        "gates": gates,
        "failed_gates": failed,
        "claim_boundary": f"Matched N=1 30-second final-AV extension route gate only. Cache-DiT quality={candidate_quality} is practical approximate denoise-reuse, not BF16 fidelity; output is extension/chunked, not native long context; no formal speedup, human-quality, product, public-comparison, or SOTA claim.",
    }
    write_json(root / "timing_summary.json", timing_summary)
    write_json(root / "cache_dit_telemetry_summary.json", cache_summary)
    write_json(root / "sol_attn_telemetry_summary.json", sol_summary)
    write_json(root / "quality_proxy_comparison.json", quality_summary)
    write_json(root / "decision.json", decision)
    write_json(
        root / "reviewer_verdict_request.json",
        {
            "schema_version": f"minimax-h3-final-av-30s-{schema_token}-reviewer-request-v1",
            "created_utc": utc_now(),
            "evidence_root": str(root),
            "requested_review": "artifact-only audit; host invokes Reviewer; Engineer did not fabricate a verdict",
            "engineer_classification": classification,
            "decision": str(root / "decision.json"),
            "required_scope": decision["claim_boundary"],
        },
    )
    lines = [
        f"# r10 Sol-Attn Cache-DiT {candidate_quality} Final-AV 30s N=1 RUN_REPORT",
        "",
        f"- Status: `{status}`",
        f"- Classification: `{classification}`",
        f"- Reference evidence: `{reference_dir}`",
        f"- Candidate evidence: `{candidate_dir}`",
        f"- Warm E2E reference/candidate: {reference_warm:.3f}s / {candidate_warm:.3f}s",
        f"- N=1 warm delta (route-gate only, not speedup): {warm_delta_pct:.3f}%",
        f"- Candidate parsed cache reuse/skip count: {parsed_reuse}; compute count: {parsed_compute}",
        f"- Failed gates: {failed}",
        f"- Claim boundary: {decision['claim_boundary']}",
        "- Reviewer acceptance: `pending_host_reviewer_not_invoked_by_engineer`.",
    ]
    (root / "RUN_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MiniMax-H3 final-AV 30s extension N=1 private runner helper")
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("container-run")
    c.add_argument("--evidence", required=True)
    c.add_argument("--manifest", required=True)
    c.add_argument("--prompt", required=True)
    c.add_argument("--model-dir", default="/models/Turbo/FL2VA")
    c.add_argument("--steps", type=int, default=8)
    c.add_argument("--flow-shift", type=float, default=12.0)
    c.add_argument("--audio-flow-shift", type=float, default=3.0)
    c.add_argument("--init-timeout-s", type=int, default=2400)
    c.add_argument("--stage-init-timeout-s", type=int, default=1800)
    c.add_argument("--readiness-timeout-s", type=int, default=3600)
    c.add_argument("--request-timeout-s", type=int, default=7200)
    c.add_argument("--dlo-resident-layers", type=int, default=12)
    c.add_argument("--attention-backend", default="CUDNN_ATTN")
    c.add_argument("--mode-label", default="dense_cudnn")
    c.add_argument("--request-quality", choices=CACHE_DIT_REQUEST_QUALITIES, default="lossless")
    c.add_argument("--server-cache-backend", choices=("none", "cache_dit"), default="none")
    c.add_argument("--enable-cache-dit-summary", type=int, choices=(0, 1), default=0)
    c.add_argument("--vae-spatial-tile-batching", type=int, choices=(0, 1), default=0)
    c.add_argument("--vae-tile-batch-size", type=int, default=0)
    c.add_argument("--regional-compile", type=int, choices=(0, 1), default=0)
    c.add_argument("--diffusion-compile-dynamic", type=int, choices=(0, 1), default=1)
    c.set_defaults(func=run_container)

    p = sub.add_parser("regional-compile-probe-mode")
    p.add_argument("--evidence", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--model-dir", default="/models/Turbo/FL2VA")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--flow-shift", type=float, default=12.0)
    p.add_argument("--audio-flow-shift", type=float, default=3.0)
    p.add_argument("--init-timeout-s", type=int, default=2400)
    p.add_argument("--stage-init-timeout-s", type=int, default=1800)
    p.add_argument("--readiness-timeout-s", type=int, default=3600)
    p.add_argument("--request-timeout-s", type=int, default=7200)
    p.add_argument("--dlo-resident-layers", type=int, default=12)
    p.add_argument("--attention-backend", default="H3_A6000_SOL_ATTN")
    p.add_argument("--mode-label", required=True)
    p.add_argument("--seed", type=int, default=4200)
    p.add_argument("--regional-compile", type=int, choices=(0, 1), default=0)
    p.add_argument("--diffusion-compile-dynamic", type=int, choices=(0, 1), default=1)
    p.set_defaults(func=run_regional_compile_probe_mode)

    f = sub.add_parser("finalize-record")
    f.add_argument("--evidence", required=True)
    f.add_argument("--manifest", required=True)
    f.add_argument("--gpu-uuid", required=True)
    f.set_defaults(func=finalize_record)

    r = sub.add_parser("recover-final-av")
    r.add_argument("--evidence", required=True)
    r.add_argument("--manifest", required=True)
    r.add_argument("--gpu-uuid", default=None)
    r.set_defaults(func=recover_final_av)

    m = sub.add_parser("finalize-matched")
    m.add_argument("--out-dir", required=True)
    m.add_argument("--dense-evidence", required=True)
    m.add_argument("--candidate-evidence", required=True)
    m.add_argument("--min-delta-pct", type=float, default=1.5)
    m.set_defaults(func=finalize_matched)

    vae = sub.add_parser("finalize-r10-vae-spatial-tile-batching-matched")
    vae.add_argument("--out-dir", required=True)
    vae.add_argument("--reference-evidence", required=True)
    vae.add_argument("--candidate-evidence", required=True)
    vae.add_argument("--min-delta-pct", type=float, default=1.0)
    vae.add_argument("--candidate-vae-tile-batch-size", type=int, default=None)
    vae.set_defaults(func=finalize_r10_vae_spatial_tile_batching_matched)

    cache = sub.add_parser("finalize-r10-cache-dit-matched")
    cache.add_argument("--out-dir", required=True)
    cache.add_argument("--reference-evidence", required=True)
    cache.add_argument("--candidate-evidence", required=True)
    cache.add_argument("--min-delta-pct", type=float, default=1.0)
    cache.add_argument("--candidate-quality", choices=tuple(CACHE_DIT_REQUEST_PROFILES.keys()), default="high")
    cache.set_defaults(func=finalize_r10_cache_dit_matched)

    a = sub.add_parser("finalize-r9-adaptive-matched")
    a.add_argument("--out-dir", required=True)
    a.add_argument("--reference-evidence", required=True)
    a.add_argument("--candidate-evidence", required=True)
    a.add_argument("--reference-profile", default=None)
    a.add_argument("--reference-tau", type=float, default=None)
    a.add_argument("--reference-step-min", type=int, default=None)
    a.add_argument("--reference-layer-min", type=int, default=None)
    a.add_argument("--reference-layer-max", type=int, default=None)
    a.add_argument("--reference-layer-range-scope", default=None)
    a.add_argument("--candidate-profile", default=None)
    a.add_argument("--candidate-tau", type=float, default=None)
    a.add_argument("--candidate-step-min", type=int, default=None)
    a.add_argument("--candidate-layer-min", type=int, default=None)
    a.add_argument("--candidate-layer-max", type=int, default=None)
    a.add_argument("--candidate-layer-range-scope", default=None)
    a.add_argument("--strict-no-automatic-red-flags", action="store_true")
    a.add_argument("--force-no-promotion", action="store_true")
    a.set_defaults(func=finalize_r9_adaptive_matched)

    n3 = sub.add_parser("finalize-r9-adaptive-matched-n3")
    n3.add_argument("--out-dir", required=True)
    n3.add_argument("--reference-profile", default="r9_current_sol_attn")
    n3.add_argument("--reference-tau", type=float, default=None)
    n3.add_argument("--reference-step-min", type=int, default=None)
    n3.add_argument("--reference-layer-min", type=int, default=None)
    n3.add_argument("--reference-layer-max", type=int, default=None)
    n3.add_argument("--reference-layer-range-scope", default=None)
    n3.add_argument("--candidate-profile", default="r9_adaptive_tau1_5_late_steps_diag")
    n3.add_argument("--candidate-tau", type=float, default=1.5)
    n3.add_argument("--candidate-step-min", type=int, default=None)
    n3.add_argument("--candidate-layer-min", type=int, default=None)
    n3.add_argument("--candidate-layer-max", type=int, default=None)
    n3.add_argument("--candidate-layer-range-scope", default=None)
    n3.add_argument("--pairs", type=int, default=3)
    n3.add_argument("--min-median-delta-pct", type=float, default=1.0)
    n3.add_argument("--max-slower-pair-pct", type=float, default=0.0)
    n3.set_defaults(func=finalize_r9_adaptive_matched_n3)

    formal = sub.add_parser("finalize-r9-adaptive-matched-formal")
    formal.add_argument("--out-dir", required=True)
    formal.add_argument("--candidate-profile", default="r9_adaptive_tau1_5_late_steps_diag")
    formal.add_argument("--candidate-tau", type=float, default=1.5)
    formal.add_argument("--candidate-step-min", type=int, default=None)
    formal.add_argument("--pairs", type=int, default=10)
    formal.add_argument("--min-required-pairs", type=int, default=10)
    formal.add_argument("--min-median-delta-pct", type=float, default=1.0)
    formal.add_argument("--max-slower-pair-pct", type=float, default=0.0)
    formal.set_defaults(func=finalize_r9_adaptive_matched_formal)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
