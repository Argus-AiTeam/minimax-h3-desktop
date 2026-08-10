# Deferred GPU test plan

GPU tests remain **external gates** for CPU/static tasks. The standalone exact-kernel GPU2 gate has been completed and absorbed from existing evidence; no GPU was used while updating this code. Future integration/model gates still require explicit operator authorization. Do not run `nvidia-smi`, model load, Docker GPU, CUDA compilation, or any GPU harness as part of CPU/static tasks.

## Later gate: single-item A6000 source-patch smoke

Prerequisites:

1. Baseline process is complete and operator authorizes a single A6000 test.
2. New disposable vLLM-Omni worktree or image is created; the locked `runtime/.../src/vllm-omni` tree remains unchanged.
3. Patch is checked with `git apply --check` and then applied only to the disposable copy.
4. All env switches remain off for import smoke, then exactly one switch is changed per test.

## Completed exact-kernel candidate sequence

Evidence path: `${PWD}/technical_report/evidence/minimax_h3_desktop/sol_engine_port/gpu_exact_20260809T155451Z`.

- Correctness: 8/8 cases compiled and launched; all `max_abs=0`, `max_rel=0`, `mismatch=0`.
- Microbenchmark: kernel-only raw latency JSON, not H3 E2E. Median speedups from the absorbed JSON are indexed modulation 22.02x, indexed gate 11.66x, RoPE 6.50x, and SwiGLU 8.09-8.11x.
- Regression note: harness path/output bugs were fixed previously (`ports` parent permissions and obsolete `parents[2]` path logic) and now have CPU/static regression coverage. The RoPE candidate guard also explicitly requires FP32 `freqs`, matching the vLLM boundary and harness inputs.

## Repeat exact-kernel candidate sequence, if needed

1. Ensure the process sees exactly one GPU and it is an A6000 SM86; run no model load.
2. Run:
   ```bash
   PYTHONPATH=ports/minimax_h3_a6000/src \
   python3 ports/minimax_h3_a6000/gpu_exact_kernel_test.py --device cuda:0 --output /tmp/h3_exact_correctness.json
   ```
3. Require the JSON `coverage_tags` to include fixed seed, random inputs, explicit per-op extremes, tag edges, non-aligned tails, and representative T/H/D shapes. Require every case to report `compile_status=compiled_and_launched` and `mismatch=0`; record `max_abs`, `max_rel`, and `numel` as raw JSON evidence.
4. Run the kernel-only benchmark:
   ```bash
   PYTHONPATH=ports/minimax_h3_a6000/src \
   python3 ports/minimax_h3_a6000/gpu_exact_kernel_bench.py --device cuda:0 --output /tmp/h3_exact_bench.json
   ```
   Keep `warmup >= 20` and `repeats >= 100`; report raw latency arrays only and do not label this H3 E2E.

## Later r3 source-patch integration sequence

Build the disposable integration image from the r2 base without GPU:

```bash
ROOT=${PWD} \
EVIDENCE_DIR=/tmp/h3_r3_build \
ports/minimax_h3_a6000/integration/r3/build_r3_overlay_image.sh
```

External GPU2 5-step same-workload integration gate, after operator authorization and after mapping physical GPU2 as the only visible in-container device:

```bash
ROOT=${PWD} \
GPU_INDEX=2 \
IMAGE=argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r3-exact-overlay \
OUT_DIR=${PWD}/technical_report/evidence/minimax_h3_desktop/sol_engine_port/r3_integration_$(date -u +%Y%m%dT%H%M%SZ) \
ports/minimax_h3_a6000/integration/run_gpu2_exact_integration_5step.sh
```

This script first runs dense reference, then enables `MINIMAX_H3_A6000_ENABLE_FUSED_ADALN=1`, `MINIMAX_H3_A6000_ENABLE_FUSED_ROPE=1`, and `MINIMAX_H3_A6000_ENABLE_FUSED_SWIGLU=1` with overlay/candidates/telemetry on. It keeps `MINIMAX_H3_A6000_ENABLE_SOL_ATTN=0` and `MINIMAX_H3_A6000_SOL_ATTN_CACHE=0`, verifies AV decode metadata for both outputs, and fails unless exact telemetry shows `calls > 0` and `candidate > 0` for AdaLN modulation/gate, RoPE, and SwiGLU. It is a 5-step functional integration gate, not a baseline or E2E benchmark.

## Later r4 per-kernel diagnostic ablation sequence

The r4 all-exact result is needs-revision, not fidelity evidence: all candidates avoided fallback and the 5-step run showed 1.79% apparent speedup, but dense-vs-exact quality differed (video PSNR 24.63 dB, audio waveform cosine 0.9776). To localize that drift after rebuilding the r4 overlay image from the current port, run the external GPU2 ablation gate:

```bash
ROOT=${PWD} \
GPU_INDEX=2 \
IMAGE=argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r4-exact-overlay \
OUT_DIR=${PWD}/technical_report/evidence/minimax_h3_desktop/sol_engine_port/r4_ablation_$(date -u +%Y%m%dT%H%M%SZ) \
ports/minimax_h3_a6000/integration/run_gpu2_exact_ablation_5step_r4.sh
```

It runs dense, indexed modulation only, indexed gate only, RoPE only, SwiGLU only, and all-exact modes with Sol-Attn/cache off. It writes `quality_vs_dense.json` for each candidate mode and `ablation_status.json`; these are diagnostic localization artifacts only.

## Later source-patch fidelity-candidate sequence

1. Import patched backend/transformer with every switch off; confirm the transformer path remains unchanged and Sol-Attn/cache calls are zero.
2. Enable one exact wrapper env at a time: `MINIMAX_H3_A6000_ENABLE_OVERLAY=1`, `MINIMAX_H3_A6000_ENABLE_TRITON_CANDIDATES=1`, plus only one of `MINIMAX_H3_A6000_ENABLE_FUSED_ADALN_MODULATE`, `MINIMAX_H3_A6000_ENABLE_FUSED_ADALN_GATE`, `MINIMAX_H3_A6000_ENABLE_FUSED_ROPE`, or `MINIMAX_H3_A6000_ENABLE_FUSED_SWIGLU`.
3. AdaLN indexed modulation/gate candidate: compare block-level tensors against reference contract for representative video/text/audio/pad tags.
4. RoPE candidate: compare Q/K after qk-norm+RoPE, including pass-through channels.
5. SwiGLU candidate: compare MLP activation output against BF16 rounded reference; vLLM boundary uses `[gate; up]` order.
6. Only after block-level parity, run a single prompt/seed/step differential against the unmodified baseline. No performance claim.

## Later practical Sol-Attn sequence

1. Enable `MINIMAX_H3_A6000_ENABLE_SOL_ATTN=1` only.
2. Keep cache disabled.
3. Confirm first 10 steps and first two layers decline dense.
4. Confirm prefix sink range derives from packed metadata, not an unguarded constant.
5. Confirm unsupported shapes/dtypes/masks/SM decline to dense with telemetry.
6. Record approximation disclosure and do not merge with fidelity-exact results.
