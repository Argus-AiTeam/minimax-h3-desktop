# MiniMax-H3 A6000 Sol-Engine overlay (exact-kernel + Sol-Attn candidate increment)

Status: **default-off standalone Triton exact-kernel candidates, a real SM86 pointer-path Sol-Attn candidate, and a supervised r7 H3 Sol-Attn diagnostic currently pending terminal evidence**. This directory does not modify the locked vLLM-Omni worktree, does not load MiniMax-H3 weights, and does not run Docker/GPU commands during CPU/static work. The external exact-kernel GPU2 gate has passed for exact kernel candidates only. The current H3 Sol-Attn supervisor-selected run is classified by r7 workload/image identity despite a stale `r6` text prefix in the run id; local status is still `running`, so no H3 E2E Sol-Attn pass, speedup, or fidelity claim is made until terminal dense+opt-in HTTP/AV evidence and `sparse_calls>0` telemetry are present.

## What is included

- `src/minimax_h3_a6000/reference_ops.py`: clear PyTorch BF16 reference contracts for AdaLN indexed modulation/gate, RoPE, and SwiGLU.
- `src/minimax_h3_a6000/exact_kernels.py`: real Triton candidate kernels adapted from the Sana Sol-Engine BF16 fusion approach, with Apache attribution, integer-domain BF16 intermediate rounding, explicit shape/dtype/device/SM86 launcher guards (including FP32 `freqs` for RoPE), default-off env gating, reference fallback for unsupported inputs, structured tensor shape/stride telemetry, and opt-in process-local candidate/fallback/decline telemetry export. Indexed modulation/gate prefer stride-aware launches, retain an explicit materialize strategy, and count materialized copy calls/bytes by source tensor. It includes only BF16 exact candidates: indexed modulation, indexed gate+residual, leading-channel H3 RoPE, and SwiGLU.
- `gpu_exact_kernel_test.py`: external single-A6000 correctness harness (`--device cuda:0 --output ...`) with fixed seed, random/extreme/tag-index/non-aligned-tail/representative T-H-D cases, JSON `coverage_tags`, `max_abs`/`max_rel`/`mismatch`/compile status, and no model load.
- `gpu_exact_kernel_bench.py`: external single-A6000 microbenchmark (`warmup >= 20`, `repeats >= 100`) reporting raw PyTorch eager vs Triton candidate latencies as JSON. It is kernel-only, not H3 E2E.
- `src/minimax_h3_a6000/sol_attn_triton_sm86.py`: real pointer-backed Triton Sol-Attn candidate adapted from Sana `triton_ref/preprocess.py` and `triton_ref/fwd.py` for A6000/SM86. It preserves the upstream SM>=8 guard and adds a strict SM86 runtime check; it is imported only after policy/tensor/metadata guards pass.
- `src/minimax_h3_a6000/sol_attn_backend.py`: CPU-testable H3 wrapper for the SM86 Sol-Attn candidate. It enforces packed contiguous BTHD BF16/head_dim=128, valid-length/video-tail metadata, prefix KV sink, prefix-query dense overwrite, first-10-step/first-2-layer dense gates, cache-disabled contract, strict dense fallback, SM86 guard, and density/sink/dense/sparse/fallback telemetry.
- `gpu_sol_attn_sm86_harness.py`: external single-A6000 correctness/bench harness for the Sol-Attn candidate. It loads no model, requires one visible A6000/SM86, verifies no fallback plus prefix-query dense behavior in correctness mode, and emits kernel-only benchmark JSON for the outer GPU2 run.
- `src/minimax_h3_a6000/patch_builder.py` plus `patches/vllm_omni_h3_a6000_opt_in.patch`: repeatable opt-in vLLM-Omni patch source. The patch wires exact wrappers at the H3 transformer's AdaLN modulation/gate, RoPE, and MLP SwiGLU boundaries, logs/export telemetry when enabled, and still leaves vLLM Sol-Attn integration blocked/skeleton because true step/layer/layout propagation has not been safely rebuilt in that patch artifact. Applying the patch is done only in disposable trees/images; CPU/static work only verifies `git apply --check` against the locked source.
- `integration/r4/Dockerfile` and `integration/r4/build_r4_overlay_image.sh`: reproducible r4 overlay image recipe starting from P0 `argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r2`, installing the local overlay, and overwriting patched vLLM-Omni files inside the derived image without editing the locked host source tree. The build script uses no GPU flags and records `gpu_flags=none` when an evidence directory is provided.
- `integration/run_gpu2_exact_integration_5step_r4.sh`: external GPU2-only 5-step same-workload script that first runs dense reference, then enables the three exact wrapper families with Sol-Attn/cache off, verifies AV decode metadata, and requires exact telemetry call/candidate counts plus indexed stride-aware strategy/layout/copy-schema telemetry. The older r3 script remains for historical repeatability only.
- `integration/run_gpu2_exact_ablation_5step_r4.sh`: external GPU2-only diagnostic ablation script for the r4 needs-revision result. It runs dense, indexed modulation only, indexed gate only, RoPE only, SwiGLU only, and all-exact modes; Sol-Attn/cache stay off, split AdaLN envs leave the non-selected AdaLN kernel on the original dense path, and output quality is recorded only as dense-vs-candidate diagnostic JSON.
- `tests/`: CPU/static correctness, patch-check, static guard, CPU launcher-guard behavior, and deferred GPU-gate tests. Real GPU tests are not counted as passed by fallback pytest; launcher behavior tests degrade to an explicit PyTorch-dependency placeholder on hosts without PyTorch.

## Default-off environment switches

Every runtime switch defaults to off:

```text
MINIMAX_H3_A6000_ENABLE_OVERLAY=0
MINIMAX_H3_A6000_ENABLE_FUSED_ADALN=0
MINIMAX_H3_A6000_ENABLE_FUSED_ADALN_MODULATE=0
MINIMAX_H3_A6000_ENABLE_FUSED_ADALN_GATE=0
MINIMAX_H3_A6000_ENABLE_FUSED_ROPE=0
MINIMAX_H3_A6000_ENABLE_FUSED_SWIGLU=0
MINIMAX_H3_A6000_ENABLE_TRITON_CANDIDATES=0
MINIMAX_H3_A6000_EXACT_INDEXED_STRATEGY=auto
MINIMAX_H3_A6000_ABLATION_DISABLE_INDEXED_MODULATE=0
MINIMAX_H3_A6000_ABLATION_DISABLE_INDEXED_GATE=0
MINIMAX_H3_A6000_ABLATION_DISABLE_ROPE=0
MINIMAX_H3_A6000_ABLATION_DISABLE_SWIGLU=0
MINIMAX_H3_A6000_ENABLE_TELEMETRY=0
MINIMAX_H3_A6000_TELEMETRY_ATEXIT=0
MINIMAX_H3_A6000_TELEMETRY_JSON=
MINIMAX_H3_A6000_ENABLE_SOL_ATTN=0
MINIMAX_H3_A6000_SOL_ATTN_CACHE=0
MINIMAX_H3_A6000_SOL_ATTN_STRICT=0
```

The optional Triton candidate gate requires overlay + Triton + per-op env toggles. Launcher SM86 checks occur only when a wrapper is called; importing the package does not initialize CUDA.

## vLLM-Omni integration boundary

The locked source under `runtime/single_a6000_bf16/src/vllm-omni` is treated read-only for this increment. The patch file adds:

1. an opt-in backend enum/file for the Sol-Attn skeleton only (real Sol-Attn vLLM integration remains blocked until denoise step, layer index, packed layout, and valid length are all propagated without defaults); and
2. opt-in exact wrapper calls in `vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py` at `_modulate_scale_shift`, `_modulate_gate`, `MiniMaxH3Attention._apply_rope`, and `MiniMaxH3MLP.forward`.

With all environment variables off, the transformer path is unchanged. If env vars are on but the local wrapper package is unavailable or a launcher guard declines, the patch falls back to the existing PyTorch/vLLM operation. When `MINIMAX_H3_A6000_ENABLE_TELEMETRY=1`, wrapper calls record process-local `calls`/`candidate`/`fallback`/`decline` counters; setting `MINIMAX_H3_A6000_TELEMETRY_JSON` plus `MINIMAX_H3_A6000_TELEMETRY_ATEXIT=1` writes an exit-time JSON summary. Do not treat `H3_A6000_SOL_ATTN` in the current patch as real integration: it is intentionally blocked/skeleton and does not import the new `sol_attn_triton_sm86` candidate.

To rebuild the patch artifact elsewhere without touching runtime:

```bash
PYTHONPATH=ports/minimax_h3_a6000/src \
python -m minimax_h3_a6000.patch_builder --output /tmp/vllm_omni_h3_a6000_opt_in.patch --print-env
```

To check applicability without applying:

```bash
git -C runtime/single_a6000_bf16/src/vllm-omni apply --check \
  ${PWD}/ports/minimax_h3_a6000/patches/vllm_omni_h3_a6000_opt_in.patch
```

## Verification

CPU/static verification for the full port, plus the focused launcher-guard and harness-coverage metadata increment:

```bash
PYTHONPATH=code:ports/minimax_h3_a6000/src python3 -m pytest -q tests ports/minimax_h3_a6000/tests
PYTHONPATH=code:ports/minimax_h3_a6000/src python -m pytest -q \
  ports/minimax_h3_a6000/tests/test_exact_kernel_launchers_cpu.py \
  ports/minimax_h3_a6000/tests/test_exact_kernels_static.py
tmp=$(mktemp -d); PYTHONPYCACHEPREFIX="$tmp" python3 -m py_compile \
  ports/minimax_h3_a6000/src/minimax_h3_a6000/*.py \
  ports/minimax_h3_a6000/gpu_exact_kernel_test.py \
  ports/minimax_h3_a6000/gpu_exact_kernel_bench.py \
  ports/minimax_h3_a6000/gpu_sol_attn_sm86_harness.py; rm -rf "$tmp"
git -C runtime/single_a6000_bf16/src/vllm-omni apply --check \
  ${PWD}/ports/minimax_h3_a6000/patches/vllm_omni_h3_a6000_opt_in.patch
```

External exact-kernel GPU evidence already absorbed (kernel-only, not H3 E2E): `${PWD}/technical_report/evidence/minimax_h3_desktop/sol_engine_port/gpu_exact_20260809T155451Z`. Correctness JSON: 8/8 cases `compiled_and_launched`, all `max_abs=0`, `max_rel=0`, `mismatch=0`. Microbenchmark median speedups: indexed modulation 22.02x, indexed gate 11.66x, RoPE 6.50x, SwiGLU 8.09-8.11x. These are raw kernel candidate timings only.

Current H3 Sol-Attn r7 CPU-only ingest: `technical_report/evidence/minimax_h3_desktop/delivery/r7_sol_attn_cpu_ingest_20260811T110523Z/r7_terminal_classification.json`. It records terminal supervisor status `complete`, selected run `sol_engine_port/sol_attn_h3_gpu2_5step_r6_20260811T110523Z`, readable r7 workload/version-label provenance, valid dense/opt-in HTTP+AV, and `sparse_calls=0`; classification remains fail-closed (`fail_closed_missing_metadata`) because Sol-Attn never entered a sparse runtime path. Opaque image/output identifiers are omitted and are not classification evidence.

External commands for repeat or later integration gates (not run by CPU/static tasks):

```bash
PYTHONPATH=ports/minimax_h3_a6000/src \
python3 ports/minimax_h3_a6000/gpu_exact_kernel_test.py --device cuda:0 --output /tmp/h3_exact_correctness.json
PYTHONPATH=ports/minimax_h3_a6000/src \
python3 ports/minimax_h3_a6000/gpu_exact_kernel_bench.py --device cuda:0 --output /tmp/h3_exact_bench.json
PYTHONPATH=ports/minimax_h3_a6000/src \
python3 ports/minimax_h3_a6000/gpu_sol_attn_sm86_harness.py --device cuda:0 --mode both --output /tmp/h3_sol_attn_sm86.json
ROOT=${PWD} \
GPU_INDEX=2 \
ports/minimax_h3_a6000/integration/run_gpu2_exact_ablation_5step_r4.sh
```
