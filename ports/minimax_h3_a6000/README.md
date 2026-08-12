# MiniMax-H3 A6000 Sol-Engine overlay (exact-kernel + Sol-Attn candidate increment)

Status: **default-off standalone Triton exact-kernel candidates, a real SM86 pointer-path Sol-Attn candidate, terminal r8 H3 Sol-Attn 5-step sparse-execution diagnostic evidence, a terminal N=3 matched-workload route gate, and a terminal formal N>=10 Sol-Attn matched-workload acceptance**. This directory does not modify the locked vLLM-Omni worktree, does not load MiniMax-H3 weights during CPU/static work, and keeps all runtime hooks default-off/fail-closed. The selected H3 Sol-Attn 5-step diagnostic run is terminal (`complete`, exit code `0`) and is attributed by readable r8 workload/version-label provenance (image tag/version/title and `H3_A6000_SOL_ATTN` workload), not opaque image/output identifier equality. The r8 diagnostic recorded dense and opt-in HTTP 200 plus structural AV evidence, `sparse_candidate_calls=192`, `sparse_calls=192`, `fallback_calls=0`, 192 density samples, and materialized-copy/resource telemetry; therefore the old r7 `fail_closed_missing_metadata` blocker is cleared for the fixed 5-step metadata gate. The r8 N=3 matched-workload route gate is terminal at `technical_report/evidence/minimax_h3_desktop/sol_engine_port/sol_attn_h3_matched_retest_r8_n3_20260812T013544Z/decision.json`: all three dense/opt-in pairs completed with HTTP 200, structural AV, sparse_calls=192/fallback_calls=0 per opt-in pair, median HTTP-time improvement 14.782455716069165% over a >3.0% route threshold, and resource envelope within the predeclared gate. The formal N>=10 run directory `technical_report/evidence/minimax_h3_desktop/sol_engine_port/sol_attn_h3_formal_n10_r8_n10_20260812T031757Z` is terminal with `formal_classification=accepted_formal_n10_same_gpu_sol_attn_speed_candidate`: 10/10 dense/opt-in pairs completed on the expected physical GPU, all HTTP/structural-AV/sparse/resource/quality gates passed, every opt-in pair recorded `sparse_calls=192` and `fallback_calls=0`, and the median HTTP-time improvement was 15.203295894% over a >3.0% threshold. This acceptance is confined to the formal matched 5-step Sol-Attn opt-in lane; it is not BF16 fidelity, Turbo/DLO/DMD evidence, public release approval, human-auditory/semantic quality certification, or quality-equivalence certification.

## What is included

- `src/minimax_h3_a6000/reference_ops.py`: clear PyTorch BF16 reference contracts for AdaLN indexed modulation/gate, RoPE, and SwiGLU.
- `src/minimax_h3_a6000/exact_kernels.py`: real Triton candidate kernels adapted from the Sana Sol-Engine BF16 fusion approach, with Apache attribution, integer-domain BF16 intermediate rounding, explicit shape/dtype/device/SM86 launcher guards (including FP32 `freqs` for RoPE), default-off env gating, reference fallback for unsupported inputs, structured tensor shape/stride telemetry, and opt-in process-local candidate/fallback/decline telemetry export. Indexed modulation/gate prefer stride-aware launches, retain an explicit materialize strategy, and count materialized copy calls/bytes by source tensor. It includes only BF16 exact candidates: indexed modulation, indexed gate+residual, leading-channel H3 RoPE, and SwiGLU.
- `gpu_exact_kernel_test.py`: external single-A6000 correctness harness (`--device cuda:0 --output ...`) with fixed seed, random/extreme/tag-index/non-aligned-tail/representative T-H-D cases, JSON `coverage_tags`, `max_abs`/`max_rel`/`mismatch`/compile status, and no model load.
- `gpu_exact_kernel_bench.py`: external single-A6000 microbenchmark (`warmup >= 20`, `repeats >= 100`) reporting raw PyTorch eager vs Triton candidate latencies as JSON. It is kernel-only, not H3 E2E.
- `src/minimax_h3_a6000/sol_attn_triton_sm86.py`: real pointer-backed Triton Sol-Attn candidate adapted from Sana `triton_ref/preprocess.py` and `triton_ref/fwd.py` for A6000/SM86. It preserves the upstream SM>=8 guard and adds a strict SM86 runtime check; it is imported only after policy/tensor/metadata guards pass.
- `src/minimax_h3_a6000/sol_attn_backend.py`: CPU-testable H3 wrapper for the SM86 Sol-Attn candidate. It enforces packed contiguous BTHD BF16/head_dim=128, valid-length/video-tail metadata, prefix KV sink, prefix-query dense overwrite, first-10-step/first-2-layer dense gates, cache-disabled contract, strict dense fallback, SM86 guard, and density/sink/dense/sparse/fallback telemetry.
- `gpu_sol_attn_sm86_harness.py`: external single-A6000 correctness/bench harness for the Sol-Attn candidate. It loads no model, requires one visible A6000/SM86, verifies no fallback plus prefix-query dense behavior in correctness mode, and emits kernel-only benchmark JSON for the outer GPU2 run.
- `src/minimax_h3_a6000/patch_builder.py` plus `patches/vllm_omni_h3_a6000_opt_in.patch`: repeatable opt-in vLLM-Omni patch source. The patch wires exact wrappers at the H3 transformer's AdaLN modulation/gate, RoPE, and MLP SwiGLU boundaries, logs/export telemetry when enabled, adds the opt-in H3 Sol-Attn backend, and propagates source-backed DiT packed-video layout, valid length, denoise step, and layer index while failing closed for missing/inconsistent metadata. Applying the patch is done only in disposable trees/images; CPU/static work only verifies `git apply --check` against the locked source.
- `integration/r4/Dockerfile` and `integration/r4/build_r4_overlay_image.sh`: reproducible r4 overlay image recipe starting from P0 `argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r2`, installing the local overlay, and overwriting patched vLLM-Omni files inside the derived image without editing the locked host source tree. The build script uses no GPU flags and records `gpu_flags=none` when an evidence directory is provided.
- `integration/run_gpu2_exact_integration_5step_r4.sh`: external GPU2-only 5-step same-workload script that first runs dense reference, then enables the three exact wrapper families with Sol-Attn/cache off, verifies AV decode metadata, and requires exact telemetry call/candidate counts plus indexed stride-aware strategy/layout/copy-schema telemetry. The older r3 script remains for historical repeatability only.
- `integration/run_gpu2_exact_ablation_5step_r4.sh`: external GPU2-only diagnostic ablation script for the r4 needs-revision result. It runs dense, indexed modulation only, indexed gate only, RoPE only, SwiGLU only, and all-exact modes; Sol-Attn/cache stay off, split AdaLN envs leave the non-selected AdaLN kernel on the original dense path, and output quality is recorded only as dense-vs-candidate diagnostic JSON.
- `tests/`: CPU/static correctness, patch-check, static guard, CPU launcher-guard behavior, and deferred GPU-gate tests. Real GPU tests are not counted as passed by fallback pytest; launcher behavior tests degrade to an explicit PyTorch-dependency placeholder on hosts without PyTorch.

## Default-off environment switches

Every enable switch defaults to off; non-boolean policy knobs keep conservative defaults:

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
MINIMAX_H3_A6000_SOL_ATTN_DENSE_FIRST_STEPS=10
MINIMAX_H3_A6000_SOL_ATTN_DENSE_FIRST_LAYERS=2
MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_MATERIALIZE=0
MINIMAX_H3_A6000_SOL_ATTN_MATERIALIZE_MAX_BYTES=67108864
```

The optional Triton candidate gate requires overlay + Triton + per-op env toggles. Launcher SM86 checks occur only when a wrapper is called; importing the package does not initialize CUDA.

## vLLM-Omni integration boundary

The locked source under `runtime/single_a6000_bf16/src/vllm-omni` is treated read-only for this increment. The patch file adds:

1. an opt-in backend enum/file for the Sol-Attn path with source-backed denoise step, layer index, packed video layout, and valid length metadata; and
2. opt-in exact wrapper calls in `vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py` at `_modulate_scale_shift`, `_modulate_gate`, `MiniMaxH3Attention._apply_rope`, and `MiniMaxH3MLP.forward`.

With all environment variables off, the transformer path is unchanged. If env vars are on but the local wrapper package is unavailable or a launcher guard declines, the patch falls back to the existing PyTorch/vLLM operation. When `MINIMAX_H3_A6000_ENABLE_TELEMETRY=1`, wrapper calls record process-local `calls`/`candidate`/`fallback`/`decline` counters; setting `MINIMAX_H3_A6000_TELEMETRY_JSON` plus `MINIMAX_H3_A6000_TELEMETRY_ATEXIT=1` writes an exit-time JSON summary. The current r8 evidence may be cited as fixed 5-step sparse-execution diagnostic evidence and, separately, as a terminal formal N>=10 matched-workload Sol-Attn opt-in lane acceptance. It must not be used as BF16 fidelity, Turbo/DLO/DMD evidence, release approval, human-auditory/semantic quality certification, or quality-equivalence certification; fail-closed dense fallback or timing jitter is never a speedup claim.

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

Clean-room deployment/lifecycle evidence is now separated from speed and quality claims. The gated non-dry local lifecycle verifier passed in `technical_report/evidence/minimax_h3_desktop/delivery/local_lifecycle_clean_room_20260812T014824Z`: a sanitized export was built, existing local FL2VA resources were inspected read-only, the locked runtime image tag/id matched, the CPU verifier fixture ran, and publication audit passed on the export root. No container was started, no model weights were loaded or modified, no GPU inference/media generation ran, and no speed/fidelity/quality claim is created by this packaging gate.

Final delivery gate evidence is recorded in `technical_report/evidence/minimax_h3_desktop/delivery/final_cpu_static_gate_20260812T020013Z`, `technical_report/evidence/minimax_h3_desktop/delivery/final_decisive_export_audit_20260812T025605Z`, and the latest post-review `formal_n10_cpu_sync_export_audit_*` directory: CPU/static tests, py_compile/patch-apply/fixture/Turbo-dry-run checks, independent Reviewer acceptance for the bounded formal-N10 lane, strict aggregation, sanitized export build, and publication audit pass from already-written evidence only. The current-stage delivery Reviewer evidence recognition repair packet `technical_report/evidence/minimax_h3_desktop/delivery/delivery_reviewer_evidence_recognition_repair_20260812T085911Z` is now first-class in the strict delivery summary, final report, and package manifest: it carries a fresh current_stage=`delivery` Reviewer verdict, a valid sealed Reviewer handoff source, and manager_recognition_check ready=True/mismatch_count=0 while preserving the operator-only Turbo listening/semantic AV-sync residual. The current Manager-hold packet is also surfaced in the strict delivery summary and manifest; the latest packet `technical_report/evidence/minimax_h3_desktop/delivery/current_manager_hold_no_gap_probe_20260812T110024Z` now explicitly names the remaining current-stage Reviewer-evidence gap: fresh independent `reviewer_verdict.json` and `manager_reviewer_handoff_crosswalk.json` are absent, so the Manager-visible fields remain `reviewer_status=pending_fresh_independent_reviewer`, independent/sealed-handoff/manager-visible-sync false until a Reviewer writes or rejects those files. These are release-readiness and Manager-visibility gates only; they do not turn r8 Sol-Attn diagnostics or the formal N>=10 opt-in lane into BF16 fidelity, public release approval, human-quality certification, or quality-equivalence claims, and the repair did not run GPU/Docker/model inference or subjective listening.

CPU/static verification for the full port, plus the focused launcher-guard and harness-coverage metadata increment:

```bash
PYTHONPATH=code:.:ports/minimax_h3_a6000/src python3 -m pytest -q tests ports/minimax_h3_a6000/tests
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

Current H3 Sol-Attn r8 CPU-only ingest: `technical_report/evidence/minimax_h3_desktop/delivery/r8_sol_attn_cpu_ingest_20260812T005600Z/r8_terminal_classification.json`. It records terminal supervisor status `complete`, selected run `sol_engine_port/sol_attn_h3_gpu2_5step_r8_prompt0644_20260812T005600Z`, readable r8 workload/version-label provenance, valid dense/opt-in HTTP+AV, and sparse runtime telemetry (`sparse_candidate_calls=192`, `sparse_calls=192`, `fallback_calls=0`, density samples=192, materialized copies=192 / 105344139264 bytes). Classification is `sparse_runtime_valid_5step_diagnostic`: the r7 missing-metadata blocker is cleared for this fixed 5-step gate, but that diagnostic alone is not a speedup, N10, BF16 fidelity, release, or quality-equivalence claim. The matched-workload r8 follow-up is terminal at `technical_report/evidence/minimax_h3_desktop/sol_engine_port/sol_attn_h3_matched_retest_r8_n3_20260812T013544Z/decision.json`, with CPU-only posthoc finalization documented in `technical_report/evidence/minimax_h3_desktop/sol_engine_port/sol_attn_h3_matched_retest_r8_n3_20260812T013544Z/posthoc_finalization_note.json` after the supervisor completed all three pairs and then failed while emitting summaries. Classification is `proceed_to_formal_n10_candidate`. The formal N>=10 run is terminal at `technical_report/evidence/minimax_h3_desktop/sol_engine_port/sol_attn_h3_formal_n10_r8_n10_20260812T031757Z/formal_n10_decision.json` with `formal_classification=accepted_formal_n10_same_gpu_sol_attn_speed_candidate`; `formal_n10_summary.json`, `RUN_REPORT.md`, `timing_summary.json`, `quality_proxy_comparison.json`, and `resource_summary.json` summarize the accepted formal lane and its limits. Historical r7 fail-closed evidence remains at `technical_report/evidence/minimax_h3_desktop/delivery/r7_sol_attn_cpu_ingest_20260811T110523Z/r7_terminal_classification.json`.

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
