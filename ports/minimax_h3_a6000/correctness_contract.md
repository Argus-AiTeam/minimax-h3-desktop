# Correctness contract

This increment defines **default-off A6000/SM86 exact-kernel candidates plus a reference differential contract**. The external single-A6000 exact-kernel harness has now been run and reviewed for these standalone kernels only; it does not declare MiniMax-H3 end-to-end output fidelity, speed, fit, or quality.

## BF16 reference operations

All payload tensors for reference ops are `torch.bfloat16`. Intermediate operations are rounded to BF16 precision at the same semantic store points the audited Sana sources identify:

1. **AdaLN indexed modulation**: `x * (1 + scale[index]) + shift[index]`
   - round `1 + scale[index]` to BF16 precision;
   - round the product to BF16 precision;
   - round the final add to BF16 precision.
2. **AdaLN indexed gate**: `residual + gate[index] * branch`
   - round the gate/product to BF16 precision;
   - round the residual add to BF16 precision.
3. **RoPE**:
   - rotate only the first `rotary_dim` head channels;
   - pass non-rotary channels through unchanged;
   - cast `cos`/`sin` to BF16 precision before multiplication;
   - round products and final add to BF16 precision.
4. **SwiGLU**:
   - split `[value; gate]` on the final dimension;
   - compute `silu(gate)`, round to BF16 precision;
   - multiply by value and round to BF16 precision.

The CPU tests cover random tensors, explicit midpoint/large/small values, shape and tag/index validation, and hand-written rounding expectations. The Triton launchers in `exact_kernels.py` use the same BF16 rounding contract and fall back to these references when disabled or unsupported.

## Triton candidate guard contract

Each public launcher is default-off and checks:

- overlay + candidate + per-op environment switch, unless an external harness passes `enable=True`;
- CUDA tensor device and SM86 capability at launcher time only;
- BF16 payload dtype;
- contiguous tensors and same device;
- table/index shape compatibility for indexed AdaLN operations;
- valid `[T,H,D]` or `[B,T,H,D]` H3 RoPE with FP32 `freqs` and even leading rotary dimension;
- even final dimension for SwiGLU.

Unsupported cases return the PyTorch reference unless `strict=True` is explicitly requested by the external harness. Importing the package must not initialize CUDA. Telemetry is default-off; when enabled it records process-local wrapper calls plus candidate/fallback/decline/strict-error counters and can export JSON directly or at process exit.

## External exact-kernel GPU gate

`gpu_exact_kernel_test.py` is the correctness harness for the standalone Triton candidates. It must be run with `--device cuda:0 --output ...` after confirming exactly one visible RTX A6000/SM86 GPU. It uses a fixed seed and covers random values, explicit extremes injected into every op family, tag index edges, non-aligned tails, and representative H3 T/H/D shapes. Its JSON output records top-level `coverage_tags` plus per-case `compile_status`, `max_abs`, `max_rel`, `mismatch`, and `numel`. The absorbed evidence at `technical_report/evidence/minimax_h3_desktop/sol_engine_port/gpu_exact_20260809T155451Z/correctness.json` passed 8/8 cases with `compiled_and_launched` and zero `max_abs`/`max_rel`/`mismatch`. A fallback pytest skip is not evidence of GPU correctness.

## Practical Sol-Attn SM86 candidate

Sol-Attn is treated as `practical_disclosed_approx` unless a later proof changes the contract. The local overlay now includes a real pointer-backed Triton candidate adapted from Sana for A6000/SM86, but it remains default-off and must not be represented as an exact/fidelity-preserving path. The wrapper enforces:

- packed H3 metadata: prefix length, `(t, h, w)` video grid, valid length, total padded length;
- contiguous BF16 BTHD tensors with `head_dim=128` and tensor length matching `total_length`;
- exact KV sink range derived from the prefix by default;
- prefix query rows overwritten by dense SDPA over the valid sequence;
- dense-first denoising steps and dense-first layers;
- SM86 candidate guard supplied by a later GPU gate, never probed at import;
- density/sink/dense/sparse/fallback telemetry;
- strict dense fallback for unsupported devices, shapes, missing metadata, non-contiguous tensors, kernel import/launch errors, or cache requests;
- cache disabled by contract.

With `MINIMAX_H3_A6000_ENABLE_SOL_ATTN=0` (default), every call declines with `env_disabled` and returns dense reference attention. With the flag on, CPU/static tests still decline on `unsupported_device`; external GPU2 must run `gpu_sol_attn_sm86_harness.py` to prove compile/launch/no-fallback behavior on one visible A6000.

## Future GPU gate required before any runtime or exactness claim

Required evidence before enabling any candidate by default or making a model-level fidelity/performance claim:

- already satisfied for standalone kernels: `gpu_exact_kernel_test.py` JSON showing Triton compile/launch and zero mismatches against PyTorch BF16 references on one visible A6000 SM86;
- already satisfied for standalone kernels: `gpu_exact_kernel_bench.py` raw latency JSON for PyTorch eager vs Triton candidates, without H3 E2E claims; median speedups were modulation 22.02x, gate 11.66x, RoPE 6.50x, and SwiGLU 8.09-8.11x;
- same model/runtime/prompt/seed/step contract for any later model-level differential;
- no hidden cache in fidelity runs;
- source patch applied to a new disposable worktree or image, not the locked runtime tree;
- exact-wrapper integration telemetry proving call counts greater than zero at the real H3 transformer boundaries;
- raw density/sink/dense/sparse/fallback telemetry for practical Sol-Attn if that backend is selected;
- explicit statement of any approximation source.
