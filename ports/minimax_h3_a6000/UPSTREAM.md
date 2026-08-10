# Upstream attribution and file mapping

This overlay is Apache-2.0-compatible original adaptation/reference code. It was authored after reading the pinned local sources below and does not copy NVLabs team work as personal work.

## Pinned sources read

- Sana/Sol-Engine: `upstreams/Sana-sol-engine` at `d00eef311670a58deb2c323fe072738fcb945600`.
  - A100 H3 source: `models/minimax_h3/a100/model.py`, `models/minimax_h3/a100/SOURCE_SNAPSHOT.json`.
  - RTX5090 H3 source: `models/minimax_h3/rtx5090/model.py`, `models/minimax_h3/rtx5090/SOURCE_SNAPSHOT.json`.
  - GB10 H3 source: `models/minimax_h3/gb10/adaln.py`, `models/minimax_h3/gb10/fusions.py`, `models/minimax_h3/gb10/transformers_compat.py`, `models/minimax_h3/gb10/SOURCE_SNAPSHOT.json`.
  - Sol-Attn source: `techniques/sparse_backends/sol_attn/interface.py`, `techniques/sparse_backends/sol_attn_backend.py`, `techniques/sparse_backends/sol_attn/triton_ref/preprocess.py`, `techniques/sparse_backends/sol_attn/triton_ref/fwd.py`, `techniques/sparse_backends/sol_attn/common/runtime.py`.
- vLLM-Omni: `runtime/single_a6000_bf16/src/vllm-omni` at `8e2e9b6b53e86e6a479ed2c0a53782f655f60e04`.
  - H3 transformer/packing: `vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py`, `packed_sequence.py`.
  - Attention interface: `vllm_omni/diffusion/attention/backends/abstract.py`, `selector.py`, `backends/registry.py`, `backends/sdpa.py`.
  - Offloader interface: `vllm_omni/diffusion/offloader/offload_plan.py`.

## Overlay file mapping

| Overlay file | Status | Upstream relationship |
|---|---|---|
| `src/minimax_h3_a6000/reference_ops.py` | Original adaptation/reference | Implements PyTorch reference contracts derived from the operation order described in Sana `gb10/fusions.py` and vLLM-Omni H3 RoPE/AdaLN/SwiGLU boundaries. |
| `src/minimax_h3_a6000/exact_kernels.py` | Adapted Triton candidate code | Implements BF16 indexed modulation, indexed gate+residual, leading-channel H3 RoPE, and SwiGLU using the Sana exact-BF16 rounding technique and Apache attribution. It intentionally does not include quantized checkpoint or pruned-model paths. |
| `gpu_exact_kernel_test.py`, `gpu_exact_kernel_bench.py` | Original external GPU harnesses | Single-visible-A6000/SM86 correctness and microbenchmark drivers; no model load. |
| `src/minimax_h3_a6000/sol_attn_triton_sm86.py` | Adapted Triton candidate code | Ports the Sana pointer-backed `triton_ref/preprocess.py` and `triton_ref/fwd.py` Sol-Attn kernels into the local A6000/SM86 overlay, preserving the upstream SM>=8 guard and adding a stricter SM86 runtime check. |
| `src/minimax_h3_a6000/sol_attn_backend.py` | Original H3 policy/wrapper adaptation | Defines packed H3 metadata, dense-first policy, prefix KV sink, prefix-query dense overwrite, strict fallback, and telemetry around the real local SM86 candidate. |
| `gpu_sol_attn_sm86_harness.py` | Original external GPU harness | Single-visible-A6000/SM86 Sol-Attn correctness/benchmark driver; no model load. |
| `src/minimax_h3_a6000/patch_builder.py` | Original tool | Copies the patch artifact without touching runtime. |
| `patches/vllm_omni_h3_a6000_opt_in.patch` | Original integration patch | Adds an opt-in vLLM-Omni backend skeleton/registry enum and wires opt-in exact wrappers into the H3 transformer AdaLN, RoPE, and SwiGLU boundaries for a future disposable tree. |
| `tests/*.py` | Original tests | CPU/static tests for reference contracts, kernel source/guards, patch applicability, and deferred GPU gates. |
| `README.md`, `correctness_contract.md`, `gpu_test_plan.md`, `NOTICE`, `UPSTREAM.md` | Original documentation | Attribution and operational contract for this overlay. |

## Non-included upstream code

No upstream model weights, CUDA/CuTe kernels, Docker images, or generated outputs are included in this overlay. The local Sol-Attn SM86 Triton candidate is source code adapted from the pinned Apache-2.0 Sana files listed above; any future application of the patch must preserve upstream license notices in the target vLLM-Omni tree.
