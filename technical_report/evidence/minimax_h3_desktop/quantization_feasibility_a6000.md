# MiniMax-H3 FL2VA quantization feasibility audit for RTX A6000 / SM86

Status: **static_audit_no_runnable_quantized_candidate**.

This is a CPU/code-only audit. It did **not** run GPU commands, Docker commands, model loading, model conversion, inference, network access, downloads, publication, or deployment. It inspects locked local sources, local configuration metadata, and already-produced evidence only.

## Verdict

**Runnable no-new-download quantized candidate for full MiniMax-H3 FL2VA on RTX A6000/SM86: `NO`.**

The locked local project contains a source-supported BF16/layerwise-offload path and diagnostic BF16/Sol-Engine-derived kernel candidates, but it does not contain a runnable full-H3 quantized A6000 deployment candidate. The strongest quantization-related source evidence is the Sol-Engine GB10 FP8 path, but that path depends on a pruned FP8 DiT checkpoint plus an external FP8 Qwen3-VL conditioner and is written/validated for a different target lane. The locked A6000 overlay intentionally keeps only BF16-exact elementwise kernels and does not implement checkpoint layout conversion or quantized model loading.

Therefore any A6000 quantization work must stay in the `practical_disclosed_approx` lane until a converter, loader/runtime integration, hardware-specific kernel support, and AV/semantic quality gates all pass. It must not be described as deployed, validated, BF16-exact, or source-supported today.

## Evidence inspected

| Evidence | Local path | Finding used in this audit |
|---|---|---|
| Runtime/source audit | `technical_report/evidence/minimax_h3_desktop/runtime_primary_source_audit.md` | The selected one-card candidate is vLLM-Omni BF16 DLO with tiled VAE and cuDNN attention; status remains `built_zero_gpu_not_a6000_validated`, not an A6000 deployment/result. |
| Baseline/track policy | `technical_report/evidence/minimax_h3_desktop/baseline_parameter_lock.json` | `fidelity_bf16_exact` forbids weight/cache approximation; quantization is allowed only as explicitly disclosed `practical_disclosed_approx` and requires a quality gate. Local FL2VA metadata totals 144,051,182,625 bytes. |
| Local model configs | `models/MiniMax-H3/FL2VA/model_index.json`, `models/MiniMax-H3/FL2VA/transformer/config.json`, `models/MiniMax-H3/FL2VA/text_encoder/config.json` | FL2VA serves `t2va`/`fl2va`; transformer config is a BF16-oriented 50-layer DiT shape (`hidden_size=5376`, `num_attention_heads=56`, `attention_head_dim=128`); text encoder config says `dtype=bfloat16`. Static grep over FL2VA JSON configs found no `quantization_config`, `float8`, `fp8`, `int8`, `awq`, `gptq`, `bitsandbytes`, `nf4`, or `fp4` metadata. |
| Download integrity | `technical_report/evidence/minimax_h3_desktop/model_prep/download_integrity_FL2VA.json` | Local expected files are complete for FL2VA: 81 files, 29 LFS-weight files, 144,051,182,625 bytes, no missing or unexpected files. This is BF16 FL2VA evidence, not quantized-checkpoint evidence. |
| Locked runtime | `technical_report/evidence/minimax_h3_desktop/runtime_lock_candidate.json`, `runtime/single_a6000_bf16/source_commit.json`, `runtime/single_a6000_bf16/requirements.lock` | Runtime is pinned to vLLM-Omni `8e2e9b6...`, torch `2.11.0+cu130`, CUDA userspace `13.0`; it includes generic quantization-related packages, but the selected H3 route and lock are explicitly BF16 DLO and zero-GPU validated only. |
| vLLM-Omni H3 loader | `runtime/single_a6000_bf16/src/vllm-omni/vllm_omni/diffusion/models/minimax_h3/*.py` | H3 DiT code accepts a `QuantizationConfig`, but H3 uses grouped QKV/fused MLP checkpoint conversions, `packed_modules_mapping = {}`, FP32 keep-lists, and `_supports_mmap_loading = False`; these are integration constraints, not a runnable quantized path. |
| A6000 overlay | `ports/minimax_h3_a6000/README.md`, `ports/minimax_h3_a6000/src/minimax_h3_a6000/exact_kernels.py`, `ports/minimax_h3_a6000/src/minimax_h3_a6000/env.py` | Overlay is default-off and BF16-exact only: indexed AdaLN modulation/gate, RoPE, and SwiGLU. It explicitly says checkpoint layout conversion paths are not implemented. Sol-Attn integration remains diagnostic/pending, not deployment. |
| Sol-Engine inventory | `technical_report/evidence/minimax_h3_desktop/sol_engine_port/h3_source_files.txt`, `h3_source_sha256.txt` | The local Sol-Engine source inventory includes GB10 FP8, GB200, RTX5090, H100/A100 source snapshots; there is no A6000-specific full quantization runtime in the inventory. |
| Sol-Engine GB10 FP8 source | `upstreams/Sana-sol-engine/models/minimax_h3/gb10/{README.md,gpu_infer.py,build.py,fp8_linear.py,paths.py}` | GB10 path uses a pruned FP8 DiT and FP8 Qwen3-VL conditioner. `Fp8Linear` uses `torch.float8_e4m3fn` and `torch._scaled_mm`; `fc2` is treated as weight-only when the checkpoint lacks `input_scale`. It is source evidence for a different quantized lane, not a no-new-download A6000 candidate. |
| Sol-Engine baseline/optimized source | `upstreams/Sana-sol-engine/models/minimax_h3/gb200/baseline/README.md`, `upstreams/Sana-sol-engine/models/minimax_h3/gb200/optimized.toml`, `upstreams/Sana-sol-engine/models/minimax_h3/rtx5090/README.md` | GB200 baseline is explicitly no quantization/offload and resident in 186 GB. GB200 optimized uses context parallelism/Sol-Attn/cache/VAE acceleration, not full-H3 quantization. RTX5090 source says released BF16 FL2VA plus layerwise component offload, not quantized weights. |
| Performance report | `technical_report/minimax_h3_a6000_performance.md` | Existing A6000 BF16 exact denominator and practical Turbo evidence are not quantization evidence. Sol-Attn is pending/not deployed; DLO candidate-50 remains pending. |

## BF16 fidelity boundary

The fidelity boundary is clear: BF16 exact results are the only evidence eligible for the `fidelity_bf16_exact` lane. `baseline_parameter_lock.json` states that weight quantization and cache approximation are not allowed in this lane. The same file permits quantization only in `practical_disclosed_approx`, with an explicit quality gate.

The locked runtime and A6000 overlay both preserve that boundary:

- `runtime_primary_source_audit.md` selects vLLM-Omni one-GPU DLO as a BF16 candidate, not a quantized candidate.
- `ports/minimax_h3_a6000/src/minimax_h3_a6000/exact_kernels.py` states that the A6000 candidate keeps only BF16-exact elementwise kernels and does not implement checkpoint layout conversion.
- `technical_report/minimax_h3_a6000_performance.md` separates BF16 exact, Turbo practical approximation, exact-kernel diagnostics, Sol-Attn diagnostics, and DLO pending stages.

Conclusion: any quantized MiniMax-H3 FL2VA A6000 result would be a new practical approximation, not a BF16 fidelity result and not a deployment claim until separately validated.

## Practical SM86 quantization candidates

| Candidate | Local evidence | A6000/SM86 no-new-download runnable? | Reason |
|---|---|---:|---|
| BF16 DLO with tiled VAE | vLLM-Omni runtime audit and runtime lock | No — not quantization | This is the current source-supported one-card route, but it is BF16/layerwise offload. It should remain the fidelity/practical runtime base, not be relabeled as quantized. |
| vLLM generic quantization of H3 DiT linears | H3 DiT constructor accepts `QuantizationConfig`; runtime has generic quant packages | No | H3 fused QKV/MLP naming and loader conversions need direct integration; local FL2VA configs contain no quantization metadata or quantized weight shards; no converter artifact exists; no SM86 quant kernels or quality evidence exist. |
| Offline INT8-style conversion from local BF16 FL2VA | Local BF16 FL2VA is complete | No | Technically plausible as future R&D, but no converter, manifest, loader patch, scale policy, calibration policy, or AV quality result exists. A converted checkpoint would be a newly generated artifact and must be audited before launch. |
| Sol-Engine GB10 FP8 path | GB10 `README.md`, `gpu_infer.py`, `build.py`, `fp8_linear.py`, `paths.py` | No | It depends on a pruned FP8 DiT and external FP8 Qwen3-VL conditioner, not the local BF16 FL2VA-only checkpoint. It is written for GB10/unified memory and uses FP8 `_scaled_mm`; it is not an A6000/SM86 source-supported route. |
| FP4/Blackwell-style low precision | No local A6000 H3 FP4 source or checkpoint metadata | No | No local source provides an FP4 H3 checkpoint, converter, A6000 kernels, or quality gate. Treating Hopper/Blackwell low-precision assumptions as SM86 support would be unsupported. |

The only practical candidate worth preserving as a future branch is **offline conversion from the local BF16 FL2VA checkpoint plus explicit vLLM-Omni H3 loader/runtime support**, but it is not runnable today.

## Rejected FP8/FP4 and Hopper/Blackwell assumptions

Rejected assumptions:

1. **“FP8 exists in Sol-Engine, so A6000 can run full H3 FP8 now.”** Rejected. The inspected FP8 implementation is under the GB10 source tree, uses a pruned FP8 DiT plus a separate FP8 conditioner, and its own comments tie behavior/performance to that lane. The A6000 overlay does not include FP8 conversion or quantized loaders.
2. **“The local BF16 FL2VA weights can be served as quantized weights without conversion.”** Rejected. The local FL2VA JSON configs contain no quantization metadata, and the downloaded FL2VA evidence is the complete BF16 partition, not quantized shards.
3. **“Generic vLLM quantization package presence is enough.”** Rejected. `requirements.lock` includes packages such as `bitsandbytes` and `compressed-tensors`, but H3-specific loader, fused-layer mapping, scale metadata, and post-load precision checks still need source-level support and validation.
4. **“FP4/Blackwell low precision can be assumed for SM86.”** Rejected. The target is RTX A6000/SM86, and no local H3 source or evidence provides an SM86 FP4 path. This audit does not infer support from unrelated architectures.
5. **“FP8/fused sparse attention evidence implies AV quality.”** Rejected. Sol-Attn and exact-kernel diagnostics in the performance report are not quantized full-H3 deployments, and quality certification remains a separate gate.

## vLLM-Omni support gaps

vLLM-Omni is still the best locked runtime base, but the inspected source shows gaps before quantized H3 can be called runnable:

- H3 DiT accepts a `QuantizationConfig`, but the checkpoint already stores grouped QKV and MLP gate/up as fused tensors, and `packed_modules_mapping = {}`. Ignored or converted layers must target fused names directly.
- Several H3 parameters and buffers must stay FP32 after load (`MINIMAX_H3_FP32_PARAM_NAMES`, `MINIMAX_H3_FP32_BUFFER_NAMES`), so a blanket full-model conversion would be invalid.
- The pipeline sets `_supports_mmap_loading = False` because the regular loader performs grouped-QKV reorder and fused-MLP packing before sharding; a quantized/offloaded loader must preserve those callbacks rather than bypass them.
- The selected runtime lock is BF16 DLO and zero-GPU validated. It has not demonstrated CUDA extension import success, A6000 fit, A6000 latency, A6000 quality, or GPU CLI success for a quantized H3 path.
- No local artifact defines an H3 component quantization manifest covering transformer, text encoder, video VAE, audio VAE, scale tensors, ignored layers, dtype exceptions, and offload storage policy.

## Sana/Sol-Engine FP8/quant relevance

Sol-Engine remains useful as source evidence, but only as a design reference for future practical approximation work:

- GB10 `fp8_linear.py` documents per-layer FP8 E4M3 metadata and honors `full_precision_matrix_mult` for `fc2`, avoiding invented activation scales.
- GB10 `build.py` constructs the DiT on `meta`, swaps quantized linears into `Fp8Linear`, rebuilds pruned AdaLN projections, and only then materializes tensors. This is the right class of strategy for avoiding BF16 peak memory during conversion/load.
- GB10 `paths.py` resolves a pruned FP8 DiT checkpoint and a separate FP8 Qwen3-VL conditioner. Those are not part of the local no-new-download BF16 FL2VA evidence.
- GB200 baseline source explicitly uses no quantization, and RTX5090 source uses released BF16 FL2VA with layerwise component offload. They do not provide full-H3 quantized A6000 support.

Implication: Sol-Engine FP8 supports the conclusion that quantization is a serious future path, but not that a no-new-download A6000 quantized deployment exists now.

## Offline conversion and storage implications

The local FL2VA partition is complete at **144,051,182,625 bytes**. Any no-new-download quantization attempt would have to generate new local artifacts from these BF16 weights rather than fetching a pre-quantized checkpoint.

Minimum safe conversion implications:

- Create a separate derived checkpoint tree; never overwrite `models/MiniMax-H3/FL2VA`.
- Preserve component boundaries (`transformer`, `text_encoder`, `video_vae`, `audio_vae`, `processor`, `tokenizer`) and `_minimax_h3.partition = fl2va` task metadata.
- Preserve FP32 exceptions and BF16/FP32 contracts for patch projections, time/final projections, RoPE buffers, and any VAE/audio paths that lack quantized kernels.
- Write a manifest with source checkpoint revision, converter commit, exact quant scheme, per-layer scale policy, ignored layers, expected file counts, byte totals, and SHA256 checksums for every generated shard.
- Budget storage for the original 144 GB plus derived shards and temporary conversion files. A rough 8-bit target may still require tens of GiB plus scales/metadata; this audit does not treat that estimate as a fit claim.
- Re-run integrity checks after conversion; do not rely on safetensors extension alone as proof that quant metadata is valid.

Because no such converter output exists locally, there is no no-new-download runnable quantized candidate today.

## Kernel and runtime integration requirements

A future A6000 quantization branch would need all of the following before any launch claim:

1. **Converter:** component-aware BF16-to-derived-checkpoint converter that preserves H3 fused QKV/MLP packing, modality/tag AdaLN behavior, FP32 keep-lists, and FL2VA partition metadata.
2. **Runtime loader:** vLLM-Omni H3 loader changes that load quantized fused tensors and scale metadata without bypassing H3 reorder/packing callbacks.
3. **Offload semantics:** DLO/offloader must store and stream the quantized representation intentionally; retaining BF16 host master copies would erase much of the host-memory benefit.
4. **A6000 kernels:** SM86-supported matmul/dequant kernels with BF16 output contracts where needed. Existing A6000 overlay kernels are BF16 exact elementwise kernels, not quantized GEMMs.
5. **Fallback and telemetry:** fail-closed default-off env gates, per-component telemetry, strict shape/dtype/device/SM86 checks, and explicit dense/BF16 fallback accounting.
6. **Compatibility checks:** one visible A6000/SM86, locked CUDA/PyTorch/vLLM-Omni versions, no unexpected network/model fetches, no CUDA extension gaps, and no accidental multi-GPU or Docker mutation outside an authorized GPU lane.
7. **AV/semantic validation:** quantized output must pass structural AV checks and prompt-matched quality checks against BF16 exact references before any practical recommendation.

## Host and GPU memory impact

Existing BF16 evidence already shows that A6000 GPU memory is not the only bottleneck. The performance report records an internal same-device BF16 exact denominator with peak GPU memory around **26.8 GiB** and peak host memory around **204.8 GiB**. The runtime audit also notes that increasing DLO resident layers does not reduce host RAM in the current implementation because resident layers keep pinned CPU master copies.

Expected quantization impact, if implemented later:

- **GPU memory:** may reduce resident-layer footprint and transfer payloads, but only if quantized kernels consume quantized weights directly or dequantize in controlled workspaces. No measured A6000 quantized peak exists.
- **Host memory:** may reduce host/pinned-memory pressure only if DLO stores quantized master copies. If the runtime retains BF16 masters, quantization may not reduce host memory materially.
- **Disk/storage:** derived checkpoint size may fall relative to 144 GB BF16 FL2VA, but it still adds a second checkpoint tree plus conversion scratch.
- **Latency:** may improve transfer or GEMM time, but may also lose time to dequantization or unsupported kernel fallback. No speedup should be claimed without a same-device measured denominator.

## AV quality gates

A quantized MiniMax-H3 A6000 result would require all gates below before practical use:

1. Same task partition (`FL2VA`), prompt/workload, seed, resolution, frame count, audio sampling, and denoising-step contract as the BF16 denominator unless a smoke run is explicitly labeled functional-only.
2. Structural AV validation: readable video, expected frame/FPS/duration envelope, readable stereo audio at 32 kHz, and no missing output.
3. Numeric comparison against BF16 reference where available: pixel/video metrics and waveform/audio metrics recorded with exact paths and hashes.
4. Semantic review: prompt adherence, visual artifacts, temporal coherence, and human auditory listening. Structural AV pass alone is not semantic quality certification.
5. Track labeling: all quantized outputs remain `practical_disclosed_approx`; they cannot be reported as BF16 exact/fidelity.
6. Evidence retention: manifest, stdout/stderr excerpts, telemetry JSON, resource samples, output hashes, and quality reports. Raw MP4/audio can be retained in evidence but must not be bundled in a future code-only release export.

No quantized output currently exists, so all quantization quality gates are **not started**.

## Runnable no-new-download candidate verdict

Final verdict: **not runnable / no candidate**.

- The local BF16 FL2VA checkpoint is complete and can support BF16/DLO work, but it is not a quantized checkpoint.
- vLLM-Omni has H3 hooks where quantization could be integrated, but no H3 quantized manifest/converter/loader path is present.
- The A6000 overlay is BF16 exact and default-off; it does not implement quantized checkpoint conversion or quantized GEMM paths.
- Sol-Engine GB10 FP8 is relevant prior art, but it requires different checkpoint artifacts and is not A6000/SM86 no-new-download evidence.
- FP8/FP4 assumptions from other architectures are rejected for SM86 unless a local source-backed implementation and A6000 validation are produced.

The only safe project statement is: **A6000 full-H3 quantization is a future practical-approximation research task, not a current deployment candidate.**

## Guarded next steps

No deployment, model-load, Docker, GPU, network, download, or conversion command is supported by this audit.

Safe CPU-only follow-up commands, if a later reviewer wants to re-check this artifact without touching GPU/Docker/network/model loading:

```bash
# Verify the audit file still contains the acceptance sections.
python3 - <<'PY'
from pathlib import Path
p = Path('technical_report/evidence/minimax_h3_desktop/quantization_feasibility_a6000.md')
text = p.read_text(encoding='utf-8')
required = [
    'BF16 fidelity boundary',
    'Practical SM86 quantization candidates',
    'Rejected FP8/FP4',
    'vLLM-Omni support gaps',
    'Sana/Sol-Engine FP8/quant relevance',
    'Offline conversion and storage implications',
    'Kernel and runtime integration requirements',
    'Host and GPU memory impact',
    'AV quality gates',
    'Runnable no-new-download candidate verdict',
    'Guarded next steps',
]
missing = [s for s in required if s not in text]
raise SystemExit(f'missing sections: {missing}' if missing else 0)
PY

# Re-check that the local FL2VA JSON configs do not advertise quantization metadata.
rg -n "quantization_config|float8|fp8|int8|awq|gptq|bitsandbytes|nf4|fp4" \
  models/MiniMax-H3/FL2VA \
  --glob '*.json' \
  --glob '!**/tokenizer.json' \
  --glob '!**/vocab.json' \
  --glob '!**/chat_template.json' \
  --glob '!**/preprocessor_config.json' \
  --glob '!**/video_preprocessor_config.json' \
  --glob '!**/tokenizer_config.json'
```

Any future quantization attempt must be separately authorized and must start with a CPU/static design and manifest review. It should not begin with an A6000 launch command.
