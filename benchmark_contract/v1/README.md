# MiniMax-H3 A6000 Benchmark Contract v1 / 基准合同 v1

Version / 版本：**1.0.0** · frozen / 冻结：**2026-08-12 15:59 UTC**

This directory is the canonical public contract for the native short clip and the 30/60-second final-AV production lanes. It replaces the old pre-gate verifier scaffold for new claims; raw historical evidence is not edited. 本目录是原生短片与 30/60 秒最终音视频生产路线的唯一 v1 公共合同。旧的 pre-gate scaffold 已被取代，但历史原始证据保持不变。

## Decisive capability finding / 能力结论

The pinned open-source path does **not** currently support native 30- or 60-second output context:

- MiniMax-H3 revision `6818f6c32d12b210915e44ad56a4228c2608f160` documents 4–15-second H3-Base output at 24 FPS with 32 kHz stereo audio.
- vLLM-Omni revision `8e2e9b6b53e86e6a479ed2c0a53782f655f60e04` enforces the same 4–15-second range and rejects values above 15 seconds.
- The prepared local model contains only `FL2VA`. It supports text-only generation plus first/last image conditioning; the separately documented `Ref2VA` partition is not locally prepared.
- Sana/Sol-Engine revision `d00eef311670a58deb2c323fe072738fcb945600` grounds the accepted 1344×768, 124-frame short cell, not native 30/60-second generation.

因此，当前 30/60 秒路线必须明确标记为 **`extension`** 或其他非原生长上下文分类：首段为 T2VA，后续段使用上一段保留的末帧作为 FL2VA 首帧条件。已验收的 30 秒 r10 结果属于六段 extension/chunked final-AV；除非未来有新的版本化源码和真实验证，否则任何 30/60 秒结果都不得标记为 `native_long_context`。

## Canonical artifacts / 规范文件

- [`contract.json`](contract.json): revisions, licenses, capabilities, timing, quality, promotion, and denominator rules.
- [`../../schemas/minimax_h3_benchmark_record_v1.schema.json`](../../schemas/minimax_h3_benchmark_record_v1.schema.json): machine-readable result shape.
- [`../../tools/validate_benchmark_record.py`](../../tools/validate_benchmark_record.py): fail-closed semantic validator.
- [`lane-manifests/native-short-1344x768-124f-24fps-v1.json`](lane-manifests/native-short-1344x768-124f-24fps-v1.json): accepted short-cell identity, no rerun.
- [`lane-manifests/final-av-30s-1344x768-24fps-v1.json`](lane-manifests/final-av-30s-1344x768-24fps-v1.json): 30-second extension workload/assembly manifest; the accepted r10 formal result is recorded separately.
- [`lane-manifests/final-av-60s-1344x768-24fps-v1.json`](lane-manifests/final-av-60s-1344x768-24fps-v1.json): 60-second extension workload manifest; a separate matched N=1 run completed as **descriptive/no-promotion** evidence, not a formal speedup.
- [`normalized-records/`](normalized-records/): bounded normalized records derived from accepted BF16, Turbo, Sol-Attn short-clip, and r10 30-second final-AV evidence. Explicit unavailable fields are not measurements.

## Frozen lanes / 冻结路线

| Lane | Final video | Final audio | Current mode | Measurement status |
|---|---:|---:|---|---|
| Native short | 1344×768, 124 frames, 24 FPS | 32 kHz stereo; historical AAC decode count 166,912 samples/channel | `native_short_clip` | accepted historical evidence exists |
| Final AV 30 s | 1344×768, exactly 720 frames | exactly 960,000 effective samples/channel | `extension`, 6 source chunks | accepted bounded r10 formal N=10 timing/structural result exists; not native long context or human-quality certification |
| Final AV 60 s | 1344×768, exactly 1,440 frames | exactly 1,920,000 effective samples/channel | `extension`, 12 source chunks | matched N=1 complete: r10 2682.008 s vs r9 2802.991 s; descriptive/no-promotion because both lanes triggered the frozen-transition proxy flag |

For extension assembly, chunk 1 retains source frames `[0,120)`; later chunks retain `[1,121)` because source frame 0 is the prior terminal-frame condition. Each chunk contributes exactly 120 final frames and 160,000 effective PCM samples per channel. AAC priming and end padding must be reported separately. Missing audio, unknown padding, wrong frame count, or decode failure is incomplete final AV and fails closed.

## Long-result labels / 长视频分类

Every 30/60-second result must declare exactly one:

- `native_long_context`: one native model context, one chunk, no assembly;
- `chunked_overlap`: independently generated chunks with declared overlap/blending;
- `extension`: later chunks are conditioned on earlier generated content;
- `montage_stitching`: independently generated shots assembled as a montage.

A result with multiple chunks or any concat/blend operation cannot claim `native_long_context`. 拼接、重叠或多段生成结果绝不能表述为原生长上下文。

## Timing hierarchy / 计时层级

Cold E2E starts before process/container startup with declared caches empty. Warm E2E starts immediately before the production request after one excluded warmup in the same service lifecycle. Both stop only after final AV is written, fully decoded, accounted, and validation metadata is durable.

Root accounting is:

```text
text conditioning + denoise + video VAE + audio VAE + encoding/mux + I/O
```

`attention` is a **nested child of denoise**. It is reported, but it is never added to denoise in E2E accounting. This prevents the common `denoise + attention` double count. `seconds_per_generated_second = warm E2E median / exact final duration`.

Historical short records retain their actual local HTTP request-to-download boundary and mark cold/component fields unavailable. They are not silently upgraded to the new E2E boundary.

## Required metrics and quality / 必填指标与质量门槛

Every result must carry explicit fields for cold/warm E2E, denoise, nested attention, video VAE, audio VAE, encoding/mux, I/O, seconds per generated second, peak GPU and host memory, power, and failures. A field may say `not_available_historical_evidence` only for a normalized historical record; omission is invalid.

Long lanes additionally require objective measurements for:

- identity/subject, background, and camera consistency;
- motion, repetition, and freezing;
- visual seams;
- loudness, silence, and audio continuity;
- an AV-sync proxy.

Automatic red flags are frozen in `contract.json`. Identity/background/camera/motion candidates use a same-lane 5% non-inferiority margin. A first baseline is descriptive, not an improvement claim. Human review is a separate gate covering all listed visual/audio categories; objective proxies cannot stand in for it.

## Tracks, topology, and promotion / 分轨、拓扑与晋级

- `fidelity_bf16_exact`: released BF16 semantics and exact/lossless changes only.
- `practical_disclosed_approx`: Turbo, cache, quantization, or sparse mechanisms, each disclosed.
- `single_a6000`: exactly one visible/used RTX A6000 and one physical GPU UUID.
- `multi_gpu_production`: separate reporting and never a single-A6000 denominator.

Promotion is N=1 → N=3 → matched N≥10. N=1 and N=3 are gates, not formal speedups. Formal results report distributions and noise. Practical product claims also require at least three prompts × three seeds plus the separate human gate.

A speedup denominator must match track, workload fingerprint, physical GPU UUID, deployment scope, timing boundary, quality threshold, and long-production mode. Only one named principal variable may differ. The validator rejects cross-track, cross-GPU, cross-workload, cold/warm, single/multi-GPU, and stitched-as-native comparisons.

## Accepted bounded evidence retained / 已验收有界证据

- BF16 warm N=10 median: **1792.2021025 s** for the native short fidelity denominator.
- Turbo 8-step N=10 median: **290.9976015 s**, practical approximation on the native short clip.
- Turbo 4-step N=10 median: **149.6191865 s**, practical approximation with greater quality risk on the native short clip.
- Sol-Attn r8 median HTTP-time improvement: **15.203295894%**, only for the formal same-GPU matched 5-step opt-in short lane.
- Final-AV 30 s r10: reviewer-accepted formal N=10 matched warm-E2E improvement **4.326262968%** versus retained `r9_current_sol_attn`; candidate warm median **1333.5752375 s** vs reference **1394.0061285 s**, cold median **1814.1335341 s** vs **1884.1419612 s**, six-chunk `extension`, complete 720-frame / 960,000-sample-per-channel accounting.

The normalized Turbo records intentionally do not create a new cross-track speedup claim. The r8 and r10 comparisons are valid only because each comparison keeps the same practical track, workload, physical GPU, timing boundary, and quality threshold inside its own lane. The 30-second r10 result is not native long context, not BF16 fidelity, not human semantic/audio/AV quality certification, not product readiness, and not public-comparison/SOTA evidence.

## Descriptive measured evidence, not promoted / 已测但未晋级

- Final-AV 60 s matched N=1 completed on one A6000: retained r9 warm E2E **2802.991 s**, r10 guarded-adaptive warm E2E **2682.008 s**, a **4.316% single-sample route signal**. Both outputs completed 1,440-frame / 1,920,000-effective-sample-per-channel accounting, but both carried the `near_frozen_transition_fraction` automatic flag. The independent Reviewer accepted only the bounded descriptive/no-promotion classification. No N=3/N≥10 or formal speedup claim exists for 60 seconds.

## Validation / 验证

```bash
python3 tools/validate_benchmark_record.py \
  benchmark_contract/v1/contract.json \
  benchmark_contract/v1/lane-manifests/*.json \
  benchmark_contract/v1/normalized-records/*.json
```

Negative fixtures under `tests/fixtures/benchmark_contract/rejected/` must return nonzero. They cover stitched-as-native, cross-track speedup, mismatched GPU/workload denominators, incomplete final AV, and missing mandatory metrics.
