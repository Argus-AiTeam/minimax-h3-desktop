# Current Work / 当前工作

> **Updated / 更新时间：2026-08-16 16:25 UTC**
>
> This page separates accepted evidence from scope limits and next work. Status text is not a new result.
> 本页区分已验收证据、边界限制与下一步工作；状态说明本身不是新结果。

## Production focus / 生产目标

The active goal is **720p-class MiniMax-H3 FL2VA long-video production on one RTX A6000**: first a reproducible 30-second audiovisual lane, then 60 seconds. The 30-second lane now has one accepted **bounded formal timing/structural result** in the practical track, but it is an `extension` / chunked final-AV workflow and **not native long context**.

当前目标是在**单张 RTX A6000**上推进 720p 级 MiniMax-H3 FL2VA 长视频生产：先 30 秒音视频链路，再 60 秒。30 秒路线已有一个已验收的**有界 formal 计时/结构结果**，但它是 `extension` / 分段 final-AV 工作流，**不是原生长上下文**。

## Accepted evidence / 已验收证据

| Lane / 路线 | Accepted result / 已验收结果 | Boundary / 边界 |
|---|---:|---|
| BF16 fidelity short clip | warm N=10 median **1792.202 s** | Native 1344×768, 124 frames, 24 FPS, 5.166667 s; same physical A6000 fidelity denominator only. |
| Turbo 8-step short clip | N=10 median **290.998 s** | Disclosed practical approximation, not BF16 fidelity. |
| Turbo 4-step short clip | N=10 median **149.619 s** | Faster approximate lane with higher quality risk; not the default. |
| Sol-Attn r8 short 5-step opt-in | median HTTP-time improvement **15.203%**, 10/10 pairs | Only the matched same-GPU 5-step opt-in lane; not 50-step BF16, long-video, or semantic quality evidence. |
| **Final-AV 30 s r10** | matched formal N=10 warm E2E median **1333.575 s** vs retained r9 **1394.006 s**; median improvement **4.326%**; cold median **1814.134 s** vs **1884.142 s**; 10/10 pairs complete | **Only** the six-chunk `extension`/chunked 30-second final-AV lane `r10_adaptive_tau1_5_step3_diag` versus retained `r9_current_sol_attn` on one A6000. Complete final AV accounting is 720 frames and 960,000 effective audio samples/channel. Not native long context, not BF16 fidelity, not human semantic/audio/AV-quality certification, not product readiness, not public-comparison or SOTA. |

Single-A6000 evidence remains separate from any multi-GPU production study. Structural AV accounting and no-reference objective proxies are not substitutes for a human quality gate.

单卡 A6000 证据与多卡生产研究分开；结构化音视频核算和无参考客观 proxy 不能替代人工质量验收。

## Stage accounting / 阶段记分边界

| Scope / 范围 | Current credit / 当前信用 | Meaning / 含义 |
|---|---:|---|
| Stage 1 foundation / Stage 1 基础 | **45/45** | Public benchmark contract, native short BF16/Turbo baselines, and bounded Sol-Attn r8 short-lane evidence are accepted. |
| Stage 2 long-video objective / Stage 2 长视频目标 | **25/55** | 30-second final-AV extension/chunked formal r10 timing/structural result accepted; 60-second lane and human/product-quality certification remain open. |
| Long-video targets / 长视频目标计数 | **1/2 bounded lanes** | 30 s: accepted as extension/chunked formal timing/structural result; 60 s: no accepted result yet. This is not a hidden quality score. |

## Recent negative results / 近期负结果

- **Sol-Attn r6:** all 208 calls declined with `unsupported_contiguity`; `sparse_calls=0`.
- **Sol-Attn r7:** packed H3 layout metadata did not reach the backend; fail-closed with `sparse_calls=0`.
- **DLO resident_layers=16:** 50-step warm candidate improved **0.456%**, below BF16 baseline CV **0.837%**; no formal N=10 promotion.
- **Exact-kernel route:** RoPE/all-exact changed audiovisual output; SwiGLU retained no accepted E2E gain.
- **Sol-Attn model-free harnesses:** stride-aware-V and pair-value-halves replay remain default-off, model-free/captured-metadata evidence only; no H3 E2E or product speedup was claimed from them.
- **Rejected r10-adjacent kernel attempts:** `num_warps=8`, paired-exact grouping, full-K exact fast path, and padded-query unmask were rejected or reverted because they were slower, not exact, or below noise.

## Hypothesis under test / 待检验假设

The accepted r10 result changed one principal variable: guarded adaptive Sol-Attn routing with tau=1.5 and `adaptive_step_min=3` versus retained `r9_current_sol_attn`. The falsifiable follow-up is whether the same bounded, default-off mechanism remains useful when the workload is extended to 60 seconds, while preserving dense fail-closed behavior, final-AV accounting, no raw tensor/media leakage, and the same claim boundaries.

已验收 r10 结果只改变一个主变量：相对保留的 `r9_current_sol_attn`，使用 tau=1.5 且 `adaptive_step_min=3` 的 guarded adaptive Sol-Attn routing。下一可证伪问题是：在 60 秒 extension/chunked 工作负载上，该默认关闭机制是否仍有价值，同时保持 dense fail-closed、final-AV 核算、无原始 tensor/media 泄漏和相同 claim 边界。

## Benchmark contract status / 基准合同状态

The canonical v1 contract is at [`benchmark_contract/v1/README.md`](benchmark_contract/v1/README.md). It still records that the pinned MiniMax-H3/vLLM-Omni path supports native output only from 4 to 15 seconds. Therefore the 30-second accepted result is explicitly `extension`/chunked, and the 60-second lane remains unmeasured. The normalized r10 record is [`benchmark_contract/v1/normalized-records/final-av-30s-r10-guarded-adaptive-step3-formal-n10.json`](benchmark_contract/v1/normalized-records/final-av-30s-r10-guarded-adaptive-step3-formal-n10.json).

规范 v1 位于 [`benchmark_contract/v1/README.md`](benchmark_contract/v1/README.md)。当前 pinned MiniMax-H3/vLLM-Omni 原生输出仍只支持 4–15 秒。因此 30 秒已验收结果明确标记为 `extension`/分段；60 秒路线仍未测。r10 规范化记录见 [`benchmark_contract/v1/normalized-records/final-av-30s-r10-guarded-adaptive-step3-formal-n10.json`](benchmark_contract/v1/normalized-records/final-av-30s-r10-guarded-adaptive-step3-formal-n10.json)。

## Next experiment / 下一实验

Next is a **60-second final-AV extension/chunked N=1 gate** using the same versioned contract style and the current retained r10/r9 evidence as context. It must measure cold/warm E2E, component times, seconds per generated second, GPU/host memory, power, failures, final-AV accounting, seams, audio continuity, and AV-sync proxy, and it must remain separate from any true native-long-context research.

下一实验是使用同一版本化合同风格执行 **60 秒 final-AV extension/chunked N=1 gate**，并以上述 r10/r9 已验收证据作为上下文。必须测量 cold/warm E2E、组件耗时、每生成秒耗时、GPU/host 内存、功耗、失败、final-AV 核算、接缝、音频连续性和 AV-sync proxy，并继续与真正原生长上下文研究分开。
