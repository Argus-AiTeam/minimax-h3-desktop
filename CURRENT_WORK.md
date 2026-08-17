# Current Work / 当前工作

> **Updated / 更新时间：2026-08-17 13:43 UTC**
>
> This page separates accepted evidence from scope limits and next work. Status text is not a new result.
> 本页区分已验收证据、边界限制与下一步工作；状态说明本身不是新结果。

## Production focus / 生产目标

The active goal is **720p-class MiniMax-H3 FL2VA production on one RTX A6000**. The 30-second lane has an accepted bounded formal N=10 timing/structural result. The 60-second lane has now completed matched N=1 end to end, but only as **descriptive/no-promotion** evidence because both lanes triggered the frozen-transition proxy flag. Both use `extension` / chunked final AV and are **not native long context**.

当前目标是在**单张 RTX A6000**上持续推进 720p 级 MiniMax-H3 FL2VA。30 秒路线已有已验收的有界 formal N=10 计时/结构结果；60 秒路线也已完成 matched N=1 端到端输出，但由于两条路线都触发冻结转场 proxy，只能记为**描述性/不晋级**证据。两者均为 `extension` / 分段 final-AV，**不是原生长上下文**。

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
| Long-video targets / 长视频目标计数 | **1/2 bounded lanes accepted** | 30 s: accepted formal result; 60 s: full N=1 output measured and Reviewer-accepted only as descriptive/no-promotion, so it does not receive accepted-lane credit. |

## Recent negative results / 近期负结果

- **Sol-Attn r6:** all 208 calls declined with `unsupported_contiguity`; `sparse_calls=0`.
- **Sol-Attn r7:** packed H3 layout metadata did not reach the backend; fail-closed with `sparse_calls=0`.
- **DLO resident_layers=16:** 50-step warm candidate improved **0.456%**, below BF16 baseline CV **0.837%**; no formal N=10 promotion.
- **Exact-kernel route:** RoPE/all-exact changed audiovisual output; SwiGLU retained no accepted E2E gain.
- **Sol-Attn model-free harnesses:** stride-aware-V and pair-value-halves replay remain default-off, model-free/captured-metadata evidence only; no H3 E2E or product speedup was claimed from them.
- **Rejected r10-adjacent routes:** r11/r12 adaptive variants, Cache-DiT, bounded VAE tile batching, VAE CUDA Graph, DLO async sync-prefetch, and regional `torch.compile` were rejected because of quality regression, no actual reuse, below-threshold E2E gain, slowdown, or timeout.
- **60-second r10 N=1:** completed at **2682.008 s** vs r9 **2802.991 s** (4.316% route signal), but both lanes triggered `near_frozen_transition_fraction`; no N=3/N≥10 promotion.

## Hypothesis under test / 待检验假设

The 60-second follow-up has been answered: the r10 mechanism produced a 4.316% N=1 timing signal with complete structural AV, but automatic freezing flags blocked promotion. The next falsifiable work must therefore use a newly pre-registered candidate and quality gate, avoid duplicate 60-second reruns, and target a measured critical-path component with a predicted ≥1% warm-E2E signal.

60 秒 follow-up 已有结论：r10 在完整结构 AV 下出现 4.316% 的 N=1 计时信号，但自动冻结红旗阻止晋级。下一步必须预注册新的候选和质量门禁，禁止重复跑同一 60 秒配置，并只选择理论上可贡献至少 1% warm-E2E 的实测关键路径。

## Benchmark contract status / 基准合同状态

The canonical v1 contract is at [`benchmark_contract/v1/README.md`](benchmark_contract/v1/README.md). The pinned path still supports only 4–15 seconds natively. The 30-second accepted result and the completed 60-second descriptive N=1 result are therefore explicitly `extension`/chunked. The normalized accepted r10 record is [`benchmark_contract/v1/normalized-records/final-av-30s-r10-guarded-adaptive-step3-formal-n10.json`](benchmark_contract/v1/normalized-records/final-av-30s-r10-guarded-adaptive-step3-formal-n10.json); the 60-second lane manifest points to its separate descriptive/no-promotion decision packet.

规范 v1 位于 [`benchmark_contract/v1/README.md`](benchmark_contract/v1/README.md)。当前 pinned 路径原生输出仍只支持 4–15 秒。30 秒已验收结果和 60 秒已完成的描述性 N=1 结果都明确标记为 `extension`/分段。30 秒 accepted r10 规范化记录见上方链接；60 秒 lane manifest 指向独立的 descriptive/no-promotion 决策包。

## Next experiment / 下一实验

Next is a **new, pre-registered critical-path candidate**, not another unchanged 60-second rerun. It must first show a source-backed ≥1% warm-E2E signal in a cheap representative gate, preserve same-input quality/non-inferiority, remain default-off and fail-closed, then advance N=1 → N=3 → N≥10 only when each frozen gate passes.

下一实验必须是**新的、预注册的关键路径候选**，不能重复不变的 60 秒配置。候选须先在低成本 representative gate 中证明有源码依据且预计至少贡献 1% warm-E2E，同时保持同输入质量 non-inferiority、默认关闭和 fail-closed；只有冻结门禁逐级通过，才可按 N=1 → N=3 → N≥10 晋级。
