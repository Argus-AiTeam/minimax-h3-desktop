# Current Work / 当前工作

> **Updated / 更新时间：2026-08-12 15:33 UTC**
>
> This page reports accepted evidence and active work separately. Status is not a result.
> 本页严格区分已验收证据与进行中状态；“正在研究”不等于“已有结果”。

## Production focus / 生产目标

The active objective is **720p-class long-video MiniMax-H3 FL2VA production on one RTX A6000**, first a reproducible 30-second audiovisual pipeline and then 60 seconds. The accepted measurements below are for the native **1344×768, 124-frame, 24 FPS, 5.166667-second** clip; they are not long-video results. No 30/60-second result is accepted yet, and no stitched output is being presented as native long-context generation.

当前目标是在**单张 RTX A6000**上完成 720p 级 MiniMax-H3 FL2VA 长视频生产：先实现可复现的 30 秒音视频管线，再推进 60 秒。下列已验收数据均来自原生 **1344×768、124 帧、24 FPS、5.166667 秒**短片，并非长视频结果。目前尚无已验收的 30/60 秒结果，也不会把拼接输出表述为原生长上下文生成。

## Accepted evidence / 已验收证据

| Lane / 路线 | Accepted result / 已验收结果 | Boundary / 边界 |
|---|---:|---|
| BF16 fidelity | warm N=10 median **1792.202 s** | Same physical A6000; fidelity denominator. / 同一物理 A6000；保真分母。 |
| Turbo 8-step | N=10 median **290.998 s** | Disclosed practical approximation, not BF16 fidelity. / 明示近似实用路线，不是 BF16 保真。 |
| Turbo 4-step | N=10 median **149.619 s** | Faster approximate lane with higher quality risk; not the default. / 更高质量风险的近似路线，不是默认方案。 |
| Sol-Attn r8 | median HTTP-time improvement **15.203%**, 10/10 pairs | **Only** the formal matched 5-step, same-GPU, opt-in lane; not a 50-step BF16, long-video, or semantic-quality result. / **仅限**正式同卡匹配的 5-step opt-in 通道；不是 50-step BF16、长视频或语义质量结论。 |

Single-A6000 evidence remains separate from any multi-GPU production study. Structural AV checks are not substitutes for an explicit human quality gate.

单卡 A6000 证据与多卡生产研究保持分离；结构化音视频检查不能替代明确的人类质量验收。

## Recent negative results / 近期负结果

- **Sol-Attn r6:** all 208 calls declined with `unsupported_contiguity`; `sparse_calls=0`. The small dense/opt-in timing difference was not accepted as acceleration.
- **Sol-Attn r7:** packed H3 layout metadata did not reach the backend; it failed closed with `sparse_calls=0`.
- **DLO resident_layers=16:** the 50-step warm candidate improved **0.456%**, below the BF16 baseline CV of **0.837%**; no formal N=10 promotion.
- **Exact-kernel route:** RoPE/all-exact changed audiovisual output; SwiGLU retained no accepted end-to-end gain. Kernel-only microbenchmarks were not promoted to product speedups.
- **Sol-Attn toy harness:** sparse median was slower than dense and was not deployed.

以上失败均保留为可复现边界，不因后续 r8 成功而删除或改写。

## Hypothesis under test / 待检验假设

The r8 real-chain profile records **192 Q/K/V materialization events totaling 105,344,139,264 bytes per 5-step run**. The falsifiable hypothesis is that a layout-aware view/packing path can remove most of this materialization while preserving metadata validation, dense fail-closed behavior, structural correctness, and the same bounded quality threshold. Success requires a matched end-to-end gain on the same physical GPU; a microbenchmark alone is insufficient.

r8 真实链路在每次 5-step 运行中记录了 **192 次 Q/K/V 实体化，共 105,344,139,264 字节**。可证伪假设是：布局感知的 view/packing 路径能够消除其中大部分复制，同时保持 metadata 校验、异常时回退 dense、结构正确性和相同质量阈值。成功必须体现为同一物理 GPU 上的匹配端到端收益；仅有微基准不成立。

## Next experiment / 下一实验

**Not started.** First freeze the versioned short-clip, 30-second, and 60-second benchmark contract. Then run one matched N=1 small gate that changes only Q/K/V materialization, checks dense fallback and output/AV correctness, and measures copy bytes plus comparable end-to-end timing. Promote to N=3 and later N≥10 only if that gate passes. Native long-context, chunked/overlap, extension, and montage/stitching outputs will be labeled separately.

**尚未启动。** 下一步先冻结短片、30 秒和 60 秒的版本化 benchmark contract；随后只改变 Q/K/V 实体化这一项，执行匹配 N=1 小门槛，验证 dense 回退与输出/音视频正确性，并同时测量复制字节数和可比端到端耗时。只有通过后才晋级 N=3，最终再考虑 N≥10。原生长上下文、分块/重叠、延展和 montage/stitching 将分别标注。

## Workflow kickoff and source tracking / 工作流启动与来源跟踪

Workflow A restarted on **2026-08-12** from the accepted real-chain profile above. The first-party source baseline is the pinned Sana Sol-Engine revision `d00eef311670a58deb2c323fe072738fcb945600` and vLLM-Omni revision `8e2e9b6b53e86e6a479ed2c0a53782f655f60e04`. The Sana repository README declares Apache-2.0 for repository code; its Sol-Attn third-party notice retains BSD-3-Clause FlashAttention attribution. MiniMax-H3 model revision `6818f6c32d12b210915e44ad56a4228c2608f160` and Turbo assets remain separately licensed and are not distributed here. See [`NOTICE`](NOTICE) and [`ports/minimax_h3_a6000/UPSTREAM.md`](ports/minimax_h3_a6000/UPSTREAM.md).

工作流 A 于 **2026-08-12** 依据上述真实链路 profile 重新启动。第一方源码基线固定为 Sana Sol-Engine `d00eef311670a58deb2c323fe072738fcb945600` 与 vLLM-Omni `8e2e9b6b53e86e6a479ed2c0a53782f655f60e04`。Sana 仓库 README 声明代码采用 Apache-2.0，Sol-Attn 的第三方通知保留 FlashAttention BSD-3-Clause 归属。MiniMax-H3 模型 revision `6818f6c32d12b210915e44ad56a4228c2608f160` 与 Turbo 资产保持独立许可，本仓库不分发其权重。
