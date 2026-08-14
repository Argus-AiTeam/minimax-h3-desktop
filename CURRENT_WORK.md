# Current Work / 当前工作

> **Updated / 更新时间：2026-08-14 09:15 UTC**
>
> This page separates accepted evidence, active Stage-2 objectives, readiness signals, and blocked work. Status is not a result.
> 本页严格区分已验收证据、Stage 2 目标、准备度信号与阻塞项；“正在研究”不等于“已有结果”。

## Production focus / 生产目标

The active objective remains **720p-class long-video MiniMax-H3 FL2VA production on one RTX A6000**: first a reproducible 30-second audiovisual lane, then a 60-second lane. The accepted measurements below are for the native **1344×768, 124-frame, 24 FPS, 5.166667-second** clip; they are **not** long-video results. No 30/60-second output has been accepted, and no stitching/montage/extension output is being presented as native long-context generation.

当前目标仍是在**单张 RTX A6000**上完成 720p 级 MiniMax-H3 FL2VA 长视频生产：先实现可复现的 30 秒音视频链路，再推进 60 秒。下列已验收数据均来自原生 **1344×768、124 帧、24 FPS、5.166667 秒**短片，**不是**长视频结果。目前没有已验收的 30/60 秒输出，也不会把拼接、montage 或 extension 输出表述为原生长上下文生成。

## Stage accounting / 阶段记分边界

| Scope / 范围 | Credit / 信用 | Meaning / 含义 |
|---|---:|---|
| Stage 1 accepted foundation / Stage 1 已验收基础 | **45/45** | Public benchmark contract, short-clip BF16/Turbo baselines, and bounded Sol-Attn r8 formal 5-step opt-in evidence are accepted. / 公开基准合同、短片 BF16/Turbo 基线与 Sol-Attn r8 formal 5-step opt-in 有界证据已验收。 |
| Stage 2 long-video objective / Stage 2 长视频目标 | **0/55** | No accepted 30s/60s long-video production result yet. / 30s/60s 长视频生产尚无已验收结果。 |
| Long-video targets / 长视频目标计数 | **0/2** | 30s: 0/1 accepted; 60s: 0/1 accepted. This is an acceptance count, not a hidden quality score. / 30 秒 0/1，60 秒 0/1；这是验收计数，不是 hidden 质量分。 |
| Readiness research / 准备度研究 | **0 Stage-2 credit** | Model-free/captured-metadata kernels, blocked-runtime checks, and negative probes inform the next gate but do not count as H3 E2E, product, BF16-fidelity, or long-video speedup. / 无模型或 captured-metadata kernel、runtime blocker 检查和负结果只服务于下一门禁，不计为 H3 端到端、产品、BF16 保真或长视频加速。 |

## Accepted evidence / 已验收证据

| Lane / 路线 | Accepted result / 已验收结果 | Boundary / 边界 |
|---|---:|---|
| BF16 fidelity | warm N=10 median **1792.202 s** | Same physical A6000; fidelity denominator. / 同一物理 A6000；保真分母。 |
| Turbo 8-step | N=10 median **290.998 s** | Disclosed practical approximation, not BF16 fidelity. / 明示近似实用路线，不是 BF16 保真。 |
| Turbo 4-step | N=10 median **149.619 s** | Faster approximate lane with higher quality risk; not the default. / 更高质量风险的近似路线，不是默认方案。 |
| Sol-Attn r8 | median HTTP-time improvement **15.203%**, 10/10 pairs | **Only** the formal matched 5-step, same-GPU, opt-in lane; not a 50-step BF16, long-video, or semantic-quality result. / **仅限**正式同卡匹配的 5-step opt-in 通道；不是 50-step BF16、长视频或语义质量结论。 |
| Sol-Attn pair-value-halves captured replay | retained default-off captured-metadata kernel candidate: replay total median **144.652 ms** vs current prefix-skip **175.216 ms**; forward pointer subphase **129.850 ms** vs **158.161 ms**; `max_abs_valid=0` | Captured-metadata/non-Docker/model-free only; no H3 E2E, long-video, BF16-fidelity, product, normal-PC, public-comparison, or SOTA claim. / 仅限 captured metadata、非 Docker、无模型 kernel 证据；不是 H3 端到端、长视频、BF16 保真、产品、普通电脑、公开对比或 SOTA 声明。 |

Single-A6000 evidence remains separate from any multi-GPU production study. Structural AV checks are not substitutes for an explicit human quality gate.

单卡 A6000 证据与多卡生产研究保持分离；结构化音视频检查不能替代明确的人类质量验收。

## Recent negative results / 近期负结果

All negative probes below are bounded by their evidence lane. Kernel/model-free timing is never reported as H3 E2E, long-video, BF16-fidelity, product, normal-PC, public-comparison, or SOTA speedup.

以下负结果均受各自证据通道限制。kernel/model-free 计时不会被写成 H3 端到端、长视频、BF16 保真、产品、普通电脑、公开对比或 SOTA 加速。

- **Sol-Attn r6:** all 208 calls declined with `unsupported_contiguity`; `sparse_calls=0`. The small dense/opt-in timing difference was not accepted as acceleration.
- **Sol-Attn r7:** packed H3 layout metadata did not reach the backend; it failed closed with `sparse_calls=0`.
- **DLO resident_layers=16:** the 50-step warm candidate improved **0.456%**, below the BF16 baseline CV of **0.837%**; no formal N=10 promotion.
- **Exact-kernel route:** RoPE/all-exact changed audiovisual output; SwiGLU retained no accepted end-to-end gain. Kernel-only microbenchmarks were not promoted to product speedups.
- **Sol-Attn legacy/model-free harnesses:** the legacy toy harness had sparse median slower than dense; the 2026-08-13 stride-aware-V SM86 harness was correctness/zero-materialization evidence only (`sparse` median **0.419968 ms** vs dense **0.131072 ms**), not an H3 E2E or product speedup.
- **`num_warps=8` forward config:** exact against the retained lane but much slower and reverted; candidate total **373.673 ms** vs current **181.965 ms**, forward pointer **357.364 ms** vs **163.835 ms**.
- **Paired-exact grouping:** reduced the modeled exact-loop iterations by **45.680%**, but regressed the whole lane by **4.955%**; candidate **152.111 ms** vs current **144.930 ms** (forward subphase also regressed, **137.183 ms** vs **130.028 ms**). Rejected and reverted.
- **Full-K exact fast path:** slower and not bit-exact; candidate **175.182 ms** vs current **145.071 ms**, forward **160.519 ms** vs **129.777 ms**, `max_abs_valid=0.0001220703125`, tail rows not exact. Rejected and reverted.
- **Padded-query unmask:** only **0.043%** apparent whole-lane difference (**145.933 ms** vs **145.996 ms**), while the forward subphase regressed (**130.639 ms** vs **130.218 ms**, **-0.323%**) and valid/tail exactness failed (`max_abs_valid=0.0233154296875`). Rejected and reverted.
- **PV128-dot:** omitted from public negative-result numbers until a separate independent review is bound to that candidate. No H3 E2E, long-video, BF16-fidelity, product, or public speedup claim is made from it here.

以上失败均保留为可复现边界，不因后续 r8 成功或 captured replay 成功而删除或改写。

## Hypothesis under test / 待检验假设

The r8 real-chain profile records **192 Q/K/V materialization events totaling 105,344,139,264 bytes per 5-step run**. The falsifiable hypothesis is that a layout-aware view/packing path can remove most of this materialization while preserving metadata validation, dense fail-closed behavior, structural correctness, and the same bounded quality threshold. The latest retained captured-metadata replay (`sol_attn_pair_value_halves_captured_replay_20260813T145731Z`) reuses the r8 H3 layout/density telemetry, `derive_h3_sol_attn_hook_metadata`, and `SolAttnPolicy.from_env`, then tests default-off `MINIMAX_H3_A6000_SOL_ATTN_PAIR_VALUE_HALVES=1`. It keeps stride-aware fused-QKV V view, full-prefix-block skip, dense-prefix overwrite, tau/routing, and exact-block order, but shares one Q/K route and online-softmax probability stream across the two BV64 value halves. It reports captured layout agreement, metadata-derivation pass, `max_abs_valid=0`, prefix/tail equality to the current path, zero padding rows, zero fallback, and zero unintended materialized copy events/bytes in the replay lanes. Its captured-metadata/model-free replay median was candidate total **144.652 ms** vs current prefix-skip **175.216 ms** (forward pointer subphase **129.850 ms** vs **158.161 ms**). It loads no H3 model and starts no Docker container; the replay is kernel/model-free only and is not a product speedup. Success still requires a matched H3 end-to-end gate on the same physical GPU; a replay or microbenchmark alone is insufficient.

r8 真实链路在每次 5-step 运行中记录了 **192 次 Q/K/V 实体化，共 105,344,139,264 字节**。可证伪假设是：布局感知的 view/packing 路径能够消除其中大部分复制，同时保持 metadata 校验、异常时回退 dense、结构正确性和相同质量阈值。最新保留的 captured-metadata replay（`sol_attn_pair_value_halves_captured_replay_20260813T145731Z`）复用 r8 H3 layout/density telemetry、`derive_h3_sol_attn_hook_metadata` 与 `SolAttnPolicy.from_env`，测试默认关闭的 `MINIMAX_H3_A6000_SOL_ATTN_PAIR_VALUE_HALVES=1`。它保留 stride-aware fused-QKV V view、完整 prefix block skip、dense-prefix overwrite、tau/routing 与 exact-block 顺序，但让两个 BV64 value halves 共用一次 Q/K 路由和 online-softmax 概率流。结果为 captured layout agreement、metadata derivation pass、`max_abs_valid=0`、prefix/tail 与当前路径相等、padding 行为 0、fallback 为 0，且 replay lanes 中无非预期 materialized copy 次数/字节。captured-metadata/无模型 replay median 为 candidate total **144.652 ms**，current prefix-skip **175.216 ms**（forward pointer 子阶段 **129.850 ms** 对 **158.161 ms**）。该 replay 没有加载 H3 模型，也没有启动 Docker 容器；仅限 kernel/model-free，不是产品级加速。成功仍必须体现为同一物理 GPU 上的 H3 匹配端到端 gate；仅有 replay 或微基准不成立。

## Benchmark contract status / 基准合同状态

The canonical public v1 contract is at [`benchmark_contract/v1/README.md`](benchmark_contract/v1/README.md), with a machine-readable schema, fail-closed validator, normalized historical records, and three dry-run lane manifests. The pinned MiniMax-H3/vLLM-Omni path supports native output only from 4 to 15 seconds; therefore the 30/60-second manifests remain explicitly **unmeasured `extension` lanes**, not native-long-context results. Review and publication status are gates, not benchmark results.

规范 v1 已公开于 [`benchmark_contract/v1/README.md`](benchmark_contract/v1/README.md)，包括机器可读 Schema、fail-closed validator、历史证据规范化记录和三个 dry-run lane manifest。当前 pinned MiniMax-H3/vLLM-Omni 原生输出仅支持 4–15 秒，因此 30/60 秒 manifest 仍明确标记为**尚未实测的 `extension` 路线**，不是原生长上下文结果。审阅与发布状态是门槛，不是 benchmark 结果。

## Pinned-runtime blocker and next real-chain gate / 固定 runtime 阻塞与下一真实链路门槛

The next real-chain experiment is still one matched N=1 small gate that changes only Q/K/V materialization, checks dense fallback and output/AV correctness, and measures copy bytes plus comparable end-to-end timing. It may run only after the Docker runtime blocker is cleared: an operator/admin must repair the Docker overlay2/layerdb conflict **or** provide a clean Docker daemon where the pinned r2 runtime image and required Sol-Attn overlay/provenance images are locally inspectable; then the r9 overlay build/inspection gate must pass, a fresh preflight must show a genuinely idle A6000, and the run must acquire a single-GPU lease. The known failed pinned r2 restore/build must not be repeated unchanged, and no GPU model inference, H3 E2E result, 30/60-second result, or new product speedup claim exists from the blocked state.

下一真实链路实验仍然是匹配 N=1 小门槛，只改变 Q/K/V 实体化这一项，验证 dense 回退与输出/音视频正确性，并同时测量复制字节数和可比端到端耗时。只有在 Docker runtime 阻塞清除后才可运行：operator/admin 需要修复 Docker overlay2/layerdb 冲突，**或**提供干净 Docker daemon，使固定 r2 runtime 镜像及所需 Sol-Attn overlay/provenance 镜像可在本地 inspect；随后 r9 overlay build/inspection gate 必须通过，fresh preflight 必须显示至少一张真正空闲的 A6000，并获得单 GPU lease。已知失败的 pinned r2 restore/build 不应原样重试；当前阻塞状态没有产生 GPU 模型推理、H3 端到端结果、30/60 秒结果或新的产品级加速声明。

## Workflow kickoff and source tracking / 工作流启动与来源跟踪

Workflow A restarted on **2026-08-12** from the accepted real-chain profile above. The first-party source baseline is the pinned Sana Sol-Engine revision `d00eef311670a58deb2c323fe072738fcb945600` and vLLM-Omni revision `8e2e9b6b53e86e6a479ed2c0a53782f655f60e04`. The Sana repository README declares Apache-2.0 for repository code; its Sol-Attn third-party notice retains BSD-3-Clause FlashAttention attribution. MiniMax-H3 model revision `6818f6c32d12b210915e44ad56a4228c2608f160` and Turbo assets remain separately licensed and are not distributed here. See [`benchmark_contract/v1/contract.json`](benchmark_contract/v1/contract.json), [`NOTICE`](NOTICE), and [`ports/minimax_h3_a6000/UPSTREAM.md`](ports/minimax_h3_a6000/UPSTREAM.md).

工作流 A 于 **2026-08-12** 依据上述真实链路 profile 重新启动。第一方源码基线固定为 Sana Sol-Engine `d00eef311670a58deb2c323fe072738fcb945600` 与 vLLM-Omni `8e2e9b6b53e86e6a479ed2c0a53782f655f60e04`。Sana 仓库 README 声明代码采用 Apache-2.0，Sol-Attn 的第三方通知保留 FlashAttention BSD-3-Clause 归属。MiniMax-H3 模型 revision `6818f6c32d12b210915e44ad56a4228c2608f160` 与 Turbo 资产保持独立许可，本仓库不分发其权重。参见 [`benchmark_contract/v1/contract.json`](benchmark_contract/v1/contract.json)、[`NOTICE`](NOTICE) 与 [`ports/minimax_h3_a6000/UPSTREAM.md`](ports/minimax_h3_a6000/UPSTREAM.md)。
