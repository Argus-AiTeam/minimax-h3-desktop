# Current Work / 当前工作

> **Updated / 更新时间：2026-08-13 14:34 UTC**
>
> This page reports accepted evidence and active work separately. Status is not a result.
> 本页严格区分已验收证据与进行中状态；“正在研究”不等于“已有结果”。

## Production focus / 生产目标

The active objective is **720p-class long-video MiniMax-H3 FL2VA production on one RTX A6000**, first a reproducible 30-second audiovisual pipeline and then 60 seconds. The accepted measurements below are for the native **1344×768, 124-frame, 24 FPS, 5.166667-second** clip; they are not long-video results. No 30/60-second result is accepted yet, and no stitched output is being presented as native long-context generation.

当前目标是在**单张 RTX A6000**上完成 720p 级 MiniMax-H3 FL2VA 长视频生产：先实现可复现的 30 秒音视频管线，再推进 60 秒。下列已验收数据均来自原生 **1344×768、124 帧、24 FPS、5.166667 秒**短片，并非长视频结果。目前尚无已验收的 30/60 秒结果，也不会把拼接输出表述为原生长上下文生成。

## Dashboard publication status / 看板发布状态

The public README live dashboard was published at `c5690f1ffd0ffb71a8581ca51afe2366a7a32687` (`docs: publish live research dashboard`, commit time 2026-08-13 14:22:33 UTC). This snapshot repairs stale pre-publication wording after that publication; it does not change accepted quantitative evidence. Commit/sync timestamps use the README's non-self-referential rule: the file records the last verified publication completed before the snapshot, while the containing commit's remote equality is checked externally and rolled into the next substantive update.

公开 README 实时看板已在 `c5690f1ffd0ffb71a8581ca51afe2366a7a32687`（`docs: publish live research dashboard`，commit time 2026-08-13 14:22:33 UTC）发布。本快照只修复发布后残留的 pre-publication wording，不改变已验收量化证据。提交/同步时间采用 README 中的非自指规则：文件记录快照前已经完成并验证的发布，包含该文件的 commit 的远端一致性由外部 Git 校验，并在下一次实质性更新中滚动写入。

## Accepted evidence / 已验收证据

| Lane / 路线 | Accepted result / 已验收结果 | Boundary / 边界 |
|---|---:|---|
| BF16 fidelity | warm N=10 median **1792.202 s** | Same physical A6000; fidelity denominator. / 同一物理 A6000；保真分母。 |
| Turbo 8-step | N=10 median **290.998 s** | Disclosed practical approximation, not BF16 fidelity. / 明示近似实用路线，不是 BF16 保真。 |
| Turbo 4-step | N=10 median **149.619 s** | Faster approximate lane with higher quality risk; not the default. / 更高质量风险的近似路线，不是默认方案。 |
| Sol-Attn r8 | median HTTP-time improvement **15.203%**, 10/10 pairs | **Only** the formal matched 5-step, same-GPU, opt-in lane; not a 50-step BF16, long-video, or semantic-quality result. / **仅限**正式同卡匹配的 5-step opt-in 通道；不是 50-step BF16、长视频或语义质量结论。 |
| Sol-Attn pair-value-halves harness | retained default-off synthetic kernel candidate: total median **145.399 ms** vs current prefix-skip **175.567 ms**; forward subphase **130.043 ms** vs **158.903 ms**; `max_abs_valid=0` | Synthetic/model-free only; no H3 E2E, long-video, BF16-fidelity, product, normal-PC, public-comparison, or SOTA claim. / 仅限合成、无模型 kernel 证据；不是 H3 端到端、长视频、BF16 保真、产品、普通电脑、公开对比或 SOTA 声明。 |

Single-A6000 evidence remains separate from any multi-GPU production study. Structural AV checks are not substitutes for an explicit human quality gate.

单卡 A6000 证据与多卡生产研究保持分离；结构化音视频检查不能替代明确的人类质量验收。

## Recent negative results / 近期负结果

- **Sol-Attn r6:** all 208 calls declined with `unsupported_contiguity`; `sparse_calls=0`. The small dense/opt-in timing difference was not accepted as acceleration.
- **Sol-Attn r7:** packed H3 layout metadata did not reach the backend; it failed closed with `sparse_calls=0`.
- **DLO resident_layers=16:** the 50-step warm candidate improved **0.456%**, below the BF16 baseline CV of **0.837%**; no formal N=10 promotion.
- **Exact-kernel route:** RoPE/all-exact changed audiovisual output; SwiGLU retained no accepted end-to-end gain. Kernel-only microbenchmarks were not promoted to product speedups.
- **Sol-Attn model-free harnesses:** the legacy toy harness had sparse median slower than dense and was not deployed; the 2026-08-13 stride-aware-V SM86 harness is correctness/zero-materialization evidence only (`sparse` median **0.419968 ms** vs dense **0.131072 ms**) and is not an H3 end-to-end or product speedup. The later pair-value-halves harness is retained only as synthetic/model-free kernel evidence, not as a product result.

以上失败均保留为可复现边界，不因后续 r8 成功而删除或改写。

## Hypothesis under test / 待检验假设

The r8 real-chain profile records **192 Q/K/V materialization events totaling 105,344,139,264 bytes per 5-step run**. The falsifiable hypothesis is that a layout-aware view/packing path can remove most of this materialization while preserving metadata validation, dense fail-closed behavior, structural correctness, and the same bounded quality threshold. The latest retained non-Docker SM86 harness result (`sol_attn_pair_value_halves_20260813T110953Z`) tests default-off `MINIMAX_H3_A6000_SOL_ATTN_PAIR_VALUE_HALVES=1`: it keeps the stride-aware fused-QKV V view, full-prefix-block skip, dense-prefix overwrite, tau/routing, and exact-block order, but shares one Q/K route and online-softmax probability stream across the two BV64 value halves. It reports `max_abs_valid=0`, prefix/tail equality to the current path, zero padding rows, `sparse_calls=7`, `fallback_calls=0`, `stride_aware_value_calls=7`, and zero materialized copy events/bytes. Its synthetic model-free median was candidate total **145.399 ms** vs current prefix-skip **175.567 ms** (forward subphase **130.043 ms** vs **158.903 ms**). It loads no H3 model; the benchmark is kernel/model-free only and is not a product speedup. Success still requires a matched H3 end-to-end gate on the same physical GPU; a microbenchmark alone is insufficient.

r8 真实链路在每次 5-step 运行中记录了 **192 次 Q/K/V 实体化，共 105,344,139,264 字节**。可证伪假设是：布局感知的 view/packing 路径能够消除其中大部分复制，同时保持 metadata 校验、异常时回退 dense、结构正确性和相同质量阈值。最新保留的非 Docker SM86 harness 结果（`sol_attn_pair_value_halves_20260813T110953Z`）测试默认关闭的 `MINIMAX_H3_A6000_SOL_ATTN_PAIR_VALUE_HALVES=1`：它保留 stride-aware fused-QKV V view、完整 prefix block skip、dense-prefix overwrite、tau/routing 与 exact-block 顺序，但让两个 BV64 value halves 共用一次 Q/K 路由和 online-softmax 概率流。结果为 `max_abs_valid=0`、prefix/tail 与当前路径相等、padding 行为 0、`sparse_calls=7`、`fallback_calls=0`、`stride_aware_value_calls=7`，且 materialized copy 次数/字节为 0。合成无模型 median 为 candidate total **145.399 ms**，current prefix-skip **175.567 ms**（forward 子阶段 **130.043 ms** 对 **158.903 ms**）。该结果没有加载 H3 模型；benchmark 仅限 kernel/model-free，不是产品级加速。成功仍必须体现为同一物理 GPU 上的 H3 匹配端到端 gate；仅有微基准不成立。

## Benchmark contract status / 基准合同状态

The canonical public v1 contract is at [`benchmark_contract/v1/README.md`](benchmark_contract/v1/README.md), with a machine-readable schema, fail-closed validator, normalized historical records, and three dry-run lane manifests. Full private and sanitized-export tests pass, as does the publication audit with the operator terms file. Independent review accepted contract correctness, first-party revision/license grounding, statistical and claim boundaries, final-AV/timing semantics, and public hygiene. The pinned MiniMax-H3/vLLM-Omni path supports native output only from 4 to 15 seconds; therefore the 30/60-second manifests remain explicitly **unmeasured `extension` lanes**, not native-long-context results. Review and publication status are gates, not benchmark results.

规范 v1 已公开于 [`benchmark_contract/v1/README.md`](benchmark_contract/v1/README.md)，包括机器可读 Schema、fail-closed validator、历史证据规范化记录和三个 dry-run lane manifest；完整私有测试、sanitized export 测试以及使用 operator terms 文件的 publication audit 均已通过。独立审阅已验收合同正确性、第一方版本与许可证依据、统计与声明边界、最终音视频与计时语义以及公共树卫生。当前 pinned MiniMax-H3/vLLM-Omni 原生输出仅支持 4–15 秒，因此 30/60 秒 manifest 仍明确标记为**尚未实测的 `extension` 路线**，不是原生长上下文结果。审阅与发布状态是门槛，不是 benchmark 结果。

## Next experiment / 下一实验

**Non-Docker pair-value-halves harness passed; real-chain Docker gate remains storage-blocked.** Contract v1 and prior stride-aware-V publication gates cleared earlier review/audit; this separate pair-value-halves Sol-Attn milestone still requires its own fresh export fingerprint, audit, and independent Reviewer acceptance before that Sol-Attn publication. The next real-chain experiment after the separate Sol-Attn publication decision is one matched N=1 small gate that changes only Q/K/V materialization, checks dense fallback and output/AV correctness, and measures copy bytes plus comparable end-to-end timing. Promote to N=3 and later N≥10 only if a real-chain gate passes. A 2026-08-13 preflight found all four A6000s idle, but the pinned r2 base image and r8/r9 Sol-Attn overlay images are not inspectable in the local Docker daemon. The operator-authorized r2 restore/rebuild from the pinned vLLM-Omni commit `8e2e9b6b53e86e6a479ed2c0a53782f655f60e04` and pinned base digest `docker.io/vllm/vllm-openai@sha256:770fe65b2c73ee74a5c42165cf3433de4048cc2cd9c57a937ca4e35aba5aa87b` failed while registering the pulled base layer with `layerdb/...: file exists`, recorded at `technical_report/evidence/minimax_h3_desktop/packaging/minimax-h3-r2-restore-build-20260813T074702Z/r2_restore_docker_storage_blocker.json`. This is an operator/admin Docker storage blocker; no identical Docker restore/build retry should be run until the daemon storage conflict is repaired or a clean daemon with the pinned r2 image is available. No pair-value-halves real-chain run, GPU model inference, H3 E2E result, or new product speedup claim was produced.

**非 Docker pair-value-halves harness 已通过；真实链路 Docker gate 仍被存储状态阻塞。** Contract v1 与此前 stride-aware-V 发布门槛已经通过早前审阅/审计；这个独立的 pair-value-halves Sol-Attn milestone 仍需要绑定自身最新 export fingerprint 的审计和独立 Reviewer 验收后，才能作为 Sol-Attn milestone 发布。该独立 Sol-Attn 发布决策后的下一真实链路实验将只改变 Q/K/V 实体化这一项，执行匹配 N=1 小门槛，验证 dense 回退与输出/音视频正确性，并同时测量复制字节数和可比端到端耗时。只有真实链路 gate 通过后才晋级 N=3，最终再考虑 N≥10。2026-08-13 的预检显示四张 A6000 均空闲，但本地 Docker daemon 中无法 inspect 到固定的 r2 基础镜像以及 r8/r9 Sol-Attn overlay 镜像；按 operator 授权从固定 vLLM-Omni commit `8e2e9b6b53e86e6a479ed2c0a53782f655f60e04` 与固定基础 digest `docker.io/vllm/vllm-openai@sha256:770fe65b2c73ee74a5c42165cf3433de4048cc2cd9c57a937ca4e35aba5aa87b` 重建 r2 时，在注册拉取的基础层时失败：`layerdb/...: file exists`，阻塞证据记录于 `technical_report/evidence/minimax_h3_desktop/packaging/minimax-h3-r2-restore-build-20260813T074702Z/r2_restore_docker_storage_blocker.json`。这属于 operator/admin 级 Docker 存储阻塞；在 daemon 存储冲突修复或干净 daemon 已有固定 r2 镜像之前，不应重复同一 Docker restore/build 命令。本轮没有产生 pair-value-halves 真实链路运行、GPU 模型推理、H3 端到端结果或新的产品级加速声明。

## Workflow kickoff and source tracking / 工作流启动与来源跟踪

Workflow A restarted on **2026-08-12** from the accepted real-chain profile above. The first-party source baseline is the pinned Sana Sol-Engine revision `d00eef311670a58deb2c323fe072738fcb945600` and vLLM-Omni revision `8e2e9b6b53e86e6a479ed2c0a53782f655f60e04`. The Sana repository README declares Apache-2.0 for repository code; its Sol-Attn third-party notice retains BSD-3-Clause FlashAttention attribution. MiniMax-H3 model revision `6818f6c32d12b210915e44ad56a4228c2608f160` and Turbo assets remain separately licensed and are not distributed here. The v1 source ledger records revision dates and license scope for MiniMax-H3, vLLM-Omni, and Sana/Sol-Engine, and records that only the FL2VA partition is prepared locally. See [`benchmark_contract/v1/contract.json`](benchmark_contract/v1/contract.json), [`NOTICE`](NOTICE), and [`ports/minimax_h3_a6000/UPSTREAM.md`](ports/minimax_h3_a6000/UPSTREAM.md).

工作流 A 于 **2026-08-12** 依据上述真实链路 profile 重新启动。第一方源码基线固定为 Sana Sol-Engine `d00eef311670a58deb2c323fe072738fcb945600` 与 vLLM-Omni `8e2e9b6b53e86e6a479ed2c0a53782f655f60e04`。Sana 仓库 README 声明代码采用 Apache-2.0，Sol-Attn 的第三方通知保留 FlashAttention BSD-3-Clause 归属。MiniMax-H3 模型 revision `6818f6c32d12b210915e44ad56a4228c2608f160` 与 Turbo 资产保持独立许可，本仓库不分发其权重。
