# MiniMax-H3 A6000 验证进度

> 更新时间：2026-08-16T20:00:31Z
> 仅汇总已落盘证据；实验中或缺失结果明确标记，不估算。

## 固定工作负载与边界

- 短片基线：单张 RTX A6000 48GB；完整 FL2VA；1344×768；5.166667 秒；124 帧；24 FPS；32 kHz 立体声音频。
- 长视频路线：当前 pinned MiniMax-H3/vLLM-Omni 原生只支持 4–15 秒；30/60 秒必须标记为 `extension`、`chunked_overlap` 或 montage/stitching，不能称为 native long context。
- 30 秒 accepted r10 结果属于 six-chunk `extension`/chunked final-AV practical lane；不是 BF16 fidelity、人类语义/音频质量、产品就绪、公开对比或 SOTA。

## 已认证性能

| 路径 | Steps | N | Median(s) | Scope |
|---|---:|---:|---:|---|
| BF16 fidelity short | 50 | 10 | 1792.202 | native short fidelity denominator |
| Turbo 8-step short | 8 | 10 | 290.998 | practical approximate short lane |
| Turbo 4-step short | 4 | 10 | 149.619 | faster practical short lane with higher quality risk |
| Sol-Attn r8 short | 5 | 10 pairs | 15.203% median HTTP-time improvement | matched 5-step opt-in only |
| Final-AV 30s r10 | 8 | 10 pairs | warm E2E 1333.575 vs retained r9 1394.006; 4.326% median improvement | matched six-chunk extension/chunked lane only |

## Final-AV 30s r10 formal N=10

- Evidence：`long_video/final-av-30s-r10-step3-guarded-adaptive-sol-attn-formal-n10-20260816T052452Z/formal_n10_summary.json`、`timing_summary.json`、`final_av_accounting_summary.json`、`resource_summary.json`、`quality_proxy_comparison.json`、`reviewer_verdict.json`。
- Reviewer：`accepted_independent_reviewer_passed`；verdict=`accept`。
- Matched design：candidate `r10_adaptive_tau1_5_step3_diag` vs retained `r9_current_sol_attn`；same workload fingerprint；same timing boundary；same single RTX A6000。
- Timing：candidate warm median **1333.5752375134907 s**；reference warm median **1394.0061285260017 s**；median warm improvement **4.326262968443439%**。candidate cold median **1814.1335341165 s**；reference cold median **1884.1419612035 s**。
- Final AV：10/10 pairs complete；每个 candidate/reference 输出均为 1344×768、720 frames、24 FPS、32 kHz stereo、960000 effective samples/channel。
- Resources：candidate median peak GPU **27946 MiB**，host **250.03582000732422 GiB**，power **300.32 W**，failures **0**。
- Boundary：只接受 matched formal N>=10 30 秒 final-AV extension/chunked practical-lane timing/structural claim；不接受 native long context、BF16 fidelity、人类语义/音频/AV质量、产品就绪、公开对比或 SOTA。

## Final-AV 30s r11 step-min=2 N=1 gate（拒绝/不晋级）

- Evidence：`long_video/final-av-30s-r10-vs-r11-step2-sol-attn-chain-20260816T185549Z/`；subagent terminal=`done`/exit_code=0。
- 决策：`n1/decision.json` classification=`reject_r11_adaptive_tau1_5_step2_diag_30s_long_lane_slower_or_proxy_regression`；`n1_route_status.json` status=`reject`、`n1_pass_for_n3=false`；未生成 `n3/decision.json`，不启动 N=3。
- Principal variable：retained r10 `r10_adaptive_tau1_5_step3_diag` vs r11 `r11_adaptive_tau1_5_step2_diag`，仅将 guarded-adaptive Sol-Attn `adaptive_step_min` 从 3 降到 2。
- Timing（N=1 route gate only，不是 speedup claim）：r10 warm E2E **1333.905 s**；r11 warm E2E **1318.763 s**；单样本 warm delta **1.135%**。
- Telemetry/proof：same physical GPU/workload/timing boundary；reference/candidate 均 sparse=2352、fallback=0、materialize=0/0 bytes、input_copy=0/0 bytes；final AV 均为 1344×768、720 frames、24 FPS、32 kHz stereo、960000 effective samples/channel。
- Failed gate：`objective_5pct_noninferiority_core_metrics`；失败指标包括 subject identity、background、camera、motion。尽管 r11 自动 proxy flags 不多于 r10，objective proxy non-inferiority 未通过，因此拒绝 r11 step-min=2 且无 promotion/reviewer/speedup 声明。
- Wrapper caveat：root `chain_decision.json` 写成 `status=complete` 并带 stale `n3_decision` path；以 `n1_route_status.json`、缺失 `n3/decision.json` 和 subagent stdout 中 GPU lease inner `returncode=17` 为准，分类为 N=1 no-promotion terminal packet。

## DLO

- 状态：dlo_candidate50_complete；resident_layers=16；warm=1784.069 秒；改善=0.456%；正式 N10 建议=False。
- 当前边界：0.456% 候选改善低于 baseline warm CV 0.837%，没有正式 DLO N10 晋级证据。

## Sol-Attn r8 short-lane

- 5-step gate 分类：sparse_runtime_valid_5step_diagnostic；sparse_candidate_calls=192；sparse_calls=192；fallback_calls=0；materialized_copy=192 次 / 105344139264 bytes。
- Formal N>=10：`accepted_formal_n10_same_gpu_sol_attn_speed_candidate`；completed_pairs=10/10；median HTTP-time improvement=15.203%；threshold=3.000%。
- 边界：仅限 formal 5-step Sol-Attn opt-in matched-workload short lane；不是 BF16 fidelity、release 或人类听感/语义质量认证。

## Non-Docker model-free evidence

- Stride-aware V harness：compile/launch、zero materialization correctness evidence only；sparse median 0.419968 ms vs dense 0.131072 ms，不是 H3 E2E 或产品 speedup。
- Pair-value-halves captured replay：default-off candidate；candidate total median 144.652 ms vs current prefix-skip 175.216 ms；`max_abs_valid=0`；仍是 captured-metadata/non-Docker/model-free，不是 H3 E2E、长视频、BF16 fidelity 或产品 speedup。

## Final delivery/publication gates

- 既有 CPU/static/export/audit gates 已通过；当前 r10 public update 必须重新通过 sanitized export、publication audit、terms/secret/large-file/model/cache/raw-log/private-path checks、Reviewer-bound claim acceptance、non-force push 和 `origin/main == HEAD`。

## 下一步

1. 保留 r11 step-min=2 N=1 no-promotion packet；不要因单样本 warm delta 或 root wrapper `complete` 字段启动 N=3、N>=10 或 speedup claim。
2. 如继续优化，需 Planner/Reviewer 重新指定一个不同的 bounded candidate 或独立 60 秒 extension/chunked gate；所有 30/60 秒输出继续与 true native long-context 分离。
3. 人类语义/音频质量 gate 与产品就绪声明仍未验收，不由结构 AV 或 objective proxies 替代。
