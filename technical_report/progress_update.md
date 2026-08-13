# MiniMax-H3 A6000 验证进度

> 更新时间：2026-08-13T15:05:00+00:00
> 仅汇总已落盘证据；实验中或缺失结果明确标记，不估算。

## 固定工作负载

- 单张RTX A6000 48GB；完整FL2VA；1344×768；5.166667秒；124帧；24FPS；32kHz立体声音频。

## 已认证性能

| 路径 | Steps | N | Median(s) | Speedup |
|---|---:|---:|---:|---:|
| BF16 fidelity | 50 | 10 | 1792.202 | 1.000× |
| Turbo 8-step | 8 | 10 | 290.998 | 6.159× |
| Turbo 4-step | 4 | 10 | 149.619 | 11.978× |

## DLO

- 状态：dlo_candidate50_complete；resident_layers=16；warm=1784.069秒；改善=0.456%；正式N10建议=False。
- 当前边界：0.456%候选改善低于baseline warm CV 0.837%，没有正式DLO N10晋级证据。

## Sol-Attn r8

- 5-step gate分类：sparse_runtime_valid_5step_diagnostic；final=candidate_pass；release_manifest_eligible=False。
- sparse_candidate_calls=192；sparse_calls=192；dense_calls=16；fallback_calls=0；declines={'dense_first_layers': 8, 'non_h3_dit_attention_prefix': 8}。
- density_samples=192；materialized_copy=192次/105344139264 bytes；peak_memory=27354.0 MiB；peak_temp=84.0C；peak_power=299.88W。
- 边界：这是5-step metadata-plumbing sparse-execution diagnostic candidate pass，不是speedup、N10、BF16 fidelity、release或质量等价声明。
- matched retest route-decision：proceed_to_formal_n10_candidate；completed_pairs=3/3；median_http_time_improvement=14.782%；threshold=3.000%；failed_gates=[]；n10_recommendation=proceed_to_formal_n10_candidate。
- matched retest evidence：decision=sol_engine_port/sol_attn_h3_matched_retest_r8_n3_20260812T013544Z/decision.json；terminal_recheck=delivery/r8_matched_retest_terminal_recheck_20260812T024043Z/summary.json。
- matched retest边界：这是terminal N=3 route gate，已导向后续formal N>=10终端接受；N=3本身仍不是正式speedup、BF16 fidelity、release或质量等价声明，也不替代人类听感。
- matched retest reason：bounded matched retest passed correctness, sparse-runtime, resource, quality-proxy, and timing gates

- formal N>=10：accepted_formal_n10_same_gpu_sol_attn_speed_candidate；completed_pairs=10/10；median_http_time_improvement=15.203%；threshold=3.000%；same_expected_gpu=True；supervisor_status=complete。
- formal N>=10 evidence：decision=sol_engine_port/sol_attn_h3_formal_n10_r8_n10_20260812T031757Z/formal_n10_decision.json；summary=sol_engine_port/sol_attn_h3_formal_n10_r8_n10_20260812T031757Z/formal_n10_summary.json；terminal_artifacts={'formal_n10_decision.json': True, 'RUN_REPORT.md': True, 'formal_n10_summary.json': True, 'timing_summary.json': True, 'quality_proxy_comparison.json': True, 'resource_summary.json': True}。
- formal N>=10边界：仅限formal 5-step Sol-Attn opt-in matched-workload lane；不是BF16 fidelity、release或人类听感/语义质量认证。

## Non-Docker stride-aware V harness / 非 Docker stride-aware V harness

- evidence：`sol_engine_port/sol_attn_stride_aware_v_harness_20260813T082456Z/summary.json` 与 `harness.json`。
- GPU：`GPU-b3425477-0877-24de-c5b8-1549ab47cd4b`；Argus lease supervision；harness 内 exactly one visible CUDA device；model_load=False。
- correctness：compile_status=compiled_and_launched；stride_aware_value_calls=1；sparse_calls=1；fallback_calls=0；materialize_copy_count=0；materialize_copy_bytes=0；prefix_rows_equal_dense=True；padding_rows_zero=True。
- benchmark：warmup=20；repeats=100；shape B=1,T_total=512,T_valid=448,H=8,D=128；sparse median=0.419968 ms；dense median=0.131072 ms。
- 边界：这是 kernel/model-free correctness 与 zero-materialization 证据；不是 H3 E2E、真实链路 speedup、长视频结果或产品质量证据。

## Non-Docker pair-value-halves captured replay / 非 Docker pair-value-halves captured replay

- evidence：`sol_engine_port/sol_attn_pair_value_halves_captured_replay_20260813T145731Z/summary.json`、`decision.json` 与 `validation.json`；前序 standalone harness evidence 仍保留在 `sol_engine_port/sol_attn_pair_value_halves_20260813T110953Z/summary.json` 与 `harness.json`。
- GPU：`GPU-b3425477-0877-24de-c5b8-1549ab47cd4b`；Argus lease supervision；replay 内 exactly one visible CUDA device；model_load=False；docker_used_for_run=False。
- captured source：复用 r8 real-chain telemetry 的 `sparse_calls=192`、`fallback_calls=0` 与 materialized_copy=192次/105344139264 bytes，并验证 captured Q/K/V layout 与 metadata helper derivation。
- mechanism：默认关闭 `MINIMAX_H3_A6000_SOL_ATTN_PAIR_VALUE_HALVES`；保留 stride-aware fused-QKV V、full-prefix-block skip、dense-prefix overwrite、tau/routing 与 exact-block order，只让两个 BV64 value halves 共用一次 Q/K route 与 online-softmax probability stream。
- correctness/telemetry：`max_abs_valid=0`；candidate prefix/tail 与 current 相等；padding rows zero；captured layout agreement=True；metadata derivation pass=True；candidate/current replay lanes 均 zero fallback 与 zero unintended materialization。
- replay benchmark：shape B=1,T_total=38272,T_valid=38247,H=56,D=128,prefix=951；candidate total median=144.652 ms；current prefix-skip total median=175.216 ms；candidate forward pointer median=129.850 ms；current forward pointer median=158.161 ms。
- 边界：这是 captured-metadata/non-Docker/model-free kernel evidence；不是 H3 E2E、长视频、BF16 fidelity、产品加速、普通电脑、公开对比或 SOTA 声明；下一步仍需要真实链路匹配 N=1 gate。

## Clean-room one-command local lifecycle

- 状态：pass；publication_audit=pass；local资源只读检查=81 files/144051182625 bytes。
- 边界：packaging/deployment evidence only；未启动容器、未加载/修改权重、未运行GPU inference/媒体生成，不产生速度/质量/保真声明。

## Final delivery gates

- CPU/static gate：pass；evidence=delivery/final_cpu_static_gate_20260812T033107Z。
- strict aggregation/export/publication audit：pass；export_file_count=82；publication_issue_count=0；evidence=delivery/final_decisive_export_audit_20260812T025605Z。
- formal N10 report-sync export/publication audit：pass；export_file_count=89；publication_issue_count=0；reviewer_status=accepted_independent_reviewer_passed；push_performed=True；evidence=delivery/formal_n10_cpu_sync_export_audit_20260812T065502Z。
- 边界：这些是CPU/static/export/audit gate，不产生GPU、Docker-run、model-load、保真或质量新声明；formal N10速度候选只来自已落盘GPU证据，不由CPU sync gate新产生。

## 当前边界

- BF16 fidelity lane仅包含baseline；Turbo属于practical_disclosed_approx，不得混入无损结论。
- Sol-Attn r8 formal N>=10 matched-workload gate已终端接受：仅限formal 5-step Sol-Attn opt-in lane；不是BF16 fidelity、release或人类听感/语义质量认证。
- 2026-08-13 non-Docker stride-aware-V harness 已证明当前 kernel path 在合成 fused-QKV V view 上 compile/launch 且 zero materialization；后续 pair-value-halves captured replay 在同一真实 H3 token/head 形状上保留默认关闭候选，candidate total median=144.652 ms vs current prefix-skip 175.216 ms，且 `max_abs_valid=0`、replay lanes zero unintended materialization。二者仍是 captured-metadata/model-free，不是 H3 E2E speedup。
- Turbo自动结构/音频指标已完成，操作者总体播放/听感验收已记录；8-step保持默认，已知4-step视觉失败继续保留。
- DMD/DMD2在无合法可复现H3 recipe/checkpoint时保持blocked。

## 下一步

1. 本次 pair-value-halves source candidate 需要 fresh sanitized export fingerprint、private/export tests、publication audit、hygiene scans 与独立 Reviewer 验收；不得复用旧 `a705...` 审阅。
2. 如果 Docker daemon 存储冲突被管理员修复或干净 daemon 已有固定 r2/r8/r9 镜像，运行只改变 Q/K/V materialization 的 real-chain N=1 gate；否则仅继续独立的 non-Docker/model-free 工作并明确边界。
3. 既有 Turbo 与 r8 formal-N10 证据边界保持不变；操作者总体播放/听感验收已完成，8-step 保持默认，已知 4-step 视觉失败继续保留。
