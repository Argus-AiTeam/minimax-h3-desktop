# MiniMax-H3 A6000 Performance Report

Status: **generated_cpu_evidence_reader**. CPU-only evidence reader: no GPU, Docker, model loading, network, downloads, publication, benchmarks, or live evidence mutation.
Schema: `argus-minimax-h3-a6000-performance-report-v1`.
Evidence root: `technical_report/evidence/minimax_h3_desktop`.

## Scope and lane boundaries

- Fidelity lane: `fidelity_bf16_exact`. Only evidence explicitly retained in this lane is treated as BF16-exact.
- Practical lane: `practical_disclosed_approx`. Turbo merged LoRA and diagnostic acceleration work stay here unless later evidence changes the boundary.
- Turbo practical results must not be relabeled as BF16-exact/fidelity results.
- Missing DLO/Sol-Attn/DMD/AV evidence is reported as **pending** or **blocked**; no missing value is estimated.

## Platform and workload

- Platform: `single_a6000_48gb_workstation`; physical device `{'compute_capability': '8.6', 'host_gpu_index': 3, 'uuid': 'GPU-ff0c8d25-2652-ce58-3ef7-e3a9aeeb3334'}`.
- Workload: 1344x768, 5.166667s, 124 frames, 24 FPS.
- Task/model: task `t2va`, partition `FL2VA`, baseline dense steps `50`.
- Audio: 2 channels at 32000 Hz.
- Prompt/workload attribution: NVlabs/Sana sol-engine t2va_example_1 at d00eef311670a58deb2c323fe072738fcb945600. Workload/prompt source preserves NVLabs/Sana Sol-Engine team attribution.

## Baseline v2 BF16-exact denominator

- Evidence: `baseline_a6000/baseline_certification.json`.
- Status: `certified_internal_same_physical_device_baseline`; track `fidelity_bf16_exact`; schema `argus-h3-a6000-fidelity-baseline-certification-v2`.
- All requests: N=13, median=1780.975666s, mean=1788.1300204615384s, CV=0.7777501736471782%.
- Warm-primary denominator: N=10, median=1792.2021025s, mean=1790.7376617s, CV=0.8371622556580874%.
- Session-first requests: N=3; service sessions=3.
- Resource envelope: peak GPU memory 26836.0 MiB; peak host memory 204.83538436889648 GiB; peak temperature 84.0 C; peak power 302.23 W.
- Structural AV pass count: 13/13.
- Claim boundary: Internal same-physical-device A6000 BF16 fidelity baseline. Warm N=10 is the primary speed denominator. Not an optimized result or external reproduction.

## Turbo GPU3 paired practical timing and quality

- Run: `gpu3_turbo_paired_n10_20260810T025102Z`; status `pass_same_physical_device_paired_n10`; track `practical_disclosed_approx`.
- Baseline denominator: {'kind': 'warm_requests_primary_denominator', 'median_s': 1792.2021025, 'n': 10, 'path': 'technical_report/evidence/minimax_h3_desktop/baseline_a6000/baseline_certification.json', 'same_physical_device': True}.
- Merge: status `completed`, strength 1.0, completed shards 13.
- Formal paired N per schedule: 10; excluded warmups: {'4step': {'http_code': 200, 'latency_s': 153.297009, 'size_download_bytes': 6858247}, '8step': {'http_code': 200, 'latency_s': 291.018631, 'size_download_bytes': 5767966}}.
- Resource envelope: peak GPU memory 26836.0 MiB; peak host memory 195.16017150878906 GiB; peak temperature 83.0 C; peak power 301.08 W.
- AV timing outputs: structural pass 20/20; missing/unreadable AV 0.

| Schedule | Steps | N | Median (s) | Mean (s) | CV (%) | Speedup vs BF16 warm N10 | Lane |
|---|---:|---:|---:|---:|---:|---:|---|
| 4-step | 4 | 10 | 149.6191865 | 149.6224431 | 0.033194157523981374 | 11.978424321268447x | `practical_disclosed_approx` |
| 8-step | 8 | 10 | 290.9976015 | 290.9859656 | 0.04878312051625522 | 6.158820874336313x | `practical_disclosed_approx` |

- Quality suite: status `structural_av_suite_pass_semantic_quality_not_certified`, cases=24, pairs=12, structural AV pass=True.
- AV/semantic boundary: human auditory listening `operator_overall_playback_listening_accepted`; semantic_quality_certified=True; quality certification `operator_accepted_practical_8step_with_known_4step_visual_failure_preserved`.
- Practical recommendation boundary: 8-step remains the default practical candidate; 4-step is ultra-fast/quality-cost experimental, not fidelity evidence.

## DLO resident-layer optimization

- Plan status: `candidate_plan_derived_from_local_weight_headers`; baseline resident_layers=12; candidates=[13, 16, 18].
- Detached continuation: {'status': 'dlo_candidate50_complete', 'context_env': {'started_utc': '2026-08-10T13:45:25Z', 'finished_utc': '2026-08-10T15:24:17Z'}, 'rl13_run_id': 'a6000_dlo_capacity_5step_rl13_20260810T072019Z', 'rl16_run_id': 'a6000_dlo_capacity_5step_rl16_20260810T113157Z', 'rl18_run_id': 'a6000_dlo_capacity_5step_rl18_20260810T134526Z', 'candidate50_run_id': 'a6000_dlo_candidate_50_rl16_20260810T141257Z', 'latest_run_id': 'a6000_dlo_candidate_50_rl16_20260810T141257Z'}.
- Capacity gates are not formal 50-step/N10 performance unless separately marked present.

| Resident layers | Stage | Result status | Gate status | Baseline 5-step (s) | Candidate 5-step (s) | Speedup | Hash match | Resource |
|---:|---|---|---|---:|---:|---:|---|---|
| 13 | capacity-5step | `present` | `pass` | 188.098444 | 186.773476 | 1.0070939837303237x | True | peak GPU memory 28060.0 MiB; peak temperature 82.0 C |
| 16 | capacity-5step | `infrastructure_interrupted_no_capacity_result` | `infrastructure_abort_no_capacity_result` | pending | pending | pending | pending | pending |
| 16 | capacity-5step | `present` | `pass` | pending | pending | pending | True | pending |
| 18 | capacity-5step | `present` | `pass` | pending | pending | pending | True | pending |

- Candidate-50: **present** — evidence present.
- Formal DLO N10: **pending** — no DLO formal N10 timing evidence found.

## Exact-kernel diagnostics and Sol-Attn

- Exact-kernel lane: `diagnostic_exact_kernel_candidates_not_deployed_as_certified_speedups`; microbenchmark scope `kernel_candidates_only_not_h3_e2e`; model_load=False.

| Kernel | N | PyTorch median (ms) | Triton candidate median (ms) | Median speedup |
|---|---:|---:|---:|---:|
| `indexed_gate_38247x5376` | 100 | 25.879040718078613 | 2.2195039987564087 | 11.659830634492517x |
| `indexed_modulate_38247x5376` | 100 | 33.603071212768555 | 1.5257600545883179 | 22.023824199431782x |
| `rope_1024x56x128_rot96` | 100 | 0.9779199957847595 | 0.15052799880504608 | 6.496598663025454x |
| `swiglu_gate_up_8192x28672` | 100 | 10.017199993133545 | 1.2379359602928162 | 8.091856375804866x |
| `swiglu_value_gate_8192x28672` | 100 | 10.027008056640625 | 1.2369920015335083 | 8.10596030064064x |

E2E diagnostic ablation quality/telemetry:
- `adaln`: video_mean_mse=0.0; audio_waveform_cosine=0.9999999999999999; latency_s=187.632019; telemetry_ops=['apply_rope_bf16', 'indexed_gate_bf16', 'indexed_modulate_bf16', 'swiglu_bf16'].
- `rope`: video_mean_mse=224.11484810613817; audio_waveform_cosine=0.9775986604966458; latency_s=188.788426; telemetry_ops=['apply_rope_bf16', 'indexed_gate_bf16', 'indexed_modulate_bf16', 'swiglu_bf16'].
- `all_exact`: video_mean_mse=224.11484810613817; audio_waveform_cosine=0.9775986604966458; latency_s=186.382043; telemetry_ops=['apply_rope_bf16', 'indexed_gate_bf16', 'indexed_modulate_bf16', 'swiglu_bf16'].
- `swiglu`: video_mean_mse=0.0; audio_waveform_cosine=0.9999999999999999; latency_s=189.553095; telemetry_ops=['apply_rope_bf16', 'indexed_gate_bf16', 'indexed_modulate_bf16', 'swiglu_bf16'].
- Exact-kernel diagnostics are retained as diagnostic/candidate evidence only unless a separate accepted N10 speed result exists.
- Sol-Attn legacy toy/kernel diagnostic: model_load=False; run_dir=`sol_engine_port/sol_attn_gpu_20260809T173323Z`.
- Sol-Attn H3 diagnostic deployment boundary: `accepted_5step_diagnostic_not_release_manifest_eligible`.
- Sol-Attn toy/kernel bench: dense median=0.12390399724245071 ms; sparse median=0.40243199467658997 ms; dense/sparse median speedup=0.30788803793302966.
- Sol-Attn stride-aware-V SM86 harness: evidence `sol_engine_port/sol_attn_stride_aware_v_harness_20260813T082456Z/summary.json` and `harness.json`; model_load=False; GPU UUID `GPU-b3425477-0877-24de-c5b8-1549ab47cd4b`; one visible GPU under Argus lease supervision. Correctness compiled/launched on SM86, exercised `stride_aware_value_calls=1`, `sparse_calls=1`, `fallback_calls=0`, `materialize_copy_count=0`, `materialize_copy_bytes=0`, `prefix_rows_equal_dense=True`, and `padding_rows_zero=True`. Harness benchmark policy was warmup=20/repeats=100 on shape B=1,T_total=512,T_valid=448,H=8,D=128; sparse median=0.41996800899505615 ms and dense median=0.131071999669075 ms. Boundary: kernel/model-free zero-materialization correctness evidence only; not H3 E2E, not real-chain speedup, not long-video, and not product-quality evidence.
- Sol-Attn r8 supervisor (current selected run by readable workload/version-label provenance, not run-id text prefix): status `complete`, latest_run_id `sol_attn_h3_gpu2_5step_r8_prompt0644_20260812T005600Z`, classified `sparse_runtime_valid_5step_diagnostic`.
- Sol-Attn r8 readable provenance: image_tag=`argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r8-sol-attn-overlay`, version=`r8`, required_version=`r8`, title=`MiniMax-H3 A6000 r8 Sol-Attn integration overlay`; opaque image/output identifiers are omitted and are not classification evidence.
- Sol-Attn r8 HTTP timing: dense=186.498762s, opt-in=158.923988s, dense/opt-in timing ratio (diagnostic only, not a speedup claim)=1.173509199882399x.
- Sol-Attn r8 telemetry: sparse_candidates=192, sparse_calls=192, fallback_calls=0, materialized_copy_calls=192, materialized_copy_bytes=105344139264, declines={'dense_first_layers': 8, 'non_h3_dit_attention_prefix': 8}, density_samples=192.
- Sol-Attn r8 resource envelope: peak GPU memory 27354.0 MiB; peak temperature 84.0 C; peak power 299.88 W.
- Sol-Attn H3 end-to-end: **sparse_runtime_valid_5step_diagnostic** — Sol-Attn sparse_calls>0 with HTTP 200, structural AV, resource, density, and materialization telemetry; this is only a 5-step sparse-execution diagnostic candidate, not a speedup, N10, BF16 fidelity, release, or quality-equivalence claim.
- Sol-Attn diagnostic boundary: the selected 5-step run may be used only as sparse-execution metadata plumbing evidence; matched-workload quality/correctness and formal performance promotion remain separate follow-up gates.
- Latest r8 matched-workload route decision: `proceed_to_formal_n10_candidate`; evidence `sol_engine_port/sol_attn_h3_matched_retest_r8_n3_20260812T013544Z/decision.json`, terminal recheck `delivery/r8_matched_retest_terminal_recheck_20260812T024043Z/summary.json`. Completed pairs=3/3; median HTTP-time improvement=14.782455716069165%; route threshold>3.0%; failed_gates=[]. This N=3 route gate led to the later accepted formal N>=10 gate; the N=3 gate itself is not formal N10, not a speedup claim, not BF16 fidelity, and not quality-equivalence certification.
- Latest r8 formal N>=10 gate CPU inspection: `accepted_formal_n10_same_gpu_sol_attn_speed_candidate`; evidence `sol_engine_port/sol_attn_h3_formal_n10_r8_n10_20260812T031757Z/formal_n10_decision.json`. Requested pairs=10; started pairs=10; completed pairs=10; supervisor_status=complete; same_expected_gpu=True. N>=10 same-GPU matched workload passed sparse-runtime, structural AV, resource, quality-proxy, and timing gates. Accepted only inside the formal matched 5-step Sol-Attn opt-in lane; not BF16 fidelity, not Turbo/DLO/DMD evidence, not release approval, and not human-auditory/semantic quality certification.

## DMD / DMD2 status

- Status: **blocked_research_only_no_go_after_turbo_unless_feasibility_changes**; track limit `practical_disclosed_approx`.
- Boundary: No DMD/DMD2 speed or quality value is reported unless a first-source H3 recipe/checkpoint appears.

## Pending stages and blockers

- `dlo.formal_n10`: **pending** — no DLO formal N10 timing evidence found.

Blockers:
- `DLO`: **pending** — no DLO formal N10 timing evidence found.
- `DMD/DMD2`: **blocked** — No DMD/DMD2 speed or quality value is reported unless a first-source H3 recipe/checkpoint appears..

## Evidence index

- `baseline_a6000/baseline_certification.json`
- `baseline_a6000/baseline_contract.json`
- `delivery/r8_matched_retest_terminal_recheck_20260812T024043Z/summary.json`
- `dlo_autotune/detached_continuation/candidate50_run_id.txt`
- `dlo_autotune/detached_continuation/context.env`
- `dlo_autotune/detached_continuation/rl13_run_id.txt`
- `dlo_autotune/detached_continuation/rl16_run_id.txt`
- `dlo_autotune/detached_continuation/rl18_run_id.txt`
- `dlo_autotune/detached_continuation/status.txt`
- `dlo_autotune/resident_layer_candidates.json`
- `dlo_autotune/runs/LATEST_RUN_ID`
- `dlo_autotune/runs/a6000_dlo_candidate_50_rl16_20260810T141257Z/candidate50_summary.json`
- `dlo_autotune/runs/a6000_dlo_capacity_5step_rl13_20260810T072019Z`
- `dlo_autotune/runs/a6000_dlo_capacity_5step_rl13_20260810T072019Z/capacity_gate_verdict.json`
- `dlo_autotune/runs/a6000_dlo_capacity_5step_rl16_20260810T111156Z`
- `dlo_autotune/runs/a6000_dlo_capacity_5step_rl16_20260810T113157Z`
- `dlo_autotune/runs/a6000_dlo_capacity_5step_rl16_20260810T113157Z/capacity_gate_verdict.json`
- `dlo_autotune/runs/a6000_dlo_capacity_5step_rl18_20260810T134526Z`
- `dlo_autotune/runs/a6000_dlo_capacity_5step_rl18_20260810T134526Z/capacity_gate_verdict.json`
- `dmd_primary_source_note.md`
- `sol_engine_port/LATEST_SOL_ATTN_GPU_DIR`
- `sol_engine_port/gpu_exact_20260809T155451Z/correctness.json`
- `sol_engine_port/gpu_exact_20260809T155451Z/microbenchmark.json`
- `sol_engine_port/r5_ablation_20260809T181515Z/adaln/av_validation.json`
- `sol_engine_port/r5_ablation_20260809T181515Z/adaln/exact_telemetry.json`
- `sol_engine_port/r5_ablation_20260809T181515Z/adaln/http_metrics.txt`
- `sol_engine_port/r5_ablation_20260809T181515Z/adaln/quality_vs_dense.json`
- `sol_engine_port/r5_ablation_20260809T181515Z/all_exact/av_validation.json`
- `sol_engine_port/r5_ablation_20260809T181515Z/all_exact/exact_telemetry.json`
- `sol_engine_port/r5_ablation_20260809T181515Z/all_exact/http_metrics.txt`
- `sol_engine_port/r5_ablation_20260809T181515Z/all_exact/quality_vs_dense.json`
- `sol_engine_port/r5_ablation_20260809T181515Z/rope/av_validation.json`
- `sol_engine_port/r5_ablation_20260809T181515Z/rope/exact_telemetry.json`
- `sol_engine_port/r5_ablation_20260809T181515Z/rope/http_metrics.txt`
- `sol_engine_port/r5_ablation_20260809T181515Z/rope/quality_vs_dense.json`
- `sol_engine_port/r5_ablation_20260809T181515Z/swiglu/av_validation.json`
- `sol_engine_port/r5_ablation_20260809T181515Z/swiglu/exact_telemetry.json`
- `sol_engine_port/r5_ablation_20260809T181515Z/swiglu/http_metrics.txt`
- `sol_engine_port/r5_ablation_20260809T181515Z/swiglu/quality_vs_dense.json`
- `sol_engine_port/sol_attn_gpu2_supervisor/exit_code`
- `sol_engine_port/sol_attn_gpu2_supervisor/latest_run_id`
- `sol_engine_port/sol_attn_gpu2_supervisor/status.txt`
- `sol_engine_port/sol_attn_gpu_20260809T173323Z/result.json`
- `sol_engine_port/sol_attn_stride_aware_v_harness_20260813T082456Z/harness.json`
- `sol_engine_port/sol_attn_stride_aware_v_harness_20260813T082456Z/summary.json`
- `sol_engine_port/sol_attn_h3_formal_n10_r8_n10_20260812T031757Z/FORMAL_N10_RUN_REPORT.md`
- `sol_engine_port/sol_attn_h3_formal_n10_r8_n10_20260812T031757Z/RUN_REPORT.md`
- `sol_engine_port/sol_attn_h3_formal_n10_r8_n10_20260812T031757Z/formal_n10_decision.json`
- `sol_engine_port/sol_attn_h3_formal_n10_r8_n10_20260812T031757Z/formal_n10_summary.json`
- `sol_engine_port/sol_attn_h3_formal_n10_r8_n10_20260812T031757Z/formal_n10_supervisor_status.json`
- `sol_engine_port/sol_attn_h3_formal_n10_r8_n10_20260812T031757Z/formal_n10_supervisor_stdout.log`
- `sol_engine_port/sol_attn_h3_formal_n10_r8_n10_20260812T031757Z/quality_proxy_comparison.json`
- `sol_engine_port/sol_attn_h3_formal_n10_r8_n10_20260812T031757Z/resource_summary.json`
- `sol_engine_port/sol_attn_h3_formal_n10_r8_n10_20260812T031757Z/timing_summary.json`
- `sol_engine_port/sol_attn_h3_gpu2_5step_r8_prompt0644_20260812T005600Z/dense_h3_backend_reference/av_validation.json`
- `sol_engine_port/sol_attn_h3_gpu2_5step_r8_prompt0644_20260812T005600Z/dense_h3_backend_reference/http_metrics.txt`
- `sol_engine_port/sol_attn_h3_gpu2_5step_r8_prompt0644_20260812T005600Z/r8_image_identity.env`
- `sol_engine_port/sol_attn_h3_gpu2_5step_r8_prompt0644_20260812T005600Z/resource_monitor.csv`
- `sol_engine_port/sol_attn_h3_gpu2_5step_r8_prompt0644_20260812T005600Z/sol_attn/av_validation.json`
- `sol_engine_port/sol_attn_h3_gpu2_5step_r8_prompt0644_20260812T005600Z/sol_attn/http_metrics.txt`
- `sol_engine_port/sol_attn_h3_gpu2_5step_r8_prompt0644_20260812T005600Z/sol_attn/sol_attn_telemetry.sol_attn.json`
- `sol_engine_port/sol_attn_h3_gpu2_5step_r8_prompt0644_20260812T005600Z/sol_attn_diagnostic_status.json`
- `sol_engine_port/sol_attn_h3_gpu2_5step_r8_prompt0644_20260812T005600Z/workload.env`
- `sol_engine_port/sol_attn_h3_matched_retest_r8_n3_20260812T013544Z/RUN_REPORT.md`
- `sol_engine_port/sol_attn_h3_matched_retest_r8_n3_20260812T013544Z/decision.json`
- `sol_engine_port/sol_attn_h3_matched_retest_r8_n3_20260812T013544Z/posthoc_finalization_note.json`
- `sol_engine_port/sol_attn_h3_matched_retest_r8_n3_20260812T013544Z/quality_proxy_comparison.json`
- `sol_engine_port/sol_attn_h3_matched_retest_r8_n3_20260812T013544Z/resource_summary.json`
- `sol_engine_port/sol_attn_h3_matched_retest_r8_n3_20260812T013544Z/timing_summary.json`
- `turbo_merged/LATEST_QUALITY_SUITE_RUN_ID`
- `turbo_merged/quality_suite_runs/gpu3_turbo_quality_20260810T005611Z/baseline_seed0_quality_comparison.json`
- `turbo_merged/quality_suite_runs/gpu3_turbo_quality_20260810T005611Z/human_review.md`
- `turbo_merged/quality_suite_runs/gpu3_turbo_quality_20260810T005611Z/operator_acceptance.json`
- `turbo_merged/quality_suite_runs/gpu3_turbo_quality_20260810T005611Z/quality_suite_analysis.json`
- `turbo_merged/timing_repeats/LATEST_RUN_ID`
- `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/merge_manifest.json`
- `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/outputs/pair01_4step_av_validation.json`
- `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/outputs/pair01_8step_av_validation.json`
- `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/outputs/pair02_4step_av_validation.json`
- `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/outputs/pair02_8step_av_validation.json`
- `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/outputs/pair03_4step_av_validation.json`
- `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/outputs/pair03_8step_av_validation.json`
- `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/outputs/pair04_4step_av_validation.json`
- `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/outputs/pair04_8step_av_validation.json`
- `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/outputs/pair05_4step_av_validation.json`
- `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/outputs/pair05_8step_av_validation.json`
- `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/outputs/pair06_4step_av_validation.json`
- `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/outputs/pair06_8step_av_validation.json`
- `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/outputs/pair07_4step_av_validation.json`
- `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/outputs/pair07_8step_av_validation.json`
- `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/outputs/pair08_4step_av_validation.json`
- `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/outputs/pair08_8step_av_validation.json`
- `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/outputs/pair09_4step_av_validation.json`
- `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/outputs/pair09_8step_av_validation.json`
- `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/outputs/pair10_4step_av_validation.json`
- `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/outputs/pair10_8step_av_validation.json`
- `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/timing_summary.json`

## Reproduction command

```bash
python3 tools/minimax_h3_a6000_performance_report.py --evidence-root technical_report/evidence/minimax_h3_desktop --out technical_report/minimax_h3_a6000_performance.md
```
