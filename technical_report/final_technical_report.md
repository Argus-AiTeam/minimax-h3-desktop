# ARGUS-IR-04 Final Technical Report

Status: **final evidence integration, not quality-complete**. This report is evidence-grounded and CPU/static-generated; it does not add measurements, run inference, or publish results.

## Scope and lane separation

- Fidelity lane: `fidelity_bf16_exact`. Only the certified same-physical-GPU3 baseline is accepted here.
- Practical lane: `practical_disclosed_approx`. GPU3 paired Turbo timing, DMD feasibility notes, and exact-kernel diagnostics live here unless future evidence changes.
- Turbo practical results must not be relabeled as fidelity/BF16-exact results.

## Baseline certification

- Evidence: `baseline_a6000/baseline_certification.json`.
- Status: `certified_internal_same_physical_device_baseline` on `single_a6000_48gb_workstation`; schema `argus-h3-a6000-fidelity-baseline-certification-v2`.
- Physical device: host GPU3, SM8.6, UUID `GPU-ff0c8d25-2652-ce58-3ef7-e3a9aeeb3334`.
- All requests: N=13, mean=1788.1300204615384s, median=1780.975666s.
- Warm-primary denominator: N=10, mean=1790.7376617s, median=1792.2021025s, CV=0.8371622556580874%.
- Session-first requests: N=3 across 3 service sessions.
- Supersedes the prior v1 warm-count interpretation; speedups below use only this warm N=10 GPU3 denominator.

## Turbo merged practical timing and quality evidence

- Timing evidence: `turbo_merged/timing_repeats/LATEST_RUN_ID` -> `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/timing_summary.json`; merge manifest `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/merge_manifest.json`.
- Same physical device: host GPU3, SM8.6, UUID `GPU-ff0c8d25-2652-ce58-3ef7-e3a9aeeb3334`.
- Merge: status `completed`, strength 1.0, completed shards 13.
- Paired timing design: two excluded warmups, then N=10 formal paired samples per schedule; strict AV pass count 20.
- 4-step paired median: 149.6191865s; speedup vs same-GPU3 BF16 warm N=10 median: 11.978424321268447x; CV=0.033194157523981374%.
- 8-step paired median: 290.9976015s; speedup vs same-GPU3 BF16 warm N=10 median: 6.158820874336313x; CV=0.04878312051625522%.
- The 8-step schedule is the practical default candidate. The 4-step schedule is ultra-fast/quality-cost experimental because the visual suite exposed a teapot-geometry failure and lower audio fidelity.
- Quality-suite evidence: `turbo_merged/LATEST_QUALITY_SUITE_RUN_ID` -> `turbo_merged/quality_suite_runs/gpu3_turbo_quality_20260810T005611Z/quality_suite_analysis.json`; baseline seed0 comparison `turbo_merged/quality_suite_runs/gpu3_turbo_quality_20260810T005611Z/baseline_seed0_quality_comparison.json`; audio envelopes `turbo_merged/quality_suite_runs/gpu3_turbo_quality_20260810T005611Z/audio_energy_envelopes.json`; human review `turbo_merged/quality_suite_runs/gpu3_turbo_quality_20260810T005611Z/human_review.md`; review contact sheets=6.
- Quality-suite coverage: 24 outputs (3 prompts x 4 seeds x 4/8 steps), N=12 per schedule; structural AV pass=True.
- Human auditory listening remains pending; semantic AV quality is not certified.
- GPU2 smoke scope: earlier GPU2 smoke is bring-up only and is not used as a speedup denominator/result.

## Clean-room one-command local lifecycle

- Evidence: `delivery/local_lifecycle_clean_room_20260812T014824Z`.
- Status: `pass_packaging_lifecycle_only`; publication audit `pass`; deploy `pass`.
- Existing local FL2VA resource inspection: 81 files, 144051182625 bytes.
- Boundary: local clean-room lifecycle verifier only; no new speedup, quality, BF16 fidelity, Sol-Attn, Turbo, DLO, DMD, GPU, Docker-run, model-load, or publication claim.

## Final CPU/static/export/audit gates

- Overall gate status: `pass`.
- CPU/static gate: `pass`; evidence `delivery/final_cpu_static_gate_20260812T020013Z/summary.txt`; summary tail=['verify_run=PASS mock-av-metadata-fixture-0001 track=fidelity_bf16_exact platform=current_a6000_reference mock=True', 'turbo_dry_run=To execute later: ARGUS_ALLOW_TURBO_QUALITY_SUITE=1 python3 tools/turbo_quality_suite_runner.py --execute --config technical_report/evidence/minimax_h3_desktop/turbo_merged/quality_suite_config.json --out technical_report/evidence/minimax_h3_desktop/turbo_merged/quality_suite_dry_run --authorization-id <FRESH_AUTHZ> --acknowledge-gpu-docker-model-inference', 'aggregate_regression_after_gate=3 collected, 3 passed, 0 skipped, 0 failed', 'release_regression_after_readme_gate=13 collected, 13 passed, 0 skipped, 0 failed'].
- Strict aggregation/export/publication audit gate: `pass`; evidence `delivery/final_decisive_export_audit_20260812T025605Z/summary.json`; export_file_count=82; publication_audit=pass; issues=0.
- Boundary: CPU/static/export/audit gates only; no GPU, Docker-run, model-load, speed, fidelity, or quality claim is created by these gates..

## Reproducible Turbo quality-suite runner

- Config: `turbo_merged/quality_suite_config.json`.
- Dry-run plan directory: `turbo_merged/quality_suite_dry_run`.
- Default behavior is fail-closed dry-run: it writes the prompt/seed/step matrix and operator commands without launching Docker/GPU/model inference.
- Non-dry execution requires a fresh operator authorization, the `ARGUS_ALLOW_TURBO_QUALITY_SUITE=1` environment gate, and the explicit acknowledgement flag printed by the runner.

## Sol / exact-kernel diagnostics

- AdaLN: `exact_output_candidate_n1_rejected_for_speedup_acceptance_not_deployed`; video MSE 0.0; audio cosine 0.9999999999999999; N=1; single-run gain 0.47683877255666474% below baseline warm CV 0.8371622556580874%.
- AdaLN disclosure preserved: `post-generation validation here-doc quoting SyntaxError`; original harness exit code 1; not an accepted N=10 speedup.
- RoPE: `rejected_for_output_drift_not_accepted_exact_kernel`; video MSE 224.11484810613817; audio cosine 0.9775986604966458.
- All-exact: `rejected_for_output_drift_not_accepted_exact_kernel`; video MSE 224.11484810613817; audio cosine 0.9775986604966458.
- SwiGLU: `rejected_no_retained_speedup_gain_not_deployed`; exact diagnostic output retained no accepted speedup gain.
- Toy Sol-Attn: `rejected_slower_not_deployed_kernel_candidate_only`; dense median 0.12390399724245071 ms, sparse median 0.40243199467658997 ms, dense/sparse median speedup 0.30788803793302966.
- H3 Sol-Attn r8: `sparse_runtime_valid_5step_diagnostic`; evidence `delivery/r8_sol_attn_cpu_ingest_20260812T005600Z/r8_terminal_classification.json`; sparse candidates=192, sparse calls=192, dense calls=16, fallback calls=0, density samples=192, materialized copies=192 / 105344139264 bytes.
- H3 Sol-Attn r8 HTTP/resource boundary: dense=186.498762s, opt-in=158.923988s, dense/opt-in ratio=1.173509199882399 (diagnostic only, not a speedup); peak GPU memory=27354.0 MiB, peak temperature=84.0 C, peak power=299.88 W.
- H3 Sol-Attn r8 claim boundary: accepted 5-step sparse-execution metadata-plumbing diagnostic only; not a speedup, N10, BF16 fidelity, release, or quality-equivalence claim.
- R8 matched-workload retest route decision: `proceed_to_formal_n10_candidate`; evidence `sol_engine_port/sol_attn_h3_matched_retest_r8_n3_20260812T013544Z/decision.json`; terminal_recheck=`delivery/r8_matched_retest_terminal_recheck_20260812T024043Z/summary.json`; completed_pairs=3/3; median_http_time_improvement=14.782455716069165%; threshold=3.0%; supervisor_status=failed; supervisor_return_code=2; posthoc_finalization_note=`sol_engine_port/sol_attn_h3_matched_retest_r8_n3_20260812T013544Z/posthoc_finalization_note.json`; n10_recommendation=proceed_to_formal_n10_candidate. Reason: bounded matched retest passed correctness, sparse-runtime, resource, quality-proxy, and timing gates. This is still only a bounded route gate, not formal N10 or a speedup/quality/fidelity claim.

## DMD / DMD2

- Evidence: `dmd_primary_source_note.md`.
- Status: **blocked_research_only_no_go_after_turbo_unless_feasibility_changes**.
- DMD remains a no-go after Turbo unless a legal H3 DMD/DMD2 recipe/checkpoint, resource profile, and AV quality bar appear in future evidence.

## Accepted / rejected / blocked matrix

### Accepted
- A single-A6000 internal BF16-exact baseline v2 is certified on physical GPU3 with N=13 total requests, N=10 true warm-primary requests, and three session-first requests. Evidence: `baseline_a6000/baseline_certification.json`. Limit: Internal same-device denominator only; not an optimized or external reproduction.
- The latest GPU3 paired Turbo timing run is accepted as the only practical speed result: two excluded warmups, paired N=10 per schedule, strict structural AV pass, and baseline-v2 warm N=10 denominator. Evidence: `turbo_merged/timing_repeats/LATEST_RUN_ID`, `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/timing_summary.json`, `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/merge_manifest.json`. Limit: Practical approximation only; semantic AV quality and human auditory listening remain pending; 8-step is the default candidate and 4-step is ultra-fast quality-cost experimental.
- AdaLN is an N=1 exact-output candidate only; the original harness-tail failure is preserved and the single-run benefit is below baseline warm-run noise. Evidence: `sol_engine_port/r5_ablation_20260809T181515Z/adaln/quality_vs_dense.json`, `sol_engine_port/a6000_adaln_candidate_50step_20260809T190402Z/av_validation_posthoc.json`, `sol_engine_port/a6000_adaln_candidate_50step_20260809T190402Z/candidate_verdict.json`. Limit: Not an accepted N=10 speedup and not deployed as a certified fidelity path.
- The gated one-command local lifecycle passed in a clean-room export/work directory using existing local locked resources only. Evidence: `delivery/local_lifecycle_clean_room_20260812T014824Z/lifecycle/stages/05_lifecycle_summary.json`, `delivery/local_lifecycle_clean_room_20260812T014824Z/lifecycle/stages/02_model_prepare.json`, `delivery/local_lifecycle_clean_room_20260812T014824Z/lifecycle/stages/03_deploy.json`. Limit: Packaging/deployment evidence only: no container start, model load, GPU inference, media generation, speed, fidelity, or quality claim.
- H3 Sol-Attn r8 metadata plumbing reached the real 5-step H3 attention boundary and executed the sparse path with valid structural AV/resource telemetry. Evidence: `delivery/r8_sol_attn_cpu_ingest_20260812T005600Z/r8_terminal_classification.json`. Limit: accepted 5-step sparse-execution metadata-plumbing diagnostic only; not a speedup, N10, BF16 fidelity, release, or quality-equivalence claim
- The r8 N=3 matched-workload route gate is terminal and recommends formal N>=10 Sol-Attn testing. Evidence: `sol_engine_port/sol_attn_h3_matched_retest_r8_n3_20260812T013544Z/decision.json`. Limit: Bounded N=3 5-step route decision only; not formal N10, not a speedup, not BF16 fidelity, and not quality-equivalence certification.

### Rejected
- Turbo merged LoRA is a BF16-exact/fidelity result. Reason: Turbo is practical_disclosed_approx and uses a statically merged LoRA approximation. Evidence: `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/timing_summary.json`.
- Treating the earlier GPU2 Turbo smoke run as an accepted speedup denominator/result. Reason: GPU2 smoke is retained only as bring-up evidence; the only speedup result uses same physical GPU3 paired timing against baseline v2 warm N=10. Evidence: `turbo_merged/runs/gpu2_turbo_20260809T195558Z/turbo_summary.json`, `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/timing_summary.json`.
- RoPE/all-exact kernels are accepted exact replacements. Reason: Same-prompt diagnostic evidence shows non-zero video/audio drift. Evidence: `sol_engine_port/r5_ablation_20260809T181515Z/rope/quality_vs_dense.json`, `sol_engine_port/r5_ablation_20260809T181515Z/all_exact/quality_vs_dense.json`.
- SwiGLU is a retained practical speed gain. Reason: Current evidence is exact diagnostic output only; no retained speedup gain is accepted or deployed. Evidence: `sol_engine_port/r5_ablation_20260809T181515Z/swiglu/quality_vs_dense.json`, `sol_engine_port/r5_ablation_20260809T181515Z/swiglu/http_metrics.txt`.
- Toy Sol-Attn is deployed or faster. Reason: Current toy harness is kernel-candidate-only, model_load=false, and sparse median is slower than dense. Evidence: `sol_engine_port/sol_attn_gpu_20260809T173323Z/result.json`.

### Blocked
- MiniMax-H3 DMD/DMD2 is a no-go after Turbo unless feasibility evidence changes. Status: blocked_research_only_no_go_after_turbo_unless_feasibility_changes Evidence: `dmd_primary_source_note.md`.
- Turbo semantic AV quality and human-auditory quality are fully certified. Status: blocked_human_auditory_listening_pending_semantic_quality_not_certified Evidence: `turbo_merged/LATEST_QUALITY_SUITE_RUN_ID`, `turbo_merged/quality_suite_runs/gpu3_turbo_quality_20260810T005611Z/quality_suite_analysis.json`, `turbo_merged/quality_suite_runs/gpu3_turbo_quality_20260810T005611Z/human_review.md`.

### Pending
- Formal DLO N10 performance promotion is complete. Status: pending_no_formal_n10_because_current_candidate_is_below_baseline_noise Evidence: `dlo_autotune/detached_continuation/status.txt`, `dlo_autotune/runs/a6000_dlo_candidate_50_rl16_20260810T141257Z/candidate50_summary.json`.
- Sol-Attn matched-workload correctness/quality and performance promotion is complete. Status: pending_formal_n10_required_after_r8_n3_candidate_before_speedup_or_quality_claim Evidence: `sol_engine_port/sol_attn_h3_matched_retest_r8_n3_20260812T013544Z/decision.json`.

## Reproduction / final gate commands

```bash
PYTHONPATH=code:.:ports/minimax_h3_a6000/src python3 -m pytest -q tests ports/minimax_h3_a6000/tests
```
```bash
python3 tools/verify_run.py tests/fixtures/minimal_av_case/run_record.json
```
```bash
python3 tools/turbo_quality_suite_runner.py --dry-run --config technical_report/evidence/minimax_h3_desktop/turbo_merged/quality_suite_config.json --out technical_report/evidence/minimax_h3_desktop/turbo_merged/quality_suite_dry_run
```
```bash
python3 tools/argus_ir04_aggregate.py --strict --input technical_report/evidence/minimax_h3_desktop --out technical_report/evidence/minimax_h3_desktop/delivery/argus_ir04_delivery_summary.json --report-out technical_report/final_technical_report.md --manifest-out technical_report/evidence/minimax_h3_desktop/delivery/package_manifest.json
```

## Test accounting

Skipped GPU placeholders are not counted as passed coverage. This delivery counts only CPU/static tests and the metadata verifier command reported by the engineer.
