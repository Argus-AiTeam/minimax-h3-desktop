# ARGUS-IR-04 Final Technical Report

Status: **final evidence integration; operator overall practical-quality gate accepted**. This report is evidence-grounded and CPU/static-generated; it does not add measurements, run inference, or publish results.

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
- Operator overall playback/listening acceptance is recorded at `turbo_merged/quality_suite_runs/gpu3_turbo_quality_20260810T005611Z/operator_acceptance.json`; 8-step remains the practical default and the known 4-step visual failure remains preserved.
- GPU2 smoke scope: earlier GPU2 smoke is bring-up only and is not used as a speedup denominator/result.
- The delivery hold/Reviewer packets below are preserved as historical pre-acceptance evidence. Their operator-action-required fields were superseded by the later operator acceptance record; they are not the current release gate.

## Turbo operator-only listening gate packet (historical pre-acceptance packet)

- Latest packet: `technical_report/evidence/minimax_h3_desktop/delivery/turbo_operator_gate_stage_closeout_packet_20260812T075720Z`; packet_status=`accepted_current_stage_delivery_boundary`; reviewer_status=`accepted_current_stage_delivery_reviewer_passed`; accepted_for_current_stage_closing=True
- Automatable delivery gates complete=True; operator listening manifest cases=24; manifest `technical_report/evidence/minimax_h3_desktop/delivery/turbo_operator_gate_stage_closeout_packet_20260812T075720Z/media_listening_manifest.json`.
- Operator-only residual: human auditory listening / semantic AV-sync gate=`operator_action_required`; agent_subjective_listening_performed=False; semantic_av_quality_certified=False; av_sync_certified=False.
- Reviewer handoff source: `<private-path>`; source_valid=True; manager_recognition_repair=`technical_report/evidence/minimax_h3_desktop/delivery/turbo_operator_gate_stage_closeout_packet_20260812T075720Z/manager_reviewer_handoff_crosswalk.json`; manager_stage_closeout_crosswalk=`technical_report/evidence/minimax_h3_desktop/delivery/turbo_operator_gate_stage_closeout_packet_20260812T075720Z/manager_stage_closeout_crosswalk.json`; manager_visibility_resolution=`technical_report/evidence/minimax_h3_desktop/delivery/turbo_operator_gate_stage_closeout_packet_20260812T075720Z/manager_visibility_resolution.json`.
- Manager recognition check: `technical_report/evidence/minimax_h3_desktop/delivery/turbo_operator_gate_stage_closeout_packet_20260812T075720Z/manager_recognition_check_20260812T082741Z.json`; status=`pass`; ready=True; mismatch_count=0.
- Boundary: current delivery-stage automatable evidence crosswalk: strict delivery summary, final report boundary, CPU/static/export/publication/private-sync gates, Turbo 24-case operator-listening manifest, canonical closeout reviewer_verdict.json, sealed Reviewer handoff crosswalk, and package manifest inclusion after regeneration; no GPU, Docker, model load, subjective listening, public release, tag, or force push is introduced.

## Current-stage delivery Reviewer evidence recognition repair

- Repair packet: `technical_report/evidence/minimax_h3_desktop/delivery/delivery_reviewer_evidence_recognition_repair_20260812T085911Z`; status=`accepted_current_stage_delivery_reviewer_evidence_repair`; ready_for_manager_recognition=True; current_stage=`delivery`; current_mission_id=`f497130bb319`.
- Fresh Reviewer verdict: `technical_report/evidence/minimax_h3_desktop/delivery/delivery_reviewer_evidence_recognition_repair_20260812T085911Z/reviewer_verdict.json`; reviewer_status=`accepted_current_stage_delivery_reviewer_passed`; decision=`done`; independent=True; sealed_handoff=`<private-path>`; sealed_handoff_valid=True.
- Manager recognition check: `technical_report/evidence/minimax_h3_desktop/delivery/delivery_reviewer_evidence_recognition_repair_20260812T085911Z/manager_recognition_check_20260812T085911Z.json`; status=`pass`; ready=True; mismatch_count=0.
- Manager-stage authority probe: `technical_report/evidence/minimax_h3_desktop/delivery/delivery_reviewer_evidence_recognition_repair_20260812T085911Z/manager_stage_authority_probe_20260812T092125Z.json`; status=`pass_authority_stage_observation`; reviewer_evidence_complete=True; transition_status=`blocked_final_delivery_stage_no_advance_target_not_reviewer_evidence_incomplete`.
- Schema/crosswalk evidence: schema_gap=`technical_report/evidence/minimax_h3_desktop/delivery/delivery_reviewer_evidence_recognition_repair_20260812T085911Z/schema_gap_analysis.json` status=`gap_identified_and_reviewer_evidence_repaired`; handoff_crosswalk=`technical_report/evidence/minimax_h3_desktop/delivery/delivery_reviewer_evidence_recognition_repair_20260812T085911Z/manager_reviewer_handoff_crosswalk.json` status=`pass`; legacy_chain_ready=True.
- Operator-only residual preserved: gate=`operator_human_auditory_listening_and_semantic_av_sync_review_for_Turbo_quality_certification`; agent_subjective_listening_performed=False; semantic_av_quality_certified=False; av_sync_certified=False.
- Boundary: Current-stage delivery Reviewer-evidence recognition repair only; it preserves prior automatable delivery evidence and leaves Turbo human listening / semantic AV-sync operator-only.

## Delivery Reviewer active-hold reconciliation

- Active packet: `technical_report/evidence/minimax_h3_desktop/delivery/delivery_reviewer_evidence_active_hold_reconciliation_20260812T094326Z`; status=`accepted_active_hold_reconciliation_stale_hold_diagnosis`; accepted_for_manager_visible_delivery_sync=True; current_stage=`delivery`; current_mission_id=`dc32594cd79e`.
- Active Reviewer verdict: `technical_report/evidence/minimax_h3_desktop/delivery/delivery_reviewer_evidence_active_hold_reconciliation_20260812T094326Z/reviewer_verdict.json`; decision=`done`; independent=True; active_hold_probe=`technical_report/evidence/minimax_h3_desktop/delivery/delivery_reviewer_evidence_active_hold_reconciliation_20260812T094326Z/active_hold_reconciliation_probe.json`; INDEX=`technical_report/evidence/minimax_h3_desktop/delivery/delivery_reviewer_evidence_active_hold_reconciliation_20260812T094326Z/INDEX.json`.
- Active hold surfaces captured: ["<private-path> life.manager.feedback.persisted fields=['diagnostic'] token=manager_hold_requires_stage_repair", "<private-path> life.planner.verdict fields=['reason', 'summary'] token=manager_hold_requires_stage_repair"].
- Manager-stage complete counter-evidence: `<private-path>`:12011 `life.manager.stage_decision` diagnostic=`intentional_hold`; classification=`stale_or_misclassified_host_planner_hold; no additional project-local Reviewer-evidence field/path/schema gap was identified in the current latest repair packet.`; raw_event_sha256_count=0.
- Operator-only residual preserved: gate=`operator_human_auditory_listening_and_semantic_av_sync_review_for_Turbo_quality_certification`; agent_subjective_listening_performed=False; semantic_av_quality_certified=False; av_sync_certified=False.
- Boundary: Accepted active-hold reconciliation/stale-hold diagnosis only; no GPU, Docker, model, benchmark, listening, semantic AV-sync certification, or Manager-owned state edit is claimed.

## Current Manager-hold no-gap probe

- Packet: `technical_report/evidence/minimax_h3_desktop/delivery/current_manager_hold_no_gap_probe_20260812T110024Z`; status=`pending_fresh_independent_reviewer`; decisive_check_passed=True; project_local_reviewer_evidence_locator_schema_mismatch=False; current_stage=`delivery`; current_mission_id=`e6de6b0b1563`.
- Reviewer boundary: reviewer_status=`pending_fresh_independent_reviewer`; decision=`None`; independent=False; sealed_handoff_valid=False; manager_visible_delivery_sync_ready=False; verdict=`technical_report/evidence/minimax_h3_desktop/delivery/current_manager_hold_no_gap_probe_20260812T110024Z/reviewer_verdict.json`; request=`technical_report/evidence/minimax_h3_desktop/delivery/current_manager_hold_no_gap_probe_20260812T110024Z/reviewer_verdict_request.json`; crosswalk=`technical_report/evidence/minimax_h3_desktop/delivery/current_manager_hold_no_gap_probe_20260812T110024Z/manager_reviewer_handoff_crosswalk.json`.
- Exact current-stage Reviewer-evidence gap: status=`open_pending_valid_independent_reviewer_source`; missing_paths=None; missing_or_unaccepted_fields=['current_manager_hold_no_gap_probe.reviewer_status must equal accepted_current_stage_delivery_reviewer_passed', 'current_manager_hold_no_gap_probe.reviewer_decision must equal done', 'current_manager_hold_no_gap_probe.reviewer_verdict_independent must equal true', 'current_manager_hold_no_gap_probe.sealed_reviewer_handoff_source_valid must equal true', 'current_manager_hold_no_gap_probe.manager_visible_delivery_sync_ready must equal true', 'current_manager_hold_no_gap_probe.accepted_for_manager_visible_delivery_sync must equal true'].
- Live Manager stage: `<private-path>`:12911 diagnostic=`intentional_hold`; reason=`Automatable delivery is complete/reviewed/pushed, but the remaining Turbo subjective listening / semantic AV-sync certification is operator-only and cannot be performed or certified by the agent. No legal advance or rollback target exists from final-stage delivery, and no operator listening result or acceptance change is present.`.
- Persisted hold token: `<private-path>`:12912 diagnostic=`manager_hold_requires_stage_repair`; reason=`Automatable delivery is complete/reviewed/pushed, but the remaining Turbo subjective listening / semantic AV-sync certification is operator-only and cannot be performed or certified by the agent. No legal advance or rollback target exists from final-stage delivery, and no operator listening result or acceptance change is present.`.
- Lower-authority planner surface: `<private-path>`:12930 status=`planned`; reason=`Manager current reality reports `manager_hold_requires_stage_repair` / Reviewer-evidence-incomplete at delivery, while reports show only operator-only Turbo human listening/semantic AV-sync remains; final report also names the latest no-gap probe as pending fresh Reviewer, so Engineer must repair or replace that current-stage Reviewer evidence packet and obtain a complete independent verdict.`.
- Operator-only residual preserved: gate=`operator_human_auditory_listening_and_semantic_av_sync_review_for_Turbo_quality_certification`; agent_subjective_listening_performed=False; semantic_av_quality_certified=False; av_sync_certified=False. Manifests still missing 0 required current-hold paths.
- Boundary: Current Manager-hold no-gap/visibility packet only; no GPU, Docker, model, benchmark, subjective listening, semantic AV-sync certification, or Manager-owned state edit is claimed.

## Clean-room one-command local lifecycle

- Evidence: `delivery/local_lifecycle_clean_room_20260812T014824Z`.
- Status: `pass_packaging_lifecycle_only`; publication audit `pass`; deploy `pass`.
- Existing local FL2VA resource inspection: 81 files, 144051182625 bytes.
- Boundary: local clean-room lifecycle verifier only; no new speedup, quality, BF16 fidelity, Sol-Attn, Turbo, DLO, DMD, GPU, Docker-run, model-load, or publication claim.

## Final CPU/static/export/audit gates

- Overall gate status: `pass`.
- CPU/static gate: `pass`; evidence `delivery/final_cpu_static_gate_20260812T033107Z/summary.txt`; summary tail=['py_compile=PASS tools/minimax_h3_a6000_performance_report.py tools/argus_ir04_aggregate.py tools/update_dual_workflow_progress.py', 'full_pytest=PASS 118 collected, 118 passed', 'verify_run=PASS mock-av-metadata-fixture-0001 track=fidelity_bf16_exact platform=current_a6000_reference mock=True', 'To execute later: ARGUS_ALLOW_TURBO_QUALITY_SUITE=1 python3 tools/turbo_quality_suite_runner.py --execute --config technical_report/evidence/minimax_h3_desktop/turbo_merged/quality_suite_config.json --out technical_report/evidence/minimax_h3_desktop/turbo_merged/quality_suite_dry_run --authorization-id <FRESH_AUTHZ> --acknowledge-gpu-docker-model-inference'].
- Strict aggregation/export/publication audit gate: `pass`; evidence `delivery/final_decisive_export_audit_20260812T025605Z/summary.json`; export_file_count=82; publication_audit=pass; issues=0.
- Formal-N10 report-sync export/publication audit gate: `pass`; evidence `delivery/formal_n10_cpu_sync_export_audit_20260812T065502Z/summary.json`; export_file_count=89; publication_audit=pass; issues=0; reviewer_status=accepted_independent_reviewer_passed; push_performed=True.
- Active-hold report-sync export/publication audit gate: `pass`; evidence `delivery/active_hold_sync_export_audit_20260812T095904Z/summary.json`; strict_aggregation=pass; export_status=built; export_file_count=89; publication_audit=pass; issues=0; active_hold_reviewer=`technical_report/evidence/minimax_h3_desktop/delivery/delivery_reviewer_evidence_active_hold_reconciliation_20260812T094326Z/reviewer_verdict.json`.
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
- Stride-aware-V SM86 harness: `accepted_model_free_zero_materialization_correctness_only`; evidence `sol_engine_port/sol_attn_stride_aware_v_harness_20260813T082456Z/summary.json` and `harness.json`; model_load=False; GPU UUID `GPU-b3425477-0877-24de-c5b8-1549ab47cd4b`; exactly one visible GPU under Argus lease supervision. Correctness compiled/launched, exercised `stride_aware_value_calls=1`, `sparse_calls=1`, `fallback_calls=0`, `materialize_copy_count=0`, `materialize_copy_bytes=0`, `prefix_rows_equal_dense=True`, and `padding_rows_zero=True`. Warmup/repeats were 20/100; sparse median=0.41996800899505615 ms, dense median=0.131071999669075 ms. Boundary: kernel/model-free evidence only; not H3 end-to-end, not long-video, not real-chain speedup, and not product-quality evidence.
- H3 Sol-Attn r8: `sparse_runtime_valid_5step_diagnostic`; evidence `delivery/r8_sol_attn_cpu_ingest_20260812T005600Z/r8_terminal_classification.json`; sparse candidates=192, sparse calls=192, dense calls=16, fallback calls=0, density samples=192, materialized copies=192 / 105344139264 bytes.
- H3 Sol-Attn r8 HTTP/resource boundary: dense=186.498762s, opt-in=158.923988s, dense/opt-in ratio=1.173509199882399 (diagnostic only, not a speedup); peak GPU memory=27354.0 MiB, peak temperature=84.0 C, peak power=299.88 W.
- H3 Sol-Attn r8 claim boundary: accepted 5-step sparse-execution metadata-plumbing diagnostic only; not a speedup, N10, BF16 fidelity, release, or quality-equivalence claim.
- R8 matched-workload retest route decision: `proceed_to_formal_n10_candidate`; evidence `sol_engine_port/sol_attn_h3_matched_retest_r8_n3_20260812T013544Z/decision.json`; terminal_recheck=`delivery/r8_matched_retest_terminal_recheck_20260812T024043Z/summary.json`; completed_pairs=3/3; median_http_time_improvement=14.782455716069165%; threshold=3.0%; supervisor_status=failed; supervisor_return_code=2; posthoc_finalization_note=`sol_engine_port/sol_attn_h3_matched_retest_r8_n3_20260812T013544Z/posthoc_finalization_note.json`; n10_recommendation=proceed_to_formal_n10_candidate. Reason: bounded matched retest passed correctness, sparse-runtime, resource, quality-proxy, and timing gates. This N=3 route gate led to the later accepted formal N>=10 gate; the N=3 gate itself remains bounded route evidence, not formal N10 or a speedup/quality/fidelity claim.
- R8 formal N>=10 matched-workload gate: `accepted_formal_n10_same_gpu_sol_attn_speed_candidate`; evidence `sol_engine_port/sol_attn_h3_formal_n10_r8_n10_20260812T031757Z/formal_n10_decision.json`; requested_pairs=10; started_pairs=10; completed_pairs=10; supervisor_status=complete. Reason: N>=10 same-GPU matched workload passed sparse-runtime, structural AV, resource, quality-proxy, and timing gates. Accepted only within the formal matched 5-step Sol-Attn opt-in lane; this is not BF16 fidelity, Turbo/DLO/DMD evidence, release approval, or human-auditory/semantic quality certification.

## DMD / DMD2

- Evidence: `dmd_primary_source_note.md`.
- Status: **blocked_research_only_no_go_after_turbo_unless_feasibility_changes**.
- DMD remains a no-go after Turbo unless a legal H3 DMD/DMD2 recipe/checkpoint, resource profile, and AV quality bar appear in future evidence.

## Accepted / rejected / blocked matrix

### Accepted
- A single-A6000 internal BF16-exact baseline v2 is certified on physical GPU3 with N=13 total requests, N=10 true warm-primary requests, and three session-first requests. Evidence: `baseline_a6000/baseline_certification.json`. Limit: Internal same-device denominator only; not an optimized or external reproduction.
- The latest GPU3 paired Turbo timing run is accepted as the only practical speed result: two excluded warmups, paired N=10 per schedule, strict structural AV pass, and baseline-v2 warm N=10 denominator. Evidence: `turbo_merged/timing_repeats/LATEST_RUN_ID`, `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/timing_summary.json`, `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/merge_manifest.json`. Limit: Practical approximation only; operator overall playback/listening acceptance is recorded; 8-step remains the default and the known 4-step visual failure remains disclosed.
- AdaLN is an N=1 exact-output candidate only; the original harness-tail failure is preserved and the single-run benefit is below baseline warm-run noise. Evidence: `sol_engine_port/r5_ablation_20260809T181515Z/adaln/quality_vs_dense.json`, `sol_engine_port/a6000_adaln_candidate_50step_20260809T190402Z/av_validation_posthoc.json`, `sol_engine_port/a6000_adaln_candidate_50step_20260809T190402Z/candidate_verdict.json`. Limit: Not an accepted N=10 speedup and not deployed as a certified fidelity path.
- The gated one-command local lifecycle passed in a clean-room export/work directory using existing local locked resources only. Evidence: `delivery/local_lifecycle_clean_room_20260812T014824Z/lifecycle/stages/05_lifecycle_summary.json`, `delivery/local_lifecycle_clean_room_20260812T014824Z/lifecycle/stages/02_model_prepare.json`, `delivery/local_lifecycle_clean_room_20260812T014824Z/lifecycle/stages/03_deploy.json`. Limit: Packaging/deployment evidence only: no container start, model load, GPU inference, media generation, speed, fidelity, or quality claim.
- The operator completed overall playback/listening review and accepted the practical Turbo release quality, with 8-step retained as default and the known 4-step visual failure preserved. Evidence: `turbo_merged/quality_suite_runs/gpu3_turbo_quality_20260810T005611Z/operator_acceptance.json`, `turbo_merged/quality_suite_runs/gpu3_turbo_quality_20260810T005611Z/human_review.md`. Limit: Overall operator acceptance without a fabricated per-case rubric transcript; Turbo remains approximate and the known 4-step failure remains rejected for default promotion.
- H3 Sol-Attn r8 metadata plumbing reached the real 5-step H3 attention boundary and executed the sparse path with valid structural AV/resource telemetry. Evidence: `delivery/r8_sol_attn_cpu_ingest_20260812T005600Z/r8_terminal_classification.json`. Limit: accepted 5-step sparse-execution metadata-plumbing diagnostic only; not a speedup, N10, BF16 fidelity, release, or quality-equivalence claim
- The r8 N=3 matched-workload route gate is terminal and led to the later accepted formal N>=10 Sol-Attn gate. Evidence: `sol_engine_port/sol_attn_h3_matched_retest_r8_n3_20260812T013544Z/decision.json`. Limit: Bounded N=3 5-step route decision only; not formal N10, not a speedup, not BF16 fidelity, and not quality-equivalence certification.
- Formal r8 Sol-Attn N>=10 matched-workload promotion is complete. Evidence: `sol_engine_port/sol_attn_h3_formal_n10_r8_n10_20260812T031757Z/formal_n10_decision.json`. Limit: Formal matched 5-step Sol-Attn lane only; not BF16 fidelity, Turbo, DLO, DMD, release, or human-auditory quality certification.
- The stride-aware-V SM86 harness is accepted as model-free correctness and zero-materialization validation for the current kernel path. Evidence: `sol_engine_port/sol_attn_stride_aware_v_harness_20260813T082456Z/summary.json`, `sol_engine_port/sol_attn_stride_aware_v_harness_20260813T082456Z/harness.json`. Limit: no MiniMax-H3 model load, no Docker real chain, no AV output, no 30/60-second result, and no H3 end-to-end speedup claim.

### Rejected
- Turbo merged LoRA is a BF16-exact/fidelity result. Reason: Turbo is practical_disclosed_approx and uses a statically merged LoRA approximation. Evidence: `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/timing_summary.json`.
- Treating the earlier GPU2 Turbo smoke run as an accepted speedup denominator/result. Reason: GPU2 smoke is retained only as bring-up evidence; the only speedup result uses same physical GPU3 paired timing against baseline v2 warm N=10. Evidence: `turbo_merged/runs/gpu2_turbo_20260809T195558Z/turbo_summary.json`, `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/timing_summary.json`.
- RoPE/all-exact kernels are accepted exact replacements. Reason: Same-prompt diagnostic evidence shows non-zero video/audio drift. Evidence: `sol_engine_port/r5_ablation_20260809T181515Z/rope/quality_vs_dense.json`, `sol_engine_port/r5_ablation_20260809T181515Z/all_exact/quality_vs_dense.json`.
- SwiGLU is a retained practical speed gain. Reason: Current evidence is exact diagnostic output only; no retained speedup gain is accepted or deployed. Evidence: `sol_engine_port/r5_ablation_20260809T181515Z/swiglu/quality_vs_dense.json`, `sol_engine_port/r5_ablation_20260809T181515Z/swiglu/http_metrics.txt`.
- Toy Sol-Attn is deployed or faster. Reason: Current toy harness is kernel-candidate-only, model_load=false, and sparse median is slower than dense. Evidence: `sol_engine_port/sol_attn_gpu_20260809T173323Z/result.json`.
- The 2026-08-13 stride-aware-V harness is an H3 product speedup. Reason: it is a model-free SM86 harness with no MiniMax-H3 model load, no AV output, and sparse median slower than dense for its synthetic benchmark shape. Evidence: `sol_engine_port/sol_attn_stride_aware_v_harness_20260813T082456Z/harness.json`.

### Blocked
- MiniMax-H3 DMD/DMD2 is a no-go after Turbo unless feasibility evidence changes. Status: blocked_research_only_no_go_after_turbo_unless_feasibility_changes Evidence: `dmd_primary_source_note.md`.

### Pending
- Formal DLO N10 performance promotion is complete. Status: pending_no_formal_n10_because_current_candidate_is_below_baseline_noise Evidence: `dlo_autotune/detached_continuation/status.txt`, `dlo_autotune/runs/a6000_dlo_candidate_50_rl16_20260810T141257Z/candidate50_summary.json`.

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
