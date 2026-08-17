# r9 Current vs r10 Guarded Adaptive Step-Min=3 Sol-Attn Final-AV 60s N=1 RUN_REPORT

- Status: `descriptive`
- Classification: `descriptive_no_promotion_r10_adaptive_tau1_5_step3_diag_60s_automatic_proxy_red_flags`
- Reference evidence: `${PWD}/technical_report/evidence/minimax_h3_desktop/long_video/final-av-60s-r9-current-vs-r10-step3-sol-attn-n1-20260816T164226Z/r9_current_sol_attn`
- Candidate evidence: `${PWD}/technical_report/evidence/minimax_h3_desktop/long_video/final-av-60s-r9-current-vs-r10-step3-sol-attn-n1-20260816T164226Z/r10_adaptive_tau1_5_step3_diag`
- Candidate profile: `r10_adaptive_tau1_5_step3_diag` (expected tau=1.5, diag, adaptive_step_min=3)
- Warm E2E reference/candidate: 2802.991s / 2682.008s
- N=1 warm delta (route-gate only, not speedup): 4.316%
- Reference sparse/fallback/materialization/input-copy: sparse=4368, fallback=0, materialize=0/0 bytes, input_copy=0/0 bytes
- Candidate sparse/fallback/materialization/input-copy: sparse=4368, fallback=0, materialize=0/0 bytes, input_copy=0/0 bytes
- Automatic proxy flags reference/candidate: ['near_frozen_transition_fraction'] / ['near_frozen_transition_fraction']
- Failed gates: ['reference_no_automatic_proxy_flags', 'candidate_no_automatic_proxy_flags']
- Same GPU/workload/timing-boundary proof: {'lane_id': 'final-av-60s-1344x768-24fps-v1', 'duration_label': '60s', 'final_frames': 1440, 'effective_audio_samples_per_channel': 1920000, 'chunk_count': 12, 'reference_physical_gpu_uuids': ['GPU-b3425477-0877-24de-c5b8-1549ab47cd4b'], 'candidate_physical_gpu_uuids': ['GPU-b3425477-0877-24de-c5b8-1549ab47cd4b'], 'reference_workload_fingerprint': '0a4ffd6b8d872ef7b9c93da6d9864e836719195f6530550196f683adcbe6b6b1', 'candidate_workload_fingerprint': '0a4ffd6b8d872ef7b9c93da6d9864e836719195f6530550196f683adcbe6b6b1', 'reference_timing_boundary': 'final_av_60s_extension_warm_after_one_excluded_warmup_v1', 'candidate_timing_boundary': 'final_av_60s_extension_warm_after_one_excluded_warmup_v1'}
- Promotion: promote_to_n3_recommended_pending_reviewer=False; force_no_promotion=True; strict_no_automatic_red_flags=True
- Claim boundary: extension output, not native long context; practical Turbo 8-step approximation; N=1 only for `r10_adaptive_tau1_5_step3_diag` vs retained `r9_current_sol_attn`; no formal speedup or human-quality claim.
- Reviewer acceptance: `accepted_independent_reviewer_passed_reconciled_wrapper_metadata_failure`; verdict=`accept`; markdown=`${PWD}/technical_report/evidence/minimax_h3_desktop/long_video/final-av-60s-r9-current-vs-r10-step3-sol-attn-n1-20260816T164226Z/reviewer_verdict.md`; json=`${PWD}/technical_report/evidence/minimax_h3_desktop/long_video/final-av-60s-r9-current-vs-r10-step3-sol-attn-n1-20260816T164226Z/reviewer_verdict.json`; wrapper retry task exited with metadata failure only after reviewer process exit_code=0 and stdout was copied verbatim.
- Reviewer boundary: accepts only the bounded N=1 descriptive/no-promotion classification; automatic proxy red flags remain visible; no formal speedup or N=3/N>=10 promotion.
