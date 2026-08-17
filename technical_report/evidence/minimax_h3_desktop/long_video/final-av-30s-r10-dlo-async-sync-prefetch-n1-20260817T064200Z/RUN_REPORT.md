# r10 DLO async sync-prefetch Final-AV 30s N=1 RUN_REPORT

Status: **reject** — reject_no_promotion_matched_n1_dlo_async_sync_prefetch_gate_failed.
Reference warm E2E: 1335.020644 s; candidate warm E2E: 1334.684101 s; delta: 0.025209% (threshold 1.000%).
Reference group-first sync-prefetch host enqueue: 297.799103 s; candidate: 0.008768 s.
Async sync-prefetch copy counts: reference 0, candidate 99.
Final AV complete: reference True, candidate True; validation pass: reference True, candidate True.
Same GPU/workload/timing/generation: True/True/True/True.
Objective noninferiority: pass; Sol-Attn invariants reference/candidate: pass/pass.
Decision gates: {"candidate_final_av_complete": true, "candidate_sol_attn_invariants_pass": true, "candidate_validation_pass": true, "dlo_async_sync_prefetch_telemetry_pass": true, "objective_noninferiority_pass": true, "reference_final_av_complete": true, "reference_sol_attn_invariants_pass": true, "reference_validation_pass": true, "same_generation_extension": true, "same_gpu": true, "same_timing_boundary": true, "same_workload": true, "warm_e2e_delta_meets_min_pct": false}

Claim boundary: N=1 route gate only; no formal speedup, BF16 fidelity, native long context, human/product quality, SOTA, N=3, or N>=10 claim.
