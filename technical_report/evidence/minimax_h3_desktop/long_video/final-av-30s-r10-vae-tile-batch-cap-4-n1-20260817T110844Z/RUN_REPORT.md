# r10 Video VAE Spatial Tile Batching Final-AV 30s N=1 RUN_REPORT

- Status: `reject`
- Classification: `reject_no_promotion_r10_video_vae_spatial_tile_batching_n1_failed_timing_or_proxy_gate`
- Reference evidence: `${PWD}/technical_report/evidence/minimax_h3_desktop/long_video/final-av-30s-r10-vae-tile-batch-cap-4-n1-20260817T110844Z/r10_adaptive_tau1_5_step3_diag_vae_serial`
- Candidate evidence: `${PWD}/technical_report/evidence/minimax_h3_desktop/long_video/final-av-30s-r10-vae-tile-batch-cap-4-n1-20260817T110844Z/r10_adaptive_tau1_5_step3_diag_vae_tile_batch_cap_4`
- Warm E2E reference/candidate: 1331.377s / 1302.506s
- N=1 warm delta (route-gate only, not speedup): 2.168% (threshold 1.000%)
- Video VAE wall reference/candidate: 202.42423862492433 / 173.88592834194424
- Sol-Attn sparse calls reference/candidate: 2352 / 2352; fallback/materialization/input-copy fixed by gates.
- Automatic proxy flags reference/candidate: ['near_frozen_transition_fraction', 'av_sync_proxy_offset'] / ['near_frozen_transition_fraction']
- Strict automatic-red-flag gate: fail (promotion requires zero flags in both lanes)
- Failed gates: ['reference_no_automatic_proxy_red_flags', 'candidate_no_automatic_proxy_red_flags', 'objective_5pct_noninferiority_core_metrics']
- Claim boundary: Matched N=1 30-second final-AV extension route gate only. Video VAE spatial tile batching or bounded tile-batch-size is practical approximate VAE decode, not exact/lossless or BF16 fidelity; output is extension/chunked, not native long context; no formal speedup, human-quality, product, public-comparison, or SOTA claim.
- Reviewer acceptance: `pending_host_reviewer_not_invoked_by_engineer`.
