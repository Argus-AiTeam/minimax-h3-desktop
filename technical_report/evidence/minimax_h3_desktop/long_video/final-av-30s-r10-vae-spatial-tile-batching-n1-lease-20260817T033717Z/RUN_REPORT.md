# r10 Video VAE Spatial Tile Batching Final-AV 30s N=1 RUN_REPORT

- Status: `pass`
- Classification: `promote_to_n3_default_off_r10_video_vae_spatial_tile_batching_n1_pending_reviewer`
- Reference evidence: `${PWD}/technical_report/evidence/minimax_h3_desktop/long_video/final-av-30s-r10-vae-spatial-tile-batching-n1-lease-20260817T033717Z/r10_adaptive_tau1_5_step3_diag_vae_serial`
- Candidate evidence: `${PWD}/technical_report/evidence/minimax_h3_desktop/long_video/final-av-30s-r10-vae-spatial-tile-batching-n1-lease-20260817T033717Z/r10_adaptive_tau1_5_step3_diag_vae_spatial_tile_batching`
- Warm E2E reference/candidate: 1335.018s / 1299.728s
- N=1 warm delta (route-gate only, not speedup): 2.643% (threshold 1.000%)
- Video VAE wall reference/candidate: 202.71950144297443 / 167.8020734799211
- Sol-Attn sparse calls reference/candidate: 2352 / 2352; fallback/materialization/input-copy fixed by gates.
- Automatic proxy flags reference/candidate: ['near_frozen_transition_fraction', 'av_sync_proxy_offset'] / ['near_frozen_transition_fraction']
- Failed gates: []
- Claim boundary: Matched N=1 30-second final-AV extension route gate only. Video VAE spatial tile batching is practical approximate VAE decode, not exact/lossless or BF16 fidelity; output is extension/chunked, not native long context; no formal speedup, human-quality, product, public-comparison, or SOTA claim.
- Reviewer acceptance: `pending_host_reviewer_not_invoked_by_engineer`.
