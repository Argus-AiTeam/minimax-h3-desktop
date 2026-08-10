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

## DMD / DMD2

- Evidence: `dmd_primary_source_note.md`.
- Status: **blocked_research_only_no_go_after_turbo_unless_feasibility_changes**.
- DMD remains a no-go after Turbo unless a legal H3 DMD/DMD2 recipe/checkpoint, resource profile, and AV quality bar appear in future evidence.

## Accepted / rejected / blocked matrix

### Accepted
- A single-A6000 internal BF16-exact baseline v2 is certified on physical GPU3 with N=13 total requests, N=10 true warm-primary requests, and three session-first requests. Evidence: `baseline_a6000/baseline_certification.json`. Limit: Internal same-device denominator only; not an optimized or external reproduction.
- The latest GPU3 paired Turbo timing run is accepted as the only practical speed result: two excluded warmups, paired N=10 per schedule, strict structural AV pass, and baseline-v2 warm N=10 denominator. Evidence: `turbo_merged/timing_repeats/LATEST_RUN_ID`, `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/timing_summary.json`, `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/merge_manifest.json`. Limit: Practical approximation only; semantic AV quality and human auditory listening remain pending; 8-step is the default candidate and 4-step is ultra-fast quality-cost experimental.
- AdaLN is an N=1 exact-output candidate only; the original harness-tail failure is preserved and the single-run benefit is below baseline warm-run noise. Evidence: `sol_engine_port/r5_ablation_20260809T181515Z/adaln/quality_vs_dense.json`, `sol_engine_port/a6000_adaln_candidate_50step_20260809T190402Z/av_validation_posthoc.json`, `sol_engine_port/a6000_adaln_candidate_50step_20260809T190402Z/candidate_verdict.json`. Limit: Not an accepted N=10 speedup and not deployed as a certified fidelity path.

### Rejected
- Turbo merged LoRA is a BF16-exact/fidelity result. Reason: Turbo is practical_disclosed_approx and uses a statically merged LoRA approximation. Evidence: `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/timing_summary.json`.
- Treating the earlier GPU2 Turbo smoke run as an accepted speedup denominator/result. Reason: GPU2 smoke is retained only as bring-up evidence; the only speedup result uses same physical GPU3 paired timing against baseline v2 warm N=10. Evidence: `turbo_merged/runs/gpu2_turbo_20260809T195558Z/turbo_summary.json`, `turbo_merged/timing_repeats/gpu3_turbo_paired_n10_20260810T025102Z/timing_summary.json`.
- RoPE/all-exact kernels are accepted exact replacements. Reason: Same-prompt diagnostic evidence shows non-zero video/audio drift. Evidence: `sol_engine_port/r5_ablation_20260809T181515Z/rope/quality_vs_dense.json`, `sol_engine_port/r5_ablation_20260809T181515Z/all_exact/quality_vs_dense.json`.
- SwiGLU is a retained practical speed gain. Reason: Current evidence is exact diagnostic output only; no retained speedup gain is accepted or deployed. Evidence: `sol_engine_port/r5_ablation_20260809T181515Z/swiglu/quality_vs_dense.json`, `sol_engine_port/r5_ablation_20260809T181515Z/swiglu/http_metrics.txt`.
- Toy Sol-Attn is deployed or faster. Reason: Current toy harness is kernel-candidate-only, model_load=false, and sparse median is slower than dense. Evidence: `sol_engine_port/sol_attn_gpu_20260809T173323Z/result.json`.

### Blocked
- MiniMax-H3 DMD/DMD2 is a no-go after Turbo unless feasibility evidence changes. Status: blocked_research_only_no_go_after_turbo_unless_feasibility_changes Evidence: `dmd_primary_source_note.md`.
- Turbo semantic AV quality and human-auditory quality are fully certified. Status: blocked_human_auditory_listening_pending_semantic_quality_not_certified Evidence: `turbo_merged/LATEST_QUALITY_SUITE_RUN_ID`, `turbo_merged/quality_suite_runs/gpu3_turbo_quality_20260810T005611Z/quality_suite_analysis.json`, `turbo_merged/quality_suite_runs/gpu3_turbo_quality_20260810T005611Z/human_review.md`.

## Exact operator commands left for follow-up

```bash
PYTHONPATH=code:. python3 -m pytest -q tests ports/minimax_h3_a6000/tests
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
