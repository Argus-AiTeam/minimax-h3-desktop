# Turbo practical quality-suite review

Status: **OPERATOR ACCEPTED — overall human playback/listening review completed; 8-step retained as the practical default, with the known 4-step visual failure preserved**.

Operator acceptance was recorded in `operator_acceptance.json` after the operator reported that review was complete and the generated results felt very good. This is an overall release acceptance, not a fabricated per-case rubric transcript: unrecorded per-case cells below remain historical PENDING entries, and `p03-object_temporal_consistency-seed1-4step` remains a visual failure. The agent did not perform subjective listening.

Scope and method:

- Every one of the 24 cases was reviewed in the six contact sheets under `contacts/`; no case was omitted or cherry-picked.
- Visual prompt adherence and obvious geometry/artifact checks use five fixed time points per clip (frames 0, 31, 62, 93, 123).
- Temporal status combines those snapshots with the analyzer's no-frozen-transition proxy. It is not a substitute for full-motion human playback.
- Audio status uses complete decode plus `audio_energy_envelopes.json` only. It detects empty/silent/clipped or event-envelope failures but is **not listening-based semantic quality**.
- AV synchronization cannot be certified without human playback. Consequently no row has a full semantic-AV PASS.

Visual summary:

- 8-step: **12/12 visual PASS**.
- 4-step: **11/12 visual PASS**.
- 4-step failure: `p03-object_temporal_consistency-seed1-4step`, where the red teapot is visibly blocky/deformed. This is why 4-step remains an ultra-fast quality-cost experimental schedule rather than the default.
- 8-step remains the practical default. The operator later recorded an overall playback/listening and semantic AV-sync acceptance in `operator_acceptance.json`; no per-case listening rubric transcript was supplied, so individual worksheet cells below remain historical PENDING rather than being silently rewritten.

## Operator listening / semantic AV-sync gate

**Gate result: satisfied by the operator's overall playback/listening acceptance.** The checklist below is retained as the exact review packet and historical per-case worksheet. Because no per-case rubric transcript was supplied, individual PENDING cells are not silently rewritten as case-level PASS results.

Agent boundary:

- The agent **did not listen to these files, did not judge subjective audio semantics, and did not certify semantic AV quality or AV sync**.
- Automated evidence only shows structural AV decode plus audio-envelope proxies. A full Turbo semantic quality PASS requires an operator's auditory/AV-sync review using this packet.
- Turbo remains `practical_disclosed_approx`; neither 4-step nor 8-step is BF16-exact/fidelity evidence.

Run directory and prompt files:

```bash
export RUN_DIR=<private-path>
```

| prompt label | prompt file | subjective listening target |
|---|---|---|
| `baseline_example_1` | `$RUN_DIR/prompts/baseline_example_1.prompt.txt` | Starship bridge: ambient hum, escalating electronic whine, bright fleet-jump boom/crackle/metal stress, then abrupt return to hollow room tone; music should build and cut with the jump. |
| `motion_audio_stress` | `$RUN_DIR/prompts/motion_audio_stress.prompt.txt` | Rainy rooftop jazz trio: saxophone/drum/music audio should be plausible for visible performers, rain/reflection atmosphere should not be replaced by unrelated audio. |
| `object_temporal_consistency` | `$RUN_DIR/prompts/object_temporal_consistency.prompt.txt` | Red teapot pouring into two cups: soft room/pour ambience should be plausible and not contain unrelated loud music/effects. |

Safe local playback guidance:

```bash
# Open one file; replace the filename with any media path from the table below.
ffplay -hide_banner -autoexit "$RUN_DIR/outputs/p01-baseline_example_1-seed0-8step.mp4"

# Alternatives if installed:
mpv --no-resume-playback "$RUN_DIR/outputs/p01-baseline_example_1-seed0-8step.mp4"
vlc "$RUN_DIR/outputs/p01-baseline_example_1-seed0-8step.mp4"
```

Start with low speaker/headphone volume: the p01 cases contain a boom/crackle event and the p02 music cases can be loud. Use normal playback first; if sync is questionable, replay the event once and optionally step/scrub around the salient moment in the same player.

Schedule boundary and recommended order:

1. Review the **12 8-step files first**. These are the practical default candidate, pending human audio/AV-sync acceptance.
2. Review the **12 4-step files second** only as ultra-fast quality-cost experimental comparisons against the same prompt/seed 8-step file.
3. The known visual failure `p03-object_temporal_consistency-seed1-4step` remains a quality failure because the red teapot is visibly blocky/deformed; do not promote 4-step as the default even if its audio/AV sync sounds acceptable.

Exact media checklist:

| case | prompt label | seed | schedule | exact playable media file |
|---|---|---:|---|---|
| `p01-baseline_example_1-seed0-4step` | `baseline_example_1` | 0 | 4-step experimental | `$RUN_DIR/outputs/p01-baseline_example_1-seed0-4step.mp4` |
| `p01-baseline_example_1-seed0-8step` | `baseline_example_1` | 0 | 8-step default candidate | `$RUN_DIR/outputs/p01-baseline_example_1-seed0-8step.mp4` |
| `p01-baseline_example_1-seed1-4step` | `baseline_example_1` | 1 | 4-step experimental | `$RUN_DIR/outputs/p01-baseline_example_1-seed1-4step.mp4` |
| `p01-baseline_example_1-seed1-8step` | `baseline_example_1` | 1 | 8-step default candidate | `$RUN_DIR/outputs/p01-baseline_example_1-seed1-8step.mp4` |
| `p01-baseline_example_1-seed2-4step` | `baseline_example_1` | 2 | 4-step experimental | `$RUN_DIR/outputs/p01-baseline_example_1-seed2-4step.mp4` |
| `p01-baseline_example_1-seed2-8step` | `baseline_example_1` | 2 | 8-step default candidate | `$RUN_DIR/outputs/p01-baseline_example_1-seed2-8step.mp4` |
| `p01-baseline_example_1-seed3-4step` | `baseline_example_1` | 3 | 4-step experimental | `$RUN_DIR/outputs/p01-baseline_example_1-seed3-4step.mp4` |
| `p01-baseline_example_1-seed3-8step` | `baseline_example_1` | 3 | 8-step default candidate | `$RUN_DIR/outputs/p01-baseline_example_1-seed3-8step.mp4` |
| `p02-motion_audio_stress-seed0-4step` | `motion_audio_stress` | 0 | 4-step experimental | `$RUN_DIR/outputs/p02-motion_audio_stress-seed0-4step.mp4` |
| `p02-motion_audio_stress-seed0-8step` | `motion_audio_stress` | 0 | 8-step default candidate | `$RUN_DIR/outputs/p02-motion_audio_stress-seed0-8step.mp4` |
| `p02-motion_audio_stress-seed1-4step` | `motion_audio_stress` | 1 | 4-step experimental | `$RUN_DIR/outputs/p02-motion_audio_stress-seed1-4step.mp4` |
| `p02-motion_audio_stress-seed1-8step` | `motion_audio_stress` | 1 | 8-step default candidate | `$RUN_DIR/outputs/p02-motion_audio_stress-seed1-8step.mp4` |
| `p02-motion_audio_stress-seed2-4step` | `motion_audio_stress` | 2 | 4-step experimental | `$RUN_DIR/outputs/p02-motion_audio_stress-seed2-4step.mp4` |
| `p02-motion_audio_stress-seed2-8step` | `motion_audio_stress` | 2 | 8-step default candidate | `$RUN_DIR/outputs/p02-motion_audio_stress-seed2-8step.mp4` |
| `p02-motion_audio_stress-seed3-4step` | `motion_audio_stress` | 3 | 4-step experimental | `$RUN_DIR/outputs/p02-motion_audio_stress-seed3-4step.mp4` |
| `p02-motion_audio_stress-seed3-8step` | `motion_audio_stress` | 3 | 8-step default candidate | `$RUN_DIR/outputs/p02-motion_audio_stress-seed3-8step.mp4` |
| `p03-object_temporal_consistency-seed0-4step` | `object_temporal_consistency` | 0 | 4-step experimental | `$RUN_DIR/outputs/p03-object_temporal_consistency-seed0-4step.mp4` |
| `p03-object_temporal_consistency-seed0-8step` | `object_temporal_consistency` | 0 | 8-step default candidate | `$RUN_DIR/outputs/p03-object_temporal_consistency-seed0-8step.mp4` |
| `p03-object_temporal_consistency-seed1-4step` | `object_temporal_consistency` | 1 | 4-step experimental; known visual FAIL | `$RUN_DIR/outputs/p03-object_temporal_consistency-seed1-4step.mp4` |
| `p03-object_temporal_consistency-seed1-8step` | `object_temporal_consistency` | 1 | 8-step default candidate | `$RUN_DIR/outputs/p03-object_temporal_consistency-seed1-8step.mp4` |
| `p03-object_temporal_consistency-seed2-4step` | `object_temporal_consistency` | 2 | 4-step experimental | `$RUN_DIR/outputs/p03-object_temporal_consistency-seed2-4step.mp4` |
| `p03-object_temporal_consistency-seed2-8step` | `object_temporal_consistency` | 2 | 8-step default candidate | `$RUN_DIR/outputs/p03-object_temporal_consistency-seed2-8step.mp4` |
| `p03-object_temporal_consistency-seed3-4step` | `object_temporal_consistency` | 3 | 4-step experimental | `$RUN_DIR/outputs/p03-object_temporal_consistency-seed3-4step.mp4` |
| `p03-object_temporal_consistency-seed3-8step` | `object_temporal_consistency` | 3 | 8-step default candidate | `$RUN_DIR/outputs/p03-object_temporal_consistency-seed3-8step.mp4` |

Pass/fail rubric for the operator:

- Mark **audio semantics PASS** only if the clip has audible, non-garbled, non-silent, non-distractingly clipped audio that matches the prompt's listening target above. Fail on wrong-prompt audio, severe dropouts, persistent distortion, missing audio, or semantically unrelated music/effects.
- Mark **AV sync PASS** only if salient sounds line up with visible events under normal playback. Conservative fail examples: a persistent obvious lead/lag, event sounds occurring before/after the visible event by roughly a quarter second or more, or audio continuing/cutting in a way that contradicts the visible scene.
- Prompt-specific sync checks: for p01, charge whine should build with thruster glow and the boom/crackle should coincide with flash/bridge shake; for p02, sax/drum/music should plausibly follow performer motion; for p03, room/pour ambience should not contradict the fixed teapot pour.
- A case's **full subjective AV PASS** requires audio semantics PASS, AV sync PASS, and no visual failure already recorded above. Therefore `p03-object_temporal_consistency-seed1-4step` cannot become a full quality PASS unless a future regenerated artifact replaces the current visually failed file.
- If only a subset can be listened to in one sitting, record the reviewed cases explicitly and leave all others PENDING; do not infer unreviewed cases from the subset.

| case | visual prompt adherence | temporal proxy | visual artifacts | audio proxy | AV sync | review verdict | notes |
|---|---|---|---|---|---|---|---|
| p01-baseline_example_1-seed0-4step | PASS | PASS proxy | PASS | PASS envelope; listening PENDING | PENDING | VISUAL PASS / FULL AV PENDING | Bridge, fleet blue charge, captain close-up retained. |
| p01-baseline_example_1-seed0-8step | PASS | PASS proxy | PASS | PASS envelope; listening PENDING | PENDING | VISUAL PASS / FULL AV PENDING | Stable bridge/fleet/captain sequence. |
| p01-baseline_example_1-seed1-4step | PASS | PASS proxy | PASS | PASS envelope; listening PENDING | PENDING | VISUAL PASS / FULL AV PENDING | Core scene and final captain close-up retained. |
| p01-baseline_example_1-seed1-8step | PASS | PASS proxy | PASS | PASS envelope; listening PENDING | PENDING | VISUAL PASS / FULL AV PENDING | Core scene retained; strong late blue-light event. |
| p01-baseline_example_1-seed2-4step | PASS | PASS proxy | PASS | PASS envelope; listening PENDING | PENDING | VISUAL PASS / FULL AV PENDING | Fleet, bridge silhouette, and close-up retained. |
| p01-baseline_example_1-seed2-8step | PASS | PASS proxy | PASS | PASS envelope; listening PENDING | PENDING | VISUAL PASS / FULL AV PENDING | Coherent fleet/bridge composition. |
| p01-baseline_example_1-seed3-4step | PASS | PASS proxy | PASS | PASS envelope; listening PENDING | PENDING | VISUAL PASS / FULL AV PENDING | Required visual sequence retained. |
| p01-baseline_example_1-seed3-8step | PASS | PASS proxy | PASS | PASS envelope; listening PENDING | PENDING | VISUAL PASS / FULL AV PENDING | Required visual sequence retained with detailed fleet. |
| p02-motion_audio_stress-seed0-4step | PASS | PASS proxy | PASS | PASS continuous-energy proxy; listening PENDING | PENDING | VISUAL PASS / FULL AV PENDING | Rainy rooftop trio/instruments/reflections visible. |
| p02-motion_audio_stress-seed0-8step | PASS | PASS proxy | PASS | PASS continuous-energy proxy; listening PENDING | PENDING | VISUAL PASS / FULL AV PENDING | Rooftop trio and wet reflections retained. |
| p02-motion_audio_stress-seed1-4step | PASS | PASS proxy | PASS | PASS continuous-energy proxy; listening PENDING | PENDING | VISUAL PASS / FULL AV PENDING | Jazz performers and instruments visible. |
| p02-motion_audio_stress-seed1-8step | PASS | PASS proxy | PASS | PASS continuous-energy proxy; listening PENDING | PENDING | VISUAL PASS / FULL AV PENDING | Clearer instrument/person detail. |
| p02-motion_audio_stress-seed2-4step | PASS | PASS proxy | PASS | PASS continuous-energy proxy; listening PENDING | PENDING | VISUAL PASS / FULL AV PENDING | Trio, skyline, rain reflections retained. |
| p02-motion_audio_stress-seed2-8step | PASS | PASS proxy | PASS | PASS continuous-energy proxy; listening PENDING | PENDING | VISUAL PASS / FULL AV PENDING | Coherent trio and city setting. |
| p02-motion_audio_stress-seed3-4step | PASS | PASS proxy | PASS | PASS continuous-energy proxy; listening PENDING | PENDING | VISUAL PASS / FULL AV PENDING | Rooftop performers and reflections retained. |
| p02-motion_audio_stress-seed3-8step | PASS | PASS proxy | PASS | PASS continuous-energy proxy; listening PENDING | PENDING | VISUAL PASS / FULL AV PENDING | Coherent rainy rooftop ensemble. |
| p03-object_temporal_consistency-seed0-4step | PASS | PASS proxy | PASS | PASS nonempty event proxy; listening PENDING | PENDING | VISUAL PASS / FULL AV PENDING | Red teapot, two cups, continuous pour retained. |
| p03-object_temporal_consistency-seed0-8step | PASS | PASS proxy | PASS | PASS nonempty event proxy; listening PENDING | PENDING | VISUAL PASS / FULL AV PENDING | Stable teapot/cups/pour geometry. |
| p03-object_temporal_consistency-seed1-4step | PARTIAL | PASS proxy | **FAIL** | PASS nonempty event proxy; listening PENDING | PENDING | **VISUAL FAIL / FULL AV PENDING** | Teapot body is visibly blocky/deformed despite correct color/action. |
| p03-object_temporal_consistency-seed1-8step | PASS | PASS proxy | PASS | PASS nonempty event proxy; listening PENDING | PENDING | VISUAL PASS / FULL AV PENDING | Stable teapot and two-cup pour. |
| p03-object_temporal_consistency-seed2-4step | PASS | PASS proxy | PASS | PASS nonempty event proxy; listening PENDING | PENDING | VISUAL PASS / FULL AV PENDING | Stable red teapot and pour sequence. |
| p03-object_temporal_consistency-seed2-8step | PASS | PASS proxy | PASS | PASS nonempty event proxy; listening PENDING | PENDING | VISUAL PASS / FULL AV PENDING | Stable geometry and cup details. |
| p03-object_temporal_consistency-seed3-4step | PASS | PASS proxy | PASS | PASS nonempty event proxy; listening PENDING | PENDING | VISUAL PASS / FULL AV PENDING | Red teapot/two cups retained; pour emerges across snapshots. |
| p03-object_temporal_consistency-seed3-8step | PASS | PASS proxy | PASS | PASS nonempty event proxy; listening PENDING | PENDING | VISUAL PASS / FULL AV PENDING | Stable teapot/cups/pour; incidental vessel marking is seed variation. |

Final review boundary: structural AV and visual contact-sheet evidence support choosing 8-step over 4-step, and the operator subsequently recorded an **overall** playback/listening and semantic AV-sync acceptance in `operator_acceptance.json`. Per-case listening/AV-sync cells remain PENDING because no per-case rubric transcript was provided; the agent did not perform subjective listening, and Turbo remains `practical_disclosed_approx`, not BF16 fidelity.
