# MiniMax-H3 on A6000 — Argus desktop release tree

> **中文定位**：这是 Argus-AiTeam 为 **单卡 NVIDIA RTX A6000 48 GB（SM86）桌面/工作站**整理的 MiniMax-H3 代码发布树。它只包含源码、补丁、测试、Schema、报告和小型证据 JSON；不包含模型权重、适配器、生成视频、Docker 层、缓存、凭据或 vendored 上游源码。
>
> **English positioning**: this is the Argus-AiTeam maintained **code-only** MiniMax-H3 desktop/A6000 release tree. It documents a conservative BF16 fidelity baseline, practical Turbo candidates, DLO capacity evidence, and default-off Sol/attention diagnostics without bundling private runtime artifacts or weights.

MiniMax-H3 is an omni-modal diffusion pipeline for synchronized video + audio generation. This export is meant to be auditable first and runnable only after a private operator supplies licensed model access, local storage, and GPU approval.

## Quick start / 快速开始

From the export root, these commands are CPU-only and do not load models, use Docker, touch GPUs, download files, publish, or call Git:

```bash
bash scripts/a6000_one_command.sh --dry-run
python3 tools/publication_audit.py --root . --max-bytes 1000000 --prohibited-terms-file <local-file-not-committed>
PYTHONPATH=code:ports/minimax_h3_a6000/src python3 -m pytest -q tests/test_github_release_tree_builder.py tests/test_minimax_h3_a6000_performance_report.py
```

The dry run prints the intended preflight/model-prepare/deploy/demo/verify stages. Non-dry model preparation, container/runtime deployment, real generation, and GPU diagnostics remain blocked unless a separate private supervised run authorizes them.

## Hardware and asset requirements / 硬件与资产要求

| Requirement | Current release boundary |
|---|---|
| GPU | Single RTX A6000 48 GB, compute capability 8.6, verified internally as the target class. |
| Host memory | Baseline evidence observed up to about 205 GiB host memory; plan for a high-memory workstation. |
| Model assets | First full MiniMax-H3 preparation is about **144 GB** of licensed weights; weights are never included here. |
| Runtime | Locked runtime metadata is included under `runtime/single_a6000_bf16/`; vendored runtime source and Docker layers are intentionally excluded. |
| Network/credentials | Required only for a later private model/runtime preparation; no credentials belong in this tree. |

## Architecture and DLO map / 架构与 DLO 示意

```mermaid
flowchart LR
    U[Prompt / media request] --> T[Qwen3-VL conditioner]
    T --> P[Packed H3 sequence
text + conditions + audio + video]
    P --> D[MiniMax-H3 DiT
50 denoising blocks]
    D --> V[Video VAE]
    D --> A[Audio VAE]
    V --> M[Muxed video]
    A --> M

    subgraph Stable[Stable evidence]
      B[BF16 dense baseline
50-step fidelity denominator]
      TB[Turbo LoRA practical lane
4/8-step timing + structural AV]
    end

    subgraph Diagnostics[Default-off diagnostics]
      DLO[DLO resident-layer candidates
capacity gates only]
      SOL[Sol-Attn r6 opt-in
fail-closed: sparse=0]
      EX[Exact Triton kernels
kernel-only microbench]
    end

    D -. evidence denominator .-> B
    D -. practical candidate .-> TB
    D -. memory/layout study .-> DLO
    D -. attention experiment .-> SOL
    D -. op fusion candidates .-> EX
```

DLO means resident-layer placement/continuation tuning. The current DLO evidence is a capacity gate, not a formal N10 performance result.

## Evidence-backed 1344x768 status / 证据支持的 1344x768 状态

All rows are read from checked-in reports/evidence for the same 1344x768, 5.166667 s, 124-frame, 24 FPS workload unless marked as diagnostic. Do **not** compare rows across lanes as if they were one benchmark leaderboard.

| Lane / item | Evidence status | Timing | Quality / structure | Release meaning |
|---|---|---:|---|---|
| BF16 dense baseline, 50 steps | certified internal same-physical-device A6000 baseline | warm N10 median **1792.2021025 s** | structural AV pass 13/13 | Fidelity denominator only; not an optimized result. |
| Turbo LoRA, 8 steps | practical paired N10 | median **290.9976015 s**; **6.158820874x** vs BF16 warm median | structural AV pass; semantic quality and human listening pending | Current practical default candidate, not BF16-exact. |
| Turbo LoRA, 4 steps | practical paired N10 | median **149.6191865 s**; **11.978424321x** vs BF16 warm median | structural AV pass; stronger quality-risk boundary | Ultra-fast experimental option, not fidelity evidence. |
| DLO resident layers 13, 5-step capacity gate | present capacity gate | dense 188.098444 s → candidate 186.773476 s | hash match true | Capacity/placement signal only; formal DLO N10 pending. |
| Sol-Attn r6 opt-in, 5-step diagnostic | **fail-closed** | dense **188.204918 s**, opt-in **187.019730 s** | identical SHA256 and structurally valid; `unsupported_contiguity=208`, `sparse_candidate_calls=0`, `sparse_calls=0` | Metadata path accepted but every opt-in attention call declined. This is **not a speedup claim** and is not release-eligible runtime evidence. |

## Stable vs experimental / 稳定与实验边界

- **Stable / 稳定**: code-only export builder, publication audit, BF16 baseline report, locked runtime metadata, dry-run workflow, and CPU/static tests.
- **Practical but not fidelity / 实用但非保真**: Turbo LoRA 8-step and 4-step timing. These outputs passed structural AV checks, but semantic/video/audio quality still needs separate review before product claims.
- **Diagnostic only / 仅诊断**: exact Triton op kernels, DLO capacity gates, and Sol-Attn r6. Kernel microbenchmarks and 5-step diagnostics are not H3 formal speed claims.
- **Blocked / 阻塞**: DMD/DMD2 remains research-only because there is no first-source H3 recipe/checkpoint basis in this tree.

## Quality limits / 质量限制

Structural AV validation means the file decodes with expected video/audio properties. It is not equivalent to semantic quality, prompt faithfulness, or human auditory approval. Turbo and diagnostic lanes must never be relabeled as BF16-exact fidelity evidence. Sol-Attn timing from r6 must not be reported as speedup until sparse calls are actually executed and accepted quality checks exist.

## Reports and evidence links / 报告与证据链接

- `technical_report/minimax_h3_a6000_performance.md` — CPU-only evidence reader with lane boundaries and pending/blocker status.
- `technical_report/final_technical_report.md` — high-level accepted/rejected/blocked summary.
- `technical_report/evidence/minimax_h3_desktop/baseline_a6000/baseline_certification.json` — BF16 baseline denominator.
- `technical_report/evidence/minimax_h3_desktop/dlo_autotune/runs/a6000_dlo_candidate_50_rl16_20260810T141257Z/candidate50_summary.json` — DLO candidate-50 artifact.
- `technical_report/evidence/minimax_h3_desktop/delivery/argus_ir04_delivery_summary.json` — delivery aggregation.
- `ports/minimax_h3_a6000/README.md` — default-off exact-kernel and Sol-Attn port details.

Raw videos, model weights, caches, Docker layers, and private run directories are intentionally outside the release manifest.

## Troubleshooting / 常见问题

| Symptom | Likely cause | Safe next step |
|---|---|---|
| Publication audit flags a private path or prohibited term | A copied text file contains local machine detail | Fix the source text or pass a local prohibited-terms file; rebuild the export. |
| `--dry-run` prints placeholders instead of running deployment | This export is intentionally code-only | Treat it as expected; non-dry stages require private approval. |
| Sol-Attn appears slightly faster in r6 timing | The opt-in path fell back before sparse kernels ran | Report it as fail-closed diagnostic timing only; do not claim speed. |
| Turbo output is fast but visually questionable | Structural AV is not semantic quality | Run human/semantic review before user-facing quality claims. |
| Model files are missing | Weights are excluded by design | Obtain official MiniMax-H3 access and prepare assets only in a private authorized environment. |

## External supervised commands for later private operators

These are documentation-only runbook commands. They are **not** run by the release builder and require a private workstation with approved GPU/runtime/model access:

```bash
# Rebuild the CPU-only performance report from already-written evidence.
python3 tools/minimax_h3_a6000_performance_report.py --evidence-root technical_report/evidence/minimax_h3_desktop --out technical_report/minimax_h3_a6000_performance.md

# Recheck the static release tree before any publication decision.
python3 tools/build_github_release_tree.py --source-root . --manifest release/github_release_manifest.json --dest <export-root> --force --json
python3 tools/publication_audit.py --root <export-root> --max-bytes 1000000 --prohibited-terms-file <local-file-not-committed>

# Optional private GPU2 diagnostic retry; keep Sol-Attn default-off unless the runbook explicitly opts in.
PYTHONPATH=ports/minimax_h3_a6000/src python3 ports/minimax_h3_a6000/gpu_exact_kernel_test.py --device cuda:0 --output <private-evidence-root>/h3_exact_correctness.json
PYTHONPATH=ports/minimax_h3_a6000/src python3 ports/minimax_h3_a6000/gpu_sol_attn_sm86_harness.py --device cuda:0 --mode both --output <private-evidence-root>/h3_sol_attn_sm86.json
```

## Roadmap / 路线图

1. Keep the release tree audit-clean and code-only.
2. Finish quality review for practical Turbo lanes before product language changes.
3. Promote DLO only after a formal same-device N10 timing result exists.
4. Advance Sol-Attn only after the contiguity gate is resolved, `sparse_calls>0`, outputs remain quality-accepted, and telemetry proves no silent math change.
5. Revisit DMD/DMD2 only if a real H3 first-source recipe/checkpoint appears.

## Attribution / 致谢

See `NOTICE`, `LICENSE`, and `ports/minimax_h3_a6000/NOTICE`. NVLabs/Sol-Engine attribution is preserved for Sol-Engine-derived design/kernel-candidate work; MiniMax, vLLM-Omni, Hugging Face ecosystem packages, PyTorch, Triton, and other upstream projects retain their own licenses and authorship.
