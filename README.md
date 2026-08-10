# MiniMax-H3 desktop code-only release tree

This is the sanitized, code-only export for the private `Argus-AiTeam/minimax-h3-desktop` repository. It contains source, patches, tests, schemas, small evidence JSON, locked runtime metadata, and reports. It does **not** contain model weights, adapters, merged checkpoints, generated media, Docker layers, caches, virtual environments, or vendored upstream/runtime source trees.

## Weight and license boundary

The first full model preparation is approximately **144 GB** and requires the official model license, authorization, and local credentials. This release tree never bundles weights and the default workflow is fail-closed: dry-run and publication-audit checks are safe; real download, container, deployment, demo, or GPU execution requires separate operator approval in the private environment.

## One-command A6000 workflow shape

Run from the export root:

```bash
bash scripts/a6000_one_command.sh --dry-run
```

The dry run prints the intended stages: preflight, model-prepare, deploy, demo, and verify. Non-dry model preparation/deploy/demo actions are placeholders until a later authorized private run supplies official credentials, model storage, runtime image access, and GPU approval.

## Local-only Git initialization for chenxi

Run only after the export has passed the publication audit. These commands are local repository setup only and do not create a remote, tag, release, or push anything:

```bash
git init
git config --local user.name chenxi
git config --local user.email cxx2216@163.com
python3 tools/publication_audit.py --root . --max-bytes 1000000 --prohibited-terms-file <local-file-not-committed>
git add .
git status --short
git commit -m "Initial private Argus MiniMax-H3 desktop release tree"
```

Remote publication is allowed only after the export passes the publication audit and `gh auth status` confirms the intended private account `Chenxxxxxx06`.

## Included evidence boundaries

- `technical_report/minimax_h3_a6000_performance.md` is a CPU-only evidence reader output; it does not claim deployment for pending lanes.
- `technical_report/final_technical_report.md` preserves accepted/rejected/blocked lane boundaries.
- `technical_report/evidence/minimax_h3_desktop/quantization_feasibility_a6000.md` records that there is no runnable no-new-download quantized A6000 candidate today.
- Runtime metadata under `runtime/single_a6000_bf16/` is lock metadata only; vendored runtime source is intentionally excluded.

## Attribution

See `NOTICE`, `LICENSE`, and the port-level `ports/minimax_h3_a6000/NOTICE`. NVLabs/Sol-Engine team attribution is preserved for Sol-Engine-derived design and kernel-candidate work; upstream projects retain their own licenses and authorship.
