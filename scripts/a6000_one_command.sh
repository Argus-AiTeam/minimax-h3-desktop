#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=1
STAGE="all"
TERMS_FILE=""

usage() {
  cat <<'EOF'
Usage: bash scripts/a6000_one_command.sh --dry-run [--stage preflight|model-prepare|deploy|demo|verify|all] [--prohibited-terms-file PATH]

Dry-run is the only enabled mode in this public-ready export. The first full
model preparation is about 144 GB and requires official license/auth and a
separate private authorization. This script never bundles weights.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --stage)
      STAGE="${2:?missing stage}"
      shift 2
      ;;
    --prohibited-terms-file)
      TERMS_FILE="${2:?missing terms file}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 64
      ;;
  esac
done

if [[ "$DRY_RUN" != "1" ]]; then
  echo "ERROR: non-dry model preparation, deploy, demo, and GPU execution are blocked in this export." >&2
  exit 2
fi

run_stage() {
  local name="$1"
  case "$name" in
    preflight)
      echo "[DRY-RUN] preflight: inspect local metadata and run publication audit; no GPU, Docker, model load, network, or download."
      ;;
    model-prepare)
      echo "[DRY-RUN] model-prepare: first full model download is about 144 GB and requires official license/auth; weights are never bundled."
      ;;
    deploy)
      echo "[DRY-RUN] deploy: locked runtime metadata is present; actual container/runtime deployment needs separate private approval."
      ;;
    demo)
      echo "[DRY-RUN] demo: real generation is disabled here; no media output is created."
      ;;
    verify)
      echo "[DRY-RUN] verify: running publication audit."
      if [[ -n "$TERMS_FILE" ]]; then
        python3 "$ROOT/tools/publication_audit.py" --root "$ROOT" --max-bytes 1000000 --prohibited-terms-file "$TERMS_FILE"
      else
        python3 "$ROOT/tools/publication_audit.py" --root "$ROOT" --max-bytes 1000000
      fi
      ;;
    *)
      echo "ERROR: unsupported stage: $name" >&2
      exit 64
      ;;
  esac
}

case "$STAGE" in
  all)
    for stage in preflight model-prepare deploy demo verify; do
      run_stage "$stage"
    done
    ;;
  preflight|model-prepare|deploy|demo|verify)
    run_stage "$STAGE"
    ;;
  *)
    echo "ERROR: unsupported --stage: $STAGE" >&2
    exit 64
    ;;
esac
