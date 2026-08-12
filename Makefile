.PHONY: help dry-run runtime models demo test audit

help:
	@printf '%s\n' \
	  'MiniMax-H3 single-A6000 commands' \
	  '  make dry-run  - print all plans without model/GPU/Docker actions' \
	  '  make runtime  - build the pinned vLLM-Omni CUDA image' \
	  '  make models   - download licensed assets and merge Turbo (requires license acknowledgement)' \
	  '  make demo     - generate a Turbo 8-step MP4 (requires license acknowledgement)' \
	  '  make test     - run CPU/static tests' \
	  '  make audit    - audit the public tree, including curated example media'

dry-run:
	bash scripts/build_runtime.sh --dry-run
	bash scripts/prepare_models.sh --dry-run
	bash scripts/run_turbo_demo.sh --dry-run

runtime:
	bash scripts/build_runtime.sh

models:
	@test "$${I_ACCEPT_MINIMAX_H3_LICENSE:-}" = YES || (echo 'Set I_ACCEPT_MINIMAX_H3_LICENSE=YES after reviewing the upstream license.' >&2; exit 2)
	bash scripts/prepare_models.sh

demo:
	@test "$${I_ACCEPT_MINIMAX_H3_LICENSE:-}" = YES || (echo 'Set I_ACCEPT_MINIMAX_H3_LICENSE=YES after reviewing the upstream license.' >&2; exit 2)
	bash scripts/run_turbo_demo.sh

test:
	PYTHONPATH=code:ports/minimax_h3_a6000/src python3 -m pytest -q tests ports/minimax_h3_a6000/tests

audit:
	python3 tools/publication_audit.py --root . --max-bytes 15000000 --json
