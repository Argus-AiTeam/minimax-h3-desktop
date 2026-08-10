# SPDX-License-Identifier: Apache-2.0
# Reproducible r4 integration overlay image.
# Build context must be the repository root. This Dockerfile starts from the
# locked P0/r2 vLLM-Omni image, installs only the local overlay package, and
# overwrites patched vLLM-Omni files inside the derived image. It does not edit
# the locked vLLM-Omni source tree on the host.

ARG BASE_IMAGE=argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r2
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="MiniMax-H3 A6000 r4 exact-kernel integration candidate"
LABEL org.opencontainers.image.description="Default-off local overlay plus patched vLLM-Omni files; no GPU required at build"

ENV MINIMAX_H3_A6000_ENABLE_OVERLAY=0 \
    MINIMAX_H3_A6000_ENABLE_TRITON_CANDIDATES=0 \
    MINIMAX_H3_A6000_EXACT_INDEXED_STRATEGY=auto \
    MINIMAX_H3_A6000_ENABLE_FUSED_ADALN=0 \
    MINIMAX_H3_A6000_ENABLE_FUSED_ROPE=0 \
    MINIMAX_H3_A6000_ENABLE_FUSED_SWIGLU=0 \
    MINIMAX_H3_A6000_ENABLE_SOL_ATTN=0 \
    MINIMAX_H3_A6000_SOL_ATTN_CACHE=0 \
    MINIMAX_H3_A6000_ENABLE_TELEMETRY=0 \
    MINIMAX_H3_A6000_TELEMETRY_ATEXIT=0 \
    MINIMAX_H3_A6000_TELEMETRY_JSON=""

COPY ports/minimax_h3_a6000 /opt/minimax_h3_a6000

RUN set -eux; \
    python3 -m pip install --no-deps --no-build-isolation /opt/minimax_h3_a6000; \
    tmp=/tmp/vllm-omni-r4-patched; \
    rm -rf "$tmp"; \
    cp -a /app/vllm-omni "$tmp"; \
    git -C "$tmp" apply /opt/minimax_h3_a6000/patches/vllm_omni_h3_a6000_opt_in.patch; \
    for f in \
      vllm_omni/diffusion/attention/backends/registry.py \
      vllm_omni/diffusion/attention/backends/sol_attn_h3_a6000.py \
      vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py; do \
        install -D -m 0644 "$tmp/$f" "/app/vllm-omni/$f"; \
    done; \
    python3 -c "from pathlib import Path; import site; patched=Path('/tmp/vllm-omni-r4-patched'); files=('vllm_omni/diffusion/attention/backends/registry.py','vllm_omni/diffusion/attention/backends/sol_attn_h3_a6000.py','vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py'); site_vllm=next((Path(p)/'vllm_omni' for p in site.getsitepackages() if (Path(p)/'vllm_omni').exists()), None); [((site_vllm.parent/rel).parent.mkdir(parents=True, exist_ok=True), (site_vllm.parent/rel).write_bytes((patched/rel).read_bytes())) for rel in files] if site_vllm else None"; \
    rm -rf /tmp/vllm-omni-r4-patched; \
    python3 -c "import minimax_h3_a6000; from minimax_h3_a6000.env import DEFAULT_ENV_SWITCHES as d; assert d['MINIMAX_H3_A6000_ENABLE_OVERLAY'] == '0'; assert d['MINIMAX_H3_A6000_ENABLE_TELEMETRY'] == '0'; print('minimax_h3_a6000_overlay_import=ok')"
