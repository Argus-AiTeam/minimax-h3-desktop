# SPDX-License-Identifier: Apache-2.0
# Reproducible r6 Sol-Attn integration overlay image.
# Build context must be the repository root. This Dockerfile starts from the
# locked P0/r2 vLLM-Omni image, installs only the local overlay package, applies
# the current Sol-Attn/exact-kernel opt-in patch to a disposable tree, and then
# overwrites every patch-changed vLLM-Omni file inside both image import
# surfaces. It does not edit the locked vLLM-Omni source tree on the host.
#
# Attribution: the local overlay preserves NVLabs/Sol-Engine, vLLM-Omni,
# MiniMax, Hugging Face, and other upstream project attribution through the
# copied NOTICE and UPSTREAM.md files; this image is only a local adapter layer.

ARG BASE_IMAGE=argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r2
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="MiniMax-H3 A6000 r6 Sol-Attn integration overlay"
LABEL org.opencontainers.image.description="Default-off local overlay plus audited patched vLLM-Omni files; no GPU required at build"
LABEL org.opencontainers.image.version="r6"
LABEL org.opencontainers.image.base.name="argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r2"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.vendor="Local MiniMax-H3 A6000 adapter preserving NVLabs/Sol-Engine and upstream attribution"

ENV MINIMAX_H3_A6000_ENABLE_OVERLAY=0 \
    MINIMAX_H3_A6000_ENABLE_TRITON_CANDIDATES=0 \
    MINIMAX_H3_A6000_EXACT_INDEXED_STRATEGY=auto \
    MINIMAX_H3_A6000_ENABLE_FUSED_ADALN=0 \
    MINIMAX_H3_A6000_ENABLE_FUSED_ADALN_MODULATE=0 \
    MINIMAX_H3_A6000_ENABLE_FUSED_ADALN_GATE=0 \
    MINIMAX_H3_A6000_ENABLE_FUSED_ROPE=0 \
    MINIMAX_H3_A6000_ENABLE_FUSED_SWIGLU=0 \
    MINIMAX_H3_A6000_ENABLE_SOL_ATTN=0 \
    MINIMAX_H3_A6000_SOL_ATTN_CACHE=0 \
    MINIMAX_H3_A6000_SOL_ATTN_STRICT=0 \
    MINIMAX_H3_A6000_ENABLE_TELEMETRY=0 \
    MINIMAX_H3_A6000_TELEMETRY_ATEXIT=0 \
    MINIMAX_H3_A6000_TELEMETRY_JSON="" \
    MINIMAX_H3_A6000_SOL_ATTN_TELEMETRY_JSON=""

COPY ports/minimax_h3_a6000 /opt/minimax_h3_a6000

RUN set -eux; \
    python3 -m pip install --no-deps --no-build-isolation /opt/minimax_h3_a6000; \
    tmp=/tmp/vllm-omni-r6-patched; \
    patch=/opt/minimax_h3_a6000/patches/vllm_omni_h3_a6000_opt_in.patch; \
    helper=/opt/minimax_h3_a6000/integration/r6/dual_install_patch_files.py; \
    rm -rf "$tmp"; \
    cp -a /app/vllm-omni "$tmp"; \
    python3 "$helper" --patch "$patch" --list-patch-files > /opt/minimax_h3_a6000/r6_patch_changed_files.txt; \
    grep -qx 'vllm_omni/diffusion/attention/backends/registry.py' /opt/minimax_h3_a6000/r6_patch_changed_files.txt; \
    grep -qx 'vllm_omni/diffusion/attention/backends/sol_attn_h3_a6000.py' /opt/minimax_h3_a6000/r6_patch_changed_files.txt; \
    grep -qx 'vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py' /opt/minimax_h3_a6000/r6_patch_changed_files.txt; \
    git -C "$tmp" apply "$patch"; \
    python3 "$helper" \
      --patch "$patch" \
      --patched-root "$tmp" \
      --app-root /app/vllm-omni \
      --manifest /opt/minimax_h3_a6000/r6_patch_changed_files.txt \
      --hash-json /opt/minimax_h3_a6000/r6_patched_source_hashes.json \
      --hash-sha256 /opt/minimax_h3_a6000/r6_patched_source_hashes.sha256; \
    test -s /opt/minimax_h3_a6000/r6_patched_source_hashes.json; \
    test -s /opt/minimax_h3_a6000/r6_patched_source_hashes.sha256; \
    rm -rf "$tmp"; \
    python3 -c "import minimax_h3_a6000; from minimax_h3_a6000.env import DEFAULT_ENV_SWITCHES as d; assert d['MINIMAX_H3_A6000_ENABLE_OVERLAY'] == '0'; assert d['MINIMAX_H3_A6000_ENABLE_TELEMETRY'] == '0'; assert d['MINIMAX_H3_A6000_ENABLE_SOL_ATTN'] == '0'; print('minimax_h3_a6000_r6_overlay_import=ok')"
