# SPDX-License-Identifier: Apache-2.0
# VAE spatial tile batching integration overlay image.
# Build context must be the repository root. This starts from the r13 split
# profiler image and installs only the default-off practical-approximate VAE
# stack_tiling switch.

ARG BASE_IMAGE=argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r13-split-profiler
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="MiniMax-H3 A6000 r15 VAE spatial tile batching overlay"
LABEL org.opencontainers.image.description="Default-off practical-approximate H3 video VAE spatial tile batching switch over the retained r10/r13 runtime; no GPU required at build"
LABEL org.opencontainers.image.version="r15-vae-spatial-batching"
LABEL org.opencontainers.image.base.name="argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r13-split-profiler"
LABEL org.opencontainers.image.licenses="Apache-2.0"

ENV MINIMAX_H3_A6000_VIDEO_VAE_SPATIAL_TILE_BATCHING=0

COPY ports/minimax_h3_a6000 /opt/minimax_h3_a6000

RUN set -eux; \
    tmp=/tmp/vllm-omni-r15-vae-spatial-batching; \
    patch=/opt/minimax_h3_a6000/patches/vllm_omni_h3_vae_spatial_tile_batching.patch; \
    helper=/opt/minimax_h3_a6000/integration/r15/install_vae_spatial_tile_batching_patch.py; \
    rm -rf "$tmp"; \
    cp -a /app/vllm-omni "$tmp"; \
    git -C "$tmp" apply "$patch"; \
    python3 "$helper" \
      --patch "$patch" \
      --patched-root "$tmp" \
      --app-root /app/vllm-omni \
      --manifest /opt/minimax_h3_a6000/r15_vae_spatial_tile_batching_changed_files.txt \
      --hash-json /opt/minimax_h3_a6000/r15_vae_spatial_tile_batching_source_hashes.json \
      --hash-sha256 /opt/minimax_h3_a6000/r15_vae_spatial_tile_batching_source_hashes.sha256; \
    test -s /opt/minimax_h3_a6000/r15_vae_spatial_tile_batching_source_hashes.json; \
    test -s /opt/minimax_h3_a6000/r15_vae_spatial_tile_batching_source_hashes.sha256; \
    rm -rf "$tmp"; \
    python3 -c "import os; assert os.environ.get('MINIMAX_H3_A6000_VIDEO_VAE_SPATIAL_TILE_BATCHING') == '0'; from vllm_omni.diffusion.models.minimax_h3 import vae as v; assert v.MINIMAX_H3_A6000_VIDEO_VAE_SPATIAL_TILE_BATCHING_ENV == 'MINIMAX_H3_A6000_VIDEO_VAE_SPATIAL_TILE_BATCHING'; print('minimax_h3_a6000_r15_vae_spatial_tile_batching_import=ok')"
