# SPDX-License-Identifier: Apache-2.0
# VAE bounded tile-batch-size integration overlay image.
# Build context must be the repository root. This starts from the r13 split
# profiler image and installs only default-off VAE adapter switches.

ARG BASE_IMAGE=argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r13-split-profiler
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="MiniMax-H3 A6000 r20 VAE bounded tile batching overlay"
LABEL org.opencontainers.image.description="Default-off practical-approximate H3 video VAE bounded tile-batch-size switch over the retained r10/r13 runtime; no GPU required at build"
LABEL org.opencontainers.image.version="r20-vae-bounded-tile-batching"
LABEL org.opencontainers.image.base.name="argus/minimax-h3-vllm-omni:8e2e9b6b53e8-r13-split-profiler"
LABEL org.opencontainers.image.licenses="Apache-2.0"

ENV MINIMAX_H3_A6000_VIDEO_VAE_SPATIAL_TILE_BATCHING=0
ENV MINIMAX_H3_A6000_VIDEO_VAE_TILE_BATCH_SIZE=0
ENV MINIMAX_H3_A6000_VIDEO_VAE_GRAPH_SAFE_REVERT=0

COPY ports/minimax_h3_a6000 /opt/minimax_h3_a6000

RUN set -eux; \
    tmp=/tmp/vllm-omni-r20-vae-bounded-tile-batching; \
    patch=/opt/minimax_h3_a6000/patches/vllm_omni_h3_vae_bounded_tile_batching.patch; \
    helper=/opt/minimax_h3_a6000/integration/r15/install_vae_spatial_tile_batching_patch.py; \
    rm -rf "$tmp"; \
    cp -a /app/vllm-omni "$tmp"; \
    git -C "$tmp" apply "$patch"; \
    python3 "$helper" \
      --patch "$patch" \
      --patched-root "$tmp" \
      --app-root /app/vllm-omni \
      --manifest /opt/minimax_h3_a6000/r20_vae_bounded_tile_batching_changed_files.txt \
      --hash-json /opt/minimax_h3_a6000/r20_vae_bounded_tile_batching_source_hashes.json \
      --hash-sha256 /opt/minimax_h3_a6000/r20_vae_bounded_tile_batching_source_hashes.sha256; \
    test -s /opt/minimax_h3_a6000/r20_vae_bounded_tile_batching_source_hashes.json; \
    test -s /opt/minimax_h3_a6000/r20_vae_bounded_tile_batching_source_hashes.sha256; \
    rm -rf "$tmp"; \
    python3 -c "import os; assert os.environ.get('MINIMAX_H3_A6000_VIDEO_VAE_SPATIAL_TILE_BATCHING') == '0'; assert os.environ.get('MINIMAX_H3_A6000_VIDEO_VAE_TILE_BATCH_SIZE') == '0'; from vllm_omni.diffusion.models.minimax_h3 import vae as v; assert v.MINIMAX_H3_A6000_VIDEO_VAE_TILE_BATCH_SIZE_ENV == 'MINIMAX_H3_A6000_VIDEO_VAE_TILE_BATCH_SIZE'; print('minimax_h3_a6000_r20_vae_bounded_tile_batching_import=ok')"
