# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PORT = ROOT / "ports" / "minimax_h3_a6000"
SRC = PORT / "src"
sys.path.insert(0, str(SRC))

from minimax_h3_a6000.patch_builder import env_switch_report, write_patch  # noqa: E402


def test_patch_apply_check_against_locked_vllm_omni_tree():
    runtime = ROOT / "runtime" / "single_a6000_bf16" / "src" / "vllm-omni"
    patch = PORT / "patches" / "vllm_omni_h3_a6000_opt_in.patch"
    proc = subprocess.run(
        ["git", "-C", str(runtime), "apply", "--check", str(patch)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""


def test_patch_builder_copies_without_runtime_write():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "candidate.patch"
        write_patch(out)
        assert out.read_text() == (PORT / "patches" / "vllm_omni_h3_a6000_opt_in.patch").read_text()
    report = env_switch_report()
    assert "MINIMAX_H3_A6000_ENABLE_SOL_ATTN=0" in report
    assert "MINIMAX_H3_A6000_ENABLE_TELEMETRY=0" in report
    assert "MINIMAX_H3_A6000_TELEMETRY_JSON=" in report
    assert "MINIMAX_H3_A6000_ENABLE_FUSED_ADALN_MODULATE=0" in report
    assert "MINIMAX_H3_A6000_ENABLE_FUSED_ADALN_GATE=0" in report
    assert "MINIMAX_H3_A6000_ABLATION_DISABLE_INDEXED_MODULATE=0" in report
    assert "MINIMAX_H3_A6000_ABLATION_DISABLE_INDEXED_GATE=0" in report
    assert "MINIMAX_H3_A6000_ENABLE_SHADOW=0" in report
    assert "MINIMAX_H3_A6000_SHADOW_STRICT=0" in report
    assert "MINIMAX_H3_A6000_SOL_ATTN_DENSE_FIRST_STEPS=10" in report
    assert "MINIMAX_H3_A6000_SOL_ATTN_DENSE_FIRST_LAYERS=2" in report
    assert "MINIMAX_H3_A6000_SOL_ATTN_DIAGNOSTIC_MATERIALIZE=0" in report
    assert "MINIMAX_H3_A6000_SOL_ATTN_MATERIALIZE_MAX_BYTES=67108864" in report
    assert not any(line.endswith("=1") for line in report.splitlines())


def test_static_guards_no_cuda_probe_or_kernel_compile_at_package_import():
    init_text = (PORT / "src" / "minimax_h3_a6000" / "__init__.py").read_text()
    package_text = "\n".join(p.read_text() for p in (PORT / "src" / "minimax_h3_a6000").glob("*.py"))
    assert "torch.cuda.get_device_capability" not in init_text
    assert "triton.jit" not in init_text
    assert "nvidia-smi" not in package_text
    assert "cuda.compile" not in package_text
    assert "DEFAULT_ENV_SWITCHES" in package_text
    assert "torch.cuda.get_device_capability" in (PORT / "src" / "minimax_h3_a6000" / "exact_kernels.py").read_text()


def _extract_added_file(patch_text: str, path: str) -> str:
    marker = f"diff --git a/{path} b/{path}"
    lines = patch_text.splitlines()
    start = lines.index(marker)
    body: list[str] = []
    for line in lines[start + 1:]:
        if line.startswith("diff --git "):
            break
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            body.append(line[1:])
    return "\n".join(body) + "\n"


def test_patch_is_opt_in_and_default_dense():
    patch_text = (PORT / "patches" / "vllm_omni_h3_a6000_opt_in.patch").read_text()
    assert "H3_A6000_SOL_ATTN" in patch_text
    assert "MINIMAX_H3_A6000_ENABLE_SOL_ATTN" in patch_text
    assert "MINIMAX_H3_A6000_ENABLE_TRITON_CANDIDATES" in patch_text
    assert "MINIMAX_H3_A6000_ENABLE_TELEMETRY" in patch_text
    assert "MINIMAX_H3_A6000_ENABLE_FUSED_ADALN_MODULATE" in patch_text
    assert "MINIMAX_H3_A6000_ENABLE_SHADOW" in patch_text
    assert "ShadowMismatchError" in patch_text
    assert "MINIMAX_H3_A6000_ENABLE_FUSED_ADALN_GATE" in patch_text
    assert "_h3_a6000_any_env_enabled" in patch_text
    assert "get_exact_kernel_telemetry" in patch_text
    assert "env_disabled" in patch_text
    assert "sol_attn_h3_sparse_candidate" in patch_text
    assert "missing_h3_hook_metadata" in patch_text
    assert "missing_step_layer_metadata" in patch_text
    assert "missing_valid_kv_length_metadata" in patch_text
    assert "h3_denoise_step_index" in patch_text
    assert "h3_layer_index" in patch_text
    assert "MINIMAX_H3_A6000_SOL_ATTN_DENSE_FIRST_STEPS" in patch_text
    assert "MINIMAX_H3_A6000_SOL_ATTN_DENSE_FIRST_LAYERS" in patch_text
    assert "dense_first_steps" in patch_text
    assert "dense_first_layers" in patch_text
    assert "density_samples" in patch_text
    assert "static_exact_block_lower_bound" in patch_text
    assert "SOL_ATTN_CACHE" in patch_text
    assert "SOL_ATTN_DIAGNOSTIC_MATERIALIZE" in patch_text
    assert "materialize_copy_bytes" in patch_text


def test_embedded_sol_attn_backend_blocks_until_real_metadata_and_kernel_path():
    patch_text = (PORT / "patches" / "vllm_omni_h3_a6000_opt_in.patch").read_text()
    source = _extract_added_file(
        patch_text,
        "vllm_omni/diffusion/attention/backends/sol_attn_h3_a6000.py",
    )
    ast.parse(source)
    assert "getattr(attn_metadata, \"video_layout\", None)" in source
    assert "non_h3_dit_attention_prefix" in source
    assert "_is_minimax_h3_dit_attention_prefix" in source
    assert "missing_h3_hook_metadata" in source
    assert "missing_attention_metadata" in source
    assert "missing_step_layer_metadata" in source
    assert "missing_valid_kv_length_metadata" in source
    assert "dense_first_steps" in source and "dense_first_layers" in source
    assert "density_samples" in source
    assert "PackedH3Metadata" in source
    assert "sol_attn_h3_sparse_candidate" in source
    assert "overlay_package_unavailable" in source
    assert source.index("non_h3_dit_attention_prefix") < source.index("_extract_sparse_metadata(attn_metadata")


def test_transformer_patch_attaches_source_backed_h3_metadata_before_attention_call():
    patch_text = (PORT / "patches" / "vllm_omni_h3_a6000_opt_in.patch").read_text()
    assert "extra[\"h3_token_layout\"] = \"prefix_video_tail_padding\"" in patch_text
    assert "extra[\"h3_valid_kv_length\"] = used" in patch_text
    assert "extra[\"h3_layer_index\"] = self._h3_a6000_layer_index" in patch_text
    assert "extra[\"h3_denoise_step_index\"] = int(step_idx)" in patch_text
    assert "video_layout=video_layout" in patch_text
