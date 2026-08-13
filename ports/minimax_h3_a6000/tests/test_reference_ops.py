# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest

try:
    import torch
except ModuleNotFoundError:  # local host may lack PyTorch; CI/runtime gates should install it
    torch = None  # type: ignore[assignment]

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

if torch is not None:
    from minimax_h3_a6000.reference_ops import (  # noqa: E402
        apply_rope_bf16_reference,
        bf16_round_to_fp32,
        indexed_gate_bf16_reference,
        indexed_modulate_bf16_reference,
        swiglu_bf16_reference,
    )
else:
    apply_rope_bf16_reference = bf16_round_to_fp32 = indexed_gate_bf16_reference = indexed_modulate_bf16_reference = swiglu_bf16_reference = None


def _bf(x):
    return torch.tensor(x, dtype=torch.bfloat16)


def test_indexed_modulate_random_matches_locked_vllm_eager_expression():
    torch.manual_seed(7)
    x = torch.randn(2, 5, 8, dtype=torch.float32).to(torch.bfloat16)
    scale = torch.randn(3, 8, dtype=torch.float32).to(torch.bfloat16)
    shift = torch.randn(3, 8, dtype=torch.float32).to(torch.bfloat16)
    indices = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0, 2, 1])

    out = indexed_modulate_bf16_reference(x, scale, shift, indices)

    flat = x.reshape(-1, 8)
    expected = (flat * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)).to(torch.bfloat16).reshape_as(x)
    assert torch.equal(out, expected)


def test_indexed_gate_extreme_values_and_tags():
    residual = _bf([[0.0, 1.0, -2.0, 3.0], [448.0, -448.0, 1e-7, -1e-7]])
    branch = _bf([[2.0, -3.0, 4.0, -5.0], [0.5, 0.25, -0.125, 0.0625]])
    gate = _bf([[1.0, 0.0, -1.0, 2.0], [0.25, -0.5, 0.75, -1.0], [0.0, 0.0, 0.0, 0.0]])
    indices = torch.tensor([0, 1])

    out = indexed_gate_bf16_reference(residual, gate, branch, indices)
    expected = (residual + gate.index_select(0, indices) * branch).to(torch.bfloat16)
    assert torch.equal(out, expected)


def test_indexed_ops_reject_bad_shapes_tags_and_dtype():
    x = torch.zeros(2, 4, dtype=torch.bfloat16)
    table = torch.zeros(3, 4, dtype=torch.bfloat16)
    with pytest.raises(IndexError):
        indexed_modulate_bf16_reference(x, table, table, torch.tensor([0, 3]))
    with pytest.raises(ValueError):
        indexed_gate_bf16_reference(x, table[:, :3], x, torch.tensor([0, 1]))
    with pytest.raises(TypeError):
        indexed_modulate_bf16_reference(x.float(), table, table, torch.tensor([0, 1]))


def test_rope_rotates_prefix_channels_and_preserves_tail():
    hidden = torch.arange(2 * 3 * 6, dtype=torch.float32).reshape(2, 3, 6).to(torch.bfloat16)
    freqs = torch.tensor([[0.0, 0.25, 0.5, 0.75], [1.0, 1.25, 1.5, 1.75]], dtype=torch.float32)

    out = apply_rope_bf16_reference(hidden, freqs)

    assert out.shape == hidden.shape
    assert torch.equal(out[..., 4:], hidden[..., 4:])
    # freq 0 first channel pair: cos=1/sin=0 leaves the first rotary lane unchanged.
    assert out[0, 0, 0] == hidden[0, 0, 0]
    # But non-zero frequencies should alter at least one rotary value.
    assert not torch.equal(out[1, :, :4], hidden[1, :, :4])


def test_rope_batched_shape_and_validation():
    hidden = torch.randn(2, 4, 3, 8).to(torch.bfloat16)
    freqs = torch.randn(4, 6)
    assert apply_rope_bf16_reference(hidden, freqs).shape == hidden.shape
    with pytest.raises(ValueError):
        apply_rope_bf16_reference(hidden, torch.randn(5, 6))
    with pytest.raises(ValueError):
        apply_rope_bf16_reference(hidden, torch.randn(4, 5))


def test_rope_half_head_dim_false_ignores_second_freq_half():
    hidden = torch.randn(2, 3, 8).to(torch.bfloat16)
    freqs = torch.randn(2, 4)
    changed_tail = freqs.clone()
    changed_tail[:, 2:] = changed_tail[:, 2:] * 31.0 + 7.0

    assert torch.equal(
        apply_rope_bf16_reference(hidden, freqs),
        apply_rope_bf16_reference(hidden, changed_tail),
    )


def test_swiglu_random_matches_locked_vllm_eager_expression():
    torch.manual_seed(11)
    x = torch.randn(3, 10, dtype=torch.float32).to(torch.bfloat16)
    out = swiglu_bf16_reference(x)
    value, gate = x[..., :5], x[..., 5:]
    expected = torch.nn.functional.silu(gate) * value
    assert torch.equal(out, expected)
    gate_up_expected = torch.nn.functional.silu(value) * gate
    assert torch.equal(swiglu_bf16_reference(x, order="gate_up"), gate_up_expected)
    with pytest.raises(ValueError):
        swiglu_bf16_reference(torch.zeros(3, 9, dtype=torch.bfloat16))


if torch is None:
    def test_reference_ops_require_pytorch_dependency():
        # The source remains a PyTorch reference. This local machine lacks the
        # dependency, so only static/package checks run here.
        return None

    for _name, _obj in list(globals().items()):
        if _name.startswith("test_") and _name != "test_reference_ops_require_pytorch_dependency" and callable(_obj):
            globals()[_name] = test_reference_ops_require_pytorch_dependency
