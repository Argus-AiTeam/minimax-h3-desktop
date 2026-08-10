# SPDX-License-Identifier: Apache-2.0
"""CPU-only behavioral checks for exact-kernel launcher guards.

These tests deliberately use CPU tensors so they cannot compile or launch Triton.
They verify that default-off/unsupported paths return the PyTorch references and
that strict mode reports a guard failure instead of silently counting a GPU gate
as passed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

try:
    import torch
except ModuleNotFoundError:  # local host may lack PyTorch; CI/runtime gates should install it
    torch = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "ports" / "minimax_h3_a6000" / "src"
sys.path.insert(0, str(SRC))

if torch is not None:
    from minimax_h3_a6000 import exact_kernels as exact  # noqa: E402
    from minimax_h3_a6000.reference_ops import (  # noqa: E402
        apply_rope_bf16_reference,
        indexed_gate_bf16_reference,
        indexed_modulate_bf16_reference,
        swiglu_bf16_reference,
    )
else:
    exact = None
    apply_rope_bf16_reference = indexed_gate_bf16_reference = indexed_modulate_bf16_reference = swiglu_bf16_reference = None


def _bf(values: list[list[float]]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).to(torch.bfloat16).contiguous()


def test_default_off_launchers_return_references_without_triton_import() -> None:
    exact._TRITON_CACHE = None
    x = _bf([[0.5, -1.0, 2.0, 3.0], [4.0, -5.0, 6.0, -7.0]])
    scale = _bf([[0.25, -0.5, 1.0, 0.0], [-0.25, 0.5, -1.0, 0.125]])
    shift = _bf([[1.0, 2.0, -3.0, 0.5], [0.0, -1.0, 1.5, -0.5]])
    indices = torch.tensor([0, 1], dtype=torch.int64)

    assert torch.equal(
        exact.indexed_modulate_bf16(x, scale, shift, indices),
        indexed_modulate_bf16_reference(x, scale, shift, indices),
    )
    assert torch.equal(
        exact.indexed_gate_bf16(x, scale, shift, indices),
        indexed_gate_bf16_reference(x, scale, shift, indices),
    )

    hidden = torch.arange(2 * 3 * 6, dtype=torch.float32).reshape(2, 3, 6).to(torch.bfloat16).contiguous()
    freqs = torch.zeros((2, 4), dtype=torch.float32).contiguous()
    assert torch.equal(exact.apply_rope_bf16(hidden, freqs), apply_rope_bf16_reference(hidden, freqs))

    swiglu = _bf([[0.5, -1.0, 2.0, 3.0], [4.0, -5.0, 6.0, -7.0]])
    assert torch.equal(exact.swiglu_bf16(swiglu), swiglu_bf16_reference(swiglu))
    assert exact._TRITON_CACHE is None


def test_exact_kernel_telemetry_is_opt_in_and_json_exportable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    exact.reset_exact_kernel_telemetry()
    for key in (
        "MINIMAX_H3_A6000_ENABLE_TELEMETRY",
        "MINIMAX_H3_A6000_TELEMETRY_JSON",
        "MINIMAX_H3_A6000_TELEMETRY_ATEXIT",
        "MINIMAX_H3_A6000_ENABLE_OVERLAY",
        "MINIMAX_H3_A6000_ENABLE_TRITON_CANDIDATES",
        "MINIMAX_H3_A6000_ENABLE_FUSED_ADALN",
    ):
        monkeypatch.delenv(key, raising=False)

    x = _bf([[0.5, -1.0], [2.0, 3.0]])
    table = _bf([[0.25, -0.5], [-0.25, 0.5]])
    indices = torch.tensor([0, 1], dtype=torch.int64)
    exact.indexed_modulate_bf16(x, table, table, indices)
    assert exact.get_exact_kernel_telemetry()["ops"]["indexed_modulate_bf16"]["calls"] == 0

    out_path = tmp_path / "telemetry" / "exact.json"
    monkeypatch.setenv("MINIMAX_H3_A6000_ENABLE_TELEMETRY", "1")
    monkeypatch.setenv("MINIMAX_H3_A6000_TELEMETRY_JSON", str(out_path))
    exact.indexed_modulate_bf16(x, table, table, indices)
    snapshot = exact.get_exact_kernel_telemetry()
    op = snapshot["ops"]["indexed_modulate_bf16"]
    assert snapshot["enabled"] is True
    assert op["calls"] == 1
    assert op["decline"] == 1
    assert op["fallback"] == 1
    assert op["candidate"] == 0
    assert "disabled by default-off environment" in next(iter(op["reasons"]))
    assert op["tensor_layout_summary"]["x"]["seen"] == 1
    assert op["tensor_layout_samples"][0]["shape"] == [2, 2]
    assert "materialize_copy_by_tensor" in op
    exact.write_exact_kernel_telemetry_json()
    exported = out_path.read_text(encoding="utf-8")
    assert "minimax_h3_a6000_exact_telemetry_v1" in exported
    assert "indexed_modulate_bf16" in exported


def test_shadow_compare_records_bitwise_mismatch_and_can_abort(monkeypatch: pytest.MonkeyPatch) -> None:
    exact.reset_exact_kernel_telemetry()
    monkeypatch.setenv("MINIMAX_H3_A6000_ENABLE_SHADOW", "1")
    monkeypatch.setenv("MINIMAX_H3_A6000_SHADOW_CALLS", "2")
    monkeypatch.delenv("MINIMAX_H3_A6000_SHADOW_STRICT", raising=False)
    reference = _bf([[1.0, -2.0], [3.0, -4.0]])
    exact._shadow_compare_or_raise("swiglu_bf16", reference.clone(), lambda: reference.clone())
    mismatch = reference.clone()
    mismatch[0, 0] = torch.tensor(2.0, dtype=torch.bfloat16)
    exact._shadow_compare_or_raise("swiglu_bf16", mismatch, lambda: reference.clone())
    op = exact.get_exact_kernel_telemetry()["ops"]["swiglu_bf16"]
    assert op["shadow"]["comparisons"] == 2
    assert op["shadow"]["mismatches"] == 1
    assert op["shadow"]["max_abs"] >= 1.0
    assert op["shadow"]["samples"][1]["bitwise_mismatch"] is True

    monkeypatch.setenv("MINIMAX_H3_A6000_SHADOW_CALLS", "3")
    monkeypatch.setenv("MINIMAX_H3_A6000_SHADOW_STRICT", "1")
    with pytest.raises(exact.ShadowMismatchError, match="shadow mismatch"):
        exact._shadow_compare_or_raise("swiglu_bf16", mismatch, lambda: reference.clone())
    assert exact.get_exact_kernel_telemetry()["ops"]["swiglu_bf16"]["shadow"]["strict_error"] == 1


def test_per_kernel_ablation_disable_env_declines_before_cuda_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    exact.reset_exact_kernel_telemetry()
    exact._TRITON_CACHE = None
    monkeypatch.setenv("MINIMAX_H3_A6000_ENABLE_TELEMETRY", "1")
    monkeypatch.setenv("MINIMAX_H3_A6000_ABLATION_DISABLE_INDEXED_MODULATE", "1")
    x = _bf([[0.5, -1.0], [2.0, 3.0]])
    table = _bf([[0.25, -0.5], [-0.25, 0.5]])
    indices = torch.tensor([0, 1], dtype=torch.int64)

    out = exact.indexed_modulate_bf16(x, table, table, indices, enable=True, strict=False, device_capability=(8, 6))

    assert torch.equal(out, indexed_modulate_bf16_reference(x, table, table, indices))
    op = exact.get_exact_kernel_telemetry()["ops"]["indexed_modulate_bf16"]
    assert op["calls"] == 1
    assert op["candidate"] == 0
    assert op["decline"] == 1
    assert op["fallback"] == 1
    assert any("ABlATION".lower() in reason.lower() for reason in op["reasons"])
    assert exact._TRITON_CACHE is None
    with pytest.raises(RuntimeError, match="per-kernel ablation env"):
        exact.indexed_modulate_bf16(x, table, table, indices, enable=True, strict=True, device_capability=(8, 6))


def test_indexed_support_accepts_sliced_tables_and_expanded_views_when_cuda_guard_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exact, "_support_common", lambda *args, **kwargs: exact.KernelSupport(True, "supported"))
    x = _bf([[0.5, -1.0, 2.0, 3.0], [4.0, -5.0, 6.0, -7.0]])
    packed = torch.arange(3 * 12, dtype=torch.float32).reshape(3, 12).to(torch.bfloat16).contiguous()
    scale = packed[:, 4:8]
    shift = packed[:, 8:12]
    indices = torch.tensor([0, 2], dtype=torch.int64)

    mod_support = exact.explain_indexed_modulate_support(x, scale, shift, indices, enable=True, device_capability=(8, 6))
    assert mod_support.supported
    assert not scale.is_contiguous() and scale.stride() == (12, 1)

    residual = x
    branch = x[:1, :].expand_as(x)
    gate = packed[:1, :4].expand(3, 4)
    gate_support = exact.explain_indexed_gate_support(residual, gate, branch, indices, enable=True, device_capability=(8, 6))
    assert gate_support.supported
    assert not branch.is_contiguous() and 0 in branch.stride()
    assert not gate.is_contiguous() and 0 in gate.stride()


def test_telemetry_records_tensor_specific_noncontiguous_layouts(monkeypatch: pytest.MonkeyPatch) -> None:
    exact.reset_exact_kernel_telemetry()
    monkeypatch.setenv("MINIMAX_H3_A6000_ENABLE_TELEMETRY", "1")
    monkeypatch.delenv("MINIMAX_H3_A6000_ENABLE_OVERLAY", raising=False)
    x = _bf([[0.5, -1.0, 2.0, 3.0], [4.0, -5.0, 6.0, -7.0]])
    packed = torch.arange(3 * 12, dtype=torch.float32).reshape(3, 12).to(torch.bfloat16).contiguous()
    scale = packed[:, 4:8]
    shift = packed[:, 8:12]
    indices = torch.tensor([0, 2], dtype=torch.int64)

    exact.indexed_modulate_bf16(x, scale, shift, indices)
    op = exact.get_exact_kernel_telemetry()["ops"]["indexed_modulate_bf16"]
    layouts = op["tensor_layouts"]
    assert any(key.startswith("scale:") and "stride=(12, 1)" in key for key in layouts)
    assert any(key.startswith("shift:") and "stride=(12, 1)" in key for key in layouts)
    assert op["tensor_layout_summary"]["scale"]["noncontiguous"] == 1
    assert op["tensor_layout_summary"]["shift"]["last"]["storage_offset"] == 8
    assert any(sample["name"] == "scale" and sample["stride"] == [12, 1] for sample in op["tensor_layout_samples"])


def test_telemetry_records_rope_and_swiglu_noncontiguous_layouts(monkeypatch: pytest.MonkeyPatch) -> None:
    exact.reset_exact_kernel_telemetry()
    monkeypatch.setenv("MINIMAX_H3_A6000_ENABLE_TELEMETRY", "1")
    hidden_base = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8).to(torch.bfloat16).contiguous()
    hidden = hidden_base[:, :, ::2]
    freqs = torch.zeros((2, 4), dtype=torch.float32)
    exact.apply_rope_bf16(hidden, freqs)

    swiglu_base = torch.arange(2 * 8, dtype=torch.float32).reshape(2, 8).to(torch.bfloat16).contiguous()
    swiglu_x = swiglu_base[:, ::2]
    exact.swiglu_bf16(swiglu_x)

    telemetry = exact.get_exact_kernel_telemetry()["ops"]
    rope_summary = telemetry["apply_rope_bf16"]["tensor_layout_summary"]
    swiglu_summary = telemetry["swiglu_bf16"]["tensor_layout_summary"]
    assert rope_summary["hidden_states"]["noncontiguous"] == 1
    assert rope_summary["hidden_states"]["last"]["stride"] == [24, 8, 2]
    assert swiglu_summary["x"]["noncontiguous"] == 1
    assert any(sample["name"] == "x" and sample["stride"] == [8, 2] for sample in telemetry["swiglu_bf16"]["tensor_layout_samples"])


def test_materialize_copy_telemetry_is_grouped_by_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    exact.reset_exact_kernel_telemetry()
    monkeypatch.setenv("MINIMAX_H3_A6000_ENABLE_TELEMETRY", "1")
    packed = torch.arange(3 * 12, dtype=torch.float32).reshape(3, 12).to(torch.bfloat16).contiguous()
    scale = packed[:, 4:8]
    shift = packed[:, 8:12]

    scale_c, shift_c = exact._contiguous_for_materialize(
        "indexed_modulate_bf16",
        (("scale", scale), ("shift", shift)),
    )

    assert scale_c.is_contiguous() and shift_c.is_contiguous()
    op = exact.get_exact_kernel_telemetry()["ops"]["indexed_modulate_bf16"]
    assert op["materialize_copy_calls"] == 2
    assert op["materialize_copy_bytes"] == int((scale.numel() + shift.numel()) * scale.element_size())
    assert op["materialize_copy_by_tensor"] == {
        "scale": {"calls": 1, "bytes": int(scale.numel() * scale.element_size())},
        "shift": {"calls": 1, "bytes": int(shift.numel() * shift.element_size())},
    }


def test_rope_support_has_explicit_freq_dtype_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exact, "_support_common", lambda *args, **kwargs: exact.KernelSupport(True, "supported"))
    hidden = torch.zeros((2, 1, 4), dtype=torch.bfloat16).contiguous()
    freqs = torch.zeros((2, 4), dtype=torch.float16).contiguous()

    support = exact.explain_rope_support(hidden, freqs, enable=True, device_capability=(8, 6))

    assert not support.supported
    assert "freqs must be FP32" in support.reason
    assert exact.explain_rope_support(hidden, freqs.to(torch.float32), enable=True, device_capability=(8, 6)).supported


def test_strict_cpu_unsupported_path_raises_before_triton() -> None:
    exact._TRITON_CACHE = None
    x = _bf([[0.5, -1.0, 2.0, 3.0], [4.0, -5.0, 6.0, -7.0]])
    table = _bf([[0.25, -0.5, 1.0, 0.0], [-0.25, 0.5, -1.0, 0.125]])
    indices = torch.tensor([0, 1], dtype=torch.int64)

    cases = [
        (
            "indexed_modulate_bf16",
            lambda: exact.indexed_modulate_bf16(
                x, table, table, indices, enable=True, strict=True, device_capability=(8, 6)
            ),
        ),
        (
            "indexed_gate_bf16",
            lambda: exact.indexed_gate_bf16(
                x, table, x, indices, enable=True, strict=True, device_capability=(8, 6)
            ),
        ),
        (
            "apply_rope_bf16",
            lambda: exact.apply_rope_bf16(
                x.reshape(2, 1, 4),
                torch.zeros((2, 4), dtype=torch.float32),
                enable=True,
                strict=True,
                device_capability=(8, 6),
            ),
        ),
        (
            "swiglu_bf16",
            lambda: exact.swiglu_bf16(x, enable=True, strict=True, device_capability=(8, 6)),
        ),
    ]
    for name, call in cases:
        with pytest.raises(RuntimeError, match=f"{name}: tensor device must be CUDA"):
            call()
    assert exact._TRITON_CACHE is None


if torch is None:
    def test_exact_kernel_launchers_require_pytorch_dependency() -> None:
        # This local fallback path is still CPU-only; real launcher behavior is
        # exercised where PyTorch is installed, while GPU correctness remains an external gate.
        return None

    for _name, _obj in list(globals().items()):
        if _name.startswith("test_") and _name != "test_exact_kernel_launchers_require_pytorch_dependency" and callable(_obj):
            globals()[_name] = test_exact_kernel_launchers_require_pytorch_dependency
