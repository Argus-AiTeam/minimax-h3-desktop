# SPDX-License-Identifier: Apache-2.0
"""Static checks for external GPU gates.

There are deliberately no pytest tests that pretend to run GPU correctness.
The real gates are the standalone scripts in the port root and must be invoked
by the outer operator with one visible A6000 SM86.
"""

from pathlib import Path

PORT = Path(__file__).resolve().parents[1]


def test_gpu_correctness_and_bench_are_external_scripts_not_pytest_placeholders():
    correctness = (PORT / "gpu_exact_kernel_test.py").read_text(encoding="utf-8")
    bench = (PORT / "gpu_exact_kernel_bench.py").read_text(encoding="utf-8")
    assert "validated_single_a6000_sm86" in correctness
    assert "coverage_tags" in correctness
    assert "explicit_extreme_values_per_op" in correctness
    assert "compile_status" in correctness
    assert "mismatch" in correctness
    assert "kernel_candidates_only_not_h3_e2e" in bench
    assert "pytest.mark.skip" not in correctness + bench
