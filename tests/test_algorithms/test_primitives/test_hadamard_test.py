"""Python-layer tests for the ``HadamardTest`` primitive.

Covers the Python responsibilities only: constructor/validation, ``build()``
orchestration (sub_blocks counts + structure, target dispatch), ``run()``
post-processing math against synthetic ``SamplingResult`` namespaces, the
``expectation_type`` property and ``__repr__``.  A small set of integration
tests exercises the full build+run pipeline through ``QarpEngine`` for the
canonical analytic values.  C++ numerics are not re-tested here.
"""

from types import SimpleNamespace

import pytest

from qarp.algorithms import HadamardTest
from qarp.blocks import ComputationalBasisStateBlock, HnBlock, PauliBlock
from qarp.engines import QarpEngine


def _sr(counts, n_shots):
    """Synthetic stand-in for ``qx.SamplingResult`` (reads .counts / .n_shots)."""
    return SimpleNamespace(counts=counts, n_shots=n_shots)


# ── Constructor / validation ────────────────────────────────────────────


def test_rejects_neither_real_nor_imaginary():
    with pytest.raises(ValueError, match="At least one of real or imaginary must be True"):
        HadamardTest(
            ket=ComputationalBasisStateBlock(basis_state=[0]),
            real=False,
            imaginary=False,
        )


def test_validate_ket_must_be_block():
    with pytest.raises(TypeError, match="ket must be a Block instance"):
        HadamardTest(ket="not a block").build()


def test_validate_bra_must_be_block_or_none():
    with pytest.raises(TypeError, match="bra must be a Block instance or None"):
        HadamardTest(
            ket=ComputationalBasisStateBlock(basis_state=[0]),
            bra="not a block",
        ).build()


def test_validate_operator_must_be_block_or_none():
    with pytest.raises(TypeError, match="operator must be a Block instance or None"):
        HadamardTest(
            ket=ComputationalBasisStateBlock(basis_state=[0]),
            operator="not a block",
        ).build()


# ── build() structure ───────────────────────────────────────────────────


def test_build_real_only_one_sub_block():
    ht = HadamardTest(
        ket=ComputationalBasisStateBlock(basis_state=[0]),
        operator=PauliBlock(pauli_string="Z", phase=None),
        real=True,
        imaginary=False,
    )
    ht.build()
    assert len(ht.sub_blocks) == 1
    assert ht.sub_blocks[0].n_cbits == 1


def test_build_imaginary_only_one_sub_block():
    ht = HadamardTest(
        ket=ComputationalBasisStateBlock(basis_state=[0]),
        operator=PauliBlock(pauli_string="Z", phase=None),
        real=False,
        imaginary=True,
    )
    ht.build()
    assert len(ht.sub_blocks) == 1
    assert ht.sub_blocks[0].n_cbits == 1


def test_build_both_two_sub_blocks():
    ht = HadamardTest(
        ket=ComputationalBasisStateBlock(basis_state=[0]),
        operator=PauliBlock(pauli_string="Z", phase=None),
        real=True,
        imaginary=True,
    )
    ht.build()
    assert len(ht.sub_blocks) == 2
    assert all(b.n_cbits == 1 for b in ht.sub_blocks)


def test_build_expectation_path_populates_sub_blocks():
    """ket + operator, bra None → EXPECTATION_VALUE; build completes."""
    ht = HadamardTest(
        ket=ComputationalBasisStateBlock(basis_state=[0]),
        operator=PauliBlock(pauli_string="Z", phase=None),
    )
    ht.build()
    from qarp.algorithms import Target

    assert ht.target == Target.EXPECTATION_VALUE
    assert len(ht.sub_blocks) == 2


def test_build_overlap_path_completes():
    """bra != ket, no operator → OVERLAP; build completes."""
    ht = HadamardTest(
        bra=ComputationalBasisStateBlock(basis_state=[0]),
        ket=HnBlock(1),
    )
    ht.build()
    from qarp.algorithms import Target

    assert ht.target == Target.OVERLAP
    assert len(ht.sub_blocks) == 2


# ── run() unit (synthetic results) ───────────────────────────────────────


def test_run_real_only_returns_float():
    ht = HadamardTest(
        ket=ComputationalBasisStateBlock(basis_state=[0]),
        operator=PauliBlock(pauli_string="Z", phase=None),
        real=True,
        imaginary=False,
    )
    ht.build()
    out = ht.run([_sr({0: 850, 1: 150}, 1000)])
    # real_part = 2 * P(bit0==0) - 1 = 2*0.85 - 1 = 0.7
    assert out == pytest.approx(0.7)
    assert ht.result_real == pytest.approx(0.7)
    assert ht.result_imaginary is None
    assert ht.result is None


def test_run_imaginary_only_returns_float():
    ht = HadamardTest(
        ket=ComputationalBasisStateBlock(basis_state=[0]),
        operator=PauliBlock(pauli_string="Z", phase=None),
        real=False,
        imaginary=True,
    )
    ht.build()
    out = ht.run([_sr({0: 850, 1: 150}, 1000)])
    assert out == pytest.approx(0.7)
    assert ht.result_imaginary == pytest.approx(0.7)
    assert ht.result_real is None
    assert ht.result is None


def test_run_both_returns_complex():
    ht = HadamardTest(
        ket=ComputationalBasisStateBlock(basis_state=[0]),
        operator=PauliBlock(pauli_string="Z", phase=None),
        real=True,
        imaginary=True,
    )
    ht.build()
    # real block: P(0)=0.85 → 0.7 ; imaginary block: P(0)=0.6 → 0.2
    out = ht.run([_sr({0: 850, 1: 150}, 1000), _sr({0: 600, 1: 400}, 1000)])
    assert out == pytest.approx(complex(0.7, 0.2))
    assert ht.result_real == pytest.approx(0.7)
    assert ht.result_imaginary == pytest.approx(0.2)
    assert ht.result == out


# ── expectation_type / repr ──────────────────────────────────────────────


def test_expectation_type_complex():
    ht = HadamardTest(
        ket=ComputationalBasisStateBlock(basis_state=[0]),
        operator=PauliBlock(pauli_string="Z", phase=None),
        real=True,
        imaginary=True,
    )
    assert ht.expectation_type == "complex"


def test_expectation_type_real():
    ht = HadamardTest(
        ket=ComputationalBasisStateBlock(basis_state=[0]),
        operator=PauliBlock(pauli_string="Z", phase=None),
        real=True,
        imaginary=False,
    )
    assert ht.expectation_type == "real"


def test_expectation_type_imaginary():
    ht = HadamardTest(
        ket=ComputationalBasisStateBlock(basis_state=[0]),
        operator=PauliBlock(pauli_string="Z", phase=None),
        real=False,
        imaginary=True,
    )
    assert ht.expectation_type == "imaginary"


def test_repr_contains_class_name():
    ht = HadamardTest(
        ket=ComputationalBasisStateBlock(basis_state=[0]),
        operator=PauliBlock(pauli_string="Z", phase=None),
    )
    assert "HadamardTest" in repr(ht)


# ── Integration (full build+run via QarpEngine) ──────────────────────────


def test_integration_z_on_zero():
    """⟨0|Z|0⟩ = 1 (real part)."""
    ht = HadamardTest(
        ket=ComputationalBasisStateBlock(basis_state=[0]),
        operator=PauliBlock(pauli_string="Z", phase=None),
        real=True,
        imaginary=False,
        n_shots=2000,
    )
    ht.build()
    eng = QarpEngine(n_shots=2000, seed=42)
    eng.build([ht])
    assert eng.run()[0] == pytest.approx(1.0)


def test_integration_z_on_one():
    """⟨1|Z|1⟩ = -1 (real part)."""
    ht = HadamardTest(
        ket=ComputationalBasisStateBlock(basis_state=[1]),
        operator=PauliBlock(pauli_string="Z", phase=None),
        real=True,
        imaginary=False,
        n_shots=2000,
    )
    ht.build()
    eng = QarpEngine(n_shots=2000, seed=42)
    eng.build([ht])
    assert eng.run()[0] == pytest.approx(-1.0)


def test_integration_complex_x_on_plus():
    """⟨+|X|+⟩ = 1 + 0i."""
    ht = HadamardTest(
        ket=HnBlock(1),
        operator=PauliBlock(pauli_string="X", phase=None),
        real=True,
        imaginary=True,
        n_shots=2000,
    )
    ht.build()
    eng = QarpEngine(n_shots=2000, seed=42)
    eng.build([ht])
    out = eng.run()[0]
    assert abs(out.real - 1.0) < 0.05
    assert abs(out.imag) < 0.05
