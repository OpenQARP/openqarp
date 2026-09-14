"""Unit + integration tests for the ``SWAPTest`` primitive.

Tests the Python layer only: constructor wiring, build()-time input
validation, build() sub_blocks orchestration, and the run()
post-processing math (``2·P(bit0=0) − 1``).  A handful of integration
tests (ported from the e2e suite) drive the full build+run pipeline
through a seeded ``QarpEngine``.
"""

from types import SimpleNamespace

import pytest

from qarp.algorithms import SWAPTest, Target
from qarp.blocks import ComputationalBasisStateBlock, HnBlock, PauliBlock
from qarp.engines import QarpEngine

# ── Constructor ─────────────────────────────────────────────────────────


def test_constructor_stores_inputs():
    bra = ComputationalBasisStateBlock(basis_state=[0])
    ket = ComputationalBasisStateBlock(basis_state=[1])
    op = PauliBlock(pauli_string="X", phase=None)
    swt = SWAPTest(bra=bra, ket=ket, operator=op, n_shots=1234)

    assert swt.bra is bra
    assert swt.ket is ket
    assert swt.operator is op
    assert swt.n_shots == 1234
    assert swt.result is None


def test_constructor_target_is_overlap():
    swt = SWAPTest(
        bra=ComputationalBasisStateBlock(basis_state=[0]),
        ket=ComputationalBasisStateBlock(basis_state=[0]),
    )
    assert swt.target == Target.OVERLAP


# ── Validation (build) ──────────────────────────────────────────────────


def test_build_rejects_non_block_bra():
    swt = SWAPTest(bra="not a block", ket=ComputationalBasisStateBlock(basis_state=[0]))
    with pytest.raises(TypeError, match="bra must be a Block instance"):
        swt.build()


def test_build_rejects_non_block_ket():
    swt = SWAPTest(bra=ComputationalBasisStateBlock(basis_state=[0]), ket="not a block")
    with pytest.raises(TypeError, match="ket must be a Block instance"):
        swt.build()


def test_build_rejects_non_block_operator():
    swt = SWAPTest(
        bra=ComputationalBasisStateBlock(basis_state=[0]),
        ket=ComputationalBasisStateBlock(basis_state=[0]),
        operator="not a block",
    )
    with pytest.raises(TypeError, match="operator must be a Block instance or None"):
        swt.build()


# ── build() orchestration ───────────────────────────────────────────────


def test_build_without_operator_has_one_sub_block():
    swt = SWAPTest(
        bra=ComputationalBasisStateBlock(basis_state=[0]),
        ket=ComputationalBasisStateBlock(basis_state=[0]),
    )
    swt.build()
    assert len(swt.sub_blocks) == 1


def test_build_with_operator_has_one_sub_block():
    swt = SWAPTest(
        bra=ComputationalBasisStateBlock(basis_state=[0]),
        ket=ComputationalBasisStateBlock(basis_state=[0]),
        operator=PauliBlock(pauli_string="X", phase=None),
    )
    swt.build()
    assert len(swt.sub_blocks) == 1


# ── run() post-processing math (synthetic results) ──────────────────────


def test_run_half_overlap():
    """counts={0:750, 1:250} → 2·0.75 − 1 = 0.5."""
    swt = SWAPTest(
        bra=ComputationalBasisStateBlock(basis_state=[0]),
        ket=ComputationalBasisStateBlock(basis_state=[0]),
    )
    sr = SimpleNamespace(counts={0: 750, 1: 250}, n_shots=1000, n_qubits=1)
    assert swt.run([sr]) == pytest.approx(0.5)
    assert swt.result == pytest.approx(0.5)


def test_run_full_overlap():
    """All shots in bit0=0 → 2·1.0 − 1 = 1.0."""
    swt = SWAPTest(
        bra=ComputationalBasisStateBlock(basis_state=[0]),
        ket=ComputationalBasisStateBlock(basis_state=[0]),
    )
    sr = SimpleNamespace(counts={0: 1000}, n_shots=1000, n_qubits=1)
    assert swt.run([sr]) == pytest.approx(1.0)


def test_run_zero_overlap():
    """50/50 split → 2·0.5 − 1 = 0.0."""
    swt = SWAPTest(
        bra=ComputationalBasisStateBlock(basis_state=[0]),
        ket=ComputationalBasisStateBlock(basis_state=[0]),
    )
    sr = SimpleNamespace(counts={0: 500, 1: 500}, n_shots=1000, n_qubits=1)
    assert swt.run([sr]) == pytest.approx(0.0)


# ── __repr__ ─────────────────────────────────────────────────────────────


def test_repr_contains_name_and_shots():
    swt = SWAPTest(
        bra=ComputationalBasisStateBlock(basis_state=[0]),
        ket=ComputationalBasisStateBlock(basis_state=[0]),
        n_shots=4242,
    )
    r = repr(swt)
    assert "SWAPTest" in r
    assert "4242" in r


# ── Integration (full build+run via QarpEngine) ─────────────────────────


def test_integration_same_state_overlap_one():
    state = [1, 0]
    swt = SWAPTest(
        bra=ComputationalBasisStateBlock(basis_state=state),
        ket=ComputationalBasisStateBlock(basis_state=state),
        n_shots=1000,
    )
    swt.build()
    eng = QarpEngine(n_shots=1000, seed=42)
    eng.build([swt])
    assert eng.run()[0] == pytest.approx(1.0)


def test_integration_orthogonal_states_overlap_zero():
    swt = SWAPTest(
        bra=ComputationalBasisStateBlock(basis_state=[0, 1]),
        ket=ComputationalBasisStateBlock(basis_state=[1, 0]),
        n_shots=2000,
    )
    swt.build()
    eng = QarpEngine(n_shots=2000, seed=42)
    eng.build([swt])
    assert abs(eng.run()[0]) < 0.05


@pytest.mark.parametrize("n", [2, 4])
def test_integration_zero_vs_hn_overlap_is_one_over_2n(n):
    """|0…0⟩ vs Hn|0…0⟩ ⇒ |⟨0…0|·⟩|² = 1/2^n."""
    swt = SWAPTest(
        bra=ComputationalBasisStateBlock(basis_state=[0] * n),
        ket=HnBlock(n),
        n_shots=8000,
    )
    swt.build()
    eng = QarpEngine(n_shots=8000, seed=42)
    eng.build([swt])
    out = eng.run()[0]
    expected = 1.0 / 2**n
    assert abs(out - expected) < 0.03, f"n={n}: got {out}, expected {expected}"


def test_integration_operator_path_x_on_zero_is_orthogonal():
    """SWAPTest of |0⟩ against X·|0⟩ = |1⟩ ⇒ overlap² = 0."""
    swt = SWAPTest(
        bra=ComputationalBasisStateBlock(basis_state=[0]),
        ket=ComputationalBasisStateBlock(basis_state=[0]),
        operator=PauliBlock(pauli_string="X", phase=None),
        n_shots=2000,
    )
    swt.build()
    eng = QarpEngine(n_shots=2000, seed=42)
    eng.build([swt])
    assert abs(eng.run()[0]) < 0.05
