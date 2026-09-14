"""Tests for whole-operator grouped transition measurements."""

import pytest

import qarp
from qarp.algorithms import GroupedTransitionHadamardTest
from qarp.blocks import (
    ComputationalBasisStateBlock,
    InterferometricMeasurementBlock,
    InterferometricStateBlock,
    SimpleBlock,
)
from qarp.engines import QarpEngine
from qarp.operators import FullyCommuting, QubitOperator


def _basis(value):
    return ComputationalBasisStateBlock(basis_state=[value])


def _run_exact(primitive):
    engine = QarpEngine(n_shots=qarp.EXACT, seed=7)
    engine.build([primitive])
    return engine.run()[0]


def test_interferometric_state_block_is_measurement_free():
    block = InterferometricStateBlock(bra=_basis(0), ket=_basis(1))
    block.build()

    assert block.n_qubits == 2
    assert block.n_cbits == 0
    assert "basis=" not in repr(block)


def test_interferometric_measurement_block_adds_qwc_measurements():
    block = InterferometricMeasurementBlock(
        bra=_basis(0),
        ket=_basis(1),
        basis={0: "X"},
    )
    block.build()

    assert block.n_qubits == 2
    assert block.n_cbits == 2
    assert "basis={0: 'X'}" in repr(block)


def test_build_groups_terms_and_creates_one_pair_per_group():
    # X0 and X1 are QWC; Z0 needs a second basis.  No per-term child
    # algorithms are created — only one block per group/quadrature.
    operator = QubitOperator("X0") + QubitOperator("X1") + QubitOperator("Z0")
    primitive = GroupedTransitionHadamardTest(
        bra=ComputationalBasisStateBlock(basis_state=[0, 0]),
        ket=ComputationalBasisStateBlock(basis_state=[0, 0]),
        operator=operator,
        real=True,
        imaginary=True,
    ).build()
    assert primitive.n_terms == 3
    assert primitive.n_groups == 2
    assert len(primitive.sub_blocks) == 4


def test_exact_real_transition_amplitude():
    # <0|X|1> = 1.
    primitive = GroupedTransitionHadamardTest(
        bra=_basis(0),
        ket=_basis(1),
        operator=QubitOperator("X0"),
        n_shots=qarp.EXACT,
    ).build()
    assert _run_exact(primitive) == pytest.approx(1.0 + 0.0j)


def test_finite_shot_real_transition_amplitude():
    primitive = GroupedTransitionHadamardTest(
        bra=_basis(0),
        ket=_basis(1),
        operator=QubitOperator("X0"),
        n_shots=4000,
        imaginary=False,
    ).build()
    engine = QarpEngine(n_shots=4000, seed=11)
    engine.build([primitive])
    assert engine.run()[0] == pytest.approx(1.0, abs=0.06)


def test_exact_imaginary_transition_amplitude():
    # S|1> = i|1>, hence <0|X S|1> = i.
    ket = SimpleBlock(1)
    ket.x(0)
    ket.s(0)
    primitive = GroupedTransitionHadamardTest(
        bra=_basis(0),
        ket=ket,
        operator=QubitOperator("X0"),
        n_shots=qarp.EXACT,
    ).build()
    assert _run_exact(primitive) == pytest.approx(1j)


def test_identity_term_uses_overlap_circuit():
    # The identity contribution is c <bra|ket>, not a classical constant.
    primitive = GroupedTransitionHadamardTest(
        bra=_basis(0),
        ket=_basis(0),
        operator=QubitOperator("", 2.5),
        n_shots=qarp.EXACT,
    ).build()
    assert primitive.n_terms == 0
    assert primitive.n_groups == 1
    assert _run_exact(primitive) == pytest.approx(2.5 + 0.0j)


def test_grouping_rejects_general_commuting_strategy():
    primitive = GroupedTransitionHadamardTest(
        bra=_basis(0),
        ket=_basis(0),
        operator=QubitOperator("X0"),
        grouping=FullyCommuting(),
    )
    with pytest.raises(ValueError, match="only qubit-wise commuting"):
        primitive.build()


def test_nonhermitian_coefficients_are_rejected():
    primitive = GroupedTransitionHadamardTest(
        bra=_basis(0),
        ket=_basis(0),
        operator=QubitOperator("X0", 1j),
    )
    with pytest.raises(ValueError, match="real operator coefficients"):
        primitive.build()


def test_wrong_result_count_is_rejected():
    primitive = GroupedTransitionHadamardTest(
        bra=_basis(0),
        ket=_basis(1),
        operator=QubitOperator("X0"),
        real=True,
        imaginary=False,
    ).build()
    with pytest.raises(ValueError, match="grouped transition results"):
        primitive.run([])
