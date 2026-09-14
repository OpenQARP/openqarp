"""Python-layer tests for the ``TermwiseHadamardTest`` primitive.

``TermwiseHadamardTest`` is a composition wrapper: one child ``HadamardTest``
per operator term, with QubitOperator operands expanded into per-term
``PauliBlock``s carrying per-term coefficients.  These tests cover the Python
responsibilities only: constructor normalization, ``build()`` validation +
orchestration (sub_algorithms / coefficients / sub_blocks), the per-term
weighted ``run()`` math against synthetic ``SamplingResult`` namespaces,
properties/accessors and ``__repr__``.  One integration test exercises the
full pipeline through ``QarpEngine``.  C++ numerics are not re-tested here.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import qarpx as qx
from qarp.algorithms import HadamardTest, TermwiseHadamardTest
from qarp.blocks import ComputationalBasisStateBlock, PauliBlock, SimpleBlock
from qarp.engines import QarpEngine
from qarp.operators import QubitOperator


def _sr(counts, n_shots):
    """Synthetic stand-in for ``qx.SamplingResult`` (reads .counts / .n_shots)."""
    return SimpleNamespace(counts=counts, n_shots=n_shots)


def _ket():
    return ComputationalBasisStateBlock(basis_state=[0])


# ── Constructor ──────────────────────────────────────────────────────────


def test_rejects_neither_real_nor_imaginary():
    with pytest.raises(ValueError, match="At least one of real or imaginary must be True"):
        TermwiseHadamardTest(
            operator=PauliBlock(pauli_string="Z", phase=None),
            ket=_ket(),
            real=False,
            imaginary=False,
        )


def test_single_block_operator_normalized_to_list():
    op = PauliBlock(pauli_string="Z", phase=None)
    th = TermwiseHadamardTest(operator=op, ket=_ket())
    assert isinstance(th.operator, list)
    assert th.operator == [op]


# ── Validation (build()) ─────────────────────────────────────────────────


def test_validate_operator_required():
    with pytest.raises(ValueError, match="operator must be provided"):
        TermwiseHadamardTest(operator=None, ket=_ket()).build()


def test_validate_operator_wrong_type():
    with pytest.raises(TypeError, match="operator must be Block, list of Blocks, or QubitOperator"):
        TermwiseHadamardTest(operator=5, ket=_ket()).build()


def test_validate_empty_list():
    with pytest.raises(ValueError, match="operator list cannot be empty"):
        TermwiseHadamardTest(operator=[], ket=_ket()).build()


def test_validate_list_element_non_block():
    op = PauliBlock(pauli_string="Z", phase=None)
    with pytest.raises(TypeError, match=r"operator\[1\] must be a Block instance"):
        TermwiseHadamardTest(operator=[op, "not a block"], ket=_ket()).build()


def test_validate_qubit_operator_rejects_coefficients():
    H = QubitOperator("Z0", 1.0)
    with pytest.raises(
        ValueError,
        match="coefficients should not be provided when operator is a QubitOperator",
    ):
        TermwiseHadamardTest(operator=H, ket=_ket(), coefficients=[1.0]).build()


def test_validate_list_coefficients_length_mismatch():
    op = PauliBlock(pauli_string="Z", phase=None)
    with pytest.raises(ValueError, match="coefficients must match the length of operator list"):
        TermwiseHadamardTest(operator=[op, op], ket=_ket(), coefficients=[1.0]).build()


# ── build() orchestration ────────────────────────────────────────────────


def test_build_qubit_operator_two_terms_real_only():
    """0.5*X0 + 0.3*Z0 on |0> → 2 children, coeffs {0.3, 0.5}, 2 sub_blocks."""
    H = QubitOperator("X0", 0.5) + QubitOperator("Z0", 0.3)
    th = TermwiseHadamardTest(operator=H, ket=_ket(), real=True, imaginary=False)
    th.build()
    assert th.n_sub_algorithms == 2
    assert all(complex(c).imag == 0 for c in th.coefficients)
    assert sorted(complex(c).real for c in th.coefficients) == [0.3, 0.5]
    # real-only → 1 sub_block per child → 2 total.
    assert len(th.sub_blocks) == 2


def test_build_qubit_operator_two_terms_both_doubles_sub_blocks():
    H = QubitOperator("X0", 0.5) + QubitOperator("Z0", 0.3)
    th = TermwiseHadamardTest(operator=H, ket=_ket(), real=True, imaginary=True)
    th.build()
    assert th.n_sub_algorithms == 2
    # real+imaginary → 2 sub_blocks per child → 4 total.
    assert len(th.sub_blocks) == 4
    assert all(isinstance(c, HadamardTest) for c in th.sub_algorithms)


# ── run() unit (synthetic results) ───────────────────────────────────────


def test_run_weighted_sum():
    """Two terms, coeffs [0.5, 0.3], children → 1.0 and -1.0 → [0.5, -0.3], sum 0.2."""
    op = PauliBlock(pauli_string="Z", phase=None)
    th = TermwiseHadamardTest(
        operator=[op, op],
        ket=_ket(),
        coefficients=[0.5, 0.3],
        real=True,
        imaginary=False,
    )
    th.build()
    # child 0 returns 1.0: 2*P(0)-1 = 1 → all counts in outcome 0.
    # child 1 returns -1.0: 2*P(0)-1 = -1 → all counts in outcome 1 (ancilla=1).
    results = [_sr({0: 1000}, 1000), _sr({1: 1000}, 1000)]
    out = th.run(results)
    assert th.result_list == pytest.approx([0.5, -0.3])
    assert th.result_sum == pytest.approx(0.2)
    assert out == pytest.approx(0.2)


def test_run_before_build_raises():
    op = PauliBlock(pauli_string="Z", phase=None)
    th = TermwiseHadamardTest(operator=op, ket=_ket())
    with pytest.raises(ValueError, match="must be built before running"):
        th.run([])


# ── Properties / accessors ───────────────────────────────────────────────


def test_len_and_n_operators():
    op = PauliBlock(pauli_string="Z", phase=None)
    th = TermwiseHadamardTest(operator=[op, op], ket=_ket())
    assert len(th) == 2
    assert th.n_operators == 2


def test_n_sub_algorithms_zero_before_build():
    op = PauliBlock(pauli_string="Z", phase=None)
    th = TermwiseHadamardTest(operator=op, ket=_ket())
    assert th.n_sub_algorithms == 0


def test_expectation_type_complex():
    op = PauliBlock(pauli_string="Z", phase=None)
    th = TermwiseHadamardTest(operator=op, ket=_ket(), real=True, imaginary=True)
    assert th.expectation_type == "complex"


def test_expectation_type_real():
    op = PauliBlock(pauli_string="Z", phase=None)
    th = TermwiseHadamardTest(operator=op, ket=_ket(), real=True, imaginary=False)
    assert th.expectation_type == "real"


def test_expectation_type_imaginary():
    op = PauliBlock(pauli_string="Z", phase=None)
    th = TermwiseHadamardTest(operator=op, ket=_ket(), real=False, imaginary=True)
    assert th.expectation_type == "imaginary"


def test_get_hadamard_test_returns_child():
    op = PauliBlock(pauli_string="Z", phase=None)
    th = TermwiseHadamardTest(operator=[op, op], ket=_ket())
    th.build()
    child = th.get_hadamard_test(0)
    assert isinstance(child, HadamardTest)
    assert child is th.sub_algorithms[0]


def test_get_hadamard_test_before_build_raises():
    op = PauliBlock(pauli_string="Z", phase=None)
    th = TermwiseHadamardTest(operator=op, ket=_ket())
    with pytest.raises(ValueError, match="must be built first"):
        th.get_hadamard_test(0)


def test_repr_contains_class_name():
    op = PauliBlock(pauli_string="Z", phase=None)
    th = TermwiseHadamardTest(operator=op, ket=_ket())
    assert "TermwiseHadamardTest" in repr(th)


# ── Integration (full build+run via QarpEngine) ──────────────────────────


def test_integration_qubit_operator():
    """H = 0.5*X0 + 0.3*Z0 on |0> → ⟨0|H|0⟩ real part ≈ 0.3."""
    H = QubitOperator("X0", 0.5) + QubitOperator("Z0", 0.3)
    th = TermwiseHadamardTest(operator=H, ket=_ket(), real=True, imaginary=False, n_shots=4000)
    th.build()
    assert len(th.sub_blocks) == 2
    eng = QarpEngine(n_shots=4000, seed=42)
    eng.build([th])
    out = eng.run()[0]
    assert abs(out - 0.3) < 0.1


def _resolved_ansatz(values):
    """One symbolic 2-qubit ansatz, ``set_symbols``-bound to ``values``."""
    b = SimpleBlock(2, name="ansatz")
    b.h(0)
    b.rz(0, qx.Param.symbol("a"))
    b.cx(0, 1)
    b.ry(1, qx.Param.symbol("b"))
    b.build()
    return b.set_symbols(dict(zip(b.symbols, values, strict=True)))


def test_integration_transition_amplitude_resolved_ansatz():
    """Transition amplitude ``⟨bra|H|ket⟩`` of one ansatz bound to different
    parameters via ``set_symbols`` for bra and ket.

    Oracle: ``⟨bra|H|ket⟩`` from ``H.sparse_matrix()`` (LSB, independent of
    ``PauliBlock``) and the two resolved statevectors.  The operator carries a
    negative-real *and* a complex coefficient, so the transition-amplitude path
    (the only test reaching ``_append_controlled_multiplexor``) actually
    exercises the coefficient split — with the old rule it double-counts the
    complex term's phase.  Regression: controlling the resolved branches used to
    resurrect the ansatz symbols, which the engine then re-bound from the
    run-time parameter map — a silently wrong result."""
    bra = _resolved_ansatz((0.3, 0.7))
    ket = _resolved_ansatz((1.1, -0.4))
    H = QubitOperator("Z0", -1.0) + QubitOperator("X0 Y1", 0.3 - 0.4j)

    sim = qx.QarpSimulator()
    e0 = np.zeros(4, complex)
    e0[0] = 1.0
    psi_bra = np.array(sim.unitary_matrix(bra.flatten(), bra.n_qubits)) @ e0
    psi_ket = np.array(sim.unitary_matrix(ket.flatten(), ket.n_qubits)) @ e0
    h_mat = np.array(H.sparse_matrix(n_qubits=2).todense())
    oracle = complex(np.vdot(psi_bra, h_mat @ psi_ket))

    th = TermwiseHadamardTest(
        bra=bra, ket=ket, operator=H, real=True, imaginary=True, n_shots=40000
    )
    th.build()
    eng = QarpEngine(n_shots=40000, seed=7)
    eng.build([th])
    out = complex(eng.run()[0])
    assert out.real == pytest.approx(oracle.real, abs=0.05)
    assert out.imag == pytest.approx(oracle.imag, abs=0.05)


@pytest.mark.parametrize(
    "terms",
    [
        [("Z0", -1.0), ("X0 Y1", 0.3 - 0.4j)],  # complex: old rule double-counts
        [("Z0", -0.5), ("X0", -0.3)],  # negative reals
        [("Z0", 0.5), ("X0", -0.3)],  # mixed signs
    ],
)
def test_integration_qubit_operator_signed_and_complex(terms):
    """``⟨ψ|H|ψ⟩`` for operators with negative and complex coefficients.

    Oracle: ``H.sparse_matrix()`` (LSB, independent of ``PauliBlock``) and the
    ansatz statevector.  The complex case is the one the old rule got wrong
    (it double-counted ``arg(c)``); the signed cases guard against a sign flip."""
    from functools import reduce

    H = reduce(lambda a, b: a + b, (QubitOperator(p, c) for p, c in terms))
    psi_block = _resolved_ansatz((0.6, 1.3))
    sim = qx.QarpSimulator()
    e0 = np.zeros(4, complex)
    e0[0] = 1.0
    psi = np.array(sim.unitary_matrix(psi_block.flatten(), psi_block.n_qubits)) @ e0
    h_mat = np.array(H.sparse_matrix(n_qubits=2).todense())
    oracle = complex(np.vdot(psi, h_mat @ psi))

    th = TermwiseHadamardTest(operator=H, ket=psi_block, real=True, imaginary=True, n_shots=40000)
    th.build()
    eng = QarpEngine(n_shots=40000, seed=11)
    eng.build([th])
    out = complex(eng.run()[0])
    assert out.real == pytest.approx(oracle.real, abs=0.05)
    assert out.imag == pytest.approx(oracle.imag, abs=0.05)


def test_qubit_operator_coefficients_are_magnitudes():
    """The expanded per-term weights are ``|c|`` — the circuit carries ``arg(c)``,
    so weighting by the full complex ``c`` would apply the phase twice.
    Oracle: ``abs()`` of the input operator's term coefficients."""
    H = QubitOperator("Z0", -2.0) + QubitOperator("X0", 0.3 - 0.4j)
    th = TermwiseHadamardTest(operator=H, ket=_ket(), real=True, imaginary=True)
    th.build()
    assert sorted(float(c) for c in th.coefficients) == pytest.approx(sorted([2.0, 0.5]))
    assert all(complex(c).imag == 0 for c in th.coefficients)  # magnitudes are real


def test_list_path_leaves_block_coefficient_unconsulted():
    """Option C (item 2c): the list path's weights come only from ``coefficients=``.
    A block's own ``.coefficient`` is not consulted, and omitting ``coefficients``
    weights every term by ``1.0`` — regression pinning the deliberately-unchanged
    behaviour (the block below carries ``0.5``, which must be ignored)."""
    op = PauliBlock("Z", coefficient=0.5)
    th = TermwiseHadamardTest(operator=[op], ket=_ket(), real=True, imaginary=False)
    th.build()
    assert th.coefficients is None  # not derived from the block
    th.run([_sr({0: 1000}, 1000)])  # child returns +1.0
    assert th.coefficients == [1.0]  # default weight, block's 0.5 unconsulted
    assert th.result_list == pytest.approx([1.0])


# ── Overlap with no operator (identity default) ─────────────────────────


def _ry_state(theta: float) -> SimpleBlock:
    b = SimpleBlock(1)
    b.ry(0, theta)
    return b.build()


def test_explicit_bra_without_operator_measures_the_bare_overlap():
    """An explicit ``bra != ket`` with no operator is the identity Hadamard test.

    Oracle: ``<Ry(a)|Ry(b)> = cos((a - b) / 2)`` for two real single-qubit
    rotations -- analytic, not another qarp path.  This configuration used to
    raise "operator must be provided", which is what kept ADAPT-VQD on
    StateVector only.
    """
    from qarp import EXACT

    a, b = 0.2, 0.7
    ht = TermwiseHadamardTest(
        bra=_ry_state(a), ket=_ry_state(b), n_shots=EXACT, real=True, imaginary=True
    )
    eng = QarpEngine(seed=0)
    eng.build([ht])
    value = complex(np.asarray(eng.run()[0]).reshape(-1)[0])
    assert value == pytest.approx(np.cos((a - b) / 2), abs=1e-12)


def test_ket_only_without_operator_still_raises():
    """The identity default is guarded: a *forgotten* operator must not become
    the expectation of the identity (1.0)."""
    ht = TermwiseHadamardTest(ket=_ry_state(0.3))
    with pytest.raises(ValueError, match="operator must be provided"):
        ht.build()
    same = _ry_state(0.3)
    degenerate = TermwiseHadamardTest(bra=same, ket=same)
    with pytest.raises(ValueError, match="operator must be provided"):
        degenerate.build()


def test_coefficients_without_operator_raise_a_direct_message():
    ht = TermwiseHadamardTest(bra=_ry_state(0.1), ket=_ry_state(0.4), coefficients=[1.0])
    with pytest.raises(ValueError, match="coefficients were given but no operator"):
        ht.build()


def test_explicit_bra_without_operator_builds_as_an_overlap_not_an_identity_amplitude():
    """The overlap is its own target, not a transition amplitude of a
    substituted identity: ``operator`` stays ``None`` after ``build()``, and the
    primitive holds one child on the cheap OVERLAP circuit (real + imaginary
    sub-blocks, no Pauli-expanded identity composite)."""
    from qarp import EXACT
    from qarp.algorithms._primitives.target import Target

    ht = TermwiseHadamardTest(
        bra=_ry_state(0.2), ket=_ry_state(0.7), n_shots=EXACT, real=True, imaginary=True
    )
    ht.build()
    assert ht.target is Target.OVERLAP
    assert ht.operator is None
    assert ht.n_sub_algorithms == 1
    assert len(ht.sub_blocks) == 2
    assert ht.get_hadamard_test(0).target is Target.OVERLAP
