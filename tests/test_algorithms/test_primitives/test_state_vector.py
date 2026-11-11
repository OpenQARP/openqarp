"""Unit + integration tests for the StateVector primitive.

StateVector computes EXACT results by contracting simulator amplitudes
directly (``consumes = Consumes.AMPLITUDES``).  These tests cover the
Python layer: capability flags, input validation, target inference,
``build()`` orchestration, ``__repr__`` and deterministic end-to-end
checks through ``QarpEngine`` (exact → small n_shots).
"""

import numpy as np
import pytest

from qarp._types import Consumes
from qarp.algorithms import StateVector, Target
from qarp.blocks import ComputationalBasisStateBlock, HnBlock
from qarp.engines import QarpEngine
from qarp.operators import QubitOperator

# ── Class capability flags ──────────────────────────────────────────────


def test_state_vector_class_flags():
    assert StateVector.consumes is Consumes.AMPLITUDES
    assert StateVector.supports_backprop_gradient is True


# ── Validation (via build()) ────────────────────────────────────────────


def test_state_vector_rejects_non_block_ket():
    with pytest.raises(TypeError, match="ket must be a Block instance"):
        StateVector(ket="not a block").build()


def test_state_vector_rejects_non_block_bra():
    with pytest.raises(TypeError, match="bra must be a Block instance or None"):
        StateVector(ket=ComputationalBasisStateBlock(basis_state=[0]), bra="not a block").build()


def test_state_vector_rejects_non_operator():
    with pytest.raises(TypeError, match="operator must be a QubitOperator, a Block, or None"):
        StateVector(ket=ComputationalBasisStateBlock(basis_state=[0]), operator="not an op").build()


# ── Target inference (after build()) ────────────────────────────────────


def test_state_vector_infers_expectation_value():
    sv = StateVector(
        ket=ComputationalBasisStateBlock(basis_state=[0]), operator=QubitOperator("Z0")
    )
    sv.build()
    assert sv.target == Target.EXPECTATION_VALUE


def test_state_vector_infers_overlap():
    sv = StateVector(bra=ComputationalBasisStateBlock(basis_state=[0]), ket=HnBlock(1))
    sv.build()
    assert sv.target == Target.OVERLAP


def test_state_vector_infers_transition_amplitude():
    sv = StateVector(
        bra=ComputationalBasisStateBlock(basis_state=[1]),
        ket=ComputationalBasisStateBlock(basis_state=[0]),
        operator=QubitOperator("X0"),
    )
    sv.build()
    assert sv.target == Target.TRANSITION_AMPLITUDE


def test_state_vector_rejects_sampling_target():
    """ket-only (SAMPLING) is not served here — Sampler(n_shots=qarp.EXACT) is."""
    from qarp.errors import CapabilityError

    sv = StateVector(ket=ComputationalBasisStateBlock(basis_state=[0]))
    with pytest.raises(CapabilityError, match=r"Sampler\(n_shots=qarp\.EXACT\)"):
        sv.build()


# ── build() sub_blocks ──────────────────────────────────────────────────


def test_state_vector_build_expectation_subblocks():
    """EXPECTATION → [ket] (length 1)."""
    ket = ComputationalBasisStateBlock(basis_state=[0])
    sv = StateVector(ket=ket, operator=QubitOperator("Z0"))
    sv.build()
    assert len(sv.sub_blocks) == 1
    assert sv.sub_blocks[0] is ket


def test_state_vector_build_overlap_subblocks():
    """OVERLAP → [ket, bra] (length 2)."""
    bra = ComputationalBasisStateBlock(basis_state=[0])
    ket = HnBlock(1)
    sv = StateVector(bra=bra, ket=ket)
    sv.build()
    assert len(sv.sub_blocks) == 2
    assert sv.sub_blocks[0] is ket
    assert sv.sub_blocks[1] is bra


# ── __repr__ ─────────────────────────────────────────────────────────────


def test_state_vector_repr():
    assert "StateVector" in repr(StateVector(ket=ComputationalBasisStateBlock(basis_state=[0])))


# ── Integration through QarpEngine (exact → deterministic) ───────────────


def test_state_vector_expectation_z_on_zero():
    """⟨0|Z|0⟩ = 1."""
    sv = StateVector(
        ket=ComputationalBasisStateBlock(basis_state=[0]), operator=QubitOperator("Z0")
    )
    sv.build()
    eng = QarpEngine(n_shots=10, seed=0)
    eng.build([sv])
    assert eng.run()[0] == pytest.approx(1.0)


def test_state_vector_expectation_z_on_one():
    """⟨1|Z|1⟩ = -1."""
    sv = StateVector(
        ket=ComputationalBasisStateBlock(basis_state=[1]), operator=QubitOperator("Z0")
    )
    sv.build()
    eng = QarpEngine(n_shots=10, seed=0)
    eng.build([sv])
    assert eng.run()[0] == pytest.approx(-1.0)


def test_state_vector_expectation_x_on_plus():
    """⟨+|X|+⟩ = 1."""
    sv = StateVector(ket=HnBlock(1), operator=QubitOperator("X0"))
    sv.build()
    eng = QarpEngine(n_shots=10, seed=0)
    eng.build([sv])
    assert abs(eng.run()[0] - 1.0) < 1e-10


def test_state_vector_expectation_survives_deepcopy_ket_rebind():
    """⟨1|Z|1⟩ = -1 after deepcopy + rebinding only ket |0⟩→|1⟩ — the
    defaulted bra must follow the new ket, not flip to a transition
    amplitude against the old one (⟨0|Z|1⟩ = 0)."""
    import copy

    sv = StateVector(
        ket=ComputationalBasisStateBlock(basis_state=[0]), operator=QubitOperator("Z0")
    )
    sv.build()
    clone = copy.deepcopy(sv)
    clone.ket = ComputationalBasisStateBlock(basis_state=[1])
    clone.build()
    assert clone.target == Target.EXPECTATION_VALUE
    eng = QarpEngine(n_shots=10, seed=0)
    eng.build([clone])
    assert eng.run()[0] == pytest.approx(-1.0)


def test_state_vector_overlap_zero_with_plus():
    """⟨0|+⟩ = 1/√2."""
    sv = StateVector(bra=ComputationalBasisStateBlock(basis_state=[0]), ket=HnBlock(1))
    sv.build()
    eng = QarpEngine(n_shots=10, seed=0)
    eng.build([sv])
    assert abs(eng.run()[0] - 1 / np.sqrt(2)) < 1e-10


def test_state_vector_multiqubit_z1_on_q1():
    """ket = |0> on q0, |1> on q1; ⟨Z1⟩ = -1 (exercises LSB matrix conversion)."""
    sv = StateVector(
        ket=ComputationalBasisStateBlock(basis_state=[0, 1]),
        operator=QubitOperator("Z1"),
    )
    sv.build()
    eng = QarpEngine(n_shots=10, seed=0)
    eng.build([sv])
    assert eng.run()[0] == pytest.approx(-1.0)


def test_state_vector_multiqubit_z0_on_q0():
    """Same ket |0>_q0 |1>_q1; ⟨Z0⟩ = +1 (qubit-ordering / endianness check)."""
    sv = StateVector(
        ket=ComputationalBasisStateBlock(basis_state=[0, 1]),
        operator=QubitOperator("Z0"),
    )
    sv.build()
    eng = QarpEngine(n_shots=10, seed=0)
    eng.build([sv])
    assert eng.run()[0] == pytest.approx(1.0)


def test_state_vector_transition_amplitude_x_one_zero():
    """⟨1|X|0⟩ = 1."""
    sv = StateVector(
        bra=ComputationalBasisStateBlock(basis_state=[1]),
        ket=ComputationalBasisStateBlock(basis_state=[0]),
        operator=QubitOperator("X0"),
    )
    sv.build()
    eng = QarpEngine(n_shots=10, seed=0)
    eng.build([sv])
    assert abs(eng.run()[0] - 1.0) < 1e-10


# ── pauli_expectation helper (B): matrix-free ⟨bra|H|ket⟩ ────────────────
#
# The helper replaced a dense O(4^n) operator-matrix build with a direct
# O(terms·2^n) Pauli sweep.  Pin it against the dense matrix as an exact
# oracle (the dense path is the pre-B implementation, kept as the reference).


def _dense_oracle(bra, ket, operator, n_qubits):
    """⟨bra|operator|ket⟩ via the dense qarpx-LSB matrix."""
    M = np.array(operator.sparse_matrix(n_qubits).toarray(), dtype=complex)
    return complex(bra.conj() @ M @ ket)


# Specs, not QubitOperator instances: parametrize values live in the module
# global AND in pytest's CallSpec2 for the whole session, so C++-backed objects
# built here are still alive at interpreter shutdown (nanobind reports them as
# leaked).  Build inside the test instead.
PAULI_OP_SPECS = [
    [("Z0", 1.0)],
    [("X0 X1", 0.5), ("Z0 Z1", -0.3)],
    [("Y0", 1.0)],  # exercises the i^{n_Y} phase
    [("Y0 Y1", 1.0), ("X0 Y1 Z2", 0.7)],
    [("", 2.0), ("Z1", 1.0)],  # identity term
]


def _op(spec):
    op = QubitOperator(*spec[0])
    for term, coeff in spec[1:]:
        op = op + QubitOperator(term, coeff)
    return op


@pytest.mark.parametrize("spec", PAULI_OP_SPECS)
def test_pauli_expectation_matches_dense_oracle(spec):
    from qarp.algorithms._primitives.state_vector import pauli_expectation

    op = _op(spec)
    n = 3
    rng = np.random.default_rng(7)
    bra = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    ket = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    got = pauli_expectation(bra, ket, op, n)
    assert got == pytest.approx(_dense_oracle(bra, ket, op, n))


def test_pauli_expectation_none_operator_is_overlap():
    from qarp.algorithms._primitives.state_vector import pauli_expectation

    n = 2
    rng = np.random.default_rng(1)
    bra = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    ket = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    assert pauli_expectation(bra, ket, None, n) == pytest.approx(complex(np.vdot(bra, ket)))


# ── pauli_apply helper: matrix-free operator|sv⟩ (QSE Gram fast path) ─────


@pytest.mark.parametrize("spec", PAULI_OP_SPECS)
def test_pauli_apply_matches_dense_oracle(spec):
    from qarp.algorithms._primitives.state_vector import pauli_apply

    op = _op(spec)
    n = 3
    rng = np.random.default_rng(7)
    sv = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    M = np.array(op.sparse_matrix(n).toarray(), dtype=complex)
    ref = M @ sv
    assert np.linalg.norm(pauli_apply(sv, op, n) - ref) < 1e-12


def test_pauli_apply_none_operator_is_identity():
    from qarp.algorithms._primitives.state_vector import pauli_apply

    n = 2
    rng = np.random.default_rng(1)
    sv = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    out = pauli_apply(sv, None, n)
    assert np.linalg.norm(out - sv) < 1e-15
    out[0] = 0  # must be a copy, not a view
    assert sv[0] != 0


def test_pauli_apply_consistent_with_pauli_expectation():
    from qarp.algorithms._primitives.state_vector import pauli_apply, pauli_expectation

    n = 3
    op = QubitOperator("X0 Y1", 0.3) + QubitOperator("Z2", -1.1) + QubitOperator("", 0.5)
    rng = np.random.default_rng(11)
    bra = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    ket = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    assert pauli_expectation(bra, ket, op, n) == pytest.approx(
        complex(np.vdot(bra, pauli_apply(ket, op, n)))
    )


# ── diagonal_pauli_vector: Z-only observable fast path ────────────────────


def test_diagonal_pauli_vector_matches_dense():
    from qarp.algorithms._primitives.state_vector import diagonal_pauli_vector

    n = 3
    op = QubitOperator("Z0 Z1", 0.7) + QubitOperator("Z2", -1.3) + QubitOperator("", 0.25)
    d = diagonal_pauli_vector(op, n)
    M = np.array(op.sparse_matrix(n).toarray(), dtype=complex)
    assert np.linalg.norm(np.diag(M) - d) < 1e-12


def test_diagonal_pauli_vector_rejects_non_diagonal():
    from qarp.algorithms._primitives.state_vector import diagonal_pauli_vector

    assert diagonal_pauli_vector(QubitOperator("Z0 X1", 1.0), 2) is None
    assert diagonal_pauli_vector(QubitOperator("Y0", 1.0), 1) is None
    assert diagonal_pauli_vector(None, 2) is None


def test_diag_fast_path_matches_pauli_expectation():
    from qarp.algorithms._primitives.state_vector import (
        diagonal_pauli_vector,
        pauli_expectation,
    )

    n = 4
    op = QubitOperator("Z0 Z2", 0.4) + QubitOperator("Z1 Z3", -0.9) + QubitOperator("", 1.5)
    d = diagonal_pauli_vector(op, n)
    rng = np.random.default_rng(5)
    bra = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    ket = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    assert complex(np.vdot(bra, d * ket)) == pytest.approx(pauli_expectation(bra, ket, op, n))
    assert complex(np.vdot(ket, d * ket)) == pytest.approx(pauli_expectation(ket, ket, op, n))


def _dense_diagonal(op, n_qubits):
    return np.diag(np.asarray(op.sparse_matrix(n_qubits).toarray(), dtype=complex))


def _z_operator(rng, op_width, n_terms, *, constant=False, complex_index=None):
    """``n_terms`` distinct Z-strings on qubits ``< op_width``, coefficients
    complex-typed; only the term at ``complex_index`` has an imaginary part."""
    masks = rng.choice(np.arange(1, 1 << op_width), size=n_terms, replace=False)
    op = QubitOperator()
    for i, mask in enumerate(masks):
        term = tuple((q, "Z") for q in range(op_width) if (int(mask) >> q) & 1)
        imag = float(rng.normal()) if i == complex_index else 0.0
        op += QubitOperator(term, complex(float(rng.normal()), imag))
    if constant:
        op += QubitOperator("", complex(float(rng.normal()), 0.0))
    return op


# (n_terms, constant, op_width, n_qubits): sparse operators, the full Z-subset
# basis, and padding qubits beyond the operator.
_DIAG_CASES = [
    (2, False, 6, 6),
    (5, False, 6, 6),
    (5, True, 6, 6),
    (6, False, 6, 6),
    (63, True, 6, 6),
    (2, False, 3, 6),
    (7, True, 3, 6),
]


@pytest.mark.parametrize(
    "n_terms, constant, op_width, n_qubits",
    _DIAG_CASES,
    ids=[f"T{t}{'+c' if c else ''}-w{w}-n{n}" for t, c, w, n in _DIAG_CASES],
)
def test_diagonal_pauli_vector_real_matches_dense_float64(n_terms, constant, op_width, n_qubits):
    from qarp.algorithms._primitives.state_vector import diagonal_pauli_vector

    rng = np.random.default_rng(100 + n_terms + op_width)
    op = _z_operator(rng, op_width, n_terms, constant=constant)
    d = diagonal_pauli_vector(op, n_qubits)
    assert d.dtype == np.float64
    assert d.shape == (1 << n_qubits,)
    np.testing.assert_allclose(d, _dense_diagonal(op, n_qubits).real, rtol=0, atol=1e-12)


@pytest.mark.parametrize("n_terms", [2, 9])
def test_diagonal_pauli_vector_complex_coefficient_matches_dense(n_terms):
    from qarp.algorithms._primitives.state_vector import diagonal_pauli_vector

    rng = np.random.default_rng(7 + n_terms)
    n_qubits = 4
    op = _z_operator(rng, n_qubits, n_terms, constant=True, complex_index=1)
    d = diagonal_pauli_vector(op, n_qubits)
    assert d.dtype == np.complex128
    np.testing.assert_allclose(d, _dense_diagonal(op, n_qubits), rtol=0, atol=1e-12)


@pytest.mark.parametrize("n_terms", [1, 4])
def test_diagonal_pauli_vector_rejects_qubit_beyond_width(n_terms):
    from qarp.algorithms._primitives.state_vector import diagonal_pauli_vector

    op = QubitOperator("Z3", 1.0)
    for q in range(n_terms - 1):
        op += QubitOperator(f"Z{q}", 0.5)
    with pytest.raises(ValueError):
        diagonal_pauli_vector(op, 3)


@pytest.mark.parametrize(
    "spec, n_qubits, want",
    [
        ([("Z0", 1.0)], 2, [1, -1, 1, -1]),
        (
            [("Z0 Z1", 0.7), ("Z2", -1.3), ("", 0.25)],
            3,
            [-0.35, -1.75, -1.75, -0.35, 2.25, 0.85, 0.85, 2.25],
        ),
    ],
    ids=["Z0-n2", "ZZ-Z-const-n3"],
)
def test_diagonal_pauli_vector_lsb_anchors(spec, n_qubits, want):
    """Hand-computed diagonals: entry k has qubit q at bit q of k."""
    from qarp.algorithms._primitives.state_vector import diagonal_pauli_vector

    op = QubitOperator()
    for label, coeff in spec:
        op += QubitOperator(label, coeff)
    np.testing.assert_allclose(diagonal_pauli_vector(op, n_qubits), want, rtol=0, atol=1e-12)


def test_state_vector_symbolic_diagonal_coefficient_fails_at_build():
    import sympy

    op = QubitOperator("Z0", sympy.Symbol("a")) + QubitOperator("Z1", 0.5)
    sv = StateVector(operator=op, ket=HnBlock(2).build())
    with pytest.raises(TypeError, match="non-numeric coefficient"):
        sv.build()


def test_state_vector_diagonal_at_recomputes_per_width():
    op = QubitOperator("Z0 Z1", 0.7) + QubitOperator("Z2", -1.3) + QubitOperator("", 0.25)
    sv = StateVector(operator=op, ket=HnBlock(3).build())
    sv.build()
    assert sv._z_only is True
    assert sv._diag is None

    d3 = sv._diagonal_at(3)
    np.testing.assert_allclose(d3, _dense_diagonal(op, 3).real, rtol=0, atol=1e-12)
    assert sv._diagonal_at(3) is d3

    d5 = sv._diagonal_at(5)
    assert d5.shape == (32,)
    np.testing.assert_allclose(d5, _dense_diagonal(op, 5).real, rtol=0, atol=1e-12)
    assert sv._diag is d5
    assert sv._diagonal_at(5) is d5

    with pytest.raises(ValueError):
        sv._diagonal_at(2)


# ── Circuit-valued (Block) operators: ⟨bra|U|ket⟩ via vdot(sv(bra), sv(ket∘U)) ─


def _x_block():
    from qarp.blocks import SimpleBlock

    b = SimpleBlock(1, name="X")
    b.x(0)
    b.build()
    return b


def _h_block():
    from qarp.blocks import SimpleBlock

    b = SimpleBlock(1, name="H")
    b.h(0)
    b.build()
    return b


def _ry_ket(theta):
    from qarp.blocks import SimpleBlock

    b = SimpleBlock(1, name="ry")
    b.ry(0, theta)
    b.build()
    return b


def test_block_operator_ev_matches_qubit_operator():
    """⟨ψ|X|ψ⟩ for ψ = Ry(0.7)|0⟩: Block-U path must equal the QubitOperator path."""
    theta = 0.7
    eng = QarpEngine(n_shots=10, seed=0)
    sv_block = StateVector(ket=_ry_ket(theta), operator=_x_block())
    sv_qop = StateVector(ket=_ry_ket(theta), operator=QubitOperator("X0"))
    eng.build([sv_block, sv_qop])
    r_block, r_qop = eng.run()
    assert r_block == pytest.approx(r_qop)
    assert r_block == pytest.approx(np.sin(theta))


def test_block_operator_ev_returns_complex_amplitude():
    """⟨0|H|0⟩ = 1/√2 — U need not be Hermitian-measurable, result is complex."""
    sv = StateVector(ket=ComputationalBasisStateBlock(basis_state=[0]), operator=_h_block())
    eng = QarpEngine(n_shots=10, seed=0)
    eng.build([sv])
    out = eng.run()[0]
    assert isinstance(out, complex)
    assert out == pytest.approx(1 / np.sqrt(2))


def test_block_operator_ev_subblocks_layout():
    """EV with Block U → [ket, ket∘U]; the bare ket is first."""
    ket = _ry_ket(0.3)
    sv = StateVector(ket=ket, operator=_x_block())
    sv.build()
    assert sv.target == Target.EXPECTATION_VALUE
    assert len(sv.sub_blocks) == 2
    assert sv.sub_blocks[0] is ket


def test_block_operator_transition_matches_qubit_operator():
    """⟨1|X|0⟩ = 1: Block-U path equals the QubitOperator path."""
    eng = QarpEngine(n_shots=10, seed=0)
    sv_block = StateVector(
        bra=ComputationalBasisStateBlock(basis_state=[1]),
        ket=ComputationalBasisStateBlock(basis_state=[0]),
        operator=_x_block(),
    )
    sv_qop = StateVector(
        bra=ComputationalBasisStateBlock(basis_state=[1]),
        ket=ComputationalBasisStateBlock(basis_state=[0]),
        operator=QubitOperator("X0"),
    )
    eng.build([sv_block, sv_qop])
    r_block, r_qop = eng.run()
    assert sv_block.target == Target.TRANSITION_AMPLITUDE
    assert r_block == pytest.approx(r_qop)
    assert r_block == pytest.approx(1.0)


def test_block_operator_transition_subblocks_layout():
    """TRANSITION with Block U → [ket∘U, bra]; no bare ket circuit."""
    bra = ComputationalBasisStateBlock(basis_state=[1])
    sv = StateVector(
        bra=bra, ket=ComputationalBasisStateBlock(basis_state=[0]), operator=_h_block()
    )
    sv.build()
    assert len(sv.sub_blocks) == 2
    assert sv.sub_blocks[1] is bra


def test_block_operator_gradient_falls_back_to_parameter_shift():
    """Block-U EV is not adjoint-eligible; parameter shift gives ∂Re⟨ψ|X|ψ⟩/∂θ.

    E(θ) = ⟨Ry(θ)0|X|Ry(θ)0⟩ = sin θ  ⇒  dE/dθ = cos θ (exact for a bare Ry).
    """
    import qarpx as qx
    from qarp.blocks import SimpleBlock

    ket = SimpleBlock(1, name="ry")
    ket.ry(0, qx.Param.symbol("theta"))
    ket.build()
    sv = StateVector(ket=ket, operator=_x_block())
    eng = QarpEngine()
    eng.build([sv])
    grad = eng.run_gradient({"theta": 0.3})[0]
    assert grad[0] == pytest.approx(np.cos(0.3), abs=1e-10)


def test_operator_wider_than_circuit_pads_idle_qubits():
    """Observable qubits beyond the circuit width are idle |0⟩ — the same
    convention as the adjoint-gradient path.  X on an idle qubit must not
    IndexError inside pauli_expectation."""
    from qarp.blocks import SimpleBlock

    ket = SimpleBlock(2, name="k")
    ket.ry(0, 0.7)
    ket.cx(0, 1)
    ket.build()
    H = QubitOperator("X2", 0.5) + QubitOperator("Z2", 2.0) + QubitOperator("Z0", 1.0)
    sv = StateVector(ket=ket, operator=H)
    eng = QarpEngine()
    eng.build([sv])
    # ⟨X2⟩=0 (idle |0⟩), ⟨Z2⟩=+1, ⟨Z0⟩=cos(0.7)
    assert complex(eng.run()[0]) == pytest.approx(2.0 + np.cos(0.7), abs=1e-10)
