"""Correctness tests for the three Synthesized*Block primitives.

Each block now inherits from :class:`SimpleBlock` and delegates synthesis to
the C++ qarpx primitives (Möttönen state-prep, Quantum Shannon Decomposition).
The tests verify end-to-end equivalence to the target object (a state vector,
a matrix, or `expm(-iHt)`).
"""

import numpy as np
import pytest
from scipy.linalg import expm

import qarpx as qx
from qarp.blocks import (
    SynthesizedStateBlock,
    SynthesizedTimeEvolutionBlock,
    SynthesizedUnitaryBlock,
)
from qarp.operators import QubitOperator


def _unitary(block):
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


def _first_column(block):
    return _unitary(block)[:, 0]


def _max_diff_up_to_phase(actual, expected):
    """Element-wise max abs diff after correcting for a global phase.

    Only for :class:`SynthesizedStateBlock`: a global phase on a *state* is
    physically unobservable, so it is genuinely outside that block's contract.
    The unitary and time-evolution blocks use :func:`_max_diff` instead — for
    them the phase is observable under a control.
    """
    nz = np.flatnonzero(np.abs(expected) > 1e-6)
    if len(nz) == 0:
        return float(np.abs(actual - expected).max())
    k = int(nz[0])
    if abs(actual.flat[k]) < 1e-15:
        return float("inf")
    phase = expected.flat[k] / actual.flat[k]
    phase /= abs(phase)
    return float(np.abs(actual * phase - expected).max())


def _max_diff(actual, expected):
    """Element-wise max abs diff — global phase included, never divided out."""
    return float(np.abs(actual - expected).max())


# ── SynthesizedStateBlock ─────────────────────────────────────────────────


def test_state_block_single_qubit_phased():
    psi = [0.6 + 0.0j, 0.8 * np.exp(1j * np.pi / 3)]
    b = SynthesizedStateBlock(n_qubits=1, amplitudes=psi).build()
    expected = np.array(psi) / np.linalg.norm(psi)
    assert _max_diff_up_to_phase(_first_column(b), expected) < 1e-10


def test_state_block_two_qubit_random():
    rng = np.random.default_rng(0xC0FFEE)
    psi = rng.standard_normal(4) + 1j * rng.standard_normal(4)
    b = SynthesizedStateBlock(n_qubits=2, amplitudes=psi).build()
    expected = psi / np.linalg.norm(psi)
    assert _max_diff_up_to_phase(_first_column(b), expected) < 1e-10


def test_state_block_three_qubit_random():
    rng = np.random.default_rng(0xBADF00D)
    psi = rng.standard_normal(8) + 1j * rng.standard_normal(8)
    b = SynthesizedStateBlock(n_qubits=3, amplitudes=psi).build()
    expected = psi / np.linalg.norm(psi)
    assert _max_diff_up_to_phase(_first_column(b), expected) < 1e-10


def test_state_block_dict_input():
    # Dict keys are LSB-first tuples (§1): element i is qubit i, the same
    # convention as list indices and Sampler keys.
    amps = {(0, 1, 1): 1.0 + 0j, (1, 0, 0): 1.0 + 0j}
    b = SynthesizedStateBlock(n_qubits=3, amplitudes=amps).build()
    # LSB-first: (0,1,1) → 2¹+2² = index 6, (1,0,0) → 2⁰ = index 1.
    expected = np.zeros(8, complex)
    expected[6] = 1
    expected[1] = 1
    expected /= np.linalg.norm(expected)
    assert _max_diff_up_to_phase(_first_column(b), expected) < 1e-10


def test_state_block_zero_amplitudes_raises():
    with pytest.raises(ValueError, match="cannot all be zero"):
        SynthesizedStateBlock(n_qubits=1, amplitudes=[0, 0])


# ── SynthesizedUnitaryBlock ───────────────────────────────────────────────


def _haar_random_unitary(N, rng):
    A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    Q, R = np.linalg.qr(A)
    d = np.diag(R) / np.abs(np.diag(R))
    return Q * d


def test_unitary_block_single_qubit_hadamard():
    H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    b = SynthesizedUnitaryBlock(unitary_matrix=H).build()
    assert _max_diff(_unitary(b), H) < 1e-10


def test_unitary_block_two_qubit_swap():
    # SWAP in qarpx LSB convention: |01⟩ ↔ |10⟩, i.e., index 1 ↔ index 2.
    SWAP = np.zeros((4, 4), complex)
    SWAP[0, 0] = 1
    SWAP[2, 1] = 1
    SWAP[1, 2] = 1
    SWAP[3, 3] = 1
    b = SynthesizedUnitaryBlock(unitary_matrix=SWAP).build()
    assert _max_diff(_unitary(b), SWAP) < 1e-10


def test_unitary_block_three_qubit_haar():
    rng = np.random.default_rng(0xBEEF)
    U = _haar_random_unitary(8, rng)
    b = SynthesizedUnitaryBlock(unitary_matrix=U).build()
    assert _max_diff(_unitary(b), U) < 1e-9


@pytest.mark.parametrize(
    "name, U",
    [
        ("diagonal", np.diag([np.exp(0.7j), np.exp(-1.3j)])),
        ("antidiagonal", np.array([[0, np.exp(0.4j)], [np.exp(2.1j), 0]])),
        ("S", np.diag([1, 1j])),
        ("T", np.diag([1, np.exp(1j * np.pi / 4)])),
        ("pauli_x", np.array([[0, 1], [1, 0]], complex)),
        ("pauli_z", np.diag([1, -1]).astype(complex)),
        ("global_phase", np.exp(1.0j) * np.eye(2)),
    ],
)
def test_unitary_block_phase_exact_on_zyz_edge_cases(name, U):
    """QSD's ZYZ base case must reproduce the global phase on its edge branches.

    Diagonal targets take the γ ≈ 0 branch and anti-diagonal ones γ ≈ π; both
    once emitted a wrong global phase.  The block carries no phase calibration
    of its own, so this pins the C++ synthesizer directly.
    """
    b = SynthesizedUnitaryBlock(unitary_matrix=U).build()
    actual = _unitary(b)
    residual = float(np.angle(np.trace(U.conj().T @ actual)))
    assert _max_diff(actual, U) < 1e-10, f"{name}: residual global phase {residual}"


def test_unitary_block_phase_exact_under_control():
    """A global phase on the inner block becomes a relative phase under control.

    Guards the property that motivated phase-exact synthesis: the ``|0⟩``-control
    subspace is the identity, so any phase error on the inner unitary shows up
    as a measurable relative phase between the two subspaces.
    """
    from qarp.blocks import ControlledBlock

    U = np.diag([np.exp(0.7j), np.exp(-1.3j)])
    inner = SynthesizedUnitaryBlock(unitary_matrix=U).build()
    ctrl = ControlledBlock(inner, num_controls=1, ctrl_state=[True])
    ctrl.build()
    U_ctrl = _unitary(ctrl)

    # LSB: control at q0 → state index = i_inner * 2 + i_ctrl.
    idx_c0 = list(range(0, 4, 2))
    idx_c1 = list(range(1, 4, 2))
    assert _max_diff(U_ctrl[np.ix_(idx_c0, idx_c0)], np.eye(2)) < 1e-10
    assert _max_diff(U_ctrl[np.ix_(idx_c1, idx_c1)], U) < 1e-10


def test_unitary_block_non_unitary_raises():
    with pytest.raises(ValueError, match="not unitary"):
        SynthesizedUnitaryBlock(unitary_matrix=np.array([[1, 0], [0, 0.5]]))


def test_unitary_block_non_power_of_two_raises():
    with pytest.raises(ValueError, match="power of 2"):
        SynthesizedUnitaryBlock(unitary_matrix=np.eye(3))


# ── SynthesizedTimeEvolutionBlock ─────────────────────────────────────────


def test_time_evolution_block_single_qubit_z():
    # H = Z on 1 qubit; expect U(t) = exp(-i Z t) = diag(e^{-it}, e^{+it}).
    op = QubitOperator("Z0")
    t = 0.7
    b = SynthesizedTimeEvolutionBlock(operator=op, n_qubits=1, time=t).build()
    Z = np.array([[1, 0], [0, -1]], complex)
    expected = expm(-1j * Z * t)
    assert _max_diff(_unitary(b), expected) < 1e-10


def test_time_evolution_block_two_qubit_zz():
    op = QubitOperator("Z0 Z1")
    t = 0.42
    b = SynthesizedTimeEvolutionBlock(operator=op, n_qubits=2, time=t).build()
    Z = np.array([[1, 0], [0, -1]], complex)
    H = np.kron(Z, Z)  # qarpx LSB: q0 = lower bit, but kron(Z, Z) is symmetric so identical.
    expected = expm(-1j * H * t)
    assert _max_diff(_unitary(b), expected) < 1e-10


def test_time_evolution_block_two_qubit_z0_only():
    """Z on qubit 0 of a 2-qubit register — asymmetric under bit reversal.

    qarpx is LSB (qubit 0 = bit 0); an MSB reading of the operator would
    evolve the wrong subspace, so this test locks in the qubit↔bit mapping.
    """
    op = QubitOperator("Z0")
    t = 0.55
    b = SynthesizedTimeEvolutionBlock(operator=op, n_qubits=2, time=t).build()

    Z = np.array([[1, 0], [0, -1]], complex)
    I = np.eye(2, dtype=complex)
    # qarpx LSB-Kron: q0 on the right, so "Z on q0" = I_{q1} ⊗ Z_{q0}.
    H_qarpx = np.kron(I, Z)
    expected = expm(-1j * H_qarpx * t)
    assert _max_diff(_unitary(b), expected) < 1e-10


def test_time_evolution_block_xy_model():
    # Mixed XY on two qubits.
    op = QubitOperator("X0 X1") + QubitOperator("Y0 Y1")
    t = 0.3
    b = SynthesizedTimeEvolutionBlock(operator=op, n_qubits=2, time=t).build()

    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    # qarpx LSB: q0 is least significant.  X0 X1 → on qubit 0 (lowest) tensor on qubit 1 (higher).
    # In LSB Kron-convention, ``op_on_q1 ⊗ op_on_q0`` puts the higher qubit on the left.
    XX = np.kron(X, X)
    YY = np.kron(Y, Y)
    expected = expm(-1j * (XX + YY) * t)
    assert _max_diff(_unitary(b), expected) < 1e-9


def test_time_evolution_block_set_time_round_trip():
    op = QubitOperator("Z0")
    b = SynthesizedTimeEvolutionBlock(operator=op, n_qubits=1, time=0.5).build()
    b2 = b.set_time(1.3)
    Z = np.array([[1, 0], [0, -1]], complex)
    expected = expm(-1j * Z * 1.3)
    assert _max_diff(_unitary(b2), expected) < 1e-10


def test_time_evolution_block_unset_time_raises():
    op = QubitOperator("Z0")
    b = SynthesizedTimeEvolutionBlock(operator=op, n_qubits=1)
    with pytest.raises(ValueError, match="Time parameter is not set"):
        b.build()


def test_time_evolution_block_non_hermitian_raises():
    op = QubitOperator("Z0", 1.0j)  # imaginary coefficient → non-Hermitian
    with pytest.raises(ValueError, match="not Hermitian"):
        SynthesizedTimeEvolutionBlock(operator=op, n_qubits=1, time=1.0)
