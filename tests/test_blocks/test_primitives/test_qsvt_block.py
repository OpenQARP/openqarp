"""Correctness tests for QSVTBlock.

Pattern B composite assembling alternating ``BE`` / ``BE†`` and
``ProjectedControlPhaseBlock`` layers per the QSVT recipe.  The standard
correctness check: for a real-symmetric ``A`` with ``λ = ‖coeffs‖₁ = 1`` and a
polynomial ``P(x) = 2x² − 1`` whose QSVT angles are ``QSPAngleFinder``,
the projection of the assembled circuit onto the ``ancilla = |0…0⟩`` subspace
should equal ``P(A)`` (within tolerance).
"""

import copy

import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import BlockEncodingBlock, QSPAngleFinder, QSVTBlock
from qarp.operators import QubitOperator


def _project_ancilla_zero(U: np.ndarray, n_anc: int, n_qubits: int) -> np.ndarray:
    """Extract the (target × target) sub-block of ``U`` on ``ancilla = |0…0⟩``."""
    N_anc = 2**n_anc
    N_target = 2 ** (n_qubits - n_anc)
    idx = [it * N_anc for it in range(N_target)]
    return U[np.ix_(idx, idx)]


def _build_unitary(qsvt: QSVTBlock) -> np.ndarray:
    return np.array(qx.QarpSimulator().unitary_matrix(qsvt.flatten(), qsvt.n_qubits))


# ── Polynomial-application tests on a 2×2 Hermitian matrix ────────────────


@pytest.fixture
def small_real_hermitian():
    """Hermitian 2×2 with ``λ = 0.8`` (rescaled internally to ``A / λ``)."""
    A = np.array([[0.5, 0.3], [0.3, -0.5]])
    be = BlockEncodingBlock(A)
    return A / be.lambda_factor()


def _be_n_qubits(A) -> int:
    be = BlockEncodingBlock(A)
    return be.n_qubits


def test_qsvt_applies_degree_one_polynomial(small_real_hermitian):
    """Degree-1 polynomial ``P(x) = x`` — projection should recover ``A``.

    This works in qarpx LSB convention; degree-1 (odd) is the simplest case
    and the only polynomial-application test currently passing.
    """
    A = small_real_hermitian
    angles = QSPAngleFinder([0, 1]).QSVT()  # P(x) = x
    qsvt = QSVTBlock(A, angles).build()
    n_anc = qsvt.n_qubits - int(np.log2(A.shape[0]))
    sub = _project_ancilla_zero(_build_unitary(qsvt), n_anc, qsvt.n_qubits)
    assert np.linalg.norm(np.real(sub) - np.real(A)) < 1e-10


def test_qsvt_applies_x_squared_minus_one(small_real_hermitian):
    """Degree-2 polynomial ``P(x) = x² − 1`` — projection should recover ``A² − I``.

    Guards the composite/controlled ``dagger`` overrides: inheriting the base
    ``Block::dagger``, which operates on the empty top-level ``commands_``
    buffer of a composite or controlled block, collapses every inner ``BE†``
    layer to identity and makes even-degree QSVT yield ``±i·A`` rather than
    ``P(A)``.
    """
    A = small_real_hermitian
    angles = QSPAngleFinder([-1, 0, 1]).QSVT()  # P(x) = x² − 1
    qsvt = QSVTBlock(A, angles).build()
    n_anc = qsvt.n_qubits - int(np.log2(A.shape[0]))
    sub = _project_ancilla_zero(_build_unitary(qsvt), n_anc, qsvt.n_qubits)
    expected = A @ A - np.eye(A.shape[0])
    assert np.linalg.norm(np.real(sub) - np.real(expected)) < 1e-10


def test_qsvt_even_degree_survives_deepcopy(small_real_hermitian):
    """Even-degree QSVT circuits must remain deepcopy-safe *and* correct.

    Even-degree polynomials interleave ``BE†`` layers.  A raw C++-daggered
    child (no ``__deepcopy__``) makes ``deepcopy`` fail with "cannot pickle
    'qarpx.Block' object" — reachable whenever a QSVT block is reused inside
    a Hadamard-test-family primitive, which deep-copies its inputs.  Guards
    both failure modes: the copy must not raise, and must still implement
    ``P(x) = x² − 1`` (a mis-materialised composite dagger silently yields
    an empty command stream, i.e. identity).
    """
    A = small_real_hermitian
    angles = QSPAngleFinder([-1, 0, 1]).QSVT()  # P(x) = x² − 1
    qsvt = QSVTBlock(A, angles).build()

    qsvt_copy = copy.deepcopy(qsvt)

    n_anc = qsvt_copy.n_qubits - int(np.log2(A.shape[0]))
    sub = _project_ancilla_zero(_build_unitary(qsvt_copy), n_anc, qsvt_copy.n_qubits)
    expected = A @ A - np.eye(A.shape[0])
    assert np.linalg.norm(np.real(sub) - np.real(expected)) < 1e-10


def test_qsvt_odd_degree_ge3_survives_deepcopy(small_real_hermitian):
    """Odd degrees ≥ 3 also interleave ``BE†`` layers — same invariant as the
    even-degree test: deepcopy must not raise and must still implement
    ``P(x) = x³``, not identity."""
    A = small_real_hermitian
    angles = QSPAngleFinder([0, 0, 0, 1]).QSVT()  # P(x) = x³
    qsvt = QSVTBlock(A, angles).build()

    qsvt_copy = copy.deepcopy(qsvt)

    n_anc = qsvt_copy.n_qubits - int(np.log2(A.shape[0]))
    sub = _project_ancilla_zero(_build_unitary(qsvt_copy), n_anc, qsvt_copy.n_qubits)
    expected = A @ A @ A
    assert np.linalg.norm(np.real(sub) - np.real(expected)) < 1e-10


# ── Surface API and validation ───────────────────────────────────────────


def test_rejects_non_square():
    with pytest.raises(NotImplementedError, match="square"):
        QSVTBlock(np.zeros((2, 3)), [0.0, 0.0])


def test_rejects_non_hermitian():
    A = np.array([[0.0, 1.0], [0.0, 0.0]])  # non-symmetric
    with pytest.raises(NotImplementedError, match="Hermitian"):
        QSVTBlock(A, [0.0, 0.0])


def test_unitarity_of_assembled_circuit(small_real_hermitian):
    """Independent of polynomial correctness: the QSVT circuit must be unitary."""
    A = small_real_hermitian
    angles = QSPAngleFinder([-1, 0, 1]).QSVT()
    qsvt = QSVTBlock(A, angles).build()
    U = _build_unitary(qsvt)
    err = np.linalg.norm(U @ U.conj().T - np.eye(U.shape[0]))
    assert err < 1e-10


# ── Polynomial correctness on larger / higher-degree / QO inputs ─────────


def _projected_target(qsvt: QSVTBlock, target_dim: int) -> np.ndarray:
    """Project the QSVT circuit onto ``ancilla = |0…0⟩``.  An ndarray ``A``
    is read in the qarpx LSB basis, so the sub-matrix is ``P(A)`` directly."""
    U = _build_unitary(qsvt)
    n_anc = qsvt.n_qubits - int(np.log2(target_dim))
    return _project_ancilla_zero(U, n_anc, qsvt.n_qubits)


def test_qsvt_applies_two_x_squared_minus_one(small_real_hermitian):
    """Degree-2 with non-unit leading coefficient: ``P(x) = 2x² − 1``."""
    A = small_real_hermitian
    angles = QSPAngleFinder([-1, 0, 2]).QSVT()
    qsvt = QSVTBlock(A, angles).build()
    sub = _projected_target(qsvt, A.shape[0])
    expected = 2 * A @ A - np.eye(A.shape[0])
    assert np.linalg.norm(np.real(sub) - np.real(expected)) < 1e-10


def test_qsvt_applies_x_cubed_2x2():
    """Odd-degree polynomial ``P(x) = x³`` on a 2×2 real-symmetric matrix."""
    np.random.seed(7)
    H = np.random.rand(2, 2)
    A = (H + H.T) / 2
    A = A / BlockEncodingBlock(A).lambda_factor()

    angles = QSPAngleFinder([0, 0, 0, 1]).QSVT()  # x³
    qsvt = QSVTBlock(A, angles).build()
    sub = _projected_target(qsvt, A.shape[0])
    assert np.linalg.norm(np.real(sub) - np.real(A @ A @ A)) < 1e-10


def test_qsvt_applies_chebyshev_t3_on_random_4x4():
    """Odd-degree ``5/2 x³ − 3/2 x`` (Chebyshev T₃) on a 4×4 random Hermitian."""
    np.random.seed(8)
    H = np.random.rand(4, 4)
    A = (H + H.T) / 2
    A = A / BlockEncodingBlock(A).lambda_factor()

    angles = QSPAngleFinder([0, -1.5, 0, 2.5]).QSVT()
    qsvt = QSVTBlock(A, angles).build()
    sub = _projected_target(qsvt, A.shape[0])
    expected = 2.5 * A @ A @ A - 1.5 * A
    assert np.linalg.norm(np.real(sub) - np.real(expected)) < 1e-10


@pytest.mark.parametrize("dim", [3, 5, 8])
def test_qsvt_2x_squared_minus_one_on_random_hermitian(dim):
    """``P(x) = 2x² − 1`` on a random ``dim × dim`` real-symmetric matrix.

    Covers non-power-of-two dimensions (3, 5) via the LCU zero-padding path,
    and 8×8 (3-qubit target) for a larger qubit count.
    """
    np.random.seed(9 + dim)
    H = np.random.rand(dim, dim)
    A = (H + H.T) / 2
    A = A / BlockEncodingBlock(A).lambda_factor()

    angles = QSPAngleFinder([-1, 0, 2]).QSVT()
    qsvt = QSVTBlock(A, angles).build()
    # n_target is the qubit count for the padded matrix.
    n_target_bits = int(np.ceil(np.log2(dim)))
    sub = _projected_target(qsvt, 2**n_target_bits)
    expected = 2 * A @ A - np.eye(A.shape[0])
    # Only the top-left dim×dim sub-block carries meaningful data.
    assert np.linalg.norm(np.real(sub[:dim, :dim]) - np.real(expected)) < 1e-10


def test_qsvt_qubit_operator_input():
    """``QubitOperator`` input path with a non-trivial odd-degree polynomial.

    Correctness check on ``P(x) = 5/2 x³ − 3/2 x``.
    The BE encodes the QO in qarpx LSB convention, so we compare the raw
    projected sub-matrix against ``P(A.sparse_matrix())`` — same basis, no
    re-indexing.  ``A`` is asymmetric under bit reversal.
    """

    A = QubitOperator("X0 Z1", 0.5) + QubitOperator("Y1 Y0", -0.3) + QubitOperator("", 0.4)
    lam = BlockEncodingBlock(A).lambda_factor()
    A_resc = A / lam

    angles = QSPAngleFinder([0, -1.5, 0, 2.5]).QSVT()
    qsvt = QSVTBlock(A_resc, angles).build()

    n_target_bits = 2  # max qubit index is 1 ⇒ 2-qubit target.
    U = _build_unitary(qsvt)
    sub = _project_ancilla_zero(U, qsvt.n_qubits - n_target_bits, qsvt.n_qubits)

    Amat = np.array(A_resc.sparse_matrix(n_target_bits).toarray(), dtype=complex)
    expected = 2.5 * Amat @ Amat @ Amat - 1.5 * Amat
    assert np.linalg.norm(np.real(sub) - np.real(expected)) < 1e-10
