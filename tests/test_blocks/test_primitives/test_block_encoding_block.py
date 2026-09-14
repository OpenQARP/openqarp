"""Correctness tests for BlockEncodingBlock.

Pattern B composite — Prep_R · Select · Prep_L†.  We verify the standard
LCU contract: projecting the synthesised circuit on the ancilla = |0…0⟩
subspace recovers ``A / λ`` (within tolerance), where ``λ = Σ_i |c_i|``.

Phase handling note: each LCU phase ``φ_i = arg(c_i)`` is carried by the
corresponding ``SelectBlock`` entry and lowered through the multi-controlled
``GPhase`` decomposition; ``Prep`` loads the real amplitudes ``√(|c_i|/λ)``.
The mixed-sign tests below specifically exercise that path.
"""

import copy

import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import BlockEncodingBlock, SimpleBlock
from qarp.operators import QubitOperator


def _project_ancilla_zero(U: np.ndarray, n_anc: int, n_qubits: int) -> np.ndarray:
    """Extract the (target × target) sub-matrix of ``U`` on ``ancilla = |0…0⟩``.

    qarpx LSB convention: bits ``0..n_anc-1`` are the ancilla register, bits
    ``n_anc..n_qubits-1`` are the target.  Index ``i`` with ancilla = 0 has
    ``i = i_target · 2^n_anc``.
    """
    N_anc = 2**n_anc
    N_target = 2 ** (n_qubits - n_anc)
    idx = [it * N_anc for it in range(N_target)]
    return U[np.ix_(idx, idx)]


def _full_matrix(operator: QubitOperator, n_target: int) -> np.ndarray:
    """Dense matrix of ``operator`` in the qarpx LSB convention."""
    return np.array(operator.sparse_matrix(n_target).toarray(), dtype=complex)


def _build_unitary(be: BlockEncodingBlock) -> np.ndarray:
    return np.array(qx.QarpSimulator().unitary_matrix(be.flatten(), be.n_qubits))


# ── Single-qubit positive-coefficient operators ──────────────────────────


def test_single_pauli_x_only():
    """H = c·X — single-term LCU."""
    op = QubitOperator("X0", 0.5)
    be = BlockEncodingBlock(op).build()
    A = _full_matrix(op, 1)
    A_recon = (
        _project_ancilla_zero(_build_unitary(be), be.num_controls, be.n_qubits) * be.lambda_norm
    )
    assert np.linalg.norm(A_recon - A) < 1e-10


def test_two_term_positive_coeffs():
    """H = 0.7 X + 0.3 Z — both coefficients positive real."""
    op = QubitOperator("X0", 0.7) + QubitOperator("Z0", 0.3)
    be = BlockEncodingBlock(op).build()
    A = _full_matrix(op, 1)
    A_recon = (
        _project_ancilla_zero(_build_unitary(be), be.num_controls, be.n_qubits) * be.lambda_norm
    )
    assert np.linalg.norm(A_recon - A) < 1e-10


def test_two_term_mixed_signs():
    """H = 0.7 X − 0.3 Z — exercises the phase-splitting (φ ∈ {0, π}) path."""
    op = QubitOperator("X0", 0.7) + QubitOperator("Z0", -0.3)
    be = BlockEncodingBlock(op).build()
    A = _full_matrix(op, 1)
    A_recon = (
        _project_ancilla_zero(_build_unitary(be), be.num_controls, be.n_qubits) * be.lambda_norm
    )
    assert np.linalg.norm(A_recon - A) < 1e-10


def test_deepcopy_preserves_circuit():
    """The Prep† child must survive deepcopy with its daggered gates intact.

    Guards two failure modes: the raw C++-daggered child with no
    ``__deepcopy__`` (crash: "cannot pickle 'qarpx.Block' object"), and a
    mis-materialised dagger yielding an empty command stream (identity).
    """
    op = QubitOperator("X0", 0.7) + QubitOperator("Z0", -0.3)
    be = BlockEncodingBlock(op).build()
    be_copy = copy.deepcopy(be)
    A = _full_matrix(op, 1)
    A_recon = (
        _project_ancilla_zero(_build_unitary(be_copy), be_copy.num_controls, be_copy.n_qubits)
        * be_copy.lambda_norm
    )
    assert np.linalg.norm(A_recon - A) < 1e-10


def test_three_term_padded_ancilla():
    """3 LCU terms → ceil(log2(3)) = 2 ancilla qubits, last slot zero-padded."""
    op = QubitOperator("X0", 0.4) + QubitOperator("Y0", 0.3) + QubitOperator("Z0", 0.3)
    be = BlockEncodingBlock(op).build()
    assert be.num_controls == 2
    A = _full_matrix(op, 1)
    A_recon = (
        _project_ancilla_zero(_build_unitary(be), be.num_controls, be.n_qubits) * be.lambda_norm
    )
    assert np.linalg.norm(A_recon - A) < 1e-10


# ── Two-qubit operators ──────────────────────────────────────────────────


def test_two_qubit_xx_zz():
    op = QubitOperator("X0 X1", 0.6) + QubitOperator("Z0 Z1", 0.4)
    be = BlockEncodingBlock(op).build()
    A = _full_matrix(op, 2)
    A_recon = (
        _project_ancilla_zero(_build_unitary(be), be.num_controls, be.n_qubits) * be.lambda_norm
    )
    assert np.linalg.norm(A_recon - A) < 1e-10


def test_two_qubit_mixed_sign_pauli_sum():
    op = (
        QubitOperator("X0 X1", 0.5)
        + QubitOperator("Y0 Y1", -0.3)
        + QubitOperator("Z0", 0.2)
        + QubitOperator("Z1", 0.4)
    )
    be = BlockEncodingBlock(op).build()
    A = _full_matrix(op, 2)
    A_recon = (
        _project_ancilla_zero(_build_unitary(be), be.num_controls, be.n_qubits) * be.lambda_norm
    )
    assert np.linalg.norm(A_recon - A) < 1e-10


# ── λ and surface API ────────────────────────────────────────────────────


def test_lambda_norm_is_sum_of_abs_coeffs():
    op = QubitOperator("X0", 0.7) + QubitOperator("Z0", -0.3)
    be = BlockEncodingBlock(op)
    assert be.lambda_norm == pytest.approx(1.0)


def test_lambda_factor_compatibility_method():
    op = QubitOperator("X0", 0.7) + QubitOperator("Z0", 0.3)
    be = BlockEncodingBlock(op)
    assert be.lambda_factor() == be.lambda_norm


def test_unitaries_attribute_preserved():
    op = QubitOperator("X0", 0.5) + QubitOperator("Z0", -0.5)
    be = BlockEncodingBlock(op)
    assert len(be.unitaries) == 2
    # Each entry is (phase, pauli_string).
    phases = sorted(round(u[0], 6) for u in be.unitaries)
    paulis = sorted(u[1] for u in be.unitaries)
    assert phases == [0.0, round(np.pi, 6)]  # 0.5 → 0, −0.5 → π
    assert paulis == ["X", "Z"]


def test_rejects_invalid_input_type():
    with pytest.raises(TypeError, match="np.ndarray or QubitOperator"):
        BlockEncodingBlock("not an operator")


# ── np.ndarray input: general LCU decomposition path ────────────────────


def _projected_target(be: BlockEncodingBlock) -> np.ndarray:
    """Return ``λ · <0…0_anc| U |0…0_anc>``.

    An ndarray ``A`` is read in the qarpx LSB basis (LCU strings are
    qubit-ordered), so the projected sub-matrix is ``A`` itself — no
    bit-reversal on the target indices.
    """
    U = _build_unitary(be)
    return _project_ancilla_zero(U, be.num_controls, be.n_qubits) * be.lambda_norm


def test_ndarray_single_qubit_recovery():
    """1-qubit dense matrix → LCU → BE recovers A within tolerance."""
    np.random.seed(0)
    A = np.random.rand(2, 2)
    be = BlockEncodingBlock(A).build()
    assert np.linalg.norm(_projected_target(be) - A) < 1e-12


def test_ndarray_two_qubit_recovery():
    """2-qubit dense matrix — asymmetric under bit reversal, so a
    kron-order (MSB) reading of ``A`` would fail here."""
    np.random.seed(1)
    A = np.random.rand(4, 4)
    be = BlockEncodingBlock(A).build()
    assert np.linalg.norm(_projected_target(be) - A) < 1e-12


def test_ndarray_three_qubit_recovery():
    np.random.seed(2)
    A = np.random.rand(8, 8)
    be = BlockEncodingBlock(A).build()
    assert np.linalg.norm(_projected_target(be) - A) < 1e-12


def test_ndarray_complex_entries_recovery():
    """Random complex matrix — exercises the full phase-splitting path."""
    np.random.seed(3)
    A = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
    be = BlockEncodingBlock(A).build()
    assert np.linalg.norm(_projected_target(be) - A) < 1e-12


def test_ndarray_non_power_of_two_padded():
    """5×7 rectangular matrix is zero-padded to 8×8 inside LCU before BE."""
    np.random.seed(4)
    A = np.random.rand(5, 7)
    be = BlockEncodingBlock(A).build()
    # n_target = 3 (padded to 8×8); the original block sits in the top-left.
    assert be.n_qubits - be.num_controls == 3
    recon = _projected_target(be)
    assert np.linalg.norm(recon[:5, :7] - A) < 1e-12
    # Padded region should be zero.
    assert np.linalg.norm(recon[5:, :]) < 1e-12
    assert np.linalg.norm(recon[:, 7:]) < 1e-12


# ── n_controls / control_state metadata ─────────────────────────────────


# ── Explicit coefficients/unitaries input ────────────────────────────────


class _XGateBlock(SimpleBlock):
    def __init__(self):
        super().__init__(1, name="XGate")

    def build_vanilla(self) -> None:
        self.x(0)


class _ZGateBlock(SimpleBlock):
    def __init__(self):
        super().__init__(1, name="ZGate")

    def build_vanilla(self) -> None:
        self.z(0)


def test_explicit_lcu_matches_qubit_operator_input():
    op = QubitOperator("X0", 0.6) + QubitOperator("Z0", 0.4)
    be_op = BlockEncodingBlock(op).build()
    be_lcu = BlockEncodingBlock(unitaries=[(0.0, "X"), (0.0, "Z")], coefficients=[0.6, 0.4]).build()

    assert np.isclose(be_lcu.lambda_norm, be_op.lambda_norm)
    U_op = _project_ancilla_zero(_build_unitary(be_op), be_op.num_controls, be_op.n_qubits)
    U_lcu = _project_ancilla_zero(_build_unitary(be_lcu), be_lcu.num_controls, be_lcu.n_qubits)
    assert np.linalg.norm(U_op - U_lcu) < 1e-12


def test_explicit_lcu_block_entries():
    be = BlockEncodingBlock(
        unitaries=[_XGateBlock(), _ZGateBlock()],
        coefficients=[0.3, 0.7],
    ).build()

    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    encoded = (
        _project_ancilla_zero(_build_unitary(be), be.num_controls, be.n_qubits) * be.lambda_norm
    )
    assert np.linalg.norm(encoded - (0.3 * X + 0.7 * Z)) < 1e-12


def test_explicit_lcu_complex_coefficient_carries_phase():
    be = BlockEncodingBlock(unitaries=["X"], coefficients=[1j]).build()

    X = np.array([[0, 1], [1, 0]], dtype=complex)
    encoded = (
        _project_ancilla_zero(_build_unitary(be), be.num_controls, be.n_qubits) * be.lambda_norm
    )
    assert np.linalg.norm(encoded - 1j * X) < 1e-12


def test_explicit_lcu_dict_entries_pad_with_identity():
    be = BlockEncodingBlock(
        unitaries=[(0.0, "XX"), (0.0, {0: "Z"})],
        coefficients=[0.5, 0.5],
    ).build()
    assert be.n_qubits == 3  # one ancilla + two target qubits

    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    expected = 0.5 * np.kron(X, X) + 0.5 * np.kron(np.eye(2), Z)
    encoded = (
        _project_ancilla_zero(_build_unitary(be), be.num_controls, be.n_qubits) * be.lambda_norm
    )
    assert np.linalg.norm(encoded - expected) < 1e-12


def test_explicit_lcu_input_validation():
    with pytest.raises(TypeError, match="both coefficients and unitaries"):
        BlockEncodingBlock(unitaries=[(0.0, "X")])

    with pytest.raises(TypeError, match="both coefficients and unitaries"):
        BlockEncodingBlock(coefficients=[1.0])

    with pytest.raises(TypeError, match="not both"):
        BlockEncodingBlock(QubitOperator("X0", 1.0), unitaries=[(0.0, "X")], coefficients=[1.0])

    with pytest.raises(ValueError, match="cannot be empty"):
        BlockEncodingBlock(unitaries=[], coefficients=[])

    with pytest.raises(ValueError, match="same length"):
        BlockEncodingBlock(unitaries=[(0.0, "X")], coefficients=[1.0, 2.0])

    with pytest.raises(ValueError, match="finite"):
        BlockEncodingBlock(unitaries=[(0.0, "X")], coefficients=[np.inf])

    # The old positional shortcuts are gone: lists are not operators.
    with pytest.raises(TypeError, match="np.ndarray or QubitOperator"):
        BlockEncodingBlock([(0.0, "X")], [1.0])
