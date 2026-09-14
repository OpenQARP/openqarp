"""Correctness tests for QubitizationBlock.

Pattern B composite — ``W = BE · R`` where ``R = 2|0…0⟩⟨0…0| - I`` acts on the
LCU control register and ``BE`` block-encodes ``A``.  Order matters:
``ReflectionBlock`` is added first (so it runs first in circuit time), then
``BlockEncodingBlock``.
"""

import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import (
    BlockEncodingBlock,
    QubitizationBlock,
    ReflectionBlock,
)
from qarp.operators import QubitOperator


def _unitary(block):
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


def _walk_reference(H: QubitOperator) -> tuple:
    """Build the reference walk operator ``BE · R`` from independently
    constructed BE and Reflection blocks."""
    be = BlockEncodingBlock(H)
    be.build()
    BE_U = _unitary(be)

    n_anc = be.num_controls
    refl = ReflectionBlock(n_anc)
    refl.build()
    R_U = _unitary(refl)

    n_qubits = be.n_qubits
    N_target = 2 ** (n_qubits - n_anc)
    # qarpx LSB convention: ancilla qubits are the low ones, so embed R on the
    # right of the kron product.
    R_full = np.kron(np.eye(N_target, dtype=complex), R_U)
    return BE_U @ R_full, n_qubits


# ── Walk-operator equivalence ────────────────────────────────────────────


def test_walk_operator_matches_be_dot_reflection_4_term():
    H = (
        QubitOperator("X0 X1", 0.5)
        + QubitOperator("Y0 Y1", -0.3)
        + QubitOperator("Z0", 0.2)
        + QubitOperator("Z1", 0.4)
    )
    qb = QubitizationBlock(H).build()
    W_expected, _ = _walk_reference(H)
    assert np.linalg.norm(_unitary(qb) - W_expected) < 1e-10


def test_walk_operator_3_term_padded():
    """3 LCU terms → n_anc = 2 (padded)."""
    H = QubitOperator("X0 X1", 0.4) + QubitOperator("Z0 Z1", 0.3) + QubitOperator("Y0 Y1", 0.3)
    qb = QubitizationBlock(H).build()
    W_expected, _ = _walk_reference(H)
    assert np.linalg.norm(_unitary(qb) - W_expected) < 1e-10


def test_walk_operator_is_unitary():
    H = (
        QubitOperator("X0 X1", 0.5)
        + QubitOperator("Y0 Y1", -0.3)
        + QubitOperator("Z0 Z1", 0.4)
        + QubitOperator("Z0", 0.2)
    )
    qb = QubitizationBlock(H).build()
    U = _unitary(qb)
    err = np.linalg.norm(U @ U.conj().T - np.eye(U.shape[0]))
    assert err < 1e-10


def test_walk_operator_2_term_lcu():
    """2-term LCU → n_anc = 1.  Exercises the ReflectionBlock(1) path (which
    short-circuits to a plain Z) that earlier qubitization builds couldn't
    reach because ``mcz`` requires ≥2 qubits."""
    H = QubitOperator("X0", 0.7) + QubitOperator("Z0", -0.3)
    qb = QubitizationBlock(H).build()
    W_expected, _ = _walk_reference(H)
    assert np.linalg.norm(_unitary(qb) - W_expected) < 1e-10


# ── Surface API ──────────────────────────────────────────────────────────


def test_lambda_factor_matches_be():
    H = QubitOperator("X0", 0.7) + QubitOperator("Z0", -0.3)
    qb = QubitizationBlock(H)
    assert qb.lambda_factor == qb.BE.lambda_norm
    assert qb.lambda_factor == pytest.approx(1.0)


def test_unitaries_qubits_attribute():
    H = (
        QubitOperator("X0 X1", 0.5)
        + QubitOperator("Y0 Y1", -0.3)
        + QubitOperator("Z0", 0.2)
        + QubitOperator("Z1", 0.4)
    )
    qb = QubitizationBlock(H)
    # 4 LCU terms → n_anc = 2; target qubits are [n_anc, n_qubits) = [2, 4) = [2, 3].
    assert qb.unitaries_qubits == [2, 3]


def test_be_attribute_preserved():
    H = QubitOperator("X0 X1", 0.5) + QubitOperator("Z0 Z1", 0.5)
    qb = QubitizationBlock(H)
    assert isinstance(qb.BE, BlockEncodingBlock)
    assert qb.BE.lambda_norm == 1.0


def test_rejects_invalid_input_type():
    with pytest.raises(TypeError, match="np.ndarray or QubitOperator"):
        QubitizationBlock("nope")


# ── ndarray input path ──────────────────────────────────────────────────


def _walk_reference_ndarray(A: np.ndarray) -> tuple:
    """``W = BE · R`` reference built from independent BE/Reflection blocks
    using an ``np.ndarray`` input.  Mirrors ``_walk_reference`` but for the
    ndarray-input LCU path."""
    be = BlockEncodingBlock(A)
    be.build()
    BE_U = _unitary(be)

    n_anc = be.num_controls
    refl = ReflectionBlock(n_anc)
    refl.build()
    R_U = _unitary(refl)

    n_qubits = be.n_qubits
    N_target = 2 ** (n_qubits - n_anc)
    R_full = np.kron(np.eye(N_target, dtype=complex), R_U)
    return BE_U @ R_full, n_qubits


def test_walk_operator_ndarray_input_2qubit():
    """Random 2-qubit ndarray ⇒ Qubitization recovers ``BE · R``."""
    np.random.seed(0)
    A = np.random.rand(4, 4)
    qb = QubitizationBlock(A).build()
    W_expected, _ = _walk_reference_ndarray(A)
    assert np.linalg.norm(_unitary(qb) - W_expected) < 1e-10


def test_walk_operator_ndarray_is_unitary():
    np.random.seed(1)
    H = np.random.rand(4, 4)
    H = (H + H.T) / 2
    qb = QubitizationBlock(H).build()
    U = _unitary(qb)
    assert np.linalg.norm(U @ U.conj().T - np.eye(U.shape[0])) < 1e-10


# ── Spectral property of the walk operator ──────────────────────────────


def test_walk_operator_eigenvalues_at_arccos_of_normalized_spectrum():
    """Qubitization theorem: for a Hermitian ``H`` with eigenvalues ``E_k`` and
    block-encoding norm ``λ``, the walk operator ``W = BE · R`` has
    ``e^{±i·arccos(E_k/λ)}`` in its spectrum.  This is the *physical* property
    qubitization is built for — pinning it rules out subtle wiring errors that
    leave ``W`` unitary but not the right walk."""
    H = QubitOperator("X0", 0.7) + QubitOperator("Z0", 0.3)
    lam = BlockEncodingBlock(H).lambda_norm

    H_mat = np.array(H.sparse_matrix(1).toarray(), dtype=complex)
    H_eigs = np.linalg.eigvalsh(H_mat)
    expected_phases = sorted(np.concatenate([np.arccos(H_eigs / lam), -np.arccos(H_eigs / lam)]))

    qb = QubitizationBlock(H).build()
    W = _unitary(qb)
    walk_phases = sorted(np.angle(np.linalg.eigvals(W)))

    # 2 H eigenvalues → 4 walk eigenvalues (±arccos for each).  W on 2 qubits
    # has 4 eigenvalues total — all of them encode H's spectrum.
    assert len(walk_phases) == len(expected_phases)
    for got, want in zip(walk_phases, expected_phases, strict=True):
        assert abs(got - want) < 1e-10, f"got {got}, want {want}"


# ── Surface API ─────────────────────────────────────────────────────────


def test_operator_name_propagates_to_inner_be():
    """``operator_name`` should flow through to the inner ``BlockEncodingBlock.name``."""
    H = QubitOperator("X0", 0.5) + QubitOperator("Z0", 0.5)
    qb = QubitizationBlock(H, operator_name="MyOp")
    assert qb.BE.name == "MyOp"
