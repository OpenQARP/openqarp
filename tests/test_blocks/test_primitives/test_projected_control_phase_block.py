"""Correctness tests for ProjectedControlPhaseBlock.

Pattern A leaf — delegates to ``self.diagonal_unitary(...)``.  Verifies the
realised circuit matches the diagonal `[e^{+iφ}] * dim + [e^{-iφ}] * (N-dim)`.
"""

import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import ProjectedControlPhaseBlock


def _unitary(block):
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


def _expected_diagonal(phase: float, dim: int, n_qubits: int) -> np.ndarray:
    N = 2**n_qubits
    plus = np.exp(1j * phase)
    minus = np.exp(-1j * phase)
    return np.array([plus] * dim + [minus] * (N - dim), dtype=complex)


def _max_diff_up_to_phase(actual_diag, expected_diag):
    nz = np.flatnonzero(np.abs(expected_diag) > 1e-6)
    k = int(nz[0])
    phase = expected_diag[k] / actual_diag[k]
    phase /= abs(phase)
    return float(np.abs(actual_diag * phase - expected_diag).max())


@pytest.mark.parametrize("phase", [0.0, 0.137, -0.42, 1.7])
def test_one_qubit_dim1(phase):
    """1-qubit, dim=1: diag = (e^{+iφ}, e^{-iφ})."""
    b = ProjectedControlPhaseBlock(phase=phase, dim=1, n_qubits=1).build()
    U = _unitary(b)
    # Should be diagonal.
    assert np.linalg.norm(U - np.diag(np.diag(U))) < 1e-12
    expected = _expected_diagonal(phase, 1, 1)
    assert _max_diff_up_to_phase(np.diag(U), expected) < 1e-10


@pytest.mark.parametrize("dim", [0, 1, 2, 3, 4])
def test_two_qubit_varied_dim(dim):
    phase = 0.31
    b = ProjectedControlPhaseBlock(phase=phase, dim=dim, n_qubits=2).build()
    U = _unitary(b)
    assert np.linalg.norm(U - np.diag(np.diag(U))) < 1e-12
    expected = _expected_diagonal(phase, dim, 2)
    assert _max_diff_up_to_phase(np.diag(U), expected) < 1e-10


def test_three_qubit_partial_subspace():
    phase = 0.65
    b = ProjectedControlPhaseBlock(phase=phase, dim=3, n_qubits=3).build()
    U = _unitary(b)
    expected = _expected_diagonal(phase, 3, 3)
    assert _max_diff_up_to_phase(np.diag(U), expected) < 1e-10


def test_full_subspace_is_pure_global_phase():
    """dim = 2^n: every entry gets e^{+iφ} → just a global phase, no
    structural rotation."""
    phase = 0.42
    b = ProjectedControlPhaseBlock(phase=phase, dim=4, n_qubits=2).build()
    U = _unitary(b)
    # Up to global phase, U should equal the identity.
    expected = _expected_diagonal(phase, 4, 2)
    assert _max_diff_up_to_phase(np.diag(U), expected) < 1e-10


def test_zero_phase_is_identity():
    b = ProjectedControlPhaseBlock(phase=0.0, dim=2, n_qubits=2).build()
    U = _unitary(b)
    assert np.linalg.norm(U - np.eye(4, dtype=complex)) < 1e-12


def test_dim_too_large_raises():
    with pytest.raises(ValueError, match="dim cannot be larger"):
        ProjectedControlPhaseBlock(phase=0.5, dim=5, n_qubits=2)
