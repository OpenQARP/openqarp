"""Tests for ReflectionBlock — Grover-style reflection about |0…0⟩."""

import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import ReflectionBlock


def _unitary(block):
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


def _expected_reflection(n: int) -> np.ndarray:
    """2|0…0⟩⟨0…0| − I — diag(+1, −1, −1, …, −1)."""
    M = -np.eye(2**n, dtype=complex)
    M[0, 0] = 1.0
    return M


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_reflection_matches_exact_unitary(n):
    """``ReflectionBlock(n)`` must reproduce ``2|0…0⟩⟨0…0| − I`` exactly
    (no global phase fudge — the GPhase(π) inside the build is precisely the
    sign flip that makes the matrix come out right)."""
    block = ReflectionBlock(n).build()
    assert np.linalg.norm(_unitary(block) - _expected_reflection(n)) < 1e-12


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_reflection_squared_is_identity(n):
    """A reflection is an involution: ``R² = I`` exactly (not just up to phase)."""
    block = ReflectionBlock(n).build()
    U = _unitary(block)
    assert np.linalg.norm(U @ U - np.eye(2**n)) < 1e-12


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_reflection_is_unitary(n):
    block = ReflectionBlock(n).build()
    U = _unitary(block)
    assert np.linalg.norm(U @ U.conj().T - np.eye(2**n)) < 1e-12


def test_n_qubits_zero_raises():
    with pytest.raises(ValueError, match="n_qubits >= 1"):
        ReflectionBlock(0).build()
