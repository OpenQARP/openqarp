"""Smoke tests for QFTBlock."""

import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import QFTBlock


def _unitary(block):
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


def _qft_matrix(n: int) -> np.ndarray:
    """Canonical textbook QFT matrix (+ω convention).

    ``F[j, k] = (1/√N) · e^{+2πi · jk/N}``.  OpenQARP's ``QFTBlock`` implements
    this convention; the previous IQFT (-ω) shape was a latent bug masked
    because QPE consumers daggered the block to recover IQFT.
    """
    N = 2**n
    F = np.zeros((N, N), dtype=complex)
    for j in range(N):
        for k in range(N):
            F[j, k] = np.exp(2j * np.pi * j * k / N)
    return F / np.sqrt(N)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_qft_unitary_matches_reference(n):
    """``QFTBlock(n)`` must implement the +ω textbook QFT exactly (not the IQFT)."""
    block = QFTBlock(n).build()
    U = _unitary(block)
    F_plus = _qft_matrix(n)
    err_plus = np.linalg.norm(U - F_plus)
    # Tripwire: if a future refactor flips the sign back to -ω the error
    # message should point at the actual culprit, not just report a generic
    # mismatch.
    err_minus = np.linalg.norm(U - F_plus.conj())
    assert err_plus < 1e-10, (
        f"QFTBlock no longer matches +ω textbook QFT (err={err_plus:.2e}); "
        f"err vs IQFT (-ω) = {err_minus:.2e}.  Check the sign on the CP "
        f"angle in qft_block.py — it should be +π / 2^(i-j)."
    )


@pytest.mark.parametrize("n", [2, 3])
def test_qft_is_unitary(n):
    block = QFTBlock(n).build()
    U = _unitary(block)
    assert np.linalg.norm(U @ U.conj().T - np.eye(2**n)) < 1e-10
