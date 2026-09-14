"""Smoke tests for HnBlock — Hadamard on every qubit."""

import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import HnBlock


def _unitary(block):
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_hn_unitary_is_hadamard_tensor_product(n):
    """``HnBlock(n)`` implements ``H⊗H⊗…⊗H`` (n times)."""
    block = HnBlock(n).build()
    U = _unitary(block)
    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    expected = H
    for _ in range(n - 1):
        expected = np.kron(H, expected)
    assert np.linalg.norm(U - expected) < 1e-12


def test_hn_construction_and_flatten():
    block = HnBlock(3).build()
    cmds = block.flatten()
    assert len(cmds) == 3
    for cmd in cmds:
        assert cmd.gate.name == "H"
