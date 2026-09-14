"""Smoke tests for IdentityBlock."""

import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import IdentityBlock


def _unitary(block):
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


@pytest.mark.parametrize("n", [1, 2, 3])
def test_identity_unitary_is_identity(n):
    block = IdentityBlock(n).build()
    U = _unitary(block)
    assert np.linalg.norm(U - np.eye(2**n)) < 1e-12


def test_identity_flatten_returns_list():
    """No assertion on cmd count — the unitary check above is the contract."""
    block = IdentityBlock(3).build()
    assert isinstance(block.flatten(), list)
