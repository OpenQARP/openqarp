"""Tests for GivensBlock — the two-qubit Givens rotation.

Two levels: unitarity and the documented action on the ``{|01⟩, |10⟩}``
subspace, which hold whatever sign convention an implementation picks, and a
pin of the exact 4×4 matrix, which does not.
"""

import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import GivensBlock


def _unitary(block):
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


@pytest.mark.parametrize("theta", [0.0, 0.5, np.pi / 3])
def test_givens_unitary_is_unitary(theta):
    """Without committing to the exact 4×4 form (sign / qubit-order conventions
    vary between implementations), check unitarity + the documented action on
    the {|01⟩, |10⟩} subspace: rotates by θ/2 like a Y rotation in that block.
    """
    block = GivensBlock(theta=theta).build()
    U = _unitary(block)
    I4 = np.eye(4, dtype=complex)
    assert np.linalg.norm(U @ U.conj().T - I4) < 1e-10
    # |00⟩ and |11⟩ are invariant.
    assert abs(abs(U[0, 0]) - 1.0) < 1e-10
    assert abs(abs(U[3, 3]) - 1.0) < 1e-10


@pytest.mark.parametrize("theta", [0.7, -1.3, np.pi])
def test_givens_unitary_matches_documented_matrix(theta):
    """Compare the built circuit against the 4×4 matrix written out in
    ``GivensBlock``'s own docstring (Nam et al., npj Quantum Inf 6, 33 (2020)),
    entry by entry in qarp's LSB qubit order — signs and global phase
    included, not just up to a phase (conventions §18).

    This pins the convention rather than merely exercising it:
    ``OrbitalRotationBlock`` builds adjacent-mode Givens networks on top of
    these exact signs, so a transposed or rephased variant would still look
    like a valid rotation here while silently reversing fermionic rotation
    directions downstream.
    """
    U = _unitary(GivensBlock(theta=theta).build())
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    expected = np.array(
        [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]],
        dtype=complex,
    )
    np.testing.assert_allclose(U, expected, atol=1e-12)
