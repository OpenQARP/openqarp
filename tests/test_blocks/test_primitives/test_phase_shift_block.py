"""Smoke tests for PhaseShiftBlock — single-qubit ``P(θ)``."""

import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import PhaseShiftBlock


def _unitary(block):
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


@pytest.mark.parametrize("theta", [0.0, 0.7, np.pi / 3, -0.4])
def test_phase_shift_is_p_gate(theta):
    block = PhaseShiftBlock(phase=theta).build()
    U = _unitary(block)
    expected = np.diag([1.0, np.exp(1j * theta)])
    assert np.linalg.norm(U - expected) < 1e-12
