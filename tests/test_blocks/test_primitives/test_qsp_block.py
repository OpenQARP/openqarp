"""Smoke tests for QSPBlock — Quantum Signal Processing."""

import pytest

from qarp.blocks import QSPBlock


def test_qsp_constructs_and_builds():
    angles = [0.1, 0.2, 0.3]
    block = QSPBlock(a=0.5, P_angles=angles).build()
    assert block.is_built
    assert block.n_qubits == 1  # QSP is single-qubit
    cmds = block.flatten()
    assert len(cmds) > 0


@pytest.mark.parametrize("n_angles", [3, 5, 7])
def test_qsp_angle_count(n_angles):
    angles = [0.1 * i for i in range(n_angles)]
    block = QSPBlock(a=0.3, P_angles=angles).build()
    assert block.is_built
