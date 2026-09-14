"""Smoke tests for LinearEntanglingBlock — linear CX/CZ chain."""

import pytest

import qarpx as qx
from qarp.blocks import LinearEntanglingBlock


@pytest.mark.parametrize(
    "circular,use_cz", [(False, False), (False, True), (True, False), (True, True)]
)
def test_linear_entangling_constructs_and_builds(circular, use_cz):
    n = 4
    block = LinearEntanglingBlock(n_qubits=n, circular=circular, use_cz=use_cz).build()
    assert block.is_built
    assert block.n_qubits == n
    cmds = block.flatten()
    expected_count = n if circular else n - 1
    assert len(cmds) == expected_count
    expected_gate = qx.GateType.CZ if use_cz else qx.GateType.CX
    for cmd in cmds:
        assert cmd.gate == expected_gate


# ── Circular wrap: only a real ring appends the wrap edge ────────────────


@pytest.mark.parametrize("use_cz", [True, False])
def test_linear_two_qubit_ring_is_the_two_qubit_chain(use_cz):
    """Regression: at n=2 the wrap duplicated the chain's only edge, and for
    CZ the pair cancelled to an identity entangler (CZ² = 1)."""
    block = LinearEntanglingBlock(n_qubits=2, circular=True, use_cz=use_cz).build()
    assert [tuple(cmd.qubits) for cmd in block.flatten()] == [(0, 1)]


def test_linear_one_qubit_ring_has_no_edges():
    """n=1 circular used to append a (0, 0) self-pair."""
    block = LinearEntanglingBlock(n_qubits=1, circular=True, use_cz=True).build()
    assert list(block.flatten()) == []


def test_linear_three_qubit_ring_keeps_the_wrap():
    """The guard must not trim real rings."""
    block = LinearEntanglingBlock(n_qubits=3, circular=True, use_cz=False).build()
    assert [tuple(cmd.qubits) for cmd in block.flatten()] == [(0, 1), (1, 2), (2, 0)]
