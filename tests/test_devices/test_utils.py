"""Device gate-set helper wrappers around the C++ full gate sets."""

import qarpx as qx
from qarp.devices._utils import get_full_gate_set_1q, get_full_gate_set_2q


def test_full_gate_set_1q_contains_single_qubit_gates():
    gs = get_full_gate_set_1q()
    assert isinstance(gs, qx.GateSet)
    assert gs.contains(qx.GateType.H)
    assert gs.contains(qx.GateType.Rz)
    assert not gs.contains(qx.GateType.CX)


def test_full_gate_set_2q_contains_two_qubit_gates():
    gs = get_full_gate_set_2q()
    assert isinstance(gs, qx.GateSet)
    assert gs.contains(qx.GateType.CX)
    assert not gs.contains(qx.GateType.H)
