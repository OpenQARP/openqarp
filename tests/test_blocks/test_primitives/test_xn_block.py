import qarpx as qx
from qarp.blocks import XnBlock


def test_xn_composition():
    n_qubits = 5
    block = XnBlock(n_qubits=n_qubits).build()

    assert block.n_qubits == n_qubits
    assert block.name == "Xn"
    cmds = list(block.commands())
    assert len(cmds) == n_qubits
    assert all(qx.gate_name(c.gate) == "X" for c in cmds)


def test_xn_composition_partial():
    n_qubits = 3
    target_qubits = [0, 2, 3]
    block = XnBlock(n_qubits=n_qubits, target_qubits=target_qubits).build()

    assert block.n_qubits == n_qubits
    assert block.name == "Xn"
    cmds = list(block.commands())
    assert len(cmds) == n_qubits
    assert all(qx.gate_name(c.gate) == "X" for c in cmds)
