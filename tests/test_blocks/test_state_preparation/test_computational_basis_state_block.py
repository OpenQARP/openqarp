import qarpx as qx
from qarp.blocks import ComputationalBasisStateBlock


def test_state_preparation():
    n_qubits = 4
    state = [0, 0, 0, 1]
    wfn = ComputationalBasisStateBlock(state, target_qubits=list(range(n_qubits)))
    wfn.build()

    assert wfn.is_built
    assert wfn.is_built
    assert wfn.n_qubits == n_qubits

    # Verify only qubit 3 has an X gate
    sim = qx.QarpSimulator()
    sv = sim.statevector(wfn.commands(), n_qubits)
    # qubit 3 = 1 → index = 2^3 = 8 (little-endian: qubit i is bit i)
    assert abs(sv[8]) > 0.99


def test_all_zeros_state():
    state = [0, 0, 0, 0]
    block = ComputationalBasisStateBlock(state)
    block.build()
    assert block.is_built
    # No X gates: empty circuit, should be identity on |0000>
    sim = qx.QarpSimulator()
    sv = sim.statevector(block.commands(), 4)
    assert abs(sv[0]) > 0.99


def test_all_ones_state():
    state = [1, 1, 1, 1]
    block = ComputationalBasisStateBlock(state)
    block.build()
    sim = qx.QarpSimulator()
    sv = sim.statevector(block.commands(), 4)
    # |1111> has index 15 (all bits set)
    assert abs(sv[15]) > 0.99
