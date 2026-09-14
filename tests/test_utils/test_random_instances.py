import random

from qarp.utils import generate_random_circuit
from qarpx import GateType as OpType


def test_random_circuit_generator():
    n_gates = random.randint(1, 50)
    n_qubits = random.randint(2, 10)

    gate_set = [
        OpType.Rx,
        OpType.Ry,
        OpType.Rz,
        OpType.X,
        OpType.Y,
        OpType.Z,
        OpType.H,
        OpType.CX,
    ]

    circuit = generate_random_circuit(n_gates, n_qubits, gate_set)

    # ``generate_random_circuit`` returns a built ``SimpleBlock``
    # (qarpx).  Iterate the flat command stream and check each gate is in the
    # requested set.
    cmds = circuit.flatten()
    for cmd in cmds:
        assert cmd.gate in gate_set

    assert circuit.n_qubits == n_qubits
    assert len(cmds) == n_gates
