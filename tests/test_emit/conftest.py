"""Shared emit-test machinery: one qarpx-side unitary and one per-SDK
unitary extractor, lifted from the four emitter test files so the
capability-contract suite and the per-SDK suites compare against the same
implementations.

Every extractor returns the unitary in qarp's LSB convention (qubit 0 =
least significant bit); the pytket and pennylane extractors fold in the
bit-reversal those SDKs need at the boundary.  SDK imports happen inside
the functions — callers guard with ``pytest.importorskip``.
"""

import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import SimpleBlock


def qarpx_unitary(cmds: list, n_qubits: int) -> np.ndarray:
    """Unitary of a command sequence via QarpSimulator, lowering to the
    native gate set first for gates csim can't dispatch directly (ECR,
    iSWAP, iSWAPdg go through their decompositions)."""
    sim = qx.QarpSimulator()
    try:
        return np.array(sim.unitary_matrix(cmds, n_qubits))
    except RuntimeError:
        b = SimpleBlock(n_qubits)
        b.set_commands(cmds)
        b.set_built(True)
        b._built = True
        lowered = b.optimize(qx.native_gateset())
        return np.array(sim.unitary_matrix(lowered.flatten(), n_qubits))


def block_unitary(block: SimpleBlock) -> np.ndarray:
    block.build()
    return qarpx_unitary(block.flatten(), block.n_qubits)


def qiskit_unitary(qc) -> np.ndarray:
    from qiskit.quantum_info import Operator

    return Operator(qc).data


def tket_unitary(circuit) -> np.ndarray:
    U = np.array(circuit.get_unitary())
    n = circuit.n_qubits
    if n > 1:
        # TKET get_unitary() is big-endian (qubit 0 = MSB); convert to
        # qarp's LSB via the bit-reversal permutation (§1).
        N = 2**n
        perm = [int(f"{i:0{n}b}"[::-1], 2) for i in range(N)]
        U = U[np.ix_(perm, perm)]
    return U


def qulacs_unitary(circuit, n_qubits: int) -> np.ndarray:
    # qulacs exposes no circuit unitary — build it column by column.
    import qulacs

    mat = np.eye(2**n_qubits, dtype=complex)
    state = qulacs.QuantumState(n_qubits)
    for col in range(2**n_qubits):
        vec = np.zeros(2**n_qubits, dtype=complex)
        vec[col] = 1.0
        state.load(vec.tolist())
        circuit.update_quantum_state(state)
        mat[:, col] = state.get_vector()
    return mat


def pennylane_unitary(tape, n_qubits: int) -> np.ndarray:
    import pennylane as qml

    # Reversed wire_order maps PennyLane's kron layout onto qarp's LSB.
    wire_order = list(range(n_qubits))[::-1]
    return qml.matrix(qml.tape.QuantumScript(tape.operations), wire_order=wire_order)


SDK_UNITARY = {
    "qiskit": lambda circ, n: qiskit_unitary(circ),
    "pytket": lambda circ, n: tket_unitary(circ),
    "qulacs": qulacs_unitary,
    "pennylane": pennylane_unitary,
}


@pytest.fixture
def sdk_unitary():
    """target name → callable(circuit, n_qubits) → LSB unitary ndarray."""
    return SDK_UNITARY
