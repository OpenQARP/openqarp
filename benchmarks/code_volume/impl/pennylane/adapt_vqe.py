"""ADAPT-VQE ground-state energy of LiH (STO-3G) in a (2e, 3o) active space."""

import numpy as np
import pennylane as qml

COORDINATES = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.59]])
N_ACTIVE_ELECTRONS, N_ACTIVE_ORBITALS = 2, 3
GRADIENT_THRESH, MAX_CYCLES = 1e-6, 50

hamiltonian, n_qubits = qml.qchem.molecular_hamiltonian(
    ["H", "Li"],
    COORDINATES,
    basis="sto-3g",
    unit="angstrom",
    method="pyscf",
    active_electrons=N_ACTIVE_ELECTRONS,
    active_orbitals=N_ACTIVE_ORBITALS,
)
singles, doubles = qml.qchem.excitations(N_ACTIVE_ELECTRONS, n_qubits)
pool = [qml.DoubleExcitation(0.0, wires=wires) for wires in doubles]
pool += [qml.SingleExcitation(0.0, wires=wires) for wires in singles]
hf_state = qml.qchem.hf_state(N_ACTIVE_ELECTRONS, n_qubits)


@qml.qnode(qml.device("default.qubit", wires=n_qubits))
def circuit():
    qml.BasisState(hf_state, wires=range(n_qubits))
    return qml.expval(hamiltonian)


optimizer = qml.AdaptiveOptimizer()
energy = None
for _ in range(MAX_CYCLES):
    circuit, energy, gradient = optimizer.step_and_cost(circuit, pool, drain_pool=True)
    if gradient < GRADIENT_THRESH:
        break

exact = np.linalg.eigvalsh(qml.matrix(hamiltonian, wire_order=range(n_qubits))).min()

print(f"pool size = {len(pool)}")
print(f"ADAPT-VQE = {energy:.10f}")
print(f"exact     = {exact:.10f}")
