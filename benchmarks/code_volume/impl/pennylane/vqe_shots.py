"""Shot-based VQE of H2 (STO-3G) with a UCCSD ansatz and qubit-wise-commuting grouping."""

import numpy as np
import pennylane as qml
from scipy.optimize import minimize

COORDINATES = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.735]])
N_ELECTRONS = 2
N_SHOTS, MAXITER, SEED = 100_000, 200, 7

hamiltonian, n_qubits = qml.qchem.molecular_hamiltonian(
    ["H", "H"], COORDINATES, basis="sto-3g", unit="angstrom", method="pyscf"
)
singles, doubles = qml.qchem.excitations(N_ELECTRONS, n_qubits)
s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
hf_state = qml.qchem.hf_state(N_ELECTRONS, n_qubits)

device = qml.device("default.qubit", wires=n_qubits, shots=N_SHOTS, seed=SEED)


@qml.qnode(device)
def energy(weights):
    qml.UCCSD(weights, range(n_qubits), s_wires=s_wires, d_wires=d_wires, init_state=hf_state)
    return qml.expval(hamiltonian)


result = minimize(
    energy,
    np.zeros(len(singles) + len(doubles)),
    method="COBYLA",
    options={"maxiter": MAXITER, "rhobeg": 0.1},
)

exact = np.linalg.eigvalsh(qml.matrix(hamiltonian, wire_order=range(n_qubits))).min()

print(f"n_qubits = {n_qubits}")
print(f"n_params = {len(singles) + len(doubles)}")
print(f"VQE-shots = {result.fun:.6f}")
print(f"exact     = {exact:.6f}")
