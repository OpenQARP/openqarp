"""SS-VQE: three lowest eigenvalues of H2 (STO-3G) from one hardware-efficient ansatz."""

import numpy as np
import pennylane as qml
from scipy.optimize import minimize

COORDINATES = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.735]])
BASIS_STATES = [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]]
WEIGHTS = [4, 2, 1]
N_LAYERS = 6

hamiltonian, n_qubits = qml.qchem.molecular_hamiltonian(
    ["H", "H"], COORDINATES, basis="sto-3g", unit="angstrom", method="pyscf"
)


@qml.qnode(qml.device("default.qubit", wires=n_qubits))
def energy(parameters, basis_state):
    qml.BasisState(np.array(basis_state), wires=range(n_qubits))
    qml.BasicEntanglerLayers(
        parameters.reshape(N_LAYERS, n_qubits), wires=range(n_qubits), rotation=qml.RY
    )
    return qml.expval(hamiltonian)


result = minimize(
    lambda parameters: sum(
        weight * energy(parameters, state)
        for weight, state in zip(WEIGHTS, BASIS_STATES, strict=True)
    ),
    np.random.default_rng(7).uniform(-0.1, 0.1, N_LAYERS * n_qubits),
    method="BFGS",
    options={"maxiter": 3000},
)

energies = [energy(result.x, state) for state in BASIS_STATES]
exact = np.linalg.eigvalsh(qml.matrix(hamiltonian, wire_order=range(n_qubits)))

print(f"n_params = {N_LAYERS * n_qubits}")
print("SS-VQE  =", np.array2string(np.sort(energies), precision=8))
print("exact   =", np.array2string(exact[:3], precision=8))
