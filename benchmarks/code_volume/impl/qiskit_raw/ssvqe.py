"""SS-VQE: three lowest eigenvalues of H2 (STO-3G) from one hardware-efficient ansatz."""

import numpy as np
from common import basis_index, build_hea, energy, molecular_hamiltonian
from openfermion import get_sparse_operator
from scipy.optimize import minimize

GEOMETRY = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.735))]
BASIS_STATES = [[], [0], [1]]
WEIGHTS = [4, 2, 1]
N_LAYERS = 6

observable, qubit_hamiltonian, molecule = molecular_hamiltonian(GEOMETRY)
n_qubits = molecule.n_qubits

circuit, parameters = build_hea(n_qubits, N_LAYERS)
references = [basis_index(occupied) for occupied in BASIS_STATES]


def weighted_energy(values):
    return sum(
        weight * energy(circuit, parameters, observable, values, reference)
        for weight, reference in zip(WEIGHTS, references, strict=True)
    )


result = minimize(
    weighted_energy,
    np.random.default_rng(7).uniform(-0.1, 0.1, len(parameters)),
    method="BFGS",
    options={"maxiter": 3000},
)

energies = [
    energy(circuit, parameters, observable, result.x, reference) for reference in references
]
exact = np.linalg.eigvalsh(get_sparse_operator(qubit_hamiltonian).toarray())

print(f"n_params = {len(parameters)}")
print("SS-VQE  =", np.array2string(np.sort(energies), precision=8))
print("exact   =", np.array2string(exact[:3], precision=8))
