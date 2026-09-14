"""SS-VQE: three lowest eigenvalues of H2 (STO-3G) from one hardware-efficient ansatz."""

import numpy as np
from common import build_hea, energy, molecular_hamiltonian, prepared_state
from openfermion import get_sparse_operator
from scipy.optimize import minimize

GEOMETRY = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.735))]
BASIS_STATES = [[], [0], [1]]
WEIGHTS = [4, 2, 1]
N_LAYERS = 6

observable, qubit_hamiltonian, molecule = molecular_hamiltonian(GEOMETRY)
n_qubits = molecule.n_qubits

circuit, layout = build_hea(n_qubits, N_LAYERS)
references = [prepared_state(n_qubits, occupied) for occupied in BASIS_STATES]


def weighted_energy(parameters):
    return sum(
        weight * energy(circuit, layout, observable, parameters, reference)
        for weight, reference in zip(WEIGHTS, references, strict=True)
    )


result = minimize(
    weighted_energy,
    np.random.default_rng(7).uniform(-0.1, 0.1, len(layout)),
    method="BFGS",
    options={"maxiter": 3000},
)

energies = [energy(circuit, layout, observable, result.x, reference) for reference in references]
exact = np.linalg.eigvalsh(get_sparse_operator(qubit_hamiltonian).toarray())

print(f"n_params = {len(layout)}")
print("SS-VQE  =", np.array2string(np.sort(energies), precision=8))
print("exact   =", np.array2string(exact[:3], precision=8))
