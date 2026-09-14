"""Shot-based VQE of H2 (STO-3G) with a UCCSD ansatz and qubit-wise-commuting grouping."""

import numpy as np
from common import (
    basis_index,
    build_ansatz,
    molecular_hamiltonian,
    qubit_wise_groups,
    sampled_energy,
    uccsd_generators,
)
from openfermion import get_sparse_operator
from scipy.optimize import minimize

GEOMETRY = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.735))]
N_SHOTS, MAXITER, SEED = 100_000, 200, 7

_, qubit_hamiltonian, molecule = molecular_hamiltonian(GEOMETRY)
n_qubits, n_electrons = molecule.n_qubits, molecule.n_electrons

generators = uccsd_generators(n_qubits, n_electrons)
circuit, parameters = build_ansatz(n_qubits, generators)
reference = basis_index(range(n_electrons))

groups = qubit_wise_groups(qubit_hamiltonian)
constant = qubit_hamiltonian.terms.get((), 0.0).real

result = minimize(
    lambda values: sampled_energy(
        circuit, parameters, groups, constant, values, reference, N_SHOTS, SEED
    ),
    np.zeros(len(generators)),
    method="COBYLA",
    options={"maxiter": MAXITER, "rhobeg": 0.1},
)

exact = np.linalg.eigvalsh(get_sparse_operator(qubit_hamiltonian).toarray()).min()

print(f"n_qubits = {n_qubits}")
print(f"n_params = {len(generators)}")
print(f"n_groups = {len(groups)}")
print(f"VQE-shots = {result.fun:.6f}")
print(f"exact     = {exact:.6f}")
