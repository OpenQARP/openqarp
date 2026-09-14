"""VQE ground-state energy of a linear H4 chain (STO-3G) with a UCCSD ansatz."""

import cirq
import numpy as np
from common import basis_index, build_ansatz, energy, molecular_hamiltonian, uccsd_generators
from scipy.optimize import minimize

GEOMETRY = [("H", (0.0, 0.0, z)) for z in (0.0, 1.0, 2.0, 3.0)]
MAXITER = 3000

observable, _, molecule = molecular_hamiltonian(GEOMETRY, run_ccsd=True)
n_qubits, n_electrons = molecule.n_qubits, molecule.n_electrons
qubits = cirq.LineQubit.range(n_qubits)

generators = uccsd_generators(n_qubits, n_electrons)
circuit, symbols = build_ansatz(qubits, generators)
reference = basis_index(n_qubits, range(n_electrons))

result = minimize(
    lambda parameters: energy(circuit, symbols, observable, parameters, reference, qubits),
    np.zeros(len(generators)),
    method="COBYLA",
    options={"maxiter": MAXITER, "rhobeg": 0.1, "tol": 1e-9},
)

print(f"n_qubits = {n_qubits}")
print(f"n_params = {len(generators)}")
print(f"HF       = {molecule.hf_energy:.10f}")
print(f"VQE      = {result.fun:.10f}")
print(f"CCSD     = {molecule.ccsd_energy:.10f}")
