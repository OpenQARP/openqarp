"""ADAPT-VQE ground-state energy of LiH (STO-3G) in a (2e, 3o) active space."""

import numpy as np
from common import (
    build_ansatz,
    energy,
    evolved_state,
    molecular_hamiltonian,
    prepared_state,
    uccsd_generators,
)
from openfermion import get_sparse_operator, hermitian_conjugated, jordan_wigner
from qulacs.observable import create_observable_from_openfermion_text
from scipy.optimize import minimize

GEOMETRY = [("H", (0.0, 0.0, 0.0)), ("Li", (0.0, 0.0, 1.59))]
ACTIVE_SPACE = ([0], [1, 2, 3])
N_ACTIVE_ELECTRONS = 2
GRADIENT_THRESH, ENERGY_THRESH, MAX_CYCLES = 1e-6, 1e-10, 50

observable, hamiltonian, _ = molecular_hamiltonian(GEOMETRY, active_space=ACTIVE_SPACE)
n_qubits = 2 * len(ACTIVE_SPACE[1])

pool = uccsd_generators(n_qubits, N_ACTIVE_ELECTRONS, spin_conserving=False)
commutators = []
for generator in pool:
    image = jordan_wigner(generator)
    commutator = hamiltonian * image - image * hamiltonian
    # [H, A] is Hermitian for anti-Hermitian A, but LiH's degenerate pi orbitals
    # leave pyscf free to rotate them differently run to run, and the residue is
    # large enough to reject.  Project onto the Hermitian part.
    commutator = (commutator + hermitian_conjugated(commutator)) / 2
    commutator.compress()
    commutators.append(create_observable_from_openfermion_text(str(commutator)))

reference = prepared_state(n_qubits, range(N_ACTIVE_ELECTRONS))
selected, parameters, previous = [], [], None

for _ in range(MAX_CYCLES):
    circuit, layout = build_ansatz(n_qubits, selected)
    state = evolved_state(circuit, layout, parameters, reference)
    gradients = [commutator.get_expectation_value(state).real for commutator in commutators]
    best = int(np.argmax(np.abs(gradients)))
    if abs(gradients[best]) < GRADIENT_THRESH:
        break
    selected.append(pool[best])
    circuit, layout = build_ansatz(n_qubits, selected)
    result = minimize(
        lambda p: energy(circuit, layout, observable, p, reference),
        np.append(parameters, 0.0),
        method="BFGS",
    )
    parameters = list(result.x)
    if previous is not None and abs(result.fun - previous) < ENERGY_THRESH:
        break
    previous = result.fun

exact = np.linalg.eigvalsh(get_sparse_operator(hamiltonian).toarray()).min()

print(f"pool size = {len(pool)}")
print(f"ADAPT-VQE = {previous:.10f}")
print(f"exact     = {exact:.10f}")
