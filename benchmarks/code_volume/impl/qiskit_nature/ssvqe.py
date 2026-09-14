"""SS-VQE: three lowest eigenvalues of H2 (STO-3G) from one hardware-efficient ansatz.

qiskit-nature ships no subspace-search VQE, so the loop is hand-written; the
driver, the mapper and the TwoLocal ansatz are the stack's own.
"""

import numpy as np
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from scipy.optimize import minimize

# Seeds for the subspace search.  These are NOT the same physical states as the
# openfermion stacks': that basis differs by orbital ordering and phase, and no
# qubit permutation relates the two, so this cell is excluded from the numeric
# cross-check (see run.py NOT_COMPARABLE and the README).
BASIS_STATES = [[], [0], [1]]
WEIGHTS = [4, 2, 1]
N_LAYERS = 6

problem = PySCFDriver(atom="H 0 0 0; H 0 0 0.735", basis="sto3g").run()
hamiltonian = JordanWignerMapper().map(problem.hamiltonian.second_q_op())
n_qubits = hamiltonian.num_qubits
# Nuclear repulsion folded in, so the objective is one operator.
hamiltonian = (
    hamiltonian + SparsePauliOp("I" * n_qubits, problem.nuclear_repulsion_energy)
).simplify()

ansatz = TwoLocal(
    n_qubits,
    "ry",
    "cx",
    entanglement=[[q, (q + 1) % n_qubits] for q in range(n_qubits)],
    reps=N_LAYERS,
    skip_final_rotation_layer=True,
)
references = [sum(1 << qubit for qubit in occupied) for occupied in BASIS_STATES]


def energy(values, reference):
    bound = ansatz.assign_parameters(values)
    state = Statevector.from_int(reference, 2**n_qubits).evolve(bound)
    return state.expectation_value(hamiltonian).real


result = minimize(
    lambda values: sum(
        weight * energy(values, reference)
        for weight, reference in zip(WEIGHTS, references, strict=True)
    ),
    np.random.default_rng(7).uniform(-0.1, 0.1, ansatz.num_parameters),
    method="BFGS",
    options={"maxiter": 3000},
)

energies = [energy(result.x, reference) for reference in references]
exact = np.linalg.eigvalsh(hamiltonian.to_matrix())

print(f"n_params = {ansatz.num_parameters}")
print("SS-VQE  =", np.array2string(np.sort(energies), precision=8))
print("exact   =", np.array2string(exact[:3], precision=8))
