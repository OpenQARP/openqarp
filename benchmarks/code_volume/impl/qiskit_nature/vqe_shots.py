"""Shot-based VQE of H2 (STO-3G) with a UCCSD ansatz and qubit-wise-commuting grouping."""

import numpy as np
from qiskit import transpile
from qiskit_aer.primitives import EstimatorV2
from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper

N_SHOTS, MAXITER, SEED = 100_000, 200, 7

problem = PySCFDriver(atom="H 0 0 0; H 0 0 0.735", basis="sto3g").run()
mapper = JordanWignerMapper()
hamiltonian = mapper.map(problem.hamiltonian.second_q_op())
ansatz = UCCSD(
    problem.num_spatial_orbitals,
    problem.num_particles,
    mapper,
    initial_state=HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper),
)
ansatz = transpile(ansatz, basis_gates=["rz", "rx", "ry", "h", "cx"])

# Aer's V2 estimator takes a target standard error, not a shot count: 1/sqrt(N).
estimator = EstimatorV2(
    options={
        "default_precision": 1.0 / np.sqrt(N_SHOTS),
        "run_options": {"seed_simulator": SEED},
    }
)
vqe = VQE(estimator, ansatz, COBYLA(maxiter=MAXITER, rhobeg=0.1))
vqe.initial_point = np.zeros(ansatz.num_parameters)
energy = problem.interpret(vqe.compute_minimum_eigenvalue(hamiltonian)).total_energies[0].real
reference = NumPyMinimumEigensolver().compute_minimum_eigenvalue(hamiltonian)
exact = problem.interpret(reference).total_energies[0].real

print(f"n_qubits = {ansatz.num_qubits}")
print(f"n_params = {ansatz.num_parameters}")
print(f"VQE-shots = {energy:.6f}")
print(f"exact     = {exact:.6f}")
