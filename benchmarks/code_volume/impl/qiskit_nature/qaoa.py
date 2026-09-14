"""QAOA MaxCut on a 6-vertex ring at p = 1, against the closed-form optimum 3n/4."""

import networkx as nx
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA

N_VERTICES, MAXITER, N_SHOTS = 6, 2000, 1_000_000

graph = nx.cycle_graph(N_VERTICES)
observable = SparsePauliOp.from_sparse_list(
    [("ZZ", [u, v], 1.0) for u, v in graph.edges], num_qubits=N_VERTICES
)

qaoa = QAOA(
    StatevectorSampler(default_shots=N_SHOTS),
    COBYLA(maxiter=MAXITER, rhobeg=0.3),
    reps=1,
    initial_point=[0.5, 0.5],
)
result = qaoa.compute_minimum_eigenvalue(observable)

# qiskit's QAOA is a *sampling* eigensolver: its reported optimum is shot-estimated
# and can sit above the analytic p=1 maximum.  Re-evaluate exactly at the optimum.
ansatz = qaoa.ansatz.remove_final_measurements(inplace=False)
state = Statevector(ansatz.assign_parameters(result.optimal_point))
cost = state.expectation_value(observable).real

print(f"n_vertices = {N_VERTICES}")
print(f"QAOA  = {N_VERTICES / 2 - cost / 2:.8f}")
print(f"exact = {3 * N_VERTICES / 4:.8f}")
