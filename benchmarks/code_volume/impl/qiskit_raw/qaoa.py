"""QAOA MaxCut on a 6-vertex ring at p = 1, against the closed-form optimum 3n/4."""

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

N_VERTICES, MAXITER = 6, 2000

graph = nx.cycle_graph(N_VERTICES)
edges = list(graph.edges)
observable = SparsePauliOp.from_sparse_list(
    [("ZZ", [u, v], 1.0) for u, v in edges], num_qubits=N_VERTICES
)


def cost(parameters):
    """<sum Z_u Z_v> for the p = 1 ansatz: |+>^n, phase separator, X mixer."""
    gamma, beta = parameters
    circuit = QuantumCircuit(N_VERTICES)
    circuit.h(range(N_VERTICES))
    for u, v in edges:
        circuit.rzz(gamma, u, v)
    circuit.rx(beta, range(N_VERTICES))
    return Statevector(circuit).expectation_value(observable).real


result = minimize(
    cost, np.array([0.5, 0.5]), method="COBYLA", options={"maxiter": MAXITER, "rhobeg": 0.3}
)

print(f"n_vertices = {N_VERTICES}")
print(f"QAOA  = {N_VERTICES / 2 - result.fun / 2:.8f}")
print(f"exact = {3 * N_VERTICES / 4:.8f}")
