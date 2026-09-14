"""QAOA MaxCut on a 6-vertex ring at p = 1, against the closed-form optimum 3n/4."""

import networkx as nx
import numpy as np
from qulacs import Observable, QuantumCircuit, QuantumState
from scipy.optimize import minimize

N_VERTICES, MAXITER = 6, 2000
PAULI_Z = 3

graph = nx.cycle_graph(N_VERTICES)
edges = list(graph.edges)

observable = Observable(N_VERTICES)
for u, v in edges:
    observable.add_operator(1.0, f"Z {u} Z {v}")


def cost(parameters):
    """<sum Z_u Z_v> for the p = 1 ansatz: |+>^n, phase separator, X mixer."""
    gamma, beta = parameters
    circuit = QuantumCircuit(N_VERTICES)
    for qubit in range(N_VERTICES):
        circuit.add_H_gate(qubit)
    for u, v in edges:
        circuit.add_multi_Pauli_rotation_gate([u, v], [PAULI_Z, PAULI_Z], gamma)
    for qubit in range(N_VERTICES):
        circuit.add_RX_gate(qubit, beta)
    state = QuantumState(N_VERTICES)
    state.set_zero_state()
    circuit.update_quantum_state(state)
    return observable.get_expectation_value(state).real


result = minimize(
    cost, np.array([0.5, 0.5]), method="COBYLA", options={"maxiter": MAXITER, "rhobeg": 0.3}
)

print(f"n_vertices = {N_VERTICES}")
print(f"QAOA  = {N_VERTICES / 2 - result.fun / 2:.8f}")
print(f"exact = {3 * N_VERTICES / 4:.8f}")
