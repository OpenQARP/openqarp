"""QAOA MaxCut on a 6-vertex ring at p = 1, against the closed-form optimum 3n/4."""

import cirq
import networkx as nx
import numpy as np
from common import SIMULATOR
from scipy.optimize import minimize

N_VERTICES, MAXITER = 6, 2000

graph = nx.cycle_graph(N_VERTICES)
qubits = cirq.LineQubit.range(N_VERTICES)
edges = [(qubits[u], qubits[v]) for u, v in graph.edges]
observable = cirq.PauliSum.from_pauli_strings(
    [cirq.PauliString({u: cirq.Z, v: cirq.Z}) for u, v in edges]
)
qubit_map = {qubit: index for index, qubit in enumerate(qubits)}


def cost(parameters):
    """<sum Z_u Z_v> for the p = 1 ansatz: |+>^n, phase separator, X mixer."""
    gamma, beta = parameters
    circuit = cirq.Circuit(cirq.H.on_each(*qubits))
    circuit.append(cirq.ZZPowGate(exponent=gamma / np.pi)(u, v) for u, v in edges)
    circuit.append(cirq.rx(beta).on_each(*qubits))
    state = SIMULATOR.simulate(circuit, qubit_order=qubits).final_state_vector
    return observable.expectation_from_state_vector(state, qubit_map).real


result = minimize(
    cost, np.array([0.5, 0.5]), method="COBYLA", options={"maxiter": MAXITER, "rhobeg": 0.3}
)

print(f"n_vertices = {N_VERTICES}")
print(f"QAOA  = {N_VERTICES / 2 - result.fun / 2:.8f}")
print(f"exact = {3 * N_VERTICES / 4:.8f}")
