"""QAOA MaxCut on a 6-vertex ring at p = 1, against the closed-form optimum 3n/4."""

import networkx as nx
import numpy as np
import pennylane as qml
from pennylane import qaoa
from scipy.optimize import minimize

N_VERTICES, MAXITER = 6, 2000

graph = nx.cycle_graph(N_VERTICES)
cost_hamiltonian, mixer_hamiltonian = qaoa.maxcut(graph)


@qml.qnode(qml.device("default.qubit", wires=N_VERTICES))
def cost(parameters):
    gamma, beta = parameters
    for wire in range(N_VERTICES):
        qml.Hadamard(wire)
    qaoa.cost_layer(gamma, cost_hamiltonian)
    qaoa.mixer_layer(beta, mixer_hamiltonian)
    return qml.expval(cost_hamiltonian)


result = minimize(
    cost, np.array([0.5, 0.5]), method="COBYLA", options={"maxiter": MAXITER, "rhobeg": 0.3}
)

# pennylane's maxcut cost is 0.5*sum(Z_u Z_v - 1), so the cut is just -<H>.
print(f"n_vertices = {N_VERTICES}")
print(f"QAOA  = {-result.fun:.8f}")
print(f"exact = {3 * N_VERTICES / 4:.8f}")
