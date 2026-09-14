"""QAOA MaxCut on a 6-vertex ring at p = 1, against the closed-form optimum.

For MaxCut C = sum (1 - Z_u Z_v)/2 at p = 1 on a triangle-free 2-regular graph the
edge formula of Wang et al. (PRA 97, 022304) collapses -- its sin^2(2*beta) term
carries [1 - cos^lambda(2*gamma)] with lambda = 0 triangles -- leaving
    <C> = n/2 + (n/4) sin(4*beta) sin(2*gamma),
whose maximum is 3n/4.  That is the oracle, and it does not depend on how any
stack chooses to parametrise its angles.
"""

import networkx as nx
import numpy as np

from qarp.algorithms import QAOA
from qarp.optimizers import ScipyOptimizer

N_VERTICES, MAXITER = 6, 2000

graph = nx.cycle_graph(N_VERTICES)
qaoa = QAOA(
    graph,
    n_layers=1,
    initial_parameters=np.array([0.5, 0.5]),
    optimizer=ScipyOptimizer("COBYLA", {"maxiter": MAXITER, "rhobeg": 0.3}),
)
qaoa.build()
cost, _ = qaoa.run()

# qarp minimises H = sum Z_u Z_v; the cut expectation is n/2 - <H>/2.
cut = N_VERTICES / 2 - cost / 2

print(f"n_vertices = {N_VERTICES}")
print(f"QAOA  = {cut:.8f}")
print(f"exact = {3 * N_VERTICES / 4:.8f}")
