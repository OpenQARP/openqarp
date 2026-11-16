"""Graphs, hypergraphs and simplicial complexes for problem construction.
Public depth: flat.  Submodules are private.
"""

import importlib.util as _importlib_util

from .._lazy import lazy_exports as _lazy_exports

from ._graph import Graph
from ._simplicial_complex import SimplicialComplex
from ._utils import (
    community_vector_to_sets,
    generate_complete_simplicial_complex,
    generate_random_simplicial_complex,
    graph_to_cost_hamiltonian,
    lists_to_tuples,
    qubit_operator_to_graph,
)

__all__ = [
    "Graph",
    "SimplicialComplex",
    "community_vector_to_sets",
    "generate_complete_simplicial_complex",
    "generate_random_simplicial_complex",
    "graph_to_cost_hamiltonian",
    "lists_to_tuples",
    "qubit_operator_to_graph",
]

# Hypergraph subclasses hypernetx's (the [hypergraph] extras, ~1 s to
# import), so it is resolved on first access rather than here; the __all__
# gate keeps `import *` and the docs to what is installed.  SimplicialComplex
# draws through matplotlib/networkx, both core dependencies, so it carries no
# optional dependency and is imported unconditionally.
if _importlib_util.find_spec("hypernetx") is not None:
    __all__ += ["Hypergraph"]

__getattr__, __dir__ = _lazy_exports(
    __name__, {"Hypergraph": ("._hypergraph", "hypernetx", "hypergraph")}
)
