import numbers
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from networkx import Graph as nxGraph
from networkx import (
    adjacency_matrix,
    draw,
    incidence_matrix,
    laplacian_matrix,
    modularity_matrix,
    normalized_laplacian_matrix,
    spring_layout,
)

from .. import config


def _check_qubit_label(label, what: str = "Node labels") -> int:
    """Validate a graph label as a qubit index and return it as a plain int.

    bool is an Integral subclass and must be rejected explicitly.
    """
    if isinstance(label, bool) or not isinstance(label, numbers.Integral) or label < 0:
        raise TypeError(f"{what} must be non-negative integers (qubit indices); got {label!r}.")
    return int(label)


def _n_qubits_from_labels(labels, what: str = "Node labels") -> int:
    """Highest qubit index + 1 over ``labels`` (0 when empty), validating each."""
    return max((_check_qubit_label(label, what) for label in labels), default=-1) + 1


class Graph(nxGraph):
    def __init__(self, *args, **kwargs):
        """Graph class extending NetworkX Graph with quantum algorithm utilities.

        Graph provides additional methods tailored for quantum optimization algorithms:
        the graph → cost-Hamiltonian and graph → graph-state constructions, sparse
        matrix representations (degree, adjacency, Laplacian, incidence) used in
        spectral partitioning and community detection, and plotting. It inherits all
        NetworkX graph functionality.  Node labels double as qubit indices wherever a
        quantum object is built from the graph, so they must then be non-negative
        integers (non-contiguous labels are allowed; see ``n_qubits``).

        Args:
            *args: Positional arguments passed to networkx.Graph constructor.
            **kwargs: Keyword arguments passed to networkx.Graph constructor.
                Common kwargs include incoming_graph_data, node/edge attributes, etc.
        """
        super().__init__(*args, **kwargs)

    # ── quantum constructions ──────────────────────────────────────────

    @property
    def n_qubits(self) -> int:
        """Qubit count implied by the node labels: highest label + 1, 0 when empty.

        Labels are qubit indices, so a non-contiguous labelling such as
        ``{0, 1, 3}`` still needs qubit 3 to exist: this is *not* the node count.

        Raises:
            TypeError: If a node label is not a non-negative integer.
        """
        return _n_qubits_from_labels(self.nodes)

    def to_cost_hamiltonian(self, *, linear=None):
        """Ising cost Hamiltonian ``Σ_e w_e Z_i Z_j + Σ_i c_i Z_i`` of this graph.

        Thin wrapper over :func:`qarp.graphs.graph_to_cost_hamiltonian`, which
        documents the convention (this is QAOA's Ising form, *not* the MaxCut
        objective: minimising it maximises the cut, ``cut = (Σ_e w_e − ⟨H⟩) / 2``).

        Args:
            linear: Mapping node → coefficient of its ``Z_i`` term.  ``None`` reads
                the per-node ``linear`` attributes (set by ``from_qubit_operator``).

        Returns:
            QubitOperator: The cost Hamiltonian.
        """
        from ._utils import graph_to_cost_hamiltonian  # lazy: utils imports Graph

        return graph_to_cost_hamiltonian(self, linear=linear)

    @classmethod
    def from_qubit_operator(cls, qubit_op) -> "Graph":
        """Graph of a Z/ZZ ``QubitOperator``: ``Z_i Z_j`` terms become weighted edges
        and ``Z_i`` terms the node attribute ``linear``; the constant is dropped.

        Inverse of ``to_cost_hamiltonian`` up to that constant.  Equal to
        :func:`qarp.graphs.qubit_operator_to_graph`.
        """
        from ._utils import qubit_operator_to_graph  # lazy: utils imports Graph

        return cls(qubit_operator_to_graph(qubit_op))

    def to_graph_state_block(self, **kwargs):
        """The graph-state preparation block: ``H`` on every qubit, then ``CZ`` per edge.

        A graph state is a hypergraph state whose hyperedges all have order 2, so this
        is the ``HypergraphStateBlock`` on this graph's edges, sized by ``n_qubits``.
        Works on a base install: hypernetx is not required.

        Args:
            **kwargs: Forwarded to ``HypergraphStateBlock`` (``name``,
                ``target_qubits``).

        Returns:
            HypergraphStateBlock: The (unbuilt) block.

        Raises:
            ValueError: On a self-loop, which is no hyperedge (``CZ`` needs two qubits).
        """
        # Lazy: blocks already import graphs; the reverse edge stays out of import time.
        from ..blocks._state_preparation.hypergraph_state_block import HypergraphStateBlock
        from ._utils import _reject_self_loop

        n_qubits = self.n_qubits
        for u, v in self.edges:
            _reject_self_loop(u, v)
        edges = [(int(u), int(v)) for u, v in self.edges]
        return HypergraphStateBlock(n_qubits=n_qubits, edges=edges, **kwargs)

    # ── sparse matrix accessors (weighted by the `weight` edge attribute) ──

    def adjacency_matrix(self):
        """Weighted adjacency matrix, rows/columns in node insertion order.

        Returns:
            SciPy sparse array. Append `.toarray()` for dense representation.
        """
        return adjacency_matrix(self)

    def laplacian_matrix(self):
        """Weighted graph Laplacian ``D − A``, rows/columns in node insertion order.

        Returns:
            SciPy sparse array. Append `.toarray()` for dense representation.
        """
        return laplacian_matrix(self)

    def normalized_laplacian_matrix(self):
        """Normalized Laplacian ``I − D^{-1/2} A D^{-1/2}``, node insertion order.

        An isolated node contributes a zero row and column (networkx's convention),
        not NaN.

        Returns:
            SciPy sparse array. Append `.toarray()` for dense representation.
        """
        return normalized_laplacian_matrix(self)

    def incidence_matrix(self):
        """Unsigned node × edge incidence matrix (entry 1 where the node is an
        endpoint), nodes in insertion order and edges in ``self.edges`` order.

        Returns:
            SciPy sparse array. Append `.toarray()` for dense representation.
        """
        return incidence_matrix(self, oriented=False)

    def degree_matrix(self):
        """
        Returns the degree matrix of the graph.
        The degree matrix is a diagonal matrix where each diagonal element is the degree of the corresponding node.

        Returns:
            SciPy sparse array: The degree matrix of the graph. Append `.toarray()` for dense representation.
        """
        return laplacian_matrix(self) + adjacency_matrix(self)

    def max_modularity_eigenvector(self):
        """
        Returns the eigenvector relative to the largest eigenvalue of the modularity matrix.

        Two behaviours are inherited from networkx's ``modularity_matrix``: entries
        are indexed by node *insertion* order (not by label), and edge weights are
        *ignored* — every edge counts 1.

        Returns:
            np.ndarray: The eigenvector relative to the largest eigenvalue of the modularity matrix.

        Raises:
            ValueError: If the graph has no nodes, or has nodes but no edges — the
                modularity matrix ``B = A − k kᵀ / 2m`` is undefined at ``2m = 0``.
        """
        if self.number_of_nodes() == 0:
            raise ValueError("graph has no nodes; the modularity matrix is undefined")
        if self.number_of_edges() == 0:
            raise ValueError(
                "graph has no edges; the modularity matrix is undefined (2m = 0). "
                "This is not an eigensolver failure."
            )
        # eigh: the modularity matrix is symmetric, and eig may return a
        # complex-dtype eigenvector for degenerate spectra.
        mod_matrix = modularity_matrix(self)
        eigenvalues, eigenvectors = np.linalg.eigh(mod_matrix)
        max_index = np.argmax(eigenvalues)
        return eigenvectors[:, max_index]

    def plot(
        self,
        highlight_nodes: Optional[Union[Dict[int, str], List[int]]] = None,
        figsize: Optional[Tuple] = (4, 3),
        pos: Optional[Dict] = None,
        with_labels: bool = True,
        node_color: str = "lightblue",
        font_size: int = 9,
        return_fig: bool = False,
    ):
        """
        Plots the graph.

        Args:
            highlight_nodes: Optional dictionary mapping node IDs to colors for highlighting.
            figsize: Size of the figure in inches.
            pos: Optional dictionary of node positions. If None, a spring layout is computed.
            with_labels: Whether to display node labels.
            node_color: Default color used for nodes not in highlight_nodes.
            font_size: Font size for node labels.
            return_fig: If True, return (fig, ax) for external saving/customization.

        Returns:
            (fig, ax) if return_fig is True, otherwise None.
        """
        # Lazy: keep matplotlib out of headless imports (mirrors blocks S-F).
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        if pos is None:
            pos = spring_layout(self, seed=config.seed)

        # Prepare node colors
        if highlight_nodes is not None:
            if isinstance(highlight_nodes, dict):
                # Use provided colors
                node_colors = [highlight_nodes.get(n, node_color) for n in self.nodes()]
            elif isinstance(highlight_nodes, list):
                # Highlight all nodes in list as red
                node_colors = ["red" if n in highlight_nodes else node_color for n in self.nodes()]
            else:
                raise TypeError("highlight_nodes must be a dict or a list of node IDs")
        else:
            node_colors = [node_color] * self.number_of_nodes()

        draw(self, pos, ax=ax, with_labels=with_labels, node_color=node_colors, font_size=font_size)
        plt.tight_layout()
        if return_fig:
            return fig, ax
        else:
            plt.show()
            return None
