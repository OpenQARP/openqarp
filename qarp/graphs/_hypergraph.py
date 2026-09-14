import hypernetx as hnx

from .. import config
from ._graph import _n_qubits_from_labels


class Hypergraph(hnx.Hypergraph):
    def __init__(self, *args, **kwargs):
        """Hypergraph class extending HyperNetX for quantum state preparation.

        Hypergraph provides a representation of higher-order graph structures where edges
        (hyperedges) can connect any number of vertices, not just pairs. This is used to
        define hypergraph states in quantum computing, where vertices correspond to qubits
        and hyperedges define multi-qubit entangling operations. The class extends HyperNetX
        functionality with the state-preparation block for the hypergraph and visualization
        utilities tailored for quantum algorithm workflows.  Incidence, dual and adjacency
        matrices are inherited from HyperNetX.

        Args:
            *args: Positional arguments passed to hypernetx.Hypergraph constructor.
            **kwargs: Keyword arguments passed to hypernetx.Hypergraph constructor.
                Common kwargs include setsystem, edges dict, weighted option, etc.
        """
        super().__init__(*args, **kwargs)

    @property
    def n_qubits(self) -> int:
        """Qubit count implied by the vertices: highest vertex index over all edges + 1,
        0 when edgeless.

        Vertices are qubit indices, so a non-contiguous labelling such as ``{0, 1, 3}``
        still needs qubit 3 to exist: this is *not* the vertex count.  This is the
        single vertex check; ``HypergraphStateBlock`` delegates here.

        Raises:
            TypeError: If a vertex label is not a non-negative integer.
        """
        vertices = {v for e in self.edges for v in self.edges[e]}
        return _n_qubits_from_labels(vertices, "Hypergraph vertex labels")

    def to_state_block(self, **kwargs):
        """The hypergraph-state preparation block for this hypergraph: ``H`` on every
        qubit, then a ``C^{k-1}Z`` per hyperedge of order ``k``.

        Args:
            **kwargs: Forwarded to ``HypergraphStateBlock`` (``name``,
                ``target_qubits``).

        Returns:
            HypergraphStateBlock: The (unbuilt) block.
        """
        # Lazy: blocks already import graphs; the reverse edge stays out of import time.
        from ..blocks._state_preparation.hypergraph_state_block import HypergraphStateBlock

        return HypergraphStateBlock(hypergraph=self, **kwargs)

    def plot(self, figsize=(4, 3), return_fig=False):
        """
        Plots the hypergraph.

        Args:
            figsize (tuple): Size of the figure.
            return_fig (bool): If True, return (fig, ax) for saving or customizing.

        Returns:
            (fig, ax) if return_fig is True, otherwise None.
        """
        # Lazy: keep matplotlib out of headless imports (mirrors blocks S-F).
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        pos = hnx.drawing.rubber_band.layout_node_link(self, seed=config.seed)
        hnx.draw(self, pos=pos, ax=ax)
        plt.tight_layout()
        if return_fig:
            return fig, ax
        else:
            plt.show()
            return None
