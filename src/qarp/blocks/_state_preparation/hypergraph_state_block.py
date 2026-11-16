from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np

from qarp.blocks._block import SimpleBlock
from qarp.blocks._prepares_known_state import prepares_known_state

if TYPE_CHECKING:
    from qarp.graphs._hypergraph import Hypergraph


@prepares_known_state
class HypergraphStateBlock(SimpleBlock):
    def __init__(
        self,
        hypergraph: Optional[Hypergraph] = None,
        n_qubits: Optional[int] = None,
        edges: Optional[List[Tuple[int, ...]]] = None,
        target_qubits: Optional[List[int]] = None,
        name: Optional[str] = None,
    ):
        """Hypergraph state preparation block for multi-qubit entangled states.

        HypergraphStateBlock prepares quantum states defined by hypergraphs, where vertices
        represent qubits and hyperedges encode multi-qubit entangling operations. Starting
        from an equal superposition (H^⊗n|0⟩^⊗n), controlled-Z gates are applied according
        to the hypergraph structure, creating states with higher-order correlations beyond
        pairwise entanglement. (See doi:10.1088/1367-2630/15/11/113022 for more details.)

        Args:
            hypergraph: A Hypergraph object defining the state structure. Takes precedence if provided.
            n_qubits: Number of qubits (required if hypergraph is None).
            edges: List of hyperedges, each a tuple of qubit indices. Last index is target for CnZ gate.
                On this path the ``hypergraph`` attribute is the equivalent ``Hypergraph``
                when hypernetx is installed and ``None`` otherwise.
            target_qubits: The target qubits the underlying block will act on when added to a circuit.
            name: Optional custom name for the block. Defaults to "HypergraphState({n_qubits}q)".

        Raises:
            TypeError: If hypergraph is provided but is not a Hypergraph instance,
                or a vertex label is not a non-negative integer.
            ValueError: If neither hypergraph nor both n_qubits and edges are provided.
        """

        if hypergraph is not None:
            from qarp.graphs import Hypergraph  # lazy: hypernetx drags matplotlib

            if not isinstance(hypergraph, Hypergraph):
                raise TypeError(
                    f"Expected 'hypergraph' to be an instance of Hypergraph, got {type(hypergraph).__name__}."
                )
            # Vertices are qubit indices, so the size is the highest index + 1, not
            # the vertex count; the check lives once, on Hypergraph.n_qubits.
            resolved_n_qubits = hypergraph.n_qubits
            resolved_edges: Union[List[Tuple[int, ...]], list[list[Any]]] = [
                list(hypergraph.edges[idx]) for idx in hypergraph.edges
            ]
            resolved_hypergraph: Optional[Hypergraph] = hypergraph
        else:
            if not isinstance(n_qubits, int) or not isinstance(edges, list):
                raise ValueError(
                    "Either 'hypergraph' or both 'n_qubits' and 'edges' must be provided."
                )
            resolved_n_qubits = n_qubits
            resolved_edges = edges
            # The block reads only n_qubits and edges; the Hypergraph is a
            # convenience that must not make a base install fail (hypernetx is
            # the optional [hypergraph] extra).
            try:
                from qarp.graphs import Hypergraph  # lazy: hypernetx drags matplotlib
            except ImportError:
                resolved_hypergraph = None
            else:
                resolved_hypergraph = Hypergraph(edges)

        super().__init__(
            resolved_n_qubits,
            target_qubits=target_qubits,
            name=name if name else f"HypergraphState({resolved_n_qubits}q)",
        )
        self.hypergraph = resolved_hypergraph
        self.edges = resolved_edges

    def build_vanilla(self) -> None:
        if self.n_qubits is None:
            raise RuntimeError("n_qubits is undefined")  # keeps mypy happy
        # H on every qubit (inline HnBlock) — bulk emit.
        self.h(list(range(self.n_qubits)))
        # CnZ per edge: decomposed into available qarpx gates
        for edge in self.edges:
            qubits = list(edge)
            n = len(qubits)
            if n == 1:
                self.z(qubits[0])
            else:
                self.mcz(qubits)

    def target_statevector(self) -> np.ndarray:
        r"""``2^{-n/2} (-1)^{e(x)}``, with ``e(x)`` the number of hyperedges
        whose qubits are all set in ``x``.

        The order-1 case is included: a single-vertex edge is "fully set" iff
        that bit is 1, which is exactly the ``Z`` the build emits.
        """
        dim = 2**self.n_qubits
        psi = np.full(dim, 2 ** (-self.n_qubits / 2), dtype=complex)
        idx = np.arange(dim)
        for edge in self.edges:
            mask = sum(1 << int(q) for q in edge)
            psi[(idx & mask) == mask] *= -1
        return psi
