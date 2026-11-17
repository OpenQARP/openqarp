from typing import Optional

from networkx import Graph

import qarpx as qx

from .._block import SimpleBlock


class CostOperatorBlock(SimpleBlock):
    """QAOA cost operator for a graph-encoded cost Hamiltonian.

    Edges contribute ZZ rotations; linear terms contribute Z rotations.
    The symbol ``γ`` is in radians: each edge of weight ``w`` emits
    ``rzz(w·γ) = exp(-i (w·γ/2) Z⊗Z)`` and each linear term ``rz(c·γ)``.
    """

    def __init__(
        self,
        n_qubits: int,
        problem: Graph,
        linear_terms: Optional[dict] = None,
        symbol_idx: int = 0,
        target_qubits=None,
        use_rzz: bool = True,
        name: Optional[str] = None,
    ):
        """Args:
        n_qubits: number of graph nodes.
        problem: NetworkX graph with edge weights.
        linear_terms: dict of single-qubit Z coefficients.
        symbol_idx: index for the symbolic parameter ``gamma_<idx>`` (ASCII so the
            circuit exports as valid OpenQASM 3).
        use_rzz: True → use RZZ; False → CX·Rz·CX decomposition.
        target_qubits, name: see ``Block``.
        """
        # A user-supplied name is honoured; the default carries the layer index.
        if name is None:
            name = f"Cost Op. (p={symbol_idx})"
        super().__init__(n_qubits, target_qubits, name=name)

        self.problem = problem
        self.linear_terms = linear_terms if linear_terms is not None else {}
        self.symbol_idx = symbol_idx
        self.use_rzz = use_rzz

    def build_vanilla(self) -> None:
        gamma_name = f"gamma_{self.symbol_idx}"
        edges = list(self.problem.edges(data=True))
        for from_, to_, data in edges:
            coef = data.get("weight", 1)
            angle = qx.Param.linear(float(coef), gamma_name)  # coef·γ, radians
            if self.use_rzz:
                self.rzz(from_, to_, angle)
            else:
                self.cx(from_, to_)
                self.rz(to_, angle)
                self.cx(from_, to_)

        for term, coeff in self.linear_terms.items():
            idx = int(term[0][0])
            self.rz(idx, qx.Param.linear(float(coeff), gamma_name))
