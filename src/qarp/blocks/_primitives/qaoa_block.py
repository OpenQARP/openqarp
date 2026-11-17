from typing import List, Optional, Union

from networkx import Graph

from .. import CompositeBlock
from . import CostOperatorBlock, HnBlock, MixedOperatorBlock


class QAOABlock(CompositeBlock):
    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        problem: Graph,
        linear_terms: Optional[dict] = None,
        use_rzz: bool = True,
        target_qubits: Optional[List[int]] = None,
        name="QAOA",
    ):
        """QAOA ansatz block combining initialization, cost, and mixer layers.

        QAOABlock constructs a complete QAOA circuit by composing initialization (Hadamard layer),
        and alternating cost and mixer operator layers. Each layer applies the cost Hamiltonian
        evolution followed by the mixer Hamiltonian evolution, with independent parameters for
        each layer. The circuit implements the QAOA ansatz |ψ(β,γ)⟩ = U(B,βₚ)U(C,γₚ)...U(B,β₁)U(C,γ₁)|+⟩,
        where p is the number of layers.

        Symbols are ``gamma_<layer>`` (cost) and ``beta_<layer>`` (mixer).  ``.symbols`` is
        sorted by name (§17), so it reads ``(beta_0, …, beta_{p-1}, gamma_0, …, gamma_{p-1})`` —
        NOT the (γ, β)-per-layer order other SDKs use positionally.  Bind by name
        (``parameter_map`` / a ``{symbol: value}`` ``initial_parameters``) rather than by position.

        Args:
            n_qubits: Number of qubits (graph nodes) in the circuit.
            n_layers: Number of QAOA layers (p parameter).
            problem: NetworkX graph defining the optimization problem with edge weights.
            linear_terms: Dictionary of linear (single-qubit) terms in the cost function.
            use_rzz: If True, each cost term is a native RZZ; if False, its CX·Rz·CX decomposition.
            target_qubits: Specific qubits to apply the block to. If None, uses all qubits.
            name: Optional custom name for the block.
        """
        blocks: list[Union[CostOperatorBlock, MixedOperatorBlock, HnBlock]] = []
        blocks += [HnBlock(n_qubits, target_qubits=list(range(n_qubits)))]

        for i in range(n_layers):
            blocks.append(
                CostOperatorBlock(
                    n_qubits=n_qubits,
                    problem=problem,
                    linear_terms=linear_terms,
                    symbol_idx=i,
                    target_qubits=list(range(n_qubits)),
                    use_rzz=use_rzz,
                )
            )
            blocks.append(
                MixedOperatorBlock(
                    n_qubits=n_qubits,
                    symbol_idx=i,
                    target_qubits=list(range(n_qubits)),
                )
            )

        super().__init__(
            n_qubits=n_qubits,
            blocks=blocks,
            target_qubits=target_qubits,
            name=name,
        )
