from typing import List, Optional

from sympy import Symbol

from .. import CompositeBlock
from . import AGateBlock, RSPBlock


class SPABlock(CompositeBlock):
    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        real: bool,
        linear: bool,
        circular: bool,
        target_qubits: Optional[List[int]] = None,
        name: Optional[str] = None,
    ):
        """Layered Separable Pair Ansatz (SPA) block with configurable entanglement geometry.

        SPABlock constructs a parameterized quantum circuit using separable pair gates
        (RSP or A-gates) applied between qubit pairs in specified patterns. The architecture
        supports linear and brickwork entanglement topologies, where each two-qubit gate can
        generate arbitrary entanglement within a pair while maintaining a structured layered
        format. This provides an efficient ansatz for quantum chemistry and optimization problems
        with controllable entanglement depth and expressibility.

        Args:
            n_qubits: Number of qubits in the circuit.
            n_layers: Number of SPA layers to stack.
            real: If True, uses RSP gates (real-valued); if False, uses A-gates (complex-valued with phase).
            linear: If True, uses linear entanglement; if False, uses brickwork entanglement.
            circular: If True, applies entanglement with periodic boundary conditions.
            target_qubits: Specific qubits to apply the block to. If None, uses all qubits.
            name: Optional custom name for the block.
        """
        self.n_layers = n_layers
        self.real = real
        self.linear = linear
        self.circular = circular
        self.blocks = []
        if name is None:
            name = f"SPA (n={n_layers})"

        for layer_index in range(n_layers):
            if not self.linear:  # pairwise / brickwork
                # odd
                for i in range(n_qubits // 2 - 1):
                    q = 2 * i + 1
                    self.blocks += [self.get_entangler_block(q, q + 1, layer_index)]
                # even
                for i in range(n_qubits // 2):
                    q = 2 * i
                    self.blocks += [self.get_entangler_block(q, q + 1, layer_index)]
                if self.circular:
                    self.blocks += [self.get_entangler_block(n_qubits - 1, 0, layer_index)]
            else:  # linear
                for i in range(n_qubits - 1):
                    self.blocks += [self.get_entangler_block(i, i + 1, layer_index)]
                if self.circular:
                    self.blocks += [self.get_entangler_block(n_qubits - 1, 0, layer_index)]

        super().__init__(
            n_qubits=n_qubits,
            blocks=self.blocks,
            target_qubits=target_qubits,
            name=name,
        )

    def get_entangler_block(self, i, j, layer_index):
        if self.real:
            theta = Symbol(f"spa_theta_{layer_index}_{i}_{j}")
            entangler = RSPBlock(theta, target_qubits=[i, j])
            # self.symbols += [theta]
        else:
            phi = Symbol(f"spa_phi_{layer_index}_{i}_{j}")
            theta = Symbol(f"spa_theta_{layer_index}_{i}_{j}")
            # self.symbols += [phi, theta]
            entangler = AGateBlock(theta=theta, phi=phi, target_qubits=[i, j])
        return entangler
