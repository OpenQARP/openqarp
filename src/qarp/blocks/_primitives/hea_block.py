from typing import List, Optional

from sympy import Symbol

from .._block import SimpleBlock, _sorted_symbols


class HEABlock(SimpleBlock):
    """Hardware-Efficient Ansatz (HEA) block with configurable entanglement patterns.

    HEABlock constructs a parameterized quantum circuit by stacking multiple HEA layers,
    each consisting of single-qubit rotations followed by entangling gates. The architecture
    supports both linear and brickwork entanglement topologies, with options for real-valued
    (Ry-only) or complex-valued (Ry-Rz) rotations. This structure provides an expressive ansatz
    suitable for variational quantum algorithms while maintaining compatibility with near-term
    quantum hardware constraints.

    Args:
        n_qubits: Number of qubits in the circuit.
        n_layers: Number of HEA layers to stack.
        real: If True, uses only Ry rotations (real ansatz); if False, includes Rz rotations (complex).
        linear: If True, uses linear entanglement; if False, uses brickwork entanglement.
        circular: If True, applies entanglement with periodic boundary conditions.
        use_cz: If True, uses CZ gates for entanglement; if False, uses CNOT gates.
        target_qubits: Specific qubits to apply the block to. If None, uses all qubits.
        name: Optional custom name for the block.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        real: bool,
        linear: bool,
        circular: bool,
        use_cz: bool,
        target_qubits: Optional[List[int]] = None,
        name: Optional[str] = None,
    ):
        self.n_layers = n_layers
        self.real = real
        self.linear = linear
        self.circular = circular
        self.use_cz = use_cz
        if name is None:
            name = f"Layered HEA (n={n_layers})"

        super().__init__(
            n_qubits=n_qubits,
            target_qubits=target_qubits,
            name=name,
        )

        # Populate symbols after super().__init__() — the base sets
        # self.symbols = None, so assigning before would be clobbered.
        syms = []
        for layer in range(n_layers):
            for q in range(n_qubits):
                syms.append(Symbol(f"ry_{layer}_{q}"))
                if not real:
                    syms.append(Symbol(f"rz_{layer}_{q}"))
        self.symbols = _sorted_symbols(syms)

    def build_vanilla(self) -> None:
        # Entangling pairs and gate method are the same for every layer — compute once.
        if self.linear:
            pairs = [(q, q + 1) for q in range(self.n_qubits - 1)]
        else:  # brickwork
            even_pairs = [(2 * q, 2 * q + 1) for q in range(self.n_qubits // 2)]
            # (n-1)//2 odd pairs, so the last qubit is entangled at odd n too;
            # leaving it out silently weakens the ansatz to single-qubit
            # rotations on that wire.
            odd_pairs = [(2 * q + 1, 2 * q + 2) for q in range((self.n_qubits - 1) // 2)]
            pairs = even_pairs + odd_pairs
        # Below 3 qubits the ring edge is already the line edge; appending it
        # would repeat a pair (an identity layer for symmetric CZ).
        if self.circular and self.n_qubits > 2:
            pairs.append((self.n_qubits - 1, 0))
        entangle = self.cz if self.use_cz else self.cx

        for layer in range(self.n_layers):
            self.ry([(q, Symbol(f"ry_{layer}_{q}")) for q in range(self.n_qubits)])
            if not self.real:
                self.rz([(q, Symbol(f"rz_{layer}_{q}")) for q in range(self.n_qubits)])
            entangle(pairs)
