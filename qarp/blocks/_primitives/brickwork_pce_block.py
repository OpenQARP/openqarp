from typing import List, Optional

from sympy import Symbol

from .._block import SimpleBlock, _sorted_symbols


class BrickworkPCEBlock(SimpleBlock):
    """Brickwork Pauli Correlation Encoding (PCE) ansatz block.

    The ansatz used by the original PCE-algorithm paper (``qarp.algorithms.PCE``):
    Sciorilli, Borges, Patti, García-Martín, Camilo, Anandkumar & Aolita,
    "Towards large-scale quantum optimization solvers with few qubits",
    Nat. Commun. 16, 476 (2025), https://doi.org/10.1038/s41467-024-55346-z.

    Constructs a parameterized quantum circuit by stacking ``n_layers`` PCE
    layers. Each layer has three single-qubit rotation sublayers (Rx, Ry, Rz)
    interleaved with brickwork Rxx entangling sublayers (native ``RXX`` gate).
    The entangling sublayers alternate between even pairs (0,1),(2,3),... and
    odd pairs (1,2),(3,4),..., matching the even/odd tiling used by
    :class:`BrickworkEntanglingBlock`.

    Args:
        n_qubits: Number of qubits in the circuit.
        n_layers: Number of PCE layers to stack.
        target_qubits: Specific qubits to apply the block to. If None, uses all qubits.
        name: Optional custom name for the block.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        target_qubits: Optional[List[int]] = None,
        name: Optional[str] = None,
    ):
        self.n_layers = n_layers
        super().__init__(
            n_qubits=n_qubits,
            target_qubits=target_qubits,
            name=name or f"BrickworkPCE (n={n_layers})",
        )

        # Populate symbols after super().__init__() — the base sets
        # self.symbols = None, so assigning before would be clobbered.
        even_pairs, odd_pairs = self._pairs()
        syms = []
        for layer in range(n_layers):
            syms += [Symbol(f"pce_rx_{layer}_{q}") for q in range(n_qubits)]
            syms += [Symbol(f"pce_Rxx0_{layer}_{c}") for c, _ in even_pairs]
            syms += [Symbol(f"pce_ry_{layer}_{q}") for q in range(n_qubits)]
            syms += [Symbol(f"pce_Rxx1_{layer}_{c}") for c, _ in odd_pairs]
            syms += [Symbol(f"pce_rz_{layer}_{q}") for q in range(n_qubits)]
            syms += [Symbol(f"pce_Rxx2_{layer}_{c}") for c, _ in even_pairs]
        self.symbols = _sorted_symbols(syms)

    def _pairs(self) -> tuple:
        n = self.n_qubits
        even_pairs = [(2 * q, 2 * q + 1) for q in range(n // 2)]
        # (n - 1) // 2, not n // 2 - 1: the latter drops the final odd pair
        # when n is odd, leaving the highest qubit unentangled.
        odd_pairs = [(2 * q + 1, 2 * q + 2) for q in range((n - 1) // 2)]
        return even_pairs, odd_pairs

    def build_vanilla(self) -> None:
        n = self.n_qubits
        even_pairs, odd_pairs = self._pairs()

        for layer in range(self.n_layers):
            self.rx([(q, Symbol(f"pce_rx_{layer}_{q}")) for q in range(n)])
            self._entangle(even_pairs, f"pce_Rxx0_{layer}")

            self.ry([(q, Symbol(f"pce_ry_{layer}_{q}")) for q in range(n)])
            self._entangle(odd_pairs, f"pce_Rxx1_{layer}")

            self.rz([(q, Symbol(f"pce_rz_{layer}_{q}")) for q in range(n)])
            self._entangle(even_pairs, f"pce_Rxx2_{layer}")

    def _entangle(self, pairs: List[tuple], prefix: str) -> None:
        """Apply an Rxx layer over disjoint qubit *pairs*, one symbol per pair."""
        if not pairs:
            return
        self.rxx([(c, t, Symbol(f"{prefix}_{c}")) for c, t in pairs])
