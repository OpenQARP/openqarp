from typing import List, Optional

from .._block import SimpleBlock


class BrickworkEntanglingBlock(SimpleBlock):
    def __init__(
        self,
        n_qubits: int,
        circular: bool,
        use_cz: bool,
        target_qubits: Optional[List[int]] = None,
        name: Optional[str] = None,
    ):
        """
        An object representing a brickwork entangling block.

        The structure entangles qubit i with qubit i + 1 for even values of i, then for odd values of i. Corresponds to the
        implementation in quri-parts and to the 'pairwise' option in qiskit, which also does even before odd.

        Args:
            n_qubits: The number of qubits to include in the block.
            circular: if True, entangles the first qubit with the last one.
            use_cz: if True replace the CX gate with a CZ gate.
            target_qubits: The target qubits will act on when added to a Block object.
        """
        self.use_cz = use_cz

        if name is None:
            if use_cz:
                name = "BrickWork Entangler (CZ)"
            else:
                name = "BrickWork Entangler (CX)"

        super().__init__(
            n_qubits=n_qubits,
            target_qubits=target_qubits,
            name=name,
        )
        self.circular = circular

    def build_vanilla(self) -> None:
        even_pairs = [(2 * i, 2 * i + 1) for i in range(self.n_qubits // 2)]
        # (n_qubits - 1) // 2, not n_qubits // 2 - 1: the latter drops the final odd
        # pair when n_qubits is odd, leaving the highest qubit unentangled.
        odd_pairs = [(2 * i + 1, 2 * i + 2) for i in range((self.n_qubits - 1) // 2)]
        pairs = even_pairs + odd_pairs
        # The wrap is a new edge only from n >= 3: a 2-ring IS the 2-chain, so
        # the wrap duplicates the only pair (CZ² = identity entangler), and a
        # 1-ring would self-pair.
        if self.circular and self.n_qubits >= 3:
            pairs.append((self.n_qubits - 1, 0))
        if self.use_cz:
            self.cz(pairs)
        else:
            self.cx(pairs)
