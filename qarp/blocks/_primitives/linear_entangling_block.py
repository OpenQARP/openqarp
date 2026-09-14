from typing import List, Optional

from .._block import SimpleBlock


class LinearEntanglingBlock(SimpleBlock):
    def __init__(
        self,
        n_qubits: int,
        circular: bool,
        use_cz: bool,
        target_qubits: Optional[List[int]] = None,
        name: Optional[str] = None,
    ):
        """
        An object representing a linear entangling block.

        The structure entangles qubit i with qubit i + 1 for all values of i.

        Args:
            n_qubits: The number of qubits to include in the block.
            circular: If True, also entangles the last qubit back to the first.
            use_cz: If True use CZ gates; otherwise use CX gates.
            target_qubits: The target qubits will act on when added to a Block object.
        """
        if name is None:
            name = "LinearEntangler(CZ)" if use_cz else "LinearEntangler(CX)"

        super().__init__(
            n_qubits=n_qubits,
            target_qubits=target_qubits,
            name=name,
        )
        self.circular = circular
        self.use_cz = use_cz

    def build_vanilla(self) -> None:
        pairs = [(i, i + 1) for i in range(self.n_qubits - 1)]
        # The wrap is a new edge only from n >= 3: a 2-ring IS the 2-chain, so
        # the wrap duplicates the only pair (CZ² = identity entangler), and a
        # 1-ring would self-pair.
        if self.circular and self.n_qubits >= 3:
            pairs.append((self.n_qubits - 1, 0))
        if self.use_cz:
            self.cz(pairs)
        else:
            self.cx(pairs)
