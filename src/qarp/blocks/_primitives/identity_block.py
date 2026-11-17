from typing import List, Optional

from .._block import SimpleBlock


class IdentityBlock(SimpleBlock):
    def __init__(
        self,
        n_qubits: int,
        target_qubits: Optional[List[int]] = None,
        name: str = "Identity",
    ):
        """Constructs an identity circuit block that does nothing.

        This block creates a circuit with the specified number of qubits but applies
        no operations, effectively implementing the identity transformation.

        Args:
            n_qubits: The number of qubits to include.
            target_qubits: The target qubits will act on when added to a Block object.
            name: The name of the block.
        """
        super().__init__(n_qubits, target_qubits, name=name)

    def build_vanilla(self) -> None:
        pass  # identity: no gates
