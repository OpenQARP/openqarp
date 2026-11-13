from typing import List, Optional

from .._block import SimpleBlock


class XnBlock(SimpleBlock):
    def __init__(
        self,
        n_qubits: int,
        target_qubits: Optional[List[int]] = None,
        name: str = "Xn",
    ):
        """Constructs a layer of X gates (Xn) circuit box.

        Args:
            n_qubits: The number of qubits to include.
            target_qubits: The target qubits will act on when added to a Block object.
            name: Optional custom name for the block.
        """
        super().__init__(n_qubits, target_qubits, name=name)

    def build_vanilla(self) -> None:
        # Bulk emit: one Python→C++ crossing for the whole layer.
        self.x(list(range(self.n_qubits)))
