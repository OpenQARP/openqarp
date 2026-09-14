from typing import List, Optional

import numpy as np

from qarp.blocks._block import SimpleBlock
from qarp.blocks._prepares_known_state import prepares_known_state


@prepares_known_state
class ComputationalBasisStateBlock(SimpleBlock):
    def __init__(
        self,
        basis_state: List[int],
        target_qubits: Optional[List[int]] = None,
        name: Optional[str] = None,
    ):
        """Prepare a basis state on an empty circuit by applying X gates on the relevant qubits.

        Args:
            basis_state: A list of 0s and 1s representing the basis state.
            target_qubits: The target qubits will act on when added to a Block object.
        """
        self.basis_state = basis_state
        if name is None:
            # Display only.  basis_state is LSB-ordered (index 0 = qubit 0), so
            # the bits are reversed before being read as an integer.
            s = [str(int(i)) for i in reversed(self.basis_state)]
            state_string = "".join(s)
            name = f"U_{int(state_string, 2)}"

        super().__init__(
            len(basis_state),
            target_qubits=target_qubits,
            name=name,
        )

    def build_vanilla(self) -> None:
        # Bulk emit X on every "1" position in one nanobind crossing.
        ones = [idx for idx, bit in enumerate(self.basis_state) if bit == 1]
        if ones:
            self.x(ones)

    def target_statevector(self) -> np.ndarray:
        """``|b⟩`` for the requested bit string."""
        psi = np.zeros(2**self.n_qubits, dtype=complex)
        psi[sum(int(bit) << i for i, bit in enumerate(self.basis_state))] = 1.0
        return psi
