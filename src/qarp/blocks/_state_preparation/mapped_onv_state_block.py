from typing import List, Optional

import numpy as np

from ...operators import JordanWigner, Mapping
from ...operators.onv import Onv
from .. import SimpleBlock
from .._prepares_known_state import prepares_known_state


@prepares_known_state
class MappedONVStateBlock(SimpleBlock):
    def __init__(
        self,
        occupation_number_vector: Onv,
        mapping: Optional[Mapping] = None,
        target_qubits: Optional[List[int]] = None,
        name=None,
    ):
        """Prepare the basis state corresponding to the input occupation number vector given a mapping.

        Args:
            occupation_number_vector: The ONV in Fock space (abab list).
            mapping: the mapping to use to obtain the relevant basis state.
            target_qubits: The target qubits will act on when added to a Block object.
        """
        if mapping is None:
            mapping = JordanWigner()
        self.mapping = mapping
        self.occupation_number_vector = occupation_number_vector
        self.basis_state = mapping.encode_state(occupation_number_vector)
        if name is None:
            # Display only.  basis_state is LSB-ordered (index 0 = qubit 0), so
            # the bits are reversed before being read as an integer.
            s = [str(int(i)) for i in reversed(self.basis_state)]
            state_string = "".join(s)
            name = f"U_{int(state_string, 2)}"

        super().__init__(
            len(self.basis_state),
            target_qubits=target_qubits,
            name=name,
        )

    def build_vanilla(self):
        ones = [idx for idx, bit in enumerate(self.basis_state) if bit == 1]
        if ones:
            self.x(ones)

    def target_statevector(self) -> np.ndarray:
        """``|b⟩`` for the ONV encoded under ``mapping``."""
        psi = np.zeros(2**self.n_qubits, dtype=complex)
        psi[sum(int(bit) << i for i, bit in enumerate(self.basis_state))] = 1.0
        return psi
