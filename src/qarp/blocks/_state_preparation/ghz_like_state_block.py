from typing import List, Optional

import numpy as np

from .._block import SimpleBlock
from .._prepares_known_state import prepares_known_state


@prepares_known_state
class GHZLikeStateBlock(SimpleBlock):
    def __init__(
        self,
        basis_state: List[int],
        dephase: bool = False,
        target_qubits: Optional[List[int]] = None,
        name=None,
    ):
        """GHZ-like state preparation with optional dephasing.

        GHZLikeStateBlock prepares a multi-qubit entangled state resembling a GHZ state but
        localized to specific basis states. It creates entanglement by applying a Hadamard
        gate to the first qubit marked as |1⟩ in the basis state, then cascading CNOT gates
        to entangle subsequent |1⟩-marked qubits. An optional S† gate provides phase control,
        enabling preparation of states like (|000...⟩ - i|111...⟩)/√2 useful for quantum
        algorithms requiring controlled superposition with specific phase relationships.

        Args:
            basis_state: A list of 0s and 1s representing which qubits participate in the GHZ-like state.
            dephase: If True, applies S† gate to introduce phase factor i in the superposition.
            target_qubits: The target qubits this block acts on when added to a larger circuit.
            name: Optional custom name for the block. If None, auto-generated from basis_state and dephase.
        """
        self.basis_state = basis_state
        if name is None:
            s = [str(int(i)) for i in self.basis_state]
            state_string = "".join(s)
            if not dephase:
                name = f"GHZ(+, {state_string})"
            else:
                name = f"GHZ(i, {state_string})"

        super().__init__(
            len(self.basis_state),
            target_qubits=target_qubits,
            name=name,
        )
        self.dephase = dephase

    def build_vanilla(self):
        """
        Build a GHZ-like state by applying H on the first qubit with basis_state=1,
        then cascading CNOTs from that qubit to all other qubits with basis_state=1.
        """
        ones_indices = [idx for idx, bit in enumerate(self.basis_state) if bit == 1]

        if ones_indices:
            first_one_idx = ones_indices[0]
            self.h(first_one_idx)
            if self.dephase:
                self.sdg(first_one_idx)

            for i in range(len(ones_indices) - 1):
                self.cx(ones_indices[i], ones_indices[i + 1])

    def target_statevector(self) -> np.ndarray:
        r"""``(|0…0⟩ + p·|mask⟩)/√2``, with ``p = -i`` when dephasing.

        ``Sdg = diag(1, -i)`` (§2.2) follows the Hadamard, so the marked branch
        picks up ``-i``, not ``+i``.  An empty mask leaves ``|0…0⟩`` unentangled.
        """
        psi = np.zeros(2**self.n_qubits, dtype=complex)
        mask = sum(1 << i for i, bit in enumerate(self.basis_state) if bit == 1)
        if mask == 0:
            psi[0] = 1.0
            return psi
        psi[0] = 1 / np.sqrt(2)
        psi[mask] = (-1j if self.dephase else 1.0) / np.sqrt(2)
        return psi
