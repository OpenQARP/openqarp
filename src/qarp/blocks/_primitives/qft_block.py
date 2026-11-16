from typing import List, Optional

import numpy as np

from qarp.blocks._block import SimpleBlock


class QFTBlock(SimpleBlock):
    def __init__(
        self,
        n_qubits: int,
        target_qubits: Optional[List[int]] = None,
        name: str = "QFT",
    ):
        """Quantum Fourier Transform (QFT) block for quantum phase estimation and related algorithms.

        Maps computational basis states to their Fourier basis via Hadamards + controlled phase
        rotations + a SWAP-based bit reversal.

        Args:
            n_qubits: The number of qubits to perform the transform over.
            target_qubits: The target qubits when added to a parent block.
            name: Optional custom name for the block.
        """
        super().__init__(n_qubits, target_qubits, name=name)

    def build_vanilla(self) -> None:
        if self.n_qubits is None:
            raise RuntimeError("Cannot build QFT: n_qubits is undefined")
        # Textbook QFT (+ω convention): H on q_{n-1}, then controlled phases
        # from the lower qubits onto it, moving down to q_0, then SWAPs to
        # reverse the register.  Block.cp(c, t, θ) applies P(θ)=diag(1, e^{iθ})
        # to the target conditioned on the control, so positive CP angles
        # directly realise the textbook R_k = diag(1, e^{2πi/2^k}) rotations.
        n = self.n_qubits
        for i in range(n - 1, -1, -1):
            self.h(i)
            for j in range(i - 1, -1, -1):
                self.cp(j, i, np.pi / 2 ** (i - j))
        for k in range(n // 2):
            self.swap(k, n - k - 1)
