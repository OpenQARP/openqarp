from typing import List, Optional

import numpy as np

from qarp.blocks._block import SimpleBlock


class ProjectedControlPhaseBlock(SimpleBlock):
    """Pattern A leaf: subspace-selective phase rotation.

    Implements the diagonal unitary that applies ``e^{+iφ}`` to the first
    ``dim`` computational basis states and ``e^{-iφ}`` to the remaining
    ``2^n_qubits − dim``.  Useful as a projection-like operator in QSVT-style
    constructions.

    Delegates to ``self.diagonal_unitary(...)`` (qarpx Shende-Bullock-Markov
    synthesis).
    """

    def __init__(
        self,
        phase: float,
        dim: int,
        n_qubits: int,
        target_qubits: Optional[List[int]] = None,
        name: str = "PCP",
    ):
        """
        Args:
            phase: Phase angle ``φ`` in radians.
            dim: Number of basis states receiving ``e^{+iφ}``.  The remaining
                ``2^n_qubits − dim`` states receive ``e^{-iφ}``.
            n_qubits: Total qubit count.
            target_qubits, name: standard Block kwargs.

        Raises:
            ValueError: If ``dim > 2**n_qubits``.
        """
        if dim > 2**n_qubits:
            raise ValueError("dim cannot be larger than 2**n_qubits in ProjectedControlPhaseBlock")
        self.phase = phase
        self.dim = dim
        super().__init__(
            n_qubits,
            target_qubits=target_qubits,
            name=name,
        )

    def build_vanilla(self) -> None:
        N = 2**self.n_qubits
        plus = np.exp(1j * self.phase)
        minus = np.exp(-1j * self.phase)
        diag = [plus] * self.dim + [minus] * (N - self.dim)
        self.diagonal_unitary(diag)
