from typing import List, Optional

import numpy as np

from qarp.blocks._block import SimpleBlock


class SynthesizedUnitaryBlock(SimpleBlock):
    """Pattern A leaf: synthesize an arbitrary `2^n × 2^n` unitary into a circuit.

    Delegates to ``self.unitary_synthesis(...)`` (qarpx C++ Quantum Shannon
    Decomposition) in ``build_vanilla()``.
    """

    def __init__(
        self,
        unitary_matrix: np.ndarray,
        target_qubits: Optional[List[int]] = None,
        name: str = "SynthUnitaryBlock",
    ):
        """
        Args:
            unitary_matrix: A square unitary matrix of dimension `2^n × 2^n`.
                Validated for unitarity (`U @ U† ≈ I`) within `1e-10`.
            target_qubits, name: standard Block kwargs.
        """
        size = unitary_matrix.shape[0]
        if size == 0 or size & (size - 1) != 0:
            raise ValueError("Unitary matrix size must be a power of 2")
        n_qubits = int(np.log2(size))

        super().__init__(
            n_qubits,
            target_qubits=target_qubits,
            name=name,
        )
        # Named `target_unitary`, not `unitary_matrix`: an attribute of that
        # name would shadow the inherited `Block.unitary_matrix()` method.
        self.target_unitary = np.asarray(unitary_matrix, dtype=complex)
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if self.target_unitary.shape[0] != self.target_unitary.shape[1]:
            raise ValueError("Unitary matrix must be square")
        size = self.target_unitary.shape[0]
        if size == 0 or size & (size - 1) != 0:
            raise ValueError("Unitary matrix size must be a power of 2")
        identity = np.eye(size)
        product = self.target_unitary @ self.target_unitary.conj().T
        tol = 1e-10
        if not np.allclose(product, identity, atol=tol):
            max_err = float(np.max(np.abs(product - identity)))
            raise ValueError(
                f"Input matrix is not unitary: max|U U† - I| = {max_err:.2e} > {tol:.0e}. "
                f"Fix the precision where the matrix is produced: prefer "
                f"scipy.linalg.svd(M, lapack_driver='gesvd') over numpy.linalg.svd "
                f"and reunitarize via V @ Wh."
            )

    def build_vanilla(self) -> None:
        self.unitary_synthesis(self.target_unitary)
