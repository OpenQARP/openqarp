from copy import deepcopy
from typing import List, Optional

import numpy as np
from scipy.linalg import expm

from qarp.blocks._block import SimpleBlock
from qarp.operators import QubitOperator


class SynthesizedTimeEvolutionBlock(SimpleBlock):
    """Pattern A leaf: synthesize ``U(t) = exp(-i H t)`` for a Hamiltonian H.

    Builds the time-evolution unitary numerically via ``scipy.linalg.expm`` and
    delegates to ``self.unitary_synthesis(...)`` (qarpx Quantum Shannon
    Decomposition) for the circuit synthesis.
    """

    def __init__(
        self,
        operator: QubitOperator,
        n_qubits: int,
        time: Optional[float] = None,
        target_qubits: Optional[List[int]] = None,
        name: str = "SynthTimeEvoBlock",
    ):
        """
        Args:
            operator: The Hamiltonian (must be Hermitian).
            n_qubits: Number of qubits the operator acts on.
            time: Evolution time `t`.  May be set later via ``set_time``;
                ``build()`` will raise if called before `time` is set.
            target_qubits, name: standard Block kwargs.
        """
        super().__init__(
            n_qubits,
            target_qubits=target_qubits,
            name=name,
        )
        self.operator = operator
        self.time = time
        # qarpx-LSB (qubit q ↔ bit q) — the convention synthesis expects.
        self.operator_matrix = operator.sparse_matrix(n_qubits).toarray()
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        expected_dim = 2**self.n_qubits
        if self.operator_matrix.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"Operator matrix dimension {self.operator_matrix.shape} does not match "
                f"expected size ({expected_dim}, {expected_dim}) for n_qubits={self.n_qubits}"
            )
        if not np.allclose(self.operator_matrix, self.operator_matrix.conj().T, atol=1e-10):
            raise ValueError("Input operator is not Hermitian (required for time evolution)")

    def set_time(self, time_value: float) -> "SynthesizedTimeEvolutionBlock":
        new_object = deepcopy(self)
        new_object.time = time_value
        # ``Block.__deepcopy__`` preserves ``built_=True`` and the C++
        # commands_ buffer, so a plain ``build()`` short-circuits on the
        # idempotency guard.  Reset both flags so ``build_vanilla`` re-emits
        # the synthesis at the new time.
        new_object._built = False
        new_object.set_commands([])
        new_object.set_built(False)
        new_object.build()
        return new_object

    def build_vanilla(self) -> None:
        if self.time is None:
            raise ValueError("Time parameter is not set. Please set time before building.")

        unitary_matrix = expm(-1j * self.operator_matrix * self.time)
        self.unitary_synthesis(unitary_matrix)
