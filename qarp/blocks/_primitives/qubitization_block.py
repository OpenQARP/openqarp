from typing import List, Optional, Union

import numpy as np

from qarp.operators import QubitOperator

from .._block import CompositeBlockBase
from .block_encoding_block import BlockEncodingBlock
from .reflection_block import ReflectionBlock


class QubitizationBlock(CompositeBlockBase):
    r"""Pattern B composite: qubitization walk operator built from a
    ``ReflectionBlock`` on the ancilla register followed by a
    ``BlockEncodingBlock``.

    The walk operator is ``W = R · BE`` where:

    * ``R = 2|0…0⟩⟨0…0| - I`` on the LCU-control register (the first
      ``num_controls`` qubits, where ``num_controls = ⌈log₂ N_LCU⌉``).
    * ``BE`` block-encodes ``A / λ`` on the full register.

    Iterating ``W`` realises a quantum walk whose spectrum encodes the
    eigenphases of the block-encoded operator, the foundation of QSP / QSVT.
    """

    def __init__(
        self,
        A: Union[np.ndarray, QubitOperator],
        operator_name: str = "Operator",
        target_qubits: Optional[List[int]] = None,
        name: str = "Qubitization",
    ):
        """Args:
        A: Operator to qubitize, either an `np.ndarray` or a
            `openfermion.QubitOperator`.
        operator_name: Name forwarded to the inner ``BlockEncodingBlock``.
        target_qubits, name: standard Block kwargs.
        """
        if not isinstance(A, (np.ndarray, QubitOperator)):
            raise TypeError("Expected a np.ndarray or QubitOperator in QubitizationBlock")

        self.A = A
        self.operator_name = operator_name

        # Construct a BE up-front for sizing and so consumers can read
        # ``self.BE.lambda_norm`` / ``self.BE.unitaries`` before .build() —
        # ``self.BE`` is stable public surface.
        self.BE = BlockEncodingBlock(A, name=operator_name)
        self.lambda_factor = self.BE.lambda_norm
        self.unitaries_qubits = list(range(self.BE.num_controls, self.BE.n_qubits))

        super().__init__(
            n_qubits=self.BE.n_qubits,
            target_qubits=target_qubits,
            name=name,
        )

    def build_vanilla(self) -> None:
        n_anc = self.BE.num_controls

        # 1. Reflection about |0…0⟩ on the ancilla register.
        refl = ReflectionBlock(n_anc)
        refl.target_qubits = list(range(n_anc))
        self.add_wired_child(refl)

        # 2. Block encoding on the full register.  We rebuild a fresh BE here
        # — the one stashed on ``self.BE`` is used only for sizing / metadata
        # exposure; building it twice is harmless (both produce identical
        # circuits) and keeps add_child semantics simple.
        be = BlockEncodingBlock(self.A, name=self.operator_name)
        be.target_qubits = list(range(self.n_qubits))
        self.add_wired_child(be)
