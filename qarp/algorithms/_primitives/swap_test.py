"""SWAPTest primitive.

Wraps :class:`qarp.blocks.SWAPTestBlock` (the qarpx composite block
that implements the SWAP test) and post-processes the engine's
:class:`qx.SamplingResult` to produce the ``|⟨bra|ket⟩|²`` estimate.

If an ``operator`` is provided, the ket register is replaced with
``ket · operator`` (so the test estimates ``|⟨bra|U|ket⟩|²``).
"""

from typing import Optional, Self, Union

import qarpx as qx

from ..._types import Shots
from ...blocks import AnyBlock
from ...blocks._block import CompositeBlockBase
from ...blocks._primitives import SWAPTestBlock
from .primitive_algorithm import PrimitiveAlgorithm
from .target import Target


class SWAPTest(PrimitiveAlgorithm):
    gradient_kind = "expectation"  # every circuit's statistic is bilinear in its state
    returns_probability = True  # run() is |⟨bra|ket⟩|², not the amplitude
    supported_targets = frozenset({Target.OVERLAP})

    def __init__(
        self,
        bra: Optional[AnyBlock] = None,
        operator: Optional[AnyBlock] = None,
        ket: Optional[AnyBlock] = None,
        n_shots: Optional[Union[int, Shots]] = None,
    ):
        """
        Args:
            bra: Block preparing the bra state ``|ψ⟩``.
            operator: Optional unitary ``U`` applied to the ket register.
            ket: Block preparing the ket state ``|φ⟩``.
            n_shots: Number of measurement shots; ``None`` defers to the engine default.
        """
        super().__init__(
            ket=ket, bra=bra, operator=operator, n_shots=n_shots, target=Target.OVERLAP
        )
        self.result: Optional[float] = None

    def _validate_inputs(self) -> None:
        if not isinstance(self.bra, qx.Block):
            raise TypeError("bra must be a Block instance")
        if not isinstance(self.ket, qx.Block):
            raise TypeError("ket must be a Block instance")
        if self.operator is not None and not isinstance(self.operator, qx.Block):
            raise TypeError("operator must be a Block instance or None")

    def build(self) -> Self:
        self._validate_inputs()

        # If operator is provided, the ket register effectively prepares
        # ``ket`` then applies ``operator`` — wrap them in a composite.
        if self.operator is not None:
            n_state = self.ket.n_qubits
            ket_built = self.ket.build()
            ket_built.target_qubits = list(range(n_state))
            op_built = self.operator.build()
            op_built.target_qubits = list(range(n_state))
            ket_register = CompositeBlockBase(n_qubits=n_state, name="ket+operator")
            ket_register.add_child(ket_built)
            ket_register.add_child(op_built)
            ket_register.build()
        else:
            ket_register = self.ket.build()

        block = SWAPTestBlock(bra=self.bra, ket=ket_register, measure=True)
        block.build()

        self.sub_blocks = [block]
        return self

    def run(self, results: list) -> float:
        """``2·P(ancilla=0) − 1 = |⟨bra|ket⟩|²``.

        ``SamplingResult.counts`` is keyed on the full all-qubit outcome
        integer; the ancilla is qubit 0 so its measured value is bit 0.
        """
        sr = results[0]
        n_shots = sr.n_shots
        zero_count = sum(c for outcome, c in sr.counts.items() if (outcome & 1) == 0)
        p0 = zero_count / n_shots
        self.result = 2 * p0 - 1
        return self.result

    def __repr__(self) -> str:
        return f"SWAPTest(target={self.target}, n_shots={self.n_shots})"
