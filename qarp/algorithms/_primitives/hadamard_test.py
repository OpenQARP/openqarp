"""HadamardTest primitive.

Wraps :class:`qarp.blocks.HadamardTestBlock` (the qarpx composite)
and post-processes the engine's :class:`qx.SamplingResult` to extract
``Re⟨ψ|U|ψ⟩`` and/or ``Im⟨ψ|U|ψ⟩``.

Submits one or two ``sub_blocks`` (real / imaginary) and combines their
ancilla statistics in :meth:`run`.
"""

from typing import Optional, Self, Union

import qarpx as qx

from ..._types import Shots
from ...blocks import AnyBlock, IdentityBlock
from ...blocks._block import CompositeBlockBase
from ...blocks._primitives import HadamardTestBlock
from .primitive_algorithm import PrimitiveAlgorithm
from .target import Target


class HadamardTest(PrimitiveAlgorithm):
    gradient_kind = "expectation"  # every circuit's statistic is bilinear in its state
    supported_targets = frozenset(
        {Target.EXPECTATION_VALUE, Target.OVERLAP, Target.TRANSITION_AMPLITUDE}
    )

    def __init__(
        self,
        bra: Optional[AnyBlock] = None,
        operator: Optional[AnyBlock] = None,
        ket: Optional[AnyBlock] = None,
        real: bool = True,
        imaginary: bool = True,
        n_shots: Optional[Union[int, Shots]] = None,
    ):
        """
        Args:
            bra: Optional state preparation block for ⟨ψ|.
            operator: The unitary operator U to test.
            ket: State preparation block for |ψ⟩.
            real: If True, sub_blocks include the real-part circuit.
            imaginary: If True, sub_blocks include the imaginary-part circuit.
            n_shots: Number of measurement shots; ``None`` defers to the engine default.
        """
        if not real and not imaginary:
            raise ValueError("At least one of real or imaginary must be True")
        super().__init__(
            ket=ket, bra=bra, operator=operator, n_shots=n_shots, target=Target.SAMPLING
        )
        self.real = real
        self.imaginary = imaginary

        self.result_real: Optional[float] = None
        self.result_imaginary: Optional[float] = None
        self.result: Optional[complex] = None

    def _validate_inputs(self) -> None:
        if not isinstance(self.ket, qx.Block):
            raise TypeError("ket must be a Block instance")
        if self.bra is not None and not isinstance(self.bra, qx.Block):
            raise TypeError("bra must be a Block instance or None")
        if self.operator is not None and not isinstance(self.operator, qx.Block):
            raise TypeError("operator must be a Block instance or None")

    def build(self) -> Self:
        self._validate_inputs()

        n_qubits = self.ket.n_qubits
        target = self.infer_target()

        if target == Target.EXPECTATION_VALUE:
            state = self.ket
            unitary = self.operator
            unitary_dagger = None
        elif target == Target.OVERLAP:
            state = IdentityBlock(n_qubits)
            unitary = self.ket
            unitary_dagger = self.bra
        elif target == Target.TRANSITION_AMPLITUDE:
            state = IdentityBlock(n_qubits)
            ko = CompositeBlockBase(n_qubits=n_qubits, name="ket+operator")
            ket_built = self.ket.build()
            ket_built.target_qubits = list(range(n_qubits))
            op_built = self.operator.build()
            op_built.target_qubits = list(range(n_qubits))
            ko.add_child(ket_built)
            ko.add_child(op_built)
            ko.build()
            unitary = ko
            unitary_dagger = self.bra
        else:
            raise RuntimeError(f"Unhandled target {target}")

        self.sub_blocks = []
        if self.real:
            blk_re = HadamardTestBlock(
                state=state,
                unitary=unitary,
                unitary_dagger=unitary_dagger,
                estimate_imaginary=False,
                measure=True,
            )
            blk_re.build()
            self.sub_blocks.append(blk_re)
        if self.imaginary:
            blk_im = HadamardTestBlock(
                state=state,
                unitary=unitary,
                unitary_dagger=unitary_dagger,
                estimate_imaginary=True,
                measure=True,
            )
            blk_im.build()
            self.sub_blocks.append(blk_im)

        return self

    @staticmethod
    def _ancilla_zero_prob(sr) -> float:
        """``P(ancilla=0) = (sum of counts where outcome bit 0 == 0) / n_shots``."""
        zero_count = sum(c for outcome, c in sr.counts.items() if (outcome & 1) == 0)
        return zero_count / sr.n_shots

    def run(self, results: list) -> complex:
        idx = 0
        real_part = 0.0
        imaginary_part = 0.0

        if self.real:
            real_part = 2 * self._ancilla_zero_prob(results[idx]) - 1
            idx += 1
        if self.imaginary:
            imaginary_part = 2 * self._ancilla_zero_prob(results[idx]) - 1

        if self.real and self.imaginary:
            self.result_real = real_part
            self.result_imaginary = imaginary_part
            self.result = complex(real_part, imaginary_part)
            return self.result
        if self.real:
            self.result_real = real_part
            return real_part
        self.result_imaginary = imaginary_part
        return imaginary_part

    @property
    def expectation_type(self) -> str:
        if self.real and self.imaginary:
            return "complex"
        if self.real:
            return "real"
        return "imaginary"

    def __repr__(self) -> str:
        return (
            f"HadamardTest(target={self.target}, real={self.real}, "
            f"imaginary={self.imaginary}, n_shots={self.n_shots})"
        )
