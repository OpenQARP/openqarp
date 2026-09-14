"""InterferometricTest primitive.

A composition wrapper that estimates ``Re⟨ψ|U|ψ⟩`` and ``Im⟨ψ|U|ψ⟩`` by
running the configured ``sampling_algorithm`` (MirrorTest or SWAPTest) on
three different state preparations: the bare ``ket``, a GHZ-like
"real" combination, and a dephased GHZ-like "imaginary" combination.
"""

from typing import Optional, Self, Union

import qarpx as qx

from ..._types import Shots
from ...blocks import AnyBlock, ComputationalBasisStateBlock, GHZLikeStateBlock
from .mirror_test import MirrorTest
from .primitive_algorithm import PrimitiveAlgorithm
from .swap_test import SWAPTest
from .target import Target


class InterferometricTest(PrimitiveAlgorithm):
    gradient_kind = "expectation"  # every circuit's statistic is bilinear in its state
    supported_targets = frozenset({Target.EXPECTATION_VALUE})

    def __init__(
        self,
        bra: Optional[ComputationalBasisStateBlock] = None,
        operator: Optional[AnyBlock] = None,
        ket: Optional[ComputationalBasisStateBlock] = None,
        real: bool = True,
        imaginary: bool = True,
        sampling_algorithm: Optional[PrimitiveAlgorithm] = None,
        n_shots: Optional[Union[int, Shots]] = None,
    ):
        """
        Args:
            bra: Optional bra state preparation block.
            operator: The unitary operator U to test.
            ket: State preparation block for |ψ⟩ (must be a
                ``ComputationalBasisStateBlock`` and non-zero).
            real: If True, include the real-part sub-circuit.
            imaginary: If True, include the imaginary-part sub-circuit.
            sampling_algorithm: ``MirrorTest()`` or ``SWAPTest()`` (default: ``MirrorTest()``).
            n_shots: Number of shots; ``None`` defers to the engine default.
        """
        if not real and not imaginary:
            raise ValueError("At least one of real or imaginary must be True")
        if sampling_algorithm is None:
            sampling_algorithm = MirrorTest()
        if not isinstance(sampling_algorithm, (MirrorTest, SWAPTest)):
            raise ValueError("sampling_algorithm must be MirrorTest() or SWAPTest()")

        super().__init__(
            ket=ket, bra=bra, operator=operator, n_shots=n_shots, target=Target.EXPECTATION_VALUE
        )
        self.real = real
        self.imaginary = imaginary
        self.sampling_algorithm = sampling_algorithm

        self.result_real: Optional[float] = None
        self.result_imaginary: Optional[float] = None
        self.result: Optional[complex] = None

        self._validate_inputs()

        if not isinstance(ket, ComputationalBasisStateBlock):
            raise TypeError("ket must be a ComputationalBasisStateBlock instance")
        ket.build()
        self.n_qubits = ket.n_qubits
        if ket.basis_state == [0] * self.n_qubits:
            raise ValueError(
                "ket must not be the all-zero state (orthogonality with the reference would fail)"
            )
        self.reference = ComputationalBasisStateBlock([0] * self.n_qubits)

    def _validate_inputs(self) -> None:
        if not isinstance(self.ket, ComputationalBasisStateBlock):
            raise TypeError("ket must be a ComputationalBasisStateBlock instance")
        if self.bra is not None and not isinstance(self.bra, ComputationalBasisStateBlock):
            raise TypeError("bra must be a ComputationalBasisStateBlock instance or None")
        if self.operator is not None and not isinstance(self.operator, qx.Block):
            raise TypeError("operator must be a Block instance or None")

    def _configure(self, sa: PrimitiveAlgorithm, *, bra: AnyBlock, ket: AnyBlock) -> AnyBlock:
        """Set up ``sa`` (MirrorTest or SWAPTest) for one sub-block build."""
        sa.bra = bra
        sa.operator = self.operator
        sa.ket = ket
        sa.n_shots = self.n_shots
        sa.build()
        return sa.sub_blocks[0]

    def build(self) -> Self:
        self._validate_inputs()
        self.sub_blocks = []

        # 1. Reference: bra = ket = ket  (gives ⟨ψ|U|ψ⟩ probability).
        self.sub_blocks.append(self._configure(self.sampling_algorithm, bra=self.ket, ket=self.ket))

        ghzlike_real = None
        if self.real:
            ghzlike_real = GHZLikeStateBlock(self.ket.basis_state)
            self.sub_blocks.append(
                self._configure(self.sampling_algorithm, bra=ghzlike_real, ket=ghzlike_real)
            )
        if self.imaginary:
            if ghzlike_real is None:
                ghzlike_real = GHZLikeStateBlock(self.ket.basis_state)
            ghzlike_imag = GHZLikeStateBlock(self.ket.basis_state, dephase=True)
            self.sub_blocks.append(
                self._configure(self.sampling_algorithm, bra=ghzlike_imag, ket=ghzlike_real)
            )

        return self

    def run(self, results: list) -> complex:
        idx = 0
        # Reference probability via the sampling_algorithm's own post-processor.
        p_ref = self.sampling_algorithm.run([results[idx]])
        idx += 1

        real_part = 0.0
        imaginary_part = 0.0
        if self.real:
            p_real = self.sampling_algorithm.run([results[idx]])
            real_part = 2 * p_real - 0.5 * (1 + p_ref)
            idx += 1
        if self.imaginary:
            p_imag = self.sampling_algorithm.run([results[idx]])
            imaginary_part = -2 * p_imag + 0.5 * (1 + p_ref)

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
            f"InterferometricTest(target={self.target}, real={self.real}, "
            f"imaginary={self.imaginary}, "
            f"sampling_algorithm={type(self.sampling_algorithm).__name__}, "
            f"n_shots={self.n_shots})"
        )
