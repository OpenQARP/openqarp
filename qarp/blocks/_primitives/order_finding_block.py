"""Measurement-free quantum order-finding circuit."""

from math import gcd
from typing import List, Optional

from .._block import CompositeBlockBase, ControlledBlock, SimpleBlock
from .hn_block import HnBlock
from .modular_multiplication_block import ModularMultiplicationBlock
from .qft_block import QFTBlock


class OrderFindingBlock(CompositeBlockBase):
    r"""Reference circuit for finding the multiplicative order of a base.

    The LSB-indexed register layout is ``counting`` followed by ``work``. The
    circuit prepares the work register in ``|1>``, computes ``a**x mod N`` by
    controlled modular multiplications, and applies an inverse QFT to the
    counting register. Measurement is deliberately left to a sampler.

    Args:
        base: Integer satisfying ``1 < base < modulus`` and coprime to it.
        modulus: Integer modulus greater than one.
        n_counting_qubits: Counting-register width. Defaults to twice the work
            width and cannot be smaller than that value.
        target_qubits: Qubits occupied when embedded in a parent block.
        name: Block name.

    Note:
        Modular arithmetic is synthesized by
        :class:`ModularMultiplicationBlock` and therefore has the same
        six-work-qubit reference limit and exponential gate cost.
    """

    def __init__(
        self,
        base: int,
        modulus: int,
        n_counting_qubits: Optional[int] = None,
        target_qubits: Optional[List[int]] = None,
        name: str = "OrderFinding",
    ) -> None:
        if isinstance(modulus, bool) or not isinstance(modulus, int):
            raise TypeError("modulus must be an integer")
        if modulus <= 1:
            raise ValueError("modulus must be greater than one")
        if isinstance(base, bool) or not isinstance(base, int):
            raise TypeError("base must be an integer")
        if not 1 < base < modulus:
            raise ValueError("base must satisfy 1 < base < modulus")
        if gcd(base, modulus) != 1:
            raise ValueError("base and modulus must be coprime")

        n_work_qubits = (modulus - 1).bit_length()
        if n_work_qubits > ModularMultiplicationBlock.MAX_REFERENCE_WORK_QUBITS:
            raise ValueError(
                "OrderFindingBlock's exact reference arithmetic supports at most "
                f"{ModularMultiplicationBlock.MAX_REFERENCE_WORK_QUBITS} work qubits; "
                f"modulus={modulus} requires {n_work_qubits}."
            )

        if n_counting_qubits is None:
            n_counting_qubits = 2 * n_work_qubits
        elif isinstance(n_counting_qubits, bool) or not isinstance(n_counting_qubits, int):
            raise TypeError("n_counting_qubits must be an integer")
        if n_counting_qubits < 2 * n_work_qubits:
            raise ValueError(
                "n_counting_qubits must be at least twice the work-register width "
                f"({2 * n_work_qubits})"
            )

        self.base = base
        self.modulus = modulus
        self.n_counting_qubits = n_counting_qubits
        self.n_work_qubits = n_work_qubits
        super().__init__(
            n_qubits=n_counting_qubits + n_work_qubits,
            target_qubits=target_qubits,
            name=name,
        )

    @property
    def counting_qubits(self) -> list[int]:
        """Local counting-register qubits, ordered least-significant first."""
        return list(range(self.n_counting_qubits))

    @property
    def work_qubits(self) -> list[int]:
        """Local work-register qubits, ordered least-significant first."""
        return list(range(self.n_counting_qubits, self.n_qubits))

    def build_vanilla(self) -> None:
        self.add_wired_child(
            HnBlock(
                self.n_counting_qubits,
                target_qubits=self.counting_qubits,
                name="CountingH",
            )
        )

        prepare_one = SimpleBlock(
            self.n_work_qubits,
            target_qubits=self.work_qubits,
            name="PrepareWorkOne",
        )
        prepare_one.x(0)
        self.add_wired_child(prepare_one)

        self.add_child(self._modular_exponentiation_block())

        inverse_qft = QFTBlock(self.n_counting_qubits).dagger().build()
        inverse_qft.target_qubits = self.counting_qubits
        self.add_child(inverse_qft)

    def _modular_exponentiation_block(self) -> CompositeBlockBase:
        """Build ``|x>|w> -> |x>|a**x*w mod N>`` in the local register frame."""
        modular_exponentiation = CompositeBlockBase(
            self.n_qubits,
            name="ModularExponentiation",
        )
        for exponent_bit, control_qubit in enumerate(self.counting_qubits):
            multiplier = pow(self.base, 2**exponent_bit, self.modulus)
            if multiplier == 1:
                continue
            multiplication = ModularMultiplicationBlock(multiplier, self.modulus).build()
            controlled = ControlledBlock(
                multiplication,
                num_controls=1,
                ctrl_state=[True],
                name=f"C-M({multiplier})@q{control_qubit}",
            ).build()
            controlled.target_qubits = [control_qubit, *self.work_qubits]
            modular_exponentiation.add_child(controlled)
        return modular_exponentiation.build()
