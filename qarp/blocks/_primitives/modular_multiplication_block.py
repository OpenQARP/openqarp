"""Exact reference synthesis of reversible modular multiplication."""

from math import gcd
from typing import List, Optional

from .._block import SimpleBlock


class ModularMultiplicationBlock(SimpleBlock):
    r"""Permutation implementing multiplication modulo a small integer.

    For ``n = (modulus - 1).bit_length()`` this block acts on the complete
    ``2**n``-dimensional Hilbert space as

    .. math::

        |x\rangle \mapsto
        \begin{cases}
            |m x \bmod N\rangle, & x < N,\\
            |x\rangle, & x \geq N.
        \end{cases}

    The reference synthesizer enumerates basis labels and is exponential in
    the work-register width.  It is intended for exact small-integer examples,
    not cryptographic-scale factoring.  ``MAX_REFERENCE_WORK_QUBITS`` caps the
    width at six; measured exact sampling of ``OrderFindingBlock(2, N)`` on the
    default engine takes 2 s at six work qubits (``N=63``), 31 s at seven
    (``N=127``) and 155 s at eight (``N=255``), the statevector of ``3n``
    total qubits dominating.  Raising the constant needs new evidence.

    Args:
        multiplier: Integer multiplier. It is normalized modulo ``modulus``
            and must be coprime to it.
        modulus: Integer modulus greater than one.
        target_qubits: Qubits occupied when embedded in a parent block.
        name: Block name.
    """

    MAX_REFERENCE_WORK_QUBITS = 6

    def __init__(
        self,
        multiplier: int,
        modulus: int,
        target_qubits: Optional[List[int]] = None,
        name: str = "ModularMultiplication",
    ) -> None:
        if isinstance(modulus, bool) or not isinstance(modulus, int):
            raise TypeError("modulus must be an integer")
        if modulus <= 1:
            raise ValueError("modulus must be greater than one")
        if isinstance(multiplier, bool) or not isinstance(multiplier, int):
            raise TypeError("multiplier must be an integer")

        normalized_multiplier = multiplier % modulus
        if gcd(normalized_multiplier, modulus) != 1:
            raise ValueError("multiplier and modulus must be coprime")

        n_work_qubits = (modulus - 1).bit_length()
        if n_work_qubits > self.MAX_REFERENCE_WORK_QUBITS:
            raise ValueError(
                "ModularMultiplicationBlock's exact reference synthesizer "
                f"supports at most {self.MAX_REFERENCE_WORK_QUBITS} work qubits; "
                f"modulus={modulus} requires {n_work_qubits}."
            )

        self.multiplier = normalized_multiplier
        self.modulus = modulus
        self.n_work_qubits = n_work_qubits
        super().__init__(
            n_qubits=n_work_qubits,
            target_qubits=target_qubits,
            name=name,
        )

    def build_vanilla(self) -> None:
        permutation = [
            self.multiplier * label % self.modulus if label < self.modulus else label
            for label in range(2**self.n_work_qubits)
        ]

        visited: set[int] = set()
        for start in range(len(permutation)):
            if start in visited:
                continue
            cycle: list[int] = []
            current = start
            while current not in visited:
                visited.add(current)
                cycle.append(current)
                current = permutation[current]

            # In circuit order these transpositions implement
            # (cycle[0] cycle[1] ... cycle[-1]).
            for endpoint in cycle[1:]:
                self._transpose_basis_states(cycle[0], endpoint)

    def _transpose_basis_states(self, first: int, second: int) -> None:
        """Swap two labels while fixing every other computational basis state."""
        if first == second:
            return

        path = [first]
        current = first
        for qubit in range(self.n_work_qubits):
            if ((first ^ second) >> qubit) & 1:
                current ^= 1 << qubit
                path.append(current)

        for left, right in zip(path, path[1:], strict=False):
            self._swap_adjacent_labels(left, right)
        for left, right in zip(reversed(path[:-2]), reversed(path[1:-1]), strict=True):
            self._swap_adjacent_labels(left, right)

    def _swap_adjacent_labels(self, first: int, second: int) -> None:
        """Swap labels differing in one bit using a mixed-polarity MCX."""
        differing = first ^ second
        target = differing.bit_length() - 1
        controls = [qubit for qubit in range(self.n_work_qubits) if qubit != target]
        zero_controls = [qubit for qubit in controls if ((first >> qubit) & 1) == 0]

        if zero_controls:
            self.x(zero_controls)
        self.h(target)
        self.mcz([*controls, target])
        self.h(target)
        if zero_controls:
            self.x(zero_controls)
