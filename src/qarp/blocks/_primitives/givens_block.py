from typing import Optional, Union

from sympy import Symbol

from .._block import SimpleBlock, as_param

_to_param = as_param  # the single §13 coercion (multi-symbol / non-linear → ValueError)


class GivensBlock(SimpleBlock):
    def __init__(
        self,
        theta: Union[Symbol, float],
        target_qubits: Optional[list[int]] = None,
        name: str = "Givens",
    ):
        """Construct the Givens Rotation gate.

        Follows Nam, npj Quantum Information (2020) 6:33 with the matrix:

        ::

            [1 0               0                0]
            [0 cos(theta / 2) -sin(theta / 2)   0]
            [0 sin(theta / 2)  cos(theta / 2)   0]
            [0 0               0                1]

        The parameter is in **radians**, matching qarpx's convention for
        single-qubit rotations (``Rx/Ry/Rz``).  ``theta`` may be float, sympy
        Symbol, or a linear sympy expression in a single symbol.

        Args:
            theta: Rotation angle in radians.
            target_qubits: The target qubits will act on when added to a Block object.
        """
        self.theta = theta
        super().__init__(
            n_qubits=2,
            target_qubits=target_qubits,
            name=name,
        )

    def build_vanilla(self) -> None:
        # Decompose the Givens rotation via the Sdg / RXX / S sandwich
        # (Nam et al.).  ``self.theta`` is in radians; the inner RXX angle is
        # ``theta/2``, implemented as ``CX(0,1) · Rx(-theta/2, q0) · CX(0,1)``
        # (since ``RXX(α) = CX·Rx(-α, q0)·CX``).
        alpha = self.theta / 2
        self.sdg(0)
        self.cx(0, 1)
        self.rx(0, _to_param(-alpha))
        self.cx(0, 1)
        self.s(0)
        self.sdg(1)
        self.cx(0, 1)
        self.rx(0, _to_param(alpha))
        self.cx(0, 1)
        self.s(1)
