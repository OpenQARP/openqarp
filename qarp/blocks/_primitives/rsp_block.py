from typing import List, Optional, Union

from sympy import Symbol

from qarp.blocks._block import SimpleBlock

from .._block import as_param

_to_param = as_param  # the single §13 coercion (multi-symbol / non-linear → ValueError)


class RSPBlock(SimpleBlock):
    def __init__(
        self,
        theta: Union[Symbol, float],
        target_qubits: Optional[List[int]] = None,
        name: str = "RSP",
    ):
        """Construct the RSP gate as described in Ibe et al., Phys. Rev. Research 4, 013173. Ensures time reversal symmetry.

        Note:
            The parameter theta may be either real (float) or symbolic (sympy Symbol).
            The parameter is in radians.

        Args:
            theta: The parameter corresponding to the rotation in the original publication.
            target_qubits: The target qubits will act on when added to a Block object.
        """
        self.theta = theta
        super().__init__(
            n_qubits=2,
            target_qubits=target_qubits,
            name=name,
        )

    def build_vanilla(self) -> None:
        alpha = self.theta
        self.cx(0, 1)
        self.ry(0, _to_param(alpha))
        self.cx(1, 0)
        self.ry(0, _to_param(-alpha))
        self.cx(0, 1)
