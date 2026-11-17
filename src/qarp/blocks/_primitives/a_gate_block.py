from typing import List, Optional, Union

import numpy as np
from sympy import Symbol

from .._block import SimpleBlock, as_param

_to_param = as_param  # the single §13 coercion (multi-symbol / non-linear → ValueError)


class AGateBlock(SimpleBlock):
    def __init__(
        self,
        theta: Union[Symbol, float],
        phi: Union[Symbol, float],
        target_qubits: Optional[List[int]] = None,
        name: str = "A-gate",
    ):
        """Construct the A-gate as described in Gard et al., npj Quantum Inf 6, 10 (2020).

        The gate is designed to efficiently prepare quantum states while preserving symmetries
        relevant to quantum chemistry and physics simulations.

        Note:
            The parameters theta and phi may be either real (float) or symbolic (sympy Symbol).
            Angles are in radians.

        Args:
            theta: The parameter corresponding to the Ry rotation.
            phi: The parameter corresponding to the Rz rotation.
            target_qubits: The target qubits will act on when added to a Block object.
        """
        self.theta = theta
        self.phi = phi
        super().__init__(
            n_qubits=2,
            target_qubits=target_qubits,
            name=name,
        )

    def build_vanilla(self) -> None:
        self.cx(1, 0)
        self.rz(1, _to_param(-self.phi - np.pi))
        self.ry(1, _to_param(-self.theta - 0.5 * np.pi))
        self.cx(0, 1)
        self.ry(1, _to_param(self.theta + 0.5 * np.pi))
        self.rz(1, _to_param(self.phi + np.pi))
        self.cx(1, 0)
