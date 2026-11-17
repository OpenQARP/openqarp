from typing import Optional

import qarpx as qx

from .._block import SimpleBlock


class MixedOperatorBlock(SimpleBlock):
    """QAOA mixer operator: ``rx(β) = exp(-i (β/2) X)`` on every qubit.

    The symbol ``β`` is in radians.
    """

    def __init__(
        self,
        n_qubits: int,
        symbol_idx: int = 0,
        target_qubits=None,
        name: Optional[str] = None,
    ):
        """Args:
        n_qubits: number of qubits in the layer.
        symbol_idx: index for the symbolic parameter ``beta_<idx>`` (ASCII so the
            circuit exports as valid OpenQASM 3).
        target_qubits, name: see ``Block``.
        """
        self.symbol_idx = symbol_idx
        # A user-supplied name is honoured; the default carries the layer index.
        if name is None:
            name = f"Mixed Op. (p={self.symbol_idx})"
        super().__init__(n_qubits, target_qubits, name=name)

    def build_vanilla(self) -> None:
        angle = qx.Param.symbol(f"beta_{self.symbol_idx}")
        # Bulk emit the layer in one nanobind crossing.
        self.rx([(i, angle) for i in range(self.n_qubits)])
