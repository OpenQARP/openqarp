from copy import deepcopy
from typing import List, Optional

from ..._structure import Repeat
from .._block import AnyBlock, CompositeBlockBase, ControlledBlock
from .._primitives import QFTBlock
from .hn_block import HnBlock
from .readout_block import ReadoutBlock


class QPEBlock(CompositeBlockBase):
    """Quantum Phase Estimation (QPE) circuit block — Pattern B composite.

    Composes:  ancilla Hadamards · state prep · controlled-U^(2^i) ladder
    · inverse QFT · (optional) ancilla measurements.
    """

    def __init__(
        self,
        eigenstate: AnyBlock,
        unitary: AnyBlock,
        n_ancilla: int,
        n_state: int,
        measure: bool = True,
        target_qubits: Optional[List[int]] = None,
        name: str = "QPE",
    ):
        """Quantum Phase Estimation: controlled-U powers on an ancilla
        register followed by an inverse QFT.

        Args:
            eigenstate: Block preparing the probe state (on the state register).
            unitary: Block of the unitary operator.
            n_ancilla: Number of ancilla qubits (phase precision bits).
            n_state: Number of qubits in the eigenstate circuit.
            measure: If True, add Measure commands on the ancilla register.
            target_qubits, name: see ``Block``.
        """
        self.eigenstate = eigenstate
        self.unitary = unitary
        self.n_ancilla = n_ancilla
        self.n_state = n_state
        self.measure_at_end = measure
        self._parts: Optional[list] = None

        super().__init__(
            n_qubits=n_ancilla + n_state,
            target_qubits=target_qubits,
            name=name,
        )

    def structure(self) -> list:
        """Hadamards on the ancillas, the eigenstate preparation, for ancilla
        ``i`` the controlled unitary repeated ``2**i`` times, the inverse QFT,
        and the ancilla readout (§13).  Builds only its parts, once."""
        if self._parts is not None:
            return self._parts
        # The parts are placed in this block's frame; the caller's block is
        # never moved, so it can serve another block unchanged.
        eigen_built = deepcopy(self.eigenstate.build())
        unit_built = self.unitary.build()
        ancilla_qubits = list(range(self.n_ancilla))
        state_qubits = list(range(self.n_ancilla, self.n_qubits))

        parts: list = [
            HnBlock(self.n_ancilla, target_qubits=ancilla_qubits, name="AncillaH").build()
        ]
        eigen_built.target_qubits = state_qubits
        parts.append(eigen_built)
        # Ancilla i gets 2^i applications of the controlled unitary → phase
        # kickback 2^i·φ, so the ancilla register encodes the phase as an
        # integer (qubit 0 = LSB).
        for i, ancilla_q in enumerate(ancilla_qubits):
            ctrl_u = ControlledBlock(
                unit_built, num_controls=1, ctrl_state=[True], name=f"C-U@a{ancilla_q}"
            )
            ctrl_u.build()
            ctrl_u.target_qubits = [ancilla_q] + state_qubits
            parts.append(Repeat(ctrl_u, 2**i))
        iqft = QFTBlock(self.n_ancilla).dagger().build()
        iqft.target_qubits = ancilla_qubits
        parts.append(iqft)
        if self.measure_at_end:
            parts.append(
                ReadoutBlock(
                    self.n_ancilla, target_qubits=ancilla_qubits, name="AncillaMeas"
                ).build()
            )
        self._parts = parts
        return parts

    def build_vanilla(self) -> None:
        for part in self.structure():
            if isinstance(part, Repeat):
                for _ in range(part.count):
                    self.add_child(part.block)
            else:
                self.add_child(part)
