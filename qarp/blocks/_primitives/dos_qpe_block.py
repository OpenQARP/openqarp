from copy import deepcopy
from typing import List, Optional

from ..._structure import Repeat
from .._block import AnyBlock, CompositeBlockBase, ControlledBlock, SimpleBlock
from .._primitives import QFTBlock
from .hn_block import HnBlock
from .readout_block import ReadoutBlock


class DOSQPEBlock(CompositeBlockBase):
    """Density Of States Quantum Phase Estimation (DOSQPE) circuit block.

    Composes:
        ancilla Hadamards · eigenstate prep · CNOT purification entanglement
        · controlled-U^(2^i) ladder (on state register) · inverse QFT
        · (optional) ancilla measurements.

    Total qubits: ``n_ancilla + 2 * n_state``.
    The registers are laid out as:

        [0 .. n_ancilla-1]                  — ancilla (time/frequency)
        [n_ancilla .. n_ancilla+n_state-1]  — state
        [n_ancilla+n_state .. n_q-1]        — purification (traced out)

    References:
        arXiv:2510.14744
    """

    def __init__(
        self,
        eigenstate: AnyBlock,
        unitary: AnyBlock,
        n_ancilla: int,
        n_state: int,
        measure: bool = False,
        target_qubits: Optional[List[int]] = None,
        name: str = "DOSQPE",
    ):
        """
        Density Of States Quantum Phase Estimation (DOSQPE) circuit class (arXiv:2510.14744).

        Args:
            eigenstate: Block that prepares the eigenstate (probe state).
            unitary: Block representing the unitary operator whose DOS is to be estimated.
            n_ancilla: Number of ancilla qubits for phase estimation.
            n_state: Number of qubits in the state register.
            measure: Whether to measure the ancilla qubits at the end.
            target_qubits: Optional list of target qubits for controlled operations.
            name: Name of the block.
        """
        self.eigenstate = eigenstate
        self.unitary = unitary
        self.n_ancilla = n_ancilla
        self.n_state = n_state
        self.measure_at_end = measure
        self._parts: Optional[list] = None
        super().__init__(
            n_qubits=n_ancilla + 2 * n_state,
            target_qubits=target_qubits,
            name=name,
        )

    def structure(self) -> list:
        """Hadamards on the ancillas, the probe preparation, the CNOT layer
        that entangles it with the purification register, for ancilla ``i``
        the controlled unitary repeated ``2**i`` times, the inverse QFT, and
        the ancilla readout (§13).  Builds only its parts, once."""
        if self._parts is not None:
            return self._parts
        # The parts are placed in this block's frame; the caller's block is
        # never moved, so it can serve another block unchanged.
        eigen_built = deepcopy(self.eigenstate.build())
        unit_built = self.unitary.build()
        n_q = self.n_qubits  # n_ancilla + 2 * n_state
        ancilla_qubits = list(range(self.n_ancilla))
        state_qubits = list(range(self.n_ancilla, self.n_ancilla + self.n_state))
        purification_qubits = list(range(self.n_ancilla + self.n_state, n_q))

        parts: list = [
            HnBlock(self.n_ancilla, target_qubits=ancilla_qubits, name="AncillaH").build()
        ]
        eigen_built.target_qubits = state_qubits
        parts.append(eigen_built)
        # Tracing out the purification register gives the desired mixed probe.
        cnot_layer = SimpleBlock(2 * self.n_state, name="PurificationCNOT")
        for i in range(self.n_state):
            cnot_layer.cx(i, self.n_state + i)
        cnot_layer.target_qubits = state_qubits + purification_qubits
        parts.append(cnot_layer.build())
        # Phase kickback 2^i·φ on ancilla i.
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
