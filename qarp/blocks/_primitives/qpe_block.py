from typing import List, Optional

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

        super().__init__(
            n_qubits=n_ancilla + n_state,
            target_qubits=target_qubits,
            name=name,
        )

    def build_vanilla(self) -> None:
        # Build children: each `built` is a real qx.Block (Python wrapper).
        eigen_built = self.eigenstate.build()
        unit_built = self.unitary.build()

        n_q = self.n_qubits  # n_ancilla + n_state
        ancilla_qubits = list(range(self.n_ancilla))
        state_qubits = list(range(self.n_ancilla, n_q))

        # 1) Hadamard layer on ancillas
        self.add_wired_child(HnBlock(self.n_ancilla, target_qubits=ancilla_qubits, name="AncillaH"))

        # 2) Eigenstate prep on state register — remap onto state qubits.
        eigen_built.target_qubits = state_qubits
        self.add_child(eigen_built)

        # 3) Controlled-U^(2^i) ladder.  Ancilla i gets 2^i applications of
        #    the controlled unitary → phase kickback 2^i·φ, so the ancilla
        #    register encodes the phase as an integer (qubit 0 = LSB).
        #    After the inverse QFT the raw outcome equals 2^n_ancilla · φ,
        #    which qpe.py reads via ``int("".join(bitstring), 2)`` (MSB-first).
        for i, ancilla_q in enumerate(ancilla_qubits):
            for _ in range(2**i):
                ctrl_u = ControlledBlock(
                    unit_built,
                    num_controls=1,
                    ctrl_state=[True],
                    name=f"C-U@a{ancilla_q}",
                )
                ctrl_u.build()
                ctrl_u.target_qubits = [ancilla_q] + state_qubits
                self.add_child(ctrl_u)

        # 4) Inverse QFT on ancilla register
        iqft = QFTBlock(self.n_ancilla).dagger().build()
        iqft.target_qubits = ancilla_qubits
        self.add_child(iqft)

        # 5) Optional ancilla measurements — ancilla q reads into cbit q.
        if self.measure_at_end:
            self.add_wired_child(
                ReadoutBlock(self.n_ancilla, target_qubits=ancilla_qubits, name="AncillaMeas")
            )
