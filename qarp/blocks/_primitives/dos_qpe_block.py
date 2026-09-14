from typing import List, Optional

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
        super().__init__(
            n_qubits=n_ancilla + 2 * n_state,
            target_qubits=target_qubits,
            name=name,
        )

    def build_vanilla(self):
        """Build the DOSQPE circuit as a qarpx CompositeBlock.

        Returns:
            qx.CompositeBlock: The assembled circuit.
        """
        # Build children — each is a real qx.Block (Python wrapper).
        eigen_built = self.eigenstate.build()
        unit_built = self.unitary.build()

        n_q = self.n_qubits  # n_ancilla + 2 * n_state
        ancilla_qubits = list(range(self.n_ancilla))
        state_qubits = list(range(self.n_ancilla, self.n_ancilla + self.n_state))
        purification_qubits = list(range(self.n_ancilla + self.n_state, n_q))

        # 1) Hadamard layer on ancilla (time/frequency) qubits
        self.add_wired_child(HnBlock(self.n_ancilla, target_qubits=ancilla_qubits, name="AncillaH"))

        # 2) Eigenstate (probe) prep on state register
        eigen_built.target_qubits = state_qubits
        self.add_child(eigen_built)

        # 3) CNOT entanglement: state[i] → purification[i].  Tracing out the
        #    purification register gives the desired mixed probe state.
        cnot_layer = SimpleBlock(2 * self.n_state, name="PurificationCNOT")
        for i in range(self.n_state):
            cnot_layer.cx(i, self.n_state + i)
        cnot_layer.target_qubits = state_qubits + purification_qubits
        self.add_wired_child(cnot_layer)

        # 4) Controlled-U^(2^i) ladder — phase kickback 2^i·φ on each ancilla qubit.
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

        # 5) Inverse QFT on ancilla register
        iqft = QFTBlock(self.n_ancilla).dagger().build()
        iqft.target_qubits = ancilla_qubits
        self.add_child(iqft)

        # 6) Optional ancilla measurements — ancilla q reads into cbit q.
        if self.measure_at_end:
            self.add_wired_child(
                ReadoutBlock(self.n_ancilla, target_qubits=ancilla_qubits, name="AncillaMeas")
            )
