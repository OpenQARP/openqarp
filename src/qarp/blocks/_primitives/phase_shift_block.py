from typing import List, Optional

from .._block import SimpleBlock


class PhaseShiftBlock(SimpleBlock):
    def __init__(
        self,
        phase: float,
        target_qubits: Optional[List[int]] = None,
        name: str = "PS",
    ):
        """Phase-shift Z-rotation: ``exp(i θ/2) Rz(θ) = diag(1, e^{iθ})``.

        Args:
            phase: Phase shift angle in radians.
            target_qubits: Specific qubits to apply the block to.
            name: Optional custom name for the block.
        """
        self.phase = phase
        super().__init__(
            n_qubits=1,
            target_qubits=target_qubits,
            name=name,
        )

    def build_vanilla(self) -> None:
        # P(θ) = diag(1, e^{iθ}) — exactly the phase-shift unitary.
        self.p(0, self.phase)
