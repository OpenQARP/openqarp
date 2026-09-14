"""Measurement-free canonical quantum amplitude estimation."""

from copy import deepcopy
from typing import Optional

from .._block import AnyBlock, CompositeBlockBase
from .amplitude_amplification_block import (
    AmplitudeAmplificationBlock,
    validate_amplification_blocks,
)
from .qpe_block import QPEBlock


def validate_n_ancilla(n_ancilla: int) -> int:
    """Validate and return a positive canonical-QAE precision width."""
    if isinstance(n_ancilla, bool) or not isinstance(n_ancilla, int):
        raise TypeError("n_ancilla must be an integer")
    if n_ancilla < 1:
        raise ValueError("n_ancilla must be positive")
    return n_ancilla


class AmplitudeEstimationBlock(CompositeBlockBase):
    r"""Canonical QAE circuit for a phase-exact amplification iterate.

    For ``A|0> = sqrt(1-a)|psi_bad> + sqrt(a)|psi_good>`` and an oracle
    implementing exactly ``O_good = I - 2 Pi_good``, the embedded
    :class:`AmplitudeAmplificationBlock` has relevant eigenphases
    ``+/- 2 theta``, where ``sin(theta)**2 = a``.  This block applies QPE to
    that iterate without adding measurements.

    Qubits ``0 .. n_ancilla-1`` form the estimation register and the state
    register follows it. Caller-owned blocks are deep-copied and never built,
    retargeted, or otherwise mutated.

    The construction follows Brassard, Hoyer, Mosca, and Tapp, *Quantum
    Amplitude Amplification and Estimation*, arXiv:quant-ph/0005055.

    Args:
        state_preparation: Unitary ``A`` preparing the initial state.
        oracle: Good-state phase oracle implementing ``I - 2 Pi_good``.
        n_ancilla: Positive number of estimation qubits.
        target_qubits: Optional placement of the complete QAE circuit.
        name: Block name.
    """

    def __init__(
        self,
        state_preparation: AnyBlock,
        oracle: AnyBlock,
        n_ancilla: int,
        target_qubits: Optional[list[int]] = None,
        name: str = "AmplitudeEstimation",
    ) -> None:
        self.n_state_qubits = validate_amplification_blocks(state_preparation, oracle)
        self.n_ancilla = validate_n_ancilla(n_ancilla)
        self.state_preparation = deepcopy(state_preparation)
        self.oracle = deepcopy(oracle)
        super().__init__(
            n_qubits=self.n_ancilla + self.n_state_qubits,
            target_qubits=target_qubits,
            name=name,
        )

    def build_vanilla(self) -> None:
        # QPEBlock builds and retargets its eigenstate in place; the iterate copies its own.
        state_preparation = deepcopy(self.state_preparation)
        iterate = AmplitudeAmplificationBlock(self.state_preparation, self.oracle)
        qpe = QPEBlock(
            eigenstate=state_preparation,
            unitary=iterate,
            n_ancilla=self.n_ancilla,
            n_state=self.n_state_qubits,
            measure=False,
            name="CanonicalQAE",
        )
        self.add_wired_child(qpe)
