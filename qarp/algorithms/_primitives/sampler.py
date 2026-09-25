"""``Sampler`` primitive — consumes a :class:`qarp.blocks.AnyBlock` and returns
a :class:`~qarp.SamplingDistribution`: probabilities normalised by n_shots,
keyed by LSB-first bit tuples over the measured qubits.
"""

from typing import Optional, Self, Union

from ..._sampling_distribution import SamplingDistribution, distribution_from_result
from ..._types import Shots
from ...blocks import AnyBlock
from .primitive_algorithm import PrimitiveAlgorithm
from .target import Target


class Sampler(PrimitiveAlgorithm):
    """qarpx-backed sampling primitive.

    Args:
        ket: Block to execute.
        n_shots: Number of measurement shots (default: use engine default).
            ``qarp.EXACT`` returns the exact Born distribution ``|ψ|²`` — note:
            probabilities, not amplitudes (phases are discarded; raw
            amplitudes are simulator internals, ``sim.statevector``).
        measured_qubits: Qubit indices whose outcomes appear in the output
            tuple.  Defaults to ``range(ket.n_qubits)``.  Non-measured
            qubits are marginalised out by summing probabilities over their
            bit values.
        initial_state: Optional LSB-indexed amplitudes seeding the register
            instead of ``|0…0⟩`` (length ``2**n_qubits``, unit norm within
            ``1e-10`` — ``ValueError`` at run otherwise).  QarpEngine only;
            other engines raise ``CapabilityError`` at build.  Mutable
            between runs for step → snapshot → re-seed loops.
    """

    supported_targets = frozenset({Target.SAMPLING})
    accepts_initial_state = True

    def __init__(
        self,
        ket: Optional[AnyBlock] = None,
        n_shots: Optional[Union[int, Shots]] = None,
        measured_qubits: Optional[list[int]] = None,
        initial_state=None,
    ):
        super().__init__(ket=ket, n_shots=n_shots, target=Target.SAMPLING)
        self.measured_qubits = measured_qubits
        self.initial_state = initial_state
        self.result: Optional[SamplingDistribution] = None

    def build(self) -> Self:
        if self.ket is None:
            raise ValueError("Sampler requires a ket Block.")
        self.ket.build()
        # The ket is itself a Block, so it becomes this sampler's sole sub-block.
        self.sub_blocks = [self.ket]
        self.n_qubits = self.ket.n_qubits
        if self.measured_qubits is None:
            self.measured_qubits = list(range(self.n_qubits))
        return self

    def run(self, results: list) -> SamplingDistribution:
        """Marginal of the first result (``qx.SamplingResult``, or the
        duck-typed ``ExactResult`` under ``n_shots=qarp.EXACT``) on
        ``measured_qubits``."""
        sr = results[0]
        self.result = distribution_from_result(sr, self.measured_qubits or range(sr.n_qubits))
        return self.result
