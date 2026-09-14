"""``Sampler`` primitive — consumes a :class:`qarp.blocks.AnyBlock` and returns
a probability distribution dict whose keys are tuples of ints (LSB = lowest-
index measured qubit) and values are probabilities normalised by n_shots.
"""

from typing import Optional, Self, Union

import numpy as np

from ..._types import SamplingDictionary, Shots, outcome_arrays
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
        self.result: Optional[SamplingDictionary] = None

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

    def run(self, results: list) -> SamplingDictionary:
        """Convert the first result (``qx.SamplingResult``, or the duck-typed
        ``ExactResult`` under ``n_shots=qarp.EXACT``) to a
        {bitstring-tuple: probability} dict."""
        sr = results[0]
        n_shots = sr.n_shots
        measured = self.measured_qubits or list(range(sr.n_qubits))

        # The vectorized path packs outcomes and keys into int64, which caps
        # the register at 63 qubits; wider registers (tensor-network backends)
        # take the exact arbitrary-precision loop instead.
        if sr.n_qubits <= 63:
            outcomes, weights = outcome_arrays(sr)
            weights = weights / n_shots
            # Project each outcome onto the measured qubits, repacked LSB-first
            # so equal projections share one integer key; marginalising over
            # non-measured qubits then reduces to accumulating weights per key.
            qubits = np.asarray(measured, dtype=np.int64)
            packed = ((outcomes[:, None] >> qubits) & 1) @ (np.int64(1) << np.arange(len(qubits)))
            unique_keys, inverse = np.unique(packed, return_inverse=True)
            probabilities = np.bincount(inverse, weights=weights)

            unique_bits = (unique_keys[:, None] >> np.arange(len(qubits))) & 1
            distribution: SamplingDictionary = dict(
                zip(map(tuple, unique_bits.tolist()), probabilities.tolist(), strict=True)
            )
        else:
            distribution = {}
            for outcome, count in sr.counts.items():
                bits = tuple((outcome >> q) & 1 for q in measured)
                distribution[bits] = distribution.get(bits, 0.0) + count / n_shots

        self.result = distribution
        return distribution
