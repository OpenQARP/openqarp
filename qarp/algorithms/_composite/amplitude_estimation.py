"""Canonical quantum amplitude estimation."""

from copy import deepcopy
from math import pi, sin
from typing import Optional, Self, cast

from ..._types import SamplingDictionary
from ...blocks import AnyBlock
from ...blocks._primitives import AmplitudeEstimationBlock
from ...blocks._primitives.amplitude_amplification_block import (
    validate_amplification_blocks,
)
from ...blocks._primitives.amplitude_estimation_block import validate_n_ancilla
from ...endianness import bits_to_label
from ...engines import Engine
from .._primitives import PrimitiveAlgorithm, Sampler
from .composite_algorithm import CompositeAlgorithm


class AmplitudeEstimation(CompositeAlgorithm):
    r"""Estimate the initial good-state probability with canonical QAE.

    The supplied oracle must implement exactly ``I - 2 Pi_good``. With
    ``M = 2**n_ancilla``, a sampled QPE label ``y`` maps to
    ``sin(pi*y/M)**2``. The conjugate phase branches are folded together at
    ``z = min(y, M-y)`` before selecting the most probable estimate.

    This is the canonical estimator of Brassard, Hoyer, Mosca, and Tapp,
    arXiv:quant-ph/0005055.

    Args:
        state_preparation: Unitary preparing the state whose amplitude is
            estimated.
        oracle: Phase-exact good-state oracle on the same register.
        n_ancilla: Positive number of estimation qubits.
        primitive: Sampling primitive. Defaults to a private ``Sampler``.
        engine: Execution engine.
    """

    def __init__(
        self,
        state_preparation: AnyBlock,
        oracle: AnyBlock,
        n_ancilla: int,
        *,
        primitive: Optional[PrimitiveAlgorithm] = None,
        engine: Optional[Engine] = None,
    ) -> None:
        self.n_state_qubits = validate_amplification_blocks(state_preparation, oracle)
        self.n_ancilla = validate_n_ancilla(n_ancilla)
        if primitive is None:
            primitive = Sampler()
        elif not isinstance(primitive, Sampler):
            raise TypeError("AmplitudeEstimation requires a sampling primitive")
        if primitive.initial_state is not None:
            raise ValueError("AmplitudeEstimation requires the sampler initial_state to be None")
        super().__init__(primitive=primitive, engine=engine)

        self.state_preparation = deepcopy(state_preparation)
        self.oracle = deepcopy(oracle)
        self.block: Optional[AnyBlock] = None
        self.distribution: Optional[SamplingDictionary] = None
        self.folded_distribution: Optional[dict[int, float]] = None
        self.phase_bin: Optional[int] = None
        self.phase: Optional[float] = None
        self.estimate: Optional[float] = None
        self.result_probability: Optional[float] = None

    def build(self) -> Self:
        """Build and compile the canonical QAE circuit."""
        self.block = AmplitudeEstimationBlock(
            self.state_preparation,
            self.oracle,
            self.n_ancilla,
        ).build()
        sampler = cast(Sampler, self.primitive)
        sampler.ket = self.block
        sampler.measured_qubits = list(range(self.n_ancilla))
        self.engine.build([self.primitive])
        self.distribution = None
        self.folded_distribution = None
        self.phase_bin = None
        self.phase = None
        self.estimate = None
        self.result_probability = None
        return self

    def run(self) -> float:
        """Execute QAE once and return the modal folded-bin estimate."""
        if self.block is None or not self.block.is_built:
            raise ValueError("Circuit not built. Call build() before run().")

        result = self.engine.run()[0]
        if not isinstance(result, dict):
            raise TypeError("AmplitudeEstimation requires a sampling primitive")
        self.distribution = cast(SamplingDictionary, result)

        modulus = 2**self.n_ancilla
        folded: dict[int, float] = {}
        for bits, probability in self.distribution.items():
            label = bits_to_label(bits)
            folded_label = min(label, modulus - label)
            folded[folded_label] = folded.get(folded_label, 0.0) + probability

        self.folded_distribution = folded
        self.phase_bin = min(folded, key=lambda label: (-folded[label], label))
        self.phase = self.phase_bin / modulus
        self.estimate = sin(pi * self.phase) ** 2
        self.result_probability = folded[self.phase_bin]
        return self.estimate
