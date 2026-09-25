"""Composite amplitude-amplification algorithm."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Optional, Self, cast

from ..._types import SamplingDictionary
from ...blocks import AmplitudeAmplificationBlock, AnyBlock, CompositeBlock
from ...blocks._primitives.amplitude_amplification_block import (
    validate_amplification_blocks,
)
from ...endianness import bits_to_label
from ...engines import Engine
from .._primitives import PrimitiveAlgorithm, Sampler
from .composite_algorithm import CompositeAlgorithm


class AmplitudeAmplification(CompositeAlgorithm):
    r"""Prepare a state and apply a fixed number of amplification iterates.

    The built circuit is ``A`` followed by ``n_iterations`` applications of
    :class:`~qarp.blocks.AmplitudeAmplificationBlock`.  The supplied ``oracle``
    must implement exactly ``I - 2 Pi_good``; see the block documentation for
    the phase convention.

    ``good_states`` is optional classical reporting metadata.  It contains
    full-register integer labels and is used only to sum the returned sampling
    probabilities.  It neither defines nor modifies the quantum oracle.

    The iterate and the probability law
    ``sin((2 * n_iterations + 1) * theta)**2`` follow Brassard, Hoyer, Mosca,
    and Tapp, *Quantum Amplitude Amplification and Estimation*,
    arXiv:quant-ph/0005055, Eqs. (1), (5), and (8).

    Args:
        state_preparation: Unitary ``A`` preparing the initial state.
        oracle: Good-state phase oracle on the same positive-width register.
        n_iterations: Explicit non-negative number of amplification iterates.
        good_states: Optional unique full-register integer labels whose
            returned probabilities are summed into ``success_probability``.
        primitive: Sampling primitive.  Defaults to a private ``Sampler``.  A
            sampler carrying ``initial_state`` is rejected: the amplification
            law assumes the circuit starts in ``|0...0>``, so ``A`` must act on
            that state and not on a seeded one.
        engine: Execution engine.  Defaults to ``QarpEngine`` through the
            composite-algorithm base class.
    """

    def __init__(
        self,
        state_preparation: AnyBlock,
        oracle: AnyBlock,
        n_iterations: int,
        *,
        good_states: Optional[Sequence[int]] = None,
        primitive: Optional[PrimitiveAlgorithm] = None,
        engine: Optional[Engine] = None,
    ) -> None:
        n_qubits = validate_amplification_blocks(state_preparation, oracle)
        if isinstance(n_iterations, bool) or not isinstance(n_iterations, int):
            raise TypeError("n_iterations must be an integer")
        if n_iterations < 0:
            raise ValueError("n_iterations must be non-negative")

        validated_good_states: Optional[tuple[int, ...]]
        if good_states is None:
            validated_good_states = None
        else:
            if isinstance(good_states, (str, bytes)) or not isinstance(good_states, Sequence):
                raise TypeError("good_states must be a sequence of integer labels")
            labels = tuple(good_states)
            for label in labels:
                if isinstance(label, bool) or not isinstance(label, int):
                    raise TypeError("good_states must contain only integer labels")
                if label < 0 or label >= 2**n_qubits:
                    raise ValueError(f"good-state label {label} is outside [0, {2**n_qubits})")
            if len(labels) != len(set(labels)):
                raise ValueError("good_states must contain unique labels")
            validated_good_states = labels

        if primitive is None:
            primitive = Sampler()
        elif not isinstance(primitive, Sampler):
            raise TypeError("AmplitudeAmplification requires a sampling primitive")
        if primitive.initial_state is not None:
            raise ValueError("AmplitudeAmplification requires the sampler initial_state to be None")
        super().__init__(primitive=primitive, engine=engine)

        # Keep private snapshots so neither circuit construction nor a later
        # caller mutation can change this algorithm's inputs.
        self.state_preparation = deepcopy(state_preparation)
        self.oracle = deepcopy(oracle)
        self.n_iterations = n_iterations
        self.n_qubits = n_qubits
        self.good_states = validated_good_states

        self.block: Optional[AnyBlock] = None
        self.distribution: Optional[SamplingDictionary] = None
        self.success_probability: Optional[float] = None

    def build(self) -> Self:
        """Build and compile ``A`` followed by the requested iterates."""
        state_preparation = deepcopy(self.state_preparation)
        state_preparation.target_qubits = list(range(self.n_qubits))

        iterate = AmplitudeAmplificationBlock(
            self.state_preparation, self.oracle, power=self.n_iterations
        )
        self.block = CompositeBlock(
            [state_preparation, iterate],
            n_qubits=self.n_qubits,
            name="AmplitudeAmplificationCircuit",
        ).build()

        self.primitive.ket = self.block
        self.engine.build([self.primitive])
        self.distribution = None
        self.success_probability = None
        return self

    def run(self) -> SamplingDictionary:
        """Execute once and return the LSB-first sampling distribution."""
        if self.block is None or not self.block.is_built:
            raise ValueError("Circuit not built. Call build() before run().")

        result = self.engine.run()[0]
        if not isinstance(result, Mapping):
            raise TypeError("AmplitudeAmplification requires a sampling primitive")
        self.distribution = cast(SamplingDictionary, result)

        if self.good_states is None:
            self.success_probability = None
        else:
            good_labels = set(self.good_states)
            self.success_probability = sum(
                probability
                for bits, probability in self.distribution.items()
                if bits_to_label(bits) in good_labels
            )
        return self.distribution
