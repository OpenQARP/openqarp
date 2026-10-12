"""Grover search with a caller-declared marked-state count."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Optional, Self, cast

from ..._types import SamplingDictionary
from ...blocks import AnyBlock
from ...blocks._primitives import GroverBlock
from ...blocks._primitives.grover_block import (
    optimal_grover_iterations,
    validate_grover_inputs,
)
from ...endianness import bits_to_label
from ...engines import Engine
from .._primitives import PrimitiveAlgorithm, Sampler
from .composite_algorithm import CompositeAlgorithm


class Grover(CompositeAlgorithm):
    """Search a uniform basis-state space with a known marked-state count.

    ``good_states`` is optional reporting metadata. It is never used to build
    or modify the oracle; when supplied, its returned probability mass is
    exposed as ``success_probability``.  It is **not** cross-checked against the
    oracle (an opaque unitary cannot be inspected without an exponential dense
    matrix), so declaring labels the oracle does not actually mark yields a
    misleading ``success_probability`` — the labels are the caller's promise.

    ``most_likely_states`` is only meaningful once amplification has
    concentrated the distribution.  When the optimal iteration count is zero
    (``n_marked >= N/2``, so the uniform state is already at or past the
    amplification optimum) the distribution stays (near-)uniform and
    ``most_likely_states`` is the entire register; read
    ``predicted_success_probability`` to see that no concentration occurred.

    The algorithm follows Grover, arXiv:quant-ph/9605043, with the known-count
    analysis of Boyer, Brassard, Hoyer, and Tapp, arXiv:quant-ph/9605034.

    Args:
        oracle: Phase oracle implementing exactly ``I - 2 Pi_good``.
        n_marked: Number of basis states marked by the oracle.
        good_states: Optional explicit marked integer labels for reporting only.
        primitive: Sampling primitive. Defaults to a private ``Sampler``.
        engine: Execution engine.
    """

    def __init__(
        self,
        oracle: AnyBlock,
        n_marked: int = 1,
        *,
        good_states: Optional[Sequence[int]] = None,
        primitive: Optional[PrimitiveAlgorithm] = None,
        engine: Optional[Engine] = None,
    ) -> None:
        self.n_qubits, search_size = validate_grover_inputs(oracle, n_marked)

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
                if label < 0 or label >= search_size:
                    raise ValueError(f"good-state label {label} is outside [0, {search_size})")
            if len(labels) != len(set(labels)):
                raise ValueError("good_states must contain unique labels")
            if len(labels) != n_marked:
                raise ValueError("good_states must contain exactly n_marked labels")
            validated_good_states = labels

        if primitive is None:
            primitive = Sampler()
        elif not isinstance(primitive, Sampler):
            raise TypeError("Grover requires a sampling primitive")
        if primitive.initial_state is not None:
            raise ValueError("Grover requires the sampler initial_state to be None")
        super().__init__(primitive=primitive, engine=engine)

        self.oracle = deepcopy(oracle)
        self.n_marked = n_marked
        self.good_states = validated_good_states
        self.n_iterations, self.predicted_success_probability = optimal_grover_iterations(
            self.n_qubits, self.n_marked
        )
        self.block: Optional[AnyBlock] = None
        self.distribution: Optional[SamplingDictionary] = None
        self.most_likely_states: Optional[tuple[int, ...]] = None
        self.success_probability: Optional[float] = None

    def build(self) -> Self:
        """Build and compile uniform preparation plus amplification."""
        self.block = GroverBlock(self.oracle, self.n_marked).build()
        sampler = cast(Sampler, self.primitive)
        sampler.ket = self.block
        sampler.measured_qubits = list(range(self.n_qubits))
        self.engine.build([self.primitive])
        self.distribution = None
        self.most_likely_states = None
        self.success_probability = None
        return self

    def run(self) -> SamplingDictionary:
        """Execute once and return the full LSB-first search distribution."""
        if self.block is None or not self.block.is_built:
            raise ValueError("Circuit not built. Call build() before run().")

        result = self.engine.run()[0]
        if not isinstance(result, Mapping):
            raise TypeError("Grover requires a sampling primitive")
        self.distribution = cast(SamplingDictionary, result)

        maximum = max(self.distribution.values())
        self.most_likely_states = tuple(
            sorted(
                bits_to_label(bits)
                for bits, probability in self.distribution.items()
                if abs(probability - maximum) <= 1e-12
            )
        )
        if self.good_states is None:
            self.success_probability = None
        else:
            labels = set(self.good_states)
            self.success_probability = sum(
                probability
                for bits, probability in self.distribution.items()
                if bits_to_label(bits) in labels
            )
        return self.distribution
