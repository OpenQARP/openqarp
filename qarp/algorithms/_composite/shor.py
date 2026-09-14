"""Exact small-integer reference implementation of Shor factoring."""

from __future__ import annotations

import copy
import random
from math import gcd, isqrt, lcm
from typing import Optional, Self, cast

from qarp._types import SamplingDictionary
from qarp.blocks import AnyBlock, ModularMultiplicationBlock, OrderFindingBlock
from qarp.endianness import bits_to_label
from qarp.engines import Engine, Runnable

from .._primitives import Sampler
from .composite_algorithm import CompositeAlgorithm


def _is_prime(number: int) -> bool:
    if number < 2:
        return False
    if number % 2 == 0:
        return number == 2
    divisor = 3
    while divisor <= isqrt(number):
        if number % divisor == 0:
            return False
        divisor += 2
    return True


def _integer_nth_root(number: int, exponent: int) -> int:
    """Largest integer ``root`` satisfying ``root**exponent <= number``."""
    lower, upper = 1, 1 << ((number.bit_length() + exponent - 1) // exponent)
    while lower <= upper:
        middle = (lower + upper) // 2
        power = middle**exponent
        if power <= number:
            lower = middle + 1
        else:
            upper = middle - 1
    return upper


def _perfect_power_factor(number: int) -> Optional[int]:
    """Return a non-trivial exact-power base, or ``None``."""
    for exponent in range(2, number.bit_length() + 1):
        root = _integer_nth_root(number, exponent)
        if root > 1 and root**exponent == number:
            return root
    return None


def _continued_fraction_denominators(
    numerator: int,
    denominator: int,
    *,
    upper_bound: int,
) -> list[int]:
    """Denominators of convergents to ``numerator / denominator`` below a bound."""
    previous_numerator, convergent_numerator = 0, 1
    previous_denominator, convergent_denominator = 1, 0
    denominators: list[int] = []

    while denominator:
        coefficient, remainder = divmod(numerator, denominator)
        next_numerator = coefficient * convergent_numerator + previous_numerator
        next_denominator = coefficient * convergent_denominator + previous_denominator
        if next_denominator >= upper_bound:
            break
        if next_denominator > 1:
            denominators.append(next_denominator)
        previous_numerator, convergent_numerator = convergent_numerator, next_numerator
        previous_denominator, convergent_denominator = (
            convergent_denominator,
            next_denominator,
        )
        numerator, denominator = denominator, remainder
    return denominators


def _order_from_multiple(base: int, modulus: int, candidate: int) -> Optional[int]:
    """Exact multiplicative order of ``base`` from a validated multiple of it.

    ``pow(base, candidate, modulus) == 1`` holds iff the order divides
    ``candidate``; dividing out every prime that keeps that identity true
    leaves the order itself.  Returns ``None`` when ``candidate`` is not a
    multiple of the order.
    """
    if not 1 < candidate < modulus or pow(base, candidate, modulus) != 1:
        return None

    order = candidate
    prime = 2
    while prime * prime <= order:
        while order % prime == 0 and pow(base, order // prime, modulus) == 1:
            order //= prime
        prime += 1
    return order


def _recover_order(
    base: int,
    modulus: int,
    distribution: SamplingDictionary,
    n_counting_qubits: int,
) -> Optional[int]:
    """Recover a validated multiplicative order from an LSB sampling distribution."""
    phase_denominator = 2**n_counting_qubits
    accumulated: set[int] = set()
    outcomes = sorted(
        distribution.items(),
        key=lambda item: (-item[1], bits_to_label(item[0])),
    )

    for bits, probability in outcomes:
        if probability <= 0.0:
            continue
        measured = bits_to_label(bits)
        if measured == 0:
            continue
        denominators = _continued_fraction_denominators(
            measured,
            phase_denominator,
            upper_bound=modulus,
        )
        new_candidates = set(denominators)
        for denominator in denominators:
            for previous in accumulated:
                combined = lcm(denominator, previous)
                if combined < modulus:
                    new_candidates.add(combined)
        accumulated.update(new_candidates)

        for candidate in sorted(new_candidates):
            order = _order_from_multiple(base, modulus, candidate)
            if order is not None:
                return order
    return None


def _factor_pair_from_order(base: int, modulus: int, order: int) -> Optional[tuple[int, int]]:
    """Convert a useful even order into a sorted non-trivial factor pair."""
    if order % 2:
        return None
    halfway = pow(base, order // 2, modulus)
    if halfway in (1, modulus - 1):
        return None

    for candidate in (gcd(halfway - 1, modulus), gcd(halfway + 1, modulus)):
        if 1 < candidate < modulus and modulus % candidate == 0:
            first, second = sorted((candidate, modulus // candidate))
            return first, second
    return None


def _sorted_pair(factor: int, number: int) -> tuple[int, int]:
    first, second = sorted((factor, number // factor))
    return first, second


class Shor(CompositeAlgorithm):
    """Exact small-integer/reference implementation of Shor factoring.

    This implementation uses an exponentially synthesized basis permutation
    for modular arithmetic and supports at most
    ``ModularMultiplicationBlock.MAX_REFERENCE_WORK_QUBITS`` work qubits
    (``number <= 64`` for the shipped value of six). It establishes a correct
    reference workflow for small examples; it does not claim
    cryptographic-scale performance or an asymptotic quantum speedup.

    ``build()`` prepares the order-finding samplers and ``run()`` produces the
    factor pair, classical shortcuts included.  ``run()`` executes the batch
    once; a repeated call returns the stored outcome, ``None`` included, so a
    finite-shot retry is a new instance.  Bases are processed in attempt
    order, as in Shor's sequential algorithm: every coprime base gets an
    order-finding circuit, and the first non-coprime base ends the attempt
    list with ``gcd(base, number)`` as a classical fallback that ``run()``
    returns only if every quantum attempt was inconclusive.  Even and
    perfect-power inputs are factored classically without any circuit.

    Args:
        number: Composite integer greater than one to factor.
        base: Optional first modular-order-finding base. Must satisfy
            ``1 < base < number``. A non-coprime base is a valid classical gcd
            shortcut unless ``force_quantum`` is set.
        n_counting_qubits: Counting-register width. Defaults to twice the work
            width and must be at least that large.
        max_attempts: Maximum number of distinct bases to try.
        base_seed: Seed for the algorithm-local base-selection RNG.
        force_quantum: Guarantee the quantum path.  Disables the gcd shortcut
            (random bases are drawn coprime; a supplied non-coprime base is
            rejected) and rejects even or perfect-power inputs, which lie
            outside the preconditions of Shor's order-finding theorem: for
            ``N = 2p`` and odd ``N = p**k`` every even order gives
            ``a**(r/2) == -1 (mod N)``, so no base can succeed.
        primitive: Sampling primitive.  A private deep copy is used, and Shor
            owns its ``ket`` and ``measured_qubits`` (the counting register);
            the caller's sampler contributes shot settings only.
        engine: Execution engine. Defaults to :class:`QarpEngine`.
    """

    def __init__(
        self,
        number: int,
        *,
        base: Optional[int] = None,
        n_counting_qubits: Optional[int] = None,
        max_attempts: int = 8,
        base_seed: Optional[int] = None,
        force_quantum: bool = False,
        primitive: Optional[Sampler] = None,
        engine: Optional[Engine] = None,
    ) -> None:
        if isinstance(number, bool) or not isinstance(number, int):
            raise TypeError("number must be an integer")
        if number <= 1:
            raise ValueError("number must be greater than one")
        if base is not None:
            if isinstance(base, bool) or not isinstance(base, int):
                raise TypeError("base must be an integer")
            if not 1 < base < number:
                raise ValueError("base must satisfy 1 < base < number")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
            raise TypeError("max_attempts must be an integer")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if base_seed is not None and (
            isinstance(base_seed, bool) or not isinstance(base_seed, int)
        ):
            raise TypeError("base_seed must be an integer or None")
        if not isinstance(force_quantum, bool):
            raise TypeError("force_quantum must be a bool")

        # The limit is checked first so no classical work of any size happens on
        # an input the quantum path cannot serve.
        n_work_qubits = (number - 1).bit_length()
        limit = ModularMultiplicationBlock.MAX_REFERENCE_WORK_QUBITS
        if n_work_qubits > limit:
            raise ValueError(
                "Shor's exact reference arithmetic supports at most "
                f"{limit} work qubits (number <= {2**limit}); number={number} "
                f"requires {n_work_qubits}."
            )
        if n_counting_qubits is None:
            n_counting_qubits = 2 * n_work_qubits
        elif isinstance(n_counting_qubits, bool) or not isinstance(n_counting_qubits, int):
            raise TypeError("n_counting_qubits must be an integer")
        if n_counting_qubits < 2 * n_work_qubits:
            raise ValueError(
                "n_counting_qubits must be at least twice the work-register width "
                f"({2 * n_work_qubits})"
            )

        if primitive is None:
            primitive = Sampler()
        if not isinstance(primitive, Sampler):
            raise TypeError("Shor requires a Sampler primitive")
        super().__init__(primitive=primitive, engine=engine)

        self.number = number
        self.base = base
        self.n_counting_qubits = n_counting_qubits
        self.n_work_qubits = n_work_qubits
        self.max_attempts = max_attempts
        self.base_seed = base_seed
        self.force_quantum = force_quantum

        self.result: Optional[tuple[int, int]] = None
        self.blocks: list[AnyBlock] = []
        self.attempted_bases: list[int] = []
        self.distributions: dict[int, SamplingDictionary] = {}
        self.orders: dict[int, Optional[int]] = {}
        self.classical_fallback: Optional[tuple[int, int]] = None
        self._precondition_factor: Optional[int] = None
        self._quantum_bases: list[int] = []
        self._samplers: list[Sampler] = []
        self._built = False
        self._executed = False

    def _candidate_bases(self) -> list[int]:
        selected: list[int] = []
        if self.base is not None:
            selected.append(self.base)
        pool = [
            candidate
            for candidate in range(2, self.number)
            if candidate != self.base
            and (not self.force_quantum or gcd(candidate, self.number) == 1)
        ]
        rng = random.Random(self.base_seed)
        draws = min(self.max_attempts - len(selected), len(pool))
        selected.extend(rng.sample(pool, draws))
        return selected

    def build(self) -> Self:
        """Choose bases and compile the order-finding samplers; never factors."""
        if self._built:
            return self
        if _is_prime(self.number):
            raise ValueError("Shor requires a composite number; prime inputs cannot be factored")

        if self.number % 2 == 0:
            self._precondition_factor = 2
        else:
            self._precondition_factor = _perfect_power_factor(self.number)
        if self._precondition_factor is not None:
            if self.force_quantum:
                raise ValueError(
                    f"number={self.number} is even or a perfect power, outside the "
                    "preconditions of Shor's order-finding theorem. Use "
                    "force_quantum=False to factor it classically."
                )
            self._built = True
            return self

        for base in self._candidate_bases():
            self.attempted_bases.append(base)
            common_factor = gcd(base, self.number)
            if common_factor == 1:
                self._quantum_bases.append(base)
                continue
            if self.force_quantum:
                raise ValueError(
                    f"base={base} shares the factor {common_factor} with "
                    f"number={self.number}; force_quantum requires a coprime base"
                )
            self.classical_fallback = _sorted_pair(common_factor, self.number)
            break

        for base in self._quantum_bases:
            block = OrderFindingBlock(
                base,
                self.number,
                n_counting_qubits=self.n_counting_qubits,
                name=f"OrderFinding({base}, {self.number})",
            )
            sampler = copy.deepcopy(cast(Sampler, self.primitive))
            sampler.ket = block
            sampler.measured_qubits = list(range(self.n_counting_qubits))
            self.blocks.append(block)
            self._samplers.append(sampler)

        self.sub_algorithms = list(self._samplers)
        if self._samplers:
            self.engine.build(cast(list[Runnable], self._samplers))
        self._built = True
        return self

    # Shor takes no results argument: it owns its engine run, like MonteCarlo.
    def run(self) -> Optional[tuple[int, int]]:
        """Execute the batch once and return the first recovered factor pair."""
        if not self._built:
            raise ValueError("Circuit not built. Call build() before run().")
        if self._executed:
            return self.result
        self._executed = True
        if self._precondition_factor is not None:
            self.result = _sorted_pair(self._precondition_factor, self.number)
            return self.result

        if self._samplers:
            raw_distributions = cast(list[SamplingDictionary], self.engine.run())
            for base, distribution in zip(
                self._quantum_bases,
                raw_distributions,
                strict=True,
            ):
                self.distributions[base] = distribution
                order = _recover_order(
                    base,
                    self.number,
                    distribution,
                    self.n_counting_qubits,
                )
                self.orders[base] = order
                if order is None:
                    continue
                factors = _factor_pair_from_order(base, self.number, order)
                if factors is not None:
                    self.result = factors
                    return factors

        if self.classical_fallback is not None:
            self.result = self.classical_fallback
        return self.result
