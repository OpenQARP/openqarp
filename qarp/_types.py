"""Type definitions or aliases for OpenQARP."""

from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property

import numpy as np

# Type alias for dictionary of sampling results
SamplingDictionary = dict[tuple[int, ...], float]


class Shots(Enum):
    """Non-integer shot-count sentinels for the ``n_shots`` knob.

    ``Shots.EXACT`` (re-exported as ``qarp.EXACT``) requests the ∞-shot limit
    of a sampling primitive: the engine computes the exact Born distribution
    |ψ|² instead of drawing samples.  Deliberately a plain ``Enum``, never
    ``IntEnum``: arithmetic or ordering on a leaked sentinel must fail loudly
    with ``TypeError``, not compute garbage.  Enum members survive ``deepcopy``
    with identity intact (``copy.n_shots is EXACT`` keeps working through the
    QSE/MonteCarlo/QMEGS deepcopy paths — a plain ``object()`` sentinel would
    not).
    """

    EXACT = "exact"


@dataclass
class ExactResult:
    """Exact Born distribution duck-typing ``qx.SamplingResult``.

    ``keys`` (sorted int64 outcome bitmasks) and ``probs`` carry the
    distribution, pruned below ~1e-12 so it sums to 1 − O(pruned) — compare
    distributions at ~1e-10, never exactly.  ``counts`` is the same data as a
    dict, built on first access: estimators that can consume the arrays
    (:func:`outcome_arrays`) never pay for a 2^n-entry dict.  ``n_shots = 1``
    is a normalization trick so every estimator's ``count / n_shots`` yields
    the probability unchanged — it is NOT a statistical claim: never infer
    shot-noise error bars or adaptive shot budgets from it (the variance of
    these numbers is exactly 0).  Check ``is_exact`` to discriminate from a
    sampled result.
    """

    n_qubits: int
    keys: np.ndarray
    probs: np.ndarray
    n_shots: int = 1
    is_exact: bool = field(default=True, init=False)

    @cached_property
    def counts(self) -> dict[int, float]:
        return dict(zip(self.keys.tolist(), self.probs.tolist(), strict=True))


def outcome_arrays(result) -> tuple[np.ndarray, np.ndarray]:
    """``(outcomes, counts)`` int64/float64 arrays of a sampling-shaped result.

    An :class:`ExactResult` hands over its arrays; a ``qx.SamplingResult``
    (whose ``counts`` property converts the C++ map on every access) is read
    once.  Outcomes are packed into int64, so registers wider than 63 qubits
    must take the dict path instead.
    """
    keys = getattr(result, "keys", None)
    if keys is not None:
        return keys, result.probs
    counts = result.counts
    return (
        np.fromiter(counts.keys(), dtype=np.int64, count=len(counts)),
        np.fromiter(counts.values(), dtype=np.float64, count=len(counts)),
    )


class Consumes(Enum):
    """What raw engine output a primitive's estimator consumes.

    Declared class-level on each primitive; engines dispatch on it.  Lives
    here, below both layers, so engines can read it without importing from
    ``qarp.algorithms``.

    Attributes:
        COUNTS: Measurement statistics from sampled circuits (protocol
            primitives: Sampler, HadamardTest, PauliAveraging, ...).
        AMPLITUDES: Simulator statevectors, contracted directly (StateVector).
            Requires an engine with amplitude access — noiseless simulation.
    """

    COUNTS = 0
    AMPLITUDES = 1
