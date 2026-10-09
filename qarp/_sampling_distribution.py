"""Array-backed sampling distribution — the ``Sampler`` output type.

Keys are LSB-first bit tuples over the measured qubits (§1).  The data lives
in two aligned arrays: ``outcomes``, the packed integers ``Σ_i b_i · 2**i`` of
the keys in strictly ascending order, and ``probabilities``.  Tuples are built
only when a caller iterates or looks one up, so wide ``qarp.EXACT``
distributions cost array operations, not one Python object per outcome.
"""

from collections.abc import ItemsView, Iterator, Mapping, Sequence, ValuesView
from functools import lru_cache
from itertools import product
from operator import add, index
from typing import Any, Optional, Union

import numpy as np

from ._types import outcome_arrays

# Packed outcomes fit int64 up to this many bits; wider keys use Python ints.
_INT64_BITS = 63
# Tuple keys are joined from two lookup tables of 2**ceil(k/2) entries each,
# which bounds the table path to this many bits.
_MAX_TABLE_BITS = 32
_CHUNK = 1 << 16
_REPR_ENTRIES = 16


@lru_cache(maxsize=None)
def _lsb_tuples(width: int) -> tuple[tuple[int, ...], ...]:
    """Every ``width``-bit LSB-first tuple, indexed by its integer value."""
    return tuple(bits[::-1] for bits in product((0, 1), repeat=width))


@lru_cache(maxsize=None)
def _lsb_index(width: int) -> dict[tuple[int, ...], int]:
    """Inverse of :func:`_lsb_tuples`: each ``width``-bit tuple to its integer."""
    return {bits: i for i, bits in enumerate(_lsb_tuples(width))}


def pack_bits(outcomes: np.ndarray, positions: Sequence[int]) -> np.ndarray:
    """Bit ``positions[i]`` of each outcome moved to bit ``i``; other bits dropped."""
    positions = list(positions)
    if positions == list(range(len(positions))):
        return outcomes & ((1 << len(positions)) - 1)
    packed = np.zeros_like(outcomes)
    for i, q in enumerate(positions):
        packed |= ((outcomes >> q) & 1) << i
    return packed


def _outcome_dtype(n_bits: int):
    return np.int64 if n_bits <= _INT64_BITS else object


def _tuples(outcomes: np.ndarray, n_bits: int) -> Iterator[tuple[int, ...]]:
    """LSB-first ``n_bits``-bit tuples of ``outcomes``, in the array's order."""
    if n_bits <= _MAX_TABLE_BITS:
        low = n_bits // 2
        lo, hi = _lsb_tuples(low), _lsb_tuples(n_bits - low)
        mask = (1 << low) - 1
        for start in range(0, len(outcomes), _CHUNK):
            block = outcomes[start : start + _CHUNK]
            yield from map(
                add,
                map(lo.__getitem__, (block & mask).tolist()),
                map(hi.__getitem__, (block >> low).tolist()),
            )
    elif n_bits <= _INT64_BITS:
        shifts = np.arange(n_bits)
        for start in range(0, len(outcomes), _CHUNK):
            block = outcomes[start : start + _CHUNK]
            yield from map(tuple, ((block[:, None] >> shifts) & 1).tolist())
    else:
        for key in outcomes.tolist():
            yield tuple((key >> i) & 1 for i in range(n_bits))


def distribution_from_result(result, measured: Sequence[int]) -> "SamplingDistribution":
    """Marginal of a sampling-shaped result on ``measured``, normalised by ``n_shots``.

    ``result`` is a ``qx.SamplingResult``, an :class:`~qarp.ExactResult`, or
    anything exposing ``counts`` / ``n_shots`` / ``n_qubits``.
    """
    measured = list(measured)
    if result.n_qubits <= _INT64_BITS:
        outcomes, weights = outcome_arrays(result)
    else:
        counts = result.counts
        outcomes = np.fromiter(counts.keys(), dtype=object, count=len(counts))
        weights = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
    weights = weights / result.n_shots
    keys, inverse = np.unique(pack_bits(outcomes, measured), return_inverse=True)
    n_shots = None if getattr(result, "is_exact", False) else int(result.n_shots)
    return SamplingDistribution._wrap(
        keys, np.bincount(inverse, weights=weights, minlength=len(keys)), len(measured), n_shots
    )


def _checked_shots(n_shots: Optional[int]) -> Optional[int]:
    if n_shots is None:
        return None
    shots = index(n_shots)
    if shots < 0:
        raise ValueError(f"n_shots must be non-negative, got {shots}")
    return shots


def _checked_width(n_bits_measured: int) -> int:
    width = index(n_bits_measured)
    if width < 0:
        raise ValueError(f"n_bits_measured must be non-negative, got {width}")
    return width


def _rebuild(outcomes, probabilities, n_bits_measured, n_shots):
    return SamplingDistribution(outcomes, probabilities, n_bits_measured, n_shots=n_shots)


def _bits_key(bits: object, n_bits: int) -> Optional[int]:
    """The packed integer of an LSB-first 0/1 tuple of length ``n_bits``, else ``None``."""
    if not isinstance(bits, tuple) or len(bits) != n_bits:
        return None
    if n_bits <= _MAX_TABLE_BITS:
        low = n_bits // 2
        try:
            return _lsb_index(low)[bits[:low]] | (_lsb_index(n_bits - low)[bits[low:]] << low)
        except (KeyError, TypeError):
            return None
    key = 0
    for i, b in enumerate(bits):
        if b == 1:
            key |= 1 << i
        elif b != 0:
            return None
    return key


class _Items(ItemsView):
    _mapping: "SamplingDistribution"

    def __iter__(self):
        return zip(iter(self._mapping), self._mapping.probabilities.tolist(), strict=True)


class _Values(ValuesView):
    _mapping: "SamplingDistribution"

    def __iter__(self):
        return iter(self._mapping.probabilities.tolist())


class SamplingDistribution(Mapping[tuple[int, ...], float]):
    """Read-only sampling distribution over LSB-first bit tuples.

    Reads like a ``{bits-tuple: probability}`` dict, iterating in ascending
    order of the packed integer.  ``outcomes`` and ``probabilities`` give the
    same data as aligned read-only arrays; ``probability_of(k)`` looks one up
    by its packed integer; ``to_dict()`` makes a plain dict, faster than
    ``dict(d)``, which looks each key up again.  Positions passed to the
    analysis methods index the key tuple, not physical qubits.

    Args:
        outcomes: Packed integers ``Σ_i b_i · 2**i`` of the keys, strictly
            ascending, each below ``2**n_bits_measured``.
        probabilities: One probability per outcome.
        n_bits_measured: Bits per key — the number of measured qubits.
        n_shots: Shots behind a sampled distribution; ``None`` for an exact
            one.  Equality and iteration ignore it.
    """

    __slots__ = ("_cursor", "_n_bits", "_n_shots", "_outcomes", "_probabilities")

    def __init__(
        self, outcomes, probabilities, n_bits_measured: int, *, n_shots: Optional[int] = None
    ):
        n_bits = _checked_width(n_bits_measured)
        wide = n_bits > _INT64_BITS
        # Past 63 bits the keys stay Python ints: numpy infers float64 for a
        # list mixing ints below and above 2**63.
        raw = np.asarray(outcomes, dtype=object if wide else None).reshape(-1)
        if raw.dtype.kind not in "iu":
            for k in raw.tolist():
                if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
                    raise TypeError(f"outcomes must be integers, got {k!r}")
        keys = (
            np.array([int(k) for k in raw.tolist()], dtype=object) if wide else raw.astype(np.int64)
        )
        probs = np.array(np.reshape(probabilities, -1), dtype=np.float64)
        if keys.shape != probs.shape:
            raise ValueError(
                f"{len(keys)} outcomes but {len(probs)} probabilities; they must align"
            )
        if len(keys):
            if np.any(keys[1:] <= keys[:-1]):
                raise ValueError("outcomes must be strictly ascending")
            if int(keys[0]) < 0 or int(keys[-1]) >> n_bits:
                raise ValueError(f"outcomes must lie in [0, 2**{n_bits})")
        self._init(keys, probs, n_bits, _checked_shots(n_shots))

    def _init(
        self, keys: np.ndarray, probs: np.ndarray, n_bits: int, n_shots: Optional[int]
    ) -> None:
        keys = keys.astype(_outcome_dtype(n_bits), copy=False)
        keys.flags.writeable = False
        probs.flags.writeable = False
        self._outcomes = keys
        self._probabilities = probs
        self._n_bits = n_bits
        self._n_shots = n_shots
        self._cursor = 0

    @classmethod
    def _wrap(
        cls, keys: np.ndarray, probs: np.ndarray, n_bits: int, n_shots: Optional[int] = None
    ) -> "SamplingDistribution":
        """Take ownership of arrays already known to be valid.  Each must own its
        memory: numpy lets a view of writable memory be made writable again."""
        self = cls.__new__(cls)
        self._init(keys, probs, n_bits, n_shots)
        return self

    @classmethod
    def from_dict(
        cls,
        mapping: Mapping,
        *,
        n_bits_measured: Optional[int] = None,
        n_shots: Optional[int] = None,
    ) -> "SamplingDistribution":
        """A distribution from a ``{bits-tuple: probability}`` mapping.

        Keys must be 0/1 tuples of one width.  ``n_bits_measured`` gives the
        width of an empty mapping and must match the keys otherwise.  A
        ``SamplingDistribution`` comes back as it is unless ``n_shots`` is given.
        """
        shots = _checked_shots(n_shots)
        if n_bits_measured is not None:
            n_bits_measured = _checked_width(n_bits_measured)
        if isinstance(mapping, SamplingDistribution):
            if n_bits_measured is not None and n_bits_measured != mapping._n_bits:
                raise ValueError(
                    f"keys have {mapping._n_bits} bits, not n_bits_measured={n_bits_measured}"
                )
            if shots is None:
                return mapping
            return cls._wrap(mapping._outcomes, mapping._probabilities, mapping._n_bits, shots)
        if not mapping:
            width = 0 if n_bits_measured is None else n_bits_measured
            return cls._wrap(np.zeros(0, dtype=np.int64), np.zeros(0), width, shots)
        for bits in mapping:
            if not isinstance(bits, tuple):
                raise ValueError(f"distribution key {bits!r} is not a tuple of 0/1 bits")
        widths = {len(bits) for bits in mapping}
        if len(widths) != 1:
            raise ValueError(f"distribution keys have mixed widths {sorted(widths)}")
        (n_bits,) = widths
        if n_bits_measured is not None and n_bits_measured != n_bits:
            raise ValueError(f"keys have {n_bits} bits, not n_bits_measured={n_bits_measured}")
        packed = []
        for bits in mapping:
            key = _bits_key(bits, n_bits)
            if key is None:
                raise ValueError(f"distribution key {bits!r} is not a tuple of 0/1 bits")
            packed.append(key)
        keys = np.array(packed, dtype=_outcome_dtype(n_bits))
        order = np.argsort(keys, kind="stable")
        probs = np.fromiter(mapping.values(), dtype=np.float64, count=len(mapping))
        return cls._wrap(keys[order], probs[order], n_bits, shots)

    @property
    def outcomes(self) -> np.ndarray:
        """Packed keys ``Σ_i b_i · 2**i``, strictly ascending, read-only."""
        # A view, so numpy refuses to make it writable: its base is read-only.
        return self._outcomes.view()

    @property
    def probabilities(self) -> np.ndarray:
        """Probabilities aligned with :attr:`outcomes`, read-only."""
        return self._probabilities.view()

    @property
    def n_bits_measured(self) -> int:
        """Bits per key: the number of measured qubits."""
        return self._n_bits

    @property
    def n_shots(self) -> Optional[int]:
        """Shots behind a sampled distribution; ``None`` for an exact one."""
        return self._n_shots

    def counts(self) -> np.ndarray:
        """Shot counts aligned with :attr:`outcomes`.  ``ValueError`` when exact."""
        if self._n_shots is None:
            raise ValueError("an exact distribution has no shot counts")
        return np.rint(self._probabilities * self._n_shots).astype(np.int64)

    def standard_errors(self) -> np.ndarray:
        """Binomial standard error ``sqrt(p(1 - p) / n_shots)`` per outcome;
        zeros for an exact distribution, whose values carry no shot noise."""
        p = self._probabilities
        if self._n_shots is None:
            return np.zeros_like(p)
        return np.sqrt(p * (1.0 - p) / self._n_shots)

    def _checked_positions(self, positions) -> list[int]:
        chosen = [index(q) for q in positions]
        if len(set(chosen)) != len(chosen):
            raise ValueError(f"positions must be distinct, got {chosen}")
        for q in chosen:
            if not 0 <= q < self._n_bits:
                raise ValueError(f"position {q} is outside the {self._n_bits}-bit keys")
        return chosen

    def marginal(self, positions) -> "SamplingDistribution":
        """The distribution over ``positions``: bit ``i`` of a new key is key
        position ``positions[i]``.  Keeps ``n_shots``."""
        chosen = self._checked_positions(positions)
        keys, inverse = np.unique(pack_bits(self._outcomes, chosen), return_inverse=True)
        probs = np.bincount(inverse, weights=self._probabilities, minlength=len(keys))
        return SamplingDistribution._wrap(keys, probs, len(chosen), self._n_shots)

    def to_dense(self, max_bits: int = 28) -> np.ndarray:
        """The length ``2**n_bits_measured`` probability vector, indexed by
        packed outcome.  ``ValueError`` past ``max_bits``, which bounds the
        allocation (``2**28`` doubles are 2 GiB)."""
        if self._n_bits > max_bits:
            raise ValueError(
                f"{self._n_bits}-bit keys need 2**{self._n_bits} entries; max_bits is {max_bits}"
            )
        dense = np.zeros(1 << self._n_bits)
        dense[self._outcomes] = self._probabilities
        return dense

    def parity_expectation(self, positions) -> float:
        """``Σ p · (−1)^(parity of the bits at positions)``: the expectation of the
        product of ``Z`` over ``positions``, probabilities as given.  Empty
        ``positions`` give the total probability."""
        parity = np.zeros_like(self._outcomes)
        for q in self._checked_positions(positions):
            parity ^= (self._outcomes >> q) & 1
        signs = 1.0 - 2.0 * parity.astype(np.float64)
        return float(signs @ self._probabilities)

    def top(self, k: int) -> list[tuple[tuple[int, ...], float]]:
        """The ``k`` most probable ``(bits, probability)`` pairs, most probable
        first; ties go to the smaller packed outcome."""
        count = index(k)
        if count < 0:
            raise ValueError(f"k must be non-negative, got {count}")
        order = np.argsort(-self._probabilities, kind="stable")[:count]
        return list(
            zip(
                _tuples(self._outcomes[order], self._n_bits),
                self._probabilities[order].tolist(),
                strict=True,
            )
        )

    def most_likely(self) -> tuple[tuple[int, ...], float]:
        """The most probable ``(bits, probability)``; the smaller packed
        outcome wins a tie.  ``ValueError`` when empty."""
        if not len(self):
            raise ValueError("an empty distribution has no most likely outcome")
        return self.top(1)[0]

    def _aligned(self, other) -> tuple[np.ndarray, np.ndarray]:
        """Both distributions' probabilities over the union of their outcomes."""
        theirs = SamplingDistribution.from_dict(other)
        if theirs._n_bits != self._n_bits and len(theirs) and len(self):
            raise ValueError(
                f"cannot compare {self._n_bits}-bit keys with {theirs._n_bits}-bit keys"
            )
        union = np.union1d(self._outcomes, theirs._outcomes)
        mine, other_probs = np.zeros(len(union)), np.zeros(len(union))
        mine[np.searchsorted(union, self._outcomes)] = self._probabilities
        other_probs[np.searchsorted(union, theirs._outcomes)] = theirs._probabilities
        return mine, other_probs

    def total_variation(self, other: Mapping) -> float:
        """``½ Σ |p − q|`` over the union of outcomes, probabilities as given."""
        p, q = self._aligned(other)
        return float(0.5 * np.abs(p - q).sum())

    def hellinger_fidelity(self, other: Mapping) -> float:
        """``(Σ √(p q))²``, qiskit's definition, probabilities as given."""
        p, q = self._aligned(other)
        return float(np.sqrt(p * q).sum() ** 2)

    def sample(
        self, n_shots: int, seed: Union[int, np.random.Generator, None] = None
    ) -> "SamplingDistribution":
        """A multinomial draw of ``n_shots`` from the probabilities renormalised
        to sum to 1 (exact results prune below 1e-12).  Outcomes drawn zero
        times are dropped."""
        shots = index(n_shots)
        if shots < 1:
            raise ValueError(f"n_shots must be positive, got {shots}")
        total = float(self._probabilities.sum())
        if not total > 0.0:
            raise ValueError("cannot sample a distribution with no probability mass")
        drawn = np.random.default_rng(seed).multinomial(shots, self._probabilities / total)
        kept = drawn > 0
        return SamplingDistribution._wrap(
            self._outcomes[kept].copy(), drawn[kept] / shots, self._n_bits, shots
        )

    def _position(self, key: int) -> Optional[int]:
        # Keys looked up in iteration order (``dict(d)``, ``d.items()`` in
        # user loops) hit the slot after the previous hit; others bisect.
        i = self._cursor
        if not (i < len(self._outcomes) and self._outcomes[i] == key):
            i = int(np.searchsorted(self._outcomes, key))
            if not (i < len(self._outcomes) and self._outcomes[i] == key):
                return None
        self._cursor = i + 1
        return i

    def _index(self, bits: object) -> Optional[int]:
        key = _bits_key(bits, self._n_bits)
        return None if key is None else self._position(key)

    def probability_of(self, outcome: int) -> float:
        """Probability of the outcome whose packed integer is ``outcome``.

        ``outcome`` is ``Σ_i b_i · 2**i`` of the key tuple; an outcome that
        never occurred has probability 0.0.  Raises ``ValueError`` outside
        ``[0, 2**n_bits_measured)``.
        """
        key = index(outcome)
        if key < 0 or key >> self._n_bits:
            raise ValueError(f"outcome {key} is outside [0, 2**{self._n_bits})")
        i = self._position(key)
        return 0.0 if i is None else float(self._probabilities[i])

    def __getitem__(self, bits: tuple[int, ...]) -> float:
        i = self._index(bits)
        if i is None:
            raise KeyError(bits)
        return float(self._probabilities[i])

    def __contains__(self, bits: object) -> bool:
        return self._index(bits) is not None

    def __len__(self) -> int:
        return len(self._outcomes)

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        return _tuples(self._outcomes, self._n_bits)

    def __reversed__(self) -> Iterator[tuple[int, ...]]:
        return _tuples(self._outcomes[::-1], self._n_bits)

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        # numpy would otherwise read the mapping as a sequence of key tuples.
        raise TypeError(
            "a SamplingDistribution is a mapping, not an array; use .probabilities, "
            ".outcomes or .to_dense()"
        )

    def items(self) -> _Items:
        return _Items(self)

    def values(self) -> _Values:
        return _Values(self)

    def to_dict(self) -> dict[tuple[int, ...], float]:
        """A plain ``{bits-tuple: probability}`` dict, in the same order."""
        return dict(self.items())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SamplingDistribution):
            if self._n_bits != other._n_bits:
                return len(self) == 0 and len(other) == 0
            return bool(
                np.array_equal(self._outcomes, other._outcomes)
                and np.array_equal(self._probabilities, other._probabilities)
            )
        if isinstance(other, Mapping):
            return self.to_dict() == dict(other.items())
        return NotImplemented

    def __reduce__(self) -> tuple[Any, ...]:
        return (_rebuild, (self._outcomes, self._probabilities, self._n_bits, self._n_shots))

    def __repr__(self) -> str:
        head = SamplingDistribution._wrap(
            self._outcomes[:_REPR_ENTRIES].copy(),
            self._probabilities[:_REPR_ENTRIES].copy(),
            self._n_bits,
        )
        body = ", ".join(f"{bits!r}: {p!r}" for bits, p in head.items())
        if len(self) > _REPR_ENTRIES:
            body += f", ... ({len(self) - _REPR_ENTRIES} more)"
        return f"{type(self).__name__}({{{body}}})"
