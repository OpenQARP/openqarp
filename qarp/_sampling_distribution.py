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
from typing import Any, Optional

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
    return SamplingDistribution._wrap(
        keys, np.bincount(inverse, weights=weights, minlength=len(keys)), len(measured)
    )


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
    ``dict(d)``, which looks each key up again.

    Args:
        outcomes: Packed integers ``Σ_i b_i · 2**i`` of the keys, strictly
            ascending, each below ``2**n_bits_measured``.
        probabilities: One probability per outcome.
        n_bits_measured: Bits per key — the number of measured qubits.
    """

    __slots__ = ("_cursor", "_n_bits", "_outcomes", "_probabilities")

    def __init__(self, outcomes, probabilities, n_bits_measured: int):
        n_bits = int(n_bits_measured)
        if n_bits < 0:
            raise ValueError(f"n_bits_measured must be non-negative, got {n_bits}")
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
        probs = np.array(probabilities, dtype=np.float64).reshape(-1)
        if keys.shape != probs.shape:
            raise ValueError(
                f"{len(keys)} outcomes but {len(probs)} probabilities; they must align"
            )
        if len(keys):
            if np.any(keys[1:] <= keys[:-1]):
                raise ValueError("outcomes must be strictly ascending")
            if int(keys[0]) < 0 or int(keys[-1]) >> n_bits:
                raise ValueError(f"outcomes must lie in [0, 2**{n_bits})")
        self._init(keys, probs, n_bits)

    def _init(self, keys: np.ndarray, probs: np.ndarray, n_bits: int) -> None:
        keys = keys.astype(_outcome_dtype(n_bits), copy=False)
        keys.flags.writeable = False
        probs.flags.writeable = False
        self._outcomes = keys
        self._probabilities = probs
        self._n_bits = n_bits
        self._cursor = 0

    @classmethod
    def _wrap(cls, keys: np.ndarray, probs: np.ndarray, n_bits: int) -> "SamplingDistribution":
        """Take ownership of arrays already known to be valid."""
        self = cls.__new__(cls)
        self._init(keys, probs, n_bits)
        return self

    @classmethod
    def _from_mapping(cls, mapping: Mapping) -> "SamplingDistribution":
        """Pack a ``{bits-tuple: probability}`` mapping whose keys share one width."""
        if isinstance(mapping, SamplingDistribution):
            return mapping
        if not mapping:
            return cls._wrap(np.zeros(0, dtype=np.int64), np.zeros(0), 0)
        for bits in mapping:
            if not isinstance(bits, tuple):
                raise ValueError(f"distribution key {bits!r} is not a tuple of 0/1 bits")
        widths = {len(bits) for bits in mapping}
        if len(widths) != 1:
            raise ValueError(f"distribution keys have mixed widths {sorted(widths)}")
        (n_bits,) = widths
        packed = []
        for bits in mapping:
            key = _bits_key(bits, n_bits)
            if key is None:
                raise ValueError(f"distribution key {bits!r} is not a tuple of 0/1 bits")
            packed.append(key)
        keys = np.array(packed, dtype=_outcome_dtype(n_bits))
        order = np.argsort(keys, kind="stable")
        probs = np.fromiter(mapping.values(), dtype=np.float64, count=len(mapping))
        return cls._wrap(keys[order], probs[order], n_bits)

    @property
    def outcomes(self) -> np.ndarray:
        """Packed keys ``Σ_i b_i · 2**i``, strictly ascending, read-only."""
        return self._outcomes

    @property
    def probabilities(self) -> np.ndarray:
        """Probabilities aligned with :attr:`outcomes`, read-only."""
        return self._probabilities

    @property
    def n_bits_measured(self) -> int:
        """Bits per key: the number of measured qubits."""
        return self._n_bits

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
        return (type(self), (self._outcomes, self._probabilities, self._n_bits))

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
