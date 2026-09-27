"""``SamplingDistribution``: the read-only mapping every sampling path returns.

Expected values are written out by hand from the §1 encoding: key tuple
``(b_0, b_1, …)`` ↔ packed integer ``Σ_i b_i · 2**i``.
"""

import copy
import pickle
import random

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from qarp import SamplingDistribution
from qarp._sampling_distribution import pack_bits

# 3-bit keys: 0 = (0,0,0), 3 = (1,1,0), 5 = (1,0,1).
_OUTCOMES = [0, 3, 5]
_PROBS = [0.2, 0.3, 0.5]
_AS_DICT = {(0, 0, 0): 0.2, (1, 1, 0): 0.3, (1, 0, 1): 0.5}


def _dist():
    return SamplingDistribution(_OUTCOMES, _PROBS, 3)


# ── reading like a dict ─────────────────────────────────────────────────


def test_lookup_membership_and_length():
    d = _dist()
    assert len(d) == 3
    assert d[(1, 1, 0)] == 0.3
    assert d.get((1, 0, 1)) == 0.5
    assert d.get((0, 1, 0), -1.0) == -1.0
    assert (0, 0, 0) in d
    assert (0, 1, 0) not in d


def test_iteration_is_ascending_in_the_packed_integer():
    d = _dist()
    assert list(d) == [(0, 0, 0), (1, 1, 0), (1, 0, 1)]
    assert list(d.items()) == list(_AS_DICT.items())
    assert list(d.values()) == _PROBS


@pytest.mark.parametrize(
    "bad_key",
    [(1, 1), (1, 1, 0, 0), (2, 0, 0), (0.5, 0, 0), ([1], 0, 0), [1, 1, 0], "110", 3],
)
def test_absent_or_malformed_keys_are_missing(bad_key):
    d = _dist()
    assert bad_key not in d
    with pytest.raises(KeyError):
        d[bad_key]


def test_lookups_in_any_order_agree_with_the_hand_written_dict():
    d = _dist()
    for bits in [(1, 0, 1), (0, 0, 0), (1, 0, 1), (1, 1, 0), (0, 0, 0)]:
        assert d[bits] == _AS_DICT[bits]


def test_dict_of_the_distribution_is_the_hand_written_dict_in_order():
    out = dict(_dist())
    assert type(out) is dict
    assert list(out.items()) == list(_AS_DICT.items())


@pytest.mark.parametrize("width", [3, 40, 70])
def test_reversed_iterates_in_descending_packed_order(width):
    """Keys are 0, 3 and 5 padded with zeros, so reversed order is 5, 3, 0."""
    pad = (0,) * (width - 3)
    d = SamplingDistribution([0, 3, 5], [0.2, 0.3, 0.5], width)
    assert list(reversed(d)) == [(1, 0, 1) + pad, (1, 1, 0) + pad, (0, 0, 0) + pad]


def test_bool_and_numpy_bits_match_like_dict_keys():
    d = _dist()
    assert d[(True, True, False)] == 0.3
    assert d[tuple(np.array([1, 0, 1]))] == 0.5


def test_probability_of_reads_by_packed_integer():
    d = _dist()
    for bits, p in _AS_DICT.items():
        assert d.probability_of(sum(b << i for i, b in enumerate(bits))) == p
    assert d.probability_of(3) == 0.3
    assert d.probability_of(np.int64(5)) == 0.5
    assert d.probability_of(1) == 0.0
    assert d.probability_of(7) == 0.0


@pytest.mark.parametrize(
    ("outcome", "error"), [(8, ValueError), (-1, ValueError), (2.0, TypeError)]
)
def test_probability_of_rejects_non_outcomes(outcome, error):
    with pytest.raises(error):
        _dist().probability_of(outcome)


# ── read-only ───────────────────────────────────────────────────────────


def test_writes_raise_and_arrays_are_frozen():
    d = _dist()
    with pytest.raises(TypeError):
        d[(0, 0, 0)] = 1.0  # type: ignore[index]
    with pytest.raises(ValueError):
        d.outcomes[0] = 1
    with pytest.raises(ValueError):
        d.probabilities[0] = 1.0


def test_constructor_copies_its_inputs():
    outcomes = np.array(_OUTCOMES)
    d = SamplingDistribution(outcomes, _PROBS, 3)
    outcomes[0] = 7
    assert d.outcomes[0] == 0
    assert outcomes.flags.writeable


# ── equality ────────────────────────────────────────────────────────────


def test_equals_dict_literal_approx_and_itself():
    d = _dist()
    assert d == _AS_DICT
    assert _AS_DICT == d
    assert d == pytest.approx({(0, 0, 0): 0.2, (1, 1, 0): 0.3, (1, 0, 1): 0.5})
    assert d == _dist()
    assert d != {(0, 0, 0): 1.0}
    assert d != SamplingDistribution([0, 3, 5], [0.2, 0.3, 0.4], 3)
    assert SamplingDistribution([], [], 2) == SamplingDistribution([], [], 5) == {}


def test_to_dict_is_a_plain_dict_in_order():
    out = _dist().to_dict()
    assert type(out) is dict
    assert list(out.items()) == list(_AS_DICT.items())


def test_unhashable_like_dict():
    with pytest.raises(TypeError):
        hash(_dist())


# ── arrays and validation ───────────────────────────────────────────────


def test_arrays_and_width():
    d = _dist()
    assert d.outcomes.tolist() == _OUTCOMES
    assert d.outcomes.dtype == np.int64
    assert d.probabilities.tolist() == _PROBS
    assert d.n_bits_measured == 3


@pytest.mark.parametrize(
    ("outcomes", "probs", "n_bits", "match"),
    [
        ([3, 0], [0.5, 0.5], 2, "strictly ascending"),
        ([1, 1], [0.5, 0.5], 2, "strictly ascending"),
        ([0, 4], [0.5, 0.5], 2, r"\[0, 2\*\*2\)"),
        ([0, 1], [1.0], 2, "align"),
        ([0], [1.0], -1, "non-negative"),
    ],
)
def test_constructor_rejects_invalid_data(outcomes, probs, n_bits, match):
    with pytest.raises(ValueError, match=match):
        SamplingDistribution(outcomes, probs, n_bits)


@pytest.mark.parametrize("outcomes", [[0.0, 1.0], [0.7, 1.9], [False, True], ["0", "1"]])
def test_constructor_rejects_non_integer_outcomes(outcomes):
    with pytest.raises(TypeError, match="outcomes must be integers"):
        SamplingDistribution(outcomes, [0.5, 0.5], 1)


def test_constructor_accepts_numpy_and_python_integers():
    assert SamplingDistribution(np.array([0, 3], dtype=np.uint8), [0.5, 0.5], 2) == {
        (0, 0): 0.5,
        (1, 1): 0.5,
    }
    wide = SamplingDistribution(np.array([1, 1 << 69], dtype=object), [0.5, 0.5], 70)
    assert wide.probability_of(1 << 69) == 0.5


def test_wide_keys_use_python_ints():
    """70-bit keys exceed int64: outcomes are Python ints, keys still tuples."""
    top = (1 << 69) | 1
    d = SamplingDistribution([1, top], [0.25, 0.75], 70)
    assert d.outcomes.dtype == object
    assert d[(1,) + (0,) * 69] == 0.25
    assert d[(1,) + (0,) * 68 + (1,)] == 0.75
    assert list(d) == [(1,) + (0,) * 69, (1,) + (0,) * 68 + (1,)]
    assert d.probability_of(top) == 0.75
    assert d.probability_of(top + 2) == 0.0


# ── copies ──────────────────────────────────────────────────────────────


def test_pickle_and_deepcopy_keep_equality_and_stay_frozen():
    """Round trips: additional to the hand-written oracles above."""
    d = _dist()
    for other in (pickle.loads(pickle.dumps(d)), copy.deepcopy(d)):
        assert other == d
        assert not other.outcomes.flags.writeable


def test_repr_truncates_wide_distributions():
    assert repr(_dist()) == (
        "SamplingDistribution({(0, 0, 0): 0.2, (1, 1, 0): 0.3, (1, 0, 1): 0.5})"
    )
    wide = SamplingDistribution(range(20), [0.05] * 20, 5)
    assert repr(wide).endswith(", ... (4 more)})")


# ── pack_bits ───────────────────────────────────────────────────────────


def test_pack_bits_moves_chosen_bits_to_the_bottom():
    # 0b1011: bit3=1 -> bit0, bit0=1 -> bit1, bit2=0 -> bit2.
    outcomes = np.array([0b1011, 0b0100, 0b1111], dtype=np.int64)
    assert pack_bits(outcomes, [3, 0, 2]).tolist() == [0b011, 0b100, 0b111]


def test_pack_bits_identity_positions_drop_higher_bits():
    outcomes = np.array([0b1011, 0b0110], dtype=np.int64)
    assert pack_bits(outcomes, [0, 1]).tolist() == [0b11, 0b10]
    assert pack_bits(outcomes, []).tolist() == [0, 0]


# ── widths at the edges ─────────────────────────────────────────────────


def test_zero_width_has_the_empty_tuple_as_its_only_key():
    d = SamplingDistribution([0], [1.0], 0)
    assert list(d) == [()] == list(reversed(d))
    assert d[()] == 1.0
    assert d.probability_of(0) == 1.0
    with pytest.raises(ValueError):
        d.probability_of(1)


@pytest.mark.parametrize("width", [63, 64, 65])
def test_keys_across_the_int64_boundary(width):
    """63 bits fit int64; from 64 the keys are Python ints, including the list
    [1, 2**63 + 1] that numpy alone would turn into floats."""
    top = (1 << (width - 1)) | 1
    top_bits = (1,) + (0,) * (width - 2) + (1,)
    d = SamplingDistribution([1, top], [0.25, 0.75], width)
    assert d.outcomes.dtype == (np.int64 if width <= 63 else object)
    assert all(type(k) is int for k in d.outcomes.tolist())
    assert list(d) == [(1,) + (0,) * (width - 1), top_bits]
    assert list(reversed(d)) == list(d)[::-1]
    assert d[top_bits] == 0.75
    assert d.probability_of(top) == 0.75
    assert pickle.loads(pickle.dumps(d)) == d


def test_wide_key_with_a_non_bit_entry_is_missing():
    d = SamplingDistribution([0], [1.0], 40)
    assert (2,) + (0,) * 39 not in d
    assert d.get((2,) + (0,) * 39) is None


# ── property: every view of a distribution agrees ───────────────────────


@st.composite
def _distributions(draw):
    """A width, a seed and an outcome count; the outcomes come from the seed."""
    return (
        draw(st.integers(min_value=0, max_value=70)),
        draw(st.integers(min_value=0, max_value=2**32 - 1)),
        draw(st.integers(min_value=0, max_value=12)),
    )


@pytest.mark.property
@example((0, 0, 1))
@example((32, 1, 12))
@example((33, 2, 12))
@example((63, 3, 12))
@example((64, 4, 12))
@example((65, 5, 12))
@given(_distributions())
def test_lookup_iteration_and_arrays_agree(spec):
    width, seed, count = spec
    rng = random.Random(seed)
    count = min(count, 2**width)
    keys = (
        sorted(rng.sample(range(2**width), count))
        if width < 20
        else sorted({rng.getrandbits(width) for _ in range(count)})
    )
    probs = [rng.random() for _ in keys]
    d = SamplingDistribution(keys, probs, width)

    def bits(k):
        return tuple((k >> i) & 1 for i in range(width))

    assert list(d) == [bits(k) for k in keys]
    assert list(reversed(d)) == [bits(k) for k in reversed(keys)]
    assert d.to_dict() == {bits(k): p for k, p in zip(keys, probs, strict=True)}
    for k, p in zip(keys, probs, strict=True):
        assert d[bits(k)] == p
        assert d.probability_of(k) == p
    if count < 2**width:
        absent = next(k for k in range(2 ** min(width, 20)) if k not in set(keys))
        assert bits(absent) not in d
        assert d.probability_of(absent) == 0.0
