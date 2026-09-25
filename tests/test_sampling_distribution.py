"""``SamplingDistribution``: the read-only mapping every sampling path returns.

Expected values are written out by hand from the §1 encoding: key tuple
``(b_0, b_1, …)`` ↔ packed integer ``Σ_i b_i · 2**i``.
"""

import copy
import pickle

import numpy as np
import pytest

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
    [(1, 1), (1, 1, 0, 0), (2, 0, 0), [1, 1, 0], "110", 3],
)
def test_absent_or_malformed_keys_are_missing(bad_key):
    d = _dist()
    assert bad_key not in d
    with pytest.raises(KeyError):
        d[bad_key]


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
