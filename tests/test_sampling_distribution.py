"""``SamplingDistribution``: the read-only mapping every sampling path returns.

Expected values are written out by hand from the §1 encoding: key tuple
``(b_0, b_1, …)`` ↔ packed integer ``Σ_i b_i · 2**i``.
"""

import copy
import pickle
import random
from functools import reduce
from itertools import product
from math import cos, prod, sin, sqrt

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

import qarp
from qarp import SamplingDistribution
from qarp._sampling_distribution import pack_bits
from qarp.algorithms import Sampler
from qarp.blocks import SimpleBlock
from qarp.engines import QarpEngine

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


def _built_by(path):
    d = _dist()
    return {
        "constructor": lambda: d,
        "from_dict": lambda: SamplingDistribution.from_dict(_AS_DICT),
        "from_dict_shared": lambda: SamplingDistribution.from_dict(d, n_shots=10),
        "marginal": lambda: d.marginal([0, 2]),
        "sample": lambda: d.sample(100, seed=1),
        "postselect": lambda: qarp.PostSelection({2: 1}).apply(d).distribution,
        "pickle": lambda: pickle.loads(pickle.dumps(d)),
        "sampler": lambda: _exact(_RyProduct()),
    }[path]()


@pytest.mark.parametrize(
    "path",
    [
        "constructor",
        "from_dict",
        "from_dict_shared",
        "marginal",
        "sample",
        "postselect",
        "pickle",
        "sampler",
    ],
)
@pytest.mark.parametrize("name", ["outcomes", "probabilities"])
def test_arrays_cannot_be_made_writable_again(path, name):
    array = getattr(_built_by(path), name)
    with pytest.raises(ValueError):
        array.flags.writeable = True


@pytest.mark.parametrize(
    "convert",
    [np.array, np.asarray, np.sum, lambda d: np.array([d, d])],
    ids=["array", "asarray", "sum", "list_of_two"],
)
def test_numpy_refuses_to_read_the_mapping_as_an_array(convert):
    with pytest.raises(TypeError, match=r"\.probabilities"):
        convert(_dist())


def test_explicit_arrays_still_convert():
    d = _dist()
    assert np.array(list(d.values())).tolist() == _PROBS
    assert np.asarray(d.probabilities).tolist() == _PROBS


_BUILDERS = {
    "constructor": lambda width: SamplingDistribution([1], [1.0], width),
    "from_dict_empty": lambda width: SamplingDistribution.from_dict({}, n_bits_measured=width),
    "from_dict": lambda width: SamplingDistribution.from_dict({(1, 0): 1.0}, n_bits_measured=width),
    "from_dict_shared": lambda width: SamplingDistribution.from_dict(
        SamplingDistribution([1], [1.0], 2), n_bits_measured=width
    ),
}


@pytest.mark.parametrize("builder", sorted(_BUILDERS))
@pytest.mark.parametrize(
    ("width", "error"), [(-1, ValueError), (2.7, TypeError), (2.0, TypeError), ("2", TypeError)]
)
def test_widths_must_be_non_negative_integers(builder, width, error):
    with pytest.raises(error):
        _BUILDERS[builder](width)


@pytest.mark.parametrize("builder", sorted(_BUILDERS))
def test_numpy_integer_widths_are_accepted(builder):
    assert _BUILDERS[builder](np.int64(2)).n_bits_measured == 2


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


def test_equal_keys_of_different_widths_are_different_outcomes():
    """Key 1 is (1, 0) at width 2 and (1, 0, 0) at width 3."""
    narrow = SamplingDistribution([1], [1.0], 2)
    assert narrow != SamplingDistribution([1], [1.0], 3)
    assert narrow == {(1, 0): 1.0}
    assert narrow != {(1, 0, 0): 1.0}


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


@pytest.mark.parametrize("width", [18, 40])
def test_iteration_across_the_chunk_boundary(width):
    """65 540 outcomes span two 65 536-key chunks, on the lookup-table path (18
    bits) and the shift path (40 bits)."""
    keys = [3 * k for k in range(65_540)]
    d = SamplingDistribution(keys, np.full(len(keys), 1 / len(keys)), width)
    tuples = list(d)
    assert len(tuples) == len(keys)
    for i in (0, 65_535, 65_536, 65_537, len(keys) - 1):
        assert tuples[i] == tuple((keys[i] >> b) & 1 for b in range(width))


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


# ── analysis utilities ──────────────────────────────────────────────────
# ⊗_q Ry(θ_q)|0⟩: bit q is 1 with probability sin²(θ_q/2), independently, so
# ⟨Z_q⟩ = cos θ_q and every marginal is a product over its positions.

_THETAS = (0.3, 1.1, 2.0, 0.7, 2.6)


class _RyProduct(SimpleBlock):
    def __init__(self):
        super().__init__(len(_THETAS))

    def build_vanilla(self):
        for q, theta in enumerate(_THETAS):
            self.ry(q, theta)


class _Ghz(SimpleBlock):
    def __init__(self, n):
        super().__init__(n)

    def build_vanilla(self):
        self.h(0)
        for q in range(self.n_qubits - 1):
            self.cx(q, q + 1)


def _exact(block):
    sampler = Sampler(ket=block, n_shots=qarp.EXACT)
    engine = QarpEngine()
    engine.build([sampler])
    return engine.run()[0]


def _bit_probability(q, bit):
    return sin(_THETAS[q] / 2) ** 2 if bit else cos(_THETAS[q] / 2) ** 2


@pytest.mark.parametrize("positions", [[0, 1, 2, 3, 4], [0, 2, 4], [3, 0, 1]])
def test_marginal_of_the_product_state(positions):
    marginal = _exact(_RyProduct()).marginal(positions)
    expected = {
        bits: prod(_bit_probability(q, b) for q, b in zip(positions, bits, strict=True))
        for bits in product((0, 1), repeat=len(positions))
    }
    assert marginal.n_bits_measured == len(positions)
    assert marginal.keys() == expected.keys()
    for bits, p in expected.items():
        assert marginal[bits] == pytest.approx(p, abs=1e-12)


@pytest.mark.parametrize("positions", [[0, 0], [5], [-1]])
def test_marginal_rejects_repeated_or_missing_positions(positions):
    with pytest.raises(ValueError):
        _dist().marginal(positions)


def test_to_dense_of_the_product_state():
    pairs = [np.array([_bit_probability(q, 0), _bit_probability(q, 1)]) for q in range(5)]
    expected = reduce(np.kron, reversed(pairs))  # qubit 0 innermost
    np.testing.assert_allclose(_exact(_RyProduct()).to_dense(), expected, atol=1e-12)


def test_to_dense_refuses_past_max_bits():
    with pytest.raises(ValueError, match="max_bits"):
        SamplingDistribution([0], [1.0], 29).to_dense()
    with pytest.raises(ValueError, match="max_bits"):
        _dist().to_dense(max_bits=2)


@pytest.mark.parametrize(
    ("positions", "expected"),
    [
        ([1], cos(_THETAS[1])),
        ([3], cos(_THETAS[3])),
        ([0, 2], cos(_THETAS[0]) * cos(_THETAS[2])),
        ([1, 4], cos(_THETAS[1]) * cos(_THETAS[4])),
        ([], 1.0),
    ],
)
def test_parity_expectation_of_the_product_state(positions, expected):
    value = _exact(_RyProduct()).parity_expectation(positions)
    assert value == pytest.approx(expected, abs=1e-10)


def test_parity_expectation_of_ghz():
    dist = _exact(_Ghz(3))
    assert dist.parity_expectation([0]) == pytest.approx(0.0, abs=1e-12)
    assert dist.parity_expectation([0, 1]) == pytest.approx(1.0, abs=1e-12)
    assert dist.parity_expectation([0, 1, 2]) == pytest.approx(0.0, abs=1e-12)


def test_top_and_most_likely_break_ties_by_smaller_outcome():
    # 0 = (0, 0), 1 = (1, 0), 2 = (0, 1), 3 = (1, 1)
    d = SamplingDistribution([0, 1, 2, 3], [0.2, 0.3, 0.3, 0.2], 2)
    assert d.top(3) == [((1, 0), 0.3), ((0, 1), 0.3), ((0, 0), 0.2)]
    assert d.most_likely() == ((1, 0), 0.3)
    assert d.top(0) == []
    assert len(d.top(10)) == 4
    with pytest.raises(ValueError):
        d.top(-1)
    with pytest.raises(ValueError):
        SamplingDistribution([], [], 2).most_likely()


def test_most_likely_outcome_of_the_product_state():
    bits = tuple(int(_bit_probability(q, 1) > _bit_probability(q, 0)) for q in range(5))
    best_bits, best_p = _exact(_RyProduct()).most_likely()
    assert best_bits == bits
    assert best_p == pytest.approx(prod(_bit_probability(q, b) for q, b in enumerate(bits)))


def test_from_dict_packs_infers_width_and_validates():
    d = SamplingDistribution.from_dict({(0, 1): 0.75, (1, 0): 0.25})
    assert d.outcomes.tolist() == [1, 2]
    assert d.probabilities.tolist() == [0.25, 0.75]
    assert d.n_bits_measured == 2
    assert SamplingDistribution.from_dict({}).n_bits_measured == 0
    assert SamplingDistribution.from_dict({}, n_bits_measured=3).n_bits_measured == 3
    assert SamplingDistribution.from_dict(d) is d
    assert SamplingDistribution.from_dict(d, n_shots=40).n_shots == 40
    for bad in ({(0, 1): 0.5, (1,): 0.5}, {3: 1.0}, {(2, 0): 1.0}):
        with pytest.raises(ValueError):
            SamplingDistribution.from_dict(bad)
    with pytest.raises(ValueError):
        SamplingDistribution.from_dict({(0, 1): 1.0}, n_bits_measured=3)


def test_distances_between_one_bit_distributions():
    """p(1) = 0.3 against q(1) = 0.8, and disjoint supports."""
    d = SamplingDistribution([0, 1], [0.7, 0.3], 1)
    other = {(0,): 0.2, (1,): 0.8}
    assert d.total_variation(other) == pytest.approx(0.5)
    assert d.hellinger_fidelity(other) == pytest.approx((sqrt(0.3 * 0.8) + sqrt(0.7 * 0.2)) ** 2)
    assert d.total_variation(d) == pytest.approx(0.0)
    assert d.hellinger_fidelity(d) == pytest.approx(1.0)
    disjoint = SamplingDistribution([0], [1.0], 1).total_variation({(1,): 1.0})
    assert disjoint == pytest.approx(1.0)
    with pytest.raises(ValueError):
        d.total_variation({(0, 0): 1.0})


def test_distances_use_partial_mass_as_given():
    """p = {0: 0.25} is not renormalised: TV = (0.25 + 0.5) / 2 and the
    fidelity is (√(0.25 · 0.5))²; renormalising would give 0.5 for both."""
    d = SamplingDistribution([0], [0.25], 1)
    fair = {(0,): 0.5, (1,): 0.5}
    assert d.total_variation(fair) == pytest.approx(0.375)
    assert d.hellinger_fidelity(fair) == pytest.approx(0.125)


def test_sample_frequencies_sit_within_five_sigma():
    exact = _exact(_RyProduct())
    n = 10**6
    sampled = exact.sample(n, seed=0)
    assert sampled.n_shots == n
    assert int(sampled.counts().sum()) == n
    for bits in product((0, 1), repeat=5):
        p = prod(_bit_probability(q, b) for q, b in enumerate(bits))
        assert abs(sampled.get(bits, 0.0) - p) <= 5 * sqrt(p * (1 - p) / n) + 1e-12


def test_sample_renormalises_missing_mass():
    """Half the mass is missing (pruned outcomes); the draw is still 50/50."""
    n = 10**6
    sampled = SamplingDistribution([0, 1], [0.25, 0.25], 1).sample(n, seed=1)
    for bits in [(0,), (1,)]:
        assert abs(sampled[bits] - 0.5) <= 5 * sqrt(0.25 / n)


def test_sample_drops_outcomes_drawn_zero_times():
    """Outcome 1 has probability 0, so no draw can hit it."""
    sampled = SamplingDistribution([0, 1, 2], [0.5, 0.0, 0.5], 2).sample(1000, seed=0)
    assert (1, 0) not in sampled
    assert sampled.outcomes.tolist() == [0, 2]
    assert int(sampled.counts().sum()) == 1000


def test_sample_is_reproducible_and_validated():
    """Reproducibility is additional to the frequency oracle above."""
    d = _dist()
    assert d.sample(1000, seed=3) == d.sample(1000, seed=np.random.default_rng(3))
    with pytest.raises(ValueError):
        d.sample(0)
    with pytest.raises(ValueError):
        SamplingDistribution([], [], 2).sample(10)


def test_counts_and_standard_errors():
    sampled = SamplingDistribution([0, 1], [0.25, 0.75], 1, n_shots=100)
    assert sampled.counts().tolist() == [25, 75]
    np.testing.assert_allclose(sampled.standard_errors(), [sqrt(0.25 * 0.75 / 100)] * 2)
    unequal = SamplingDistribution([0, 1, 2], [0.1, 0.3, 0.6], 2, n_shots=100)
    np.testing.assert_allclose(unequal.standard_errors(), [0.03, sqrt(0.0021), sqrt(0.0024)])
    exact = SamplingDistribution([0, 1], [0.25, 0.75], 1)
    assert exact.n_shots is None
    np.testing.assert_array_equal(exact.standard_errors(), [0.0, 0.0])
    with pytest.raises(ValueError, match="no shot counts"):
        exact.counts()


def test_n_shots_is_kept_validated_and_ignored_by_equality():
    sampled = SamplingDistribution([0, 1], [0.25, 0.75], 1, n_shots=100)
    assert sampled == SamplingDistribution([0, 1], [0.25, 0.75], 1)
    assert pickle.loads(pickle.dumps(sampled)).n_shots == 100
    assert sampled.marginal([0]).n_shots == 100
    with pytest.raises(ValueError):
        SamplingDistribution([0], [1.0], 1, n_shots=-1)
