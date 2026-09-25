"""Unit tests for the ``Sampler`` primitive.

Covers the Python logic only: ``build()`` validation + ``measured_qubits``
default, and ``run()`` converting a ``SamplingResult`` into a
:class:`~qarp.SamplingDistribution` with marginalisation over non-measured
qubits.  ``run()`` is driven with a synthetic result so the C++
simulator is not involved; a single integration test exercises the full
engine path.
"""

from itertools import product
from math import cos, prod, sin
from types import SimpleNamespace

import numpy as np
import pytest

from qarp import EXACT, ExactResult, SamplingDistribution
from qarp.algorithms import Sampler
from qarp.blocks import ComputationalBasisStateBlock, HnBlock, SimpleBlock
from qarp.engines import QarpEngine

# ── build() ─────────────────────────────────────────────────────────────


def test_build_without_ket_raises():
    s = Sampler(ket=None)
    with pytest.raises(ValueError, match="Sampler requires a ket Block"):
        s.build()


def test_build_defaults_measured_qubits_to_all():
    s = Sampler(ket=HnBlock(2))
    s.build()
    assert s.measured_qubits == [0, 1]


# ── run(): basic conversion ─────────────────────────────────────────────


def test_run_basic_single_qubit():
    sr = SimpleNamespace(counts={0: 800, 1: 200}, n_shots=1000, n_qubits=1)
    s = Sampler(ket=ComputationalBasisStateBlock(basis_state=[0]), measured_qubits=[0])
    dist = s.run([sr])
    assert dist == {(0,): pytest.approx(0.8), (1,): pytest.approx(0.2)}


# ── run(): marginalisation ──────────────────────────────────────────────


def test_run_marginalizes_over_qubit_0():
    """measured_qubits=[0] marginalises out qubit 1.

    counts (bit-encoded, qubit q at bit q):
        0b00 → 500   (q0=0)
        0b10 → 300   (q0=0)   ← q1=1 but marginalised out
        0b01 → 200   (q0=1)
    → (0,): (500+300)/1000 = 0.8 ; (1,): 200/1000 = 0.2
    """
    sr = SimpleNamespace(counts={0b00: 500, 0b10: 300, 0b01: 200}, n_shots=1000, n_qubits=2)
    s = Sampler(ket=ComputationalBasisStateBlock(basis_state=[0, 0]), measured_qubits=[0])
    dist = s.run([sr])
    assert dist == {(0,): pytest.approx(0.8), (1,): pytest.approx(0.2)}
    assert sum(dist.values()) == pytest.approx(1.0)


def test_run_marginalizes_over_qubit_1():
    """Same counts, measured_qubits=[1] marginalises the other way.

        0b00 → 500   (q1=0)
        0b10 → 300   (q1=1)
        0b01 → 200   (q1=0)
    → (0,): (500+200)/1000 = 0.7 ; (1,): 300/1000 = 0.3
    """
    sr = SimpleNamespace(counts={0b00: 500, 0b10: 300, 0b01: 200}, n_shots=1000, n_qubits=2)
    s = Sampler(ket=ComputationalBasisStateBlock(basis_state=[0, 0]), measured_qubits=[1])
    dist = s.run([sr])
    assert dist == {(0,): pytest.approx(0.7), (1,): pytest.approx(0.3)}
    assert sum(dist.values()) == pytest.approx(1.0)


# ── run(): registers wider than the int64 fast path ─────────────────────


def test_run_wide_register_beyond_int64():
    """70-qubit GHZ counts: outcomes exceed 2**63, so the vectorized int64
    packing cannot apply and the arbitrary-precision path must handle it."""
    n = 70
    sr = SimpleNamespace(counts={0: 600, (1 << n) - 1: 400}, n_shots=1000, n_qubits=n)
    s = Sampler(ket=None, measured_qubits=list(range(n)))
    dist = s.run([sr])
    assert dist == {
        (0,) * n: pytest.approx(0.6),
        (1,) * n: pytest.approx(0.4),
    }


def test_run_wide_register_marginalizes():
    """Same GHZ counts, measuring only the top qubit (index 69 ≥ bit 63)."""
    n = 70
    sr = SimpleNamespace(counts={0: 600, (1 << n) - 1: 400}, n_shots=1000, n_qubits=n)
    s = Sampler(ket=None, measured_qubits=[n - 1])
    dist = s.run([sr])
    assert dist == {(0,): pytest.approx(0.6), (1,): pytest.approx(0.4)}
    assert sum(dist.values()) == pytest.approx(1.0)


# ── canonical-class identity ────────────────────────────────────────────


def test_canonical_sampler_module():
    assert Sampler.__module__ == "qarp.algorithms._primitives.sampler"


# ── integration through the engine ──────────────────────────────────────


def test_sampler_integration_through_engine():
    """|1⟩ sampled 1000 times → essentially all mass on outcome (1,)."""
    s = Sampler(ket=ComputationalBasisStateBlock(basis_state=[1]), n_shots=1000)
    s.build()
    eng = QarpEngine(n_shots=1000, seed=42)
    eng.build([s])
    out = eng.run()[0]
    assert out[(1,)] == pytest.approx(1.0)
    assert out.get((0,), 0.0) == pytest.approx(0.0)


# ── exact distributions against analytic marginals ──────────────────────

_THETAS = (0.3, 1.1, 2.0, 0.7, 2.6, 1.6)


class _RyProduct(SimpleBlock):
    """``⊗_q Ry(θ_q)|0⟩``: qubit ``q`` reads 1 with probability ``sin²(θ_q/2)``."""

    def __init__(self, thetas):
        super().__init__(len(thetas))
        self._thetas = thetas

    def build_vanilla(self):
        for q, theta in enumerate(self._thetas):
            self.ry(q, theta)


class _Ghz(SimpleBlock):
    def __init__(self, n):
        super().__init__(n)

    def build_vanilla(self):
        self.h(0)
        for q in range(self.n_qubits - 1):
            self.cx(q, q + 1)


def _packed(bits):
    """The §1 integer of an LSB-first bit tuple."""
    return sum(b << i for i, b in enumerate(bits))


def _analytic_marginal(thetas, measured):
    one = [sin(t / 2) ** 2 for t in thetas]
    zero = [cos(t / 2) ** 2 for t in thetas]
    return {
        bits: prod(one[q] if b else zero[q] for q, b in zip(measured, bits, strict=True))
        for bits in product((0, 1), repeat=len(measured))
    }


def _run(block, n_shots, measured=None, seed=None):
    s = Sampler(ket=block, n_shots=n_shots, measured_qubits=measured)
    eng = QarpEngine(seed=seed) if n_shots is EXACT else QarpEngine(n_shots=n_shots, seed=seed)
    eng.build([s])
    return eng.run()[0]


_MEASURED = [None, [0, 2, 5], [4, 1, 3]]


@pytest.mark.parametrize("measured", _MEASURED)
def test_exact_distribution_matches_analytic_marginals(measured):
    dist = _run(_RyProduct(_THETAS), EXACT, measured)
    expected = _analytic_marginal(_THETAS, measured or list(range(len(_THETAS))))
    assert isinstance(dist, SamplingDistribution)
    assert dist.keys() == expected.keys()
    for bits, p in expected.items():
        assert dist[bits] == pytest.approx(p, abs=1e-12)


@pytest.mark.parametrize("measured", _MEASURED)
def test_exact_arrays_match_analytic_marginals_in_ascending_order(measured):
    dist = _run(_RyProduct(_THETAS), EXACT, measured)
    expected = _analytic_marginal(_THETAS, measured or list(range(len(_THETAS))))
    order = sorted(expected, key=_packed)
    assert dist.n_bits_measured == len(order[0])
    assert dist.outcomes.tolist() == [_packed(bits) for bits in order]
    np.testing.assert_allclose(dist.probabilities, [expected[b] for b in order], atol=1e-12)
    assert list(dist) == order


def test_exact_ghz_marginal_on_two_of_four_qubits():
    assert _run(_Ghz(4), EXACT, [1, 3]) == {
        (0, 0): pytest.approx(0.5, abs=1e-12),
        (1, 1): pytest.approx(0.5, abs=1e-12),
    }


def test_sampled_bell_stays_on_the_bell_support():
    n_shots = 2000
    dist = _run(_Ghz(2), n_shots, seed=0)
    assert dist.outcomes.tolist() == [0b00, 0b11]
    assert dist.probabilities.sum() == pytest.approx(1.0, abs=1e-12)
    counts = dist.probabilities * n_shots
    np.testing.assert_allclose(counts, np.round(counts), atol=1e-9)


# ── wide measured sets: lookup-table boundary and the row-wise path ─────


@pytest.mark.parametrize("width", [31, 32, 33, 40])
def test_wide_measured_set_keys_are_the_set_bits(width):
    """Outcome bits are chosen, not computed: each key lists its set bits."""
    set_bits = [(), (0, width - 1), tuple(range(width)), (1, width // 2)]
    outcomes = [sum(1 << q for q in bits) | (1 << 45) for bits in set_bits]
    order = np.argsort(outcomes)
    sr = ExactResult(
        n_qubits=50,
        keys=np.array(outcomes, dtype=np.int64)[order],
        probs=np.array([0.1, 0.2, 0.3, 0.4])[order],
    )
    dist = Sampler(ket=None, measured_qubits=list(range(width))).run([sr])
    expected = {
        tuple(1 if q in bits else 0 for q in range(width)): p
        for bits, p in zip(set_bits, (0.1, 0.2, 0.3, 0.4), strict=True)
    }
    assert dist == expected
    assert list(dist) == sorted(expected, key=_packed)


def test_run_wide_register_iterates_in_ascending_order():
    """70-qubit counts listed high-first still come back ascending."""
    n = 70
    sr = SimpleNamespace(counts={(1 << n) - 1: 400, 0: 600}, n_shots=1000, n_qubits=n)
    dist = Sampler(ket=None, measured_qubits=[n - 1, 0, 5]).run([sr])
    assert list(dist) == [(0, 0, 0), (1, 1, 1)]
    assert dist.outcomes.tolist() == [0, 0b111]
