"""Unit tests for the ``Sampler`` primitive.

Covers the Python logic only: ``build()`` validation + ``measured_qubits``
default, and ``run()`` converting a ``SamplingResult`` into a
``{bitstring-tuple: probability}`` distribution with marginalisation over
non-measured qubits.  ``run()`` is driven with a synthetic result so the C++
simulator is not involved; a single integration test exercises the full
engine path.
"""

from types import SimpleNamespace

import pytest

from qarp.algorithms import Sampler
from qarp.blocks import ComputationalBasisStateBlock, HnBlock
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
