"""``n_shots=qarp.EXACT`` — the ∞-shot readout limit of sampling primitives.

Pins the Phase 5 contract of the sampler/statevec refactor: EXACT feeds each
protocol's unchanged estimator the exact Born probabilities (an ``ExactResult``
with ``n_shots = 1``), so exact-protocol results agree with ``StateVector`` to
machine precision on scalar targets, and ``Sampler(n_shots=qarp.EXACT)`` IS
the exact distribution.  Distributions are pruned below ~1e-12 → compare at
~1e-10, never exactly.
"""

from copy import deepcopy

import numpy as np
import pytest

import qarp
import qarpx as qx
from qarp.algorithms import (
    CuttingPrimitive,
    HadamardTest,
    MirrorTest,
    PauliAveraging,
    Sampler,
    StateVector,
    SWAPTest,
    TermwiseHadamardTest,
)
from qarp.blocks import SimpleBlock
from qarp.devices import Device, NoiseModel, get_nearest_neighbour_architecture
from qarp.engines import QarpEngine
from qarp.errors import CapabilityError
from qarp.operators import QubitOperator

ATOL = 1e-10


def _ry_ket(theta=0.7, n=1):
    b = SimpleBlock(n, name="ry")
    b.ry(0, theta)
    b.build()
    return b


def _entangled_ket(theta=0.7):
    """cos(θ/2)|00⟩ + sin(θ/2)|11⟩."""
    b = SimpleBlock(2, name="ent")
    b.ry(0, theta)
    b.cx(0, 1)
    b.build()
    return b


# ── Sampler(EXACT) is the exact Born distribution ────────────────────────


def test_sampler_exact_equals_born_distribution():
    theta = 0.7
    samp = Sampler(ket=_entangled_ket(theta), n_shots=qarp.EXACT)
    eng = QarpEngine()
    eng.build([samp])
    dist = eng.run()[0]
    expected = {(0, 0): np.cos(theta / 2) ** 2, (1, 1): np.sin(theta / 2) ** 2}
    assert set(dist) == set(expected)
    for k, v in expected.items():
        assert dist[k] == pytest.approx(v, abs=ATOL)
    assert sum(dist.values()) == pytest.approx(1.0, abs=ATOL)


def test_sampler_exact_marginalizes_measured_qubits():
    samp = Sampler(ket=_entangled_ket(0.7), n_shots=qarp.EXACT, measured_qubits=[0])
    eng = QarpEngine()
    eng.build([samp])
    dist = eng.run()[0]
    assert dist[(0,)] == pytest.approx(np.cos(0.35) ** 2, abs=ATOL)
    assert dist[(1,)] == pytest.approx(np.sin(0.35) ** 2, abs=ATOL)


def test_sampler_exact_agrees_with_large_shot_sampling():
    samp_exact = Sampler(ket=_ry_ket(), n_shots=qarp.EXACT)
    samp_shots = Sampler(ket=_ry_ket(), n_shots=200_000)
    eng = QarpEngine(seed=3)
    eng.build([samp_exact, samp_shots])
    d_exact, d_shots = eng.run()
    for k in d_exact:
        assert d_shots.get(k, 0.0) == pytest.approx(d_exact[k], abs=5e-3)


def test_exact_is_seed_independent():
    dists = []
    for seed in (1, 99):
        eng = QarpEngine(seed=seed)
        samp = Sampler(ket=_entangled_ket(), n_shots=qarp.EXACT)
        eng.build([samp])
        dists.append(eng.run()[0])
    assert dists[0] == dists[1]  # bit-reproducible, no RNG involved


# ── exact protocol == StateVector on scalar targets ──────────────────────


def test_pauli_averaging_exact_equals_state_vector():
    H = QubitOperator("Z0", 0.75) + QubitOperator("X0", -0.4)
    pa = PauliAveraging(ket=_ry_ket(), operator=H, n_shots=qarp.EXACT)
    sv = StateVector(ket=_ry_ket(), operator=H)
    eng = QarpEngine()
    eng.build([pa, sv])
    r_pa, r_sv = eng.run()
    assert complex(r_pa) == pytest.approx(complex(r_sv), abs=ATOL)


def test_hadamard_test_exact_equals_state_vector():
    """Same Block-U target, protocol vs direct evaluation — equal at EXACT."""
    u = SimpleBlock(1, name="X")
    u.x(0)
    u.build()
    ht = HadamardTest(ket=_ry_ket(), operator=u, n_shots=qarp.EXACT)
    sv = StateVector(ket=_ry_ket(), operator=u)
    eng = QarpEngine()
    eng.build([ht, sv])
    r_ht, r_sv = eng.run()
    assert complex(r_ht) == pytest.approx(complex(r_sv), abs=ATOL)
    assert complex(r_ht) == pytest.approx(np.sin(0.7), abs=ATOL)


def test_overlap_protocols_exact_equal_state_vector():
    """SWAPTest / MirrorTest at EXACT equal |⟨bra|ket⟩|² from StateVector."""
    sw = SWAPTest(bra=_ry_ket(0.3), ket=_ry_ket(1.1), n_shots=qarp.EXACT)
    mt = MirrorTest(bra=_ry_ket(0.3), ket=_ry_ket(1.1), n_shots=qarp.EXACT)
    sv = StateVector(bra=_ry_ket(0.3), ket=_ry_ket(1.1))
    eng = QarpEngine()
    eng.build([sw, mt, sv])
    r_sw, r_mt, r_sv = eng.run()
    expected = abs(complex(r_sv)) ** 2  # = cos²((1.1−0.3)/2)
    assert expected == pytest.approx(np.cos(0.4) ** 2, abs=ATOL)
    assert r_sw == pytest.approx(expected, abs=ATOL)
    assert r_mt == pytest.approx(expected, abs=ATOL)


def test_termwise_hadamard_exact_equals_state_vector():
    """Per-term Hadamard tests at EXACT sum to the exact ⟨ψ|H|ψ⟩."""
    H = QubitOperator("Z0", 0.75) + QubitOperator("X0", -0.4)
    th = TermwiseHadamardTest(ket=_ry_ket(), operator=H, n_shots=qarp.EXACT)
    sv = StateVector(ket=_ry_ket(), operator=H)
    eng = QarpEngine()
    eng.build([th, sv])
    r_th, r_sv = eng.run()
    assert complex(r_th) == pytest.approx(complex(r_sv), abs=ATOL)


# ── resolution order: engine-wide default, per-primitive override ────────


def test_engine_wide_exact_default():
    """QarpEngine(n_shots=EXACT): primitives left at None go exact; an explicit
    int stays sampled (its distribution is quantized in units of 1/n_shots)."""
    eng = QarpEngine(n_shots=qarp.EXACT, seed=0)
    default_prim = Sampler(ket=_ry_ket())
    explicit_prim = Sampler(ket=_ry_ket(), n_shots=50)
    eng.build([default_prim, explicit_prim])
    d_exact, d_sampled = eng.run()
    assert d_exact[(0,)] == pytest.approx(np.cos(0.35) ** 2, abs=ATOL)
    for v in d_sampled.values():
        assert (v * 50) == pytest.approx(round(v * 50), abs=1e-9)


def test_batch_run_exact_override():
    """batch_run(n_shots=EXACT) sweeps exactly; matches StateVector per point."""
    b = SimpleBlock(1, name="rx")
    b.rx(0, qx.Param.symbol("theta"))
    b.build()
    pa = PauliAveraging(ket=b, operator=QubitOperator("Z0"), n_shots=123)
    eng = QarpEngine(seed=0)
    thetas = [0.0, 0.4, np.pi / 2]
    results = eng.batch_run([pa], [{"theta": t} for t in thetas], n_shots=qarp.EXACT)
    for t, per_set in zip(thetas, results, strict=True):
        assert complex(per_set[0]) == pytest.approx(np.cos(t), abs=ATOL)


def test_batch_run_honors_primitive_exact():
    samp = Sampler(ket=_ry_ket(), n_shots=qarp.EXACT)
    eng = QarpEngine(seed=0)
    results = eng.batch_run([samp], [{}])
    assert results[0][0][(0,)] == pytest.approx(np.cos(0.35) ** 2, abs=ATOL)


# ── routed devices: exact counts reindexed back to logical order ─────────


def test_exact_reindexing_on_routed_device():
    """cx(0,3) on a 2×2 grid must route; the EXACT distribution must come back
    in logical qubit order, identical to the unrouted run."""
    b = SimpleBlock(4, name="routed")
    b.ry(0, 0.7)
    b.cx(0, 3)  # 0–3 not adjacent on the 2×2 grid
    b.build()

    plain = QarpEngine()
    s1 = Sampler(ket=b, n_shots=qarp.EXACT)
    plain.build([s1])
    d_plain = plain.run()[0]

    arch = get_nearest_neighbour_architecture(2, 2)
    routed = QarpEngine(device=Device(4, architecture=arch))
    s2 = Sampler(ket=b, n_shots=qarp.EXACT)
    routed.build([s2])
    d_routed = routed.run()[0]

    assert set(d_plain) == set(d_routed)
    for k in d_plain:
        assert d_routed[k] == pytest.approx(d_plain[k], abs=ATOL)


# ── rejections ───────────────────────────────────────────────────────────


def test_cutting_primitive_rejects_exact():
    with pytest.raises(CapabilityError, match="no ∞-shot limit"):
        CuttingPrimitive(n_shots=qarp.EXACT)


def test_noisy_engine_rejects_exact_readout():
    eng = QarpEngine(n_qubits=1, noise_model=NoiseModel.bit_flip(0.05))
    samp = Sampler(ket=_ry_ket(), n_shots=qarp.EXACT)
    with pytest.raises(CapabilityError, match="exact amplitudes"):
        eng.build([samp])


def test_noisy_engine_rejects_engine_wide_exact_at_init():
    with pytest.raises(CapabilityError, match="self-contradictory"):
        QarpEngine(n_qubits=1, noise_model=NoiseModel.bit_flip(0.05), n_shots=qarp.EXACT)


def test_exact_with_true_mcm_rejected():
    b = SimpleBlock(1, name="mcm")
    b.h(0)
    b.measure(0, 0)
    b.h(0)
    b.build()
    samp = Sampler(ket=b, n_shots=qarp.EXACT)
    with pytest.raises(CapabilityError, match="mid-circuit"):
        QarpEngine().build([samp])


# ── sentinel mechanics ───────────────────────────────────────────────────


def test_exact_sentinel_survives_deepcopy():
    """QSE / MonteCarlo / QMEGS deepcopy primitives; identity must survive."""
    prim = Sampler(ket=_ry_ket(), n_shots=qarp.EXACT)
    assert deepcopy(prim).n_shots is qarp.EXACT


def test_exact_sentinel_rejects_arithmetic():
    """Plain Enum, not IntEnum: leaked-sentinel arithmetic fails loudly."""
    with pytest.raises(TypeError):
        qarp.EXACT * 2
    with pytest.raises(TypeError):
        qarp.EXACT > 0


def test_rebuild_false_keeps_routing_map():
    """build(rebuild=False) must preserve the logical→physical map — dropping
    it returned physical-qubit-ordered results on routed devices."""
    b = SimpleBlock(4, name="routed")
    b.ry(0, 0.7)
    b.cx(0, 3)
    b.build()
    arch = get_nearest_neighbour_architecture(2, 2)
    eng = QarpEngine(device=Device(4, architecture=arch))
    samp = Sampler(ket=b, n_shots=qarp.EXACT)
    eng.build([samp])
    d_first = eng.run()[0]
    eng.build([samp], rebuild=False)
    d_reused = eng.run()[0]
    assert d_first == d_reused


# ── P2.5: ExactResult carries arrays; the dict is lazy ───────────────────


def _random_ket(n, seed):
    from qarp.blocks import HEABlock

    rng = np.random.default_rng(seed)
    ket = HEABlock(n, 2, real=False, linear=True, circular=False, use_cz=False)
    ket.build()
    return ket.set_symbols(
        {s: float(v) for s, v in zip(ket.symbols, rng.normal(size=len(ket.symbols)), strict=True)}
    )


def test_exact_result_arrays_are_the_pruned_born_distribution():
    """``keys`` sorted and unique, ``probs = |ψ_k|²`` at those keys, the
    pair summing to 1 − O(1e-12); ``ψ`` from the simulator directly."""
    from qarp.engines._engine import _exact_result

    n = 6
    ket = _random_ket(n, 0)
    sim = qx.QarpSimulator()
    er = _exact_result(sim, ket.flatten(), n)
    psi = np.asarray(sim.statevector(ket.flatten(), n)).flatten()
    assert er.keys.dtype == np.int64
    assert np.all(np.diff(er.keys) > 0)
    np.testing.assert_allclose(er.probs, np.abs(psi[er.keys]) ** 2, atol=1e-14)
    assert er.probs.sum() == pytest.approx(1.0, abs=1e-12)
    assert er.n_shots == 1 and er.is_exact


def test_exact_result_counts_dict_is_lazy_and_bypassed_by_array_consumers():
    """``Sampler.run`` and ``PauliAveraging.run`` read the arrays; the dict
    is only built when something asks for ``counts`` — and then it is the
    same distribution."""
    from qarp.engines._engine import _exact_result

    n = 5
    ket = _random_ket(n, 1)
    er = _exact_result(qx.QarpSimulator(), ket.flatten(), n)
    assert "counts" not in vars(er)

    sampler = Sampler(ket=ket, n_shots=qarp.EXACT)
    sampler.build()
    dist = sampler.run([er])
    # The parity estimator expects post-Clifford outcomes; its value is
    # pinned against StateVector above — here only that it reads arrays.
    pa = PauliAveraging(ket=ket, operator=QubitOperator("Z0 Z1") + QubitOperator("Z3"))
    pa.build()
    pa.run([er] * pa.n_groups)
    assert "counts" not in vars(er)

    counts = er.counts
    assert counts == dict(zip(er.keys.tolist(), er.probs.tolist(), strict=True))
    assert sum(counts.values()) == pytest.approx(1.0, abs=1e-12)
    for bits, p in dist.items():
        key = sum(b << q for q, b in enumerate(bits))
        assert counts[key] == pytest.approx(p, abs=ATOL)


def test_reindex_exact_matches_per_key_bit_loop():
    """Random 8-bit distributions under (a) a random full permutation and
    (b) a map that drops physical bits, so logical keys collide and must
    accumulate.  Oracle: the per-key Python bit loop the vectorised version
    replaced."""
    from qarp import ExactResult
    from qarp.engines._engine import _reindex_exact

    rng = np.random.default_rng(2)
    keys = np.sort(rng.choice(256, 100, replace=False)).astype(np.int64)
    probs = rng.random(100)
    probs /= probs.sum()
    for l2p in (rng.permutation(8).tolist(), [0, 3, 5]):
        expected = {}
        for k, p in zip(keys.tolist(), probs.tolist(), strict=True):
            logical = sum(((k >> phys) & 1) << l for l, phys in enumerate(l2p))
            expected[logical] = expected.get(logical, 0.0) + p
        out = _reindex_exact(ExactResult(n_qubits=8, keys=keys, probs=probs), l2p)
        assert np.all(np.diff(out.keys) > 0)
        assert out.keys.tolist() == sorted(expected)
        np.testing.assert_allclose(out.probs, [expected[k] for k in out.keys.tolist()], atol=1e-15)
        assert "counts" not in vars(out)


def test_outcome_arrays_reads_a_sampling_result_once():
    """A finite-shot result has no arrays; ``outcome_arrays`` converts its
    ``counts`` mapping exactly once and in the mapping's own order."""
    from types import SimpleNamespace

    from qarp._types import outcome_arrays

    reads = []

    class Counts(dict):
        def keys(self):
            reads.append("keys")
            return super().keys()

    sr = SimpleNamespace(counts=Counts({5: 3, 0: 7, 2: 1}), n_qubits=3, n_shots=11)
    outcomes, weights = outcome_arrays(sr)
    assert outcomes.tolist() == [5, 0, 2] and weights.tolist() == [3.0, 7.0, 1.0]
    assert outcomes.dtype == np.int64 and weights.dtype == np.float64
    assert reads == ["keys"]


@pytest.mark.bench
def test_exact_result_costs_little_more_than_the_statevector():
    """P2.5 at 20 qubits: building the exact distribution (|ψ|², prune,
    arrays — no dict) is ≤ 3× the statevector itself.  The Sampler's
    ``{bit-tuple: p}`` output is not in this bound (see the plan)."""
    import time

    from qarp.engines._engine import _exact_result

    n = 20
    ket = _random_ket(n, 3)
    cmds = ket.flatten()
    sim = qx.QarpSimulator()
    sim.statevector(cmds, n)  # warm
    best_sv, best_er = float("inf"), float("inf")
    for _ in range(3):
        t = time.perf_counter()
        sim.statevector(cmds, n)
        best_sv = min(best_sv, time.perf_counter() - t)
        t = time.perf_counter()
        _exact_result(sim, cmds, n)
        best_er = min(best_er, time.perf_counter() - t)
    assert best_er < 3 * best_sv, (
        f"exact {best_er * 1e3:.0f} ms vs statevector {best_sv * 1e3:.0f} ms"
    )
