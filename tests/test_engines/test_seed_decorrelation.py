"""B2: a fixed engine seed must decorrelate shot noise across the circuits
of one run()/batch_run() fan-out (identical seeds gave every circuit the
identical random tape — byte-identical counts, correlated errors in any
multi-circuit sum) while staying reproducible call-to-call.
"""

from qarp.algorithms import Sampler
from qarp.blocks import SimpleBlock
from qarp.engines import QarpEngine


def _hadamard_sampler(n=3, shots=4096):
    b = SimpleBlock(n)
    b.h(list(range(n)))
    return Sampler(ket=b, n_shots=shots)


def test_identical_circuits_draw_distinct_tapes_in_one_run():
    s1, s2 = _hadamard_sampler(), _hadamard_sampler()
    eng = QarpEngine(seed=11)
    eng.build([s1, s2])
    d1, d2 = eng.run()
    # Byte-identical counts across the fan-out = the zero-spread illusion.
    assert dict(d1) != dict(d2)


def test_batch_run_decorrelates_across_circuits():
    s1, s2 = _hadamard_sampler(), _hadamard_sampler()
    eng = QarpEngine(seed=11)
    (per_set,) = eng.batch_run([s1, s2], [{}])
    d1, d2 = per_set
    assert dict(d1) != dict(d2)


def test_fanout_seeding_is_reproducible():
    """The per-circuit derivation must reset per call and be engine-stable."""
    eng_a = QarpEngine(seed=11)
    eng_a.build([_hadamard_sampler(), _hadamard_sampler()])
    first = [dict(d) for d in eng_a.run()]
    second = [dict(d) for d in eng_a.run()]
    assert first == second

    eng_b = QarpEngine(seed=11)
    eng_b.build([_hadamard_sampler(), _hadamard_sampler()])
    assert [dict(d) for d in eng_b.run()] == first


def test_batch_run_seeds_match_run_after_an_exact_readout():
    # An EXACT readout draws no seed in either call, so a lowered circuit and
    # a gate-path circuit after it sample the same tapes in both.
    import numpy as np

    import qarp
    from qarp.blocks import ComputationalBasisStateBlock, HnBlock, QPEBlock

    u = SimpleBlock(1, name="U")
    u.p(0, 2 * np.pi * 0.3)
    lowered = Sampler(QPEBlock(ComputationalBasisStateBlock([1]), u, 3, 1), n_shots=200)
    prims = [Sampler(HnBlock(3), n_shots=qarp.EXACT), lowered, Sampler(HnBlock(3), n_shots=50)]
    eng = QarpEngine(seed=5)
    eng.build(prims)
    assert eng._programs[id(lowered)][0] is not None and not lowered.compiled_circuits[0]
    by_run = [dict(r) for r in eng.run()]
    (by_sweep,) = eng.batch_run(prims, [{}])
    assert [dict(r) for r in by_sweep[1:]] == by_run[1:]
    # φ = 0.3 is not on the 3-bit grid: the shots spread over several bins.
    assert len(by_run[1]) > 1
