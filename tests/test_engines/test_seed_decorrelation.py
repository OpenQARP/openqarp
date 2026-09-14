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
