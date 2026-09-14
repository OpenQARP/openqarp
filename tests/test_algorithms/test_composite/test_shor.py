"""Independent mathematical and lifecycle tests for Shor factoring."""

import pytest

from qarp import EXACT
from qarp.algorithms import Sampler, Shor, StateVector
from qarp.algorithms._composite.shor import (
    _factor_pair_from_order,
    _recover_order,
)
from qarp.endianness import label_to_bits


def _phase_distribution(labels: list[int], n_counting: int) -> dict[tuple[int, ...], float]:
    probability = 1.0 / len(labels)
    return {tuple(label_to_bits(label, n_counting)): probability for label in labels}


class _SpyEngine:
    def __init__(self, results=None):
        self.build_calls = 0
        self.run_calls = 0
        self.primitives = []
        self.results = results or []

    def build(self, primitives):
        self.build_calls += 1
        self.primitives = list(primitives)

    def run(self):
        self.run_calls += 1
        return list(self.results)


@pytest.mark.parametrize(
    ("base", "modulus", "n_counting", "labels", "expected"),
    [
        (4, 15, 8, [0, 128], 2),
        (2, 15, 8, [0, 64, 128, 192], 4),
        (2, 21, 10, [341, 512], 6),
    ],
)
def test_continued_fractions_recover_validated_orders(base, modulus, n_counting, labels, expected):
    distribution = _phase_distribution(labels, n_counting)
    assert _recover_order(base, modulus, distribution, n_counting) == expected


def test_order_recovery_combines_reduced_denominators_by_lcm():
    n_counting = 10
    one_third = _phase_distribution([341], n_counting)
    one_half = _phase_distribution([512], n_counting)

    assert _recover_order(2, 21, one_third, n_counting) is None
    assert _recover_order(2, 21, one_half, n_counting) is None
    combined = {**one_third, **one_half}
    assert _recover_order(2, 21, combined, n_counting) == 6


def test_inconclusive_distribution_is_rejected_without_brute_force():
    distribution = _phase_distribution([0, 1], 8)
    assert _recover_order(2, 15, distribution, 8) is None


def test_exact_end_to_end_quantum_path_factors_fifteen():
    algorithm = Shor(
        15,
        base=2,
        max_attempts=1,
        primitive=Sampler(n_shots=EXACT),
    ).build()

    assert algorithm.run() == (3, 5)
    assert algorithm.attempted_bases == [2]
    assert algorithm.orders == {2: 4}
    assert set(algorithm.distributions) == {2}
    assert len(algorithm.blocks) == 1


def test_order_six_factors_twenty_one_with_production_postprocessor():
    distribution = _phase_distribution([341, 512], 10)
    order = _recover_order(2, 21, distribution, 10)
    assert order == 6
    assert _factor_pair_from_order(2, 21, order) == (3, 7)


@pytest.mark.parametrize(
    ("number", "base", "expected"),
    [(10, None, (2, 5)), (9, None, (3, 3)), (15, 3, (3, 5)), (55, 5, (5, 11))],
)
def test_classical_shortcuts_do_not_execute_the_engine(number, base, expected):
    engine = _SpyEngine()
    algorithm = Shor(number, base=base, engine=engine).build()

    # build() prepares only; the classical answer appears at run().
    assert algorithm.result is None
    assert algorithm.run() == expected
    assert algorithm.blocks == []
    assert engine.build_calls == 0
    assert engine.run_calls == 0


def test_coprime_supplied_base_reaches_the_quantum_path_with_default_attempts():
    # Regression: the original sieve gcd-checked every pre-selected base first,
    # so a supplied coprime base was skipped whenever a later draw shared a factor.
    engine = _SpyEngine()
    algorithm = Shor(15, base=2, base_seed=0, engine=engine).build()

    assert algorithm.attempted_bases[0] == 2
    assert algorithm.blocks[0].name == "OrderFinding(2, 15)"
    assert engine.build_calls == 1

    engine.results = [_phase_distribution([0, 64, 128, 192], 8)] + [
        _phase_distribution([0], 8) for _ in algorithm.blocks[1:]
    ]
    assert algorithm.run() == (3, 5)
    assert algorithm.orders == {2: 4}


def test_attempt_order_falls_back_to_gcd_only_after_quantum_attempts(monkeypatch):
    # The attempt list is pinned directly: random.sample's draw order is not a
    # cross-version guarantee.  ord_15(14) = 2 with 14 == -1 mod 15 is
    # inconclusive, so gcd(3, 15) decides.
    monkeypatch.setattr(Shor, "_candidate_bases", lambda self: [14, 3])
    engine = _SpyEngine(results=[_phase_distribution([0, 128], 8)])
    algorithm = Shor(15, base=14, engine=engine).build()

    assert algorithm.attempted_bases == [14, 3]
    assert algorithm.classical_fallback == (3, 5)
    assert len(algorithm.blocks) == 1
    assert algorithm.run() == (3, 5)
    assert algorithm.orders == {14: 2}
    assert engine.run_calls == 1


def test_attempt_order_prefers_a_conclusive_quantum_result_over_the_fallback(monkeypatch):
    # Base 3 follows the supplied 2; the quantum result for 2 wins.
    monkeypatch.setattr(Shor, "_candidate_bases", lambda self: [2, 3])
    engine = _SpyEngine(results=[_phase_distribution([0, 64, 128, 192], 8)])
    algorithm = Shor(15, base=2, engine=engine).build()

    assert algorithm.attempted_bases == [2, 3]
    assert algorithm.classical_fallback == (3, 5)
    assert algorithm.run() == (3, 5)
    assert algorithm.orders == {2: 4}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"number": 10}, "even or a perfect power"),
        ({"number": 27}, "even or a perfect power"),
        ({"number": 15, "base": 3}, "coprime base"),
    ],
)
def test_force_quantum_rejects_inputs_outside_the_theorem(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Shor(force_quantum=True, engine=_SpyEngine(), **kwargs).build()


def test_force_quantum_draws_only_coprime_bases_and_builds_once():
    from math import gcd

    for seed in range(200):
        bases = Shor(21, max_attempts=8, base_seed=seed, force_quantum=True)._candidate_bases()
        assert all(gcd(base, 21) == 1 for base in bases)
        assert len(bases) == len(set(bases)) == 8

    engine = _SpyEngine()
    algorithm = Shor(15, max_attempts=8, base_seed=0, force_quantum=True, engine=engine).build()
    assert algorithm.result is None
    assert algorithm.classical_fallback is None
    assert len(algorithm.blocks) == 7  # phi(15) - 1 coprime bases in [2, 15)
    assert engine.build_calls == 1

    engine.results = [_phase_distribution([0], 8)] * 7
    assert algorithm.run() is None
    assert engine.run_calls == 1


def test_force_quantum_exact_end_to_end_factors_fifteen():
    algorithm = Shor(
        15,
        base=2,
        force_quantum=True,
        primitive=Sampler(n_shots=EXACT),
    ).build()

    assert algorithm.run() == (3, 5)
    assert algorithm.orders[2] == 4
    assert algorithm.classical_fallback is None


def test_inconclusive_base_permits_the_next_configured_attempt(monkeypatch):
    # Both bases are coprime, as force_quantum requires of the draw.
    monkeypatch.setattr(Shor, "_candidate_bases", lambda self: [14, 2])
    n_counting = 8
    engine = _SpyEngine(
        results=[
            _phase_distribution([0, 128], n_counting),
            _phase_distribution([0, 64, 128, 192], n_counting),
        ]
    )
    algorithm = Shor(
        15,
        base=14,
        n_counting_qubits=n_counting,
        max_attempts=2,
        force_quantum=True,
        engine=engine,
    ).build()

    assert algorithm.attempted_bases == [14, 2]
    assert algorithm.run() == (3, 5)
    assert algorithm.orders == {14: 2, 2: 4}
    assert engine.run_calls == 1


def test_odd_order_minus_one_and_missing_evidence_are_inconclusive():
    assert _factor_pair_from_order(2, 7, 3) is None
    assert _factor_pair_from_order(14, 15, 2) is None
    assert _recover_order(2, 15, _phase_distribution([0], 8), 8) is None


def test_seeded_base_selection_is_reproducible_distinct_and_local():
    first = Shor(15, max_attempts=8, base_seed=7)._candidate_bases()
    second = Shor(15, max_attempts=8, base_seed=7)._candidate_bases()
    different = Shor(15, max_attempts=8, base_seed=8)._candidate_bases()

    assert first == second
    assert first != different
    assert len(first) == len(set(first)) == 8


def test_sampler_is_private_and_per_attempt_copies_are_isolated():
    caller = Sampler(n_shots=EXACT)
    engine = _SpyEngine()
    algorithm = Shor(15, base=2, max_attempts=1, primitive=caller, engine=engine).build()

    assert algorithm.primitive is not caller
    assert algorithm._samplers[0] is not algorithm.primitive
    algorithm._samplers[0].measured_qubits.append(99)
    assert algorithm.primitive.measured_qubits is None
    assert caller.ket is None
    assert caller.measured_qubits is None


def test_inconclusive_run_is_cached_and_never_re_executes_the_engine():
    engine = _SpyEngine(results=[_phase_distribution([0], 8)])
    algorithm = Shor(15, base=2, max_attempts=1, engine=engine).build()

    assert algorithm.run() is None
    assert algorithm.run() is None
    assert engine.run_calls == 1


def test_build_is_idempotent_and_run_before_build_fails():
    engine = _SpyEngine(results=[_phase_distribution([0], 8)])
    algorithm = Shor(15, base=2, max_attempts=1, engine=engine)
    with pytest.raises(ValueError, match="Call build"):
        algorithm.run()

    assert algorithm.build() is algorithm
    assert algorithm.build() is algorithm
    assert engine.build_calls == 1


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"number": True}, TypeError, "number"),
        ({"number": 1.5}, TypeError, "number"),
        ({"number": 1}, ValueError, "greater than one"),
        ({"number": 15, "base": True}, TypeError, "base"),
        ({"number": 15, "base": 1}, ValueError, "1 < base"),
        ({"number": 15, "base": 15}, ValueError, "1 < base"),
        ({"number": 15, "max_attempts": True}, TypeError, "max_attempts"),
        ({"number": 15, "max_attempts": 0}, ValueError, "positive"),
        ({"number": 15, "base_seed": True}, TypeError, "base_seed"),
        ({"number": 15, "force_quantum": 1}, TypeError, "force_quantum"),
        ({"number": 65}, ValueError, "at most 6 work qubits"),
        ({"number": 2**60 + 1}, ValueError, "at most 6 work qubits"),
        (
            {"number": 15, "n_counting_qubits": 7},
            ValueError,
            "at least twice",
        ),
    ],
)
def test_invalid_constructor_inputs(kwargs, error, message):
    with pytest.raises(error, match=message):
        Shor(**kwargs)


def test_non_sampler_primitive_is_rejected():
    with pytest.raises(TypeError, match="Sampler primitive"):
        Shor(15, primitive=StateVector())


def test_prime_input_fails_at_build():
    with pytest.raises(ValueError, match="composite number"):
        Shor(13).build()
