from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from qarp import EXACT
from qarp.algorithms import SelectedPairReadout
from qarp.engines import QarpEngine
from qarp.errors import CapabilityError
from qarp.operators import QubitOperator, pair_generator
from tests.selected_pair_oracles import pair_unitary, pauli_matrix


def _readout(hamiltonian=None, rotations=((0, 1, 0.31),), **kwargs):
    return SelectedPairReadout(
        QubitOperator("Z0") if hamiltonian is None else hamiltonian,
        4,
        rotations,
        engine=QarpEngine(seed=7),
        n_shots=kwargs.pop("n_shots", EXACT),
        **kwargs,
    )


def test_exact_simulated_residual_matches_scipy_including_complex_reversal():
    h = pair_generator(0, 1) + QubitOperator("Z0", 0.4) + QubitOperator((), 1.7)
    readout = _readout(h)
    u = pair_unitary(4, readout.rotations)
    expected = u.conj().T @ pauli_matrix(h.terms, 4) @ u
    pairs = [(3, 3), (3, 12), (12, 3), (12, 12), (0, 0), (0, 1)]
    for item in readout.estimate(pairs):
        assert item.value == pytest.approx(expected[item.target, item.source], abs=1e-12)
        assert item.standard_error_real == item.standard_error_imaginary == 0
    before = dict(readout.diagnostics)
    readout.estimate(pairs)
    assert readout.diagnostics["submissions"] == before["submissions"]
    assert readout.diagnostics["measurement_settings"] == before["measurement_settings"]
    assert readout.diagnostics["unique_keys"] == 5
    assert readout.diagnostics["shots"] == 0


def test_identity_commutation_and_structural_zero_do_not_execute(monkeypatch):
    h = QubitOperator((), 2) + QubitOperator("Z0 Z1", 0.5)
    readout = _readout(h)
    monkeypatch.setattr(readout.engine, "run", lambda: pytest.fail("certified residual executed"))
    diagonal, off_diagonal = readout.estimate([(3, 3), (3, 12)])
    assert diagonal.value == 2.5
    assert diagonal.status == "exact" and not diagonal.certified_zero
    assert off_diagonal.certified_zero and off_diagonal.value == 0
    assert readout.diagnostics["measurement_settings"] == 0
    assert readout.diagnostics["unique_keys"] == 2
    with pytest.raises(FrozenInstanceError):
        diagonal.value = 0


def test_sampled_and_simulated_zero_are_never_structural_certificates(monkeypatch):
    for shots in (EXACT, 100):
        readout = _readout(n_shots=shots)
        original_run = readout.engine.run

        def zero_run():
            original_run()
            return [0j]

        monkeypatch.setattr(readout.engine, "run", zero_run)
        estimate = readout.estimate([(3, 12)])[0]
        assert estimate.value == 0
        assert not estimate.certified_zero
        assert estimate.status == ("simulated" if shots is EXACT else "sampled")
        assert readout.diagnostics["unique_keys"] == 1
        assert (12, 12) not in readout._cache


def test_finite_shots_counts_uncertainty_cache_and_seed_stream(monkeypatch):
    readout = _readout(n_shots=2000, real_only=True)
    seeds = []
    original_run = readout.engine.run

    def recorded_run():
        seeds.append(readout.engine._seed)
        return original_run()

    monkeypatch.setattr(readout.engine, "run", recorded_run)
    first = readout.estimate([(3, 3)])[0]
    second = readout.estimate([(3, 12)])[0]
    again = readout.estimate([(12, 3)])[0]
    assert first.value.real == pytest.approx(-np.cos(0.62), abs=6 * first.standard_error_real)
    assert second.value.real == pytest.approx(np.sin(0.62), abs=6 * second.standard_error_real)
    assert again.value == second.value.conjugate()
    assert seeds == [7, 100010]
    assert readout.engine._seed == 7
    counts = readout.diagnostics
    assert counts["measurement_settings"] == counts["built_circuits"] == 2
    assert counts["submissions"] == 2 and counts["shots"] == 4000
    assert counts["cache_hits"] == 1


@pytest.mark.parametrize("change", ["H", "U", "angles", "shots", "seed", "limits", "compiler"])
def test_frozen_mutation_rejected_even_on_cache_hit(change):
    readout = _readout()
    readout.estimate([(0, 0)])
    if change == "H":
        readout.hamiltonian += QubitOperator((), 1)
    elif change == "U":
        readout.U = readout.U.dagger()
    elif change == "angles":
        readout.rotations = ((0, 1, 0.4),)
    elif change == "shots":
        readout.n_shots = 200
    elif change == "seed":
        readout.engine._seed = 12
    elif change == "limits":
        readout.max_requests += 1
    else:
        readout.engine._structured = not readout.engine._structured
    with pytest.raises(ValueError, match="changed"):
        readout.estimate([(0, 0)])


def test_live_noise_is_rejected_before_cache(monkeypatch):
    from types import SimpleNamespace

    readout = _readout()
    readout.estimate([(0, 0)])
    monkeypatch.setattr(
        QarpEngine, "noise_model", property(lambda self: SimpleNamespace(enabled=True))
    )
    with pytest.raises(CapabilityError, match="noiseless"):
        readout.estimate([(0, 0)])


@pytest.mark.parametrize(
    "option,value,match",
    [
        ("max_settings", 1, "setting budget"),
        ("max_total_shots", 2, "shot budget"),
        ("max_term_visits", 1, "term-visit"),
    ],
)
def test_readout_caps_precede_execution(monkeypatch, option, value, match):
    readout = _readout(QubitOperator("Z0") + QubitOperator("X0"), n_shots=10, **{option: value})
    monkeypatch.setattr(readout.engine, "run", lambda: pytest.fail("budget exceeded before run"))
    with pytest.raises(ValueError, match=match):
        readout.estimate([(3, 3)])
    assert readout.diagnostics["submissions"] == 0


def test_request_caps_and_cache_eviction():
    readout = _readout(QubitOperator((), 2), max_requests=2, cache_max_entries=1)
    readout.estimate(
        [
            (0, 0),
            (1, 1),
        ]
    )
    assert readout.diagnostics["cache_entries"] == 1
    with pytest.raises(ValueError, match="unique-request"):
        readout.estimate([(2, 2)])
    with pytest.raises(ValueError, match="batch budget"):
        readout.estimate([(0, 0)] * 3)


def test_real_only_and_shot_validation():
    with pytest.raises(CapabilityError, match="even-Y"):
        _readout(QubitOperator("Y0"), real_only=True)
    for shots in (1, True, -1, 1.5):
        with pytest.raises(ValueError, match="n_shots"):
            _readout(n_shots=shots)


def test_large_register_certificate_does_not_execute_or_discover_numerically(monkeypatch):
    import qarp.operators._selected_pairs as algebra

    def forbidden(*args, **kwargs):
        pytest.fail("structural certificate attempted numerical support discovery")

    monkeypatch.setattr(algebra, "selected_pair_amplitudes", forbidden)
    monkeypatch.setattr(algebra, "_requested_coefficient", forbidden)
    readout = SelectedPairReadout(
        QubitOperator("X0"),
        40,
        ((0, 19, 0.2), (18, 19, -0.3)),
        engine=QarpEngine(seed=7),
        n_shots=100,
    )
    monkeypatch.setattr(readout.engine, "run", forbidden)
    assert readout.estimate([(3, 3 ^ (1 << 8))])[0].certified_zero
    assert readout.diagnostics["submissions"] == 0


def test_failed_submission_retains_budget_and_restores_seed(monkeypatch):
    readout = _readout(n_shots=10)

    def fail():
        raise RuntimeError("interrupted simulator")

    monkeypatch.setattr(readout.engine, "run", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        readout.estimate([(3, 12)])
    counts = readout.diagnostics
    assert readout.engine._seed == 7
    assert counts["submissions"] == counts["failed_submissions"] == 1
    assert counts["shots"] == 0 and counts["submitted_shots"] == 20
    assert not counts["execution_counts_complete"]
    assert counts["cache_entries"] == 0


def test_grouping_budget_rejected_before_circuit_build(monkeypatch):
    readout = _readout(pair_generator(0, 1), max_grouping_pairs=1)
    monkeypatch.setattr(readout.engine, "build", lambda *a: pytest.fail("unbounded grouping"))
    with pytest.raises(ValueError, match="grouping-pair"):
        readout.estimate([(3, 12)])
