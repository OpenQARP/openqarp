"""ShadowProtocol / PauliShadow as a primitive: contract, guards, lifecycle.

Engine-integrated but deliberately small and seeded so the assertions are
deterministic (contract facts, not sampled convergence).
"""

from __future__ import annotations

import numpy as np
import pytest

from qarp import EXACT
from qarp.algorithms import PauliShadow
from qarp.blocks import SimpleBlock
from qarp.devices import NoiseModel
from qarp.engines import QarpEngine
from qarp.errors import CapabilityError
from qarp.operators import QubitOperator


def _bell():
    ket = SimpleBlock(2)
    ket.h(0)
    ket.cx(0, 1)
    return ket


# --- primitive contract --------------------------------------------------------


def test_build_emits_one_circuit_per_setting():
    s = PauliShadow(QubitOperator("Z0"), _bell(), n_settings=20, seed=0)
    s.build()
    assert len(s.sub_blocks) == 20


def test_engine_run_returns_a_float_list():
    s = PauliShadow(QubitOperator("Z0 Z1"), _bell(), n_settings=200, seed=0)
    eng = QarpEngine(seed=1)
    eng.build([s])
    out = eng.run()
    assert isinstance(out, list) and len(out) == 1
    assert isinstance(out[0], float)


def test_dataset_well_formed_after_run():
    s = PauliShadow(QubitOperator("Z0 Z1"), _bell(), n_settings=100, seed=0)
    eng = QarpEngine(seed=1)
    eng.build([s])
    eng.run()
    ds = s.dataset
    assert ds.n_qubits == 2
    assert ds.n_settings == 100
    assert not ds.shot_exact


def test_collected_dataset_is_inert_and_survives_rebuild():
    # F3: inertness asserted on a REAL collected campaign (base.py run()), not a
    # fabricated one — and a detached reference is immune to rebuilding the
    # primitive.  (test_dataset.py's structural check only proves the test helper
    # writes plain data; this proves run() does.)
    s = PauliShadow(QubitOperator("Z0 Z1"), _bell(), n_settings=30, seed=0)
    eng = QarpEngine(seed=1)
    eng.build([s])
    eng.run()
    ds = s.dataset  # detached handle to the collected dataset

    # inert: the collected records hold only int8 settings and int→int counts,
    # no qarpx block or circuit — so the dataset is picklable/serializable.
    for setting, counts in ds.records:
        assert isinstance(setting, np.ndarray) and setting.dtype == np.int8
        assert all(isinstance(k, int) and isinstance(v, int) for k, v in counts.items())
    before = ds.estimator().expval("Z0 Z1").value

    # rebuilding the primitive clears ITS handle (build() nulls _dataset) but the
    # already-detached dataset stays live and estimates identically.
    s.build()
    with pytest.raises(RuntimeError, match="collect one, or pass"):
        _ = s.dataset
    assert ds.estimator().expval("Z0 Z1").value == before


def test_two_paths_agree_bit_for_bit():
    H = QubitOperator("Z0 Z1") - 0.5 * QubitOperator("X0 X1") + 2.0 * QubitOperator("")
    s = PauliShadow(H, _bell(), n_settings=300, seed=5)
    eng = QarpEngine(seed=9)
    eng.build([s])
    run_value = eng.run()[0]
    core = s.dataset.estimator().expval(H).value
    assert run_value == core  # exactly one estimation core


# --- guards --------------------------------------------------------------------


def test_parameterized_ket_rejected():
    import sympy

    ket = SimpleBlock(1)
    ket.rz(0, sympy.Symbol("t"))  # leaves a free symbol
    s = PauliShadow(QubitOperator("Z0"), ket, n_settings=20, seed=0)
    with pytest.raises(ValueError, match="concrete"):
        s.build()


def test_n_settings_below_floor_rejected():
    s = PauliShadow(QubitOperator("Z0"), _bell(), n_settings=3, seed=0)
    with pytest.raises(ValueError, match="below the default batch count"):
        s.build()


def test_dataset_before_collect_raises():
    s = PauliShadow(QubitOperator("Z0"), _bell(), n_settings=20, seed=0)
    with pytest.raises(RuntimeError, match="collect one, or pass"):
        _ = s.dataset


def test_collect_mode_without_ket_raises():
    s = PauliShadow(QubitOperator("Z0"), n_settings=20, seed=0)
    with pytest.raises(ValueError, match="needs a ket"):
        s.build()


# --- reproducibility -----------------------------------------------------------


def test_same_seed_identical_settings_independent_of_engine_seed():
    a = PauliShadow(QubitOperator("Z0"), _bell(), n_settings=50, seed=123)
    b = PauliShadow(QubitOperator("Z0"), _bell(), n_settings=50, seed=123)
    a.build()
    b.build()
    assert np.array_equal(a._settings, b._settings)
    # engine seed only affects sampling, not the setting sequence
    e1, e2 = QarpEngine(seed=1), QarpEngine(seed=2)
    e1.build([a])
    e2.build([b])
    assert np.array_equal(a._settings, b._settings)


# --- shot-exact, not setting-exact --------------------------------------------


def test_shot_exact_is_not_setting_exact():
    # EXACT removes shot noise (dataset.shot_exact) but NOT the finite-ensemble
    # draw: the settings still depend on the collector seed, and the estimate is
    # never the exact true value (1.0 for <Z0 Z1> on a Bell state).
    sa = _collect_exact(seed=0)
    sb = _collect_exact(seed=2)
    assert sa.dataset.shot_exact and sb.dataset.shot_exact  # no shot noise
    assert not np.array_equal(sa._settings, sb._settings)  # but the ensemble varies
    va = sa.dataset.estimator().expval("Z0 Z1", n_batches=1).value  # mean, fine-grained
    assert va != 1.0  # a finite ensemble never lands exactly on the true value
    assert abs(va - 1.0) < 0.5  # yet converging toward it


def _collect_exact(seed):
    s = PauliShadow(QubitOperator("Z0 Z1"), _bell(), n_settings=300, seed=seed, n_shots=EXACT)
    eng = QarpEngine(seed=0)
    eng.build([s])
    eng.run()
    return s


# --- noisy-engine refusal (requires_noiseless) --------------------------------


def test_noisy_engine_refused():
    s = PauliShadow(QubitOperator("Z0"), _bell(), n_settings=20, seed=0)
    eng = QarpEngine(n_qubits=2, noise_model=NoiseModel.bit_flip(0.05), n_shots=100, seed=0)
    with pytest.raises(CapabilityError, match="noiseless"):
        eng.build([s])


def test_reuse_mode_allowed_on_noisy_engine():
    # Reuse does no measurement (pure arithmetic on stored snapshots), so a noisy
    # engine is irrelevant and must NOT be refused — unlike collect mode.
    src = PauliShadow(QubitOperator("Z0 Z1"), _bell(), n_settings=200, seed=0)
    clean = QarpEngine(seed=1)
    clean.build([src])
    clean.run()
    ds = src.dataset

    reuse = PauliShadow(QubitOperator("Z0 Z1"), dataset=ds)
    noisy = QarpEngine(n_qubits=2, noise_model=NoiseModel.bit_flip(0.05), n_shots=100, seed=0)
    noisy.build([reuse])  # must not raise, despite active noise
    assert isinstance(noisy.run()[0], float)
    # collect mode on the same noisy engine is still refused (guard intact)
    with pytest.raises(CapabilityError, match="noiseless"):
        noisy.build([PauliShadow(QubitOperator("Z0"), _bell(), n_settings=20, seed=0)])


def test_disabled_noise_allows_collection():
    s = PauliShadow(QubitOperator("Z0"), _bell(), n_settings=20, seed=0)
    eng = QarpEngine(n_qubits=2, noise_model=NoiseModel.bit_flip(0.05), n_shots=100, seed=0)
    eng.noise_model.enabled = False
    eng.build([s])  # must not raise
    assert isinstance(eng.run()[0], float)


def test_noise_re_checked_per_run_not_only_at_build():
    # The noiseless guard must re-check at run(): noise enabled AFTER a clean build
    # cannot sneak a biased campaign through the ideal inverse channel.
    s = PauliShadow(QubitOperator("Z0"), _bell(), n_settings=20, seed=0)
    eng = QarpEngine(n_qubits=2, noise_model=NoiseModel.bit_flip(0.05), n_shots=100, seed=0)
    eng.noise_model.enabled = False
    eng.build([s])  # clean at build time
    eng.noise_model.enabled = True  # turn noise on after the build
    with pytest.raises(CapabilityError, match="noiseless"):
        eng.run()


# --- memory lifecycle ----------------------------------------------------------


def test_run_does_not_clear_circuits_and_release_does():
    s = PauliShadow(QubitOperator("Z0 Z1"), _bell(), n_settings=50, seed=0)
    eng = QarpEngine(seed=1)
    eng.build([s])
    first = eng.run()[0]
    # run() left the compiled circuits intact → a second dispatch still works
    assert s.compiled_circuits
    second = eng.run()[0]
    assert isinstance(second, float)
    # the dataset survives, and release() frees the circuits in place
    _ = s.dataset
    s.release()
    assert s.sub_blocks == [] and s.compiled_circuits == []
    _ = s.dataset  # still valid after release
    assert first == first  # (sanity; first is a number)
