"""Median-of-means machinery, the observable contract, and estimator gating.

All deterministic — synthetic datasets and hand-built values, no engine.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qarp.algorithms import PauliKernel, ShadowDataset
from qarp.algorithms._primitives.shadows._estimators import median_of_means, n_batches_for
from qarp.algorithms._primitives.shadows.estimator import _decompose
from qarp.errors import CapabilityError
from qarp.operators import QubitOperator

# --- n_batches derivation ------------------------------------------------------


def test_n_batches_derivation():
    assert n_batches_for(0.05, 1) == math.ceil(2 * math.log(2 / 0.05))  # == 8
    assert n_batches_for(0.05, 10) == math.ceil(2 * math.log(2 * 10 / 0.05))  # union bound
    assert n_batches_for(0.05, 10) > n_batches_for(0.05, 1)
    assert n_batches_for(0.5, 1) >= 1


# --- median-of-means ------------------------------------------------------------


def test_n_batches_1_is_the_mean():
    y = np.array([1.0, 2.0, 3.0, 10.0])
    value, _ = median_of_means(y, 1)
    assert value == pytest.approx(y.mean())


def test_median_of_means_is_outlier_robust():
    # 99 good values, one wild outlier: MoM ignores it, the mean does not.
    y = np.concatenate([np.ones(99), [1000.0]])
    mom, _ = median_of_means(y, 9)
    assert abs(mom - 1.0) < 1e-9
    assert abs(y.mean() - 1.0) > 5  # the plain mean is wrecked


def test_round_robin_partition_interleaves_a_merged_campaign():
    # Merged campaign: A = 6 settings of 0 appended with B = 3 settings of 9.
    # Round-robin (setting i -> batch i % K) puts one B into EVERY batch, so all
    # three batch means are 3 -> median 3.  A CONTIGUOUS partition would segregate
    # A and B: batches [0,0,0], [0,0,0], [9,9,9] -> median 0.  The two rules give
    # DIFFERENT medians here, so this test fails if the partition were contiguous
    # (the old all-0-then-all-2 pattern gave 1.0 under both and could not tell them
    # apart — the very property the row exists to pin).
    y = np.concatenate([np.zeros(6), np.full(3, 9.0)])
    k = 3
    round_robin = float(np.median([y[b::k].mean() for b in range(k)]))
    contiguous = float(np.median([y[b * 3 : b * 3 + 3].mean() for b in range(k)]))
    assert round_robin == 3.0 and contiguous == 0.0  # the pattern truly distinguishes
    value, _ = median_of_means(y, k)
    assert value == pytest.approx(round_robin)
    assert value != pytest.approx(contiguous)


def test_n_batches_exceeding_settings_raises():
    with pytest.raises(ValueError):
        median_of_means(np.ones(5), 6)


def test_error_is_hkp_theorem1_halfwidth():
    # Oracle: HKP (arXiv:2002.08953) Thm 1 half-width eps = sqrt(34 * V_hat / N),
    # V_hat the single-setting sample variance (ddof=1), N = n_settings // K.
    # Computed here from the paper's closed form, independent of the code path.
    rng = np.random.default_rng(0)
    y = rng.normal(size=120)
    k = 8
    _, error = median_of_means(y, k)
    expected = math.sqrt(34.0 * float(np.var(y, ddof=1)) / (120 // k))
    assert error == pytest.approx(expected)
    # A single setting has no variance estimate → reported error is 0.0.
    assert median_of_means(np.array([2.0]), 1)[1] == 0.0


# --- observable contract (_decompose) ------------------------------------------


def test_decompose_signs_identity_and_empty():
    const, terms, coeffs = _decompose(-1.0 * QubitOperator("Z0") + 0.5 * QubitOperator("X1"))
    assert const == 0.0
    assert len(terms) == 2 and len(coeffs) == 2
    assert -1.0 in coeffs and 0.5 in coeffs  # sign preserved (not abs)

    const2, terms2, _ = _decompose(2.0 * QubitOperator("") + QubitOperator("Z0"))
    assert const2 == 2.0 and len(terms2) == 1  # identity → exact constant

    const3, terms3, coeffs3 = _decompose(QubitOperator())  # empty operator
    assert const3 == 0.0 and terms3 == [] and coeffs3 == []


def test_decompose_non_hermitian_raises():
    with pytest.raises(ValueError, match="not Hermitian"):
        _decompose(1j * QubitOperator("Z0"))


# --- estimator aggregation on a synthetic (biased) dataset ---------------------
#
# All settings measure Z with outcome 0 → each single-Z snapshot is +3.  These
# records are NOT Born-sampled, so the *values* are not physical expectations —
# the point is to lock the estimator's aggregation and identity handling, which
# are deterministic.


def _all_z_zero_dataset(n_qubits=2, n_settings=50):
    setting = np.full(n_qubits, 2, dtype=np.int8)  # all Z
    records = [(setting.copy(), {0: 1}) for _ in range(n_settings)]
    return ShadowDataset(PauliKernel(n_qubits), n_qubits, records, shot_exact=False)


def test_expval_aggregation_and_identity():
    est = _all_z_zero_dataset().estimator()
    r = est.expval("Z0")
    assert r.value == pytest.approx(3.0)
    assert r.error == pytest.approx(0.0)  # all per-setting values equal
    # identity constant adds exactly, sign preserved
    r2 = est.expval(2.0 * QubitOperator("") - 1.0 * QubitOperator("Z0"))
    assert r2.value == pytest.approx(2.0 - 3.0)


def test_multi_shot_within_setting_is_weighted_averaged():
    # n_shots>1 gives a setting several outcomes as {outcome: count}; the estimator
    # averages the per-shot snapshots weighted by count.  Single-shot campaigns
    # (every other test) never exercise this weighting.
    zz = np.array([2, 2], dtype=np.int8)  # both Z
    # 3 shots -> outcome 00 (Z0 = +3), 1 shot -> 01 (Z0 = -3): weighted mean 1.5
    ds = ShadowDataset(PauliKernel(2), 2, [(zz, {0b00: 3, 0b01: 1})], shot_exact=False)
    assert ds.estimator().expval("Z0", n_batches=1).value == pytest.approx((3 * 3 - 3) / 4)


def test_expval_many_values_match_expval_at_joint_batches():
    # expval_many must actually estimate each observable at the union-bound batch
    # count — not merely report the count.  Pin the values, not just n_batches.
    est = _all_z_zero_dataset(n_settings=200).estimator()
    obs = [QubitOperator("Z0"), QubitOperator("Z1")]
    joint = est.expval_many(obs, delta=0.05)
    k = n_batches_for(0.05, len(obs))
    for o, r in zip(obs, joint, strict=True):
        direct = est.expval(o, n_batches=k)
        assert r.value == direct.value and r.error == direct.error


def test_expval_pure_constant_is_exact():
    est = _all_z_zero_dataset().estimator()
    r = est.expval(5.0 * QubitOperator(""))
    assert r.value == pytest.approx(5.0)
    assert r.error == 0.0


def test_expval_many_union_bound_batches():
    est = _all_z_zero_dataset(n_settings=200).estimator()
    obs = [QubitOperator("Z0"), QubitOperator("Z1")]
    joint = est.expval_many(obs, delta=0.05)
    marginal = est.expval_many(obs, delta=0.05, marginal=True)
    assert joint[0].n_batches == n_batches_for(0.05, len(obs))
    assert marginal[0].n_batches == n_batches_for(0.05, 1)
    assert joint[0].n_batches > marginal[0].n_batches


def test_observable_wider_than_dataset_raises():
    est = _all_z_zero_dataset(n_qubits=2).estimator()
    with pytest.raises(ValueError, match="only 2 qubits"):
        est.expval("Z9")


def test_future_capabilities_raise_on_pauli_dataset():
    est = _all_z_zero_dataset().estimator()
    for call in (est.one_rdm, est.two_rdm, est.purity, est.renyi2_entropy):
        with pytest.raises(CapabilityError):
            call()
    with pytest.raises(CapabilityError):
        est.fidelity(np.zeros(4))
