"""Independent oracles for the sampler track (§18).

The track compares a sampled draw to exact Born probabilities inside shot
noise, so the oracle here is the *noise model itself*: every tolerance band
`checks.agree` applies must be at least `SIGMA_K` true standard errors wide,
with the standard error taken from the exact distribution — not from a proxy.
A band narrower than that turns honest draws into nightly failures.

Usage:  python -m benchmarks.sampler.verify
"""

import math
import sys

import numpy as np

from benchmarks.sampler import _exact, checks, inputs, spec


def _exact_weight_probs(family: str) -> tuple[np.ndarray, int]:
    n, ops, shots = inputs.workload(family, spec.FAMILIES[family]["smoke"])
    probs, _ = _exact.to_weights(n, _exact.run_sample(n, _exact.build_sampler(n, ops, shots)))
    return probs / probs.sum(), shots


def _true_variance_se(probs: np.ndarray, shots: int) -> float:
    """Standard error of a sample variance: sqrt((mu4 - sigma^4) / shots).

    Textbook result for i.i.d. draws; the fourth central moment is what makes
    a distribution piled on one outcome (weight 0 here) noisier in its
    variance than `sigma^2 / sqrt(shots)` suggests.
    """
    w = np.arange(probs.size, dtype=float)
    mean = float((probs * w).sum())
    var = float((probs * (w - mean) ** 2).sum())
    mu4 = float((probs * (w - mean) ** 4).sum())
    return math.sqrt(max(mu4 - var * var, 0.0) / shots)


def check_variance_band_is_calibrated() -> list[str]:
    """`agree` must accept a draw SIGMA_K-0.5 true sigma off and reject one
    SIGMA_K+0.5 off, on every smoke cell's exact distribution."""
    failures = []
    for family in spec.FAMILIES:
        probs, shots = _exact_weight_probs(family)
        oracle = checks.weight_moments(probs, shots)
        se = _true_variance_se(probs, shots)
        var = oracle["var_weight"]
        inside = {**oracle, "var_weight": var + (checks.SIGMA_K - 0.5) * se}
        outside = {**oracle, "var_weight": var + (checks.SIGMA_K + 0.5) * se}
        if not checks.agree(inside, oracle, spec.CHECK_RTOL):
            failures.append(
                f"{family}: variance band rejects a {checks.SIGMA_K - 0.5:.1f}-sigma draw "
                f"(true SE {se:.4f})"
            )
        if checks.agree(outside, oracle, spec.CHECK_RTOL):
            failures.append(
                f"{family}: variance band accepts a {checks.SIGMA_K + 0.5:.1f}-sigma draw"
            )
    return failures


def check_incident_trotter_qulacs_draw_is_honest() -> list[str]:
    """Nightly 2026-09-13: qulacs on trotter_sample[n=4] failed the gate with
    this draw.  It sits 2.9 true sigma from exact — inside a 6-sigma gate."""
    probs, shots = _exact_weight_probs("trotter_sample")
    oracle = checks.weight_moments(probs, shots)
    observed = {
        "mean_weight": 0.28125,
        "var_weight": 0.258544921875,
        "p_zero": 0.7464599609375,
        "shots": shots,
    }
    observed = {k: observed.get(k, oracle[k]) for k in oracle}
    if not checks.agree(observed, oracle, spec.CHECK_RTOL):
        return ["trotter_sample[n=4]: the 2026-09-13 qulacs draw is rejected by the gate"]
    return []


CHECKS = (check_variance_band_is_calibrated, check_incident_trotter_qulacs_draw_is_honest)


def run_all() -> list[str]:
    failures = []
    for check in CHECKS:
        found = check()
        print(f"  {'FAIL' if found else 'ok  '} sampler/verify/{check.__name__}")
        failures += found
    return failures


if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)
