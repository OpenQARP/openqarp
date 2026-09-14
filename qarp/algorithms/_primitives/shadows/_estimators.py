"""Median-of-means machinery shared by the shadow primitive and estimator.

Median-of-means (MoM) is the estimator of the classical-shadow literature: the
single-snapshot estimates are unbiased but heavy-tailed, so a plain mean is
fragile.  Splitting the *settings* (the independent random unit) into batches,
averaging within each, and taking the median across batches concentrates the
estimate exponentially in the batch count even under heavy tails.

The batch count derives from a confidence target ``delta`` (``~2 ln(2M/delta)``,
HKP); the partition is **round-robin by setting index** (``setting i -> batch
i % n_batches``) so every batch is a representative cross-section — this keeps a
merged campaign's batches evenly mixed rather than campaign-segregated.  (HKP
Eq. (S17) partitions into *contiguous* blocks; for a single exchangeable campaign
the two are statistically identical, and round-robin is the choice that also
serves merged datasets.)

The reported ``error`` is exactly the accuracy that Theorem 1 of Huang, Kueng &
Preskill (arXiv:2002.08953) controls — not an ad-hoc band.  The proof of that
theorem states: for a variable of variance ``sigma^2``, ``K`` sample means of
size ``N = 34 sigma^2 / eps^2`` give a median-of-means estimator obeying
``Pr[|mu_hat - E| >= eps] <= 2 e^{-K/2}``.  Inverting for the ``eps`` a collected
dataset actually achieves — with ``V_hat`` the empirical single-setting variance
(the natural estimator of ``Var[o_hat]``, the very ``sigma^2`` of the theorem),
``N`` the per-batch size, and ``K = n_batches`` fixed by ``delta`` via
``2 e^{-K/2} = delta`` — gives the half-width

    eps = sqrt(34 * V_hat / N),

for which ``Pr[|o_hat - tr(O rho)| >= eps] <= delta``.  It is the paper's own
constant and its own confidence level.

That promise attaches to the *true* single-setting variance ``sigma^2``; what
ships substitutes the empirical ``V_hat`` for it, so the reported half-width is a
plug-in of a certified bound rather than itself certified.  (The theorem is
usually quoted against the shadow norm ``||O||^2_shadow = max_rho E[o_hat^2]``,
which upper-bounds ``Var[o_hat]`` uniformly over states and so fixes the sample
budget a priori.  The MoM concentration itself runs on the variance at the actual
state, which is what ``V_hat`` estimates; a state-independent shadow-norm bound
is future work.)  HKP's own Remark notes the constant 34 is a loose worst case.
"""

from __future__ import annotations

import math

import numpy as np


def n_batches_for(delta: float, n_observables: int = 1) -> int:
    """HKP batch count for failure probability ``delta`` over ``n_observables``.

    ``~2 ln(2M/delta)``; the ``M`` factor is the union-bound correction applied
    when a whole family is estimated together (``expval_many``).  Clamped to at
    least 1.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if n_observables < 1:
        raise ValueError(f"n_observables must be >= 1, got {n_observables}")
    return max(1, math.ceil(2.0 * math.log(2.0 * n_observables / delta)))


def median_of_means(values: np.ndarray, n_batches: int) -> tuple[float, float]:
    """Median-of-means point estimate and the HKP Theorem-1 accuracy half-width.

    Args:
        values: per-setting estimates ``y(s)`` (the independent unit is the
            setting; never split by shot).
        n_batches: number of batches ``K``; ``1`` reduces to the plain mean.

    Returns:
        ``(value, error)`` — the median-of-means value and the half-width
        ``eps = sqrt(34 * V_hat / N)`` (module docstring), with ``V_hat`` the
        empirical single-setting variance and ``N`` the per-batch size
        (``n_settings // n_batches``).  This is exactly the ``eps`` that
        Theorem 1 of arXiv:2002.08953 controls at confidence ``delta`` (with
        ``delta`` tied to ``n_batches`` by ``2 e^{-K/2} = delta``), evaluated
        with the empirical ``V_hat`` in place of the true variance ``sigma^2``
        that the theorem's promise actually attaches to.

    Raises:
        ValueError: if ``n_batches`` exceeds the number of settings (a
            degenerate partition with empty/size-1 batches).
    """
    y = np.asarray(values, dtype=float)
    n = y.size
    if n == 0:
        raise ValueError("median_of_means: no settings to estimate from")
    if n_batches < 1:
        raise ValueError(f"n_batches must be >= 1, got {n_batches}")
    if n_batches > n:
        raise ValueError(
            f"n_batches ({n_batches}) exceeds the number of settings ({n}); "
            "collect more settings or lower n_batches / raise delta."
        )
    # Round-robin partition: batch b holds y[b], y[b+n_batches], ... — a
    # representative cross-section, so a merged A+B campaign stays evenly mixed.
    batch_means = np.array([y[b::n_batches].mean() for b in range(n_batches)])
    value = float(np.median(batch_means))
    # eps = sqrt(34 * Var[o_hat] / N) inverted from HKP's N = 34 sigma^2/eps^2.
    # V_hat is the empirical single-setting variance (estimator of Var[o_hat]);
    # N is the guaranteed minimum batch size, floor(n / K).
    if n < 2:
        error = 0.0
    else:
        v_hat = float(np.var(y, ddof=1))
        n_per_batch = n // n_batches
        error = math.sqrt(34.0 * v_hat / n_per_batch)
    return value, error
