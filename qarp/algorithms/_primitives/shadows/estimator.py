"""``ShadowEstimator`` — turn one snapshot dataset into many observable estimates.

Pure post-processing over a :class:`~.dataset.ShadowDataset` via its stateless
kernel.  This is the "estimate many from one campaign" surface; the
:class:`~.base.ShadowProtocol` primitive's own ``run()`` delegates here for its
bound operator, so the two entry points share one estimation core and agree by
construction.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence, Union

import numpy as np

from qarp.errors import CapabilityError
from qarp.operators import QubitOperator

from ._estimators import median_of_means, n_batches_for


class ShadowEstimate(NamedTuple):
    """One estimate with the HKP Theorem-1 accuracy half-width.

    ``error`` is the ``eps = sqrt(34 * V_hat / N)`` of arXiv:2002.08953 Theorem 1
    (``V_hat`` the empirical single-setting variance, ``N`` the per-batch size) —
    the half-width for which ``Pr[|value - true| >= error] <= delta`` at the
    ``delta`` that fixed ``n_batches``.  That promise holds for the *true*
    variance ``sigma^2``; ``V_hat`` is a plug-in for it, so the reported bar
    inherits whatever error that estimate carries (and HKP's own Remark that the
    constant 34 is a loose worst case).
    """

    value: float
    error: float  # HKP Theorem-1 half-width sqrt(34 * V_hat / N) at confidence delta
    delta: float
    n_batches: int
    ensemble_only: bool  # True on a shot-exact dataset (V_hat is ensemble spread only)


def _decompose(observable: Union[str, QubitOperator]) -> tuple[float, list, list[float]]:
    """Split a Hermitian observable into ``(constant, terms, real_coeffs)``.

    Mirrors ``PauliAveraging._decompose_operator`` but **fails loud** on a
    non-negligible imaginary part rather than silently taking the real part.
    The identity term becomes an exact additive constant (zero variance), never
    routed through the kernel.
    """
    if isinstance(observable, str):
        observable = QubitOperator(observable)
    constant = 0.0
    terms: list = []
    coeffs: list[float] = []
    for term, coeff in observable.terms.items():
        c = complex(coeff)
        if abs(c.imag) > 1e-10:
            raise ValueError(
                f"observable is not Hermitian: term {term} has coefficient {coeff} "
                "with a non-negligible imaginary part."
            )
        c = float(c.real)
        if not term:
            constant += c
        else:
            terms.append(term)
            coeffs.append(c)
    return constant, terms, coeffs


class ShadowEstimator:
    """Estimate observables from a collected :class:`ShadowDataset`."""

    def __init__(self, dataset):
        self._dataset = dataset

    def _validate_fits(self, terms: list) -> None:
        n = self._dataset.n_qubits
        for term in terms:
            for qubit, _ in term:
                if qubit >= n:
                    raise ValueError(
                        f"observable acts on qubit {qubit}, but the dataset has only {n} qubits."
                    )

    def _setting_values(self, terms: list, coeffs: list[float], constant: float) -> np.ndarray:
        """Per-setting estimate ``y(s) = constant + <sum_k c_k P_k>`` over the
        setting's shots (the independent MoM unit)."""
        kernel = self._dataset.kernel
        records = self._dataset.records
        ys = np.empty(len(records), dtype=float)
        for i, (setting, counts) in enumerate(records):
            total = 0.0
            acc = 0.0
            for outcome, weight in counts.items():
                snap = 0.0
                for term, c in zip(terms, coeffs, strict=True):
                    snap += c * kernel.snapshot_estimate(setting, outcome, term)
                acc += weight * snap
                total += weight
            ys[i] = constant + (acc / total if total else 0.0)
        return ys

    def expval(
        self,
        observable: Union[str, QubitOperator],
        *,
        delta: float = 0.05,
        n_batches: int | None = None,
    ) -> ShadowEstimate:
        """Median-of-means estimate of ``<observable>`` (sum-inside for a
        multi-term operator)."""
        constant, terms, coeffs = _decompose(observable)
        self._validate_fits(terms)
        nb = n_batches if n_batches is not None else n_batches_for(delta, 1)
        if not terms:  # pure constant: exact, zero variance
            return ShadowEstimate(constant, 0.0, delta, 1, self._dataset.shot_exact)
        ys = self._setting_values(terms, coeffs, constant)
        value, error = median_of_means(ys, nb)
        return ShadowEstimate(value, error, delta, nb, self._dataset.shot_exact)

    def expval_many(
        self,
        observables: Sequence[Union[str, QubitOperator]],
        *,
        delta: float = 0.05,
        n_batches: int | None = None,
        marginal: bool = False,
    ) -> list[ShadowEstimate]:
        """Estimate a family of observables from the one dataset.

        By default this is exactly Theorem 1 of HKP applied to the family: the
        batch count uses the union bound ``K = ~2 ln(2M/delta)`` over the ``M``
        supplied, so each ``error`` is the half-width for which *all* ``M``
        estimates lie within their band **jointly** with probability at least
        ``1 - delta``.  ``marginal=True`` opts out to a per-observable ``delta``
        (``K = ~2 ln(2/delta)``, no union correction).

        Looping :meth:`expval` yourself does **not** give the joint guarantee —
        use this method for a family.  And choosing observables *after* seeing
        results from the same dataset voids it: collect fresh or split.

        An explicit ``n_batches`` overrides the derived count for every estimate
        (power-user knob; it then defeats the union bound this method exists to
        apply, so the joint guarantee no longer holds at ``delta``).
        """
        m = len(observables)
        if n_batches is None:
            n_batches = n_batches_for(delta, 1 if marginal else max(1, m))
        return [self.expval(o, delta=delta, n_batches=n_batches) for o in observables]

    # --- capability-gated estimands (future ensembles) -------------------------
    #
    # These dispatch on the kernel's ``capabilities`` flag set rather than on
    # hardcoded ensemble identity (seam S3).  On a random-Pauli dataset they raise
    # CapabilityError; the matchgate / global-Clifford ensembles add the capability and
    # the implementation.

    def _require(self, capability: str, feature: str):
        if capability not in self._dataset.kernel.capabilities:
            raise CapabilityError(
                f"{feature} is not available for the {self._dataset.kernel.ensemble!r} "
                f"ensemble (needs capability {capability!r}); it arrives with a future "
                "shadow ensemble."
            )

    def one_rdm(self):
        """1-RDM (matchgate ensemble; future)."""
        self._require("rdm", "one_rdm")

    def two_rdm(self):
        """2-RDM (matchgate ensemble; future)."""
        self._require("rdm", "two_rdm")

    def purity(self):
        """Tr(ρ²) (global-Clifford ensemble; future)."""
        self._require("purity", "purity")

    def renyi2_entropy(self):
        """Rényi-2 entropy (global-Clifford ensemble; future)."""
        self._require("purity", "renyi2_entropy")

    def fidelity(self, pure_state):
        """Fidelity to a pure state (global-Clifford ensemble; future)."""
        self._require("fidelity", "fidelity")
