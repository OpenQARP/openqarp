"""Phase 4b of the pipeline property suite: optimizer agreement.

Every exported optimizer, run through the real pipeline (LayerBlock ansatz →
StateVector → QarpEngine EXACT, analytic gradient where the optimizer takes
one), must reach the same closed-form minimum: f(θ) = ⟨Z₀+Z₁⟩ = cos θ₀ +
cos θ₁, min −2 at θ = (π, π).  The analytic minimum is the oracle;
per-optimizer settings were tuned once so the tolerance is convergence
accuracy, never slack.  SPSA is deterministic here because a user-provided
gradient replaces its stochastic estimator.

A completeness guard mirrors the block-registry one: a newly exported
optimizer that is in neither the table nor EXCLUDED fails this module.
"""

import numpy as np
import pytest
from sympy import Symbol

import qarpx as qx
from qarp.algorithms import StateVector
from qarp.blocks import LayerBlock
from qarp.engines import QarpEngine
from qarp.operators import QubitOperator
from qarp.optimizers import Optimizer

pytestmark = pytest.mark.property

F_MIN = -2.0
X_MIN = np.pi

# name → (constructor, pass the analytic gradient, |f − F_MIN| tolerance).
# Settings tuned 2026-08-21 (tune probe): all reach ≤ 1e-6 except RMSProp,
# whose normalised steps oscillate near the minimum (1e-4 measured; 1e-3
# asserted).
OPTIMIZERS: dict = {
    "ScipyOptimizer": (lambda o: o.ScipyOptimizer("BFGS"), True, 1e-6),
    "RotosolveOptimizer": (lambda o: o.RotosolveOptimizer(), False, 1e-6),
    "SGDOptimizer": (lambda o: o.SGDOptimizer({"lr": 0.3, "maxiter": 400}), True, 1e-6),
    "AdamOptimizer": (lambda o: o.AdamOptimizer({"lr": 0.2, "maxiter": 400}), True, 1e-6),
    "RMSPropOptimizer": (lambda o: o.RMSPropOptimizer({"lr": 0.02, "maxiter": 3000}), True, 1e-3),
    "AdaGradOptimizer": (lambda o: o.AdaGradOptimizer({"lr": 0.5, "maxiter": 800}), True, 1e-6),
    "AdamaxOptimizer": (lambda o: o.AdamaxOptimizer({"lr": 0.2, "maxiter": 500}), True, 1e-6),
    "NadamOptimizer": (lambda o: o.NadamOptimizer({"lr": 0.2, "maxiter": 500}), True, 1e-6),
    "SPSAOptimizer": (lambda o: o.SPSAOptimizer({"spsa_a0": 1.0, "maxiter": 800}), True, 1e-6),
}

# Exported Optimizer subclasses deliberately not swept, with the reason.
EXCLUDED_OPTIMIZERS = {
    "Optimizer": "the ABC itself",
    "GradientDescentOptimizer": "abstract base — compute_update not implemented",
}


def test_optimizer_registry_is_complete():
    """Every exported Optimizer subclass is swept or excluded with a reason."""
    import qarp.optimizers as qo

    unaccounted = [
        name
        for name in dir(qo)
        if isinstance(getattr(qo, name), type)
        and issubclass(getattr(qo, name), Optimizer)
        and name not in OPTIMIZERS
        and name not in EXCLUDED_OPTIMIZERS
    ]
    assert not unaccounted, (
        f"Optimizers unregistered in the agreement sweep: {unaccounted}. "
        "Add an OPTIMIZERS entry (or EXCLUDED_OPTIMIZERS with a reason) in "
        "test_optimizer_agreement.py."
    )


def _problem():
    ket = LayerBlock(qx.GateType.Ry, 2, parameters=[Symbol("t0"), Symbol("t1")])
    ket.build()
    prim = StateVector(ket=ket, operator=QubitOperator("Z0") + QubitOperator("Z1"))
    engine = QarpEngine()
    engine.build([prim])

    def f(x):
        return float(np.real(engine.run(ket.parameter_map([float(v) for v in x]))[0]))

    def g(x):
        pm = ket.parameter_map([float(v) for v in x])
        return np.asarray(engine.run_gradient(pm)[0], dtype=float)

    return f, g


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_every_optimizer_reaches_the_analytic_minimum(name):
    import qarp.optimizers as qo

    make, use_gradient, tol = OPTIMIZERS[name]
    f, g = _problem()
    seen: list = []
    kwargs: dict = {"callback": lambda p: seen.append(np.asarray(p, dtype=float))}
    if use_gradient:
        kwargs["gradient"] = g
    result = make(qo).minimize(f, np.array([2.0, 2.6]), **kwargs)

    assert abs(result.fun - F_MIN) <= tol, f"{name}: fun={result.fun} vs {F_MIN} (tol {tol})"
    wrapped = (np.asarray(result.x, dtype=float) - X_MIN + np.pi) % (2 * np.pi) - np.pi
    assert np.all(np.abs(wrapped) < 0.2), f"{name}: x={result.x} not at the π minimum"
    assert seen and all(np.all(np.isfinite(p)) for p in seen), (
        f"{name}: callback never invoked with finite parameters"
    )
