"""Tests for RotosolveOptimizer.

Rotosolve is coordinate descent with an analytic per-parameter update that
assumes the cost is sinusoidal in each parameter (period 2π).  These tests use
plain synthetic costs (no circuits) to pin the Python logic: basic convergence,
the deterministic symmetry-breaking restart that escapes coordinate-wise
saddles, preservation of genuine minima, flat-direction skipping, and
determinism.
"""

import numpy as np
import pytest

from qarp.optimizers import RotosolveOptimizer


def _periodic(target):
    """Separable sinusoidal cost minimised (value -len(target)) at ``target``."""
    target = np.asarray(target, dtype=float)

    def cost(x):
        return float(np.sum(-np.cos(np.asarray(x) - target)))

    return cost


# Coordinate-wise saddle at the origin: each 1D slice through (0, 0) is minimised
# at 0 (the coupling vanishes when either argument is 0), yet a joint move lowers
# the cost.  Global minimum is -2.5 at (±π/3, ±π/3).  This is the synthetic analog
# of an all-zero hardware-efficient-ansatz start.
def _coordinate_saddle(v):
    x, y = v
    return float(-np.cos(x) - np.cos(y) - 2.0 * np.sin(x) * np.sin(y))


def test_rotosolve_minimizes_separable_cost():
    target = [0.5, -1.0, 2.0]
    opt = RotosolveOptimizer(maxiter=50)
    res = opt.minimize(_periodic(target), np.zeros(3))
    assert res.fun == pytest.approx(-3.0, abs=1e-4)
    # each parameter recovered up to a 2π wrap
    diff = (np.asarray(res.x) - np.asarray(target) + np.pi) % (2 * np.pi) - np.pi
    assert np.allclose(diff, 0.0, atol=1e-3)


def test_rotosolve_escapes_coordinate_saddle():
    """The deterministic restart escapes a coordinate-wise saddle at the origin."""
    res = RotosolveOptimizer(maxiter=50, max_restarts=3).minimize(_coordinate_saddle, [0.0, 0.0])
    assert res.fun == pytest.approx(-2.5, abs=1e-3)
    assert res.fun < -2.0 + 1e-6  # strictly below the saddle value


def test_rotosolve_without_restarts_stays_stuck_at_saddle():
    """Plain coordinate descent (no restarts) cannot leave the saddle — this is
    the behaviour the hardening fixes."""
    res = RotosolveOptimizer(maxiter=50, max_restarts=0).minimize(_coordinate_saddle, [0.0, 0.0])
    assert res.fun == pytest.approx(-2.0, abs=1e-6)
    assert np.allclose(res.x, 0.0, atol=1e-6)


def test_rotosolve_preserves_genuine_minimum():
    """Starting at the true minimum, the kicks are rejected and the minimum is
    returned unchanged."""
    target = [0.3, -0.7]
    opt = RotosolveOptimizer(maxiter=50, max_restarts=3)
    res = opt.minimize(_periodic(target), np.asarray(target, dtype=float))
    assert res.fun == pytest.approx(-2.0, abs=1e-6)
    diff = (np.asarray(res.x) - np.asarray(target) + np.pi) % (2 * np.pi) - np.pi
    assert np.allclose(diff, 0.0, atol=1e-4)


def test_rotosolve_skips_flat_direction():
    """A parameter the cost does not depend on is left at its start, and the
    informative parameter is still optimised."""

    def cost(x):
        return float(-np.cos(x[1] - 0.4))  # independent of x[0]

    res = RotosolveOptimizer(maxiter=50).minimize(cost, np.array([1.234, 0.0]))
    assert res.fun == pytest.approx(-1.0, abs=1e-4)
    assert res.x[0] == pytest.approx(1.234, abs=1e-9)  # flat coord untouched


def test_rotosolve_is_deterministic():
    """No RNG in the restart — repeated runs match exactly."""
    a = RotosolveOptimizer(maxiter=50, max_restarts=3).minimize(_coordinate_saddle, [0.0, 0.0])
    b = RotosolveOptimizer(maxiter=50, max_restarts=3).minimize(_coordinate_saddle, [0.0, 0.0])
    assert np.array_equal(a.x, b.x)
    assert a.fun == b.fun


def test_rotosolve_keeps_its_class_docstring():
    """``supports_bounds`` was once placed above the docstring, which silently
    turned ``__doc__`` into ``None``."""
    assert RotosolveOptimizer.__doc__ is not None
    assert "Rotosolve" in RotosolveOptimizer.__doc__
    assert RotosolveOptimizer.supports_bounds is False
