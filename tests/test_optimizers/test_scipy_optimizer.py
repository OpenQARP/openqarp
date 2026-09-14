import numpy as np

from qarp.optimizers import ScipyOptimizer


def rosenbrock_function(x):
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def rosenbrock_gradient(x):
    jac = np.zeros_like(x)
    jac[:-1] += 200 * (x[1:] - x[:-1] ** 2) * (-2 * x[:-1]) - 2 * (1 - x[:-1])
    jac[1:] += 200 * (x[1:] - x[:-1] ** 2)
    return jac


def test_scipy_minimizer_rosen_nograd():
    optimizer = ScipyOptimizer("BFGS", options=None)
    res = optimizer.minimize(
        rosenbrock_function, np.array([0, 0], dtype=float), callback=None, gradient=None
    )
    assert np.allclose(res.x, [1, 1], rtol=1e-5, atol=1e-5)
    assert np.isclose(rosenbrock_function(res.x), 0)
    optimizer = ScipyOptimizer("L-BFGS-B", options=None)
    res = optimizer.minimize(
        rosenbrock_function, np.array([0, 0], dtype=float), callback=None, gradient=None
    )
    assert np.allclose(res.x, [1, 1], rtol=1e-5, atol=1e-5)
    assert np.isclose(rosenbrock_function(res.x), 0)


def test_scipy_minimizer_rosen_grad():
    optimizer = ScipyOptimizer("BFGS", options=None)
    res = optimizer.minimize(
        rosenbrock_function,
        np.array([0, 0], dtype=float),
        callback=None,
        gradient=rosenbrock_gradient,
    )
    assert np.allclose(res.x, [1, 1])
    assert np.isclose(rosenbrock_function(res.x), 0)
    optimizer = ScipyOptimizer("L-BFGS-B", options=None)
    res = optimizer.minimize(
        rosenbrock_function,
        np.array([0, 0], dtype=float),
        callback=None,
        gradient=rosenbrock_gradient,
    )
    assert np.allclose(res.x, [1, 1])
    assert np.isclose(rosenbrock_function(res.x), 0)
