"""Smoke tests for QarpEngine: parameter coercion, gradient API, batch_run.

The VQA family (VQE / VQD / ADAPT-* / PCE / SSVQE / VFF) drives QarpEngine
via ``engine.run({Symbol(θ): val, ...})`` and ``engine.run_gradient(...)``.
These tests pin the contract those callers rely on.
"""

import numpy as np
from sympy import Symbol

import qarpx as qx
from qarp.algorithms import VQE, StateVector
from qarp.blocks import SimpleBlock
from qarp.engines import QarpEngine
from qarp.operators import QubitOperator
from qarp.optimizers import ScipyOptimizer


def _rx_ry_ansatz():
    b = SimpleBlock(1, name="rxry")
    b.rx(0, qx.Param.symbol("theta"))
    b.ry(0, qx.Param.symbol("phi"))
    b.build()
    return b


def test_engine_run_accepts_sympy_symbol_keys():
    """The VQA family builds dicts as ``{Symbol(name): value, ...}``; the
    engine must normalise the keys without forcing per-caller coercion."""
    H = QubitOperator("Z0", 1.0)
    ket = _rx_ry_ansatz()
    sv = StateVector(operator=H, ket=ket)
    eng = QarpEngine()
    eng.build([sv])

    # Sympy-keyed dict
    out_sym = eng.run({Symbol("theta"): 0.0, Symbol("phi"): 0.0})[0]
    # str-keyed dict
    out_str = eng.run({"theta": 0.0, "phi": 0.0})[0]
    assert abs(out_sym - out_str) < 1e-12
    # ⟨0|Z|0⟩ = 1
    assert abs(out_sym.real - 1.0) < 1e-12


def test_engine_run_gradient_matches_finite_difference():
    """Parameter-shift gradient must match a finite-difference reference."""
    H = QubitOperator("Z0", 1.0) + QubitOperator("X0", 0.5)
    ket = _rx_ry_ansatz()
    sv = StateVector(operator=H, ket=ket)
    eng = QarpEngine()
    eng.build([sv])

    point = {"theta": 0.3, "phi": -0.7}
    grad_ps = eng.run_gradient(point)[0]

    eps = 1e-5
    grad_fd = np.zeros(2)
    for k, sym in enumerate(("theta", "phi")):
        plus = dict(point)
        plus[sym] += eps
        minus = dict(point)
        minus[sym] -= eps
        grad_fd[k] = (eng.run(plus)[0].real - eng.run(minus)[0].real) / (2 * eps)

    # Parameter-shift is exact (same precision as run() itself); FD has
    # truncation error ~eps² so 1e-6 is comfortable.
    assert np.allclose(grad_ps, grad_fd, atol=1e-6)


def test_vqe_finds_ground_with_parameter_shift_gradient():
    """End-to-end VQE with the new analytical-gradient path."""
    H = QubitOperator("Z0", 1.0) + QubitOperator("X0", 0.5)
    ket = _rx_ry_ansatz()

    vqe = VQE(
        operator=H,
        ket=ket,
        primitive=StateVector(),
        initial_parameters=[0.1, 0.1],
        gradient=True,
        optimizer=ScipyOptimizer(method="CG"),
        verbose=False,
    )
    vqe.build()
    e_vqe, _ = vqe.run()

    exact = -np.sqrt(1.25)  # ground of [[1, 0.5],[0.5, -1]]
    e_vqe = float(e_vqe.real if hasattr(e_vqe, "real") else e_vqe)
    assert abs(e_vqe - exact) < 1e-8
