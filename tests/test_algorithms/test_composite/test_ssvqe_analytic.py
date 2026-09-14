"""SSVQE pyscf-free: analytic two-state spectrum of H = Z0 + 2·Z1 with a
shared RY⊗RY ansatz over orthogonal basis states, plus contracts and the
verbose path.
"""

import numpy as np
import pytest
from sympy import Symbol

from qarp.algorithms import SSVQE, TermwiseHadamardTest
from qarp.blocks import ComputationalBasisStateBlock, SimpleBlock
from qarp.operators import QubitOperator
from qarp.optimizers import ScipyOptimizer


def _ham():
    return QubitOperator("Z0") + QubitOperator("Z1", 2.0)


def _ansatz():
    b = SimpleBlock(2)
    b.ry(0, Symbol("a"))
    b.ry(1, Symbol("b"))
    return b.build()


def _basis(n_states):
    states = [[0, 0], [1, 0]][:n_states]
    return [ComputationalBasisStateBlock(bits).build() for bits in states]


def _exact_spectrum():
    # Independent oracle: numpy-built diagonal H (spectrum ordering-invariant).
    z = np.diag([1.0, -1.0])
    eye = np.eye(2)
    h = np.kron(eye, z) + 2.0 * np.kron(z, eye)
    return np.sort(np.linalg.eigvalsh(h))


def test_ssvqe_analytic_two_lowest_states():
    exact = _exact_spectrum()
    ssvqe = SSVQE(
        _ham(),
        _basis(2),
        _ansatz(),
        weights=[2.0, 1.0],
        initial_parameters=[0.4, 0.4],
        verbose=False,
    )
    ssvqe.build()
    energies, x = ssvqe.run()

    assert abs(np.real(energies[0]) - exact[0]) < 1e-5  # −3
    assert abs(np.real(energies[1]) - exact[1]) < 1e-5  # −1
    # Order-proof surface round-trips the raw vector.
    assert ssvqe.optimal_parameters == ssvqe.ansatz_block.parameter_map(x)


def test_ssvqe_gradient_with_sampling_primitive_matches_finite_differences():
    """Formerly refused ("Auto-differentiation not supported"); a COUNTS
    primitive now differentiates through the batched parameter shift."""
    from qarp import EXACT

    ssvqe = SSVQE(
        _ham(),
        _basis(1),
        _ansatz(),
        weights=[1.0],
        initial_parameters=[0.4, 0.4],
        gradient=True,
        verbose=False,
        primitive=TermwiseHadamardTest(n_shots=EXACT),
    )
    ssvqe.build()
    x = np.array([0.4, 0.4])
    g = ssvqe.objective_gradient(x)
    assert g.dtype == np.float64
    h = 1e-6
    for i in range(2):
        up, dn = x.copy(), x.copy()
        up[i] += h
        dn[i] -= h
        fd = (ssvqe.objective(up) - ssvqe.objective(dn)) / (2 * h)
        assert g[i] == pytest.approx(fd, abs=1e-5)


def test_ssvqe_optimal_parameters_before_run_raises():
    ssvqe = SSVQE(
        _ham(), _basis(1), _ansatz(), weights=[1.0], initial_parameters=[0.4, 0.4], verbose=False
    )
    with pytest.raises(RuntimeError, match="call run"):
        ssvqe.optimal_parameters


def test_ssvqe_verbose_build_gradient_banner(capsys):
    ssvqe = SSVQE(
        _ham(),
        _basis(1),
        _ansatz(),
        weights=[1.0],
        initial_parameters=[0.4, 0.4],
        gradient=True,
        verbose=True,
    )
    ssvqe.build()
    assert "\tGradient: analytic (default) via " in capsys.readouterr().out


def test_ssvqe_verbose_build_and_run(capsys):
    ssvqe = SSVQE(
        _ham(),
        _basis(1),
        _ansatz(),
        weights=[1.0],
        initial_parameters=[0.4, 0.4],
        verbose=True,
        # COBYLA needs headroom past its n+2 minimum budget before it
        # reports a completed iteration; too tight → zero callbacks.
        optimizer=ScipyOptimizer("COBYLA", options={"maxiter": 20}),
    )
    ssvqe.build()
    ssvqe.run()
    out = capsys.readouterr().out

    assert "SSVQE Build:" in out
    assert "\tGradient: No analytic gradients." in out
    assert "SSVQE Run:" in out
    assert "\t\tIteration\t\tEnergy\t\t\t\t  dE\t\t\t\t|step|" in out
