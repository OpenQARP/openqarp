"""The jax recipe from ``docs/source/gradients.rst``, executed.

qarp has no jax dependency: the recipe is a ``jax.custom_vjp`` the caller
owns, whose backward rule is the cotangent times ``run_gradient``'s Jacobian.
Oracles: ``run_gradient`` itself for ``jax.grad(E)``, the closed-form chain
rule ``2·E·∂E`` for ``jax.grad(E²)``, and central finite differences of the
squared overlap for the OVERLAP branch (nightly ``[integrations]`` job).
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

import qarpx as qx  # noqa: E402
from qarp.algorithms import StateVector  # noqa: E402
from qarp.blocks import ComputationalBasisStateBlock, SimpleBlock  # noqa: E402
from qarp.engines import QarpEngine  # noqa: E402

jax.config.update("jax_enable_x64", True)


def _ansatz():
    b = SimpleBlock(2)
    b.ry(0, qx.Param.symbol("a"))
    b.rx(1, qx.Param.symbol("b"))
    b.cx(0, 1)
    b.ry(1, qx.Param.symbol("c"))
    b.build()
    return b


def make_energy(engine, block, *, square_overlap=False):
    """The documented recipe: forward = run(), backward = ct @ J."""

    def _value(x):
        v = engine.run(block.parameter_map(np.asarray(x)))[0]
        return float(abs(v) ** 2) if square_overlap else float(np.real(v))

    @jax.custom_vjp
    def energy(x):
        return jnp.asarray(_value(x))

    def fwd(x):
        return jnp.asarray(_value(x)), x

    def bwd(x, ct):
        jac = np.real(engine.run_gradient(block.parameter_map(np.asarray(x)))[0])
        return (ct * jnp.asarray(jac),)

    energy.defvjp(fwd, bwd)
    return energy


def test_jax_grad_equals_run_gradient_and_composes():
    block = _ansatz()
    engine = QarpEngine()
    engine.build(
        [StateVector(ket=block, operator=qx.QubitOperator("Z0") + qx.QubitOperator("X1", 0.5))]
    )
    energy = make_energy(engine, block)
    x = jnp.array([0.4, -0.7, 1.1])

    g = np.asarray(jax.grad(energy)(x))
    ref = engine.run_gradient(block.parameter_map(np.asarray(x)))[0]
    assert np.allclose(g, ref, atol=1e-12)

    e = float(energy(x))
    g_sq = np.asarray(jax.grad(lambda y: energy(y) ** 2)(x))
    assert np.allclose(g_sq, 2.0 * e * ref, atol=1e-12)


def test_jax_grad_of_overlap_recipe_squares_the_forward():
    """run() returns ⟨bra|ket⟩ while run_gradient differentiates |⟨bra|ket⟩|²:
    the recipe's forward must square, or the pair is inconsistent."""
    block = _ansatz()
    engine = QarpEngine()
    engine.build([StateVector(ket=block, bra=ComputationalBasisStateBlock(basis_state=[0, 0]))])
    energy = make_energy(engine, block, square_overlap=True)
    x = np.array([0.4, -0.7, 1.1])

    g = np.asarray(jax.grad(energy)(jnp.asarray(x)))
    h = 1e-6
    for k in range(3):
        up, dn = x.copy(), x.copy()
        up[k] += h
        dn[k] -= h
        fd = (
            abs(engine.run(block.parameter_map(up))[0]) ** 2
            - abs(engine.run(block.parameter_map(dn))[0]) ** 2
        ) / (2 * h)
        assert g[k] == pytest.approx(fd, abs=1e-6)
