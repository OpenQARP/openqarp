"""Gradient-ordering contract.

``Engine.run_gradient`` returns arrays ordered by the params mapping's
insertion order (``engine.py: symbol_names = list(params.keys())``) — the
contract every variational algorithm's ``objective_gradient`` relies on when
zipping gradients against its own parameter vector.  Pinned here for both
the adjoint-backprop and the parameter-shift branches (both allocate per ``symbol_names`` pre-branch).
"""

import numpy as np
import pytest
from sympy import Symbol

import qarpx as qx
from qarp.algorithms import StateVector
from qarp.engines import QarpEngine
from qarp.operators import QubitOperator


def _two_param_ket():
    from qarp.blocks import SimpleBlock

    b = SimpleBlock(2)
    b.rx(0, qx.Param.symbol("a"))
    b.ry(1, qx.Param.symbol("b"))
    b.cx(0, 1)
    b.build()
    return b


def _ev_primitive():
    ket = _two_param_ket()
    prim = StateVector()
    prim.ket = ket
    prim.bra = ket
    prim.operator = QubitOperator("Z0") + QubitOperator("X1", 0.5)
    return prim


def _finite_difference(engine, params, sym, h=1e-6):
    up = dict(params)
    up[sym] = params[sym] + h
    down = dict(params)
    down[sym] = params[sym] - h
    return (np.real(engine.run(up)[0]) - np.real(engine.run(down)[0])) / (2 * h)


def test_gradient_follows_params_insertion_order_adjoint():
    engine = QarpEngine()
    engine.build([_ev_primitive()])

    p_ab = {Symbol("a"): 0.3, Symbol("b"): 1.1}
    p_ba = {Symbol("b"): 1.1, Symbol("a"): 0.3}
    g_ab = engine.run_gradient(p_ab)[0]
    g_ba = engine.run_gradient(p_ba)[0]

    # Same point, permuted insertion order → permuted array.
    assert g_ab[0] == pytest.approx(g_ba[1], abs=1e-12)
    assert g_ab[1] == pytest.approx(g_ba[0], abs=1e-12)

    # Pin which index is which against finite differences.
    assert g_ab[0] == pytest.approx(_finite_difference(engine, p_ab, Symbol("a")), abs=1e-5)
    assert g_ab[1] == pytest.approx(_finite_difference(engine, p_ab, Symbol("b")), abs=1e-5)


def test_gradient_follows_params_insertion_order_adjoint_overlap():
    """OVERLAP branch: run() returns the complex amplitude ⟨bra|ket⟩, but
    run_gradient returns ∂|⟨bra|ket⟩|²/∂θ — the finite-difference oracle
    must differentiate the squared modulus."""
    from qarp.blocks import ComputationalBasisStateBlock

    ket = _two_param_ket()
    prim = StateVector(bra=ComputationalBasisStateBlock(basis_state=[0, 0]), ket=ket)

    engine = QarpEngine()
    engine.build([prim])

    p_ab = {Symbol("a"): 0.3, Symbol("b"): 1.1}
    p_ba = {Symbol("b"): 1.1, Symbol("a"): 0.3}
    g_ab = engine.run_gradient(p_ab)[0]
    g_ba = engine.run_gradient(p_ba)[0]

    assert g_ab[0] == pytest.approx(g_ba[1], abs=1e-12)
    assert g_ab[1] == pytest.approx(g_ba[0], abs=1e-12)

    def _fd_sq_overlap(params, sym, h=1e-6):
        up = dict(params)
        up[sym] = params[sym] + h
        down = dict(params)
        down[sym] = params[sym] - h
        f_up = abs(engine.run(up)[0]) ** 2
        f_down = abs(engine.run(down)[0]) ** 2
        return (f_up - f_down) / (2 * h)

    assert g_ab[0] == pytest.approx(_fd_sq_overlap(p_ab, Symbol("a")), abs=1e-5)
    assert g_ab[1] == pytest.approx(_fd_sq_overlap(p_ab, Symbol("b")), abs=1e-5)


def test_gradient_follows_params_insertion_order_parameter_shift():
    # A Block observable (not QubitOperator) forces the parameter-shift
    # fallback: supports_backprop EV requires a QubitOperator operator.
    from qarp.blocks import SimpleBlock

    obs = SimpleBlock(2, name="Zobs")
    obs.z(0)
    obs.build()
    ket = _two_param_ket()
    prim = StateVector()
    prim.ket = ket
    prim.bra = ket
    prim.operator = obs

    engine = QarpEngine()
    engine.build([prim])

    p_ab = {Symbol("a"): 0.3, Symbol("b"): 1.1}
    p_ba = {Symbol("b"): 1.1, Symbol("a"): 0.3}
    g_ab = engine.run_gradient(p_ab)[0]
    g_ba = engine.run_gradient(p_ba)[0]

    assert g_ab[0] == pytest.approx(g_ba[1], abs=1e-10)
    assert g_ab[1] == pytest.approx(g_ba[0], abs=1e-10)
    assert g_ab[0] == pytest.approx(_finite_difference(engine, p_ab, Symbol("a")), abs=1e-5)


# ── Optimizer must not fold distinct parameters out of the gradient path ────


@pytest.mark.parametrize("n_qubits", [2, 3, 4, 5])
def test_hea_ansatz_gradients_survive_optimization(n_qubits):
    """A multi-layer HEA must stay differentiable through the default
    optimizer at every width and topology.

    Redundant entanglers (a duplicated ring edge, or a qubit the brickwork
    misses) let two layers' Ry rotations become wire-adjacent; folding them
    into ``Ry(a + b)`` is unitary-exact but leaves a gate the adjoint path
    cannot differentiate.  Oracle: central finite differences of the energy.
    """
    from qarp.algorithms import VQE
    from qarp.blocks import HEABlock

    for linear, circular in ((False, True), (False, False), (True, True)):
        ansatz = HEABlock(
            n_qubits, 2, real=True, linear=linear, circular=circular, use_cz=True
        ).build()
        operator = sum((QubitOperator(f"Z{q}") for q in range(n_qubits)), QubitOperator())
        x0 = np.linspace(0.2, 1.1, len(ansatz.symbols))
        vqe = VQE(operator=operator, ket=ansatz, gradient=True, initial_parameters=x0)
        vqe.suppress_success_message = True
        vqe.build()

        params = dict(zip(ansatz.symbols, x0, strict=True))
        grad = np.asarray(vqe.engine.run_gradient(params)[0], dtype=float)
        assert grad.shape == (len(ansatz.symbols),)

        h = 1e-6
        for i in range(len(x0)):
            up, dn = x0.copy(), x0.copy()
            up[i] += h
            dn[i] -= h
            e_up = vqe.engine.run(dict(zip(ansatz.symbols, up, strict=True)))[0].real
            e_dn = vqe.engine.run(dict(zip(ansatz.symbols, dn, strict=True)))[0].real
            assert abs(grad[i] - (e_up - e_dn) / (2 * h)) < 1e-5, (
                f"n={n_qubits} linear={linear} circular={circular} param {i}"
            )
