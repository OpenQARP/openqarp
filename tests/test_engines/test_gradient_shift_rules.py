"""Gate-level gradient oracle: every parametric gate × method × angle form.

This is the test that would have caught the three silent wrong gradients
(symbolic ``U``, controlled rotations, transition amplitudes): the property
sweep in ``test_gradient_consistency.py`` is over blocks, and no block emits
a symbolic ``U`` or a controlled rotation.  Oracle: central finite
differences of the very objective the engine evaluates.

States and observables are chosen so every frequency component of the
objective is non-zero — an observable diagonal on a controlled gate's
control qubit hides the half-frequency terms, and a wrong coefficient on a
vanishing term would pass.  Each case asserts the finite-difference value
itself is non-negligible for that reason.
"""

import numpy as np
import pytest

import qarpx as qx
from qarp.algorithms import StateVector
from qarp.blocks import SimpleBlock
from qarp.engines import QarpEngine
from qarp.errors import CapabilityError

ONE_QUBIT = ("Rx", "Ry", "Rz", "P", "GPhase", "U")
TWO_QUBIT = ("CRx", "CRy", "CRz", "CP", "RXX", "RYY", "RZZ", "CU")
GATES = ONE_QUBIT + TWO_QUBIT
# angle forms: (label, factory of the symbolic angle from a symbol name)
FORMS = ("plain", "scaled", "half", "shared", "compound")
METHODS = ("adjoint", "parameter-shift", "finite-diff")
H = 1e-6
ATOL = 1e-7
NONDEGENERATE = 1e-3


def _angle(form: str, sym: str):
    if form == "plain":
        return qx.Param.symbol(sym)
    if form == "scaled":
        return qx.Param.linear(-2.0, sym, 0.3)
    if form == "half":
        return qx.Param.linear(0.5, sym, -0.2)
    if form == "compound":
        return (qx.Param.symbol(sym) + qx.Param.symbol("b")) * qx.Param(0.5)
    return qx.Param.symbol(sym)  # "shared": the symbol is reused on a second gate


def _apply(block, gate: str, angle):
    """Apply ``gate`` with ``angle``; single-qubit gates on qubit 0,
    two-qubit gates on (0, 1)."""
    if gate == "GPhase":
        block.gphase(angle)
        return
    if gate == "U":  # θ = angle under test, φ carries the symbol too (compound phase)
        block.u(0, angle, qx.Param.symbol("x"), 0.4)
        return
    if gate == "CU":
        block.cu(0, 1, angle, qx.Param.symbol("x"), 0.4, 0.2)
        return
    method = getattr(block, gate.lower())
    if gate in ONE_QUBIT:
        method(0, angle)
    else:
        method(0, 1, angle)


def _ket(gate: str, form: str):
    """A generic 2-qubit dressing around the gate under test: nothing is
    diagonal, so every frequency component survives."""
    b = SimpleBlock(2)
    b.ry(0, 0.6)  # no qubit sits in an eigenstate of the gate's generator
    b.rx(1, 0.4)
    b.rz(0, 0.5)
    b.cx(0, 1)
    b.ry(1, 0.9)
    _apply(b, gate, _angle(form, "x"))
    if form == "shared":
        b.ry(1, qx.Param.symbol("x"))  # second occurrence, different generator
    b.cx(0, 1)
    b.ry(0, 0.7)
    b.build()
    return b


def _values(form: str) -> dict:
    p = {"x": 0.83}
    if form == "compound":
        p["b"] = -0.41
    return p


def _pauli_sum():
    # Off-diagonal on both qubits: no generator's half-frequency term vanishes.
    return (
        qx.QubitOperator("X0", 1.0)
        + qx.QubitOperator("Y1", 0.7)
        + qx.QubitOperator("Z0 Z1", 0.4)
        + qx.QubitOperator("X0 Y1", 0.3)
    )


def _block_observable():
    obs = SimpleBlock(2, name="U")
    obs.h(0)
    obs.s(1)
    obs.cx(0, 1)
    obs.build()
    return obs


def _bra():
    b = SimpleBlock(2)
    b.ry(0, 0.9)
    b.h(1)
    b.build()
    return b


def _fd(engine, params, sym, part=np.real):
    up, dn = dict(params), dict(params)
    up[sym] += H
    dn[sym] -= H
    return (part(engine.run(up)[0]) - part(engine.run(dn)[0])) / (2 * H)


def _check(engine, params, grad, complex_out):
    for k, sym in enumerate(params):
        fd_re = _fd(engine, params, sym)
        assert grad[k].real == pytest.approx(fd_re, abs=ATOL), f"∂Re/∂{sym}"
        if complex_out:
            fd_im = _fd(engine, params, sym, np.imag)
            assert grad[k].imag == pytest.approx(fd_im, abs=ATOL), f"∂Im/∂{sym}"
            assert abs(fd_re) + abs(fd_im) > NONDEGENERATE, "degenerate oracle"
        else:
            assert grad.dtype == np.float64
            assert abs(fd_re) > NONDEGENERATE, "degenerate oracle"


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("gate", GATES)
def test_expectation_value_over_qubit_operator(gate, method, form):
    engine = QarpEngine()
    engine.build([StateVector(ket=_ket(gate, form), operator=_pauli_sum())])
    params = _values(form)
    if gate == "GPhase" and form != "shared":
        # A global phase drops out of ⟨ψ|H|ψ⟩: the derivative is exactly zero.
        grad = engine.run_gradient(params, method=method)[0]
        assert np.allclose(grad, 0.0, atol=1e-9)
        return
    grad = engine.run_gradient(params, method=method)[0]
    _check(engine, params, grad, complex_out=False)


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("method", ("parameter-shift", "finite-diff"))
@pytest.mark.parametrize("gate", GATES)
def test_expectation_value_over_block_operator(gate, method, form):
    """⟨ψ|U|ψ⟩ with a circuit-valued U: complex, AMPLITUDE kind, the symbol
    appears in both compiled circuits ([ket, ket∘U])."""
    engine = QarpEngine()
    engine.build([StateVector(ket=_ket(gate, form), operator=_block_observable())])
    params = _values(form)
    grad = engine.run_gradient(params, method=method)[0]
    assert grad.dtype == np.complex128
    if gate == "GPhase" and form != "shared":
        assert np.allclose(grad, 0.0, atol=1e-9)
        return
    _check(engine, params, grad, complex_out=True)


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("method", ("parameter-shift", "finite-diff"))
@pytest.mark.parametrize("gate", GATES)
def test_transition_amplitude(gate, method, form):
    """⟨bra|H|ket⟩, bra ≠ ket: linear in the ket's amplitudes (frequency ½
    for a Pauli rotation) — the case the old two-term rule got wrong by √2.
    A global phase now matters."""
    engine = QarpEngine()
    engine.build([StateVector(ket=_ket(gate, form), bra=_bra(), operator=_pauli_sum())])
    params = _values(form)
    grad = engine.run_gradient(params, method=method)[0]
    assert grad.dtype == np.complex128
    _check(engine, params, grad, complex_out=True)


# ── pinned regressions (the old wrong numbers, from the plan's *Why*) ────────


def test_regression_controlled_rotation_two_term_was_off_by_sqrt2():
    """CRx/CRy/CRz with an off-diagonal control observable: the old two-term
    shift returned −0.223325 where central FD gives −0.157915 (ratio √2)."""
    for gate in ("crx", "cry", "crz"):
        k = SimpleBlock(2)
        k.h(0)
        k.ry(1, 0.4)
        getattr(k, gate)(0, 1, qx.Param.symbol("t"))
        k.build()
        obs = SimpleBlock(2)
        obs.x(0)
        obs.z(1)
        obs.build()
        engine = QarpEngine()
        engine.build([StateVector(ket=k, operator=obs)])
        g = engine.run_gradient({"t": 0.7}, method="parameter-shift")[0]
        assert g[0].real == pytest.approx(_fd(engine, {"t": 0.7}, "t"), abs=ATOL), gate


def test_regression_transition_amplitude_two_term_was_off_by_sqrt2():
    """bra = Ry(0.9)|0⟩, ket = Rx(t)|0⟩, Z observable: the old shift gave
    −0.157525 where central FD of the real part gives −0.111387."""
    ket = SimpleBlock(1)
    ket.rx(0, qx.Param.symbol("t"))
    ket.build()
    bra = SimpleBlock(1)
    bra.ry(0, 0.9)
    bra.build()
    engine = QarpEngine()
    engine.build([StateVector(ket=ket, bra=bra, operator=qx.QubitOperator("Z0"))])
    g = engine.run_gradient({"t": 0.5}, method="parameter-shift")[0]
    assert g[0].real == pytest.approx(_fd(engine, {"t": 0.5}, "t"), abs=ATOL)
    assert g[0].imag == pytest.approx(_fd(engine, {"t": 0.5}, "t", np.imag), abs=ATOL)


def test_regression_overlap_adjoint_dropped_the_bra_term():
    """OVERLAP with one symbol driving both circuits (bra ``Ry(t)Rz(0.3)``,
    ket ``Ry(0.4)Rx(t)``): the adjoint swept the ket only and returned
    −0.298186 where central FD of |⟨bra|ket⟩|² gives −0.466776 — the shift and
    finite-difference methods were already right.  Every method now agrees."""
    ket = SimpleBlock(1)
    ket.ry(0, 0.4)
    ket.rx(0, qx.Param.symbol("t"))
    ket.build()
    bra = SimpleBlock(1)
    bra.ry(0, qx.Param.symbol("t"))
    bra.rz(0, 0.3)
    bra.build()
    engine = QarpEngine()
    engine.build([StateVector(ket=ket, bra=bra)])
    p = {"t": 0.8}
    fd = _fd(engine, p, "t", lambda v: abs(v) ** 2)
    assert abs(fd) > NONDEGENERATE
    for method in METHODS:
        g = engine.run_gradient(p, method=method)[0]
        assert g.dtype == np.float64
        assert g[0] == pytest.approx(fd, abs=ATOL), method


def test_regression_compound_angle_on_the_adjoint_raised_runtime_error():
    """``Ry((a+b)/2)`` through the adjoint used to surface as a bare
    ``RuntimeError`` ("depends on 2 symbols"); the compound chain rule now
    differentiates it and the C++ refusals are ``CapabilityError``."""
    engine = QarpEngine()
    engine.build([StateVector(ket=_ket("Ry", "compound"), operator=_pauli_sum())])
    params = _values("compound")
    grad = engine.run_gradient(params, method="adjoint")[0]
    _check(engine, params, grad, complex_out=False)


def test_zero_coefficient_occurrence_contributes_nothing():
    """``Param.linear(0, "z", 0.4)`` is an angle the symbol does not move:
    the shift rule used to divide by the coefficient (``ZeroDivisionError``);
    the occurrence is now skipped and every method returns 0 for ``z``."""
    ket = SimpleBlock(1)
    ket.ry(0, qx.Param.symbol("t"))
    ket.rx(0, qx.Param.linear(0.0, "z", 0.4))
    ket.build()
    engine = QarpEngine()
    engine.build([StateVector(ket=ket, operator=_block_observable_1q())])
    p = {"t": 0.3, "z": 0.5}
    for method in ("parameter-shift", "finite-diff"):
        g = engine.run_gradient(p, method=method)[0]
        assert g[0].real == pytest.approx(_fd(engine, p, "t"), abs=ATOL), method
        assert g[1] == 0.0, method


def _block_observable_1q():
    obs = SimpleBlock(1, name="U1")
    obs.h(0)
    obs.build()
    return obs


def test_non_affine_angle_is_refused_by_both_paths_with_capability_error():
    sq = qx.Param.symbol("t") * qx.Param.symbol("t")
    ket = SimpleBlock(1)
    ket.ry(0, sq)
    ket.build()
    engine = QarpEngine()
    engine.build([StateVector(ket=ket, operator=qx.QubitOperator("Z0"))])
    with pytest.raises(CapabilityError, match="affine"):
        engine.run_gradient({"t": 0.3}, method="adjoint")
    with pytest.raises(CapabilityError, match="affine"):
        engine.run_gradient({"t": 0.3}, method="parameter-shift")
    # Finite differences do not care about the angle's form.
    g = engine.run_gradient({"t": 0.3}, method="finite-diff")[0]
    assert g[0] == pytest.approx(-np.sin(0.09) * 0.6, abs=1e-7)  # ⟨Z⟩ = cos(t²)


def test_formerly_refused_shapes_now_differentiate():
    """Multi-occurrence (UCC-shaped) and differing coefficients across circuits
    (bra Rx(2θ) vs ket Rx(θ)) were CapabilityErrors of the single-occurrence
    rule; per-occurrence shifting handles both."""
    from tests.strategies import FACTORIES

    ucc = FACTORIES["UCCBlock"]().build()
    obs = SimpleBlock(ucc.n_qubits, name="Zobs")
    obs.z(0)
    obs.build()
    engine = QarpEngine()
    engine.build([StateVector(ket=ucc, operator=obs)])
    rng = np.random.default_rng(0)
    params = ucc.parameter_map(rng.uniform(-1, 1, len(ucc.symbols)))
    grad = engine.run_gradient(params, method="parameter-shift")[0]
    for k, sym in enumerate(params):
        assert grad[k].real == pytest.approx(_fd(engine, params, sym), abs=1e-6), str(sym)

    ket = SimpleBlock(1)
    ket.rx(0, qx.Param.symbol("t"))
    ket.build()
    bra = SimpleBlock(1)
    bra.rx(0, qx.Param.linear(2.0, "t"))
    bra.h(0)
    bra.build()
    engine = QarpEngine()
    engine.build([StateVector(ket=ket, bra=bra)])  # OVERLAP → |⟨bra|ket⟩|²
    g = engine.run_gradient({"t": 0.4}, method="parameter-shift")[0]
    fd = _fd(engine, {"t": 0.4}, "t", part=lambda v: abs(v) ** 2)
    assert g[0] == pytest.approx(fd, abs=ATOL)


def test_regression_symbolic_u_adjoint_returned_zeros():
    """The plan's first *Why* row: ``U(θ,φ,λ)`` after ``ry(0.3)`` with
    ``⟨X + 0.7Y + 0.4Z⟩`` — the adjoint returned ``[0, 0, 0]`` where central
    FD gives ``[0.680, −0.013, 0.069]``; a symbolic ``CU`` on two qubits
    returned ``[0, 0, 0]`` the same way.  Both paths now rewrite ``U``/``CU``
    into ``GPhase·Rz·Ry·Rz`` / the ``P``-``CX`` ladder before differentiating
    (the compound ``(φ±λ)/2`` phases are what Step 4 made differentiable)."""
    ket = SimpleBlock(1)
    ket.ry(0, 0.3)
    ket.u(0, qx.Param.symbol("th"), qx.Param.symbol("ph"), qx.Param.symbol("la"))
    ket.build()
    op = qx.QubitOperator("X0") + qx.QubitOperator("Y0", 0.7) + qx.QubitOperator("Z0", 0.4)
    engine = QarpEngine()
    engine.build([StateVector(ket=ket, operator=op)])
    p = {"th": 0.4, "ph": 0.9, "la": -0.6}
    fd = np.array([_fd(engine, p, s) for s in p])
    assert np.allclose(fd, [0.680, -0.013, 0.069], atol=2e-3)  # the numbers in the plan
    for method in ("adjoint", "parameter-shift", "finite-diff"):
        g = engine.run_gradient(p, method=method)[0]
        assert np.allclose(g, fd, atol=ATOL), method
    # The compiled circuit still carries the native U: the rewrite is the
    # gradient path's own, and never touches what run() executes.
    assert [c.gate.name for c in engine._primitives[0].compiled_circuits[0]].count("U") == 1
    cu = SimpleBlock(2)
    cu.ry(0, 0.5)
    cu.ry(1, 0.9)
    cu.cu(0, 1, qx.Param.symbol("th"), qx.Param.symbol("ph"), qx.Param.symbol("la"))
    cu.build()
    engine = QarpEngine()
    engine.build(
        [StateVector(ket=cu, operator=qx.QubitOperator("X1") + qx.QubitOperator("Z0 Y1", 0.5))]
    )
    params = {"th": 0.4, "ph": 0.9, "la": -0.7}
    for method in METHODS:
        grad = engine.run_gradient(params, method=method)[0]
        for k, sym in enumerate(params):
            assert grad[k] == pytest.approx(_fd(engine, params, sym), abs=ATOL), (method, sym)
        assert np.abs(grad).max() > NONDEGENERATE


# ── §9 named-inverse pairs in the backward sweep ─────────────────────────
#
# The adjoint steps |ψ⟩ and |φ⟩ back through *every* gate, parametric or
# not.  A gate whose dagger is a different GateType (§9) therefore has to be
# inverted as that type; treating it as self-adjoint corrupts both states
# and silently flips the sign of its contribution.  Nothing here carries a
# symbol — the gate under test is a fixed gate sitting between the
# parameter and the observable.
NAMED_INVERSE_GATES = ("s", "sdg", "t", "tdg", "sx", "sxdg", "cs", "csdg", "csx", "csxdg")
_TWO_QUBIT_NAMED_INVERSE = {"cs", "csdg", "csx", "csxdg"}


def _circuit_around(gate, seed):
    """A 3-qubit circuit with one symbol and gate placed mid-circuit.

    The placement is seeded rather than fixed because a single placement can
    leave the gate decoupled from the observable, where an un-inverted gate
    cancels out and the bug hides (it did, for five of these six, until the
    sweep was widened).
    """
    rng = np.random.default_rng(seed)
    block = SimpleBlock(3)
    block.ry(0, qx.Param.symbol("t"))
    for _ in range(4):
        block.rz(int(rng.integers(0, 3)), float(rng.uniform(0, 2 * np.pi)))
        a, b = (int(x) for x in rng.permutation(3)[:2])
        block.cx(a, b)
    a, b = (int(x) for x in rng.permutation(3)[:2])
    if gate in _TWO_QUBIT_NAMED_INVERSE:
        getattr(block, gate)(a, b)
    else:
        getattr(block, gate)(a)
    for _ in range(2):
        block.ry(int(rng.integers(0, 3)), float(rng.uniform(0, 2 * np.pi)))
    block.build()
    return block


@pytest.mark.parametrize("gate", NAMED_INVERSE_GATES)
def test_adjoint_inverts_named_inverse_gates(gate):
    """§9: SX/SXdg, CS/CSdg and CSX/CSXdg are named-inverse pairs,
    not self-adjoint gates.  The backward sweep used to fall through to a
    default: break labelled "self-adjoint" for all six, so any circuit
    containing one returned a silently wrong adjoint gradient — a sign flip
    on that gate's contribution, agreeing with the oracle only when the gate
    happened not to couple to the observable.  S/T were always
    handled and are swept here as controls."""
    observable = _pauli_sum()
    seen_nondegenerate = False
    for seed in range(1000, 1008):
        block = _circuit_around(gate, seed)
        engine = QarpEngine()
        engine.build([StateVector(ket=block, operator=observable)])
        params = {"t": 0.41}
        fd = _fd(engine, params, "t")
        if abs(fd) > NONDEGENERATE:
            seen_nondegenerate = True
        for method in METHODS:
            got = engine.run_gradient(params, method=method)[0][0]
            assert got == pytest.approx(fd, abs=ATOL), f"{gate} / {method} / seed {seed}"
    assert seen_nondegenerate, f"{gate}: no seed produced a non-negligible gradient"
