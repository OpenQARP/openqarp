"""Gradient method registry: names, per-engine declarations, options,
``gradient_kind``, and the batched finite-difference / SPSA methods.

Oracles: exception contracts for the registry rows; the *optimizer-side*
helpers ``compute_fd_gradients`` / ``compute_spsa_gradients`` (independent
code paths, sequential objective calls) and closed forms for the numbers.
"""

import numpy as np
import pytest
from sympy import Symbol

import qarpx as qx
from qarp import EXACT
from qarp.algorithms import PauliAveraging, PauliShadow, Sampler, StateVector
from qarp.blocks import ComputationalBasisStateBlock, SimpleBlock
from qarp.engines import CudaqEngine, Engine, QarpEngine
from qarp.engines._gradients import GRADIENT_KINDS, GRADIENT_METHODS, RESERVED_METHODS
from qarp.errors import CapabilityError
from qarp.optimizers import compute_fd_gradients, compute_spsa_gradients


def _ry_ket():
    b = SimpleBlock(1)
    b.ry(0, Symbol("t"))
    b.build()
    return b


def _two_param_ket():
    b = SimpleBlock(2)
    b.rx(0, qx.Param.symbol("a"))
    b.ry(1, qx.Param.symbol("b"))
    b.cx(0, 1)
    b.build()
    return b


def _z_block(n):
    obs = SimpleBlock(n, name="Zobs")
    obs.z(0)
    obs.build()
    return obs


def _fd(engine, params, sym, h=1e-6, transform=lambda v: np.real(v)):
    up, dn = dict(params), dict(params)
    up[sym] += h
    dn[sym] -= h
    return (transform(engine.run(up)[0]) - transform(engine.run(dn)[0])) / (2 * h)


# ── registry contract ─────────────────────────────────────────────────────


def test_registry_names():
    assert GRADIENT_METHODS == ("default", "adjoint", "parameter-shift", "finite-diff", "spsa")
    assert set(RESERVED_METHODS).isdisjoint(GRADIENT_METHODS)
    assert QarpEngine.gradient_methods == frozenset(GRADIENT_METHODS)
    assert "adjoint" not in CudaqEngine.gradient_methods  # class attribute; no GPU needed
    assert Engine.gradient_methods == frozenset()


def test_unknown_method_is_a_value_error_and_reserved_is_a_capability_error():
    engine = QarpEngine()
    engine.build([StateVector(ket=_ry_ket(), operator=qx.QubitOperator("Z0"))])
    with pytest.raises(ValueError, match="unknown gradient method"):
        engine.run_gradient({"t": 0.3}, method="parameter_shift")
    for name in RESERVED_METHODS:
        with pytest.raises(CapabilityError, match="reserved"):
            engine.run_gradient({"t": 0.3}, method=name)


def test_options_are_validated_against_the_requested_method():
    engine = QarpEngine()
    engine.build([StateVector(ket=_ry_ket(), operator=qx.QubitOperator("Z0"))])
    with pytest.raises(ValueError, match="takes no options"):
        engine.run_gradient({"t": 0.3}, method="adjoint", options={"fd_eps": 1e-3})
    with pytest.raises(ValueError, match="not valid for gradient method 'finite-diff'"):
        engine.run_gradient({"t": 0.3}, method="finite-diff", options={"spsa_c0": 1e-3})
    with pytest.raises(ValueError, match="fd_order"):
        engine.run_gradient({"t": 0.3}, method="finite-diff", options={"fd_order": 3})


def test_undeclared_method_is_refused_naming_the_engine_that_has_it():
    class _Stub(Engine):
        gradient_methods = CudaqEngine.gradient_methods

        def _sweep(self, *a, **k):
            raise NotImplementedError

    prim = StateVector(ket=_ry_ket(), operator=qx.QubitOperator("Z0")).build()
    with pytest.raises(CapabilityError, match="QarpEngine provides 'adjoint'"):
        _Stub().resolve_gradient_method(prim, "adjoint")


def test_adjoint_refused_on_non_eligible_primitive_and_default_picks_shift():
    engine = QarpEngine()
    prim = StateVector(ket=_two_param_ket(), operator=_z_block(2))  # Block observable
    engine.build([prim])
    assert engine.resolve_gradient_method(prim, "default") == "parameter-shift"
    with pytest.raises(CapabilityError, match="not adjoint-differentiable"):
        engine.run_gradient({"a": 0.3, "b": 1.1}, method="adjoint")

    ev = StateVector(ket=_two_param_ket(), operator=qx.QubitOperator("Z0"))
    engine.build([ev])
    assert engine.resolve_gradient_method(ev, "default") == "adjoint"


# ── gradient_kind ─────────────────────────────────────────────────────────


def test_every_shipped_primitive_declares_a_gradient_kind():
    import qarp.algorithms as alg
    from qarp.algorithms import PrimitiveAlgorithm

    classes = [
        getattr(alg, name)
        for name in alg.__all__
        if isinstance(getattr(alg, name), type)
        and issubclass(getattr(alg, name), PrimitiveAlgorithm)
    ]
    assert len(classes) >= 12
    for cls in classes:
        assert cls.gradient_kind in GRADIENT_KINDS, cls
    assert PrimitiveAlgorithm.gradient_kind == "none"  # the safe default for a new primitive
    assert Sampler.gradient_kind == "none"
    assert PauliShadow.gradient_kind == "none"  # median-of-means is non-linear
    assert PauliAveraging.gradient_kind == "expectation"


def test_state_vector_kind_follows_its_target():
    ket = _two_param_ket()
    ev = StateVector(ket=ket, operator=qx.QubitOperator("Z0")).build()
    assert ev.gradient_kind == "expectation"
    block_ev = StateVector(ket=ket, operator=_z_block(2)).build()
    assert block_ev.gradient_kind == "amplitude"
    bra = ComputationalBasisStateBlock(basis_state=[0, 0])
    assert StateVector(ket=ket, bra=bra).build().gradient_kind == "squared_overlap"
    assert (
        StateVector(ket=ket, bra=bra, operator=qx.QubitOperator("Z0")).build().gradient_kind
        == "amplitude"
    )


def test_projected_state_vector_refuses_parameter_shift():
    from qarp.algorithms._composite.projected_vqe import _ProjectedStateVector

    # A Rayleigh quotient is a ratio: the shift rule would return a wrong number.
    assert _ProjectedStateVector.gradient_kind == "none"
    # Pinned on a *built* instance: StateVector.build() would overwrite the
    # declaration by target, and _ProjectedStateVector.build() must not call it.
    from qarp.algorithms import ProjectedVQE

    pvqe = ProjectedVQE(
        operator=qx.QubitOperator("Z0") + qx.QubitOperator("X0", 0.5),
        ket=_ry_ket(),
        initial_parameters=[0.3],
    ).build()
    assert pvqe.primitive.gradient_kind == "none"
    with pytest.raises(CapabilityError, match="gradient_kind='none'"):
        pvqe.engine.run_gradient({"t": 0.3}, method="parameter-shift")
    (g,) = pvqe.engine.run_gradient({"t": 0.3}, method="finite-diff")
    assert np.isfinite(g).all()


def test_shadow_gradient_kind_is_unreachable_because_it_needs_a_bound_ket():
    """A shadow's ``"none"`` is a declaration only: its ``build()`` refuses a
    ket with free symbols, so no gradient method — not even finite
    differences — can reach a shadow primitive."""
    engine = QarpEngine()
    with pytest.raises(ValueError, match="concrete"):
        engine.build([PauliShadow(qx.QubitOperator("Z0"), _ry_ket(), n_settings=64, seed=0)])


def test_kind_none_refuses_parameter_shift_but_scalar_check_guards_finite_diff():
    engine = QarpEngine()
    engine.build([Sampler(ket=_ry_ket(), n_shots=EXACT)])
    with pytest.raises(CapabilityError, match="gradient_kind='none'"):
        engine.run_gradient({"t": 0.3}, method="parameter-shift")
    with pytest.raises(CapabilityError, match="not a scalar"):
        engine.run_gradient({"t": 0.3}, method="finite-diff")


# ── finite-diff / spsa: batched engine methods vs the optimizer helpers ────


def _energy_objective(engine, ket):
    def objective(x):
        return float(np.real(engine.run(ket.parameter_map(x))[0]))

    return objective


def test_forward_finite_difference_equals_optimizer_helper():
    ket = _two_param_ket()
    engine = QarpEngine()
    engine.build([StateVector(ket=ket, operator=qx.QubitOperator("Z0") + qx.QubitOperator("X1"))])
    x = np.array([0.4, -0.7])
    g_engine = engine.run_gradient(
        ket.parameter_map(x), method="finite-diff", options={"fd_order": 1, "fd_eps": 1e-5}
    )[0]
    g_helper = compute_fd_gradients(_energy_objective(engine, ket), x, fd_eps=1e-5)
    assert np.allclose(g_engine, g_helper, atol=1e-12)


def test_central_finite_difference_matches_closed_form():
    ket = _ry_ket()  # ⟨Z⟩ = cos t on Ry(t)|0⟩
    engine = QarpEngine()
    engine.build([StateVector(ket=ket, operator=qx.QubitOperator("Z0"))])
    t = 0.83
    g = engine.run_gradient({"t": t}, method="finite-diff")[0]
    assert g[0] == pytest.approx(-np.sin(t), abs=1e-8)


def test_spsa_equals_optimizer_helper_under_one_seed():
    ket = _two_param_ket()
    engine = QarpEngine()
    engine.build([StateVector(ket=ket, operator=qx.QubitOperator("Z0") + qx.QubitOperator("X1"))])
    x = np.array([0.4, -0.7])
    opts = {"spsa_c0": 1e-2, "num_spsa": 3, "spsa_seed": 11}
    g_engine = engine.run_gradient(ket.parameter_map(x), method="spsa", options=opts)[0]
    g_helper = compute_spsa_gradients(
        _energy_objective(engine, ket), x, ck=1e-2, num_perturbations=3, seed=11
    )
    assert np.allclose(g_engine, g_helper, atol=1e-12)


def test_spsa_default_seed_comes_from_the_engine_seed():
    ket = _two_param_ket()
    point = ket.parameter_map([0.4, -0.7])
    op = qx.QubitOperator("Z0") + qx.QubitOperator("X1")
    grads = []
    for _ in range(2):
        engine = QarpEngine(seed=7)
        engine.build([StateVector(ket=ket, operator=op)])
        grads.append(engine.run_gradient(point, method="spsa")[0])
    assert np.array_equal(grads[0], grads[1])  # equally seeded engines agree across processes
    unseeded = []
    for _ in range(2):
        engine = QarpEngine()
        engine.build([StateVector(ket=ket, operator=op)])
        unseeded.append(engine.run_gradient(point, method="spsa", options={"num_spsa": 4})[0])
    assert np.array_equal(unseeded[0], unseeded[1])  # an unseeded engine still draws one stream
    # The stream advances between calls (values may coincide: δ and −δ give
    # the same estimate), so pin the generator state instead.
    before = engine._gradient_rng().bit_generator.state["state"]["state"]
    engine.run_gradient(point, method="spsa")
    assert engine._gradient_rng().bit_generator.state["state"]["state"] != before


# ── batching, ordering, targets ───────────────────────────────────────────


def test_shift_path_evaluates_every_point_in_one_sweep(monkeypatch):
    ket = _two_param_ket()
    engine = QarpEngine()
    engine.build([StateVector(ket=ket, operator=_z_block(2))])
    calls = []
    original = engine._sweep

    def spy(prims, circuits, l2p, points, override, **kw):
        calls.append(len(points))
        return original(prims, circuits, l2p, points, override, **kw)

    monkeypatch.setattr(engine, "_sweep", spy)
    engine.run_gradient({"a": 0.3, "b": 1.1})  # default → parameter shift
    # ⟨ψ|U|ψ⟩ compiles to [ket, ket∘U]: each symbol is one occurrence per
    # circuit, two points per occurrence — 8 points, still one sweep.
    assert calls == [8]
    calls.clear()
    engine.run_gradient({"a": 0.3, "b": 1.1}, method="finite-diff", options={"fd_order": 1})
    assert calls == [3]  # base point + one forward step per symbol
    calls.clear()
    engine.build([StateVector(ket=ket, operator=qx.QubitOperator("Z0"))])  # one circuit
    engine.run_gradient({"a": 0.3, "b": 1.1}, method="parameter-shift")
    assert calls == [4]  # ±shift per occurrence, one occurrence per symbol
    calls.clear()
    crx = SimpleBlock(2)
    crx.h(0)
    crx.crx(0, 1, qx.Param.symbol("a"))
    crx.build()
    engine.build([StateVector(ket=crx, operator=qx.QubitOperator("X0 Z1"))])
    engine.run_gradient({"a": 0.3}, method="parameter-shift")
    assert calls == [4]  # the four-term rule: ±π/2 and ±3π/2 for one occurrence
    calls.clear()
    engine.build([StateVector(ket=crx, operator=_z_block(2))])  # [ket, ket∘U]: two occurrences
    engine.run_gradient({"a": 0.3}, method="parameter-shift")
    assert calls == [4]  # amplitude kind: every gate is two-term (±π), CRx included


def test_rebuild_never_reuses_a_stale_shift_plan():
    """The plan cache is keyed by build generation: after ``build()`` with a
    different circuit under the same symbols, the shift gradient must be that
    of the new circuit (a stale plan would return the old circuit's number)."""
    first = _two_param_ket()
    second = SimpleBlock(2)
    second.ry(0, qx.Param.symbol("a"))
    second.cx(0, 1)
    second.rx(1, qx.Param.linear(2.0, "b", 0.4))
    second.ry(0, qx.Param.symbol("b"))
    second.build()
    engine = QarpEngine()
    p = {"a": 0.3, "b": 1.1}
    engine.build([StateVector(ket=first, operator=_z_block(2))])
    g_first = engine.run_gradient(p, method="parameter-shift")[0]
    engine.build([StateVector(ket=second, operator=_z_block(2))])
    g_second = engine.run_gradient(p, method="parameter-shift")[0]
    fd = [_fd(engine, p, s) for s in ("a", "b")]
    assert np.allclose(g_second.real, fd, atol=1e-6)
    assert not np.allclose(g_first.real, fd, atol=1e-3)  # the circuits do differ


@pytest.mark.parametrize("method", ["adjoint", "parameter-shift", "finite-diff"])
def test_columns_follow_params_insertion_order(method):
    ket = _two_param_ket()
    engine = QarpEngine()
    op = qx.QubitOperator("Z0") + qx.QubitOperator("X1", 0.5)
    engine.build([StateVector(ket=ket, operator=op)])
    p_ab = {Symbol("a"): 0.3, Symbol("b"): 1.1}
    p_ba = {Symbol("b"): 1.1, Symbol("a"): 0.3}
    g_ab = engine.run_gradient(p_ab, method=method)[0]
    g_ba = engine.run_gradient(p_ba, method=method)[0]
    assert g_ab[0] == pytest.approx(g_ba[1], abs=1e-12)
    assert g_ab[1] == pytest.approx(g_ba[0], abs=1e-12)
    assert g_ab[0] == pytest.approx(_fd(engine, p_ab, Symbol("a")), abs=1e-5)


def test_overlap_is_differentiated_as_squared_modulus_on_every_method():
    ket = _two_param_ket()
    bra = ComputationalBasisStateBlock(basis_state=[0, 0])
    engine = QarpEngine()
    engine.build([StateVector(ket=ket, bra=bra)])
    p = {"a": 0.3, "b": 1.1}
    fd = [_fd(engine, p, s, transform=lambda v: abs(v) ** 2) for s in ("a", "b")]
    for method in ("adjoint", "parameter-shift", "finite-diff"):
        g = engine.run_gradient(p, method=method)[0]
        assert g.dtype == np.float64
        assert np.allclose(g, fd, atol=1e-5), method


def test_block_operator_expectation_is_complex_by_target_and_matches_fd():
    ket = _two_param_ket()
    obs = SimpleBlock(2, name="obs")
    obs.z(0)
    obs.s(1)  # non-Hermitian U: ⟨ψ|U|ψ⟩ is genuinely complex
    obs.build()
    engine = QarpEngine()
    engine.build([StateVector(ket=ket, operator=obs)])
    p = {"a": 0.3, "b": 1.1}
    g = engine.run_gradient(p)[0]
    assert g.dtype == np.complex128
    for k, s in enumerate(("a", "b")):
        assert g[k].real == pytest.approx(_fd(engine, p, s), abs=1e-5)
        assert g[k].imag == pytest.approx(_fd(engine, p, s, transform=np.imag), abs=1e-5)


def test_transition_amplitude_is_complex_by_target():
    ket = _ry_ket()
    bra = SimpleBlock(1)
    bra.h(0)
    bra.build()
    engine = QarpEngine()
    engine.build([StateVector(ket=ket, bra=bra, operator=qx.QubitOperator("Z0"))])
    assert engine.run_gradient({"t": 0.5})[0].dtype == np.complex128


def test_transition_amplitude_shift_matches_fd():
    """Linear in the ket (frequency ½): the old two-term rule was off by √2."""
    ket = _ry_ket()
    bra = SimpleBlock(1)
    bra.h(0)
    bra.build()
    engine = QarpEngine()
    engine.build([StateVector(ket=ket, bra=bra, operator=qx.QubitOperator("Z0"))])
    p = {"t": 0.5}
    g = engine.run_gradient(p)[0]
    assert g[0].real == pytest.approx(_fd(engine, p, "t"), abs=1e-5)


def test_pauli_averaging_exact_shift_matches_fd():
    """A COUNTS primitive under n_shots=EXACT differentiates through the
    batched shift with no shot noise (previously refused by VQA)."""
    ket = _two_param_ket()
    op = qx.QubitOperator("Z0") + qx.QubitOperator("X1", 0.5)
    engine = QarpEngine()
    engine.build([PauliAveraging(ket=ket, operator=op, n_shots=EXACT)])
    p = {"a": 0.3, "b": 1.1}
    g = engine.run_gradient(p)[0]
    assert g.dtype == np.float64
    assert np.allclose(g, [_fd(engine, p, s) for s in ("a", "b")], atol=1e-5)
