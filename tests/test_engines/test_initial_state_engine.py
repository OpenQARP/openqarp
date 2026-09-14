"""Primitive-level ``initial_state=`` through the engines.

Ket-seeding semantics: ``Sampler`` / ``StateVector`` carry the seed; the
engine threads it through the sampled, EXACT, and amplitudes paths.
Oracles are numpy ``kron``-literal linear algebra (LSB ordering per §1)
and analytic Born probabilities; the gradient row uses central finite
differences.
"""

import numpy as np
import pytest
from sympy import Symbol

import qarp
from qarp.algorithms import Sampler, StateVector
from qarp.blocks import SimpleBlock
from qarp.devices import Device, get_nearest_neighbour_architecture
from qarp.engines import QarpEngine
from qarp.errors import CapabilityError
from qarp.operators import QubitOperator
from tests.conftest import _StubEngine

ATOL = 1e-10

RNG = np.random.default_rng(20260731)

I2 = np.eye(2, dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
Z = np.diag([1.0, -1.0]).astype(complex)


def _ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _random_state(n_qubits):
    dim = 2**n_qubits
    psi = RNG.normal(size=dim) + 1j * RNG.normal(size=dim)
    return psi / np.linalg.norm(psi)


def _h0_block(n=2):
    b = SimpleBlock(n)
    b.h(0)
    return b


# ── Sampler through QarpEngine ───────────────────────────────────────────


def test_sampler_exact_from_seeded_state():
    psi = _random_state(2)
    u = np.kron(I2, H)  # H on qubit 0 (rightmost kron factor)
    expected = np.abs(u @ psi) ** 2

    sampler = Sampler(ket=_h0_block(), n_shots=qarp.EXACT, initial_state=psi)
    eng = QarpEngine()
    eng.build([sampler])
    dist = eng.run()[0]

    for idx, p in enumerate(expected):
        bits = ((idx >> 0) & 1, (idx >> 1) & 1)
        assert dist.get(bits, 0.0) == pytest.approx(p, abs=ATOL)


def test_sampler_finite_shots_from_seeded_state():
    psi = np.array([0.6, 0.0, 0.0, 0.8], dtype=complex)
    n_shots = 20_000
    sampler = Sampler(ket=SimpleBlock(2).build(), n_shots=n_shots, initial_state=psi)
    eng = QarpEngine(seed=7)
    eng.build([sampler])
    dist = eng.run()[0]

    for bits, p in (((0, 0), 0.36), ((1, 1), 0.64)):
        sigma = np.sqrt(p * (1 - p) / n_shots)
        assert abs(dist.get(bits, 0.0) - p) < 5 * sigma


def test_seed_is_mutable_between_runs():
    """Step-loop shape: build once, re-seed per run."""
    sampler = Sampler(ket=SimpleBlock(1).build(), n_shots=qarp.EXACT)
    eng = QarpEngine()
    eng.build([sampler])
    assert eng.run()[0] == pytest.approx({(0,): 1.0}, abs=ATOL)
    sampler.initial_state = np.array([0, 1], dtype=complex)
    assert eng.run()[0] == pytest.approx({(1,): 1.0}, abs=ATOL)


# ── StateVector through QarpEngine ───────────────────────────────────────


def test_statevector_expectation_from_seeded_state():
    psi = _random_state(1)
    theta = 0.4
    ket = SimpleBlock(1)
    ket.ry(0, theta)
    expected = np.vdot(_ry(theta) @ psi, Z @ (_ry(theta) @ psi)).real

    prim = StateVector(ket=ket, operator=QubitOperator("Z0"), initial_state=psi)
    eng = QarpEngine()
    eng.build([prim])
    assert eng.run()[0] == pytest.approx(expected, abs=ATOL)


def test_statevector_overlap_seeds_ket_only():
    """⟨bra|ket⟩ with a seeded ket: the bra stays |0…0⟩-rooted."""
    psi = _random_state(1)
    ket = SimpleBlock(1)
    ket.ry(0, 0.8)
    bra = SimpleBlock(1)
    bra.h(0)
    expected = np.vdot(H @ np.array([1, 0], dtype=complex), _ry(0.8) @ psi)

    prim = StateVector(ket=ket, bra=bra, initial_state=psi)
    eng = QarpEngine()
    eng.build([prim])
    assert eng.run()[0] == pytest.approx(expected, abs=ATOL)


# ── rejections (contract) ────────────────────────────────────────────────


def test_base_engine_rejects_seeded_primitive():
    """``supports_initial_state`` is ``False`` on the base; an engine that does
    not opt in must reject a seeded primitive at ``build()`` (§14)."""
    sampler = Sampler(ket=_h0_block(), n_shots=100, initial_state=_random_state(2))
    with pytest.raises(CapabilityError, match="cannot seed its register"):
        _StubEngine().build([sampler])


def test_routed_device_rejects_seeded_primitive():
    b = SimpleBlock(4)
    b.ry(0, 0.7)
    b.cx(0, 3)  # 0–3 not adjacent on the 2×2 grid → routing runs
    sampler = Sampler(ket=b, n_shots=qarp.EXACT, initial_state=_random_state(4))
    arch = get_nearest_neighbour_architecture(2, 2)
    eng = QarpEngine(device=Device(4, architecture=arch))
    with pytest.raises(CapabilityError, match="logical qubit order"):
        eng.build([sampler])


def test_batch_run_finite_shots_rejects_seeded_primitive():
    sampler = Sampler(ket=_h0_block(), initial_state=_random_state(2))
    eng = QarpEngine()
    with pytest.raises(CapabilityError, match="batch_run"):
        eng.batch_run([sampler], [{}], n_shots=100)


def test_routed_run_rejects_post_build_initial_state_mutation():
    """Build unseeded on a routed device passes; seeding afterwards must
    raise at run() instead of consuming amplitudes in physical order."""
    b = SimpleBlock(4)
    b.ry(0, 0.7)
    b.cx(0, 3)  # 0–3 not adjacent on the 2×2 grid → routing runs
    sampler = Sampler(ket=b, n_shots=qarp.EXACT)
    arch = get_nearest_neighbour_architecture(2, 2)
    eng = QarpEngine(device=Device(4, architecture=arch))
    eng.build([sampler])
    sampler.initial_state = _random_state(4)
    with pytest.raises(CapabilityError, match="logical qubit order"):
        eng.run()


def test_routed_batch_run_norebuild_rejects_post_build_mutation():
    """rebuild=False reuses compiled circuits — the routed initial_state guard
    must still fire on the EXACT branch."""
    b = SimpleBlock(4)
    b.ry(0, 0.7)
    b.cx(0, 3)
    sampler = Sampler(ket=b, n_shots=qarp.EXACT)
    arch = get_nearest_neighbour_architecture(2, 2)
    eng = QarpEngine(device=Device(4, architecture=arch))
    eng.build([sampler])
    sampler.initial_state = _random_state(4)
    with pytest.raises(CapabilityError, match="logical qubit order"):
        eng.batch_run([sampler], [{}], rebuild=False)


def test_other_primitives_reject_the_kwarg():
    from qarp.algorithms import HadamardTest

    with pytest.raises(TypeError):
        HadamardTest(initial_state=_random_state(1))


def test_attribute_assignment_on_non_accepting_primitive_is_rejected():
    """The restriction is enforced at the consumption point, not just the
    constructor: assigning ``ht.initial_state = psi`` must be caught, not
    pass every engine check and silently corrupt the ancilla-based estimate."""
    from qarp.algorithms import HadamardTest

    ht = HadamardTest(bra=_h0_block(), ket=_h0_block(), n_shots=100)
    ht.initial_state = _random_state(3)
    with pytest.raises(CapabilityError, match="does not accept initial_state"):
        QarpEngine().build([ht])


# ── seeded gradients (adjoint backprop, finite-difference oracles) ───────


def test_gradient_of_seeded_primitive_matches_finite_differences():
    psi = _random_state(1)
    theta = Symbol("theta")
    ket = SimpleBlock(1)
    ket.ry(0, theta)
    prim = StateVector(ket=ket, operator=QubitOperator("Z0"), initial_state=psi)
    eng = QarpEngine()
    eng.build([prim])

    val = 0.3
    (grad,) = eng.run_gradient({theta: val})

    eps = 1e-6
    fd = (eng.run({theta: val + eps})[0] - eng.run({theta: val - eps})[0]) / (2 * eps)
    assert grad[0] == pytest.approx(fd, abs=1e-5)


def test_seeded_shared_symbol_ev_gradient_matches_finite_differences():
    """θ appears in two gates: the 2-term parameter shift is invalid here, so
    this fails under the old seeded→shift demotion and requires the seeded
    adjoint kernel."""
    psi = _random_state(2)
    theta = Symbol("theta")
    ket = SimpleBlock(2)
    ket.ry(0, theta)
    ket.cx(0, 1)
    ket.ry(1, theta)
    prim = StateVector(ket=ket, operator=QubitOperator("Z1"), initial_state=psi)
    eng = QarpEngine()
    eng.build([prim])

    val = 0.37
    (grad,) = eng.run_gradient({theta: val})

    eps = 1e-6
    fd = (eng.run({theta: val + eps})[0] - eng.run({theta: val - eps})[0]).real / (2 * eps)
    assert grad[0] == pytest.approx(fd, abs=1e-6)


def test_seeded_overlap_gradient_is_d_abs_overlap_squared():
    """Seeded OVERLAP must return the documented ∂|⟨bra|ψ⟩|²/∂θ; the old
    demotion returned the parameter shift of Re⟨bra|ψ⟩ — a different scalar."""
    psi = _random_state(1)
    theta = Symbol("theta")
    ket = SimpleBlock(1)
    ket.ry(0, theta)
    bra = SimpleBlock(1)
    bra.h(0)
    prim = StateVector(ket=ket, bra=bra, initial_state=psi)
    eng = QarpEngine()
    eng.build([prim])

    val = 0.53
    (grad,) = eng.run_gradient({theta: val})

    eps = 1e-6
    op = abs(eng.run({theta: val + eps})[0]) ** 2
    om = abs(eng.run({theta: val - eps})[0]) ** 2
    fd = (op - om) / (2 * eps)
    assert grad[0] == pytest.approx(fd, abs=1e-6)


# ── The routed guard keys on the *initial* placement, not the final map ──────


def test_routed_guard_names_the_initial_placement():
    """The refusal is about where qubits are placed, not where they end."""
    b = SimpleBlock(4)
    b.ry(0, 0.7)
    b.cx(0, 3)  # 0–3 not adjacent on the 2×2 grid → the router places them adjacent
    sampler = Sampler(ket=b, n_shots=qarp.EXACT, initial_state=_random_state(4))
    arch = get_nearest_neighbour_architecture(2, 2)
    eng = QarpEngine(device=Device(4, architecture=arch))
    with pytest.raises(CapabilityError, match="initial_logical_to_physical"):
        eng.build([sampler])


def test_routed_guard_accepts_identity_placement_even_when_swaps_move_the_register():
    """Seeding is safe whenever the placement is the identity: the seed lands on
    the right wires, and the final map only reindexes the readout."""
    from qarp.engines._qarp_engine import _Layout, _reject_routed_initial_state

    sampler = Sampler(ket=_h0_block(), n_shots=qarp.EXACT, initial_state=_random_state(2))
    _reject_routed_initial_state(sampler, None)
    _reject_routed_initial_state(sampler, _Layout(initial=None, final=[1, 0]))
    with pytest.raises(CapabilityError, match="initial_logical_to_physical"):
        _reject_routed_initial_state(sampler, _Layout(initial=[1, 0], final=None))
