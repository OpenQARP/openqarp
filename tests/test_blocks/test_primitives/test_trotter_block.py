"""Regression tests for TrotterBlock.

Focus: the constant-(identity) term branch of ``TrotterBlock.build_vanilla``,
which is easy to leave unexercised by zeroing it out (``qham.terms[()] = 0``).
The convention contract is:

    Trotterized U(t) ≈ exp(-i H t)

so the constant term ``c·I`` must contribute ``exp(-i c t)`` to the global
phase — *not* ``exp(+i c t)``.
"""

import random

import numpy as np
import pytest
from scipy.linalg import expm
from sympy import Symbol

import qarpx as qx
from qarp.blocks import ControlledBlock, TrotterBlock
from qarp.operators import (
    FullyCommuting,
    GroupingStrategy,
    NoGrouping,
    QubitOperator,
    QubitWiseCommuting,
)

# Physics assertions that must hold under any partition are parametrised over
# these; `None` is the default (FullyCommuting).
_GROUPINGS = [None, NoGrouping(), FullyCommuting()]


def _unitary(block):
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


def _exact_evolution(op: QubitOperator, n_qubits: int, t: float) -> np.ndarray:
    """Reference exp(-i H t) in qarpx LSB ordering (``sparse_matrix``)."""
    H = np.array(op.sparse_matrix(n_qubits).todense())
    return expm(-1j * H * t)


# ── Constant-only Hamiltonian ───────────────────────────────────────────


@pytest.mark.parametrize("c", [0.0, 0.137, -0.42, 1.7])
@pytest.mark.parametrize("t", [0.0, 0.5, 1.0])
def test_constant_only_hamiltonian_emits_exp_minus_ict(c, t):
    """H = c·I on n qubits should yield U = exp(-i c t) · I."""
    op = QubitOperator("", c)
    n_qubits = 2
    b = TrotterBlock(operator=op, n_qubits=n_qubits, steps=1, time=t).build()
    expected = np.exp(-1j * c * t) * np.eye(2**n_qubits, dtype=complex)
    assert np.linalg.norm(_unitary(b) - expected) < 1e-12


# ── Constant + single Pauli (composition with Pauli-exp branch) ─────────


def test_constant_plus_single_z_composes_correctly():
    """H = c·I + a·Z₀ should produce exp(-i(c·I + a·Z₀)·t).

    Verifies that the constant gphase and the Pauli-exp branch use the
    same sign convention.
    """
    from scipy.linalg import expm

    c, a, t = 0.3, 0.7, 0.5
    op = QubitOperator("", c) + QubitOperator("Z0", a)
    b = TrotterBlock(operator=op, n_qubits=1, steps=1, time=t).build()

    Z = np.array([[1, 0], [0, -1]], complex)
    I = np.eye(2, dtype=complex)
    expected = expm(-1j * (c * I + a * Z) * t)
    assert np.linalg.norm(_unitary(b) - expected) < 1e-12


# ── Constant term respects steps (it does not step-multiply: it's emitted
#     once at the end of build_vanilla, since constant·t is exact). ────


def test_constant_term_emitted_once_regardless_of_steps():
    """The identity branch is exact at any step count — emitting it once
    at the end with angle ``c·t`` is correct."""
    c, t = 0.5, 1.0
    op = QubitOperator("", c)
    U1 = _unitary(TrotterBlock(operator=op, n_qubits=1, steps=1, time=t).build())
    U5 = _unitary(TrotterBlock(operator=op, n_qubits=1, steps=5, time=t).build())
    assert np.linalg.norm(U1 - U5) < 1e-12
    expected = np.exp(-1j * c * t) * np.eye(2, dtype=complex)
    assert np.linalg.norm(U1 - expected) < 1e-12


def test_constant_term_respects_imaginary_flag():
    """When ``imaginary=True``, the constant ``a + b·j`` must contribute the
    *imaginary* part ``b`` to the global phase, not the real part — matching
    how the Pauli-term coefficients are coerced."""
    a, b, t = 0.3, 0.7, 0.5
    op = QubitOperator("", complex(a, b))

    U_real = _unitary(
        TrotterBlock(operator=op, n_qubits=1, steps=1, time=t, imaginary=False).build()
    )
    U_imag = _unitary(
        TrotterBlock(operator=op, n_qubits=1, steps=1, time=t, imaginary=True).build()
    )
    expected_real = np.exp(-1j * a * t) * np.eye(2, dtype=complex)
    expected_imag = np.exp(-1j * b * t) * np.eye(2, dtype=complex)
    assert np.linalg.norm(U_real - expected_real) < 1e-12
    assert np.linalg.norm(U_imag - expected_imag) < 1e-12


# ── Higher-order Suzuki–Yoshida recursion (order ∈ {2, 4, 6, 8}) ─────────
#
# ``_arbitrary_order_sequence`` implements the Suzuki 5-term recursion
#     S_{p}(t) = S_{p-2}(u·t)^2 · S_{p-2}((1-4u)·t) · S_{p-2}(u·t)^2
# with u = 1/(4 − 4^(1/(p−1))).  The mathematical guarantee is order-``p``
# accuracy: ``‖U_trotter(t,s) − exp(−iHt)‖ = O(t^(p+1) / s^p)`` in the
# small-t / large-s regime.  Tests below pin that scaling.


@pytest.fixture
def _noncommuting_hamiltonian():
    """A non-commuting 2-qubit Hamiltonian (each pair of terms anticommutes
    on at least one qubit).  Coefficients chosen unequal to avoid accidental
    symmetries that flatten the Trotter error."""
    return QubitOperator("X0", 0.37) + QubitOperator("Z0 Z1", 0.51) + QubitOperator("Y0 Y1", 0.42)


def _trotter_error(op, n_qubits, t, steps, order, grouping=None):
    """‖U_trotter(t, steps, order) − exp(−iHt)‖₂ for the given Hamiltonian."""
    block = TrotterBlock(
        operator=op,
        n_qubits=n_qubits,
        steps=steps,
        time=t,
        order=order,
        grouping=grouping,
    ).build()
    U = _unitary(block)
    return np.linalg.norm(U - _exact_evolution(op, n_qubits, t))


_ORDER_TIME_GRID = {
    # Time grids chosen per order so err(t) lies in a clean asymptotic band:
    # well above the ~1e-13 machine-precision floor and well below 1
    # (where the leading-order Taylor truncation dominates).  Each range
    # spans ~3× in t to give the log-log fit usable lever-arm.
    2: np.array([0.05, 0.04, 0.03, 0.025, 0.02]),
    4: np.array([0.20, 0.16, 0.13, 0.10, 0.08]),
    6: np.array([0.45, 0.40, 0.35, 0.30, 0.25]),
}


@pytest.mark.parametrize("order, expected_slope", [(2, 3), (4, 5), (6, 7)])
@pytest.mark.parametrize("grouping", _GROUPINGS)
def test_high_order_convergence_rate(_noncommuting_hamiltonian, order, expected_slope, grouping):
    """Order-p Trotter error scales as O(t^(p+1)) at fixed steps=1, small t.

    Fits log(err) vs log(t) and asserts slope ≈ p+1.  The Suzuki recursion
    must produce the textbook accuracy order — this catches recursion math
    bugs (wrong sub-time coefficients, wrong composition order, missing
    leg) that the constant-term tests never see.
    """
    op = _noncommuting_hamiltonian
    times = _ORDER_TIME_GRID[order]
    errs = np.array(
        [_trotter_error(op, 2, t, steps=1, order=order, grouping=grouping) for t in times]
    )
    slope, _ = np.polyfit(np.log(times), np.log(errs), 1)
    # Tolerance: ±0.5 absorbs finite-precision noise and finite-Hamiltonian
    # higher-order correction terms at the small but not infinitesimal t
    # used here.
    assert abs(slope - expected_slope) < 0.5, (
        f"order={order} grouping={grouping!r}: slope={slope:.3f}, expected≈{expected_slope}"
    )


@pytest.mark.parametrize("grouping", _GROUPINGS)
def test_higher_order_beats_lower_order(_noncommuting_hamiltonian, grouping):
    """At matched (t, steps) in the small-t regime, err(6) < err(4) < err(2)."""
    op = _noncommuting_hamiltonian
    t, steps = 0.3, 1
    e2 = _trotter_error(op, 2, t, steps, order=2, grouping=grouping)
    e4 = _trotter_error(op, 2, t, steps, order=4, grouping=grouping)
    e6 = _trotter_error(op, 2, t, steps, order=6, grouping=grouping)
    assert e4 < e2, f"grouping={grouping!r}: e4={e4:.3e} ≥ e2={e2:.3e}"
    assert e6 < e4, f"grouping={grouping!r}: e6={e6:.3e} ≥ e4={e4:.3e}"


def test_order_6_matches_manual_suzuki_recursion(_noncommuting_hamiltonian):
    """Order 6 = Suzuki S_6(t) = S_4(u·t)² · S_4((1−4u)·t) · S_4(u·t)² with
    u₆ = 1/(4 − 4^(1/5)).  Build that composition by hand from order-4
    TrotterBlocks and pin equality with the order=6 build.

    This cross-checks the recursion math (reduction coefficient, outer/inner
    composition shape) independent of the convergence-rate fit.
    """
    op = _noncommuting_hamiltonian
    t = 0.4

    u6 = 1.0 / (4.0 - 4.0 ** (1.0 / 5.0))
    sub_times = [u6 * t, u6 * t, (1.0 - 4.0 * u6) * t, u6 * t, u6 * t]
    # Build the order-4 sub-unitaries and multiply right-to-left (the qarpx
    # circuit applies the first emitted gate first, so its matrix product
    # is reversed vs the build-order list).
    U_manual = np.eye(4, dtype=complex)
    for ts in sub_times:
        sub = TrotterBlock(
            operator=op, n_qubits=2, steps=1, time=ts, order=4, grouping=NoGrouping()
        ).build()
        U_manual = _unitary(sub) @ U_manual

    U_block = _unitary(
        TrotterBlock(
            operator=op, n_qubits=2, steps=1, time=t, order=6, grouping=NoGrouping()
        ).build()
    )
    assert np.linalg.norm(U_block - U_manual) < 1e-10


@pytest.mark.filterwarnings("ignore:Trotter order=.*:RuntimeWarning")
def test_order_8_higher_than_order_6(_noncommuting_hamiltonian):
    """Order 8 (one level deeper in the recursion) must still produce a
    sensible error.  At small t, err(8) ≤ err(6) — drop a regression marker
    so a bad recursion at p>6 trips even without an explicit slope fit."""
    op = _noncommuting_hamiltonian
    t = 0.2
    e6 = _trotter_error(op, 2, t, steps=1, order=6)
    e8 = _trotter_error(op, 2, t, steps=1, order=8)
    assert e8 <= e6 + 1e-12


@pytest.mark.parametrize("bad_order", [3, 5, 7])
def test_odd_order_gt_one_rejected(_noncommuting_hamiltonian, bad_order):
    """Odd orders > 1 have no defined Suzuki recursion — must raise at
    construction time (before any build)."""
    op = _noncommuting_hamiltonian
    with pytest.raises(ValueError, match="order must be 1 or an even integer"):
        TrotterBlock(operator=op, n_qubits=2, steps=1, time=0.5, order=bad_order)


@pytest.mark.parametrize("bad_order", [0, -1, 1.5])
def test_invalid_order_rejected(_noncommuting_hamiltonian, bad_order):
    """Order must be a positive int (1 or even >= 2); reject 0, negatives,
    and non-integers at construction time."""
    op = _noncommuting_hamiltonian
    with pytest.raises(ValueError, match="order must be 1 or an even integer"):
        TrotterBlock(operator=op, n_qubits=2, steps=1, time=0.5, order=bad_order)


def test_order_8_emits_runtime_warning(_noncommuting_hamiltonian):
    """Suzuki recursion at order >= 8 unfolds into many sub-blocks; the
    block emits a RuntimeWarning so users notice the gate-count cliff."""
    op = _noncommuting_hamiltonian
    with pytest.warns(RuntimeWarning, match="Trotter order=8"):
        TrotterBlock(operator=op, n_qubits=2, steps=1, time=0.5, order=8)


# ── First-order convergence with steps ──────────────────────────────────
#
# The classical Trotter convergence statement: at fixed time, the
# first-order Lie-product error scales as O(t²/s), so doubling the step
# count should roughly halve the error.  Pin monotone decrease plus a
# small absolute bound at the largest step count.


def test_first_order_convergence_with_steps(_noncommuting_hamiltonian):
    """Order-1 error decreases monotonically as ``steps`` increases."""
    op = _noncommuting_hamiltonian
    t = 1.0
    errs = []
    for s in [1, 2, 4, 8, 16]:
        block = TrotterBlock(operator=op, n_qubits=2, steps=s, time=t, order=1).build()
        errs.append(np.linalg.norm(_unitary(block) - _exact_evolution(op, 2, t)))
    for i in range(1, len(errs)):
        assert errs[i] < errs[i - 1], f"step {i}: {errs[i]:.3e} ≥ {errs[i - 1]:.3e}"
    assert errs[-1] < 0.05, (
        f"order-1 with steps=16 should drive error below 0.05; got {errs[-1]:.3e}"
    )


# ── Controlled Trotter via ControlledBlock ───────────────────────────────
#
# qarpx routes "controlled X" through the ``ControlledBlock`` wrapper
# rather than the ``n_controls`` constructor kwarg (which is metadata only).
# The invariant ``C[True](U) · C[False](U) = I_ctrl ⊗ U`` holds because
# the two ControlledBlocks act on disjoint control sub-blocks and together
# cover the entire control register.


def _lifted_identity_tensor_inner(U_inner: np.ndarray, n_controls: int) -> np.ndarray:
    """Reference matrix for ``I_control ⊗ U_inner`` in qarpx LSB ordering.

    qarpx places controls at the lowest qubit indices (q0..q_{n_controls-1}
    inside the ControlledBlock — see ``hadamard_test_block.py``).  In LSB
    ordering this puts the control register at the low bits, so the lifted
    matrix is ``np.kron(U_inner, I_{2**n_controls})``.
    """
    return np.kron(U_inner, np.eye(2**n_controls, dtype=complex))


def test_single_control_reconstructs_uncontrolled_z_only():
    """``C[True](U) · C[False](U) = I_control ⊗ U`` — sequencing both
    control states recovers the uncontrolled operator on every fiber.

    Restricted to a Z-only Hamiltonian so the inner Trotter circuit emits
    only ``CX`` and ``Rz`` (no ``H`` basis change).  qarpx's controlled-H
    decomposition is currently buggy (controlled-H sign
    follow-up), so general (X/Y-bearing) Hamiltonians break this invariant
    until that fix lands.
    """
    op = QubitOperator("Z0", 0.3) + QubitOperator("Z1", 0.5) + QubitOperator("Z0 Z1", 0.4)
    t = 0.4
    inner = TrotterBlock(operator=op, n_qubits=2, steps=2, time=t, order=2).build()
    U_inner = _unitary(inner)

    c_true = ControlledBlock(inner, num_controls=1, ctrl_state=[True])
    c_true.build()
    c_false = ControlledBlock(inner, num_controls=1, ctrl_state=[False])
    c_false.build()

    combined = _unitary(c_true) @ _unitary(c_false)  # disjoint blocks — order is immaterial
    expected = _lifted_identity_tensor_inner(U_inner, n_controls=1)
    assert np.linalg.norm(combined - expected) < 1e-10


def test_single_control_reconstructs_uncontrolled_general(_noncommuting_hamiltonian):
    """Same invariant as the Z-only variant, with an inner Hamiltonian that
    exercises ``H``/``Sdg`` basis changes. Locks in the qarpx C-H decomposition."""
    op = _noncommuting_hamiltonian
    t = 0.4
    inner = TrotterBlock(operator=op, n_qubits=2, steps=2, time=t, order=2).build()
    U_inner = _unitary(inner)

    c_true = ControlledBlock(inner, num_controls=1, ctrl_state=[True])
    c_true.build()
    c_false = ControlledBlock(inner, num_controls=1, ctrl_state=[False])
    c_false.build()

    combined = _unitary(c_true) @ _unitary(c_false)
    expected = _lifted_identity_tensor_inner(U_inner, n_controls=1)
    assert np.linalg.norm(combined - expected) < 1e-10


def test_two_controls_all_states_reconstruct_uncontrolled(_noncommuting_hamiltonian):
    """For ``n_controls=2``, sequencing all four control-state polarities
    ``[T,T] · [T,F] · [F,T] · [F,F]`` reconstructs ``I_ctrl ⊗ U``.

    Locks in the auto-transpile path for multi-control: any inner block
    (here a non-commuting H with X/Y/Z/CX terms) is lowered to the
    multi-control basis (``{X, Y, Z, Rx, Ry, Rz, P, CX, GPhase, Barrier}``)
    in ``ControlledBlock::flatten`` before wrapping.
    """
    op = _noncommuting_hamiltonian
    t = 0.3
    inner = TrotterBlock(operator=op, n_qubits=2, steps=2, time=t, order=2).build()
    U_inner = _unitary(inner)

    states = [[True, True], [True, False], [False, True], [False, False]]
    combined = np.eye(2 ** (inner.n_qubits + 2), dtype=complex)
    for s in states:
        cb = ControlledBlock(inner, num_controls=2, ctrl_state=s)
        cb.build()
        combined = _unitary(cb) @ combined

    expected = _lifted_identity_tensor_inner(U_inner, n_controls=2)
    assert np.linalg.norm(combined - expected) < 1e-10


# ── Symbolic time ────────────────────────────────────────────────────────
#
# ``time=Symbol("t")`` defers angle substitution to ``set_time(value)``.
# Under the qarpx convention ``set_time(v)`` substitutes ``v`` directly
# (no π factor — that was a pytket half-turn artefact from old qarp).


def test_symbolic_time_substitution_matches_concrete(_noncommuting_hamiltonian):
    """``set_time(v)`` on a symbolic-time block matches a fresh build
    with ``time=v`` to floating-point precision."""
    op = _noncommuting_hamiltonian
    t_val = 0.5
    sym = TrotterBlock(operator=op, n_qubits=2, steps=2, time=Symbol("t"), order=2).build()
    concrete = TrotterBlock(operator=op, n_qubits=2, steps=2, time=t_val, order=2).build()

    substituted = sym.set_time(t_val)

    assert np.linalg.norm(_unitary(substituted) - _unitary(concrete)) < 1e-10


def test_symbolic_time_physical_correctness(_noncommuting_hamiltonian):
    """After ``set_time``, the substituted block evolves states under
    ``exp(-i H t_val)`` to within the expected Trotter error."""
    op = _noncommuting_hamiltonian
    t_val = 0.1
    sym = TrotterBlock(operator=op, n_qubits=2, steps=4, time=Symbol("t"), order=2).build()
    substituted = sym.set_time(t_val)
    err = np.linalg.norm(_unitary(substituted) - _exact_evolution(op, 2, t_val))
    # order=2 with steps=4 at t=0.1 gives error ~ t^3/s^2 ≈ 6e-5; bound at 0.01.
    assert err < 0.01, f"symbolic-time set_time correctness: err={err:.3e}"


def test_set_time_requires_symbolic_time(_noncommuting_hamiltonian):
    """``set_time`` on a concrete-time block raises — caller picked the
    wrong API."""
    op = _noncommuting_hamiltonian
    block = TrotterBlock(operator=op, n_qubits=2, steps=1, time=0.5, order=1).build()
    with pytest.raises(ValueError, match="not built with a symbolic time"):
        block.set_time(0.7)


# ── Foundational: commuting Hamiltonian → Trotter is exact ──────────────


# ── Block.optimize() peephole pipeline ──────────────────────────────────


def test_optimize_reduces_fh_trotter_command_count(_noncommuting_hamiltonian):
    """``Block.optimize()`` runs the standard transpile + identity-elimination
    + single-qubit fusion pipeline.  On a representative non-commuting
    Hamiltonian (3 Pauli terms, order=2 Trotter) the optimizer cuts the
    command count meaningfully; pin a ratio threshold rather than the exact
    number to allow harmless small drifts in future decomposition tweaks."""
    op = _noncommuting_hamiltonian
    block = TrotterBlock(operator=op, n_qubits=2, steps=1, time=0.5, order=2).build()
    n_orig = len(block.flatten())
    optimized = block.optimize()
    n_opt = len(optimized.flatten())
    assert n_opt < n_orig, f"optimize did not reduce: {n_orig} → {n_opt}"
    # Sanity: at least 20% reduction on this case (observed ~25-30%).
    assert n_opt <= int(n_orig * 0.85), f"optimize reduced less than 15%: {n_orig} → {n_opt}"


def test_optimize_preserves_unitary(_noncommuting_hamiltonian):
    """Whatever the transpile + peephole pipeline does, the resulting block's
    unitary must equal the original's to floating-point precision (the
    invariant the engines rely on)."""
    op = _noncommuting_hamiltonian
    block = TrotterBlock(operator=op, n_qubits=2, steps=2, time=0.4, order=4).build()
    optimized = block.optimize()
    assert np.linalg.norm(_unitary(block) - _unitary(optimized)) < 1e-10


def test_optimize_returns_built_simpleblock(_noncommuting_hamiltonian):
    """The returned object is a fresh built SimpleBlock with the local qubit
    frame — ready for ``flatten()`` and engine consumption without further
    setup."""
    op = _noncommuting_hamiltonian
    block = TrotterBlock(operator=op, n_qubits=2, steps=1, time=0.3, order=2).build()
    optimized = block.optimize()
    assert optimized.is_built
    assert optimized.n_qubits == block.n_qubits
    assert list(optimized.target_qubits) == [0, 1]
    # Receiver is not mutated.
    assert block is not optimized


def test_optimize_on_unbuilt_block_raises(_noncommuting_hamiltonian):
    """``optimize()`` requires a built block — flatten() is undefined otherwise."""
    op = _noncommuting_hamiltonian
    block = TrotterBlock(operator=op, n_qubits=2, steps=1, time=0.3, order=1)
    with pytest.raises(RuntimeError, match="not built"):
        block.optimize()


def test_optimize_levels(_noncommuting_hamiltonian):
    """``level=`` selects the optimization tier: 0 = transpile only,
    1 = wire-adjacent cancellation + fusion (default), 2 = + commutation-aware
    cancellation.  Reductions are monotone, every level preserves the
    unitary, and invalid levels raise."""
    op = _noncommuting_hamiltonian
    block = TrotterBlock(operator=op, n_qubits=2, steps=2, time=0.4, order=2).build()

    n0 = len(block.optimize(level=0).flatten())
    n1 = len(block.optimize(level=1).flatten())
    n2 = len(block.optimize(level=2).flatten())
    assert n0 >= n1 >= n2
    assert n1 < n0  # this circuit has known peephole wins

    for level in (0, 1, 2):
        optimized = block.optimize(level=level)
        assert np.linalg.norm(_unitary(block) - _unitary(optimized)) < 1e-10

    with pytest.raises(ValueError, match="level must be 0, 1, or 2"):
        block.optimize(level=3)


def test_all_commuting_hamiltonian_is_exact_at_any_steps():
    """When all terms in H mutually commute, Trotter ``= exp(-iHt)`` for
    any positive ``steps`` and order.  This pins the commuting-term path
    independent of the Suzuki-recursion correctness."""
    # All-Z Hamiltonian — every term pairwise commutes.
    op = QubitOperator("Z0", 0.3) + QubitOperator("Z1", 0.5) + QubitOperator("Z0 Z1", 0.4)
    t = 0.7
    exact = _exact_evolution(op, 2, t)
    for steps in [1, 3]:
        for order in [1, 2, 4]:
            for grouping in _GROUPINGS:
                U = _unitary(
                    TrotterBlock(
                        operator=op,
                        n_qubits=2,
                        steps=steps,
                        time=t,
                        order=order,
                        grouping=grouping,
                    ).build()
                )
                err = np.linalg.norm(U - exact)
                assert err < 1e-10, (
                    f"commuting H, steps={steps}, order={order}, {grouping!r}: err={err:.3e}"
                )


# ── Yoshida composition (order ∈ {6, 8}) ────────────────────────────────
#
# ``composition="yoshida"`` opts into the literature-tuned weight chain
# ``∏_j S₂(w_j · t)`` (Yoshida 1990 for order 6, Morales 2025 for order 8).
# Cheaper than Suzuki at the same order with smaller leading-error constant;
# only valid at orders 6 and 8.


@pytest.mark.parametrize("order", [6, 8])
@pytest.mark.parametrize("c, t", [(0.0, 0.5), (0.137, 0.5), (-0.42, 0.7)])
def test_yoshida_constant_only_global_phase(order, c, t):
    """Constant ``c·I`` contributes the same ``exp(-i c t)`` global phase
    under the Yoshida composition as under Suzuki."""
    op = QubitOperator("", c)
    block = TrotterBlock(
        operator=op, n_qubits=2, steps=1, time=t, order=order, composition="yoshida"
    ).build()
    expected = np.exp(-1j * c * t) * np.eye(4, dtype=complex)
    assert np.linalg.norm(_unitary(block) - expected) < 1e-10


@pytest.mark.parametrize("order", [6, 8])
@pytest.mark.parametrize("grouping", _GROUPINGS)
@pytest.mark.parametrize(
    "op_str, coeff",
    [("Z0", 0.5), ("X0", 0.3), ("Y0", -0.4), ("Z0 Z1", 0.7), ("X0 Y1", 0.6)],
)
def test_yoshida_single_pauli_term_exact(order, grouping, op_str, coeff):
    """A single-Pauli H has no commutators — Yoshida is exact at any order/steps."""
    op = QubitOperator(op_str, coeff)
    t = 0.4
    block = TrotterBlock(
        operator=op,
        n_qubits=2,
        steps=1,
        time=t,
        order=order,
        composition="yoshida",
        grouping=grouping,
    ).build()
    assert np.linalg.norm(_unitary(block) - _exact_evolution(op, 2, t)) < 1e-9


@pytest.mark.parametrize("order", [6, 8])
def test_yoshida_grouping_invariance_on_all_commuting_h(order):
    """All-Z Hamiltonian: grouped and ungrouped Yoshida chains must produce
    identical unitaries (commuting terms have no Trotter splitting error,
    and the chain shape collapses to the same product)."""
    op = QubitOperator("Z0", 0.3) + QubitOperator("Z1", 0.4) + QubitOperator("Z0 Z1", 0.2)
    t = 0.5
    b1 = TrotterBlock(
        operator=op,
        n_qubits=2,
        steps=1,
        time=t,
        order=order,
        composition="yoshida",
        grouping=FullyCommuting(),
    ).build()
    b2 = TrotterBlock(
        operator=op,
        n_qubits=2,
        steps=1,
        time=t,
        order=order,
        composition="yoshida",
        grouping=NoGrouping(),
    ).build()
    assert np.linalg.norm(_unitary(b1) - _unitary(b2)) < 1e-9


@pytest.mark.parametrize("grouping", _GROUPINGS)
def test_yoshida_6_better_than_first_order(grouping):
    """Yoshida 6 must produce a smaller error than 1st-order Lie split on a
    non-commuting Hamiltonian, even at modest step count."""
    op = QubitOperator("X0", 1.0) + QubitOperator("Z0", 0.5)
    t, steps = 0.3, 3
    ref = _exact_evolution(op, 1, t)

    b_first = TrotterBlock(
        operator=op,
        n_qubits=1,
        steps=steps,
        time=t,
        order=1,
        grouping=grouping,
    ).build()
    b_yosh = TrotterBlock(
        operator=op,
        n_qubits=1,
        steps=steps,
        time=t,
        order=6,
        composition="yoshida",
        grouping=grouping,
    ).build()
    err_first = np.max(np.abs(_unitary(b_first) - ref))
    err_yosh = np.max(np.abs(_unitary(b_yosh) - ref))
    assert err_yosh < err_first, (
        f"Yoshida 6 (err={err_yosh:.2e}) should beat first-order (err={err_first:.2e})"
    )


@pytest.mark.parametrize("order", [6, 8])
def test_yoshida_convergence_with_step_count(order):
    """Increasing ``steps`` must monotonically decrease the Yoshida Trotter error."""
    op = QubitOperator("X0", 0.7) + QubitOperator("Z0", -0.4)
    t = 0.5
    ref = _exact_evolution(op, 1, t)
    errors = []
    for steps in (1, 2, 4):
        block = TrotterBlock(
            operator=op,
            n_qubits=1,
            steps=steps,
            time=t,
            order=order,
            composition="yoshida",
            grouping=FullyCommuting(),
        ).build()
        errors.append(np.max(np.abs(_unitary(block) - ref)))
    for prev, curr in zip(errors[:-1], errors[1:], strict=True):
        assert curr <= prev + 1e-12, f"Yoshida error did not decrease: {errors}"


def test_yoshida_symbolic_time_substitution_matches_concrete():
    """``time=Symbol('t')`` + ``set_time(t_val)`` matches a direct build with
    ``time=t_val`` for the Yoshida composition (parity with Suzuki)."""
    op = QubitOperator("Z0 Z1", 0.3) + QubitOperator("X0", 0.4)
    t_val = 0.6
    b_sym = TrotterBlock(
        operator=op,
        n_qubits=2,
        steps=1,
        time=Symbol("t"),
        order=6,
        composition="yoshida",
    ).build()
    b_concrete = TrotterBlock(
        operator=op,
        n_qubits=2,
        steps=1,
        time=t_val,
        order=6,
        composition="yoshida",
    ).build()
    b_subbed = b_sym.set_time(t_val)
    assert np.linalg.norm(_unitary(b_subbed) - _unitary(b_concrete)) < 1e-12


@pytest.mark.parametrize("bad_order", [1, 2, 4, 10, 12])
def test_yoshida_rejects_unsupported_order(bad_order):
    """Yoshida weight tables exist only at orders 6 and 8."""
    with pytest.raises(ValueError, match="composition='yoshida' is only defined"):
        TrotterBlock(
            operator=QubitOperator("Z0", 1.0),
            n_qubits=1,
            order=bad_order,
            composition="yoshida",
        )


def test_invalid_composition_rejected():
    with pytest.raises(ValueError, match="composition must be"):
        TrotterBlock(
            operator=QubitOperator("Z0", 1.0),
            n_qubits=1,
            order=1,
            composition="strang",  # not a valid choice
        )


# ── Grouping-strategy injection ─────────────────────────────────────────


def test_fully_commuting_strategy_exact_on_commuting_hamiltonian():
    """XX+YY+ZZ mutually commute but are pairwise non-QWC: FullyCommuting
    packs them into one commuting_pauli_set (QWC keeps 3 singleton groups),
    and both stay exactly ``exp(-iHt)`` for one step."""
    from qarp.operators import FullyCommuting

    op = QubitOperator("X0 X1", 0.3) + QubitOperator("Y0 Y1", 0.5) + QubitOperator("Z0 Z1", 0.4)
    t = 0.7
    exact = _exact_evolution(op, 2, t)
    for strategy in (FullyCommuting(), QubitWiseCommuting()):
        U = _unitary(
            TrotterBlock(
                operator=op, n_qubits=2, steps=1, time=t, order=1, grouping=strategy
            ).build()
        )
        err = np.linalg.norm(U - exact)
        assert err < 1e-10, f"{strategy!r}: err={err:.3e}"


def test_grouping_default_is_fully_commuting():
    """``grouping=None`` uses general commutation — QWC would break number
    conservation, see the JW leak test."""
    from qarp.operators import FullyCommuting

    op = QubitOperator("X0 X1", 0.3) + QubitOperator("Z0", 0.5) + QubitOperator("Z0 Z1", 0.4)
    default = TrotterBlock(operator=op, n_qubits=2, steps=2, time=0.3, order=2)
    assert isinstance(default.grouping, FullyCommuting)
    explicit = TrotterBlock(
        operator=op, n_qubits=2, steps=2, time=0.3, order=2, grouping=FullyCommuting()
    ).build()
    assert [str(c) for c in default.build().flatten()] == [str(c) for c in explicit.flatten()]


def test_group_commuting_kwarg_is_gone():
    """``group_commuting`` was a two-valued projection of ``grouping``; the
    hard cut means the old spelling fails loudly, not silently."""
    with pytest.raises(TypeError, match="group_commuting"):
        TrotterBlock(
            operator=QubitOperator("X0 X1", 0.3),
            n_qubits=2,
            time=0.3,
            group_commuting=False,
        )


class _NonCommutingGrouping(GroupingStrategy):
    """Deliberately invalid: lumps every term into one group regardless of
    commutation.  Exercises the strategy contract, not a shipped strategy."""

    def group(self, terms, n_qubits):
        return [list(range(len(terms)))]


def test_non_commuting_group_is_rejected():
    """A strategy violating the ``GroupingStrategy`` contract (groups must be
    exponentiable together) is caught by ``commuting_pauli_set_exp`` rather
    than silently emitting a first-order product."""
    op = QubitOperator("X0", 0.3) + QubitOperator("Z0", 0.4)  # anticommute
    with pytest.raises(Exception, match="(?i)commut"):
        TrotterBlock(operator=op, n_qubits=1, time=0.3, grouping=_NonCommutingGrouping()).build()


# ── Grouping buys depth ──────────────────────────────────────────────────


def _gate_counts(block):
    counts: dict = {}
    for c in block.flatten():
        name = str(c).split()[0].split("(")[0]
        counts[name] = counts.get(name, 0) + 1
    return counts


def test_commuting_group_shares_one_basis_change():
    """{XX, YY, ZZ} on 2 qubits is mutually commuting, so the group costs one
    shared basis-change Clifford instead of three independent ladders.

    The termwise side is hand-derived exactly; the grouped side asserts the
    invariant (same rotations, strictly less overhead) rather than a literal
    Clifford size — the AG Clifford is an algorithmic output, and the planned
    graysynth/PMH replacement is expected to shrink it further.
    """
    op = QubitOperator("X0 X1", 0.3) + QubitOperator("Y0 Y1", 0.5) + QubitOperator("Z0 Z1", 0.4)
    grouped = _gate_counts(
        TrotterBlock(
            operator=op, n_qubits=2, steps=1, time=0.6, order=1, grouping=FullyCommuting()
        ).build()
    )
    termwise = _gate_counts(
        TrotterBlock(
            operator=op, n_qubits=2, steps=1, time=0.6, order=1, grouping=NoGrouping()
        ).build()
    )
    # Termwise, hand-derived: XX = 2 H + (CX, Rz, CX) + 2 H = 7;
    # YY = 2·(Sdg,H) + (CX, Rz, CX) + 2·(H,S) = 11; ZZ = (CX, Rz, CX) = 3.
    assert sum(termwise.values()) == 7 + 11 + 3
    assert termwise == {"H": 8, "Sdg": 2, "S": 2, "CX": 6, "Rz": 3}
    # One Rz per term either way — the rotations are the physics, the Clifford
    # is the overhead that grouping amortises.
    assert grouped["Rz"] == termwise["Rz"] == 3
    assert sum(grouped.values()) < sum(termwise.values())


def test_no_grouping_matches_the_termwise_ladder_cost():
    """``NoGrouping()`` must not pay a penalty for going through the
    commuting-set builder — its k=1 fast path is the plain Pauli ladder
    (pinned in C++ by ``test_synthesis_pauli_exp.cpp``)."""
    op = QubitOperator("X0 Y1", 0.3) + QubitOperator("Z0", 0.5)
    b = TrotterBlock(
        operator=op, n_qubits=2, steps=1, time=0.6, order=1, grouping=NoGrouping()
    ).build()
    # X0Y1: H + (Sdg,H) basis change either side (6) + 2 CX + 1 Rz = 9.
    # Z0:   1 Rz.
    assert len(list(b.flatten())) == 9 + 1


def _hamming_leak(U, n_qubits, occupied):
    """Weight of the evolved basis state outside its Hamming-weight sector."""
    idx = sum(1 << q for q in occupied)
    psi = U[:, idx]
    return sum(abs(psi[k]) ** 2 for k in range(2**n_qubits) if bin(k).count("1") != len(occupied))


def test_grouped_trotter_conserves_particle_number():
    """JW hopping pairs ``X_pX_q``/``Y_pY_q`` commute generally but not
    qubit-wise.  QWC grouping separates them and leaks Hamming weight at
    finite steps (~1e-2 on this H); the general-commuting default must keep
    each pair together and conserve particle number exactly.  Splitting them
    surfaces downstream as wrong-N determinants in TE-QSCI."""

    H = (
        0.5 * QubitOperator("X0 X1")
        + 0.5 * QubitOperator("Y0 Y1")
        + 0.5 * QubitOperator("X1 X2")
        + 0.5 * QubitOperator("Y1 Y2")
        + 0.3 * QubitOperator("Z0")
        + 0.2 * QubitOperator("Z2")
    )
    for steps in (1, 3):
        b = TrotterBlock(operator=H, n_qubits=3, steps=steps, time=0.8, order=1).build()
        leak = _hamming_leak(_unitary(b), 3, occupied=[0])
        assert leak < 1e-12, f"steps={steps}: leak={leak:.3e}"
    # Sanity: the QWC criterion really does leak on this H (pins that this
    # test would catch the regression).
    b = TrotterBlock(
        operator=H, n_qubits=3, steps=1, time=0.8, order=1, grouping=QubitWiseCommuting()
    ).build()
    assert _hamming_leak(_unitary(b), 3, occupied=[0]) > 1e-3


_JW_CHAIN_TERMS = [
    ("X0 X1", 0.5),
    ("Y0 Y1", 0.5),
    ("X1 X2", 0.5),
    ("Y1 Y2", 0.5),
    ("Z0", 0.3),
    ("Z2", 0.2),
]


def _chain_from(order):
    """The 3-site JW chain assembled in a caller-chosen term order.

    Same operator every time — only the order the terms are added differs,
    which is exactly what must not change the partition.
    """
    H = QubitOperator()
    for label, coeff in order:
        H += coeff * QubitOperator(label)
    return H


def _particle_number_operator(n_qubits):
    """JW particle number ``N = Σ_q (I − Z_q)/2``."""
    N = QubitOperator()
    for q in range(n_qubits):
        N += 0.5 * QubitOperator(()) - 0.5 * QubitOperator(f"Z{q}")
    return N


def _commutator(a, b):
    return a * b - b * a


def _group_sums(H, n_qubits, grouping):
    """Weighted Pauli sum of each group the strategy produces for ``H``."""
    items = [(t, c) for t, c in H.terms.items() if t]
    dicts = [{q: p for q, p in t} for t, _ in items]
    return [
        sum((items[i][1] * QubitOperator(items[i][0]) for i in grp), QubitOperator())
        for grp in grouping.group(dicts, n_qubits)
    ]


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_grouping_preserves_particle_number_symmetry_oracle(seed):
    """Analytic oracle for conservation, independent of any circuit.

    Each group is exponentiated *exactly* by ``commuting_pauli_set_exp``, so
    the Trotterised evolution commutes with a symmetry ``S`` iff every group's
    weighted sum does.  Checking ``[G, N] = 0`` per group is therefore the
    algebraic statement of "this partition cannot leak particle number" — it
    never builds a circuit, so it cannot be fooled by the emitter agreeing with
    itself.  Must hold whatever order the terms were written in.
    """
    shuffled = _JW_CHAIN_TERMS[:]
    random.Random(seed).shuffle(shuffled)
    H = _chain_from(shuffled)
    N = _particle_number_operator(3)

    for G in _group_sums(H, 3, FullyCommuting()):
        assert _commutator(G, N) == QubitOperator(), f"group {G} does not conserve N"


def test_symmetry_oracle_has_teeth():
    """The oracle must reject a partition that really does break the symmetry —
    otherwise the test above passes vacuously.  QWC splits the JW hopping pair,
    and a lone ``X_pX_q`` does not commute with N."""
    H = _chain_from(_JW_CHAIN_TERMS)
    N = _particle_number_operator(3)
    violations = [
        G for G in _group_sums(H, 3, QubitWiseCommuting()) if _commutator(G, N) != QubitOperator()
    ]
    assert violations, "QWC was expected to break particle-number conservation"


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_grouped_trotter_conserves_particle_number_under_any_term_order(seed):
    """End-to-end companion to the oracle: the circuit itself must not leak,
    however the Hamiltonian was written.

    Regression for the order-sensitivity of greedy first-fit — before
    ``canonical_term_order`` the default ``FullyCommuting`` grouped
    ``[X0X1, X1X2]`` and ``[Y0Y1, Y1Y2]`` on some orderings, splitting both
    hopping pairs and leaking ~1.2e-2 with no warning.
    """
    shuffled = _JW_CHAIN_TERMS[:]
    random.Random(seed).shuffle(shuffled)
    b = TrotterBlock(operator=_chain_from(shuffled), n_qubits=3, steps=1, time=0.8, order=1).build()
    assert _hamming_leak(_unitary(b), 3, occupied=[0]) < 1e-12


def test_partition_is_identical_across_term_orders():
    """The partition depends on the term *set*, not on how the operator was
    assembled — the property that makes the two tests above stable."""
    reference = None
    for seed in range(6):
        shuffled = _JW_CHAIN_TERMS[:]
        random.Random(seed).shuffle(shuffled)
        H = _chain_from(shuffled)
        items = [(t, c) for t, c in H.terms.items() if t]
        dicts = [{q: p for q, p in t} for t, _ in items]
        groups = FullyCommuting().group(dicts, 3)
        as_terms = sorted(sorted(items[i][0] for i in grp) for grp in groups)
        if reference is None:
            reference = as_terms
        assert as_terms == reference


def test_yoshida_path_accepts_fully_commuting_strategy():
    """Yoshida grouped emission goes through commuting_pauli_set_exp, which
    validates mutual commutation — a general-commuting (non-QWC) group must
    pass that validation and stay exact on a commuting H."""
    from qarp.operators import FullyCommuting

    op = QubitOperator("X0 X1", 0.3) + QubitOperator("Y0 Y1", 0.5) + QubitOperator("Z0 Z1", 0.4)
    t = 0.6
    exact = _exact_evolution(op, 2, t)
    U = _unitary(
        TrotterBlock(
            operator=op,
            n_qubits=2,
            steps=1,
            time=t,
            order=6,
            composition="yoshida",
            grouping=FullyCommuting(),
        ).build()
    )
    assert np.linalg.norm(U - exact) < 1e-8


def test_steps_replay_identical_command_sequences():
    """S-J: steps repeat one synthesized sequence — steps=3 at time=3 must be
    exactly three copies of the steps=1, time=1 stream (identical per-step
    angle factor 2·time/steps)."""
    op = qx.QubitOperator("X0 X1", 0.31) + qx.QubitOperator("Z0 Z1", 0.42)
    b1 = TrotterBlock(operator=op, n_qubits=2, steps=1, time=1.0).build()
    b3 = TrotterBlock(operator=op, n_qubits=2, steps=3, time=3.0).build()
    one = [str(c) for c in b1.flatten()]
    three = [str(c) for c in b3.flatten()]
    assert three == one * 3
