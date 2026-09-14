"""QITE (Motta imaginary-time evolution) — phase 1, statevector-exact.

Oracles are independent of the implementation: scipy ``expm`` for the exact
imaginary-time trajectory, ``eigvalsh`` for the ground-state energy, and the
transverse-field-only analytic limit.
"""

import numpy as np
import pytest
from scipy.linalg import expm

import qarpx as qx
from qarp.algorithms import QITE
from qarp.algorithms._composite.qite import full_pauli_pool, qite_linear_system
from qarp.blocks import ComputationalBasisStateBlock, SimpleBlock
from qarp.operators.models import transverse_field_ising

# ── helpers ──────────────────────────────────────────────────────────────────


def _sv(block, n):
    return np.asarray(qx.QarpSimulator().statevector(block.build().flatten(), n))


def _exact_ite(Hmat, psi0, dtau, n_steps):
    """Exact normalized imaginary-time trajectory: energies[k] and final ψ."""
    step = expm(-dtau * Hmat)
    psi = psi0.astype(complex).copy()
    energies = [float((psi.conj() @ Hmat @ psi).real)]
    for _ in range(n_steps):
        psi = step @ psi
        psi = psi / np.linalg.norm(psi)
        energies.append(float((psi.conj() @ Hmat @ psi).real))
    return np.array(energies), psi


def _fit_slope(dtaus, errs):
    return float(np.polyfit(np.log(dtaus), np.log(errs), 1)[0])


def _unitary(block, n):
    """Dense unitary of a built block, one column per basis state."""
    cols = [
        np.asarray(
            qx.QarpSimulator().statevector(
                block.flatten(), n, initial_state=np.ascontiguousarray(e)
            )
        )
        for e in np.eye(2**n, dtype=np.complex128)
    ]
    return np.array(cols).T


# ── row 1: final energy → ground state (eigh) ───────────────────────────────


def test_final_energy_converges_to_ground_state():
    H = transverse_field_ising((2,), j=1.0, h_x=0.8)
    E0 = float(np.linalg.eigvalsh(H.sparse_matrix(2).toarray())[0])
    qite = QITE(H, ComputationalBasisStateBlock([0, 0]), dtau=0.04, n_steps=200)
    E_final, _ = qite.run()
    assert E_final == pytest.approx(E0, abs=1e-3)


# ── row 2: qite_linear_system reproduces the ITE step, O(dtau²) ──────────────


def test_qite_linear_system_reproduces_ite_step_second_order():
    # Pins b's sign, the 2·Re S factor, and the dropped <H> term: a wrong one
    # makes the step O(dtau) instead of O(dtau²).
    n = 2
    H = transverse_field_ising((2,), j=1.0, h_x=0.7)
    Hmat = H.sparse_matrix(n).toarray()
    pool = full_pauli_pool(n)
    rng = np.random.default_rng(0)
    psi = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
    psi = psi / np.linalg.norm(psi)

    dtaus = np.array([0.08, 0.04, 0.02])
    errs = []
    for dtau in dtaus:
        re_s, b = qite_linear_system(psi, pool, H, n)
        a = np.linalg.solve(re_s + 1e-10 * np.eye(len(pool)), b)
        A = np.zeros((2**n, 2**n), dtype=complex)
        for coeff, sigma in zip(a, pool, strict=True):
            A += coeff * sigma.sparse_matrix(n).toarray()
        lhs = expm(-1j * dtau * A) @ psi
        rhs = expm(-dtau * Hmat) @ psi
        rhs = rhs / np.linalg.norm(rhs)
        errs.append(np.linalg.norm(lhs - rhs))
    assert _fit_slope(dtaus, np.array(errs)) == pytest.approx(2.0, abs=0.3)


# ── layer wiring: the simulated TrotterBlock layer == exp(-i·dtau·Â) ────────


def test_layer_wiring_matches_exact_exponential():
    # Wiring check (not a re-test of TrotterBlock): QITE must hand the block the
    # solved Â, n_qubits, time AND trotter_order.  Two traps, both measured:
    # end-to-end rows cannot see an order-1 fallback (its O(dtau²) rendering error
    # sits one order below QITE's own O(dtau)), and neither can _step's carried
    # state — at |00> the order-1 commutator term annihilates the state, so the
    # projected error is O(dtau³) for BOTH orders at any tolerance.  In operator
    # norm they separate: Strang is O(dtau³), order 1 is O(dtau²).
    n = 2
    H = transverse_field_ising((2,), j=1.0, h_x=0.8)
    psi0 = _sv(ComputationalBasisStateBlock([0, 0]), n)

    dtaus = np.array([0.08, 0.04, 0.02])
    errs = []
    for dtau in dtaus:
        qite = QITE(H, ComputationalBasisStateBlock([0, 0]), dtau=dtau, n_steps=1)
        qite.build()
        layer, psi_next = qite._step(psi0)
        A = np.zeros((2**n, 2**n), dtype=complex)
        for coeff, sigma in zip(qite._last_a, qite._pool, strict=True):
            A += coeff * sigma.sparse_matrix(n).toarray()
        exact = expm(-1j * dtau * A)
        assert np.allclose(psi_next, exact @ psi0, atol=1e-4)
        errs.append(float(np.linalg.norm(_unitary(layer, n) - exact, 2)))
    assert _fit_slope(dtaus, np.array(errs)) == pytest.approx(3.0, abs=0.3)
    assert errs[-1] < 1e-5  # order 1 lands at 1.3e-4 here

    # trotter_steps reaches the block too: Strang over s substeps is O(dtau³/s²),
    # so doubling the substeps cuts the error ~4x.  The sweep above cannot see
    # this — every dtau there uses the default steps=1.
    dtau = dtaus[0]
    qite = QITE(H, ComputationalBasisStateBlock([0, 0]), dtau=dtau, n_steps=1, trotter_steps=2)
    qite.build()
    layer_2, _ = qite._step(psi0)
    A = np.zeros((2**n, 2**n), dtype=complex)
    for coeff, sigma in zip(qite._last_a, qite._pool, strict=True):
        A += coeff * sigma.sparse_matrix(n).toarray()
    err_2 = float(np.linalg.norm(_unitary(layer_2, n) - expm(-1j * dtau * A), 2))
    assert err_2 < errs[0] / 3.0


# ── row 3: per-step energy tracks the exact ITE trajectory ──────────────────


def test_per_step_energy_tracks_exact_trajectory():
    n, dtau, n_steps = 2, 0.01, 25
    H = transverse_field_ising((2,), j=1.0, h_x=0.9)
    psi0 = _sv(ComputationalBasisStateBlock([0, 0]), n)
    exact, _ = _exact_ite(H.sparse_matrix(n).toarray(), psi0, dtau, n_steps)
    qite = QITE(H, ComputationalBasisStateBlock([0, 0]), dtau=dtau, n_steps=n_steps)
    qite.run()
    assert np.allclose(qite.energy_history, exact, atol=3e-3)


# ── row 4: energy non-increasing (+ tol for Trotter/Tikhonov) ───────────────


def test_energy_non_increasing():
    H = transverse_field_ising((2,), j=1.0, h_x=1.0)
    qite = QITE(H, ComputationalBasisStateBlock([0, 0]), dtau=0.05, n_steps=80)
    qite.run()
    hist = qite.energy_history
    assert all(b <= a + 1e-6 for a, b in zip(hist, hist[1:], strict=False))


# ── row 5: whole-run convergence is first order in dtau ─────────────────────


def test_whole_run_first_order_in_dtau():
    # Global error of a first-order integrator is O(dtau) in the phase-aligned
    # state distance, not infidelity: infidelity ~ (state error)² ~ dtau²; the
    # state distance is the O(dtau) quantity.  Moderate beta so the exact state is not yet near the ground
    # state (where the gap would collapse faster than O(dtau)).
    n, beta = 2, 0.6
    H = transverse_field_ising((2,), j=1.0, h_x=0.8)
    psi0 = _sv(ComputationalBasisStateBlock([0, 0]), n)
    _, psi_exact = _exact_ite(H.sparse_matrix(n).toarray(), psi0, beta / 40, 40)

    dtaus = np.array([0.06, 0.03, 0.015])
    dists = []
    for dtau in dtaus:
        qite = QITE(H, ComputationalBasisStateBlock([0, 0]), dtau=dtau, n_steps=round(beta / dtau))
        _, psi_q = qite.run()
        overlap = abs(np.vdot(psi_exact, psi_q))
        dists.append(np.sqrt(max(2.0 * (1.0 - overlap), 0.0)))
    assert _fit_slope(dtaus, np.array(dists)) == pytest.approx(1.0, abs=0.3)


# ── row 6: get_final_state_block re-simulation == carried ψ (round-trip) ─────


def test_final_block_resimulation_matches_carried_state():
    n = 2
    H = transverse_field_ising((2,), j=1.0, h_x=0.8)
    qite = QITE(H, ComputationalBasisStateBlock([0, 0]), dtau=0.05, n_steps=30)
    _, psi = qite.run()
    psi_blk = _sv(qite.get_final_state_block(), n)
    assert abs(np.vdot(psi_blk, psi)) == pytest.approx(1.0, abs=1e-8)


# ── row 7: pool= override — real H/ψ0 → Y-odd pool suffices ──────────────────


def test_y_odd_pool_matches_full_pool():
    n = 2
    H = transverse_field_ising((2,), j=1.0, h_x=0.9)

    def _n_y(sigma):
        (term,) = sigma.terms
        return sum(1 for _, p in term if p == "Y")

    y_odd = [s for s in full_pauli_pool(n) if _n_y(s) % 2 == 1]
    full = QITE(H, ComputationalBasisStateBlock([0, 0]), dtau=0.05, n_steps=40)
    reduced = QITE(H, ComputationalBasisStateBlock([0, 0]), dtau=0.05, n_steps=40, pool=y_odd)
    full.run()
    reduced.run()
    assert np.allclose(full.energy_history, reduced.energy_history, atol=1e-8)


# ── row 8: analytic value (transverse-field-only limit) ─────────────────────


def test_transverse_only_analytic_limit():
    # The Pfeuty periodic momentum sum was rejected as the oracle: its finite-N
    # sector handling is error-prone, so the analytic oracle here is the
    # decoupled limit j=0 -> H = -h_x·Σ X, ground energy exactly -h_x·N.  The
    # eigh oracle already covers the interacting ground state.
    n, h_x = 3, 1.3
    H = transverse_field_ising((3,), j=0.0, h_x=h_x)
    qite = QITE(H, ComputationalBasisStateBlock([0, 0, 0]), dtau=0.05, n_steps=120)
    E_final, _ = qite.run()
    assert E_final == pytest.approx(-h_x * n, abs=1e-3)


def test_two_site_tfim_analytic_ground_energy():
    # Interacting analytic oracle, independent of eigh: for H = -j Z0Z1
    # - h_x (X0 + X1) the {(|00>+|11>)/√2, (|01>+|10>)/√2} block is
    # [[-j, -2h_x], [-2h_x, j]], so E0 = -sqrt(j² + 4 h_x²) exactly.
    j, h_x = 1.0, 0.6
    H = transverse_field_ising((2,), j=j, h_x=h_x)
    qite = QITE(H, ComputationalBasisStateBlock([0, 0]), dtau=0.04, n_steps=200)
    E_final, _ = qite.run()
    assert E_final == pytest.approx(-np.sqrt(j**2 + 4 * h_x**2), abs=1e-3)


# ── row 9: error paths (patch coverage) ─────────────────────────────────────


def test_symbolic_initial_block_rejected():
    from sympy import Symbol

    sym = SimpleBlock(2)
    sym.ry(0, Symbol("theta"))
    H = transverse_field_ising((2,), j=1.0, h_x=1.0)
    with pytest.raises(ValueError):
        QITE(H, sym, dtau=0.1, n_steps=5)


def test_non_qubit_operator_hamiltonian_rejected():
    with pytest.raises(TypeError):
        QITE("not-an-operator", ComputationalBasisStateBlock([0, 0]), dtau=0.1, n_steps=5)


def test_zero_steps_rejected():
    H = transverse_field_ising((2,), j=1.0, h_x=1.0)
    with pytest.raises(ValueError):
        QITE(H, ComputationalBasisStateBlock([0, 0]), dtau=0.1, n_steps=0)


def test_final_state_block_before_run_raises():
    H = transverse_field_ising((2,), j=1.0, h_x=1.0)
    qite = QITE(H, ComputationalBasisStateBlock([0, 0]), dtau=0.1, n_steps=5)
    with pytest.raises(RuntimeError):
        qite.get_final_state_block()


def test_engine_with_noise_model_rejected():
    from qarp.devices import NoiseModel
    from qarp.engines import QarpEngine
    from qarp.errors import CapabilityError

    H = transverse_field_ising((2,), j=1.0, h_x=1.0)
    engine = QarpEngine(n_qubits=2, noise_model=NoiseModel.bit_flip(0.05))
    with pytest.raises(CapabilityError):
        QITE(H, ComputationalBasisStateBlock([0, 0]), dtau=0.1, n_steps=5, engine=engine)


def test_non_qarp_engine_rejected():
    from qarp.errors import CapabilityError
    from tests.conftest import _StubEngine

    H = transverse_field_ising((2,), j=1.0, h_x=1.0)
    with pytest.raises(CapabilityError):
        QITE(H, ComputationalBasisStateBlock([0, 0]), dtau=0.1, n_steps=5, engine=_StubEngine())


def test_routed_device_rejected():
    from qarp.devices import Device, get_nearest_neighbour_architecture
    from qarp.engines import QarpEngine
    from qarp.errors import CapabilityError

    H = transverse_field_ising((2,), j=1.0, h_x=1.0)
    engine = QarpEngine(device=Device(2, architecture=get_nearest_neighbour_architecture(1, 2)))
    with pytest.raises(CapabilityError):
        QITE(H, ComputationalBasisStateBlock([0, 0]), dtau=0.1, n_steps=5, engine=engine)


def test_hamiltonian_wider_than_initial_block_rejected():
    # Without the guard this is a bare IndexError from inside pauli_apply.
    H = transverse_field_ising((3,), j=1.0, h_x=1.0)
    with pytest.raises(ValueError, match="3 qubits"):
        QITE(H, ComputationalBasisStateBlock([0, 0]), dtau=0.1, n_steps=5)


def test_non_unit_coefficient_pool_rejected():
    # Silently halves Â otherwise: the solve sees the coefficient, _step does not.
    from qarp.operators import QubitOperator

    H = transverse_field_ising((2,), j=1.0, h_x=1.0)
    qite = QITE(
        H,
        ComputationalBasisStateBlock([0, 0]),
        dtau=0.1,
        n_steps=5,
        pool=[QubitOperator("Y0", 0.5)],
    )
    with pytest.raises(ValueError, match="unit coefficient"):
        qite.build()


def test_multi_term_pool_entry_rejected():
    from qarp.operators import QubitOperator

    H = transverse_field_ising((2,), j=1.0, h_x=1.0)
    qite = QITE(
        H,
        ComputationalBasisStateBlock([0, 0]),
        dtau=0.1,
        n_steps=5,
        pool=[QubitOperator("Y0", 1.0) + QubitOperator("Y1", 1.0)],
    )
    with pytest.raises(ValueError, match="single Pauli strings"):
        qite.build()


def test_verbose_logs_one_row_per_step(capsys):
    # |step| is the update magnitude ‖a‖·dtau — there is no optimizer step here.
    H = transverse_field_ising((2,), j=1.0, h_x=1.0)
    qite = QITE(H, ComputationalBasisStateBlock([0, 0]), dtau=0.1, n_steps=3, verbose=True)
    qite.run()
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert lines[0].startswith("QITE Run:")
    assert "|step|" in lines[1]
    assert len(lines) == 2 + qite.n_steps


def test_disabled_noise_model_is_accepted():
    """§14: a disabled noise model keeps the exact path open; the result is
    the eigh ground energy, as on a plain engine."""
    import qarpx as qx
    from qarp.devices import NoiseModel
    from qarp.engines import QarpEngine

    H = transverse_field_ising((2,), j=1.0, h_x=0.8)
    E0 = float(np.linalg.eigvalsh(H.sparse_matrix(2).toarray())[0])
    engine = QarpEngine(n_qubits=2, noise_model=NoiseModel.bit_flip(0.05, [qx.GateType.X]))
    engine.noise_model.enabled = False
    qite = QITE(H, ComputationalBasisStateBlock([0, 0]), dtau=0.04, n_steps=200, engine=engine)
    E_final, _ = qite.run()
    assert E_final == pytest.approx(E0, abs=1e-3)
