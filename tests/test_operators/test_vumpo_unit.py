"""VUMPO unit tests that need quimb only (the main suite also needs pyscf):
parameter-length contracts, cost-mode contracts, the global-optimizer path,
and the diagnostics — `_check_diag`'s exact spectrum is checked against
`numpy.linalg.eigvalsh` of a dense Hamiltonian built in-test.
"""

import numpy as np
import pytest

qtn = pytest.importorskip("quimb.tensor")

from qarp.operators import VUMPO  # noqa: E402


def _tfim_dense(L=2, j=1.0, h=0.7):
    z = np.diag([1.0, -1.0])
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    eye = np.eye(2)
    ham = -j * np.kron(z, z) - h * (np.kron(x, eye) + np.kron(eye, x))
    return ham


def _tfim_vumpo(**kwargs):
    ham = _tfim_dense()
    mpo = qtn.MatrixProductOperator.from_dense(ham, dims=(2, 2))
    defaults = dict(H_mpo=mpo, n_layers=1, hwp=False)
    defaults.update(kwargs)
    return VUMPO(**defaults), ham


def test_skew_hermitian_needs_d_squared_parameters():
    v = VUMPO(qtn.MPO_identity(2))
    with pytest.raises(ValueError, match="Not enough parameters"):
        v.skew_hermitian(np.zeros(3), d=2)


def test_build_gate_parameter_length_contracts():
    v_hwp = VUMPO(qtn.MPO_identity(2), hwp=True)
    with pytest.raises(ValueError, match="HWP mode requires 6"):
        v_hwp.build_gate(np.zeros(5))
    v_full = VUMPO(qtn.MPO_identity(2), hwp=False)
    with pytest.raises(ValueError, match="requires 16"):
        v_full.build_gate(np.zeros(15))


def test_flat_parameter_vector_too_short_raises():
    v = VUMPO(qtn.MPO_identity(2), n_layers=1, hwp=False)
    with pytest.raises(ValueError, match="expected"):
        v.layerise_params(np.zeros(v.n_gates * v.params_per_gate - 1))


def test_cost_fn_unknown_mode_raises():
    v, _ = _tfim_vumpo()
    v.mode = "bogus"
    with pytest.raises(ValueError, match="Unknown mode"):
        v.cost_fn(np.zeros(v.n_gates * v.params_per_gate))


def test_cost_fn_gs_deflation_uses_stored_initial_state():
    # The deflation loop falls back to self.initial_state (set as a side
    # effect of compute_energy_tn); the in-loop None-raise is unreachable
    # through cost_fn and stays uncovered by design.
    v, _ = _tfim_vumpo(mode="gs", beta=1.0)
    n = v.n_gates * v.params_per_gate
    prev = [(np.zeros(n), [0, 0])]
    cost = v.cost_fn(np.zeros(n), initial_state=None, prev_states=prev)
    # Zero params -> identity U: energy <00|H|00> = -1, overlap |<00|00>|^2 = 1.
    assert cost == pytest.approx(-1.0 + 1.0)


def test_cost_fn_diag_alpha_adds_energy_penalty():
    v, _ = _tfim_vumpo(mode="diag", alpha=0.5)
    params = np.full(v.n_gates * v.params_per_gate, 0.1)
    with_alpha = v.cost_fn(params, initial_state=[0, 0])
    v.alpha = 0.0
    base = v.cost_fn(params, initial_state=[0, 0])
    v.alpha = 0.5
    energy = v.compute_energy_tn(params, [0, 0])
    assert with_alpha == pytest.approx(base + 0.5 * energy)


def test_compute_energy_tn_defaults_initial_state():
    """The default ket is |0..0> (full) or |10..0> (hwp), used for the
    evaluation without being written onto the instance."""
    v, _ = _tfim_vumpo()
    assert v.initial_state is None
    # Zero params -> identity U: E = <00|H|00> = -J = -1 exactly.
    energy = v.compute_energy_tn(np.zeros(v.n_gates * v.params_per_gate))
    assert energy == pytest.approx(-1.0)
    assert v.initial_state is None

    # hwp default |10>: with U = I and H = Z x I the energy is <10|Z x I|10> = -1.
    z = np.diag([1.0, -1.0])
    v_hwp = VUMPO(
        qtn.MatrixProductOperator.from_dense(np.kron(z, np.eye(2)), dims=(2, 2)),
        n_layers=1,
        hwp=True,
    )
    energy = v_hwp.compute_energy_tn(np.zeros(v_hwp.n_gates * v_hwp.params_per_gate))
    assert energy == pytest.approx(-1.0)
    assert v_hwp.initial_state is None


def test_build_circuit_matches_tensor_network_unitary():
    v, _ = _tfim_vumpo()
    rng = np.random.default_rng(2)
    params = rng.normal(scale=0.3, size=v.n_gates * v.params_per_gate)

    circuit = v._build_circuit(params)
    # One raw gate per brickwork slot, on a 2-qubit register.
    assert circuit.N == 2
    assert len(circuit.gates) == v.n_gates


def test_check_diag_exact_spectrum_matches_eigvalsh(capsys):
    v, ham = _tfim_vumpo(mode="diag")
    params = np.full(v.n_gates * v.params_per_gate, 0.05)
    exact, approx, gs_idx = v._check_diag(params)

    assert np.allclose(exact, np.sort(np.linalg.eigvalsh(ham)), atol=1e-10)
    assert len(approx) == 4
    assert 0 <= gs_idx < 4
    out = capsys.readouterr().out
    assert "Off diagonal ratio, original" in out
    assert "Diagonalization error" in out


def test_build_global_optimizer_and_valid_diag(capsys):
    v, ham = _tfim_vumpo(mode="gs", opt="global", initial_state=[0, 0])
    v.optimizer_options_global = {"maxiter": 5}
    v.build(valid_diag=True)

    assert np.allclose(v.exact_eigs, np.sort(np.linalg.eigvalsh(ham)), atol=1e-10)
    assert v.approx_eigs is not None and len(v.approx_eigs) == 4
    out = capsys.readouterr().out
    assert "Final cost" in out
    assert "Diagonalization check complete." in out


def test_build_unknown_optimizer_raises():
    v, _ = _tfim_vumpo(opt="bogus")
    with pytest.raises(NotImplementedError, match="local' or 'global"):
        v.build()


# ── protocol fidelity: step (ii) "locally minimize", step (iv) "until convergence" ──


def test_local_minimisation_default_is_five_iterations_and_tol_is_exposed():
    v, _ = _tfim_vumpo()
    assert v.maxiter_local == 5
    assert v.optimizer_options_local == {"maxiter": 5}
    assert v.tol == 1e-6


def test_tol_stops_the_sweep_before_n_sweeps_on_a_converged_problem():
    """TFIM on two sites with one full gate is exactly diagonalisable, so the
    cost stops moving after the first sweeps and ``tol`` must end the loop."""
    v, _ = _tfim_vumpo(n_layers=1, n_sweeps=50, maxiter_local=20, tol=1e-8)
    gate_params = [np.ones(v.params_per_gate) for _ in range(v.n_gates)]
    v.optimize_local_sweep(gate_params, initial_state=[0, 0])
    assert 1 < v.sweeps_run < 50


def test_n_sweeps_caps_a_zero_tolerance_run():
    v, _ = _tfim_vumpo(n_layers=1, n_sweeps=4, maxiter_local=1, tol=0.0)
    gate_params = [np.ones(v.params_per_gate) for _ in range(v.n_gates)]
    v.optimize_local_sweep(gate_params, initial_state=[0, 0])
    assert v.sweeps_run == 4


def test_check_diag_refuses_to_densify_above_max_dense_qubits():
    v = VUMPO(qtn.MPO_ham_heis(6), n_layers=1)
    params = np.zeros(v.n_gates * v.params_per_gate)
    with pytest.raises(ValueError, match="energy_variance"):
        v._check_diag(params, max_dense_qubits=5)
    v._check_diag(params, max_dense_qubits=6)  # at the limit it still runs


def test_diag_alpha_term_uses_the_default_initial_state_when_none_is_given():
    """The alpha bias must not silently vanish for direct calls that omit
    ``initial_state``: ``build()`` passes ``self.initial_state`` and the two
    entry points must optimise the same objective."""
    v, _ = _tfim_vumpo(mode="diag", alpha=0.5, initial_state=[0, 1])
    rng = np.random.default_rng(3)
    p = rng.normal(size=v.n_gates * v.params_per_gate)
    assert v.cost_fn(p) == pytest.approx(v.cost_fn(p, initial_state=[0, 1]), rel=1e-12)
    assert v.cost_fn(p) != pytest.approx(-v._cost_tn(p), rel=1e-6)
    terms = v._terms(v._gates(p), None, None)
    v._fill_boundaries(terms, left=True, right=True)
    assert v._local_objective(0, terms)(p[: v.params_per_gate])[0] == pytest.approx(
        v.cost_fn(p), rel=1e-12
    )


def test_layerise_params_rejects_a_vector_that_is_too_long():
    """A four-layer vector on a two-layer instance, or a full-gate vector on
    an hwp instance, must not be silently truncated to a wrong circuit."""
    v = VUMPO(qtn.MPO_ham_heis(4), n_layers=2, hwp=True)
    total = v.n_gates * v.params_per_gate
    with pytest.raises(ValueError, match=f"expected {total}"):
        v.layerise_params(np.ones(total + 6))
    with pytest.raises(ValueError, match=f"expected {total}"):
        v.layerise_params(np.ones(total - 1))
    assert len(v.layerise_params(np.ones(total))) == 2


def test_hwp_is_fixed_at_construction():
    """params_per_gate and the gate builder must never disagree: flipping the
    flag on a live instance is refused rather than half-applied."""
    v, _ = _tfim_vumpo(hwp=False)
    assert v.hwp is False and v.params_per_gate == 16
    with pytest.raises(AttributeError):
        v.hwp = True
    assert v.params_per_gate == 16


def test_read_only_evaluation_does_not_cache_a_default_initial_state():
    v, _ = _tfim_vumpo(hwp=False)
    assert v.initial_state is None
    v.cost_fn(np.zeros(v.n_gates * v.params_per_gate))
    v.compute_energy_tn(np.zeros(v.n_gates * v.params_per_gate))
    assert v.initial_state is None
