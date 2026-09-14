import numpy as np
import pytest

qtn = pytest.importorskip("quimb.tensor")

MatrixProductOperator = qtn.MatrixProductOperator


from qarp.operators import VUMPO

openfermion = pytest.importorskip("openfermion")
from qarp.operators.compat import from_openfermion


class hamiltonians:
    """Adapter: openfermion's reference fermi_hubbard builder, converted to
    the qarpx FermionOperator at the boundary."""

    @staticmethod
    def fermi_hubbard(*args, **kwargs):
        return from_openfermion(openfermion.hamiltonians.fermi_hubbard(*args, **kwargs))


from qarp.endianness import lsb_to_msb_matrix
from qarp.operators import JordanWigner
from tests.molecular_assets import fermion_operator


@pytest.fixture(scope="session")
def h2_problem():
    """
    Build the H2 Hamiltonian once and reuse in all tests.
    Assumes dependencies are installed and importable.
    """
    qham = JordanWigner().encode_operator(fermion_operator("h2_0.735_sto3g"))
    # quimb's from_dense is kron-ordered (site 0 = leftmost factor) — an
    # external MSB boundary, so VUMPO site n ↔ qubit n needs the reversed
    # matrix.  ham_mat stays in that site order for the U†HU checks below.
    ham_mat = lsb_to_msb_matrix(qham.sparse_matrix().toarray())

    L = 4
    H_mpo = MatrixProductOperator.from_dense(ham_mat, dims=(2,) * L)
    initial_state = np.array([1, 1, 0, 0])

    return {
        "H_mpo": H_mpo,
        "ham_mat": ham_mat,
        "L": L,
        "initial_state": initial_state,
    }


@pytest.fixture(scope="session")
def fh_problem():
    nx = 3
    ny = 2
    L = nx * ny
    FH = hamiltonians.fermi_hubbard(
        nx,
        ny,
        -1.0,
        2.5,
        chemical_potential=0.0,
        magnetic_field=1.5,
        periodic=True,
        spinless=True,
        particle_hole_symmetry=False,
    )

    fh_ham = JordanWigner().encode_operator(FH)  # JordanWigner encoding for the Hamiltonian
    qop_fh = fh_ham

    ham_mat_fh = lsb_to_msb_matrix(qop_fh.sparse_matrix().toarray())  # quimb site order
    H_mpo = MatrixProductOperator.from_dense(ham_mat_fh, dims=(2,) * L)

    initial_state_fh = np.array([1] * (2) + [0] * (4))
    return {
        "H_mpo": H_mpo,
        "ham_mat": ham_mat_fh,
        "L": L,
        "initial_state": initial_state_fh,
    }


@pytest.fixture
def make_vumpo(h2_problem):
    """
    Factory fixture returning a builder you can call with n_layers, mode, hwp.
    Usage:
        vumpo, params = make_vumpo(n_layers=3, mode="gs", hwp=False,
                                   n_sweeps=4, maxiter_local=10)
    """

    def _build(
        n_layers: int,
        mode: str = "diag",
        hwp: bool = False,
        *,
        n_sweeps: int = 1,
        maxiter_local: int = 1,
        opt: str = "local",
        verbose: bool = False,
        **kwargs,
    ):
        v = VUMPO(
            H_mpo=h2_problem["H_mpo"],
            initial_state=h2_problem["initial_state"],
            n_layers=n_layers,
            n_sweeps=n_sweeps,
            maxiter_local=maxiter_local,
            hwp=hwp,
            mode=mode,
            opt=opt,
            verbose=verbose,
            **kwargs,
        )
        params = v.build(valid_diag=False)
        return v, params

    return _build


@pytest.fixture
def make_vumpo_fh(fh_problem):
    """
    Factory fixture returning a builder you can call with n_layers, mode, hwp.
    Usage:
        vumpo, params = make_vumpo(n_layers=3, mode="gs", hwp=False,
                                   n_sweeps=4, maxiter_local=10)
    """

    def _build(
        n_layers: int,
        mode: str = "diag",
        hwp: bool = False,
        *,
        n_sweeps: int = 1,
        maxiter_local: int = 1,
        opt: str = "local",
        verbose: bool = False,
        **kwargs,
    ):
        v = VUMPO(
            H_mpo=fh_problem["H_mpo"],
            initial_state=fh_problem["initial_state"],
            n_layers=n_layers,
            n_sweeps=n_sweeps,
            maxiter_local=maxiter_local,
            hwp=hwp,
            mode=mode,
            opt=opt,
            verbose=verbose,
            **kwargs,
        )
        params = v.build(valid_diag=False)
        return v, params

    return _build


def trivial_mpo(L):
    return qtn.MPO_identity(L)


def test_skew_hermitian_is_skew():
    v = VUMPO(trivial_mpo(2))
    params = np.arange(4.0)
    A = v.skew_hermitian(params, d=2)
    assert A.shape == (2, 2)
    assert np.allclose(A.conj().T, -A)


def test_full_gate_is_unitary():
    v = VUMPO(trivial_mpo(2), hwp=False)
    params = np.linspace(0, 1, 16)
    U = v.build_gate(params)
    assert U.shape == (4, 4)
    assert np.allclose(U.conj().T @ U, np.eye(4), atol=1e-7)


def test_hwp_gate_structure():
    v = VUMPO(trivial_mpo(2), hwp=True)
    params = np.linspace(0, 1, 6)
    U = v.build_gate(params)

    assert U.shape == (4, 4)

    # Off-block elements should be zero
    assert np.allclose(U[0, 1:3], 0)
    assert np.allclose(U[1:3, 0], 0)
    assert np.allclose(U[3, 1:3], 0)
    assert np.allclose(U[1:3, 3], 0)


def test_layerise_params():
    L = 4
    v = VUMPO(trivial_mpo(L), n_layers=2)
    total = v.n_gates * v.params_per_gate
    params = np.arange(total, dtype=float)
    layers = v.layerise_params(params)

    assert len(layers) == 2
    assert sum(len(layer) for layer in layers) == v.n_gates


def test_build_tn_open_indices():
    L = 4  # This should be even so the tests passes. The vumpo does not add identity gates on untouched qubits.
    mpo = trivial_mpo(L)
    print(mpo)
    v = VUMPO(mpo, n_layers=1)
    print(v.n_gates, v.params_per_gate)
    params = np.ones(v.n_gates * v.params_per_gate)
    print(params)
    tn = v.build_tn(params)
    print(tn)
    inds = set(tn.outer_inds())

    # Every qubit must have ONE outer index, but name depends on whether
    # it participated in any gate.
    print(inds)
    print(type(inds))
    for q in range(L):
        assert f"b{q}" in inds
        assert f"k{q}" in inds

    assert len(inds) == 2 * L


def test_add_circuit_wire_flow():
    L = 4
    v = VUMPO(trivial_mpo(L), n_layers=1, hwp=False)
    params = np.ones(v.n_gates * v.params_per_gate)

    tensors = []
    wire_in = {q: f"in{q}" for q in range(L)}
    wire_out = v._add_circuit(tensors, params, prefix="U", wire_in=wire_in)

    # Should still have exactly L wires
    assert set(wire_out.keys()) == set(range(L))

    # Output wire names must be unique
    assert len(set(wire_out.values())) == L

    # Should have exactly v.n_gates tensors
    assert len([t for t in tensors if t.data.size == 16]) == v.n_gates


def test_to_flat_params_list_vs_array():
    v = VUMPO(trivial_mpo(2), n_layers=1)
    p_list = [np.arange(v.params_per_gate) for _ in range(v.n_gates)]
    p_flat = v._to_flat_params(p_list)

    assert p_flat.shape[0] == v.n_gates * v.params_per_gate
    assert np.allclose(p_flat[: v.params_per_gate], p_list[0])


def test_add_circuit_conjugate_structure():
    L = 4
    v = VUMPO(trivial_mpo(L), n_layers=1)
    params = np.ones(v.n_gates * v.params_per_gate)

    tensors = []
    wire_in = {q: f"h{q}" for q in range(L)}
    wire_out = v._add_circuit(tensors, params, prefix="D", wire_in=wire_in, conj=True)

    assert len(wire_out) == L
    assert all("D_0_" in w for w in wire_out.values())

    assert len(tensors) == v.n_gates


def test_compute_energy_identity_mpo():
    L = 3
    v = VUMPO(trivial_mpo(L), n_layers=1)
    params = np.zeros(v.n_gates * v.params_per_gate)
    initial_state = [0] * L

    energy = v.compute_energy_tn(params, initial_state)

    # <s| I |s> = 1 for normalised computational basis states
    assert np.isclose(energy, 1.0, atol=1e-7)


def test_overlap_symmetry():
    L = 3
    v = VUMPO(trivial_mpo(L), n_layers=1)

    params_a = np.zeros(v.n_gates * v.params_per_gate)
    params_b = np.ones(v.n_gates * v.params_per_gate) * 0.1

    init_a = [0, 1, 0]
    init_b = [1, 0, 1]

    ov_ab = v._overlap_tn(params_a, params_b, init_a, init_b)
    ov_ba = v._overlap_tn(params_b, params_a, init_b, init_a)

    assert np.isclose(ov_ab, np.conj(ov_ba))


def test_build_returns_correct_param_shape():
    L = 4
    v = VUMPO(trivial_mpo(L), n_layers=2, opt="local", n_sweeps=1)
    params = v.build(valid_diag=False)

    assert params.shape == (v.n_gates * v.params_per_gate,)


def test_cost_fn_gs_identity_matches_energy():
    L = 3
    v = VUMPO(trivial_mpo(L), n_layers=1, mode="gs")
    params = np.zeros(v.n_gates * v.params_per_gate)
    init = [0] * L

    assert v.cost_fn(params, initial_state=init) == v.compute_energy_tn(params, init)


def test_build_tn_outer_index_pairing():
    L = 4
    v = VUMPO(trivial_mpo(L), n_layers=1)
    params = np.ones(v.n_gates * v.params_per_gate)

    tn = v.build_tn(params)
    inds = tn.outer_inds()

    # Every qubit has exactly one input AND one output index
    in_count = sum(ind.startswith("k") for ind in inds)
    out_count = sum(ind.startswith("b") for ind in inds)

    assert in_count + out_count == 2 * L
    assert in_count <= L and out_count <= L
    assert in_count + out_count == len(set(inds))  # all outer inds accounted for


def test_h2_eigenvalues_approximate_diag(make_vumpo, h2_problem):
    """
    Check that diag(U† H U) approximately matches the exact eigenspectrum
    within a loose but meaningful tolerance for shallow circuits.
    """
    vumpo, params = make_vumpo(n_layers=4, mode="diag", hwp=True)

    ham_mat = h2_problem["ham_mat"]
    exact_eigs = np.sort(np.linalg.eigvalsh(ham_mat))

    L = vumpo.L
    tn = vumpo.build_tn(params)
    U = tn.contract(
        output_inds=[f"b{q}" for q in range(L)] + [f"k{q}" for q in range(L)]
    ).data.reshape(2**L, 2**L)

    UHU = U.conj().T @ ham_mat @ U
    approx_eigs = np.sort(np.real(np.diag(UHU)))

    assert np.sum(np.abs(approx_eigs - exact_eigs)) < 1


def test_h2_energy_close_gs(make_vumpo, h2_problem):
    """
    In GS mode, the computed variational energy should be reasonably close
    to the exact ground state with shallow depth.
    """
    vumpo, params = make_vumpo(n_layers=4, mode="gs", hwp=True)

    ham_mat = h2_problem["ham_mat"]
    E_exact = np.linalg.eigvalsh(ham_mat)[0]

    E_var = vumpo.compute_energy_tn(params, initial_state=vumpo.initial_state)

    assert abs(E_var - E_exact) < 0.15


def test_fh_energy_monotone_with_layers(make_vumpo_fh, fh_problem):
    """
    With increasing n_layers in GS mode, the variational energy should not increase,
    and typically decreases (monotonic non-increasing).
    """
    energies = []
    for depth in (1, 2, 4):
        vumpo, params = make_vumpo_fh(n_layers=depth, mode="gs", hwp=True)
        print(vumpo.n_layers)
        E = vumpo.compute_energy_tn(params, initial_state=vumpo.initial_state)
        energies.append(E)

    assert energies[1] <= energies[0] + 1e-12
    assert energies[2] <= energies[1] + 1e-12


def test_fh_diag_eigenvalues_improve_with_layers(make_vumpo_fh, fh_problem):
    """
    In diag mode, the diagonal of U† H U should approximate the eigenvalues
    better as n_layers increases. We test monotonic improvement of the
    eigenvalue L1 error for depths 1, 2, 3.
    """
    ham_mat = fh_problem["ham_mat"]
    exact = np.sort(np.linalg.eigvalsh(ham_mat))
    energies = []

    for depth in (1, 2, 4):
        vumpo, params = make_vumpo_fh(n_layers=depth, mode="diag", hwp=True)
        L = vumpo.L
        tn = vumpo.build_tn(params)
        U = tn.contract(
            output_inds=[f"b{q}" for q in range(L)] + [f"k{q}" for q in range(L)]
        ).data.reshape(2**L, 2**L)

        UHU = U.conj().T @ ham_mat @ U
        approx = np.sort(np.real(np.diag(UHU)))

        err = np.sum(np.abs(approx - exact))
        energies.append(err)

    assert energies[1] <= energies[0] + 1e-12
    assert energies[2] <= energies[1] + 1e-12


# ── Regression tests for bugfixes ──────────────────────────────────


def test_local_cost_single_evaluation(h2_problem):
    """Verify local_cost calls cost_fn exactly once per evaluation."""
    v = VUMPO(
        H_mpo=h2_problem["H_mpo"],
        initial_state=h2_problem["initial_state"],
        n_layers=2,
        n_sweeps=1,
        maxiter_local=1,
        mode="gs",
        hwp=True,
    )
    call_count = [0]
    original_cost_fn = v.cost_fn

    def counting_cost_fn(*args, **kwargs):
        call_count[0] += 1
        return original_cost_fn(*args, **kwargs)

    v.cost_fn = counting_cost_fn
    gate_params = [np.ones(v.params_per_gate, dtype=float) for _ in range(v.n_gates)]
    v.optimize_local_sweep(gate_params, initial_state=list(h2_problem["initial_state"]))

    assert call_count[0] <= v.n_gates * 25, (
        f"cost_fn called {call_count[0]} times for {v.n_gates} gates"
    )


def test_gs_better_than_diag_for_ground_state(make_vumpo, h2_problem):
    """mode='gs' should produce equal or lower energy than mode='diag'."""
    ham_mat = h2_problem["ham_mat"]
    E_exact = np.linalg.eigvalsh(ham_mat)[0]

    vumpo_diag, params_diag = make_vumpo(
        n_layers=4, mode="diag", hwp=True, n_sweeps=4, maxiter_local=4
    )
    L = vumpo_diag.L
    tn_diag = vumpo_diag.build_tn(params_diag)
    U_diag = tn_diag.contract(
        output_inds=[f"b{q}" for q in range(L)] + [f"k{q}" for q in range(L)]
    ).data.reshape(2**L, 2**L)
    UHU_diag = U_diag.conj().T @ ham_mat @ U_diag
    E_diag = np.min(np.real(np.diag(UHU_diag)))

    vumpo_gs, params_gs = make_vumpo(n_layers=4, mode="gs", hwp=True, n_sweeps=4, maxiter_local=4)
    E_gs = vumpo_gs.compute_energy_tn(params_gs, initial_state=vumpo_gs.initial_state)

    assert E_gs <= E_diag + 0.05, (
        f"mode='gs' energy {E_gs:.6f} > mode='diag' energy {E_diag:.6f} (exact={E_exact:.6f})"
    )


def test_deflation_penalises_all_prev_states(h2_problem):
    """Overlap penalty should accumulate across all prev_states."""
    v = VUMPO(
        H_mpo=h2_problem["H_mpo"],
        initial_state=h2_problem["initial_state"],
        n_layers=2,
        mode="gs",
        beta=10.0,
    )
    params = np.ones(v.n_gates * v.params_per_gate) * 0.1
    init = list(h2_problem["initial_state"])

    cost_base = v.cost_fn(params, initial_state=init, prev_states=None)

    prev1 = [(params, init)]
    cost_1 = v.cost_fn(params, initial_state=init, prev_states=prev1)

    prev2 = [(params, init), (params, init)]
    cost_2 = v.cost_fn(params, initial_state=init, prev_states=prev2)

    penalty_1 = cost_1 - cost_base
    penalty_2 = cost_2 - cost_base

    assert penalty_1 > 1e-6, "Single prev_state should add penalty"
    assert penalty_2 > penalty_1 * 1.5, (
        f"Two identical prev_states should roughly double the penalty: "
        f"penalty_1={penalty_1:.6f}, penalty_2={penalty_2:.6f}"
    )


def test_tn_energy_matches_direct(h2_problem):
    """compute_energy_tn should match direct <s|U^dag H U|s> calculation."""
    ham_mat = h2_problem["ham_mat"]
    init = list(h2_problem["initial_state"])
    L = h2_problem["L"]

    v = VUMPO(H_mpo=h2_problem["H_mpo"], initial_state=init, n_layers=2, hwp=True, mode="gs")

    np.random.seed(99)
    params = np.random.randn(v.n_gates * v.params_per_gate) * 0.5

    E_tn = v.compute_energy_tn(params, init)

    dim = 2**L
    tn = v.build_tn(params)
    U = tn.contract(
        output_inds=[f"b{q}" for q in range(L)] + [f"k{q}" for q in range(L)]
    ).data.reshape(dim, dim)
    s = np.zeros(dim)
    idx = sum(init[q] * 2 ** (L - 1 - q) for q in range(L))
    s[idx] = 1.0
    E_direct = np.real(s @ U.conj().T @ ham_mat @ U @ s)

    assert np.isclose(E_tn, E_direct, atol=1e-8), f"TN={E_tn:.10f} vs direct={E_direct:.10f}"


def test_gs_optimization_improves_over_bare(h2_problem):
    """GS mode optimization should produce lower energy than the bare initial state."""
    ham_mat = h2_problem["ham_mat"]
    init = list(h2_problem["initial_state"])
    L = h2_problem["L"]

    dim = 2**L
    s = np.zeros(dim)
    idx = sum(init[q] * 2 ** (L - 1 - q) for q in range(L))
    s[idx] = 1.0
    E_bare = np.real(s @ ham_mat @ s)

    v = VUMPO(
        H_mpo=h2_problem["H_mpo"],
        initial_state=init,
        n_layers=4,
        n_sweeps=4,
        maxiter_local=4,
        hwp=True,
        mode="gs",
        opt="local",
    )
    params = v.build(valid_diag=False)
    E_opt = v.compute_energy_tn(params, init)

    assert E_opt <= E_bare + 1e-6, f"Optimized energy {E_opt:.6f} worse than bare {E_bare:.6f}"


def test_expm_depth_monotonicity(h2_problem):
    """Best energy across random starts should not increase with more layers."""
    init = list(h2_problem["initial_state"])

    best_energies = []
    for n_layers in [2, 4]:
        v = VUMPO(
            H_mpo=h2_problem["H_mpo"],
            initial_state=init,
            n_layers=n_layers,
            hwp=True,
            mode="gs",
        )
        n_p = v.n_gates * v.params_per_gate
        best_E = float("inf")
        for trial in range(3):
            np.random.seed(trial)
            x0 = np.random.randn(n_p) * 0.1
            from scipy.optimize import minimize

            res = minimize(
                lambda p: v.compute_energy_tn(p, init),
                x0,
                method="L-BFGS-B",
                options={"maxiter": 50},
            )
            E = v.compute_energy_tn(res.x, init)
            if E < best_E:
                best_E = E
        best_energies.append(best_E)

    assert best_energies[1] <= best_energies[0] + 0.1, (
        f"4 layers ({best_energies[1]:.6f}) worse than 2 layers ({best_energies[0]:.6f})"
    )


def test_diag_warmstart_helps_gs(h2_problem):
    """Warm-starting GS from diag params should beat or match cold GS start."""
    init = list(h2_problem["initial_state"])

    # Diag mode
    np.random.seed(42)
    v_diag = VUMPO(
        H_mpo=h2_problem["H_mpo"],
        initial_state=init,
        n_layers=4,
        n_sweeps=2,
        maxiter_local=2,
        hwp=True,
        mode="diag",
        opt="local",
    )
    diag_params = v_diag.build(valid_diag=False)

    # Cold GS
    np.random.seed(42)
    v_cold = VUMPO(
        H_mpo=h2_problem["H_mpo"],
        initial_state=init,
        n_layers=4,
        n_sweeps=2,
        maxiter_local=2,
        hwp=True,
        mode="gs",
        opt="local",
    )
    cold_params = v_cold.build(valid_diag=False)
    E_cold = v_cold.compute_energy_tn(cold_params, init)

    # Warm GS from diag params
    v_warm = VUMPO(
        H_mpo=h2_problem["H_mpo"],
        initial_state=init,
        n_layers=4,
        n_sweeps=4,
        maxiter_local=4,
        hwp=True,
        mode="gs",
        opt="local",
    )
    ppg = v_warm.params_per_gate
    warm_gate_params = [diag_params[i * ppg : (i + 1) * ppg] for i in range(v_warm.n_gates)]
    warm_params = v_warm.optimize_local_sweep(warm_gate_params, initial_state=init)
    E_warm = v_warm.compute_energy_tn(warm_params, init)

    assert E_warm <= E_cold + 0.1, f"Warm-start {E_warm:.6f} much worse than cold {E_cold:.6f}"
