"""Boundary-contraction engine for the VUMPO networks.

Oracles are dense linear algebra at small L: the circuit unitary from
``VUMPO.build_tn`` and the Hamiltonian from the MPO, never the engine itself.
"""

import re
import sys

import numpy as np
import pytest

qtn = pytest.importorskip("quimb.tensor")

from qarp.operators import VUMPO
from qarp.operators._vumpo_network import NetworkSpec, make_engine


def _heisenberg_vumpo(L, n_layers, hwp, seed=0):
    rng = np.random.default_rng(seed)
    v = VUMPO(qtn.MPO_ham_heis(L), n_layers=n_layers, hwp=hwp)
    params = rng.normal(size=v.n_gates * v.params_per_gate)
    gates = [v.build_gate(p) for layer in v.layerise_params(params) for _, p in layer]
    return v, params, gates


def _dense_unitary(v, params):
    dim = 2**v.L
    tn = v.build_tn(params)
    out = [f"b{q}" for q in range(v.L)]
    inn = [f"k{q}" for q in range(v.L)]
    return tn.contract(output_inds=out + inn).data.reshape(dim, dim)


def _dense_hamiltonian(H):
    L = H.L
    dim = 2**L
    upper = [H.upper_ind(q) for q in range(L)]
    lower = [H.lower_ind(q) for q in range(L)]
    return H.contract(output_inds=upper + lower).data.reshape(dim, dim)


@pytest.mark.parametrize(
    "L,n_layers,hwp", [(4, 1, False), (5, 2, False), (6, 2, True), (6, 3, True)]
)
def test_doubled_network_value_matches_dense_sum_of_squared_diagonal(L, n_layers, hwp):
    v, params, gates = _heisenberg_vumpo(L, n_layers, hwp)
    U = _dense_unitary(v, params)
    Hd = _dense_hamiltonian(v.H_mpo)
    expected = float(np.sum(np.real(np.diag(U.conj().T @ Hd @ U)) ** 2))

    spec = NetworkSpec.doubled(v.H_mpo, n_layers)
    engine = make_engine(spec)
    value = engine.value(spec.arrays(gates))

    assert value == pytest.approx(expected, rel=1e-12)


def _ket(bits):
    vec = np.zeros(1, dtype=complex)
    vec[0] = 1.0
    for b in bits:  # quimb / VUMPO site order: site 0 is the leftmost kron factor
        e = np.zeros(2, dtype=complex)
        e[b] = 1.0
        vec = np.kron(vec, e)
    return vec


@pytest.mark.parametrize("L,n_layers,hwp", [(4, 2, False), (5, 3, True)])
def test_energy_network_value_matches_dense_expectation(L, n_layers, hwp):
    v, params, gates = _heisenberg_vumpo(L, n_layers, hwp, seed=1)
    ket = [1, 0, 1, 1, 0][:L]
    U = _dense_unitary(v, params)
    Hd = _dense_hamiltonian(v.H_mpo)
    s = _ket(ket)
    expected = float(np.real(s.conj() @ U.conj().T @ Hd @ U @ s))

    spec = NetworkSpec.energy(v.H_mpo, n_layers)
    value = make_engine(spec).value(spec.arrays(gates, ket=ket))

    assert value == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("L,n_layers,hwp", [(4, 2, False), (5, 3, True)])
def test_overlap_network_value_matches_dense_inner_product(L, n_layers, hwp):
    va, params_a, gates_a = _heisenberg_vumpo(L, n_layers, hwp, seed=2)
    _, params_b, gates_b = _heisenberg_vumpo(L, n_layers, hwp, seed=3)
    ket_a, ket_b = [0, 1, 1, 0, 1][:L], [1, 1, 0, 0, 1][:L]
    Ua, Ub = _dense_unitary(va, params_a), _dense_unitary(va, params_b)
    expected = complex(_ket(ket_a).conj() @ Ua.conj().T @ Ub @ _ket(ket_b))

    spec = NetworkSpec.overlap(L, n_layers)
    value = make_engine(spec).value(spec.arrays(gates_a, gates_b=gates_b, ket=ket_a, ket_b=ket_b))

    assert value == pytest.approx(expected, rel=1e-12, abs=1e-14)


# ── regime selection ─────────────────────────────────────────────────────


def test_deep_narrow_network_uses_searched_engine_and_matches_dense():
    from qarp.operators._vumpo_network import BOUNDARY_WIDTH_MAX, SearchedEngine

    v, params, gates = _heisenberg_vumpo(6, 8, True, seed=4)
    spec = NetworkSpec.doubled(v.H_mpo, 8)
    assert spec.width > BOUNDARY_WIDTH_MAX
    engine = make_engine(spec)
    assert isinstance(engine, SearchedEngine)

    U = _dense_unitary(v, params)
    Hd = _dense_hamiltonian(v.H_mpo)
    expected = float(np.sum(np.real(np.diag(U.conj().T @ Hd @ U)) ** 2))
    assert engine.value(spec.arrays(gates)) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("kind", ["doubled", "energy", "overlap"])
def test_searched_engine_agrees_with_boundary_engine(kind):
    """Invariance across regimes, additional to the dense rows above."""
    from qarp.operators._vumpo_network import BoundaryEngine, SearchedEngine

    v, params, gates = _heisenberg_vumpo(10, 3, False, seed=5)
    _, _, gates_b = _heisenberg_vumpo(10, 3, False, seed=55)
    spec = (
        NetworkSpec.overlap(10, 3) if kind == "overlap" else getattr(NetworkSpec, kind)(v.H_mpo, 3)
    )
    arrays = spec.arrays(gates, gates_b=gates_b, ket=[0, 1] * 5, ket_b=[1, 0] * 5)
    a = BoundaryEngine(spec).value(arrays)
    b = SearchedEngine(spec).value(arrays)
    assert b == pytest.approx(a, rel=1e-12)


# ── VUMPO routed through the engine ──────────────────────────────────────


def test_vumpo_cost_sites_use_the_engine_not_quimb_contraction(monkeypatch):
    """The three scalar evaluations go through the spec/engine; a per-call
    ``TensorNetwork.contract`` would put path search back on the hot path."""
    v, params, _ = _heisenberg_vumpo(5, 2, True, seed=6)
    U = _dense_unitary(v, params)
    Hd = _dense_hamiltonian(v.H_mpo)
    ket = [1, 0, 0, 1, 0]
    s = _ket(ket)
    expected_cost = float(np.sum(np.real(np.diag(U.conj().T @ Hd @ U)) ** 2))
    expected_energy = float(np.real(s.conj() @ U.conj().T @ Hd @ U @ s))
    expected_overlap = float(np.real(s.conj() @ U.conj().T @ U @ s))

    def forbidden(*args, **kwargs):
        raise AssertionError("TensorNetwork.contract called on the evaluation path")

    monkeypatch.setattr(qtn.TensorNetwork, "contract", forbidden)

    assert v._cost_tn(params) == pytest.approx(expected_cost, rel=1e-12)
    assert v.compute_energy_tn(params, ket) == pytest.approx(expected_energy, rel=1e-12)
    assert v._overlap_tn(params, params, ket, ket) == pytest.approx(expected_overlap, rel=1e-12)


# ── environments and the polynomial in one gate ───────────────────────────


def _replace_gate(gates, k, G):
    out = list(gates)
    out[k] = G
    return out


def _random_unitary(rng, hwp):
    if hwp:
        g = np.zeros((4, 4), dtype=complex)
        g[0, 0], g[3, 3] = np.exp(1j * rng.normal()), np.exp(1j * rng.normal())
        g[1:3, 1:3] = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
        return g
    return np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]


def _dense_unitary_from_gates(v, gates):
    """Circuit unitary from explicit gate matrices, pure numpy: site 0 is the
    leftmost kron factor and layer 0 acts first."""
    dim = 2**v.L
    U = np.eye(dim, dtype=complex)
    for k, (m, n) in enumerate(
        [(m, n) for m in range(v.n_layers) for n in range(m % 2, v.L - 1, 2)]
    ):
        full = np.kron(np.kron(np.eye(2**n), gates[k]), np.eye(2 ** (v.L - n - 2)))
        U = full @ U
    return U


def test_dense_helper_matches_build_tn():
    v, params, gates = _heisenberg_vumpo(5, 3, False, seed=99)
    np.testing.assert_allclose(
        _dense_unitary_from_gates(v, gates), _dense_unitary(v, params), atol=1e-13
    )


@pytest.mark.parametrize("L,n_layers,hwp", [(5, 2, False), (6, 3, True)])
def test_doubled_environment_gives_cost_as_quartic_in_the_gate(L, n_layers, hwp):
    from qarp.operators._vumpo_network import polynomial_value_and_w

    v, params, gates = _heisenberg_vumpo(L, n_layers, hwp, seed=7)
    rng = np.random.default_rng(8)
    Hd = _dense_hamiltonian(v.H_mpo)
    spec = NetworkSpec.doubled(v.H_mpo, n_layers)
    engine = make_engine(spec)
    arrays = spec.arrays(gates)
    for k in range(spec.n_gates):
        env = engine.environment_full(k, arrays)
        assert env.shape == (16,) * 4
        G = _random_unitary(rng, hwp)
        U = _dense_unitary_from_gates(v, _replace_gate(gates, k, G))
        expected = float(np.sum(np.real(np.diag(U.conj().T @ Hd @ U)) ** 2))
        value, _ = polynomial_value_and_w(env, G, spec.slot_conj(k))
        assert value.real == pytest.approx(expected, rel=1e-12)


def test_energy_and_overlap_environments_give_cost_as_polynomial_in_the_gate():
    from qarp.operators._vumpo_network import polynomial_value_and_w

    L, n_layers = 5, 2
    v, params, gates = _heisenberg_vumpo(L, n_layers, False, seed=9)
    _, _, gates_b = _heisenberg_vumpo(L, n_layers, False, seed=10)
    rng = np.random.default_rng(11)
    Hd = _dense_hamiltonian(v.H_mpo)
    ket_a, ket_b = [1, 0, 1, 0, 0], [0, 1, 1, 0, 0]
    sa, sb = _ket(ket_a), _ket(ket_b)
    e_spec = NetworkSpec.energy(v.H_mpo, n_layers)
    o_spec = NetworkSpec.overlap(L, n_layers)
    e_arrays = e_spec.arrays(gates, ket=ket_a)
    o_arrays = o_spec.arrays(gates, gates_b=gates_b, ket=ket_a, ket_b=ket_b)
    e_engine, o_engine = make_engine(e_spec), make_engine(o_spec)
    Ub = _dense_unitary_from_gates(v, gates_b)
    for k in range(e_spec.n_gates):
        G = _random_unitary(rng, False)
        U = _dense_unitary_from_gates(v, _replace_gate(gates, k, G))
        env = e_engine.environment_full(k, e_arrays)
        assert env.shape == (16, 16)
        value, _ = polynomial_value_and_w(env, G, e_spec.slot_conj(k))
        assert value.real == pytest.approx(
            float(np.real(sa.conj() @ U.conj().T @ Hd @ U @ sa)), rel=1e-12
        )
        env = o_engine.environment_full(k, o_arrays)
        assert env.shape == (16,)
        value, _ = polynomial_value_and_w(env, G, o_spec.slot_conj(k))
        assert value == pytest.approx(
            complex(sa.conj() @ U.conj().T @ Ub @ sb), rel=1e-12, abs=1e-14
        )


@pytest.mark.parametrize("degree", [1, 2, 4])
def test_polynomial_w_matches_finite_differences_in_the_gate_matrix(degree):
    """d Re(value) = Re Tr(W dG) for complex perturbations of every entry."""
    from qarp.operators._vumpo_network import polynomial_value_and_w

    rng = np.random.default_rng(12)
    env = rng.normal(size=(16,) * degree) + 1j * rng.normal(size=(16,) * degree)
    conj = tuple(bool(i % 2) for i in range(degree))  # (False, True, False, True)[:degree]
    G = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    _, W = polynomial_value_and_w(env, G, conj)
    eps = 1e-6
    for a in range(4):
        for b in range(4):
            for d in (1.0, 1j):
                dG = np.zeros((4, 4), dtype=complex)
                dG[a, b] = d
                plus = polynomial_value_and_w(env, G + eps * dG, conj)[0].real
                minus = polynomial_value_and_w(env, G - eps * dG, conj)[0].real
                assert (plus - minus) / (2 * eps) == pytest.approx(
                    np.real(np.trace(W @ dG)), rel=1e-6, abs=1e-8
                )


def test_incremental_boundaries_equal_rebuilt_boundaries():
    from qarp.operators._vumpo_network import BoundaryEngine

    v, params, gates = _heisenberg_vumpo(8, 2, False, seed=13)
    spec = NetworkSpec.doubled(v.H_mpo, 2)
    engine = BoundaryEngine(spec)
    arrays = spec.arrays(gates)
    left = engine.left_boundaries(arrays)
    right = engine.right_boundaries(arrays)
    assert left[0] is None and right[spec.L] is None
    for c in range(1, spec.L):
        B = None
        for col in range(c):
            B = engine.advance(B, arrays, col)
        np.testing.assert_allclose(left[c].data, B.data, rtol=1e-13)
        assert left[c].inds == B.inds
        R = None
        for col in range(spec.L - 1, c - 1, -1):
            R = engine.advance(R, arrays, col)
        np.testing.assert_allclose(right[c].data, R.data, rtol=1e-13)
        assert right[c].inds == R.inds
    # closing any left/right pair gives the value
    for c in range(1, spec.L):
        closed = engine.close(left[c], right[c])
        assert closed == pytest.approx(engine.value(arrays), rel=1e-12)


# ── gradient in the gate parameters, and the sweep ───────────────────────


@pytest.mark.parametrize("hwp", [False, True])
@pytest.mark.parametrize("degenerate", [False, True])
def test_gate_gradient_matches_scipy_expm_frechet(hwp, degenerate):
    """Re Tr(W dG/dθ_i) via Daleckii–Krein against scipy's Fréchet derivative,
    including a degenerate generator spectrum where the divided difference
    collapses to the exponential itself."""
    from scipy.linalg import expm_frechet

    rng = np.random.default_rng(14)
    v = VUMPO(qtn.MPO_ham_heis(4), n_layers=1, hwp=hwp)
    ppg = v.params_per_gate
    p = np.zeros(ppg) if degenerate else rng.normal(size=ppg)
    if degenerate and not hwp:
        p[:4] = 0.7  # equal diagonal generator entries, no off-diagonal terms
    W = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))

    def dG(i):
        if hwp:
            d = np.zeros((4, 4), dtype=complex)
            if i == 0:
                d[0, 0] = 1j * np.exp(1j * p[0])
            elif i == 1:
                d[3, 3] = 1j * np.exp(1j * p[1])
            else:
                e = np.zeros(4)
                e[i - 2] = 1.0
                d[1:3, 1:3] = expm_frechet(
                    v.skew_hermitian(p[2:6], 2), v.skew_hermitian(e, 2), compute_expm=False
                )
            return d
        e = np.zeros(16)
        e[i] = 1.0
        return expm_frechet(v.skew_hermitian(p, 4), v.skew_hermitian(e, 4), compute_expm=False)

    expected = np.array([np.real(np.trace(W @ dG(i))) for i in range(ppg)])
    np.testing.assert_allclose(v._gate_gradient(W, p), expected, rtol=1e-10, atol=1e-12)


def _configs():
    for hwp in (False, True):
        yield dict(hwp=hwp, mode="diag", alpha=0.0, beta=0.0)
        yield dict(hwp=hwp, mode="diag", alpha=0.3, beta=0.0)
        yield dict(hwp=hwp, mode="gs", alpha=0.0, beta=0.0)
        yield dict(hwp=hwp, mode="gs", alpha=0.0, beta=0.7)


@pytest.mark.parametrize(
    "cfg",
    list(_configs()),
    ids=lambda c: f"{'hwp' if c['hwp'] else 'full'}-{c['mode']}-a{c['alpha']}-b{c['beta']}",
)
def test_local_objective_matches_cost_fn_and_central_differences(cfg):
    L, n_layers = 5, 2
    rng = np.random.default_rng(15)
    v = VUMPO(qtn.MPO_ham_heis(L), n_layers=n_layers, **cfg)
    ppg = v.params_per_gate
    flat = rng.normal(size=v.n_gates * ppg)
    init = [1, 0, 1, 0, 0]
    prev = [
        (rng.normal(size=v.n_gates * ppg), [0, 1, 0, 1, 0]),
        (rng.normal(size=v.n_gates * ppg), [1, 1, 0, 0, 0]),
    ]
    prev_states = prev if cfg["beta"] else None

    gates = v._gates(flat)
    terms = v._terms(gates, init, prev_states)
    v._fill_boundaries(terms, left=True, right=True)
    for k in (0, v.n_gates // 2, v.n_gates - 1):
        f = v._local_objective(k, terms)
        pk = flat[k * ppg : (k + 1) * ppg]

        def full(p):
            g = flat.copy()
            g[k * ppg : (k + 1) * ppg] = p
            return v.cost_fn(g, init, prev_states)

        value, grad = f(pk)
        assert value == pytest.approx(full(pk), rel=1e-12)
        eps = 1e-5
        fd = np.array(
            [
                (full(pk + eps * np.eye(ppg)[i]) - full(pk - eps * np.eye(ppg)[i])) / (2 * eps)
                for i in range(ppg)
            ]
        )
        np.testing.assert_allclose(grad, fd, rtol=1e-6, atol=1e-8)


def test_visit_order_is_column_major_and_mirrored_on_odd_sweeps():
    v = VUMPO(qtn.MPO_ham_heis(6), n_layers=3)
    # gates: layer 0 at sites 0,2,4 -> k 0,1,2; layer 1 at 1,3 -> k 3,4; layer 2 at 0,2,4 -> k 5,6,7
    assert v._visit_order(0) == [(0, [0, 5]), (1, [3]), (2, [1, 6]), (3, [4]), (4, [2, 7])]
    assert v._visit_order(1) == [(4, [7, 2]), (3, [4]), (2, [6, 1]), (1, [3]), (0, [5, 0])]


@pytest.mark.parametrize("mode", ["diag", "gs"])
def test_one_sweep_never_increases_the_cost(mode):
    rng = np.random.default_rng(16)
    v = VUMPO(
        qtn.MPO_ham_heis(6), n_layers=2, mode=mode, n_sweeps=1, maxiter_local=3, hwp=(mode == "gs")
    )
    init = [1, 0, 1, 0, 1, 0]
    v.initial_state = init
    gate_params = [rng.normal(size=v.params_per_gate) for _ in range(v.n_gates)]
    before = v.cost_fn(np.concatenate(gate_params), init)
    after_params = v.optimize_local_sweep(gate_params, initial_state=init)
    after = v.cost_fn(after_params, init)
    assert after <= before + 1e-9 * abs(before)
    assert after < before  # random start: the sweep actually moves


def test_global_gradient_matches_finite_differences_and_reaches_the_same_optimum():
    from scipy.optimize import minimize

    rng = np.random.default_rng(17)
    v = VUMPO(qtn.MPO_ham_heis(4), n_layers=2, maxiter_global=20)
    flat = rng.normal(size=v.n_gates * v.params_per_gate)
    value, grad = v._value_and_gradient(flat, None, None)
    assert value == pytest.approx(v.cost_fn(flat), rel=1e-12)
    eps = 1e-5
    n = flat.size
    fd = np.array(
        [
            (v.cost_fn(flat + eps * np.eye(n)[i]) - v.cost_fn(flat - eps * np.eye(n)[i]))
            / (2 * eps)
            for i in range(n)
        ]
    )
    np.testing.assert_allclose(grad, fd, rtol=1e-6, atol=1e-8)

    start = [flat[k * 16 : (k + 1) * 16] for k in range(v.n_gates)]
    ours = v.optimize_global(start)
    ref = minimize(lambda p: v.cost_fn(p), flat, method="L-BFGS-B", options={"maxiter": 20})
    assert v.cost_fn(ours) <= ref.fun + 1e-6 * abs(ref.fun)


@pytest.mark.bench
def test_per_gate_visit_cost_is_flat_in_L():
    """The paper's step (iv): a sweep is O(L), so the per-visit cost must not
    grow with L.  Ratio bound 3 absorbs cache and allocator noise."""
    import time

    per_visit = {}
    for L in (8, 20, 50):
        rng = np.random.default_rng(L)
        v = VUMPO(qtn.MPO_ham_heis(L), n_layers=2, n_sweeps=1, maxiter_local=5)
        gate_params = [rng.normal(size=v.params_per_gate) for _ in range(v.n_gates)]
        v.optimize_local_sweep(gate_params, initial_state=[0] * L)  # warm the expression caches
        t = time.perf_counter()
        v.optimize_local_sweep(gate_params, initial_state=[0] * L)
        per_visit[L] = (time.perf_counter() - t) / v.n_gates
    print({L: f"{t * 1e3:.2f} ms" for L, t in per_visit.items()})
    assert per_visit[50] < 3 * per_visit[8]


# ── MPO route and the paper's diagnostics ─────────────────────────────────


def _random_qubit_operator(L, n_terms, rng):
    from qarp.operators import QubitOperator

    op = QubitOperator()
    for _ in range(n_terms):
        qs = sorted(rng.choice(L, size=rng.integers(1, 4), replace=False))
        term = " ".join(f"{rng.choice(['X', 'Y', 'Z'])}{q}" for q in qs)
        op += QubitOperator(term, float(rng.normal()))
    return op


@pytest.mark.parametrize("L", [3, 5, 6])
def test_qubit_operator_to_mpo_matches_dense_construction(L):
    from qarp.endianness import lsb_to_msb_matrix
    from qarp.operators import qubit_operator_to_mpo

    rng = np.random.default_rng(L)
    op = _random_qubit_operator(L, 12, rng)
    expected = qtn.MatrixProductOperator.from_dense(
        lsb_to_msb_matrix(op.sparse_matrix().toarray()), dims=(2,) * L
    )
    mpo = qubit_operator_to_mpo(op, L)
    np.testing.assert_allclose(_dense_hamiltonian(mpo), _dense_hamiltonian(expected), atol=1e-10)


def heisenberg_chain(L, fields, J=1.0):
    """Eq. (5) of the paper with spin-1/2 operators S = sigma / 2."""
    from qarp.operators import QubitOperator

    op = QubitOperator()
    for n in range(L - 1):
        for p in "XYZ":
            op += QubitOperator(f"{p}{n} {p}{n + 1}", J / 4)
    for n, h in enumerate(fields):
        op += QubitOperator(f"Z{n}", -h / 2)
    return op


@pytest.mark.parametrize("L", [8, 16, 28])
def test_heisenberg_mpo_bond_dimension_is_five_at_every_length(L):
    """Appendix A's chi is set by the operator's locality, not by L."""
    from qarp.operators import qubit_operator_to_mpo

    fields = np.random.default_rng(L).uniform(-8, 8, size=L)
    assert qubit_operator_to_mpo(heisenberg_chain(L, fields), L).max_bond() == 5


def test_trace_h2_matches_dense_trace():
    from qarp.operators import qubit_operator_to_mpo

    L = 5
    op = _random_qubit_operator(L, 10, np.random.default_rng(21))
    v = VUMPO(qubit_operator_to_mpo(op, L), n_layers=1)
    Hd = op.sparse_matrix().toarray()
    assert v.trace_h2() == pytest.approx(float(np.real(np.trace(Hd @ Hd))), rel=1e-12)


def test_energy_variance_matches_dense_summed_variance_and_is_nonnegative():
    v, params, _ = _heisenberg_vumpo(5, 2, False, seed=22)
    U = _dense_unitary(v, params)
    Hd = _dense_hamiltonian(v.H_mpo)
    UHU = U.conj().T @ Hd @ U
    expected = float(np.real(np.trace(UHU @ UHU)) - np.sum(np.real(np.diag(UHU)) ** 2))
    f = v.energy_variance(params)
    assert f == pytest.approx(expected, rel=1e-12)
    assert f >= 0.0


def test_off_diagonal_ratio_matches_dense_ratio():
    v, params, _ = _heisenberg_vumpo(5, 2, True, seed=23)
    U = _dense_unitary(v, params)
    Hd = _dense_hamiltonian(v.H_mpo)
    UHU = U.conj().T @ Hd @ U
    off = UHU - np.diag(np.diag(UHU))
    expected = np.linalg.norm(off) / np.linalg.norm(UHU)
    assert v.off_diagonal_ratio(params) == pytest.approx(expected, rel=1e-10)


def test_one_local_minimisation_never_increases_the_cost():
    """Per visit, not only per sweep: L-BFGS with the exact gradient starts
    from the current gate and can only descend."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(31)
    v = VUMPO(qtn.MPO_ham_heis(6), n_layers=2, maxiter_local=3)
    flat = rng.normal(size=v.n_gates * v.params_per_gate)
    ppg = v.params_per_gate
    terms = v._terms(v._gates(flat), None, None)
    v._fill_boundaries(terms, left=True, right=True)
    for k in range(v.n_gates):
        f = v._local_objective(k, terms)
        p0 = flat[k * ppg : (k + 1) * ppg]
        res = minimize(f, p0, jac=True, method="L-BFGS-B", options={"maxiter": 3})
        assert f(res.x)[0] <= f(p0)[0] + 1e-12 * abs(f(p0)[0])


def test_qubit_operator_to_mpo_rejects_terms_outside_the_register():
    from qarp.operators import QubitOperator, qubit_operator_to_mpo

    with pytest.raises(ValueError, match="qubit 4"):
        qubit_operator_to_mpo(QubitOperator("X1 Z4", 1.0), 3)


def test_engine_third_party_imports_are_declared_in_the_mps_extra():
    """`import qarp.operators` pulls the engine in whenever quimb is present,
    so every third-party module the engine imports must be installed by the
    same extra (or by the core dependencies).  opt_einsum once was not."""
    import ast
    import pathlib
    import tomllib

    root = pathlib.Path(__file__).resolve().parents[2]
    project = tomllib.load((root / "pyproject.toml").open("rb"))["project"]
    declared = {
        re.split(r"[<>=!\[ ;]", dep)[0].lower().replace("-", "_")
        for dep in project["dependencies"] + project["optional-dependencies"]["mps"]
    }
    tree = ast.parse((root / "qarp" / "operators" / "_vumpo_network.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    third_party = {m for m in imported if m not in sys.stdlib_module_names and m != "qarp"}
    assert third_party, "sanity: the engine imports numpy and opt_einsum"
    assert third_party <= declared, f"undeclared: {third_party - declared}"
