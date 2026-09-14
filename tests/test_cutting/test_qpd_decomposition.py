"""Tests for QPDDecomposition and CuttingPrimitive."""

import numpy as np
import pytest

openfermion = pytest.importorskip("openfermion")
from qarp.operators import QubitOperator
from qarp.operators.compat import from_openfermion


def fermi_hubbard(*args, **kwargs):
    """openfermion's reference builder, converted at the boundary."""
    return from_openfermion(openfermion.fermi_hubbard(*args, **kwargs))


from sympy import Symbol

import qarpx as qx
from qarp.cutting import EAPartitioning, QPDDecomposition
from qarp.cutting._qpd_decomposition import _post_process_shot
from qarp.operators import JordanWigner

# ── Circuit builders ────────────────────────────────────────────────────────


def _toy_circuit_1() -> tuple[list, int]:
    n = 3
    b = qx.SimpleBlock(n, "toy1")
    b.h(0)
    b.h(1)
    b.h(2)
    b.cx(0, 1)
    b.cx(1, 2)
    b.cx(0, 1)
    b.cx(1, 2)
    b.set_built(True)
    return list(b.flatten()), n


def _toy_circuit_2() -> tuple[list, int]:
    n = 3
    b = qx.SimpleBlock(n, "toy2")
    b.h(0)
    b.h(1)
    b.h(2)
    b.cx(0, 1)
    b.cx(1, 2)
    b.cx(1, 0)
    b.cx(2, 1)
    b.set_built(True)
    return list(b.flatten()), n


def _toy_circuit_rzz() -> tuple[list, int, float]:
    n = 2
    theta = np.random.rand()
    b = qx.SimpleBlock(n, "rzz")
    b.h(0)
    b.h(1)
    b.rx(0, 0.2)
    b.rzz(0, 1, theta)
    b.set_built(True)
    return list(b.flatten()), n, theta


def _hamiltonian():
    """Factory, not a module global: a module-scope qarpx object stays alive to
    interpreter shutdown and nanobind reports it leaked (AGENTS.md landmine)."""
    return (
        +0.16988452027940318 * QubitOperator("Z0")
        + -0.21886306781219608 * QubitOperator("Z0 Z1 Z2")
        + 0.04544288414432624 * QubitOperator("Y0 Y1 Y2")
        + 0.04544288414432624
    )


def _make_qpd(circuit_cmds, n_qubits, manual_setting, n_shots=1000):
    cutter = EAPartitioning(circuit_cmds, n_qubits, [len(s) + 1 for s in manual_setting])
    result = cutter.cut(manual_setting=manual_setting)
    return QPDDecomposition(result, _hamiltonian(), verbose=False, n_shots=n_shots)


# ── Unit tests ──────────────────────────────────────────────────────────────


def test_generate_experiments_full_circuit():
    cmds, n = _toy_circuit_1()
    qpd = _make_qpd(cmds, n, [[0], [1, 2]])
    qpd.decompose()
    # decompose() does not materialise experiments; use the n_experiments property
    assert qpd.n_experiments == 6**qpd.n_cuts
    # coefficients are filled lazily by compute()
    qpd.compute()
    assert len(qpd.coefficients) == 6**qpd.n_cuts


def test_split_observable():
    cmds, n = _toy_circuit_1()
    qpd = _make_qpd(cmds, n, [[0], [1, 2]])
    qpd._split_observable()
    assert len(qpd.sub_observables.keys()) == qpd.n_subcircuits


def test_generate_experiments_subcircuits():
    """self.jobs is not materialised; verify via n_experiments and n_subcircuits."""
    cmds, n = _toy_circuit_1()
    qpd = _make_qpd(cmds, n, [[0], [1, 2]])
    qpd.decompose()
    assert qpd.n_subcircuits == 2
    assert qpd.n_experiments == 6**qpd.n_cuts


def test_qpd_factor_computation():
    """_post_process_shot returns ±1 based on QPD bit parity."""
    assert _post_process_shot([0]) == 1
    assert _post_process_shot([1]) == -1
    assert _post_process_shot([0, 0]) == 1
    assert _post_process_shot([0, 1]) == -1
    assert _post_process_shot([1, 0]) == -1
    assert _post_process_shot([1, 1]) == 1


def test_overhead():
    cmds, n = _toy_circuit_1()
    cutter = EAPartitioning(cmds, n, [2, 2])
    result = cutter.cut(manual_setting=[[0], [1, 2]])
    qpd = QPDDecomposition(result, _hamiltonian(), verbose=False, n_shots=100)
    assert qpd.overhead() == 6**cutter.n_cuts


def test_qpd_compute_shape():
    """compute() returns a float and reconstruction_exp_val has correct shape."""
    cmds, n = _toy_circuit_1()
    qpd = _make_qpd(cmds, n, [[0], [1, 2]], n_shots=500)
    qpd.decompose()
    qpd.compute()
    assert len(qpd.reconstruction_exp_val) == qpd.n_valid_obs_terms


def test_reconstruct_expectation_value_2():
    cmds, n = _toy_circuit_2()
    qpd = _make_qpd(cmds, n, [[0], [1, 2]], n_shots=500)
    qpd.decompose()
    qpd.compute()
    assert len(qpd.reconstruction_exp_val) == qpd.n_valid_obs_terms


def test_random_seed_reproducibility():
    """Standalone shot simulation is seeded via ``shot_seed=`` (explicit,
    per-experiment-derived), not via the process-wide config.seed."""
    cmds, n = _toy_circuit_1()
    cutter = EAPartitioning(cmds, n, [2, 2])
    result = cutter.cut(manual_setting=[[0], [1, 2]])

    qpd1 = QPDDecomposition(result, _hamiltonian(), verbose=False, n_shots=500, shot_seed=1234)
    qpd1.decompose()
    ev1 = qpd1.compute()

    qpd2 = QPDDecomposition(result, _hamiltonian(), verbose=False, n_shots=500, shot_seed=1234)
    qpd2.decompose()
    ev2 = qpd2.compute()

    assert ev1 == ev2

    qpd3 = QPDDecomposition(result, _hamiltonian(), verbose=False, n_shots=500, shot_seed=99)
    qpd3.decompose()
    assert qpd3.compute() != ev1  # different seed → different tape


def test_ghz_circuit():
    from qarp import config

    config.seed = 42

    def build_circ(nsites):
        hubbard = fermi_hubbard(1, nsites, tunneling=2.87, coulomb=3.13, periodic=False)
        hubbard_qubit = JordanWigner().encode_operator(hubbard)
        n = 2 * nsites
        b = qx.SimpleBlock(n, "ghz")
        b.h(0)
        for i in range(n - 1):
            b.cx(i, i + 1)
        b.set_built(True)
        return list(b.flatten()), n, hubbard_qubit

    m_c = [[0], [1], [2, 3]]
    cmds, n, hubbard_qubit = build_circ(nsites=2)
    cutter = EAPartitioning(cmds, n, [len(s) + 1 for s in m_c])
    result = cutter.cut(manual_setting=m_c)
    qpd = QPDDecomposition(result, hubbard_qubit, n_shots=10000, verbose=False)
    qpd.decompose()
    ev = qpd.compute()
    assert abs(ev - 3.1299) <= 0.2


def test_ghz_statevector_reconstruction():
    n = 4
    b = qx.SimpleBlock(n, "ghz4")
    b.h(0)
    b.cx(0, 1)
    b.cx(1, 2)
    b.cx(2, 3)
    b.set_built(True)
    cmds = list(b.flatten())

    toy_obs = QubitOperator("X0 X0", 1.0)
    cuts = [[0, 1], [2, 3]]

    cutter = EAPartitioning(cmds, n, [2, 2])
    result = cutter.cut(manual_setting=cuts)
    qpd = QPDDecomposition(result, toy_obs, n_shots=5000, verbose=False)
    qpd.decompose()

    SV = qpd.reconstruct_statevector()
    assert abs(sum(SV.values()) - 1.0) <= 0.1
    assert abs(SV.get("0000", 0) - 0.5) <= 0.1
    assert abs(SV.get("1111", 0) - 0.5) <= 0.1


def test_decompose_parameters():
    beta1 = Symbol("b1")
    beta2 = Symbol("b2")
    n = 2
    b = qx.SimpleBlock(n, "param")
    b.rx(0, qx.Param.symbol("b1"))
    b.rx(1, qx.Param.symbol("b2"))
    b.cx(0, 1)
    b.set_built(True)
    cmds = list(b.flatten())

    toy_obs = QubitOperator("Z0 Z1")
    cutter = EAPartitioning(cmds, n, [1, 1])
    result = cutter.cut(manual_setting=[[0], [1]])
    qpd = QPDDecomposition(result, toy_obs, n_shots=5000, verbose=False)
    qpd.decompose()

    evs = []
    # (0,0): Rx(0)=I → Z0Z1=+1.  (0,pi): Rx(pi) flips qubit1 → Z0Z1=-1.
    for beta1_val, beta2_val in [(0.0, 0.0), (0.0, np.pi)]:
        ev = qpd.compute(symbol_map={beta1: beta1_val, beta2: beta2_val})
        evs.append(ev)

    assert evs[0] > evs[1]


def test_qwc_grouping_structure():
    """QWC groups collapse commuting observable terms into shared measurement circuits.

    For a Heisenberg + transverse-field Hamiltonian (XX, YY, ZZ two-body +
    X, Z single-site terms), the terms partition into exactly three QWC groups
    per subcircuit — one per measurement basis:
      - Z-basis group: ZZ and Z terms (no rotation needed)
      - X-basis group: XX and X terms (H rotation)
      - Y-basis group: YY terms (Rx(π/2) rotation)
    """
    import math

    n = 6

    # Build a Heisenberg + transverse-field Hamiltonian with all three bases
    H = QubitOperator()
    for i in range(n - 1):
        H += 1.0 * QubitOperator(f"X{i} X{i + 1}")  # X-basis
        H += 1.0 * QubitOperator(f"Y{i} Y{i + 1}")  # Y-basis
        H += 1.0 * QubitOperator(f"Z{i} Z{i + 1}")  # Z-basis
    for i in range(n):
        H += 0.8 * QubitOperator(f"Z{i}")  # Z-basis
        H += 0.5 * QubitOperator(f"X{i}")  # X-basis
    # 27 non-identity terms: 5×XX + 5×YY + 5×ZZ + 6×Z + 6×X

    b = qx.SimpleBlock(n, "heis")
    for q in range(n):
        b.ry(q, math.pi / 3)
    for q in range(n - 1):
        b.cx(q, q + 1)
    b.set_built(True)
    cmds = list(b.flatten())

    # 1 cut: device = 3 qubits → subcircuits [[0,1,2], [3,4,5]]
    cutter = EAPartitioning(cmds, n, [4, 4])
    result = cutter.cut(manual_setting=[[0, 1, 2], [3, 4, 5]])

    qpd = QPDDecomposition(result, H, n_shots=100, verbose=False)
    qpd._split_observable()

    # Both subcircuits should have exactly 3 QWC groups (X / Y / Z basis)
    for i_sub in range(qpd.n_subcircuits):
        groups = qpd.sub_observable_qwc_groups[str(i_sub)]
        bases = qpd.sub_observable_qwc_bases[str(i_sub)]

        assert len(groups) == 3, (
            f"Subcircuit {i_sub}: expected 3 QWC groups, got {len(groups)}. Bases found: {bases}"
        )

        # Each group must cover distinct Pauli bases (X, Y, Z — one each)
        pauli_sets = set()
        for basis in bases:
            paulis = frozenset(basis.values())
            assert paulis, "QWC group has no basis — should be at least one Pauli"
            pauli_sets.add(paulis)
        assert len(pauli_sets) == 3, (
            f"Subcircuit {i_sub}: expected 3 distinct basis sets, got {pauli_sets}"
        )

        # Total term coverage: all 27 terms must appear in some group
        all_covered = set()
        for grp in groups:
            all_covered.update(grp)
        assert all_covered == set(range(qpd.n_valid_obs_terms)), (
            "Not all observable terms covered by QWC groups"
        )


def test_qwc_grouping_reduces_blocks():
    """QWC grouping produces fewer blocks than the naïve per-term approach."""

    n = 4
    H = QubitOperator()
    for i in range(n - 1):
        H += 1.0 * QubitOperator(f"X{i} X{i + 1}")
        H += 1.0 * QubitOperator(f"Y{i} Y{i + 1}")
        H += 1.0 * QubitOperator(f"Z{i} Z{i + 1}")
    for i in range(n):
        H += 0.8 * QubitOperator(f"Z{i}")
        H += 0.5 * QubitOperator(f"X{i}")

    b = qx.SimpleBlock(n, "h4")
    for q in range(n):
        b.h(q)
    for q in range(n - 1):
        b.cx(q, q + 1)
    b.set_built(True)
    cmds = list(b.flatten())

    cutter = EAPartitioning(cmds, n, [3, 3])
    result = cutter.cut(manual_setting=[[0, 1], [2, 3]])

    qpd = QPDDecomposition(result, H, n_shots=100, verbose=False)
    qpd._split_observable()

    n_exps = 6**qpd.n_cuts
    n_terms = qpd.n_valid_obs_terms
    n_subs = qpd.n_subcircuits
    naïve_blocks = n_exps * n_subs * n_terms

    total_qwc_groups = sum(len(qpd.sub_observable_qwc_groups[str(i)]) for i in range(n_subs))
    qwc_blocks = n_exps * total_qwc_groups

    assert qwc_blocks < naïve_blocks, (
        f"QWC grouping should reduce block count: {qwc_blocks} vs naïve {naïve_blocks}"
    )
    # For this Hamiltonian (3 bases) the reduction should be at least 3×
    assert naïve_blocks / qwc_blocks >= 3.0


def test_experiment_fraction_subsampling():
    """experiment_fraction runs a fraction of experiments with auto strategy selection."""
    from qarp import config

    config.seed = 42

    cmds, n = _toy_circuit_1()
    cutter = EAPartitioning(cmds, n, [2, 2])
    result = cutter.cut(manual_setting=[[0], [1, 2]])

    total = 6**result.n_cuts

    # Full run
    qpd_full = QPDDecomposition(result, _hamiltonian(), n_shots=500, verbose=False)
    qpd_full.decompose()
    ev_full = qpd_full.compute()

    # 50% subsampled run via fraction
    qpd_sub = QPDDecomposition(
        result,
        _hamiltonian(),
        n_shots=500,
        verbose=False,
        experiment_fraction=0.5,
        rng_seed=0,
    )
    qpd_sub.decompose()

    # CX circuit → auto-selects "uniform"
    assert qpd_sub.sampling_strategy == "uniform"
    assert qpd_sub.n_experiments == max(1, int(total * 0.5))
    ev_sub = qpd_sub.compute()
    assert len(qpd_sub.coefficients) == qpd_sub.n_experiments

    # Both results should be in the same rough ballpark
    assert abs(ev_sub - ev_full) < 2.0


def test_experiment_fraction_auto_strategy():
    """CX cuts → uniform; RZZ cuts → top_k auto-selection."""
    import math

    cmds, n = _toy_circuit_1()  # CX circuit
    cutter = EAPartitioning(cmds, n, [2, 2])
    result_cx = cutter.cut(manual_setting=[[0], [1, 2]])
    qpd_cx = QPDDecomposition(
        result_cx, _hamiltonian(), n_shots=100, verbose=False, experiment_fraction=0.5
    )
    assert qpd_cx.sampling_strategy == "uniform"

    # RZZ circuit → top_k
    b = qx.SimpleBlock(2, "rzz")
    b.h(0)
    b.h(1)
    b.rzz(0, 1, math.pi / 4)
    b.set_built(True)
    rzz_cmds = list(b.flatten())
    rzz_obs = QubitOperator("Z0 Z1")
    cutter2 = EAPartitioning(rzz_cmds, 2, [1, 1])
    result_rzz = cutter2.cut(manual_setting=[[0], [1]])
    qpd_rzz = QPDDecomposition(
        result_rzz, rzz_obs, n_shots=100, verbose=False, experiment_fraction=0.5
    )
    assert qpd_rzz.sampling_strategy == "top_k"


def test_generate_single_experiment():
    """_generate_single_experiment(k) matches the k-th result from _iter_experiments."""
    cmds, n = _toy_circuit_1()
    cutter = EAPartitioning(cmds, n, [2, 2])
    result = cutter.cut(manual_setting=[[0], [1, 2]])
    qpd = QPDDecomposition(result, _hamiltonian(), n_shots=100, verbose=False)

    iter_exps = list(qpd._iter_experiments())
    for k in range(6):
        single_cmds, single_coeff = qpd._generate_single_experiment(k)
        iter_cmds, iter_coeff = iter_exps[k]
        assert abs(single_coeff - iter_coeff) < 1e-12
        assert len(single_cmds) == len(iter_cmds)


# ── Grouping-strategy guard ─────────────────────────────────────────────


def test_rejects_non_qubit_wise_grouping():
    """Cutting derives per-qubit measurement bases per subcircuit — an
    entangling-Clifford strategy cannot apply and must be rejected."""
    from qarp.operators import FullyCommuting, QubitWiseCommuting

    cmds, n = _toy_circuit_1()
    cutter = EAPartitioning(cmds, n, [2, 3])
    result = cutter.cut(manual_setting=[[0], [1, 2]])
    with pytest.raises(ValueError, match="qubit-wise"):
        QPDDecomposition(result, _hamiltonian(), verbose=False, grouping=FullyCommuting())
    qpd = QPDDecomposition(result, _hamiltonian(), verbose=False, grouping=QubitWiseCommuting())
    assert qpd.grouping.qubit_wise


def test_rejects_openfermion_qubit_operator():
    """`observable` must be a qarp.operators.QubitOperator, not openfermion's —
    the two are same-named, structurally similar (both duck-type on `.terms`/
    `.constant`), but distinct incompatible types; silently accepting either
    was an oversight (PostProcessing.__init__ had no isinstance check), not a
    documented interop feature."""
    of_hamiltonian = 0.169884520279 * openfermion.QubitOperator(
        "Z0"
    ) + -0.218863067812 * openfermion.QubitOperator("Z0 Z1 Z2")
    cmds, n = _toy_circuit_1()
    cutter = EAPartitioning(cmds, n, [2, 3])
    result = cutter.cut(manual_setting=[[0], [1, 2]])
    with pytest.raises(TypeError, match="qarp.operators.QubitOperator"):
        QPDDecomposition(result, of_hamiltonian, verbose=False)


# ── Sampling-strategy execution, ranking contracts, statevector probs ───────


def test_experiment_fraction_out_of_range_rejected():
    cmds, n = _toy_circuit_1()
    cutter = EAPartitioning(cmds, n, [2, 3])
    result = cutter.cut(manual_setting=[[0], [1, 2]])
    for bad in (0.0, 1.5, -0.2):
        with pytest.raises(ValueError, match="experiment_fraction must be in"):
            QPDDecomposition(result, _hamiltonian(), verbose=False, experiment_fraction=bad)


def test_compute_all_coefficients_gamma_is_three_per_cx_cut():
    """Mitarai & Fujii (NJP 23, 023021): the CNOT quasi-probability
    decomposition has sampling overhead gamma = 3, so the ranked coefficient
    vector must satisfy sum|c| = 3^n_cuts."""
    qpd = _make_qpd(*_toy_circuit_1()[:2], manual_setting=[[0], [1, 2]])
    qpd.decompose()
    all_coeffs = qpd._compute_all_coefficients()
    assert len(all_coeffs) == 6**qpd.n_cuts
    assert np.isclose(np.sum(np.abs(all_coeffs)), 3.0**qpd.n_cuts, atol=1e-12)


def test_top_k_compute_matches_dense_expectation():
    """RZZ cut → auto top_k. Oracle: dense numpy evaluation of
    <psi|Z0 Z1|psi> for psi = RZZ(theta)(Rx(0.2) x I)(H x H)|00>, built from
    explicit gate matrices only."""
    cmds, n, theta = _toy_circuit_rzz()

    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    rx = np.array([[np.cos(0.1), -1j * np.sin(0.1)], [-1j * np.sin(0.1), np.cos(0.1)]])  # Rx(0.2)
    zz = np.diag([1.0, -1.0, -1.0, 1.0])
    rzz = np.diag(np.exp(-1j * theta / 2 * np.diag(zz)))
    psi = rzz @ np.kron(np.eye(2), rx) @ np.kron(h, h) @ np.eye(4)[:, 0]
    exact = float(np.real(psi.conj() @ zz @ psi))

    toy_obs = QubitOperator("Z0 Z1")
    cutter = EAPartitioning(cmds, n, [1, 1])
    result = cutter.cut(manual_setting=[[0], [1]])
    qpd = QPDDecomposition(
        result,
        toy_obs,
        verbose=True,
        n_shots=20000,
        shot_seed=7,
        experiment_fraction=0.9,  # drops the lowest-|coeff| branches
    )
    assert qpd.sampling_strategy == "top_k"
    qpd.decompose()
    ev = qpd.compute()
    assert abs(ev - exact) < 0.06  # shot noise + bounded top_k drop


def test_top_k_ranking_rejects_symbolic_cut_parameter():
    n = 2
    b = qx.SimpleBlock(n, "sym-rzz")
    b.h(0)
    b.rzz(0, 1, qx.Param.symbol("t"))
    b.set_built(True)
    cutter = EAPartitioning(list(b.flatten()), n, [1, 1])
    result = cutter.cut(manual_setting=[[0], [1]])
    qpd = QPDDecomposition(result, QubitOperator("Z0 Z1"), verbose=False, n_shots=100)
    qpd.decompose()
    with pytest.raises(ValueError, match="concrete .numeric. gate parameters"):
        qpd._compute_all_coefficients()


def test_locate_qubit_outside_subcircuits_raises():
    qpd = _make_qpd(*_toy_circuit_1()[:2], manual_setting=[[0], [1, 2]])
    qpd.decompose()
    with pytest.raises(RuntimeError, match="not found in any subcircuit"):
        qpd._locate_qubit_idx_subcircuit(99)


def test_normalize_sv_probs_clips_and_handles_zero_mass():
    qpd = _make_qpd(*_toy_circuit_1()[:2], manual_setting=[[0], [1, 2]])
    out = qpd.normalize_SV_probs({"00": -0.5, "11": 0.75})
    assert out["00"] == 0.0
    assert out["11"] == 1.0
    zero = qpd.normalize_SV_probs({"00": -1.0, "11": -2.0})
    assert zero == {"00": 0, "11": 0}
