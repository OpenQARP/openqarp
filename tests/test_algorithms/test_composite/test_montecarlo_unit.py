"""MonteCarlo pyscf-free unit tests: constructor validation, walker algebra
(annihilation / spawning / death-cloning / energy correction) against
hand-computed arithmetic, and an exact end-to-end semiclassical run — a
diagonal H with identity U leaves the reference eigenstate stationary, so
every energy estimate is exactly the eigenvalue.
"""

import numpy as np
import pytest

from qarp.algorithms import MonteCarlo, WalkerState
from qarp.blocks import SimpleBlock
from qarp.operators import QubitOperator

_E0 = np.array([1.0, 0.0])
_E1 = np.array([0.0, 1.0])


def _identity_u():
    return SimpleBlock(1, name="U")


def _mc(**overrides):
    defaults = dict(
        hamiltonian=np.diag([1.0, -1.0]),  # Z in the computational basis
        approx_ground_state_energy=-1.0,
        total_time=0.2,
        time_step=0.1,
        reference_walker_label=1,
        unitary_block=_identity_u(),
        initial_walker_count=5,
        num_trajectories=1,
        verbose=False,
        seed=1,
    )
    defaults.update(overrides)
    return MonteCarlo(**defaults)


# ── constructor validation ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "overrides, exc, match",
    [
        ({"mode": "bogus"}, ValueError, "Invalid mode"),
        ({"reference_walker_label": [1]}, TypeError, "single reference label"),
        (
            {"num_target_states": 2, "reference_walker_label": [1]},
            ValueError,
            "equal to the number of target states",
        ),
        (
            {"num_target_states": 2, "reference_walker_label": 1},
            ValueError,
            "equal to the number of target states",
        ),
        ({"mode": "Quantum"}, TypeError, "only valid for Semiclassical"),
        ({"hamiltonian": np.array([1.0, -1.0])}, ValueError, "2-dimensional"),
        ({"hamiltonian": np.ones((2, 3))}, ValueError, "square"),
        ({"hamiltonian": np.array([[0.0, 1.0], [0.0, 0.0]])}, ValueError, "Hermitian"),
        ({"hamiltonian": "bad-type"}, TypeError, "must be QubitOperator or np.ndarray"),
        ({"qdrift": True}, ValueError, "not supported for np.ndarray"),
        # "QOP" resolves to QubitOperator("Z0") inside the test: qarpx-backed
        # objects must never live in a parametrize list (module scope keeps
        # them alive at shutdown -> nanobind leak report).
        (
            {"hamiltonian": "QOP", "qdrift": True, "qdrift_samples": 0},
            ValueError,
            "positive integer",
        ),
        (
            {"hamiltonian": "QOP", "qdrift": True, "qdrift_ratio": 2.0},
            ValueError,
            "between 0 and 1",
        ),
    ],
)
def test_constructor_validation(overrides, exc, match):
    if isinstance(overrides.get("hamiltonian"), str) and overrides["hamiltonian"] == "QOP":
        overrides = {**overrides, "hamiltonian": QubitOperator("Z0")}
    with pytest.raises(exc, match=match):
        _mc(**overrides)


def test_scalar_parameters_wrapped_into_lists():
    mc = _mc()
    assert mc.approx_ground_state_energy == [-1.0]
    assert mc.reference_walker_label == [1]
    assert mc.initial_walker_count == [5]
    assert mc.shift_damping == [0.1]


def test_list_parameters_kept():
    mc = _mc(
        num_target_states=2,
        reference_walker_label=[0, 1],
        approx_ground_state_energy=[1.0, -1.0],
        initial_walker_count=[5, 7],
        shift_damping=[0.1, 0.2],
    )
    assert mc.approx_ground_state_energy == [1.0, -1.0]
    assert mc.initial_walker_count == [5, 7]
    assert mc.shift_damping == [0.1, 0.2]


def test_qdrift_preparation_branches(capsys):
    # Quantum mode keeps the Hamiltonian a QubitOperator so the type survives
    # qDRIFT for inspection.
    ham = QubitOperator("Z0") + QubitOperator("X0", 0.5)
    mc = _mc(hamiltonian=ham, mode="Quantum", qdrift=True, qdrift_samples=50, verbose=True)
    assert isinstance(mc.hamiltonian, QubitOperator)
    assert "Applying qDRIFT to Hamiltonian..." in capsys.readouterr().out

    mc = _mc(
        hamiltonian=ham,
        mode="Quantum",
        qdrift=True,
        qdrift_samples=50,
        qdrift_ratio=0.5,
        verbose=True,
    )
    assert isinstance(mc.hamiltonian, QubitOperator)
    assert "partially-randomized qDRIFT" in capsys.readouterr().out


# ── walker-basis validation and build ───────────────────────────────────


def test_validate_walker_basis_raises():
    mc = _mc()
    with pytest.raises(ValueError, match="cannot be empty"):
        mc._validate_walker_basis([], "Semiclassical")
    block_walker = WalkerState(state_data=_identity_u().build(), sign=1.0, label="0")
    array_walker = WalkerState(state_data=_E0, sign=1.0, label="0")
    with pytest.raises(TypeError, match="to be arrays"):
        mc._validate_walker_basis([block_walker], "Semiclassical")
    with pytest.raises(TypeError, match="circuit blocks"):
        mc._validate_walker_basis([array_walker], "Quantum")


def _basis_walkers():
    return [
        WalkerState(state_data=_E0, sign=1.0, label="0"),
        WalkerState(state_data=_E1, sign=1.0, label="1"),
    ]


def test_build_with_provided_walker_basis():
    mc = _mc(walker_basis=_basis_walkers()).build()
    assert mc.labels_to_indices == {0: 0, 1: 1}
    assert mc.reference_walker_index == [1]


def test_build_missing_reference_label_raises():
    with pytest.raises(ValueError, match="not found in the walker basis"):
        _mc(walker_basis=_basis_walkers(), reference_walker_label=5).build()


def test_build_rejects_mixed_walker_types():
    mixed = [
        WalkerState(state_data=_E0, sign=1.0, label="0"),
        WalkerState(state_data=_identity_u().build(), sign=1.0, label="1"),
    ]
    with pytest.raises(TypeError, match="np.ndarray instances"):
        _mc(walker_basis=mixed).build()


def test_build_rejects_mixed_walker_types_quantum():
    # First walker passes the basis pre-check; the per-walker build check
    # must still catch the stray ndarray.
    mixed = [
        WalkerState(state_data=_identity_u().build(), sign=1.0, label="0"),
        WalkerState(state_data=_E1, sign=1.0, label="1"),
    ]
    with pytest.raises(TypeError, match="Block instances"):
        _mc(
            hamiltonian=QubitOperator("Z0"),
            mode="Quantum",
            walker_basis=mixed,
            reference_walker_label=0,
        ).build()


# ── walker algebra, hand-computed ───────────────────────────────────────


def _w(label, sign):
    return WalkerState(state_data=_E0, sign=sign, label=label)


def test_annihilation_positive_majority():
    walkers = [_w("a", 1.0)] * 3 + [_w("a", -1.0)]
    survivors = _mc().remove_opposite_sign_pairs(walkers)
    assert len(survivors) == 2
    assert all(w.sign > 0 for w in survivors)


def test_annihilation_negative_majority():
    walkers = [_w("b", 1.0)] + [_w("b", -1.0)] * 2
    survivors = _mc().remove_opposite_sign_pairs(walkers)
    assert len(survivors) == 1
    assert survivors[0].sign < 0


def test_annihilation_balanced_and_label_isolation():
    walkers = [_w("a", 1.0), _w("a", -1.0), _w("b", 1.0)]
    survivors = _mc().remove_opposite_sign_pairs(walkers)
    assert len(survivors) == 1
    assert survivors[0].label == "b"


def test_spawning_semiclassical_with_certain_probability():
    mc = _mc().build()
    mc._spawning_cache = {0: (np.array([1]), np.array([1.0]))}
    mc._sign_cache = np.array([[0.0, -1.0], [0.0, 0.0]], dtype=complex)
    visited = {0}
    spawned = mc._apply_spawning_semiclassical([_w("0", 1.0)], 0, visited)
    assert len(spawned) == 1
    assert spawned[0].label == "1"
    assert spawned[0].sign == -1.0  # walker sign x sign factor
    assert visited == {0, 1}


def test_spawning_semiclassical_edge_cases():
    mc = _mc().build()
    mc._spawning_cache = {}
    assert mc._apply_spawning_semiclassical([_w("0", 1.0)], 0, set()) == []
    mc._spawning_cache = {0: (np.array([]), np.array([]))}
    assert mc._apply_spawning_semiclassical([_w("0", 1.0)], 0, set()) == []
    # A cached target outside the walker basis is skipped, not KeyError'd.
    mc._spawning_cache = {0: (np.array([7]), np.array([1.0]))}
    assert mc._apply_spawning_semiclassical([_w("0", 1.0)], 0, set()) == []


def test_death_and_cloning_semiclassical():
    mc = _mc(time_step=1.0).build()
    mc.hamiltonian_estimate = np.diag([1.0, -1.0])
    assert mc._apply_death_and_cloning_semiclassical([], 0, 0.0) == []
    # H_11 - shift = -1 with tau=1: cloning probability 1 doubles the walkers.
    walkers = [_w("1", 1.0)] * 3
    assert len(mc._apply_death_and_cloning_semiclassical(walkers, 1, 0.0)) == 6
    # H_00 - shift = +1 with tau=1: death probability 1 kills them all.
    assert mc._apply_death_and_cloning_semiclassical(walkers, 0, 0.0) == []


def test_estimate_ground_state_energy_hand_computed():
    h = np.array([[1.0, 0.3], [0.3, -1.0]])
    mc = _mc(hamiltonian=h, reference_walker_label=0, walker_basis=_basis_walkers()).build()
    walkers_by_label = {
        "0": [_w("0", 1.0)] * 4,
        "1": [WalkerState(state_data=_E1, sign=-1.0, label="1")] * 2,
    }
    # correction = <1|H|0> * (sign * n1 / n0) = 0.3 * (-2/4) = -0.15
    energy = mc.estimate_ground_state_energy(walkers_by_label, 1.0, {0, 1})
    assert energy == pytest.approx(1.0 - 0.15)


def test_estimate_ground_state_energy_empty_reference_returns_initial():
    mc = _mc(walker_basis=_basis_walkers()).build()
    assert mc.estimate_ground_state_energy({}, -0.5, {0, 1}) == -0.5
    assert mc.estimate_ground_state_energy_quantum([], -0.5, {0, 1}) == -0.5


# ── quantum-mode death/cloning via real 1-qubit circuits ────────────────


def test_death_and_cloning_quantum_cache_miss_computes_h_ii():
    """U = H (Hadamard) rotates Z to X, so h_00 = <0|X|0> = 0 up to shot
    noise; with shift -2 and tau=1 the death probability saturates and every
    walker dies. The diagonal element lands in the cache."""
    u = SimpleBlock(1, name="U")
    u.h(0)
    mc = _mc(
        hamiltonian=QubitOperator("Z0"),
        mode="Quantum",
        unitary_block=u,
        reference_walker_label=0,
        time_step=1.0,
        total_time=1.0,
    ).build()

    assert mc._apply_death_and_cloning_quantum([], 0, 0.0) == []

    walkers = [_w("0", 1.0)] * 3
    survivors = mc._apply_death_and_cloning_quantum(walkers, 0, -2.0)
    assert survivors == []
    h_00, _ = mc._hamiltonian_cache[(0, 0)]
    assert abs(h_00) < 0.2  # 0 up to shot noise


# ── exact end-to-end semiclassical run ──────────────────────────────────


def test_semiclassical_run_on_diagonal_h_is_exact(capsys):
    """Diagonal H with identity U: the reference eigenstate is stationary
    (no off-diagonal spawning, death probability 0 at shift = E), so every
    recorded energy is exactly the eigenvalue -1."""
    mc = _mc(
        total_time=0.3,
        time_step=0.1,
        verbose=True,
        num_trajectories=2,
        population_threshold=0,  # forces the shift-update branch (log 1 = 0)
        save_walker_history=True,
    ).build()
    final = mc.run()

    assert final == [-1.0]
    for trajectory in mc.energy_estimates_trajectories:
        assert all(e == -1.0 for e in trajectory[0])
    out = capsys.readouterr().out
    assert "=== Trajectory 1/2 ===" in out
    assert "Step 0/" in out and "Energy: N/A" in out
    assert len(mc.walker_history_trajectories) == 2


def test_shot_branch_samples_through_the_engine():
    """The shot branch of ``_sample_probabilities`` goes through
    ``self.engine`` (seeded, noise-aware), not a fresh simulator: the engine's
    ``run`` is called and the marginal matches the analytic Bell-pair
    distribution up to shot noise."""
    u = SimpleBlock(1, name="U")
    u.h(0)
    mc = _mc(hamiltonian=QubitOperator("Z0"), mode="Quantum", unitary_block=u, n_shots=4000).build()
    runs = {"n": 0}
    real_run = mc.engine.run

    def counting_run(*a, **k):
        runs["n"] += 1
        return real_run(*a, **k)

    mc.engine.run = counting_run
    bell = SimpleBlock(2)
    bell.h(0)
    bell.cx(0, 1)
    bell.build()
    probs = mc._sample_probabilities(bell, 2)
    assert runs["n"] == 1
    assert set(probs) == {0, 3}
    assert abs(probs[0] - 0.5) < 0.05 and abs(probs[3] - 0.5) < 0.05

    # Endianness pin (§1, LSB): the Bell keys are bit-reversal invariant, so
    # an asymmetric state must fix the convention — X on qubit 0 is integer
    # key 1 (bit 0), X on qubit 1 is key 2.
    for q, key in ((0, 1), (1, 2)):
        flip = SimpleBlock(2)
        flip.x(q)
        flip.build()
        assert mc._sample_probabilities(flip, 2) == {key: 1.0}


def test_shot_branch_names_the_ancilla_when_the_engine_is_too_narrow():
    """The overlap circuits are one qubit wider than the system.  An engine
    sized to the system refuses them at ``check_fits``; the refusal must say
    why and what width fixes it, and the widened engine must succeed."""
    from qarp.engines import QarpEngine
    from qarp.errors import CapabilityError

    u = SimpleBlock(1, name="U")
    u.h(0)
    kwargs = dict(hamiltonian=QubitOperator("Z0"), mode="Quantum", unitary_block=u, n_shots=200)

    narrow = _mc(engine=QarpEngine(n_qubits=1), **kwargs).build()
    with pytest.raises(CapabilityError, match=r"exposes only 1.*n_qubits \+ 1 = 2"):
        narrow._estimate_Hij_circuit([0])

    wide = _mc(engine=QarpEngine(n_qubits=2), **kwargs).build()
    assert sum(wide._sample_probabilities(SimpleBlock(2).build(), 2).values()) == pytest.approx(1.0)
