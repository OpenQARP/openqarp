from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import truncnorm

import qarp
from qarp.algorithms import MMQCELS
from qarp.blocks import ComputationalBasisStateBlock, SimpleBlock, TrotterBlock
from qarp.engines import QarpEngine
from qarp.operators import JordanWigner, QubitOperator
from tests.molecular_assets import fermion_operator, reference_onv


@pytest.fixture(scope="module")
def h2_problem():
    mapping = JordanWigner()
    hamiltonian = mapping.encode_operator(fermion_operator("h2_0.735_sto3g"))
    state = ComputationalBasisStateBlock(
        mapping.encode_state(reference_onv("h2_0.735_sto3g"))
    ).build()
    # Independent oracle: numpy.linalg.eigh of the snapshot Hamiltonian and
    # |<eigvec|HF>|^2 overlaps, kept only where the weight exceeds 1e-8.
    reference = np.array([-1.1373060357533997, 0.4950577416181092])
    weights = np.array([0.98755973, 0.01244027])
    return hamiltonian, state, reference, weights


def _common(state):
    return dict(
        state=state,
        T0=1.25,
        N0=120,
        Nj=80,
        n_dominant_eigenvalues=2,
        error_rate=1e-2,
        initial_eigenvalues=[-1.0, 0.5],
        n_initial_guesses=10,
        seed=1,
        verbose=False,
    )


def test_mmqcels_classical_h2(h2_problem):
    hamiltonian, state, reference, weights = h2_problem
    algorithm = MMQCELS(
        operator=hamiltonian,
        execution_mode="classical",
        n_levels=5,
        **_common(state),
    )
    result = algorithm.run()
    assert result == pytest.approx(reference, abs=1e-5)
    assert algorithm.amplitudes == pytest.approx(weights, abs=1e-5)


def test_mmqcels_statevector_h2_reports_trotter_error(h2_problem):
    hamiltonian, state, reference, _ = h2_problem
    trotter = TrotterBlock(state.n_qubits, hamiltonian, steps=24, order=2).build()
    algorithm = MMQCELS(
        operator=trotter,
        execution_mode="statevector",
        n_levels=4,
        engine=QarpEngine(seed=1),
        **_common(state),
    )
    result = algorithm.run()
    # Regression pin, not an oracle: the bound covers fitting plus the
    # deliberately approximate Trotter evolution and matches the MWE's check.
    assert np.max(np.abs(result - reference)) < 6e-3


def test_mmqcels_exact_hadamard_matches_statevector_h2(h2_problem):
    hamiltonian, state, _, _ = h2_problem
    trotter = TrotterBlock(state.n_qubits, hamiltonian, steps=24, order=2).build()
    common = _common(state)
    statevector = MMQCELS(
        operator=trotter,
        execution_mode="statevector",
        n_levels=4,
        engine=QarpEngine(seed=1),
        **common,
    )
    statevector_eigenvalues = statevector.run()
    hadamard = MMQCELS(
        operator=trotter,
        execution_mode="hadamard",
        n_shots=qarp.EXACT,
        n_levels=4,
        engine=QarpEngine(seed=1),
        **common,
    )
    hadamard_eigenvalues = hadamard.run()
    assert hadamard_eigenvalues == pytest.approx(statevector_eigenvalues, abs=1e-8)
    assert hadamard.amplitudes == pytest.approx(statevector.amplitudes, abs=1e-8)


def _classical_alg(**kwargs):
    defaults = dict(
        operator=np.diag([-0.7, 1.1]),
        state=np.array([np.sqrt(0.65), np.sqrt(0.35)]),
        execution_mode="classical",
        T0=1.0,
        N0=24,
        Nj=20,
        n_levels=3,
        n_dominant_eigenvalues=2,
        initial_eigenvalues=[-0.5, 0.9],
        n_initial_guesses=3,
        error_rate=1e-2,
        seed=7,
        verbose=False,
    )
    defaults.update(kwargs)
    return MMQCELS(**defaults)


def _z_problem():
    state = SimpleBlock(1)
    state.x(0)
    return (
        TrotterBlock(1, QubitOperator("Z0"), steps=1).build(),
        state.build(),
    )


@pytest.mark.parametrize("missing", ["N0", "Nj"])
def test_standard_mode_requires_sample_counts(missing):
    kwargs = dict(N0=10, Nj=10)
    kwargs[missing] = None
    with pytest.raises(ValueError, match=missing):
        _classical_alg(**kwargs)


def test_standard_mode_requires_exactly_one_level_selector():
    with pytest.raises(ValueError, match="exactly one"):
        _classical_alg(n_levels=None, q=None)
    with pytest.raises(ValueError, match="exactly one"):
        _classical_alg(n_levels=3, q=0.2)


def test_standard_q_schedule_matches_pinned_theorem_one_values():
    alg = _classical_alg(T0=0.5, error_rate=0.01, n_levels=None, q=0.2)
    assert alg.n_levels == 7
    assert tuple(alg.time_scale(j) for j in range(alg.n_levels)) == (
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
    )


def test_error_rate_mode_is_opt_in_and_uses_documented_formulas():
    alg = _classical_alg(
        parameter_mode="error_rate",
        error_rate=1e-2,
        N0=None,
        Nj=None,
        n_levels=None,
    )
    assert alg.optimal_number_of_iterations() == 8
    assert alg.optimal_dataset_length() == 10
    assert alg.n_levels == 8
    assert alg.Nj == 10
    assert alg.N0 == 5
    with pytest.raises(ValueError, match="q is not valid"):
        _classical_alg(parameter_mode="error_rate", q=0.2)


@pytest.mark.parametrize(
    ("error_rate", "expected"),
    [(2**-10, 11), (1e-2, 8), (0.5, 2)],
)
def test_optimal_number_of_iterations_regression_points(error_rate, expected):
    alg = _classical_alg(error_rate=error_rate)
    assert alg.optimal_number_of_iterations() == expected


def test_explicit_error_rate_mode_overrides_win():
    alg = _classical_alg(parameter_mode="error_rate", N0=7, Nj=[8, 9], n_levels=3)
    assert alg._sample_counts == (7, 8, 9)


def test_nj_sequence_must_match_subsequent_levels():
    with pytest.raises(ValueError, match="Nj sequence length"):
        _classical_alg(Nj=[10], n_levels=3)
    with pytest.raises(ValueError, match="must not be empty"):
        _classical_alg(Nj=[], n_levels=1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"N0": 2}, "N0"),
        ({"Nj": 2}, "Nj at level 1"),
        (
            {
                "parameter_mode": "error_rate",
                "error_rate": 0.5,
                "N0": None,
                "Nj": None,
                "n_levels": None,
            },
            "N0",
        ),
    ],
)
def test_every_evaluated_level_requires_more_samples_than_modes(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _classical_alg(**kwargs)


@pytest.mark.parametrize(
    ("name", "value", "error"),
    [
        ("error_rate", 0.0, ValueError),
        ("T0", "invalid", TypeError),
        ("T0", np.complex128(1 + 2j), TypeError),
        ("T0", np.inf, ValueError),
        ("gamma", -1.0, ValueError),
        ("N0", True, TypeError),
        ("Nj", 0, ValueError),
        ("Nj", "10", TypeError),
        ("Nj", 1.5, TypeError),
        ("n_levels", 1.5, TypeError),
        ("n_initial_guesses", 0, ValueError),
    ],
)
def test_invalid_common_parameters_raise_with_name(name, value, error):
    with pytest.raises(error, match=name):
        _classical_alg(**{name: value})


def test_bounds_and_initial_eigenvalue_validation():
    with pytest.raises(TypeError, match="lam_min"):
        _classical_alg(lam_min=True)
    with pytest.raises(ValueError, match="lam_max"):
        _classical_alg(lam_max=np.inf)
    with pytest.raises(ValueError, match="lam_min"):
        _classical_alg(lam_min=2.0, lam_max=1.0)
    with pytest.raises(TypeError, match="lam_min"):
        _classical_alg(lam_min=np.complex128(-2.0 + 1.0j))
    with pytest.raises(ValueError, match="exactly 2"):
        _classical_alg(initial_eigenvalues=[0.0])
    with pytest.raises(ValueError, match="within"):
        _classical_alg(initial_eigenvalues=[-4.0, 0.0])
    with pytest.raises(TypeError, match="initial_eigenvalues"):
        _classical_alg(initial_eigenvalues=[np.complex128(-0.5 + 1.0j), 0.9])
    with pytest.raises(ValueError, match="initial_eigenvalues"):
        _classical_alg(initial_eigenvalues=[np.nan, 0.9])


def test_execution_and_shot_contracts():
    with pytest.raises(ValueError, match="execution_mode"):
        _classical_alg(execution_mode="unknown")
    with pytest.raises(ValueError, match="parameter_mode"):
        _classical_alg(parameter_mode="paper")
    with pytest.raises(ValueError, match="parameter_mode"):
        _classical_alg(parameter_mode="epsilon_heuristics")
    with pytest.raises(ValueError, match="n_shots"):
        _classical_alg(n_shots=10)
    with pytest.raises(TypeError, match="operator"):
        _classical_alg(operator="invalid")
    with pytest.raises(TypeError, match="state"):
        _classical_alg(state="invalid")
    with pytest.raises(TypeError, match="optimizer"):
        _classical_alg(optimizer=object())
    # A bounds-less qarp optimizer is refused at construction, not after the
    # dataset is generated (it would raise ValueError inside the level fit).
    from qarp.optimizers import GradientDescentOptimizer, RotosolveOptimizer

    for opt in (RotosolveOptimizer(), GradientDescentOptimizer()):
        with pytest.raises(TypeError, match="honour bounds"):
            _classical_alg(optimizer=opt)

    operator, state = _z_problem()
    with pytest.raises(ValueError, match="n_shots"):
        MMQCELS(
            operator,
            state,
            execution_mode="statevector",
            T0=1.0,
            N0=4,
            Nj=4,
            n_levels=2,
            n_shots=10,
        )
    with pytest.raises(ValueError, match="n_shots"):
        MMQCELS(
            operator,
            state,
            execution_mode="hadamard",
            T0=1.0,
            N0=4,
            Nj=4,
            n_levels=2,
            n_shots=0,
        )


def test_circuit_modes_require_symbolic_trotter_and_block_state():
    state = SimpleBlock(1).build()
    with pytest.raises(TypeError, match="TrotterBlock"):
        MMQCELS(
            QubitOperator("Z0"),
            state,
            execution_mode="statevector",
            T0=1.0,
            N0=4,
            Nj=4,
            n_levels=2,
        )
    frozen = TrotterBlock(1, QubitOperator("Z0"), steps=1, time=1.0).build()
    with pytest.raises(ValueError, match="symbolic time"):
        MMQCELS(
            frozen,
            state,
            execution_mode="hadamard",
            T0=1.0,
            N0=4,
            Nj=4,
            n_levels=2,
        )


def test_truncated_gaussian_matches_scipy_reference_and_support():
    alg = _classical_alg(seed=19, gamma=1.5)
    actual = alg.generate_times(2.0, 128)
    expected = truncnorm.rvs(
        -1.5,
        1.5,
        loc=0.0,
        scale=2.0,
        size=128,
        random_state=np.random.default_rng(19),
    )
    assert actual == pytest.approx(expected)
    assert np.all(np.abs(actual) <= 3.0)
    assert len(np.unique(actual)) == len(actual)


def test_truncated_gaussian_has_the_analytic_variance():
    gamma = 1.5
    scale = 2.0
    alg = _classical_alg(seed=23, gamma=gamma)
    actual = alg.generate_times(scale, 50_000)
    expected_variance = scale**2 * truncnorm.var(-gamma, gamma)
    assert np.mean(actual) == pytest.approx(0.0, abs=0.02)
    assert np.var(actual) == pytest.approx(expected_variance, rel=0.02)


def test_time_scale_validates_level():
    alg = _classical_alg()
    with pytest.raises(TypeError, match="level"):
        alg.time_scale(True)
    with pytest.raises(ValueError, match="level"):
        alg.time_scale(-1)


@pytest.mark.parametrize(
    ("operator", "state", "message"),
    [
        (np.ones((2, 3)), np.array([1.0, 0.0]), "square"),
        (np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([1.0, 0.0]), "Hermitian"),
        (np.diag([0.0, 1.0]), np.array([1.0, 0.0, 0.0]), "length"),
        (np.diag([0.0, 1.0]), np.array([1.0, 1.0]), "normalized"),
    ],
)
def test_classical_build_validates_operator_and_state(operator, state, message):
    with pytest.raises(ValueError, match=message):
        _classical_alg(operator=operator, state=state).build()


def test_classical_qubit_operator_infers_width_from_ndarray_state():
    alg = _classical_alg(
        operator=QubitOperator("Z0"),
        state=np.array([1.0, 0.0, 0.0, 0.0]),
        n_dominant_eigenvalues=1,
        initial_eigenvalues=[0.8],
    ).build()
    dataset = alg.generate_dataset(1.0, 8)
    for time, value in dataset:
        assert value == pytest.approx(np.exp(-1j * time), abs=1e-13)


def test_classical_qubit_operator_rejects_non_power_of_two_state():
    with pytest.raises(ValueError, match="power of two"):
        _classical_alg(
            operator=QubitOperator("Z0"),
            state=np.array([1.0, 0.0, 0.0]),
        ).build()


def test_objective_validates_shape_and_dataset():
    alg = _classical_alg()
    with pytest.raises(ValueError, match="exactly 2"):
        alg.objective([0.0], [(0.0, 1.0)])
    with pytest.raises(ValueError, match="must not be empty"):
        alg.objective([0.0, 1.0], [])
    with pytest.raises(ValueError, match="finite"):
        alg.objective([0.0, 1.0], [(np.nan, 1.0)])


def test_complex_loss_distinguishes_signed_frequency():
    alg = _classical_alg(
        operator=np.diag([0.7]),
        state=np.array([1.0]),
        n_dominant_eigenvalues=1,
        initial_eigenvalues=[-0.5],
    )
    times = np.linspace(-2.0, 2.0, 31)
    dataset = list(zip(times, np.exp(-1j * 0.7 * times), strict=True))
    assert alg.objective([0.7], dataset) < 1e-25
    assert alg.objective([-0.7], dataset) > 0.1


def test_signed_frequency_is_recovered_from_opposite_initial_guess():
    alg = _classical_alg(
        operator=np.diag([0.7]),
        state=np.array([1.0]),
        n_dominant_eigenvalues=1,
        initial_eigenvalues=[-0.5],
        n_initial_guesses=1,
        N0=40,
        Nj=30,
    )
    assert alg.run() == pytest.approx([0.7], abs=1e-6)


def test_two_mode_variable_projection_is_independent_complex_oracle():
    alg = _classical_alg()
    times = np.linspace(-3.0, 3.0, 61)
    amplitudes = np.array([0.6 + 0.1j, 0.3 - 0.2j])
    eigenvalues = np.array([-0.7, 1.1])
    values = np.exp(-1j * np.outer(times, eigenvalues)) @ amplitudes
    dataset = list(zip(times, values, strict=True))
    loss, fitted = alg._variable_projection(eigenvalues, times, values)
    assert loss < 1e-25
    assert fitted == pytest.approx(amplitudes)
    assert alg.objective(eigenvalues, dataset) < 1e-25


def test_optimizer_recovers_two_mode_complex_signal():
    alg = _classical_alg()
    times = np.linspace(-3.0, 3.0, 61)
    amplitudes = np.array([0.6 + 0.1j, 0.3 - 0.2j])
    eigenvalues = np.array([-0.7, 1.1])
    values = np.exp(-1j * np.outer(times, eigenvalues)) @ amplitudes
    dataset = list(zip(times, values, strict=True))
    fitted_eigenvalues, fitted_amplitudes, loss, failed_starts = alg._fit_level(
        dataset,
        [np.array([-0.5, 0.9])],
        [(-np.pi, np.pi), (-np.pi, np.pi)],
    )
    assert fitted_eigenvalues == pytest.approx(eigenvalues, abs=1e-6)
    assert fitted_amplitudes == pytest.approx(amplitudes, abs=1e-6)
    assert loss < 1e-12
    assert failed_starts == 0


def test_objective_is_mean_squared_residual_not_sum():
    alg = _classical_alg(
        operator=np.diag([0.0]),
        state=np.array([1.0]),
        n_dominant_eigenvalues=1,
        initial_eigenvalues=[0.0],
    )
    dataset = [(0.0, 0.0), (1.0, 0.0), (2.0, 3.0)]
    assert alg.objective([0.0], dataset) == pytest.approx(2.0)


def test_classical_dataset_matches_analytic_weighted_signal():
    alg = _classical_alg(seed=3).build()
    dataset = alg.generate_dataset(1.2, 12)
    for time, value in dataset:
        expected = 0.65 * np.exp(0.7j * time) + 0.35 * np.exp(-1.1j * time)
        assert value == pytest.approx(expected, abs=1e-13)


def test_statevector_and_exact_hadamard_match_analytic_signal():
    operator, state = _z_problem()
    common = dict(
        operator=operator,
        state=state,
        T0=1.0,
        N0=6,
        Nj=6,
        n_levels=2,
        seed=5,
        verbose=False,
    )
    statevector = MMQCELS(execution_mode="statevector", **common).build()
    hadamard = MMQCELS(execution_mode="hadamard", n_shots=qarp.EXACT, **common).build()
    data_sv = statevector.generate_dataset(1.0, 8)
    data_ht = hadamard.generate_dataset(1.0, 8)
    assert np.asarray(data_ht) == pytest.approx(np.asarray(data_sv), abs=1e-12)
    for time, value in data_sv:
        assert value == pytest.approx(np.exp(1j * time), abs=1e-12)


def test_finite_shot_hadamard_is_statistically_consistent():
    operator, state = _z_problem()
    algorithm = MMQCELS(
        operator,
        state,
        execution_mode="hadamard",
        T0=1.0,
        N0=8,
        Nj=8,
        n_levels=2,
        n_shots=4000,
        seed=13,
        verbose=False,
        engine=QarpEngine(seed=13),
    ).build()
    for time, value in algorithm.generate_dataset(1.0, 12):
        # Each quadrature is an independent binomial mean. This loose bound
        # is about 3.8 standard deviations at maximal variance.
        assert value == pytest.approx(np.exp(1j * time), abs=0.06)


def test_default_one_shot_hadamard_run_is_statistically_consistent():
    operator, state = _z_problem()
    algorithm = MMQCELS(
        operator,
        state,
        execution_mode="hadamard",
        T0=1.0,
        N0=400,
        Nj=400,
        n_levels=3,
        initial_eigenvalues=[-0.8],
        n_initial_guesses=1,
        seed=17,
        verbose=False,
        engine=QarpEngine(seed=17),
    )
    assert algorithm.n_shots is None
    assert algorithm.run() == pytest.approx([-1.0], abs=0.12)


def test_circuit_run_batches_once_per_level(monkeypatch):
    operator, state = _z_problem()
    engine = QarpEngine(seed=2)
    alg = MMQCELS(
        operator,
        state,
        execution_mode="statevector",
        T0=1.0,
        N0=8,
        Nj=8,
        n_levels=3,
        initial_eigenvalues=[-0.8],
        n_initial_guesses=1,
        seed=2,
        verbose=False,
        engine=engine,
    ).build()
    calls = 0
    rebuild_values = []
    original = engine.batch_run

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        rebuild_values.append(kwargs.get("rebuild"))
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "batch_run", counted)
    result = alg.run()
    assert calls == alg.n_levels
    assert rebuild_values == [False] * alg.n_levels
    assert result == pytest.approx([-1.0], abs=1e-6)


def test_run_is_reproducible_and_populates_qpe_style_result_attributes():
    alg = _classical_alg(N0=60, Nj=50, n_levels=4, n_initial_guesses=6)
    assert alg.result is None
    assert alg.eigenvalues is None
    assert alg.amplitudes is None
    first = alg.run()
    first_losses = alg.level_losses
    second = alg.run()
    assert first == pytest.approx([-0.7, 1.1], abs=2e-5)
    assert second == pytest.approx(first)
    assert alg.result is alg.eigenvalues
    assert alg.result is second
    assert alg.amplitudes == pytest.approx([0.65, 0.35], abs=2e-5)
    assert alg.level_losses == pytest.approx(first_losses)
    assert alg.sample_counts == (60, 50, 50, 50)
    assert len(alg.sampled_max_times) == alg.n_levels
    assert len(alg.sampled_total_times) == alg.n_levels
    assert alg.optimizer_failed_starts == (0,) * alg.n_levels


class _FailingOptimizer:
    def minimize(self, objective, initial_parameters, **kwargs):
        del objective, kwargs
        return SimpleNamespace(
            x=np.asarray(initial_parameters), fun=np.inf, success=False, message="forced"
        )


def test_optimizer_failure_is_not_silently_accepted():
    alg = _classical_alg(optimizer=_FailingOptimizer())
    with pytest.raises(RuntimeError, match="forced"):
        alg.run()


class _OneFailureOptimizer:
    def __init__(self):
        self.calls = 0

    def minimize(self, objective, initial_parameters, **kwargs):
        del kwargs
        self.calls += 1
        if self.calls == 1:
            return SimpleNamespace(
                x=np.asarray(initial_parameters), fun=np.inf, success=False, message="first failed"
            )
        parameters = np.asarray(initial_parameters)
        return SimpleNamespace(
            x=parameters, fun=objective(parameters), success=True, message="accepted"
        )


def test_discarded_optimizer_starts_are_reported():
    optimizer = _OneFailureOptimizer()
    alg = _classical_alg(optimizer=optimizer, n_initial_guesses=2)
    alg.run()
    assert alg.optimizer_failed_starts == (1, 0, 0)


class _RecordingOptimizer:
    def __init__(self):
        self.bounds = []

    def minimize(self, objective, initial_parameters, **kwargs):
        self.bounds.append(tuple(kwargs["bounds"]))
        parameters = np.asarray(initial_parameters)
        return SimpleNamespace(
            x=parameters, fun=objective(parameters), success=True, message="recorded"
        )


def test_later_level_bounds_follow_algorithm_two_without_global_clipping():
    optimizer = _RecordingOptimizer()
    alg = _classical_alg(
        operator=np.diag([0.7]),
        state=np.array([1.0]),
        n_dominant_eigenvalues=1,
        initial_eigenvalues=[3.0],
        n_initial_guesses=1,
        N0=2,
        Nj=2,
        n_levels=2,
        optimizer=optimizer,
    )
    alg.run()
    assert optimizer.bounds[0] == ((-np.pi, np.pi),)
    assert optimizer.bounds[1][0] == pytest.approx((3.0 - np.pi / 2, 3.0 + np.pi / 2))
