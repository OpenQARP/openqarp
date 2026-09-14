"""Tests for the QMEGS (Quantum Multiple Eigenvalue Gaussian filtered Search) algorithm."""

import numpy as np
import pytest

import qarp
from qarp.algorithms import QMEGS, get_overlaps
from qarp.algorithms._composite.qmegs import truncated_gaussian_density
from qarp.blocks import SimpleBlock, SynthesizedTimeEvolutionBlock
from qarp.operators import QubitOperator
from tests.operator_test_utils import eigenspectrum

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_2qubit_hamiltonian():
    """A simple 2-qubit Hamiltonian with known eigenvalues."""
    hamiltonian = (QubitOperator("Z0 X1") + QubitOperator("Y0 Y1") + QubitOperator("X0 X1")) * 0.25
    return hamiltonian


@pytest.fixture
def simple_trial_state():
    """A simple trial state |0⟩ ⊗ |+⟩ — H on qubit 1."""
    trial_state = SimpleBlock(2, name="trial")
    trial_state.h(1)
    trial_state.build()
    return trial_state


@pytest.fixture
def synthesized_unitary(simple_2qubit_hamiltonian):
    """SynthesizedTimeEvolutionBlock for exact time evolution."""
    return SynthesizedTimeEvolutionBlock(
        n_qubits=2,
        operator=simple_2qubit_hamiltonian,
    )


@pytest.fixture
def target_indices_fixture(simple_trial_state, simple_2qubit_hamiltonian):
    """Compute target indices based on overlaps."""
    overlaps, eigenvalues = get_overlaps(
        simple_trial_state, simple_2qubit_hamiltonian, return_eigenvalues=True
    )
    sorted_indices = np.argsort(overlaps)[::-1]
    target_indices = sorted_indices[:2].tolist()
    return target_indices, overlaps, eigenvalues


# =============================================================================
# Tests for get_overlaps helper function
# =============================================================================


class TestGetOverlaps:
    """Tests for the get_overlaps helper function."""

    def test_get_overlaps_returns_array(self, simple_trial_state, simple_2qubit_hamiltonian):
        overlaps = get_overlaps(simple_trial_state, simple_2qubit_hamiltonian)
        assert isinstance(overlaps, np.ndarray)
        assert len(overlaps) == 4  # 2^2 eigenvalues for 2 qubits

    def test_get_overlaps_sum_to_one(self, simple_trial_state, simple_2qubit_hamiltonian):
        overlaps = get_overlaps(simple_trial_state, simple_2qubit_hamiltonian)
        assert np.isclose(np.sum(overlaps), 1.0)

    def test_get_overlaps_with_eigenvalues(self, simple_trial_state, simple_2qubit_hamiltonian):
        overlaps, eigenvalues = get_overlaps(
            simple_trial_state, simple_2qubit_hamiltonian, return_eigenvalues=True
        )
        assert isinstance(eigenvalues, np.ndarray)
        assert len(eigenvalues) == 4
        expected_spectrum = eigenspectrum(simple_2qubit_hamiltonian)
        assert np.allclose(sorted(eigenvalues), sorted(expected_spectrum))


# =============================================================================
# Tests for truncated_gaussian_density function
# =============================================================================


class TestTruncatedGaussianDensity:
    def test_truncated_gaussian_returns_values(self):
        t = np.linspace(-10, 10, 100)
        result = truncated_gaussian_density(t, sigma=1.0, T=100)
        assert isinstance(result, np.ndarray)
        assert len(result) == 100

    def test_truncated_gaussian_positive(self):
        t = np.linspace(-100, 100, 100)
        result = truncated_gaussian_density(t, sigma=1.0, T=100)
        assert np.all(result >= 0)

    def test_truncated_gaussian_infinite_sigma(self):
        t = np.array([0.0, 1.0, 2.0])
        result = truncated_gaussian_density(t, sigma=np.inf, T=100)
        expected = np.exp(-100 * t**2 / 2)
        assert np.allclose(result, expected)


# =============================================================================
# Tests for QMEGS constructor and validation
# =============================================================================


class TestQMEGSConstructor:
    def test_qmegs_basic_construction(
        self, synthesized_unitary, simple_trial_state, target_indices_fixture
    ):
        target_indices, _, _ = target_indices_fixture
        qmegs = QMEGS(
            unitary=synthesized_unitary,
            state=simple_trial_state,
            n_shots=qarp.EXACT,
            target_indices=target_indices,
        )
        assert qmegs.n_qubits == 2
        assert qmegs.target_indices == target_indices
        assert qmegs.mode_dataset == "analytical"
        assert qmegs.mode_time == "rvs"

    def test_qmegs_custom_parameters(
        self, synthesized_unitary, simple_trial_state, target_indices_fixture
    ):
        target_indices, _, _ = target_indices_fixture
        qmegs = QMEGS(
            unitary=synthesized_unitary,
            state=simple_trial_state,
            n_shots=100,
            target_indices=target_indices,
            sigma=2.0,
            eta=0.05,
            T=200,
            mode_dataset="analytical",
            mode_time="rejection_sampling",
            verbose=False,
        )
        assert qmegs.sigma == 2.0
        assert qmegs.eta == 0.05
        assert qmegs.T == 200
        assert qmegs.n_shots == 100
        assert qmegs.mode_time == "rejection_sampling"

    def test_qmegs_invalid_target_indices_type(self, synthesized_unitary, simple_trial_state):
        with pytest.raises(ValueError, match="target_indices must be a list"):
            QMEGS(
                unitary=synthesized_unitary,
                state=simple_trial_state,
                n_shots=qarp.EXACT,
                target_indices=(0, 1),  # tuple instead of list
            )

    def test_qmegs_invalid_target_indices_range(self, synthesized_unitary, simple_trial_state):
        with pytest.raises(ValueError, match="target_indices must be a list"):
            QMEGS(
                unitary=synthesized_unitary,
                state=simple_trial_state,
                n_shots=qarp.EXACT,
                target_indices=[0, 10],  # 10 is out of range for 2 qubits
            )


# =============================================================================
# Tests for QMEGS build method
# =============================================================================


class TestQMEGSBuild:
    def test_build_computes_parameters(
        self, synthesized_unitary, simple_trial_state, target_indices_fixture
    ):
        target_indices, _, _ = target_indices_fixture
        qmegs = QMEGS(
            unitary=synthesized_unitary,
            state=simple_trial_state,
            n_shots=qarp.EXACT,
            target_indices=target_indices,
        ).build()

        assert qmegs.p is not None
        assert qmegs.eigenvalues is not None
        assert qmegs.pmin is not None
        assert qmegs.ptail is not None
        assert qmegs.alpha is not None
        assert qmegs.q is not None
        assert qmegs.n_samples is not None
        assert qmegs.n_samples > 0

    def test_build_returns_self(
        self, synthesized_unitary, simple_trial_state, target_indices_fixture
    ):
        target_indices, _, _ = target_indices_fixture
        qmegs = QMEGS(
            unitary=synthesized_unitary,
            state=simple_trial_state,
            n_shots=qarp.EXACT,
            target_indices=target_indices,
        )
        result = qmegs.build()
        assert result is qmegs

    def test_build_pmin_greater_than_ptail(
        self, synthesized_unitary, simple_trial_state, target_indices_fixture
    ):
        target_indices, _, _ = target_indices_fixture
        qmegs = QMEGS(
            unitary=synthesized_unitary,
            state=simple_trial_state,
            n_shots=qarp.EXACT,
            target_indices=target_indices,
        ).build()
        assert qmegs.pmin > qmegs.ptail

    def test_build_fails_bad_target_indices(self, synthesized_unitary, simple_trial_state):
        with pytest.raises(ValueError, match="Algorithm assumption violated"):
            QMEGS(
                unitary=synthesized_unitary,
                state=simple_trial_state,
                n_shots=qarp.EXACT,
                target_indices=[0, 1],  # These have low overlap
            ).build()


# =============================================================================
# Tests for QMEGS run method - Physical correctness
# =============================================================================


class TestQMEGSRun:
    def test_run_analytical_mode_finds_eigenvalues(
        self, synthesized_unitary, simple_trial_state, target_indices_fixture
    ):
        target_indices, _, eigenvalues = target_indices_fixture
        qmegs = QMEGS(
            unitary=synthesized_unitary,
            state=simple_trial_state,
            n_shots=qarp.EXACT,
            target_indices=target_indices,
            eta=0.01,
            T=100,
            mode_dataset="analytical",
        ).build()

        result = qmegs.run()
        assert len(result) == len(target_indices)

        expected_eigenvalues = sorted(eigenvalues[target_indices])
        found_eigenvalues = sorted(result)
        for found, expected in zip(found_eigenvalues, expected_eigenvalues, strict=True):
            assert abs(found - expected) < 0.1, f"Found {found}, expected {expected}"

    def test_run_multiple_times_consistent(
        self, synthesized_unitary, simple_trial_state, target_indices_fixture
    ):
        target_indices, _, eigenvalues = target_indices_fixture
        np.random.seed(42)

        qmegs = QMEGS(
            unitary=synthesized_unitary,
            state=simple_trial_state,
            n_shots=qarp.EXACT,
            target_indices=target_indices,
            eta=0.01,
            T=100,
            mode_dataset="analytical",
        ).build()

        results = []
        for _ in range(3):
            result = qmegs.run()
            results.append(sorted(result))

        expected_eigenvalues = sorted(eigenvalues[target_indices])
        for run_result in results:
            for found, expected in zip(run_result, expected_eigenvalues, strict=True):
                assert abs(found - expected) < 0.15

    def test_run_clears_dataset_between_runs(
        self, synthesized_unitary, simple_trial_state, target_indices_fixture
    ):
        target_indices, _, _ = target_indices_fixture
        qmegs = QMEGS(
            unitary=synthesized_unitary,
            state=simple_trial_state,
            n_shots=qarp.EXACT,
            target_indices=target_indices,
            eta=0.01,
            T=100,
        ).build()

        qmegs.run()
        n_samples_first = len(qmegs.dataset)

        qmegs.run()
        n_samples_second = len(qmegs.dataset)

        assert n_samples_first == n_samples_second == qmegs.n_samples

    def test_run_without_build_raises(
        self, synthesized_unitary, simple_trial_state, target_indices_fixture
    ):
        target_indices, _, _ = target_indices_fixture
        qmegs = QMEGS(
            unitary=synthesized_unitary,
            state=simple_trial_state,
            n_shots=qarp.EXACT,
            target_indices=target_indices,
        )
        with pytest.raises(ValueError, match="Call build"):
            qmegs.run()


# =============================================================================
# Tests for QMEGS sampling mode
# =============================================================================


class TestQMEGSSampling:
    def test_sampling_mode_with_engine(
        self, synthesized_unitary, simple_trial_state, target_indices_fixture
    ):
        np.random.seed(0)
        target_indices, _, eigenvalues = target_indices_fixture
        qmegs = QMEGS(
            unitary=synthesized_unitary,
            state=simple_trial_state,
            n_shots=10_000,
            target_indices=target_indices,
            eta=0.01,
            T=200,
            mode_dataset="sampling",
            mode_time="rejection_sampling",
        ).build()

        result = qmegs.run()

        expected_eigenvalues = sorted(eigenvalues[target_indices])
        found_eigenvalues = sorted(result)
        for found, expected in zip(found_eigenvalues, expected_eigenvalues, strict=True):
            assert abs(found - expected) < 0.15

    def test_sampling_mode_hadamard_test_opt_in(
        self, synthesized_unitary, simple_trial_state, target_indices_fixture
    ):
        """Explicit ``primitive=HadamardTest()`` — the finite-shot protocol path
        (the pre-refactor default; the default primitive is now exact StateVector)."""
        from qarp.algorithms import HadamardTest

        np.random.seed(0)
        target_indices, _, eigenvalues = target_indices_fixture
        qmegs = QMEGS(
            unitary=synthesized_unitary,
            state=simple_trial_state,
            n_shots=10_000,
            target_indices=target_indices,
            eta=0.01,
            T=200,
            mode_dataset="sampling",
            mode_time="rejection_sampling",
            primitive=HadamardTest(),
        ).build()

        result = qmegs.run()

        expected_eigenvalues = sorted(eigenvalues[target_indices])
        found_eigenvalues = sorted(result)
        for found, expected in zip(found_eigenvalues, expected_eigenvalues, strict=True):
            assert abs(found - expected) < 0.15

    def test_rejection_sampling_mode(
        self, synthesized_unitary, simple_trial_state, target_indices_fixture
    ):
        target_indices, _, _ = target_indices_fixture
        qmegs = QMEGS(
            unitary=synthesized_unitary,
            state=simple_trial_state,
            n_shots=qarp.EXACT,
            target_indices=target_indices,
            eta=0.05,
            T=50,
            mode_dataset="analytical",
            mode_time="rejection_sampling",
        ).build()

        result = qmegs.run()
        assert len(result) == len(target_indices)


# =============================================================================
# Tests for filtered density function
# =============================================================================


class TestFilteredDensityFunction:
    def test_filtered_density_peaks_at_eigenvalues(
        self, synthesized_unitary, simple_trial_state, target_indices_fixture
    ):
        target_indices, _, eigenvalues = target_indices_fixture
        qmegs = QMEGS(
            unitary=synthesized_unitary,
            state=simple_trial_state,
            n_shots=qarp.EXACT,
            target_indices=target_indices,
            eta=0.01,
            T=100,
        ).build()

        dataset = qmegs._generate_data_analytical()

        J = int(np.floor(2 * np.pi * qmegs.T / qmegs.q))
        theta_js = np.array([-np.pi + j * qmegs.q / qmegs.T for j in range(J + 1)])
        G_js = qmegs.filtered_density_function(dataset, theta_js, qmegs.n_samples)

        peak_indices = np.argsort(G_js)[-len(target_indices) :]
        peak_thetas = theta_js[peak_indices]

        target_eigs = eigenvalues[target_indices]
        for peak in peak_thetas:
            min_dist = min(abs(peak - eig) for eig in target_eigs)
            assert min_dist < 0.1


class TestFilteredDensityVectorization:
    """The vectorised filtered_density_function must reproduce the original
    per-theta Python loop exactly."""

    def test_vectorized_matches_loop(self, synthesized_unitary, simple_trial_state):
        import numpy as np

        qmegs = QMEGS(
            unitary=synthesized_unitary,
            state=simple_trial_state,
            n_shots=qarp.EXACT,
            target_indices=[0],
        )
        rng = np.random.default_rng(3)
        times = rng.uniform(-50, 50, size=37)
        meas = rng.normal(size=37) + 1j * rng.normal(size=37)
        dataset = list(zip(times, meas, strict=True))
        theta_js = np.linspace(-np.pi, np.pi, 101)
        n_samples = 37

        got = qmegs.filtered_density_function(dataset, theta_js, n_samples)

        ref = np.zeros(len(theta_js))
        for idx, theta_j in enumerate(theta_js):
            summ = sum(m * np.exp(1j * theta_j * t) for t, m in dataset)
            ref[idx] = np.abs(summ) / n_samples

        assert got.shape == ref.shape
        assert np.linalg.norm(got - ref) < 1e-12


# =============================================================================
# Density delta term, contracts, shot-noise path, diagnostics
# =============================================================================


def _z_qmegs(**kwargs):
    """1-qubit H = Z on |+⟩: p = [1/2, 1/2], eigenvalues {-1, +1}."""
    trial = SimpleBlock(1, name="plus")
    trial.h(0)
    defaults = dict(
        unitary=SynthesizedTimeEvolutionBlock(n_qubits=1, operator=QubitOperator("Z0")),
        state=trial.build(),
        n_shots=qarp.EXACT,
        target_indices=[0, 1],
        T=40,
    )
    defaults.update(kwargs)
    return QMEGS(**defaults)


class TestTruncatedGaussianDensityDelta:
    def test_delta_contribution_at_zero(self):
        """At t = 0 the density carries the truncation deficit as a delta term:
        (1 − (2Φ(σ) − 1)) + 1/(√(2π)·T), with Φ the standard normal CDF —
        derived independently via scipy.stats.norm."""
        from scipy.stats import norm

        sigma, T = 1.0, 5.0
        expected = (1.0 - (2.0 * norm.cdf(sigma) - 1.0)) + 1.0 / (np.sqrt(2 * np.pi) * T)
        got = truncated_gaussian_density(np.array([0.0, T / 2]), sigma, T)
        assert got[0] == pytest.approx(expected, rel=1e-9)
        # Non-zero t inside the window: pure Gaussian term, no delta.
        assert got[1] == pytest.approx(
            np.exp(-((T / 2) ** 2) / (2 * T**2)) / (np.sqrt(2 * np.pi) * T), rel=1e-9
        )

    def test_outside_window_is_zero(self):
        assert truncated_gaussian_density(np.array([100.0]), 1.0, 5.0)[0] == 0.0


class TestValidateInputs:
    def test_unitary_without_set_time_rejected(self):
        bare = SimpleBlock(1, name="U")
        bare.x(0)
        with pytest.raises(ValueError, match="'operator' attribute and 'set_time'"):
            _z_qmegs(unitary=bare.build())

    def test_unitary_operator_wrong_type_rejected(self):
        class _Stub:
            n_qubits = 1
            operator = np.eye(2)

            def set_time(self, t):
                return self

        with pytest.raises(ValueError, match="must be a QubitOperator"):
            _z_qmegs(unitary=_Stub())


class TestSamplingContracts:
    def test_sample_times_before_build_raises(self):
        with pytest.raises(ValueError, match="Call build"):
            _z_qmegs()._sample_times()

    def test_unknown_mode_time_raises(self):
        alg = _z_qmegs()
        alg.n_samples = 5
        alg.mode_time = "bogus"
        with pytest.raises(ValueError, match="Unknown mode_time"):
            alg._sample_times()

    def test_unknown_mode_dataset_raises(self):
        alg = _z_qmegs().build()
        alg.mode_dataset = "bogus"
        with pytest.raises(ValueError, match="Unknown mode_dataset"):
            alg.generate_data()

    def test_generate_analytical_before_build_raises(self):
        alg = _z_qmegs()
        alg.n_samples = 5
        with pytest.raises(ValueError, match="Call build"):
            alg._generate_data_analytical()

    def test_rejection_sampling_exhaustion_raises(self):
        alg = _z_qmegs(filtering_function=lambda x, sigma, T: 0.0)
        with pytest.raises(RuntimeError, match="Rejection sampling failed"):
            alg._sample_time_rejection(max_attempts=50)


class TestShotNoisePath:
    def test_shot_noise_dataset_tracks_true_signal(self):
        """Finite-shot analytical data: each Z_had is a 2·Binomial(n)/n − 1
        estimate of Σ pₖ e^{−iλₖt}; with n_shots = 4000 a 5σ band is ±0.08
        per quadrature."""
        alg = _z_qmegs(n_shots=4000).build()
        alg.n_samples = 6  # keep the mask matrices small
        np.random.seed(21)
        dataset = alg._generate_data_analytical()
        assert len(dataset) == 6
        for tn, z_had in dataset:
            true_z = 0.5 * np.exp(-1j * (-1.0) * tn) + 0.5 * np.exp(-1j * (1.0) * tn)
            assert abs(z_had.real - true_z.real) < 0.08
            assert abs(z_had.imag - true_z.imag) < 0.08


class TestDiagnostics:
    def test_perform_measurement_is_cos_t(self):
        # ⟨+|e^{−iZt}|+⟩ = cos t — exact with the StateVector primitive.
        alg = _z_qmegs().build()
        for t in (0.3, 1.1):
            assert alg._perform_measurement(t) == pytest.approx(np.cos(t), abs=1e-9)

    def test_unitary_at_time_verbose_prints_synthesis_error(self, capsys):
        alg = _z_qmegs().build()
        alg._unitary_at_time(0.5, verbose=True)
        assert "Unitary approximation error at t=0.5" in capsys.readouterr().out

    def test_repr_mentions_configuration(self):
        text = repr(_z_qmegs())
        assert text.startswith("QMEGS(n_qubits=1")
        assert "mode_dataset='analytical'" in text

    def test_verbose_build_prints_overlaps_and_samples(self, capsys):
        _z_qmegs(verbose=True).build()
        out = capsys.readouterr().out
        assert "Overlaps:" in out
        assert "p_min:" in out
        assert "Number of samples:" in out


class TestRunBranches:
    def test_exhausted_search_space_breaks_early(self, capsys):
        """An exclusion radius wider than the grid wipes the search space after
        the first hit: run() must stop early instead of indexing into nothing."""
        alg = _z_qmegs(verbose=True).build()
        np.random.seed(4)
        alg.alpha = 1e6  # exclusion radius alpha/T covers the whole grid
        res = alg.run()
        assert len(res) == 1  # second target unreachable -> loop broke
        assert "Found eigenphase 1:" in capsys.readouterr().out


# =============================================================================
# overlaps= (pipeline_hardening_plan.md P1.2)
# =============================================================================


class TestQMEGSOverlaps:
    def test_tuple_form_skips_diagonalisation(
        self, synthesized_unitary, simple_trial_state, target_indices_fixture, monkeypatch
    ):
        target_indices, overlaps, _ = target_indices_fixture
        pmin = float(min(overlaps[i] for i in target_indices))
        ptail = float(sum(overlaps[i] for i in range(len(overlaps)) if i not in target_indices))

        def _boom(*a, **k):
            raise AssertionError("eigh called on the tuple path")

        monkeypatch.setattr(np.linalg, "eigh", _boom)
        qmegs = QMEGS(
            unitary=synthesized_unitary,
            state=simple_trial_state,
            n_shots=None,
            target_indices=target_indices,
            T=100,
            eta=0.01,
            mode_dataset="sampling",
            overlaps=(pmin, ptail),
        ).build()
        assert qmegs.p is None and qmegs.eigenvalues is None
        # Hand-computed parameter formulas from arXiv:2402.01013.
        assert qmegs.alpha == pytest.approx(np.sqrt(np.log(1 / (pmin - ptail))))
        assert qmegs.q == pytest.approx(np.sqrt(np.log(2 * pmin / (pmin + ptail))))
        expected_n = int(
            round(np.log((100 / qmegs.q + len(target_indices)) / 0.01) / (pmin - ptail) ** 2)
        )
        assert qmegs.n_samples == expected_n

    def test_sampling_mode_on_noisy_engine_with_supplied_overlaps(
        self, synthesized_unitary, simple_trial_state, target_indices_fixture
    ):
        import qarpx as qx
        from qarp.algorithms import HadamardTest
        from qarp.devices import NoiseModel
        from qarp.engines import QarpEngine

        np.random.seed(0)
        target_indices, overlaps, eigenvalues = target_indices_fixture
        pmin = float(min(overlaps[i] for i in target_indices))
        ptail = float(sum(overlaps[i] for i in range(len(overlaps)) if i not in target_indices))
        engine = QarpEngine(
            n_qubits=3, noise_model=NoiseModel.depolarizing(1e-3, [qx.GateType.CX]), seed=0
        )
        qmegs = QMEGS(
            unitary=synthesized_unitary,
            state=simple_trial_state,
            n_shots=10_000,
            target_indices=target_indices,
            eta=0.01,
            T=200,
            mode_dataset="sampling",
            mode_time="rejection_sampling",
            overlaps=(pmin, ptail),
            primitive=HadamardTest(),
            engine=engine,
        ).build()
        result = qmegs.run()
        for found, expected in zip(
            sorted(result), sorted(eigenvalues[target_indices]), strict=True
        ):
            assert abs(found - expected) < 0.15

    def test_classical_overlaps_refused_on_noisy_engine(
        self, synthesized_unitary, simple_trial_state, target_indices_fixture
    ):
        import qarpx as qx
        from qarp.devices import NoiseModel
        from qarp.engines import QarpEngine
        from qarp.errors import CapabilityError

        target_indices, _, _ = target_indices_fixture
        engine = QarpEngine(n_qubits=3, noise_model=NoiseModel.depolarizing(1e-3, [qx.GateType.CX]))
        with pytest.raises(CapabilityError, match="overlaps="):
            QMEGS(
                unitary=synthesized_unitary,
                state=simple_trial_state,
                n_shots=1000,
                target_indices=target_indices,
                mode_dataset="sampling",
                engine=engine,
            ).build()

    @pytest.mark.parametrize(
        "overlaps, mode, match",
        [
            ((0.2, 0.3), "sampling", "ptail < pmin"),
            ((1.2, 0.1), "sampling", "ptail < pmin"),
            ((-0.1, 0.0), "sampling", "ptail < pmin"),
            ((0.7, 0.4), "sampling", "pmin \\+ ptail"),
            ((float("nan"), 0.1), "sampling", "finite"),
            ("bogus", "sampling", "classical"),
            ((0.6, 0.1), "analytical", "analytical"),
        ],
    )
    def test_overlaps_validation(
        self, synthesized_unitary, simple_trial_state, target_indices_fixture, overlaps, mode, match
    ):
        target_indices, _, _ = target_indices_fixture
        with pytest.raises(ValueError, match=match):
            QMEGS(
                unitary=synthesized_unitary,
                state=simple_trial_state,
                n_shots=None,
                target_indices=target_indices,
                mode_dataset=mode,
                overlaps=overlaps,
            )

    def test_numpy_pair_is_validated_not_truth_tested(
        self, synthesized_unitary, simple_trial_state, target_indices_fixture
    ):
        """A numpy ``(pmin, ptail)`` used to hit ``overlaps != "classical"``,
        which returns an array whose truth value is ambiguous — the user saw
        numpy's error instead of the validation message or a working build."""
        target_indices, overlaps, _ = target_indices_fixture
        pmin = float(min(overlaps[i] for i in target_indices))
        ptail = float(sum(overlaps[i] for i in range(len(overlaps)) if i not in target_indices))
        kwargs = dict(
            unitary=synthesized_unitary,
            state=simple_trial_state,
            n_shots=None,
            target_indices=target_indices,
            T=100,
            eta=0.01,
            mode_dataset="sampling",
        )
        good = QMEGS(overlaps=np.array([pmin, ptail]), **kwargs).build()
        assert good.p is None
        assert good.alpha == pytest.approx(np.sqrt(np.log(1 / (pmin - ptail))))
        with pytest.raises(ValueError, match="ptail < pmin"):
            QMEGS(overlaps=np.array([0.1, 0.6]), **kwargs)

    def test_classical_numerics_unchanged(
        self,
        synthesized_unitary,
        simple_trial_state,
        simple_2qubit_hamiltonian,
        target_indices_fixture,
    ):
        """``p`` from build() equals |<v_k|psi>|^2 computed in-test from the
        dense spectrum and the analytic |0>(x)|+> = [1,0,1,0]/sqrt(2)."""
        target_indices, _, _ = target_indices_fixture
        qmegs = QMEGS(
            unitary=synthesized_unitary,
            state=simple_trial_state,
            n_shots=None,
            target_indices=target_indices,
            mode_dataset="analytical",
        ).build()
        _, vecs = np.linalg.eigh(simple_2qubit_hamiltonian.sparse_matrix(2).toarray())
        psi = np.array([1, 0, 1, 0]) / np.sqrt(2)
        expected = np.abs(vecs.conj().T @ psi) ** 2
        np.testing.assert_allclose(qmegs.p, expected, atol=1e-12)
