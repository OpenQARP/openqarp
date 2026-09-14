import numpy as np
import pytest

# Must precede the qarp import: the SpectrumEstimator re-export in
# qarp.algorithms is itself gated on cvxpy, so without cvxpy the import below
# raises and the module fails to *collect* rather than skipping.
pytest.importorskip("cvxpy")

from qarp.algorithms import SpectrumEstimator


@pytest.fixture
def simple_distribution():
    """Create a simple synthetic distribution for testing."""
    # Single peak at phase 0.3 with degeneracy 4 (for 2 qubits, hilbert_dim=4)
    return {
        0.25: 0.2,
        0.3125: 0.6,  # Main peak
        0.5: 0.2,
    }


@pytest.fixture
def multi_peak_distribution():
    """Create a distribution with multiple peaks."""
    return {
        0.0: 0.05,
        0.2: 0.3,  # First peak
        0.25: 0.2,
        0.5: 0.05,
        0.6: 0.25,  # Second peak
        0.65: 0.15,
    }


# Initialization tests
def test_init_default_parameters():
    """Test initialization with default parameters."""
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3)

    assert optimizer.n_qubits == 2
    assert optimizer.n_ancilla == 3
    assert optimizer.hilbert_dim == 4
    assert optimizer.mode == "auto"
    assert optimizer.verbose is False


def test_init_custom_parameters():
    """Test initialization with custom parameters."""
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3, mode="l1", verbose=True)

    assert optimizer.n_qubits == 2
    assert optimizer.hilbert_dim == 4
    assert optimizer.mode == "l1"
    assert optimizer.verbose is True


def test_invalid_mode_raises_error():
    """Test that invalid mode raises ValueError."""
    with pytest.raises(ValueError, match="Invalid mode"):
        SpectrumEstimator(n_qubits=2, n_ancilla=3, mode="invalid_mode")


def test_valid_modes():
    """Test all valid modes can be initialized."""
    valid_modes = ["l2", "l1", "l2_adaptive", "l2_relaxed"]

    for mode in valid_modes:
        optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3, mode=mode)
        assert optimizer.mode == mode


def test_custom_kernel_function():
    """Test initialization with custom kernel function."""
    custom_kernel = lambda delta: np.exp(-(delta**2))
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3, kernel_fn=custom_kernel)

    # Test kernel is callable
    test_delta = np.array([0.1, 0.2])
    result = optimizer.kernel_fn(test_delta)
    assert result.shape == test_delta.shape


# Dirichlet kernel tests
def test_dirichlet_kernel_shape():
    """Test kernel output shape matches input."""
    delta = np.linspace(-0.5, 0.5, 20)
    result = SpectrumEstimator._dirichlet_kernel(delta, M=3)

    assert result.shape == delta.shape


def test_dirichlet_kernel_peak_at_zero():
    """Test kernel has maximum at delta=0."""
    delta = np.linspace(-0.1, 0.1, 41)
    result = SpectrumEstimator._dirichlet_kernel(delta, M=3)

    max_idx = np.argmax(result)
    assert abs(delta[max_idx]) < 0.01  # Peak near zero


def test_dirichlet_kernel_nonnegative():
    """Test kernel is non-negative everywhere."""
    delta = np.linspace(-0.5, 0.5, 20)
    result = SpectrumEstimator._dirichlet_kernel(delta, M=3)

    assert np.all(result >= 0)


def test_dirichlet_kernel_peak_value_is_one():
    """Kernel must equal its peak value 1 at delta=0 (removable singularity).

    Regression: the old +eps form returned 0 there, which zeroed the system
    matrix diagonal when candidates coincided with the measured grid.
    """
    for M in (2, 3, 5):
        assert np.isclose(SpectrumEstimator._dirichlet_kernel(np.array([0.0]), M)[0], 1.0)
        # Value must not exceed 1 anywhere (0 is the global maximum).
        delta = np.linspace(-0.5, 0.5, 101)
        assert np.max(SpectrumEstimator._dirichlet_kernel(delta, M)) <= 1.0 + 1e-9


def _synthetic_dos_distribution(true_phases, true_degs, n_ancilla):
    """Build a measured DOS-QPE distribution by convolving a spectrum with the kernel."""
    freqs = np.array([k / (2**n_ancilla) for k in range(2**n_ancilla)])
    probs = np.zeros_like(freqs)
    dim = float(np.sum(true_degs))
    for theta, d in zip(true_phases, true_degs, strict=True):
        delta = (freqs - theta + 0.5) % 1.0 - 0.5
        probs += (d / dim) * SpectrumEstimator._dirichlet_kernel(delta, n_ancilla)
    probs /= probs.sum()
    return {f: p for f, p in zip(freqs, probs, strict=True)}, freqs


def test_debias_total_degeneracy_matches_dim():
    """Debiased weights respect the sum == Hilbert-dimension constraint."""
    dist, freqs = _synthetic_dos_distribution([0.25, 0.5], [3, 1], n_ancilla=3)
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3, mode="l2", verbose=False)
    phases, degeneracies = optimizer.estimate(
        dist, freqs=freqs, cluster=False, debias=True, threshold=0.2
    )
    assert len(phases) > 0
    assert np.isclose(np.sum(degeneracies), optimizer.hilbert_dim, atol=1e-6)
    # The dominant true peak (degeneracy 3 at 0.25) should be recovered on-grid.
    assert np.any(np.isclose(phases, 0.25, atol=1.0 / 2**3))


def test_integer_degeneracies_sum_preserved():
    """integer_degeneracies yields integers summing exactly to the Hilbert dim."""
    dist, freqs = _synthetic_dos_distribution([0.25, 0.5], [3, 1], n_ancilla=3)
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3, mode="l2", verbose=False)
    _, degeneracies = optimizer.estimate(
        dist, freqs=freqs, cluster=True, integer_degeneracies=True, threshold=0.2
    )
    assert np.all(degeneracies == np.round(degeneracies))  # integer-valued
    assert int(np.sum(degeneracies)) == optimizer.hilbert_dim


# Data preparation tests
def test_prepare_with_float_keys():
    """Test preparation with float frequency keys."""
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3)
    distribution = {0.25: 0.5, 0.75: 0.5}

    sample_bins, observed_data = optimizer.prepare_observed_data(distribution, n_bins=50)

    assert len(sample_bins) == 50
    assert len(observed_data) == len(sample_bins)
    assert np.sum(observed_data) > 0  # Data was created


def test_prepare_with_string_keys():
    """Test preparation with bitstring keys."""
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=2)
    # For 2 ancilla bits: "00" = 0/4, "10" = 2/4 = 0.5
    distribution = {"0000": 0.6, "1000": 0.4}

    sample_bins, observed_data = optimizer.prepare_observed_data(distribution, n_bins=50)

    assert len(observed_data) > 0
    assert np.sum(observed_data) > 0


def test_prepare_with_custom_n_bins():
    """Test custom number of bins."""
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3)
    distribution = {0.5: 1.0}

    sample_bins, _ = optimizer.prepare_observed_data(distribution, n_bins=30)

    assert len(sample_bins) == 30


def test_prepare_with_precomputed_freqs():
    """Test with pre-computed frequencies."""
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3)
    freqs = np.array([0.25, 0.75])
    distribution = {"dummy1": 0.3, "dummy2": 0.7}

    _, observed_data = optimizer.prepare_observed_data(distribution, freqs=freqs, n_bins=50)

    assert np.sum(observed_data) > 0


def test_ensure_numpy_array():
    """Test with pre-computed frequencies."""
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3)
    freqs = [0.25, 0.75]
    distribution = {"dummy1": 0.3, "dummy2": 0.7}

    with pytest.raises(ValueError, match="If frequencies are provided, they must be a numpy array"):
        _, observed_data = optimizer.prepare_observed_data(distribution, freqs=freqs, n_bins=50)


def test_prepare_with_subset_distribution():
    """Test preparation when distribution is a subset of freqs (finite-sampling with filtering)."""
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3)

    # Full freqs array (simulating dosqpe.freqs)
    freqs = np.array([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])

    # Filtered distribution with only some frequencies (simulating filtering low-probability outcomes)
    # Keys are float frequencies, values are probabilities
    distribution = {0.25: 0.6, 0.5: 0.3, 0.75: 0.1}

    sample_bins, observed_data = optimizer.prepare_observed_data(
        distribution, freqs=freqs, n_bins=100
    )

    # Verify the data was prepared correctly
    assert len(sample_bins) == 100
    assert np.sum(observed_data) > 0

    # The probs array should be padded to match freqs length
    assert len(optimizer.probs) == len(freqs)

    # Check that the probabilities are in the correct positions
    # freqs[2] = 0.25 should have prob 0.6
    # freqs[4] = 0.5 should have prob 0.3
    # freqs[6] = 0.75 should have prob 0.1
    assert np.isclose(optimizer.probs[2], 0.6)
    assert np.isclose(optimizer.probs[4], 0.3)
    assert np.isclose(optimizer.probs[6], 0.1)

    # Other positions should be zero
    assert np.isclose(optimizer.probs[0], 0.0)
    assert np.isclose(optimizer.probs[1], 0.0)
    assert np.isclose(optimizer.probs[3], 0.0)
    assert np.isclose(optimizer.probs[5], 0.0)
    assert np.isclose(optimizer.probs[7], 0.0)


# Clustering tests
def test_cluster_fixed_method():
    """Test fixed-tolerance clustering."""
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3)

    # Three phases: two close together, one far
    phases = np.array([0.1, 0.11, 0.5])
    degeneracies = np.array([1.0, 1.0, 2.0])

    clustered_phases, clustered_degs = optimizer.cluster_estimates(
        phases, degeneracies, tolerance=0.05, method="fixed"
    )

    # Should merge first two
    assert len(clustered_phases) == 2
    assert clustered_degs[0] == 2  # 1 + 1


def test_cluster_gap_method():
    """Test gap-based clustering."""
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3)

    # Phases with natural gap
    phases = np.array([0.1, 0.12, 0.5, 0.52])
    degeneracies = np.array([1.0, 1.0, 1.0, 1.0])

    clustered_phases, clustered_degs = optimizer.cluster_estimates(
        phases, degeneracies, method="gap", gap_threshold=2.0
    )

    # Should create two clusters
    assert len(clustered_phases) == 2


def test_cluster_empty_input():
    """Test clustering with empty arrays."""
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3)

    phases = np.array([])
    degeneracies = np.array([])

    clustered_phases, clustered_degs = optimizer.cluster_estimates(phases, degeneracies)

    assert len(clustered_phases) == 0
    assert len(clustered_degs) == 0


def test_cluster_single_phase():
    """Test clustering with single phase."""
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3)

    phases = np.array([0.5])
    degeneracies = np.array([4.0])

    clustered_phases, clustered_degs = optimizer.cluster_estimates(phases, degeneracies)

    assert len(clustered_phases) == 1
    assert clustered_phases[0] == 0.5
    assert clustered_degs[0] == 4


def test_cluster_weighted_average():
    """Test that cluster centers use weighted average."""
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3)

    phases = np.array([0.1, 0.2])
    degeneracies = np.array([1.0, 3.0])

    clustered_phases, _ = optimizer.cluster_estimates(
        phases, degeneracies, tolerance=0.2, method="fixed"
    )

    # Weighted average: (0.1*1 + 0.2*3) / 4 = 0.175
    assert len(clustered_phases) == 1
    assert abs(clustered_phases[0] - 0.175) < 1e-10


# L2 optimization tests
def test_l2_basic_optimization(simple_distribution):
    """Test basic L2 optimization."""
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3, mode="l2", verbose=False)

    phases, degeneracies = optimizer.estimate(
        simple_distribution, n_candidates=20, n_bins=50, cluster=False
    )

    assert len(phases) > 0
    assert len(degeneracies) == len(phases)
    assert np.sum(degeneracies) <= 4 + 1  # Close to hilbert_dim


def test_l2_with_clustering(simple_distribution):
    """Test L2 optimization with clustering."""
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3, mode="l2", verbose=False)

    phases, degeneracies = optimizer.estimate(
        simple_distribution, n_candidates=20, n_bins=50, cluster=True
    )

    assert optimizer.clustered_phases is not None
    assert optimizer.clustered_degeneracies is not None
    assert len(phases) <= 20  # Clustered should be fewer


def test_l2_optimization_info_stored(simple_distribution):
    """Test that optimization info is stored."""
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3, mode="l2", verbose=False)

    optimizer.estimate(simple_distribution, n_candidates=20, n_bins=50, cluster=False)

    assert "status" in optimizer.optimization_info
    assert "objective_value" in optimizer.optimization_info


# L1 optimization tests
def test_l1_basic_optimization(simple_distribution):
    """Test basic L1 optimization."""
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3, mode="l1", verbose=False)

    phases, degeneracies = optimizer.estimate(
        simple_distribution, n_candidates=20, n_bins=50, cluster=False
    )

    assert len(phases) > 0
    assert len(degeneracies) == len(phases)


# L2 relaxed optimization tests
def test_l2_relaxed_optimization(simple_distribution):
    """Test L2 relaxed optimization."""
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3, mode="l2_relaxed", verbose=False)

    phases, degeneracies = optimizer.estimate(
        simple_distribution,
        n_candidates=20,
        n_bins=50,
        cluster=False,
        constraint_penalty=10.0,
    )

    assert len(phases) > 0
    # Relaxed constraint allows deviation from hilbert_dim
    assert np.sum(degeneracies) <= 4 * 1.1


# L2 adaptive optimization tests
def test_l2_adaptive_optimization(simple_distribution):
    """Test L2 adaptive optimization."""
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3, mode="l2_adaptive", verbose=False)

    phases, degeneracies = optimizer.estimate(
        simple_distribution,
        n_bins=50,
        cluster=False,
        initial_candidates=15,
        n_refinements=2,
    )

    assert len(phases) > 0
    assert "refinement_1_grid_size" in optimizer.optimization_info
    assert "refinement_2_grid_size" in optimizer.optimization_info


# Optimization kwargs tests
def test_custom_l1_penalty(simple_distribution):
    """Test custom L1 penalty parameter."""
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3, mode="l2", verbose=False)

    phases, _ = optimizer.estimate(
        simple_distribution,
        n_candidates=20,
        n_bins=50,
        cluster=False,
        l1_penalty=10.0,  # High penalty for more sparsity
    )

    assert len(phases) > 0


def test_custom_threshold(simple_distribution):
    """Test custom threshold parameter."""
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3, mode="l2", verbose=False)

    phases_low, _ = optimizer.estimate(
        simple_distribution,
        n_candidates=20,
        n_bins=50,
        cluster=False,
        threshold=0.01,  # Low threshold, keep more peaks
    )

    optimizer2 = SpectrumEstimator(n_qubits=2, n_ancilla=3, mode="l2", verbose=False)
    phases_high, _ = optimizer2.estimate(
        simple_distribution,
        n_candidates=20,
        n_bins=50,
        cluster=False,
        threshold=1.0,  # High threshold, fewer peaks
    )

    assert len(phases_low) >= len(phases_high)


# Multi-peak reconstruction tests
def test_two_peak_reconstruction(multi_peak_distribution):
    """Test reconstruction of two distinct peaks."""
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3, mode="l2", verbose=False)

    phases, degeneracies = optimizer.estimate(
        multi_peak_distribution, n_candidates=30, n_bins=50, cluster=True
    )

    # Should find approximately 2 peaks
    assert 1 <= len(phases) <= 4


# Edge cases tests
def test_single_frequency_distribution():
    """Test with single frequency in distribution."""
    optimizer = SpectrumEstimator(
        n_qubits=2, n_ancilla=2, mode="l2", peak_guess="uniform", verbose=False
    )
    distribution = {0.5: 1.0}

    phases, degeneracies = optimizer.estimate(
        distribution, n_candidates=20, n_bins=30, cluster=False
    )

    assert len(phases) > 0


def test_uniform_distribution():
    """Test with uniform distribution."""
    optimizer = SpectrumEstimator(
        n_qubits=2, n_ancilla=2, mode="l2", peak_guess="uniform", verbose=False
    )
    distribution = {i / 8: 1 / 8 for i in range(8)}

    phases, degeneracies = optimizer.estimate(
        distribution, n_candidates=20, n_bins=30, cluster=False
    )

    assert len(phases) > 0


def test_optimization_without_cluster(simple_distribution):
    """Test that results are stored correctly without clustering."""
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3, mode="l2", verbose=False)

    phases, degeneracies = optimizer.estimate(
        simple_distribution, n_candidates=20, n_bins=50, cluster=False
    )

    assert optimizer.estimated_phases is not None
    assert optimizer.estimated_degeneracies is not None
    assert optimizer.clustered_phases is None
    assert optimizer.clustered_degeneracies is None
    assert np.array_equal(phases, optimizer.estimated_phases)


# Results storage tests
def test_results_accessible_after_optimization(simple_distribution):
    """Test that results are accessible after optimization."""
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3, mode="l2", verbose=False)

    phases, degeneracies = optimizer.estimate(
        simple_distribution, n_candidates=20, n_bins=50, cluster=True
    )

    # Check clustered results are stored
    assert optimizer.clustered_phases is not None
    assert optimizer.clustered_degeneracies is not None
    assert np.array_equal(phases, optimizer.clustered_phases)
    assert np.array_equal(degeneracies, optimizer.clustered_degeneracies)

    # Check raw results are also stored
    assert optimizer.estimated_phases is not None
    assert optimizer.estimated_degeneracies is not None
    assert optimizer.full_weights is not None


# Hamming weight tests
def test_hamming_weight_hilbert_dim():
    """Test that hamming_weight correctly sets Hilbert dimension via C(n, k)."""
    from math import comb

    optimizer = SpectrumEstimator(n_qubits=6, n_ancilla=3, hamming_weight=2)
    assert optimizer.hilbert_dim == comb(6, 2)  # C(6,2) = 15
    assert optimizer.hamming_weight == 2

    optimizer2 = SpectrumEstimator(n_qubits=4, n_ancilla=3, hamming_weight=1)
    assert optimizer2.hilbert_dim == comb(4, 1)  # C(4,1) = 4

    # Without hamming_weight, should use 2^n_qubits
    optimizer3 = SpectrumEstimator(n_qubits=4, n_ancilla=3)
    assert optimizer3.hilbert_dim == 2**4  # 16
    assert optimizer3.hamming_weight is None


# Peak guess tests
def test_invalid_peak_guess_raises_error():
    """Test that invalid peak_guess raises ValueError."""
    with pytest.raises(ValueError, match="Invalid peak guess method"):
        SpectrumEstimator(n_qubits=2, n_ancilla=3, peak_guess="invalid")


def test_find_peaks_guess(multi_peak_distribution):
    """Test convex optimization with peak_guess='find_peaks' (the default guess)."""
    optimizer = SpectrumEstimator(
        n_qubits=2, n_ancilla=3, mode="l2", peak_guess="find_peaks", verbose=False
    )

    phases, degeneracies = optimizer.estimate(multi_peak_distribution, n_bins=50, cluster=True)

    assert len(phases) >= 1
    assert len(degeneracies) == len(phases)


def test_uniform_guess(multi_peak_distribution):
    """Test convex optimization with peak_guess='uniform'."""
    optimizer = SpectrumEstimator(
        n_qubits=2, n_ancilla=3, mode="l2", peak_guess="uniform", verbose=False
    )

    phases, degeneracies = optimizer.estimate(
        multi_peak_distribution, n_candidates=30, n_bins=50, cluster=True
    )

    assert len(phases) >= 1
    assert len(degeneracies) == len(phases)


def test_find_peaks_default():
    """Test that find_peaks is the default peak_guess strategy."""
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3)
    assert optimizer.peak_guess == "find_peaks"


# ESPRIT (parametric, off-grid) tests
def test_default_mode_is_auto():
    """'auto' is the default estimator."""
    assert SpectrumEstimator(n_qubits=2, n_ancilla=3).mode == "auto"


def test_esprit_recovers_offgrid_phases():
    """ESPRIT resolves phases between grid points (super-resolution)."""
    true_phases = [0.2, 0.55]  # neither lies on the 1/16 grid
    dist, freqs = _synthetic_dos_distribution(true_phases, [3, 1], n_ancilla=4)
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=4, mode="esprit")
    phases, degeneracies = optimizer.estimate(dist, freqs=freqs, cluster=False, threshold=0.2)

    grid_spacing = 1.0 / 2**4
    for tp in true_phases:
        circular_err = np.min(np.abs(((phases - tp + 0.5) % 1.0) - 0.5))
        assert circular_err < grid_spacing / 2  # better than nearest-grid resolution


def test_esprit_model_order_override():
    """Explicit model_order is honoured."""
    dist, freqs = _synthetic_dos_distribution([0.2, 0.55], [3, 1], n_ancilla=4)
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=4, mode="esprit")
    optimizer.estimate(dist, freqs=freqs, cluster=False, model_order=2, threshold=0.0)
    assert optimizer.optimization_info["esprit_model_order"] == 2


def test_esprit_requires_full_grid():
    """ESPRIT rejects sparse / off-grid distributions with a clear error."""
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3, mode="esprit")
    with pytest.raises(ValueError, match="uniform grid"):
        optimizer.estimate({0.25: 0.6, 0.5: 0.4})  # only 2 of 8 grid points


def test_convex_mode_still_available():
    """The convex modes remain selectable alongside the default."""
    dist, freqs = _synthetic_dos_distribution([0.25, 0.5], [3, 1], n_ancilla=3)
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=3, mode="l2")
    phases, _ = optimizer.estimate(dist, freqs=freqs, cluster=False, threshold=0.2)
    assert len(phases) > 0


# Atomic-norm (gridless) tests
def test_atomic_norm_recovers_offgrid_phases():
    """Atomic-norm denoising resolves off-grid phases without a fixed line budget."""
    true_phases = [0.2, 0.55]
    dist, freqs = _synthetic_dos_distribution(true_phases, [3, 1], n_ancilla=4)
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=4, mode="atomic_norm")
    phases, _ = optimizer.estimate(dist, freqs=freqs, cluster=False, threshold=0.2)

    grid_spacing = 1.0 / 2**4
    for tp in true_phases:
        circular_err = np.min(np.abs(((phases - tp + 0.5) % 1.0) - 0.5))
        assert circular_err < grid_spacing  # off-grid recovery


# Auto-selection (Wasserstein) tests
def test_circular_wasserstein_zero_for_identical():
    """Circular W1 of a distribution with itself is zero."""
    a = np.array([0.1, 0.4, 0.2, 0.3])
    assert SpectrumEstimator._circular_wasserstein1(a, a) == 0.0


def test_auto_selects_and_records_mode():
    """'auto' runs candidates, records the Wasserstein scores, and picks the best."""
    dist, freqs = _synthetic_dos_distribution([0.2, 0.55], [3, 1], n_ancilla=4)
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=4)  # auto default
    phases, degeneracies = optimizer.estimate(dist, freqs=freqs, cluster=False, threshold=0.2)

    info = optimizer.optimization_info
    assert info["auto_selected_mode"] in optimizer.DEFAULT_AUTO_MODES
    # The selected mode is the argmin of the recorded Wasserstein scores.
    assert info["auto_selected_mode"] == min(info["auto_scores"], key=info["auto_scores"].get)
    assert len(phases) > 0


def test_auto_custom_candidate_list():
    """auto_modes restricts which estimators 'auto' considers."""
    dist, freqs = _synthetic_dos_distribution([0.2, 0.55], [3, 1], n_ancilla=4)
    optimizer = SpectrumEstimator(n_qubits=2, n_ancilla=4)
    optimizer.estimate(dist, freqs=freqs, cluster=False, threshold=0.2, auto_modes=["esprit", "l2"])
    assert set(optimizer.optimization_info["auto_scores"]).issubset({"esprit", "l2"})


# =============================================================================
# Key conversions, rounding/clustering algebra, estimate/plot branches
# =============================================================================


class TestKeyConversion:
    def test_str_key_is_msb_first(self):
        est = SpectrumEstimator(n_qubits=2, n_ancilla=3)
        assert est._key_to_freq("101") == 5 / 8

    def test_tuple_key_is_lsb_first(self):
        est = SpectrumEstimator(n_qubits=2, n_ancilla=3)
        assert est._key_to_freq((1, 0, 1)) == 5 / 8  # bit q at position q
        assert est._key_to_freq((1, 0, 0)) == 1 / 8

    def test_numeric_keys_pass_through(self):
        est = SpectrumEstimator(n_qubits=2, n_ancilla=3)
        assert est._key_to_freq(0.375) == 0.375

    def test_unsupported_key_raises(self):
        est = SpectrumEstimator(n_qubits=2, n_ancilla=3)
        with pytest.raises(ValueError, match="Unsupported distribution key type"):
            est._key_to_freq([1, 0])

    def test_key_is_freq_like(self):
        est = SpectrumEstimator(n_qubits=2, n_ancilla=3)
        assert est._key_is_freq_like((0, 1, 0))
        assert est._key_is_freq_like(0.5)
        assert est._key_is_freq_like("010xyz")  # leading n_ancilla chars binary
        assert not est._key_is_freq_like("peak_a")
        assert not est._key_is_freq_like(None)


class TestRequireCvxpy:
    def test_raises_when_unavailable(self, monkeypatch):
        import qarp.algorithms._spectral_estimation as mod

        monkeypatch.setattr(mod, "CVXPY_AVAILABLE", False)
        with pytest.raises(ImportError, match="cvxpy is required for testing"):
            SpectrumEstimator._require_cvxpy("testing")


class TestRoundDegeneracies:
    def test_independent_rounding(self):
        est = SpectrumEstimator(n_qubits=2, n_ancilla=3)  # hilbert_dim 4
        out = est._round_degeneracies(np.array([1.4, 2.6]), preserve_sum=False)
        assert list(out) == [1, 3]

    def test_largest_remainder_hands_out_deficit(self):
        est = SpectrumEstimator(n_qubits=2, n_ancilla=3)
        # floors [1,1,1] sum 3, hilbert 4: one unit to a largest fraction —
        # never to the smallest.
        out = est._round_degeneracies(np.array([1.4, 1.4, 1.2]), preserve_sum=True)
        assert sorted(out) == [1, 1, 2]
        assert out[2] == 1
        assert out.sum() == 4

    def test_largest_remainder_reclaims_excess(self):
        est = SpectrumEstimator(n_qubits=2, n_ancilla=3)
        # floors [3,2] sum 5 > hilbert 4: reclaim one from the smallest fraction.
        out = est._round_degeneracies(np.array([3.1, 2.1]), preserve_sum=True)
        assert out.sum() == 4

    def test_empty_input(self):
        est = SpectrumEstimator(n_qubits=2, n_ancilla=3)
        assert len(est._round_degeneracies(np.array([]), preserve_sum=True)) == 0


class TestClusterEstimates:
    def _est(self, verbose=False):
        return SpectrumEstimator(n_qubits=2, n_ancilla=3, verbose=verbose)

    def test_empty_and_single(self):
        est = self._est()
        phases, degs = est.cluster_estimates(np.array([]), np.array([]))
        assert len(phases) == 0 and len(degs) == 0
        phases, degs = est.cluster_estimates(np.array([0.3]), np.array([2.0]))
        assert list(phases) == [0.3] and list(degs) == [2.0]

    def test_fixed_method_default_tolerance_weighted_mean(self, capsys):
        est = self._est(verbose=True)
        # tol = 1/16: 0.1 and 0.13 merge (weighted mean), 0.5 stays alone.
        phases, degs = est.cluster_estimates(
            np.array([0.1, 0.13, 0.5]), np.array([1.0, 1.0, 2.0]), method="fixed"
        )
        assert np.allclose(phases, [0.115, 0.5])
        assert list(degs) == [2, 2]
        assert "Clustering (fixed)" in capsys.readouterr().out

    def test_fixed_method_zero_degeneracy_cluster_uses_mean(self):
        est = self._est()
        phases, degs = est.cluster_estimates(
            np.array([0.1, 0.11]), np.array([0.0, 0.0]), method="fixed"
        )
        assert np.allclose(phases, [0.105])
        assert list(degs) == [0]

    def test_gap_method_zero_degeneracy_clusters_use_mean(self):
        est = self._est()
        # gaps [0.02, 0.78], threshold 1.5*0.4: [0.1, 0.12] cluster then [0.9].
        phases, degs = est.cluster_estimates(
            np.array([0.1, 0.12, 0.9]), np.array([0.0, 0.0, 0.0]), method="gap"
        )
        assert np.allclose(phases, [0.11, 0.9])
        assert list(degs) == [0, 0]

    def test_gap_method_degenerate_grid_falls_back_to_tolerance(self):
        est = self._est()
        phases, degs = est.cluster_estimates(
            np.array([0.1, 0.1 + 1e-12, 0.1 + 2e-12]),
            np.array([1.0, 1.0, 2.0]),
            method="gap",
        )
        assert len(phases) == 1
        assert list(degs) == [4]

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown clustering method"):
            self._est().cluster_estimates(
                np.array([0.1, 0.5]), np.array([1.0, 1.0]), method="bogus"
            )

    def test_preserve_integer_sum(self):
        est = self._est()
        phases, degs = est.cluster_estimates(
            np.array([0.1, 0.5, 0.9]),
            np.array([1.4, 1.4, 1.2]),
            method="fixed",
            preserve_integer_sum=True,
        )
        assert degs.sum() == 4


class TestPrepareObservedDataBranches:
    def test_freqs_must_be_ndarray(self):
        est = SpectrumEstimator(n_qubits=2, n_ancilla=3)
        with pytest.raises(ValueError, match="must be a numpy array"):
            est.prepare_observed_data({0.25: 1.0}, freqs=[0.25, 0.5])

    def test_unmatched_frequency_warns_in_verbose(self, capsys):
        est = SpectrumEstimator(n_qubits=2, n_ancilla=3, verbose=True)
        est.prepare_observed_data({0.3: 1.0}, n_bins=32, freqs=np.array([0.0, 0.5]))
        assert "not found in freqs array" in capsys.readouterr().out


class TestEstimateBranches:
    def test_l2_adaptive_recovers_single_peak(self, simple_distribution):
        est = SpectrumEstimator(n_qubits=2, n_ancilla=3, mode="l2_adaptive")
        phases, degs = est.estimate(
            simple_distribution, n_bins=60, cluster=True, initial_candidates=40, n_refinements=1
        )
        assert len(phases) >= 1
        # Dominant reconstructed phase within a bin of the main 0.3125 peak.
        top = phases[np.argmax(degs)]
        assert abs(top - 0.3125) < 1 / 8

    def test_integer_degeneracies_without_cluster(self, simple_distribution):
        est = SpectrumEstimator(n_qubits=2, n_ancilla=3, mode="l2")
        phases, degs = est.estimate(
            simple_distribution, n_bins=60, cluster=False, integer_degeneracies=True
        )
        assert degs.dtype.kind == "i"
        assert degs.sum() == est.hilbert_dim


class TestPlotStructural:
    def test_plot_before_estimate_raises(self, simple_distribution):
        est = SpectrumEstimator(n_qubits=2, n_ancilla=3)
        with pytest.raises(ValueError, match="Run estimate"):
            est.plot(simple_distribution)

    def test_plot_returns_fig_and_overlays(self, simple_distribution):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        est = SpectrumEstimator(n_qubits=2, n_ancilla=3)
        est.estimate(simple_distribution, n_bins=60)
        result = est.plot(
            simple_distribution,
            true_phases=np.array([0.3125]),
            true_degeneracies=np.array([4.0]),
            n_bins=60,
            show_deconvolved=True,
            return_fig=True,
        )
        assert result is not None
        fig, ax = result
        assert len(ax.get_lines()) >= 1  # observed signal + overlays drawn
        plt.close(fig)

    def test_plot_show_path(self, simple_distribution, monkeypatch):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        shown = []
        monkeypatch.setattr(plt, "show", lambda: shown.append(True))
        est = SpectrumEstimator(n_qubits=2, n_ancilla=3)
        est.estimate(simple_distribution, n_bins=60)
        assert est.plot(simple_distribution, n_bins=60) is None
        assert shown == [True]
        plt.close("all")


def test_module_works_without_tqdm(monkeypatch):
    """Without tqdm the fallback must stay callable and pass iterables through.

    Regression: the guard used to bind ``tqdm = None``, so every call site
    raised TypeError whenever tqdm was not installed (it is undeclared).
    """
    import importlib
    import sys

    import qarp.algorithms._spectral_estimation as se

    monkeypatch.setitem(sys.modules, "tqdm", None)
    monkeypatch.setitem(sys.modules, "tqdm.auto", None)
    try:
        importlib.reload(se)
        assert se.TQDM_AVAILABLE is False
        assert list(se.tqdm(range(3), desc="d", total=3, disable=True)) == [0, 1, 2]
    finally:
        monkeypatch.undo()
        importlib.reload(se)
