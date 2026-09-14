import importlib
import importlib.util
from math import comb
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, cast

import numpy as np
from scipy.optimize import nnls
from scipy.signal import find_peaks

CVXPY_AVAILABLE = importlib.util.find_spec("cvxpy") is not None


class _LazyCvxpy:
    """``cvxpy`` imported on first attribute access (~0.5 s), so the default
    'esprit' mode and ``import qarp.algorithms`` never pay for it."""

    def __getattr__(self, name: str) -> Any:
        if not CVXPY_AVAILABLE:
            raise ImportError("cvxpy is required: pip install 'openqarp[convex-optim]'")
        return getattr(importlib.import_module("cvxpy"), name)


cp = _LazyCvxpy()

try:
    from tqdm.auto import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    # tqdm is optional and undeclared: the fallback must stay callable so the
    # call sites work without it (progress kwargs dropped, iteration preserved).
    def tqdm(iterable, **_kwargs):  # type: ignore[no-redef]
        return iterable


ModeName = Literal["auto", "esprit", "l2", "l1", "l2_adaptive", "l2_relaxed", "atomic_norm"]


class SpectrumEstimator:
    # Parametric (off-grid) and convex (grid-based / SDP) families.  'atomic_norm'
    # is gridless but solved as an SDP, so it lives with the cvxpy-requiring convex
    # modes.  'auto' runs several and keeps the best by a Wasserstein data-fit score.
    PARAMETRIC_MODES = ["esprit"]
    CONVEX_MODES = ["l2", "l1", "l2_adaptive", "l2_relaxed", "atomic_norm"]
    VALID_MODES = PARAMETRIC_MODES + CONVEX_MODES + ["auto"]

    # Candidate estimators tried by 'auto' (filtered to what the install supports).
    DEFAULT_AUTO_MODES: List[ModeName] = ["esprit", "l2", "atomic_norm"]

    VALID_PEAK_GUESSES = ["uniform", "find_peaks"]

    def __init__(
        self,
        n_qubits: int,
        n_ancilla: int,
        hamming_weight: Optional[int] = None,
        mode: ModeName = "auto",
        kernel_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        peak_guess: Literal["uniform", "find_peaks"] = "find_peaks",
        solver: Any = None,
        verbose: bool = False,
    ) -> None:
        """
        Spectral estimator for DOS-QPE and spectral reconstruction.

        Reconstructs the eigenvalue spectrum (eigenphases + degeneracies) from quantum
        phase estimation measurements, modelling the measurement as the true spectrum
        convolved with the QPE (Fejér) kernel.  Several estimators are provided.

        Auto (default):
        ---------------
        - 'auto' (default): run several estimators and keep the one whose reconstruction,
          pushed back through the kernel, best reproduces the measured distribution under
          a circular 1-Wasserstein (earth-mover) score.  This sidesteps the per-method
          failure modes — e.g. ESPRIT silently dropping a weak line scores poorly and is
          not selected.  Candidates default to {esprit, l2, atomic_norm}, filtered to
          what the install supports (the convex ones need cvxpy).

        Parametric:
        -----------
        - 'esprit': off-grid line-spectral estimation via the matrix-pencil / ESPRIT
          method.  The inverse-DFT of the distribution gives the (window-tapered) moments
          ``c_t = Σ_j (d_j/D)·e^{2πiθ_j t}`` — i.e. ``Tr(ρUᵗ)``, exactly what QPE encodes.
          A Hankel-matrix SVD recovers the eigenphases in closed form, *without a candidate
          grid* (no basis-mismatch / peak-splitting) and *without cvxpy*.  Degeneracies come
          from a non-negative least-squares fit against the measured distribution with the
          exact kernel.  Best when the number of distinct eigenvalues is modest relative to
          2^n_ancilla; its fixed model order can drop weak lines in dense spectra.

        Gridless convex (requires cvxpy):
        ---------------------------------
        - 'atomic_norm': off-grid like ESPRIT but with no fixed line budget.  An atomic-norm
          (total-variation) SDP denoises the moments under a sparse-spikes prior; the model
          order emerges from the recovered Toeplitz rank, so weak lines are not dropped by a
          hard cutoff.  Eigenphases via the matrix pencil on the denoised moments, then the
          same kernel NNLS for degeneracies.  The most robust single method, at SDP cost.

        Grid-based convex (require cvxpy):
        ----------------------------------
        The convex modes solve a sparse deconvolution on a candidate grid of the form
            minimize: loss(Ax - b) + λ·sparsity(x)
            subject to: sum(x) = Hilbert_dimension
        where A is the kernel system matrix and x the degeneracies.  They degrade more
        gracefully than 'esprit' for dense spectra (more lines than the ancilla resolves).

        - 'l2': LASSO-type optimization with L2 loss and L1 sparsity penalty.
            minimize: ||Ax - b||₂² + λ||x||₁
            subject to: sum(x) = dim
          Best for Gaussian noise, produces smooth solutions. Computationally efficient.

        - 'l1': Robust variant using L1 loss instead of L2.
            minimize: ||Ax - b||₁ + λ||x||₁
            subject to: sum(x) = dim
          More robust to outliers and non-Gaussian noise. Can produce sharper peaks.

        - 'l2_adaptive': Iterative grid refinement strategy.
          Starts with coarse grid, runs l2 optimization, then refines grid around
          significant peaks. Repeats for better resolution. Best for high-precision
          peak localization when approximate locations are unknown.

        - 'l2_relaxed': Soft-constraint variant of l2.
            minimize: ||Ax - b||₂² + λ₁||x||₁ + λ₂(sum(x) - dim)²
            subject to: sum(x) ≤ 1.1 * dim
          Relaxes the hard equality constraint to a penalty term. Useful when the
          strict degeneracy constraint causes numerical issues or when total degeneracy
          may deviate slightly from the Hilbert dimension.

        Post-Processing:
        ----------------
        After optimization, nearby phases can be clustered to merge peaks within a
        specified tolerance (default: 1/(2^(n_ancilla+1))). This reduces noise-induced
        splitting and produces cleaner final results.

        Args:
            n_qubits: Number of system qubits (determines Hilbert dimension = 2^n_qubits,
                      or C(n_qubits, hamming_weight) if hamming_weight is set)
            n_ancilla: Number of ancilla qubits used in QPE (determines resolution = 1/2^n_ancilla)
            hamming_weight: If set, restricts to the Dicke subspace with this Hamming weight.
                           Hilbert dimension becomes C(n_qubits, hamming_weight) instead of 2^n_qubits.
                           Use None (default) for a maximally-mixed-state probe.
            mode: Estimator - 'auto' (default), 'esprit', 'atomic_norm', 'l2', 'l1',
                  'l2_adaptive', or 'l2_relaxed'. 'auto' picks the best by a Wasserstein
                  data-fit score. All except 'esprit' require cvxpy.
            kernel_fn: Custom kernel function f(Δφ) → intensity. If None, uses Dirichlet kernel
                      appropriate for QPE: K(Δφ) = sin²(π·2^M·Δφ) / (2^M·sin(π·Δφ))²
            peak_guess: Strategy for generating initial candidate phases (default: 'find_peaks').
                       Only used by the grid-based convex modes; ignored by 'esprit'/'atomic_norm'.
                       - 'find_peaks': uses scipy.signal.find_peaks on the observed distribution.
                         Much faster and more robust for larger problems.
                       - 'uniform': places candidates on a uniform grid. The number of candidates
                         is controlled by n_candidates in estimate().
            solver: CVXPY solver instance (default: cp.CLARABEL). Can use cp.MOSEK, cp.SCS, etc.
            verbose: If True, print optimization progress, diagnostics, and iteration details
        """
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {self.VALID_MODES}")

        # cvxpy is only needed by the convex modes; 'esprit' is dependency-light.
        if mode in self.CONVEX_MODES and not CVXPY_AVAILABLE:
            raise ImportError(
                f"cvxpy is required for the '{mode}' mode. Install with: pip install cvxpy "
                "(or use the default 'esprit' mode, which does not need it)."
            )

        if peak_guess not in self.VALID_PEAK_GUESSES:
            raise ValueError(
                f"Invalid peak guess method '{peak_guess}'. Must be one of {self.VALID_PEAK_GUESSES}"
            )

        self.n_qubits = n_qubits
        self.n_ancilla = n_ancilla
        self.peak_guess = peak_guess
        self.hamming_weight = hamming_weight
        if hamming_weight is None:
            self.hilbert_dim = 2**n_qubits
        else:
            self.hilbert_dim = comb(n_qubits, hamming_weight)
        self.mode = mode
        self.solver = solver if solver is not None else (cp.CLARABEL if CVXPY_AVAILABLE else None)
        self.verbose = verbose

        # Default Dirichlet kernel
        if kernel_fn is None:
            self.kernel_fn = lambda delta: self._dirichlet_kernel(delta, n_ancilla)
        else:
            self.kernel_fn = kernel_fn

        # Results storage
        self.estimated_phases: Optional[np.ndarray] = None
        self.estimated_degeneracies: Optional[np.ndarray] = None
        self.full_weights: Optional[np.ndarray] = None
        self.clustered_phases: Optional[np.ndarray] = None
        self.clustered_degeneracies: Optional[np.ndarray] = None
        self.optimization_info: Dict[str, Any] = {}

    def _print(self, *args, **kwargs):
        """Print only if verbose is enabled."""
        if self.verbose:
            print(*args, **kwargs)

    @staticmethod
    def _require_cvxpy(context: str) -> None:
        """Raise a helpful error if cvxpy is needed but unavailable."""
        if not CVXPY_AVAILABLE:
            raise ImportError(
                f"cvxpy is required for {context}. Install with: pip install cvxpy "
                "(or use the default 'esprit' mode, which does not need it)."
            )

    @staticmethod
    def _dirichlet_kernel(delta_phi: np.ndarray, M: int) -> np.ndarray:
        """Fejér (squared-Dirichlet) kernel used as the QPE kernel function.

        ``K(Δφ) = [sin(π·2ᴹ·Δφ) / (2ᴹ·sin(π·Δφ))]²`` has a removable singularity
        at ``Δφ = 0`` (and integer Δφ) where it equals its peak value ``1``.  The
        naive ``+eps`` denominator instead evaluates to ``0`` there — which zeroes
        the diagonal of the system matrix when candidates coincide with the
        measured grid — so the limit is substituted explicitly.
        """
        delta_phi = np.asarray(delta_phi, dtype=float)
        result = np.ones_like(delta_phi)  # limit value at Δφ → 0 (or integer)
        sin_d = np.sin(np.pi * delta_phi)
        mask = ~np.isclose(sin_d, 0.0, atol=1e-12)
        result[mask] = (np.sin(np.pi * (2**M) * delta_phi[mask]) / ((2**M) * sin_d[mask])) ** 2
        return result

    def _get_candidate_phases(self, n_candidates: int, threshold: float = 1e-1) -> np.ndarray:
        """Get candidate phases based on peak_guess strategy.

        Must be called after prepare_observed_data() (which sets self.probs and self.phase_freqs).

        Args:
            n_candidates: Number of uniform grid points (only used when peak_guess='uniform').
            threshold: Significance threshold; the 0th data point is included if its
                      probability exceeds 1% of this value (only used when peak_guess='find_peaks').

        Returns:
            Array of candidate phase positions.
        """
        if self.peak_guess == "uniform":
            return np.linspace(0, 1, n_candidates, endpoint=False)
        elif self.peak_guess == "find_peaks":
            peaks = find_peaks(self.probs)[0]
            # Include the 0th data point if its probability is non-negligible
            # (find_peaks cannot detect a peak at the boundary)
            if self.probs[0] > threshold * 0.01:
                peaks = np.concatenate((np.asarray([0]), peaks))
            return self.phase_freqs[peaks]
        else:
            raise RuntimeError(
                f"Unrecognized peak guess method '{self.peak_guess}'. "
                f"Must be one of {self.VALID_PEAK_GUESSES}"
            )

    def _key_is_freq_like(self, key: Any) -> bool:
        """Whether a distribution key encodes a frequency (vs. a dummy label).

        int/float/tuple keys always encode a frequency; str keys do only if the
        leading ``n_ancilla`` characters are binary digits.
        """
        if isinstance(key, (int, float, tuple)):
            return True
        if isinstance(key, str):
            bits = key[: self.n_ancilla]
            return len(bits) > 0 and all(c in "01" for c in bits)
        return False

    def _key_to_freq(self, key: Union[str, int, float, Tuple[int, ...]]) -> float:
        """Convert a distribution key to its phase frequency in [0, 1).

        Tuple keys come from DOS-QPE and are LSB-first (``bits[q]`` is the q-th
        bit of the measured outcome), so they are read with bit 0 as the least
        significant bit — matching the ``k[::-1]`` reversal in
        :meth:`DOSQPE.plot`. String keys keep the MSB-first reading.
        """
        if isinstance(key, str):
            return int(key[: self.n_ancilla], 2) / (2**self.n_ancilla)
        if isinstance(key, tuple):
            ancilla = key[: self.n_ancilla]
            outcome = sum(int(bit) << q for q, bit in enumerate(ancilla))
            return outcome / (2**self.n_ancilla)
        if isinstance(key, (int, float)):
            return float(key)
        raise ValueError(f"Unsupported distribution key type: {type(key)}")

    def prepare_observed_data(
        self,
        distribution: Dict[Union[str, int, float, Tuple[int, ...]], float],
        n_bins: Optional[int] = None,
        freqs: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare observed data from DOS-QPE distribution.

        Args:
            distribution: DOS-QPE distribution {outcome: probability}
            n_bins: Number of bins for sampling (default: 100 * 2^n_ancilla)
            freqs: Pre-computed frequencies (optional). If None, will be extracted from distribution keys.
                   If provided and len(distribution) == len(freqs), frequencies are taken from freqs
                   and probabilities from distribution values (in iteration order).
                   If provided and len(distribution) < len(freqs), the distribution is assumed to be
                   defined on a subset of freqs (e.g., from finite-sampling experiments with filtering).

        Returns:
            sample_bins: Array of sample bin positions
            observed_data: Array of observed intensities
        """
        if n_bins is None:
            n_bins = 100 * (2**self.n_ancilla)

        if n_bins is None:
            raise ValueError("n_bins must be specified or computed")
        sample_bins = np.linspace(0, 1, n_bins, endpoint=False)
        observed_data = np.zeros_like(sample_bins)

        if freqs is not None:
            if not isinstance(freqs, np.ndarray):
                raise ValueError("If frequencies are provided, they must be a numpy array")

        # Decide how to read the distribution.  When freqs is provided and the
        # keys carry no frequency information (dummy labels), pair the values
        # with freqs in iteration order.  Otherwise the keys themselves encode
        # the frequency and must be parsed (DOS-QPE tuple keys are LSB-first).
        keys_are_freq_like = all(self._key_is_freq_like(k) for k in distribution.keys())
        dummy_mode = freqs is not None and not keys_are_freq_like

        # `dummy_mode` implies `freqs is not None`; the explicit check narrows
        # the type so `phase_freqs` (and `self.phase_freqs`) is never Optional.
        if freqs is not None and dummy_mode and len(distribution) == len(freqs):
            phase_freqs = freqs
            probs = np.array(list(distribution.values()))
        else:
            # Extract frequencies from distribution keys
            phase_freqs_list = []
            probs_list = []

            for outcome, probability in distribution.items():
                phase_freqs_list.append(self._key_to_freq(outcome))
                probs_list.append(probability)

            phase_freqs = np.array(phase_freqs_list)
            probs = np.array(probs_list)

        # Store original probabilities (before any snapping)
        self.probs_orig = probs.copy()

        # When freqs is provided and the keys encode frequencies, snap the parsed
        # frequencies onto the freqs grid.  This both pads subset distributions
        # (finite-sampling experiments where low-probability outcomes are dropped)
        # and reorders full distributions whose keys are out of grid order
        # (DOS-QPE counts arrive keyed by outcome, not sorted by frequency).
        if freqs is not None and not dummy_mode:
            # Use np.isclose for robust floating-point comparison
            padded_probs = np.zeros(len(freqs))
            for dist_freq, prob in zip(phase_freqs, probs, strict=True):
                # Find the matching index in freqs using approximate comparison
                matches = np.isclose(freqs, dist_freq, rtol=1e-9, atol=1e-12)
                if np.any(matches):
                    idx = np.argmax(matches)  # Get first matching index
                    padded_probs[idx] += prob
                else:
                    # If no exact match found, this is unexpected but we can still proceed
                    # by using the distribution frequency directly (won't be padded to freqs grid)
                    self._print(
                        f"Warning: frequency {dist_freq} from distribution not found in freqs array"
                    )
            probs = padded_probs
            phase_freqs = freqs
        elif freqs is None:
            freqs = phase_freqs

        self.probs = probs
        self.phase_freqs = phase_freqs
        degeneracies = probs * self.hilbert_dim

        # Build observed signal
        iterator = tqdm(  # type: ignore
            zip(phase_freqs, degeneracies, strict=True),
            total=len(phase_freqs),
            desc="Building observed signal",
            disable=not (self.verbose and TQDM_AVAILABLE),
        )

        for phi, d in iterator:
            delta = (sample_bins - phi + 0.5) % 1.0 - 0.5
            observed_data += d * self.kernel_fn(delta)

        return sample_bins, observed_data

    def _build_system_matrix(
        self,
        sample_bins: np.ndarray,
        candidate_phases: np.ndarray,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Build the system matrix A for deconvolution."""
        sample_bins = np.asarray(sample_bins).reshape(-1)
        candidate_phases = np.asarray(candidate_phases).reshape(-1)

        n_samples = sample_bins.size
        n_candidates = candidate_phases.size

        A = np.zeros((n_samples, n_candidates))

        iterator = tqdm(  # type: ignore
            range(n_samples),
            desc="Building matrix",
            disable=not (show_progress and self.verbose and TQDM_AVAILABLE),
        )

        for i in iterator:
            delta = (sample_bins[i] - candidate_phases + 0.5) % 1.0 - 0.5
            A[i, :] = self.kernel_fn(delta)

        return A

    def _optimize_l2_sparse(
        self,
        sample_bins: np.ndarray,
        observed_data: np.ndarray,
        candidate_phases: np.ndarray,
        l1_penalty: float = 1e-1,
        threshold: float = 1e-1,
        show_details: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """L2 residual minimization with L1 sparsity penalty."""
        if show_details:
            self._print(
                f"\nL2-Sparse: {len(candidate_phases)} candidates, λ={l1_penalty}, threshold={threshold}"
            )

        A = self._build_system_matrix(sample_bins, candidate_phases, show_progress=show_details)
        n_candidates = len(candidate_phases)

        # Scale the penalty by the data scale for better numerical behavior
        data_scale = np.linalg.norm(observed_data)
        scaled_l1_penalty = l1_penalty * data_scale / self.hilbert_dim

        weights = cp.Variable(n_candidates, nonneg=True)  # type: ignore
        residual = A @ weights - observed_data
        sparsity_penalty = scaled_l1_penalty * cp.norm1(weights)  # type: ignore
        objective = cp.Minimize(cp.sum_squares(residual) + sparsity_penalty)  # type: ignore
        constraints = [cp.sum(weights) == self.hilbert_dim]  # type: ignore

        problem = cp.Problem(objective, constraints)  # type: ignore
        problem.solve(solver=self.solver, verbose=False)

        self.optimization_info["status"] = problem.status
        self.optimization_info["objective_value"] = problem.value
        self.optimization_info["data_term"] = cp.sum_squares(residual).value  # type: ignore
        self.optimization_info["sparsity_term"] = sparsity_penalty.value

        weights_optim = cast(np.ndarray, weights.value)
        if weights_optim is None:
            raise RuntimeError("Optimization failed: weights.value is None")
        significant_indices = np.where(weights_optim > threshold)[0]

        if show_details:
            n_nonzero = np.sum(weights_optim > 1e-6)
            self._print(f"  Status: {problem.status}, Objective: {problem.value:.4e}")
            self._print(
                f"  Data term: {self.optimization_info['data_term']:.4e}, "
                f"Sparsity term: {self.optimization_info['sparsity_term']:.4e}"
            )
            self._print(
                f"  Non-zero weights: {n_nonzero}/{n_candidates}, "
                f"Significant: {len(significant_indices)}/{n_candidates}, "
                f"Total deg: {np.sum(weights_optim):.2f}"
            )

        return (
            candidate_phases[significant_indices],
            weights_optim[significant_indices],
            weights_optim,
        )

    def _optimize_l1_sparse(
        self,
        sample_bins: np.ndarray,
        observed_data: np.ndarray,
        candidate_phases: np.ndarray,
        l1_penalty: float = 1e-1,
        threshold: float = 1e-1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """L1 residual minimization with L1 sparsity penalty."""
        self._print(
            f"\nL1-Sparse: {len(candidate_phases)} candidates, λ={l1_penalty}, threshold={threshold}"
        )

        A = self._build_system_matrix(sample_bins, candidate_phases, show_progress=True)
        n_samples = len(sample_bins)
        n_candidates = len(candidate_phases)

        # Scale the penalty by the data scale
        data_scale = np.sum(np.abs(observed_data))
        scaled_l1_penalty = l1_penalty * data_scale / self.hilbert_dim

        weights = cp.Variable(n_candidates, nonneg=True)  # type: ignore
        residual = A @ weights - observed_data
        abs_residual = cp.Variable(n_samples)  # type: ignore

        constraints = [
            cp.sum(weights) == self.hilbert_dim,  # type: ignore
            abs_residual >= residual,
            abs_residual >= -residual,
        ]

        objective = cp.Minimize(  # type: ignore
            cp.sum(abs_residual) + scaled_l1_penalty * cp.norm1(weights)  # type: ignore
        )
        problem = cp.Problem(objective, constraints)  # type: ignore
        problem.solve(solver=self.solver, verbose=False)

        self.optimization_info["status"] = problem.status
        self.optimization_info["objective_value"] = problem.value

        weights_optim = cast(np.ndarray, weights.value)
        if weights_optim is None:
            raise RuntimeError("Optimization failed: weights.value is None")
        significant_indices = np.where(weights_optim > threshold)[0]

        n_nonzero = np.sum(weights_optim > 1e-6)
        self._print(f"  Status: {problem.status}, Objective: {problem.value:.4e}")
        self._print(
            f"  Non-zero weights: {n_nonzero}/{n_candidates}, "
            f"Significant: {len(significant_indices)}/{n_candidates}, "
            f"Total deg: {np.sum(weights_optim):.2f}"
        )

        return (
            candidate_phases[significant_indices],
            weights_optim[significant_indices],
            weights_optim,
        )

    def _optimize_relaxed(
        self,
        sample_bins: np.ndarray,
        observed_data: np.ndarray,
        candidate_phases: np.ndarray,
        l1_penalty: float = 1e-1,
        constraint_penalty: float = 10.0,
        threshold: float = 1e-1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Relaxed constraint optimization."""
        self._print(
            f"\nRelaxed: {len(candidate_phases)} candidates, "
            f"λ_L1={l1_penalty}, λ_constraint={constraint_penalty}, threshold={threshold}"
        )

        A = self._build_system_matrix(sample_bins, candidate_phases, show_progress=True)
        n_candidates = len(candidate_phases)

        weights = cp.Variable(n_candidates, nonneg=True)  # type: ignore
        residual = A @ weights - observed_data

        data_scale = np.linalg.norm(observed_data)
        sparsity_penalty = l1_penalty * data_scale / self.hilbert_dim * cp.norm1(weights)  # type: ignore
        constraint_violation = (
            constraint_penalty * data_scale * cp.square(cp.sum(weights) - self.hilbert_dim)  # type: ignore
        )

        objective = cp.Minimize(  # type: ignore
            cp.sum_squares(residual) + sparsity_penalty + constraint_violation  # type: ignore
        )
        constraints = [cp.sum(weights) <= self.hilbert_dim * 1.1]  # type: ignore

        problem = cp.Problem(objective, constraints)  # type: ignore
        problem.solve(solver=self.solver, verbose=False)

        self.optimization_info["status"] = problem.status
        self.optimization_info["objective_value"] = problem.value

        weights_optim = cast(np.ndarray, weights.value)
        if weights_optim is None:
            raise RuntimeError("Optimization failed: weights.value is None")
        significant_indices = np.where(weights_optim > threshold)[0]

        n_nonzero = np.sum(weights_optim > 1e-6)
        self._print(f"  Status: {problem.status}, Objective: {problem.value:.4e}")
        self._print(
            f"  Non-zero weights: {n_nonzero}/{n_candidates}, "
            f"Significant: {len(significant_indices)}/{n_candidates}, "
            f"Total deg: {np.sum(weights_optim):.2f}"
        )

        return (
            candidate_phases[significant_indices],
            weights_optim[significant_indices],
            weights_optim,
        )

    def _refine_grid(
        self,
        grid: np.ndarray,
        weights: np.ndarray,
        threshold_ratio: float = 1e-2,
        zoom_width: float = 0.02,
        points_per_peak: int = 30,
    ) -> np.ndarray:
        """Refine grid around significant peaks."""
        max_w = np.max(weights)
        support_indices = np.where(weights > max_w * threshold_ratio)[0]
        refined: list[float] = []

        for idx in support_indices:
            center = grid[idx]
            start = (center - zoom_width / 2) % 1
            end = (center + zoom_width / 2) % 1

            if start < end:
                local_grid = np.linspace(start, end, points_per_peak, endpoint=False)
            else:
                local_grid = np.sort(
                    np.concatenate(
                        [
                            np.linspace(start, 1, points_per_peak // 2, endpoint=False),
                            np.linspace(0, end, points_per_peak // 2, endpoint=False),
                        ]
                    )
                )
            refined.extend(local_grid)

        return np.unique(refined)

    def _optimize_adaptive_grid(
        self,
        sample_bins: np.ndarray,
        observed_data: np.ndarray,
        initial_candidates: int = 100,
        n_refinements: int = 3,
        l1_penalty: float = 1.0,
        threshold: float = 0.05,
        zoom_width: float = 0.02,
        points_per_peak: int = 30,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Adaptive grid refinement optimization."""

        phase_grid = self._get_candidate_phases(initial_candidates, threshold)
        if self.peak_guess == "find_peaks":
            self._print(
                f"\nAdaptive Grid: {n_refinements} refinements, "
                f"initial candidates={len(phase_grid)} found by find_peaks, λ={l1_penalty}"
            )
        else:
            self._print(
                f"\nAdaptive Grid: {n_refinements} refinements, "
                f"initial candidates={initial_candidates}, λ={l1_penalty}"
            )

        weights: np.ndarray = np.array([])

        refinement_iterator = tqdm(  # type: ignore
            range(n_refinements),
            desc="Refinement iterations",
            disable=not (self.verbose and TQDM_AVAILABLE),
        )

        for i in refinement_iterator:
            show_details = i == 0 or i == n_refinements - 1  # Only show first and last

            phi_est, deg_est, weights = self._optimize_l2_sparse(
                sample_bins,
                observed_data,
                phase_grid,
                l1_penalty,
                threshold=0,
                show_details=show_details,
            )

            n_peaks = np.sum(weights > np.max(weights) * 1e-2)
            self.optimization_info[f"refinement_{i + 1}_grid_size"] = len(phase_grid)
            self.optimization_info[f"refinement_{i + 1}_peaks"] = n_peaks

            if self.verbose and not show_details:
                self._print(
                    f"  Refinement {i + 1}/{n_refinements}: grid={len(phase_grid)}, peaks={n_peaks}"
                )

            if i < n_refinements - 1:
                phase_grid = self._refine_grid(
                    phase_grid,
                    weights,
                    zoom_width=zoom_width,
                    points_per_peak=points_per_peak,
                )

        # Final thresholding
        significant_indices = np.where(weights > threshold)[0]
        self._print(
            f"  Final: {len(significant_indices)} significant phases (threshold={threshold})"
        )

        return phase_grid[significant_indices], weights[significant_indices], weights

    def _grid_probs(self, probs: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Return the probabilities ordered on the full uniform 2^M phase grid.

        ESPRIT works on the moment sequence of the distribution and therefore
        needs the probabilities sampled on the complete uniform grid
        ``{k / 2^M : k = 0 .. 2^M − 1}`` (the DOS-QPE ancilla outcomes), sorted by
        frequency.  Raises if the data is not on that grid (e.g. a sparse synthetic
        distribution) — in that case use a convex mode instead.
        """
        n = 2**self.n_ancilla
        freqs = np.asarray(freqs, dtype=float)
        probs = np.asarray(probs, dtype=float)
        if len(probs) != n:
            raise ValueError(
                f"ESPRIT needs the distribution on the full uniform grid of "
                f"{n} = 2^{self.n_ancilla} points, got {len(probs)}. Pass "
                "freqs=dosqpe.freqs (the standard DOS-QPE workflow), or use a convex mode."
            )
        order = np.argsort(freqs)
        expected = np.arange(n) / n
        if not np.allclose(freqs[order], expected, atol=1e-9):
            raise ValueError(
                "ESPRIT needs probabilities on the uniform grid k/2^M; the provided "
                "frequencies are not that grid. Use a convex mode for off-grid data."
            )
        return probs[order]

    def _esprit_model_order(self, singular_values: np.ndarray, rel_tol: float) -> int:
        """Pick the number of eigenphases from the Hankel singular-value spectrum.

        The count of singular values above ``rel_tol × σ_max`` separates the signal
        subspace from the noise floor.  Clamped to ``[1, len(σ) − 1]`` so the
        rotational-invariance shift always has a non-empty subspace to act on.
        """
        s = np.asarray(singular_values, dtype=float)
        if s.size == 0 or s[0] == 0:
            return 1
        r = int(np.sum(s > rel_tol * s[0]))
        return int(np.clip(r, 1, s.size - 1))

    def _window_deconvolved_moments(self, grid_probs: np.ndarray, n_moments: int) -> np.ndarray:
        """Pure-exponential moments ``c_t = Σ_j (d_j/D) e^{2πiθ_j t}``, t = 0..n_moments-1.

        The inverse-DFT of the distribution gives the moments tapered by the
        triangular Fejér window; dividing the window out leaves a clean sum of
        complex exponentials (whose frequencies are the eigenphases).  The lag
        range is kept short enough that the taper stays well-conditioned.
        """
        n = len(grid_probs)
        c = np.fft.ifft(grid_probs) * n  # c[t] = Σ_k p_k e^{+2πi k t / n}
        t = np.arange(n_moments)
        return c[:n_moments] / (1.0 - t / n)

    @staticmethod
    def _pencil_phases(moments: np.ndarray, order: int) -> np.ndarray:
        """Eigenphases of a sum of exponentials via the matrix-pencil / ESPRIT shift.

        Builds a Hankel matrix from ``moments``, takes its rank-``order`` signal
        subspace, and reads the eigenphases off the rotational-invariance shift as
        ``angle(λ) / 2π``.  Off-grid by construction.
        """
        m = np.asarray(moments)
        n_m = len(m)
        n_cols = n_m // 2 + 1
        n_rows = n_m - n_cols + 1
        hankel = np.array([m[i : i + n_cols] for i in range(n_rows)])
        u, _, _ = np.linalg.svd(hankel)
        order = int(np.clip(order, 1, min(u.shape[1], n_cols) - 1)) if min(hankel.shape) > 1 else 1
        u_sig = u[:, :order]
        shift = np.linalg.pinv(u_sig[:-1]) @ u_sig[1:]
        return (np.angle(np.linalg.eigvals(shift)) / (2 * np.pi)) % 1.0

    def _default_n_moments(self, n: int) -> int:
        """Lag count for the moment-based estimators (parametric family)."""
        return int(np.clip(round(0.7 * n), 4, n))

    def _esprit_estimate(
        self,
        probs: np.ndarray,
        freqs: np.ndarray,
        model_order: Optional[int] = None,
        n_moments: Optional[int] = None,
        rel_tol: float = 1e-2,
        threshold: float = 1e-1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Off-grid eigenphase estimation via the matrix-pencil / ESPRIT method.

        Eigenphases come from a matrix pencil on the (window-deconvolved) moment
        sequence; the model order is taken from the Hankel singular-value spectrum.
        Degeneracies are a non-negative least-squares fit of the recovered phases
        against the measured distribution with the exact QPE kernel.  No cvxpy.

        Args:
            probs: Measured probabilities (any order; re-gridded internally).
            freqs: Frequencies for ``probs`` (the uniform 2^M grid).
            model_order: Number of eigenphases. If None, from the singular spectrum.
            n_moments: Number of moments (lags). If None, ~0.7·2^M.
            rel_tol: Relative singular-value threshold for automatic model order.
            threshold: Drop recovered phases whose degeneracy is below this value.

        Returns:
            (phases, degeneracies, degeneracies) — the third mirrors ``full_weights``.
        """
        p = self._grid_probs(probs, freqs)
        n_moments = (
            self._default_n_moments(len(p)) if n_moments is None else int(min(n_moments, len(p)))
        )
        moments = self._window_deconvolved_moments(p, n_moments)

        # Model order from the Hankel singular-value spectrum.
        n_cols = n_moments // 2 + 1
        n_rows = n_moments - n_cols + 1
        hankel = np.array([moments[i : i + n_cols] for i in range(n_rows)])
        s = np.linalg.svd(hankel, compute_uv=False)
        r = model_order if model_order is not None else self._esprit_model_order(s, rel_tol)
        self.optimization_info["esprit_singular_values"] = s
        self.optimization_info["esprit_model_order"] = r

        phases = self._pencil_phases(moments, r)
        degeneracies = self._fit_degeneracies(phases, p)

        self._print(
            f"\nESPRIT: model order {r} (of {n_moments} moments), "
            f"{int(np.sum(degeneracies > threshold))} phases above threshold."
        )
        keep = degeneracies > threshold
        return phases[keep], degeneracies[keep], degeneracies[keep]

    def _atomic_norm_estimate(
        self,
        probs: np.ndarray,
        freqs: np.ndarray,
        tau: float = 1.0,
        n_moments: Optional[int] = None,
        rel_tol: float = 1e-2,
        threshold: float = 1e-1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gridless eigenphase estimation via atomic-norm (total-variation) denoising.

        Solves the atomic-norm soft-thresholding (AST) semidefinite program to denoise
        the moment sequence under a sparse-spikes prior (Bhaskar–Tang–Recht).  The
        eigenphases are then read off the denoised moments with the matrix pencil, and
        the model order is taken from the rank of the recovered PSD Toeplitz matrix —
        so, unlike a fixed-order method, weak lines are not dropped by a hard cutoff.
        Degeneracies come from the same kernel NNLS fit as 'esprit'.  Requires cvxpy.

        Args:
            probs: Measured probabilities (any order; re-gridded internally).
            freqs: Frequencies for ``probs`` (the uniform 2^M grid).
            tau: AST regularisation. Larger → smoother / fewer lines. The moments are
                normalised (c_0 = 1), so the scale is dataset-independent; the default
                works across spectra and a poor value only loses the 'auto' selection.
            n_moments: Number of moments (the SDP is this size). If None, ~0.7·2^M.
            rel_tol: Relative eigenvalue threshold for the Toeplitz-rank model order.
            threshold: Drop recovered phases whose degeneracy is below this value.

        Returns:
            (phases, degeneracies, degeneracies) — the third mirrors ``full_weights``.
        """
        self._require_cvxpy("the 'atomic_norm' mode")
        p = self._grid_probs(probs, freqs)
        n_moments = (
            self._default_n_moments(len(p)) if n_moments is None else int(min(n_moments, len(p)))
        )
        y = self._window_deconvolved_moments(p, n_moments)
        n = len(y)

        # AST semidefinite program: minimise ½‖x−y‖² + τ·½(t + u₀) s.t. the bordered
        # Hermitian-Toeplitz matrix is PSD.  x is the denoised moment sequence.
        x = cp.Variable(n, complex=True)  # type: ignore
        u = cp.Variable(n, complex=True)  # type: ignore
        t_var = cp.Variable()  # type: ignore
        toeplitz = cp.bmat(  # type: ignore
            [[u[i - j] if i >= j else cp.conj(u[j - i]) for j in range(n)] for i in range(n)]
        )
        bordered = cp.bmat(  # type: ignore
            [
                [toeplitz, cp.reshape(x, (n, 1), order="C")],
                [cp.reshape(cp.conj(x), (1, n), order="C"), cp.reshape(t_var, (1, 1), order="C")],
            ]
        )
        data_fit = 0.5 * (cp.sum_squares(cp.real(x - y)) + cp.sum_squares(cp.imag(x - y)))  # type: ignore
        objective = cp.Minimize(data_fit + tau * 0.5 * (t_var + cp.real(u[0])))  # type: ignore
        cp.Problem(objective, [bordered >> 0, cp.imag(u[0]) == 0]).solve(solver=cp.SCS)  # type: ignore

        if x.value is None:
            raise RuntimeError("Atomic-norm SDP failed to solve.")
        denoised = np.asarray(x.value)

        # Model order = numerical rank of the recovered PSD Toeplitz matrix.
        u_val = np.asarray(u.value)
        toeplitz_val = np.array(
            [
                [u_val[i - j] if i >= j else np.conj(u_val[j - i]) for j in range(n)]
                for i in range(n)
            ]
        )
        eigs = np.linalg.eigvalsh(toeplitz_val).real[::-1]
        r = int(np.sum(eigs > rel_tol * eigs[0])) if eigs[0] > 0 else 1
        self.optimization_info["atomic_toeplitz_eigenvalues"] = eigs
        self.optimization_info["atomic_model_order"] = r

        phases = self._pencil_phases(denoised, r)
        degeneracies = self._fit_degeneracies(phases, p)

        self._print(
            f"\nAtomic-norm: Toeplitz rank {r} (τ={tau}), "
            f"{int(np.sum(degeneracies > threshold))} phases above threshold."
        )
        keep = degeneracies > threshold
        return phases[keep], degeneracies[keep], degeneracies[keep]

    def _fit_degeneracies(self, phases: np.ndarray, grid_probs: np.ndarray) -> np.ndarray:
        """Non-negative least-squares degeneracies for fixed phases.

        Solves ``min_{w≥0} ‖A w − p‖²`` with ``A[k, j] = K(φ_k − phase_j)`` the exact
        QPE kernel on the measured grid, then scales to the Hilbert dimension
        (degeneracies = D·w, since the probabilities sum to one).  Uses the correct
        single-kernel forward model and needs no cvxpy.
        """
        n = len(grid_probs)
        grid = np.arange(n) / n
        delta = (grid[:, None] - phases[None, :] + 0.5) % 1.0 - 0.5
        a_matrix = self.kernel_fn(delta.ravel()).reshape(delta.shape)
        weights, _ = nnls(a_matrix, grid_probs)
        return weights * self.hilbert_dim

    def _forward_distribution(
        self, phases: np.ndarray, degeneracies: np.ndarray, freqs: np.ndarray
    ) -> np.ndarray:
        """Predicted (kernel-broadened) distribution a reconstruction would produce.

        Pushes the estimated spectrum back through the QPE kernel on the ``freqs``
        grid and normalises, so it can be compared against the measured distribution
        on equal footing (both are kernel-broadened).
        """
        freqs = np.asarray(freqs, dtype=float)
        pred = np.zeros(len(freqs))
        for theta, d in zip(phases, degeneracies, strict=True):
            pred += d * self.kernel_fn((freqs - theta + 0.5) % 1.0 - 0.5)
        total = pred.sum()
        return pred / total if total > 0 else pred

    @staticmethod
    def _circular_wasserstein1(a: np.ndarray, b: np.ndarray) -> float:
        """1-Wasserstein distance between two distributions on the unit circle.

        Uses the closed form for the circular earth-mover distance (Werman et al.):
        the optimal cyclic offset is the median of the cumulative difference, so
        ``W₁ = (1/N) Σ_i |F_i − median(F)|`` with ``F = cumsum(a − b)``.  Both inputs
        are assumed sampled on the same uniform grid and normalised.
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        cdf_diff = np.cumsum(a - b)
        return float(np.sum(np.abs(cdf_diff - np.median(cdf_diff))) / len(a))

    def _debias_weights(
        self,
        sample_bins: np.ndarray,
        observed_data: np.ndarray,
        phases: np.ndarray,
    ) -> np.ndarray:
        """Re-fit amplitudes on a fixed support without the L1 penalty.

        The L1 term used during support selection shrinks the recovered weights
        (the usual LASSO amplitude bias).  Once the support (the significant
        phases) is fixed, re-solving the *unpenalised* non-negative least squares
        problem — keeping the ``sum == hilbert_dim`` constraint — removes that
        bias, so the degeneracies and their integer rounding are accurate.

        Args:
            sample_bins: Grid the fit is evaluated on (the measured frequencies).
            observed_data: Target signal on ``sample_bins`` (degeneracies = D·p).
            phases: The selected support to re-fit amplitudes for.

        Returns:
            Debiased non-negative weights aligned with ``phases``.  Falls back to
            uniform weights summing to the Hilbert dimension if the solve fails.
        """
        A = self._build_system_matrix(sample_bins, phases)
        weights = cp.Variable(len(phases), nonneg=True)  # type: ignore
        objective = cp.Minimize(cp.sum_squares(A @ weights - observed_data))  # type: ignore
        constraints = [cp.sum(weights) == self.hilbert_dim]  # type: ignore
        problem = cp.Problem(objective, constraints)  # type: ignore
        problem.solve(solver=self.solver, verbose=False)

        if weights.value is None:
            self._print("  Debias solve failed; keeping penalised weights.")
            return np.full(len(phases), self.hilbert_dim / max(len(phases), 1))
        self._print(f"  Debiased {len(phases)} weights (total deg: {np.sum(weights.value):.2f}).")
        return np.asarray(weights.value)

    def _round_degeneracies(self, degeneracies: np.ndarray, preserve_sum: bool) -> np.ndarray:
        """Round degeneracies to integers.

        Degeneracies are physically integers.  With ``preserve_sum=False`` each
        value is rounded independently.  With
        ``preserve_sum=True`` the largest-remainder method is used so the rounded
        integers still sum to the Hilbert dimension.
        """
        degs = np.asarray(degeneracies, dtype=float)
        if not preserve_sum or len(degs) == 0:
            return np.round(degs).astype(int)

        floored = np.floor(degs).astype(int)
        deficit = int(round(self.hilbert_dim - floored.sum()))
        frac = degs - floored
        if deficit > 0:
            # Hand the leftover units to the largest fractional parts.
            for i in np.argsort(frac)[::-1][:deficit]:
                floored[i] += 1
        elif deficit < 0:
            # Reclaim units from the smallest fractional parts, never going below 0.
            for i in np.argsort(frac):
                if deficit == 0:
                    break
                if floored[i] > 0:
                    floored[i] -= 1
                    deficit += 1
        return floored

    def cluster_estimates(
        self,
        estimated_phases: np.ndarray,
        estimated_degeneracies: np.ndarray,
        tolerance: Optional[float] = None,
        method: Literal["fixed", "gap"] = "gap",
        gap_threshold: float = 1.5,
        preserve_integer_sum: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Group nearby eigenphase estimates into clusters.

        Args:
            estimated_phases: Array of estimated phases
            estimated_degeneracies: Array of estimated degeneracies
            tolerance: Clustering tolerance for 'fixed' method (default: 1/(2^(n_ancilla+1)))
            method: Clustering method - 'fixed' (traditional) or 'gap' (adaptive, default)
            gap_threshold: For 'gap' method, merge if gap < threshold × median_gap (default: 1.5)
            preserve_integer_sum: If True, round the per-cluster degeneracies with the
                largest-remainder method so they still sum to the Hilbert dimension.
                If False (default) each cluster total is rounded independently.

        Returns:
            clustered_phases: Array of cluster centers
            clustered_degeneracies: Array of integer degeneracies per cluster
        """
        if len(estimated_phases) == 0:
            return np.array([]), np.array([])

        if len(estimated_phases) == 1:
            return estimated_phases.copy(), estimated_degeneracies.copy()

        sorted_indices = np.argsort(estimated_phases)
        phi_sorted = estimated_phases[sorted_indices]
        d_sorted = estimated_degeneracies[sorted_indices]

        if method == "fixed":
            if tolerance is None:
                tolerance = 1.0 / (2 ** (self.n_ancilla + 1))

            clustered_phases = []
            clustered_degeneracies = []

            i = 0
            while i < len(phi_sorted):
                cluster_phi_list = [phi_sorted[i]]
                cluster_deg_list = [d_sorted[i]]
                j = i + 1

                while j < len(phi_sorted) and abs(phi_sorted[j] - phi_sorted[i]) < tolerance:
                    cluster_phi_list.append(phi_sorted[j])
                    cluster_deg_list.append(d_sorted[j])
                    j += 1

                cluster_phi_arr = np.array(cluster_phi_list)
                cluster_deg_arr = np.array(cluster_deg_list)
                total_deg = np.sum(cluster_deg_arr)

                if total_deg > 0:
                    weighted_avg = np.sum(cluster_phi_arr * cluster_deg_arr) / total_deg
                else:
                    weighted_avg = np.mean(cluster_phi_arr)

                clustered_phases.append(weighted_avg)
                clustered_degeneracies.append(total_deg)
                i = j

            self._print(
                f"Clustering (fixed): {len(estimated_phases)} → {len(clustered_phases)} "
                f"(tolerance={tolerance:.6f})"
            )

        elif method == "gap":
            # Gap-based adaptive clustering
            # Compute gaps between consecutive sorted phases
            gaps = np.diff(phi_sorted)

            if len(gaps) == 0:
                return phi_sorted.copy(), d_sorted.copy()

            # Compute characteristic gap scale (use median to be robust to outliers)
            median_gap = np.median(gaps)

            # If all gaps are small, use default tolerance
            if median_gap < 1e-10:
                if tolerance is None:
                    tolerance = 1.0 / (2 ** (self.n_ancilla + 1))
                characteristic_gap = tolerance
            else:
                characteristic_gap = gap_threshold * median_gap

            # Identify merge boundaries (small gaps)
            merge_mask = gaps < characteristic_gap

            # Build clusters by grouping consecutive phases with small gaps
            clustered_phases = []
            clustered_degeneracies = []

            cluster_start = 0
            for i in range(len(merge_mask)):
                if not merge_mask[i]:  # Large gap - end current cluster
                    # Cluster from cluster_start to i (inclusive)
                    cluster_phi_arr = phi_sorted[cluster_start : i + 1]
                    cluster_deg_arr = d_sorted[cluster_start : i + 1]

                    total_deg = np.sum(cluster_deg_arr)
                    if total_deg > 0:
                        weighted_avg = np.sum(cluster_phi_arr * cluster_deg_arr) / total_deg
                    else:
                        weighted_avg = np.mean(cluster_phi_arr)

                    clustered_phases.append(weighted_avg)
                    clustered_degeneracies.append(total_deg)

                    cluster_start = i + 1

            cluster_phi_arr = phi_sorted[cluster_start:]
            cluster_deg_arr = d_sorted[cluster_start:]
            total_deg = np.sum(cluster_deg_arr)
            if total_deg > 0:
                weighted_avg = np.sum(cluster_phi_arr * cluster_deg_arr) / total_deg
            else:
                weighted_avg = np.mean(cluster_phi_arr)

            clustered_phases.append(weighted_avg)
            clustered_degeneracies.append(total_deg)

            self._print(
                f"Clustering (gap): {len(estimated_phases)} → {len(clustered_phases)} "
                f"(median_gap={median_gap:.6f}, threshold={characteristic_gap:.6f})"
            )

        else:
            raise ValueError(f"Unknown clustering method: {method}. Use 'fixed' or 'gap'.")

        self.clustered_phases = np.array(clustered_phases)
        self.clustered_degeneracies = self._round_degeneracies(
            np.asarray(clustered_degeneracies, dtype=float), preserve_integer_sum
        )

        return self.clustered_phases, self.clustered_degeneracies

    def _optimize_auto(
        self,
        distribution: Dict[Union[str, int, float, Tuple[int, ...]], float],
        auto_modes: Optional[Sequence[ModeName]] = None,
        **call_kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run several estimators and keep the best by a Wasserstein data-fit score.

        Each candidate mode is run on the same distribution; its reconstruction is pushed
        back through the kernel and compared to the measured distribution with the circular
        1-Wasserstein distance.  The lowest-distance reconstruction wins — so a method that
        drops a line (leaving unexplained measured mass) is automatically rejected.  A
        candidate that errors (e.g. ESPRIT on a non-grid distribution) is skipped.
        """
        modes = auto_modes if auto_modes is not None else self.DEFAULT_AUTO_MODES
        # Keep only runnable candidates: 'esprit' always, convex/SDP modes need cvxpy.
        modes = [
            m for m in modes if m != "auto" and (m not in self.CONVEX_MODES or CVXPY_AVAILABLE)
        ]
        if not modes:
            raise RuntimeError("No runnable 'auto' candidate modes (cvxpy unavailable?).")

        # Measured distribution on its grid, for scoring.
        self.prepare_observed_data(
            distribution, call_kwargs.get("n_bins"), call_kwargs.get("freqs")
        )
        grid = self.phase_freqs
        total = self.probs.sum()
        p_meas = self.probs / total if total > 0 else self.probs

        results: Dict[str, Any] = {}
        scores: Dict[str, float] = {}
        for m in modes:
            try:
                sub = type(self)(
                    n_qubits=self.n_qubits,
                    n_ancilla=self.n_ancilla,
                    hamming_weight=self.hamming_weight,
                    mode=m,
                    kernel_fn=self.kernel_fn,
                    peak_guess=self.peak_guess,
                    solver=self.solver,
                    verbose=False,  # keep candidates quiet; auto reports the scores itself
                )
                phases, degeneracies = sub.estimate(distribution, **call_kwargs)
                predicted = self._forward_distribution(phases, degeneracies, grid)
                score = self._circular_wasserstein1(predicted, p_meas)
                results[m] = (sub, np.asarray(phases), np.asarray(degeneracies))
                scores[m] = score
                self._print(f"  auto: {m:>11s} → Wasserstein {score:.5f} ({len(phases)} lines)")
            except Exception as exc:  # a candidate that can't run is simply skipped
                self._print(f"  auto: {m:>11s} → skipped ({type(exc).__name__}: {exc})")

        if not scores:
            raise RuntimeError("All 'auto' candidate modes failed.")

        best_mode = min(scores, key=lambda k: scores[k])
        sub, phases, degeneracies = results[best_mode]

        # Adopt the winning sub-estimator's results.
        self.estimated_phases = sub.estimated_phases
        self.estimated_degeneracies = sub.estimated_degeneracies
        self.full_weights = sub.full_weights
        self.clustered_phases = sub.clustered_phases
        self.clustered_degeneracies = sub.clustered_degeneracies
        self.optimization_info.update(sub.optimization_info)
        self.optimization_info["auto_selected_mode"] = best_mode
        self.optimization_info["auto_scores"] = scores
        self._print(f"\nAUTO selected '{best_mode}' (Wasserstein {scores[best_mode]:.5f}).")

        return phases, degeneracies

    def estimate(
        self,
        distribution: Dict[Union[str, int, float, Tuple[int, ...]], float],
        n_candidates: int = 200,
        n_bins: Optional[int] = None,
        cluster: bool = True,
        cluster_tolerance: Optional[float] = None,
        cluster_gap_threshold: float = 1.5,
        freqs: Optional[np.ndarray] = None,
        cluster_method: Literal["fixed", "gap"] = "gap",
        debias: bool = True,
        integer_degeneracies: bool = False,
        auto_modes: Optional[Sequence[ModeName]] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run spectral estimation in the configured mode.

        For 'auto' (default) several estimators are run and the one whose reconstruction
        best reproduces the measured distribution (circular Wasserstein) is returned.
        For 'esprit'/'atomic_norm' the eigenphases are recovered off-grid from the moment
        sequence and the degeneracies from a kernel NNLS fit.  For the grid-based convex
        modes the *measured* distribution is fit directly: ``Σ_c x_c · K(φ_k − cand_c) ≈
        D · p_k`` at the measured frequencies ``φ_k`` (``K`` the QPE/Fejér kernel, ``D`` the
        Hilbert dimension) — the single convolution the QPE measurement actually performs.

        Args:
            distribution: DOS-QPE distribution {outcome: probability}
            n_candidates: Number of candidate phases for the convex grid methods.
                         Only used when peak_guess='uniform'; ignored when peak_guess='find_peaks'
                         and for the parametric/gridless modes.
            n_bins: Number of bins for the (plotting-only) kernel-convolved signal
            cluster: Whether to apply post-processing clustering
            cluster_method: Clustering method - 'fixed' (traditional) or 'gap' (adaptive, default).
                Forced to 'fixed' for the off-grid modes ('esprit'/'atomic_norm').
            cluster_tolerance: Tolerance for clustering (default: 1/(2^(n_ancilla+1)))
            cluster_gap_threshold: Tolerance for 'gap' method, merge if gap < threshold × median_gap (default: 1.5)
            freqs: Pre-computed frequencies (optional). If None, will be extracted from distribution.
                The off-grid modes require the full uniform 2^n_ancilla grid (pass freqs=dosqpe.freqs).
            debias: If True (default), re-fit convex grid-mode amplitudes on the selected support
                without the L1 penalty so the degeneracies are not LASSO-shrunk. Ignored for
                'l2_relaxed', 'esprit' and 'atomic_norm' (already unbiased NNLS).
            integer_degeneracies: If True, round the final degeneracies to integers
                with the largest-remainder method so they sum to the Hilbert dimension.
            auto_modes: For mode='auto', the candidate estimators to try (default
                {esprit, l2, atomic_norm}, filtered to what the install supports).
            **kwargs: Mode-specific parameters. For 'esprit': model_order, n_moments.
                For 'atomic_norm': tau, n_moments. For the grid-based convex modes:
                l1_penalty, threshold, constraint_penalty, initial_candidates, n_refinements.

        Returns:
            phases: Array of estimated eigenphases
            degeneracies: Array of estimated degeneracies
        """
        if self.mode == "auto":
            return self._optimize_auto(
                distribution,
                n_candidates=n_candidates,
                n_bins=n_bins,
                cluster=cluster,
                cluster_tolerance=cluster_tolerance,
                cluster_gap_threshold=cluster_gap_threshold,
                freqs=freqs,
                cluster_method=cluster_method,
                debias=debias,
                integer_degeneracies=integer_degeneracies,
                auto_modes=auto_modes,
                **kwargs,
            )
        self._print("\n" + "=" * 70)
        self._print(f"SPECTRAL OPTIMIZATION: {self.mode.upper()}")
        self._print(
            f"System: {self.n_qubits} qubits | Ancilla: {self.n_ancilla} | "
            f"Hilbert dim: {self.hilbert_dim} | Solver: {self.solver}"
        )
        self._print("=" * 70)

        # Prepare data.  prepare_observed_data() also sets self.probs / self.phase_freqs
        # and returns a fine kernel-convolved signal used only for plotting.
        self.prepare_observed_data(distribution, n_bins, freqs)

        # Fit on the measured grid with a single QPE kernel (consistent forward
        # model).  Fitting against the fine convolved signal instead would smear
        # the already-convolved data with a second kernel (a K∗K vs K model
        # mismatch) and needlessly inflate the system matrix to ~100·2^M rows.
        fit_bins = self.phase_freqs
        fit_target = self.probs * self.hilbert_dim

        threshold = kwargs.get("threshold", 1e-1)

        # Run estimation based on mode
        if self.mode == "esprit":
            self.estimated_phases, self.estimated_degeneracies, self.full_weights = (
                self._esprit_estimate(
                    self.probs,
                    self.phase_freqs,
                    model_order=kwargs.get("model_order", None),
                    n_moments=kwargs.get("n_moments", None),
                    threshold=threshold,
                )
            )

        elif self.mode == "atomic_norm":
            self.estimated_phases, self.estimated_degeneracies, self.full_weights = (
                self._atomic_norm_estimate(
                    self.probs,
                    self.phase_freqs,
                    tau=kwargs.get("tau", 1.0),
                    n_moments=kwargs.get("n_moments", None),
                    threshold=threshold,
                )
            )

        elif self.mode == "l2":
            candidate_phases = self._get_candidate_phases(n_candidates, threshold)
            l1_penalty = kwargs.get("l1_penalty", 1.0)

            self.estimated_phases, self.estimated_degeneracies, self.full_weights = (
                self._optimize_l2_sparse(
                    fit_bins, fit_target, candidate_phases, l1_penalty, threshold
                )
            )

        elif self.mode == "l1":
            candidate_phases = self._get_candidate_phases(n_candidates, threshold)
            l1_penalty = kwargs.get("l1_penalty", 1.0)

            self.estimated_phases, self.estimated_degeneracies, self.full_weights = (
                self._optimize_l1_sparse(
                    fit_bins, fit_target, candidate_phases, l1_penalty, threshold
                )
            )

        elif self.mode == "l2_relaxed":
            candidate_phases = self._get_candidate_phases(n_candidates, threshold)
            l1_penalty = kwargs.get("l1_penalty", 1.0)
            constraint_penalty = kwargs.get("constraint_penalty", 10.0)

            self.estimated_phases, self.estimated_degeneracies, self.full_weights = (
                self._optimize_relaxed(
                    fit_bins,
                    fit_target,
                    candidate_phases,
                    l1_penalty,
                    constraint_penalty,
                    threshold,
                )
            )

        elif self.mode == "l2_adaptive":
            initial_candidates = kwargs.get("initial_candidates", 100)
            n_refinements = kwargs.get("n_refinements", 3)
            l1_penalty = kwargs.get("l1_penalty", 1.0)
            threshold = kwargs.get("threshold", 0.05)

            self.estimated_phases, self.estimated_degeneracies, self.full_weights = (
                self._optimize_adaptive_grid(
                    fit_bins,
                    fit_target,
                    initial_candidates,
                    n_refinements,
                    l1_penalty,
                    threshold,
                )
            )

        # Debias the grid-mode amplitudes on the selected support (remove LASSO
        # shrinkage).  Skipped for 'l2_relaxed' (soft sum constraint is intentional)
        # and the off-grid modes (degeneracies already come from an unpenalised NNLS).
        if (
            debias
            and self.mode not in ("l2_relaxed", "esprit", "atomic_norm")
            and self.estimated_phases is not None
            and len(self.estimated_phases) > 0
        ):
            self.estimated_degeneracies = self._debias_weights(
                fit_bins, fit_target, self.estimated_phases
            )

        # Apply clustering if requested
        if cluster:
            if self.estimated_phases is None or self.estimated_degeneracies is None:
                raise RuntimeError("Cannot cluster: estimated phases/degeneracies not available")
            # The off-grid modes already return distinct lines, so adaptive gap
            # clustering (sized to the spacing between lines) would wrongly fuse
            # genuine eigenvalues.  Only collapse near-duplicates from an
            # over-specified model order via a tight fixed tolerance.
            method: Literal["fixed", "gap"] = (
                "fixed" if self.mode in ("esprit", "atomic_norm") else cluster_method
            )
            self.clustered_phases, self.clustered_degeneracies = self.cluster_estimates(
                self.estimated_phases,
                self.estimated_degeneracies,
                cluster_tolerance,
                method=method,
                gap_threshold=cluster_gap_threshold,
                preserve_integer_sum=integer_degeneracies,
            )

            self._print("\n" + "=" * 70)
            self._print(
                f"COMPLETE: {len(self.clustered_phases)} clustered phases | "
                f"Total degeneracy: {np.sum(self.clustered_degeneracies)}"
            )
            self._print("=" * 70 + "\n")

            return self.clustered_phases, self.clustered_degeneracies
        else:
            if self.estimated_phases is None or self.estimated_degeneracies is None:
                raise RuntimeError(
                    "Optimization failed: estimated phases/degeneracies not available"
                )
            if integer_degeneracies:
                self.estimated_degeneracies = self._round_degeneracies(
                    self.estimated_degeneracies, preserve_sum=True
                )
            self._print("\n" + "=" * 70)
            self._print(
                f"COMPLETE: {len(self.estimated_phases)} phases | "
                f"Total degeneracy: {np.sum(self.estimated_degeneracies):.2f}"
            )
            self._print("=" * 70 + "\n")

            return self.estimated_phases, self.estimated_degeneracies

    def plot(
        self,
        distribution: Dict[Union[str, int, float, Tuple[int, ...]], float],
        true_phases: Optional[np.ndarray] = None,
        true_degeneracies: Optional[np.ndarray] = None,
        n_bins: Optional[int] = None,
        freqs: Optional[np.ndarray] = None,
        show_deconvolved: bool = False,
        figsize: Tuple[int, int] = (10, 5),
        return_fig: bool = False,
    ) -> Optional[Tuple[Any, Any]]:
        """
        import matplotlib.pyplot as plt  # deferred: ~0.2 s, plotting only

        Plot optimization results.

        Args:
            distribution: Original DOS-QPE distribution
            true_phases: True eigenphases for comparison (optional)
            true_degeneracies: True degeneracies for comparison (optional)
            n_bins: Number of bins for plotting
            freqs: Pre-computed frequencies (optional)
            show_deconvolved: If True, show kernel-convolved distribution in addition to raw data (default: False)
            figsize: Figure size
            return_fig: If True, return (fig, ax)
        """
        if self.estimated_phases is None:
            raise ValueError("No results to plot. Run estimate() first.")

        import matplotlib.pyplot as plt  # deferred: ~0.2 s of import, plotting only

        sample_bins, observed_data = self.prepare_observed_data(distribution, n_bins, freqs)

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Plot raw data (probabilities at measured frequencies)
        # phase_freqs and probs are set by prepare_observed_data
        raw_degeneracies = self.probs * self.hilbert_dim
        # Only plot non-zero probabilities
        nonzero_mask = raw_degeneracies > 1e-10
        if np.any(nonzero_mask):
            ax.bar(
                self.phase_freqs[nonzero_mask],
                raw_degeneracies[nonzero_mask],
                width=1 / (2 ** (self.n_ancilla + 2)),
                alpha=0.3,
                label="Raw data",
                color="C0",
            )

        # Optionally plot kernel-convolved distribution
        if show_deconvolved:
            # Mask near-zero values to avoid vertical line artifacts
            # when connecting zeros to peaks
            plot_data = observed_data.copy()
            threshold = np.max(observed_data) * 1e-5
            plot_data[plot_data < threshold] = np.nan
            ax.plot(
                sample_bins,
                plot_data,
                "-",
                alpha=0.6,
                linewidth=1.5,
                label="Kernel-convolved data",
                color="C3",
            )

        # Plot true spectrum if provided
        if true_phases is not None and true_degeneracies is not None:
            # Scale true degeneracies to Hilbert dimension if they are normalized probabilities
            if np.max(true_degeneracies) <= 1.0:
                scaled_true_degeneracies = np.array(true_degeneracies) * self.hilbert_dim
            else:
                scaled_true_degeneracies = true_degeneracies

            for i in range(len(true_phases)):
                ax.plot(
                    [true_phases[i], true_phases[i]],
                    [0, scaled_true_degeneracies[i]],
                    "-",
                    lw=4,
                    c="C2",
                    zorder=0,
                    alpha=0.7,
                    label="True spectrum" if i == 0 else "",
                )

        # Plot estimated phases
        if self.estimated_phases is None or self.estimated_degeneracies is None:
            raise RuntimeError("Cannot plot: no optimization results available")
        ax.scatter(
            self.estimated_phases,
            self.estimated_degeneracies,
            color="gray",
            marker="x",
            s=100,
            label="Estimated phases",
            zorder=5,
        )

        # Plot clustered results if available
        if self.clustered_phases is not None:
            if self.clustered_degeneracies is None:
                raise RuntimeError(
                    "Inconsistent state: clustered_phases exists but clustered_degeneracies is None"
                )
            ax.scatter(
                self.clustered_phases,
                self.clustered_degeneracies,
                color="red",
                marker="*",
                s=200,
                label="Clustered phases",
                zorder=10,
            )

        ax.set_xlabel("Eigenphase")
        ax.set_ylabel("Degeneracy")
        ax.set_title(f"Spectral Deconvolution ({self.mode})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        if return_fig:
            return fig, ax
        else:
            plt.show()
            return None
