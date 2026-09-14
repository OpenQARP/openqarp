from typing import Callable, List, Literal, Optional, Tuple, Union, cast

import numpy as np
import scipy
from scipy.integrate import quad

from qarp.operators import QubitOperator

from ..._types import Shots
from ...blocks import AnyBlock
from ...engines import Engine
from ...engines._runnable import Runnable
from ...errors import CapabilityError
from .._primitives import PrimitiveAlgorithm, StateVector
from .composite_algorithm import CompositeAlgorithm


def get_overlaps(
    trial_state: AnyBlock, hamiltonian: QubitOperator, return_eigenvalues: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Compute overlaps of trial state with eigenstates of the Hamiltonian.

    Args:
        trial_state: Block representing the trial state.
        hamiltonian: QubitOperator representing the Hamiltonian.
        return_eigenvalues: If True, also return the eigenvalues.

    Returns:
        Array of overlap probabilities with each eigenstate, sorted by ascending
        eigenvalue. If ``return_eigenvalues`` is True, returns a
        ``(overlaps, eigenvalues)`` tuple instead.

    Note:
        Uses np.linalg.eigh which returns eigenvectors as COLUMNS of the matrix.
        The overlaps are computed correctly by iterating over columns.
    """
    import qarpx as qx

    # Classical utility: a fresh simulator is the point (no engine involved).
    n_qubits = trial_state.n_qubits
    trial_state.build()
    trial_sv = np.asarray(qx.QarpSimulator().statevector(trial_state.flatten(), n_qubits)).flatten()
    return _overlaps_from_statevector(trial_sv, hamiltonian, n_qubits, return_eigenvalues)


def _overlaps_from_statevector(
    trial_sv: np.ndarray,
    hamiltonian: QubitOperator,
    n_qubits: int,
    return_eigenvalues: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """``|⟨v_k|ψ⟩|²`` over the dense spectrum of ``hamiltonian`` — O(4^n)."""
    # LSB — eigenvectors must share the qarpx statevector basis for the
    # p/λ table to match the exp(-iHt) HadamardTest measurements.
    ham = hamiltonian.sparse_matrix(n_qubits).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(ham)
    overlaps = np.array(
        [
            np.abs(np.vdot(eigenvectors[:, i].flatten(), trial_sv)) ** 2
            for i in range(eigenvectors.shape[1])
        ]
    )
    if return_eigenvalues:
        return overlaps, eigenvalues
    return overlaps


def truncated_gaussian_density(
    t: Union[float, np.ndarray], sigma: float, T: float
) -> Union[float, np.ndarray]:
    """
    Compute the truncated Gaussian density function a(t) as defined in equation (6) of arxiv2402.01013.

    Args:
        t: Input value(s) where to evaluate the density function.
        sigma: Truncation parameter (level of truncation).
        T: Time parameter (time window).

    Returns:
        Value(s) of the truncated Gaussian density function a(t).
    """
    t = np.asarray(t)

    # Handle the case when sigma = infinity (no truncation)
    if np.isinf(sigma):
        # When σ -> infinity, we get F(x) = exp(-T*x^2/2)
        return np.exp(-T * t**2 / 2)

    def integrand(s):
        return (1 / np.sqrt(2 * np.pi) / T) * np.exp(-(s**2) / (2 * T**2))

    integral_result, _ = quad(integrand, -sigma * T, sigma * T)

    # Dirac delta function contribution at t=0 - approx with a narrow Gaussian
    delta_t = np.zeros_like(t)
    # Only add delta contribution if t contains 0 (within numerical precision)
    if np.any(np.abs(t) < 1e-10):
        delta_indices = np.abs(t) < 1e-10
        delta_t[delta_indices] = 1.0

    first_term = (1 - integral_result) * delta_t

    second_term_coeff = 1 / np.sqrt(2 * np.pi) / T
    second_term_exp = np.exp(-(t**2) / (2 * T**2))
    indicator = np.logical_and(t >= -sigma * T, t <= sigma * T).astype(float)
    second_term = second_term_coeff * second_term_exp * indicator

    return first_term + second_term


class QMEGS(CompositeAlgorithm):
    """Quantum Multiple Eigenvalue Gaussian filtered Search (QMEGS) algorithm.

    Implementation based on arxiv2402.01013. This algorithm finds multiple eigenphases
    of a unitary operator using Hadamard tests with Gaussian filtered time sampling.

    Args:
        unitary: Block representing the time-evolution unitary.
        state: Block representing the trial state.
        n_shots: Number of measurement shots — same meaning in both modes:
            an int adds shot noise (synthetic in "analytical" mode, sampled in
            "sampling" mode with a sampling primitive); ``qarp.EXACT`` means no
            shot noise; ``None`` means the default shot count (10_000 /
            engine default).  The default exact primitive ignores it in
            "sampling" mode.
        target_indices: List of indices of target eigenvalues to find (Dset in the paper).
        sigma: Truncation level for time sampling.
        eta: Error tolerance for the algorithm.
        T: Time window for time sampling.
        filtering_function: Custom filtering function for time sampling.
            Defaults to truncated_gaussian_density.
        mode_dataset: Dataset generation mode.
            - "analytical": Generate data using classical simulation (faster, for testing).
            - "sampling": Generate data using actual sampling execution.
        mode_time: Time sampling mode.
            - "rvs": Use scipy's truncnorm.rvs for fast sampling.
            - "rejection_sampling": Use rejection sampling (enables arbitrary filtering functions).
        overlaps: Where the algorithm's a-priori inputs ``p_min`` / ``p_tail``
            (Theorem 1 of arXiv:2402.01013) come from.
            - ``"classical"`` (default): ``build()`` dense-diagonalises the
              Hamiltonian (O(4^n)) and reads the trial statevector through
              the engine's simulator — a validation harness, refused on a
              noisy or routed engine.  Required by ``mode_dataset="analytical"``.
            - ``(pmin, ptail)``: the two overlaps supplied by the user; no
              diagonalisation, no statevector.  Only ``len(target_indices)``
              is consulted (the number of eigenvalues to extract) — the index
              values themselves are not used.
        verbose: Whether to print progress information.
        primitive: Primitive used to evaluate ⟨ψ|U(t)|ψ⟩ in "sampling" mode.
            Defaults to ``StateVector()`` — exact, noise-free amplitudes.
            Pass ``HadamardTest()`` for finite-shot protocol realism.
        engine: Quantum engine for circuit execution.
    """

    def __init__(
        self,
        unitary: AnyBlock,
        state: AnyBlock,
        n_shots: Union[int, Shots, None],
        target_indices: List[int],  # target eigenvalue indices (Dset in the paper)
        sigma: float = 1.0,  # truncation level
        eta: float = 0.01,  # error tolerance
        T: int = 100,  # time window
        filtering_function: Optional[Callable] = None,
        mode_dataset: Literal["analytical", "sampling"] = "analytical",
        mode_time: Literal["rvs", "rejection_sampling"] = "rvs",
        overlaps: Union[Literal["classical"], Tuple[float, float]] = "classical",
        verbose: bool = False,
        primitive: Optional[PrimitiveAlgorithm] = None,
        engine: Optional[Engine] = None,
    ):
        if primitive is None:
            primitive = StateVector()
        super().__init__(primitive=primitive, engine=engine)
        self.overlaps = overlaps

        # Core components
        self.unitary = unitary
        self.state = state
        self.n_qubits = self.unitary.n_qubits
        self.n_shots = n_shots
        self.target_indices = target_indices  # target eigenvalue indices
        self.verbose = verbose

        # Mode settings
        self.mode_dataset = mode_dataset
        self.mode_time = mode_time

        # Set filtering function
        self.filtering_function = filtering_function or truncated_gaussian_density

        # Algorithm parameters (will be computed in build)
        self.p = None  # overlap probabilities with each eigenstate
        self.eigenvalues = None  # Store eigenvalues for consistent ordering
        self.pmin = None  # minimum overlap with target eigenstates
        self.ptail = None  # total probability in non-target eigenstates
        self.alpha = None  # exclusion radius factor
        self.q = None  # grid spacing factor
        self.sigma = sigma  # truncation level
        self.eta = eta  # error tolerance
        self._T_initial = T  # store initial time window for reset
        self.T = T  # time window (may be updated in run)
        self.n_samples = None

        # Results storage
        self.dataset = None
        self.res = None
        self.x_to_plot: List[float] = []
        self.y_to_plot: List[complex] = []

        self.validate_inputs()

    def validate_inputs(self):
        """Validate input parameters for correctness.

        Raises:
            ValueError: If any input parameter is invalid.
        """
        if self.n_qubits is None:
            raise ValueError("unitary.n_qubits must not be None")
        if not isinstance(self.target_indices, list) or not all(
            isinstance(i, int) and 0 <= i < 2**self.n_qubits for i in self.target_indices
        ):
            raise ValueError(
                f"target_indices must be a list of valid integer indices within [0, {2**self.n_qubits - 1}]"
            )
        if not hasattr(self.unitary, "operator") or not hasattr(self.unitary, "set_time"):
            raise ValueError("unitary must have 'operator' attribute and 'set_time' method.")
        if not isinstance(self.unitary.operator, QubitOperator):  # type: ignore[attr-defined]
            raise ValueError("unitary.operator must be a QubitOperator.")
        # isinstance, not ``!= "classical"``: a numpy pair makes ``!=``
        # return an array and the truth test raises before validation.
        if isinstance(self.overlaps, str):
            if self.overlaps != "classical":
                raise ValueError(
                    "overlaps must be 'classical' or a (pmin, ptail) pair of floats; "
                    f"got {self.overlaps!r}"
                )
        else:
            try:
                pmin, ptail = (float(v) for v in self.overlaps)  # type: ignore[union-attr]
            except (TypeError, ValueError):
                raise ValueError(
                    "overlaps must be 'classical' or a (pmin, ptail) pair of floats; "
                    f"got {self.overlaps!r}"
                ) from None
            if not (np.isfinite(pmin) and np.isfinite(ptail)):
                raise ValueError(f"overlaps must be finite; got ({pmin}, {ptail})")
            if not (0.0 <= ptail < pmin <= 1.0) or pmin + ptail > 1.0:
                raise ValueError(
                    "overlaps=(pmin, ptail) needs 0 <= ptail < pmin <= 1 and "
                    f"pmin + ptail <= 1; got ({pmin}, {ptail})"
                )
            if self.mode_dataset == "analytical":
                raise ValueError(
                    "mode_dataset='analytical' simulates Σ p_k e^{-iλ_k t} classically "
                    "and needs overlaps='classical'; use mode_dataset='sampling' with "
                    "user-supplied overlaps."
                )

    def build(self):
        """Build the algorithm by computing necessary parameters.

        Calculates overlaps, probabilities, and algorithm parameters based on
        the input unitary and trial state.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If the algorithm's assumptions are not satisfied
                (pmin must be greater than ptail).
        """
        if self.state.n_qubits is None:
            raise ValueError("state.n_qubits must not be None")

        if isinstance(self.overlaps, str):
            # Classical validation step: O(4^n) diagonalisation plus one exact
            # trial statevector.  The statevector goes through the shared
            # amplitude gate so a noisy or routed engine refuses instead of
            # silently reporting ideal overlaps.
            # _amplitude_simulator below would raise too; this check exists
            # only to name the overlaps=(pmin, ptail) remedy in the message.
            if not self._amplitudes_available(check_primitive=False):
                raise CapabilityError(
                    "QMEGS overlaps='classical' needs an exact trial statevector "
                    f"from the engine, but {type(self.engine).__name__} cannot "
                    "provide one (noise enabled or routed device).  Pass "
                    "overlaps=(pmin, ptail) instead."
                )
            n_qubits = self.state.n_qubits
            self.state.build()
            sim = self._amplitude_simulator(n_qubits, check_primitive=False)
            trial_sv = np.asarray(sim.statevector(self.state.flatten(), n_qubits)).flatten()
            # non_target_indices: complement set (Dc in the paper)
            self.non_target_indices = [
                i for i in range(2**n_qubits) if i not in self.target_indices
            ]
            # self.p[i] always corresponds to self.eigenvalues[i]
            self.p, self.eigenvalues = _overlaps_from_statevector(
                trial_sv, self.unitary.operator, n_qubits, return_eigenvalues=True
            )
            # pmin: minimum overlap with target eigenstates
            self.pmin = np.min([self.p[i] for i in self.target_indices])
            # ptail: total probability in non-target eigenstates
            self.ptail = sum([self.p[i] for i in self.non_target_indices])
            if self.verbose:
                print(f"Overlaps: {self.p}")
        else:
            # User-supplied a-priori inputs (Theorem 1): no spectrum is computed.
            self.p = None
            self.eigenvalues = None
            self.non_target_indices = []
            self.pmin, self.ptail = (float(v) for v in self.overlaps)

        if self.verbose:
            print(f"p_min: {self.pmin}")
            print(f"p_tail: {self.ptail}")

        # Validate algorithm assumptions
        if self.pmin <= self.ptail:
            raise ValueError(
                f"Algorithm assumption violated: pmin ({self.pmin:.6f}) must be greater than "
                f"ptail ({self.ptail:.6f}). This means the trial state does not have sufficient "
                f"overlap with the target eigenstates in target_indices={self.target_indices}. "
                f"Consider choosing different target indices or a better trial state. "
                f"Overlaps: {self.p}"
            )

        # alpha: exclusion radius factor = sqrt(log(1/(pmin - ptail)))
        self.alpha = np.sqrt(np.log(1 / (self.pmin - self.ptail)))

        # q: grid spacing factor = sqrt(log(2*pmin/(ptail + pmin)))
        self.q = np.sqrt(np.log(2 * self.pmin / (self.ptail + self.pmin)))

        # n_samples: number of samples = (1/(pmin - ptail)^2) * log((T/q + |D|)/eta)
        numerator = 1 / (self.pmin - self.ptail) ** 2
        log_arg = (self.T / self.q + len(self.target_indices)) / self.eta
        self.n_samples = int(round((numerator * np.log(log_arg)).real))

        if self.verbose:
            print(f"Number of samples: {self.n_samples}")

        return self

    def _sample_times(self) -> np.ndarray:
        """Sample time points according to the selected mode.

        Returns:
            Array of sampled time points.
        """
        if self.n_samples is None:
            raise ValueError("n_samples must be set before sampling times. Call build() first.")
        if self.mode_time == "rvs":
            from scipy.stats import truncnorm  # deferred: scipy.stats is ~0.25 s of import

            t_list = truncnorm.rvs(
                -self.sigma,  # sigma: truncation level
                self.sigma,
                loc=0,
                scale=self.T,  # T: time window
                size=self.n_samples,
            )
            # Ensure we always return an array, even for size=1
            return np.atleast_1d(t_list)
        elif self.mode_time == "rejection_sampling":
            max_attempts = 10000
            t_list = []
            for _ in range(self.n_samples):
                t = self._sample_time_rejection(max_attempts)
                t_list.append(t)
            return np.array(t_list)
        else:
            raise ValueError(f"Unknown mode_time: {self.mode_time}")

    def _sample_time_rejection(self, max_attempts: int) -> float:
        """Sample a single time point using rejection sampling.

        Args:
            max_attempts: Maximum number of rejection sampling attempts.

        Returns:
            Sampled time point.

        Raises:
            RuntimeError: If rejection sampling fails after max_attempts.
        """
        for _ in range(max_attempts):
            x = np.random.uniform(-self.sigma * self.T, self.sigma * self.T)
            y = np.random.uniform(0, 1)
            if y < self.filtering_function(x, self.sigma, self.T):
                return x
        raise RuntimeError(f"Rejection sampling failed after {max_attempts} attempts")

    def _generate_data_analytical(self) -> List[tuple]:
        """Generate dataset using classical simulation.

        This mode computes expectation values analytically and optionally simulates
        measurement noise, without running actual quantum circuits.

        If n_shots is qarp.EXACT, exact expectation values are returned (no
        shot noise); ``None`` means the default 10_000 shots.

        Returns:
            List of (time, measurement) tuples.
        """
        t_list = self._sample_times()
        N = len(t_list)  # number of time points

        # Compute true expectation values: <ψ|U(t)|ψ> = Σ_k p_k * e^(-i*λ_k*t).
        if self.eigenvalues is None or self.p is None:
            raise ValueError(
                "eigenvalues and p must be set before generating data. Call build() first."
            )
        z = self.p.dot(np.exp(-1j * np.outer(self.eigenvalues, t_list)))  # expectation values

        # qarp.EXACT: return exact expectation values (no shot noise)
        if self.n_shots is Shots.EXACT:
            return list(zip(t_list, z, strict=True))

        # Otherwise, simulate shot noise (None → the default shot count)
        n_shots = self.n_shots if self.n_shots is not None else 10_000
        N_list = np.array(n_shots * np.ones_like(t_list)).flatten()  # shots per time point
        Nsample = int(max(N_list))  # max number of shots

        # Convert to Hadamard test probabilities
        Re_true = (1 + np.real(z)) / 2  # real part probability
        Im_true = (1 + np.imag(z)) / 2  # imaginary part probability

        # Construct sampling masks for different shot numbers
        N_check = np.arange(Nsample).reshape([Nsample, 1]) * np.ones((1, N))  # shot indices
        Sign_check = np.ones((Nsample, 1)) * (N_list - 0.5)  # shot threshold
        Re_check = (np.sign(N_check - Sign_check) - 1) / (-2)  # real mask
        Im_check = (np.sign(N_check - Sign_check) - 1) / (-2)  # imaginary mask

        # Apply masks to true probabilities
        Re_true = np.multiply(Re_check, np.ones((Nsample, 1)) * Re_true)  # masked real prob
        Im_true = np.multiply(Im_check, np.ones((Nsample, 1)) * Im_true)  # masked imag prob

        # Simulate binary measurements
        Re_random = np.random.uniform(0, 1, (Nsample, N))  # random values for real
        Im_random = np.random.uniform(0, 1, (Nsample, N))  # random values for imaginary

        Re = np.sum(Re_random < Re_true, axis=0) / N_list  # measured real part
        Im = np.sum(Im_random < Im_true, axis=0) / N_list  # measured imaginary part

        # Convert back to expectation values
        Z_Had = (2 * Re - 1) + 1j * (2 * Im - 1)  # Hadamard test results

        return list(zip(t_list, Z_Had, strict=True))

    def _generate_data_sampling(self) -> List[tuple]:
        """Generate dataset using actual sampling execution.

        This mode runs Hadamard test circuits on the quantum engine.  All time
        samples are compiled and executed in ONE engine pass (one primitive per
        time point) instead of a build/run round-trip per sample — same
        measurements, far less per-call Python overhead.

        Returns:
            List of (time, measurement) tuples.
        """
        from copy import deepcopy

        t_list = self._sample_times()
        primitives: list[Runnable] = []
        for tn in t_list:
            unitary_at_t = self._unitary_at_time(tn)
            prim = deepcopy(self.primitive)
            prim.ket = self.state
            prim.bra = self.state
            prim.operator = unitary_at_t
            prim.n_shots = self.n_shots
            primitives.append(prim)

        # engine.build() builds each primitive itself — no pre-build needed.
        self.engine.build(primitives)
        results = self.engine.run()
        return [
            (tn, cast(complex, r[0] if isinstance(r, (list, tuple)) else r))
            for tn, r in zip(t_list, results, strict=True)
        ]

    def _unitary_at_time(self, time: float, verbose: Optional[bool] = None) -> AnyBlock:
        """Concrete-time unitary block, with optional synthesis-error diagnostic."""
        nonsymbolic_unitary = self.unitary.set_time(time)  # type: ignore[attr-defined, call-arg]
        nonsymbolic_unitary.build()

        if verbose is None:
            verbose = self.verbose
        if verbose:
            # LSB — same basis as unitary_matrix below.
            unitary_exact = self.unitary.operator.sparse_matrix(  # type: ignore[attr-defined]
                self.unitary.n_qubits
            ).todense()
            unitary_exact = scipy.linalg.expm(-1j * unitary_exact * time)
            import qarpx as qx

            unitary_synth = np.array(
                qx.QarpSimulator().unitary_matrix(
                    nonsymbolic_unitary.flatten(), nonsymbolic_unitary.n_qubits
                )
            )
            error = np.linalg.norm(unitary_synth - unitary_exact)
            print(f"Unitary approximation error at t={time}: {error}")
        return nonsymbolic_unitary

    def _perform_measurement(self, time: float, verbose: Optional[bool] = None) -> complex:
        """Perform a single measurement at the given time.

        Kept for direct/diagnostic use — the sampling dataset path batches all
        time samples through one engine pass in :meth:`_generate_data_sampling`.

        Args:
            time: Evolution time for the unitary.
            verbose: Whether to print debug information for this measurement.

        Returns:
            Complex measurement result from the primitive.
        """
        # Configure primitive with current operator and state.  engine.build()
        # builds the primitive itself — building here too would compile every
        # circuit twice.
        self.primitive.ket = self.state
        self.primitive.bra = self.state
        self.primitive.operator = self._unitary_at_time(time, verbose)
        self.primitive.n_shots = self.n_shots

        self.engine.build([self.primitive])
        result = self.engine.run()
        if isinstance(result, (list, tuple)):
            # Expectation-value primitive → scalar, not a Sampler distribution.
            return cast(complex, result[0])
        return result

    def generate_data(self) -> List[tuple]:
        """Generate the measurement dataset according to selected mode.

        Returns:
            List of (time, measurement) tuples.
        """
        if self.mode_dataset == "analytical":
            return self._generate_data_analytical()
        elif self.mode_dataset == "sampling":
            return self._generate_data_sampling()
        else:
            raise ValueError(f"Unknown mode_dataset: {self.mode_dataset}")

    def filtered_density_function(
        self, dataset: List[tuple], theta_js: np.ndarray, n_samples: int
    ) -> np.ndarray:
        """Compute the filtered density function G_j for given theta values.

        Args:
            dataset: List of (time, measurement) tuples from quantum measurements.
            theta_js: Array of theta values to evaluate.
            n_samples: Number of samples for normalization.

        Returns:
            G_js: filtered density values for each theta.
        """
        # Measurements are <ψ|exp(-iHt)|ψ> = Σ p_k exp(-i*λ_k*t);
        # use conjugate phase exp(+i*θ*t) to extract eigenvalues +λ_k.
        # Vectorised: G = |exp(i θ⊗t) @ m| / n — one (J×N)·(N,) product
        # instead of a J×N Python double loop.
        n = len(dataset)
        times = np.fromiter((t for t, _ in dataset), dtype=float, count=n)
        measurements = np.fromiter((m for _, m in dataset), dtype=complex, count=n)
        phases = np.exp(1j * np.outer(np.asarray(theta_js, dtype=float), times))
        return np.abs(phases @ measurements) / n_samples

    def run(self) -> List[float]:
        """Run the QMEGS algorithm to find eigenphases.

        Follows Algorithm 2 in arxiv2402.01013.

        Returns:
            List of found eigenphases.
        """
        # Validate that build() has been called
        if self.q is None or self.alpha is None or self.n_samples is None:
            raise ValueError("Algorithm parameters not set. Call build() before run().")

        # Clear previous results and reset T to initial value
        self.T = self._T_initial
        self.dataset = None
        self.res = None
        self.x_to_plot = []
        self.y_to_plot = []

        # Generate measurement dataset
        self.dataset = self.generate_data()

        # Update T (time window) based on actual data range
        t_list = np.array([x[0] for x in self.dataset])
        self.T = max(t_list) - min(t_list)

        # Generate theta grid
        # J: number of grid points = floor(2π*T/q)
        J = int(np.floor(2 * np.pi * self.T / self.q))
        # theta_js: grid of theta values = -π + j*q/T for j in [0, J]
        theta_js = np.array([-np.pi + j * self.q / self.T for j in range(J + 1)])
        # G_js: filtered density values
        G_js = self.filtered_density_function(self.dataset, theta_js, self.n_samples)

        # Initialize search parameters
        to_search = [True] * len(theta_js)
        res = []
        self.x_to_plot = []
        self.y_to_plot = []

        # Find eigenphases iteratively
        for k in range(len(self.target_indices)):
            # Find maximum in remaining search space
            search_indices = [i for i in range(len(theta_js)) if to_search[i]]
            if not search_indices:
                break

            j_k = search_indices[np.argmax(G_js[search_indices])]  # index of peak
            eigenphase = theta_js[j_k]
            res.append(eigenphase)

            # d: exclusion radius = alpha/T
            d = self.alpha / self.T
            interval = (eigenphase - d, eigenphase + d)  # exclusion interval

            # Exclude points in the interval from future searches
            for i in range(len(theta_js)):
                if interval[0] < theta_js[i] < interval[1]:
                    to_search[i] = False

            # Store data for plotting
            self.x_to_plot.append(theta_js[search_indices])
            self.y_to_plot.append(G_js[search_indices])

            if self.verbose:
                print(f"Found eigenphase {k + 1}: {eigenphase:.6f}")

        self.res = res

        return res

    def __repr__(self) -> str:
        """Return a string representation of the QMEGS instance."""
        return (
            f"QMEGS(n_qubits={self.n_qubits}, n_shots={self.n_shots}, "
            f"n_samples={self.n_samples}, mode_dataset='{self.mode_dataset}', "
            f"mode_time='{self.mode_time}')"
        )
