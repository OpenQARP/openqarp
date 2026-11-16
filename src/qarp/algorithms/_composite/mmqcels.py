"""Multi-modal, multi-level quantum complex exponential least squares."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Optional, Union, cast

import numpy as np
from sympy import Symbol

import qarpx as qx

from ..._types import Shots
from ...blocks import AnyBlock, TrotterBlock
from ...engines import Engine, QarpEngine
from ...operators import QubitOperator
from ...optimizers import Optimizer, ScipyOptimizer
from .._primitives import HadamardTest, StateVector
from .composite_algorithm import CompositeAlgorithm

ExecutionMode = Literal["classical", "statevector", "hadamard"]
ParameterMode = Literal["standard", "error_rate"]
SampleCounts = Union[int, Sequence[int]]


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _positive_finite(name: str, value: object) -> float:
    result = _finite_real(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


class MMQCELS(CompositeAlgorithm):
    """Estimate several dominant eigenvalues with MM-QCELS.

    The implementation follows Algorithm 2 and Eq. (13) of Ding and Lin,
    Quantum 7, 1136 (2023): times are independent truncated-Gaussian draws,
    observations are complex, and the time scale doubles between levels.

    ``parameter_mode="standard"`` is the recommended default. It requires
    ``T0``, ``N0``, ``Nj``, and exactly one of ``q`` or ``n_levels``.
    ``parameter_mode="error_rate"`` retains OpenQARP's historical
    epsilon-derived level and sample-count helpers as an explicitly opt-in
    convenience. It is not a theorem-backed MM-QCELS parameter calibration.

    Args:
        operator: Hamiltonian in classical mode, or a symbolic-time
            :class:`TrotterBlock` in circuit modes.
        state: Statevector/block in classical mode, or a built block in
            circuit modes.
        execution_mode: Required data-generation strategy: ``"classical"``,
            ``"statevector"``, or ``"hadamard"``.
        T0: Positive initial Gaussian time scale.
        parameter_mode: ``"standard"`` or the opt-in ``"error_rate"``
            compatibility preset.
        N0: Number of samples at level zero. Required in standard mode.
        Nj: Positive sample count reused after level zero, or one count per
            subsequent level. Required in standard mode, even though a scalar
            value is unused when ``n_levels=1``.
        n_dominant_eigenvalues: Number of dominant modes ``K``.
        error_rate: Accuracy ``epsilon`` used by ``parameter_mode="error_rate"``
            and by standard mode's ``q`` schedule. It does not affect a
            standard schedule with an explicit ``n_levels``.
        n_levels: Explicit number of evaluated levels. Mutually exclusive
            with ``q`` in standard mode.
        q: Theorem-1 schedule parameter. Standard mode derives
            ``l=max(ceil(log2(q/(epsilon*T0))), 1)`` and evaluates ``l+1``
            levels.
        gamma: Truncation radius in standard deviations. Sampled times lie in
            ``[-gamma*T_j, gamma*T_j]``.
        initial_eigenvalues: Optional length-``K`` first-level phase guess.
        n_initial_guesses: Total number of first-level optimization starts,
            including ``initial_eigenvalues`` when supplied.
        n_shots: Hadamard shots per quadrature. ``None`` means one shot, as in
            the paper; :data:`qarp.EXACT` evaluates the exact protocol.
        lam_min: Initial lower eigenvalue bound.
        lam_max: Initial upper eigenvalue bound.
        optimizer: SciPy-compatible bounded optimizer.
        seed: Seed for time sampling and generated optimizer starts. When no
            engine is supplied, it also seeds the default QarpEngine.
        verbose: Print per-level progress.
        engine: Optional configured execution engine.

    Attributes:
        result: Eigenvalues returned by the latest successful :meth:`run`.
        eigenvalues: Alias of :attr:`result`, following the QPE output style.
        amplitudes: Complex amplitudes fitted alongside the eigenvalues.
        level_losses: Final complex least-squares loss at every level.
        sample_counts: Number of observations used at every level.
        sampled_max_times: Largest absolute sampled time at every level.
        sampled_total_times: Sum of absolute sampled times at every level.
        optimizer_failed_starts: Number of discarded optimizer starts at every
            level. A level for which every start fails raises instead.

    Every evaluated level must contain more observations than dominant modes;
    otherwise the variable-projection residual cannot identify the frequencies.
    """

    def __init__(
        self,
        operator: Union[QubitOperator, np.ndarray, TrotterBlock],
        state: Union[np.ndarray, AnyBlock],
        *,
        execution_mode: ExecutionMode,
        T0: float,
        parameter_mode: ParameterMode = "standard",
        N0: Optional[int] = None,
        Nj: Optional[SampleCounts] = None,
        n_dominant_eigenvalues: int = 1,
        error_rate: float = 1e-3,
        n_levels: Optional[int] = None,
        q: Optional[float] = None,
        gamma: float = 1.0,
        initial_eigenvalues: Optional[Sequence[float]] = None,
        n_initial_guesses: int = 10,
        n_shots: Optional[Union[int, Shots]] = None,
        lam_min: float = -np.pi,
        lam_max: float = np.pi,
        optimizer: Optional[Optimizer] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
        engine: Optional[Engine] = None,
    ) -> None:
        self._validate_modes(execution_mode, parameter_mode)
        self.execution_mode = execution_mode
        self.parameter_mode = parameter_mode
        self.error_rate = _positive_finite("error_rate", error_rate)
        self.T0 = _positive_finite("T0", T0)
        self.gamma = _positive_finite("gamma", gamma)
        self.K = _positive_int("n_dominant_eigenvalues", n_dominant_eigenvalues)
        self.n_initial_guesses = _positive_int("n_initial_guesses", n_initial_guesses)
        self.lam_min, self.lam_max = self._validate_bounds(lam_min, lam_max)
        self.seed = seed
        self.verbose = bool(verbose)
        self.n_shots = self._validate_shots(execution_mode, n_shots)

        self._validate_operator_state(operator, state, execution_mode)
        self.operator = operator
        self.state = state

        self.Nj, self.N0, self.n_levels, self.q = self._resolve_parameters(
            N0=N0, Nj=Nj, n_levels=n_levels, q=q
        )
        self._sample_counts = self._expand_sample_counts()
        self._validate_sample_counts()
        self.initial_eigenvalues = self._validate_initial_eigenvalues(initial_eigenvalues)

        if optimizer is None:
            optimizer = ScipyOptimizer("L-BFGS-B")
        if not hasattr(optimizer, "minimize"):
            raise TypeError("optimizer must provide a minimize method")
        # The level fit is a bounded search ([lam_min, lam_max] per mode), so
        # an optimizer that rejects bounds= would fail only after the dataset
        # is generated; refuse it here.
        if not getattr(optimizer, "supports_bounds", True):
            raise TypeError(
                f"MMQCELS optimizer must honour bounds=; {type(optimizer).__name__} "
                "does not (Optimizer.supports_bounds is False).  Use ScipyOptimizer "
                "with a bounded method such as 'L-BFGS-B'."
            )
        self.optimizer = optimizer

        # CompositeAlgorithm requires these collaborators even though the
        # classical signal path does not execute either of them.
        primitive = self._make_primitive()
        if engine is None:
            engine = QarpEngine(seed=seed)
        super().__init__(primitive=primitive, engine=engine)

        self._rng = np.random.default_rng(seed)
        self._built = False
        self._classical_eigenvalues: Optional[np.typing.NDArray[np.float64]] = None
        self._classical_weights: Optional[np.typing.NDArray[np.float64]] = None
        self.result: Optional[np.typing.NDArray[np.float64]] = None
        self.eigenvalues: Optional[np.typing.NDArray[np.float64]] = None
        self.amplitudes: Optional[np.typing.NDArray[np.complex128]] = None
        self.level_losses: tuple[float, ...] = ()
        self.sample_counts: tuple[int, ...] = self._sample_counts
        self.sampled_max_times: tuple[float, ...] = ()
        self.sampled_total_times: tuple[float, ...] = ()
        self.optimizer_failed_starts: tuple[int, ...] = ()

    @staticmethod
    def _validate_modes(execution_mode: object, parameter_mode: object) -> None:
        if execution_mode not in ("classical", "statevector", "hadamard"):
            raise ValueError("execution_mode must be 'classical', 'statevector', or 'hadamard'")
        if parameter_mode not in ("standard", "error_rate"):
            raise ValueError("parameter_mode must be 'standard' or 'error_rate'")

    @staticmethod
    def _validate_bounds(lam_min: object, lam_max: object) -> tuple[float, float]:
        lower = _finite_real("lam_min", lam_min)
        upper = _finite_real("lam_max", lam_max)
        if lower >= upper:
            raise ValueError("lam_min must be smaller than lam_max")
        return lower, upper

    @staticmethod
    def _validate_shots(
        execution_mode: ExecutionMode, n_shots: Optional[Union[int, Shots]]
    ) -> Optional[Union[int, Shots]]:
        if execution_mode != "hadamard":
            if n_shots is not None:
                raise ValueError("n_shots is only valid when execution_mode='hadamard'")
            return None
        if n_shots is None or n_shots is Shots.EXACT:
            return n_shots
        return _positive_int("n_shots", n_shots)

    @staticmethod
    def _validate_operator_state(
        operator: object, state: object, execution_mode: ExecutionMode
    ) -> None:
        if execution_mode == "classical":
            if not isinstance(operator, (QubitOperator, np.ndarray)):
                raise TypeError("operator must be a QubitOperator or ndarray in classical mode")
            if not isinstance(state, (np.ndarray, qx.Block)):
                raise TypeError("state must be an ndarray or Block in classical mode")
            return
        if not isinstance(operator, TrotterBlock):
            raise TypeError("operator must be a TrotterBlock in circuit modes")
        if not isinstance(state, qx.Block):
            raise TypeError("state must be a Block in circuit modes")
        if not isinstance(operator.time, Symbol):
            raise ValueError("operator must use a symbolic time in circuit modes")

    @staticmethod
    def _normalise_nj(Nj: object) -> Union[int, tuple[int, ...]]:
        if isinstance(Nj, (int, np.integer)) and not isinstance(Nj, bool):
            return _positive_int("Nj", Nj)
        if isinstance(Nj, (str, bytes)):
            raise TypeError("Nj must be a positive integer or a sequence of them")
        try:
            values: tuple[object, ...] = tuple(Nj)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError("Nj must be a positive integer or a sequence of them") from exc
        if not values:
            raise ValueError("Nj sequence must not be empty")
        return tuple(_positive_int(f"Nj[{i}]", value) for i, value in enumerate(values))

    def _resolve_parameters(
        self,
        *,
        N0: Optional[int],
        Nj: Optional[SampleCounts],
        n_levels: Optional[int],
        q: Optional[float],
    ) -> tuple[Union[int, tuple[int, ...]], int, int, Optional[float]]:
        if self.parameter_mode == "standard":
            if N0 is None:
                raise ValueError("N0 is required when parameter_mode='standard'")
            if Nj is None:
                raise ValueError("Nj is required when parameter_mode='standard'")
            if (n_levels is None) == (q is None):
                raise ValueError("standard mode requires exactly one of n_levels or q")
            resolved_nj = self._normalise_nj(Nj)
            resolved_n0 = _positive_int("N0", N0)
            resolved_q = _positive_finite("q", q) if q is not None else None
            if n_levels is None:
                assert resolved_q is not None
                final_index = max(
                    int(np.ceil(np.log2(resolved_q / (self.error_rate * self.T0)))), 1
                )
                resolved_levels = final_index + 1
            else:
                resolved_levels = _positive_int("n_levels", n_levels)
        else:
            if q is not None:
                raise ValueError("q is not valid when parameter_mode='error_rate'")
            resolved_nj = self.optimal_dataset_length() if Nj is None else self._normalise_nj(Nj)
            if N0 is None:
                first_nj = resolved_nj if isinstance(resolved_nj, int) else resolved_nj[0]
                resolved_n0 = max(first_nj // 2, 1)
            else:
                resolved_n0 = _positive_int("N0", N0)
            resolved_levels = (
                self.optimal_number_of_iterations()
                if n_levels is None
                else _positive_int("n_levels", n_levels)
            )
            resolved_q = None

        if isinstance(resolved_nj, tuple) and len(resolved_nj) != resolved_levels - 1:
            raise ValueError(
                "Nj sequence length must equal n_levels - 1 "
                f"({resolved_levels - 1}), got {len(resolved_nj)}"
            )
        return resolved_nj, resolved_n0, resolved_levels, resolved_q

    def _expand_sample_counts(self) -> tuple[int, ...]:
        if isinstance(self.Nj, int):
            return (self.N0,) + (self.Nj,) * (self.n_levels - 1)
        return (self.N0,) + self.Nj

    def _validate_sample_counts(self) -> None:
        for level, count in enumerate(self._sample_counts):
            if count <= self.K:
                name = "N0" if level == 0 else f"Nj at level {level}"
                raise ValueError(
                    f"{name} must exceed n_dominant_eigenvalues ({self.K}); got {count}"
                )

    def _validate_initial_eigenvalues(
        self, values: Optional[Sequence[float]]
    ) -> Optional[np.typing.NDArray[np.float64]]:
        if values is None:
            return None
        array = np.asarray(values)
        if array.ndim != 1 or len(array) != self.K:
            raise ValueError(f"initial_eigenvalues must contain exactly {self.K} values")
        if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
            raise TypeError("initial_eigenvalues must be finite real numbers")
        array = np.asarray(array, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError("initial_eigenvalues must be finite")
        if np.any(array < self.lam_min) or np.any(array > self.lam_max):
            raise ValueError("initial_eigenvalues must lie within [lam_min, lam_max]")
        return np.sort(array.astype(float, copy=True))

    def _make_primitive(self):
        if self.execution_mode == "hadamard":
            shots = 1 if self.n_shots is None else self.n_shots
            return HadamardTest(real=True, imaginary=True, n_shots=shots)
        return StateVector()

    def optimal_number_of_iterations(self) -> int:
        """Return the predecessor-QCELS epsilon-only level heuristic."""

        return max(int(np.ceil(np.log2(1.0 / self.error_rate))) + 1, 1)

    def optimal_dataset_length(self) -> int:
        """Return OpenQARP's historical epsilon-only sample-count heuristic."""

        return max(int(np.round(1.0 / np.sqrt(self.error_rate))), 1)

    def time_scale(self, level: int) -> float:
        """Return the MM-QCELS scale ``T_j = 2**j * T0``."""

        if isinstance(level, bool) or not isinstance(level, (int, np.integer)):
            raise TypeError("level must be a non-negative integer")
        if level < 0 or level >= self.n_levels:
            raise ValueError(f"level must be in [0, {self.n_levels - 1}]")
        return float(2 ** int(level) * self.T0)

    def generate_times(self, T: float, n: int) -> np.typing.NDArray[np.float64]:
        """Draw ``n`` independent samples from Eq. (3)'s distribution."""

        from scipy.stats import truncnorm  # deferred: scipy.stats is ~0.25 s of import

        scale = _positive_finite("T", T)
        count = _positive_int("n", n)
        return np.asarray(
            truncnorm.rvs(
                -self.gamma,
                self.gamma,
                loc=0.0,
                scale=scale,
                size=count,
                random_state=self._rng,
            ),
            dtype=float,
        )

    def _prepare_classical_cache(self) -> None:
        if isinstance(self.operator, QubitOperator):
            if isinstance(self.state, qx.Block):
                width = self.state.n_qubits
            else:
                state_size = int(np.asarray(self.state).size)
                width = state_size.bit_length() - 1
                if state_size <= 0 or 2**width != state_size:
                    raise ValueError("state length must be a positive power of two")
            hamiltonian = self.operator.sparse_matrix(width).toarray()
        else:
            hamiltonian = np.asarray(self.operator, dtype=complex)
        if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
            raise ValueError("operator ndarray must be square")
        if not np.allclose(hamiltonian, hamiltonian.conj().T):
            raise ValueError("operator must be Hermitian in classical mode")

        if isinstance(self.state, qx.Block):
            self.state.build()
            state = np.asarray(
                qx.QarpSimulator().statevector(self.state.flatten(), self.state.n_qubits),
                dtype=complex,
            )
        else:
            state = np.asarray(self.state, dtype=complex)
        state = state.reshape(-1)
        if state.size != hamiltonian.shape[0]:
            raise ValueError("state length must match the operator dimension")
        norm = float(np.linalg.norm(state))
        if not np.isfinite(norm) or not np.isclose(norm, 1.0, atol=1e-10):
            raise ValueError("state must be normalized")

        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        overlaps = eigenvectors.conj().T @ state
        self._classical_eigenvalues = np.asarray(eigenvalues, dtype=float)
        self._classical_weights = np.asarray(np.abs(overlaps) ** 2, dtype=float)

    def build(self) -> MMQCELS:
        """Build and cache the selected dataset evaluator."""

        if self.execution_mode == "classical":
            self._prepare_classical_cache()
        else:
            self.primitive.ket = self.state
            self.primitive.bra = None
            self.primitive.operator = self.operator
            self.engine.build([self.primitive])
        self._built = True
        if self.verbose:
            print("MMQCELS build")
            print(f"  execution mode: {self.execution_mode}")
            print(f"  parameter mode: {self.parameter_mode}")
            print(f"  dominant modes: {self.K}")
            print(f"  levels: {self.n_levels}")
        return self

    def _classical_signal(
        self, times: np.typing.NDArray[np.float64]
    ) -> np.typing.NDArray[np.complex128]:
        if self._classical_eigenvalues is None or self._classical_weights is None:
            self._prepare_classical_cache()
        assert self._classical_eigenvalues is not None
        assert self._classical_weights is not None
        phases = np.exp(-1j * np.outer(times, self._classical_eigenvalues))
        return np.asarray(phases @ self._classical_weights, dtype=complex)

    def _circuit_signal(
        self, times: np.typing.NDArray[np.float64]
    ) -> np.typing.NDArray[np.complex128]:
        if not self._built:
            self.build()
        assert isinstance(self.operator, TrotterBlock)
        assert isinstance(self.operator.time, Symbol)
        symbol = str(self.operator.time)
        param_sets = [{symbol: float(time)} for time in times]
        results = self.engine.batch_run([self.primitive], param_sets, rebuild=False)
        return np.asarray(
            [complex(cast("float | complex", per_set[0])) for per_set in results],
            dtype=np.complex128,
        )

    def generate_dataset(self, T: float, n: Optional[int] = None) -> list[tuple[float, complex]]:
        """Generate one independent MM-QCELS dataset at scale ``T``."""

        count = self.N0 if n is None else _positive_int("n", n)
        times = self.generate_times(T, count)
        values = (
            self._classical_signal(times)
            if self.execution_mode == "classical"
            else self._circuit_signal(times)
        )
        return [(float(time), complex(value)) for time, value in zip(times, values, strict=True)]

    @staticmethod
    def _dataset_arrays(
        dataset: Sequence[tuple[float, complex]],
    ) -> tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.complex128]]:
        if not dataset:
            raise ValueError("dataset must not be empty")
        times = np.asarray([item[0] for item in dataset], dtype=float)
        values = np.asarray([item[1] for item in dataset], dtype=complex)
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
            raise ValueError("dataset values must be finite")
        return times, values

    @staticmethod
    def _variable_projection(
        eigenvalues: np.typing.ArrayLike,
        times: np.typing.NDArray[np.float64],
        values: np.typing.NDArray[np.complex128],
    ) -> tuple[float, np.typing.NDArray[np.complex128]]:
        theta = np.asarray(eigenvalues, dtype=float)
        design = np.exp(-1j * np.outer(times, theta))
        amplitudes, *_ = np.linalg.lstsq(design, values, rcond=None)
        residual = values - design @ amplitudes
        loss = float(np.mean(np.abs(residual) ** 2))
        return loss, np.asarray(amplitudes, dtype=complex)

    def objective(
        self, eigenvalues: np.typing.ArrayLike, dataset: Sequence[tuple[float, complex]]
    ) -> float:
        """Evaluate the full complex Eq. (13) loss after eliminating amplitudes."""

        theta = np.asarray(eigenvalues, dtype=float)
        if theta.shape != (self.K,):
            raise ValueError(f"eigenvalues must contain exactly {self.K} values")
        times, values = self._dataset_arrays(dataset)
        return self._variable_projection(theta, times, values)[0]

    def _random_start(self, bounds: Sequence[tuple[float, float]]) -> np.ndarray:
        return np.sort(
            np.asarray([self._rng.uniform(low, high) for low, high in bounds], dtype=float)
        )

    @staticmethod
    def _optimizer_message(result: object) -> str:
        return str(getattr(result, "message", "optimizer reported failure"))

    def _fit_level(
        self,
        dataset: Sequence[tuple[float, complex]],
        starts: Sequence[np.typing.NDArray[np.float64]],
        bounds: Sequence[tuple[float, float]],
    ) -> tuple[
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.complex128],
        float,
        int,
    ]:
        times, values = self._dataset_arrays(dataset)

        def loss(theta):
            return self._variable_projection(theta, times, values)[0]

        successful = []
        failures = []
        for start in starts:
            result = self.optimizer.minimize(loss, start, bounds=cast("Sequence[float]", bounds))
            success = bool(getattr(result, "success", True))
            finite = np.isfinite(float(getattr(result, "fun", np.inf))) and np.all(
                np.isfinite(np.asarray(getattr(result, "x", []), dtype=float))
            )
            if success and finite:
                successful.append(result)
            else:
                failures.append(self._optimizer_message(result))
        if not successful:
            detail = "; ".join(failures) if failures else "no optimizer result"
            raise RuntimeError(f"MMQCELS optimization failed: {detail}")

        best = min(successful, key=lambda result: float(result.fun))
        theta_unsorted = np.asarray(best.x, dtype=float)
        order = np.argsort(theta_unsorted)
        theta = theta_unsorted[order]
        final_loss, amplitudes = self._variable_projection(theta, times, values)
        return theta, amplitudes, final_loss, len(failures)

    def run(self) -> np.typing.NDArray[np.float64]:
        """Run all levels and return the sorted eigenvalues.

        The returned array is also stored in :attr:`result` and
        :attr:`eigenvalues`. Fitted amplitudes and per-level diagnostics are
        available on the corresponding instance attributes.
        """

        if not self._built:
            self.build()
        self._rng = np.random.default_rng(self.seed)

        full_bounds = [(self.lam_min, self.lam_max)] * self.K
        starts = []
        if self.initial_eigenvalues is not None:
            starts.append(self.initial_eigenvalues.copy())
        while len(starts) < self.n_initial_guesses:
            starts.append(self._random_start(full_bounds))

        level_losses = []
        sampled_max_times = []
        sampled_total_times = []
        optimizer_failed_starts = []
        theta: Optional[np.typing.NDArray[np.float64]] = None
        amplitudes: Optional[np.typing.NDArray[np.complex128]] = None

        for level, count in enumerate(self._sample_counts):
            scale = self.time_scale(level)
            dataset = self.generate_dataset(scale, count)
            times = np.asarray([item[0] for item in dataset], dtype=float)
            if level == 0:
                bounds = full_bounds
                level_starts = starts
            else:
                assert theta is not None
                bounds = [
                    (
                        float(value - np.pi / scale),
                        float(value + np.pi / scale),
                    )
                    for value in theta
                ]
                level_starts = [theta.copy()]

            theta, amplitudes, loss, failed_starts = self._fit_level(dataset, level_starts, bounds)
            level_losses.append(loss)
            sampled_max_times.append(float(np.max(np.abs(times))))
            sampled_total_times.append(float(np.sum(np.abs(times))))
            optimizer_failed_starts.append(failed_starts)
            if self.verbose:
                print(
                    f"MMQCELS level {level}: T={scale:.6g}, N={count}, "
                    f"loss={loss:.6g}, eigenvalues={theta}"
                )

        assert theta is not None and amplitudes is not None
        self.eigenvalues = np.asarray(theta, dtype=float)
        self.amplitudes = np.asarray(amplitudes, dtype=complex)
        self.level_losses = tuple(level_losses)
        self.sample_counts = self._sample_counts
        self.sampled_max_times = tuple(sampled_max_times)
        self.sampled_total_times = tuple(sampled_total_times)
        self.optimizer_failed_starts = tuple(optimizer_failed_starts)
        self.result = self.eigenvalues
        return self.result
