import copy
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Optional, Self, Union

from ... import MPIConfig
from ..._types import Consumes
from ...engines import Engine, QarpEngine
from ...errors import CapabilityError
from .. import PrimitiveAlgorithm


class CompositeAlgorithm(ABC):
    def __init__(
        self,
        primitive: PrimitiveAlgorithm,
        engine: Optional[Engine] = None,
    ):
        """
        Base class for composite algorithms that manage multiple sub-algorithms or require
        more complex Block compositions.

        Args:
            engine: Optional quantum engine to execute circuits. If None, a default
                ``QarpEngine`` is used.  To target a specific device pass a
                pre-configured engine, e.g. ``QarpEngine(device=qx.Device(...))``.
            primitive: Optional primitive algorithm to use for circuit execution.
                The base stores a private deepcopy — ``build()`` implementations
                rebind ket/bra/operator on it, and one primitive shared across
                algorithms must not cross-corrupt.
        """
        # Under a multi-rank launcher every rank would run the same serial
        # optimisation; that is not parallelism, so refuse up front.
        if MPIConfig.is_mpi_env() and MPIConfig.world_size() > 1:
            raise CapabilityError(
                f"{type(self).__name__}: MPI parallelism is not implemented — "
                f"{MPIConfig.world_size()} ranks detected.  Run OpenQARP in a single "
                "process, or set QARP_DISABLE_MPI=1 on the one rank that drives it."
            )
        if engine is None:
            engine = QarpEngine()
        self.engine = engine
        self.primitive = copy.deepcopy(primitive) if primitive is not None else None

        # Last-evaluation caches: objective/gradient wrappers write these so
        # verbose callbacks can print without re-running the quantum workload.
        self._last_objective: Optional[float] = None
        self._last_gradnorm: Optional[float] = None

        # List to store sub-algorithms (can be primitive or composite)
        self.sub_algorithms: list[Union[PrimitiveAlgorithm, CompositeAlgorithm]] = []

    @abstractmethod
    def build(self) -> Self:
        """
        Build the composite algorithm by constructing all sub-algorithms.

        This method should:
        1. Create and configure all necessary sub-algorithms
        2. Build each sub-algorithm

        Returns:
            Self for method chaining
        """

    @abstractmethod
    def run(self) -> Any:
        """Execute the algorithm after ``build()``.

        The return type is algorithm-specific (documented per class);
        subclasses may add optional keyword-only arguments such as
        ``max_iter`` but take no positional arguments.
        """

    # ── Statevector fast paths ──────────────────────────────────────────

    def _amplitude_block_reason(self, *, check_primitive: bool = True) -> Optional[str]:
        """Why a statevector fast path is not legitimate, or None when it is:
        the engine must hand exact amplitudes in logical qubit order and —
        unless ``check_primitive=False`` — the primitive must contract
        amplitudes (or the algorithm is amplitude-native, ``primitive is None``).

        ``check_primitive=False`` is for a classical step that is independent
        of how the algorithm's own circuits are measured (QMEGS's a-priori
        overlaps): the engine must still be exact, the primitive may sample.
        """
        eng = self.engine
        if (
            check_primitive
            and self.primitive is not None
            and self.primitive.consumes is not Consumes.AMPLITUDES
        ):
            return f"{type(self.primitive).__name__} consumes counts, not amplitudes"
        if not eng.provides_amplitudes:
            return (
                f"{type(eng).__name__} does not provide amplitudes "
                "(noise model enabled, or an inherently noisy engine)"
            )
        if eng._routed():
            return (
                f"{type(eng).__name__} routes through a device architecture, so "
                "amplitudes are not guaranteed to be in logical qubit order"
            )
        return None

    def _amplitudes_available(self, *, check_primitive: bool = True) -> bool:
        """True iff a statevector fast path is legitimate (see
        ``_amplitude_block_reason``)."""
        return self._amplitude_block_reason(check_primitive=check_primitive) is None

    def _amplitude_simulator(self, n_qubits: int, *, check_primitive: bool = True):
        """The engine's live simulator for a statevector fast path over an
        ``n_qubits``-wide circuit.

        Every algorithm-level bypass of the engine pipeline reads amplitudes
        through here, never through a fresh ``qx.QarpSimulator()``, so
        noise, routing and engine kind are honoured exactly as the engine
        path would (§14: capability checks re-validate at run time).  The
        simulator comes from ``Engine._host_statevector_simulator``, so an
        engine's host-transfer cap applies here as on its primitive path.
        """
        reason = self._amplitude_block_reason(check_primitive=check_primitive)
        if reason is not None:
            raise CapabilityError(
                f"{type(self).__name__} statevector fast path needs exact amplitudes "
                f"in logical qubit order, but {reason}.  Use a sampling primitive "
                "with finite shots, disable the engine's noise model where it can "
                "be disabled, or use an architecture-free device."
            )
        self.engine._pre_run()  # QarpEngine re-syncs _sim with the live noise toggle
        sim = self.engine._host_statevector_simulator(n_qubits)
        if sim is None:
            # provides_amplitudes is a promise about the engine's primitives;
            # the fast path additionally needs a host statevector API.
            raise CapabilityError(
                f"{type(self).__name__} statevector fast path needs a simulator "
                f"with a host statevector API, but {type(self.engine).__name__} "
                "exposes none.  Use QarpEngine, or a sampling primitive."
            )
        return sim

    # ── Shared optimization machinery ───────────────────────────────────

    def _minimize(
        self,
        objective: Callable,
        initial_parameters: Iterable[float],
        optimizer,
        gradient: Optional[Callable] = None,
        callback: Optional[Callable] = None,
        success_label: Optional[str] = None,
        suppress_success: bool = False,
    ):
        """``optimizer.minimize`` plus the shared success/failure line.

        ``success_label`` prints "<label> minimization finished successfully"
        (unless ``suppress_success``) or the "did NOT finish" line; None
        prints nothing.  Returns the optimizer's scipy-shaped result.
        """
        if gradient is not None:
            result = optimizer.minimize(
                objective, initial_parameters, callback=callback, gradient=gradient
            )
        else:
            result = optimizer.minimize(objective, initial_parameters, callback=callback)

        if success_label is not None:
            if result.success:
                if not suppress_success:
                    print(f"{success_label} minimization finished successfully")
            else:
                print(f"{success_label} minimization did NOT finish successfully")
        return result

    def _log_iteration(
        self,
        iteration: int,
        energy: float,
        dE: float,
        step_size: float,
        gradnorm: Optional[float] = None,
        cost: Any = None,
        label: Optional[str] = None,
    ) -> None:
        """Tab-formatted optimizer-progress lines shared by every composite.

        Iteration 0 prints the header only; the optional ``gradnorm`` /
        ``cost`` columns appear iff their values are not None.
        """
        if iteration == 0:
            print((label or type(self).__name__) + " Run:")
            cols = "\t\tIteration\t\tEnergy\t\t\t\t  dE\t\t\t\t|step|"
            if gradnorm is not None:
                cols += "\t\t\t\t|grad|"
            if cost is not None:
                cols += "\t\t\t\tCost"
            print(cols)
            return

        def _f(v):
            return f"{v:{'.10f' if v < 0 else ' .10f'}}"

        row = f"\t\t\t{iteration}\t\t{_f(energy)}\t\t{_f(dE)}\t\t{_f(step_size)}"
        if gradnorm is not None:
            row += f"\t\t{_f(gradnorm)}"
        if cost is not None:
            row += f"\t\t{cost}"
        print(row)
