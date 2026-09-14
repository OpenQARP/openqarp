"""Abstract PostProcessing base class for QPD-based circuit cutting."""

from abc import ABC, abstractmethod

from qarp.operators import QubitOperator

from ._auto_cut_finder import CutterResult


class PostProcessing(ABC):
    def __init__(
        self,
        cutter_result: CutterResult,
        observable: QubitOperator,
        verbose: bool = True,
    ) -> None:
        """
        Args:
            cutter_result: Result of the circuit cutting step.
            observable: Hamiltonian / observable for the expectation value, as a
                ``qarp.operators.QubitOperator`` (not openfermion's — convert
                with ``qarp.operators.compat.from_openfermion()`` first).
            verbose: Print runtime information.
        """
        assert cutter_result.subcircuits is not None
        assert len(cutter_result.subcircuits) > 1, "Circuit has not been cut yet."
        if not isinstance(observable, QubitOperator):
            obs_type = type(observable)
            raise TypeError(
                f"observable must be a qarp.operators.QubitOperator instance, got "
                f"{obs_type.__module__}.{obs_type.__qualname__} — same class name, "
                "different (incompatible) type. If converting from openfermion, use "
                "qarp.operators.compat.from_openfermion() first."
            )

        # idxs[i] = global qubit indices belonging to subcircuit i
        self.idxs = cutter_result.subcircuit_qubits
        self.custom_commands = cutter_result.custom_commands
        self.subcircuits = cutter_result.subcircuits  # list of (cmds, n_qubits)
        self.n_subcircuits = cutter_result.n_subcircuits
        self.n_cuts = cutter_result.n_cuts
        self.cut_info = cutter_result.cut_info
        self.verbose = verbose
        self.observable = observable

        self._experiments = None  # list of command lists (one per full experiment)
        self._jobs = None  # dict {str(i_sub): list of (cmds, n_qubits)}
        self._coefficients = None

    @abstractmethod
    def decompose(self):
        raise NotImplementedError

    @abstractmethod
    def compute(self, **kwargs) -> float:
        raise NotImplementedError

    @property
    def jobs(self):
        if self._jobs is not None:
            return self._jobs
        raise RuntimeError("Cuts have not been processed yet; call decompose() first.")

    @jobs.setter
    def jobs(self, value):
        if not isinstance(value, dict):
            raise ValueError("jobs must be a dict")
        self._jobs = value

    @property
    def experiments(self):
        if self._experiments is not None:
            return self._experiments
        raise RuntimeError("Cuts have not been processed yet; call decompose() first.")

    @experiments.setter
    def experiments(self, value):
        if not isinstance(value[0], list):
            raise ValueError("experiments must be a list of command lists")
        self._experiments = value

    @property
    def coefficients(self):
        if self._coefficients is not None:
            return self._coefficients
        raise RuntimeError("Cuts have not been processed yet; call decompose() first.")

    @coefficients.setter
    def coefficients(self, value):
        self._coefficients = value
