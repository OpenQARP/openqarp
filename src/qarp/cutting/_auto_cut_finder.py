"""Abstract AutoCutFinder and CutterResult, operating on lists of qx.Command."""

from abc import ABC, abstractmethod
from typing import Optional

import qarpx as qx


def remove_barriers(commands: list) -> list:
    """Remove all Barrier commands from a command list."""
    return [
        c for c in commands if not (isinstance(c, qx.Command) and c.gate == qx.GateType.Barrier)
    ]


class AutoCutFinder(ABC):
    def __init__(
        self,
        commands: list,
        n_qubits: int,
        max_size_subcircuits: list,
        penalization_term: float,
        verbose: bool = True,
    ) -> None:
        """Abstract base for automatic circuit cut finders.

        Args:
            commands: Flat list of qx.Command objects representing the circuit
                      to cut (barriers already stripped, measurements removed).
            n_qubits: Number of qubits in the circuit.
            max_size_subcircuits: Maximum qubit count per subcircuit, e.g. [2, 2].
            penalization_term: Penalty weight for invalid partitions in the
                               optimisation objective.
            verbose: Print progress information.
        """
        self.commands = remove_barriers(commands)
        self.n_qubits = n_qubits

        if max_size_subcircuits:
            self.n_subcircuits = len(max_size_subcircuits)
            assert sum(max_size_subcircuits) >= n_qubits, (
                "max_size_subcircuits total capacity is less than n_qubits."
            )
            assert self.n_subcircuits > 1, "At least 2 subcircuits required for circuit cutting."
        assert penalization_term > 1, "penalization_term must be > 1."

        self.max_size_subcircuits = max_size_subcircuits
        self.penalization_term = penalization_term
        self.verbose = verbose

        # Results populated by cut()
        self._subcircuits_cmds: list = []  # list of (commands, n_qubits) per subcircuit
        self._subcircuit_qubits: list = []  # global qubit indices per subcircuit
        self._cut_cmds_2q: list = []  # full circuit with 2q _CutMarker sentinels
        self._cut_cmds_1q: list = []  # same with 1q markers (for filtering)
        self._cut_performed: bool = False
        self.cut_names: dict = {}
        self.n_cuts: int = 0

    @abstractmethod
    def cut(self, manual_setting: Optional[list] = None):
        """Find cut locations and partition the circuit into subcircuits.

        Args:
            manual_setting: Optional list of qubit-index lists, one per subcircuit.

        Returns:
            CutterResult
        """
        raise NotImplementedError

    @property
    def subcircuits(self):
        """List of (commands, n_qubits) tuples for each subcircuit."""
        if self._cut_performed:
            return self._subcircuits_cmds
        raise RuntimeError("Circuit has not been cut yet.")

    @subcircuits.setter
    def subcircuits(self, value):
        assert len(value) >= self.n_subcircuits
        self._subcircuits_cmds = value

    @property
    def subcircuit_qubits(self):
        """Global qubit indices for each subcircuit."""
        if self._cut_performed:
            return self._subcircuit_qubits
        raise RuntimeError("Circuit has not been cut yet.")

    @property
    def cut_qc(self):
        """2q _CutMarker command list (for decomposition iterator)."""
        if self._cut_performed:
            return self._cut_cmds_2q
        raise RuntimeError("Circuit has not been cut yet.")

    @cut_qc.setter
    def cut_qc(self, value):
        self._cut_cmds_2q = value


def check_cut_budget(n_cuts: int, *, force: bool = False) -> None:
    """Reject cut counts past ``config.max_number_of_cuts``.

    Every cut multiplies the experiment count by 6 (6^n_cuts total) — the
    guard lives here, in ``qarp.cutting``, so both the engine-integrated and
    the standalone execution paths consult it.
    """
    from qarp import config

    if n_cuts >= config.max_number_of_cuts and not force:
        raise RuntimeError(
            f"Number of cuts ({n_cuts}) exceeds "
            f"config.max_number_of_cuts ({config.max_number_of_cuts}). "
            "Set force_max_number_cuts=True to override."
        )


class CutterResult:
    """Result of a circuit cutting operation."""

    def __init__(self, cutter_obj: AutoCutFinder) -> None:
        self.custom_commands = cutter_obj.cut_qc  # list with _CutMarker (2q form)
        self.subcircuits = cutter_obj.subcircuits  # list of (cmds, n_qubits)
        self.subcircuit_qubits = cutter_obj.subcircuit_qubits  # list of qubit-index lists
        self.cut_info = cutter_obj.cut_names
        self.n_subcircuits = cutter_obj.n_subcircuits
        self.n_cuts = len(cutter_obj.cut_names)
        self.n_qubits = cutter_obj.n_qubits

    def __str__(self) -> str:
        return (
            f"Original circuit width: {self.n_qubits}\n"
            f"Number of subcircuits: {self.n_subcircuits}\n"
            f"Number of cuts: {self.n_cuts}"
        )
