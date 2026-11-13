from dataclasses import dataclass
from typing import Dict, List

from .._config import PlotConfig
from .positioning import get_x_positions_list, get_y_positions


class CircuitValidationError(Exception):
    """Custom exception for circuit validation errors."""

    pass


@dataclass
class CircuitData:
    """Container for processed circuit data."""

    qubits: List
    commands: List
    non_measure_cmds: List
    x_positions: List
    y_positions: Dict
    n_qubits: int
    n_measures: int
    n_layers: int
    global_phase: float


class CircuitProcessor:
    def __init__(self, config: PlotConfig):
        """Processes and validates circuits for plotting."""
        self.config = config

    def process_circuit(
        self, circ, decompose_boxes: bool, flatten_layers: bool, invert_order: bool
    ) -> CircuitData:
        """Process circuit into plotting data with validation."""
        # Validate circuit
        self._validate_circuit(circ)

        qubits = circ.qubits

        # Box decomposition is handled upstream by ``CircuitAdapter`` (it
        # decides per-child whether to emit a box or flatten), so by the time
        # the command stream reaches the processor the choice is already baked
        # in.  Accepted here only for signature compatibility.
        del decompose_boxes

        commands = circ.get_commands()
        non_measure_cmds = [cmd for cmd in commands if str(cmd.op).lower() != "measure"]
        # Filter out global phase gates (no qubits) for positioning purposes
        visible_cmds = [cmd for cmd in non_measure_cmds if len(cmd.qubits) > 0]

        y_positions = get_y_positions(qubits, invert_order)
        n_measures = self._count_and_check_measurements(commands)

        if flatten_layers:
            x_positions = get_x_positions_list(visible_cmds, qubits)
        else:
            x_positions = list(range(len(visible_cmds)))

        n_layers = max(x_positions) if x_positions else 0

        return CircuitData(
            qubits=qubits,
            commands=commands,
            non_measure_cmds=visible_cmds,  # Only include visible commands
            x_positions=x_positions,
            y_positions=y_positions,
            n_qubits=len(qubits),
            n_measures=n_measures,
            n_layers=n_layers,
            global_phase=circ.phase,
        )

    def _validate_circuit(self, circ):
        """Validate circuit structure."""
        if not circ.qubits:
            raise CircuitValidationError("Circuit has no qubits")

    def _count_and_check_measurements(self, commands) -> int:
        """Count measurements and validate their placement."""
        measured_qubits = set()
        n_measures = 0

        for cmd in commands:
            if str(cmd.op).lower() == "measure":
                n_measures += 1
                for qubit in cmd.qubits:
                    if qubit in measured_qubits:
                        raise CircuitValidationError(
                            f"Measurement on qubit {qubit} is not the last operation."
                        )
                    measured_qubits.add(qubit)
            else:
                for qubit in cmd.qubits:
                    if qubit in measured_qubits:
                        raise CircuitValidationError(
                            f"Operation on qubit {qubit} occurs after a measurement."
                        )

        return n_measures
