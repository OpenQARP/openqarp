from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

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
    # Commands drawn in the closing measurement column, in command order.
    terminal_measures: List = field(default_factory=list)
    # Per drawn command: its x in data units, and whether it links to the classical wire.
    x_coords: List[float] = field(default_factory=list)
    reads_classical: List[bool] = field(default_factory=list)
    writes_classical: List[bool] = field(default_factory=list)
    measure_x: float = 0.0
    wire_end: float = 0.0
    has_classical_wire: bool = False


def _is_measure(cmd) -> bool:
    return str(cmd.op).lower() == "measure"


class CircuitProcessor:
    def __init__(self, config: PlotConfig):
        """Processes and validates circuits for plotting."""
        self.config = config

    def process_circuit(
        self,
        circ,
        decompose_boxes: bool,
        flatten_layers: bool,
        invert_order: bool,
        width_of: Optional[Callable[[object, bool, bool], float]] = None,
    ) -> CircuitData:
        """Process circuit into plotting data with validation.

        ``width_of(cmd, reads, writes)`` gives the width a drawn command needs,
        in data units, given whether it links to the classical wire; its
        column grows to hold it.  Without it every column is one spacing wide.
        """
        # Validate circuit
        self._validate_circuit(circ)

        qubits = circ.qubits

        # Box decomposition is handled upstream by ``CircuitAdapter`` (it
        # decides per-child whether to emit a box or flatten), so by the time
        # the command stream reaches the processor the choice is already baked
        # in.  Accepted here only for signature compatibility.
        del decompose_boxes

        commands = circ.get_commands()
        terminal = self._terminal_measurements(commands)
        # Drawn in time order: everything on a wire except the closing
        # measurements (global phase gates have no wire).
        visible_cmds = [
            cmd for i, cmd in enumerate(commands) if i not in terminal and len(cmd.qubits) > 0
        ]

        reads = [bool(getattr(cmd, "condition_bits", ())) for cmd in visible_cmds]
        writes = self._read_later(visible_cmds)
        linked = [r or w for r, w in zip(reads, writes, strict=True)]

        y_positions = get_y_positions(qubits, invert_order)

        # The classical wire is drawn under the lowest wire: the last qubit,
        # or the first one when the order is inverted.
        lowest = 0 if invert_order else len(qubits) - 1
        if flatten_layers:
            reach = [lowest if link else None for link in linked]
            x_positions = get_x_positions_list(visible_cmds, qubits, reach)
        else:
            x_positions = list(range(len(visible_cmds)))

        n_layers = max(x_positions) if x_positions else 0

        spacing = self.config.spacing
        n_columns = int(n_layers) + 1 if x_positions else 0
        if width_of is None:
            widths = [spacing] * n_columns
        else:
            # A column holding only zero-width marks (barriers) is half as wide.
            widths = [spacing / 2] * n_columns
            for i, (cmd, x) in enumerate(zip(visible_cmds, x_positions, strict=True)):
                need = width_of(cmd, reads[i], writes[i])
                if need > 0:
                    widths[int(x)] = max(widths[int(x)], spacing, need + 0.4 * spacing)
        # A column one spacing wide is centered a quarter spacing past its left
        # edge plus half its width: the first gate sits at spacing / 4.
        edge = -0.25 * spacing
        centers = []
        for width in widths:
            centers.append(edge + width / 2)
            edge += width
        x_coords = [centers[int(x)] for x in x_positions]
        measure_x = edge + spacing / 2
        wire_end = measure_x if terminal else edge + 0.25 * spacing

        return CircuitData(
            qubits=qubits,
            commands=commands,
            non_measure_cmds=visible_cmds,
            x_positions=x_positions,
            y_positions=y_positions,
            n_qubits=len(qubits),
            n_measures=len(terminal),
            n_layers=n_layers,
            global_phase=circ.phase,
            terminal_measures=[commands[i] for i in sorted(terminal)],
            x_coords=x_coords,
            reads_classical=reads,
            writes_classical=writes,
            measure_x=measure_x,
            wire_end=wire_end,
            has_classical_wire=any(linked),
        )

    def _validate_circuit(self, circ):
        """Validate circuit structure."""
        if not circ.qubits:
            raise CircuitValidationError("Circuit has no qubits")

    @staticmethod
    def _terminal_measurements(commands) -> set:
        """Indices of the measurements that close their wire.

        A measurement is terminal when nothing acts on its qubit afterwards and
        no later condition reads its bit; those share the last column.  Every
        other measurement is drawn where it happens.
        """
        terminal = set()
        used_later: set = set()
        read_later: set = set()
        for i in range(len(commands) - 1, -1, -1):
            cmd = commands[i]
            condition_bits = getattr(cmd, "condition_bits", ())
            cbits = getattr(cmd, "cbits", ())
            if (
                _is_measure(cmd)
                and not condition_bits
                and not used_later.intersection(cmd.qubits)
                and not read_later.intersection(cbits)
            ):
                terminal.add(i)
            used_later.update(cmd.qubits)
            # An unconditional write hides every earlier write from later reads.
            if not condition_bits:
                read_later.difference_update(cbits)
            read_later.update(condition_bits)
        return terminal

    @staticmethod
    def _read_later(commands) -> List[bool]:
        """For each command, whether a later condition reads a bit it writes."""
        flags = [False] * len(commands)
        read_later: set = set()
        for i in range(len(commands) - 1, -1, -1):
            cbits = getattr(commands[i], "cbits", ())
            condition_bits = getattr(commands[i], "condition_bits", ())
            flags[i] = bool(read_later.intersection(cbits))
            # An unconditional write hides every earlier write from later reads.
            if not condition_bits:
                read_later.difference_update(cbits)
            read_later.update(condition_bits)
        return flags
