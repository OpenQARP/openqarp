"""Direct tests for CircuitProcessor.

Reached only through ``test_circuit_plotter.py``, so its validation errors —
the ones that tell a user *why* their circuit will not draw — had no named
coverage.  A stub circuit stands in for the plot adapter's duck type
(``qubits`` / ``get_commands()`` / ``phase``).
"""

import pytest

from qarp.plotting._config import PlotConfig
from qarp.plotting._processors.circuit_processor import (
    CircuitProcessor,
    CircuitValidationError,
)


class _Cmd:
    def __init__(self, op, qubits, cbits=(), condition=None, condition_bits=()):
        self.op = op
        self.qubits = qubits
        self.cbits = list(cbits)
        self.condition = condition
        self.condition_bits = list(condition_bits)


class _Circuit:
    def __init__(self, qubits, commands, phase=0.0):
        self.qubits = qubits
        self._commands = commands
        self.phase = phase

    def get_commands(self):
        return self._commands


@pytest.fixture
def processor():
    return CircuitProcessor(PlotConfig())


def _process(processor, circ, flatten_layers=True):
    return processor.process_circuit(
        circ, decompose_boxes=False, flatten_layers=flatten_layers, invert_order=False
    )


def test_a_circuit_without_qubits_is_refused(processor):
    with pytest.raises(CircuitValidationError, match="no qubits"):
        _process(processor, _Circuit([], []))


def test_a_measurement_followed_by_a_gate_is_drawn_where_it_happens(processor):
    """A mid-circuit measurement takes its own column, before the gate."""
    circ = _Circuit([0], [_Cmd("Measure", [0]), _Cmd("h", [0])])
    data = _process(processor, circ)
    assert data.n_measures == 0
    assert data.non_measure_cmds == circ.get_commands()
    assert data.x_positions == [0.25, 1.25]


def test_only_the_last_of_two_measurements_closes_the_wire(processor):
    first, second = _Cmd("Measure", [0]), _Cmd("Measure", [0])
    data = _process(processor, _Circuit([0], [first, second]))
    assert data.non_measure_cmds == [first]
    assert data.terminal_measures == [second]


def test_a_measurement_closes_only_its_own_wire(processor):
    """Measuring q0 stays terminal when the later gate is on q1."""
    circ = _Circuit([0, 1], [_Cmd("Measure", [0]), _Cmd("h", [1])])
    data = _process(processor, circ)
    assert data.n_measures == 1
    assert len(data.non_measure_cmds) == 1


def test_a_measurement_whose_bit_is_read_is_drawn_before_its_reader(processor):
    """The measured qubit is idle afterwards, but a condition reads the bit:
    the measurement is drawn in place, and both link to the classical wire."""
    measure = _Cmd("Measure", [0], cbits=[0])
    gate = _Cmd("x", [1], condition="c0=1", condition_bits=[0])
    data = _process(processor, _Circuit([0, 1], [measure, gate]))
    assert data.n_measures == 0
    assert data.x_positions == [0.25, 1.25]
    assert data.writes_classical == [True, False]
    assert data.reads_classical == [False, True]
    assert data.has_classical_wire


def test_a_bit_written_twice_links_only_the_write_that_is_read(processor):
    """The second measurement overwrites c0 before the condition reads it, so
    the first one feeds nothing: it closes its wire like any last measurement."""
    first = _Cmd("Measure", [0], cbits=[0])
    second = _Cmd("Measure", [1], cbits=[0])
    gate = _Cmd("x", [2], condition="c0=1", condition_bits=[0])
    data = _process(processor, _Circuit([0, 1, 2], [first, second, gate]))
    assert data.terminal_measures == [first]
    assert data.non_measure_cmds == [second, gate]
    assert data.writes_classical == [True, False]


def test_a_bit_nobody_reads_draws_no_classical_wire(processor):
    measure = _Cmd("Measure", [0], cbits=[0])
    data = _process(processor, _Circuit([0], [measure, _Cmd("h", [0])]))
    assert data.writes_classical == [False, False]
    assert not data.has_classical_wire


def test_a_column_grows_to_its_widest_command(processor):
    """Columns are one spacing wide unless a command needs more; a wide first
    column pushes the second one along by the extra width."""
    wide, narrow = _Cmd("ry", [0]), _Cmd("h", [0])
    spacing = processor.config.spacing
    need = {id(wide): 2 * spacing, id(narrow): 0.1 * spacing}
    data = processor.process_circuit(
        _Circuit([0], [wide, narrow]),
        False,
        True,
        False,
        width_of=lambda cmd, reads, writes: need[id(cmd)],
    )
    first_width = 2 * spacing + 0.4 * spacing
    assert data.x_coords == pytest.approx(
        [-0.25 * spacing + first_width / 2, -0.25 * spacing + first_width + spacing / 2]
    )


def test_a_link_label_widens_the_column_only_when_the_link_is_drawn(processor):
    """``width_of`` is told whether each command links to the classical wire:
    a measurement whose bit nobody reads does not, the one that is read does."""
    unread = _Cmd("Measure", [0], cbits=[0])
    read = _Cmd("Measure", [0], cbits=[1])
    gate = _Cmd("x", [0], condition="c1=1", condition_bits=[1])
    seen = []
    processor.process_circuit(
        _Circuit([0], [unread, read, gate]),
        False,
        True,
        False,
        width_of=lambda cmd, reads, writes: seen.append((reads, writes)) or 0.0,
    )
    assert seen == [(False, False), (False, True), (True, False)]


def test_default_columns_are_one_spacing_apart(processor):
    """Without widths the first gate sits a quarter spacing in, one spacing apart."""
    spacing = processor.config.spacing
    data = _process(processor, _Circuit([0], [_Cmd("h", [0]), _Cmd("x", [0])]))
    assert data.x_coords == pytest.approx([0.25 * spacing, 1.25 * spacing])


def test_global_phase_commands_are_excluded_from_the_layout(processor):
    """A phase gate carries no wire: it is not a visible command and takes no
    column, but its value still reaches the plot."""
    circ = _Circuit([0], [_Cmd("gphase", []), _Cmd("h", [0])], phase=0.5)
    data = _process(processor, circ)
    assert data.non_measure_cmds == [circ.get_commands()[1]]
    assert data.x_positions == [0.25]
    assert data.global_phase == 0.5


def test_unflattened_layout_gives_one_column_per_visible_command(processor):
    circ = _Circuit([0, 1], [_Cmd("h", [0]), _Cmd("h", [1]), _Cmd("x", [0])])
    data = _process(processor, circ, flatten_layers=False)
    assert data.x_positions == [0, 1, 2]
    assert data.n_layers == 2


def test_an_empty_command_list_reports_zero_layers(processor):
    data = _process(processor, _Circuit([0], []))
    assert data.x_positions == []
    assert data.n_layers == 0
    assert data.n_qubits == 1
