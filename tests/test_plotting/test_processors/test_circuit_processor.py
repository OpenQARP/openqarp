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
    def __init__(self, op, qubits):
        self.op = op
        self.qubits = qubits


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


def test_an_operation_after_a_measurement_is_refused(processor):
    circ = _Circuit([0], [_Cmd("Measure", [0]), _Cmd("h", [0])])
    with pytest.raises(CircuitValidationError, match="after a measurement"):
        _process(processor, circ)


def test_measuring_the_same_qubit_twice_is_refused(processor):
    circ = _Circuit([0], [_Cmd("Measure", [0]), _Cmd("Measure", [0])])
    with pytest.raises(CircuitValidationError, match="not the last operation"):
        _process(processor, circ)


def test_an_operation_after_a_measurement_on_another_wire_is_fine(processor):
    """The guard is per qubit: measuring q0 must not fence q1."""
    circ = _Circuit([0, 1], [_Cmd("Measure", [0]), _Cmd("h", [1])])
    data = _process(processor, circ)
    assert data.n_measures == 1
    assert len(data.non_measure_cmds) == 1


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
