"""Direct tests for the layout helpers.

These were reached only through ``test_circuit_plotter.py``, where a layout
regression surfaces as a whole-plot diff rather than a named failure.  The
three functions are pure, so the expected layouts here are computed by hand.
"""

from qarp.plotting._processors.positioning import (
    flatten_qubits,
    get_x_positions_list,
    get_y_positions,
)


class _Cmd:
    """Minimal stand-in for the plot adapter's command (``op`` + ``qubits``)."""

    def __init__(self, op, qubits):
        self.op = op
        self.qubits = qubits


def test_flatten_qubits_flattens_arbitrary_nesting():
    assert flatten_qubits([0, [1, 2], [[3], [4, [5]]]]) == [0, 1, 2, 3, 4, 5]


def test_flatten_qubits_treats_strings_as_scalars():
    """A str is Iterable; recursing into it would explode a label into chars."""
    assert flatten_qubits(["q0", ["q1", "q2"]]) == ["q0", "q1", "q2"]


def test_get_y_positions_puts_qubit_zero_at_the_top_by_default():
    # n − 1 − i: the first qubit gets the highest y, so it draws topmost.
    assert get_y_positions([0, 1, 2]) == {0: 2, 1: 1, 2: 0}


def test_get_y_positions_inverted_counts_upward():
    assert get_y_positions([0, 1, 2], invert_order=True) == {0: 0, 1: 1, 2: 2}


def test_disjoint_gates_share_one_layer():
    """Gates on disjoint wires occupy the same column: both at 0 + 1/4."""
    cmds = [_Cmd("h", [0]), _Cmd("h", [1])]
    assert get_x_positions_list(cmds, [0, 1]) == [0.25, 0.25]


def test_gates_on_one_wire_stack_in_successive_layers():
    cmds = [_Cmd("h", [0]), _Cmd("x", [0]), _Cmd("z", [0])]
    assert get_x_positions_list(cmds, [0]) == [0.25, 1.25, 2.25]


def test_a_two_qubit_gate_blocks_every_wire_it_spans():
    """cx(0, 2) spans q1 too, so a later h(1) cannot share its column."""
    cmds = [_Cmd("cx", [0, 2]), _Cmd("h", [1])]
    assert get_x_positions_list(cmds, [0, 1, 2]) == [0.25, 1.25]


def test_measure_commands_take_no_column():
    cmds = [_Cmd("Measure", [0]), _Cmd("h", [0])]
    assert get_x_positions_list(cmds, [0]) == [0.25]


def test_qubitless_commands_are_skipped():
    """A global-phase command carries no wire and must not consume a column."""
    cmds = [_Cmd("gphase", []), _Cmd("h", [0])]
    assert get_x_positions_list(cmds, [0]) == [0.25]
