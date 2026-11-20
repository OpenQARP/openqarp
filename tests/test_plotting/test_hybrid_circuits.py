"""Plots of hybrid programs and of labelled boxes.

A hybrid program measures mid-circuit, resets, and guards gates with classical
conditions; the plot has to keep those in time order and show which bit each
condition reads.  The expected conditions and bit numbers below are written
out by hand from the programs, and the label-fit checks compare against the
extents matplotlib reports for the rendered text.
"""

import matplotlib
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import qarpx as qx
from qarp.blocks import (
    CompositeBlock,
    ComputationalBasisStateBlock,
    ConditionalBlock,
    HEABlock,
    ReadoutBlock,
    SimpleBlock,
)
from qarp.plotting import CircuitAdapter, plot
from qarp.plotting._interactive import build_payload


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _measure(n_qubits, qubit, cbit=0):
    block = SimpleBlock(n_qubits)
    block.h(qubit).measure(qubit, cbit)
    return block.build()


def _gate(n_qubits, name, *qubits):
    block = SimpleBlock(n_qubits)
    getattr(block, name)(*qubits)
    return block.build()


def _feed_forward(else_body=None):
    """H and measure on q0 into c0, then X on q1 if c0 is 1."""
    cond = ConditionalBlock(
        cbits=[0], values=[True], then_body=_gate(2, "x", 1), else_body=else_body
    )
    cond.target_cbits = [0]
    return CompositeBlock([_measure(2, 0), cond], 2).build()


def _drawn(block, **kwargs):
    """Plot and return (plotter, axes) with the figure rendered."""
    plotter = plot(block, return_plotter=True, **kwargs)
    fig = plt.gcf()
    fig.canvas.draw()
    return plotter, fig.axes[0]


def _texts(ax):
    return [t.get_text() for t in ax.texts]


def _summary(block, **kwargs):
    """(gate, qubits, bits written, condition) per adapter command."""
    return [
        (str(c.op), [q.idx for q in c.qubits], c.cbits, c.condition)
        for c in CircuitAdapter(block, **kwargs).get_commands()
    ]


# ── Measurement and reset in time order ──────────────────────────────────────


def test_a_qubit_reused_after_measurement_plots_in_time_order():
    block = SimpleBlock(1)
    block.h(0).measure(0, 0)
    block.reset(0).x(0)
    block.build()

    plotter, ax = _drawn(block)

    assert [g.gate_type for g in plotter.gates] == ["H", "Measure", "Reset", "X"]
    xs = [g.x_position for g in plotter.gates]
    assert xs == sorted(xs) and len(set(xs)) == 4
    assert plotter._measure_payload == []  # no closing measurement column
    assert "|0>" in _texts(ax) and "Reset" not in _texts(ax)


def test_measuring_a_qubit_twice_plots_the_first_in_place():
    block = SimpleBlock(2)
    block.h(0).measure(0, 0).cx(0, 1).measure(0, 1).measure(1, 0)
    block.build()

    plotter, _ = _drawn(block)

    assert [g.gate_type for g in plotter.gates] == ["H", "Measure", "CX"]
    assert len(plotter._measure_payload) == 2  # q0 again, and q1


def test_a_gate_after_a_readout_block_plots():
    block = CompositeBlock([ReadoutBlock(n_qubits=2), _gate(2, "cx", 0, 1)], 2).build()
    plotter, _ = _drawn(block)
    assert [g.gate_type for g in plotter.gates] == ["Measure", "Measure", "Block"]


# ── Classical conditions ─────────────────────────────────────────────────────


@pytest.mark.parametrize("decompose_boxes", [False, True])
def test_a_conditional_gate_carries_its_condition(decompose_boxes):
    assert _summary(_feed_forward(), decompose_boxes=decompose_boxes) == [
        ("H", [0], [], None),
        ("Measure", [0], [0], None),
        ("X", [1], [], "c0=1"),
    ]


def test_an_else_arm_carries_the_negated_condition():
    program = _feed_forward(else_body=_gate(2, "h", 1))
    assert _summary(program)[2:] == [("X", [1], [], "c0=1"), ("H", [1], [], "c0=0")]


def test_a_two_bit_condition_is_an_and_and_its_else_a_not():
    cond = ConditionalBlock(
        cbits=[0, 1],
        values=[True, False],
        then_body=_gate(2, "x", 0),
        else_body=_gate(2, "z", 0),
    )
    cond.target_cbits = [0, 1]
    program = CompositeBlock([_measure(2, 0), _measure(2, 1), cond], 2).build()

    assert _summary(program)[4:] == [
        ("X", [0], [], "c0=1, c1=0"),
        ("Z", [0], [], "not(c0=1, c1=0)"),
    ]


def test_children_number_their_bits_in_the_composite():
    """Each child writes its own bit 0; the composite places them one after the
    other, so the second child's measurement is c1 and the condition reads it."""
    cond = ConditionalBlock(cbits=[0], values=[True], then_body=_gate(2, "x", 0))
    cond.target_cbits = [1]
    program = CompositeBlock([_measure(2, 0), _measure(2, 1), cond], 2).build()

    assert _summary(program) == [
        ("H", [0], [], None),
        ("Measure", [0], [0], None),
        ("H", [1], [], None),
        ("Measure", [1], [1], None),
        ("X", [0], [], "c1=1"),
    ]


def test_a_boxed_child_reports_the_bits_it_writes_and_its_condition():
    pre = SimpleBlock(3)
    pre.h(1).cx(1, 2).cx(0, 1).h(0)
    pre.measure(0, 0).measure(1, 1)
    pre.build()
    fix = ConditionalBlock(cbits=[0], values=[True], then_body=_gate(3, "cx", 1, 2))
    fix.target_cbits = [1]
    both = ConditionalBlock(
        cbits=[0],
        values=[False],
        then_body=_gate(3, "cx", 0, 2),
        else_body=_gate(3, "cz", 0, 2),
    )
    both.target_cbits = [0]
    program = CompositeBlock([pre, fix, both], 3).build()

    boxes = CircuitAdapter(program).get_commands()

    assert [str(b.op) for b in boxes] == ["Block", "Block", "Block"]
    assert [(b.cbits, b.condition, b.condition_bits) for b in boxes] == [
        ([0, 1], None, []),
        ([], "c1=1", [1]),
        ([], "c0=0 / else", [0]),
    ]


def test_the_classical_wire_names_the_bit_and_the_condition():
    plotter, ax = _drawn(_feed_forward())

    texts = _texts(ax)
    assert texts.count("c") == 2  # the wire's label at both ends
    assert "c0" in texts and "c0=1" in texts
    measure, gate = plotter.gates[1], plotter.gates[2]
    assert measure.x_position < gate.x_position  # written before it is read
    assert gate.condition == "c0=1" and measure.cbits == [0]


def test_no_classical_wire_without_a_condition():
    block = SimpleBlock(2)
    block.h(0).cx(0, 1).measure([(0, 0), (1, 1)])
    block.build()
    _, ax = _drawn(block)
    assert "c" not in _texts(ax)


def test_tooltips_name_the_classical_side():
    plotter, _ = _drawn(_feed_forward())
    payload = build_payload(plotter.gates, plotter._measure_payload)
    assert [payload[f"g{i}"]["label"] for i in range(3)] == ["H", "Measure -> c0", "X if c0=1"]


# ── Barriers ─────────────────────────────────────────────────────────────────


def _barrier(*qubits):
    cmd = qx.Command(qx.GateType.Barrier, *qubits) if qubits else qx.Command()
    cmd.gate = qx.GateType.Barrier
    return cmd


def test_a_barrier_is_a_line_not_a_gate():
    cmds = [qx.Command(qx.GateType.H, 0), _barrier(0, 1), qx.Command(qx.GateType.CX, 0, 1)]
    plotter, ax = _drawn(CircuitAdapter(cmds, n_qubits=2))

    assert [g.gate_type for g in plotter.gates] == ["H", "Barrier", "CX"]
    assert "Barrier" not in _texts(ax)
    squares = [line for line in ax.lines if line.get_marker() == "s"]
    assert len(squares) == 2  # H and the CX target: the barrier draws no box


def test_a_barrier_without_qubits_fences_every_wire():
    cmds = [qx.Command(qx.GateType.H, 0), _barrier(), qx.Command(qx.GateType.H, 2)]
    adapter = CircuitAdapter(cmds, n_qubits=3)
    assert [q.idx for q in adapter.get_commands()[1].qubits] == [0, 1, 2]

    plotter, _ = _drawn(adapter)
    xs = [g.x_position for g in plotter.gates]
    assert xs == sorted(xs) and len(set(xs)) == 3  # the second H waits for the barrier


# ── Labels fit their boxes ───────────────────────────────────────────────────


def _box_extents(ax):
    renderer = ax.figure.canvas.get_renderer()
    return [p.get_window_extent(renderer) for p in ax.patches if isinstance(p, Rectangle)]


def _holding_box(ax, text):
    """The box whose extent holds the rendered ``text``, or None."""
    renderer = ax.figure.canvas.get_renderer()
    (artist,) = [t for t in ax.texts if t.get_text() == text]
    label = artist.get_window_extent(renderer)
    for box in _box_extents(ax):
        if box.x0 <= label.x0 and label.x1 <= box.x1 and box.y0 <= label.y0 and label.y1 <= box.y1:
            return box
    return None


@pytest.mark.parametrize("figsize", [None, (10, 4), (2.5, 1.2)])
def test_a_parameterized_gate_label_fits_inside_its_box(figsize):
    """Also in a figure wider or narrower than the default one."""
    block = SimpleBlock(2)
    block.ry(0, 0.8).crz(0, 1, 0.3)
    block.build()

    _, ax = _drawn(block, figsize=figsize)

    assert _holding_box(ax, "Ry(0.8)") is not None
    assert _holding_box(ax, "RZ(0.3)") is not None


def test_block_boxes_hold_their_labels_and_stay_apart():
    """Each block label stays inside its own box, the boxes stay apart, and the
    measurements start right of the last box."""
    block = CompositeBlock(
        [
            ComputationalBasisStateBlock([1, 1, 0, 0]),
            HEABlock(4, 2, True, True, True, False),
            ReadoutBlock(n_qubits=4),
        ]
    ).build()

    plotter, ax = _drawn(block)

    assert [g.gate_type for g in plotter.gates] == ["Block", "Block"]
    labels = [t for t in _texts(ax) if len(t) > 1 and not t.startswith("q[")]
    first, second = (_holding_box(ax, label) for label in labels)
    assert first is not None and second is not None
    assert first.x1 < second.x0
    meter_x = [line.get_xdata()[0] for line in ax.lines if line.get_marker() == "s"]
    right_edge = max(p.get_x() + p.get_width() for p in ax.patches if isinstance(p, Rectangle))
    assert len(meter_x) == 4 and min(meter_x) > right_edge


def test_a_controlled_block_box_holds_its_label():
    from qarp.blocks import ControlledBlock, HnBlock

    controlled = ControlledBlock(HnBlock(2).build(), 1, [True])
    block = CompositeBlock([controlled], 3).build()

    plotter, ax = _drawn(block)

    assert [g.gate_type for g in plotter.gates] == ["ControlledBlock"]
    (label,) = [t for t in _texts(ax) if len(t) > 1 and not t.startswith("q[")]
    assert _holding_box(ax, label) is not None


def test_a_label_above_a_two_qubit_gate_clears_the_next_column():
    """Rzz draws its label between its two wires; the H after it starts right
    of where that label ends."""
    block = SimpleBlock(2)
    block.rzz(0, 1, 0.123).h(0)
    block.build()

    _, ax = _drawn(block)

    renderer = ax.figure.canvas.get_renderer()
    (label,) = [t for t in ax.texts if t.get_text().upper().startswith("RZZ")]
    (h_text,) = [t for t in ax.texts if t.get_text() == "H"]
    assert label.get_window_extent(renderer).x1 < h_text.get_window_extent(renderer).x0


def _reset_block():
    block = SimpleBlock(1)
    block.x(0).measure(0, 0).reset(0).h(0)
    return block.build()


def test_the_reset_label_is_math_under_latex():
    """TeX text mode sets ``|`` and ``>`` as other glyphs, so the label goes
    through math mode there; the box is laid out without running TeX."""
    plot(_reset_block(), use_latex=True)
    texts = _texts(plt.gcf().axes[0])
    assert r"$|0\rangle$" in texts and "|0>" not in texts


def test_a_reset_renders_with_latex():
    import shutil

    if not (shutil.which("latex") and shutil.which("dvipng")):
        pytest.skip("needs a LaTeX installation")
    _drawn(_reset_block(), use_latex=True)


def test_a_filled_scheme_draws_a_hybrid_program():
    """The filled schemes draw the meter as a ``>`` marker; a measurement in
    place uses it too, and the classical side is still named."""
    _, ax = _drawn(_feed_forward(), color_scheme="classic")
    assert [line.get_marker() for line in ax.lines].count(">") == 1
    assert "c0=1" in _texts(ax)


def test_a_measurement_in_place_takes_a_gate_index():
    """It is indexed and can be queried like a gate; closing measurements are not."""
    block = SimpleBlock(2)
    block.h(0).measure(0, 0).x(0).measure(1, 1)
    block.build()

    plotter, ax = _drawn(block, show_gate_indices=True)

    assert plotter.query_gate(1).gate_type == "Measure"
    assert [g.index for g in plotter.gates] == [0, 1, 2]
    assert {"0", "1", "2"} <= set(_texts(ax)) and "3" not in _texts(ax)


@pytest.mark.parametrize("figsize", [None, (12, 3), (3, 1.5)])
def test_qubit_labels_stay_inside_the_figure(figsize):
    block = SimpleBlock(2)
    block.h(0).cx(0, 1)
    block.build()

    _, ax = _drawn(block, figsize=figsize)

    renderer = ax.figure.canvas.get_renderer()
    width = ax.figure.bbox.width
    for label in (t for t in ax.texts if t.get_text().startswith("q[")):
        extent = label.get_window_extent(renderer)
        assert 0 <= extent.x0 and extent.x1 <= width


# ── Composite frames ─────────────────────────────────────────────────────────


def test_boxed_and_decomposed_plots_agree_on_the_classical_bits():
    """The decomposed plot reads the bits from the program the core flattens;
    the boxed plot places each child's bits itself.  A child with
    ``target_cbits`` in between takes no range of its own."""
    first = ConditionalBlock(cbits=[0], values=[True], then_body=_gate(2, "x", 1))
    first.target_cbits = [0]
    second = ConditionalBlock(cbits=[0], values=[False], then_body=_gate(2, "z", 0))
    second.target_cbits = [1]
    program = CompositeBlock([_measure(2, 0), first, _measure(2, 1), second], 2).build()

    boxed = _summary(program)

    assert boxed == _summary(program, decompose_boxes=True)
    assert [row[2:] for row in boxed] == [
        ([], None),
        ([0], None),
        ([], "c0=1"),
        ([], None),
        ([1], None),
        ([], "c1=0"),
    ]


def test_a_composite_with_its_own_target_qubits_plots_in_its_own_frame():
    block = CompositeBlock([_gate(2, "h", 0), _gate(2, "cx", 0, 1)], 2)
    block.target_qubits = [3, 4]
    block.build()

    adapter = CircuitAdapter(block)

    assert adapter.n_qubits == 2
    assert _summary(block) == [("H", [0], [], None), ("Block", [0, 1], [], None)]


def test_a_pending_dagger_keeps_each_child_on_its_own_wires():
    two = SimpleBlock(3)
    two.cx(1, 2).t(2)
    two.build()
    block = CompositeBlock([_gate(3, "h", 0), two], 3).build().dagger()

    drawn = [(row[0], row[1]) for row in _summary(block)]

    assert sorted(drawn) == [("Block", [0, 1, 2]), ("H", [0])]


def test_a_conditional_with_only_an_else_arm_shows_the_negated_condition():
    cond = ConditionalBlock(
        cbits=[0], values=[True], then_body=None, else_body=_gate(3, "cx", 1, 2)
    )
    cond.target_cbits = [0]
    program = CompositeBlock([_measure(3, 0), cond], 3).build()

    box = CircuitAdapter(program).get_commands()[-1]

    assert (str(box.op), box.condition) == ("Block", "c0=0")


def test_with_inverted_order_a_link_does_not_cross_a_gate_under_it():
    """Inverted, q0 is the lowest wire: the link from the measurement on q2
    runs down through q0, so the second H on q0 takes another column."""
    block = SimpleBlock(3)
    block.h(2).h(0).measure(2, 0).h(0)
    block.build()
    cond = ConditionalBlock(cbits=[0], values=[True], then_body=_gate(3, "x", 1))
    cond.target_cbits = [0]
    program = CompositeBlock([block, cond], 3).build()

    plotter, _ = _drawn(program, decompose_boxes=True, invert_order=True)

    (measure,) = [g for g in plotter.gates if g.gate_type == "Measure"]
    on_q0 = [g.x_position for g in plotter.gates if g.qubits == ["q[0]"]]
    assert len(on_q0) == 2 and measure.x_position not in on_q0
