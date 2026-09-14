"""Direct tests for the renderer base class and the global-phase renderer.

``base.py``, ``boxed.py`` and ``global_phase.py`` were reached only through
``test_circuit_plotter.py``, where a regression surfaces as a plot-level diff.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from qarp.blocks import SimpleBlock
from qarp.plotting import CircuitAdapter, PlotConfig
from qarp.plotting._renderers.base import GateRenderer
from qarp.plotting._renderers.global_phase import GlobalPhaseRenderer


class _Renderer(GateRenderer):
    """Concrete stand-in: GateRenderer is abstract on ``render`` alone."""

    def render(self, ax, cmd, x, y_positions, prev_label_y):
        return prev_label_y


def test_gate_renderer_is_abstract_on_render():
    with pytest.raises(TypeError):
        GateRenderer(PlotConfig())


def test_format_label_rounds_floats_to_the_configured_precision():
    config = PlotConfig()
    config.label_decimal_places = 2
    renderer = _Renderer(config)
    # Only the numeric run is rewritten; the gate name is left alone.
    assert renderer._format_label("Rz(3.14159)") == "Rz(3.14)"


def test_format_label_strips_trailing_zeros():
    config = PlotConfig()
    config.label_decimal_places = 3
    renderer = _Renderer(config)
    assert renderer._format_label("Rx(0.5000)") == "Rx(0.5)"
    assert renderer._format_label("Ry(2.0)") == "Ry(2)"


def test_format_label_handles_negative_and_scientific_notation():
    config = PlotConfig()
    config.label_decimal_places = 2
    renderer = _Renderer(config)
    assert renderer._format_label("Rz(-1.5708)") == "Rz(-1.57)"
    assert renderer._format_label("Rz(1.0e-3)") == "Rz(0)"


def test_format_label_leaves_integers_untouched():
    """The pattern requires a decimal point, so a bare integer is not a match."""
    renderer = _Renderer(PlotConfig())
    assert renderer._format_label("CX q0 q1") == "CX q0 q1"
    assert renderer._format_label("P(2)") == "P(2)"


def test_format_label_delegates_to_the_label_manager_when_one_is_set():
    class _Manager:
        def format_label(self, label):
            return f"managed:{label}"

    renderer = _Renderer(PlotConfig())
    renderer.set_label_manager(_Manager())
    assert renderer._format_label("Rz(3.14159)") == "managed:Rz(3.14159)"


def test_global_phase_renderer_draws_nothing_and_passes_the_label_state_through():
    """A global phase acts on no qubit, so it must consume no ink and leave the
    label cursor where it was — otherwise it shifts every later label."""
    block = SimpleBlock(1)
    block.h(0)
    block.build()
    cmd = CircuitAdapter(block).get_commands()[0]

    fig, ax = plt.subplots()
    try:
        before = len(ax.get_children())
        sentinel = (1.0, True)
        result = GlobalPhaseRenderer(PlotConfig()).render(
            ax, cmd, x=0.0, y_positions={0: 0.0}, prev_label_y=sentinel
        )
        assert result is sentinel
        assert len(ax.get_children()) == before
    finally:
        plt.close(fig)


class _Op:
    def __init__(self, circuit_name, n_commands=0):
        self.circuit_name = circuit_name
        self._n_commands = n_commands

    def get_circuit(self):
        class _Circ:
            def __init__(self, n):
                self._n = n

            def get_commands(self):
                return [None] * self._n

        return _Circ(self._n_commands)


class _Qubit:
    """Adapter qubit duck type: hashable, and carries an ``index`` tuple."""

    def __init__(self, i):
        self.index = (i,)

    def __hash__(self):
        return hash(self.index)

    def __eq__(self, other):
        return isinstance(other, _Qubit) and self.index == other.index


class _BoxCmd:
    def __init__(self, op, qubits):
        self.op = op
        self.qubits = [_Qubit(q) for q in qubits]


def test_boxed_renderer_refuses_a_non_empty_identity_box():
    """An "identity" box that still carries commands is a contradiction: the
    drawing would claim the block is empty when it is not."""
    from qarp.plotting._renderers.boxed import BoxedGateRenderer

    fig, ax = plt.subplots()
    try:
        cmd = _BoxCmd(_Op("Identity", n_commands=3), [0, 1])
        with pytest.raises(ValueError, match="Identity gate is not empty"):
            BoxedGateRenderer(PlotConfig()).render(
                ax,
                cmd,
                x=0.0,
                y_positions={_Qubit(0): 1.0, _Qubit(1): 0.0},
                prev_label_y=(None, None),
            )
    finally:
        plt.close(fig)


def test_boxed_renderer_draws_a_rectangle_spanning_its_qubits():
    from matplotlib.patches import Rectangle

    from qarp.plotting._renderers.boxed import BoxedGateRenderer

    fig, ax = plt.subplots()
    try:
        cmd = _BoxCmd(_Op("MyBlock", n_commands=2), [0, 1])
        BoxedGateRenderer(PlotConfig()).render(
            ax, cmd, x=0.0, y_positions={_Qubit(0): 1.0, _Qubit(1): 0.0}, prev_label_y=(None, None)
        )
        rectangles = [c for c in ax.get_children() if isinstance(c, Rectangle)]
        # One box for the gate (plus matplotlib's own axes patch).
        assert any(r.get_height() >= 1.0 for r in rectangles)
    finally:
        plt.close(fig)
