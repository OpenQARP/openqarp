"""Direct tests for BarrierRenderer.

Reached only through ``test_circuit_plotter.py`` otherwise, where a regression
surfaces as a whole-plot diff rather than a named failure.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from qarp.plotting import PlotConfig
from qarp.plotting._renderers.barrier import BarrierRenderer


class _Cmd:
    """Minimal stand-in for the plot adapter's command (``op`` + ``qubits``)."""

    def __init__(self, op, qubits):
        self.op = op
        self.qubits = qubits


def test_whole_register_barrier_spans_every_wire():
    """An empty-qubit Barrier (the whole-register-fence convention) must draw
    its dashed line across every wire, not just qubits it happens to name —
    it names none directly."""
    cmd = _Cmd("Barrier", [])
    y_positions = {0: 2.0, 1: 1.0, 2: 0.0}
    sentinel = (None, None)

    fig, ax = plt.subplots()
    try:
        result = BarrierRenderer(PlotConfig()).render(
            ax, cmd, x=0.5, y_positions=y_positions, prev_label_y=sentinel
        )
        assert result is sentinel  # label cursor passed through untouched, like global phase

        segments = [c.get_segments()[0] for c in ax.collections]
        assert len(segments) == 1
        (x0, y0), (x1, y1) = segments[0]
        assert x0 == x1 == 0.5
        assert sorted((y0, y1)) == [0.0 - 0.5, 2.0 + 0.5]
    finally:
        plt.close(fig)


def test_qubit_scoped_barrier_spans_only_its_own_qubits():
    """A Barrier built with explicit qubits (``qx.Command(GateType.Barrier, q)``)
    must span only the qubits it names, not the whole register."""
    cmd = _Cmd("Barrier", [1, 2])
    y_positions = {0: 2.0, 1: 1.0, 2: 0.0}

    fig, ax = plt.subplots()
    try:
        BarrierRenderer(PlotConfig()).render(
            ax, cmd, x=0.0, y_positions=y_positions, prev_label_y=(None, None)
        )
        (_, y0), (_, y1) = ax.collections[0].get_segments()[0]
        assert sorted((y0, y1)) == [0.0 - 0.5, 1.0 + 0.5]
    finally:
        plt.close(fig)
