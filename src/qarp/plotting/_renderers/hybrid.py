"""Renderers for the non-unitary commands of a hybrid program."""

import matplotlib as mpl

from .. import _geometry as geometry
from .._config import PlotConfig
from ..styles import get_text_color
from .base import GateRenderer, LabelInfo

RESET_LABEL = "|0>"


def _reset_text() -> str:
    """The reset label as drawn: TeX text mode has no ``|`` or ``>``."""
    return r"$|0\rangle$" if mpl.rcParams["text.usetex"] else RESET_LABEL


def draw_meter(ax, config: PlotConfig, x: float, y: float) -> None:
    """Draw a measurement on one wire: the amber meter box."""
    assert config.gate_colors is not None  # __post_init__ fills it
    measure_color = config.gate_colors["measure"]
    if config.gate_style == "outline":
        # Meter box: amber outline on a tinted face, ink glyph.
        face, edge = config.gate_face_edge(measure_color)
        ax.plot(
            x, y, "s", mfc=face, mec=edge, mew=config.gate_edge_width, markersize=config.gate_size
        )
        ax.text(
            x, y, "M", ha="center", va="center", fontsize=config.text_size, color=config.text_color
        )
    else:
        ax.plot(x, y, ">", color=measure_color, markersize=config.gate_size)
        ax.text(
            x,
            y,
            "M",
            ha="right",
            va="center",
            fontsize=config.text_size,
            color=get_text_color(measure_color),
        )


class MeasureRenderer(GateRenderer):
    """Renderer for a measurement drawn where it happens."""

    def render(self, ax, cmd, x: float, y_positions: dict, prev_label_y: LabelInfo) -> LabelInfo:
        draw_meter(ax, self.config, x, y_positions[cmd.qubits[0]])
        return prev_label_y


class ResetRenderer(GateRenderer):
    """Renderer for a reset: a box in the measurement color holding ``|0>``."""

    def footprint(self, cmd) -> float:
        return geometry.pt_to_x(self._gate_box_size(RESET_LABEL)[0])

    def render(self, ax, cmd, x: float, y_positions: dict, prev_label_y: LabelInfo) -> LabelInfo:
        y = y_positions[cmd.qubits[0]]
        face, edge = self.config.gate_face_edge(self.config.gate_color("reset"))
        self._draw_gate_box(ax, x, y, RESET_LABEL, face, edge)
        text_color = (
            self.config.text_color if self.config.gate_style == "outline" else get_text_color(face)
        )
        ax.text(
            x,
            y,
            _reset_text(),
            ha="center",
            va="center",
            fontsize=self.config.text_size,
            color=text_color,
            zorder=3,
        )
        return prev_label_y


class BarrierRenderer(GateRenderer):
    """Renderer for a barrier: a dashed line across the wires it fences."""

    def footprint(self, cmd) -> float:
        return 0.0

    def render(self, ax, cmd, x: float, y_positions: dict, prev_label_y: LabelInfo) -> LabelInfo:
        for qubit in cmd.qubits:
            y = y_positions[qubit]
            ax.vlines(
                x,
                y - 0.5,
                y + 0.5,
                color=self.config.muted_color,
                linewidth=1.2,
                linestyles=(0, (3, 2)),
                zorder=1,
            )
        return prev_label_y
