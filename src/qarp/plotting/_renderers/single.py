from ..styles import get_text_color
from .base import GateRenderer, LabelInfo


class SingleGateRenderer(GateRenderer):
    """Renderer for single-qubit gates."""

    def render(self, ax, cmd, x: float, y_positions: dict, prev_label_y: LabelInfo) -> LabelInfo:
        original_label = str(cmd.op)
        y = y_positions[cmd.qubits[0]]

        # Use new label positioning system
        y_text, x_offset, label = self._get_label_position(x, y, original_label, prev_label_y)

        is_short_text = len(label) < self.config.max_chars
        name = label.lower().split("(")[0]

        color = self.config.gate_color(name)
        face, edge = self.config.gate_face_edge(color)
        ax.plot(
            x,
            y,
            "s",
            mfc=face,
            mec=edge,
            mew=self.config.gate_edge_width,
            markersize=self.config.gate_size,
        )

        # Draw connector line if label is significantly displaced
        self._draw_label_connector(ax, x, y, y_text)

        # In-gate text sits on the (tinted) face; displaced labels use plain ink.
        if abs(y_text - y) < 0.1:
            text_color = (
                self.config.text_color
                if self.config.gate_style == "outline"
                else get_text_color(face)
            )
        else:
            text_color = self.config.text_color

        ax.text(
            x + x_offset,
            y_text,
            label,
            ha="center",
            va="center",
            fontsize=self.config.text_size,
            color=text_color,
            zorder=3,
        )

        return (y_text, is_short_text)
