from ..styles import get_text_color
from .base import GateRenderer, LabelInfo


class ControlledGateRenderer(GateRenderer):
    """Renderer for controlled gates."""

    # Controlled-Z is symmetric in all its qubits, so it has no distinguished
    # target: the standard notation is a control dot on every wire joined by a
    # single vertical line.
    _SYMMETRIC_CONTROL = {"cz", "mcz"}

    def render(self, ax, cmd, x: float, y_positions: dict, prev_label_y: LabelInfo) -> LabelInfo:
        original_label = str(cmd.op)
        # Stem drops any "(0.3)" suffix: it selects the palette entry, while the
        # drawn label keeps the parameter.
        stem = original_label.lower().split("(")[0]
        structure = self.config.structure_color

        y_coords = [y_positions[q] for q in cmd.qubits]

        if stem in self._SYMMETRIC_CONTROL:
            ax.vlines(x, min(y_coords), max(y_coords), color=structure, linewidth=1.2)
            for y in y_coords:
                ax.plot(x, y, "o", color=structure, markersize=self.config.control_size)
            return prev_label_y

        n_prefix = 2 if stem.startswith("cc") else 1
        base = stem[n_prefix:]
        ctrl_y = y_coords[:-1]
        tgt_y = y_coords[-1]

        # Use new label positioning system - use base name as the label
        base_label = original_label[n_prefix:]
        y_text, x_offset, label = self._get_label_position(
            x, tgt_y, base_label.upper(), prev_label_y
        )
        is_short_text = len(label) < self.config.max_chars

        # Draw control line
        ax.vlines(x, min(ctrl_y + [tgt_y]), max(ctrl_y + [tgt_y]), color=structure, linewidth=1.2)

        # Draw control dots
        for y in ctrl_y:
            ax.plot(x, y, "o", color=structure, markersize=self.config.control_size)

        # Draw target
        color = self.config.gate_color(base)
        face, edge = self.config.gate_face_edge(color)
        ax.plot(
            x,
            tgt_y,
            "s",
            mfc=face,
            mec=edge,
            mew=self.config.gate_edge_width,
            markersize=self.config.gate_size,
        )

        # Draw connector line if label is significantly displaced
        self._draw_label_connector(ax, x, tgt_y, y_text)

        if abs(y_text - tgt_y) < 0.1:
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
            fontsize=self.config.text_size,
            ha="center",
            va="center",
            color=text_color,
            zorder=3,
        )

        return (y_text, is_short_text)
