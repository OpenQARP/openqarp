from matplotlib.patches import Rectangle

from ..styles import get_text_color
from .base import GateRenderer, LabelInfo


class ControlledBlockRenderer(GateRenderer):
    """Renderer for quantum-controlled block boxes."""

    def render(self, ax, cmd, x: float, y_positions: dict, prev_label_y: LabelInfo) -> LabelInfo:
        original_label = str(cmd.op.get_op().get_circuit().name)
        n_controls = cmd.op.get_n_controls()
        # LSB-first per conventions §6: bit k is control qubit k, matching the
        # order of cmd.qubits.  bin()/zfill would read it MSB-first.
        packed = cmd.op.get_control_state()
        control_state = [(packed >> k) & 1 for k in range(n_controls)]

        y_coords = [y_positions[q] for q in cmd.qubits]
        ctrl_y = y_coords[:n_controls]
        tgt_y = y_coords[n_controls:]

        y_min, y_max = min(tgt_y), max(tgt_y)
        padding = 0.4
        box_height = max(y_max - y_min + 2 * padding, 1.0)
        y_box_min = (y_min + y_max - box_height) / 2
        base_y = (y_box_min + y_box_min + box_height) / 2

        # Use new label positioning system
        y_text, x_offset, label = self._get_label_position(x, base_y, original_label, prev_label_y)
        is_short_text = len(label) < self.config.max_chars

        structure = self.config.structure_color

        # Draw control line
        ax.vlines(
            x, min(ctrl_y + tgt_y), max(ctrl_y + tgt_y), color=structure, linewidth=1.2, zorder=1
        )

        # Draw control dots with proper state
        for idx, y in enumerate(ctrl_y):
            if control_state[idx] == 1:
                ax.plot(
                    x,
                    y,
                    "o",
                    color=structure,
                    markersize=self.config.control_size,
                    zorder=1,
                )
            elif control_state[idx] == 0:
                ax.plot(
                    x,
                    y,
                    "o",
                    color=structure,
                    markersize=self.config.control_size,
                    mfc="white",
                    zorder=1,
                )

        # Draw box
        ax.add_patch(
            Rectangle(
                (x - self.config.spacing / 3, y_box_min),
                self.config.spacing * 2 / 3,
                box_height,
                facecolor=self.config.gate_colors["box"],  # type: ignore[index]
                edgecolor=structure,
                linewidth=self.config.box_linewidth,
                zorder=2,
            )
        )

        # Draw connector line if label is significantly displaced
        self._draw_label_connector(ax, x, base_y, y_text)

        # Determine text color based on box background
        box_color = self.config.gate_colors.get(  # type: ignore
            "box", "#EFEFEF"
        )  # type: ignore[union-attr]
        text_color = (
            self.config.text_color
            if self.config.gate_style == "outline"
            else get_text_color(box_color)
        )

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

        # Add qubit numbers at the corresponding y positions
        for qubit in cmd.qubits[n_controls:]:
            qubit_y = y_positions[qubit]
            number_x = x - (self.config.spacing / 3) * 0.9

            ax.text(
                number_x,
                qubit_y,
                str(qubit.index[0]),
                fontsize=self.config.text_size * 0.7,
                ha="left",
                va="center",
                color=self.config.muted_color,
                weight="light",
                zorder=3,
            )

        return (y_text, is_short_text)
