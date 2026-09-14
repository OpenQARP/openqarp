from matplotlib.patches import Rectangle

from ..styles import get_text_color
from .base import GateRenderer, LabelInfo


class BoxedGateRenderer(GateRenderer):
    """Renderer for boxed gates."""

    def render(self, ax, cmd, x: float, y_positions: dict, prev_label_y: LabelInfo) -> LabelInfo:
        original_label = getattr(cmd.op, "circuit_name", "_no_name_")
        if original_label is None:
            print(f"Gate with None circuit_name: {cmd.op}, type: {type(cmd.op)}")
            original_label = "_no_name_"
        is_identity = original_label.lower() == "identity"

        is_empty_box = False
        if hasattr(cmd.op, "get_circuit"):
            try:
                circuit = cmd.op.get_circuit()
                is_empty_box = len(circuit.get_commands()) == 0
            except Exception:
                pass  # If we can't access the circuit, assume it's not empty

        if is_identity and not is_empty_box:
            raise ValueError("Identity gate is not empty.")

        y_coords = [y_positions[q] for q in cmd.qubits]
        y_min, y_max = min(y_coords), max(y_coords)
        padding = 0.4
        box_height = max(y_max - y_min + 2 * padding, 1.0)
        y_box_min = (y_min + y_max - box_height) / 2
        base_y = (y_box_min + y_box_min + box_height) / 2

        # Use new label positioning system
        y_text, x_offset, label = self._get_label_position(x, base_y, original_label, prev_label_y)
        is_short_text = len(label) < self.config.max_chars

        # Draw identity box or empty block box as transparent with dashed outline
        # the rest as white with solid outline
        if is_identity:
            facecolor = "none"
            linestyle = "--"
        else:
            facecolor = self.config.gate_colors["box"]  # type: ignore[index]
            linestyle = "-"

        ax.add_patch(
            Rectangle(
                (x - self.config.spacing / 3, y_box_min),
                self.config.spacing * 2 / 3,
                box_height,
                facecolor=facecolor,
                edgecolor=self.config.structure_color,
                linewidth=self.config.box_linewidth,
                linestyle=linestyle,
                zorder=1,
            )
        )

        # Draw connector line if label is significantly displaced
        self._draw_label_connector(ax, x, base_y, y_text)

        # Determine text color based on box background
        box_color = (self.config.gate_colors or {}).get("box", "#EFEFEF")  # type: ignore[union-attr]
        if is_identity or self.config.gate_style == "outline":
            text_color = self.config.text_color
        else:
            text_color = get_text_color(box_color)

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
        for qubit in cmd.qubits:
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
