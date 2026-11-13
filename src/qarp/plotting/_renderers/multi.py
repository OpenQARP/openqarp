from .base import GateRenderer, LabelInfo


class MultiGateRenderer(GateRenderer):
    """Renderer for multi-qubit gates."""

    def render(self, ax, cmd, x: float, y_positions: dict, prev_label_y: LabelInfo) -> LabelInfo:
        original_label = str(cmd.op)
        y_coords = [y_positions[q] for q in cmd.qubits]
        y_min, y_max = min(y_coords), max(y_coords)
        base_y = (y_min + y_max) / 2
        structure = self.config.structure_color

        # Use new label positioning system
        y_text, x_offset, label = self._get_label_position(x, base_y, original_label, prev_label_y)
        is_short_text = len(label) < self.config.max_chars

        ax.vlines(x, y_min, y_max, color=structure, linewidth=1.2)

        if label.lower() == "swap":
            for y in y_coords:
                ax.plot(x, y, "x", color=structure, markersize=self.config.swap_size)
        elif label.lower() == "cswap":
            for i, y in enumerate(y_coords):
                if i == 0:
                    ax.plot(x, y, "o", color=structure, markersize=self.config.control_size)
                else:
                    ax.plot(x, y, "x", color=structure, markersize=self.config.swap_size)
        else:
            for y in y_coords:
                ax.plot(
                    x,
                    y,
                    "s",
                    mec=structure,
                    mew=self.config.gate_edge_width,
                    mfc="white",
                    markersize=self.config.gate_size,
                )

            # Draw connector line if label is significantly displaced
            self._draw_label_connector(ax, x, base_y, y_text)

            ax.text(
                x + x_offset,
                y_text,
                label,
                ha="center",
                va="bottom",
                fontsize=self.config.text_size,
                color=self.config.text_color,
                zorder=3,
            )

        return (y_text, is_short_text)
