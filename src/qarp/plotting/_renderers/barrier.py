from .base import GateRenderer, LabelInfo


class BarrierRenderer(GateRenderer):
    """Renderer for Barrier commands: a dashed vertical fence line.

    An empty qubit list is the whole-register-fence convention (``qx.Command()``'s
    default), spanning every wire; a non-empty list spans just the qubits it names.
    """

    def render(self, ax, cmd, x: float, y_positions: dict, prev_label_y: LabelInfo) -> LabelInfo:
        ys = [y_positions[q] for q in cmd.qubits] if cmd.qubits else list(y_positions.values())
        y_min, y_max = min(ys), max(ys)

        ax.vlines(
            x,
            y_min - 0.5,
            y_max + 0.5,
            color=self.config.muted_color,
            linestyle="--",
            linewidth=1.2,
        )

        return prev_label_y
