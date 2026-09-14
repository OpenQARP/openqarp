from .base import GateRenderer, LabelInfo


class GlobalPhaseRenderer(GateRenderer):
    """Renderer for global phase gates (gates with no qubits)."""

    def render(self, ax, cmd, x: float, y_positions: dict, prev_label_y: LabelInfo) -> LabelInfo:
        """Render a global phase gate.

        Global phase gates don't act on any qubits, so they are not visually
        rendered in the circuit diagram. We simply return the previous label
        info unchanged.

        Args:
            ax: Matplotlib axes to render on.
            cmd: The gate command.
            x: X position (unused for global phase).
            y_positions: Dictionary mapping qubits to y positions.
            prev_label_y: Previous label positioning info.

        Returns:
            The unchanged prev_label_y tuple.
        """
        # Global phase gates are not rendered visually
        # They don't act on any qubits, so nothing to draw
        return prev_label_y
