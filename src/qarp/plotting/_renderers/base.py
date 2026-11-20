import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from matplotlib.patches import Rectangle

from .. import _geometry as geometry
from .._config import PlotConfig

if TYPE_CHECKING:
    from .label_manager import LabelManager

# (label y-position, is-short-text) threaded between consecutive gate renders.
LabelInfo = Tuple[Optional[float], Optional[bool]]


class GateRenderer(ABC):
    def __init__(self, config: PlotConfig, label_manager: Optional["LabelManager"] = None):
        """Abstract base class for gate renderers.

        Args:
            config: Plot configuration settings.
            label_manager: Optional label manager for smart label positioning.
                          If not provided, the fallback overlap-avoidance positioning is used.
        """
        self.config = config
        self._label_manager = label_manager

    def set_label_manager(self, label_manager: "LabelManager") -> None:
        """Set the label manager for this renderer."""
        self._label_manager = label_manager

    @abstractmethod
    def render(self, ax, cmd, x: float, y_positions: Dict, prev_label_y: LabelInfo) -> LabelInfo:
        """Render a gate and return label position info."""
        pass

    def footprint(self, cmd) -> float:
        """Width the drawn command needs along the circuit, in data units."""
        return geometry.pt_to_x(self.config.gate_size)

    def _gate_box_size(self, label: str) -> Tuple[float, float, bool]:
        """(width, height) in points of the box holding ``label`` inside it, and
        whether that is the plain square marker."""
        side = float(self.config.gate_size)
        width = geometry.text_width_pt(label, self.config.text_size)
        if "\n" not in label and width + geometry.BOX_PAD_PT <= side:
            return side, side, True
        height = geometry.text_height_pt(label, self.config.text_size) + geometry.BOX_PAD_PT
        return width + 2 * geometry.BOX_PAD_PT, max(side, height), False

    def _draw_gate_box(self, ax, x: float, y: float, label: str, face: str, edge: str) -> None:
        """Draw the box of a gate: the square marker, or a box widened to its label."""
        width, height, square = self._gate_box_size(label)
        if square:
            ax.plot(
                x,
                y,
                "s",
                mfc=face,
                mec=edge,
                mew=self.config.gate_edge_width,
                markersize=self.config.gate_size,
            )
            return
        w, h = geometry.pt_to_x(width, ax), geometry.pt_to_y(height, ax)
        ax.add_patch(
            Rectangle(
                (x - w / 2, y - h / 2),
                w,
                h,
                facecolor=face,
                edgecolor=edge,
                linewidth=self.config.gate_edge_width,
                zorder=2,
            )
        )

    def _block_box_width(self, label: str, qubits, ax=None) -> float:
        """Width in data units of a block box: its label between two gutters,
        the left one holding the qubit numbers."""
        digits = max((len(str(q.index[0])) for q in qubits), default=1)
        gutter = geometry.text_width_pt("0" * digits, self.config.text_size * 0.7) + 8
        width = geometry.text_width_pt(label, self.config.text_size) + 2 * gutter
        return max(self.config.spacing * 2 / 3, geometry.pt_to_x(width, ax))

    def _get_label_position(
        self, x: float, base_y: float, label: str, prev_label_y: LabelInfo
    ) -> Tuple[float, float, str]:
        """Get the optimal label position using the label manager, or the fallback method when none is set.

        Args:
            x: X position of the gate
            base_y: Y position of the gate
            label: Original label text
            prev_label_y: Previous label's (y, is_short_text); used by the fallback positioner

        Returns:
            Tuple of (y_position, x_offset, formatted_label)
        """
        if self._label_manager is not None:
            return self._label_manager.compute_position(x, base_y, label)
        else:
            # Fallback: no label manager, so place labels with simple overlap avoidance.
            formatted_label = self._format_label(label)
            is_short_text = len(formatted_label) < self.config.max_chars
            y = self._adjust_text_position(prev_label_y, base_y, is_short_text)
            return y, 0.0, formatted_label

    def _draw_label_connector(self, ax, x: float, gate_y: float, label_y: float) -> None:
        """Draw connector line if label is displaced (when using label manager)."""
        if self._label_manager is not None:
            self._label_manager.draw_label_connector(ax, x, gate_y, label_y)

    def _adjust_text_position(
        self,
        prev_label_y: LabelInfo,
        curr_y: float,
        curr_is_short_text: bool,
        threshold: float = 0.5,
        shift_val: float = 0.6,
    ) -> float:
        """Adjust text position to avoid overlaps (fallback positioner)."""
        prev_y, prev_is_short_text = prev_label_y
        if (
            prev_y is not None
            and not prev_is_short_text
            and not curr_is_short_text
            and abs(prev_y - curr_y) < threshold
        ):
            return curr_y + shift_val if prev_y <= curr_y else curr_y - shift_val
        return curr_y

    def _format_label(self, label: str) -> str:
        """Format label by rounding numerical parameters to configured decimal places."""
        # If using label manager, delegate to it
        if self._label_manager is not None:
            return self._label_manager.format_label(label)

        # Fallback formatting when no label manager is set.
        decimal_places = self.config.label_decimal_places

        def round_match(match: re.Match) -> str:
            num_str = match.group(0)
            try:
                num = float(num_str)
                # Format with specified decimal places, removing trailing zeros
                formatted = f"{num:.{decimal_places}f}".rstrip("0").rstrip(".")
                return formatted
            except ValueError:
                return num_str

        # Match floating point numbers (including negative and scientific notation)
        pattern = r"-?\d+\.\d+(?:[eE][+-]?\d+)?"
        return re.sub(pattern, round_match, label)
