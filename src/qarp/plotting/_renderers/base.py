import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, Tuple

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
