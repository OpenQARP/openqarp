"""Label manager for context-aware label positioning in circuit plots."""

from dataclasses import dataclass
from typing import List, Tuple

from .._config import LabelMode, PlotConfig


@dataclass
class LabelInfo:
    """Information about a rendered label for tracking and overlap avoidance."""

    x: float  # X position of the label
    y: float  # Y position of the label (possibly adjusted)
    original_y: float  # Original Y position (gate position)
    text: str  # The label text
    width_estimate: float  # Estimated width in axis units
    height_estimate: float  # Estimated height in axis units


class LabelManager:
    """Manages label positioning to avoid overlaps in circuit diagrams.

    This class tracks all placed labels and provides methods to compute
    optimal positions for new labels, avoiding collisions with existing ones.
    """

    def __init__(self, config: PlotConfig):
        self.config = config
        self._placed_labels: List[LabelInfo] = []
        # Estimate character width/height in axis units based on text size
        # These are rough estimates that work for typical matplotlib figures
        self._char_width = config.text_size * 0.006  # Reduced from 0.008
        self._char_height = config.text_size * 0.010  # Reduced from 0.012
        # Maximum number of lines to break a label into
        self._max_lines = 2

    def reset(self):
        """Clear all tracked labels. Call this when starting a new plot."""
        self._placed_labels = []

    def format_label(self, label: str) -> str:
        """Format a label according to the current label mode.

        Args:
            label: The original label text (e.g., "Ry(0.5*pi)")

        Returns:
            Formatted label text based on label_mode setting.
        """
        if self.config.label_mode == LabelMode.TRUNCATE:
            return self._truncate_label(label)
        elif self.config.label_mode == LabelMode.SMART:
            return self._smart_format_label(label)
        else:  # FULL mode
            return label

    def _truncate_label(self, label: str) -> str:
        """Truncate label to show only gate name without parameters.

        Examples:
            "Ry(0.5*pi)" -> "Ry"
            "Rz(theta)" -> "Rz"
            "U3(0.1, 0.2, 0.3)" -> "U3"
        """
        # Find the position of the first opening parenthesis
        paren_idx = label.find("(")
        if paren_idx > 0:
            return label[:paren_idx]
        return label

    def _smart_format_label(self, label: str) -> str:
        """Format label for smart mode - may add line breaks for long labels.

        Limits line breaks to max 2 lines to prevent excessive vertical space usage.
        """
        # Round numerical values first (using the existing logic)
        import re

        decimal_places = self.config.label_decimal_places

        def round_match(match: re.Match) -> str:
            # The pattern only matches valid float syntax, so float() cannot fail.
            num = float(match.group(0))
            return f"{num:.{decimal_places}f}".rstrip("0").rstrip(".")

        pattern = r"-?\d+\.\d+(?:[eE][+-]?\d+)?"
        label = re.sub(pattern, round_match, label)

        # If label is within max width, return as-is
        if len(label) <= self.config.label_max_width:
            return label

        # Try to break at logical points for long labels (max 2 lines)
        return self._break_long_label(label)

    def _break_long_label(self, label: str) -> str:
        """Break a long label into at most 2 lines at logical points.

        If the label is too long for 2 lines, truncate with ellipsis.
        """
        max_width = self.config.label_max_width

        # Check if it's a gate with parameters: Name(params)
        paren_idx = label.find("(")
        if paren_idx > 0 and label.endswith(")"):
            gate_name = label[:paren_idx]
            params = label[paren_idx + 1 : -1]

            # If gate name alone fits, put name on first line, params on second
            if len(gate_name) <= max_width:
                # Truncate params if too long for second line
                if len(params) > max_width:
                    params = params[: max_width - 3] + "..."
                return f"{gate_name}\n({params})"

        # For very long labels, try to break into 2 lines with ellipsis on second
        if len(label) > max_width * 2:
            # Break at max_width for first line, rest goes to second with ellipsis
            line1 = label[:max_width]
            line2 = label[max_width : max_width * 2 - 3] + "..."
            return f"{line1}\n{line2}"

        # Try breaking at break characters (comma, operators, spaces)
        break_chars = ",+-*/ "
        best_break = -1

        # Look for break point near the middle
        search_start = max(1, len(label) // 2 - max_width // 2)
        search_end = min(len(label) - 1, len(label) // 2 + max_width // 2)

        for i in range(search_start, search_end):
            if label[i] in break_chars:
                best_break = i
                # Prefer breaks closer to middle
                if i >= len(label) // 2:
                    break

        if best_break > 0:
            line1 = label[: best_break + 1].strip()
            line2 = label[best_break + 1 :].strip()
            # Truncate lines if still too long
            if len(line1) > max_width:
                line1 = line1[: max_width - 2] + ".."
            if len(line2) > max_width:
                line2 = line2[: max_width - 2] + ".."
            return f"{line1}\n{line2}"

        # Try breaking at capital letters (e.g., BlockEncoding -> Block\nEncoding)
        for i in range(1, len(label)):
            if label[i].isupper():
                line1 = label[:i]
                line2 = label[i:]
                # Only use this break if both lines fit within max_width
                if len(line1) <= max_width and len(line2) <= max_width:
                    return f"{line1}\n{line2}"

        # Last resort: break into 2 lines with ellipsis on second line
        if len(label) > max_width:
            line1 = label[:max_width]
            remaining = label[max_width:]
            if len(remaining) > max_width - 3:
                line2 = remaining[: max_width - 3] + "..."
            else:
                line2 = remaining
            return f"{line1}\n{line2}"

        # Label fits on one line
        return label

    def compute_position(self, x: float, base_y: float, label: str) -> Tuple[float, float, str]:
        """Compute optimal position for a label to avoid overlaps.

        Args:
            x: X position of the gate
            base_y: Y position of the gate (natural label position)
            label: The label text to place

        Returns:
            Tuple of (adjusted_y, x_offset, final_label_text)
            where x_offset is 0 for centered labels or non-zero for shifted labels
        """
        formatted_label = self.format_label(label)

        if self.config.label_mode == LabelMode.TRUNCATE:
            # Truncated labels are short enough, minimal adjustment needed
            y, x_offset = self._simple_position(x, base_y, formatted_label)
            self._register_label(x + x_offset, y, base_y, formatted_label)
            return y, x_offset, formatted_label

        elif self.config.label_mode == LabelMode.FULL:
            # Full mode - no adjustments
            self._register_label(x, base_y, base_y, formatted_label)
            return base_y, 0.0, formatted_label

        else:  # SMART mode
            y, x_offset = self._smart_position(x, base_y, formatted_label)
            self._register_label(x + x_offset, y, base_y, formatted_label)
            return y, x_offset, formatted_label

    def _simple_position(self, x: float, base_y: float, label: str) -> Tuple[float, float]:
        """Simple positioning with basic overlap avoidance."""
        # Check for nearby labels and shift if needed
        y = base_y
        x_offset = 0.0

        for placed in self._placed_labels:
            if self._labels_overlap(x, y, label, placed):
                # Try shifting y
                y = self._find_free_y(x, base_y, label)
                break

        return y, x_offset

    def _smart_position(self, x: float, base_y: float, label: str) -> Tuple[float, float]:
        """Smart positioning with multiple strategies to avoid overlaps."""
        # Strategy 1: Try original position
        if not self._has_overlap(x, base_y, label):
            return base_y, 0.0

        # Strategy 2: Try small y adjustments (up and down)
        for dy in [
            self.config.label_y_margin,
            -self.config.label_y_margin,
            self.config.label_y_margin * 2,
            -self.config.label_y_margin * 2,
        ]:
            test_y = base_y + dy
            if not self._has_overlap(x, test_y, label):
                return test_y, 0.0

        # Strategy 3: Try x offsets combined with y adjustments
        for dx in [
            self.config.label_x_margin,
            -self.config.label_x_margin,
        ]:
            for dy in [0, self.config.label_y_margin, -self.config.label_y_margin]:
                test_y = base_y + dy
                if not self._has_overlap(x + dx, test_y, label):
                    return test_y, dx

        # Strategy 4: Find any free position nearby
        y = self._find_free_y(x, base_y, label)
        return y, 0.0

    def _has_overlap(self, x: float, y: float, label: str) -> bool:
        """Check if placing a label at (x, y) would overlap with existing labels."""
        for placed in self._placed_labels:
            if self._labels_overlap(x, y, label, placed):
                return True
        return False

    def _labels_overlap(self, x: float, y: float, label: str, placed: LabelInfo) -> bool:
        """Check if two labels would overlap.

        Only considers labels that are very close in x position (same gate column
        or immediately adjacent). Labels at different x positions in the circuit
        are unlikely to overlap visually.
        """
        # Estimate dimensions of new label
        lines = label.split("\n")
        new_height = len(lines) * self._char_height

        # X distance threshold - only consider overlap if labels are very close in x
        # Use spacing as the primary measure - labels separated by more than half
        # a gate spacing are unlikely to overlap
        x_threshold = self.config.spacing * 0.4

        x_distance = abs(x - placed.x)

        # If labels are far apart in x, they don't overlap
        if x_distance > x_threshold:
            return False

        # For labels at similar x positions, check y overlap
        y_margin = self.config.label_y_margin
        y_distance = abs(y - placed.y)
        y_threshold = (new_height + placed.height_estimate) / 2 + y_margin

        return y_distance < y_threshold

    def _find_free_y(self, x: float, base_y: float, label: str) -> float:
        """Find a free y position for a label, searching outward from base_y."""
        step = self.config.label_y_margin
        max_attempts = 20

        for i in range(1, max_attempts):
            # Try above
            test_y = base_y + i * step
            if not self._has_overlap(x, test_y, label):
                return test_y

            # Try below
            test_y = base_y - i * step
            if not self._has_overlap(x, test_y, label):
                return test_y

        # If no free position found, return a position far enough away
        return base_y + max_attempts * step

    def _register_label(self, x: float, y: float, original_y: float, label: str) -> None:
        """Register a placed label for future overlap checks."""
        lines = label.split("\n")
        width = max(len(line) for line in lines) * self._char_width
        height = len(lines) * self._char_height

        self._placed_labels.append(
            LabelInfo(
                x=x,
                y=y,
                original_y=original_y,
                text=label,
                width_estimate=width,
                height_estimate=height,
            )
        )

    def draw_label_connector(self, ax, x: float, gate_y: float, label_y: float) -> None:
        """Draw a subtle connector line from gate to displaced label if needed.

        This helps users understand which gate a displaced label belongs to.
        """
        if abs(label_y - gate_y) > self.config.label_y_margin * 1.5:
            # Draw a thin dotted line connecting gate to label
            ax.plot(
                [x, x],
                [gate_y, label_y],
                color=self.config.muted_color,
                linestyle=":",
                linewidth=0.5,
                alpha=0.5,
                zorder=1,
            )
