from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

from .styles import DEFAULT_COLORS
from .styles import _theme as theme


class LabelMode(Enum):
    """Determines how gate labels are handled in circuit plots.

    Attributes:
        TRUNCATE: Truncates labels to show only the gate name without parameters.
                  E.g., "Ry(0.5*pi)" becomes "Ry". This is the safest option
                  that avoids overlap issues.
        SMART: Uses intelligent positioning to avoid overlaps. Labels may be
               shifted vertically, line-broken, or repositioned as needed.
        FULL: Displays full labels without any adjustments. May result in
              overlapping labels for complex circuits.
    """

    TRUNCATE = "truncate"
    SMART = "smart"
    FULL = "full"


@dataclass
class PlotConfig:
    """Configuration for circuit plotting."""

    gate_colors: Optional[Dict[str, str]] = None
    spacing: float = 0.25
    gate_size: int = 16
    control_size: int = 4
    swap_size: int = 6
    text_size: int = 9
    box_linewidth: float = 1.1
    # "outline" draws gates as white-tinted boxes with a role-colored edge
    # (OpenQARP theme); "filled" is the solid-chip look used by the
    # classic/colorblind/grayscale schemes and custom color dicts.
    gate_style: str = "outline"
    gate_edge_width: float = 1.3
    default_gate_color: str = theme.INK  # fallback for gates missing from the scheme
    wire_color: str = theme.WIRE
    structure_color: str = theme.INK  # control dots, connector vlines, swaps
    text_color: str = theme.INK
    muted_color: str = theme.MUTED
    gate_index_color: str = theme.INDEX_RED
    max_chars: int = 8
    label_decimal_places: int = 3
    label_mode: LabelMode = LabelMode.SMART
    label_max_width: int = 12  # Max characters before line breaking in SMART mode
    label_y_margin: float = 0.4  # Minimum vertical distance between labels
    label_x_margin: float = 0.15  # Minimum horizontal distance between labels
    show_gate_indices: bool = False  # Show small index numbers near gates for querying
    gate_index_size: int = 6  # Font size for gate index numbers

    def __post_init__(self):
        if self.gate_colors is None:
            self.gate_colors = DEFAULT_COLORS

    def gate_color(self, name: str) -> str:
        """Scheme color for a gate name, with the config fallback."""
        return (self.gate_colors or {}).get(name, self.default_gate_color)

    def gate_face_edge(self, color: str) -> Tuple[str, str]:
        """(facecolor, edgecolor) for a gate marker under the active gate_style."""
        if self.gate_style == "outline":
            return theme.tint(color, 0.08), color
        return color, color
