"""Natural scale of a circuit figure: how many inches one data unit takes.

Text is sized in points while gates sit at data coordinates, so a box can only
be made to fit its label if the two are tied together.  The plotter sizes its
default figure from these constants and the layout converts label sizes
through them, so a column is as wide as the box drawn in it.  Drawing converts
through the axes it draws on, so a box still fits its label in a figure of
another size; the columns keep the natural layout, so in a narrower figure
neighbouring boxes can meet.
"""

from functools import lru_cache
from typing import Tuple

import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

X_INCH = 2.4  # inches per data unit along the circuit
Y_INCH = 0.4  # inches between neighbouring wires
LINE_HEIGHT = 1.25  # line pitch of a multi-line label, in font sizes
BOX_PAD_PT = 4.0  # space between a label and the edge of its fitted box


@lru_cache(maxsize=64)
def _advance(family: Tuple[str, ...], fontsize: float) -> float:
    """Advance width of one character of the theme's mono face, in points."""
    try:
        path = TextPath((0, 0), "0" * 10, size=fontsize, prop=FontProperties(family=list(family)))
        return float(path.get_extents().width) / 10
    except Exception:
        return 0.6 * fontsize


def text_width_pt(text: str, fontsize: float) -> float:
    """Width of the widest line of ``text``, in points."""
    n_chars = max(len(line) for line in text.split("\n"))
    if mpl.rcParams["text.usetex"]:
        return 0.55 * fontsize * n_chars
    family = mpl.rcParams["font.family"]
    family = (family,) if isinstance(family, str) else tuple(family)
    return _advance(family, float(fontsize)) * n_chars


def text_height_pt(text: str, fontsize: float) -> float:
    """Height of ``text`` over all its lines, in points."""
    return (text.count("\n") + 1) * LINE_HEIGHT * fontsize


def pt_to_x(points: float, ax=None) -> float:
    """Length along the circuit in data units: on ``ax``, or at the natural scale."""
    if ax is None:
        return points / 72 / X_INCH
    low, high = ax.get_xlim()
    return points / 72 * ax.figure.dpi * abs(high - low) / ax.get_window_extent().width


def pt_to_y(points: float, ax=None) -> float:
    """Length across the wires in data units: on ``ax``, or at the natural scale."""
    if ax is None:
        return points / 72 / Y_INCH
    low, high = ax.get_ylim()
    return points / 72 * ax.figure.dpi * abs(high - low) / ax.get_window_extent().height
