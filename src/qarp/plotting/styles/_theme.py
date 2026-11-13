"""OpenQARP visual theme: design tokens, matplotlib rc params, and bundled fonts.

Single source of truth for the package's plotting identity (shared with the
OpenQARP website): paper/ink surfaces, cobalt as the working accent, amber
reserved for measurement, hairline chrome, mono type for data labels.
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.font_manager as fm

# ---------------------------------------------------------------- tokens

PAPER = "#f4f7f9"  # page / box-fill surface
CARD = "#ffffff"  # figure surface
INK = "#101b2d"  # primary ink: text, structural gates, controls
SLATE = "#44566c"  # secondary text
MUTED = "#5d6f86"  # axis labels, qubit indices
COBALT = "#2e4fd8"  # working accent: parameterized gates, highlights
COBALT_INK = "#1d33a8"
AMBER = "#c77d02"  # measurement semantics ONLY: measure/reset/classical
WIRE = "#b6c2d1"  # qubit wires
GRID = "#e7edf3"  # hairline gridlines
BASELINE = "#c3cedb"  # axis baseline / spines
INDEX_RED = "#d03b3b"  # gate-index debug annotations (kept off the role palette)

_FONT_DIR = Path(__file__).parent / "fonts"
_MONO_FALLBACKS = ["JetBrains Mono", "DejaVu Sans Mono", "monospace"]


def tint(color: str, amount: float) -> str:
    """Blend ``color`` toward white; ``amount`` is the retained color fraction."""
    hex_color = color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    mix = lambda c: round(255 - (255 - c) * amount)  # noqa: E731
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


@lru_cache(maxsize=1)
def register_fonts() -> Optional[str]:
    """Register the bundled Spline Sans Mono faces with matplotlib.

    Returns the usable mono family name (bundled face, or the first system
    fallback found), or None if only the generic ``monospace`` is available.
    Cached: registration mutates matplotlib's global font manager once.
    """
    registered = False
    for ttf in sorted(_FONT_DIR.glob("*.ttf")):
        try:
            fm.fontManager.addfont(str(ttf))
            registered = True
        except Exception:
            pass  # fall through to system fonts
    if registered:
        return "Spline Sans Mono"
    available = {f.name for f in fm.fontManager.ttflist}
    for family in _MONO_FALLBACKS[:-1]:
        if family in available:
            return family
    return None


def mono_family() -> List[str]:
    """Font-family list for rc params: bundled mono first, then fallbacks."""
    primary = register_fonts()
    return ([primary] if primary else []) + ["monospace"]


def qarp_rc(use_latex: bool = False) -> Dict:
    """Scoped rc params for OpenQARP figures — apply via ``plt.rc_context``."""
    rc = {
        "font.family": mono_family(),
        "text.color": INK,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": SLATE,
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titleweight": "semibold",
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "grid.color": GRID,
        "grid.linewidth": 0.9,
        "grid.linestyle": "-",
        "figure.facecolor": CARD,
        "axes.facecolor": CARD,
        "savefig.facecolor": CARD,
        "legend.frameon": False,
        # Text as paths: notebook SVG output renders identically without the font.
        "svg.fonttype": "path",
        "text.usetex": use_latex,
    }
    if use_latex:
        rc["font.family"] = "serif"
        rc["font.serif"] = ["Computer Modern Roman"]
    return rc


# ------------------------------------------------- role-based gate scheme
# Color encodes role, not gate name: structural/Clifford gates stay quiet
# (ink outline), parameterized gates carry the variational story (cobalt),
# and amber is measurement semantics. Boxes read as grouped paper.

_STRUCTURAL = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "swap"]
_PARAMETERIZED = ["rx", "ry", "rz", "p", "u"]
_MEASUREMENT = ["measure", "reset"]

QARP_COLORS: Dict[str, str] = {
    **{name: INK for name in _STRUCTURAL},
    **{name: COBALT for name in _PARAMETERIZED},
    **{name: AMBER for name in _MEASUREMENT},
    "box": PAPER,
}
