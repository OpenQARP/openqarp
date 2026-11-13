"""Plot styles: the named colour schemes and the qarp matplotlib theme.
Public depth: flat.  Submodules are private.
"""

from ._colors import (
    DEFAULT_COLORS,
    QARP_COLORS,
    CLASSIC_COLORS,
    COLORBLIND_COLORS,
    PROTANOPIA_COLORS,
    DEUTERANOPIA_COLORS,
    TRITANOPIA_COLORS,
    HIGH_CONTRAST_COLORS,
    GRAYSCALE_COLORS,
    COLOR_SCHEMES,
    get_text_color,
)
from ._theme import qarp_rc, register_fonts, tint

__all__ = [
    "CLASSIC_COLORS",
    "COLORBLIND_COLORS",
    "COLOR_SCHEMES",
    "DEFAULT_COLORS",
    "DEUTERANOPIA_COLORS",
    "GRAYSCALE_COLORS",
    "HIGH_CONTRAST_COLORS",
    "PROTANOPIA_COLORS",
    "QARP_COLORS",
    "TRITANOPIA_COLORS",
    "get_text_color",
    "qarp_rc",
    "register_fonts",
    "tint",
]
