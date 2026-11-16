"""Color schemes for quantum gates."""

from ._theme import QARP_COLORS


def get_text_color(background_color: str) -> str:
    """Determine optimal text color (black or white) based on background luminance.

    Uses the relative luminance formula from WCAG 2.0 guidelines to determine
    whether black or white text provides better contrast.

    Args:
        background_color: Hex color string (e.g., '#FF5500' or 'FF5500')

    Returns:
        '#000000' (black) for light backgrounds, '#FFFFFF' (white) for dark backgrounds
    """
    # Remove '#' if present
    hex_color = background_color.lstrip("#")

    # Parse RGB values
    try:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
    except (ValueError, IndexError):
        # Default to black text if color parsing fails
        return "#000000"

    # Apply sRGB to linear RGB conversion
    def to_linear(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r_lin = to_linear(r)
    g_lin = to_linear(g)
    b_lin = to_linear(b)

    # Calculate relative luminance (WCAG formula)
    luminance = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin

    # Use white text for dark backgrounds (luminance < 0.179)
    # This threshold provides approximately 4.5:1 contrast ratio
    return "#FFFFFF" if luminance < 0.1 else "#000000"


# Pre-theme per-gate palette, kept selectable as color_scheme="classic".
CLASSIC_COLORS = {
    "h": "#B9DAFB",
    "x": "#EAB8E4",
    "y": "#CF8C8F",
    "z": "#B5E1BD",
    "rx": "#B8E1F4",
    "ry": "#F2C94C",
    "rz": "#C89DF1",
    "s": "#F8C66B",
    "sdg": "#F8C66B",
    "t": "#94FBE1",
    "tdg": "#94FBE1",
    "p": "#F8C66B",
    "u": "#F8C66B",
    "measure": "#FF8579",
    "reset": "#FF8579",
    "box": "#EFEFEF",
}

# General colorblind-friendly palette (Wong palette)
# Works well for most types of color vision deficiency
COLORBLIND_COLORS = {
    "h": "#E69F00",  # Orange
    "x": "#56B4E9",  # Sky blue
    "y": "#009E73",  # Bluish green
    "z": "#F0E442",  # Yellow
    "rx": "#0072B2",  # Blue
    "ry": "#D55E00",  # Vermillion
    "rz": "#CC79A7",  # Reddish purple
    "s": "#999999",  # Gray
    "sdg": "#999999",
    "t": "#E69F00",
    "tdg": "#E69F00",
    "p": "#D55E00",
    "u": "#CC79A7",
    "measure": "#FF0000",
    "reset": "#FF0000",
    "box": "#EFEFEF",
}

# Protanopia-friendly palette (red-blind)
# Avoids red-green confusion, uses blue-yellow distinction
PROTANOPIA_COLORS = {
    "h": "#0077BB",  # Blue
    "x": "#33BBEE",  # Cyan
    "y": "#EE7733",  # Orange
    "z": "#FFDD44",  # Yellow
    "rx": "#004488",  # Dark blue
    "ry": "#DDAA33",  # Gold
    "rz": "#BB5566",  # Muted pink
    "s": "#999999",  # Gray
    "sdg": "#999999",
    "t": "#0077BB",
    "tdg": "#0077BB",
    "p": "#DDAA33",
    "u": "#BB5566",
    "measure": "#CC3311",  # Dark orange-red
    "reset": "#CC3311",
    "box": "#EFEFEF",
}

# Deuteranopia-friendly palette (green-blind, most common)
# Similar to protanopia, avoids red-green confusion
DEUTERANOPIA_COLORS = {
    "h": "#0077BB",  # Blue
    "x": "#33BBEE",  # Cyan
    "y": "#EE7733",  # Orange
    "z": "#FFDD44",  # Yellow
    "rx": "#004488",  # Dark blue
    "ry": "#DDAA33",  # Gold
    "rz": "#AA3377",  # Purple-pink
    "s": "#BBBBBB",  # Light gray
    "sdg": "#BBBBBB",
    "t": "#0077BB",
    "tdg": "#0077BB",
    "p": "#DDAA33",
    "u": "#AA3377",
    "measure": "#EE3377",  # Magenta
    "reset": "#EE3377",
    "box": "#EFEFEF",
}

# Tritanopia-friendly palette (blue-blind, rare)
# Avoids blue-yellow confusion, uses red-green distinction
TRITANOPIA_COLORS = {
    "h": "#EE3377",  # Magenta
    "x": "#33BB33",  # Green
    "y": "#EE7733",  # Orange
    "z": "#FF6677",  # Salmon
    "rx": "#117733",  # Dark green
    "ry": "#CC3311",  # Red-orange
    "rz": "#AA4499",  # Purple
    "s": "#999999",  # Gray
    "sdg": "#999999",
    "t": "#EE3377",
    "tdg": "#EE3377",
    "p": "#CC3311",
    "u": "#AA4499",
    "measure": "#CC3311",
    "reset": "#CC3311",
    "box": "#EFEFEF",
}

# High contrast palette for maximum visibility
# Uses bold, saturated colors with maximum distinction
HIGH_CONTRAST_COLORS = {
    "h": "#0000FF",  # Pure blue
    "x": "#FF00FF",  # Magenta
    "y": "#FF8800",  # Orange
    "z": "#00FF00",  # Green
    "rx": "#000088",  # Dark blue
    "ry": "#FFCC00",  # Gold
    "rz": "#8800FF",  # Purple
    "s": "#888888",  # Gray
    "sdg": "#888888",
    "t": "#00FFFF",  # Cyan
    "tdg": "#00FFFF",
    "p": "#FFCC00",
    "u": "#8800FF",
    "measure": "#FF0000",  # Red
    "reset": "#FF0000",
    "box": "#DDDDDD",
}

# Grayscale palette for black-and-white printing
GRAYSCALE_COLORS = {
    "h": "#E0E0E0",  # Light gray
    "x": "#A0A0A0",  # Medium gray
    "y": "#808080",  # Gray
    "z": "#C0C0C0",  # Silver
    "rx": "#B0B0B0",  # Light-medium gray
    "ry": "#909090",  # Medium-dark gray
    "rz": "#707070",  # Dark gray
    "s": "#D0D0D0",
    "sdg": "#D0D0D0",
    "t": "#E8E8E8",
    "tdg": "#E8E8E8",
    "p": "#989898",
    "u": "#787878",
    "measure": "#404040",  # Dark gray
    "reset": "#404040",
    "box": "#F0F0F0",
}

# Dictionary mapping scheme names to palettes for easy access
# Package default: the OpenQARP theme's role-based scheme (see styles/theme.py).
DEFAULT_COLORS = QARP_COLORS

COLOR_SCHEMES = {
    "default": DEFAULT_COLORS,
    "qarp": QARP_COLORS,
    "classic": CLASSIC_COLORS,
    "colorblind": COLORBLIND_COLORS,
    "protanopia": PROTANOPIA_COLORS,
    "deuteranopia": DEUTERANOPIA_COLORS,
    "tritanopia": TRITANOPIA_COLORS,
    "high_contrast": HIGH_CONTRAST_COLORS,
    "grayscale": GRAYSCALE_COLORS,
}
