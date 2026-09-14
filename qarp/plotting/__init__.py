"""Circuit and histogram plotting.  Public depth: flat for the plotters and their
configuration, plus one qualified namespace — ``styles``, the colour schemes
(including the accessibility palettes) and the matplotlib theme, which users
import directly.  Renderers, processors and the interactive widget are private.
"""

from ._plot import plot
from ._circuit_plotter import CircuitPlotter, GateInfo
from ._config import PlotConfig, LabelMode
from ._circuit_adapter import CircuitAdapter
from ._plot_histogram import plot_histogram

__all__ = [
    "CircuitAdapter",
    "CircuitPlotter",
    "GateInfo",
    "LabelMode",
    "PlotConfig",
    "plot",
    "plot_histogram",
    "styles",
]
