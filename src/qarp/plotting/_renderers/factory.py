from .._config import PlotConfig
from .boxed import BoxedGateRenderer
from .controlled import ControlledGateRenderer
from .controlled_block import ControlledBlockRenderer
from .global_phase import GlobalPhaseRenderer
from .label_manager import LabelManager
from .multi import MultiGateRenderer
from .single import SingleGateRenderer


class RendererFactory:
    """Factory for creating appropriate gate renderers."""

    def __init__(self, config: PlotConfig):
        self.config = config
        self.label_manager = LabelManager(config)
        self._renderers = {
            "single": SingleGateRenderer(config, self.label_manager),
            "controlled": ControlledGateRenderer(config, self.label_manager),
            "boxed": BoxedGateRenderer(config, self.label_manager),
            "controlled_block": ControlledBlockRenderer(config, self.label_manager),
            "multi": MultiGateRenderer(config, self.label_manager),
            "global_phase": GlobalPhaseRenderer(config, self.label_manager),
        }

    def reset_label_manager(self) -> None:
        """Reset the label manager. Call this before plotting a new circuit."""
        self.label_manager.reset()

    def get_renderer(self, gate_type: str):
        """Get appropriate renderer for gate type."""
        return self._renderers[gate_type]

    # Controlled gates with a single distinguished target, drawn as control
    # dots plus one target marker.  CSWAP is deliberately absent: it has two
    # targets, so the multi renderer draws its swap pair.
    _CONTROLLED = {
        "cx",
        "cy",
        "cz",
        "ch",
        "cs",
        "csdg",
        "csx",
        "csxdg",
        "cp",
        "crx",
        "cry",
        "crz",
        "cu",
        "ccx",
        "mcz",
    }

    def classify_gate(self, cmd) -> str:
        """Classify gate type for renderer selection."""
        # Parameterized labels arrive as "crz(0.3)"; match on the stem alone.
        name = str(cmd.op).lower().split("(")[0]

        # Handle global phase gates (no qubits)
        if len(cmd.qubits) == 0:
            return "global_phase"
        elif name in self._CONTROLLED:
            return "controlled"
        elif name == "controlledblock":
            return "controlled_block"
        elif name == "block":
            return "boxed"
        elif len(cmd.qubits) == 1:
            return "single"
        else:
            return "multi"
