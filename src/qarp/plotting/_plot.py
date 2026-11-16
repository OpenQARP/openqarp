from typing import Dict, List, Optional, Union

from ._circuit_adapter import CircuitAdapter
from ._circuit_plotter import CircuitPlotter, PlotConfig
from ._config import LabelMode
from .styles import COLOR_SCHEMES, DEFAULT_COLORS

# Classic schemes (and custom color dicts) map gate names to *fill* colors,
# so they use the solid-chip treatment and chrome.
_CLASSIC_SCHEMES = {
    "classic",
    "colorblind",
    "protanopia",
    "deuteranopia",
    "tritanopia",
    "high_contrast",
    "grayscale",
}


def _apply_color_scheme(config: PlotConfig, color_scheme: Union[str, Dict]) -> None:
    """Resolve a scheme name or dict onto the config, including its treatment."""
    classic = False
    if isinstance(color_scheme, str):
        config.gate_colors = COLOR_SCHEMES.get(color_scheme, DEFAULT_COLORS)
        classic = color_scheme in _CLASSIC_SCHEMES
    elif isinstance(color_scheme, dict):
        config.gate_colors = color_scheme
        classic = True
    else:
        config.gate_colors = DEFAULT_COLORS
    if classic:
        config.gate_style = "filled"
        config.box_linewidth = 0.3
        config.default_gate_color = "skyblue"
        config.wire_color = "gray"
        config.structure_color = "black"
        config.text_color = "#000000"
        config.muted_color = "gray"


# Convenience function
def plot(
    circ,
    color_scheme: Union[str, Dict] = "default",
    label_mode: Optional[LabelMode] = None,
    show_gate_indices: bool = False,
    return_plotter: bool = False,
    inner_index: Optional[Union[int, List[int]]] = None,
    **kwargs,
) -> Optional[CircuitPlotter]:
    """Plot the provided circuit.

    Args:
        circ: The quantum circuit to plot.
        color_scheme: Color scheme for gates. Options are 'default' (the OpenQARP
            theme: role-based colors, outlined gates), 'qarp' (alias of default),
            'classic' (per-gate solid-chip palette), 'colorblind', 'protanopia',
            'deuteranopia', 'tritanopia', 'high_contrast', 'grayscale', or a custom
            dict mapping gate names to hex fill colors (drawn in the classic
            solid-chip treatment).
        label_mode: How to handle gate labels. Options are LabelMode.TRUNCATE
            (show only gate names), LabelMode.SMART (intelligent positioning, default),
            or LabelMode.FULL (full labels without adjustment).
        show_gate_indices: If True, display small index numbers near each gate.
            Use the returned plotter's query_gate(index) or list_gates() to get full info.
        return_plotter: If True, returns the plotter instance for gate querying.
        inner_index: Index or path to the inner circuit of a box gate. Can be
            a single int (plots inner circuit at that index) or a list of ints
            (recursively zooms into nested boxes following the path, e.g., [1, 0, 3]).
            Use with show_gate_indices=True first to identify box gate indices.
            Works with any boxed sub-block (plain or quantum-controlled).
        figsize: Size of the figure (width, height). If None, size is auto-calculated.
        save_fig: If provided, saves the figure to the given file path.
        decompose_boxes: Whether to decompose box gates before plotting.
        flatten_layers: Whether to flatten layers in the circuit.
        invert_order: Whether to invert the order of qubits in the plot.
        scrollable: If True, display the plot in a scrollable HTML container (for Jupyter).
            If None (default), auto-detects: scrollable inside a Jupyter notebook,
            plain plt.show() otherwise.
        interactive: If True (default whenever the scrollable HTML path is used),
            the notebook output gains hover tooltips — full gate expression, qubits,
            gate index — and dims the rest of the circuit while a gate is hovered.
            Pure client-side, so it survives nbconvert to HTML. False for plain SVG.
        use_latex: Whether to use LaTeX for rendering text.
        spacing: Horizontal spacing between gates. If None, uses default from config.
        verbose: If True, prints additional information about the circuit.

    Returns:
        CircuitPlotter: The plotter instance if return_plotter=True, otherwise None.
            Use query_gate(index) or list_gates() to inspect gate information.

    Example:
        >>> # First, plot with indices to find box gates
        >>> plot(circuit, show_gate_indices=True)
        >>> # Then plot the inner circuit of the box at index 5
        >>> plot(circuit, inner_index=5)
        >>> # Zoom into nested boxes: box 1 -> box 0 inside -> box 3 inside
        >>> plot(circuit, inner_index=[1, 0, 3])
        >>> # Use specific colorblind palette
        >>> plot(circuit, color_scheme='deuteranopia')
    """

    # Accept a raw Block directly (e.g. a drill-in inner circuit): wrap it in
    # the renderer-facing adapter.  Already-wrapped circuits / command-list
    # adapters expose ``get_commands`` and pass through untouched.
    if not hasattr(circ, "get_commands") and hasattr(circ, "n_qubits"):
        circ = CircuitAdapter(circ, decompose_boxes=kwargs.get("decompose_boxes", False))

    config = PlotConfig()
    _apply_color_scheme(config, color_scheme)

    # Apply label_mode and show_gate_indices to config
    if label_mode is not None:
        config.label_mode = label_mode
    if show_gate_indices:
        config.show_gate_indices = show_gate_indices

    plotter = CircuitPlotter(config=config)

    # If inner_index is specified, we need to first plot to get the gate registry,
    # then extract and plot the inner circuit
    if inner_index is not None:
        # Normalize to a list for uniform handling
        if isinstance(inner_index, int):
            index_path = [inner_index]
        else:
            index_path = list(inner_index)

        # Recursively navigate through the path
        current_circ = circ
        for i, idx in enumerate(index_path):
            # Do a silent plot to build the gate registry (without displaying)
            plotter.plot(current_circ, _show=False)

            # Get the inner circuit at this index
            inner_circ = plotter.get_inner_circuit(idx)
            if inner_circ is None:
                print(
                    f"Could not find inner circuit at index {idx} (path step {i + 1}/{len(index_path)})"
                )
                return None

            current_circ = inner_circ
            # Create a fresh plotter for the next iteration
            plotter = CircuitPlotter(config=config)

        # Plot the final inner circuit
        plotter.plot(current_circ, **kwargs)

        if return_plotter:
            return plotter
        return None

    # Normal plot
    plotter.plot(circ, **kwargs)

    if return_plotter:
        return plotter
    return None
