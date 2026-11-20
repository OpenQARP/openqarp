import uuid
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from IPython.display import HTML, display

from . import _geometry as geometry
from ._config import LabelMode, PlotConfig
from ._processors.circuit_processor import CircuitProcessor, CircuitValidationError
from ._renderers import LabelInfo, RendererFactory
from ._renderers.hybrid import draw_meter
from .styles._theme import qarp_rc

# The classical wire sits one wire pitch below the last qubit.
_CLASSICAL_Y = -1.0


def _link_label(cmd, reads: bool, writes: bool) -> str:
    """Text under a command's link to the classical wire: the bits it writes,
    then the condition it reads."""
    parts = []
    if writes:
        parts.append(", ".join(f"c{b}" for b in cmd.cbits))
    if reads:
        parts.append(cmd.condition or ", ".join(f"c{b}" for b in cmd.condition_bits))
    return "; ".join(parts)


def _in_notebook() -> bool:
    """True when running inside a Jupyter/IPython notebook kernel."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


# matplotlib's file/headless backends.  Calling ``plt.show()`` under any of
# these cannot open a window and only emits a ``UserWarning`` (e.g.
# "FigureCanvasAgg is non-interactive, and thus cannot be shown") — which is
# what surfaced under pytest/CI, where the backend defaults to Agg.
_NON_INTERACTIVE_BACKENDS = {"agg", "pdf", "ps", "svg", "template", "cairo", "pgf"}


def _backend_can_display() -> bool:
    """True if the active matplotlib backend can actually open a window.

    Used to skip ``plt.show()`` on headless/file backends, where it would be a
    no-op that only warns.
    """
    return plt.get_backend().lower() not in _NON_INTERACTIVE_BACKENDS


@dataclass
class GateInfo:
    """Information about a gate in the circuit plot for querying."""

    index: int  # Gate index in the plot
    gate_type: str  # Type of gate (e.g., "Ry", "CX", "U3")
    full_label: str  # Full label including parameters
    displayed_label: str  # Label as displayed in the plot (may be truncated)
    qubits: List[str]  # List of qubit names the gate acts on
    x_position: float  # X position in the plot
    y_position: float  # Y position (center) in the plot
    command: Any  # The original qx.Command object
    cbits: List[int] = field(default_factory=list)  # Classical bits the gate writes
    condition: Optional[str] = None  # Classical condition guarding the gate, e.g. "c0=1"

    @property
    def classical(self) -> str:
        """The gate's classical side as text: ``-> c0`` and/or ``if c0=1``."""
        parts = []
        if self.cbits:
            parts.append("-> " + ", ".join(f"c{b}" for b in self.cbits))
        if self.condition:
            parts.append(f"if {self.condition}")
        return " ".join(parts)

    def __repr__(self) -> str:
        qubits_str = ", ".join(self.qubits)
        return f"Gate[{self.index}]: {self.full_label} on [{qubits_str}]"

    def __str__(self) -> str:
        qubits_str = ", ".join(self.qubits)
        lines = [
            f"Gate Index: {self.index}",
            f"Gate Type: {self.gate_type}",
            f"Full Expression: {self.full_label}",
            f"Displayed As: {self.displayed_label}",
            f"Qubits: [{qubits_str}]",
        ]
        return "\n".join(lines)


class CircuitPlotter:
    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        self.processor = CircuitProcessor(self.config)
        self.renderer_factory = RendererFactory(self.config)
        self._gate_registry: List[GateInfo] = []  # Stores gate info for querying
        self._measure_payload: List[Dict] = []  # Terminal-measure tooltip entries
        self._plot_uid: str = ""  # Per-plot id scoping SVG gids for interactivity

    @property
    def gates(self) -> List[GateInfo]:
        """Get list of all gates from the last plotted circuit.

        Returns:
            List of GateInfo objects containing full gate information.
        """
        return self._gate_registry

    def query_gate(self, index: int) -> Optional[GateInfo]:
        """Query a gate by its index to get full information.

        Args:
            index: The gate index as shown in the plot.

        Returns:
            GateInfo object with full gate details, or None if index not found.

        Example:
            >>> plotter.plot(circuit, show_gate_indices=True)
            >>> gate = plotter.query_gate(3)
            >>> print(gate)
            Gate Index: 3
            Gate Type: Ry
            Full Expression: Ry(0.123456789*pi + theta)
            Displayed As: Ry
            Qubits: [q[0]]
        """
        for gate in self._gate_registry:
            if gate.index == index:
                return gate
        return None

    def get_inner_circuit(self, index: int):
        """Extract the inner circuit from a box gate by its index.

        This method works with boxed sub-blocks (plain or quantum-controlled)
        that contain inner circuits.

        Args:
            index: The gate index as shown in the plot.

        Returns:
            The inner Circuit if the gate is a box type, None otherwise.

        Example:
            >>> plotter.plot(circuit, show_gate_indices=True)
            >>> inner = plotter.get_inner_circuit(5)  # Get inner circuit of gate #5
            >>> if inner:
            ...     plot(inner)  # Plot the inner circuit
        """
        gate = self.query_gate(index)
        if gate is None:
            print(f"No gate found at index {index}")
            return None

        cmd = gate.command
        op = cmd.op

        # Try to get the inner circuit from the operation
        # Works for plain and quantum-controlled block boxes
        try:
            # First, try to get circuit directly (plain block box)
            if hasattr(op, "get_circuit"):
                return op.get_circuit()

            # For controlled block boxes, get the inner op first
            if hasattr(op, "get_op"):
                inner_op = op.get_op()
                if hasattr(inner_op, "get_circuit"):
                    return inner_op.get_circuit()

            print(
                f"Gate '{gate.gate_type}' at index {index} is not a box type with an inner circuit."
            )
            return None
        except Exception as e:
            print(f"Could not extract inner circuit: {e}")
            return None

    def list_gates(self) -> None:
        """Print a summary of all gates from the last plotted circuit."""
        if not self._gate_registry:
            print("No gates recorded. Plot a circuit first.")
            return

        print(f"Total gates: {len(self._gate_registry)}\n")
        for gate in self._gate_registry:
            print(repr(gate))

    def plot_inner(self, index: int, **kwargs) -> Optional["CircuitPlotter"]:
        """Plot the inner circuit of a box gate.

        This is a convenience method that extracts and plots the inner circuit
        of a boxed sub-block (plain or quantum-controlled) in a single call.

        Args:
            index: The gate index as shown in the plot.
            **kwargs: Additional arguments passed to plot() (e.g., figsize, save_fig).

        Returns:
            A new CircuitPlotter instance for the inner circuit, or None if the gate
            is not a box type.

        Example:
            >>> plotter = plot(circuit, show_gate_indices=True, return_plotter=True)
            >>> inner_plotter = plotter.plot_inner(5)  # Plot inner circuit of gate #5
        """
        inner_circuit = self.get_inner_circuit(index)
        if inner_circuit is None:
            return None

        # Create a new plotter for the inner circuit
        inner_plotter = CircuitPlotter(config=self.config)
        inner_plotter.plot(inner_circuit, **kwargs)
        return inner_plotter

    def plot(
        self,
        circ,
        figsize: Optional[Tuple[float, float]] = None,
        save_fig: Optional[str] = None,
        decompose_boxes: bool = False,
        flatten_layers: bool = True,
        invert_order: bool = False,
        scrollable: Optional[bool] = None,
        interactive: Optional[bool] = None,
        use_latex: bool = False,
        spacing: Optional[float] = None,
        verbose: bool = False,
        label_mode: Optional[LabelMode] = None,
        show_gate_indices: Optional[bool] = None,
        _show: bool = True,
    ):
        """Main class for plotting quantum circuits.

        Args:
            circ: The quantum circuit to plot.
            figsize: Size of the figure (width, height). If None, size is auto-calculated.
            save_fig: If provided, saves the figure to the given file path.
            decompose_boxes: Whether to decompose box gates before plotting.
            flatten_layers: Whether to flatten layers in the circuit.
            invert_order: Whether to invert the order of qubits in the plot.
            scrollable: If True, display the plot in a scrollable HTML container (for Jupyter).
                If None (default), auto-detects: scrollable inside a Jupyter notebook,
                plain ``plt.show()`` otherwise.
            interactive: If True (default when the scrollable HTML path is used),
                the notebook output gains hover tooltips — full gate expression,
                qubits, gate index — with the rest of the circuit dimmed while a
                gate is hovered. Pure client-side (no widgets), so it survives
                ``nbconvert`` to HTML. Set False for the plain static SVG.
            use_latex: Whether to use LaTeX for rendering text.
            spacing: Horizontal spacing between gates. If None, uses default from config.
            verbose: If True, prints additional information about the circuit.
            label_mode: How to handle gate labels. Options:
                - LabelMode.TRUNCATE: Show only gate names without parameters (e.g., "Ry" instead of "Ry(0.5)")
                - LabelMode.SMART: Intelligent positioning to avoid overlaps (default)
                - LabelMode.FULL: Full labels without adjustment (may overlap)
            show_gate_indices: If True, display small index numbers near each gate.
                Use query_gate(index) or list_gates() to get full gate information.
        """
        # Override spacing if provided
        if spacing is not None:
            self.config.spacing = spacing

        # Override label_mode if provided
        if label_mode is not None:
            self.config.label_mode = label_mode
            # Recreate the renderer factory with updated config
            self.renderer_factory = RendererFactory(self.config)

        # Override show_gate_indices if provided
        if show_gate_indices is not None:
            self.config.show_gate_indices = show_gate_indices

        # Reset label manager and gate registry before each plot
        self.renderer_factory.reset_label_manager()
        self._gate_registry = []
        self._measure_payload = []
        self._plot_uid = uuid.uuid4().hex[:8]

        # Theme rc params scoped to this figure only — never mutate the user's
        # global rcParams.  Layout measures labels in the theme's font and
        # rendering resolves fonts at draw time, so both run inside the context.
        with plt.rc_context(qarp_rc(use_latex)):
            try:
                circuit_data = self.processor.process_circuit(
                    circ, decompose_boxes, flatten_layers, invert_order, width_of=self._width_of
                )
            except CircuitValidationError as e:
                # Surface validation failures as ValueError to callers
                raise ValueError(str(e)) from e

            if verbose:
                self._print_circuit_info(circuit_data)

            spacing = self.config.spacing
            label_pt = 6 + max(
                geometry.text_width_pt(str(q), self.config.text_size) for q in circuit_data.qubits
            )
            x_low, x_high = -spacing, circuit_data.wire_end + spacing / 2
            y_limits = (
                _CLASSICAL_Y - 0.75 if circuit_data.has_classical_wire else -0.75,
                circuit_data.n_qubits - 0.25,
            )

            # The axes fill the figure and the default figure is sized from the
            # natural scale, so a label measured in points fits the box drawn
            # for it in data units.  The margins hold the qubit labels: in a
            # figure of a given width they take the same points of it.
            if figsize is None:
                margin = geometry.pt_to_x(label_pt)
                figsize = (
                    (x_high - x_low + 2 * margin) * geometry.X_INCH,
                    (y_limits[1] - y_limits[0]) * geometry.Y_INCH,
                )
            else:
                room_pt = max(figsize[0] * 72 - 2 * label_pt, label_pt)
                margin = label_pt * (x_high - x_low) / room_pt
            x_limits = (x_low - margin, x_high + margin)
            fig, ax = plt.subplots(figsize=figsize)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.set_xlim(*x_limits)
            ax.set_ylim(*y_limits)
            ax.axis("off")

            # Draw circuit elements
            self._draw_qubit_lines(
                ax, circuit_data.qubits, circuit_data.y_positions, circuit_data.wire_end
            )
            if circuit_data.has_classical_wire:
                self._draw_classical_wire(ax, circuit_data.wire_end)

            prev_label_y: LabelInfo = (None, None)
            for idx, cmd in enumerate(circuit_data.non_measure_cmds):
                prev_label_y = self._draw_gate(
                    ax,
                    cmd,
                    circuit_data.x_coords[idx],
                    circuit_data.y_positions,
                    prev_label_y,
                    gate_index=idx,
                    reads=circuit_data.reads_classical[idx],
                    writes=circuit_data.writes_classical[idx],
                )

            if circuit_data.n_measures > 0:
                self._draw_measurements(
                    ax,
                    circuit_data.terminal_measures,
                    circuit_data.measure_x,
                    circuit_data.y_positions,
                )

            # Handle output
            if scrollable is None:
                scrollable = _in_notebook()
            if _show:
                if scrollable:
                    self._create_scrollable_output(
                        fig, interactive=interactive if interactive is not None else True
                    )
                elif _backend_can_display():
                    plt.show()
                # else: headless/file backend (e.g. Agg under pytest/CI) — plt.show()
                # can't render and would only emit a UserWarning, so skip it.  The
                # figure is left open exactly as the no-op plt.show() would leave it.
            else:
                plt.close(fig)

            if save_fig:
                fig.savefig(save_fig, bbox_inches="tight")
                if verbose:
                    print(f"Figure saved to {save_fig}")

    def _draw_gate(
        self,
        ax,
        cmd,
        x: float,
        y_positions: dict,
        prev_label_y: LabelInfo,
        gate_index: Optional[int] = None,
        reads: bool = False,
        writes: bool = False,
    ) -> LabelInfo:
        """Draw a gate using appropriate renderer and optionally record gate info.

        ``reads`` / ``writes`` add the gate's link to the classical wire: it is
        guarded by a classical condition, or it writes a bit a later one reads.
        """
        gate_type = self.renderer_factory.classify_gate(cmd)
        renderer = self.renderer_factory.get_renderer(gate_type)
        before = list(ax.get_children())
        result = renderer.render(ax, cmd, x, y_positions, prev_label_y)
        if reads or writes:
            self._draw_classical_link(ax, cmd, x, y_positions, reads, writes)

        # Record gate info for querying
        if gate_index is not None:
            # Tag this gate's artists so the notebook SVG can attach hover
            # behavior to them (gids survive into matplotlib's SVG output).
            before_ids = {id(a) for a in before}
            for j, artist in enumerate(a for a in ax.get_children() if id(a) not in before_ids):
                artist.set_gid(f"qarp-{self._plot_uid}-g{gate_index}-a{j}")
            full_label = str(cmd.op)
            # Get displayed label from the label manager
            displayed_label = self.renderer_factory.label_manager.format_label(full_label)

            # Extract gate type name
            paren_idx = full_label.find("(")
            gate_type_name = full_label[:paren_idx] if paren_idx > 0 else full_label

            # Calculate y position (center of gate)
            y_coords = [y_positions[q] for q in cmd.qubits]
            y_center = sum(y_coords) / len(y_coords) if y_coords else 0.0

            # Store gate info
            gate_info = GateInfo(
                index=gate_index,
                gate_type=gate_type_name,
                full_label=full_label,
                displayed_label=displayed_label,
                qubits=[str(q) for q in cmd.qubits],
                x_position=x,
                y_position=y_center,
                command=cmd,
                cbits=list(getattr(cmd, "cbits", ())),
                condition=getattr(cmd, "condition", None),
            )
            self._gate_registry.append(gate_info)

            # Draw gate index if enabled (skip for global phase gates with no qubits)
            if self.config.show_gate_indices and y_coords:
                # Position the index slightly above and to the right of the gate
                y_top = max(y_coords)
                ax.text(
                    x + self.config.spacing * 0.15,
                    y_top + 0.25,
                    str(gate_index),
                    fontsize=self.config.gate_index_size,
                    color=self.config.gate_index_color,
                    ha="left",
                    va="bottom",
                    weight="bold",
                    zorder=10,
                    # bbox=dict(
                    #     boxstyle="circle,pad=0.15",
                    #     facecolor="white",
                    #     edgecolor="red",
                    #     linewidth=0.5,
                    #     alpha=0.8,
                    # ),
                )

        return result

    def _width_of(self, cmd, reads: bool = False, writes: bool = False) -> float:
        """Width a drawn command needs in its column, in data units: its box,
        or the label under its link to the classical wire if that is wider."""
        renderer = self.renderer_factory.get_renderer(self.renderer_factory.classify_gate(cmd))
        width = renderer.footprint(cmd)
        label = _link_label(cmd, reads, writes)
        if label:
            link = geometry.text_width_pt(label, self.config.text_size * 0.8)
            width = max(width, geometry.pt_to_x(link))
        return width

    def _draw_qubit_lines(self, ax, qubits, y_positions, wire_end):
        """Draw horizontal lines for qubits."""
        for q in qubits:
            y = y_positions[q]
            ax.hlines(
                y,
                -self.config.spacing / 2,
                wire_end,
                color=self.config.wire_color,
                linewidth=1.4,
                zorder=0,
            )
            label_kwargs = dict(
                va="center", fontsize=self.config.text_size, color=self.config.muted_color
            )
            ax.text(-self.config.spacing, y, str(q), ha="right", **label_kwargs)
            ax.text(wire_end + self.config.spacing / 2, y, str(q), ha="left", **label_kwargs)

    def _draw_classical_wire(self, ax, wire_end):
        """Draw the classical wire: one double line for all classical bits."""
        color = self.config.gate_color("measure")
        offset = geometry.pt_to_y(1.3, ax)
        for y in (_CLASSICAL_Y - offset, _CLASSICAL_Y + offset):
            ax.hlines(y, -self.config.spacing / 2, wire_end, color=color, linewidth=1.0, zorder=0)
        label_kwargs = dict(va="center", fontsize=self.config.text_size, color=color)
        ax.text(-self.config.spacing, _CLASSICAL_Y, "c", ha="right", **label_kwargs)
        ax.text(wire_end + self.config.spacing / 2, _CLASSICAL_Y, "c", ha="left", **label_kwargs)

    def _draw_classical_link(self, ax, cmd, x, y_positions, reads, writes):
        """Join a command to the classical wire with a double line.

        A dot on the wire marks a condition being read, an arrowhead a bit
        being written; the bits or the condition are named under the wire.
        """
        color = self.config.gate_color("measure")
        y_gate = min(y_positions[q] for q in cmd.qubits)
        offset = geometry.pt_to_x(1.3, ax)
        for dx in (-offset, offset):
            ax.vlines(x + dx, _CLASSICAL_Y, y_gate, color=color, linewidth=1.0, zorder=0.5)
        if writes:
            ax.plot(x, _CLASSICAL_Y + geometry.pt_to_y(3.5, ax), "v", color=color, markersize=5)
        if reads:
            ax.plot(x, _CLASSICAL_Y, "o", color=color, markersize=self.config.control_size)
        ax.text(
            x,
            _CLASSICAL_Y - geometry.pt_to_y(5, ax),
            _link_label(cmd, reads, writes),
            ha="center",
            va="top",
            fontsize=self.config.text_size * 0.8,
            color=color,
            zorder=3,
        )

    def _draw_measurements(self, ax, commands, x, y_positions):
        """Draw the closing measurements in one column."""
        for k, cmd in enumerate(commands):
            before_ids = {id(a) for a in ax.get_children()}
            draw_meter(ax, self.config, x, y_positions[cmd.qubits[0]])
            for j, artist in enumerate(a for a in ax.get_children() if id(a) not in before_ids):
                artist.set_gid(f"qarp-{self._plot_uid}-m{k}-a{j}")
            self._measure_payload.append({"key": f"m{k}", "qubits": [str(q) for q in cmd.qubits]})

    def _print_circuit_info(self, circuit_data):
        """Print circuit information."""
        print("Circuit information:")
        print(f"  Qubits: {circuit_data.n_qubits}")
        print(f"  Operations (closing measurements excluded): {len(circuit_data.non_measure_cmds)}")
        print(f"  Global phase: {circuit_data.global_phase}")

    def _create_scrollable_output(self, fig, interactive: bool = False):
        """Create scrollable (optionally hover-interactive) HTML output."""
        f = StringIO()
        fig.savefig(f, format="svg", bbox_inches="tight")
        svg = f.getvalue()
        f.close()

        if interactive and (self._gate_registry or self._measure_payload):
            from ._interactive import build_payload, interactive_html

            payload = build_payload(self._gate_registry, self._measure_payload)
            html = interactive_html(svg, payload, self._plot_uid)
        else:
            html = f"""
            <div style="overflow-x:auto; border:1px solid #ccc; width:100%;">
                {svg}
            </div>
            """
        display(HTML(html))
        plt.close(fig)
