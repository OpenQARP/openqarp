"""Tests for gate renderers.

Uses qarp Blocks wrapped via :class:`CircuitAdapter` to obtain the
``_Command`` objects the renderers expect.
"""

import matplotlib
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing

import matplotlib.pyplot as plt

from qarp.blocks import SimpleBlock
from qarp.plotting import CircuitAdapter, LabelMode, PlotConfig
from qarp.plotting._renderers import (
    ControlledGateRenderer,
    MultiGateRenderer,
    RendererFactory,
    SingleGateRenderer,
)
from qarp.plotting._renderers.label_manager import LabelManager


def _first_command(block):
    """Build the block and return the first wrapped command from CircuitAdapter."""
    block.build()
    return CircuitAdapter(block).get_commands()[0]


class TestRendererFactory:
    """Tests for RendererFactory."""

    def test_factory_creates_label_manager(self):
        """Test that factory creates a label manager."""
        config = PlotConfig()
        factory = RendererFactory(config)

        assert factory.label_manager is not None
        assert isinstance(factory.label_manager, LabelManager)

    def test_factory_provides_all_renderers(self):
        """Test that factory provides all renderer types."""
        config = PlotConfig()
        factory = RendererFactory(config)

        renderer_types = ["single", "controlled", "boxed", "controlled_block", "multi"]
        for rtype in renderer_types:
            renderer = factory.get_renderer(rtype)
            assert renderer is not None

    def test_factory_reset_label_manager(self):
        """Test that reset_label_manager clears state."""
        config = PlotConfig(label_mode=LabelMode.SMART)
        factory = RendererFactory(config)

        # Add some labels
        factory.label_manager.compute_position(0, 1.0, "Test")
        factory.label_manager.compute_position(0, 1.0, "Test2")

        # Reset
        factory.reset_label_manager()

        # After reset, first position should be unchanged
        y, x_offset, _ = factory.label_manager.compute_position(0, 1.0, "Test")
        assert y == 1.0
        assert x_offset == 0.0

    def test_classify_single_qubit_gate(self):
        """Test classification of single qubit gates."""
        config = PlotConfig()
        factory = RendererFactory(config)

        b = SimpleBlock(1, name="h_only")
        b.h(0)
        cmd = _first_command(b)

        assert factory.classify_gate(cmd) == "single"

    def test_classify_controlled_gate(self):
        """Test classification of controlled gates."""
        config = PlotConfig()
        factory = RendererFactory(config)

        b = SimpleBlock(2, name="cx_only")
        b.cx(0, 1)
        cmd = _first_command(b)

        assert factory.classify_gate(cmd) == "controlled"

    def test_classify_multi_gate(self):
        """Test classification of multi-qubit gates."""
        config = PlotConfig()
        factory = RendererFactory(config)

        b = SimpleBlock(2, name="swap_only")
        b.swap(0, 1)
        cmd = _first_command(b)

        assert factory.classify_gate(cmd) == "multi"

    def test_classify_controlled_clifford_singles_as_controlled(self):
        """CH / CS / CSdg / CSX / CSXdg draw as control dot + target box."""
        config = PlotConfig()
        factory = RendererFactory(config)
        for name in ("ch", "cs", "csdg", "csx", "csxdg"):
            b = SimpleBlock(2, name=name)
            getattr(b, name)(0, 1)
            assert factory.classify_gate(_first_command(b)) == "controlled", name

    def test_classify_mcz_as_controlled(self):
        """MCZ routes to the controlled renderer (standard multi-control dots),
        not the multi (linked-box) renderer."""
        config = PlotConfig()
        factory = RendererFactory(config)

        b = SimpleBlock(4, name="mcz_only")
        b.mcz([0, 1, 2, 3])
        cmd = _first_command(b)

        assert factory.classify_gate(cmd) == "controlled"


class TestSingleGateRenderer:
    """Tests for SingleGateRenderer."""

    @pytest.fixture
    def renderer_with_manager(self):
        """Create a renderer with label manager."""
        config = PlotConfig(label_mode=LabelMode.SMART)
        label_manager = LabelManager(config)
        return SingleGateRenderer(config, label_manager)

    @pytest.fixture
    def renderer_without_manager(self):
        """Create a renderer without a label manager."""
        config = PlotConfig()
        return SingleGateRenderer(config)

    def test_render_single_gate(self, renderer_with_manager):
        """Test rendering a single qubit gate."""
        fig, ax = plt.subplots()

        b = SimpleBlock(1, name="ry")
        b.ry(0, 0.5)
        cmd = _first_command(b)

        y_positions = {cmd.qubits[0]: 0}
        prev_label_y = (None, None)

        result = renderer_with_manager.render(ax, cmd, 0.0, y_positions, prev_label_y)

        assert result is not None
        assert len(result) == 2  # (y_text, is_short_text)

        plt.close(fig)

    def test_render_without_label_manager(self, renderer_without_manager):
        """Rendering without a label manager falls back to unmanaged placement."""
        fig, ax = plt.subplots()

        b = SimpleBlock(1, name="h")
        b.h(0)
        cmd = _first_command(b)

        y_positions = {cmd.qubits[0]: 0}
        prev_label_y = (None, None)

        result = renderer_without_manager.render(ax, cmd, 0.0, y_positions, prev_label_y)

        assert result is not None

        plt.close(fig)


class TestControlledGateRenderer:
    """Tests for ControlledGateRenderer."""

    @pytest.fixture
    def renderer(self):
        """Create a renderer with label manager."""
        config = PlotConfig(label_mode=LabelMode.SMART)
        label_manager = LabelManager(config)
        return ControlledGateRenderer(config, label_manager)

    def test_render_cx_gate(self, renderer):
        """Test rendering a CX gate."""
        fig, ax = plt.subplots()

        b = SimpleBlock(2, name="cx")
        b.cx(0, 1)
        cmd = _first_command(b)

        y_positions = {cmd.qubits[0]: 0, cmd.qubits[1]: 1}
        prev_label_y = (None, None)

        result = renderer.render(ax, cmd, 0.0, y_positions, prev_label_y)

        assert result is not None
        # CX keeps a distinguished target: a square marker plus a label.
        assert "s" in [line.get_marker() for line in ax.lines]

        plt.close(fig)

    def test_render_mcz_as_symmetric_dots(self, renderer):
        """MCZ renders as the standard symmetric form: a control dot on every
        wire and a single vertical line — no target box, no label."""
        fig, ax = plt.subplots()

        b = SimpleBlock(4, name="mcz")
        b.mcz([0, 1, 2, 3])
        cmd = _first_command(b)

        y_positions = {q: i for i, q in enumerate(cmd.qubits)}
        renderer.render(ax, cmd, 0.0, y_positions, (None, None))

        markers = [line.get_marker() for line in ax.lines]
        assert markers == ["o", "o", "o", "o"]  # one dot per qubit, no squares
        assert len(ax.texts) == 0  # symmetric form carries no gate label

        plt.close(fig)


class TestMultiGateRenderer:
    """Tests for MultiGateRenderer."""

    @pytest.fixture
    def renderer(self):
        """Create a renderer with label manager."""
        config = PlotConfig(label_mode=LabelMode.SMART)
        label_manager = LabelManager(config)
        return MultiGateRenderer(config, label_manager)

    def test_render_swap_gate(self, renderer):
        """Test rendering a SWAP gate."""
        fig, ax = plt.subplots()

        b = SimpleBlock(2, name="swap")
        b.swap(0, 1)
        cmd = _first_command(b)

        y_positions = {cmd.qubits[0]: 0, cmd.qubits[1]: 1}
        prev_label_y = (None, None)

        result = renderer.render(ax, cmd, 0.0, y_positions, prev_label_y)

        assert result is not None

        plt.close(fig)


class TestRendererLabelModeIntegration:
    """Integration tests for renderers with different label modes."""

    @pytest.fixture
    def first_param_command(self):
        """First gate of a 2-qubit Ry/Rz block, wrapped via CircuitAdapter."""
        b = SimpleBlock(2, name="params")
        b.ry(0, 0.123456789)
        b.rz(1, 0.987654321)
        return _first_command(b)

    def test_truncate_mode_shortens_labels(self, first_param_command):
        """Test that truncate mode produces shorter labels."""
        config = PlotConfig(label_mode=LabelMode.TRUNCATE)
        factory = RendererFactory(config)

        fig, ax = plt.subplots()

        cmd = first_param_command
        y_positions = {cmd.qubits[0]: 0}

        renderer = factory.get_renderer("single")
        result = renderer.render(ax, cmd, 0.0, y_positions, (None, None))

        # In truncate mode, label should be short
        y_text, is_short_text = result
        assert is_short_text  # Truncated labels should be short

        plt.close(fig)

    def test_full_mode_preserves_labels(self, first_param_command):
        """Test that full mode preserves complete labels."""
        config = PlotConfig(label_mode=LabelMode.FULL)
        factory = RendererFactory(config)

        fig, ax = plt.subplots()

        cmd = first_param_command
        y_positions = {cmd.qubits[0]: 0}

        renderer = factory.get_renderer("single")
        result = renderer.render(ax, cmd, 0.0, y_positions, (None, None))

        # Result should be valid
        assert result is not None

        plt.close(fig)


# =============================================================================
# MultiGateRenderer branches, ControlledBlockRenderer
# =============================================================================
from matplotlib.patches import Rectangle

from qarp.blocks import CompositeBlock, ControlledBlock, XnBlock
from qarp.plotting._renderers import ControlledBlockRenderer


def _multi_renderer():
    config = PlotConfig(label_mode=LabelMode.SMART)
    return MultiGateRenderer(config, LabelManager(config))


def _marker_counts(ax):
    from collections import Counter

    return Counter(line.get_marker() for line in ax.lines)


class TestMultiGateRendererBranches:
    def test_swap_draws_two_x_markers(self):
        fig, ax = plt.subplots()
        b = SimpleBlock(2, name="swap")
        b.swap(0, 1)
        cmd = _first_command(b)
        y_positions = {q: i for i, q in enumerate(cmd.qubits)}

        result = _multi_renderer().render(ax, cmd, 0.0, y_positions, (None, None))
        assert len(result) == 2
        assert _marker_counts(ax)["x"] == 2
        plt.close(fig)

    def test_cswap_draws_control_dot_and_two_x_markers(self):
        fig, ax = plt.subplots()
        b = SimpleBlock(3, name="cswap")
        b.cswap(0, 1, 2)
        cmd = _first_command(b)
        y_positions = {q: i for i, q in enumerate(cmd.qubits)}

        _multi_renderer().render(ax, cmd, 0.0, y_positions, (None, None))
        counts = _marker_counts(ax)
        assert counts["o"] == 1  # control dot on the first qubit
        assert counts["x"] == 2  # swap crosses on the targets
        plt.close(fig)

    def test_generic_multi_gate_draws_boxes_and_label(self):
        fig, ax = plt.subplots()
        b = SimpleBlock(2, name="rzz")
        b.rzz(0, 1, 0.4)
        cmd = _first_command(b)
        y_positions = {q: i for i, q in enumerate(cmd.qubits)}

        _multi_renderer().render(ax, cmd, 0.0, y_positions, (None, None))
        assert _marker_counts(ax)["s"] == 2  # one square per qubit
        assert any("RZZ" in t.get_text().upper() for t in ax.texts)
        plt.close(fig)


class TestControlledBlockRenderer:
    def _controlled_block_cmd(self):
        cb = ControlledBlock(XnBlock(2).build(), 2, [True, False])
        parent = CompositeBlock([cb], 4)
        parent.build()
        for cmd in CircuitAdapter(parent).get_commands():
            if str(cmd.op).lower() == "controlledblock":
                return cmd
        raise AssertionError("no controlled-block command found")

    def test_render_draws_box_controls_and_qubit_numbers(self):
        fig, ax = plt.subplots()
        cmd = self._controlled_block_cmd()
        y_positions = {q: i for i, q in enumerate(cmd.qubits)}

        config = PlotConfig(label_mode=LabelMode.SMART)
        renderer = ControlledBlockRenderer(config, LabelManager(config))
        result = renderer.render(ax, cmd, 0.0, y_positions, (None, None))

        assert len(result) == 2
        boxes = [p for p in ax.patches if isinstance(p, Rectangle)]
        assert len(boxes) == 1
        # Two control dots: one filled (state 1), one open (state 0).
        dots = [line for line in ax.lines if line.get_marker() == "o"]
        assert len(dots) == 2
        face_colors = {str(d.get_markerfacecolor()) for d in dots}
        assert "white" in face_colors  # the open (state-0) control
        # Target-qubit index annotations inside the box: 2 targets.
        number_texts = [t for t in ax.texts if t.get_text() in {"0", "1", "2", "3"}]
        assert len(number_texts) == 2
        plt.close(fig)


def test_controlled_block_dot_fill_matches_control_state_positionally():
    """Open/filled control dots must land on the right wires.

    §6 orders controls LSB-first (q0 = lowest bit), so an asymmetric state
    distinguishes a correct reading from a reversed one — a count-only
    assertion cannot.
    """
    from qarp.blocks import CompositeBlock, ControlledBlock, XnBlock

    state = [True, False, False]
    cb = ControlledBlock(XnBlock(1).build(), len(state), list(state))
    parent = CompositeBlock([cb], 4)
    parent.build()
    cmd = next(
        c for c in CircuitAdapter(parent).get_commands() if str(c.op).lower() == "controlledblock"
    )

    fig, ax = plt.subplots()
    config = PlotConfig(label_mode=LabelMode.SMART)
    ControlledBlockRenderer(config, LabelManager(config)).render(
        ax, cmd, 0.0, {q: i for i, q in enumerate(cmd.qubits)}, (None, None)
    )
    dots = [line for line in ax.lines if line.get_marker() == "o"]
    assert len(dots) == len(state)
    filled = [str(d.get_markerfacecolor()) != "white" for d in dots]
    assert filled == list(state), f"dots {filled} do not match control state {state}"
    plt.close(fig)


class TestControlledGateCoverage:
    """Every controlled ``GateType`` must reach the controlled renderer.

    The classifier used to match a hand-written name list against the full
    gate label, so parameterized controls (whose label reads ``"CRz(0.3)"``)
    and unlisted ones silently fell through to the multi renderer and drew as
    anonymous boxes with no control dot.
    """

    # (builder method, args, n_qubits) — built inside the test, never at module
    # scope, so no qarpx object outlives the call.
    SINGLE_TARGET = [
        ("cx", (0, 1), 2),
        ("cy", (0, 1), 2),
        ("cz", (0, 1), 2),
        ("cp", (0, 1, 0.3), 2),
        ("crx", (0, 1, 0.3), 2),
        ("cry", (0, 1, 0.3), 2),
        ("crz", (0, 1, 0.3), 2),
        ("ccx", (0, 1, 2), 3),
    ]

    @pytest.mark.parametrize(("method", "args", "n_qubits"), SINGLE_TARGET)
    def test_single_target_controls_route_to_controlled(self, method, args, n_qubits):
        b = SimpleBlock(n_qubits, name=method)
        getattr(b, method)(*args)
        cmd = _first_command(b)

        assert RendererFactory(PlotConfig()).classify_gate(cmd) == "controlled"

    def test_cswap_stays_multi(self):
        """CSWAP has two targets, so the multi renderer draws its swap pair;
        the controlled renderer would draw a single target marker and lose one
        of the swapped wires."""
        b = SimpleBlock(3, name="cswap")
        b.cswap(0, 1, 2)
        cmd = _first_command(b)

        assert RendererFactory(PlotConfig()).classify_gate(cmd) == "multi"

    def test_controlled_rotation_draws_dot_and_keeps_angle(self):
        """CRz(θ) renders as control dot + target marker, with θ still shown."""
        fig, ax = plt.subplots()
        b = SimpleBlock(2, name="crz")
        b.crz(0, 1, 0.3)
        cmd = _first_command(b)
        config = PlotConfig()

        ControlledGateRenderer(config, LabelManager(config)).render(
            ax, cmd, 0.0, {q: i for i, q in enumerate(cmd.qubits)}, (None, None)
        )

        counts = _marker_counts(ax)
        assert counts["o"] == 1  # control dot
        assert counts["s"] == 1  # single target marker
        assert any("0.3" in t.get_text() for t in ax.texts), "rotation angle dropped from label"
        plt.close(fig)

    def test_controlled_gate_target_takes_base_gate_color(self):
        """The target marker is colored by the base gate, so CRz matches Rz.

        Stripping only the leading "c" left "rz(0.3)" as the palette key, which
        missed every scheme and fell back to the default color.
        """
        config = PlotConfig(gate_colors={"rz": "#123456"}, gate_style="filled")

        fig, ax = plt.subplots()
        b = SimpleBlock(2, name="crz")
        b.crz(0, 1, 0.3)
        cmd = _first_command(b)
        ControlledGateRenderer(config, LabelManager(config)).render(
            ax, cmd, 0.0, {q: i for i, q in enumerate(cmd.qubits)}, (None, None)
        )

        target = next(line for line in ax.lines if line.get_marker() == "s")
        assert target.get_markerfacecolor() == "#123456"
        plt.close(fig)


def test_color_schemes_only_name_real_gates():
    """Palette keys must be gate names the backend can actually emit.

    The schemes carried entries for gates from other SDKs (``sx``, ``tk1``,
    ``u1``…) that no ``GateType`` produces, so those colors were unreachable
    while real gates fell back to the default color.
    """
    import qarpx as qx
    from qarp.plotting.styles import COLOR_SCHEMES

    real = {qx.gate_name(g).lower() for g in qx.GateType.__members__.values()}
    real.add("box")  # structural key for block/box fills, not a gate

    for scheme_name, scheme in COLOR_SCHEMES.items():
        unknown = sorted(set(scheme) - real)
        assert not unknown, f"{scheme_name} names non-existent gates: {unknown}"
