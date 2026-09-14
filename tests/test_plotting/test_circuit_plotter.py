"""Tests for the CircuitPlotter class.

Uses qarp Blocks (wrapped via CircuitAdapter) as the input fixture.  The
plotter accepts the adapter via a duck-typed protocol (``.qubits``,
``.get_commands()``, ``.phase``).
"""

import os
import tempfile

import matplotlib
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing

from qarp.blocks import SimpleBlock
from qarp.plotting import CircuitAdapter, CircuitPlotter, LabelMode, PlotConfig, plot


def _adapter(block):
    """Build the block (if not built) and wrap it for the plotter."""
    if not getattr(block, "_built", False):
        block.build()
    return CircuitAdapter(block)


class TestCircuitPlotterModes:
    """Tests for CircuitPlotter with different label modes."""

    @pytest.fixture
    def simple_circuit(self):
        """3-qubit block: Ry, Rz, CX, H."""
        b = SimpleBlock(3, name="simple")
        b.ry(0, 0.5)
        b.rz(1, 0.3)
        b.cx(0, 1)
        b.h(2)
        return _adapter(b)

    def test_truncate_mode_processes_circuit(self, simple_circuit):
        """Test that TRUNCATE mode processes circuit successfully."""
        config = PlotConfig(label_mode=LabelMode.TRUNCATE)
        plotter = CircuitPlotter(config)

        circuit_data = plotter.processor.process_circuit(simple_circuit, False, True, False)

        assert circuit_data.n_qubits == 3
        assert len(circuit_data.non_measure_cmds) == 4

    def test_smart_mode_processes_circuit(self, simple_circuit):
        """Test that SMART mode processes circuit successfully."""
        config = PlotConfig(label_mode=LabelMode.SMART)
        plotter = CircuitPlotter(config)

        circuit_data = plotter.processor.process_circuit(simple_circuit, False, True, False)

        assert circuit_data.n_qubits == 3
        assert len(circuit_data.non_measure_cmds) == 4

    def test_full_mode_processes_circuit(self, simple_circuit):
        """Test that FULL mode processes circuit successfully."""
        config = PlotConfig(label_mode=LabelMode.FULL)
        plotter = CircuitPlotter(config)

        circuit_data = plotter.processor.process_circuit(simple_circuit, False, True, False)

        assert circuit_data.n_qubits == 3
        assert len(circuit_data.non_measure_cmds) == 4


class TestCircuitPlotterRendering:
    """Tests for CircuitPlotter rendering to file."""

    @pytest.fixture
    def test_circuit(self):
        """4-qubit block with mixed parametric and non-parametric gates."""
        b = SimpleBlock(4, name="render_test")
        b.ry(0, 0.123456789)
        b.rz(0, 0.987654321)
        b.rx(0, 0.555555555)
        b.ry(1, 0.111111111)
        b.cx(0, 1)
        b.h(2)
        b.cx(1, 2)
        b.rz(3, 0.333)
        return _adapter(b)

    def test_render_truncate_mode(self, test_circuit):
        """Test rendering with TRUNCATE mode saves a file."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            filename = f.name

        try:
            plot(test_circuit, label_mode=LabelMode.TRUNCATE, save_fig=filename)
            assert os.path.exists(filename)
            assert os.path.getsize(filename) > 0
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_render_smart_mode(self, test_circuit):
        """Test rendering with SMART mode saves a file."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            filename = f.name

        try:
            plot(test_circuit, label_mode=LabelMode.SMART, save_fig=filename)
            assert os.path.exists(filename)
            assert os.path.getsize(filename) > 0
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_render_full_mode(self, test_circuit):
        """Test rendering with FULL mode saves a file."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            filename = f.name

        try:
            plot(test_circuit, label_mode=LabelMode.FULL, save_fig=filename)
            assert os.path.exists(filename)
            assert os.path.getsize(filename) > 0
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_render_different_file_sizes(self, test_circuit):
        """Test that TRUNCATE mode produces smaller files than FULL mode."""
        files = {}

        try:
            for mode in [LabelMode.TRUNCATE, LabelMode.SMART, LabelMode.FULL]:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    filename = f.name
                files[mode] = filename
                plot(test_circuit, label_mode=mode, save_fig=filename)

            # TRUNCATE should generally produce smaller files due to shorter labels
            truncate_size = os.path.getsize(files[LabelMode.TRUNCATE])
            full_size = os.path.getsize(files[LabelMode.FULL])

            # This is a soft assertion - truncate should be smaller or equal
            assert truncate_size <= full_size * 1.1  # Allow 10% tolerance

        finally:
            for filename in files.values():
                if os.path.exists(filename):
                    os.unlink(filename)


class TestCircuitPlotterLabelModeOverride:
    """Tests for label_mode override in plot method."""

    @pytest.fixture
    def simple_circuit(self):
        b = SimpleBlock(2, name="override")
        b.ry(0, 0.5)
        b.h(1)
        return _adapter(b)

    def test_label_mode_override(self, simple_circuit):
        """Test that label_mode parameter overrides config."""
        # Create plotter with FULL mode in config
        config = PlotConfig(label_mode=LabelMode.FULL)
        plotter = CircuitPlotter(config)

        # Initially should be FULL mode
        assert plotter.config.label_mode == LabelMode.FULL

        # Plot with TRUNCATE override
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            filename = f.name

        try:
            plotter.plot(simple_circuit, label_mode=LabelMode.TRUNCATE, save_fig=filename)
            # Config should now be TRUNCATE
            assert plotter.config.label_mode == LabelMode.TRUNCATE
        finally:
            if os.path.exists(filename):
                os.unlink(filename)


class TestPlotConvenienceFunction:
    """Tests for the plot() convenience function."""

    @pytest.fixture
    def simple_circuit(self):
        b = SimpleBlock(2, name="conv")
        b.ry(0, 0.5)
        b.cx(0, 1)
        return _adapter(b)

    def test_plot_with_label_mode(self, simple_circuit):
        """Test plot function accepts label_mode parameter."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            filename = f.name

        try:
            # Should not raise
            plot(simple_circuit, label_mode=LabelMode.TRUNCATE, save_fig=filename)
            assert os.path.exists(filename)
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_plot_with_color_scheme_and_label_mode(self, simple_circuit):
        """Test plot function with both color_scheme and label_mode."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            filename = f.name

        try:
            plot(
                simple_circuit,
                color_scheme="colorblind",
                label_mode=LabelMode.SMART,
                save_fig=filename,
            )
            assert os.path.exists(filename)
        finally:
            if os.path.exists(filename):
                os.unlink(filename)


class TestGateIndexing:
    """Tests for gate indexing and querying feature."""

    @pytest.fixture
    def parameterized_circuit(self):
        """5-gate circuit with parametric and non-parametric gates."""
        b = SimpleBlock(3, name="param")
        b.ry(0, 0.123456789)  # Gate 0
        b.rz(1, 0.987654321)  # Gate 1
        b.cx(0, 1)  # Gate 2
        b.h(2)  # Gate 3
        b.rx(0, 0.555)  # Gate 4
        return _adapter(b)

    def test_gate_registry_populated(self, parameterized_circuit):
        """Test that gate registry is populated after plotting."""
        config = PlotConfig()
        plotter = CircuitPlotter(config)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            filename = f.name

        try:
            plotter.plot(parameterized_circuit, save_fig=filename)

            # Should have 5 gates registered
            assert len(plotter.gates) == 5
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_query_gate_returns_correct_info(self, parameterized_circuit):
        """Test that query_gate returns correct gate information."""
        config = PlotConfig()
        plotter = CircuitPlotter(config)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            filename = f.name

        try:
            plotter.plot(parameterized_circuit, save_fig=filename)

            # Query first gate (Ry)
            gate = plotter.query_gate(0)
            assert gate is not None
            assert gate.gate_type == "Ry"
            # Check that the full_label contains "Ry" and a parameter value
            assert "Ry" in gate.full_label
            assert "0.12" in gate.full_label
            assert len(gate.qubits) == 1

            # Query CX gate.  The CircuitAdapter preserves qarp's flat build
            # order: Ry, Rz, CX, H, Rx — so CX sits at index 2 (vs index 3
            # under pytket's parallelism-aware ordering).
            gate2 = plotter.query_gate(2)
            assert gate2 is not None
            assert gate2.gate_type == "CX"
            assert len(gate2.qubits) == 2
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_query_invalid_index_returns_none(self, parameterized_circuit):
        """Test that querying invalid index returns None."""
        config = PlotConfig()
        plotter = CircuitPlotter(config)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            filename = f.name

        try:
            plotter.plot(parameterized_circuit, save_fig=filename)

            # Query non-existent gate
            gate = plotter.query_gate(999)
            assert gate is None
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_show_gate_indices_renders(self, parameterized_circuit):
        """Test that show_gate_indices renders without error."""
        config = PlotConfig(show_gate_indices=True)
        plotter = CircuitPlotter(config)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            filename = f.name

        try:
            plotter.plot(parameterized_circuit, save_fig=filename)
            assert os.path.exists(filename)
            # File should be larger with indices
            assert os.path.getsize(filename) > 0
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_show_gate_indices_via_plot_param(self, parameterized_circuit):
        """Test enabling gate indices via plot parameter."""
        config = PlotConfig(show_gate_indices=False)
        plotter = CircuitPlotter(config)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            filename = f.name

        try:
            plotter.plot(parameterized_circuit, show_gate_indices=True, save_fig=filename)
            # Config should be updated
            assert plotter.config.show_gate_indices
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_gate_registry_reset_on_new_plot(self, parameterized_circuit):
        """Test that gate registry is reset when plotting a new circuit."""
        config = PlotConfig()
        plotter = CircuitPlotter(config)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            filename = f.name

        try:
            # First plot
            plotter.plot(parameterized_circuit, save_fig=filename)
            assert len(plotter.gates) == 5

            # Plot a smaller circuit
            small = SimpleBlock(1, name="small")
            small.h(0)
            plotter.plot(_adapter(small), save_fig=filename)

            # Registry should be reset to 1 gate
            assert len(plotter.gates) == 1
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_convenience_function_returns_plotter(self, parameterized_circuit):
        """Test that plot() convenience function returns plotter for querying."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            filename = f.name

        try:
            plotter = plot(
                parameterized_circuit,
                show_gate_indices=True,
                return_plotter=True,
                save_fig=filename,
            )

            # Should return CircuitPlotter
            assert isinstance(plotter, CircuitPlotter)

            # Should be able to query gates
            gate = plotter.query_gate(0)
            assert gate is not None
        finally:
            if os.path.exists(filename):
                os.unlink(filename)

    def test_gate_info_str_representation(self, parameterized_circuit):
        """Test GateInfo string representation."""
        config = PlotConfig()
        plotter = CircuitPlotter(config)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            filename = f.name

        try:
            plotter.plot(parameterized_circuit, save_fig=filename)

            gate = plotter.query_gate(0)

            # Test __str__
            str_repr = str(gate)
            assert "Gate Index: 0" in str_repr
            assert "Gate Type: Ry" in str_repr
            assert "Full Expression:" in str_repr

            # Test __repr__
            repr_str = repr(gate)
            assert "Gate[0]:" in repr_str
        finally:
            if os.path.exists(filename):
                os.unlink(filename)


def _box_labels(adapter):
    """Names of the box (Block / ControlledBlock) commands in an adapter."""
    return [
        cmd.op.circuit_name for cmd in adapter.get_commands() if hasattr(cmd.op, "circuit_name")
    ]


class TestCompositeBlockBoxing:
    """A built CompositeBlock renders one box per child sub-block; leaves and
    ``decompose_boxes=True`` render primitive gates."""

    @pytest.fixture
    def composite(self):
        """3-child composite over 3 qubits: two named leaves + a CX leaf."""
        from qarp.blocks import CompositeBlock

        a = SimpleBlock(3, name="Prep")
        a.h(0)
        a.h(1)
        b = SimpleBlock(3, name="Entangle")
        b.cx(0, 1)
        b.cx(1, 2)
        return CompositeBlock([a, b], 3, name="Comp").build()

    def test_children_render_as_boxes(self, composite):
        adapter = CircuitAdapter(composite)
        cmds = adapter.get_commands()
        # One box per child, not the 4 flattened primitive gates.
        assert len(cmds) == 2
        assert all(str(cmd.op).lower() == "block" for cmd in cmds)
        assert _box_labels(adapter) == ["Prep", "Entangle"]

    def test_box_spans_child_target_qubits(self, composite):
        adapter = CircuitAdapter(composite)
        # Both children act on the full 3-qubit frame.
        for cmd in adapter.get_commands():
            assert sorted(q.idx for q in cmd.qubits) == [0, 1, 2]

    def test_decompose_boxes_flattens(self, composite):
        adapter = CircuitAdapter(composite, decompose_boxes=True)
        cmds = adapter.get_commands()
        assert len(cmds) == 4  # h, h, cx, cx
        assert not any(hasattr(cmd.op, "circuit_name") for cmd in cmds)

    def test_leaf_block_is_not_boxed(self):
        leaf = SimpleBlock(2, name="leaf")
        leaf.h(0)
        leaf.cx(0, 1)
        leaf.build()
        adapter = CircuitAdapter(leaf)
        cmds = adapter.get_commands()
        assert [str(c.op) for c in cmds] == ["H", "CX"]

    def test_drill_into_box_returns_inner_circuit(self, composite):
        plotter = CircuitPlotter()
        plotter.plot(CircuitAdapter(composite), _show=False)
        inner = plotter.get_inner_circuit(0)  # "Prep"
        assert inner is not None
        assert inner.name == "Prep"
        assert [str(c.op) for c in inner.get_commands()] == ["H", "H"]


class TestControlledAndMeasureChildren:
    """Controlled children render as controlled-block boxes; measurement-only
    children pass through as measure commands (the plot's ``M`` markers)."""

    def test_controlled_block_op_exposes_control_metadata(self):
        """The box op for a controlled child exposes the control metadata the
        ``ControlledBlockRenderer`` reads."""
        from qarp.blocks import ControlledBlock, XnBlock
        from qarp.plotting._circuit_adapter import _ControlledBlockOp

        cb = ControlledBlock(XnBlock(2).build(), 2, [True, False]).build()
        op = _ControlledBlockOp(cb, cb.inner())
        assert str(op).lower() == "controlledblock"  # → routed to controlled_block renderer
        assert op.get_n_controls() == 2
        # §6: controls are LSB-first, bit k = control qubit k, so
        # [True, False] packs to 0b01 — q0 requires 1, q1 requires 0.
        assert op.get_control_state() == 0b01
        assert op.get_op().get_circuit().name == cb.inner().name

    def test_measure_only_child_passes_through(self):
        from qarp.blocks import CompositeBlock, HnBlock, ReadoutBlock

        comp = CompositeBlock([HnBlock(2).build(), ReadoutBlock(2).build()], 2).build()
        cmds = CircuitAdapter(comp).get_commands()
        # 1 box (Hn) + 2 measure commands (not a box around the readout)
        assert sum(str(c.op).lower() == "block" for c in cmds) == 1
        assert sum(str(c.op).lower() == "measure" for c in cmds) == 2

    def test_single_qubit_leaf_child_renders_as_gate(self):
        """A single-qubit leaf child (an ``AncillaH``-style wrapper) shows its
        bare gate on its target wire — not an opaque box."""
        from qarp.blocks import CompositeBlock

        ancilla_h = SimpleBlock(1, name="AncillaH")
        ancilla_h.h(0)
        ancilla_h.build()
        ancilla_h.target_qubits = [2]  # parent-frame wire

        entangle = SimpleBlock(3, name="Entangle")
        entangle.cx(0, 1)

        comp = CompositeBlock([entangle, ancilla_h], 3, name="Comp").build()
        cmds = CircuitAdapter(comp).get_commands()

        # Entangle stays boxed; the single-qubit AncillaH is a bare H on wire 2.
        assert _box_labels(CircuitAdapter(comp)) == ["Entangle"]
        h_cmds = [c for c in cmds if str(c.op) == "H"]
        assert len(h_cmds) == 1
        assert [q.idx for q in h_cmds[0].qubits] == [2]
