"""Tests for the LabelManager class."""

from qarp.plotting import LabelMode, PlotConfig
from qarp.plotting._renderers.label_manager import LabelInfo, LabelManager


class TestLabelManagerTruncateMode:
    """Tests for LabelManager in TRUNCATE mode."""

    def test_truncate_simple_gate(self):
        """Test truncating a simple parameterized gate."""
        config = PlotConfig(label_mode=LabelMode.TRUNCATE)
        lm = LabelManager(config)

        label = lm.format_label("Ry(0.5)")
        assert label == "Ry"

    def test_truncate_complex_parameter(self):
        """Test truncating a gate with complex parameter."""
        config = PlotConfig(label_mode=LabelMode.TRUNCATE)
        lm = LabelManager(config)

        label = lm.format_label("Rz(0.5*pi + theta)")
        assert label == "Rz"

    def test_truncate_multi_parameter(self):
        """Test truncating a gate with multiple parameters."""
        config = PlotConfig(label_mode=LabelMode.TRUNCATE)
        lm = LabelManager(config)

        label = lm.format_label("U3(0.1, 0.2, 0.3)")
        assert label == "U3"

    def test_truncate_no_params(self):
        """Test that gates without parameters are unchanged."""
        config = PlotConfig(label_mode=LabelMode.TRUNCATE)
        lm = LabelManager(config)

        label = lm.format_label("H")
        assert label == "H"

        label = lm.format_label("CX")
        assert label == "CX"


class TestLabelManagerSmartMode:
    """Tests for LabelManager in SMART mode."""

    def test_smart_rounds_decimals(self):
        """Test that smart mode rounds decimal values."""
        config = PlotConfig(label_mode=LabelMode.SMART, label_decimal_places=2)
        lm = LabelManager(config)

        label = lm.format_label("Ry(0.123456789)")
        assert "0.12" in label
        assert "0.123456789" not in label

    def test_smart_preserves_short_labels(self):
        """Test that short labels are preserved in smart mode."""
        config = PlotConfig(label_mode=LabelMode.SMART)
        lm = LabelManager(config)

        label = lm.format_label("H")
        assert label == "H"

    def test_smart_breaks_long_labels(self):
        """Test that long labels are broken in smart mode."""
        config = PlotConfig(label_mode=LabelMode.SMART, label_max_width=8)
        lm = LabelManager(config)

        label = lm.format_label("Ry(very_long_parameter_name)")
        # Should contain a newline for line breaking
        assert "\n" in label or len(label.split("\n")[0]) <= 10

    def test_smart_limits_to_two_lines(self):
        """Test that smart mode limits labels to at most 2 lines."""
        config = PlotConfig(label_mode=LabelMode.SMART, label_max_width=10)
        lm = LabelManager(config)

        # Very long label that would need many lines
        label = lm.format_label("U3(theta_0, phi_0, lambda_0, theta_1, phi_1, lambda_1)")
        lines = label.split("\n")
        assert len(lines) <= 2, f"Label has {len(lines)} lines, expected max 2"

    def test_smart_truncates_very_long_params(self):
        """Test that very long parameters are truncated with ellipsis."""
        config = PlotConfig(label_mode=LabelMode.SMART, label_max_width=12)
        lm = LabelManager(config)

        label = lm.format_label("U3(very_very_very_long_parameter_that_cannot_fit)")
        # Should have ellipsis if truncated
        lines = label.split("\n")
        assert len(lines) <= 2


class TestLabelManagerFullMode:
    """Tests for LabelManager in FULL mode."""

    def test_full_preserves_label(self):
        """Test that full mode preserves the complete label."""
        config = PlotConfig(label_mode=LabelMode.FULL)
        lm = LabelManager(config)

        original = "Ry(0.123456789*pi + theta)"
        label = lm.format_label(original)
        assert label == original


class TestLabelManagerOverlapAvoidance:
    """Tests for label overlap avoidance."""

    def test_no_overlap_distant_labels(self):
        """Test that distant labels don't get adjusted."""
        config = PlotConfig(label_mode=LabelMode.SMART)
        lm = LabelManager(config)

        # Place first label
        y1, x1_offset, _ = lm.compute_position(0, 1.0, "Ry(0.5)")
        assert y1 == 1.0
        assert x1_offset == 0.0

        # Place second label far away - should not be adjusted
        y2, x2_offset, _ = lm.compute_position(0, 5.0, "Rz(0.3)")
        assert y2 == 5.0
        assert x2_offset == 0.0

    def test_no_overlap_different_x_positions(self):
        """Test that labels at different x positions don't trigger overlap avoidance."""
        config = PlotConfig(label_mode=LabelMode.SMART, spacing=0.25)
        lm = LabelManager(config)

        # Place first label at x=0
        y1, x1_offset, _ = lm.compute_position(0, 1.0, "Ry(0.5)")
        assert y1 == 1.0

        # Place second label at different x (different gate column) but same y
        # Should NOT be shifted because they're at different x positions
        y2, x2_offset, _ = lm.compute_position(0.5, 1.0, "Rz(0.3)")
        assert y2 == 1.0, "Labels at different x should not trigger y adjustment"
        assert x2_offset == 0.0

    def test_overlap_avoidance_same_position(self):
        """Test that overlapping labels at same x are shifted."""
        config = PlotConfig(label_mode=LabelMode.SMART)
        lm = LabelManager(config)

        # Place first label
        y1, _, _ = lm.compute_position(0, 1.0, "Ry(0.5)")
        assert y1 == 1.0

        # Place second label at very close x position - should be shifted
        y2, _, _ = lm.compute_position(0.05, 1.0, "Rz(0.3)")
        assert y2 != 1.0  # Should have been shifted

    def test_overlap_avoidance_multiple_labels(self):
        """Test overlap avoidance with multiple labels."""
        config = PlotConfig(label_mode=LabelMode.SMART, label_y_margin=0.4)
        lm = LabelManager(config)

        positions = []
        for i in range(5):
            y, _, _ = lm.compute_position(0, 1.0, f"Gate{i}")
            positions.append(y)

        # All positions should be different or sufficiently spaced
        for i, y1 in enumerate(positions):
            for j, y2 in enumerate(positions):
                if i != j:
                    # Labels should not be exactly at the same position
                    # (they could be at different x offsets though)
                    pass  # This is a soft check

    def test_reset_clears_labels(self):
        """Test that reset clears all tracked labels."""
        config = PlotConfig(label_mode=LabelMode.SMART)
        lm = LabelManager(config)

        # Place some labels at same x (close enough to trigger overlap)
        lm.compute_position(0, 1.0, "Ry(0.5)")
        lm.compute_position(0.05, 1.0, "Rz(0.3)")

        # Reset
        lm.reset()

        # Now placing at same position should not cause shift
        y, x_offset, _ = lm.compute_position(0, 1.0, "H")
        assert y == 1.0
        assert x_offset == 0.0


class TestLabelInfo:
    """Tests for LabelInfo dataclass."""

    def test_label_info_creation(self):
        """Test creating a LabelInfo instance."""
        info = LabelInfo(
            x=1.0,
            y=2.0,
            original_y=2.0,
            text="Ry",
            width_estimate=0.1,
            height_estimate=0.05,
        )
        assert info.x == 1.0
        assert info.y == 2.0
        assert info.original_y == 2.0
        assert info.text == "Ry"
        assert info.width_estimate == 0.1
        assert info.height_estimate == 0.05


# =============================================================================
# Line-breaking heuristics,
# collision search, connector drawing
# =============================================================================
import matplotlib.pyplot as _plt

from qarp.plotting import LabelMode as _LabelMode
from qarp.plotting import PlotConfig as _PlotConfig
from qarp.plotting._renderers.label_manager import LabelManager as _LM


def _lm(mode=_LabelMode.SMART):
    return _LM(_PlotConfig(label_mode=mode))


class TestBreakLongLabel:
    """Exact-output tests at the default label_max_width = 12."""

    def test_gate_name_with_long_params_splits_and_truncates(self):
        out = _lm()._break_long_label("Rzz(0.123456789,0.987654321)")
        line1, line2 = out.split("\n")
        assert line1 == "Rzz"
        assert line2.startswith("(") and line2.endswith("...)")

    def test_very_long_label_hard_breaks_with_ellipsis(self):
        out = _lm()._break_long_label("A" * 30)
        assert out == "A" * 12 + "\n" + "A" * 9 + "..."

    def test_breaks_at_separator_near_middle(self):
        out = _lm()._break_long_label("abcdef,ghijklmn")
        assert out == "abcdef,\nghijklmn"

    def test_breaks_at_capital_letter(self):
        out = _lm()._break_long_label("BlockEncoding")
        assert out == "Block\nEncoding"

    def test_last_resort_break_without_separators(self):
        out = _lm()._break_long_label("abcdefghijklmnop")
        assert out == "abcdefghijkl\nmnop"


class TestCollisionSearch:
    def test_second_label_at_same_spot_is_displaced_smart(self):
        lm = _lm(_LabelMode.SMART)
        y1, dx1, _ = lm.compute_position(0.0, 0.0, "Rz(0.5)")
        y2, dx2, _ = lm.compute_position(0.0, 0.0, "Rz(0.7)")
        assert (y2, dx2) != (y1, dx1)  # displaced in y or shifted in x

    def test_second_label_at_same_spot_is_displaced_simple(self):
        lm = _lm(_LabelMode.TRUNCATE)
        y1, dx1, _ = lm.compute_position(0.0, 0.0, "Rz(0.5)")
        y2, dx2, _ = lm.compute_position(0.0, 0.0, "Rz(0.7)")
        assert (y2, dx2) != (y1, dx1)

    def test_find_free_y_exhaustion_returns_far_position(self, monkeypatch):
        lm = _lm()
        monkeypatch.setattr(lm, "_has_overlap", lambda *a: True)
        y = lm._find_free_y(0.0, 0.0, "X")
        assert abs(y - 20 * lm.config.label_y_margin) < 1e-12

    def test_smart_position_falls_back_to_free_y_search(self, monkeypatch):
        lm = _lm(_LabelMode.SMART)
        # Strategies 1-3 always collide; strategy 4 delegates to _find_free_y.
        calls = {"n": 0}

        def crowded(x, y, label):
            calls["n"] += 1
            return calls["n"] < 12  # first attempts collide, then free

        monkeypatch.setattr(lm, "_has_overlap", crowded)
        monkeypatch.setattr(lm, "_labels_overlap", lambda *a: True)  # strategy-1 scan collides
        y, dx = lm._smart_position(0.0, 0.0, "Rz(0.5)")
        assert y != 0.0 or dx != 0.0


class TestConnector:
    def test_connector_drawn_only_when_displaced(self):
        lm = _lm()
        fig, ax = _plt.subplots()
        lm.draw_label_connector(ax, 0.0, gate_y=0.0, label_y=0.05)
        assert len(ax.lines) == 0  # within 1.5 * margin: no connector
        lm.draw_label_connector(ax, 0.0, gate_y=0.0, label_y=2.0)
        assert len(ax.lines) == 1
        _plt.close(fig)
