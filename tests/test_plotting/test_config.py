"""Tests for plotting configuration."""

from qarp.plotting import LabelMode, PlotConfig


class TestLabelMode:
    """Tests for LabelMode enum."""

    def test_label_mode_values(self):
        """Test that LabelMode has the expected values."""
        assert LabelMode.TRUNCATE.value == "truncate"
        assert LabelMode.SMART.value == "smart"
        assert LabelMode.FULL.value == "full"

    def test_label_mode_members(self):
        """Test that all expected members exist."""
        members = [m.value for m in LabelMode]
        assert "truncate" in members
        assert "smart" in members
        assert "full" in members
        assert len(members) == 3


class TestPlotConfig:
    """Tests for PlotConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PlotConfig()
        assert config.spacing == 0.25
        assert config.gate_size == 16
        assert config.control_size == 4
        assert config.swap_size == 6
        assert config.text_size == 9
        assert config.box_linewidth == 1.1
        assert config.gate_style == "outline"
        assert config.max_chars == 8
        assert config.label_decimal_places == 3
        assert config.label_mode == LabelMode.SMART
        assert config.label_max_width == 12
        assert config.label_y_margin == 0.4
        assert config.label_x_margin == 0.15

    def test_custom_label_mode(self):
        """Test setting custom label mode."""
        config = PlotConfig(label_mode=LabelMode.TRUNCATE)
        assert config.label_mode == LabelMode.TRUNCATE

        config2 = PlotConfig(label_mode=LabelMode.FULL)
        assert config2.label_mode == LabelMode.FULL

    def test_custom_label_margins(self):
        """Test setting custom label margins."""
        config = PlotConfig(
            label_y_margin=0.6,
            label_x_margin=0.2,
            label_max_width=20,
        )
        assert config.label_y_margin == 0.6
        assert config.label_x_margin == 0.2
        assert config.label_max_width == 20

    def test_gate_colors_initialized(self):
        """Test that gate colors are initialized."""
        config = PlotConfig()
        assert config.gate_colors is not None
        assert isinstance(config.gate_colors, dict)
