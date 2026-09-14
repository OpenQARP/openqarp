"""plot() API surface and CircuitPlotter drill-in / output: color-scheme
resolution, raw-Block acceptance, inner_index navigation, gate registry
queries, measurement drawing, and the scrollable/interactive HTML output
(display monkeypatched — no notebook).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import qarp.plotting._circuit_plotter as plotter_mod
from qarp.blocks import CompositeBlock, SimpleBlock
from qarp.plotting import CircuitPlotter, PlotConfig, plot
from qarp.plotting._plot import _apply_color_scheme
from qarp.plotting.styles import DEFAULT_COLORS


def _boxed_composite():
    a = SimpleBlock(2, name="Prep")
    a.h(0)
    b = SimpleBlock(2, name="Entangle")
    b.cx(0, 1)
    comp = CompositeBlock([a, b], 2)
    comp.build()
    return comp


class TestApplyColorScheme:
    def test_legacy_named_scheme_switches_to_filled(self):
        config = PlotConfig()
        _apply_color_scheme(config, "classic")
        assert config.gate_style == "filled"
        assert config.default_gate_color == "skyblue"

    def test_custom_dict_is_used_verbatim_and_filled(self):
        config = PlotConfig()
        palette = {"h": "#123456"}
        _apply_color_scheme(config, palette)
        assert config.gate_colors == palette
        assert config.gate_style == "filled"

    def test_non_string_non_dict_falls_back_to_defaults(self):
        config = PlotConfig()
        _apply_color_scheme(config, None)
        assert config.gate_colors == DEFAULT_COLORS


class TestPlotEntryPoint:
    def test_accepts_raw_block(self):
        block = SimpleBlock(1)
        block.h(0)
        block.build()
        plotter = plot(block, return_plotter=True)
        assert isinstance(plotter, CircuitPlotter)
        plt.close("all")

    def test_inner_index_drills_into_boxed_child(self):
        # The adapter inlines the 1q "Prep" child; the boxed "Entangle" child
        # sits at gate index 1 and drills to its single-CX inner circuit.
        plotter = plot(_boxed_composite(), inner_index=1, return_plotter=True)
        assert isinstance(plotter, CircuitPlotter)
        assert len(plotter._gate_registry) == 1
        plt.close("all")

    def test_inner_index_not_found_reports_and_returns_none(self, capsys):
        result = plot(_boxed_composite(), inner_index=99)
        assert result is None
        assert "Could not find inner circuit at index 99" in capsys.readouterr().out
        plt.close("all")


class TestGateRegistryQueries:
    def _plotter(self):
        return plot(_boxed_composite(), return_plotter=True)

    def test_get_inner_circuit_unknown_index_reports(self, capsys):
        plotter = self._plotter()
        assert plotter.get_inner_circuit(99) is None
        assert "No gate found at index 99" in capsys.readouterr().out
        plt.close("all")

    def test_get_inner_circuit_non_box_gate_reports(self, capsys):
        block = SimpleBlock(1)
        block.h(0)
        block.build()
        plotter = plot(block, return_plotter=True)
        assert plotter.get_inner_circuit(0) is None
        assert "not a box type" in capsys.readouterr().out
        plt.close("all")

    def test_list_gates_empty_and_populated(self, capsys):
        empty = CircuitPlotter(PlotConfig())
        empty.list_gates()
        assert "No gates recorded" in capsys.readouterr().out

        plotter = self._plotter()
        plotter.list_gates()
        out = capsys.readouterr().out
        assert "Total gates: 2" in out
        plt.close("all")

    def test_plot_inner_valid_and_invalid(self, capsys):
        plotter = self._plotter()
        inner = plotter.plot_inner(1)
        assert isinstance(inner, CircuitPlotter)
        assert plotter.plot_inner(99) is None
        plt.close("all")


class TestMeasurementDrawing:
    def _measured_block(self):
        block = SimpleBlock(1)
        block.h(0)
        block.measure(0, 0)
        block.build()
        return block

    def test_outline_and_filled_measure_styles(self):
        for scheme in ("default", "classic"):
            plotter = plot(self._measured_block(), color_scheme=scheme, return_plotter=True)
            assert len(plotter._measure_payload) == 1
            assert plotter._measure_payload[0]["key"] == "m0"
            plt.close("all")


class TestOutputs:
    def test_verbose_plot_prints_circuit_info(self, capsys):
        block = SimpleBlock(1)
        block.h(0)
        block.build()
        plot(block, verbose=True)
        out = capsys.readouterr().out
        assert "Circuit information:" in out
        assert "Qubits: 1" in out
        plt.close("all")

    def test_scrollable_output_plain_and_interactive(self, monkeypatch):
        captured = []
        monkeypatch.setattr(plotter_mod, "display", lambda obj: captured.append(obj))
        monkeypatch.setattr(plotter_mod, "HTML", lambda html: html)

        plotter = plot(_boxed_composite(), return_plotter=True)
        fig = plt.figure()
        plotter._create_scrollable_output(fig, interactive=False)
        assert "<svg" in captured[-1]

        fig = plt.figure()
        plotter._create_scrollable_output(fig, interactive=True)
        assert "qarp-plot-" in captured[-1]  # interactive container id
        plt.close("all")
