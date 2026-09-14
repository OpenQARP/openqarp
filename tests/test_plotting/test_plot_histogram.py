"""plot_histogram structural tests: bar heights/labels mirror the input
distribution, ordering and filtering options, highlight colors, value texts.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from qarp.plotting import plot_histogram
from qarp.plotting.styles import _theme as theme

_PROBS = {(0, 0): 0.1, (1, 0): 0.6, (0, 1): 0.3}


def _heights_and_labels(ax):
    heights = [p.get_height() for p in ax.patches]
    labels = [t.get_text() for t in ax.get_xticklabels()]
    return heights, labels


def test_bars_mirror_distribution():
    fig, ax = plot_histogram(_PROBS, return_plotter=True)
    heights, labels = _heights_and_labels(ax)
    assert heights == [0.1, 0.6, 0.3]
    assert labels == ["00", "10", "01"]
    plt.close(fig)


def test_empty_probs_raises():
    with pytest.raises(ValueError, match="empty"):
        plot_histogram({})


def test_show_all_solutions_pads_missing_keys():
    fig, ax = plot_histogram({(0, 0): 1.0}, show_all_solutions=True, return_plotter=True)
    heights, labels = _heights_and_labels(ax)
    assert len(heights) == 4
    assert heights[labels.index("00")] == 1.0
    assert sum(heights) == 1.0
    plt.close(fig)


def test_sort_by_prob_descending():
    fig, ax = plot_histogram(_PROBS, sort_by_prob=True, return_plotter=True)
    heights, labels = _heights_and_labels(ax)
    assert heights == sorted(heights, reverse=True)
    assert labels[0] == "10"
    plt.close(fig)


def test_top_k_implies_sorting_and_limits():
    fig, ax = plot_histogram(_PROBS, top_k=2, return_plotter=True)
    heights, _ = _heights_and_labels(ax)
    assert heights == [0.6, 0.3]
    plt.close(fig)


def test_highlight_max_colors_only_the_peak():
    fig, ax = plot_histogram(_PROBS, highlight_max=True, return_plotter=True)
    colors = [p.get_facecolor() for p in ax.patches]
    cobalt = matplotlib.colors.to_rgba(theme.COBALT)
    amber = matplotlib.colors.to_rgba(theme.AMBER)
    assert colors[1] == cobalt  # the 0.6 peak
    assert colors[0] == amber and colors[2] == amber
    plt.close(fig)


def test_value_texts_skip_near_zero():
    probs = {(0,): 0.9995, (1,): 0.0005}
    fig, ax = plot_histogram(probs, return_plotter=True)
    values = [t.get_text() for t in ax.texts]
    assert "1.000" in values or "0.999" in values  # 0.9995 formatted
    assert len(values) == 1  # 0.0005 below the 1e-3 display floor
    plt.close(fig)


def test_no_value_texts_when_disabled_and_default_returns_none():
    fig, ax = plot_histogram(_PROBS, show_values=False, title="t", return_plotter=True)
    assert len(ax.texts) == 0
    # The OpenQARP theme places titles left-aligned.
    assert ax.get_title(loc="left") == "t"
    plt.close(fig)
    assert plot_histogram(_PROBS) is None
    plt.close("all")


def test_heights_sum_matches_distribution_mass():
    fig, ax = plot_histogram(_PROBS, return_plotter=True)
    heights, _ = _heights_and_labels(ax)
    assert np.isclose(sum(heights), sum(_PROBS.values()))
    plt.close(fig)
