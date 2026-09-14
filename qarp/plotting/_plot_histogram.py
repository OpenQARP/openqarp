import itertools
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt

from .styles import _theme as theme
from .styles._theme import qarp_rc


def plot_histogram(
    probs: Dict[Tuple[int, ...], float],
    title: str = "",
    figsize: Tuple[int, int] = (10, 5),
    show_all_solutions: bool = False,
    return_plotter: bool = False,
    show_values: bool = True,
    sort_by_prob: bool = False,
    top_k: Optional[int] = None,
    highlight_max: bool = False,
):
    """
    Plots a histogram of the probabilities of different solutions.

    Args:
        probs: A dictionary mapping tuples of binary values (representing solutions) to their probabilities.
        title: The title of the histogram.
        figsize: The size of the figure to create.
        show_all_solutions: If True, includes all possible solutions in the histogram, even those with zero probability.
        return_plotter: If True, returns the figure and axis objects for further manipulation.
        show_values: If True, displays probability values above each bar.
        sort_by_prob: If True, sorts bars by probability in descending order.
        top_k: If set, only show the top k solutions by probability. Implies sort_by_prob=True.
        highlight_max: If True, highlights the most probable solution(s) in a different color.
    """
    if not probs:
        raise ValueError("The 'probs' dictionary is empty.")

    n_nodes = len(next(iter(probs.keys())))

    # Generate all keys if requested
    keys = (
        list(itertools.product([0, 1], repeat=n_nodes))
        if show_all_solutions
        else list(probs.keys())
    )
    complete_data = {k: probs.get(k, 0.0) for k in keys}

    if top_k is not None or sort_by_prob:
        sorted_items = sorted(complete_data.items(), key=lambda x: x[1], reverse=True)
    else:
        sorted_items = list(complete_data.items())

    if top_k is not None:
        sorted_items = sorted_items[:top_k]

    sorted_keys, sorted_values = zip(*sorted_items, strict=True)

    # Bitstring labels (e.g. "0011") instead of tuple notation
    key_labels = ["".join(str(b) for b in k) for k in sorted_keys]

    # Amber = measurement results; cobalt highlights the most probable solution(s).
    bar_color = theme.AMBER
    highlight_color = theme.COBALT
    if highlight_max:
        max_val = max(sorted_values)
        colors = [highlight_color if v == max_val and v > 0 else bar_color for v in sorted_values]
    else:
        colors = [bar_color] * len(sorted_values)

    # Plot — theme rc params scoped to this figure only.
    with plt.rc_context(qarp_rc()):
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(
            key_labels,
            sorted_values,
            color=colors,
            width=0.62,
            linewidth=0,
        )

        # Axis labels and title
        ax.set_xlabel("Solutions")
        ax.set_ylabel("Probability")
        if title:
            ax.set_title(label=title)

        # Improve readability: hairline y-grid, baseline-only spines.
        plt.xticks(rotation=45, ha="right")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="both", length=0)

        # Show numeric values above bars, skipping near-zero values
        if show_values:
            for bar, val in zip(bars, sorted_values, strict=True):
                if val > 1e-3:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color=theme.INK,
                    )

        plt.tight_layout()

    if return_plotter:
        return fig, ax
    return None
