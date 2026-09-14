"""``benchmarks/statevector/tune.py`` is the subject under test here (§15's
one-direction exception); its stack adapters import lazily, so nothing from
the SDK extras is touched."""

import math

import pytest

from benchmarks.statevector import tune


def _grid(cells: dict[str, list[float]]) -> dict:
    """label -> one value per (family, n) cell, in FAMILIES x SIZES order."""
    grid = {}
    for label, values in cells.items():
        keys = [(f, n) for f in tune.FAMILIES for n in tune.SIZES]
        assert len(values) == len(keys)
        for (f, n), v in zip(keys, values, strict=True):
            grid[(f, n, label)] = v
    return grid


def test_score_is_the_geomean_over_the_selection_sizes():
    # n = 12 cells (index 0, 3, 6) are excluded by SELECT_FROM; the rest are
    # 1, 4 -> geomean 2 for every family.
    grid = _grid({"a": [99.0, 1.0, 4.0] * 3})
    assert tune._score("a", grid) == pytest.approx(2.0)


def test_non_positive_cell_is_dropped_not_logged(capsys):
    """A run phase measured below its subtracted overhead used to raise
    ``ValueError: math domain error`` and abort the whole sweep (review
    2026-09-14)."""
    grid = _grid({"a": [5.0, 2.0, 8.0] * 3, "b": [5.0, -0.3, 0.0, 5.0, 3.0, 3.0, 5.0, 3.0, 3.0]})
    assert tune._score("a", grid) == pytest.approx(4.0)
    assert tune._score("b", grid) == pytest.approx(3.0)
    assert "b: 2 cell(s) <= 0 ms" in capsys.readouterr().out
    assert tune._select(["a", "b"], grid) == "b"


def test_candidate_with_no_usable_cell_never_wins():
    grid = _grid({"a": [1.0, 10.0, 10.0] * 3, "b": [1.0, 0.0, -1.0] * 3})
    assert math.isinf(tune._score("b", grid))
    assert tune._select(["a", "b"], grid) == "a"
