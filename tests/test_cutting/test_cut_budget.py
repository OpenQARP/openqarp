"""The cut-budget guard: every cut multiplies the experiment count by 6, so
``check_cut_budget`` is the one gate standing between a plausible-looking cut
count and 6^n circuits.  Enforced on both cutting paths (``CuttingPrimitive``
and ``QPDDecomposition``) and, until now, pinned by nothing.
"""

import pytest

from qarp import config
from qarp.cutting import check_cut_budget


@pytest.fixture
def restore_budget():
    original = config.max_number_of_cuts
    yield
    config.max_number_of_cuts = original


def test_cut_counts_below_the_budget_are_accepted(restore_budget):
    config.max_number_of_cuts = 3
    for n_cuts in (0, 1, 2):
        check_cut_budget(n_cuts)  # must not raise


def test_the_budget_itself_is_already_rejected(restore_budget):
    """The comparison is ``>=``: ``max_number_of_cuts`` cuts is one too many,
    not the last allowed value."""
    config.max_number_of_cuts = 3
    with pytest.raises(RuntimeError):
        check_cut_budget(3)


def test_the_error_names_the_count_the_budget_and_the_override(restore_budget):
    config.max_number_of_cuts = 2
    with pytest.raises(RuntimeError) as excinfo:
        check_cut_budget(5)
    message = str(excinfo.value)
    assert "5" in message
    assert "2" in message
    assert "force_max_number_cuts=True" in message


def test_force_overrides_the_budget(restore_budget):
    config.max_number_of_cuts = 1
    check_cut_budget(99, force=True)  # must not raise


def test_default_budget_rejects_six_cuts():
    """6 cuts is 6^6 = 46656 experiments; the shipped default refuses it."""
    assert config.max_number_of_cuts == 6
    check_cut_budget(5)
    with pytest.raises(RuntimeError):
        check_cut_budget(6)
