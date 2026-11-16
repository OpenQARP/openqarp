"""Opt-in frozen-energy comparisons, not independent molecular/FCI validation."""

import os
from pathlib import Path

import pytest

from scripts.validate_selected_pair_preparations import reproduce


@pytest.mark.parametrize(
    "case,gain",
    [
        ("h2o_one_pair", 12.60087336822835),
        ("n2_one_pair", 36.48998932641234),
        ("n2_two_pairs", 72.94544197807795),
        ("lih_one_pair", 14.456498739770396),
    ],
)
def test_frozen_reproduction_script_energy(case, gain):
    # The script is the subject under test; no fixtures are imported from examples.
    root = os.environ.get("QARP_SELECTED_PAIR_PACKAGE")
    if root is None:
        pytest.skip("External selected-pair handoff not configured")
    result = reproduce(Path(root), case)
    assert result["energy_lowering_mHa"] == pytest.approx(gain, abs=2e-6)
    assert result["fidelity"] is None
    assert result["settings"] == result["submissions"] == result["shots"] == 0
    assert not result["ground_state_used_for_selection"]
