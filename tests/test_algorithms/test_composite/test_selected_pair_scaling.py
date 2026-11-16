"""Independent small oracles and resource guards for the bounded research harness."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validate_selected_pair_scaling import (
    assess_roots,
    determinant_address,
    independent_jw,
    independent_preparation,
    request_panel,
    spin_counts,
    supervised,
)


@pytest.mark.parametrize(
    "label,address", [(3, (0, 0, 1)), (6, (1, 0, -1)), (9, (0, 1, 1)), (12, (1, 1, 1))]
)
def test_signed_determinant_address_literal(label, address):
    pytest.importorskip("pyscf")
    assert determinant_address(label, 2, (1, 1)) == address
    with pytest.raises(ValueError, match="sector"):
        determinant_address(1, 2, (1, 1))


def test_literal_exponential_sign_phase_and_order():
    theta = 0.3
    forward = independent_preparation(3, [(0, 1, theta)])
    reverse = independent_preparation(12, [(0, 1, theta)])
    assert forward == pytest.approx({3: np.cos(theta), 12: np.sin(theta)}, abs=1e-14)
    assert reverse == pytest.approx({3: -np.sin(theta), 12: np.cos(theta)}, abs=1e-14)
    ordered = independent_preparation(3, [(0, 1, np.pi / 2), (1, 2, np.pi / 2)])
    assert ordered[48] == pytest.approx(1)
    assert (
        abs(independent_preparation(3, [(1, 2, np.pi / 2), (0, 1, np.pi / 2)]).get(48, 0)) < 1e-14
    )


def test_independent_integral_jw_against_literal_number_operator():
    pytest.importorskip("openfermion")
    from tests.selected_pair_oracles import pauli_matrix

    h1 = np.diag([-1.0, 0.5])
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 0, 0, 0], eri[1, 1, 1, 1] = 0.4, 0.6
    terms, branches = independent_jw(h1, eri, 0.7)
    diagonal = []
    for label in range(16):
        a, b, c, d = [(label >> bit) & 1 for bit in range(4)]
        diagonal.append(0.7 - a - b + 0.5 * (c + d) + 0.4 * a * b + 0.6 * c * d)
    np.testing.assert_allclose(pauli_matrix(terms, 4), np.diag(diagonal), atol=1e-14)
    assert branches == 264


def test_root_validation_rejects_bad_or_incomplete_ground_space():
    matrix = np.diag([-2.0, -2.0, 1.0])
    action = lambda v: matrix @ v
    report, ground = assess_roots([-2, -2, 1], list(np.eye(3)), action, True)
    assert report["numerically_accepted"] and report["ground_manifold_complete"]
    assert len(ground) == 2
    report, _ = assess_roots([-2, -2], list(np.eye(3))[:2], action, True)
    assert not report["ground_manifold_complete"]
    report, _ = assess_roots([-2, 1], [np.array([0.9, 0, 0]), np.array([0, 0, 1])], action, True)
    assert not report["numerically_accepted"]
    report, _ = assess_roots([-2, 1], [np.array([1, 0, 0]), np.array([0, 1, 0])], action, True)
    assert not report["numerically_accepted"]


def test_structural_panel_is_matched_nested_charge_valid_and_not_value_ranked():
    terms = {((i, "X"), (j, "X")): 1.0 for i in range(8) for j in range(i + 1, 8) if i % 2 == j % 2}
    fixture = {"n_qubits": 8, "reference": 15, "electron_counts": [2, 2]}
    arms = {"bare": [], "one_pair": [(0, 3, 0.2)], "two_pairs": [(0, 3, 0.2), (1, 2, -0.1)]}
    keys = request_panel(fixture, terms, arms)
    assert len(set(keys)) == 128
    assert keys[0] == (15, 15)
    assert all(a == b for a, b in keys[:8])
    assert all(spin_counts(label, 4) == (2, 2) for key in keys for label in key)
    assert keys == request_panel(fixture, {k: -3.7 * v for k, v in terms.items()}, arms)


def test_supervisor_terminates_at_wall_budget(tmp_path):
    report = supervised(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path / "sleep.log",
        0.15,
        2**30,
        dict(os.environ),
    )
    assert report["termination_reason"] == "wall_budget_exceeded"
    assert report["returncode"] != 0
    assert report["wall_seconds"] < 3


def test_planning_counts_literal_qwc_without_execution(monkeypatch, tmp_path):
    import scripts.validate_selected_pair_scaling as harness
    from qarp.engines import QarpEngine

    fixture = {"n_qubits": 8, "fixture_sha256": "synthetic"}
    monkeypatch.setattr(
        harness,
        "frozen",
        lambda *_: (fixture, {((0, "Z"),): 1.0}, {"bare": [], "one_pair": [(0, 1, 0.3)]}),
    )
    monkeypatch.setattr(harness, "request_panel", lambda *_: [(i, i) for i in range(8)])
    monkeypatch.setattr(QarpEngine, "run", lambda *_: pytest.fail("Planning executed a circuit"))
    report = harness.planning_check(tmp_path, "synthetic", tmp_path)
    assert report["results"]["bare"]["logical_QWC_settings"] == 0
    assert report["results"]["one_pair"]["logical_QWC_settings"] == 8
    assert report["shots"] == report["submissions"] == report["executed_settings"] == 0


def test_supervisor_terminates_at_rss_budget(tmp_path):
    report = supervised(
        [sys.executable, "-c", "import time; a=bytearray(16*1024**2); time.sleep(30)"],
        tmp_path / "memory.log",
        3,
        1024,
        dict(os.environ),
    )
    assert report["termination_reason"] == "RSS_budget_exceeded"
    assert report["returncode"] != 0


def test_exhausted_stage_does_not_start_later_stage(monkeypatch, tmp_path):
    import scripts.validate_selected_pair_scaling as harness

    clock = iter([0, 2701])
    monkeypatch.setattr(harness.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        harness, "supervised", lambda *_: pytest.fail("Exhausted stage started work")
    )
    args = SimpleNamespace(output_dir=tmp_path / "results", package_root=tmp_path)
    assert harness.supervise(args) == 1
    assert '"status": "budget_exhausted"' in (args.output_dir / "manifest.json").read_text()
