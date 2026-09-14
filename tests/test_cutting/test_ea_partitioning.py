"""Tests for EAPartitioning."""

import pytest

import qarpx as qx
from qarp.cutting import EAPartitioning
from qarp.cutting._internals import _cmd, _CutMarker


def _toy_circuit_cmds(n_qubits: int, cx_pairs: list) -> tuple[list, int]:
    cmds = [_cmd(qx.GateType.H, i) for i in range(n_qubits)]
    for c, t in cx_pairs:
        cmds.append(qx.Command(qx.GateType.CX, c, t))
    return cmds, n_qubits


def test_empty_circuit_raises():
    """EAPartitioning requires at least one entangling gate."""
    cmds = [_cmd(qx.GateType.H, 0), _cmd(qx.GateType.H, 1)]
    with pytest.raises(AssertionError):
        EAPartitioning(cmds, 2, [2, 2])


def test_wrong_max_sizes_raises():
    cmds, n = _toy_circuit_cmds(4, [(0, 1), (1, 2), (2, 3)])
    with pytest.raises(AssertionError):
        EAPartitioning(cmds, n, [2, 1])  # sum < n_qubits


def test_manual_cut_n_cuts():
    cmds, n = _toy_circuit_cmds(3, [(0, 1), (1, 2), (0, 1), (1, 2)])
    cutter = EAPartitioning(cmds, n, [2, 2])
    result = cutter.cut(manual_setting=[[0, 1], [2]])
    # The (1,2) connection is the cut — 2 CX gates on (1,2)
    assert result.n_cuts == 2


def test_manual_cut_subcircuit_count():
    cmds, n = _toy_circuit_cmds(3, [(0, 1), (1, 2)])
    cutter = EAPartitioning(cmds, n, [2, 2])
    result = cutter.cut(manual_setting=[[0], [1, 2]])
    assert result.n_subcircuits == 2


def test_manual_cut_subcircuit_qubit_indices():
    cmds, n = _toy_circuit_cmds(3, [(0, 1), (1, 2)])
    cutter = EAPartitioning(cmds, n, [2, 2])
    result = cutter.cut(manual_setting=[[0], [1, 2]])
    assert result.subcircuit_qubits[0] == [0]
    assert sorted(result.subcircuit_qubits[1]) == [1, 2]


def test_cutter_result_str():
    cmds, n = _toy_circuit_cmds(3, [(0, 1), (1, 2)])
    cutter = EAPartitioning(cmds, n, [2, 2])
    result = cutter.cut(manual_setting=[[0], [1, 2]])
    s = str(result)
    assert "subcircuits" in s.lower() or "cuts" in s.lower()


def test_cut_markers_present_in_custom_commands():
    cmds, n = _toy_circuit_cmds(3, [(0, 1), (1, 2)])
    cutter = EAPartitioning(cmds, n, [2, 2])
    result = cutter.cut(manual_setting=[[0], [1, 2]])
    markers = [c for c in result.custom_commands if isinstance(c, _CutMarker)]
    assert len(markers) == result.n_cuts


def test_manual_setting_invalid_qubits():
    cmds, n = _toy_circuit_cmds(3, [(0, 1), (1, 2)])
    cutter = EAPartitioning(cmds, n, [2, 2])
    with pytest.raises(AssertionError):
        cutter.cut(manual_setting=[[0, 1], [1, 3]])  # qubit 3 out of range


def test_subcircuits_are_filtered_correctly():
    """After cutting, each subcircuit's commands only reference local qubit indices."""
    cmds, n = _toy_circuit_cmds(4, [(0, 1), (2, 3), (1, 2)])
    cutter = EAPartitioning(cmds, n, [2, 2])
    result = cutter.cut(manual_setting=[[0, 1], [2, 3]])
    for sub_cmds, local_n in result.subcircuits:
        for cmd in sub_cmds:
            if isinstance(cmd, qx.Command):
                assert all(q < local_n for q in cmd.qubits), (
                    f"qubit out of local range in subcircuit (local_n={local_n})"
                )
