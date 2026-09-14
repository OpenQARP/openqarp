"""Tests for the command-list Reconstructer."""

import pytest

import qarpx as qx
from qarp.cutting import Reconstructer
from qarp.cutting._internals import _cmd, _CutMarker, _measure_cmd


def _build_commands(n_qubits: int, n_layers: int) -> tuple[list, int]:
    """Build a layered all-to-all CX circuit as a command list."""
    cmds = [_cmd(qx.GateType.H, i) for i in range(n_qubits)]
    for _ in range(n_layers):
        for i in range(n_qubits - 1):
            for j in range(i + 1, n_qubits):
                cmds.append(qx.Command(qx.GateType.CX, i, j))
                cmds.append(qx.Command(qx.GateType.CX, j, i))
    return cmds, n_qubits


def _count_1q(cmds):
    return sum(1 for c in cmds if isinstance(c, qx.Command) and len(c.qubits) == 1)


def _count_2q(cmds):
    return sum(1 for c in cmds if isinstance(c, qx.Command) and len(c.qubits) == 2)


def _count_gates(cmds):
    return sum(1 for c in cmds if isinstance(c, qx.Command))


def test_empty_circuit_raises():
    with pytest.raises(ValueError):
        Reconstructer([], 0)
    with pytest.raises(ValueError):
        Reconstructer([_cmd(qx.GateType.Barrier, 0)], 1)


def test_reconstruct_delete_gates():
    for n_qubits in [3, 4, 5, 6]:
        for n_layers in [2, 4, 6]:
            cmds, n = _build_commands(n_qubits, n_layers)
            rec = Reconstructer(cmds, n)

            to_delete = [[0, 1]]
            new_cmds, n_cuts = rec.reconstruct_delete_gates(to_delete)
            assert _count_2q(cmds) == _count_2q(new_cmds) + n_layers * 2
            assert _count_1q(cmds) == _count_1q(new_cmds)
            assert n_cuts == n_layers * 2

            to_delete = [[1, 2]]
            new_cmds, n_cuts = rec.reconstruct_delete_gates(to_delete)
            assert _count_2q(cmds) == _count_2q(new_cmds) + n_layers * 2
            assert n_cuts == n_layers * 2

            to_delete = [[0, 1], [1, 2], [0, 2]]
            new_cmds, n_cuts = rec.reconstruct_delete_gates(to_delete)
            assert _count_2q(cmds) - len(to_delete) * 2 * n_layers == _count_2q(new_cmds)


def test_reconstruct_swap_gates_by_customs():
    for n_qubits in [3, 4, 5]:
        for n_layers in [2, 4]:
            cmds, n = _build_commands(n_qubits, n_layers)
            rec = Reconstructer(cmds, n)

            to_delete = [[0, 1]]
            cut1q, cut2q, n_cuts, cut_names = rec.reconstruct_swap_gates_by_customs(to_delete)
            # Each cut gate → one 2q _CutMarker in cut2q and two 1q markers in cut1q
            cut_markers_2q = [c for c in cut2q if isinstance(c, _CutMarker)]
            assert len(cut_markers_2q) == n_cuts
            assert n_cuts == len(cut_names)
            # Original gate count preserved in plot representation
            assert _count_gates(cmds) == _count_gates(cut2q) + n_cuts  # 2q cut → 1 marker

            to_delete = [[1, 2]]
            _, cut2q, n_cuts, cut_names = rec.reconstruct_swap_gates_by_customs(to_delete)
            assert n_cuts == len(cut_names)


def test_reconstruct_filter_qubits():
    for n_qubits in [3, 4, 5]:
        for n_layers in [2, 4]:
            cmds, n = _build_commands(n_qubits, n_layers)
            rec = Reconstructer(cmds, n)

            # Filtering to a single connected qubit — must cut first
            to_filter = [0]
            with pytest.raises(RuntimeError):
                Reconstructer.reconstruct_filter_qubits(cmds, n, to_filter)

            # After cutting, filter works
            new_cmds, _ = rec.reconstruct_delete_gates(
                [[to_filter[0], i] for i in range(n_qubits) if i != to_filter[0]]
            )
            filtered, local_n = Reconstructer.reconstruct_filter_qubits(new_cmds, n, to_filter)
            assert local_n == len(to_filter)

            # Full circuit filter = no change
            to_filter_all = list(range(n_qubits))
            filtered_all, local_n_all = Reconstructer.reconstruct_filter_qubits(
                cmds, n, to_filter_all
            )
            assert local_n_all == n_qubits
            assert _count_gates(filtered_all) == _count_gates(cmds)


def test_remove_measure_gates():
    for n_q in range(2, 5):
        for n_lay in range(1, 3):
            cmds, n = _build_commands(n_q, n_lay)
            meas_cmds = list(cmds) + [_measure_cmd(q, q) for q in range(n_q)]
            new_cmds, measured = Reconstructer.remove_measure_gates(meas_cmds)
            assert _count_gates(new_cmds) == _count_gates(meas_cmds) - n_q
            assert len(measured) == n_q
