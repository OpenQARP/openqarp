"""Tests for ReadoutBlock (Python-layer collection wrapper)."""

import pytest

import qarpx as qx
from qarp.blocks import ReadoutBlock


def _measure_pairs(block):
    """Return [(qubit, cbit), ...] for every Measure command in `block`'s flat command stream."""
    pairs = []
    for cmd in block.flatten():
        if cmd.gate == qx.GateType.Measure:
            pairs.append((cmd.qubits[0], cmd.cbits[0]))
    return pairs


def test_default_reads_out_full_register():
    block = ReadoutBlock(3).build()
    assert block.n_qubits == 3
    assert _measure_pairs(block) == [(0, 0), (1, 1), (2, 2)]


def test_explicit_qubits_subset():
    block = ReadoutBlock(5, qubits=[0, 2, 4]).build()
    # cbits default to list(qubits)
    assert _measure_pairs(block) == [(0, 0), (2, 2), (4, 4)]


def test_explicit_qubits_and_cbits_independent():
    block = ReadoutBlock(4, qubits=[1, 3], cbits=[0, 1]).build()
    assert _measure_pairs(block) == [(1, 0), (3, 1)]


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        ReadoutBlock(3, qubits=[0, 1], cbits=[0])


def test_qubit_out_of_range_raises():
    with pytest.raises(ValueError, match="out of range|lie in"):
        ReadoutBlock(3, qubits=[0, 5])


def test_n_qubits_must_be_positive():
    with pytest.raises(ValueError, match="positive integer"):
        ReadoutBlock(0)


def test_composes_in_compositeblock_without_explicit_n_qubits():
    """Composing one ReadoutBlock(4) child should not trip n_qubits inference."""
    from qarp.blocks import CompositeBlock, HnBlock

    block = CompositeBlock([HnBlock(4), ReadoutBlock(4)]).build()
    assert block.n_qubits == 4
    pairs = _measure_pairs(block)
    assert pairs == [(0, 0), (1, 1), (2, 2), (3, 3)]
