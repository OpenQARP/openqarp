from math import comb

import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import DickeStateBlock


def _run_dicke(n, k, n_shots=100000):
    block = DickeStateBlock(n, k)
    block.build()
    sim = qx.QarpSimulator()
    result = sim.run(block.commands(), n, n_shots, seed=42)
    return result.counts


def test_dicke_4_qubits():
    counts = _run_dicke(4, 2)
    # Every outcome must have Hamming weight 2
    for outcome, _ in counts.items():
        assert bin(outcome).count("1") == 2, f"Unexpected Hamming weight in outcome {outcome:04b}"
    # Should have exactly C(4,2)=6 distinct outcomes
    assert len(counts) == 6
    # Each outcome should appear with roughly equal probability
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    assert np.allclose(probs, [1 / 6] * 6, atol=2e-2)


def test_dicke_3_1():
    counts = _run_dicke(3, 1)
    for outcome, _ in counts.items():
        assert bin(outcome).count("1") == 1
    assert len(counts) == comb(3, 1)


def test_dicke_4_4():
    """All qubits = 1: unique state |1111>."""
    block = DickeStateBlock(4, 4)
    block.build()
    sim = qx.QarpSimulator()
    sv = sim.statevector(block.commands(), 4)
    # |1111> is index 15 (little-endian)
    assert abs(sv[15]) > 0.99


@pytest.mark.parametrize("n", [2, 3, 4])
def test_dicke_k0_is_the_all_zero_state(n):
    """``hamming_weight=0`` is exactly |0...0>: the old ``qubits[-0:]`` slice put
    an X on every qubit (10.1)."""
    block = DickeStateBlock(n, 0)
    block.build()
    sv = np.asarray(qx.QarpSimulator().statevector(block.flatten(), n))
    expected = np.zeros(2**n, dtype=complex)
    expected[0] = 1.0
    np.testing.assert_allclose(sv, expected, atol=1e-12)
