"""Arithmetic composition operators on ``Block``: ``*``, ``|``, ``**``/``^``, ``~``.

All four operators build on ``composite_block.CompositeBlock`` and reuse the
operand's C++ command buffer via structural sharing (``ref<Block>`` children)
rather than flattening or deep-copying — see the docstrings in
``qarp/blocks/_block.py`` next to each operator.
"""

import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import CompositeBlock, HnBlock, XnBlock


def _unitary(block):
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


# ── __mul__: sequential composition, placed by target_qubits ────────────


def test_mul_composes_sequentially():
    """``h * x`` should apply h then x, matching matrix product x @ h."""
    h = HnBlock(2).build()
    x = XnBlock(2).build()
    s = (h * x).build()
    assert isinstance(s, CompositeBlock)
    assert s.n_qubits == 2
    assert np.allclose(_unitary(s), _unitary(x) @ _unitary(h), atol=1e-10)


def test_mul_places_smaller_block_by_target_qubits():
    """A smaller operand embeds at its own ``target_qubits`` within the wider result."""
    h4 = HnBlock(4).build()
    x2 = XnBlock(2).build()
    x2.target_qubits = [1, 3]

    combined = (h4 * x2).build()
    assert combined.n_qubits == 4

    ref = HnBlock(4).build()
    for q in (1, 3):
        ref.x(q)
    assert np.allclose(_unitary(combined), _unitary(ref), atol=1e-10)


def test_mul_defaults_to_low_qubits_when_target_qubits_unset():
    """Without an explicit ``target_qubits``, the smaller operand embeds at [0, n)."""
    h4 = HnBlock(4).build()
    x2 = XnBlock(2).build()

    combined = (h4 * x2).build()
    assert combined.n_qubits == 4

    ref = HnBlock(4).build()
    ref.x(0)
    ref.x(1)
    assert np.allclose(_unitary(combined), _unitary(ref), atol=1e-10)


def test_mul_rejects_non_block_operand():
    h = HnBlock(2).build()
    with pytest.raises(TypeError):
        h * "not a block"


# ── __or__: parallel composition, disjoint qubits ───────────────────────


def test_or_places_operands_on_disjoint_qubits():
    h = HnBlock(1).build()
    x = XnBlock(1).build()
    p = (h | x).build()
    assert p.n_qubits == 2
    sv = np.array(qx.QarpSimulator().statevector(p.flatten(), p.n_qubits))
    # qubit 1 (X, shifted by the offset) is pinned to |1>; qubit 0 (H) is in
    # superposition, so only the two amplitudes with the qubit-1 bit set survive.
    expected = np.zeros(4, dtype=complex)
    expected[0b10] = 1 / np.sqrt(2)
    expected[0b11] = 1 / np.sqrt(2)
    assert np.allclose(np.abs(sv), np.abs(expected), atol=1e-10)


def test_or_does_not_mutate_operands():
    h = HnBlock(1).build()
    x = XnBlock(1).build()
    original_target = list(x.target_qubits)
    _ = h | x
    assert x.target_qubits == original_target


# ── __pow__ / __xor__: repeat ────────────────────────────────────────────


def test_pow_repeats_and_reuses_instance():
    h = HnBlock(2).build()
    r = h**3
    assert r.n_qubits == 2
    # Structural sharing: the same built instance is reused, not copied.
    assert r.blocks[0] is r.blocks[1] is r.blocks[2] is h


def test_xor_matches_pow():
    h = HnBlock(2).build()
    assert (h**3).n_qubits == (h ^ 3).n_qubits
    r = h ^ 3
    assert r.blocks[0] is r.blocks[1] is r.blocks[2] is h


def test_pow_by_one_returns_self():
    h = HnBlock(2).build()
    assert (h**1) is h


def test_pow_rejects_non_positive_count():
    h = HnBlock(2).build()
    with pytest.raises(ValueError, match="positive"):
        h**0


# ── __invert__: dagger shorthand ─────────────────────────────────────────


def test_invert_matches_dagger():
    h = HnBlock(2).build()
    assert (~h)._is_dagger == h.dagger()._is_dagger
    assert (~h)._is_dagger is True


# ── chaining: normal operator precedence/associativity applies ──────────


def test_chained_operators_preserve_order():
    """``h * x * (i ** 3)`` should apply h, x, then i three times, in order."""
    h = HnBlock(2).build()
    x = XnBlock(2).build()
    from qarp.blocks import IdentityBlock

    i = IdentityBlock(2).build()

    chained = (h * x * i**3).build()
    assert chained.n_qubits == 2

    manual = _unitary(i) @ _unitary(i) @ _unitary(i) @ _unitary(x) @ _unitary(h)
    assert np.allclose(_unitary(chained), manual, atol=1e-10)
