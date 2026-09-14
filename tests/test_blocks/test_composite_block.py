"""Smoke tests for the user-facing CompositeBlock.

Coverage of the bare-base ``qarp.blocks.CompositeBlockBase`` (Pattern B
subclasses with ``add_child``) is in ``test_block_refactor.py``.  This file
focuses on the user-facing wrapper that takes a ``blocks=`` list and
auto-handles the ControlledBlock-wrapping orchestration.
"""

import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import CompositeBlock, ControlledBlock, HnBlock, IdentityBlock, SimpleBlock


def _unitary(block):
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


def test_composite_chains_two_blocks():
    """``CompositeBlock([H_n, I_n])`` should produce the same unitary as ``H_n`` alone."""
    n = 2
    comp = CompositeBlock([HnBlock(n).build(), IdentityBlock(n).build()]).build()
    U_comp = _unitary(comp)
    U_hn = _unitary(HnBlock(n).build())
    assert np.linalg.norm(U_comp - U_hn) < 1e-12


def test_composite_n_qubits_inferred_from_children():
    """``n_qubits`` is inferred from the children's footprint when not explicit."""
    comp = CompositeBlock([HnBlock(3).build()]).build()
    assert comp.n_qubits == 3


def test_composite_rejects_non_block_element():
    with pytest.raises(TypeError, match="Block instance"):
        CompositeBlock([HnBlock(2).build(), "not a block"]).build()


# ── Control wiring ────────────────────────────────────────────────────────
# A child's ``n_controls`` means "wrap me"; a ControlledBlock's records the
# controls it has already applied.  Conflating the two double-controls it.


def _ry(theta):
    """Analytic Ry(theta) = exp(-i·theta·Y/2) (conventions §1)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _controlled(u, n_target):
    """Analytic control-on-qubit-0 embedding of ``u`` (targets on qubits 1..n).

    LSB (§1): index ``i = b0 + 2·i_target``, so the control-1 subspace is the
    odd indices.  Built independently of qarp's own ControlledBlock.
    """
    dim = 2 ** (n_target + 1)
    m = np.eye(dim, dtype=complex)
    for row in range(2**n_target):
        for col in range(2**n_target):
            m[2 * row + 1, 2 * col + 1] = u[row, col]
    return m


def _ry_block(theta):
    block = SimpleBlock(1, name="inner")
    block.ry(0, theta)
    return block.build()


def test_controlled_child_is_not_re_controlled():
    """A ControlledBlock child keeps its own controls — it is not wrapped again."""
    theta = 0.7
    comp = CompositeBlock([ControlledBlock(_ry_block(theta), 1).build()], name="solo").build()

    assert comp.n_qubits == 2
    np.testing.assert_allclose(_unitary(comp), _controlled(_ry(theta), 1), atol=1e-12)


def test_controlled_child_in_wider_composite_is_not_re_controlled():
    """Regression: a spectator child left room for the spurious extra control, so
    the double wrap silently produced C-C-U instead of raising on the remap."""
    theta = 0.7
    spectator = SimpleBlock(1, name="spectator")
    spectator.x(0)
    spectator.build()
    spectator.target_qubits = [2]

    comp = CompositeBlock(
        [ControlledBlock(_ry_block(theta), 1).build(), spectator], name="wide"
    ).build()
    assert comp.n_qubits == 3

    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    expected = np.kron(pauli_x, _controlled(_ry(theta), 1))  # X on q2, the high bit
    np.testing.assert_allclose(_unitary(comp), expected, atol=1e-12)


def test_controlled_leaf_sizes_the_parent():
    """Controlisation is explicit (§13): a ControlledBlock child reports its
    full width, so the inferred parent width includes the control."""
    theta = 0.35
    leaf = SimpleBlock(1, name="leaf")
    leaf.ry(0, theta)
    leaf.build()

    comp = CompositeBlock([ControlledBlock(leaf, 1, [True])], name="leaf_parent").build()
    assert comp.n_qubits == 2
    np.testing.assert_allclose(_unitary(comp), _controlled(_ry(theta), 1), atol=1e-12)


def test_standalone_and_composed_blocks_agree():
    """A block's circuit never depends on its container: HnBlock alone equals
    HnBlock inside a CompositeBlock, and neither is controlled."""
    hn = HnBlock(2).build()
    comp = CompositeBlock([HnBlock(2)]).build()
    assert [str(c) for c in hn.flatten()] == [str(c) for c in comp.flatten()]
    np.testing.assert_allclose(_unitary(hn), _unitary(comp), atol=1e-12)


def test_unbuilt_controlled_block_reports_its_width():
    """``n_qubits`` is knowable at construction — a parent sizing itself from an
    unbuilt ControlledBlock child must not read 0."""
    ctrl_u = ControlledBlock(_ry_block(0.2), 2)
    assert ctrl_u.n_qubits == 3
    ctrl_u.build()
    assert ctrl_u.n_qubits == 3


def test_composite_sizes_itself_from_an_unbuilt_controlled_child():
    theta = 0.15
    comp = CompositeBlock([ControlledBlock(_ry_block(theta), 1)], name="unbuilt").build()
    assert comp.n_qubits == 2
    np.testing.assert_allclose(_unitary(comp), _controlled(_ry(theta), 1), atol=1e-12)


def test_nested_unbuilt_controlled_blocks_report_their_width():
    """Widths resolve bottom-up: each wrapper is constructed after its inner."""
    theta = 0.2
    single = ControlledBlock(_ry_block(theta), 1)
    double = ControlledBlock(single, 1)
    assert (single.n_qubits, double.n_qubits) == (2, 3)
    np.testing.assert_allclose(
        _unitary(double.build()), _controlled(_controlled(_ry(theta), 1), 2), atol=1e-12
    )
