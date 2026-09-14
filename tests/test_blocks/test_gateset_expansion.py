"""Python surface of the gateset expansion.

The numerical contract for SX/SXdg/Id/CH/CS/CSdg/CSX/CSXdg is pinned at the
C++ layer (unitaries, dagger, decomposition exactness, controlled lowering)
and in ``tests/test_conventions_gates.py``.  This file covers what only the
Python layer adds: the ``ccz`` / ``mcx`` sugar, the one-gate controlled
flattening seen from Python, and the QASM 3 spellings through ``to_qasm3``.

Oracles are numpy permutation matrices built here (§18).  Blocks are built
inside tests, never at module scope.
"""

import numpy as np

import qarpx as qx
from qarp.blocks import ControlledBlock, SimpleBlock


def _unitary(block):
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


def _cnx_matrix(controls, target, n_qubits):
    """C^n(X) as an explicit permutation on the LSB-first basis (§1)."""
    dim = 2**n_qubits
    m = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        j = i ^ (1 << target) if all((i >> c) & 1 for c in controls) else i
        m[j, i] = 1
    return m


def test_ccz_is_sugar_for_mcz():
    a = SimpleBlock(3, name="ccz")
    a.ccz(0, 1, 2)
    a.build()
    b = SimpleBlock(3, name="mcz")
    b.mcz([0, 1, 2])
    b.build()
    assert [c.gate for c in a.flatten()] == [qx.GateType.MCZ]
    assert a == b
    np.testing.assert_allclose(_unitary(a), np.diag([1] * 7 + [-1]), atol=1e-12)


def test_mcx_three_controls_is_the_four_qubit_toffoli():
    b = SimpleBlock(4, name="c3x")
    b.mcx(0, 1, 2, 3)
    b.build()
    np.testing.assert_allclose(_unitary(b), _cnx_matrix([0, 1, 2], 3, 4), atol=1e-12)


def test_mcx_four_controls():
    b = SimpleBlock(5, name="c4x")
    b.mcx(0, 1, 2, 3, 4)
    b.build()
    np.testing.assert_allclose(_unitary(b), _cnx_matrix([0, 1, 2, 3], 4, 5), atol=1e-12)


def test_mcx_honours_permuted_qubit_arguments():
    # Controls at global qubits 3, 1, 0; target at 2 (the July c3x case).
    b = SimpleBlock(4, name="c3x_perm")
    b.mcx(3, 1, 0, 2)
    b.build()
    np.testing.assert_allclose(_unitary(b), _cnx_matrix([3, 1, 0], 2, 4), atol=1e-12)


def test_mcx_accepts_a_list_and_degenerates_to_cx_and_ccx():
    one = SimpleBlock(2, name="mcx1")
    one.mcx([0, 1])
    one.build()
    np.testing.assert_allclose(_unitary(one), _cnx_matrix([0], 1, 2), atol=1e-12)
    two = SimpleBlock(3, name="mcx2")
    two.mcx(0, 1, 2)
    two.build()
    np.testing.assert_allclose(_unitary(two), _cnx_matrix([0, 1], 2, 3), atol=1e-12)


def test_mcx_rejects_fewer_than_two_qubits():
    import pytest

    b = SimpleBlock(2, name="mcx_bad")
    with pytest.raises(ValueError, match="at least one control"):
        b.mcx(1)
    with pytest.raises(ValueError, match="at least one control"):
        b.mcx([0])


def test_mcx_is_one_mcz_between_two_hadamards():
    b = SimpleBlock(4, name="shape")
    b.mcx(0, 1, 2, 3)
    b.build()
    gates = [c.gate for c in b.flatten()]
    assert gates == [qx.GateType.H, qx.GateType.MCZ, qx.GateType.H]
    assert list(b.flatten()[1].qubits) == [0, 1, 2, 3]


def test_single_controlled_h_flattens_to_one_ch_from_python():
    inner = SimpleBlock(1, name="h")
    inner.h(0)
    inner.build()
    ctrl = ControlledBlock(inner, num_controls=1)
    ctrl.build()
    cmds = ctrl.flatten()
    assert [c.gate for c in cmds] == [qx.GateType.CH]
    assert list(cmds[0].qubits) == [0, 1]


def test_sx_squared_is_x_through_the_block_api():
    b = SimpleBlock(1, name="sx2")
    b.sx(0).sx(0)
    b.build()
    np.testing.assert_allclose(_unitary(b), np.array([[0, 1], [1, 0]]), atol=1e-12)


def test_to_qasm_uses_stdgates_names_and_modifiers():
    b = SimpleBlock(2, name="qasm_expansion")
    b.sx(0).sxdg(0).id(1).ch(0, 1).cs(0, 1).csdg(0, 1).csx(0, 1).csxdg(0, 1)
    b.build()
    qasm = b.to_qasm3()
    for line in [
        "sx q[0];",
        "inv @ sx q[0];",
        "id q[1];",
        "ch q[0], q[1];",
        "ctrl @ s q[0], q[1];",
        "ctrl @ sdg q[0], q[1];",
        "ctrl @ sx q[0], q[1];",
        "ctrl @ inv @ sx q[0], q[1];",
    ]:
        assert line in qasm, line
