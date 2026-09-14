"""``CompositeBlock.add_child`` after ``build()`` wires exactly the new child
(pipeline_hardening_plan.md P1.9).  Oracle: analytic single-qubit matrices."""

import numpy as np

from qarp.blocks import CompositeBlock, SimpleBlock

H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]])


def _gate(name):
    b = SimpleBlock(1, name=name)
    getattr(b, name)(0)
    b.build()
    return b


def test_add_child_after_build_adds_only_the_new_child():
    top = CompositeBlock([_gate("h")], n_qubits=1).build()
    top.add_child(_gate("x"))
    top.build()
    assert [str(c) for c in top.flatten()] == ["H [q0]", "X [q0]"]
    np.testing.assert_allclose(top.unitary_matrix(), X @ H, atol=1e-12)


def test_repeated_add_child_and_build_is_linear_not_quadratic():
    top = CompositeBlock([_gate("h")], n_qubits=1).build()
    for _ in range(3):
        top.add_child(_gate("x"))
        top.build()
    assert [str(c) for c in top.flatten()] == ["H [q0]", "X [q0]", "X [q0]", "X [q0]"]
    np.testing.assert_allclose(top.unitary_matrix(), X @ H, atol=1e-12)  # X^3 = X


def test_build_without_new_children_is_idempotent():
    top = CompositeBlock([_gate("h"), _gate("x")], n_qubits=1).build()
    before = [str(c) for c in top.flatten()]
    top.build()
    top.build()
    assert [str(c) for c in top.flatten()] == before


def test_add_child_refreshes_symbols():
    from sympy import Symbol

    a = Symbol("a")
    top = CompositeBlock([_gate("h")], n_qubits=1).build()
    assert tuple(top.symbols) == ()
    rz = SimpleBlock(1)
    rz.rz(0, a)
    rz.build()
    top.add_child(rz)
    top.build()
    assert tuple(top.symbols) == (a,)
