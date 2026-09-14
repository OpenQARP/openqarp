"""block.py contracts: structural-block
deepcopy, constructor validation, unbuilt-raises, parameter coercion edges,
bulk parametric layers, and operator-protocol fallbacks.
"""

import copy

import numpy as np
import pytest
import sympy
from sympy import Symbol

import qarpx as qx
from qarp.blocks import (
    CompositeBlockBase,
    ConditionalBlock,
    ControlledBlock,
    MeasureBlock,
    ResetBlock,
    SimpleBlock,
)


def _ry_block(n=1, sym="a"):
    b = SimpleBlock(n)
    b.ry(0, Symbol(sym))
    return b


# ── deepcopy of structural primitives ───────────────────────────────────


def test_deepcopy_measure_block():
    m = MeasureBlock(0, 0)
    m.build()
    c = copy.deepcopy(m)
    assert c is not m
    assert c.qubit_index == 0 and c.cbit_index == 0
    assert qx.commands_equal(c.flatten(), m.flatten())


def test_deepcopy_reset_block():
    r = ResetBlock(0)
    r.build()
    c = copy.deepcopy(r)
    assert c is not r
    assert c.qubit_index == 0
    assert qx.commands_equal(c.flatten(), r.flatten())


def test_deepcopy_conditional_block():
    then_body = SimpleBlock(1)
    then_body.x(0)
    then_body.build()
    cond = ConditionalBlock([0], [True], then_body, None)
    cond.build()
    c = copy.deepcopy(cond)
    assert c is not cond
    assert list(c.condition_cbits) == [0]
    assert c.then_body is not cond.then_body
    assert qx.commands_equal(c.flatten(), cond.flatten())


# ── constructor / finalize validation ───────────────────────────────────


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"target_qubits": [0, 0]}, "unique indices"),
    ],
)
def test_validation_raises(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SimpleBlock(1, **kwargs).build()


def test_target_qubits_length_mismatch_at_build():
    with pytest.raises(ValueError, match="does not match n_qubits"):
        SimpleBlock(2, target_qubits=[0, 1, 2]).build()


# ── symbols surface edges ───────────────────────────────────────────────


def test_refresh_symbols_contracts():
    b = _ry_block().build()
    with pytest.raises(ValueError, match="Postfix cannot be empty"):
        b.refresh_symbols("   ")
    plain = SimpleBlock(1)
    plain.h(0)
    plain.build()
    with pytest.raises(RuntimeError, match="no symbols defined"):
        plain.refresh_symbols("_1")


# ── equality / export / accessor preconditions ──────────────────────────


def test_eq_different_qubit_counts_is_false():
    a = SimpleBlock(1)
    a.h(0)
    a.build()
    b = SimpleBlock(2)
    b.h(0)
    b.build()
    assert (a == b) is False


def test_to_qasm_requires_build_and_writes_output(tmp_path):
    b = SimpleBlock(1)
    b.h(0)
    with pytest.raises(RuntimeError, match="not built"):
        b.to_qasm3()
    b.build()
    out = tmp_path / "circ.qasm"
    text = b.to_qasm3(output=str(out))
    assert "OPENQASM 3" in text
    assert out.read_text() == text


def test_n_gates_requires_build():
    b = SimpleBlock(1)
    b.h(0)
    with pytest.raises(RuntimeError, match="not built"):
        b.n_gates()


# ── parameter coercion edges ────────────────────────────────────────────


def test_numeric_sympy_expression_becomes_concrete_angle():
    # §2 Rx(θ) = exp(−iθX/2): at θ = π/2 the unitary is
    # [[cos, −i sin], [−i sin, cos]] with cos = sin = 1/√2.
    b = SimpleBlock(1)
    b.rx(0, sympy.pi / 2)
    b.build()
    u = np.array(qx.QarpSimulator().unitary_matrix(b.flatten(), 1))
    s = 1 / np.sqrt(2)
    assert np.allclose(u, np.array([[s, -1j * s], [-1j * s, s]]), atol=1e-12)


@pytest.mark.parametrize("expr", [Symbol("a") * Symbol("b"), Symbol("a") ** 2])
def test_non_linear_symbolic_angles_rejected(expr):
    # One coercion, one named error (P1.12): no more bare sympy TypeError.
    b = SimpleBlock(1)
    with pytest.raises(ValueError, match="linear in one symbol"):
        b.rx(0, expr)


def test_cpp_call_unknown_method_raises():
    with pytest.raises(AttributeError, match="not found on any qarpx base"):
        SimpleBlock(1)._cpp_call("no_such_method")


# ── bulk parametric layers ──────────────────────────────────────────────


def test_bulk_two_qubit_parametric_matches_scalar_calls():
    bulk = SimpleBlock(3)
    bulk.rzz([(0, 1, 0.3), (1, 2, 0.5)])
    bulk.build()

    scalar = SimpleBlock(3)
    scalar.rzz(0, 1, 0.3)
    scalar.rzz(1, 2, 0.5)
    scalar.build()

    assert qx.commands_equal(bulk.flatten(), scalar.flatten())


# ── operator protocol fallbacks ─────────────────────────────────────────


def test_add_non_block_raises_type_error():
    b = SimpleBlock(1)
    b.h(0)
    b.build()
    with pytest.raises(TypeError):
        b + 3


def test_binary_operators_defer_to_the_reflected_operand():
    """``__mul__``/``__or__`` return NotImplemented for a non-Block rather than
    raising, so Python falls through to the right operand's reflected method.

    There is no C++ operator slot to shadow them (``qx.SimpleBlock.__mul__`` is
    absent); the mixin's are the bound ones.  Replacing the ``NotImplemented``
    returns with a ``raise`` would still satisfy the plain TypeError tests above,
    so the protocol needs its own pin.
    """

    class Reflected:
        def __rmul__(self, other):
            return "rmul"

        def __ror__(self, other):
            return "ror"

    b = SimpleBlock(1)
    b.h(0)
    b.build()
    assert b * Reflected() == "rmul"
    assert b | Reflected() == "ror"


def test_mul_non_int_raises_type_error():
    b = SimpleBlock(1)
    b.h(0)
    b.build()
    with pytest.raises(TypeError):
        b * 1.5


def test_repr_names_shape():
    text = repr(SimpleBlock(2, name="probe"))
    assert text.startswith("SimpleBlock(n_qubits=2")
    assert "name='probe'" in text


# ── composite wiring and controlled defaults ────────────────────────────


def test_add_wired_child_with_and_without_targets():
    parent = CompositeBlockBase(n_qubits=2, name="parent")
    plain = SimpleBlock(1)
    plain.h(0)
    targeted = SimpleBlock(1, target_qubits=[1])
    targeted.x(0)
    parent.add_wired_child(plain)
    parent.add_wired_child(targeted)
    parent.build()
    assert len(parent.flatten()) == 2


def test_controlled_block_default_ctrl_state_is_all_true():
    inner = SimpleBlock(1)
    inner.x(0)
    inner.build()
    implicit = ControlledBlock(inner, 2).build()
    explicit = ControlledBlock(inner, 2, [True, True]).build()
    assert list(implicit.ctrl_state) == [True, True]
    # Phase-exact equality (§18): controllable blocks compare exactly.
    u1 = np.array(qx.QarpSimulator().unitary_matrix(implicit.flatten(), 3))
    u2 = np.array(qx.QarpSimulator().unitary_matrix(explicit.flatten(), 3))
    assert np.allclose(u1, u2, atol=1e-14)


def test_negative_target_qubits_rejected_with_a_message_naming_the_argument():
    """The Python guard runs before the C++ setter, whose own rejection names
    neither the argument nor the offending value."""
    with pytest.raises(ValueError, match=r"target_qubits must be a list of non-negative"):
        SimpleBlock(2, target_qubits=[-1, 0])
    with pytest.raises(ValueError, match=r"target_qubits must be a list of non-negative"):
        SimpleBlock(2, target_qubits=[0, 1.5])
    with pytest.raises(ValueError, match=r"target_qubits must be a list of non-negative"):
        SimpleBlock(2, target_qubits=[0, True])
    with pytest.raises(ValueError, match="unique indices"):
        SimpleBlock(2, target_qubits=[1, 1])


def test_negative_target_qubits_after_construction_still_hit_the_binding():
    """Known gap, pinned: the Python guard runs only at construction.  Direct
    assignment on a built block reaches the nanobind setter, whose TypeError
    names neither the argument nor the value.  Closing it needs a property
    override on a hot attribute — judged not worth it in the to-do sweep."""
    b = SimpleBlock(1)
    b.h(0)
    b.build()
    with pytest.raises(TypeError, match="incompatible function arguments"):
        b.target_qubits = [-1]


def test_numpy_integer_target_qubits_are_accepted_and_normalised():
    """A guard checking ``isinstance(q, int)`` would refuse ``np.int64``, which
    the C++ setter accepts and narrows — indices built with numpy are a normal
    way to call this."""
    block = SimpleBlock(2, target_qubits=list(np.arange(2)))
    block.h(0)
    block.build()
    assert block.target_qubits == [0, 1]
    assert all(type(q) is int for q in block.target_qubits)
