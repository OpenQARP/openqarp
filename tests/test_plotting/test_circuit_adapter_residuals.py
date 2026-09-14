"""CircuitAdapter wrapper dunders,
classification fallbacks, target inference, and the command-list constructor.
"""

import pytest

import qarpx as qx
from qarp.blocks import CompositeBlock, SimpleBlock
from qarp.plotting import CircuitAdapter
from qarp.plotting._circuit_adapter import (
    _BoxCommand,
    _BoxOp,
    _Command,
    _ControlledBlockOp,
    _GateOp,
    _is_measure_only,
    _Qubit,
)


class _FlattenRaises:
    """Stub block whose flatten() always fails."""

    n_qubits = 1
    target_qubits = None
    name = "stub"

    def flatten(self):
        raise RuntimeError("boom")


def test_qubit_wrapper_dunders():
    q0, q1 = _Qubit(0), _Qubit(1)
    assert repr(q1) == "q[1]"
    assert q0 < q1
    assert sorted([q1, q0]) == [q0, q1]


def test_gate_op_symbolic_param_repr():
    op = _GateOp(qx.GateType.Rx, [qx.Param.symbol("theta")])
    text = repr(op)
    assert text.startswith("Rx(")
    assert "theta" in text


def test_command_repr_names_qubits():
    b = SimpleBlock(2)
    b.cx(0, 1)
    b.build()
    cmd = _Command(list(b.flatten())[0])
    assert repr(cmd) == "CX [q[0], q[1]]"


def test_box_wrappers_reprs():
    inner = SimpleBlock(1, name="Inner")
    inner.h(0)
    inner.build()
    assert repr(_BoxOp(inner)) == "Block(Inner)"
    box_cmd = _BoxCommand(_BoxOp(inner), [0])
    assert repr(box_cmd) == "Block [q[0]]"

    from qarp.blocks import ControlledBlock

    cb = ControlledBlock(inner, 1, [True]).build()
    op = _ControlledBlockOp(cb, cb.inner())
    assert repr(op).startswith("ControlledBlock(")


def test_command_list_constructor_requires_n_qubits():
    b = SimpleBlock(1)
    b.h(0)
    b.build()
    cmds = list(b.flatten())
    with pytest.raises(ValueError, match="n_qubits is required"):
        CircuitAdapter(cmds)
    adapter = CircuitAdapter(cmds, n_qubits=1)
    assert adapter.n_qubits == 1
    assert len(adapter.get_commands()) == 1
    adapter.name = "renamed"
    assert adapter.name == "renamed"


def test_is_measure_only_fallbacks():
    comp = CompositeBlock([SimpleBlock(1, name="a")], 1)
    assert not _is_measure_only(comp)  # composites are never measure-only
    assert not _is_measure_only(_FlattenRaises())  # unreadable → not measure-only


def test_infer_target_qubits_fallback_chain():
    infer = CircuitAdapter._child_span

    class _NQubitsOnly:
        target_qubits = None
        n_qubits = 3

    assert infer(_NQubitsOnly()) == [0, 1, 2]

    class _ReferencedOnly:
        target_qubits = None
        n_qubits = 0

        def flatten(self):
            b = SimpleBlock(3)
            b.h(2)
            b.build()
            return list(b.flatten())

    assert infer(_ReferencedOnly()) == [2]

    class _Empty:
        target_qubits = None
        n_qubits = 0

        def flatten(self):
            return []

    assert infer(_Empty()) == [0]


def test_wrap_controlled_child_falls_back_to_plain_box():
    wrapped = CircuitAdapter._wrap_controlled_child(_FlattenRaises(), [0])
    assert isinstance(wrapped, _BoxCommand)
    assert str(wrapped.op) == "Block"  # plain box, not a controlled box


def test_control_state_int_follows_conventions_lsb_order():
    """§6: control qubits are LSB-first — bit ``k`` of the packed integer is
    control qubit ``k``.  An MSB packing renders control dots reversed the
    moment the reader is corrected."""
    from qarp.blocks import ControlledBlock, XnBlock

    for state in ([True, False], [True, False, False], [True, True, False]):
        cb = ControlledBlock(XnBlock(1).build(), len(state), list(state)).build()
        packed = _ControlledBlockOp(cb, cb.inner()).get_control_state()
        per_convention = [(packed >> k) & 1 for k in range(len(state))]
        assert per_convention == [int(b) for b in state], (
            f"state={state} packed={packed:b} reads as {per_convention} under §6"
        )
