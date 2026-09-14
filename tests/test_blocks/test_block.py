"""Smoke tests for the Block base hierarchy.

A Python block IS its qarpx C++ block — there is no separate ``.circuit``
attribute.  Fuller coverage of the contract lives in ``test_block_refactor.py``
(Pattern A/B/C subclass smoke + sympy lifecycle + deepcopy + dagger
preservation).  This file keeps a small set of basic invariants that test the
Python wrapper layer directly.
"""

import numpy as np
import pytest
from sympy import Symbol

import qarpx as qx
from qarp.blocks import SimpleBlock


def test_simple_block_construction():
    b = SimpleBlock(2, name="test")
    assert b.n_qubits == 2
    assert b.name == "test"
    assert not b.is_built


def test_simple_block_build_idempotency():
    """Calling ``build()`` twice must not double-emit gates."""
    b = SimpleBlock(2)
    b.h(0)
    b.cx(0, 1)
    b.build()
    n_first = len(b.flatten())
    b.build()  # second call: no-op
    assert len(b.flatten()) == n_first


def test_any_block_is_the_qarpx_base():
    """``AnyBlock`` is the hint/isinstance type: the shared qarpx base."""
    from qarp.blocks import AnyBlock

    assert AnyBlock is qx.Block
    assert issubclass(SimpleBlock, AnyBlock)


def _assert_locates_the_bad_wire(msg: str) -> None:
    assert "streaming" in msg  # which block
    assert "n_qubits=2" in msg  # its declared width
    assert "command #2 of 3" in msg  # where in append order
    assert "H acts on qubit 5" in msg  # which gate, which index


def _streaming_block_with_a_bad_wire() -> SimpleBlock:
    b = SimpleBlock(2, name="streaming")
    b.h(0)
    b.cx(0, 1)
    b.h(5)
    return b


def test_build_out_of_range_qubit_names_block_position_and_gate():
    """An out-of-range qubit is rejected at build(); the message must locate it.

    The bare index ("qubit 5 out of range for mapping of size 2") does not say
    which gate call produced the command, so the diagnostic carries the block
    name and declared width, the command's position in append order, and the
    gate itself.  build() used to swallow this into symbols=() and let it
    surface later as "ket has no parameters to optimize".
    """
    with pytest.raises(IndexError) as excinfo:
        _streaming_block_with_a_bad_wire().build()
    _assert_locates_the_bad_wire(str(excinfo.value))


def test_flatten_out_of_range_qubit_names_block_position_and_gate():
    """The original case: the same diagnostic from flatten() on a block whose
    commands were recorded without the build() checks — mark_built() plus the
    identity target mapping _finalize would have set, i.e. the reconstruct
    protocol.  The remap error needs a mapping to remap through."""
    b = _streaming_block_with_a_bad_wire()
    b.mark_built()
    b.target_qubits = [0, 1]

    with pytest.raises(IndexError) as excinfo:
        b.flatten()
    _assert_locates_the_bad_wire(str(excinfo.value))


def test_composite_child_on_an_out_of_range_wire_raises_at_build_not_empty_symbols():
    """A composite's flat-scan used to hit the remap error and publish
    symbols=() — the child's real symbol silently lost."""
    from qarp.blocks import CompositeBlock

    child = SimpleBlock(4)
    child.rz(0, qx.Param.symbol("t"))
    child.cx(0, 3)
    child.build()

    parent = CompositeBlock([child], n_qubits=2)
    with pytest.raises(IndexError, match=r"CX acts on qubit 3 .* only 2 entries"):
        parent.build()
    assert parent.symbols is None  # never published as ()


def test_qaoa_block_sized_below_its_labels_raises_at_build():
    """The incident: QAOABlock(n_qubits=3) on edges {(0,1),(1,3)} built with
    symbols=() and surfaced as "ket has no parameters to optimize"."""
    import networkx as nx

    from qarp.blocks import QAOABlock

    g = nx.Graph()
    g.add_edge(0, 1, weight=1.0)
    g.add_edge(1, 3, weight=1.0)
    with pytest.raises(IndexError, match=r"qubit 3 .* only 3 entries"):
        QAOABlock(n_qubits=3, problem=g, n_layers=1).build()


def test_controlled_block_over_a_measurement_raises_at_build_not_empty_symbols():
    """ControlledBlock's flatten raises RuntimeError for a non-unitary inner —
    the same type C++ uses for "called before build()".  The flat-scan used to
    catch that type and publish symbols=(), losing the inner's 't'."""
    from qarp.blocks import ControlledBlock

    inner = SimpleBlock(1)
    inner.rz(0, qx.Param.symbol("t"))
    inner.measure(0, 0)
    inner.build()
    with pytest.raises(RuntimeError, match=r"cannot quantum-control a 'Measure'"):
        ControlledBlock(inner, num_controls=1).build()


def test_unbuilt_composite_free_symbols_is_empty_not_an_error():
    """The case the removed catch was really for: before build() the flat-scan
    has nothing to scan, detected by the built flag rather than by exception."""
    from qarp.blocks import CompositeBlock

    parent = CompositeBlock([SimpleBlock(1)], n_qubits=1)
    assert not parent.is_built
    assert parent.free_symbols() == []


def test_simple_block_flatten_ordering():
    b = SimpleBlock(2)
    b.h(0)
    b.cx(0, 1)
    b.h(1)
    b.build()
    cmds = b.flatten()
    assert [cmd.gate.name for cmd in cmds] == ["H", "CX", "H"]
    assert cmds[0].qubits[0] == 0
    assert list(cmds[1].qubits) == [0, 1]
    assert cmds[2].qubits[0] == 1


def test_cp_gate_convention_e_plus_i_theta_on_11():
    """Pin the canonical convention: ``cp(c, t, θ)`` puts ``e^{+iθ}`` on |11⟩.

    Tripwire for the QFT-sign episode: ``QFTBlock`` was once written assuming
    the opposite convention.  If the simulator's `CP` dispatch (or the
    `decompose_cp` step) ever flips sign, this test fails before any
    downstream algorithm starts misbehaving silently.
    """
    b = SimpleBlock(2)
    b.cp(0, 1, np.pi / 4)
    b.build()
    U = np.array(qx.QarpSimulator().unitary_matrix(b.flatten(), b.n_qubits))
    # Basis order |00⟩, |01⟩, |10⟩, |11⟩.  CP is diagonal: identity on the
    # first three basis states, phase on |11⟩.
    assert np.allclose(U, np.diag([1.0, 1.0, 1.0, np.exp(1j * np.pi / 4)]))


def test_set_symbols_returns_a_new_block_with_substituted_params():
    b = SimpleBlock(1)
    b.rx(0, qx.Param.symbol("θ"))
    b.build()
    sub = b.set_symbols({Symbol("θ"): 0.5}).build()
    cmds = sub.flatten()
    assert len(cmds) == 1
    assert not cmds[0].is_parametric()


def test_set_symbols_updates_free_symbols_and_symbols():
    """Lazy set_symbols must be visible through free_symbols() / .symbols.

    Dispatching straight to the C++ scan of the canonical (unsubstituted)
    buffer would report bound symbols as free.
    """
    b = SimpleBlock(1)
    b.rx(0, qx.Param.symbol("θ"))
    b.ry(0, qx.Param.symbol("φ"))
    b.build()

    partial = b.set_symbols({Symbol("θ"): 0.5})
    assert partial.free_symbols() == ["φ"]
    assert partial.symbols == (Symbol("φ"),)

    full = partial.set_symbols({Symbol("φ"): 0.3})
    assert full.free_symbols() == []
    assert full.symbols == ()

    # Original untouched.
    assert sorted(b.free_symbols()) == sorted(["θ", "φ"])


def test_replace_then_set_symbols_free_symbols():
    """Rename → bind-the-new-name replays in flatten() order."""
    b = SimpleBlock(1)
    b.rx(0, qx.Param.symbol("θ"))
    b.build()
    renamed = b.replace_symbols({Symbol("θ"): Symbol("φ")})
    assert renamed.free_symbols() == ["φ"]
    bound = renamed.set_symbols({Symbol("φ"): 1.0})
    assert bound.free_symbols() == []
    assert not bound.flatten()[0].is_parametric()


def test_free_symbols_recurses_into_composite_children():
    from qarp.blocks import CompositeBlock

    child = SimpleBlock(1)
    child.rx(0, qx.Param.symbol("θ"))
    child.build()
    comp = CompositeBlock([child]).build()
    assert comp.free_symbols() == ["θ"]
    assert comp.set_symbols({Symbol("θ"): 0.5}).free_symbols() == []


def test_build_after_set_symbols_prunes_bound_symbols():
    """symbols derived at build time must honour a pre-build pending bind."""
    b = SimpleBlock(1)
    b.rx(0, qx.Param.symbol("θ"))
    b.ry(0, qx.Param.symbol("φ"))
    late = b.set_symbols({Symbol("θ"): 0.5})
    late.build()
    assert late.symbols == (Symbol("φ"),)
    assert late.free_symbols() == ["φ"]


def test_set_symbols_propagates_to_python_blocks_children():
    """Binding a composite must also rebind its ``.blocks`` bookkeeping copies.

    A parent lazy queue covering only the C++ buffer leaves
    ``vqe.final_block.blocks[1]`` unbound.
    """
    from qarp.blocks import CompositeBlock

    inner = SimpleBlock(1)
    inner.rx(0, qx.Param.symbol("θ"))
    inner.build()
    comp = CompositeBlock([inner]).build()
    nested = CompositeBlock([comp]).build()

    bound = nested.set_symbols({Symbol("θ"): 0.5})
    assert bound.free_symbols() == []
    assert bound.blocks[0].free_symbols() == []
    assert bound.blocks[0].blocks[0].free_symbols() == []
    assert not bound.blocks[0].blocks[0].flatten()[0].is_parametric()

    renamed = nested.replace_symbols({Symbol("θ"): Symbol("φ")})
    assert renamed.blocks[0].blocks[0].free_symbols() == ["φ"]

    # Originals untouched.
    assert nested.blocks[0].blocks[0].free_symbols() == ["θ"]
    assert inner.free_symbols() == ["θ"]


# ── Raw qarpx block deepcopy (hybrid _copy_base_state_from path) ──────────


def test_raw_simple_block_deepcopy():
    """C++-created blocks must deepcopy without the mixin (else pickle fails)."""
    import copy

    raw = qx.SimpleBlock(2, "raw")
    raw.h(0)
    raw.rx(1, qx.Param.symbol("t"))
    raw.build()
    clone = copy.deepcopy(raw)

    assert type(clone) is qx.SimpleBlock
    assert clone.is_built()
    assert [str(c) for c in clone.commands()] == [str(c) for c in raw.commands()]
    # Independent buffers: mutating the original must not touch the clone.
    raw.x(0)
    assert len(clone.commands()) == 2
    assert len(raw.commands()) == 3


def test_transformed_block_deepcopy_as_simple_shell():
    """C++ set_symbols returns a ctor-less base qx.Block (TransformedBlock);
    its clone is a built SimpleBlock shell with identical behaviour."""
    import copy

    raw = qx.SimpleBlock(1, "src")
    raw.rx(0, qx.Param.symbol("t"))
    raw.build()
    tb = raw.set_symbols({"t": 0.5})
    assert type(tb) is qx.Block

    clone = copy.deepcopy(tb)
    assert isinstance(clone, qx.Block)
    assert type(clone) is qx.SimpleBlock
    assert clone.is_built()
    assert [str(c) for c in clone.flatten()] == [str(c) for c in tb.flatten()]


def test_raw_block_deepcopy_memo_aliasing():
    """A child shared between two slots must stay shared in the copy."""
    import copy

    shared = qx.SimpleBlock(1, "shared")
    shared.h(0)
    shared.build()
    comp = qx.CompositeBlock([shared, shared], 1, "comp")
    comp.build()

    clone = copy.deepcopy(comp)
    kids = clone.children()
    assert kids[0] is kids[1]
    assert kids[0] is not shared


def test_raw_wrapper_blocks_deepcopy():
    """Controlled / Conditional / Measure / Reset raw kinds round-trip."""
    import copy

    inner = qx.SimpleBlock(1, "inner")
    inner.h(0)
    inner.build()

    ctl = qx.ControlledBlock(inner, 2, [True, False])
    ctl.build()
    c = copy.deepcopy(ctl)
    assert type(c) is qx.ControlledBlock
    assert c.num_controls == 2
    assert list(c.ctrl_state) == [True, False]
    assert c.inner() is not inner
    assert [str(x) for x in c.flatten()] == [str(x) for x in ctl.flatten()]

    cond = qx.ConditionalBlock([0], [True], inner)
    cond.build()
    q = copy.deepcopy(cond)
    assert type(q) is qx.ConditionalBlock
    assert list(q.condition_cbits) == [0]
    assert q.then_body is not inner

    m = qx.MeasureBlock(0, 0)
    m.build()
    mc = copy.deepcopy(m)
    assert type(mc) is qx.MeasureBlock and mc.qubit_index == 0

    r = qx.ResetBlock(0)
    r.build()
    rc = copy.deepcopy(r)
    assert type(rc) is qx.ResetBlock and rc.qubit_index == 0


def test_python_composite_with_raw_child_survives_transforms():
    """The original failure: set_symbols on a composite holding a raw child."""
    from qarp.blocks import CompositeBlock

    child = SimpleBlock(1)
    child.rx(0, qx.Param.symbol("a"))
    child.build()
    raw = qx.SimpleBlock(1, "raw")
    raw.h(0)
    raw.build()

    mixed = CompositeBlock([child], n_qubits=1)
    mixed.blocks.append(raw)

    bound = mixed.set_symbols({Symbol("a"): 0.3})
    assert bound.blocks[0].free_symbols() == []
    # Raw child: cloned (not aliased), left as-is — no mixin lazy queue.
    assert bound.blocks[1] is not raw
    assert bound.blocks[1].name == "raw"

    dag = mixed.dagger()
    assert dag.blocks[1] is not raw


def test_deepcopy_base_state_canary():
    """Tripwire: every base Block field must survive a clone.

    ``_copy_base_state_from`` copies the base state via the compiler-generated
    assignment, so a new C++ field is copied automatically — but this test's
    known-property set must then be extended, which forces a conscious check
    that any *subclass* ctor field or Python-side handling is updated too.
    (The base fields ``n_controls``/``control_state`` are unbound and hence
    unverifiable from Python; they ride along in the same C++ assignment.)
    """
    import copy

    bound_props = {n for n, v in vars(qx.Block).items() if isinstance(v, property)}
    assert bound_props == {"name", "n_qubits", "n_cbits", "target_qubits", "target_cbits"}, (
        "qx.Block grew/lost a bound field — extend this canary and verify "
        "clone handling (subclass ctor fields are NOT auto-copied)."
    )

    b = qx.SimpleBlock(3, "canary")
    b.rx(0, qx.Param.symbol("t"))
    b.cx(0, 1)
    b.measure(2, 1)
    b.build()
    b.n_cbits = 2
    b.target_qubits = [2, 0, 1]
    b.target_cbits = [1, 0]

    clone = copy.deepcopy(b)
    assert clone.name == "canary"
    assert clone.n_qubits == 3
    assert clone.n_cbits == 2
    assert list(clone.target_qubits) == [2, 0, 1]
    assert list(clone.target_cbits) == [1, 0]
    assert clone.is_built()
    assert [str(c) for c in clone.commands()] == [str(c) for c in b.commands()]


# ── Simulation views: statevector() / unitary_matrix() ────────────────────


def test_statevector_view():
    b = SimpleBlock(2)
    b.h(0)
    b.cx(0, 1)
    b.build()
    expected = np.zeros(4, dtype=complex)
    expected[0] = expected[3] = 1 / np.sqrt(2)
    np.testing.assert_allclose(b.statevector(), expected, atol=1e-12)


def test_unitary_matrix_view_includes_global_phase():
    b = SimpleBlock(1)
    b.gphase(np.pi / 2)
    b.build()
    np.testing.assert_allclose(b.unitary_matrix(), 1j * np.eye(2), atol=1e-12)


def test_simulation_views_guards():
    b = SimpleBlock(1)
    b.rx(0, qx.Param.symbol("t"))
    with pytest.raises(RuntimeError, match="not built"):
        b.statevector()
    b.build()
    with pytest.raises(ValueError, match="free symbols"):
        b.statevector()
    with pytest.raises(ValueError, match="free symbols"):
        b.unitary_matrix()
    # Pending lazy substitution is applied via flatten(): Rx(π)|0⟩ = -i|1⟩.
    bound = b.set_symbols({Symbol("t"): np.pi})
    np.testing.assert_allclose(np.abs(bound.statevector()), [0.0, 1.0], atol=1e-12)


def test_statevector_view_rejects_mid_circuit_measurement():
    b = SimpleBlock(1)
    b.measure(0, 0)
    b.x(0)  # gate after measure → true mid-circuit operation
    b.build()
    with pytest.raises(RuntimeError, match="statevector"):
        b.statevector()


# ── Circuit DAG introspection ────────────────────────────────────────────


def test_block_depth():
    """``Block.depth()`` counts the longest dependency path: parallel gates
    on disjoint qubits share a time step; Barrier/GPhase weigh 0."""
    b = SimpleBlock(2)
    b.h(0)
    b.h(1)  # parallel with the first H
    b.cx(0, 1)
    b.h(0)
    b.build()
    assert b.depth() == 3
    assert len(b.flatten()) == 4  # depth < gate count


def test_block_depth_requires_built():
    b = SimpleBlock(1)
    b.h(0)
    with pytest.raises(RuntimeError, match="not built"):
        b.depth()


def test_block_gate_counts():
    """``n_1q_gates``/``n_2q_gates`` count by actual qubit footprint; a
    3-qubit gate (CCX) lands in neither bucket but is picked up by the
    general ``n_nqb_gates``."""
    b = SimpleBlock(2)
    b.h(0)
    b.h(1)
    b.cx(0, 1)
    b.rz(0, 0.3)
    b.build()
    assert b.n_1q_gates() == 3  # H, H, Rz
    assert b.n_2q_gates() == 1  # CX
    assert b.n_nqb_gates(1) == b.n_1q_gates()
    assert b.n_nqb_gates(2) == b.n_2q_gates()

    b3 = SimpleBlock(3)
    b3.ccx(0, 1, 2)
    b3.build()
    assert b3.n_1q_gates() == 0
    assert b3.n_2q_gates() == 0
    assert b3.n_nqb_gates(3) == 1


def test_block_gate_counts_exclude_non_gate_commands():
    """Barrier/Measure/Reset/GPhase are not gates: excluded from
    n_1q_gates/n_2q_gates/n_nqb_gates, but n_gates_of_type still counts them
    (unfiltered, like ``qx.CircuitDAG.count_ops()``)."""
    b = SimpleBlock(2)
    b.h(0)
    b.h(1)
    b.cx(0, 1)
    b.measure(0, 0)
    b.reset(1)
    b.gphase(np.pi / 4)
    b.build()
    cmds = list(b.flatten())
    cmds.append(qx.Command(qx.GateType.Barrier, 0))  # 1-qubit Barrier, not a gate

    fresh = SimpleBlock(2)
    fresh.set_commands(cmds)
    fresh.mark_built()

    assert fresh.n_1q_gates() == 2  # H, H only — Barrier/Measure/Reset excluded
    assert fresh.n_2q_gates() == 1  # CX
    assert fresh.n_gates_of_type(qx.GateType.H) == 2
    assert fresh.n_gates_of_type(qx.GateType.CX) == 1
    assert fresh.n_gates_of_type(qx.GateType.Barrier) == 1
    assert fresh.n_gates_of_type(qx.GateType.Measure) == 1
    assert fresh.n_gates_of_type(qx.GateType.Reset) == 1
    assert fresh.n_gates_of_type(qx.GateType.GPhase) == 1


def test_block_gate_counts_sum_over_composite_children():
    from qarp.blocks import CompositeBlock

    child1 = SimpleBlock(2)
    child1.h(0)
    child1.h(1)
    child1.cx(0, 1)
    child1.build()

    child2 = SimpleBlock(2)
    child2.rz(0, 0.3)
    child2.build()

    comp = CompositeBlock([child1, child2]).build()
    assert comp.n_1q_gates() == 3  # H, H, Rz
    assert comp.n_2q_gates() == 1  # CX
    assert comp.n_gates_of_type(qx.GateType.H) == 2
    assert comp.n_gates_of_type(qx.GateType.Rz) == 1


def test_block_gate_counts_require_built():
    b = SimpleBlock(1)
    b.h(0)
    with pytest.raises(RuntimeError, match="not built"):
        b.n_1q_gates()
    with pytest.raises(RuntimeError, match="not built"):
        b.n_2q_gates()
    with pytest.raises(RuntimeError, match="not built"):
        b.n_nqb_gates(1)
    with pytest.raises(RuntimeError, match="not built"):
        b.n_gates_of_type(qx.GateType.H)


def test_circuit_dag_readonly_introspection():
    """``qx.CircuitDAG`` exposes read-only structure: exact round-trip,
    ASAP layers, op counts, front layer."""
    b = SimpleBlock(2)
    b.h(0)
    b.h(1)
    b.cx(0, 1)
    b.build()
    cmds = b.flatten()

    dag = qx.CircuitDAG.from_commands(cmds)
    assert dag.n_qubits == 2
    assert dag.n_nodes == 3

    # RT-1: exact round-trip.
    back = dag.to_commands()
    assert [c.gate for c in back] == [c.gate for c in cmds]

    assert dag.depth() == 2
    assert dag.layers() == [[0, 1], [2]]
    assert dag.front_layer() == [0, 1]
    assert dag.count_ops() == {"H": 2, "CX": 1}
    assert dag.wires(2) == [0, 1]
    assert not dag.is_region(0)


# ── Command.to_string shows classical conditions (pipeline_hardening_plan.md P1.10) ──


def test_command_to_string_shows_condition_bits():
    """A conditioned command names the cbit and value it reads, so a
    flattened stream reveals a sibling-offset mismatch; unconditioned
    commands print as before."""
    from qarp.blocks import CompositeBlock, ConditionalBlock, MeasureBlock

    x = SimpleBlock(1)
    x.x(0)
    x.build()
    cond = ConditionalBlock([0], [True], x)
    cond.target_cbits = [0]
    top = CompositeBlock([MeasureBlock(0, 0), cond], n_qubits=1).build()
    texts = [str(c) for c in top.flatten()]
    assert any(t.endswith(" if c0==1") for t in texts), texts
    assert "if" not in str(top.flatten()[0])


# ── one angle coercion (pipeline_hardening_plan.md P1.12) ─────────────────────


@pytest.mark.parametrize("expr", ["a + b", "a**2", "a * b", "sin(a)"])
def test_non_linear_or_multi_symbol_angles_raise_named_error(expr):
    from sympy import sympify

    blk = SimpleBlock(1)
    with pytest.raises(ValueError, match="linear in one symbol"):
        blk.rz(0, sympify(expr))


def test_linear_expression_becomes_linear_param():
    a = Symbol("a")
    blk = SimpleBlock(1)
    blk.rz(0, 2 * a + 1)
    blk.build()
    bound = blk.set_symbols({a: 0.25}).build()
    theta = 2 * 0.25 + 1
    np.testing.assert_allclose(
        bound.unitary_matrix(),
        np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)]),
        atol=1e-12,
    )


def test_non_numeric_angle_raises_named_error():
    blk = SimpleBlock(1)
    with pytest.raises(ValueError, match="qarp gates accept"):
        blk.rz(0, object())


# ── controlisation is explicit (pipeline_hardening_plan.md P1.8) ─────────────


def _exported_block_classes():
    import inspect

    from qarp import blocks

    out = []
    for name in blocks.__all__:
        obj = getattr(blocks, name)
        if inspect.isclass(obj) and issubclass(obj, qx.Block) and obj is not qx.Block:
            out.append(obj)
    return out


def test_no_block_constructor_accepts_control_arguments():
    """§13: only ControlledBlock carries control metadata.  Every exported
    block class's constructor is free of n_controls / control_state."""
    import inspect

    from qarp.blocks import ControlledBlock

    classes = _exported_block_classes()
    assert len(classes) > 40
    for cls in classes:
        if cls is ControlledBlock:
            continue
        params = inspect.signature(cls.__init__).parameters
        assert "n_controls" not in params, cls.__name__
        assert "control_state" not in params, cls.__name__


@pytest.mark.parametrize(
    "make",
    [
        lambda: SimpleBlock(2, n_controls=1),  # type: ignore[call-arg]
        lambda: __import__("qarp.blocks", fromlist=["HnBlock"]).HnBlock(2, n_controls=1),
        lambda: __import__("qarp.blocks", fromlist=["CompositeBlock"]).CompositeBlock(
            [SimpleBlock(1)], n_controls=1
        ),
        lambda: SimpleBlock(2, None, None, None, "x"),  # type: ignore[call-arg, misc]  # stale five-positional call
    ],
)
def test_control_arguments_and_stale_positional_calls_fail_loudly(make):
    with pytest.raises(TypeError):
        make()


def test_controlled_block_positional_order_matches_other_blocks():
    """target_qubits precedes name, as in every other block's signature."""
    from qarp.blocks import ControlledBlock

    ctl = ControlledBlock(SimpleBlock(1), 1, [True], [0, 1], "x")
    assert ctl.target_qubits == [0, 1]
    assert ctl.name == "x"


def test_controlled_block_matches_analytic_block_diagonal():
    """ControlledBlock(HnBlock(2), 1) with the control at qubit 0 (LSB) is
    |0><0| ⊗ I + |1><1| ⊗ (H⊗H) with the control as the innermost factor."""
    from qarp.blocks import ControlledBlock, HnBlock

    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    P0 = np.diag([1.0, 0.0])
    P1 = np.diag([0.0, 1.0])
    expected = np.kron(np.eye(4), P0) + np.kron(np.kron(H, H), P1)
    block = ControlledBlock(HnBlock(2).build(), 1, [True]).build()
    assert block.n_controls == 1 and block.control_state == [True]
    np.testing.assert_allclose(block.unitary_matrix(), expected, atol=1e-12)


class _TwoQubitRotations(SimpleBlock):
    """Inner gates outside the single-control direct table (CRz, CP, RZZ):
    each one is lowered to the multi-control basis on its own."""

    def __init__(self, n_gates):
        super().__init__(4)
        self.n_gates = n_gates

    def build_vanilla(self):
        for i in range(self.n_gates):
            if i % 3 == 0:
                self.crz(0, 1, 0.3)
            elif i % 3 == 1:
                self.cp(1, 2, 0.4)
            else:
                self.rzz(2, 3, 0.5)
        return self


def test_controlled_block_lowers_off_table_gates_exactly():
    """The per-command lowering path (pipeline_hardening_plan.md P2.3) is
    phase-exact: C(U) equals |0><0| ⊗ I + |1><1| ⊗ U for an inner block
    of CRz/CP/RZZ, control at qubit 0 (LSB)."""
    from qarp.blocks import ControlledBlock

    inner = _TwoQubitRotations(9).build()
    U = inner.unitary_matrix()
    expected = np.kron(np.eye(16), np.diag([1.0, 0.0])) + np.kron(U, np.diag([0.0, 1.0]))
    block = ControlledBlock(inner, 1).build()
    np.testing.assert_allclose(block.unitary_matrix(), expected, atol=1e-12)


@pytest.mark.bench
def test_controlled_block_per_command_lowering_cost():
    """P2.3: the multi-control-basis transpiler is built once, not per
    lowered command.  Rebuilding it cost ~2.9 µs per command on top of
    ~3.3 µs of lowering (M-series laptop); the ceiling sits between."""
    import time

    from qarp.blocks import ControlledBlock

    inner = _TwoQubitRotations(3000).build()
    n_inner = len(inner.flatten())
    ControlledBlock(inner, 1).build().flatten()  # warm
    reps = 5
    t = time.perf_counter()
    for _ in range(reps):
        ControlledBlock(inner, 1).build().flatten()
    per_command = (time.perf_counter() - t) / reps / n_inner
    assert per_command < 5e-6, f"{per_command * 1e6:.2f} µs per inner command"


def test_default_block_names_carry_no_newlines():
    """Plot wrapping belongs to the plotting layer; ``name`` is also the QASM
    header comment and the ``_dag`` suffix base (pipeline_hardening_plan.md P1.13)."""
    from qarp.blocks import (
        BrickworkEntanglingBlock,
        HEABlock,
        MixedOperatorBlock,
        SPABlock,
        TrotterBlock,
    )
    from qarp.operators import QubitOperator

    blocks = [
        BrickworkEntanglingBlock(4, circular=False, use_cz=True),
        BrickworkEntanglingBlock(4, circular=False, use_cz=False),
        HEABlock(3, 1, real=True, linear=True, circular=False, use_cz=False),
        MixedOperatorBlock(2),
        SPABlock(2, 1, real=True, linear=True, circular=False),
        TrotterBlock(1, QubitOperator("Z0"), time=0.1),
    ]
    for b in blocks:
        assert "\n" not in b.name, b.name
