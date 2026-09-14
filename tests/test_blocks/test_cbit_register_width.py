"""Agreement pins for the classical-register width across every surface.

`block.n_cbits`, `qx.SamplingResult.n_cbits`, `cbit_history` width and the
OpenQASM 3 `bit[k] c;` declaration are all derived from one shared helper
(`cbit_register_width`, core/command.h), so they must agree: a
builder-`measure()`d block reporting 0 while the simulator reports the true
width is the failure this guards.  Oracle is a hand-counted width, plus the simulator's and emitter's
independently-derived values.
"""

import re

import pytest

import qarpx as qx
from qarp.blocks import CompositeBlockBase, ConditionalBlock, MeasureBlock, ResetBlock, SimpleBlock


def _qasm_register_width(block) -> int:
    """The `bit[k] c;` width the OpenQASM 3 emitter declares (0 if absent)."""
    match = re.search(r"^bit\[(\d+)\] c;", block.to_qasm3(), re.MULTILINE)
    return int(match.group(1)) if match else 0


def _sampled_width(block, n_qubits) -> tuple:
    result = qx.QarpSimulator().run(block.flatten(), n_qubits, 4, seed=1)
    return result.n_cbits, len(result.cbit_history[0])


def test_builder_measures_set_n_cbits():
    """Two `measure()` calls on a leaf: the block reports the analytic width 2.

    This is the regression the whole change exists for — it reported 0.
    """
    block = SimpleBlock(3)
    block.h(0)
    block.measure(0, 0)
    block.measure(1, 1)
    block.build()
    assert block.n_cbits == 2


def test_all_surfaces_agree_on_a_hand_counted_width():
    """block / simulator / cbit_history / QASM all equal the same counted k."""
    block = SimpleBlock(3)
    block.h(0)
    block.measure(0, 0)
    block.measure(1, 1)
    block.build()

    expected = 2  # cbits 0 and 1 written ⇒ 1 + max index
    sampled_n_cbits, history_width = _sampled_width(block, 3)
    assert block.n_cbits == expected
    assert sampled_n_cbits == expected
    assert history_width == expected
    assert _qasm_register_width(block) == expected


def test_non_contiguous_cbits_size_to_the_largest_index():
    """`measure(0, 3)` needs a 4-bit register, not a 1-bit one."""
    block = SimpleBlock(2)
    block.measure(0, 3)
    block.build()

    sampled_n_cbits, history_width = _sampled_width(block, 2)
    assert block.n_cbits == 4
    assert sampled_n_cbits == 4
    assert history_width == 4
    assert _qasm_register_width(block) == 4


def test_explicit_width_survives_build_and_rebuild():
    """A declared register wider than the circuit writes is never overwritten.

    The absorbers rely on this to round-trip a source `bit[n] c;`.
    """
    block = SimpleBlock(2)
    block.measure(0, 0)
    block.n_cbits = 5
    block.build()
    assert block.n_cbits == 5

    block.build()  # idempotent rebuild must not shrink it
    assert block.n_cbits == 5


def test_explicit_width_narrower_than_the_circuit_is_widened():
    """A declared register may only widen, never shrink.

    `measure(0, 3)` needs 4 bits.  An explicit `n_cbits = 1` used to survive
    build, leaving the block reporting 1 while the simulator, both emitters and
    `cbit_register_width` all sized from the commands and used 4.
    """
    block = SimpleBlock(1)
    block.measure(0, 3)
    block.n_cbits = 1
    block.build()

    expected = 4
    sampled_n_cbits, history_width = _sampled_width(block, 1)
    assert block.n_cbits == expected
    assert qx.cbit_register_width(block.flatten()) == expected
    assert sampled_n_cbits == expected
    assert history_width == expected
    assert _qasm_register_width(block) == expected


def test_reinference_requires_a_fresh_block():
    """Inference happens at first build only.

    ``Block.build()`` is idempotent at the Python layer — it returns early on an
    already-built block so ``build_vanilla()`` cannot double-append commands —
    so clearing ``n_cbits`` afterwards does *not* re-arm inference.  Build a new
    block instead.
    """
    block = SimpleBlock(2)
    block.measure(0, 0)
    block.n_cbits = 5
    block.build()
    block.n_cbits = 0
    block.build()  # no-op: the block is already built
    assert block.n_cbits == 0

    fresh = SimpleBlock(2)
    fresh.measure(0, 0)
    fresh.build()
    assert fresh.n_cbits == 1


def test_condition_only_block_counts_condition_bits():
    """A block that only *reads* a cbit still needs a register that wide."""
    body = SimpleBlock(1)
    body.x(0)
    body.build()
    cond = ConditionalBlock(cbits=[2], values=[True], then_body=body)
    cond.build()
    assert cond.n_cbits == 3


def test_gate_only_and_reset_blocks_stay_zero():
    gates = SimpleBlock(2)
    gates.h(0)
    gates.cx(0, 1)
    gates.build()
    assert gates.n_cbits == 0

    reset = ResetBlock(qubit=0)
    reset.build()
    assert reset.n_cbits == 0


def test_composite_of_measure_blocks_still_offsets():
    """`resolve_cbits` aggregation is unchanged: two children ⇒ width 2."""
    composite = CompositeBlockBase(2)
    for qubit in range(2):
        child = MeasureBlock(qubit=0, cbit=0)
        child.build()
        child.target_qubits = [qubit]
        composite.add_child(child)
    composite.build()
    assert composite.n_cbits == 2

    sampled_n_cbits, _ = _sampled_width(composite, 2)
    assert sampled_n_cbits == 2


def test_repeated_writes_to_one_cbit_do_not_double_count():
    block = SimpleBlock(3)
    block.measure(0, 1)
    block.measure(1, 1)
    block.build()
    assert block.n_cbits == 2


def test_uninitialised_condition_cbits_flags_the_unwritten_read():
    """A condition reading a cbit no Measure wrote is reported by index."""
    body = SimpleBlock(2)
    body.x(1)
    body.build()
    cond = ConditionalBlock(cbits=[0], values=[True], then_body=body)
    cond.build()

    # Standalone: nothing writes cbit 0, so the branch can never fire.
    assert list(qx.uninitialised_condition_cbits(cond.flatten())) == [0]

    # Preceded by the write it reads: clean.
    prelude = SimpleBlock(2)
    prelude.h(0)
    prelude.measure(0, 0)
    prelude.build()
    cond2 = ConditionalBlock(cbits=[0], values=[True], then_body=body)
    cond2.build()
    cond2.target_cbits = [0]
    good = CompositeBlockBase(2)
    good.add_child(prelude)
    good.add_child(cond2)
    good.build()
    assert list(qx.uninitialised_condition_cbits(good.flatten())) == []


def test_sibling_composed_condition_is_reported():
    """The real footgun: composing a conditional as a sibling of its measure.

    `CompositeBlock` offsets children onto disjoint cbit ranges, so without an
    explicit ``target_cbits`` alias the condition reads past the write.
    """
    prelude = SimpleBlock(2)
    prelude.h(0)
    prelude.measure(0, 0)
    prelude.build()
    body = SimpleBlock(2)
    body.x(1)
    body.build()
    cond = ConditionalBlock(cbits=[0], values=[True], then_body=body)
    cond.build()

    composite = CompositeBlockBase(2)
    composite.add_child(prelude)
    composite.add_child(cond)
    # Offset to cbit 1, which nothing writes: build() refuses, naming the
    # child and the cbit it reads in its own frame (local 0 = parent 1).
    with pytest.raises(ValueError, match=r"Conditional reads its cbit\(s\) \[0\]"):
        composite.build()

    # The free-function query still reports it on the raw stream.
    composite._built = True
    assert list(qx.uninitialised_condition_cbits(composite.flatten())) == [1]
    # The C++ method on CompositeBlock agrees (it delegates to the free function).
    assert list(composite.uninitialised_condition_cbits()) == [1]


def _dead_branch(cbit_alias):
    """A conditional nothing writes, opted back in by an explicit alias."""
    body = SimpleBlock(2)
    body.x(1)
    body.build()
    cond = ConditionalBlock(cbits=[0], values=[True], then_body=body)
    cond.build()
    cond.target_cbits = [cbit_alias]
    return cond


def test_explicit_dead_branch_alias_survives_nesting():
    """§8: an explicit ``target_cbits`` keeps a deliberately dead branch — one
    composite level down too.  The inner build attributes the alias; the
    outer build sees only the inner composite's flattened stream, in which
    the alias is invisible, and must not rescan it."""
    from qarp.blocks import CompositeBlock

    prelude = SimpleBlock(2)
    prelude.h(0)
    prelude.build()

    direct = CompositeBlock([prelude, _dead_branch(5)], 2)
    direct.build()
    assert list(qx.uninitialised_condition_cbits(direct.flatten())) == [5]

    inner = CompositeBlock([_dead_branch(5)], 2)
    nested = CompositeBlock([SimpleBlock(2).h(0), inner], 2)
    nested.build()
    assert list(qx.uninitialised_condition_cbits(nested.flatten())) == [5]

    # The unaliased nested read still raises — at the inner composite's own
    # build, naming the conditional.
    body = SimpleBlock(2)
    body.x(1)
    body.build()
    unaliased = ConditionalBlock(cbits=[0], values=[True], then_body=body)
    unaliased.build()
    with pytest.raises(ValueError, match=r"Conditional reads its cbit\(s\) \[0\]"):
        CompositeBlock([SimpleBlock(2).h(0), CompositeBlock([unaliased], 2)], 2).build()


@pytest.mark.parametrize("width", [1, 4])
def test_emitted_qasm_parses_back(width):
    """The emitter's declaration is wide enough for every cbit it references."""
    block = SimpleBlock(2)
    block.measure(0, width - 1)
    block.build()

    absorbed = SimpleBlock.from_qasm3(block.to_qasm3())
    absorbed.build()
    assert absorbed.n_cbits >= width
