"""Execution-level pins for the classical-register result contract.

The C++ suite pins these through test_functional_mcm.cpp; these tests pin the
Python-visible contract on `n_cbits` / `cbit_history` through blocks +
`QarpSimulator.run`, against analytic expectations.
"""

import qarpx as qx
from qarp.blocks import CompositeBlock, ConditionalBlock, SimpleBlock

N_SHOTS = 1000
SEED = 7


def _run(block, n_qubits, n_shots=N_SHOTS, seed=SEED):
    return qx.QarpSimulator().run(block.flatten(), n_qubits, n_shots, seed=seed)


def _dead(cond):
    """Structural tests below never write the cbit they condition on; an
    explicit target_cbits keeps such a deliberately dead branch legal (§8)."""
    cond.target_cbits = list(range(cond.n_cbits))  # identity alias into the parent space
    return cond


def test_terminal_measurement_populates_cbit_history():
    """Terminal-only measurement takes the sample-once fast path — the
    classical register must still be recorded per shot."""
    b = SimpleBlock(2)
    b.x(0)
    b.measure(0, 0)
    b.measure(1, 1)
    b.build()
    res = _run(b, 2)
    assert res.n_cbits == 2
    assert len(res.cbit_history) == N_SHOTS
    assert all(h == [True, False] for h in res.cbit_history)
    assert res.counts == {1: N_SHOTS}


def test_mid_circuit_remeasure_records_both_outcomes():
    """X;Measure→c0;X;Measure→c1 on one qubit: history is [True, False]
    every shot and the final state is |0⟩."""
    b = SimpleBlock(1)
    b.x(0)
    b.measure(0, 0)
    b.x(0)
    b.measure(0, 1)
    b.build()
    res = _run(b, 1)
    assert res.n_cbits == 2
    assert all(h == [True, False] for h in res.cbit_history)
    assert res.counts == {0: N_SHOTS}


def test_feedforward_cbit_history_matches_counts():
    """H;Measure→c0; if c0: X(1).  Outcomes are only |00⟩ and |11⟩, the number
    of recorded True cbits equals the |11⟩ count exactly, and the branch rate
    is ~1/2 (5σ binomial bound).

    Composite children get *disjoint* cbit ranges by default, so reading a
    sibling's cbit needs an explicit ``target_cbits`` alias — without it the
    condition is offset past the measurement and silently never fires.
    """
    prelude = SimpleBlock(2)
    prelude.h(0)
    prelude.measure(0, 0)
    body = SimpleBlock(2)
    body.x(1)
    body.build()
    cond = ConditionalBlock(cbits=[0], values=[True], then_body=body)
    cond.build()
    cond.target_cbits = [0]  # alias onto the prelude's cbit 0
    circ = CompositeBlock([prelude, cond])
    circ.build()

    res = _run(circ, 2)
    assert res.n_cbits == 1
    assert set(res.counts) <= {0b00, 0b11}
    n_true = sum(h[0] for h in res.cbit_history)
    assert n_true == res.counts.get(0b11, 0)
    assert abs(n_true / N_SHOTS - 0.5) < 5 * 0.5 / N_SHOTS**0.5


def test_mid_circuit_reset_clears_qubit_and_cbit():
    """H;Reset;Measure→c0: the qubit is deterministically |0⟩ afterwards."""
    b = SimpleBlock(1)
    b.h(0)
    b.reset(0)
    b.measure(0, 0)
    b.build()
    res = _run(b, 1)
    assert res.n_cbits == 1
    assert res.counts == {0: N_SHOTS}
    assert not any(h[0] for h in res.cbit_history)


# ── dagger over branch regions ────────────────────────────────────────────


def _names(cmds):
    return [qx.gate_name(c.gate) for c in cmds]


def _markers_are_wellformed(cmds):
    """Grammar check: every BranchBegin closes, nothing closes unopened, and
    a BranchElse only appears inside an open region."""
    depth = 0
    for name in _names(cmds):
        if name == "BranchBegin":
            depth += 1
        elif name == "BranchEnd":
            depth -= 1
            if depth < 0:
                return False
        elif name == "BranchElse" and depth == 0:
            return False
    return depth == 0


def test_dagger_keeps_branch_markers_in_order_and_negates_bodies():
    """A branch region is atomic under dagger: markers and condition keep their
    order and only the bodies invert.  Reversing the raw stream would put
    BranchEnd before its BranchBegin."""
    then_body = SimpleBlock(1, name="then")
    then_body.rx(0, 0.7)
    else_body = SimpleBlock(1, name="else")
    else_body.ry(0, 1.1)
    cond = ConditionalBlock([0], [True], then_body, else_body)
    cond.build()

    cmds = cond.dagger().flatten()

    assert _names(cmds) == ["BranchBegin", "Rx", "BranchElse", "Ry", "BranchEnd"]
    assert cmds[1].params[0].value() == -0.7
    assert cmds[3].params[0].value() == -1.1
    # Condition travels with the marker, unnegated.
    assert list(cmds[0].condition_bits) == [0]
    assert list(cmds[0].condition_values) == [True]


def test_dagger_of_composite_child_region_stays_wellformed():
    """The region must survive daggering the *parent*, not only itself — the
    parent daggers an already-flattened stream where the region is just
    three loose marker commands."""
    body = SimpleBlock(1, name="body")
    body.rx(0, 0.7)
    pre = SimpleBlock(1, name="pre")
    pre.h(0)
    post = SimpleBlock(1, name="post")
    post.t(0)
    parent = CompositeBlock([pre, _dead(ConditionalBlock([0], [True], body)), post], 1)
    parent.build()

    cmds = parent.dagger().flatten()

    assert _markers_are_wellformed(cmds)
    # Whole items reverse; the region does not turn inside out.
    assert _names(cmds) == ["Tdg", "BranchBegin", "Rx", "BranchEnd", "H"]


def test_dagger_of_nested_regions_stays_wellformed():
    """Nested regions pair with their own markers, not the outermost."""
    inner_body = SimpleBlock(1, name="inner")
    inner_body.rx(0, 0.3)
    outer_body = CompositeBlock([_dead(ConditionalBlock([1], [True], inner_body))], 1)
    parent = CompositeBlock([_dead(ConditionalBlock([0], [True], outer_body))], 1)
    parent.build()

    cmds = parent.dagger().flatten()

    assert _markers_are_wellformed(cmds)
    assert _names(cmds).count("BranchBegin") == _names(cmds).count("BranchEnd") == 2


def test_dagger_is_an_involution_over_branch_regions():
    """Daggering twice restores the original stream — an implementation-free
    check that the region transform is exactly inverted."""
    body = SimpleBlock(1, name="body")
    body.rx(0, 0.7)
    pre = SimpleBlock(1, name="pre")
    pre.h(0)
    parent = CompositeBlock([pre, _dead(ConditionalBlock([0], [True], body))], 1)
    parent.build()

    once = parent.dagger().flatten()
    twice = parent.dagger().dagger().flatten()

    assert _names(twice) == _names(parent.flatten())
    assert _names(once) != _names(twice) or not any(c.params for c in once)
    for got, want in zip(twice, parent.flatten(), strict=True):
        assert got.gate == want.gate
        assert [p.value() for p in got.params] == [p.value() for p in want.params]


def test_daggered_branch_stream_is_accepted_by_the_dag():
    """The consequence that makes this a bug rather than a cosmetic ordering:
    an inverted region is rejected by CircuitDAG with 'stray BranchEnd'."""
    body = SimpleBlock(1, name="body")
    body.rx(0, 0.7)
    parent = CompositeBlock([_dead(ConditionalBlock([0], [True], body))], 1)
    parent.build()

    cmds = list(parent.dagger().flatten())

    optimized = qx.Transpiler(qx.native_gateset()).transpile_and_optimize(cmds, qx.OptLevel.O1)
    assert _markers_are_wellformed(optimized)


# ── sibling cbit aliasing (pipeline_hardening_plan.md P1.10) ─────────────────


def _x_on(q, n=2):
    b = SimpleBlock(n)
    b.x(q)
    b.build()
    return b


def test_sibling_conditional_without_alias_raises_at_build():
    import pytest

    from qarp.blocks import MeasureBlock

    cond = ConditionalBlock([0], [True], _x_on(1))
    with pytest.raises(ValueError, match="target_cbits"):
        CompositeBlock([_x_on(0), MeasureBlock(0, 0), cond], n_qubits=2).build()


def test_sibling_conditional_with_explicit_alias_fires_every_shot():
    """X(0); measure q0 -> c0; if c0 == 1: X(1).  With the alias the branch
    reads the measurement and the outcome is (1, 1) on every shot."""
    from qarp.blocks import MeasureBlock

    cond = ConditionalBlock([0], [True], _x_on(1))
    cond.target_cbits = [0]
    top = CompositeBlock([_x_on(0), MeasureBlock(0, 0), cond], n_qubits=2).build()
    sr = _run(top, 2)
    assert dict(sr.counts) == {3: N_SHOTS}


def test_explicit_alias_to_an_unwritten_cbit_is_a_deliberate_dead_branch():
    """§8: an explicit target_cbits keeps a dead branch legal; it never fires
    (cbit 1 is never written), so the outcome is (1, 0)."""
    from qarp.blocks import MeasureBlock

    cond = ConditionalBlock([0], [True], _x_on(1))
    cond.target_cbits = [1]
    top = CompositeBlock([_x_on(0), MeasureBlock(0, 0), cond], n_qubits=2).build()
    sr = _run(top, 2)
    assert dict(sr.counts) == {1: N_SHOTS}


def test_self_contained_conditional_next_to_explicit_dead_branch_builds():
    """Attribution: a child that measures and reads its OWN cbit is clean;
    a sibling with an explicit dead-branch alias is legal; the composite
    must build and blame neither."""
    from qarp.blocks import MeasureBlock

    inner = CompositeBlock(
        [_x_on(0), MeasureBlock(0, 0), _alias(ConditionalBlock([0], [True], _x_on(1)), [0])],
        n_qubits=2,
    )
    inner.build()
    dead = _alias(ConditionalBlock([0], [True], _x_on(1)), [5])
    top = CompositeBlock([inner, dead], n_qubits=2).build()
    sr = _run(top, 2)
    assert dict(sr.counts) == {3: N_SHOTS}


def test_offending_child_is_named():
    import pytest

    from qarp.blocks import MeasureBlock

    cond = ConditionalBlock([0], [True], _x_on(1), name="feedforward")
    with pytest.raises(ValueError, match="feedforward reads its cbit"):
        CompositeBlock([_x_on(0), MeasureBlock(0, 0), cond], n_qubits=2).build()


def _alias(cond, cbits):
    cond.target_cbits = list(cbits)
    return cond
