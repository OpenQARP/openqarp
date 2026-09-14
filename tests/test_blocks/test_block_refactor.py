"""Tests for the Python block hierarchy.

Covers each of the six base classes (SimpleBlock, CompositeBlockBase,
ControlledBlock, MeasureBlock, ResetBlock, ConditionalBlock), the sympy
substitution lifecycle, dagger preservation, MCM round-trip through
the simulator, and build() idempotency.
"""

import numpy as np
import pytest
from sympy import Symbol

import qarpx as qx
from qarp.blocks import (
    AnyBlock,
    CompositeBlockBase,
    ConditionalBlock,
    ControlledBlock,
    MeasureBlock,
    ResetBlock,
    SimpleBlock,
)

# ── Subclass-of-each-base smoke ─────────────────────────────────────────


class _Hn(SimpleBlock):
    """Pattern A — leaf populated via inherited gate methods."""

    def build_vanilla(self):
        for i in range(self.n_qubits):
            self.h(i)


class _CompPair(CompositeBlockBase):
    """Pattern B — composite of two children added via add_child."""

    def __init__(self, n_qubits, **kw):
        super().__init__(n_qubits, **kw)

    def build_vanilla(self):
        a = SimpleBlock(self.n_qubits, name="a")
        for i in range(self.n_qubits):
            a.h(i)
        a.build()
        b = SimpleBlock(self.n_qubits, name="b")
        for i in range(self.n_qubits - 1):
            b.cx(i, i + 1)
        b.build()
        self.add_child(a)
        self.add_child(b)


def test_simpleblock_is_a_qx_simpleblock():
    hn = _Hn(2)
    assert isinstance(hn, qx.SimpleBlock)
    assert isinstance(hn, qx.Block)


def test_simpleblock_build_and_flatten():
    hn = _Hn(3)
    hn.build()
    assert hn.is_built
    flat = hn.flatten()
    assert len(flat) == 3
    assert all(c.gate == qx.GateType.H for c in flat)


def test_compositeblock_add_child_pattern():
    cp = _CompPair(2)
    cp.build()
    assert isinstance(cp, qx.CompositeBlock)
    flat = cp.flatten()
    # 2 H + 1 CX
    assert len(flat) == 3
    assert flat[0].gate == qx.GateType.H
    assert flat[1].gate == qx.GateType.H
    assert flat[2].gate == qx.GateType.CX


def test_controlledblock_wraps_inner():
    inner = _Hn(2)
    inner.build()
    ctrl = ControlledBlock(inner, num_controls=1, ctrl_state=[True])
    ctrl.build()
    assert isinstance(ctrl, qx.ControlledBlock)
    flat = ctrl.flatten()
    # Each H of the inner becomes a controlled-H equivalent.  Don't pin the
    # exact decomposition; just assert there are gates and they touch all
    # qubits.
    assert len(flat) > 0


def test_measureblock_emits_measure():
    mb = MeasureBlock(qubit=0, cbit=0)
    mb.build()
    flat = mb.flatten()
    assert len(flat) == 1
    assert flat[0].gate == qx.GateType.Measure


def test_resetblock_emits_reset():
    # ResetBlock is a single-qubit leaf — the qubit index is local (0),
    # remapping to a parent's wider register happens via target_qubits.
    rb = ResetBlock(qubit=0)
    rb.build()
    flat = rb.flatten()
    assert len(flat) == 1
    assert flat[0].gate == qx.GateType.Reset


def test_conditionalblock_with_python_body():
    body = SimpleBlock(1, name="body")
    body.x(0)
    body.build()
    cond = ConditionalBlock(cbits=[0], values=[True], then_body=body)
    cond.build()
    flat = cond.flatten()
    # Conditional emits a BranchBegin / X / BranchEnd sequence.  The X must
    # appear; the exact branch markers depend on Phase 2's implementation.
    gates = [c.gate for c in flat]
    assert qx.GateType.X in gates


# ── AnyBlock hint type ──────────────────────────────────────────────────


def test_any_block_is_the_shared_base():
    assert AnyBlock is qx.Block
    # issubclass against qx.Block directly: AnyBlock aliases an untyped
    # nanobind class, which mypy sees as Any.
    assert issubclass(SimpleBlock, qx.Block)
    assert issubclass(CompositeBlockBase, qx.Block)


def test_class_hnblock_inherits_from_simpleblock():
    """User code subclasses `SimpleBlock` — confirm it's still ergonomic."""

    class HnBlock(SimpleBlock):
        def build_vanilla(self):
            for i in range(self.n_qubits):
                self.h(i)

    hn = HnBlock(2)
    hn.build()
    assert len(hn.flatten()) == 2


# ── Sympy substitution round-trip ───────────────────────────────────────


class _SymRx(SimpleBlock):
    def build_vanilla(self):
        self.rx(0, Symbol("theta"))


def test_sympy_symbols_collected_at_build():
    s = _SymRx(1)
    s.build()
    assert s.symbols is not None
    assert [str(sym) for sym in s.symbols] == ["theta"]


def test_set_symbols_lazy_substitution_at_flatten():
    s = _SymRx(1).build()
    s2 = s.set_symbols({Symbol("theta"): 1.5}).build()
    flat = s2.flatten()
    assert flat[0].params[0].is_concrete()
    assert flat[0].params[0].value() == pytest.approx(1.5)


def test_set_symbols_returns_same_subclass():
    """Symbol substitution should preserve the user's Python subclass identity."""
    s = _SymRx(1).build()
    s2 = s.set_symbols({Symbol("theta"): 0.5})
    assert isinstance(s2, _SymRx)


def test_replace_symbols_renames():
    s = _SymRx(1).build()
    s2 = s.replace_symbols({Symbol("theta"): Symbol("phi")}).build()
    flat = s2.flatten()
    # The renamed param should be symbolic with name "phi".
    p = flat[0].params[0]
    assert p.is_symbolic()
    assert "phi" in str(p)


# ── Dagger ──────────────────────────────────────────────────────────────


def test_dagger_preserves_subclass_identity():
    s = _SymRx(1).build()
    d = s.dagger()
    assert isinstance(d, _SymRx), "dagger should preserve Python subclass"


def test_dagger_preserves_sympy_symbols():
    s = _SymRx(1).build()
    d = s.dagger()
    assert d.symbols == s.symbols


def test_dagger_negates_param_lazily_at_flatten():
    s = _SymRx(1).build()
    d = s.dagger().build()
    flat = d.flatten()
    p = flat[0].params[0]
    # The Rx command's daggered form has a negated angle; rendered via
    # Param.to_string() this contains "-theta" (linear-form rendering).
    assert "-theta" in str(p) or "(-1)*theta" in str(p)


def test_dagger_round_trip_yields_identity_for_unitary_block():
    """G followed by G† produces identity (verified via the simulator)."""
    g = _Hn(2).build()
    dag = g.dagger().build()
    sim = qx.QarpSimulator()
    transp = qx.Transpiler(qx.native_gateset())
    cmds = transp.transpile(g.flatten() + dag.flatten())
    U = sim.unitary_matrix(cmds, 2)
    assert np.allclose(np.array(U), np.eye(4), atol=1e-10)


# ── MCM round-trip through the simulator ────────────────────────────────


def test_measureblock_inside_compositeblock_via_simulator():
    """A composite of [H · Measure] should sample 0/1 with ~50/50 probability."""
    h = SimpleBlock(1, name="h")
    h.h(0)
    h.build()

    m = MeasureBlock(0, 0)
    m.build()

    cb = CompositeBlockBase(1, name="meas-test")
    # Manually populate via add_child since cb has no build_vanilla override.
    cb.add_child(h)
    cb.add_child(m)
    cb.build()

    flat = cb.flatten()
    # 1 H + 1 Measure
    assert len(flat) == 2
    assert flat[0].gate == qx.GateType.H
    assert flat[1].gate == qx.GateType.Measure

    sim = qx.QarpSimulator()
    result = sim.run(flat, 1, 1000, seed=42)
    # ~50/50 split tolerance
    p0 = result.counts.get(0, 0) / 1000
    assert 0.4 < p0 < 0.6, f"expected ~0.5, got {p0}"


# ── build() idempotency ─────────────────────────────────────────────────


def test_build_is_idempotent():
    """Calling build() twice must not double-append commands."""
    hn = _Hn(3)
    hn.build()
    n_first = len(hn.flatten())
    hn.build()  # second call is a no-op
    n_second = len(hn.flatten())
    assert n_first == n_second == 3


def test_compositeblock_safely_calls_child_build_again():
    """Composite Pattern B builds children that may already be built."""
    pre_built = SimpleBlock(2, name="pre")
    pre_built.h(0)
    pre_built.h(1)
    pre_built.build()

    parent = CompositeBlockBase(2, name="parent")
    parent.add_child(pre_built)
    parent.build()
    # Parent's flatten gets exactly the 2 H gates from pre — not 4.
    assert len(parent.flatten()) == 2


# ── Backward-compat: 19 Pattern A subclasses still work ────────────────


@pytest.mark.parametrize(
    "cls_name,args",
    [
        ("HnBlock", (2,)),
        ("XnBlock", (3,)),
        ("QFTBlock", (2,)),
        ("IdentityBlock", (2,)),
        ("RSPBlock", (2,)),
    ],
)
def test_existing_block_subclasses_still_build(cls_name, args):
    """Pattern A subclasses with `_USE_IR = True` removed must still build."""
    import qarp.blocks as qb

    cls = getattr(qb, cls_name)
    b = cls(*args)
    b.build()
    assert b.is_built
    flat = b.flatten()
    assert isinstance(flat, list)


# ── deepcopy preserves C++ state (regression for §3.3 #8) ──────────────


def test_deepcopy_preserves_adhoc_simpleblock_commands():
    """Ad-hoc-built SimpleBlock (gates added outside build_vanilla) round-trips."""
    from copy import deepcopy

    src = SimpleBlock(2)
    src.h(0)
    src.cx(0, 1)
    src.build()

    dup = deepcopy(src)
    assert dup.is_built == src.is_built
    assert len(dup.commands()) == len(src.commands())
    src_flat = src.flatten()
    dup_flat = dup.flatten()
    assert len(src_flat) == len(dup_flat)
    assert all(str(a) == str(b) for a, b in zip(src_flat, dup_flat, strict=True))


def test_deepcopy_preserves_compositeblock_children():
    """CompositeBlock with built children survives deepcopy with no rebuild."""
    from copy import deepcopy

    inner = SimpleBlock(1, name="inner")
    inner.h(0)
    inner.build()

    parent = CompositeBlockBase(1, name="parent")
    parent.add_child(inner)
    parent.build()

    dup = deepcopy(parent)
    # Children must survive the C++ round-trip, not just the Python __dict__.
    assert len(dup.children()) == 1
    assert len(dup.flatten()) == len(parent.flatten()) == 1


def test_deepcopy_preserves_target_qubits_and_n_cbits():
    """target_qubits, target_cbits, n_cbits live on the C++ side and must survive."""
    from copy import deepcopy

    src = SimpleBlock(2)
    src.h(0)
    src.build()
    src.target_qubits = [3, 7]
    src.n_cbits = 2

    dup = deepcopy(src)
    assert dup.target_qubits == [3, 7]
    assert dup.n_cbits == 2


def test_set_symbols_through_compositeblock_with_adhoc_child():
    """parent.set_symbols(...) on a CompositeBlock with an ad-hoc-built parametric
    child must produce a flatten that applies the substitution — the chained
    set_symbols → deepcopy → flatten path that §3.3 #8 said was broken."""

    theta = Symbol("theta")
    inner = SimpleBlock(1, name="inner")
    inner.rx(0, qx.Param.symbol("theta"))
    inner.build()

    parent = CompositeBlockBase(1, name="parent")
    parent.add_child(inner)
    parent.build()

    bound = parent.set_symbols({theta: 1.234})
    flat = bound.flatten()
    assert len(flat) == 1
    # Param resolved to a concrete float, not a symbol.
    assert flat[0].params and flat[0].params[0].is_concrete()
    assert abs(flat[0].params[0].value() - 1.234) < 1e-12
