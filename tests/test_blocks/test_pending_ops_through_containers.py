"""Lazy Python-level transforms survive embedding into C++ containers.

``set_symbols`` / ``replace_symbols`` / ``dagger`` are applied lazily in
``Block.flatten()`` — the canonical C++ command buffer stays untransformed.
Any C++ container that lifts a child's *raw* buffer (``CompositeBlock`` via
``add_child``, ``ControlledBlock``, ``ConditionalBlock``) must first bake those
pending ops (``_materialise_pending_ops``) or they are silently dropped:
``set_symbols`` resurrects the original symbols, ``dagger`` reverts to the
un-daggered gate.  Regression for the ``ControlledBlock`` /
``ConditionalBlock`` fixes — pinned NUMERICALLY (full-unitary oracle), not just
by symbol bookkeeping.
"""

from copy import deepcopy

import numpy as np
import pytest
from sympy import Symbol

import qarpx as qx
from qarp.blocks import (
    CompositeBlock,
    ConditionalBlock,
    ControlledBlock,
    SimpleBlock,
)

_BRANCH_MARKERS = {"BranchBegin", "BranchEnd", "BranchElse"}


def _unitary(block):
    """Unitary of a block's gate stream (classical branch markers stripped)."""
    cmds = [c for c in block.flatten() if c.gate.name not in _BRANCH_MARKERS]
    return np.array(qx.QarpSimulator().unitary_matrix(cmds, block.n_qubits))


def _controlled_oracle(inner_unitary, n_inner):
    """``|0><0|⊗I + |1><1|⊗U`` with control = qubit 0 (qarpx LSB convention)."""
    dim = 2 ** (n_inner + 1)
    m = np.zeros((dim, dim), complex)
    for i in range(dim):
        if i & 1 == 0:
            m[i, i] = 1.0
        else:
            for t in range(2**n_inner):
                m[(t << 1) | 1, i] = inner_unitary[t, i >> 1]
    return m


def _symbolic_inner():
    """A two-qubit block with two free symbols (``a``, ``b``)."""
    b = SimpleBlock(2, name="inner")
    b.h(0)
    b.rz(0, qx.Param.symbol("a"))
    b.cx(0, 1)
    b.ry(1, qx.Param.symbol("b"))
    b.build()
    return b


# Each container: builder(inner) -> built container, and the oracle mapping the
# inner's own unitary onto the container's expected unitary.  Composite and
# Conditional apply the body directly; Controlled lifts it to |1>-controlled.
_CONTAINERS = {
    "composite": (
        lambda inner: CompositeBlock([deepcopy(inner)]).build(),
        lambda u_inner, n: u_inner,
    ),
    "controlled": (
        lambda inner: ControlledBlock(deepcopy(inner), num_controls=1, ctrl_state=[True]).build(),
        _controlled_oracle,
    ),
    "conditional": (
        lambda inner: ConditionalBlock(cbits=[0], values=[True], then_body=deepcopy(inner)).build(),
        lambda u_inner, n: u_inner,
    ),
}


@pytest.mark.parametrize("container", list(_CONTAINERS))
def test_set_symbols_survives_container_embedding(container):
    """A ``set_symbols``-bound block keeps its angles once embedded, and does
    not re-expose the bound symbols."""
    wrap, oracle = _CONTAINERS[container]
    inner = _symbolic_inner()
    resolved = inner.set_symbols(dict(zip(inner.symbols, [0.6, -0.9], strict=True)))
    assert resolved.symbols == ()

    wrapped = wrap(resolved)
    assert wrapped.symbols == (), f"{container} resurrected the bound symbols"
    assert np.allclose(_unitary(wrapped), oracle(_unitary(resolved), resolved.n_qubits)), (
        f"{container} dropped the set_symbols binding"
    )


@pytest.mark.parametrize("container", list(_CONTAINERS))
def test_dagger_survives_container_embedding(container):
    """A daggered block stays daggered once embedded (regression: it reverted
    to the un-daggered gate)."""
    wrap, oracle = _CONTAINERS[container]
    inner = SimpleBlock(2, name="d")
    inner.s(0)  # S; S† = Sdg carries the tell-tale opposite phase
    inner.cx(0, 1)
    inner.build()
    daggered = inner.dagger()

    wrapped = wrap(daggered)
    assert np.allclose(_unitary(wrapped), oracle(_unitary(daggered), daggered.n_qubits)), (
        f"{container} dropped the dagger"
    )


@pytest.mark.parametrize("container", list(_CONTAINERS))
def test_replace_symbols_survives_container_embedding(container):
    """A ``replace_symbols`` rename survives embedding: the container exposes
    the new name (not the old), and binding it evolves correctly."""
    wrap, oracle = _CONTAINERS[container]
    inner = _symbolic_inner()
    renamed = inner.replace_symbols({Symbol("a"): Symbol("zz")})

    wrapped_symbolic = wrap(renamed)
    assert set(map(str, wrapped_symbolic.symbols)) == {"zz", "b"}, (
        f"{container} did not carry the rename (old symbol resurrected)"
    )

    resolved = renamed.set_symbols({Symbol("zz"): 0.4, Symbol("b"): 0.8})
    wrapped = wrap(resolved)
    assert np.allclose(_unitary(wrapped), oracle(_unitary(resolved), resolved.n_qubits))


# ── unbuilt composite children ────────────────────────────────────────────
#
# A composite's C++ command buffer stays empty until build(), and the container
# lifts that raw buffer.  An unbuilt composite child therefore contributed
# nothing at all — a ConditionalBlock flattened to a body-less
# ``BranchBegin/BranchEnd`` pair and a ControlledBlock to an empty stream, with
# no error either way.  ``CompositeBlock`` itself always handled this, so the
# two wrappers were the outliers.


def _unbuilt_composite_with_rx(theta=0.3):
    leaf = SimpleBlock(1, name="leaf")
    leaf.rx(0, theta)
    return CompositeBlock([leaf], 1)  # deliberately not built


def test_conditional_keeps_an_unbuilt_composite_body():
    """The guarded body must survive, not collapse to an empty branch region."""
    cond = ConditionalBlock([0], [True], _unbuilt_composite_with_rx())
    cond.target_cbits = [0]  # deliberately dead branch (§8): opt in explicitly
    parent = CompositeBlock([cond], 1)
    parent.build()

    names = [qx.gate_name(c.gate) for c in parent.flatten()]

    assert names == ["BranchBegin", "Rx", "BranchEnd"]


def test_controlled_keeps_an_unbuilt_composite_inner():
    """The controlled inner must survive; an empty stream is silent data loss."""
    block = ControlledBlock(_unbuilt_composite_with_rx(), 1, [True])
    block.build()

    names = [qx.gate_name(c.gate) for c in block.flatten()]

    assert names == ["CRx"]


@pytest.mark.parametrize("container", ["conditional", "controlled"])
def test_unbuilt_composite_child_matches_prebuilt_child(container):
    """Building the child first must make no difference to the result — the
    oracle is the pre-built stream, which was always correct."""

    def make(prebuilt):
        child = _unbuilt_composite_with_rx()
        if prebuilt:
            child.build()
        if container == "controlled":
            block = ControlledBlock(child, 1, [True])
        else:
            cond = ConditionalBlock([0], [True], child)
            cond.target_cbits = [0]  # deliberately dead branch (§8)
            block = CompositeBlock([cond], 1)
        block.build()
        return [
            (qx.gate_name(c.gate), tuple(c.qubits), tuple(p.value() for p in c.params))
            for c in block.flatten()
        ]

    assert make(prebuilt=False) == make(prebuilt=True)
