"""End-to-end smoke test for the Phase 3 MCM bindings.

Run with PYTHONPATH pointing at the build's python output dir, e.g.

    PYTHONPATH=build/python pytest tests/python/test_mcm_bindings.py

The C++ tests already cover correctness in detail; this file just confirms
the binding surface exists and the simulator returns the right shape from
Python.
"""

from __future__ import annotations

import ast
import contextlib

import qarpx as qx

# Lightweight stand-in for pytest.raises when running outside a pytest harness.
# When pytest is available, prefer the real one; the project's existing Python
# test infrastructure picks it up automatically.
try:
    import pytest

    _raises = pytest.raises
except ImportError:

    @contextlib.contextmanager
    def _raises(exc_type):
        try:
            yield
        except exc_type:
            return
        raise AssertionError(f"expected {exc_type.__name__}, none raised")


def test_measure_block_constructs_and_flattens():
    m = qx.MeasureBlock(qubit=2, cbit=5)
    m.build()
    cmds = m.flatten()
    assert len(cmds) == 1
    assert cmds[0].gate == qx.GateType.Measure
    assert cmds[0].qubits == [2]
    assert cmds[0].cbits == [5]
    assert m.n_qubits == 1
    assert m.n_cbits == 1


def test_reset_block_constructs_and_flattens():
    r = qx.ResetBlock(qubit=3)
    r.build()
    cmds = r.flatten()
    assert len(cmds) == 1
    assert cmds[0].gate == qx.GateType.Reset
    assert r.n_cbits == 0


def test_block_reset_builder_shortcut():
    b = qx.SimpleBlock(1)
    b.h(0).reset(0).measure(0, 0)
    b.build()
    gates = [qx.gate_name(c.gate) for c in b.flatten()]
    assert gates == ["H", "Reset", "Measure"]


def test_conditional_block_emits_branch_markers():
    then_body = qx.SimpleBlock(1)
    then_body.x(0)
    then_body.build()
    cond = qx.ConditionalBlock(cbits=[0], values=[True], then_body=then_body)
    cond.build()
    gates = [qx.gate_name(c.gate) for c in cond.flatten()]
    assert gates == ["BranchBegin", "X", "BranchEnd"]


def test_conditional_with_else_emits_branch_else():
    cond = qx.conditional(
        cbits=[0],
        values=[True],
        then=lambda b: b.x(0),
        else_=lambda b: b.z(0),
        n_qubits=1,
    )
    gates = [qx.gate_name(c.gate) for c in cond.flatten()]
    assert gates == ["BranchBegin", "X", "BranchElse", "Z", "BranchEnd"]


def test_conditional_factory_supports_then_only_and_else_only():
    then_only = qx.conditional([0], [True], then=lambda b: b.x(0), n_qubits=1)
    assert [qx.gate_name(c.gate) for c in then_only.flatten()] == [
        "BranchBegin",
        "X",
        "BranchEnd",
    ]

    else_only = qx.conditional([0], [True], else_=lambda b: b.z(0), n_qubits=1)
    assert [qx.gate_name(c.gate) for c in else_only.flatten()] == [
        "BranchBegin",
        "BranchElse",
        "Z",
        "BranchEnd",
    ]


def test_conditional_factory_rejects_both_none():
    with _raises(Exception):
        qx.conditional([0], [True], n_qubits=1)


def test_conditional_advertises_a_parseable_signature():
    """Regression: `n_qubits` is required but follows two defaulted arguments,
    so the signature nanobind advertises in __doc__ was not legal Python.
    Oracle is CPython's own grammar, via ast — nothing here reads the C++.

    nanobind does not set __signature__ (inspect.signature reports
    ``(*args, **kwargs)``), so this __doc__ line is the only machine-readable
    signature the binding exposes; anything that parses it — help(), IDE
    tooltips, doc and stub generators — needs it to be valid.
    """
    signature = qx.conditional.__doc__.splitlines()[0]
    ast.parse(f"def {signature}: ...")


def test_conditional_requires_keywords_past_the_cbit_operands():
    """The kw_only boundary is the fix's user-visible contract: everything from
    `then` on must be named, so `n_qubits` can never be reached positionally."""
    with _raises(Exception):
        qx.conditional([0], [True], lambda b: b.x(0), None, 1)


def test_target_cbits_attribute_round_trips():
    m = qx.MeasureBlock(qubit=0, cbit=0)
    assert m.target_cbits is None
    m.target_cbits = [3]
    assert m.target_cbits == [3]


def test_teleportation_shape_simulates_correctly():
    """H(0) → Measure(q=0, c=0) → if (c==1) X(0) — every shot lands in |0⟩."""
    h = qx.SimpleBlock(1)
    h.h(0)
    h.build()

    meas = qx.MeasureBlock(qubit=0, cbit=0)
    meas.target_cbits = [0]

    correction = qx.conditional(
        cbits=[0],
        values=[True],
        then=lambda b: b.x(0),
        n_qubits=1,
    )
    correction.target_cbits = [0]
    correction.target_qubits = [0]

    root = qx.CompositeBlock([h, meas, correction], n_qubits=1)
    root.n_cbits = 1
    root.build()
    flat = root.flatten()

    sim = qx.QarpSimulator()
    r = sim.run(flat, n_qubits=1, n_shots=500, seed=3)
    assert r.counts.get(0, 0) == 500
    assert r.counts.get(1, 0) == 0


def test_cbit_history_shape_matches_n_shots_and_n_cbits():
    """A two-bit measurement records both bits per shot in cbit_history."""
    setup = qx.SimpleBlock(2)
    setup.h(0).h(1)
    setup.build()

    m0 = qx.MeasureBlock(qubit=0, cbit=0)
    m0.target_cbits = [0]
    m1 = qx.MeasureBlock(qubit=1, cbit=0)
    m1.target_cbits = [1]

    root = qx.CompositeBlock([setup, m0, m1], n_qubits=2)
    root.n_cbits = 2
    root.build()
    flat = root.flatten()

    sim = qx.QarpSimulator()
    r = sim.run(flat, n_qubits=2, n_shots=100, seed=7)
    assert r.n_cbits == 2
    assert len(r.cbit_history) == 100
    assert all(len(reg) == 2 for reg in r.cbit_history)


def test_multi_bit_conditional_with_else_runs():
    """Multi-bit + else: the case branch markers were added to support."""
    setup = qx.SimpleBlock(2)
    setup.h(0).h(1)
    setup.build()

    m0 = qx.MeasureBlock(qubit=0, cbit=0)
    m0.target_cbits = [0]
    m1 = qx.MeasureBlock(qubit=1, cbit=0)
    m1.target_cbits = [1]

    cond = qx.conditional(
        cbits=[0, 1],
        values=[True, False],
        then=lambda b: b.x(0),
        else_=lambda b: b.z(0),
        n_qubits=1,
    )
    cond.target_cbits = [0, 1]
    cond.target_qubits = [0]

    root = qx.CompositeBlock([setup, m0, m1, cond], n_qubits=2)
    root.n_cbits = 2
    root.build()
    flat = root.flatten()

    sim = qx.QarpSimulator()
    r = sim.run(flat, n_qubits=2, n_shots=400, seed=11)
    # Doesn't throw, returns the right shapes.
    assert r.n_cbits == 2
    assert len(r.cbit_history) == 400
