"""Tests for PytketEmitter.

Guards: all tests are skipped when pytket is not installed.

Key coverage:
  - Half-turn angle conversion correctness (verified via pytket circuit unitary)
  - Sign convention for ZZPhase/XXPhase/YYPhase (θ vs -θ/π)
  - Symbolic parameters via sympy
  - CapabilityError on CU with non-zero gamma (validate() rejection)
"""

import math

import numpy as np
import pytest

import qarpx as qx
from qarp.errors import CapabilityError

pytket = pytest.importorskip("pytket")

from qarp.absorb import PytketAbsorber
from qarp.blocks import SimpleBlock
from qarp.emit import PytketEmitter


@pytest.fixture
def emitter() -> PytketEmitter:
    return PytketEmitter()


from tests.test_emit.conftest import block_unitary as _qarpx_unitary
from tests.test_emit.conftest import tket_unitary as _tket_unitary


@pytest.mark.parametrize(
    "gate,builder",
    [
        ("Rx", lambda b: b.rx(0, math.pi / 3)),
        ("Ry", lambda b: b.ry(0, math.pi / 4)),
        ("Rz", lambda b: b.rz(0, math.pi / 5)),
        ("P", lambda b: b.p(0, math.pi / 6)),
        ("H", lambda b: b.h(0)),
        ("SX", lambda b: b.sx(0)),
        ("SXdg", lambda b: b.sxdg(0)),
        ("Id", lambda b: b.id(0)),
        ("CX", lambda b: b.cx(0, 1) or None),
        ("CH", lambda b: b.ch(0, 1)),
        ("CS", lambda b: b.cs(0, 1)),
        ("CSdg", lambda b: b.csdg(0, 1)),
        ("CSX", lambda b: b.csx(0, 1)),
        ("CSXdg", lambda b: b.csxdg(0, 1)),
    ],
)
def test_gate_unitary(gate, builder, emitter):
    n = 2 if gate in {"CX", "CH", "CS", "CSdg", "CSX", "CSXdg"} else 1
    b = SimpleBlock(n)
    builder(b)
    b.build()
    cmds = b.flatten()
    circ = emitter.emit(cmds, n)
    U_tket = _tket_unitary(circ)
    U_qarpx = _qarpx_unitary(b)
    assert np.allclose(U_tket, U_qarpx, atol=1e-10), f"Unitary mismatch for {gate}"


def test_rzz_sign_convention(emitter):
    """ZZPhase(t) in TKET uses opposite sign: verify -θ/π conversion."""
    theta = math.pi / 4
    b = SimpleBlock(2)
    b.rzz(0, 1, theta)
    b.build()
    circ = emitter.emit(b.flatten(), 2)
    U_tket = _tket_unitary(circ)
    U_qarpx = _qarpx_unitary(b)
    assert np.allclose(U_tket, U_qarpx, atol=1e-10)


def test_rxx_ryy_sign_convention(emitter):
    theta = math.pi / 3
    for name, builder in [
        ("rxx", lambda b: b.rxx(0, 1, theta)),
        ("ryy", lambda b: b.ryy(0, 1, theta)),
    ]:
        b = SimpleBlock(2)
        builder(b)
        b.build()
        circ = emitter.emit(b.flatten(), 2)
        assert np.allclose(_tket_unitary(circ), _qarpx_unitary(b), atol=1e-10), (
            f"Sign mismatch for {name}"
        )


def test_symbolic_param(emitter):
    pytest.importorskip("sympy")
    b = SimpleBlock(1)
    b.rx(0, qx.Param.symbol("theta"))
    b.build()
    circ = emitter.emit(b.flatten(), 1)
    # Circuit should have a free symbol
    assert len(circ.free_symbols()) > 0


def test_symbolic_param_unitary_after_binding(emitter):
    """Binding the free symbol to a concrete value must reproduce the same
    unitary as building the gate directly with that value."""
    pytest.importorskip("sympy")
    b = SimpleBlock(1)
    b.rx(0, qx.Param.symbol("theta"))
    b.build()
    circ = emitter.emit(b.flatten(), 1)
    sym = list(circ.free_symbols())[0]
    theta_val = math.pi / 3
    circ.symbol_substitution({sym: theta_val})

    concrete_b = SimpleBlock(1)
    concrete_b.rx(0, theta_val)
    concrete_b.build()
    assert np.allclose(_tket_unitary(circ), _qarpx_unitary(concrete_b), atol=1e-10)


@pytest.mark.parametrize(
    "gate,builder",
    [
        ("CRx", lambda b: b.crx(0, 1, math.pi / 4)),
        ("CRy", lambda b: b.cry(0, 1, math.pi / 4)),
        ("CRz", lambda b: b.crz(0, 1, math.pi / 4)),
        ("CP", lambda b: b.cp(0, 1, math.pi / 4)),
        ("CU", lambda b: b.cu(0, 1, math.pi / 4, math.pi / 5, math.pi / 6, 0.0)),
    ],
)
def test_parametric_2q_gate_unitary(gate, builder, emitter):
    b = SimpleBlock(2)
    builder(b)
    b.build()
    circ = emitter.emit(b.flatten(), 2)
    assert np.allclose(_tket_unitary(circ), _qarpx_unitary(b), atol=1e-10), (
        f"Unitary mismatch for {gate}"
    )


# ── Round-trip via absorber (block equality) ────────────────────────────────
# emit() + PytketAbsorber().absorb() must reconstruct exactly the same
# circuit (Block.__eq__), not just an equivalent unitary — covering plain
# gates, parametric (half-turn-converted) gates, symbols, mid-circuit
# measurement, and nested CompositeBlocks.


def test_round_trip_block_equality_basic(emitter):
    b = SimpleBlock(2)
    b.h(0).cx(0, 1).rz(1, math.pi / 4)
    b.build()
    circ = emitter.emit(b.flatten(), 2)
    absorbed = PytketAbsorber().absorb(circ)
    assert absorbed == b


def test_round_trip_block_equality_parametric(emitter):
    b = SimpleBlock(2)
    b.crx(0, 1, math.pi / 4).rzz(0, 1, math.pi / 6)
    b.build()
    circ = emitter.emit(b.flatten(), 2)
    absorbed = PytketAbsorber().absorb(circ)
    assert absorbed == b


def test_round_trip_block_equality_symbolic(emitter):
    pytest.importorskip("sympy")
    b = SimpleBlock(1)
    b.rx(0, qx.Param.symbol("theta"))
    b.build()
    circ = emitter.emit(b.flatten(), 1)
    absorbed = PytketAbsorber().absorb(circ)
    assert absorbed == b


def test_round_trip_block_equality_mid_circuit_measure(emitter):
    b = SimpleBlock(1)
    b.x(0)
    b.measure(0, 0)
    b.x(0)
    b.measure(0, 1)
    b.build()
    circ = emitter.emit(b.flatten(), 1)
    absorbed = PytketAbsorber().absorb(circ)
    assert absorbed == b


def test_round_trip_block_equality_composite(emitter):
    from qarp.blocks import CompositeBlock

    sub1 = SimpleBlock(2)
    sub1.h(0).cx(0, 1)
    sub1.build()
    sub2 = SimpleBlock(2)
    sub2.rz(1, math.pi / 4).cz(0, 1)
    sub2.build()
    comp = CompositeBlock([sub1, sub2], 2)
    comp.build()

    circ = emitter.emit(comp.flatten(), comp.n_qubits)
    absorbed = PytketAbsorber().absorb(circ)
    assert absorbed == comp


def test_error_cu_with_gamma(emitter):
    b = SimpleBlock(2)
    b.cu(0, 1, math.pi / 4, math.pi / 4, math.pi / 4, 0.1)
    b.build()
    with pytest.raises(CapabilityError, match="global phase"):
        emitter.emit(b.flatten(), 2)


# ── Measurement / mid-circuit measurement ──────────────────────────────────────


def test_measure_basic(emitter):
    b = SimpleBlock(1)
    b.h(0)
    b.measure(0, 0)
    b.build()
    circ = emitter.emit(b.flatten(), 1)
    ops = [
        (cmd.op.type, [q.index[0] for q in cmd.qubits], [c.index[0] for c in cmd.bits])
        for cmd in circ
    ]
    from pytket import OpType

    assert (OpType.Measure, [0], [0]) in ops


def test_measure_distinct_cbit(emitter):
    """TKET has real classical bits — qubit and cbit need not match."""
    b = SimpleBlock(2)
    b.h(0)
    b.measure(0, 1)
    b.build()
    circ = emitter.emit(b.flatten(), 2)
    from pytket import OpType

    ops = [
        (cmd.op.type, [q.index[0] for q in cmd.qubits], [c.index[0] for c in cmd.bits])
        for cmd in circ
    ]
    assert (OpType.Measure, [0], [1]) in ops


def test_mid_circuit_measure(emitter):
    b = SimpleBlock(1)
    b.x(0)
    b.measure(0, 0)
    b.x(0)
    b.measure(0, 1)
    b.build()
    circ = emitter.emit(b.flatten(), b.n_qubits)
    from pytket import OpType

    n_measures = sum(1 for cmd in circ if cmd.op.type == OpType.Measure)
    assert n_measures == 2


# ── Composite (flattened) blocks ───────────────────────────────────────────────


def test_composite_block_flatten(emitter):
    from qarp.blocks import CompositeBlock

    sub1 = SimpleBlock(2)
    sub1.h(0).cx(0, 1)
    sub1.build()

    sub2 = SimpleBlock(2)
    sub2.rz(1, math.pi / 4).cz(0, 1)
    sub2.build()

    comp = CompositeBlock([sub1, sub2], 2)
    comp.build()

    circ = emitter.emit(comp.flatten(), comp.n_qubits)
    assert np.allclose(_tket_unitary(circ), _qarpx_unitary(comp), atol=1e-10)


def test_measurement_endianness_matches_qarpx(emitter):
    """TKET's get_shots()/get_counts() return per-cbit values directly (no
    aggregate-string convention to reconcile, unlike Qiskit) — verify they
    agree with qarp.endianness's LSB-first per-qubit convention. Requires an
    execution backend (pytket-qiskit's AerBackend); skipped if unavailable.
    """
    pytest.importorskip("pytket.extensions.qiskit")
    from pytket.extensions.qiskit import AerBackend

    from qarp.endianness import label_to_bits

    n = 4
    b = SimpleBlock(n)
    b.x(0)
    b.x(2)
    for q in range(n):
        b.measure(q, q)
    b.build()

    sim = qx.QarpSimulator()
    result = sim.run(b.flatten(), n, 1)
    qarpx_bits = label_to_bits(list(result.counts.keys())[0], n)

    circ = emitter.emit(b.flatten(), n)
    backend = AerBackend()
    compiled = backend.get_compiled_circuit(circ)
    shots = backend.run_circuit(compiled, n_shots=1).get_shots()
    tket_bits = [int(v) for v in shots[0]]

    assert qarpx_bits == tket_bits == [1, 0, 1, 0]


def test_error_custom_gate(emitter):
    cmd = qx.Command()
    cmd.gate = qx.GateType.Custom
    with pytest.raises(CapabilityError, match="Custom"):
        emitter.emit([cmd], 1)


def test_import_error_with_hint(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "pytket", None)
    with pytest.raises(ImportError, match="pip install"):
        PytketEmitter().emit([], 1)


@pytest.mark.parametrize("kind", ["reset", "measure"])
def test_conditional_measure_reset_keep_condition(emitter, kind):
    """A conditional Reset/Measure must stay conditional in TKET. Measure/Reset
    are non-unitary, so there is no matrix to compare — the physical property
    under test is that the emitted op is wrapped in OpType.Conditional (the
    branch condition is forwarded, not dropped)."""
    from pytket.circuit import OpType

    from qarp.blocks import ConditionalBlock, MeasureBlock, ResetBlock

    body = ResetBlock(0) if kind == "reset" else MeasureBlock(0, 0)
    body.build()
    cond = ConditionalBlock(cbits=[0], values=[True], then_body=body)
    cond.build()
    circ = emitter.emit(cond.flatten(), cond.n_qubits)
    ops = [c.op.type for c in circ.get_commands()]
    assert OpType.Conditional in ops, f"{kind} lost its classical condition: {ops}"


# ── Parameter bridge linearity (pipeline_hardening_plan.md P1.14) ─────────


def test_nonlinear_single_symbol_param_is_refused(emitter):
    """t*t has one free symbol but is not c*x + d; the bridge probes a third
    point and refuses instead of emitting a wrong linear form."""
    b = SimpleBlock(1)
    b.rx(0, qx.Param.symbol("t") * qx.Param.symbol("t"))
    b.build()
    assert "linear" in (b.can_emit_to("pytket") or "")
    with pytest.raises(CapabilityError, match="linear"):
        emitter.emit(b.flatten(), 1)


def test_linear_param_with_offset_binds_to_analytic_rz(emitter, sdk_unitary):
    """Rz(2t + 0.5) at t = 0.3 is Rz(1.1) = diag(e^{-0.55i}, e^{0.55i}), through
    the half-turn scaling both ways."""
    b = SimpleBlock(1)
    b.rz(0, qx.Param.linear(2.0, "t", 0.5))
    b.build()
    circ = emitter.emit(b.flatten(), 1)
    sym = list(circ.free_symbols())[0]
    circ.symbol_substitution({sym: 0.3})
    expected = np.diag([np.exp(-0.55j), np.exp(0.55j)])
    np.testing.assert_allclose(sdk_unitary["pytket"](circ, 1), expected, atol=1e-12)
