"""Tests for PennylaneEmitter."""

import math

import numpy as np
import pytest

import qarpx as qx
from qarp.errors import CapabilityError

pennylane = pytest.importorskip("pennylane")

from qarp.absorb import PennylaneAbsorber
from qarp.blocks import SimpleBlock
from qarp.emit import PennylaneEmitter


@pytest.fixture
def emitter() -> PennylaneEmitter:
    return PennylaneEmitter()


from tests.test_emit.conftest import block_unitary as _qarpx_unitary


@pytest.mark.parametrize(
    "gate,builder",
    [
        ("X", lambda b: b.x(0)),
        ("Y", lambda b: b.y(0)),
        ("Z", lambda b: b.z(0)),
        ("H", lambda b: b.h(0)),
        ("S", lambda b: b.s(0)),
        ("Sdg", lambda b: b.sdg(0)),
        ("T", lambda b: b.t(0)),
        ("Tdg", lambda b: b.tdg(0)),
        ("SX", lambda b: b.sx(0)),
        ("SXdg", lambda b: b.sxdg(0)),
        ("Id", lambda b: b.id(0)),
        ("Rx", lambda b: b.rx(0, math.pi / 3)),
        ("Ry", lambda b: b.ry(0, math.pi / 4)),
        ("Rz", lambda b: b.rz(0, math.pi / 5)),
        ("P", lambda b: b.p(0, math.pi / 6)),
    ],
)
def test_1q_gate(gate, builder, emitter):
    b = SimpleBlock(1)
    builder(b)
    b.build()
    tape = emitter.emit(b.flatten(), 1)
    import pennylane as qml

    U_pl = qml.matrix(qml.tape.QuantumScript(tape.operations), wire_order=[0])
    U_qarpx = _qarpx_unitary(b)
    assert np.allclose(U_pl, U_qarpx, atol=1e-10), f"Mismatch for {gate}"


@pytest.mark.parametrize(
    "gate,builder",
    [
        ("CX", lambda b: b.cx(0, 1)),
        ("CY", lambda b: b.cy(0, 1)),
        ("CZ", lambda b: b.cz(0, 1)),
        ("SWAP", lambda b: b.swap(0, 1)),
        ("iSWAP", lambda b: b.iswap(0, 1)),
        ("iSWAPdg", lambda b: b.iswapdg(0, 1)),
        ("CH", lambda b: b.ch(0, 1)),
        ("CS", lambda b: b.cs(0, 1)),
        ("CSdg", lambda b: b.csdg(0, 1)),
        ("CSX", lambda b: b.csx(0, 1)),
        ("CSXdg", lambda b: b.csxdg(0, 1)),
        ("RZZ", lambda b: b.rzz(0, 1, math.pi / 4)),
        ("RXX", lambda b: b.rxx(0, 1, math.pi / 4)),
        ("RYY", lambda b: b.ryy(0, 1, math.pi / 4)),
    ],
)
def test_2q_gate(gate, builder, emitter):
    b = SimpleBlock(2)
    builder(b)
    b.build()
    import pennylane as qml

    tape = emitter.emit(b.flatten(), 2)
    # PennyLane's wire_order[0] is the MSB; QARPx's qubit 0 is the LSB.
    U_pl = qml.matrix(qml.tape.QuantumScript(tape.operations), wire_order=[1, 0])
    U_qarpx = _qarpx_unitary(b)
    assert np.allclose(U_pl, U_qarpx, atol=1e-10), f"Mismatch for {gate}"


@pytest.mark.parametrize(
    "gate,builder",
    [
        ("CRx", lambda b: b.crx(0, 1, math.pi / 4)),
        ("CRy", lambda b: b.cry(0, 1, math.pi / 4)),
        ("CRz", lambda b: b.crz(0, 1, math.pi / 4)),
        ("CP", lambda b: b.cp(0, 1, math.pi / 4)),
        ("CU", lambda b: b.cu(0, 1, math.pi / 4, math.pi / 5, math.pi / 6, 0.1)),
    ],
)
def test_parametric_2q_gate(gate, builder, emitter):
    b = SimpleBlock(2)
    builder(b)
    b.build()
    import pennylane as qml

    tape = emitter.emit(b.flatten(), 2)
    # PennyLane's wire_order[0] is the MSB; QARPx's qubit 0 is the LSB.
    U_pl = qml.matrix(qml.tape.QuantumScript(tape.operations), wire_order=[1, 0])
    U_qarpx = _qarpx_unitary(b)
    assert np.allclose(U_pl, U_qarpx, atol=1e-10), f"Mismatch for {gate}"


# ── Round-trip via absorber (block equality) ────────────────────────────────
# emit() + PennylaneAbsorber().absorb() must reconstruct exactly the same
# circuit (Block.__eq__), not just an equivalent unitary — covering plain
# gates, parametric gates, mid-circuit measurement, and nested
# CompositeBlocks. PennyLane has no symbolic-parameter support (see
# test_error_on_symbolic_param below), so that complexity isn't applicable.


def test_round_trip_block_equality_basic(emitter):
    b = SimpleBlock(2)
    b.h(0).cx(0, 1).rz(1, math.pi / 4)
    b.build()
    tape = emitter.emit(b.flatten(), 2)
    absorbed = PennylaneAbsorber().absorb(tape)
    assert absorbed == b


def test_round_trip_block_equality_parametric(emitter):
    b = SimpleBlock(2)
    b.crx(0, 1, math.pi / 4).rzz(0, 1, math.pi / 6)
    b.build()
    tape = emitter.emit(b.flatten(), 2)
    absorbed = PennylaneAbsorber().absorb(tape)
    assert absorbed == b


def test_round_trip_block_equality_mid_circuit_measure(emitter):
    """cbit must equal qubit on both sides — see test_error_on_distinct_cbit_measure."""
    b = SimpleBlock(1)
    b.x(0)
    b.measure(0, 0)
    b.x(0)
    b.measure(0, 0)
    b.build()
    tape = emitter.emit(b.flatten(), 1)
    absorbed = PennylaneAbsorber().absorb(tape)
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

    tape = emitter.emit(comp.flatten(), comp.n_qubits)
    absorbed = PennylaneAbsorber().absorb(tape)
    assert absorbed == comp


def test_barrier_silently_skipped(emitter):
    """Barrier has no PennyLane equivalent: the emitter drops it while preserving
    the surrounding gates and their order."""
    cmds = [
        qx.Command(qx.GateType.H, 0),
        qx.Command(qx.GateType.Barrier, 0),
        qx.Command(qx.GateType.X, 0),
    ]
    tape = emitter.emit(cmds, 1)
    assert [type(op).__name__ for op in tape.operations] == ["Hadamard", "PauliX"]


def test_error_on_symbolic_param(emitter):
    b = SimpleBlock(1)
    b.rx(0, qx.Param.symbol("theta"))
    b.build()
    with pytest.raises(CapabilityError, match="symbolic"):
        emitter.emit(b.flatten(), 1)


def test_error_on_custom_gate(emitter):
    cmd = qx.Command()
    cmd.gate = qx.GateType.Custom
    with pytest.raises(CapabilityError, match="Custom"):
        emitter.emit([cmd], 1)


def test_error_on_reset(emitter):
    b = SimpleBlock(1)
    b.reset(0)
    b.build()
    with pytest.raises(CapabilityError, match="Reset"):
        emitter.emit(b.flatten(), 1)


def test_import_error_with_hint(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "pennylane", None)
    with pytest.raises(ImportError, match="pip install"):
        PennylaneEmitter().emit([], 1)


# ── Measurement / mid-circuit measurement ──────────────────────────────────────


def test_measure_basic(emitter):
    b = SimpleBlock(1)
    b.h(0)
    b.measure(0, 0)
    b.build()
    tape = emitter.emit(b.flatten(), 1)
    names = [type(op).__name__ for op in tape.operations]
    assert "MidMeasure" in names


def test_error_on_distinct_cbit_measure(emitter):
    """PennyLane has no classical register — qml.measure(wire) ties the
    MeasurementValue to the wire only, so cbit must equal the qubit index."""
    b = SimpleBlock(2)
    b.h(0)
    b.measure(0, 1)
    b.build()
    with pytest.raises(CapabilityError, match="classical"):
        emitter.emit(b.flatten(), 2)


def test_mid_circuit_measure(emitter):
    b = SimpleBlock(1)
    b.x(0)
    b.measure(0, 0)
    b.x(0)
    b.measure(0, 0)
    b.build()
    tape = emitter.emit(b.flatten(), 1)
    n_measures = sum(1 for op in tape.operations if type(op).__name__ == "MidMeasure")
    assert n_measures == 2


def test_measurement_endianness_matches_qarpx(emitter):
    """qml.sample(wires=q) reads one qubit at a time, so there's no
    aggregate-string convention here (unlike Qiskit) — but verify it agrees
    with qarp.endianness's LSB-first per-qubit convention for every qubit.
    """
    import pennylane as qml

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

    tape = emitter.emit(b.flatten(), n)
    dev = qml.device("default.qubit", wires=n)

    @qml.set_shots(1)
    @qml.qnode(dev)
    def circuit():
        for op in tape.operations:
            qml.apply(op)
        return [qml.sample(wires=q) for q in range(n)]

    pennylane_bits = [int(np.asarray(x).reshape(-1)[0]) for x in circuit()]

    assert qarpx_bits == pennylane_bits == [1, 0, 1, 0]


# ── Composite (flattened) blocks ───────────────────────────────────────────────


def test_composite_block_flatten(emitter):
    import pennylane as qml

    from qarp.blocks import CompositeBlock

    sub1 = SimpleBlock(2)
    sub1.h(0).cx(0, 1)
    sub1.build()

    sub2 = SimpleBlock(2)
    sub2.rz(1, math.pi / 4).cz(0, 1)
    sub2.build()

    comp = CompositeBlock([sub1, sub2], 2)
    comp.build()

    tape = emitter.emit(comp.flatten(), comp.n_qubits)
    U_pl = qml.matrix(qml.tape.QuantumScript(tape.operations), wire_order=[1, 0])
    assert np.allclose(U_pl, _qarpx_unitary(comp), atol=1e-10)


def test_gphase_unitary_matches_qarpx(emitter):
    """QARPx GPhase(θ) = e^{+iθ}; qml.GlobalPhase(φ) = e^{-iφ}. The emitter must
    negate so the (observable) global phase matches (conventions §7). Compared on
    the full 1-qubit unitary, where the global phase is visible."""
    import pennylane as qml

    b = SimpleBlock(1)
    b.gphase(0.5)
    b.h(0)
    b.build()
    tape = emitter.emit(b.flatten(), 1)
    U_pl = qml.matrix(qml.tape.QuantumScript(tape.operations), wire_order=[0])
    assert np.allclose(U_pl, _qarpx_unitary(b), atol=1e-10)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_mcz_unitary_matches_qarpx(emitter, n):
    """Multi-controlled Z (conventions §5). The emitter must control an
    *instantiated* qml.PauliZ(wires=target) — the bare class yields an empty op.
    Covers CZ (n=2), CCZ (n=3), and the generic Controlled path (n=4)."""
    import pennylane as qml

    b = SimpleBlock(n)
    b.mcz(list(range(n)))
    b.build()
    tape = emitter.emit(b.flatten(), n)
    U_pl = qml.matrix(qml.tape.QuantumScript(tape.operations), wire_order=list(range(n))[::-1])
    assert np.allclose(U_pl, _qarpx_unitary(b), atol=1e-10)
