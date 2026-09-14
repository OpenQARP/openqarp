"""Tests for PennylaneAbsorber."""

import math

import numpy as np
import pytest

import qarpx as qx

pennylane = pytest.importorskip("pennylane")

from qarp.absorb import PennylaneAbsorber
from qarp.blocks import SimpleBlock
from qarp.emit import PennylaneEmitter


@pytest.fixture
def emitter() -> PennylaneEmitter:
    return PennylaneEmitter()


@pytest.fixture
def absorber() -> PennylaneAbsorber:
    return PennylaneAbsorber()


def _qarpx_unitary(block) -> np.ndarray:
    block.build()
    sim = qx.QarpSimulator()
    try:
        return np.array(sim.unitary_matrix(block.flatten(), block.n_qubits))
    except RuntimeError as e:
        pytest.skip(f"QarpSimulator does not support gate in circuit: {e}")


def test_round_trip_basic(emitter, absorber):
    b = SimpleBlock(2)
    b.h(0).cx(0, 1).rz(1, math.pi / 4)
    b.build()

    tape = emitter.emit(b.flatten(), 2)
    absorbed = absorber.absorb(tape)

    assert absorbed == b
    assert np.allclose(_qarpx_unitary(b), _qarpx_unitary(absorbed), atol=1e-10)


def test_round_trip_1q(emitter, absorber):
    b = SimpleBlock(1)
    b.h(0).rx(0, math.pi / 3).s(0)
    b.build()

    tape = emitter.emit(b.flatten(), 1)
    absorbed = absorber.absorb(tape)

    assert absorbed == b
    assert np.allclose(_qarpx_unitary(b), _qarpx_unitary(absorbed), atol=1e-10)


def test_round_trip_gateset_expansion(emitter, absorber):
    b = SimpleBlock(2)
    b.sx(0).sxdg(1).id(0).ch(0, 1).cs(0, 1).csdg(1, 0).csx(0, 1).csxdg(1, 0)
    b.build()
    tape = emitter.emit(b.flatten(), 2)
    absorbed = absorber.absorb(tape)
    assert absorbed == b
    assert np.allclose(_qarpx_unitary(b), _qarpx_unitary(absorbed), atol=1e-10)


def test_adjoint_sdg_tdg(emitter, absorber):
    b = SimpleBlock(1)
    b.sdg(0).tdg(0)
    b.build()

    tape = emitter.emit(b.flatten(), 1)
    absorbed = absorber.absorb(tape)

    assert absorbed == b
    assert np.allclose(_qarpx_unitary(b), _qarpx_unitary(absorbed), atol=1e-10)


def test_iswap_absorbed(emitter, absorber):
    b = SimpleBlock(2)
    b.iswap(0, 1)
    b.build()

    tape = emitter.emit(b.flatten(), 2)
    absorbed = absorber.absorb(tape)

    assert absorbed == b
    assert np.allclose(_qarpx_unitary(b), _qarpx_unitary(absorbed), atol=1e-10)


def test_ising_gates_absorbed(emitter, absorber):
    for builder in [
        lambda b: b.rzz(0, 1, math.pi / 4),
        lambda b: b.rxx(0, 1, math.pi / 4),
        lambda b: b.ryy(0, 1, math.pi / 4),
    ]:
        b = SimpleBlock(2)
        builder(b)
        b.build()
        tape = emitter.emit(b.flatten(), 2)
        absorbed = absorber.absorb(tape)
        assert absorbed == b
        assert np.allclose(_qarpx_unitary(b), _qarpx_unitary(absorbed), atol=1e-10)


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
def test_parametric_2q_round_trip(gate, builder, emitter, absorber):
    b = SimpleBlock(2)
    builder(b)
    b.build()
    tape = emitter.emit(b.flatten(), 2)
    absorbed = absorber.absorb(tape)
    assert absorbed == b, f"Round-trip block mismatch for {gate}"
    assert np.allclose(_qarpx_unitary(b), _qarpx_unitary(absorbed), atol=1e-10), (
        f"Round-trip unitary mismatch for {gate}"
    )


# ── Measurement / mid-circuit measurement round-trip ──────────────────────────


def test_round_trip_measure_basic(emitter, absorber):
    b = SimpleBlock(1)
    b.h(0)
    b.measure(0, 0)
    b.build()
    tape = emitter.emit(b.flatten(), 1)
    absorbed = absorber.absorb(tape)
    assert absorbed == b


def test_round_trip_mid_circuit_measure(emitter, absorber):
    """PennyLane ties measurement to the wire only — cbit must equal qubit
    on both sides of the round-trip."""
    b = SimpleBlock(1)
    b.x(0)
    b.measure(0, 0)
    b.x(0)
    b.measure(0, 0)
    b.build()
    tape = emitter.emit(b.flatten(), 1)
    absorbed = absorber.absorb(tape)
    assert absorbed == b
    measures = [
        (list(c.qubits), list(c.cbits)) for c in absorbed.flatten() if c.gate == qx.GateType.Measure
    ]
    assert measures == [([0], [0]), ([0], [0])]


# ── Composite (flattened) blocks ───────────────────────────────────────────────


def test_round_trip_composite_block(emitter, absorber):
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
    absorbed = absorber.absorb(tape)
    assert absorbed == comp
    assert np.allclose(_qarpx_unitary(comp), _qarpx_unitary(absorbed), atol=1e-10)


def test_gphase_absorbed_preserves_unitary():
    """qml.GlobalPhase(φ) = e^{-iφ}; the absorbed block must reproduce that phase
    (the absorber negates to QARPx's e^{+iθ} convention, §7). Built directly."""
    import pennylane as qml

    tape = qml.tape.QuantumScript([qml.GlobalPhase(0.5), qml.Hadamard(0)])
    block = PennylaneAbsorber().absorb(tape)
    U_block = _qarpx_unitary(block)
    U_pl = qml.matrix(tape, wire_order=[0])
    assert np.allclose(U_block, U_pl, atol=1e-10)


@pytest.mark.parametrize("n", [3, 4])
def test_mcz_absorbed_preserves_unitary(n):
    """A PennyLane multi-controlled Z (CCZ for n=3, Controlled for n>=4) must
    absorb to a block with the same unitary. Built directly, not via our emitter."""
    import pennylane as qml

    tape = qml.tape.QuantumScript([qml.ctrl(qml.PauliZ(wires=n - 1), control=list(range(n - 1)))])
    block = PennylaneAbsorber().absorb(tape)
    U_block = _qarpx_unitary(block)
    U_pl = qml.matrix(tape, wire_order=list(range(n))[::-1])
    assert np.allclose(U_block, U_pl, atol=1e-10)
