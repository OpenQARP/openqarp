"""Tests for QulacsEmitter.

Guards: all tests are skipped when qulacs is not installed.

Key coverage:
  - Angle negation correctness for Rx/Ry/Rz
  - DenseMatrix fallback gates (P, U, iSWAP, …) via unitary comparison
  - RuntimeError on symbolic params, conditionals, Reset, MCZ
"""

import math

import numpy as np
import pytest

import qarpx as qx
from qarp.errors import CapabilityError

qulacs = pytest.importorskip("qulacs")

from qarp.absorb import QulacsAbsorber
from qarp.blocks import SimpleBlock
from qarp.emit import QulacsEmitter


@pytest.fixture
def emitter() -> QulacsEmitter:
    return QulacsEmitter()


from tests.test_emit.conftest import block_unitary as _qarpx_unitary
from tests.test_emit.conftest import qulacs_unitary as _qulacs_unitary


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
    ],
)
def test_1q_clifford(gate, builder, emitter):
    b = SimpleBlock(1)
    builder(b)
    b.build()
    qc = emitter.emit(b.flatten(), 1)
    assert np.allclose(_qulacs_unitary(qc, 1), _qarpx_unitary(b), atol=1e-10), (
        f"Mismatch for {gate}"
    )


def test_rx_angle_negation(emitter):
    """Rx(π/3) in QARPx must produce the correct unitary via Qulacs -θ negation."""
    theta = math.pi / 3
    b = SimpleBlock(1)
    b.rx(0, theta)
    b.build()
    qc = emitter.emit(b.flatten(), 1)
    assert np.allclose(_qulacs_unitary(qc, 1), _qarpx_unitary(b), atol=1e-10)


def test_ry_rz_angle_negation(emitter):
    for name, builder in [
        ("ry", lambda b: b.ry(0, math.pi / 4)),
        ("rz", lambda b: b.rz(0, math.pi / 5)),
    ]:
        b = SimpleBlock(1)
        builder(b)
        b.build()
        qc = emitter.emit(b.flatten(), 1)
        assert np.allclose(_qulacs_unitary(qc, 1), _qarpx_unitary(b), atol=1e-10), (
            f"Angle mismatch for {name}"
        )


def test_cx_cz_swap(emitter):
    for name, builder in [
        ("CX", lambda b: b.cx(0, 1)),
        ("CZ", lambda b: b.cz(0, 1)),
        ("SWAP", lambda b: b.swap(0, 1)),
    ]:
        b = SimpleBlock(2)
        builder(b)
        b.build()
        qc = emitter.emit(b.flatten(), 2)
        assert np.allclose(_qulacs_unitary(qc, 2), _qarpx_unitary(b), atol=1e-10), (
            f"Mismatch for {name}"
        )


def test_controlled_clifford_singles_via_dense_matrix(emitter):
    # Control on the higher qubit too: catches a transposed dense-matrix order.
    for name, builder in [
        ("CH", lambda b: b.ch(0, 1)),
        ("CS", lambda b: b.cs(0, 1)),
        ("CSdg", lambda b: b.csdg(0, 1)),
        ("CSX", lambda b: b.csx(0, 1)),
        ("CSXdg", lambda b: b.csxdg(0, 1)),
        ("CSX_rev", lambda b: b.csx(1, 0)),
    ]:
        b = SimpleBlock(2)
        builder(b)
        b.build()
        qc = emitter.emit(b.flatten(), 2)
        assert np.allclose(_qulacs_unitary(qc, 2), _qarpx_unitary(b), atol=1e-10), (
            f"Mismatch for {name}"
        )


def test_iswap_dense_matrix(emitter):
    b = SimpleBlock(2)
    b.iswap(0, 1)
    b.build()
    qc = emitter.emit(b.flatten(), 2)
    assert np.allclose(_qulacs_unitary(qc, 2), _qarpx_unitary(b), atol=1e-10)


def test_p_gate_dense_matrix(emitter):
    b = SimpleBlock(1)
    b.p(0, math.pi / 3)
    b.build()
    qc = emitter.emit(b.flatten(), 1)
    assert np.allclose(_qulacs_unitary(qc, 1), _qarpx_unitary(b), atol=1e-10)


# ── Parametric DenseMatrix-fallback gates ──────────────────────────────────────
# Qulacs has no native gate for these — they go through DenseMatrix, which is
# where a controlled-gate qubit-ordering slip (control vs. target swapped in
# the local 2x2 sub-block) hides undetected for the asymmetric ones.


@pytest.mark.parametrize(
    "gate,n,builder",
    [
        ("CRx", 2, lambda b: b.crx(0, 1, math.pi / 4)),
        ("CRy", 2, lambda b: b.cry(0, 1, math.pi / 4)),
        ("CRz", 2, lambda b: b.crz(0, 1, math.pi / 4)),
        ("CP", 2, lambda b: b.cp(0, 1, math.pi / 4)),
        ("RZZ", 2, lambda b: b.rzz(0, 1, math.pi / 4)),
        ("RXX", 2, lambda b: b.rxx(0, 1, math.pi / 4)),
        ("RYY", 2, lambda b: b.ryy(0, 1, math.pi / 4)),
        ("CU", 2, lambda b: b.cu(0, 1, math.pi / 4, math.pi / 5, math.pi / 6, 0.1)),
    ],
)
def test_parametric_2q_dense_matrix(gate, n, builder, emitter):
    b = SimpleBlock(n)
    builder(b)
    b.build()
    qc = emitter.emit(b.flatten(), n)
    assert np.allclose(_qulacs_unitary(qc, n), _qarpx_unitary(b), atol=1e-10), (
        f"Unitary mismatch for {gate}"
    )


def test_ccx_cswap_dense_matrix(emitter):
    for name, builder in [("CCX", lambda b: b.ccx(0, 1, 2)), ("CSWAP", lambda b: b.cswap(0, 1, 2))]:
        b = SimpleBlock(3)
        builder(b)
        b.build()
        qc = emitter.emit(b.flatten(), 3)
        assert np.allclose(_qulacs_unitary(qc, 3), _qarpx_unitary(b), atol=1e-10), (
            f"Unitary mismatch for {name}"
        )


# ── Round-trip via absorber (block equality) ────────────────────────────────
# emit() + QulacsAbsorber().absorb() must reconstruct exactly the same
# circuit (Block.__eq__), not just an equivalent unitary — covering plain
# gates, parametric (incl. DenseMatrix-fallback) gates, mid-circuit
# measurement, and nested CompositeBlocks. Qulacs has no symbolic-parameter
# support (see test_error_on_symbolic_param below), so that complexity isn't
# applicable here.


def test_round_trip_block_equality_basic(emitter):
    b = SimpleBlock(2)
    b.h(0).cx(0, 1).s(0).t(1)
    b.build()
    qc = emitter.emit(b.flatten(), 2)
    absorbed = QulacsAbsorber().absorb(qc)
    assert absorbed == b


def test_round_trip_block_equality_parametric(emitter):
    b = SimpleBlock(2)
    b.crx(0, 1, math.pi / 4).rzz(0, 1, math.pi / 6)
    b.build()
    qc = emitter.emit(b.flatten(), 2)
    absorbed = QulacsAbsorber().absorb(qc)
    assert absorbed == b


def test_round_trip_block_equality_mid_circuit_measure(emitter):
    b = SimpleBlock(1)
    b.x(0)
    b.measure(0, 0)
    b.x(0)
    b.measure(0, 1)
    b.build()
    qc = emitter.emit(b.flatten(), 1)
    absorbed = QulacsAbsorber().absorb(qc)
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

    qc = emitter.emit(comp.flatten(), comp.n_qubits)
    absorbed = QulacsAbsorber().absorb(qc)
    assert absorbed == comp


# ── Error cases ───────────────────────────────────────────────────────────────


def test_error_on_symbolic_param(emitter):
    b = SimpleBlock(1)
    b.rx(0, qx.Param.symbol("theta"))
    b.build()
    with pytest.raises(CapabilityError, match="symbolic"):
        emitter.emit(b.flatten(), 1)


def test_error_on_reset(emitter):
    b = SimpleBlock(1)
    b.reset(0)
    b.build()
    with pytest.raises(CapabilityError, match="Reset"):
        emitter.emit(b.flatten(), 1)


def test_error_on_mcz(emitter):
    b = SimpleBlock(3)
    b.mcz([0, 1, 2])
    b.build()
    with pytest.raises(CapabilityError, match="MCZ"):
        emitter.emit(b.flatten(), 3)


def test_error_on_conditional(emitter):
    from qarp.blocks import ConditionalBlock

    body = SimpleBlock(1)
    body.x(0)
    body.build()
    cond = ConditionalBlock(cbits=[0], values=[True], then_body=body)
    cond.build()
    with pytest.raises(CapabilityError, match="conditional"):
        emitter.emit(cond.flatten(), cond.n_qubits)


def test_import_error_with_hint(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "qulacs", None)
    with pytest.raises(ImportError, match="pip install"):
        QulacsEmitter().emit([], 1)


# ── Measurement / mid-circuit measurement ──────────────────────────────────────
# qulacs.QuantumCircuit has no add_measurement(); the gate must be built via
# qulacs.gate.Measurement(qubit, cbit) + add_gate(). These tests execute the
# circuit (not just construct it): construction alone does not catch the
# emitter crashing on a Measure command.


def test_measure_basic_executes(emitter):
    b = SimpleBlock(1)
    b.h(0)
    b.measure(0, 0)
    b.build()
    qc = emitter.emit(b.flatten(), 1)
    # H + Measure = 2 gates; a dropped measurement would leave only 1.
    assert qc.get_gate_count() == 2
    state = qulacs.QuantumState(1)
    qc.update_quantum_state(state)  # executes without raising


def test_measure_distinct_cbit_deterministic(emitter):
    """qubit and cbit may differ; classical_register_address must be honored.

    X|0> -> |1> measured deterministically as 1; verified via the gate's own
    to_json() (the only way qulacs exposes the classical register address)."""
    import json

    b = SimpleBlock(2)
    b.x(0)
    b.measure(0, 1)
    b.build()
    qc = emitter.emit(b.flatten(), 2)
    gate = qc.get_gate(1)
    info = json.loads(gate.to_json())
    assert info["classical_register_address"] == "1"


def test_mid_circuit_measure_deterministic(emitter):
    """X; measure(->c0); X; measure(->c1): qubit flips between measurements,
    so both classical outcomes are deterministic and must differ."""
    b = SimpleBlock(1)
    b.x(0)
    b.measure(0, 0)
    b.x(0)
    b.measure(0, 1)
    b.build()
    qc = emitter.emit(b.flatten(), 1)
    state = qulacs.QuantumState(1)
    state.set_zero_state()
    qc.update_quantum_state(state)
    # The classical outcomes must differ (c0=1 while the qubit is |1>, c1=0 after
    # the second X) — a silently dropped measurement would not record these.
    assert state.get_classical_value(0) == 1
    assert state.get_classical_value(1) == 0
    # And after X;X the qubit is back to |0> with certainty.
    assert np.allclose(state.get_vector(), [1, 0], atol=1e-9)


def test_measurement_endianness_matches_qarpx(emitter):
    """Qulacs's get_classical_value(cbit) reads one classical bit at a time,
    so there's no aggregate-string convention to reconcile here (unlike
    Qiskit) — but verify it agrees with qarp.endianness's LSB-first
    per-qubit convention for every qubit, not just qubit 0.
    """
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

    qc = emitter.emit(b.flatten(), n)
    state = qulacs.QuantumState(n)
    state.set_zero_state()
    qc.update_quantum_state(state)
    qulacs_bits = [state.get_classical_value(c) for c in range(n)]

    assert qarpx_bits == qulacs_bits == [1, 0, 1, 0]


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
    cmds = comp.flatten()

    qc = emitter.emit(cmds, comp.n_qubits)
    assert np.allclose(_qulacs_unitary(qc, comp.n_qubits), _qarpx_unitary(comp), atol=1e-10)


# ── Asymmetric 2q gates: control/target ordering ─────────────────────────────


def test_cy_unitary_matches_qarpx(emitter):
    """CY is asymmetric under control/target swap (conventions §3.1); the emitted
    qulacs DenseMatrix must place the control on the correct qubit. Compared
    against the QARPx unitary."""
    b = SimpleBlock(2)
    b.cy(0, 1)
    b.build()
    qc = emitter.emit(b.flatten(), 2)
    assert np.allclose(_qulacs_unitary(qc, 2), _qarpx_unitary(b), atol=1e-10)


def test_ecr_unitary_matches_convention(emitter):
    """ECR is non-symmetric (§3.2) and not simulable by QarpSimulator, so pin the
    emitted qulacs unitary against the analytic ECR(0,1) matrix in the LSB basis
    {|00>, |01>, |10>, |11>}."""
    b = SimpleBlock(2)
    b.ecr(0, 1)
    b.build()
    qc = emitter.emit(b.flatten(), 2)
    s = 1.0 / math.sqrt(2.0)
    ecr01 = s * np.array(
        [[0, 1, 0, 1j], [1, 0, -1j, 0], [0, 1j, 0, 1], [-1j, 0, 1, 0]], dtype=complex
    )
    assert np.allclose(_qulacs_unitary(qc, 2), ecr01, atol=1e-10)
