"""Tests for QiskitEmitter.

Guards: all tests are skipped when qiskit is not installed.

Coverage:
  - Gate-level unitary comparison (emitted circuit == QARPx simulator)
  - Symbolic parameters round-trip
  - CapabilityError on Custom gate (validate() rejection)
  - ImportError when qiskit is absent (mocked)
"""

import math

import numpy as np
import pytest

import qarpx as qx
from qarp.errors import CapabilityError

qiskit = pytest.importorskip("qiskit")

from qarp.absorb import QiskitAbsorber
from qarp.blocks import SimpleBlock
from qarp.emit import QiskitEmitter

# ── Fixture ───────────────────────────────────────────────────────────────────


@pytest.fixture
def emitter() -> QiskitEmitter:
    return QiskitEmitter()


from tests.test_emit.conftest import qarpx_unitary as _unitary_from_qarpx
from tests.test_emit.conftest import qiskit_unitary as _unitary_from_qiskit


def _block_unitary(block: SimpleBlock) -> np.ndarray:
    block.build()
    cmds = block.flatten()
    return _unitary_from_qarpx(cmds, block.n_qubits)


# ── Basic 1-qubit gate tests ──────────────────────────────────────────────────


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
def test_1q_no_param_gate(gate, builder, emitter):
    b = SimpleBlock(1)
    builder(b)
    b.build()
    cmds = b.flatten()

    qc = emitter.emit(cmds, 1)

    U_qiskit = _unitary_from_qiskit(qc)
    U_qarpx = _unitary_from_qarpx(cmds, 1)
    assert np.allclose(U_qiskit, U_qarpx, atol=1e-10), f"Unitary mismatch for {gate}"


@pytest.mark.parametrize(
    "gate,builder",
    [
        ("Rx", lambda b: b.rx(0, math.pi / 3)),
        ("Ry", lambda b: b.ry(0, math.pi / 4)),
        ("Rz", lambda b: b.rz(0, math.pi / 5)),
        ("P", lambda b: b.p(0, math.pi / 6)),
    ],
)
def test_1q_param_gate(gate, builder, emitter):
    b = SimpleBlock(1)
    builder(b)
    b.build()
    cmds = b.flatten()

    qc = emitter.emit(cmds, 1)

    U_qiskit = _unitary_from_qiskit(qc)
    U_qarpx = _unitary_from_qarpx(cmds, 1)
    assert np.allclose(U_qiskit, U_qarpx, atol=1e-10), f"Unitary mismatch for {gate}"


def test_u_gate(emitter):
    b = SimpleBlock(1)
    b.u(0, math.pi / 3, math.pi / 4, math.pi / 5)
    b.build()
    cmds = b.flatten()
    qc = emitter.emit(cmds, 1)
    assert np.allclose(_unitary_from_qiskit(qc), _unitary_from_qarpx(cmds, 1), atol=1e-10)


# ── Basic 2-qubit gate tests ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "gate,builder",
    [
        ("CX", lambda b: b.cx(0, 1)),
        ("CY", lambda b: b.cy(0, 1)),
        ("CZ", lambda b: b.cz(0, 1)),
        ("SWAP", lambda b: b.swap(0, 1)),
        ("ECR", lambda b: b.ecr(0, 1)),
        ("iSWAP", lambda b: b.iswap(0, 1)),
        ("iSWAPdg", lambda b: b.iswapdg(0, 1)),
        ("CH", lambda b: b.ch(0, 1)),
        ("CS", lambda b: b.cs(0, 1)),
        ("CSdg", lambda b: b.csdg(0, 1)),
        ("CSX", lambda b: b.csx(0, 1)),
        ("CSXdg", lambda b: b.csxdg(0, 1)),
    ],
)
def test_2q_no_param_gate(gate, builder, emitter):
    b = SimpleBlock(2)
    builder(b)
    b.build()
    cmds = b.flatten()
    qc = emitter.emit(cmds, 2)
    assert np.allclose(_unitary_from_qiskit(qc), _unitary_from_qarpx(cmds, 2), atol=1e-10), (
        f"Unitary mismatch for {gate}"
    )


@pytest.mark.parametrize(
    "gate,builder",
    [
        ("CRx", lambda b: b.crx(0, 1, math.pi / 4)),
        ("CRy", lambda b: b.cry(0, 1, math.pi / 4)),
        ("CRz", lambda b: b.crz(0, 1, math.pi / 4)),
        ("CP", lambda b: b.cp(0, 1, math.pi / 4)),
        ("RZZ", lambda b: b.rzz(0, 1, math.pi / 4)),
        ("RXX", lambda b: b.rxx(0, 1, math.pi / 4)),
        ("RYY", lambda b: b.ryy(0, 1, math.pi / 4)),
    ],
)
def test_2q_param_gate(gate, builder, emitter):
    b = SimpleBlock(2)
    builder(b)
    b.build()
    cmds = b.flatten()
    qc = emitter.emit(cmds, 2)
    assert np.allclose(_unitary_from_qiskit(qc), _unitary_from_qarpx(cmds, 2), atol=1e-10), (
        f"Unitary mismatch for {gate}"
    )


def test_ccx_gate(emitter):
    b = SimpleBlock(3)
    b.ccx(0, 1, 2)
    b.build()
    cmds = b.flatten()
    qc = emitter.emit(cmds, 3)
    assert np.allclose(_unitary_from_qiskit(qc), _unitary_from_qarpx(cmds, 3), atol=1e-10)


def test_cswap_gate(emitter):
    b = SimpleBlock(3)
    b.cswap(0, 1, 2)
    b.build()
    cmds = b.flatten()
    qc = emitter.emit(cmds, 3)
    assert np.allclose(_unitary_from_qiskit(qc), _unitary_from_qarpx(cmds, 3), atol=1e-10)


# ── Symbolic parameter round-trip ─────────────────────────────────────────────


def test_symbolic_param_creates_qiskit_parameter(emitter):

    b = SimpleBlock(1)
    b.rx(0, qx.Param.symbol("theta"))
    b.build()
    cmds = b.flatten()
    qc = emitter.emit(cmds, 1)
    assert len(qc.parameters) == 1
    assert list(qc.parameters)[0].name == "theta"


def test_symbolic_param_linear_form(emitter):

    b = SimpleBlock(1)
    b.rx(0, qx.Param.linear(2.0, "theta", 0.5))
    b.build()
    cmds = b.flatten()
    qc = emitter.emit(cmds, 1)
    assert len(qc.parameters) == 1
    # Bind the parameter and check unitary
    theta_val = math.pi / 4
    bound = qc.assign_parameters({"theta": theta_val})
    concrete_b = SimpleBlock(1)
    concrete_b.rx(0, 2.0 * theta_val + 0.5)
    concrete_b.build()
    U_bound = _unitary_from_qiskit(bound)
    U_qarpx = _unitary_from_qarpx(concrete_b.flatten(), 1)
    assert np.allclose(U_bound, U_qarpx, atol=1e-10)


def test_symbolic_param_2q_gate_unitary(emitter):
    """A symbolic param on a 2-qubit controlled gate must bind to the same
    unitary as building the gate directly with the concrete value."""
    b = SimpleBlock(2)
    b.crx(0, 1, qx.Param.symbol("theta"))
    b.build()
    qc = emitter.emit(b.flatten(), 2)
    theta_val = math.pi / 3
    bound = qc.assign_parameters({"theta": theta_val})
    concrete_b = SimpleBlock(2)
    concrete_b.crx(0, 1, theta_val)
    concrete_b.build()
    assert np.allclose(
        _unitary_from_qiskit(bound), _unitary_from_qarpx(concrete_b.flatten(), 2), atol=1e-10
    )


# ── Multi-gate circuit ────────────────────────────────────────────────────────


def test_multi_gate_circuit(emitter):
    b = SimpleBlock(3)
    b.h(0).cx(0, 1).rz(2, math.pi / 3).ccx(0, 1, 2)
    b.build()
    cmds = b.flatten()
    qc = emitter.emit(cmds, 3)
    assert np.allclose(_unitary_from_qiskit(qc), _unitary_from_qarpx(cmds, 3), atol=1e-10)


# ── Measurement / mid-circuit measurement ──────────────────────────────────────


def test_measure_basic(emitter):
    b = SimpleBlock(1)
    b.h(0)
    b.measure(0, 0)
    b.build()
    qc = emitter.emit(b.flatten(), 1)
    assert qc.num_clbits == 1


def test_measure_distinct_cbit(emitter):
    """Qiskit has real classical registers — qubit and cbit need not match."""
    b = SimpleBlock(2)
    b.h(0)
    b.measure(0, 1)
    b.build()
    qc = emitter.emit(b.flatten(), 2)
    ops = [
        (
            inst.operation.name,
            [qc.find_bit(q).index for q in inst.qubits],
            [qc.find_bit(c).index for c in inst.clbits],
        )
        for inst in qc.data
    ]
    assert ("measure", [0], [1]) in ops


def test_mid_circuit_measure(emitter):
    b = SimpleBlock(1)
    b.x(0)
    b.measure(0, 0)
    b.x(0)
    b.measure(0, 1)
    b.build()
    qc = emitter.emit(b.flatten(), b.n_qubits)
    # Should not raise and should contain two measure instructions.
    n_measures = sum(1 for inst in qc.data if inst.operation.name == "measure")
    assert n_measures == 2


def test_conditional_emits_if_else(emitter):
    """A ConditionalBlock exports to a qiskit IfElseOp (documented branch path)."""
    from qarp.blocks import ConditionalBlock

    body = SimpleBlock(1)
    body.x(0)
    body.build()
    cond = ConditionalBlock(cbits=[0], values=[True], then_body=body)
    cond.build()
    qc = emitter.emit(cond.flatten(), cond.n_qubits)
    assert any(inst.operation.name == "if_else" for inst in qc.data)


def test_measurement_endianness_matches_qarpx(emitter):
    """Qiskit's measurement bitstrings are written MSB-first (the highest
    classical-bit index is the leftmost character) — the reverse of
    qarp.endianness's LSB-first per-qubit container convention (qubit/cbit 0
    is the first element). That is a documented string-notation difference,
    NOT a translation bug: decoded correctly with each system's own
    convention, every individual qubit's value must still agree.
    """
    from qiskit.quantum_info import Statevector

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
    sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False))
    outcome, _ = sv.measure()
    # outcome is MSB-first ("cbit3 cbit2 cbit1 cbit0"); reverse to get
    # qarp's LSB-first per-qubit list (bits[q] == cbit q).
    qiskit_bits = [int(c) for c in outcome[::-1]]

    assert qarpx_bits == qiskit_bits == [1, 0, 1, 0]


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
    assert np.allclose(
        _unitary_from_qiskit(qc), _unitary_from_qarpx(cmds, comp.n_qubits), atol=1e-10
    )


# ── Round-trip via absorber (block equality) ────────────────────────────────
# emit() + QiskitAbsorber().absorb() must reconstruct exactly the same
# circuit (Block.__eq__), not just an equivalent unitary, across a range of
# complexities: plain/parametric gates, symbols, mid-circuit measurement, and
# nested CompositeBlocks.


def test_round_trip_block_equality_basic(emitter):
    b = SimpleBlock(2)
    b.h(0).cx(0, 1).rz(1, math.pi / 4)
    b.build()
    qc = emitter.emit(b.flatten(), 2)
    absorbed = QiskitAbsorber().absorb(qc)
    assert absorbed == b


def test_round_trip_block_equality_parametric(emitter):
    b = SimpleBlock(2)
    b.crx(0, 1, math.pi / 4).ryy(0, 1, math.pi / 6)
    b.build()
    qc = emitter.emit(b.flatten(), 2)
    absorbed = QiskitAbsorber().absorb(qc)
    assert absorbed == b


def test_round_trip_block_equality_symbolic(emitter):
    b = SimpleBlock(1)
    b.rx(0, qx.Param.symbol("theta"))
    b.build()
    qc = emitter.emit(b.flatten(), 1)
    absorbed = QiskitAbsorber().absorb(qc)
    assert absorbed == b


def test_round_trip_block_equality_mid_circuit_measure(emitter):
    b = SimpleBlock(1)
    b.x(0)
    b.measure(0, 0)
    b.x(0)
    b.measure(0, 1)
    b.build()
    qc = emitter.emit(b.flatten(), 1)
    absorbed = QiskitAbsorber().absorb(qc)
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
    absorbed = QiskitAbsorber().absorb(qc)
    assert absorbed == comp


# ── Error cases ───────────────────────────────────────────────────────────────


def test_error_on_custom_gate(emitter):
    cmd = qx.Command()
    cmd.gate = qx.GateType.Custom
    with pytest.raises(CapabilityError, match="Custom"):
        emitter.emit([cmd], 1)


def test_import_error_with_hint(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "qiskit", None)
    with pytest.raises(ImportError, match="pip install"):
        QiskitEmitter().emit([], 1)


@pytest.mark.parametrize("n", [3, 4])
def test_mcz_unitary_matches_qarpx(emitter, n):
    """MCZ (§5) exported to Qiskit (H + MCX + H) must match the QARPx unitary."""
    b = SimpleBlock(n)
    b.mcz(list(range(n)))
    b.build()
    qc = emitter.emit(b.flatten(), n)
    assert np.allclose(_unitary_from_qiskit(qc), _block_unitary(b), atol=1e-10)


# ── Parameter bridge linearity (pipeline_hardening_plan.md P1.14) ─────────


def test_nonlinear_single_symbol_param_is_refused(emitter):
    """t*t has one free symbol but is not c*x + d; the bridge probes a third
    point and refuses instead of emitting a wrong linear form."""
    b = SimpleBlock(1)
    b.rx(0, qx.Param.symbol("t") * qx.Param.symbol("t"))
    b.build()
    assert "linear" in (b.can_emit_to("qiskit") or "")
    with pytest.raises(CapabilityError, match="linear"):
        emitter.emit(b.flatten(), 1)


def test_linear_param_with_offset_binds_to_analytic_rz(emitter, sdk_unitary):
    """Rz(2t + 0.5) at t = 0.3 is Rz(1.1) = diag(e^{-0.55i}, e^{0.55i})."""
    b = SimpleBlock(1)
    b.rz(0, qx.Param.linear(2.0, "t", 0.5))
    b.build()
    qc = emitter.emit(b.flatten(), 1)
    bound = qc.assign_parameters({"t": 0.3})
    expected = np.diag([np.exp(-0.55j), np.exp(0.55j)])
    np.testing.assert_allclose(sdk_unitary["qiskit"](bound, 1), expected, atol=1e-12)
