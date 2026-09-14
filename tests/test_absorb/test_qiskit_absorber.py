"""Tests for QiskitAbsorber.

Guards: all tests are skipped when qiskit is not installed.

Coverage:
  - Basic gate round-trip (emit → absorb → unitary comparison)
  - Symbolic parameter absorption
  - AbsorbError on unrecognized gate
"""

import math

import numpy as np
import pytest

import qarpx as qx

qiskit = pytest.importorskip("qiskit")

from qarp.absorb import QiskitAbsorber
from qarp.blocks import SimpleBlock
from qarp.emit import QiskitEmitter

from .conftest import feedforward_block

# ── Fixture ───────────────────────────────────────────────────────────────────


@pytest.fixture
def emitter() -> QiskitEmitter:
    return QiskitEmitter()


@pytest.fixture
def absorber() -> QiskitAbsorber:
    return QiskitAbsorber()


def _unitary_from_qarpx(block) -> np.ndarray:
    block.build()
    sim = qx.QarpSimulator()
    return np.array(sim.unitary_matrix(block.flatten(), block.n_qubits))


# ── Round-trip tests ──────────────────────────────────────────────────────────


def test_round_trip_basic(emitter, absorber):
    """Emit a basic block to Qiskit then absorb back; unitaries must match."""
    b = SimpleBlock(2)
    b.h(0).cx(0, 1).rz(1, math.pi / 4)
    b.build()
    cmds = b.flatten()

    qc = emitter.emit(cmds, 2)
    absorbed = absorber.absorb(qc)

    assert absorbed == b, "absorbed block must be the same circuit as the original"
    U_orig = _unitary_from_qarpx(b)
    U_absorbed = _unitary_from_qarpx(absorbed)
    assert np.allclose(U_orig, U_absorbed, atol=1e-10)


def test_round_trip_gateset_expansion(emitter, absorber):
    """SX/SXdg/Id and the controlled Clifford singles come back as themselves."""
    b = SimpleBlock(2)
    b.sx(0).sxdg(1).id(0).ch(0, 1).cs(0, 1).csdg(1, 0).csx(0, 1).csxdg(1, 0)
    b.build()
    qc = emitter.emit(b.flatten(), 2)
    absorbed = absorber.absorb(qc)
    assert absorbed == b
    assert np.allclose(_unitary_from_qarpx(b), _unitary_from_qarpx(absorbed), atol=1e-10)


def test_foreign_qiskit_gateset_expansion_names(absorber):
    """A circuit built with qiskit's own gate classes, judged by qiskit's Operator."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import CSXGate
    from qiskit.quantum_info import Operator

    qc = QuantumCircuit(2)
    qc.sx(0), qc.sxdg(1), qc.id(0), qc.ch(0, 1), qc.cs(0, 1), qc.csdg(1, 0), qc.csx(0, 1)
    qc.append(CSXGate().inverse(), [1, 0])
    absorbed = absorber.absorb(qc)
    names = [c.gate.name for c in absorbed.flatten()]
    assert names == ["SX", "SXdg", "Id", "CH", "CS", "CSdg", "CSX", "CSXdg"]
    assert np.allclose(_unitary_from_qarpx(absorbed), Operator(qc).data, atol=1e-10)


def test_round_trip_parametric(emitter, absorber):
    b = SimpleBlock(1)
    b.rx(0, math.pi / 3)
    b.ry(0, math.pi / 5)
    b.build()
    cmds = b.flatten()

    qc = emitter.emit(cmds, 1)
    absorbed = absorber.absorb(qc)

    assert absorbed == b
    assert np.allclose(_unitary_from_qarpx(b), _unitary_from_qarpx(absorbed), atol=1e-10)


def test_round_trip_diverse_gates(emitter, absorber):
    b = SimpleBlock(3)
    b.h(0).cy(0, 1).crz(1, 2, math.pi / 6).ccx(0, 1, 2)
    b.build()
    qc = emitter.emit(b.flatten(), 3)
    absorbed = absorber.absorb(qc)
    assert absorbed == b
    assert np.allclose(_unitary_from_qarpx(b), _unitary_from_qarpx(absorbed), atol=1e-10)


def test_symbolic_param_absorbed_as_symbol(absorber):
    from qiskit.circuit import Parameter, QuantumCircuit
    from qiskit.circuit.library import RXGate

    qc = QuantumCircuit(1)
    theta = Parameter("theta")
    qc.append(RXGate(theta), [0])

    absorbed = absorber.absorb(qc)
    cmds = absorbed.flatten()
    assert any(len(cmd.params) > 0 and any(p.is_symbolic() for p in cmd.params) for cmd in cmds), (
        "Expected symbolic param in absorbed block"
    )

    expected = SimpleBlock(1)
    expected.rx(0, qx.Param.symbol("theta"))
    expected.build()
    assert absorbed == expected


def test_symbolic_param_round_trip_unitary(emitter, absorber):
    """Symbolic round-trip: bind the Qiskit-side parameter to a concrete value
    after absorbing, and check the unitary matches the same value substituted
    directly in QARPx."""
    b = SimpleBlock(1)
    b.rx(0, qx.Param.symbol("theta"))
    b.build()

    qc = emitter.emit(b.flatten(), 1)
    theta_val = math.pi / 3
    bound = qc.assign_parameters({"theta": theta_val})
    absorbed = absorber.absorb(bound)

    concrete = SimpleBlock(1)
    concrete.rx(0, theta_val)
    concrete.build()
    assert absorbed == concrete
    assert np.allclose(_unitary_from_qarpx(absorbed), _unitary_from_qarpx(concrete), atol=1e-10)


# ── Measurement / mid-circuit measurement round-trip ──────────────────────────


def test_round_trip_measure_basic(emitter, absorber):
    b = SimpleBlock(1)
    b.h(0)
    b.measure(0, 0)
    b.build()
    qc = emitter.emit(b.flatten(), 1)
    absorbed = absorber.absorb(qc)
    assert absorbed == b


def test_round_trip_measure_distinct_cbit(emitter, absorber):
    b = SimpleBlock(2)
    b.h(0)
    b.measure(0, 1)
    b.build()
    qc = emitter.emit(b.flatten(), 2)
    absorbed = absorber.absorb(qc)
    assert absorbed == b


def test_round_trip_mid_circuit_measure(emitter, absorber):
    """Gate -> measure -> gate -> measure on the same qubit must round-trip
    with both measurements preserved in order with correct cbits."""
    b = SimpleBlock(1)
    b.x(0)
    b.measure(0, 0)
    b.x(0)
    b.measure(0, 1)
    b.build()
    qc = emitter.emit(b.flatten(), 1)
    absorbed = absorber.absorb(qc)
    assert absorbed == b
    measures = [
        (list(c.qubits), list(c.cbits)) for c in absorbed.flatten() if c.gate == qx.GateType.Measure
    ]
    assert measures == [([0], [0]), ([0], [1])]


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

    qc = emitter.emit(comp.flatten(), comp.n_qubits)
    absorbed = absorber.absorb(qc)

    # Block.__eq__ compares flattened commands, so a CompositeBlock (nested
    # structure) compares equal to a flat SimpleBlock with the same commands.
    assert absorbed == comp

    sim = qx.QarpSimulator()
    U_orig = np.array(sim.unitary_matrix(comp.flatten(), comp.n_qubits))
    U_absorbed = np.array(sim.unitary_matrix(absorbed.flatten(), absorbed.n_qubits))
    assert np.allclose(U_orig, U_absorbed, atol=1e-10)


@pytest.mark.parametrize("n_ctrl", [2, 3])
def test_mcx_absorbed_preserves_unitary(n_ctrl):
    """A Qiskit MCX (named 'ccx' for 2 controls, 'mcx' for >=3) must absorb to a
    block with the same unitary — the >=3 case via H*MCZ*H. Built directly, not
    via our emitter."""
    import qiskit
    from qiskit.circuit.library import MCXGate
    from qiskit.quantum_info import Operator

    qc = qiskit.QuantumCircuit(n_ctrl + 1)
    qc.append(MCXGate(n_ctrl), list(range(n_ctrl + 1)))
    block = QiskitAbsorber().absorb(qc)
    assert np.allclose(_unitary_from_qarpx(block), Operator(qc).data, atol=1e-10)


# ── Parameter bridge linearity (pipeline_hardening_plan.md P1.14) ─────────


def test_nonlinear_parameter_expression_is_refused(absorber):
    """theta**2 is one symbol but not c*x + d; a two-point probe would silently
    absorb it as Rx(theta), the third point refuses it."""
    from qiskit.circuit import Parameter, QuantumCircuit

    from qarp.errors import CapabilityError

    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.rx(theta * theta, 0)
    with pytest.raises(CapabilityError, match="not linear"):
        absorber.absorb(qc)


def test_linear_param_with_offset_round_trips(emitter, absorber):
    b = SimpleBlock(1)
    b.rz(0, qx.Param.linear(2.0, "t", 0.5))
    b.build()
    absorbed = absorber.absorb(emitter.emit(b.flatten(), 1))
    assert absorbed == b
    bound = absorbed.set_symbols({s: 0.3 for s in absorbed.symbols})
    expected = np.diag([np.exp(-0.55j), np.exp(0.55j)])  # Rz(1.1)
    np.testing.assert_allclose(_unitary_from_qarpx(bound), expected, atol=1e-12)


# ── global_phase and the legacy singles (pipeline_hardening_plan.md P1.15) ─


def test_circuit_global_phase_is_absorbed_exactly():
    """`circuit.global_phase = 0.7` on an X circuit is e^{0.7i}·X, phase-exact
    (§13: the absorbed block must be controllable without a phase shift)."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.x(0)
    qc.global_phase = 0.7
    block = QiskitAbsorber().absorb(qc)
    expected = np.exp(0.7j) * np.array([[0, 1], [1, 0]])
    np.testing.assert_allclose(_unitary_from_qarpx(block), expected, atol=1e-12)


def test_symbolic_global_phase_is_absorbed():
    from qiskit.circuit import Parameter, QuantumCircuit

    t = Parameter("t")
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.global_phase = 2 * t + 0.5
    block = QiskitAbsorber().absorb(qc)
    expected = SimpleBlock(1)
    expected.gphase(qx.Param.linear(2.0, "t", 0.5))
    expected.x(0)
    expected.build()
    assert block == expected


def test_transpiled_circuit_matches_qiskit_operator():
    """qiskit's transpiler folds phases into `global_phase`; the absorbed block
    equals `Operator(tq)` in qarp's own LSB layout — no bit reversal."""
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Operator

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.3, 1)
    qc.sx(0)
    tq = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
    assert abs(float(tq.global_phase)) > 1e-9, "fixture must exercise global_phase"
    block = QiskitAbsorber().absorb(tq)
    np.testing.assert_allclose(_unitary_from_qarpx(block), Operator(tq).data, atol=1e-10)


def test_u1_u2_u3_r_match_qiskit_operator():
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RGate, U1Gate, U2Gate, U3Gate
    from qiskit.quantum_info import Operator

    for gate in (U1Gate(0.3), U2Gate(0.4, 0.7), U3Gate(0.2, 0.5, 0.9), RGate(0.3, 1.1)):
        qc = QuantumCircuit(1)
        qc.append(gate, [0])
        block = QiskitAbsorber().absorb(qc)
        np.testing.assert_allclose(
            _unitary_from_qarpx(block), Operator(qc).data, atol=1e-12, err_msg=gate.name
        )


# ── Conditionals and barriers (pipeline_hardening_plan.md P1.16) ──────────


@pytest.mark.parametrize("with_else", [False, True])
def test_conditional_round_trips_through_if_else(emitter, absorber, with_else):
    block = feedforward_block(with_else)
    qc = emitter.emit(block.flatten(), 2)
    assert any(inst.operation.name == "if_else" for inst in qc.data)
    absorbed = absorber.absorb(qc)
    assert absorbed == block
    # Feed-forward outcome per shot: c0=1 → X(1) → |11⟩ (key 3); c0=0 → Y(1)
    # → |10⟩ (key 2) with an else arm, |00⟩ (key 0) without.
    res = qx.QarpSimulator().run(absorbed.flatten(), 2, 400, seed=7)
    assert set(res.counts) == ({2, 3} if with_else else {0, 3})
    assert sum(h[0] for h in res.cbit_history) == res.counts[3]


def test_if_else_on_wide_register_is_refused(absorber):
    """A 2-bit ClassicalRegister condition compares an integer against
    several bits at once; only single-Clbit conditions cross."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    from qarp.errors import CapabilityError

    qr, cr = QuantumRegister(1), ClassicalRegister(2)
    qc = QuantumCircuit(qr, cr)
    qc.measure(0, 0)
    with qc.if_test((cr, 3)):
        qc.x(0)
    with pytest.raises(CapabilityError, match="2-bit ClassicalRegister"):
        absorber.absorb(qc)


def test_barrier_round_trips(emitter, absorber):
    b = SimpleBlock(2)
    b.h(0).cx(0, 1)
    b.build()
    cmds = list(b.flatten())
    cmds.insert(1, qx.Command(qx.GateType.Barrier, 0, 1))
    with_barrier = SimpleBlock(2)
    with_barrier.set_commands(cmds)
    with_barrier.mark_built()
    with_barrier._finalize()

    absorbed = absorber.absorb(emitter.emit(with_barrier.flatten(), 2))
    assert absorbed == with_barrier
    assert absorbed.n_gates_of_type(qx.GateType.Barrier) == 1
    # The barrier is a scheduling fence only: the unitary is the barrier-free one.
    np.testing.assert_allclose(_unitary_from_qarpx(absorbed), _unitary_from_qarpx(b), atol=1e-12)
