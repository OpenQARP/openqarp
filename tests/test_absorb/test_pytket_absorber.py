"""Tests for PytketAbsorber."""

import math

import numpy as np
import pytest

import qarpx as qx

pytket = pytest.importorskip("pytket")

from qarp.absorb import PytketAbsorber
from qarp.blocks import SimpleBlock
from qarp.emit import PytketEmitter

from .conftest import feedforward_block


@pytest.fixture
def emitter() -> PytketEmitter:
    return PytketEmitter()


@pytest.fixture
def absorber() -> PytketAbsorber:
    return PytketAbsorber()


def _qarpx_unitary(block) -> np.ndarray:
    block.build()
    sim = qx.QarpSimulator()
    return np.array(sim.unitary_matrix(block.flatten(), block.n_qubits))


def _tket_unitary(circuit) -> np.ndarray:
    U = np.array(circuit.get_unitary())
    n = circuit.n_qubits
    if n > 1:
        # TKET get_unitary() is big-endian (qubit 0 = MSB); QARPx is little-endian
        # (qubit 0 = LSB) — convert via the bit-reversal permutation.
        perm = [int(f"{i:0{n}b}"[::-1], 2) for i in range(2**n)]
        U = U[np.ix_(perm, perm)]
    return U


def test_round_trip_basic(emitter, absorber):
    b = SimpleBlock(2)
    b.h(0).cx(0, 1).rz(1, math.pi / 4)
    b.build()

    circ = emitter.emit(b.flatten(), 2)
    absorbed = absorber.absorb(circ)

    assert absorbed == b
    assert np.allclose(_qarpx_unitary(b), _qarpx_unitary(absorbed), atol=1e-10)


def test_round_trip_gateset_expansion(emitter, absorber):
    b = SimpleBlock(2)
    b.sx(0).sxdg(1).id(0).ch(0, 1).cs(0, 1).csdg(1, 0).csx(0, 1).csxdg(1, 0)
    b.build()
    circ = emitter.emit(b.flatten(), 2)
    absorbed = absorber.absorb(circ)
    assert absorbed == b
    assert np.allclose(_qarpx_unitary(b), _qarpx_unitary(absorbed), atol=1e-10)


def test_pytket_v_vdg_noop_absorb_exactly(absorber):
    """pytket's V is Rx(π/2) (no e^{iπ/4}: not SX); noop is the explicit identity."""
    from pytket import Circuit
    from pytket.circuit import OpType

    circ = Circuit(1)
    circ.add_gate(OpType.V, [0])
    circ.add_gate(OpType.Vdg, [0])
    circ.add_gate(OpType.noop, [0])
    circ.add_gate(OpType.SX, [0])
    absorbed = absorber.absorb(circ)
    names = [c.gate.name for c in absorbed.flatten()]
    assert names == ["Rx", "Rx", "Id", "SX"]
    assert np.allclose(_qarpx_unitary(absorbed), _tket_unitary(circ), atol=1e-10)


def test_round_trip_parametric(emitter, absorber):
    b = SimpleBlock(1)
    b.rx(0, math.pi / 3).ry(0, math.pi / 5)
    b.build()

    circ = emitter.emit(b.flatten(), 1)
    absorbed = absorber.absorb(circ)

    assert absorbed == b
    assert np.allclose(_qarpx_unitary(b), _qarpx_unitary(absorbed), atol=1e-10)


def test_rzz_sign_round_trip(emitter, absorber):
    """ZZPhase sign convention must cancel on round-trip."""
    b = SimpleBlock(2)
    b.rzz(0, 1, math.pi / 4)
    b.build()

    circ = emitter.emit(b.flatten(), 2)
    absorbed = absorber.absorb(circ)

    assert absorbed == b
    assert np.allclose(_qarpx_unitary(b), _qarpx_unitary(absorbed), atol=1e-10)


def test_rxx_ryy_round_trip(emitter, absorber):
    for builder in [lambda b: b.rxx(0, 1, math.pi / 4), lambda b: b.ryy(0, 1, math.pi / 4)]:
        b = SimpleBlock(2)
        builder(b)
        b.build()
        circ = emitter.emit(b.flatten(), 2)
        absorbed = absorber.absorb(circ)
        assert absorbed == b
        assert np.allclose(_qarpx_unitary(b), _qarpx_unitary(absorbed), atol=1e-10)


def test_symbolic_param_round_trip(emitter, absorber):
    pytest.importorskip("sympy")
    b = SimpleBlock(1)
    b.rx(0, qx.Param.symbol("theta"))
    b.build()

    circ = emitter.emit(b.flatten(), 1)
    absorbed = absorber.absorb(circ)

    cmds = absorbed.flatten()
    assert any(p.is_symbolic() for cmd in cmds for p in cmd.params), (
        "Expected symbolic param after absorb"
    )
    assert absorbed == b


def test_symbolic_param_round_trip_unitary(emitter, absorber):
    """Bind the TKET-side free symbol to a concrete value after emission,
    then absorb and compare against the same value substituted in QARPx."""
    pytest.importorskip("sympy")
    b = SimpleBlock(1)
    b.rx(0, qx.Param.symbol("theta"))
    b.build()

    circ = emitter.emit(b.flatten(), 1)
    sym = list(circ.free_symbols())[0]
    theta_val = math.pi / 3
    circ.symbol_substitution({sym: theta_val})
    absorbed = absorber.absorb(circ)

    concrete = SimpleBlock(1)
    concrete.rx(0, theta_val)
    concrete.build()
    assert absorbed == concrete
    assert np.allclose(_qarpx_unitary(absorbed), _qarpx_unitary(concrete), atol=1e-10)


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
def test_parametric_2q_round_trip(gate, builder, emitter, absorber):
    b = SimpleBlock(2)
    builder(b)
    b.build()
    circ = emitter.emit(b.flatten(), 2)
    absorbed = absorber.absorb(circ)
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
    circ = emitter.emit(b.flatten(), 1)
    absorbed = absorber.absorb(circ)
    assert absorbed == b


def test_round_trip_measure_distinct_cbit(emitter, absorber):
    b = SimpleBlock(2)
    b.h(0)
    b.measure(0, 1)
    b.build()
    circ = emitter.emit(b.flatten(), 2)
    absorbed = absorber.absorb(circ)
    assert absorbed == b


def test_round_trip_mid_circuit_measure(emitter, absorber):
    b = SimpleBlock(1)
    b.x(0)
    b.measure(0, 0)
    b.x(0)
    b.measure(0, 1)
    b.build()
    circ = emitter.emit(b.flatten(), 1)
    absorbed = absorber.absorb(circ)
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

    circ = emitter.emit(comp.flatten(), comp.n_qubits)
    absorbed = absorber.absorb(circ)

    assert absorbed == comp
    sim = qx.QarpSimulator()
    U_orig = np.array(sim.unitary_matrix(comp.flatten(), comp.n_qubits))
    U_absorbed = np.array(sim.unitary_matrix(absorbed.flatten(), absorbed.n_qubits))
    assert np.allclose(U_orig, U_absorbed, atol=1e-10)


def test_global_phase_absorbed_preserves_unitary():
    """pytket carries a circuit's global phase on ``circuit.phase`` (half-turns),
    not as a command, so the absorber must read that attribute back. Built
    directly, not via our emitter."""
    from pytket import Circuit

    circ = Circuit(1)
    circ.H(0)
    circ.add_phase(0.7 / np.pi)  # half-turns -> global phase e^{i*0.7}
    block = PytketAbsorber().absorb(circ)
    assert np.allclose(_qarpx_unitary(block), _tket_unitary(circ), atol=1e-10)


# ── Parameter bridge linearity (pipeline_hardening_plan.md P1.14) ─────────


def test_nonlinear_parameter_expression_is_refused(absorber):
    """t**2 is one sympy symbol but not c*x + d; the third probe refuses it."""
    import sympy
    from pytket import Circuit

    from qarp.errors import CapabilityError

    t = sympy.Symbol("t")
    circ = Circuit(1)
    circ.Rx(t * t, 0)
    with pytest.raises(CapabilityError, match="not linear"):
        absorber.absorb(circ)


def test_linear_param_with_offset_round_trips(emitter, absorber):
    b = SimpleBlock(1)
    b.rz(0, qx.Param.linear(2.0, "t", 0.5))
    b.build()
    absorbed = absorber.absorb(emitter.emit(b.flatten(), 1))
    assert absorbed == b
    bound = absorbed.set_symbols({s: 0.3 for s in absorbed.symbols})
    expected = np.diag([np.exp(-0.55j), np.exp(0.55j)])  # Rz(1.1)
    np.testing.assert_allclose(_qarpx_unitary(bound), expected, atol=1e-12)


# ── Conditionals and barriers (pipeline_hardening_plan.md P1.16) ──────────


@pytest.mark.parametrize("with_else", [False, True])
def test_conditional_round_trips_through_conditional_ops(emitter, absorber, with_else):
    """pytket has per-command conditions, so the emitter writes the else arm
    as a second run on the complemented bit; the absorber folds it back."""
    from pytket.circuit import OpType

    block = feedforward_block(with_else)
    circ = emitter.emit(block.flatten(), 2)
    assert any(cmd.op.type == OpType.Conditional for cmd in circ.get_commands())
    absorbed = absorber.absorb(circ)
    assert absorbed == block
    # c0=1 → X(1) → |11⟩ (key 3); c0=0 → Y(1) → |10⟩ (key 2) with else, |00⟩ without.
    res = qx.QarpSimulator().run(absorbed.flatten(), 2, 400, seed=7)
    assert set(res.counts) == ({2, 3} if with_else else {0, 3})
    assert sum(h[0] for h in res.cbit_history) == res.counts[3]


def test_two_bit_conditional_round_trips(emitter, absorber):
    """H(0); c0 = M(0); X(0); c1 = M(0); if (c0, c1) == (1, 0): X(1).  One
    serial wire, so pytket's DAG order is unique and `==` is order-safe."""
    from qarp.blocks import CompositeBlock, ConditionalBlock

    meas = SimpleBlock(2)
    meas.h(0)
    meas.measure(0, 0)
    meas.x(0)
    meas.measure(0, 1)
    meas.build()
    then_body = SimpleBlock(2)
    then_body.x(1)
    then_body.build()
    cond = ConditionalBlock(cbits=[0, 1], values=[True, False], then_body=then_body)
    cond.target_cbits = [0, 1]
    block = CompositeBlock([meas, cond], 2)
    block.build()

    absorbed = absorber.absorb(emitter.emit(block.flatten(), 2))
    assert absorbed == block
    # First outcome 1 → q0 flips to 0, c1=0, condition true → X(1) → |10⟩ (key
    # 2); first outcome 0 → q0=1, c1=1, condition false → |01⟩ (key 1).
    res = qx.QarpSimulator().run(absorbed.flatten(), 2, 400, seed=7)
    assert set(res.counts) == {1, 2}
    assert sum(h == [True, False] for h in res.cbit_history) == res.counts[2]


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
    np.testing.assert_allclose(_qarpx_unitary(absorbed), _qarpx_unitary(b), atol=1e-12)


def test_phase_getter_survives_qarpx_import():
    """Regression for the SymEngine symbol leak: with qarpx loaded first,
    pytket's own `Circuit.phase` read 1.0 (its SymEngine calls resolved into
    qarpx's copy) and the absorber then segfaulted.  qarpx now exports only
    its module entry point."""
    from pytket import Circuit

    assert Circuit(1).phase == 0.0
