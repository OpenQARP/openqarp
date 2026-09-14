"""Tests for QulacsAbsorber."""

import math

import numpy as np
import pytest

import qarpx as qx

qulacs = pytest.importorskip("qulacs")

from qarp.absorb import QulacsAbsorber
from qarp.blocks import SimpleBlock
from qarp.emit import QulacsEmitter


@pytest.fixture
def emitter() -> QulacsEmitter:
    return QulacsEmitter()


@pytest.fixture
def absorber() -> QulacsAbsorber:
    return QulacsAbsorber()


def _qarpx_unitary(block) -> np.ndarray:
    block.build()
    sim = qx.QarpSimulator()
    try:
        return np.array(sim.unitary_matrix(block.flatten(), block.n_qubits))
    except RuntimeError as e:
        pytest.skip(f"QarpSimulator does not support gate in circuit: {e}")


def _qulacs_unitary(circuit, n_qubits: int) -> np.ndarray:
    mat = np.eye(2**n_qubits, dtype=complex)
    state = qulacs.QuantumState(n_qubits)
    for col in range(2**n_qubits):
        vec = np.zeros(2**n_qubits, dtype=complex)
        vec[col] = 1.0
        state.load(vec)
        circuit.update_quantum_state(state)
        mat[:, col] = state.get_vector()
    return mat


def test_round_trip_clifford(emitter, absorber):
    b = SimpleBlock(2)
    b.h(0).cx(0, 1).s(0).t(1)
    b.build()

    qc = emitter.emit(b.flatten(), 2)
    absorbed = absorber.absorb(qc)

    assert absorbed == b
    assert np.allclose(_qarpx_unitary(b), _qarpx_unitary(absorbed), atol=1e-10)


def test_round_trip_sqrt_x_pair_and_identity(emitter, absorber):
    """qulacs has named sqrtX / sqrtXdag / I gates: exact GateType round trip."""
    b = SimpleBlock(2)
    b.sx(0).sxdg(1).id(0)
    b.build()
    qc = emitter.emit(b.flatten(), 2)
    absorbed = absorber.absorb(qc)
    assert absorbed == b
    assert np.allclose(_qarpx_unitary(b), _qarpx_unitary(absorbed), atol=1e-10)


def test_controlled_clifford_singles_round_trip_up_to_unitary(emitter, absorber):
    """These cross qulacs as dense matrices, so only the unitary is pinned."""
    b = SimpleBlock(2)
    b.ch(0, 1).cs(0, 1).csdg(1, 0).csx(0, 1).csxdg(1, 0)
    b.build()
    qc = emitter.emit(b.flatten(), 2)
    absorbed = absorber.absorb(qc)
    assert np.allclose(_qarpx_unitary(b), _qarpx_unitary(absorbed), atol=1e-10)


def test_round_trip_cz_qubits_preserved(emitter, absorber):
    """add_CZ_gate splits its two qubits across target_index_list (1 elem)
    and control_index_list (1 elem) — like CNOT, not like SWAP. A 2-qubit
    CZ(0,1) round-trip can pass by accident (CZ is symmetric, so swapping
    qubits 0/1 yields the same gate); use qubits {1,2} in a 3-qubit register
    where misreading the second qubit as 0 is not a no-op."""
    b = SimpleBlock(3)
    b.cz(1, 2)
    b.build()

    qc = emitter.emit(b.flatten(), 3)
    absorbed = absorber.absorb(qc)
    cmds = absorbed.flatten()

    assert any(c.gate == qx.GateType.CZ and set(c.qubits) == {1, 2} for c in cmds)
    assert absorbed == b
    assert np.allclose(_qarpx_unitary(b), _qarpx_unitary(absorbed), atol=1e-10)


def test_round_trip_rotation(emitter, absorber):
    b = SimpleBlock(1)
    b.rx(0, math.pi / 3).ry(0, math.pi / 4).rz(0, math.pi / 5)
    b.build()

    qc = emitter.emit(b.flatten(), 1)
    absorbed = absorber.absorb(qc)

    assert absorbed == b
    assert np.allclose(_qarpx_unitary(b), _qarpx_unitary(absorbed), atol=1e-10)


def test_angle_negation_cancels(emitter, absorber):
    """Emit+absorb angle negation must cancel: Rx(θ) → Qulacs → Rx(θ) again."""
    theta = math.pi / 3
    b = SimpleBlock(1)
    b.rx(0, theta)
    b.build()

    qc = emitter.emit(b.flatten(), 1)
    absorbed = absorber.absorb(qc)
    cmds = absorbed.flatten()

    # The absorbed rotation angle should equal the original
    assert any(abs(float(p.value()) - theta) < 1e-9 for cmd in cmds for p in cmd.params)
    assert absorbed == b


def test_iswap_dense_matrix_round_trip(emitter, absorber):
    b = SimpleBlock(2)
    b.iswap(0, 1)
    b.build()

    qc = emitter.emit(b.flatten(), 2)
    absorbed = absorber.absorb(qc)

    assert absorbed == b
    assert np.allclose(_qarpx_unitary(b), _qarpx_unitary(absorbed), atol=1e-10)


# ── Parametric DenseMatrix-fallback round-trips ────────────────────────────────


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
def test_parametric_2q_round_trip(gate, n, builder, emitter, absorber):
    b = SimpleBlock(n)
    builder(b)
    b.build()
    qc = emitter.emit(b.flatten(), n)
    absorbed = absorber.absorb(qc)
    assert absorbed == b, f"Round-trip block mismatch for {gate}"
    assert np.allclose(_qarpx_unitary(b), _qarpx_unitary(absorbed), atol=1e-10), (
        f"Round-trip unitary mismatch for {gate}"
    )


def test_ccx_cswap_round_trip(emitter, absorber):
    for name, builder in [("CCX", lambda b: b.ccx(0, 1, 2)), ("CSWAP", lambda b: b.cswap(0, 1, 2))]:
        b = SimpleBlock(3)
        builder(b)
        b.build()
        qc = emitter.emit(b.flatten(), 3)
        absorbed = absorber.absorb(qc)
        assert absorbed == b, f"Round-trip block mismatch for {name}"
        assert np.allclose(_qarpx_unitary(b), _qarpx_unitary(absorbed), atol=1e-10), (
            f"Round-trip unitary mismatch for {name}"
        )


# ── Measurement / mid-circuit measurement round-trip ──────────────────────────
# The gate's classical register address is only readable via to_json() — see
# the "CPTP" branch in qulacs_absorber.cpp. These tests guard against losing
# that mapping (assuming cbit == qubit silently loses it).


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
    assert absorbed == comp
    assert np.allclose(_qarpx_unitary(comp), _qarpx_unitary(absorbed), atol=1e-10)


# ── Asymmetric 2q gates absorbed independently ───────────────────────────────


def _qarpx_unitary_native(block) -> np.ndarray:
    """Unitary via QarpSimulator, lowering to the native gate set first so gates
    csim cannot simulate directly (e.g. ECR) go through their decomposition."""
    lowered = block.optimize(qx.native_gateset())
    return np.array(qx.QarpSimulator().unitary_matrix(lowered.flatten(), block.n_qubits))


def test_cy_absorbed_preserves_unitary():
    """A qulacs CY DenseMatrix (control on the local MSB) absorbs to a block with
    the same unitary. The circuit is built directly and qulacs computes the
    reference unitary."""
    from qulacs import QuantumCircuit
    from qulacs.gate import DenseMatrix

    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    cy_local = np.eye(4, dtype=complex)
    cy_local[2:, 2:] = y  # control on the local MSB (qubit_list[1])
    qc = QuantumCircuit(2)
    qc.add_gate(DenseMatrix([0, 1], cy_local))
    block = QulacsAbsorber().absorb(qc)
    assert np.allclose(_qarpx_unitary(block), _qulacs_unitary(qc, 2), atol=1e-10)


def test_ecr_absorbed_preserves_unitary():
    """A qulacs ECR DenseMatrix absorbs to a block with the same unitary (ECR is
    non-simulable, so lower to native before comparing)."""
    from qulacs import QuantumCircuit
    from qulacs.gate import DenseMatrix

    s = 1.0 / math.sqrt(2.0)
    ecr_local = s * np.array(
        [[0, 0, 1, 1j], [0, 0, 1j, 1], [1, -1j, 0, 0], [-1j, 1, 0, 0]], dtype=complex
    )
    qc = QuantumCircuit(2)
    qc.add_gate(DenseMatrix([0, 1], ecr_local))
    block = QulacsAbsorber().absorb(qc)
    assert np.allclose(_qarpx_unitary_native(block), _qulacs_unitary(qc, 2), atol=1e-10)


def test_swap_on_high_qubits_absorbed_preserves_unitary():
    """SWAP on qubits (1, 2) in a 3-qubit register — a misread of the SWAP qubit
    pair (a symmetric swap(0,1) would hide it) changes the unitary."""
    from qulacs import QuantumCircuit

    qc = QuantumCircuit(3)
    qc.add_SWAP_gate(1, 2)
    block = QulacsAbsorber().absorb(qc)
    assert np.allclose(_qarpx_unitary(block), _qulacs_unitary(qc, 3), atol=1e-10)


def test_general_1q_densematrix_absorbed_preserves_unitary():
    """A general single-qubit U (which qulacs carries as a 1-qubit DenseMatrix)
    must absorb to a block with the same unitary via the U3+phase decomposition,
    not raise 'unrecognized 1-qubit DenseMatrix gate'. Built directly."""
    from qulacs import QuantumCircuit
    from qulacs.gate import DenseMatrix

    th, ph, lam = 0.3, 0.4, 0.5
    c, s = math.cos(th / 2), math.sin(th / 2)
    u_mat = np.array(
        [[c, -np.exp(1j * lam) * s], [np.exp(1j * ph) * s, np.exp(1j * (ph + lam)) * c]],
        dtype=complex,
    )
    qc = QuantumCircuit(1)
    qc.add_gate(DenseMatrix([0], u_mat))
    block = QulacsAbsorber().absorb(qc)
    assert np.allclose(_qarpx_unitary(block), _qulacs_unitary(qc, 1), atol=1e-10)
