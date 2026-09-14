"""Per-target decomposition rules travel with the ``GateSet``.

Before this, ``qx.Transpiler(qx.clifford_t_rz_gateset())`` had no ``Rx``/``Ry`` rule
until the caller remembered ``install_clifford_t_rz_decompositions()``; two of the five
call sites did not, so ``Block.optimize(target_gateset=star)`` and a
``QarpEngine`` on a Clifford+T+Rz-gateset device failed on the first ``Rx``.

Oracles are analytic matrices assembled here with numpy (§18); the
implementation's own output is never the reference.  Blocks are built inside
tests, never at module scope.
"""

import math

import numpy as np
import pytest

import qarpx as qx
from qarp.algorithms import StateVector
from qarp.blocks import SimpleBlock
from qarp.engines import QarpEngine
from qarp.operators import QubitOperator

TH, PHI = 0.7, -0.4

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _rx(t):
    return np.cos(t / 2) * I2 - 1j * np.sin(t / 2) * X


def _ry(t):
    return np.cos(t / 2) * I2 - 1j * np.sin(t / 2) * Y


def _rz(t):
    return np.diag([np.exp(-1j * t / 2), np.exp(1j * t / 2)])


def _unitary(cmds, n):
    return np.array(qx.QarpSimulator().unitary_matrix(cmds, n))


def _rx_ry_block():
    b = SimpleBlock(1, name="rxry")
    b.rx(0, TH)
    b.ry(0, PHI)
    b.build()
    return b


# ── Rz-only rules come with the Clifford+T+Rz target ─────────────────────


def test_clifford_t_rz_transpiler_lowers_rx_without_an_install_call():
    t = qx.Transpiler(qx.clifford_t_rz_gateset())
    out = t.transpile([qx.Command(qx.GateType.Rx, 0, qx.Param(TH))])
    assert all(c.gate in (qx.GateType.H, qx.GateType.Rz) for c in out)
    # Phase-exact: Rx(θ) = H·Rz(θ)·H on the nose.
    np.testing.assert_allclose(_unitary(out, 1), _rx(TH), atol=1e-12)


def test_install_clifford_t_rz_decompositions_is_idempotent():
    t = qx.Transpiler(qx.clifford_t_rz_gateset())
    before = t.transpile([qx.Command(qx.GateType.Ry, 0, qx.Param(PHI))])
    t.install_clifford_t_rz_decompositions()
    after = t.transpile([qx.Command(qx.GateType.Ry, 0, qx.Param(PHI))])
    assert [c.gate for c in before] == [c.gate for c in after]
    np.testing.assert_allclose(_unitary(after, 1), _ry(PHI), atol=1e-12)


def test_block_optimize_to_clifford_t_rz_gateset_no_longer_fails_on_rx():
    lowered = _rx_ry_block().optimize(target_gateset=qx.clifford_t_rz_gateset())
    assert all(c.gate != qx.GateType.Rx for c in lowered.flatten())
    expected = _ry(PHI) @ _rx(TH)  # circuit order rx then ry
    np.testing.assert_allclose(_unitary(lowered.flatten(), 1), expected, atol=1e-12)


def test_qarp_engine_on_clifford_t_rz_gateset_device_runs_rotations():
    op = QubitOperator("Z0", 1.0) + QubitOperator("X0", 0.5)
    eng = QarpEngine(n_qubits=1, gate_set=qx.clifford_t_rz_gateset())
    eng.build([StateVector(operator=op, ket=_rx_ry_block())])
    got = eng.run({})[0]
    psi = _ry(PHI) @ _rx(TH) @ np.array([1, 0], dtype=complex)
    want = (psi.conj() @ (Z + 0.5 * X) @ psi).real
    assert abs(got.real - want) < 1e-10


# ── Python-registered rules ──────────────────────────────────────────────


def _gphase(theta):
    # Command.params is read-only from Python; route through the builder.
    b = qx.SimpleBlock(1, "gphase")
    b.gphase(qx.Param(theta))
    b.build()
    return b.flatten()[0]


def _rx_rz_target():
    gs = qx.GateSet()
    gs.name = "rx_rz_cx"
    gs.allowed = {
        qx.GateType.Rx,
        qx.GateType.Rz,
        qx.GateType.CX,
        qx.GateType.GPhase,
        qx.GateType.Measure,
        qx.GateType.Barrier,
    }
    return gs


def test_custom_target_without_a_rule_for_h_raises():
    """A gate the rule table cannot reach is a capability of the target, not
    an internal failure (P1.21); the diagnostic lists every gap, sorted."""
    from qarp.errors import CapabilityError

    t = qx.Transpiler(_rx_rz_target())
    with pytest.raises(CapabilityError, match="no decomposition for gate H") as info:
        t.transpile([qx.Command(qx.GateType.H, 0)])
    listed = str(info.value).split("unreachable gates for this target: ")[1].split(", ")
    assert listed == sorted(listed) and "H" in listed
    assert info.value.command.gate == qx.GateType.H
    names = [g.name for g in t.unreachable_gates()]
    assert names == sorted(names) == listed


def test_python_registered_rule_is_applied_and_phase_exact():
    t = qx.Transpiler(_rx_rz_target())

    def h_rule(cmd):
        # H = e^{iπ/2} · Rz(π/2) · Rx(π/2) · Rz(π/2)  (matrix product); emitted in circuit order.
        q = cmd.qubits[0]
        return [
            qx.Command(qx.GateType.Rz, q, qx.Param(math.pi / 2)),
            qx.Command(qx.GateType.Rx, q, qx.Param(math.pi / 2)),
            qx.Command(qx.GateType.Rz, q, qx.Param(math.pi / 2)),
            _gphase(math.pi / 2),
        ]

    t.register_decomposition(qx.GateType.H, h_rule)
    out = t.transpile([qx.Command(qx.GateType.H, 0)])
    assert {c.gate for c in out} <= _rx_rz_target().allowed
    np.testing.assert_allclose(_unitary(out, 1), H, atol=1e-12)


def test_python_registered_rule_overrides_a_builtin():
    # cudaq has no Sdg; the built-in lowers it to P.  A registered rule wins.
    t = qx.Transpiler(qx.cudaq_gateset())
    t.register_decomposition(
        qx.GateType.Sdg,
        lambda cmd: [qx.Command(qx.GateType.Rz, cmd.qubits[0], qx.Param(-math.pi / 2))],
    )
    out = t.transpile([qx.Command(qx.GateType.Sdg, 0)])
    assert [c.gate for c in out] == [qx.GateType.Rz]
    # Sdg = e^{-iπ/4} · Rz(-π/2): equal up to the global phase the rule drops.
    got = _unitary(out, 1)
    want = np.diag([1, -1j])
    overlap = np.trace(want.conj().T @ got) / 2
    assert abs(abs(overlap) - 1) < 1e-12
