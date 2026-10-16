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


# ── The sx_rz hardware target ────────────────────────────────────────────

SX_RZ_ANGLES = (0.7, -0.4, 2.9)


def _p(t):
    return np.diag([1, np.exp(1j * t)])


def _u(th, ph, la):
    return np.array(
        [
            [np.cos(th / 2), -np.exp(1j * la) * np.sin(th / 2)],
            [np.exp(1j * ph) * np.sin(th / 2), np.exp(1j * (ph + la)) * np.cos(th / 2)],
        ]
    )


# (builder method, analytic matrix, number of angles) — §2 tables.
SX_RZ_ONE_QUBIT = {
    "h": (lambda: H, 0),
    "s": (lambda: np.diag([1, 1j]), 0),
    "sdg": (lambda: np.diag([1, -1j]), 0),
    "t": (lambda: _p(math.pi / 4), 0),
    "tdg": (lambda: _p(-math.pi / 4), 0),
    "y": (lambda: Y, 0),
    "z": (lambda: Z, 0),
    "rx": (_rx, 1),
    "ry": (_ry, 1),
    "p": (_p, 1),
    "u": (_u, 3),
}

SX_RZ_ALLOWED = {
    qx.GateType.X,
    qx.GateType.SX,
    qx.GateType.Rz,
    qx.GateType.CX,
    qx.GateType.Measure,
    qx.GateType.Barrier,
}


def _one_qubit_command(gate, angles):
    b = qx.SimpleBlock(1, gate)
    getattr(b, gate)(0, *angles)
    b.build()
    return b.flatten()[0]


def _kron_lsb(mat, q, n=2):
    # qarp is LSB: qubit 0 is the least significant bit, i.e. the rightmost kron factor.
    factors = [I2] * n
    factors[n - 1 - q] = mat
    out = factors[0]
    for f in factors[1:]:
        out = np.kron(out, f)
    return out


def _phase_free_allclose(got, want, atol=1e-10):
    overlap = np.trace(want.conj().T @ got) / want.shape[0]
    assert abs(abs(overlap) - 1) < atol, f"not unitarily equal: |overlap| = {abs(overlap)}"
    np.testing.assert_allclose(got * np.conj(overlap / abs(overlap)), want, atol=atol)


def test_sx_rz_gateset_shape():
    gs = qx.sx_rz_gateset()
    assert gs.name == "sx_rz"
    assert gs.rules == "sx_rz"
    assert gs.allowed == SX_RZ_ALLOWED


@pytest.mark.parametrize("gate", sorted(SX_RZ_ONE_QUBIT))
def test_sx_rz_phase_exact(gate):
    """Each override is exact, GPhase included: admit GPhase in a copy of the
    target so the compensating phase survives, then compare on the nose."""
    from itertools import product

    matrix, n_angles = SX_RZ_ONE_QUBIT[gate]
    gs = qx.sx_rz_gateset()
    gs.allowed = gs.allowed | {qx.GateType.GPhase}
    t = qx.Transpiler(gs)
    for angles in product(SX_RZ_ANGLES, repeat=n_angles):
        out = t.transpile([_one_qubit_command(gate, angles)])
        assert {c.gate for c in out} <= {
            qx.GateType.X,
            qx.GateType.SX,
            qx.GateType.Rz,
            qx.GateType.GPhase,
        }
        np.testing.assert_allclose(_unitary(out, 1), matrix(*angles), atol=1e-12)


def test_sx_rz_target_closure():
    """A sweep over the native single- and two-qubit gates lands in the target
    with the GPhase dropped, equal to the analytic unitary up to global phase."""
    th, ph, la = SX_RZ_ANGLES
    b = SimpleBlock(2, name="sweep")
    b.h(0).s(1).t(0).y(1).z(0).rx(0, th).ry(1, ph).p(0, la).u(1, th, ph, la)
    b.sdg(0).tdg(1).cz(0, 1).swap(0, 1).cy(1, 0).sx(0).sxdg(1)
    b.build()

    cz = np.diag([1, 1, 1, -1]).astype(complex)
    swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
    p0, p1 = np.diag([1, 0]).astype(complex), np.diag([0, 1]).astype(complex)
    cy_1_0 = np.kron(p0, I2) + np.kron(p1, Y)  # control = qubit 1 (high bit), target = qubit 0
    sx = np.exp(1j * math.pi / 4) * _rx(math.pi / 2)
    steps = [
        _kron_lsb(H, 0),
        _kron_lsb(np.diag([1, 1j]), 1),
        _kron_lsb(_p(math.pi / 4), 0),
        _kron_lsb(Y, 1),
        _kron_lsb(Z, 0),
        _kron_lsb(_rx(th), 0),
        _kron_lsb(_ry(ph), 1),
        _kron_lsb(_p(la), 0),
        _kron_lsb(_u(th, ph, la), 1),
        _kron_lsb(np.diag([1, -1j]), 0),
        _kron_lsb(_p(-math.pi / 4), 1),
        cz,
        swap,
        cy_1_0,
        _kron_lsb(sx, 0),
        _kron_lsb(sx.conj().T, 1),
    ]
    want = np.eye(4, dtype=complex)
    for m in steps:
        want = m @ want

    out = qx.Transpiler(qx.sx_rz_gateset()).transpile(b.flatten())
    assert {c.gate for c in out} <= SX_RZ_ALLOWED
    _phase_free_allclose(_unitary(out, 2), want)


def test_sx_rz_routed_device():
    """`compile_for_device` on a directed line with the sx_rz target: the
    routed program re-expressed in logical order equals the unrouted unitary
    up to phase (§14), and every emitted CX respects the edge direction."""
    from qarp.devices import Device

    arch = qx.Architecture(3, [(0, 1), (1, 2)], "line")
    dev = Device(3, architecture=arch, gate_set=qx.sx_rz_gateset(), directedness=True)
    b = SimpleBlock(3, name="routed")
    b.h(0).cx(0, 2).cz(2, 1).ry(2, 0.7).cx(1, 0)
    b.build()

    out = qx.compile_for_device(b.flatten(), 3, dev)
    assert {c.gate for c in out.commands} <= SX_RZ_ALLOWED
    qx.assert_directions(out.commands, arch)

    n = 3
    initial = list(out.initial_logical_to_physical)
    final = list(out.final_logical_to_physical)

    def basis(l2p):
        return [sum(((j >> q) & 1) << l2p[q] for q in range(n)) for j in range(2**n)]

    u_physical = _unitary(list(out.commands), n)
    u_logical = u_physical[np.ix_(basis(final), basis(initial))]
    _phase_free_allclose(u_logical, _unitary(b.flatten(), n))
