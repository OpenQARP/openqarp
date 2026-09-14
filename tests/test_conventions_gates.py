"""Executable form of the gate contract in ``qarp_conventions.md`` §2–§5.

The conventions document is authoritative — when code and doc disagree, the
doc wins (§15).  These tests make that enforceable: every matrix, ordering
and identity the document states is checked against an oracle built here
from first principles with numpy/scipy, never against qarpx's own output
(§18).

LSB convention (§1): basis index = q0 + 2·q1 + 4·q2 …, so the operator for
``A`` on the higher qubit and ``B`` on the lower is ``kron(A, B)``.
"""

import numpy as np
import pytest
from scipy.linalg import expm

import qarpx as qx

# Generic angles: irrational-ish and distinct, so a swapped or dropped
# parameter cannot coincidentally satisfy the comparison.
TH, PHI, LAM, GAM = 0.7, 1.1, -0.4, 0.25

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
P0 = np.diag([1, 0]).astype(complex)
P1 = np.diag([0, 1]).astype(complex)


def _rx(t):
    return np.cos(t / 2) * I2 - 1j * np.sin(t / 2) * X


def _ry(t):
    return np.cos(t / 2) * I2 - 1j * np.sin(t / 2) * Y


def _rz(t):
    return np.diag([np.exp(-1j * t / 2), np.exp(1j * t / 2)])


def _phase(t):
    return np.diag([1, np.exp(1j * t)])


def _u(th, ph, la):
    """§2.5 U(θ, φ, λ), the OpenQASM 3 general single-qubit gate."""
    return np.array(
        [
            [np.cos(th / 2), -np.exp(1j * la) * np.sin(th / 2)],
            [np.exp(1j * ph) * np.sin(th / 2), np.exp(1j * (ph + la)) * np.cos(th / 2)],
        ]
    )


def _controlled(u):
    """§3.1 control on q0 (LSB), target on q1."""
    return np.kron(I2, P0) + np.kron(u, P1)


# One-qubit gate name → (builder args, analytic matrix).  Kept as plain data;
# blocks are constructed inside each test (never at module scope) so no qarpx
# object outlives the call.
ONE_QUBIT = {
    "x": ((), X),
    "y": ((), Y),
    "z": ((), Z),
    "h": ((), np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)),
    "s": ((), np.diag([1, 1j])),
    "sdg": ((), np.diag([1, -1j])),
    "t": ((), np.diag([1, np.exp(1j * np.pi / 4)])),
    "tdg": ((), np.diag([1, np.exp(-1j * np.pi / 4)])),
    # §2.6: SX = e^{iπ/4}·Rx(π/2) = ½[[1+i, 1−i], [1−i, 1+i]], the principal √X.
    "sx": ((), np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2),
    "sxdg": ((), np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]]) / 2),
    "id": ((), I2),
    "rx": ((TH,), _rx(TH)),
    "ry": ((TH,), _ry(TH)),
    "rz": ((TH,), _rz(TH)),
    "p": ((TH,), _phase(TH)),
    "u": ((TH, PHI, LAM), _u(TH, PHI, LAM)),
}

TWO_QUBIT_CONTROLLED = {
    "cx": ((), X),
    "cy": ((), Y),
    "cz": ((), Z),
    "ch": ((), np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)),
    "cs": ((), np.diag([1, 1j])),
    "csdg": ((), np.diag([1, -1j])),
    "csx": ((), np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2),
    "csxdg": ((), np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]]) / 2),
    "crx": ((TH,), _rx(TH)),
    "cry": ((TH,), _ry(TH)),
    "crz": ((TH,), _rz(TH)),
    "cp": ((TH,), _phase(TH)),
    "cu": ((TH, PHI, LAM, GAM), np.exp(1j * GAM) * _u(TH, PHI, LAM)),
}


def _unitary(build, n_qubits):
    """Full unitary of a freshly built block, including global phase."""
    block = qx.SimpleBlock(n_qubits, "conv")
    build(block)
    block.build()
    cmds = block.flatten()
    try:
        return np.array(qx.QarpSimulator().unitary_matrix(cmds, n_qubits))
    except RuntimeError:
        # Gates with no direct simulator kernel go through native lowering,
        # which §11 states is phase-exact.
        lowered = qx.Transpiler(qx.native_gateset()).transpile(cmds)
        return np.array(qx.QarpSimulator().unitary_matrix(lowered, n_qubits))


def _assert_close(got, want, label):
    err = np.max(np.abs(got - want))
    assert err < 1e-12, f"{label}: max|Δ| = {err:.3e}\ngot\n{got}\nwant\n{want}"


# ── §2 single-qubit gates ────────────────────────────────────────────────


@pytest.mark.parametrize("gate", sorted(ONE_QUBIT))
def test_single_qubit_matrix_matches_convention(gate):
    """Each §2 gate equals its documented matrix, global phase included."""
    args, oracle = ONE_QUBIT[gate]
    got = _unitary(lambda b: getattr(b, gate)(0, *args), 1)
    _assert_close(got, oracle, f"§2 {gate}")


def test_rz_equals_phase_up_to_the_documented_global_phase():
    """§2.4: Rz(θ) = e^{-iθ/2}·P(θ) — exactly, not merely up to phase."""
    rz = _unitary(lambda b: b.rz(0, TH), 1)
    p = _unitary(lambda b: b.p(0, TH), 1)
    _assert_close(rz, np.exp(-1j * TH / 2) * p, "§2.4 Rz vs P")


def test_sx_is_the_principal_square_root_of_x():
    """§2.6: SX² = X, and SX is scipy's principal matrix square root of X —
    an independent oracle for the e^{iπ/4} that distinguishes SX from Rx(π/2)."""
    from scipy.linalg import sqrtm

    sx = _unitary(lambda b: b.sx(0), 1)
    _assert_close(sx @ sx, X, "§2.6 SX² = X")
    _assert_close(sx, sqrtm(X), "§2.6 SX = sqrtm(X)")
    _assert_close(_unitary(lambda b: b.sxdg(0), 1), sx.conj().T, "§2.6 SXdg = SX†")


@pytest.mark.parametrize(("gate", "angle"), [("s", np.pi / 2), ("t", np.pi / 4)])
def test_s_and_t_are_exact_phase_gates(gate, angle):
    """§2.4: S ≡ P(π/2) and T ≡ P(π/4), including global phase."""
    got = _unitary(lambda b: getattr(b, gate)(0), 1)
    _assert_close(got, _unitary(lambda b: b.p(0, angle), 1), f"§2.4 {gate}")


# ── gate-set coverage: the document must describe every unitary GateType ──

# Non-unitary and structural types; §7/§8 govern these, not §2/§3.
NON_UNITARY = {"barrier", "custom", "measure", "reset"}


def _unitary_gate_names(arity):
    return {
        qx.gate_name(g).lower()
        for g in qx.GateType.__members__.values()
        if qx.gate_num_qubits(g) == arity and qx.gate_name(g).lower() not in NON_UNITARY
    }


def test_section_2_covers_every_unitary_single_qubit_gate_type():
    """§2 documents the whole unitary 1q gate set.

    Guards against a new GateType reaching the backend without a matching
    entry in the conventions document.
    """
    assert _unitary_gate_names(1) == set(ONE_QUBIT)


def test_section_3_covers_every_unitary_two_qubit_gate_type():
    """§3 documents the whole unitary 2q gate set: controlled gates (§3.1),
    symmetric gates (§3.2) and symmetric rotations (§3.3).  MCZ reports the
    minimum width of its variadic tuple and is governed by §5."""
    documented = set(TWO_QUBIT_CONTROLLED) | {
        "swap",
        "iswap",
        "iswapdg",
        "ecr",
        "rzz",
        "rxx",
        "ryy",
        "mcz",
    }
    assert _unitary_gate_names(2) == documented


# ── §3 two-qubit gates ───────────────────────────────────────────────────


@pytest.mark.parametrize("gate", sorted(TWO_QUBIT_CONTROLLED))
def test_controlled_gate_targets_q1_with_control_on_q0(gate):
    """§3.1: q0 is the control, q1 the target, in the LSB-first basis."""
    args, base = TWO_QUBIT_CONTROLLED[gate]
    got = _unitary(lambda b: getattr(b, gate)(0, 1, *args), 2)
    _assert_close(got, _controlled(base), f"§3.1 {gate}")


@pytest.mark.parametrize("gate", ["swap", "iswap", "iswapdg", "cz"])
def test_symmetric_two_qubit_gates_ignore_argument_order(gate):
    """§3.1/§3.2: these are symmetric in (q0, q1) at the unitary level."""
    forward = _unitary(lambda b: getattr(b, gate)(0, 1), 2)
    reversed_ = _unitary(lambda b: getattr(b, gate)(1, 0), 2)
    _assert_close(forward, reversed_, f"§3.2 {gate} symmetry")


@pytest.mark.parametrize("gate", ["swap", "iswap"])
def test_command_preserves_user_supplied_qubit_order(gate):
    """§3.2: symmetry is a property of the unitary, not of the IR — the
    command still stores the qubits in the order the caller gave them."""
    block = qx.SimpleBlock(2, "order")
    getattr(block, gate)(1, 0)
    block.build()
    assert [list(c.qubits) for c in block.flatten()] == [[1, 0]]


def test_ecr_is_argument_asymmetric_and_self_adjoint():
    """§3.2: ECR carries an implicit (control, target) structure, so swapping
    its arguments changes the unitary; but ECR² = I, so it is its own dagger."""
    forward = _unitary(lambda b: b.ecr(0, 1), 2)
    reversed_ = _unitary(lambda b: b.ecr(1, 0), 2)
    assert np.max(np.abs(forward - reversed_)) > 1e-6, "ECR must not be argument-symmetric"
    _assert_close(forward @ forward, np.eye(4), "§3.2 ECR² = I")


def test_iswapdg_is_the_adjoint_of_iswap():
    """§3.2/§11: iSWAPdg is a dedicated GateType equal to iSWAP†."""
    iswap = _unitary(lambda b: b.iswap(0, 1), 2)
    _assert_close(_unitary(lambda b: b.iswapdg(0, 1), 2), iswap.conj().T, "§3.2 iSWAPdg")


# ── §3.2 gates the simulator lowers on the fly ───────────────────────────
#
# iSWAP, iSWAPdg and ECR have no csim kernel; QarpSimulator lowers each
# through its canonical §11 decomposition instead.  That lowering has to be
# phase-exact and correctly oriented — ECR is argument-asymmetric, and a
# conjugated iSWAP would still satisfy the identities above — so the
# unitaries are pinned against closed forms built here from numpy (§18).

_LOWERED_2Q_ORACLES = {
    # ECR = (1/√2)(X0 − Y0 X1)
    "ecr": (np.kron(I2, X) - np.kron(X, Y)) / np.sqrt(2),
    # iSWAP = exp(i·π/4·(X0X1 + Y0Y1))
    "iswap": expm(1j * np.pi / 4 * (np.kron(X, X) + np.kron(Y, Y))),
    "iswapdg": expm(-1j * np.pi / 4 * (np.kron(X, X) + np.kron(Y, Y))),
}


@pytest.mark.parametrize("gate", sorted(_LOWERED_2Q_ORACLES))
def test_lowered_two_qubit_gate_matches_its_closed_form(gate):
    """§3.2: what the simulator applies is the documented unitary."""
    got = _unitary(lambda b, g=gate: getattr(b, g)(0, 1), 2)
    _assert_close(got, _LOWERED_2Q_ORACLES[gate], f"§3.2 {gate}")


@pytest.mark.parametrize("gate", sorted(_LOWERED_2Q_ORACLES))
def test_simulator_accepts_lowered_two_qubit_gates(gate):
    """These reach QarpSimulator directly — the caller need not pre-lower.

    Pins the dispatch itself: without it the gates raise "unsupported gate"
    and _unitary silently falls back to its own transpiler lowering, so
    the matrix test above would keep passing while the simulator rejected
    the gate outright.
    """
    block = qx.SimpleBlock(2, "native")
    getattr(block, gate)(0, 1)
    block.build()
    qx.QarpSimulator().unitary_matrix(block.flatten(), 2)


@pytest.mark.parametrize(("gate", "pauli"), [("rzz", "Z"), ("rxx", "X"), ("ryy", "Y")])
def test_symmetric_rotation_equals_matrix_exponential(gate, pauli):
    """§3.3: RPP(θ) = exp(-i θ/2 · P⊗P), oracle by scipy matrix exponential."""
    p = {"X": X, "Y": Y, "Z": Z}[pauli]
    oracle = expm(-1j * TH / 2 * np.kron(p, p))
    _assert_close(_unitary(lambda b: getattr(b, gate)(0, 1, TH), 2), oracle, f"§3.3 {gate}")


def test_rzz_is_diagonal_with_the_documented_eigenphases():
    """§3.3: e^{-iθ/2} on the +1 eigenstates of Z⊗Z (|00⟩, |11⟩) and e^{+iθ/2}
    on the -1 eigenstates (|01⟩, |10⟩)."""
    oracle = np.diag(np.exp(-1j * TH / 2 * np.array([1, -1, -1, 1])))
    _assert_close(_unitary(lambda b: b.rzz(0, 1, TH), 2), oracle, "§3.3 RZZ diagonal")


def test_documented_canonical_decompositions_are_exact():
    """§3.3 states one canonical decomposition per symmetric rotation; each
    must reproduce the gate exactly, global phase included."""
    cases = [
        (
            "RZZ = CX·Rz(q1)·CX",
            lambda b: b.rzz(0, 1, TH),
            lambda b: (b.cx(0, 1), b.rz(1, TH), b.cx(0, 1)),
        ),
        (
            "RXX = H⊗H·RZZ·H⊗H",
            lambda b: b.rxx(0, 1, TH),
            lambda b: (b.h(0), b.h(1), b.rzz(0, 1, TH), b.h(0), b.h(1)),
        ),
        (
            "RYY = Rx(π/2)⊗Rx(π/2)·RZZ·Rx(-π/2)⊗Rx(-π/2)",
            lambda b: b.ryy(0, 1, TH),
            lambda b: (
                b.rx(0, np.pi / 2),
                b.rx(1, np.pi / 2),
                b.rzz(0, 1, TH),
                b.rx(0, -np.pi / 2),
                b.rx(1, -np.pi / 2),
            ),
        ),
    ]
    for label, gate, decomposition in cases:
        _assert_close(_unitary(gate, 2), _unitary(decomposition, 2), f"§3.3 {label}")


# ── §4 three-qubit gates ─────────────────────────────────────────────────


def test_ccx_treats_the_first_two_qubits_as_controls():
    """§4: CCX(c0, c1, t) — target is the last argument."""
    oracle = np.eye(8, dtype=complex)
    for col in range(8):
        if (col & 0b1) and (col & 0b10):
            oracle[:, col] = 0
            oracle[col ^ 0b100, col] = 1
    _assert_close(_unitary(lambda b: b.ccx(0, 1, 2), 3), oracle, "§4 CCX")


def test_cswap_swaps_its_last_two_qubits():
    """§4: CSWAP(c, q0, q1) — control is the first argument."""
    oracle = np.eye(8, dtype=complex)
    for col in range(8):
        if col & 0b1:
            b1, b2 = (col >> 1) & 1, (col >> 2) & 1
            oracle[:, col] = 0
            oracle[(col & 0b1) | (b2 << 1) | (b1 << 2), col] = 1
    _assert_close(_unitary(lambda b: b.cswap(0, 1, 2), 3), oracle, "§4 CSWAP")


# ── §5 multi-controlled gates ────────────────────────────────────────────


@pytest.mark.parametrize("width", [2, 3, 4, 5])
def test_mcz_phases_only_the_all_ones_state(width):
    """§5: MCZ applies -1 to the all-ones basis state (amplitude index 2^n-1).

    Checked exactly rather than up to phase: width ≥ 4 lowers ancilla-free via
    Barenco, which §5 requires to stay phase-exact on GPhase-carrying targets.
    """
    oracle = np.eye(2**width, dtype=complex)
    oracle[-1, -1] = -1
    got = _unitary(lambda b: b.mcz(list(range(width))), width)
    _assert_close(got, oracle, f"§5 MCZ width {width}")


# ── §11 canonical decompositions ─────────────────────────────────────────


def test_section_11_identities_hold_in_circuit_order():
    """§11's bullets read left-to-right in circuit (time) order — the leftmost
    factor is applied first.  Each identity must hold exactly, global phase
    included; the exactness of the *installed* rules is pinned separately in
    C++ (test_decomposition_table.cpp), so this test guards the document's own
    formulas."""
    cases = [
        (
            "P(θ) = GPhase(θ/2)·Rz(θ)",
            lambda b: b.p(0, TH),
            lambda b: (b.gphase(TH / 2), b.rz(0, TH)),
            1,
        ),
        (
            "U(θ,φ,λ) = GPhase((φ+λ)/2)·Rz(λ)·Ry(θ)·Rz(φ)",
            lambda b: b.u(0, TH, PHI, LAM),
            lambda b: (b.gphase((PHI + LAM) / 2), b.rz(0, LAM), b.ry(0, TH), b.rz(0, PHI)),
            1,
        ),
        (
            "CP(θ) = GPhase(θ/4)·Rz(θ/2,t)·CX·Rz(-θ/2,t)·CX·Rz(θ/2,c)",
            lambda b: b.cp(0, 1, TH),
            lambda b: (
                b.gphase(TH / 4),
                b.rz(1, TH / 2),
                b.cx(0, 1),
                b.rz(1, -TH / 2),
                b.cx(0, 1),
                b.rz(0, TH / 2),
            ),
            2,
        ),
        (
            "CU(θ,φ,λ,γ) = P(γ,c)·P((φ+λ)/2,c)·P((λ-φ)/2,t)·CX·U(-θ/2,0,-(φ+λ)/2)·CX·U(θ/2,φ,0)",
            lambda b: b.cu(0, 1, TH, PHI, LAM, GAM),
            lambda b: (
                b.p(0, GAM),
                b.p(0, (PHI + LAM) / 2),
                b.p(1, (LAM - PHI) / 2),
                b.cx(0, 1),
                b.u(1, -TH / 2, 0.0, -(PHI + LAM) / 2),
                b.cx(0, 1),
                b.u(1, TH / 2, PHI, 0.0),
            ),
            2,
        ),
        (
            "iSWAPdg = the iSWAP sequence reversed and daggered",
            lambda b: b.iswapdg(0, 1),
            lambda b: (b.h(1), b.cx(1, 0), b.cx(0, 1), b.h(0), b.sdg(1), b.sdg(0)),
            2,
        ),
    ]
    for label, gate, decomposition, n in cases:
        _assert_close(_unitary(gate, n), _unitary(decomposition, n), f"§11 {label}")


def test_controlling_a_u_produces_cu_with_zero_gamma():
    """§11: `ControlledBlock` on a `U` lifts to `CU(θ, φ, λ, γ=0)` — the
    global-phase slot starts empty, not folded into φ/λ."""
    inner = qx.SimpleBlock(1, "inner")
    inner.u(0, TH, PHI, LAM)
    inner.build()
    controlled = qx.ControlledBlock(inner, 1, [True])
    controlled.build()
    cmds = controlled.flatten()
    assert [qx.gate_name(c.gate) for c in cmds] == ["CU"]
    assert [p.value() for p in cmds[0].params] == [TH, PHI, LAM, 0.0]


def test_targets_without_gphase_drop_it_at_final_lowering():
    """§11: `decompose_gphase → []` is the per-target boundary for phase loss
    — cudaq's gate set cannot express a global phase, so it vanishes there."""
    block = qx.SimpleBlock(1, "gp")
    block.gphase(TH)
    block.build()
    assert qx.Transpiler(qx.cudaq_gateset()).transpile(block.flatten()) == []
