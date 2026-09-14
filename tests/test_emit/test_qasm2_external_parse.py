"""§12.2 external-oracle tests: emitted OpenQASM 2 through qiskit's importer.

The in-repo QASM2 round-trip (emit → absorb) shares both endpoints with the
code under test, so it cannot catch a program that only *qarp* understands.
Parsing with qiskit's strict importer is the independent check (§18); the
semantic tests then compare qiskit's evaluation of the program against
analytic matrices.

That importer matters more here than it does for QASM 3.  ``qiskit.qasm2``
implements the *original spec's* ``qelib1.inc`` — 23 gates — and rejects the
names qiskit added later (``swap``, ``crx``, ``rzz``, ``p``, ``cp``, …).  Every
gate outside those 23 therefore needs an emitted ``gate`` definition, and these
tests are what proves each definition is the gate qarp means (§11 phase-exact,
not merely up to phase — a QASM 2 file may be read back and controlled).

qiskit's ``Operator`` is little-endian like qarpx, so matrices compare
directly — the MSB reversal landmine is openfermion's, not qiskit's.
"""

import numpy as np
import pytest
from scipy.linalg import expm

import qarp  # noqa: F401 - ABI check + CapabilityError registration
import qarpx as qx
from qarp.errors import CapabilityError

pytest.importorskip("qiskit")

import qiskit.qasm2
from qiskit.circuit.library import ECRGate
from qiskit.quantum_info import Operator

TH, PHI, LAM, GAM = 0.7, 1.1, -0.4, 0.25

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)


def _emit(build, n_qubits, n_cbits=0):
    block = qx.SimpleBlock(n_qubits, "external")
    if n_cbits:
        block.n_cbits = n_cbits
    build(block)
    block.build()
    return qx.QASM2Emitter().emit(block.flatten(), n_qubits)


# Every §12.2 table row that has a QASM 2 representation.  The three rejected
# constructs (symbolic, GPhase, MCZ) and Custom are covered separately below.
ROWS = {
    "1q-fixed": (
        lambda b: (b.x(0), b.y(0), b.z(0), b.h(0), b.s(0), b.sdg(0), b.t(0), b.tdg(0)),
        1,
        0,
    ),
    "1q-parametric": (lambda b: (b.rx(0, TH), b.ry(0, TH), b.rz(0, TH), b.p(0, TH)), 1, 0),
    "u-as-u3": (lambda b: b.u(0, TH, PHI, LAM), 1, 0),
    "sqrt-x-and-id": (lambda b: (b.sx(0), b.sxdg(0), b.id(0)), 1, 0),
    "controlled-clifford": (
        lambda b: (b.ch(0, 1), b.cs(0, 1), b.csdg(0, 1), b.csx(0, 1), b.csxdg(0, 1)),
        2,
        0,
    ),
    "2q-fixed": (lambda b: (b.cx(0, 1), b.cy(0, 1), b.cz(0, 1), b.swap(0, 1)), 2, 0),
    "ecr": (lambda b: b.ecr(0, 1), 2, 0),
    "iswap": (lambda b: b.iswap(0, 1), 2, 0),
    "iswapdg": (lambda b: b.iswapdg(0, 1), 2, 0),
    "2q-parametric": (
        lambda b: (b.crx(0, 1, TH), b.cry(0, 1, TH), b.crz(0, 1, TH), b.cp(0, 1, TH)),
        2,
        0,
    ),
    "cu": (lambda b: b.cu(0, 1, TH, PHI, LAM, GAM), 2, 0),
    "rzz": (lambda b: b.rzz(0, 1, TH), 2, 0),
    "rxx": (lambda b: b.rxx(0, 1, TH), 2, 0),
    "ryy": (lambda b: b.ryy(0, 1, TH), 2, 0),
    "3q": (lambda b: (b.ccx(0, 1, 2), b.cswap(0, 1, 2)), 3, 0),
    "measure-reset": (lambda b: (b.h(0), b.measure(0, 0), b.reset(0)), 1, 1),
}


@pytest.mark.parametrize("row", sorted(ROWS))
def test_every_documented_row_parses_externally(row):
    build, n_qubits, n_cbits = ROWS[row]
    qiskit.qasm2.loads(_emit(build, n_qubits, n_cbits))


# ── Semantics of the emitted gate definitions ────────────────────────────
# Only 23 gate names exist in the spec qelib1.inc, so the emitter ships a
# `gate` definition for every other name it writes.  qiskit evaluates those
# definitions — not qarpx's kernels — making this an external check that each
# definition is the documented gate, global phase included.

_RZX = lambda t: expm(-1j * t / 2 * np.kron(X, Z))  # noqa: E731 - local matrix shorthand
_SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
_ISWAP = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex)


def _ctrl(u):
    """Control on qubit 0 (LSB), target qubit 1 — the §3.1 ordering."""
    p0, p1 = np.diag([1, 0]).astype(complex), np.diag([0, 1]).astype(complex)
    return np.kron(I2, p0) + np.kron(u, p1)


def _u3(theta, phi, lam):
    return np.array(
        [
            [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
            [
                np.exp(1j * phi) * np.sin(theta / 2),
                np.exp(1j * (phi + lam)) * np.cos(theta / 2),
            ],
        ]
    )


# CSWAP with control on qubit 0 (LSB): swaps |011> and |101>.
_CSWAP = np.eye(8, dtype=complex)
_CSWAP[[3, 5]] = _CSWAP[[5, 3]]

_SX = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2

DEFINED_GATES = {
    "sx": (lambda b: b.sx(0), _SX, 1),
    "sxdg": (lambda b: b.sxdg(0), _SX.conj().T, 1),
    "cs": (lambda b: b.cs(0, 1), _ctrl(np.diag([1, 1j])), 2),
    "csdg": (lambda b: b.csdg(0, 1), _ctrl(np.diag([1, -1j])), 2),
    "csx": (lambda b: b.csx(0, 1), _ctrl(_SX), 2),
    "csxdg": (lambda b: b.csxdg(0, 1), _ctrl(_SX.conj().T), 2),
    "swap": (lambda b: b.swap(0, 1), _SWAP, 2),
    "cswap": (lambda b: b.cswap(0, 1, 2), _CSWAP, 3),
    "crx": (lambda b: b.crx(0, 1, TH), _ctrl(expm(-1j * TH / 2 * X)), 2),
    "cry": (lambda b: b.cry(0, 1, TH), _ctrl(expm(-1j * TH / 2 * Y)), 2),
    "rzz": (lambda b: b.rzz(0, 1, TH), expm(-1j * TH / 2 * np.kron(Z, Z)), 2),
    "rxx": (lambda b: b.rxx(0, 1, TH), expm(-1j * TH / 2 * np.kron(X, X)), 2),
    "ryy": (lambda b: b.ryy(0, 1, TH), expm(-1j * TH / 2 * np.kron(Y, Y)), 2),
    "iswap": (lambda b: b.iswap(0, 1), _ISWAP, 2),
    "iswapdg": (lambda b: b.iswapdg(0, 1), _ISWAP.conj().T, 2),
    # Argument-asymmetric, so this also pins that `a` maps to the first qubit.
    "ecr": (lambda b: b.ecr(0, 1), _RZX(-np.pi / 4) @ np.kron(I2, X) @ _RZX(np.pi / 4), 2),
    "cu": (
        lambda b: b.cu(0, 1, TH, PHI, LAM, GAM),
        np.kron(I2, np.diag([1, 0]))
        + np.exp(1j * GAM) * np.kron(_u3(TH, PHI, LAM), np.diag([0, 1])),
        2,
    ),
}


@pytest.mark.parametrize("gate", sorted(DEFINED_GATES))
def test_emitted_definition_is_the_documented_gate(gate):
    build, want, n_qubits = DEFINED_GATES[gate]
    circuit = qiskit.qasm2.loads(_emit(build, n_qubits))
    got = np.array(Operator(circuit).data)
    err = np.max(np.abs(got - want))
    assert err < 1e-12, (
        f"{gate}: qiskit evaluates the definition to a different matrix, max|Δ|={err:.3e}"
    )


def test_ecr_definition_matches_qiskits_own_gate():
    """§3.2: pinned against an external ECR, not only against qarp's convention."""
    got = np.array(Operator(qiskit.qasm2.loads(_emit(lambda b: b.ecr(0, 1), 2))).data)
    assert np.max(np.abs(got - ECRGate().to_matrix())) < 1e-12


def test_rzz_definition_is_phase_exact_not_merely_up_to_phase():
    """The qiskit-qelib1 body `cx; u1(θ); cx` is RZZ only up to e^{-iθ/2}.

    Emitting it would pass any up-to-phase check and break §11 exactness —
    which becomes a *relative* phase the moment the program is controlled.
    """
    got = np.array(Operator(qiskit.qasm2.loads(_emit(lambda b: b.rzz(0, 1, TH), 2))).data)
    wrong = np.diag([1, np.exp(1j * TH), np.exp(1j * TH), 1]).astype(complex)
    assert np.max(np.abs(got - expm(-1j * TH / 2 * np.kron(Z, Z)))) < 1e-12
    assert np.max(np.abs(got - wrong)) > 1e-3, "emitted rzz lost the e^{-iθ/2} factor"


def test_whole_circuit_matches_qarps_own_evaluation():
    """End-to-end: every gate at once, qiskit's reading vs qarp's simulator.

    Per-gate tests can all pass while the program as a whole is wrong (a
    mis-ordered prelude, a shadowed definition), so this pins the composition.
    """
    block = qx.SimpleBlock(3, "whole")
    block.h(0), block.rx(0, TH), block.p(1, LAM), block.swap(0, 1)
    block.cswap(0, 1, 2), block.crx(0, 1, TH), block.cry(0, 2, TH)
    block.crz(1, 2, TH), block.rzz(0, 1, TH), block.rxx(0, 1, PHI)
    block.ryy(0, 1, TH), block.ecr(0, 1), block.iswap(0, 1), block.iswapdg(0, 1)
    block.cu(0, 1, TH, PHI, LAM, GAM), block.u(2, TH, PHI, LAM)
    block.cp(0, 1, TH), block.ccx(0, 1, 2), block.cy(0, 1), block.cz(0, 1)
    block.build()
    cmds = block.flatten()

    from_qasm3 = np.array(Operator(qiskit.qasm2.loads(qx.QASM2Emitter().emit(cmds, 3))).data)
    native = qx.Transpiler(qx.native_gateset()).transpile(cmds)
    from_qarp = np.array(qx.QarpSimulator().unitary_matrix(native, 3))
    assert np.max(np.abs(from_qasm3 - from_qarp)) < 1e-12


# ── Registers and conditionals ───────────────────────────────────────────


def test_single_creg_when_no_conditionals():
    """§12.2: the conventional layout, which is what a QASM 2 consumer reads."""
    prog = _emit(lambda b: (b.h(0), b.measure(0, 0), b.measure(1, 1)), 2, 2)
    assert "creg c[2];" in prog
    assert "measure q[0] -> c[0];" in prog
    circuit = qiskit.qasm2.loads(prog)
    assert [r.name for r in circuit.cregs] == ["c"]


def _conditional_program():
    then_body = qx.SimpleBlock(2, "then")
    then_body.x(1)
    then_body.build()
    else_body = qx.SimpleBlock(2, "else")
    else_body.h(1)
    else_body.build()
    measured = qx.SimpleBlock(2, "m")
    measured.n_cbits = 1
    measured.h(0)
    measured.measure(0, 0)
    measured.build()
    # target_cbits aliases the branch onto the bit the measurement writes;
    # without it CompositeBlock offsets the child past that write (§8).
    conditional = qx.ConditionalBlock([0], [True], then_body, else_body)
    conditional.target_cbits = [0]
    program = qx.CompositeBlock([measured, conditional], 2)
    program.build()
    return program


def test_conditionals_use_per_bit_cregs_and_negate_the_else():
    """§12.2: `if` compares a whole creg, so a single-bit condition needs a
    single-bit register; `else` is the same register against the other value."""
    prog = qx.QASM2Emitter().emit(_conditional_program().flatten(), 2)
    assert "creg c0[1];" in prog
    assert "if (c0 == 1) x q[1];" in prog
    assert "if (c0 == 0) h q[1];" in prog
    assert "else" not in prog  # OpenQASM 2 has no else keyword

    circuit = qiskit.qasm2.loads(prog)
    guarded = [i.operation for i in circuit.data if i.operation.name == "if_else"]
    assert len(guarded) == 2
    assert {op.condition[1] for op in guarded} == {0, 1}
    assert {op.condition[0].name for op in guarded} == {"c0"}


def test_guarded_bodies_are_the_gates_they_guard():
    """The `if`/negated-`if` pair must carry the right *bodies*, not just parse.

    qiskit cannot evolve a statevector through control flow, so the oracle is
    qiskit's own reading of each guarded block: its condition tuple and the
    unitary of the operation it wraps, against analytic X and H.
    """
    circuit = qiskit.qasm2.loads(qx.QASM2Emitter().emit(_conditional_program().flatten(), 2))
    guarded = [i.operation for i in circuit.data if i.operation.name == "if_else"]
    assert len(guarded) == 2

    hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    want = {1: X, 0: hadamard}  # then-branch is x, else-branch is h
    for op in guarded:
        register, value = op.condition
        assert register.name == "c0"
        body = op.blocks[0]
        assert body.num_qubits == 1, "each guard wraps exactly one 1q operation"
        got = np.array(Operator(body).data)
        assert np.max(np.abs(got - want[value])) < 1e-12, (
            f"guard on c0 == {value} wraps the wrong gate"
        )


# ── Rejections (§12.2) ───────────────────────────────────────────────────
# Asserted against the conventions table, not against the implementation's
# own output: each of these has no OpenQASM 2 spelling at all.

REJECTED = {
    "symbolic": (lambda b: b.rx(0, qx.symbol("theta")), 1, "symbolic"),
    "linear-param": (lambda b: b.rx(0, qx.Param.linear(2.0, "theta", 0.5)), 1, "symbolic"),
    "gphase": (lambda b: b.gphase(TH), 1, "GPhase"),
    "mcz": (lambda b: b.mcz([0, 1, 2]), 3, "MCZ"),
}


@pytest.mark.parametrize("case", sorted(REJECTED))
def test_unrepresentable_constructs_raise_capability_error(case):
    build, n_qubits, needle = REJECTED[case]
    with pytest.raises(CapabilityError) as excinfo:
        _emit(build, n_qubits)
    assert needle in str(excinfo.value)


@pytest.mark.parametrize("case", sorted(REJECTED))
def test_can_emit_to_answers_without_emitting(case):
    """`can_emit_to` is the pre-flight form — same verdict, no exception."""
    build, n_qubits, needle = REJECTED[case]
    from qarp.blocks import SimpleBlock

    block = SimpleBlock(n_qubits, name="preflight")
    build(block)
    block.build()
    reason = block.can_emit_to("qasm2")
    assert reason is not None and needle in reason
    assert block.can_emit_to("qasm3") is None, "QASM 3 still represents it"


def test_nested_conditionals_are_rejected():
    """OpenQASM 2's `if` governs a single quantum operation, which cannot
    itself be an `if` — no EmitterCapabilities flag covers this, so the
    emitter checks it explicitly."""
    inner_body = qx.SimpleBlock(2, "inner")
    inner_body.x(1)
    inner_body.build()
    inner = qx.ConditionalBlock([1], [True], inner_body)
    inner.build()
    outer = qx.ConditionalBlock([0], [True], inner)
    outer.target_cbits = [0, 1]
    program = qx.CompositeBlock([outer], 2)
    program.build()
    # can_emit_to must agree with emit(): the pre-flight adds the same shape
    # check, since no EmitterCapabilities flag can express it.
    reason = qx.QASM2Emitter().validate(program.flatten()).reason
    assert "nested classical conditionals" in reason
    with pytest.raises(CapabilityError, match="nested classical conditionals"):
        qx.QASM2Emitter().emit(program.flatten(), 2)


def test_definitions_absent_when_unused():
    prog = _emit(lambda b: (b.h(0), b.cx(0, 1)), 2)
    assert "gate " not in prog


def test_rzz_definition_accompanies_rxx_and_ryy():
    """`rxx`/`ryy` bodies call `rzz`, so requesting one requests the other."""
    prog = _emit(lambda b: b.rxx(0, 1, TH), 2)
    assert "gate rzz(theta)" in prog
    assert prog.index("gate rzz(theta)") < prog.index("gate rxx(theta)")


def test_multiline_circuit_name_stays_a_comment():
    """Composite block names contain newlines; `// <name>` verbatim would emit
    lines that are not comments at all — an invalid program.

    Regression: this shipped broken in the QASM 3 emitter and was only masked
    by that absorber tolerating unknown statements (2026-09-01).
    """
    block = qx.SimpleBlock(2, "Layered\nHEA\n(n=1)_optimized")
    block.ry(0, TH)
    block.cx(0, 1)
    block.build()

    for emitter in (qx.QASM2Emitter(), qx.QASM3Emitter()):
        source = emitter.emit(block.flatten(), 2, block.name)
        header = source.split("OPENQASM")[0].splitlines()
        assert header and all(line.startswith("//") for line in header if line), (
            f"{emitter.target_name()}: header line is not a comment: {header}"
        )

    # And the strict external parser agrees, for the one it can read.
    qiskit.qasm2.loads(qx.QASM2Emitter().emit(block.flatten(), 2, block.name))
