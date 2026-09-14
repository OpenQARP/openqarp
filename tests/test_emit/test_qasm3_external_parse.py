"""§12 external-oracle tests: emitted OpenQASM 3 through qiskit's importer.

The in-repo QASM3 round-trip (emit → absorb) shares both endpoints with the
code under test, so it cannot catch a program that only *qarp* understands —
emitting a gate name ``stdgates.inc`` does not define produces exactly that.
Parsing with qiskit's strict importer is the independent check (§18); the
semantic tests then compare qiskit's evaluation of the program against
analytic matrices.

qiskit's ``Operator`` is little-endian like qarpx, so matrices compare
directly — the MSB reversal landmine is openfermion's, not qiskit's.

Skipped entirely when qiskit or qiskit_qasm3_import is not installed —
``qiskit.qasm3.loads`` needs the latter at call time (lazy), so guarding
only qiskit turns every test into a MissingOptionalLibraryError failure.
"""

import numpy as np
import pytest
from scipy.linalg import expm

import qarpx as qx

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_qasm3_import")

import qiskit.qasm3
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
    return qx.QASM3Emitter().emit(block.flatten(), n_qubits)


# Every §12 table row (Custom excepted — rejected by validate(), §12).
# Multi-bit `&&` conditions are excluded: qiskit's importer has no accepted
# spelling for them, single-bit conditions are covered below.
ROWS = {
    "1q-fixed": (
        lambda b: (b.x(0), b.y(0), b.z(0), b.h(0), b.s(0), b.sdg(0), b.t(0), b.tdg(0)),
        1,
        0,
    ),
    "1q-parametric": (lambda b: (b.rx(0, TH), b.ry(0, TH), b.rz(0, TH), b.p(0, TH)), 1, 0),
    "u-builtin": (lambda b: b.u(0, TH, PHI, LAM), 1, 0),
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
    "mcz": (lambda b: b.mcz([0, 1, 2]), 3, 0),
    "gphase": (lambda b: b.gphase(TH), 1, 0),
    "measure-reset": (lambda b: (b.h(0), b.measure(0, 0), b.reset(0)), 1, 1),
    "symbolic": (lambda b: b.rx(0, qx.symbol("theta")), 1, 0),
    "linear-param": (lambda b: b.rx(0, qx.Param.linear(2.0, "theta", 0.5)), 1, 0),
}


@pytest.mark.parametrize("row", sorted(ROWS))
def test_every_documented_row_parses_externally(row):
    build, n_qubits, n_cbits = ROWS[row]
    qiskit.qasm3.loads(_emit(build, n_qubits, n_cbits))


def test_conditional_region_parses_externally():
    """§12: BranchBegin/Else/End emit `if (c[i] == true) { … } else { … }` —
    the bool comparison is the form strict external parsers accept for a bit."""
    then_body = qx.SimpleBlock(2, "then")
    then_body.x(1)
    then_body.build()
    else_body = qx.SimpleBlock(2, "else")
    else_body.h(1)
    else_body.build()
    measured = qx.SimpleBlock(2, "m")
    measured.n_cbits = 2
    measured.h(0)
    measured.measure(0, 0)
    measured.build()
    program = qx.CompositeBlock(
        [measured, qx.ConditionalBlock([0], [True], then_body, else_body)], 2
    )
    program.build()
    prog = qx.QASM3Emitter().emit(program.flatten(), 2)
    assert "== true" in prog and "} else {" in prog
    qiskit.qasm3.loads(prog)


# ── Semantics of the emitted gate definitions ────────────────────────────
# ecr / iswap / rzz / rxx / ryy have no stdgates.inc symbol, so the emitter
# ships a `gate` definition for each.  qiskit evaluates those definitions —
# not qarpx's kernels — making this an external check that each definition
# is the documented gate, global phase included.

_RZX = lambda t: expm(-1j * t / 2 * np.kron(X, Z))  # noqa: E731 - local matrix shorthand

DEFINED_GATES = {
    "rzz": (lambda b: b.rzz(0, 1, TH), expm(-1j * TH / 2 * np.kron(Z, Z))),
    "rxx": (lambda b: b.rxx(0, 1, TH), expm(-1j * TH / 2 * np.kron(X, X))),
    "ryy": (lambda b: b.ryy(0, 1, TH), expm(-1j * TH / 2 * np.kron(Y, Y))),
    "iswap": (
        lambda b: b.iswap(0, 1),
        np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex),
    ),
    "iswapdg": (
        lambda b: b.iswapdg(0, 1),
        np.array([[1, 0, 0, 0], [0, 0, -1j, 0], [0, -1j, 0, 0], [0, 0, 0, 1]], dtype=complex),
    ),
    # OpenQASM stdgates definition; argument-asymmetric, so this also pins
    # that a maps to the first qubit.
    "ecr": (lambda b: b.ecr(0, 1), _RZX(-np.pi / 4) @ np.kron(I2, X) @ _RZX(np.pi / 4)),
}


@pytest.mark.parametrize("gate", sorted(DEFINED_GATES))
def test_emitted_definition_is_the_documented_gate(gate):
    build, want = DEFINED_GATES[gate]
    circuit = qiskit.qasm3.loads(_emit(build, 2))
    got = np.array(Operator(circuit).data)
    err = np.max(np.abs(got - want))
    assert err < 1e-12, (
        f"{gate}: qiskit evaluates the definition to a different matrix, max|Δ|={err:.3e}"
    )


# §2.6 / §3.1 names that ride on stdgates symbols or `inv @` / `ctrl @`
# modifiers: qiskit's reading of the modifier form must be the documented gate.
_SX = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _ctrl(u):
    p0, p1 = np.diag([1, 0]).astype(complex), np.diag([0, 1]).astype(complex)
    return np.kron(I2, p0) + np.kron(u, p1)


MODIFIER_GATES = {
    "sx": (lambda b: b.sx(0), _SX, 1),
    "sxdg": (lambda b: b.sxdg(0), _SX.conj().T, 1),
    "id": (lambda b: b.id(0), I2, 1),
    "ch": (lambda b: b.ch(0, 1), _ctrl(_H), 2),
    "cs": (lambda b: b.cs(0, 1), _ctrl(np.diag([1, 1j])), 2),
    "csdg": (lambda b: b.csdg(0, 1), _ctrl(np.diag([1, -1j])), 2),
    "csx": (lambda b: b.csx(0, 1), _ctrl(_SX), 2),
    "csxdg": (lambda b: b.csxdg(0, 1), _ctrl(_SX.conj().T), 2),
    # Control on the higher qubit: pins operand order through the modifier.
    "csx-reversed": (
        lambda b: b.csx(1, 0),
        np.kron(np.diag([1, 0]), I2) + np.kron(np.diag([0, 1]), _SX),
        2,
    ),
}


@pytest.mark.parametrize("gate", sorted(MODIFIER_GATES))
def test_modifier_form_is_the_documented_gate(gate):
    build, want, n_qubits = MODIFIER_GATES[gate]
    circuit = qiskit.qasm3.loads(_emit(build, n_qubits))
    got = np.array(Operator(circuit).data)
    err = np.max(np.abs(got - want))
    assert err < 1e-12, f"{gate}: qiskit reads the modifier form differently, max|Δ|={err:.3e}"


def test_definitions_absent_when_unused():
    prog = _emit(lambda b: (b.h(0), b.cx(0, 1)), 2)
    assert "gate " not in prog


def test_symbolic_input_becomes_a_parameter():
    """§10/§12: `input float[64] theta;` — qiskit surfaces it as a Parameter."""
    circuit = qiskit.qasm3.loads(_emit(lambda b: b.rx(0, qx.symbol("theta")), 1))
    assert [p.name for p in circuit.parameters] == ["theta"]
