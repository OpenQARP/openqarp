"""§12.2 oracle tests for QASM2Absorber: foreign OpenQASM 2 → qarp.

The oracle is a file qarp did not write.  A circuit is built in qiskit, dumped
by ``qiskit.qasm2.dumps``, absorbed, and the resulting block's unitary compared
against qiskit's own ``Operator`` — neither endpoint is qarp's emitter, so this
cannot be satisfied by an emitter and absorber that agree on something wrong
(§18).  The emit → absorb round-trip is asserted too, but only *in addition*.

That matters most for the names with no direct qarp gate: ``ch`` becomes
``CU(π/2, 0, π, 0)`` and ``sx``/``sxdg`` become ``U`` plus a compensating
``GPhase`` — mappings whose correctness is exactly what qiskit's evaluation
checks here.
"""

import numpy as np
import pytest

import qarpx as qx
from qarp.absorb import QASM2Absorber
from qarp.blocks import SimpleBlock

pytest.importorskip("qiskit")

import qiskit.qasm2
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator

TH, PHI, LAM, GAM = 0.7, 1.1, -0.4, 0.25


def _unitary(commands, n_qubits):
    """qarp's own evaluation of an absorbed command list.

    Transpiled to the native set first: the simulator has no kernel for the
    composite gates (ECR, iSWAP, …) the absorber may produce.
    """
    native = qx.Transpiler(qx.native_gateset()).transpile(commands)
    return np.array(qx.QarpSimulator().unitary_matrix(native, n_qubits))


def _assert_absorbs_to(circuit):
    """qiskit circuit → its own QASM 2 → qarp → same unitary."""
    source = qiskit.qasm2.dumps(circuit)
    commands, n_qubits, _, symbols = qx.QASM2Absorber().absorb(source)
    assert symbols == [], "OpenQASM 2 has no symbolic parameters"
    assert n_qubits == circuit.num_qubits
    got = _unitary(commands, n_qubits)
    want = Operator(circuit).data
    err = float(np.abs(got - want).max())
    assert err < 1e-12, f"absorbed circuit differs from qiskit's own reading, max|Δ|={err:.3e}"


# ── Foreign files: the independent oracle ────────────────────────────────
# Parametrized over plain names, never over built objects: a qarpx object in a
# parametrize list is retained for the whole session and leaks at shutdown.

FOREIGN_CASES = [
    "spec-qelib1-1q",
    "spec-qelib1-2q",
    "controlled-hadamard",
    "sqrt-x-pair",
    "controlled-clifford-singles",
    "qiskit-extended-names",
    "general-single-qubit",
    "controlled-u-family",
    "three-qubit",
]


def _build_foreign(case):
    qc = QuantumCircuit(3)
    if case == "spec-qelib1-1q":
        qc.x(0), qc.y(0), qc.z(0), qc.h(0), qc.s(0), qc.sdg(0), qc.t(0), qc.tdg(0)
        qc.rx(TH, 0), qc.ry(TH, 0), qc.rz(TH, 0), qc.id(1)
    elif case == "spec-qelib1-2q":
        qc.cx(0, 1), qc.cy(0, 1), qc.cz(0, 1), qc.crz(TH, 0, 1)
    elif case == "controlled-hadamard":
        # `ch` is spec qelib1 and reads as CH (§3.1).
        qc.h(0), qc.ch(0, 1), qc.ch(1, 2)
    elif case == "sqrt-x-pair":
        # sx / sxdg read as SX / SXdg, phase included (§2.6).
        qc.sx(0), qc.sxdg(1), qc.sx(2), qc.sx(0)
    elif case == "controlled-clifford-singles":
        # qiskit dumps CSXGate().inverse() with a nested `mcphase` helper
        # definition, which the absorber's no-unknown-definitions contract
        # refuses; csxdg is covered by the direct-source test below instead.
        qc.cs(0, 1), qc.csdg(1, 2), qc.csx(0, 2)
    elif case == "qiskit-extended-names":
        # Not in the spec's 23 gates; a foreign file may still use them.
        qc.swap(0, 1), qc.cswap(0, 1, 2), qc.crx(TH, 0, 1), qc.cry(TH, 0, 1)
        qc.rxx(TH, 0, 1), qc.rzz(TH, 0, 1), qc.p(TH, 2), qc.cp(TH, 0, 2)
    elif case == "general-single-qubit":
        qc.u(TH, PHI, LAM, 0)
        qc.p(LAM, 1)
    elif case == "controlled-u-family":
        qc.cu(TH, PHI, LAM, GAM, 0, 1)
        qc.cp(TH, 1, 2)
    elif case == "three-qubit":
        qc.ccx(0, 1, 2), qc.cswap(0, 1, 2)
    else:  # pragma: no cover - guarded by the parametrize list
        raise AssertionError(case)
    return qc


@pytest.mark.parametrize("case", FOREIGN_CASES)
def test_foreign_qiskit_file_absorbs_to_the_same_unitary(case):
    _assert_absorbs_to(_build_foreign(case))


def test_multiple_quantum_registers_concatenate_in_declaration_order():
    """qarp has one flat index space; registers join it in declaration order."""
    a, b = QuantumRegister(2, "a"), QuantumRegister(1, "b")
    qc = QuantumCircuit(a, b)
    qc.h(a[0]), qc.cx(a[0], a[1]), qc.cx(a[1], b[0])
    _assert_absorbs_to(qc)

    commands, n_qubits, _, _ = qx.QASM2Absorber().absorb(qiskit.qasm2.dumps(qc))
    assert n_qubits == 3
    # b[0] is the third declared bit, so it is qarp qubit 2.
    assert list(commands[-1].qubits) == [1, 2]


# ── Classical registers, measurement, conditionals ───────────────────────


def test_multiple_classical_registers_and_whole_register_measure():
    source = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg meas[2];
creg flag[1];
h q[0];
barrier q;
measure q -> meas;
reset q[0];
"""
    commands, n_qubits, n_cbits, _ = qx.QASM2Absorber().absorb(source)
    assert (n_qubits, n_cbits) == (2, 3)

    kinds = [str(c.gate).rsplit(".", 1)[-1] for c in commands]
    assert kinds == ["H", "Barrier", "Measure", "Measure", "Reset"]
    # `barrier q;` covers the whole register.
    assert list(commands[1].qubits) == [0, 1]
    # `measure q -> meas;` expands to one Measure per bit, in index order.
    assert [(list(c.qubits), list(c.cbits)) for c in commands[2:4]] == [
        ([0], [0]),
        ([1], [1]),
    ]


def test_single_bit_condition_reads_back_as_one_condition_bit():
    """The emitter's per-bit `creg` layout is what makes this a 1-bit frame."""
    source = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c0[1];
creg c1[1];
measure q[0] -> c0[0];
if (c1 == 1) x q[1];
"""
    commands, _, n_cbits, _ = qx.QASM2Absorber().absorb(source)
    assert n_cbits == 2
    begin = commands[1]
    assert str(begin.gate).endswith("BranchBegin")
    assert list(begin.condition_bits) == [1]
    assert list(begin.condition_values) == [True]
    assert str(commands[-1].gate).endswith("BranchEnd")


def test_wide_register_condition_becomes_the_integers_bits():
    """§12.2: `if` compares a whole creg, so the frame is that integer's bits,
    bit 0 of the register being the least significant."""
    source = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[3];
if (c == 5) x q[0];
"""
    commands, _, n_cbits, _ = qx.QASM2Absorber().absorb(source)
    assert n_cbits == 3
    begin = commands[0]
    assert list(begin.condition_bits) == [0, 1, 2]
    assert list(begin.condition_values) == [True, False, True]  # 5 == 0b101


def test_condition_value_wider_than_its_register_is_rejected():
    source = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
if (c == 4) x q[0];
"""
    with pytest.raises(RuntimeError, match="wider than the register"):
        qx.QASM2Absorber().absorb(source)


# ── Gate definitions ─────────────────────────────────────────────────────


def test_emitted_prelude_definitions_are_skipped_and_resolved_by_name():
    """The emitter's own `gate` bodies round-trip without being parsed."""
    block = qx.SimpleBlock(2, "prelude")
    block.ecr(0, 1)
    block.iswap(0, 1)
    block.ryy(0, 1, TH)
    block.build()
    source = qx.QASM2Emitter().emit(block.flatten(), 2, "prelude")
    assert "gate ecr a, b" in source

    commands, n_qubits, _, _ = qx.QASM2Absorber().absorb(source)
    assert [str(c.gate).rsplit(".", 1)[-1] for c in commands] == ["ECR", "iSWAP", "RYY"]
    assert n_qubits == 2


def test_definition_of_an_unknown_name_is_rejected():
    source = """OPENQASM 2.0;
include "qelib1.inc";
gate mygate(theta) a, b { cx a, b; rz(theta) b; cx a, b; }
qreg q[2];
mygate(0.5) q[0], q[1];
"""
    with pytest.raises(RuntimeError, match="mygate"):
        qx.QASM2Absorber().absorb(source)


# ── Rejections ───────────────────────────────────────────────────────────

UNSUPPORTED = {
    "opaque": ("opaque custom a, b;\nqreg q[2];\n", "opaque"),
    "qasm3-input": ("input float[64] theta;\nqreg q[1];\n", "OpenQASM 3"),
    "qasm3-gphase": ("qreg q[1];\ngphase(0.5);\n", "OpenQASM 3"),
    "qasm3-ctrl": ("qreg q[3];\nctrl(2) @ z q[0], q[1], q[2];\n", "OpenQASM 3"),
    "qasm3-registers": ("qubit[2] q;\n", "OpenQASM 3"),
    "unknown-gate": ("qreg q[1];\nfrobnicate q[0];\n", "frobnicate"),
    "undeclared-register": ("qreg q[1];\nx r[0];\n", "undeclared"),
    "index-out-of-range": ("qreg q[1];\nx q[4];\n", "out of range"),
}


@pytest.mark.parametrize("case", sorted(UNSUPPORTED))
def test_unsupported_constructs_name_themselves(case):
    body, needle = UNSUPPORTED[case]
    source = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n' + body
    with pytest.raises(RuntimeError, match=needle):
        qx.QASM2Absorber().absorb(source)


def test_symbolic_identifier_in_an_expression_is_rejected():
    source = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
rx(theta) q[0];
"""
    with pytest.raises(RuntimeError, match="no symbolic parameters"):
        qx.QASM2Absorber().absorb(source)


# ── Round-trip (additional to the oracle above, never instead of it) ──────


def test_emit_absorb_emit_is_byte_identical():
    block = qx.SimpleBlock(3, "rt")
    block.h(0), block.rx(0, TH), block.p(1, LAM), block.swap(0, 1)
    block.cswap(0, 1, 2), block.crx(0, 1, TH), block.cry(0, 2, TH)
    block.crz(1, 2, TH), block.rzz(0, 1, TH), block.rxx(0, 1, PHI)
    block.ryy(0, 1, TH), block.ecr(0, 1), block.iswap(0, 1), block.iswapdg(0, 1)
    block.cu(0, 1, TH, PHI, LAM, GAM), block.u(2, TH, PHI, LAM)
    block.cp(0, 1, TH), block.ccx(0, 1, 2), block.cy(0, 1), block.cz(0, 1)
    block.build()

    first = qx.QASM2Emitter().emit(block.flatten(), 3, "rt")
    commands, n_qubits, _, _ = qx.QASM2Absorber().absorb(first)
    assert qx.QASM2Emitter().emit(commands, n_qubits, "rt") == first


def test_measure_and_conditional_round_trip():
    source = """// rt
OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c0[1];

h q[0];
measure q[0] -> c0[0];
if (c0 == 1) x q[1];
if (c0 == 0) h q[1];
"""
    commands, n_qubits, _, _ = qx.QASM2Absorber().absorb(source)
    assert qx.QASM2Emitter().emit(commands, n_qubits, "rt") == source


# ── Python surface ───────────────────────────────────────────────────────


def test_from_qasm2_builds_a_simple_block():
    block = SimpleBlock(2, name="src")
    block.h(0)
    block.cx(0, 1)
    block.build()

    rebuilt = SimpleBlock.from_qasm2(block.to_qasm2())
    assert isinstance(rebuilt, SimpleBlock)
    assert rebuilt.n_qubits == 2
    assert rebuilt.to_qasm2() == block.to_qasm2()


def test_absorber_source_name():
    assert QASM2Absorber().source_name() == "qasm2"
    assert qx.QASM2Absorber().source_name() == "qasm2"


def test_circuit_name_is_read_back_from_the_header_comment():
    block = SimpleBlock(1, name="my_circuit")
    block.h(0)
    block.build()
    assert SimpleBlock.from_qasm2(block.to_qasm2()).name == "my_circuit"


def test_sx_absorbs_to_its_own_gate_and_re_emits():
    """§12.2: `sx` reads as SX (phase included) and writes back out — the
    former U + GPhase asymmetry is gone now that SX is a GateType."""
    from qiskit.circuit.library import SXGate

    source = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nsx q[0];\n'
    commands, n_qubits, _, _ = qx.QASM2Absorber().absorb(source)
    assert [str(c.gate).rsplit(".", 1)[-1] for c in commands] == ["SX"]
    got = _unitary(commands, n_qubits)
    assert np.max(np.abs(got - SXGate().to_matrix())) < 1e-12

    out = qx.QASM2Emitter().emit(commands, n_qubits)
    assert "sx q[0];" in out
    again, _, _, _ = qx.QASM2Absorber().absorb(out)
    assert [str(c.gate).rsplit(".", 1)[-1] for c in again] == ["SX"]


def test_ch_and_controlled_clifford_names_absorb_to_their_own_gates():
    source = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n'
        "ch q[0], q[1];\ncs q[0], q[1];\ncsdg q[1], q[0];\ncsx q[0], q[1];\ncsxdg q[1], q[0];\n"
    )
    commands, _, _, _ = qx.QASM2Absorber().absorb(source)
    assert [str(c.gate).rsplit(".", 1)[-1] for c in commands] == [
        "CH",
        "CS",
        "CSdg",
        "CSX",
        "CSXdg",
    ]
    assert [list(c.qubits) for c in commands] == [[0, 1], [0, 1], [1, 0], [0, 1], [1, 0]]


def test_id_absorbs_to_the_identity_gate_and_u0_to_nothing():
    """§12.2: `id` is the explicit identity (Id GateType); `u0` is an idle of
    `length` cycles with no gate meaning and still contributes no command."""
    source = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nid q[0];\nu0(3) q[0];\nh q[0];\n'
    commands, _, _, _ = qx.QASM2Absorber().absorb(source)
    assert [str(c.gate).rsplit(".", 1)[-1] for c in commands] == ["Id", "H"]


def test_python_wrapper_wraps_parse_errors():
    """The wrapper re-raises with its own prefix so the source is obvious."""
    with pytest.raises(RuntimeError, match=r"QASM2Absorber: parse error:"):
        QASM2Absorber().absorb('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nnope q[0];\n')


def test_missing_header_comment_falls_back_to_a_default_name():
    """A foreign file has no `// <name>` line; the block still gets a name."""
    source = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nh q[0];\n'
    assert QASM2Absorber().absorb(source).name == "Circuit"
