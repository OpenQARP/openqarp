"""Tests for QASM3Absorber.

This absorber wraps the C++ ``qx.QASM3Absorber``.
Tests are skipped gracefully if the C++ class is not compiled.

Key coverage:
  - Round-trip: emit(cmds, n) → absorb → emit must produce identical QASM3
  - Symbolic params are preserved through the round-trip
  - Conditional blocks (BranchBegin/Else/End) round-trip
  - All supported gate types
"""

import math

import numpy as np
import pytest

import qarpx as qx
from qarp.absorb import QASM3Absorber
from qarp.blocks import ConditionalBlock, SimpleBlock

# Skip entire module if the C++ absorber is not compiled yet
if not hasattr(qx, "QASM3Absorber"):
    pytest.skip(
        "qx.QASM3Absorber not available (C++ absorb module not compiled). "
        "Build with the absorb module to enable these tests.",
        allow_module_level=True,
    )


@pytest.fixture
def absorber() -> QASM3Absorber:
    return QASM3Absorber()


def _qasm3_emit(block: SimpleBlock) -> str:
    block.build()
    return qx.QASM3Emitter().emit(block.flatten(), block.n_qubits, block.name)


def _same_circuit(a: SimpleBlock, b: SimpleBlock) -> bool:
    """Block equality at the same default tolerance the SDK emitters use.

    ``Param::to_string`` emits shortest-round-trippable doubles
    (``std::to_chars``), so QASM3 angles round-trip bit-exactly: an angle like
    ``pi/3`` comes back as ``1.0471975511965976`` rather than a 6-significant-
    digit ``1.0472``.  No widened tolerance is needed: the ``commands_equal``
    default (1e-9) holds, aligning QASM with the SDK round-trips.
    """
    return a.n_qubits == b.n_qubits and qx.commands_equal(a.flatten(), b.flatten())


# ── Round-trip string equality ────────────────────────────────────────────────


def test_round_trip_basic(absorber):
    b = SimpleBlock(2, name="test")
    b.h(0).cx(0, 1).rz(1, math.pi / 4)
    b.build()

    qasm = qx.QASM3Emitter().emit(b.flatten(), 2, "test")
    result = absorber.absorb(qasm)
    re_emitted = qx.QASM3Emitter().emit(result.flatten(), result.n_qubits, result.name)

    assert qasm == re_emitted
    assert _same_circuit(result, b)


def test_round_trip_all_1q_gates(absorber):
    b = SimpleBlock(1, name="gates1q")
    b.x(0).y(0).z(0).h(0).s(0).sdg(0).t(0).tdg(0)
    b.rx(0, math.pi / 3).ry(0, math.pi / 4).rz(0, math.pi / 5).p(0, math.pi / 6)
    b.u(0, math.pi / 3, math.pi / 4, math.pi / 5)
    b.build()
    qasm = qx.QASM3Emitter().emit(b.flatten(), 1, "gates1q")
    result = absorber.absorb(qasm)
    assert qx.QASM3Emitter().emit(result.flatten(), result.n_qubits, result.name) == qasm
    assert _same_circuit(result, b)


def test_round_trip_2q_gates(absorber):
    b = SimpleBlock(2, name="gates2q")
    b.cx(0, 1).cy(0, 1).cz(0, 1).swap(0, 1)
    b.rzz(0, 1, math.pi / 4).rxx(0, 1, math.pi / 4).ryy(0, 1, math.pi / 4)
    b.crx(0, 1, math.pi / 3).cry(0, 1, math.pi / 3).crz(0, 1, math.pi / 3).cp(0, 1, math.pi / 3)
    b.build()
    qasm = qx.QASM3Emitter().emit(b.flatten(), 2, "gates2q")
    result = absorber.absorb(qasm)
    assert qx.QASM3Emitter().emit(result.flatten(), result.n_qubits, result.name) == qasm
    assert _same_circuit(result, b)


def test_round_trip_3q_gates(absorber):
    b = SimpleBlock(3, name="gates3q")
    b.ccx(0, 1, 2).cswap(0, 1, 2)
    b.build()
    qasm = qx.QASM3Emitter().emit(b.flatten(), 3, "gates3q")
    result = absorber.absorb(qasm)
    assert qx.QASM3Emitter().emit(result.flatten(), result.n_qubits, result.name) == qasm
    assert result == b  # exact match: no rotation angles to lose precision on


def test_round_trip_special_forms(absorber):
    """Special QASM3 syntax: MCZ -> ``ctrl(n) @ z``, iSWAPdg -> ``inv @ iswap``,
    plus iswap / ecr / gphase."""
    b = SimpleBlock(4, name="special")
    b.mcz([0, 1, 2, 3])
    b.iswapdg(0, 1)
    b.iswap(2, 3)
    b.ecr(0, 1)
    b.gphase(0.3)
    b.build()
    qasm = qx.QASM3Emitter().emit(b.flatten(), 4, "special")
    result = absorber.absorb(qasm)
    assert qx.QASM3Emitter().emit(result.flatten(), result.n_qubits, result.name) == qasm
    assert _same_circuit(result, b)


def test_round_trip_gateset_expansion_forms(absorber):
    """§2.6/§3.1 names: sx / id / ch as stdgates symbols, sxdg as ``inv @ sx``,
    cs / csdg / csx / csxdg as ``ctrl @`` forms — all read back to their own
    GateType, not to a lowered equivalent."""
    b = SimpleBlock(2, name="expansion")
    b.sx(0)
    b.sxdg(1)
    b.id(0)
    b.ch(0, 1)
    b.cs(0, 1)
    b.csdg(1, 0)
    b.csx(0, 1)
    b.csxdg(1, 0)
    b.build()
    qasm = qx.QASM3Emitter().emit(b.flatten(), 2, "expansion")
    result = absorber.absorb(qasm)
    assert qx.QASM3Emitter().emit(result.flatten(), result.n_qubits, result.name) == qasm
    assert _same_circuit(result, b)
    assert [c.gate for c in result.flatten()] == [c.gate for c in b.flatten()]


def test_round_trip_conditional_branch(absorber):
    """A conditional round-trips through the ``if (c[i] == v) { ... } else { ... }``
    branch parser (BranchBegin / BranchElse / BranchEnd)."""
    then_body = SimpleBlock(1, name="then")
    then_body.x(0)
    else_body = SimpleBlock(1, name="else")
    else_body.h(0)
    cond = ConditionalBlock([0], [True], then_body, else_body, name="cond")
    cond.build()
    qasm = qx.QASM3Emitter().emit(cond.flatten(), cond.n_qubits, cond.name)
    result = absorber.absorb(qasm)
    assert qx.QASM3Emitter().emit(result.flatten(), result.n_qubits, result.name) == qasm
    assert _same_circuit(result, cond)


def test_round_trip_symbolic_params(absorber):
    b = SimpleBlock(1, name="sym")
    b.rx(0, qx.Param.symbol("theta"))
    b.ry(0, qx.Param.linear(2.0, "phi", 0.5))
    b.build()
    qasm = qx.QASM3Emitter().emit(b.flatten(), 1, "sym")
    result = absorber.absorb(qasm)
    assert qx.QASM3Emitter().emit(result.flatten(), result.n_qubits, result.name) == qasm
    assert _same_circuit(result, b)

    # Symbols must be preserved
    free = result.free_symbols()
    assert "theta" in free
    assert "phi" in free


def test_round_trip_mid_circuit_measure(absorber):
    b = SimpleBlock(1, name="mid_meas")
    b.x(0)
    b.measure(0, 0)
    b.x(0)
    b.measure(0, 1)
    b.build()
    qasm = qx.QASM3Emitter().emit(b.flatten(), 1, "mid_meas")
    result = absorber.absorb(qasm)
    assert qx.QASM3Emitter().emit(result.flatten(), result.n_qubits, result.name) == qasm
    assert result == b
    measures = [
        (list(c.qubits), list(c.cbits)) for c in result.flatten() if c.gate == qx.GateType.Measure
    ]
    assert measures == [([0], [0]), ([0], [1])]


def test_symbolic_params_unitary_after_substitution(absorber):
    """Substituting the absorbed symbols to concrete values must reproduce
    the same unitary as building the gates directly with those values."""
    from sympy import Symbol

    b = SimpleBlock(1, name="sym")
    b.rx(0, qx.Param.symbol("theta"))
    b.ry(0, qx.Param.linear(2.0, "phi", 0.5))
    b.build()
    qasm = qx.QASM3Emitter().emit(b.flatten(), 1, "sym")
    result = absorber.absorb(qasm)

    theta_val, phi_val = math.pi / 3, math.pi / 5
    bound = result.set_symbols({Symbol("theta"): theta_val, Symbol("phi"): phi_val})

    concrete = SimpleBlock(1)
    concrete.rx(0, theta_val)
    concrete.ry(0, 2.0 * phi_val + 0.5)
    concrete.build()

    sim = qx.QarpSimulator()
    U_bound = np.array(sim.unitary_matrix(bound.flatten(), 1))
    U_concrete = np.array(sim.unitary_matrix(concrete.flatten(), 1))
    assert np.allclose(U_bound, U_concrete, atol=1e-10)


def test_round_trip_measurement(absorber):
    b = SimpleBlock(1, name="meas")
    b.h(0).measure(0, 0)
    b.build()
    qasm = qx.QASM3Emitter().emit(b.flatten(), 1, "meas")
    result = absorber.absorb(qasm)
    assert qx.QASM3Emitter().emit(result.flatten(), result.n_qubits, result.name) == qasm
    assert result == b


# ── Unitary correctness ───────────────────────────────────────────────────────


def test_unitary_preserved(absorber):
    b = SimpleBlock(2, name="u")
    b.h(0).cx(0, 1).rz(1, math.pi / 3)
    b.build()
    qasm = qx.QASM3Emitter().emit(b.flatten(), 2, "u")
    absorbed = absorber.absorb(qasm)
    absorbed.build()

    sim = qx.QarpSimulator()
    U_orig = np.array(sim.unitary_matrix(b.flatten(), 2))
    U_absorbed = np.array(sim.unitary_matrix(absorbed.flatten(), 2))
    assert np.allclose(U_orig, U_absorbed, atol=1e-10)


# ── Header handling (regression, 2026-09-01) ─────────────────────────────
# `skip_header` used to skip tokens until the first *recognized* keyword, and
# an unknown statement fell through to a silent skip-to-semicolon.  Together
# those let an OpenQASM 2 file run past its `qreg`, so the gates that followed
# were absorbed against n_qubits == 0 and the block only failed much later, at
# flatten(), with an IndexError pointing nowhere near the cause.


def test_openqasm2_source_is_rejected_by_name_not_silently_mis_parsed():
    source = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0],q[1];\n'
    with pytest.raises(RuntimeError, match="OpenQASM 2 syntax"):
        qx.QASM3Absorber().absorb(source)


@pytest.mark.parametrize("keyword", ["qreg", "creg", "opaque"])
def test_each_openqasm2_only_keyword_points_at_from_qasm2(keyword):
    source = f"OPENQASM 3.0;\n{keyword} c[2];\n"
    with pytest.raises(RuntimeError, match="from_qasm2"):
        qx.QASM3Absorber().absorb(source)


def test_unknown_statement_raises_instead_of_being_skipped():
    """The silent skip is what let malformed programs through."""
    with pytest.raises(RuntimeError, match="unknown statement or gate 'frobnicate'"):
        qx.QASM3Absorber().absorb("OPENQASM 3.0;\nfrobnicate q[0];\n")


def test_header_only_program_is_still_valid_and_empty():
    commands, n_qubits, n_cbits, symbols = qx.QASM3Absorber().absorb(
        'OPENQASM 3.0;\ninclude "stdgates.inc";\n'
    )
    assert (list(commands), n_qubits, n_cbits, symbols) == ([], 0, 0, [])


def test_out_of_range_wire_is_rejected_at_absorb_not_later():
    """QASM3 declares ``qubit[2] q`` and then uses ``q[5]``.  The parser does no
    range check, so reconstruction used to return a block with symbols=() that
    only failed at flatten(); the remap error now surfaces from absorb()."""
    src = "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\nx q[5];\n"
    with pytest.raises(IndexError, match=r"X acts on qubit 5 .* only 2 entries"):
        QASM3Absorber().absorb(src)
