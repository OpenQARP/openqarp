"""The emit capability contract — declarations are falsifiable (§18).

Row 1 (SDK-oracle): every gate an emitter declares in its gate set emits,
and the emitted one-gate circuit's unitary matches qarpx — judged by the
SDK's own unitary, not by this codebase.  Row 2: everything *not*
declared (gates outside the set, capability flags that are off) is
rejected by ``validate()``; the expected accept/reject table below is
transcribed from the per-SDK prose contract written before this code
existed.  Row 3 is explicitly *not* an oracle: a
differential cross-check that ``validate(cmds) is None`` ⟺ ``emit(cmds)``
does not raise, guarding declaration-vs-dispatch drift.

Everything is constructed inside tests (never at module scope — nanobind
leak landmine); parametrize carries plain strings only.
"""

import math
import random

import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import ConditionalBlock, SimpleBlock
from qarp.errors import CapabilityError
from qarp.utils import generate_random_circuit
from tests.test_emit.conftest import SDK_UNITARY, qarpx_unitary

# Angle chosen to avoid the symmetric values that mask sign/order bugs.
ANGLE = 0.61 * math.pi

SDK_TARGETS = ["qiskit", "pytket", "qulacs", "pennylane"]
ALL_TARGETS = SDK_TARGETS + ["qasm3", "qir"]

# Non-unitary / meta entries: emitted, but not comparable as a unitary.
META_GATES = {"Measure", "Reset", "Barrier", "GPhase"}


def _emitter(target: str):
    return {
        "qiskit": qx.QiskitEmitter,
        "pytket": qx.PytketEmitter,
        "qulacs": qx.QulacsEmitter,
        "pennylane": qx.PennylaneEmitter,
        "qasm3": qx.QASM3Emitter,
        "qir": qx.QIREmitter,
    }[target]()


def _skip_unless_sdk(target: str):
    if target in SDK_TARGETS:
        pytest.importorskip(target)


def _one_gate_block(gate_name: str) -> SimpleBlock:
    """A built one-gate circuit for every emittable GateType."""
    a, a2, a3 = ANGLE, 0.7 * ANGLE, 0.4 * ANGLE
    builders = {
        "X": (1, lambda b: b.x(0)),
        "Y": (1, lambda b: b.y(0)),
        "Z": (1, lambda b: b.z(0)),
        "H": (1, lambda b: b.h(0)),
        "S": (1, lambda b: b.s(0)),
        "Sdg": (1, lambda b: b.sdg(0)),
        "T": (1, lambda b: b.t(0)),
        "Tdg": (1, lambda b: b.tdg(0)),
        "SX": (1, lambda b: b.sx(0)),
        "SXdg": (1, lambda b: b.sxdg(0)),
        "Id": (1, lambda b: b.id(0)),
        "Rx": (1, lambda b: b.rx(0, a)),
        "Ry": (1, lambda b: b.ry(0, a)),
        "Rz": (1, lambda b: b.rz(0, a)),
        "P": (1, lambda b: b.p(0, a)),
        "U": (1, lambda b: b.u(0, a, a2, a3)),
        "CX": (2, lambda b: b.cx(0, 1)),
        "CY": (2, lambda b: b.cy(0, 1)),
        "CZ": (2, lambda b: b.cz(0, 1)),
        "SWAP": (2, lambda b: b.swap(0, 1)),
        "ECR": (2, lambda b: b.ecr(0, 1)),
        "iSWAP": (2, lambda b: b.iswap(0, 1)),
        "iSWAPdg": (2, lambda b: b.iswapdg(0, 1)),
        "CH": (2, lambda b: b.ch(0, 1)),
        "CS": (2, lambda b: b.cs(0, 1)),
        "CSdg": (2, lambda b: b.csdg(0, 1)),
        "CSX": (2, lambda b: b.csx(0, 1)),
        "CSXdg": (2, lambda b: b.csxdg(0, 1)),
        "CRx": (2, lambda b: b.crx(0, 1, a)),
        "CRy": (2, lambda b: b.cry(0, 1, a)),
        "CRz": (2, lambda b: b.crz(0, 1, a)),
        "CP": (2, lambda b: b.cp(0, 1, a)),
        "RZZ": (2, lambda b: b.rzz(0, 1, a)),
        "RXX": (2, lambda b: b.rxx(0, 1, a)),
        "RYY": (2, lambda b: b.ryy(0, 1, a)),
        # γ = 0 so the one-gate sweep stays inside every backend's CU
        # capability; non-zero γ is a dedicated capability scenario below.
        "CU": (2, lambda b: b.cu(0, 1, a, a2, a3, 0.0)),
        "CCX": (3, lambda b: b.ccx(0, 1, 2)),
        "CSWAP": (3, lambda b: b.cswap(0, 1, 2)),
        "MCZ": (3, lambda b: b.mcz(0, 1, 2)),
        "GPhase": (1, lambda b: b.gphase(a)),
        "Measure": (1, lambda b: b.measure(0, 0)),
        "Reset": (1, lambda b: b.reset(0)),
    }
    n, build = builders[gate_name]
    b = SimpleBlock(n)
    build(b)
    b.build()
    return b


def _declared_unitary_gate_names(target: str) -> list:
    gs = _emitter(target).gate_set()
    names = sorted(
        qx.gate_name(g)
        for g in gs.allowed
        if qx.gate_name(g) not in META_GATES and qx.gate_name(g) != "Custom"
    )
    return names


# ── Row 1: every declared gate emits, unitary judged by the SDK ──────────


@pytest.mark.parametrize("target", SDK_TARGETS)
def test_declared_gates_emit_and_match_sdk_unitary(target):
    _skip_unless_sdk(target)
    emitter = _emitter(target)
    for gate_name in _declared_unitary_gate_names(target):
        block = _one_gate_block(gate_name)
        cmds = block.flatten()
        circ = emitter.emit(cmds, block.n_qubits)
        U_sdk = SDK_UNITARY[target](circ, block.n_qubits)
        U_ref = qarpx_unitary(cmds, block.n_qubits)
        np.testing.assert_allclose(
            U_sdk,
            U_ref,
            atol=1e-10,
            err_msg=f"{target}: unitary mismatch for declared gate {gate_name}",
        )


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_declared_meta_gates_emit(target):
    # Measure / Reset / Barrier / GPhase have no unitary to compare; the
    # declaration is still falsifiable: emission must not raise.
    _skip_unless_sdk(target)
    emitter = _emitter(target)
    declared = {qx.gate_name(g) for g in emitter.gate_set().allowed}
    for gate_name in sorted(META_GATES & declared):
        if gate_name == "Barrier":
            cmds = [qx.Command(qx.GateType.Barrier, 0)]
            n = 1
        else:
            block = _one_gate_block(gate_name)
            cmds, n = block.flatten(), block.n_qubits
        emitter.emit(cmds, n)


# ── Row 2: everything not declared is rejected by validate() ─────────────


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_undeclared_gates_are_rejected(target):
    emitter = _emitter(target)
    declared = {qx.gate_name(g) for g in emitter.gate_set().allowed}
    all_names = set(qx.GateType.__members__) - {
        "BranchBegin",
        "BranchElse",
        "BranchEnd",
        "NUM_GATE_TYPES",
    }
    undeclared = sorted(all_names - declared)
    for gate_name in undeclared:
        if gate_name == "Custom":
            cmds = [qx.Command(qx.GateType.Custom, 0)]
        else:
            block = _one_gate_block(gate_name)
            cmds = block.flatten()
        inc = emitter.validate(cmds)
        assert inc is not None, f"{target}: undeclared {gate_name} passed validate()"
        assert gate_name in inc.reason or "Custom" in inc.reason


# Transcribed from the per-SDK prose contract:
# which capability scenarios each backend accepts.  Keys: see _SCENARIOS.
_PROSE_CONTRACT = {
    "qiskit": {
        "symbolic": True,
        "multi_symbol": False,
        "nonlinear": False,
        "distinct_cbit": True,
        "conditional": True,
        "conditional_2bit": False,
        "cu_gamma": True,
    },
    "pytket": {
        "symbolic": True,
        "multi_symbol": False,
        "nonlinear": False,
        "distinct_cbit": True,
        "conditional": True,
        "conditional_2bit": True,
        "cu_gamma": False,
    },
    "qulacs": {
        "symbolic": False,
        "multi_symbol": False,
        "nonlinear": False,
        "distinct_cbit": True,
        "conditional": False,
        "conditional_2bit": False,
        "cu_gamma": True,
    },
    "pennylane": {
        "symbolic": False,
        "multi_symbol": False,
        "nonlinear": False,
        "distinct_cbit": False,
        "conditional": False,
        "conditional_2bit": False,
        "cu_gamma": True,
    },
    "qasm3": {
        "symbolic": True,
        "multi_symbol": True,
        "nonlinear": True,
        "distinct_cbit": True,
        "conditional": True,
        "conditional_2bit": True,
        "cu_gamma": True,
    },
    "qir": {
        "symbolic": False,
        "multi_symbol": False,
        "nonlinear": False,
        "distinct_cbit": True,
        "conditional": True,
        "conditional_2bit": True,
        "cu_gamma": None,
    },
}


def _scenario_commands(scenario: str):
    if scenario == "symbolic":
        b = SimpleBlock(1)
        b.rx(0, qx.Param.symbol("theta"))
    elif scenario == "multi_symbol":
        b = SimpleBlock(1)
        b.rx(0, qx.Param.symbol("a") + qx.Param.symbol("b"))
    elif scenario == "nonlinear":
        # One symbol, but not c*x + d: the linear bridge cannot carry it.
        b = SimpleBlock(1)
        b.rx(0, qx.Param.symbol("t") * qx.Param.symbol("t"))
    elif scenario == "distinct_cbit":
        b = SimpleBlock(1)
        b.measure(0, 1)
    elif scenario == "conditional":
        body = SimpleBlock(1)
        body.x(0)
        b = ConditionalBlock(cbits=[0], values=[True], then_body=body)
    elif scenario == "conditional_2bit":
        body = SimpleBlock(1)
        body.x(0)
        b = ConditionalBlock(cbits=[0, 1], values=[True, False], then_body=body)
    elif scenario == "cu_gamma":
        b = SimpleBlock(2)
        b.cu(0, 1, ANGLE, 0.3, 0.2, 0.5)
    else:
        raise AssertionError(scenario)
    b.build()
    return b.flatten()


@pytest.mark.parametrize("target", ALL_TARGETS)
@pytest.mark.parametrize(
    "scenario",
    [
        "symbolic",
        "multi_symbol",
        "nonlinear",
        "distinct_cbit",
        "conditional",
        "conditional_2bit",
        "cu_gamma",
    ],
)
def test_capability_flags_match_prose_contract(target, scenario):
    expected_ok = _PROSE_CONTRACT[target][scenario]
    if expected_ok is None:
        pytest.skip(f"{target}: scenario {scenario} outside its gate set")
    emitter = _emitter(target)
    inc = emitter.validate(_scenario_commands(scenario))
    if expected_ok:
        assert inc is None, f"{target}/{scenario}: unexpectedly rejected — {inc}"
    else:
        assert inc is not None, f"{target}/{scenario}: unexpectedly accepted"


# ── Row 3: differential — validate() is None ⟺ emit() does not raise ────


def _corpus(seed: int):
    """A deterministic mixed corpus: random circuits plus hand-made edge
    circuits that exercise the reject paths of at least one backend."""
    random.seed(seed)
    gate_set = [
        qx.GateType.H,
        qx.GateType.X,
        qx.GateType.Y,
        qx.GateType.Z,
        qx.GateType.S,
        qx.GateType.Sdg,
        qx.GateType.T,
        qx.GateType.Tdg,
        qx.GateType.Rx,
        qx.GateType.Ry,
        qx.GateType.Rz,
        qx.GateType.CX,
        qx.GateType.CZ,
        qx.GateType.SWAP,
        qx.GateType.CRx,
        qx.GateType.CRy,
        qx.GateType.CRz,
    ]
    blocks = [generate_random_circuit(12, 3, gate_set)]
    for scenario in ("symbolic", "distinct_cbit", "conditional", "cu_gamma"):
        blocks.append(_scenario_commands(scenario))
    out = []
    for b in blocks:
        out.append((b.flatten(), b.n_qubits) if hasattr(b, "flatten") else (b, 3))
    return out


@pytest.mark.parametrize("target", ALL_TARGETS)
@pytest.mark.parametrize("seed", range(3))
def test_validate_iff_emit_succeeds(target, seed):
    _skip_unless_sdk(target)
    emitter = _emitter(target)
    for cmds, n_qubits in _corpus(seed):
        inc = emitter.validate(cmds)
        try:
            emitter.emit(cmds, n_qubits)
            emitted = True
        except CapabilityError:
            emitted = False
        assert (inc is None) == emitted, (
            f"{target}: validate said {inc}, emit {'succeeded' if emitted else 'raised'}"
        )


# ── Row 4: qulacs_gateset() names only gates qulacs actually accepts ─────


def test_qulacs_gateset_names_only_accepted_gates():
    qulacs = pytest.importorskip("qulacs")
    ctor = {
        "X": lambda g: g.X(0),
        "Y": lambda g: g.Y(0),
        "Z": lambda g: g.Z(0),
        "H": lambda g: g.H(0),
        "S": lambda g: g.S(0),
        "Sdg": lambda g: g.Sdag(0),
        "T": lambda g: g.T(0),
        "Tdg": lambda g: g.Tdag(0),
        "Rx": lambda g: g.RX(0, ANGLE),
        "Ry": lambda g: g.RY(0, ANGLE),
        "Rz": lambda g: g.RZ(0, ANGLE),
        "CX": lambda g: g.CNOT(0, 1),
        "CZ": lambda g: g.CZ(0, 1),
        "SWAP": lambda g: g.SWAP(0, 1),
        "Measure": lambda g: g.Measurement(0, 0),
    }
    # GPhase and Barrier are emitter-level concessions (silently dropped on
    # emission), not qulacs.gate symbols — asserted in the meta-gate test.
    emitter_level = {"GPhase", "Barrier"}
    declared = {qx.gate_name(g) for g in qx.qulacs_gateset().allowed}
    assert declared == set(ctor) | emitter_level
    for name in sorted(declared - emitter_level):
        ctor[name](qulacs.gate)  # constructing must not throw


# ── Row 9: can_emit_to() agrees with to_<sdk>() and needs no SDK ─────────


@pytest.mark.parametrize("target", SDK_TARGETS)
def test_can_emit_to_agrees_with_emit(target):
    _skip_unless_sdk(target)
    to_method = {
        "qiskit": "to_qiskit",
        "pytket": "to_pytket",
        "qulacs": "to_qulacs",
        "pennylane": "to_pennylane",
    }[target]
    for build in (
        lambda b: [b.h(0), b.cx(0, 1)],
        lambda b: b.rx(0, qx.Param.symbol("theta")),
        lambda b: b.cu(0, 1, ANGLE, 0.3, 0.2, 0.5),
    ):
        b = SimpleBlock(2)
        build(b)
        b.build()
        reason = b.can_emit_to(target)
        if reason is None:
            getattr(b, to_method)()
        else:
            with pytest.raises(CapabilityError):
                getattr(b, to_method)()


@pytest.mark.parametrize("target", SDK_TARGETS)
def test_can_emit_to_needs_no_sdk(target, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, target, None)
    b = SimpleBlock(2)
    b.h(0)
    b.cx(0, 1)
    b.build()
    assert b.can_emit_to(target) is None

    sym = SimpleBlock(1)
    sym.rx(0, qx.Param.symbol("theta"))
    sym.build()
    reason = sym.can_emit_to(target)
    if target in ("qulacs", "pennylane"):
        assert "symbolic" in reason
    else:
        assert reason is None
