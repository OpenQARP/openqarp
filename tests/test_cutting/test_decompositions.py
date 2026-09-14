"""Tests for QPD gate decompositions."""

import numpy as np
import pytest
import sympy

import qarpx as qx
from qarp.cutting import decompose_cx, decompose_rzz


def _gate_name(cmd) -> str:
    """Return the gate name string in lowercase for assertion comparison."""
    return cmd.gate.name.lower()


def _qubit(cmd) -> int:
    return cmd.qubits[0]


def _is_measure(cmd) -> bool:
    return isinstance(cmd, qx.Command) and cmd.gate == qx.GateType.Measure


def test_decompose_cx_count():
    """decompose_cx returns exactly 6 experiments with 6 coefficients."""
    prefix = []
    idx = [0, 1]
    exps, coeffs = decompose_cx(prefix, 2, idx, np.zeros(6, dtype=int))
    assert len(exps) == 6
    assert len(coeffs) == 6


def test_decompose_cx_coefficients():
    expected = [0.5, 0.5, 0.5, -0.5, 0.5, -0.5]
    _, coeffs = decompose_cx([], 2, [0, 1], np.zeros(6, dtype=int))
    assert all(np.isclose(float(a), b, atol=1e-9) for a, b in zip(coeffs, expected, strict=True))


def test_decompose_cx_gate_structure():
    """Check per-experiment gate sequences match the expected CX decomposition."""
    expected_gates = [
        [["sdg"], ["h", "sdg", "h"]],
        [["s"], ["h", "s", "h"]],
        [["sdg"], []],
        [["sdg"], ["h", "z", "h"]],
        [[], ["h", "sdg", "h"]],
        [["z"], ["h", "sdg", "h"]],
    ]

    exps, _ = decompose_cx([], 2, [0, 1], np.zeros(6, dtype=int))

    for circuit_idx, exp_cmds in enumerate(exps):
        q0_gates, q1_gates = [], []
        for cmd in exp_cmds:
            if not isinstance(cmd, qx.Command) or _is_measure(cmd):
                continue
            if _qubit(cmd) == 0:
                q0_gates.append(_gate_name(cmd))
            else:
                q1_gates.append(_gate_name(cmd))
        assert q0_gates == expected_gates[circuit_idx][0], f"exp {circuit_idx} q0"
        assert q1_gates == expected_gates[circuit_idx][1], f"exp {circuit_idx} q1"


def test_decompose_cx_measures_in_correct_experiments():
    """Experiments 1-2 have no QPD Measure; experiments 3-6 have exactly one."""
    exps, _ = decompose_cx([], 2, [0, 1], np.zeros(6, dtype=int))
    for i, cmds in enumerate(exps):
        meas_count = sum(1 for c in cmds if _is_measure(c))
        if i < 2:
            assert meas_count == 0, f"exp {i} should have no QPD Measure"
        else:
            assert meas_count == 1, f"exp {i} should have exactly 1 QPD Measure"


def test_decompose_cx_prefix_propagated():
    """Gates in the prefix appear at the start of every experiment."""
    prefix = [qx.Command(qx.GateType.H, 0), qx.Command(qx.GateType.H, 1)]
    exps, _ = decompose_cx(prefix, 2, [0, 1], np.zeros(6, dtype=int))
    for i, cmds in enumerate(exps):
        q_cmds = [c for c in cmds if isinstance(c, qx.Command)]
        assert q_cmds[0].gate == qx.GateType.H and q_cmds[0].qubits == [0], f"exp {i} prefix[0]"
        assert q_cmds[1].gate == qx.GateType.H and q_cmds[1].qubits == [1], f"exp {i} prefix[1]"


def test_decompose_cx_qpd_cbit_index():
    """QPD Measure in experiments 3-6 uses the cbit index from idx_to_be_used_bit."""
    exps, _ = decompose_cx([], 2, [0, 1], np.full(6, 3, dtype=int))
    for i in range(2, 6):
        meas_cmds = [c for c in exps[i] if _is_measure(c)]
        assert len(meas_cmds) == 1
        assert meas_cmds[0].cbits == [3], f"exp {i} QPD cbit should be 3"


def test_decompose_rzz_count():
    param = np.random.uniform(0.1, np.pi)
    exps, coeffs = decompose_rzz([], 2, [0, 1], np.zeros(6, dtype=int), param)
    assert len(exps) == 6
    assert len(coeffs) == 6


def test_decompose_rzz_coefficients():
    param = np.random.uniform(0.1, np.pi)
    parameter_rad = -param  # the negated angle used in QPD formula
    expected = [
        sympy.cos(parameter_rad / 2.0) ** 2,
        sympy.sin(parameter_rad / 2.0) ** 2,
        (-1) * sympy.sin(parameter_rad) / 2.0,
        sympy.sin(parameter_rad) / 2.0,
        (-1) * sympy.sin(parameter_rad) / 2.0,
        sympy.sin(parameter_rad) / 2.0,
    ]
    _, coeffs = decompose_rzz([], 2, [0, 1], np.zeros(6, dtype=int), param)
    for i, (got, exp) in enumerate(zip(coeffs, expected, strict=True)):
        assert abs(float(got) - float(exp)) < 1e-9, f"coeff {i}: {got} != {exp}"


def test_decompose_rzz_gate_structure():
    expected_gates = [
        [[], []],
        [["z"], ["z"]],
        [[], ["s"]],
        [[], ["sdg"]],
        [["s"], []],
        [["sdg"], []],
    ]
    param = np.random.uniform(0.1, np.pi)
    exps, _ = decompose_rzz([], 2, [0, 1], np.zeros(6, dtype=int), param)
    for circuit_idx, exp_cmds in enumerate(exps):
        q0_gates, q1_gates = [], []
        for cmd in exp_cmds:
            if not isinstance(cmd, qx.Command) or _is_measure(cmd):
                continue
            if _qubit(cmd) == 0:
                q0_gates.append(_gate_name(cmd))
            else:
                q1_gates.append(_gate_name(cmd))
        assert q0_gates == expected_gates[circuit_idx][0], f"exp {circuit_idx} q0"
        assert q1_gates == expected_gates[circuit_idx][1], f"exp {circuit_idx} q1"


# ── Unsupported-gate boundary ────────────────────────────────────────────
# The five stubs are reachable only through the QPD registry, so nothing named
# the supported-gate boundary.  Pinning it here keeps a half-finished
# implementation (a stub that returns None instead of raising) from reaching
# the reconstructer, where it would surface as a silent coefficient mismatch.


@pytest.mark.parametrize("gate", ["cy", "cz", "crx", "cry", "crz"])
def test_unsupported_cut_gates_raise_not_implemented(gate):
    import qarp.cutting as cutting

    decompose = getattr(cutting, f"decompose_{gate}")
    with pytest.raises(NotImplementedError, match="not yet supported with circuit cutting"):
        decompose([], 2, [0, 1], np.zeros(6, dtype=int))
