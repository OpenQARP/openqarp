"""synthesize_clifford_t: analytic-Rz oracle, fold exactness, T-count law.

Oracle discipline (qarp_conventions §18): every synthesized sequence is
compared against the analytic ``Rz(θ) = diag(e^{-iθ/2}, e^{+iθ/2})`` of
§2.3 in *exact* operator norm — phase included (§13: these streams may sit
under control).  The simulator only multiplies the emitted H/S/T/X/GPhase
commands; the target matrix shares no code with the synthesis.
"""

import math

import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import SimpleBlock
from qarp.resources import Stage, estimate, synthesize_clifford_t

pytest.importorskip("pygridsynth")

CLIFFORD_T = {"H", "S", "T", "X", "Y", "Z", "Sdg", "Tdg", "CX", "CZ"}


def _unitary(cmds, n_qubits: int) -> np.ndarray:
    return np.asarray(qx.QarpSimulator().unitary_matrix(list(cmds), n_qubits))


def _rz(theta: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])


def _rz_cmds(theta: float) -> list:
    b = SimpleBlock(1)
    b.rz(0, theta)
    b.build()
    return list(b.flatten())


@pytest.mark.parametrize("theta", [0.3, -0.7, 1.234, math.pi / 3])
def test_synthesized_rz_within_epsilon_phase_exact(theta):
    eps = 1e-6
    out = synthesize_clifford_t(_rz_cmds(theta), epsilon=eps)
    dist = np.linalg.norm(_unitary(out, 1) - _rz(theta), 2)
    assert dist <= eps


def test_clifford_fold_is_exact_across_branches():
    """θ = θ_r + kπ/2 must land on the same residual sequence with an exactly
    compensated S^k / phase — the fold identity Rz(kπ/2) = e^{-ikπ/4}·S^k,
    including negative k and the k=4 → -identity wrap."""
    eps = 1e-6
    for k in (-3, -1, 1, 2, 4, 7):
        theta = 0.3 + k * math.pi / 2
        out = synthesize_clifford_t(_rz_cmds(theta), epsilon=eps)
        dist = np.linalg.norm(_unitary(out, 1) - _rz(theta), 2)
        assert dist <= eps, f"k={k}: {dist}"


@pytest.mark.parametrize("k", [0, 1, 2, 3, -1])
def test_clifford_angles_need_no_synthesis(k):
    """Multiples of π/2 are free: exact S-power + GPhase, zero T gates."""
    theta = k * math.pi / 2
    out = synthesize_clifford_t(_rz_cmds(theta), epsilon=1e-10)
    names = [qx.gate_name(c.gate) for c in out]
    assert "T" not in names and "H" not in names
    assert np.allclose(_unitary(out, 1), _rz(theta), atol=1e-14)


@pytest.mark.parametrize("j", [1, 3, -1, 5, 7, -3])
def test_odd_quarter_turns_are_exact_single_t(j):
    """Rz(jπ/4) for odd j is T^j up to phase — one T, no approximation.

    Oracle is the analytic Rz matrix at machine precision (1e-14), far below
    any gridsynth ε, so passing proves the exact path ran.  Folding only to
    π/2 sent these to gridsynth: O(log 1/ε) T gates and ε error for a
    rotation with an exact one-gate form.
    """
    theta = j * math.pi / 4
    out = synthesize_clifford_t(_rz_cmds(theta), epsilon=1e-10)
    names = [qx.gate_name(c.gate) for c in out]
    assert "H" not in names  # gridsynth sequences always carry H
    assert names.count("T") == 1
    assert np.allclose(_unitary(out, 1), _rz(theta), atol=1e-14)


def test_eighth_turns_still_need_synthesis():
    """π/8 has no exact Clifford+T form, so the π/4 fold must NOT claim it —
    the boundary of the exact path, asserted rather than assumed."""
    out = synthesize_clifford_t(_rz_cmds(math.pi / 8), epsilon=1e-6)
    names = [qx.gate_name(c.gate) for c in out]
    assert "H" in names  # went to gridsynth
    assert np.linalg.norm(_unitary(out, 1) - _rz(math.pi / 8), 2) <= 1e-6


def test_quarter_fold_is_exact_across_branches():
    """θ = θ_r + jπ/4 lands on the same residual with an exactly compensated
    T^j / phase, including negative j and the j=8 → identity wrap."""
    eps = 1e-6
    for j in (-5, -1, 1, 3, 8, 11):
        theta = 0.3 + j * math.pi / 4
        out = synthesize_clifford_t(_rz_cmds(theta), epsilon=eps)
        dist = np.linalg.norm(_unitary(out, 1) - _rz(theta), 2)
        assert dist <= eps, f"j={j}: {dist}"


def test_t_count_follows_ross_selinger_law():
    """T-count ≈ 3·log2(1/ε) + O(loglog) — the published scaling (Ross &
    Selinger 2016)."""
    for eps in (1e-4, 1e-8):
        out = synthesize_clifford_t(_rz_cmds(0.3), epsilon=eps)
        n_t = sum(1 for c in out if c.gate == qx.GateType.T)
        expected = 3 * math.log2(1 / eps)
        assert 0.5 * expected <= n_t <= expected + 25


def test_multi_qubit_stream_passthrough_and_width():
    """Non-Rz gates of the set pass through; the whole-stream unitary matches the
    analytic composition (LSB kron: qubit 0 innermost, §1)."""
    theta, eps = 0.9, 1e-6
    b = SimpleBlock(2)
    b.h(0)
    b.rz(1, theta)
    b.cx(0, 1)
    b.build()
    out = synthesize_clifford_t(b.flatten(), epsilon=eps)

    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    cx = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])  # q0=ctrl (§3.1)
    ident = np.eye(2)
    expected = cx @ np.kron(_rz(theta), ident) @ np.kron(ident, h)
    dist = np.linalg.norm(_unitary(out, 2) - expected, 2)
    assert dist <= eps


def test_output_is_clifford_t_pure():
    out = synthesize_clifford_t(_rz_cmds(0.3), epsilon=1e-4)
    for cmd in out:
        assert qx.gate_name(cmd.gate) in CLIFFORD_T | {"GPhase"}


def test_symbolic_angle_raises():
    from sympy import Symbol

    b = SimpleBlock(1)
    b.rz(0, Symbol("theta"))
    b.build()
    with pytest.raises(ValueError, match="bound angles"):
        synthesize_clifford_t(b.flatten(), epsilon=1e-4)


def test_gate_outside_clifford_t_rz_raises():
    b = SimpleBlock(1)
    b.rx(0, 0.3)
    b.build()
    with pytest.raises(ValueError, match="Clifford\\+T\\+Rz-format"):
        synthesize_clifford_t(b.flatten(), epsilon=1e-4)


@pytest.mark.parametrize("eps", [0.0, 1.0, -1e-3])
def test_bad_epsilon_raises(eps):
    with pytest.raises(ValueError, match="epsilon"):
        synthesize_clifford_t([], epsilon=eps)


# ── estimator integration ─────────────────────────────────────────────────


def _mixed_block() -> SimpleBlock:
    b = SimpleBlock(2)
    b.h(0)
    b.rz(0, 0.3)
    b.cx(0, 1)
    b.rz(1, 1.1)
    b.build()
    return b


def test_estimator_synthesized_stage():
    report = estimate(_mixed_block(), gateset=qx.clifford_t_rz_gateset(), synthesis_epsilon=1e-4)
    assert report.stages[-1] == Stage.SYNTHESIZED
    vec = report.final
    assert vec.provenance.synthesis == "gridsynth:eps=0.0001"
    assert "Rz" not in vec.op_histogram
    assert vec.t_count is not None and vec.t_count > 0  # countable again: no Rz left
    # pre-synthesis stage still reports the un-synthesized truth
    assert report[Stage.OPTIMIZED].t_count is None


def test_estimator_synthesis_requires_clifford_t_rz_gateset():
    with pytest.raises(ValueError, match="clifford_t_rz"):
        estimate(_mixed_block(), synthesis_epsilon=1e-4)
    with pytest.raises(ValueError, match="clifford_t_rz"):
        estimate(_mixed_block(), gateset=qx.clifford_t_gateset(), synthesis_epsilon=1e-4)
