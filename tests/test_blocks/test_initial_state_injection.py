"""Caller-supplied initial states: simulator bindings + block convenience.

Oracles are independent of the implementation: hand-computed amplitudes,
numpy ``kron`` products of gate-matrix literals (LSB ordering per §1 —
qubit ``q`` is bit ``q`` of the amplitude index, so a gate on qubit 0 is
the *rightmost* kron factor), and analytic Born probabilities.  The N-step
reinjection test is a declared round-trip, additional to the oracles.
"""

import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import SimpleBlock

RNG = np.random.default_rng(20260730)

# 2×2 gate-matrix literals (exp(-iθP/2) convention, §2).
I2 = np.eye(2, dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta):
    return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)]).astype(complex)


# CX with control qubit 0 (bit 0), target qubit 1 (bit 1), LSB indexing:
# swaps indices 1 (q0=1,q1=0) and 3 (q0=1,q1=1).
CX01 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)


def _embed_1q(u, q, n):
    """kron-embed a 1-qubit gate on qubit ``q`` (qubit 0 = rightmost factor)."""
    full = np.eye(1, dtype=complex)
    for k in reversed(range(n)):
        full = np.kron(full, u if k == q else I2)
    return full


def _random_state(n_qubits):
    dim = 2**n_qubits
    psi = RNG.normal(size=dim) + 1j * RNG.normal(size=dim)
    return psi / np.linalg.norm(psi)


# ── identity oracle ──────────────────────────────────────────────────────


def test_empty_block_returns_input_exactly():
    psi = _random_state(2)
    b = SimpleBlock(2)
    b.build()
    out = b.statevector(initial_state=psi)
    np.testing.assert_array_equal(out, psi)


def test_output_feeds_back_in_unchanged():
    """statevector output re-enters as initial_state with no conversion."""
    b = SimpleBlock(1)
    b.h(0)
    b.build()
    once = b.statevector()
    twice = b.statevector(initial_state=once)
    # H² = I: back to |0⟩.
    np.testing.assert_allclose(twice, [1.0, 0.0], atol=1e-12)


# ── analytic oracles ─────────────────────────────────────────────────────


def test_h_on_seeded_basis_state():
    # |q1=1, q0=0⟩ = e_2; H on qubit 1 → (e_0 − e_2)/√2.
    b = SimpleBlock(2)
    b.h(1)
    b.build()
    psi = np.zeros(4, dtype=complex)
    psi[2] = 1.0
    out = b.statevector(initial_state=psi)
    expected = np.array([1, 0, -1, 0], dtype=complex) / np.sqrt(2)
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_cx_on_seeded_plus_state_gives_bell():
    # (e_0 + e_1)/√2 = (|0⟩+|1⟩)_q0/√2 ⊗ |0⟩_q1; CX(0→1) → (e_0 + e_3)/√2.
    b = SimpleBlock(2)
    b.cx(0, 1)
    b.build()
    psi = np.array([1, 1, 0, 0], dtype=complex) / np.sqrt(2)
    out = b.statevector(initial_state=psi)
    expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_random_circuit_matches_dense_linear_algebra():
    psi = _random_state(3)
    b = SimpleBlock(3)
    b.h(0)
    b.rx(1, 0.7)
    b.cx(0, 1)
    b.rz(2, -1.1)
    b.ry(0, 0.3)
    b.build()

    u = _embed_1q(H, 0, 3)
    u = np.kron(I2, CX01) @ _embed_1q(_rx(0.7), 1, 3) @ u
    u = _embed_1q(_ry(0.3), 0, 3) @ _embed_1q(_rz(-1.1), 2, 3) @ u

    out = b.statevector(initial_state=psi)
    np.testing.assert_allclose(out, u @ psi, atol=1e-12)


# ── run: sampling from a seeded state ────────────────────────────────────


def test_run_from_seeded_basis_state_is_deterministic():
    b = SimpleBlock(1)
    b.measure(0, 0)
    b.build()
    psi = np.array([0, 1], dtype=complex)
    res = qx.QarpSimulator().run(b.flatten(), 1, 64, 5, initial_state=psi)
    assert dict(res.counts) == {1: 64}
    assert all(reg[0] for reg in res.cbit_history)


def test_run_distribution_matches_born_probabilities():
    psi = _random_state(2)
    n_shots = 20_000
    res = qx.QarpSimulator().run([], 2, n_shots, 11, initial_state=psi)
    probs = np.abs(psi) ** 2
    for outcome, p in enumerate(probs):
        freq = res.counts.get(outcome, 0) / n_shots
        sigma = np.sqrt(p * (1 - p) / n_shots)
        assert abs(freq - p) < 5 * sigma + 1e-12


def test_trajectory_measure_then_reuse_per_branch_probabilities():
    # ψ = (1/2)|0⟩ + (√3/2)|1⟩; measure (p₁ = 3/4) then X: final outcome
    # flips the record, so counts(0) ≈ 3/4 and cbit c0 ≈ 3/4 True.
    psi = np.array([0.5, np.sqrt(3) / 2], dtype=complex)
    b = SimpleBlock(1)
    b.measure(0, 0)
    b.x(0)
    b.build()
    n_shots = 20_000
    res = qx.QarpSimulator().run(b.flatten(), 1, n_shots, 23, initial_state=psi)
    sigma = np.sqrt(0.75 * 0.25 / n_shots)
    assert abs(res.counts.get(0, 0) / n_shots - 0.75) < 5 * sigma
    c0_freq = sum(reg[0] for reg in res.cbit_history) / n_shots
    assert abs(c0_freq - 0.75) < 5 * sigma


# ── validation (contract) ────────────────────────────────────────────────


def test_wrong_length_raises_valueerror():
    b = SimpleBlock(2)
    b.build()
    with pytest.raises(ValueError, match="2\\^n_qubits"):
        b.statevector(initial_state=np.array([1, 0], dtype=complex))
    with pytest.raises(ValueError, match="length"):
        qx.QarpSimulator().run([], 2, 8, 0, initial_state=np.array([1, 0], dtype=complex))


def test_non_unit_norm_raises_valueerror():
    b = SimpleBlock(1)
    b.build()
    with pytest.raises(ValueError, match="normalised"):
        b.statevector(initial_state=np.array([0.5, 0.5], dtype=complex))
    with pytest.raises(ValueError, match="normalised"):
        qx.QarpSimulator().run([], 1, 8, 0, initial_state=np.array([0.5, 0.5], dtype=complex))


def test_block_convenience_coerces_real_input():
    """A real-valued (float) array is coerced to complex128, not rejected."""
    b = SimpleBlock(1)
    b.build()
    out = b.statevector(initial_state=np.array([0.0, 1.0]))
    np.testing.assert_array_equal(out, [0.0 + 0.0j, 1.0 + 0.0j])


# ── N-step reinjection (declared round-trip, additional) ─────────────────


def test_step_loop_reinjection_equals_single_long_circuit():
    psi0 = _random_state(2)
    step = SimpleBlock(2)
    step.ry(0, 0.4)
    step.cx(0, 1)
    step.rz(1, 0.9)
    step.build()
    cmds = step.flatten()

    psi = psi0
    for _ in range(4):
        psi = step.statevector(initial_state=psi)

    single = qx.QarpSimulator().statevector(list(cmds) * 4, 2, initial_state=psi0)
    np.testing.assert_allclose(psi, single, atol=1e-12)
