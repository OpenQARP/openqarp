"""Tests for RSPBlock — 2-qubit Real Symmetric Preserving gate
(Ibe et al., Phys. Rev. Research 4, 013173).

The block takes ``theta`` in radians (effective angle in
In qarpx LSB convention the closed-form matrix is:

    [[1,                  0,                   0,                   0],
     [0,  -cos(α - π/2),   sin(α + π/2),       0],
     [0,   sin(α + π/2),   cos(α - π/2),       0],
     [0,                  0,                   0,                   1]]

with ``α = θ`` (θ in radians).  Sign flip on the (1,1)/(2,2) diagonal block is purely
the bit-reversal between pytket MSB and qarpx LSB on the two-qubit basis.
"""

import numpy as np
import pytest
import sympy

import qarpx as qx
from qarp.blocks import RSPBlock


def _unitary(block):
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


def _expected_unitary(theta: float) -> np.ndarray:
    """Closed-form RSP matrix in qarpx LSB convention (effective angle α = θ)."""
    alpha = theta
    return np.array(
        [
            [1, 0, 0, 0],
            [0, -np.cos(alpha - np.pi / 2), np.sin(alpha + np.pi / 2), 0],
            [0, np.sin(alpha + np.pi / 2), np.cos(alpha - np.pi / 2), 0],
            [0, 0, 0, 1],
        ],
        dtype=complex,
    )


# ── Exact-matrix correctness ────────────────────────────────────────────


@pytest.mark.parametrize("theta", [0.0, 0.25, 0.5, 1 / np.pi, 0.75, 1.0])
def test_rsp_matches_closed_form(theta):
    block = RSPBlock(theta=theta).build()
    assert np.linalg.norm(_unitary(block) - _expected_unitary(theta)) < 1e-12


@pytest.mark.parametrize("theta", [0.0, 0.25, 0.5, np.pi / 4])
def test_rsp_unitary_is_unitary(theta):
    block = RSPBlock(theta=theta).build()
    U = _unitary(block)
    assert np.linalg.norm(U @ U.conj().T - np.eye(4)) < 1e-10


def test_rsp_n_qubits_is_2():
    block = RSPBlock(theta=0.3).build()
    assert block.n_qubits == 2


# ── Identity special case ────────────────────────────────────────────────


def test_rsp_zero_theta_is_swap():
    """At θ = 0 both Ry layers collapse to identity, leaving
    ``CX(0,1) · CX(1,0) · CX(0,1)`` — the textbook 3-CX SWAP decomposition."""
    block = RSPBlock(theta=0.0).build()
    U = _unitary(block)
    swap = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=complex,
    )
    assert np.linalg.norm(U - swap) < 1e-12


# ── Symbolic-θ path ──────────────────────────────────────────────────────


def test_rsp_symbolic_theta_builds_and_specialises_to_concrete_value():
    """Build with sympy Symbol, then bind to a concrete value via the
    ``set_symbols`` API and verify the resulting unitary matches the
    closed-form for that value."""
    t = sympy.Symbol("t")
    block = RSPBlock(theta=t).build()
    bound = block.set_symbols({t: 0.4})
    U = _unitary(bound)
    assert np.linalg.norm(U - _expected_unitary(0.4)) < 1e-10


def test_rsp_linear_symbolic_expression():
    """RSPBlock should accept linear expressions in a single symbol via
    ``Param.linear``."""
    t = sympy.Symbol("t")
    block = RSPBlock(theta=2 * t + 0.1).build()
    bound = block.set_symbols({t: 0.1})  # effective theta = 0.3
    U = _unitary(bound)
    assert np.linalg.norm(U - _expected_unitary(0.3)) < 1e-10


# ── Validation ───────────────────────────────────────────────────────────


def test_rsp_rejects_multi_symbol_expression():
    a, b = sympy.symbols("a b")
    with pytest.raises(ValueError, match="linear in one symbol"):
        RSPBlock(theta=a + b).build()
