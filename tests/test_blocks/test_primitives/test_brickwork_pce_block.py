"""Contract tests for BrickworkPCEBlock — brickwork PCE ansatz.

Parameter-count formula, entangler topology (even/odd tiling shared with
BrickworkEntanglingBlock/HEABlock), and an independent oracle (built from
plain numpy/scipy analytic Rx/Ry/Rz/Rxx matrices, never from qarpx's own
gate matrices) pinning the layer ordering and pair-to-symbol mapping (§18).
"""

import numpy as np
import pytest
from scipy.linalg import expm
from sympy import Symbol

import qarpx as qx
from qarp.blocks import BrickworkPCEBlock

# ── construction smoke ──────────────────────────────────────────────────────


@pytest.mark.parametrize("n_qubits", [2, 3, 5])
@pytest.mark.parametrize("n_layers", [1, 2])
def test_pce_constructs_and_builds(n_qubits, n_layers):
    block = BrickworkPCEBlock(n_qubits=n_qubits, n_layers=n_layers).build()
    assert block.is_built
    assert block.n_qubits == n_qubits
    cmds = block.flatten()
    assert any(cmd.is_parametric() for cmd in cmds)


# ── parameter count formula ─────────────────────────────────────────────────


def _expected_symbol_count(n_qubits: int, n_layers: int) -> int:
    """3 rotation layers (n_qubits each) + 2 even-pair + 1 odd-pair entangling
    sublayer per PCE layer, one symbol per entangling pair."""
    n_even = n_qubits // 2
    n_odd = (n_qubits - 1) // 2
    return n_layers * (3 * n_qubits + 2 * n_even + n_odd)


@pytest.mark.parametrize("n_qubits", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("n_layers", [1, 3])
def test_pce_symbol_count_matches_formula(n_qubits, n_layers):
    block = BrickworkPCEBlock(n_qubits=n_qubits, n_layers=n_layers)
    expected = _expected_symbol_count(n_qubits, n_layers)
    # Pre-build: regression for the symbols-clobbered-by-super bug (as in HEABlock).
    assert len(block.symbols) == expected
    block.build()
    assert len(block.symbols) == expected


def test_pce_symbols_distinct_across_layers():
    """Distinct variational parameters per layer must stay separately
    differentiable — a naming collision would silently fold two layers."""
    block = BrickworkPCEBlock(n_qubits=4, n_layers=2)
    assert len(set(block.symbols)) == len(block.symbols)


# ── entangler topology ──────────────────────────────────────────────────────


def _rxx_pairs(block):
    return [tuple(c.qubits) for c in block.flatten() if qx.gate_name(c.gate) == "RXX"]


def test_pce_entangler_pairs_4_qubits_single_layer():
    """One RXX gate per pair, per entangling sublayer: even pairs (0,1),(2,3),
    then odd pair (1,2), then even pairs again — Rx-Rxx0-Ry-Rxx1-Rz-Rxx2,
    even-then-odd-then-even as in HEA/BrickworkEntangling."""
    block = BrickworkPCEBlock(n_qubits=4, n_layers=1).build()
    assert _rxx_pairs(block) == [
        (0, 1),
        (2, 3),
        (1, 2),
        (0, 1),
        (2, 3),
    ]


def test_pce_entangler_touches_every_qubit():
    """Every qubit must appear in at least one entangling pair, or it is
    only ever single-qubit rotated — strictly weaker than requested."""
    for n_qubits in (3, 4, 5, 6, 7):
        block = BrickworkPCEBlock(n_qubits=n_qubits, n_layers=1).build()
        touched = {q for pair in _rxx_pairs(block) for q in pair}
        assert touched == set(range(n_qubits)), n_qubits


def test_pce_two_layers_repeats_the_pattern():
    block = BrickworkPCEBlock(n_qubits=4, n_layers=2).build()
    pairs = _rxx_pairs(block)
    assert len(pairs) == 10
    assert pairs[:5] == pairs[5:]


# ── independent oracle: layer ordering and pair-to-symbol mapping ──────────

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)


def _rx(t):
    return expm(-0.5j * t * X)


def _ry(t):
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    return expm(-0.5j * t * Y)


def _rz(t):
    Z = np.diag([1, -1]).astype(complex)
    return expm(-0.5j * t * Z)


def _rxx(t):
    return expm(-0.5j * t * np.kron(X, X))


def _embed(op: np.ndarray, low: int, n: int) -> np.ndarray:
    """Place *op* on qubits ``[low, low + log2(dim(op)))``, LSB convention
    (§1: qubit 0 = least significant, so the higher-index block is the left
    kron factor)."""
    k = int(np.log2(op.shape[0]))
    high = n - low - k
    return np.kron(np.eye(2**high, dtype=complex), np.kron(op, np.eye(2**low, dtype=complex)))


def _rot_layer(angles, n):
    out = np.eye(2**n, dtype=complex)
    for q, matrix in angles:
        out = _embed(matrix, q, n) @ out
    return out


def _oracle_unitary(n_qubits: int, params: dict) -> np.ndarray:
    """Independent reference for one BrickworkPCEBlock layer (n_layers=1),
    built from Rx/Ry/Rz/Rxx analytic matrices — never from qarpx's own gate
    matrices."""
    n = n_qubits
    even_pairs = [(2 * q, 2 * q + 1) for q in range(n // 2)]
    odd_pairs = [(2 * q + 1, 2 * q + 2) for q in range((n - 1) // 2)]

    u = np.eye(2**n, dtype=complex)
    u = _rot_layer([(q, _rx(params[f"pce_rx_0_{q}"])) for q in range(n)], n) @ u
    for c, _ in even_pairs:
        u = _embed(_rxx(params[f"pce_Rxx0_0_{c}"]), c, n) @ u
    u = _rot_layer([(q, _ry(params[f"pce_ry_0_{q}"])) for q in range(n)], n) @ u
    for c, _ in odd_pairs:
        u = _embed(_rxx(params[f"pce_Rxx1_0_{c}"]), c, n) @ u
    u = _rot_layer([(q, _rz(params[f"pce_rz_0_{q}"])) for q in range(n)], n) @ u
    for c, _ in even_pairs:
        u = _embed(_rxx(params[f"pce_Rxx2_0_{c}"]), c, n) @ u
    return u


@pytest.mark.parametrize("n_qubits", [2, 3, 4])
def test_pce_unitary_matches_independent_oracle(n_qubits):
    block = BrickworkPCEBlock(n_qubits=n_qubits, n_layers=1).build()
    rng = np.random.default_rng(0)
    values = {str(s): float(rng.uniform(-np.pi, np.pi)) for s in block.symbols}

    bound = block.set_symbols({Symbol(name): v for name, v in values.items()})
    got = bound.unitary_matrix()
    expected = _oracle_unitary(n_qubits, values)
    np.testing.assert_allclose(got, expected, atol=1e-10)
