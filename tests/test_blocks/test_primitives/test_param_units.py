"""Parameter-unit convention: user-facing symbol values are RADIANS.

Locks the QAOA blocks (CostOperatorBlock, MixedOperatorBlock) to radians
rather than half-turns, plus an HEABlock control.  A symbol value ``v`` must produce exactly the rotation the block's
docstring states with ``v`` in radians — each test compares the full unitary
against a ``scipy.linalg.expm`` reference, so a reintroduced ×π factor fails
loudly.  (TrotterAnsatzBlock / UCCBlock / RSPBlock radians semantics are
pinned the same way in their own test modules.)
"""

import networkx as nx
import numpy as np
from scipy.linalg import expm
from sympy import Symbol

import qarpx as qx
from qarp.blocks import CostOperatorBlock, HEABlock, MixedOperatorBlock

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0 + 0j, -1.0])
I2 = np.eye(2)

V = 0.731  # arbitrary radian value, deliberately not a rational multiple of π


def _unitary(block):
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


def test_cost_operator_edge_symbol_is_radians():
    """γ in radians: an edge of weight w emits rzz(w·γ) = exp(-i (w·γ/2) Z⊗Z)."""
    w = 1.7
    g = nx.Graph()
    g.add_edge(0, 1, weight=w)
    blk = CostOperatorBlock(2, g)
    blk.build()
    sub = blk.set_symbols({Symbol("gamma_0"): V}).build()
    ref = expm(-1j * (w * V / 2) * np.kron(Z, Z))  # ZZ is qubit-order symmetric
    assert np.linalg.norm(_unitary(sub) - ref) < 1e-10


def test_cost_operator_linear_term_symbol_is_radians():
    """γ in radians: a linear term of coefficient c emits rz(c·γ)."""
    c = 0.9
    g = nx.Graph()
    g.add_node(0)
    blk = CostOperatorBlock(1, g, linear_terms={((0, "Z"),): c})
    blk.build()
    sub = blk.set_symbols({Symbol("gamma_0"): V}).build()
    ref = expm(-1j * (c * V / 2) * Z)
    assert np.linalg.norm(_unitary(sub) - ref) < 1e-10


def test_mixed_operator_symbol_is_radians():
    """β in radians: rx(β) = exp(-i (β/2) X) on every qubit."""
    blk = MixedOperatorBlock(2)
    blk.build()
    sub = blk.set_symbols({Symbol("beta_0"): V}).build()
    ref = expm(-1j * (V / 2) * (np.kron(X, I2) + np.kron(I2, X)))
    assert np.linalg.norm(_unitary(sub) - ref) < 1e-10


def test_hea_symbol_is_radians():
    """Control: HEABlock was already radians — ry symbol v gives exp(-i (v/2) Y)."""
    blk = HEABlock(n_qubits=1, n_layers=1, real=True, linear=True, circular=False, use_cz=True)
    blk.build()
    sub = blk.set_symbols({Symbol("ry_0_0"): V}).build()
    ref = expm(-1j * (V / 2) * Y)
    assert np.linalg.norm(_unitary(sub) - ref) < 1e-10
