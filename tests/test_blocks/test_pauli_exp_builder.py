"""Python-side smoke tests for the Pauli-exponential builder API.

Covers the bindings added in Phase 6.5:
  - ``qx.Pauli`` enum
  - ``qx.parse_pauli_string`` / ``qx.commutes``
  - ``Block.pauli_exp(pauli, angle)`` accepting str or list[Pauli]
  - ``Block.commuting_pauli_set_exp(paulis, angles)`` accepting list[str]

Reference unitaries are built via scipy's matrix exponential of
``Σ θᵢ Pᵢ`` in the computational basis.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

import qarpx as qx

# ── Helpers ──────────────────────────────────────────────────────────────


PAULI_MATRICES = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_matrix(p: str) -> np.ndarray:
    """Build the n-qubit Pauli matrix in qarpx LSB-first basis (qubit 0 innermost)."""
    m = np.array([[1.0 + 0j]])
    for q in p:
        m = np.kron(PAULI_MATRICES[q], m)
    return m


def reference_commuting_unitary(paulis: list[str], angles: list[float]) -> np.ndarray:
    """Reference U = exp(-i/2 · Σᵢ θᵢ Pᵢ) via direct matrix exponential."""
    n = len(paulis[0])
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    for p, theta in zip(paulis, angles, strict=True):
        H += theta * pauli_matrix(p)
    return scipy.linalg.expm(-0.5j * H)


def build_block_unitary(block: qx.Block) -> np.ndarray:
    """Build the full unitary by simulating the block's flattened commands."""
    if not block.is_built():
        block.build()
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


# ── Pauli enum + parse helpers ────────────────────────────────────────────


def test_pauli_enum_values():
    assert qx.Pauli.I.value == 0
    assert qx.Pauli.X.value == 1
    assert qx.Pauli.Y.value == 2
    assert qx.Pauli.Z.value == 3


def test_parse_pauli_string_round_trip():
    parsed = qx.parse_pauli_string("IXYZ")
    assert parsed == [qx.Pauli.I, qx.Pauli.X, qx.Pauli.Y, qx.Pauli.Z]


def test_parse_pauli_string_rejects_invalid():
    with pytest.raises(ValueError):
        qx.parse_pauli_string("XYW")


def test_commutes_predicate():
    assert qx.commutes(qx.parse_pauli_string("XX"), qx.parse_pauli_string("YY"))
    assert not qx.commutes(qx.parse_pauli_string("X"), qx.parse_pauli_string("Y"))


# ── Block.pauli_exp ──────────────────────────────────────────────────────


@pytest.mark.parametrize("pauli", ["X", "Y", "Z", "I"])
def test_pauli_exp_single_qubit(pauli):
    angle = 0.5
    b = qx.SimpleBlock(1)
    b.pauli_exp(pauli, angle)
    b.build()
    np.testing.assert_allclose(
        build_block_unitary(b),
        reference_commuting_unitary([pauli], [angle]),
        atol=1e-10,
    )


@pytest.mark.parametrize("pauli", ["XYZ", "ZZZ", "XXX", "YII"])
def test_pauli_exp_three_qubit(pauli):
    angle = 0.7
    b = qx.SimpleBlock(3)
    b.pauli_exp(pauli, angle)
    b.build()
    np.testing.assert_allclose(
        build_block_unitary(b),
        reference_commuting_unitary([pauli], [angle]),
        atol=1e-10,
    )


def test_pauli_exp_returns_block_for_chaining():
    b = qx.SimpleBlock(2)
    # Three chained Paulis on a single-line builder expression.
    result = b.pauli_exp("XX", 0.1).pauli_exp("YY", 0.2).pauli_exp("ZZ", 0.3)
    assert result is b


def test_pauli_exp_accepts_pauli_enum_list():
    b1 = qx.SimpleBlock(3)
    b2 = qx.SimpleBlock(3)
    b1.pauli_exp("XYZ", 0.4)
    b2.pauli_exp([qx.Pauli.X, qx.Pauli.Y, qx.Pauli.Z], 0.4)
    b1.build()
    b2.build()
    np.testing.assert_allclose(build_block_unitary(b1), build_block_unitary(b2))


# ── Block.commuting_pauli_set_exp ────────────────────────────────────────


def test_commuting_pauli_set_exp_bell_stabilisers():
    # {XX, YY, ZZ} on 2 qubits.
    paulis = ["XX", "YY", "ZZ"]
    angles = [0.4, -0.2, 0.7]
    b = qx.SimpleBlock(2)
    b.commuting_pauli_set_exp(paulis, angles)
    b.build()
    np.testing.assert_allclose(
        build_block_unitary(b),
        reference_commuting_unitary(paulis, angles),
        atol=1e-10,
    )


def test_commuting_pauli_set_exp_zz_family_n3():
    paulis = ["ZZI", "IZZ", "ZIZ"]
    angles = [0.3, 0.5, 0.1]
    b = qx.SimpleBlock(3)
    b.commuting_pauli_set_exp(paulis, angles)
    b.build()
    np.testing.assert_allclose(
        build_block_unitary(b),
        reference_commuting_unitary(paulis, angles),
        atol=1e-10,
    )


def test_commuting_pauli_set_exp_symbolic_round_trip():
    paulis = ["XX", "ZZ"]
    theta = 0.4
    phi = 0.9

    b = qx.SimpleBlock(2)
    b.commuting_pauli_set_exp(paulis, [qx.Param.symbol("theta"), qx.Param.symbol("phi")])
    b.build()
    substituted = qx.substitute_all(b.flatten(), {"theta": theta, "phi": phi})
    actual = np.array(qx.QarpSimulator().unitary_matrix(substituted, 2))
    np.testing.assert_allclose(
        actual, reference_commuting_unitary(paulis, [theta, phi]), atol=1e-10
    )


def test_commuting_pauli_set_exp_rejects_non_commuting():
    b = qx.SimpleBlock(1)
    with pytest.raises(ValueError):
        b.commuting_pauli_set_exp(["X", "Y"], [0.1, 0.2])


def test_commuting_pauli_set_exp_rejects_length_mismatch():
    b = qx.SimpleBlock(2)
    with pytest.raises(ValueError):
        b.commuting_pauli_set_exp(["XX", "ZZ"], [0.1])


def test_commuting_pauli_set_exp_rejects_wrong_pauli_length():
    b = qx.SimpleBlock(2)
    with pytest.raises(ValueError):
        b.commuting_pauli_set_exp(["XXX"], [0.1])


# ── Gate-count regression pins (graysynth acceptance oracle) ──────────────
#
# `commuting_pauli_set_exp` synthesises its diagonal block with graysynth
# (shared CX cascades + Patel-Markov-Hayes restoration).  These pins are its
# acceptance oracle: they need no external library, so they gate CI, unlike the
# alternative of benchmarking against pytket (not a declared dependency —
# conventions §15).
#
# Ceilings are `<=`, so an improvement passes and only a regression fails; they
# are pinned with zero slack, so any regression fails immediately.  Every pin
# also asserts the unitary against the scipy oracle, so a count cannot be
# reduced by emitting a wrong circuit.
#
# Graysynth's win is concentrated in dense groups, which is where UCC and
# Trotter actually live: `ucc_double_4q` went 32 → 16 CX, while the sparse and
# already-diagonal workloads were untouched.

# Fixed, deterministic workloads.  The UCC entries are the real generators from
# `UCCBlock([1, 1, 0, 0]).symbol_qop_pairs`.
COUNT_WORKLOADS = {
    "pair_2q": (["XX", "YY"], 4),
    "triple_3q": (["XXI", "YYI", "ZZI"], 4),
    "ucc_single_4q": (["YZXI", "XZYI"], 6),
    # 32 under the naive per-row ladder; halved when graysynth landed.
    "ucc_double_4q": (
        ["YYYX", "YXYY", "XXYX", "XYYY", "YXXX", "YYXY", "XYXX", "XXXY"],
        16,
    ),
    "zdiag_6q": (["ZZIIII", "IZZIII", "IIZZII", "IIIZZI", "IIIIZZ"], 10),
    "hopping_8q": (
        [
            "XXIIIIII",
            "YYIIIIII",
            "IIXXIIII",
            "IIYYIIII",
            "IIIIXXII",
            "IIIIYYII",
            "IIIIIIXX",
            "IIIIIIYY",
        ],
        16,
    ),
}


@pytest.mark.parametrize("workload", sorted(COUNT_WORKLOADS))
def test_commuting_set_cx_count_within_ceiling(workload):
    """CX count must not regress, and the circuit must still be correct."""
    paulis, ceiling = COUNT_WORKLOADS[workload]
    angles = [0.1 * (i + 1) for i in range(len(paulis))]
    n_qubits = len(paulis[0])

    block = qx.SimpleBlock(n_qubits)
    block.commuting_pauli_set_exp(paulis, angles)
    block.build()

    gates = [str(c.gate).rsplit(".", 1)[-1] for c in block.commands()]
    n_cx = gates.count("CX")
    assert n_cx <= ceiling, f"{workload}: CX count rose to {n_cx} (ceiling {ceiling})"

    np.testing.assert_allclose(
        build_block_unitary(block), reference_commuting_unitary(paulis, angles), atol=1e-10
    )


@pytest.mark.parametrize("workload", sorted(COUNT_WORKLOADS))
def test_commuting_set_emits_one_rz_per_pauli(workload):
    """Structural invariant independent of the diagonal-block algorithm: one
    rotation per Pauli.  Graysynth reshapes the CX ladder, never the Rz count,
    so this must survive the swap unchanged."""
    paulis, _ = COUNT_WORKLOADS[workload]
    angles = [0.1 * (i + 1) for i in range(len(paulis))]

    block = qx.SimpleBlock(len(paulis[0]))
    block.commuting_pauli_set_exp(paulis, angles)
    block.build()

    gates = [str(c.gate).rsplit(".", 1)[-1] for c in block.commands()]
    assert gates.count("Rz") == len(paulis)
