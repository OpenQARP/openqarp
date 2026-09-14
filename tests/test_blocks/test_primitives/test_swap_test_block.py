"""Correctness tests for SWAPTestBlock.

Layout: q0 = ancilla, q1..n_state = ket register, q(1+n_state)..2*n_state =
bra register.  After ``H · ket · bra · CSWAP-each · H``, the ancilla
expectation ``P(0) - P(1)`` equals ``|⟨bra|ket⟩|²``.
"""

import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import SimpleBlock, SWAPTestBlock


def _unitary(block):
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


def _ancilla_expectation(U, n_qubits: int) -> float:
    """⟨0|q0 - 1|q0⟩ on the all-zeros input.  q0 = LSB."""
    psi0 = np.zeros(2**n_qubits, complex)
    psi0[0] = 1.0
    psi_out = U @ psi0
    p0 = sum(abs(psi_out[i]) ** 2 for i in range(2**n_qubits) if (i & 1) == 0)
    p1 = sum(abs(psi_out[i]) ** 2 for i in range(2**n_qubits) if (i & 1) == 1)
    return p0 - p1


# ── Construction validation ──────────────────────────────────────────────


def test_qubit_mismatch_raises():
    bra = SimpleBlock(2, name="b")
    bra.build()
    ket = SimpleBlock(1, name="k")
    ket.build()
    with pytest.raises(ValueError, match="Qubit count mismatch"):
        SWAPTestBlock(bra=bra, ket=ket)


def test_n_qubits_layout():
    bra = SimpleBlock(3, name="b")
    bra.build()
    ket = SimpleBlock(3, name="k")
    ket.build()
    sw = SWAPTestBlock(bra=bra, ket=ket)
    assert sw.n_qubits == 1 + 2 * 3


# ── Standard overlap cases ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "bra_factory, ket_factory, expected",
    [
        # |0⟩ vs |0⟩  → |⟨0|0⟩|² = 1
        (lambda: SimpleBlock(1, name="0"), lambda: SimpleBlock(1, name="0"), 1.0),
        # |0⟩ vs |1⟩  → 0
        (lambda: SimpleBlock(1, name="0"), lambda: SimpleBlock(1, name="1").x(0), 0.0),
        # |0⟩ vs |+⟩  → |⟨0|+⟩|² = 1/2
        (lambda: SimpleBlock(1, name="0"), lambda: SimpleBlock(1, name="+").h(0), 0.5),
        # |+⟩ vs |+⟩  → 1
        (lambda: SimpleBlock(1, name="+").h(0), lambda: SimpleBlock(1, name="+").h(0), 1.0),
    ],
)
def test_overlap_matches_known(bra_factory, ket_factory, expected):
    bra = bra_factory()
    bra.build()
    ket = ket_factory()
    ket.build()
    sw = SWAPTestBlock(bra=bra, ket=ket)
    sw.build()
    U = _unitary(sw)
    assert _ancilla_expectation(U, sw.n_qubits) == pytest.approx(expected, abs=1e-12)


# ── Multi-qubit ket / bra ────────────────────────────────────────────────


def test_two_qubit_orthogonal_states_zero_overlap():
    """|00⟩ vs |11⟩ are orthogonal → overlap² = 0."""
    bra = SimpleBlock(2, name="00")
    bra.build()
    ket = SimpleBlock(2, name="11")
    ket.x(0)
    ket.x(1)
    ket.build()
    sw = SWAPTestBlock(bra=bra, ket=ket)
    sw.build()
    U = _unitary(sw)
    assert abs(_ancilla_expectation(U, sw.n_qubits)) < 1e-12


def test_two_qubit_identical_states_unit_overlap():
    """|01⟩ vs |01⟩ → overlap² = 1."""
    bra = SimpleBlock(2, name="01")
    bra.x(0)
    bra.build()
    ket = SimpleBlock(2, name="01")
    ket.x(0)
    ket.build()
    sw = SWAPTestBlock(bra=bra, ket=ket)
    sw.build()
    U = _unitary(sw)
    assert _ancilla_expectation(U, sw.n_qubits) == pytest.approx(1.0, abs=1e-12)


# ── Measurement option ───────────────────────────────────────────────────


def test_measure_option_emits_measure_command():
    bra = SimpleBlock(1, name="b")
    bra.build()
    ket = SimpleBlock(1, name="k")
    ket.build()
    sw = SWAPTestBlock(bra=bra, ket=ket, measure=True)
    sw.build()
    cmds = sw.flatten()
    assert any(c.gate.name == "Measure" for c in cmds)
    assert sw.n_cbits >= 1


# ── Complex-overlap regression ──────────────────────────────────────────


def test_complex_overlap_one_vs_s_plus_is_one_half():
    """|1⟩ vs S·H|0⟩ = (|0⟩ + i|1⟩)/√2 ⇒ |⟨1|·⟩|² = |i/√2|² = 1/2."""
    bra = SimpleBlock(1, name="1")
    bra.x(0)
    bra.build()
    ket = SimpleBlock(1, name="S+")
    ket.h(0)
    ket.s(0)
    ket.build()
    sw = SWAPTestBlock(bra=bra, ket=ket)
    sw.build()
    U = _unitary(sw)
    assert _ancilla_expectation(U, sw.n_qubits) == pytest.approx(0.5, abs=1e-12)


# ── Multi-qubit partial overlap ─────────────────────────────────────────


@pytest.mark.parametrize("n", [2, 3, 4])
def test_zero_state_vs_hn_overlap_is_one_over_2n(n):
    """|0…0⟩ vs Hn|0…0⟩ = uniform superposition ⇒ |⟨0…0|·⟩|² = 1/2^n."""
    bra = SimpleBlock(n, name="0…0")
    bra.build()
    ket = SimpleBlock(n, name="++…+")
    for q in range(n):
        ket.h(q)
    ket.build()
    sw = SWAPTestBlock(bra=bra, ket=ket)
    sw.build()
    U = _unitary(sw)
    assert _ancilla_expectation(U, sw.n_qubits) == pytest.approx(1.0 / 2**n, abs=1e-12)


def test_two_qubit_bell_states_orthogonal_overlap_zero():
    """|Φ⁺⟩ = (|00⟩+|11⟩)/√2 vs |Ψ⁺⟩ = (|01⟩+|10⟩)/√2 — orthogonal Bell pair."""
    phi_plus = SimpleBlock(2, name="Φ+")
    phi_plus.h(0)
    phi_plus.cx(0, 1)
    phi_plus.build()

    psi_plus = SimpleBlock(2, name="Ψ+")
    psi_plus.h(0)
    psi_plus.x(1)
    psi_plus.cx(0, 1)
    psi_plus.build()

    sw = SWAPTestBlock(bra=phi_plus, ket=psi_plus)
    sw.build()
    U = _unitary(sw)
    assert abs(_ancilla_expectation(U, sw.n_qubits)) < 1e-12


# ── Surface API ─────────────────────────────────────────────────────────


def test_bra_ket_attributes_preserved():
    bra = SimpleBlock(1, name="b")
    bra.build()
    ket = SimpleBlock(1, name="k")
    ket.build()
    sw = SWAPTestBlock(bra=bra, ket=ket)
    assert sw.bra is bra
    assert sw.ket is ket
