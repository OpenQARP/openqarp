"""Smoke tests for SynthesizedTimeEvolutionBlock — exact U(t)=exp(-iHt)."""

import numpy as np
import pytest
from scipy.linalg import expm

import qarpx as qx
from qarp.blocks import ControlledBlock, SynthesizedTimeEvolutionBlock
from qarp.operators import JordanWigner, QubitOperator
from qarp.operators.models import fermi_hubbard


def _unitary(block):
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


def _max_diff(actual, expected):
    """Element-wise max abs diff — global phase included, never divided out.

    A synthesized block that is right only up to ``e^{iγ}`` reads as an
    eigenphase shift once it sits under a control (QPE, QMEGS, Hadamard
    tests), so the global phase is part of the contract here.
    """
    return float(np.abs(actual - expected).max())


def _residual_phase(actual, expected):
    """Global phase between two unitaries — for failure diagnostics only."""
    return float(np.angle(np.trace(expected.conj().T @ actual)))


def _expected_lsb_unitary(operator, n_qubits, t):
    """``expm(-i H t)`` in qarpx (LSB) convention, derived from an OpenFermion operator."""
    H_lsb = operator.sparse_matrix(n_qubits).toarray()
    return expm(-1j * H_lsb * t)


# ── Basic correctness ────────────────────────────────────────────────────


@pytest.mark.parametrize("t", [0.0, 0.3, 1.5])
def test_time_evolution_matches_expm_for_z(t):
    """``U(t) = exp(-i Z₀ t)`` on 1 qubit."""
    op = QubitOperator("Z0")
    block = SynthesizedTimeEvolutionBlock(operator=op, n_qubits=1, time=t).build()
    U = _unitary(block)
    expected = expm(-1j * np.array([[1, 0], [0, -1]], dtype=complex) * t)
    assert _max_diff(U, expected) < 1e-10, f"residual phase {_residual_phase(U, expected)}"


def test_time_evolution_is_unitary():
    op = QubitOperator("Z0", 0.7) + QubitOperator("X0 X1", 0.3)
    block = SynthesizedTimeEvolutionBlock(operator=op, n_qubits=2, time=0.5).build()
    U = _unitary(block)
    assert np.linalg.norm(U @ U.conj().T - np.eye(4)) < 1e-10


# ── Constructor kwargs round-trip ─────────────────────────────────────────


def test_kwargs_round_trip():
    op = QubitOperator("Z0")
    block = SynthesizedTimeEvolutionBlock(
        operator=op,
        n_qubits=1,
        time=0.5,
        target_qubits=[2],
        name="CustomTE",
    )
    assert block.n_qubits == 1
    assert block.target_qubits == [2]
    assert block.name == "CustomTE"
    assert block.time == 0.5


def test_default_name():
    op = QubitOperator("Z0")
    assert SynthesizedTimeEvolutionBlock(operator=op, n_qubits=1, time=0.5).name == (
        "SynthTimeEvoBlock"
    )


# ── set_time semantics ────────────────────────────────────────────────────


def test_set_time_does_not_mutate_original():
    """set_time returns a fresh deepcopy; the original block's time is unchanged."""
    op = QubitOperator("Z0")
    original = SynthesizedTimeEvolutionBlock(operator=op, n_qubits=1, time=0.5).build()
    updated = original.set_time(1.3)
    assert original.time == 0.5
    assert updated.time == 1.3
    assert updated is not original
    assert isinstance(updated, SynthesizedTimeEvolutionBlock)


def test_set_time_chain_updates_unitary():
    """Successive set_time calls each rebuild with the new time; the synthesized
    unitary at each step matches ``expm(-i H t_k)`` exactly."""
    op = QubitOperator("Z0")
    block = SynthesizedTimeEvolutionBlock(operator=op, n_qubits=1, time=0.1).build()
    for t in [0.5, 1.0, 2.0, 0.25]:
        block = block.set_time(t)
        assert block.time == t
        U = _unitary(block)
        expected = _expected_lsb_unitary(op, 1, t)
        assert _max_diff(U, expected) < 1e-9


# ── Wider correctness coverage ────────────────────────────────────────────


def test_negative_time_correctness():
    """Backward evolution: t < 0 still synthesizes ``expm(-i H t)`` correctly."""
    op = QubitOperator("Z0")
    t = -0.5
    block = SynthesizedTimeEvolutionBlock(operator=op, n_qubits=1, time=t).build()
    U = _unitary(block)
    expected = _expected_lsb_unitary(op, 1, t)
    assert _max_diff(U, expected) < 1e-10


def test_large_time_correctness():
    """Stays correct at t=10 — no precision drift across many wraps."""
    op = QubitOperator("Z0")
    t = 10.0
    block = SynthesizedTimeEvolutionBlock(operator=op, n_qubits=1, time=t).build()
    U = _unitary(block)
    expected = _expected_lsb_unitary(op, 1, t)
    assert _max_diff(U, expected) < 1e-9


def test_fermi_hubbard_correctness():
    """Realistic Hamiltonian: 1-site Fermi-Hubbard via Jordan-Wigner (2 qubits)."""
    fham = fermi_hubbard((1,), t=0.14, U=0.231)
    qham = JordanWigner().encode_operator(fham)
    t = 0.5
    block = SynthesizedTimeEvolutionBlock(operator=qham, n_qubits=2, time=t).build()
    U = _unitary(block)
    expected = _expected_lsb_unitary(qham, 2, t)
    assert _max_diff(U, expected) < 1e-9


def test_three_qubit_operator_correctness():
    """Mixed 3-qubit Hamiltonian: ZZ chain + transverse field."""
    op = (
        QubitOperator("Z0 Z1", 0.5)
        + QubitOperator("Z1 Z2", 0.5)
        + QubitOperator("X0", 0.3)
        + QubitOperator("X1", 0.3)
        + QubitOperator("X2", 0.3)
    )
    t = 0.4
    block = SynthesizedTimeEvolutionBlock(operator=op, n_qubits=3, time=t).build()
    U = _unitary(block)
    expected = _expected_lsb_unitary(op, 3, t)
    assert _max_diff(U, expected) < 1e-8


# ── Global-phase exactness regressions ───────────────────────────────────
#
# QSD's ZYZ base case splits on γ = 2·atan2(|u10|, |u00|); its γ ≈ 0 and
# γ ≈ π branches once emitted a wrong global phase.  Only *structured*
# operators reach them — a Haar-random or generic target lands on the correct
# generic branch — so these cases use diagonal / permutation-like Hamiltonians
# and compare against scipy's expm including the phase.


@pytest.mark.parametrize("t", [0.3, 1.0, 2.7])
def test_diagonal_hamiltonian_phase_exact(t):
    """Purely diagonal H drives every ZYZ call onto the γ ≈ 0 branch."""
    op = QubitOperator("Z0", 0.7) + QubitOperator("Z1", -0.4) + QubitOperator("Z0 Z1", 0.9)
    block = SynthesizedTimeEvolutionBlock(operator=op, n_qubits=2, time=t).build()
    U = _unitary(block)
    expected = _expected_lsb_unitary(op, 2, t)
    assert _max_diff(U, expected) < 1e-10, f"residual phase {_residual_phase(U, expected)}"


def test_identity_term_contributes_its_global_phase():
    """A constant term in H is pure global phase: U(t) = e^{-i c t}·exp(-i H' t).

    Nothing but the synthesized global phase carries it, so dropping the phase
    loses the constant entirely.
    """
    const = 0.75
    t = 1.1
    op = QubitOperator("Z0", 0.6) + QubitOperator("X0 X1", 0.3)
    op_with_const = op + QubitOperator("", const)

    U_plain = _unitary(SynthesizedTimeEvolutionBlock(operator=op, n_qubits=2, time=t).build())
    U_const = _unitary(
        SynthesizedTimeEvolutionBlock(operator=op_with_const, n_qubits=2, time=t).build()
    )
    assert _max_diff(U_const, np.exp(-1j * const * t) * U_plain) < 1e-10


def test_four_qubit_hubbard_phase_exact():
    """2-site Fermi-Hubbard (4 qubits) — the shape that exposed the bug.

    A structured many-body ``expm(-iHt)`` sends the QSD recursion onto diagonal
    and anti-diagonal single-qubit sub-blocks, where the wrong-phase branches
    lived.  Oracle is scipy's expm, independent of qarp.
    """
    fham = fermi_hubbard((2,), t=1.0, U=8.0)
    qham = JordanWigner().encode_operator(fham)
    for t in (0.25, 1.0, 2.0):
        block = SynthesizedTimeEvolutionBlock(operator=qham, n_qubits=4, time=t).build()
        U = _unitary(block)
        expected = _expected_lsb_unitary(qham, 4, t)
        assert _max_diff(U, expected) < 1e-9, (
            f"t={t}: residual phase {_residual_phase(U, expected)}"
        )


def test_eigenstate_survival_amplitude_matches_analytic_phase():
    """``⟨ψ|U(t)|ψ⟩ = e^{-iEt}`` for an eigenstate ``|ψ⟩`` of H.

    This is the QMEGS/Hadamard-test observable: unlike ``|⟨ψ|U|ψ⟩|`` it is
    phase-sensitive, so a synthesis that is right only up to ``e^{iγ}`` returns
    the wrong eigenvalue while still having unit modulus.  Oracle is the
    analytic ``e^{-iEt}`` from a numpy eigendecomposition.
    """
    op = QubitOperator("Z0", 0.6) + QubitOperator("Z1", -0.35) + QubitOperator("Z0 Z1", 0.8)
    n_qubits = 2
    H_lsb = op.sparse_matrix(n_qubits).toarray()
    evals, evecs = np.linalg.eigh(H_lsb)

    for k in range(len(evals)):
        psi, energy = evecs[:, k], evals[k]
        for t in (0.4, 1.3):
            U = _unitary(
                SynthesizedTimeEvolutionBlock(operator=op, n_qubits=n_qubits, time=t).build()
            )
            z = complex(np.vdot(psi, U @ psi))
            assert abs(z - np.exp(-1j * energy * t)) < 1e-9, (
                f"eigenstate {k}, t={t}: got phase {np.angle(z)}, "
                f"expected {np.angle(np.exp(-1j * energy * t))}"
            )


# ── LSB-convention anchor (first-principles Kron, no sparse_matrix) ──────


def test_bit_reversal_anchor_z_on_q0_lsb():
    """``Z`` on qubit 0 of a 2-qubit register — asymmetric under bit
    reversal, so it pins the qubit↔bit mapping.

    The expected unitary is built directly via Kron in the qarpx LSB
    convention (``I_{q1} ⊗ Z_{q0}``), independent of ``sparse_matrix``, so
    an MSB reading anywhere in the block would evolve the wrong subspace.
    """
    op = QubitOperator("Z0")
    t = 0.55
    block = SynthesizedTimeEvolutionBlock(operator=op, n_qubits=2, time=t).build()

    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    # qarpx LSB-Kron: low qubit on the right.  "Z on q0" = I_{q1} ⊗ Z_{q0}.
    H_qarpx = np.kron(I, Z)
    expected = expm(-1j * H_qarpx * t)

    U = _unitary(block)
    assert _max_diff(U, expected) < 1e-10


# ── Controlled wrapping via ControlledBlock ──────────────────────────────


def test_controlled_block_wrapping_subspace_decomposition():
    """``ControlledBlock(SynthesizedTimeEvolutionBlock, ctrl_state=[True])``:

    * On the ``|0⟩``-control subspace the wrapped circuit must act as the
      identity (controls deactivated).
    * On the ``|1⟩``-control subspace it must act as ``exp(-iHt)`` exactly.
      This is the assertion that matters: a global phase on the inner block
      becomes a *relative* phase between the two control subspaces, which
      QPE/DOSQPE/QMEGS read directly as an eigenphase shift.

    qarpx places controls at the lowest qubit indices, so in LSB ordering
    the control bits live at the low end and the inner subspaces are
    selected by ``i_ctrl ∈ {0, 1}`` in the LSB position.
    """
    op = QubitOperator("Z0", 0.4) + QubitOperator("Z0 Z1", 0.7)
    t = 0.5
    inner = SynthesizedTimeEvolutionBlock(operator=op, n_qubits=2, time=t).build()
    expected_inner = _expected_lsb_unitary(op, 2, t)

    ctrl = ControlledBlock(inner, num_controls=1, ctrl_state=[True])
    ctrl.build()
    U_ctrl = _unitary(ctrl)
    assert U_ctrl.shape == (8, 8)

    # LSB: control at q0 → state index = i_inner * 2 + i_ctrl.
    idx_c0 = list(range(0, 8, 2))  # control = |0⟩
    idx_c1 = list(range(1, 8, 2))  # control = |1⟩

    U_c0 = U_ctrl[np.ix_(idx_c0, idx_c0)]
    U_c1 = U_ctrl[np.ix_(idx_c1, idx_c1)]

    # Control = |0⟩ subspace: identity, no phase ambiguity.
    assert np.linalg.norm(U_c0 - np.eye(4)) < 1e-10

    # Control = |1⟩ subspace: matches the inner unitary exactly, phase included.
    assert _max_diff(U_c1, expected_inner) < 1e-9, (
        f"relative phase between control subspaces: {_residual_phase(U_c1, expected_inner)}"
    )


# ── Validation edge cases ─────────────────────────────────────────────────


def test_operator_qubit_index_out_of_range_raises():
    """OpenFermion raises when the operator references a qubit beyond n_qubits;
    the error must surface to the caller rather than being silently absorbed."""
    op = QubitOperator("Z5", 1.0)
    with pytest.raises(ValueError, match="Invalid number of qubits"):
        SynthesizedTimeEvolutionBlock(operator=op, n_qubits=2, time=0.5)
