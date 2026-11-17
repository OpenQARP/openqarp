#pragma once

#include "../block/block.h"
#include "../core/command.h"
#include "../core/param.h"
#include "../core/pauli.h"

#include <vector>

namespace qarpx::synthesis {

// ── Public API ───────────────────────────────────────────────────────────

/// Append gates to `block` that implement
///
///     U = exp( -i/2 · Σᵢ angles[i] · paulis[i] )
///
/// for a set of mutually commuting Pauli strings.
///
/// Preconditions (validated; throws `std::invalid_argument` on violation):
///   - `paulis.size() == angles.size()`.
///   - Every `paulis[i].size() == block.n_qubits`.
///   - Every pair `(paulis[i], paulis[j])` commutes.
///
/// Empty input is a no-op.  Identity rows (all-I) collapse into a single
/// `GPhase(-Σ_id angles[i] / 2)` for the identity contribution.
///
/// Algorithm:
///   1. (k = 1 fast path)  Delegate to `pauli_exp` — emit the canonical
///      basis-change → CX-ladder → Rz(angle) → reverse pattern (the single
///      Pauli-exponential form the Trotter blocks emit).
///   2. (k ≥ 2)  Aaronson-Gottesman §7-style basis-change Clifford brings
///      every Pauli to a Z-only string, then per-row CX-ladder + Rz
///      synthesises each diagonal exponential, then the inverse Clifford
///      restores the original basis.
///
/// A future optimisation (Amy-Azimzadeh-Mosca graysynth +
/// Patel-Markov-Hayes, arXiv:1712.01859 + arXiv:quant-ph/0302002) will
/// replace the per-row diagonal-block synthesis with a shared cascade that
/// approaches pytket's `PauliExpCommutingSetBox` CX count.  See
/// `detail::synthesise_diagonal_block_graysynth`.
void commuting_pauli_set_exp(
    Block& block,
    const std::vector<PauliString>& paulis,
    const std::vector<Param>& angles);

/// Append gates to `block` that implement
///     U = exp( -i/2 · angle · pauli )
/// for a single multi-qubit Pauli operator.  Equivalent to
/// `commuting_pauli_set_exp(block, {pauli}, {angle})` (and is the k = 1
/// fast path inside that function).
///
/// `pauli.size() == block.n_qubits` is required; throws otherwise.
void pauli_exp(
    Block& block,
    const PauliString& pauli,
    const Param& angle);

namespace detail {

/// Symplectic tableau for a set of Paulis.
///
/// Each row encodes a Pauli `P_r = ⊗_q P_q^{(r)}` via two binary vectors:
///   x[r][q] = 1 iff `P_q^{(r)} ∈ {X, Y}`.
///   z[r][q] = 1 iff `P_q^{(r)} ∈ {Y, Z}`.
/// And a sign bit:
///   sign[r] = false  →  the row represents +P_r,
///   sign[r] = true   →  the row represents -P_r.
///
/// Single-qubit Cliffords act locally:
///   H(q):    swaps x[r][q] ↔ z[r][q];
///            phase flips for rows with Y at qubit q.
///   S(q):    z[r][q] ^= x[r][q];     // X → Y, Y → X, Z → Z, I → I
///            phase flips for rows with Y at qubit q.        (S Y S† = -X)
///   Sdg(q):  z[r][q] ^= x[r][q];     // same symplectic action as S
///            phase flips for rows with X at qubit q.        (Sdg X S = -Y)
///
/// Two-qubit Clifford CX(c, t):
///   x[r][t] ^= x[r][c];   z[r][c] ^= z[r][t];
///   phase update per Aaronson-Gottesman (`apply_cx` below).
struct SymplecticTableau {
    int n_qubits = 0;
    int n_rows = 0;
    std::vector<std::vector<bool>> x;     // [n_rows][n_qubits]
    std::vector<std::vector<bool>> z;     // [n_rows][n_qubits]
    std::vector<bool> sign;                // [n_rows]: true == -1
};

/// Build a tableau from `paulis`.  All rows start with `sign = false`.
/// Throws `std::invalid_argument` if any row's length differs from `n_qubits`.
SymplecticTableau pauli_strings_to_tableau(
    const std::vector<PauliString>& paulis, int n_qubits);

void apply_h  (SymplecticTableau& T, int q);
void apply_s  (SymplecticTableau& T, int q);
void apply_sdg(SymplecticTableau& T, int q);
void apply_cx (SymplecticTableau& T, int c, int t);

/// Result of `compute_basis_change_clifford`:
///   - `clifford` is the gate sequence to apply BEFORE the diagonal Rz layer
///     (in execution order).  The inverse of this sequence is applied after
///     the Rz layer to undo the basis change.
///   - `z_rows[r][q] = true` iff the diagonalised row r has Z at qubit q.
///   - `signs[r] = true` iff the conjugated row equals `-Z_only` rather than
///     `+Z_only`.  Caller must negate the rotation angle for sign-flipped
///     rows so the emitted exponential implements the original Pauli.
struct BasisChangeResult {
    std::vector<Command> clifford;
    std::vector<std::vector<bool>> z_rows;
    std::vector<bool> signs;
};

/// Compute Cliffords that diagonalise a set of mutually commuting Paulis to
/// a Z-only basis.  Throws `std::invalid_argument` if the set is not
/// pairwise commuting — the elimination would otherwise return wrong
/// z_rows/signs with no error (a non-commuting row can silently collapse to
/// identity, so a consumer would read its expectation as +1).
///
/// Algorithm: Aaronson-Gottesman §7-style symplectic Gauss elimination.  For
/// each row r in order, find an "unprocessed" qubit q with non-I on row r;
/// apply Sdg/H to convert row r at qubit q to Z; apply CX(qp, q) for each
/// other unprocessed qubit qp where row r has non-I to merge it into the
/// pivot.  Maintains the invariant that processed rows have Z at exactly
/// their pivot qubit and I at every unprocessed qubit, so subsequent
/// Cliffords on unprocessed qubits never disturb already-processed rows.
BasisChangeResult compute_basis_change_clifford(
    const std::vector<PauliString>& paulis, int n_qubits);

// ── Diagonal-block synthesis ──────────────────────────────────────────────
//
// Given a set of Z-only Pauli strings (the output of
// `compute_basis_change_clifford`'s `z_rows`/`signs`) and per-row angles,
// emit a gate sequence implementing
//
//     Π_i exp( -i · effective_angle_i / 2 · Z^{z_rows[i]} )
//
// where `effective_angle_i = signs[i] ? -angles[i] : angles[i]` and
// `Z^v = ⊗_q (z_rows[i][q] ? Z : I)`.  Rows with `z_rows[i]` all false
// contribute a global phase `gphase(-effective_angle_i / 2)`.
//
// Two implementations are provided.  Both produce equivalent unitaries;
// they differ in CX count.
//
//   - `synthesise_diagonal_block_naive` emits an independent CX ladder per
//     row.  Cost: Σᵢ 2·(hamming(z_rows[i]) − 1) CX + k Rz.  Used as the
//     correctness reference for the graysynth variant.
//
//   - `synthesise_diagonal_block_graysynth` applies the
//     Amy-Azimzadeh-Mosca 2018 recursion (arXiv:1712.01859 §4) to share
//     CX cascades across rows, followed by Patel-Markov-Hayes
//     (arXiv:quant-ph/0302002) for any residual linear permutation.
//     Equivalent unitary, typically much fewer CX.

void synthesise_diagonal_block_naive(
    Block& block,
    const std::vector<std::vector<bool>>& z_rows,
    const std::vector<bool>& signs,
    const std::vector<Param>& angles);

void synthesise_diagonal_block_graysynth(
    Block& block,
    const std::vector<std::vector<bool>>& z_rows,
    const std::vector<bool>& signs,
    const std::vector<Param>& angles);

/// Synthesise with whichever of the two above emits fewer CX for this
/// particular group, and emit the winner.  Graysynth is better on dense
/// groups but pays a Patel-Markov-Hayes restoration that the naive ladder
/// avoids by uncomputing, so it loses on small groups spread over many
/// qubits.  This is what `commuting_pauli_set_exp` calls, making "never worse
/// than the naive ladder" a guarantee rather than a typical case.
void synthesise_diagonal_block_best(
    Block& block,
    const std::vector<std::vector<bool>>& z_rows,
    const std::vector<bool>& signs,
    const std::vector<Param>& angles);

}  // namespace detail

}  // namespace qarpx::synthesis
