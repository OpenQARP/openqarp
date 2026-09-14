#include "qarpx/synthesis/pauli_exponential.h"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qarpx::synthesis::detail {

SymplecticTableau pauli_strings_to_tableau(
    const std::vector<PauliString>& paulis, int n_qubits)
{
    SymplecticTableau T;
    T.n_qubits = n_qubits;
    T.n_rows = static_cast<int>(paulis.size());
    T.x.assign(T.n_rows, std::vector<bool>(n_qubits, false));
    T.z.assign(T.n_rows, std::vector<bool>(n_qubits, false));
    T.sign.assign(T.n_rows, false);

    for (int r = 0; r < T.n_rows; ++r) {
        if (static_cast<int>(paulis[r].size()) != n_qubits) {
            throw std::invalid_argument(
                "pauli_strings_to_tableau: row " + std::to_string(r)
                + " has length " + std::to_string(paulis[r].size())
                + " but n_qubits is " + std::to_string(n_qubits));
        }
        for (int q = 0; q < n_qubits; ++q) {
            switch (paulis[r][q]) {
                case Pauli::I: /* x=0, z=0 */ break;
                case Pauli::X: T.x[r][q] = true; break;
                case Pauli::Z: T.z[r][q] = true; break;
                case Pauli::Y: T.x[r][q] = true; T.z[r][q] = true; break;
            }
        }
    }
    return T;
}

// ── Clifford updates on the tableau ───────────────────────────────────────
//
// All three update the symplectic representation in place AND update the
// row signs to track the ±1 phase that would otherwise be lost.
//
// Sign rules derived from:
//   H X H = Z,    H Y H = -Y,    H Z H = X.
//   S X S† = Y,   S Y S† = -X,   S Z S† = Z.
//   Sdg X Sdg† = -Y,  Sdg Y Sdg† = X,  Sdg Z Sdg† = Z.
//   CX(c,t): standard symplectic action on (x_c, z_c, x_t, z_t) plus the
//   Aaronson-Gottesman 2004 phase formula
//      sign ^= x[c] · z[t] · (x[t] ⊕ z[c] ⊕ 1)
//   evaluated with PRE-update values.

void apply_h(SymplecticTableau& T, int q) {
    for (int r = 0; r < T.n_rows; ++r) {
        // Phase flips iff row r has Y at qubit q (H Y H = -Y).
        if (T.x[r][q] && T.z[r][q]) {
            T.sign[r] = !T.sign[r];
        }
        // Symplectic action: swap x and z bits.
        bool tmp = T.x[r][q];
        T.x[r][q] = T.z[r][q];
        T.z[r][q] = tmp;
    }
}

void apply_s(SymplecticTableau& T, int q) {
    for (int r = 0; r < T.n_rows; ++r) {
        // Phase flips iff row r has Y at qubit q (S Y S† = -X).
        if (T.x[r][q] && T.z[r][q]) {
            T.sign[r] = !T.sign[r];
        }
        // Symplectic action: z[r][q] ^= x[r][q].
        if (T.x[r][q]) {
            T.z[r][q] = !T.z[r][q];
        }
    }
}

void apply_sdg(SymplecticTableau& T, int q) {
    for (int r = 0; r < T.n_rows; ++r) {
        // Phase flips iff row r has X at qubit q (Sdg X Sdg† = -Y).
        if (T.x[r][q] && !T.z[r][q]) {
            T.sign[r] = !T.sign[r];
        }
        // Symplectic action is the same as S: z[r][q] ^= x[r][q].
        if (T.x[r][q]) {
            T.z[r][q] = !T.z[r][q];
        }
    }
}

void apply_cx(SymplecticTableau& T, int c, int t) {
    if (c == t) {
        throw std::invalid_argument("apply_cx: control and target must differ");
    }
    for (int r = 0; r < T.n_rows; ++r) {
        // PRE-update values for the AG phase formula.
        const bool xc = T.x[r][c];
        const bool zc = T.z[r][c];
        const bool xt = T.x[r][t];
        const bool zt = T.z[r][t];
        // sign ^= xc · zt · (xt ⊕ zc ⊕ 1)
        if (xc && zt && (xt ^ zc ^ true)) {
            T.sign[r] = !T.sign[r];
        }
        // Symplectic update.
        T.x[r][t] = xt ^ xc;
        T.z[r][c] = zc ^ zt;
    }
}

// ── Helpers for the basis-change algorithm ─────────────────────────────────

namespace {

// Append a single-qubit Clifford command and update the tableau.
void emit_h(SymplecticTableau& T, std::vector<Command>& out, int q) {
    apply_h(T, q);
    out.emplace_back(GateType::H, static_cast<uint32_t>(q));
}

void emit_sdg(SymplecticTableau& T, std::vector<Command>& out, int q) {
    apply_sdg(T, q);
    out.emplace_back(GateType::Sdg, static_cast<uint32_t>(q));
}

void emit_cx(SymplecticTableau& T, std::vector<Command>& out, int c, int t) {
    apply_cx(T, c, t);
    out.emplace_back(GateType::CX,
                     static_cast<uint32_t>(c),
                     static_cast<uint32_t>(t));
}

// Convert (x[r][q], z[r][q]) to (0, 1) — i.e., row r's qubit q is Z.
// Picks Sdg over S so that Y → X is phase-clean (Sdg Y S = X with no sign flip).
void diagonalise_to_z(SymplecticTableau& T, std::vector<Command>& out, int r, int q) {
    if (T.x[r][q] && T.z[r][q]) {
        // Y at (r, q): Sdg sends Y → X.
        emit_sdg(T, out, q);
    }
    if (T.x[r][q]) {
        // X at (r, q): H sends X → Z.
        emit_h(T, out, q);
    }
    // Now T.x[r][q] = false and T.z[r][q] = true.
}

}  // anonymous namespace

BasisChangeResult compute_basis_change_clifford(
    const std::vector<PauliString>& paulis, int n_qubits)
{
    SymplecticTableau T = pauli_strings_to_tableau(paulis, n_qubits);

    // The elimination assumes mutual commutation; on non-commuting input it
    // returns wrong z_rows/signs with no error.  This is the chokepoint every
    // consumer routes through (diagonalise_group, commuting_pauli_set_exp),
    // so the check lives here rather than in each caller.
    for (std::size_t i = 0; i + 1 < paulis.size(); ++i) {
        for (std::size_t j = i + 1; j < paulis.size(); ++j) {
            if (!commutes(paulis[i], paulis[j])) {
                throw std::invalid_argument(
                    "compute_basis_change_clifford: paulis[" + std::to_string(i)
                    + "] = " + to_string(paulis[i])
                    + " does not commute with paulis[" + std::to_string(j)
                    + "] = " + to_string(paulis[j]));
            }
        }
    }

    BasisChangeResult result;
    result.clifford.reserve(static_cast<std::size_t>(2 * n_qubits * T.n_rows));

    std::vector<bool> qubit_processed(n_qubits, false);

    for (int r = 0; r < T.n_rows; ++r) {
        // Step 1: pick an unprocessed qubit q where row r has non-I.
        int pivot = -1;
        for (int q = 0; q < n_qubits; ++q) {
            if (qubit_processed[q]) continue;
            if (T.x[r][q] || T.z[r][q]) {
                pivot = q;
                break;
            }
        }
        if (pivot < 0) {
            // Row r has only I at unprocessed qubits.  Whatever it has at
            // processed qubits is already Z-only (by the invariant); leave
            // the tableau row alone — `z_rows` for this row will just be
            // its current Z-pattern at processed qubits.
            continue;
        }

        // Step 2: convert row r at qubit pivot to Z.
        diagonalise_to_z(T, result.clifford, r, pivot);

        // Step 3: clean up unprocessed qubits qp ≠ pivot in row r so that
        // row r ends with Z at pivot only and I at all other unprocessed
        // qubits.
        for (int qp = 0; qp < n_qubits; ++qp) {
            if (qp == pivot || qubit_processed[qp]) continue;
            // Convert any non-I at (r, qp) to Z, then merge into pivot via CX.
            if (T.x[r][qp] || T.z[r][qp]) {
                diagonalise_to_z(T, result.clifford, r, qp);
                // Row r now has Z at qp.  CX(qp, pivot) sets z[r][qp] ^= z[r][pivot] = 1,
                // clearing it; pivot's Z is preserved (z[r][pivot] is unchanged by
                // CX(qp, pivot) on the z side).
                emit_cx(T, result.clifford, qp, pivot);
            }
        }

        qubit_processed[pivot] = true;
    }

    // Extract the diagonalised z-pattern and signs.
    result.z_rows = std::move(T.z);
    result.signs = std::move(T.sign);
    return result;
}

// ── Diagonal-block synthesis ──────────────────────────────────────────────

namespace {

/// Validate the three input arrays have matching size.  Throws on mismatch.
void check_diagonal_block_input(
    const std::vector<std::vector<bool>>& z_rows,
    const std::vector<bool>& signs,
    const std::vector<Param>& angles)
{
    if (z_rows.size() != signs.size() || z_rows.size() != angles.size()) {
        throw std::invalid_argument(
            "synthesise_diagonal_block: z_rows / signs / angles must all have the same length");
    }
    if (!z_rows.empty()) {
        const std::size_t n = z_rows[0].size();
        for (std::size_t i = 1; i < z_rows.size(); ++i) {
            if (z_rows[i].size() != n) {
                throw std::invalid_argument(
                    "synthesise_diagonal_block: rows of z_rows must all have the same length");
            }
        }
    }
}

/// Effective rotation angle for row i: negated when `signs[i]` is true so
/// that the emitted `exp(-i·angle_eff/2 · Z^v)` equals the requested
/// `exp(-i·angles[i]/2 · sign · Z^v)` in the original basis.
Param effective_angle(const Param& angle, bool flip_sign) {
    return flip_sign ? -angle : angle;
}

}  // anonymous namespace

void synthesise_diagonal_block_naive(
    Block& block,
    const std::vector<std::vector<bool>>& z_rows,
    const std::vector<bool>& signs,
    const std::vector<Param>& angles)
{
    check_diagonal_block_input(z_rows, signs, angles);

    for (std::size_t i = 0; i < z_rows.size(); ++i) {
        const Param angle = effective_angle(angles[i], signs[i]);

        // Collect the qubits at which row i has Z.
        std::vector<uint32_t> zq;
        zq.reserve(z_rows[i].size());
        for (std::size_t q = 0; q < z_rows[i].size(); ++q) {
            if (z_rows[i][q]) zq.push_back(static_cast<uint32_t>(q));
        }

        if (zq.empty()) {
            // Identity row → global phase `exp(-i·angle/2) = gphase(-angle/2)`.
            block.gphase(-angle / Param(2.0));
            continue;
        }
        if (zq.size() == 1) {
            block.rz(zq[0], angle);
            continue;
        }

        // Multi-qubit Z: CX-ladder to last qubit, Rz, reverse CX-ladder.
        const uint32_t target = zq.back();
        for (std::size_t j = 0; j + 1 < zq.size(); ++j) {
            block.cx(zq[j], target);
        }
        block.rz(target, angle);
        for (std::size_t j = zq.size() - 1; j > 0; --j) {
            block.cx(zq[j - 1], target);
        }
    }
}

namespace {

using BitMatrix = std::vector<std::vector<bool>>;

/// `A[t] ^= A[c]` — the row operation a `CX(c, t)` performs on the parity
/// state (qubit `q` holds the parity given by row `q`).
void add_row(BitMatrix& A, int c, int t) {
    const std::size_t n = A[t].size();
    for (std::size_t j = 0; j < n; ++j) {
        A[t][j] = A[t][j] != A[c][j];
    }
}

BitMatrix identity_matrix(int n) {
    BitMatrix I(static_cast<std::size_t>(n), std::vector<bool>(static_cast<std::size_t>(n), false));
    for (int i = 0; i < n; ++i) I[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = true;
    return I;
}

int hamming(const std::vector<bool>& v) {
    int h = 0;
    for (bool b : v) if (b) ++h;
    return h;
}

/// Row operations reducing `A` (invertible over GF(2)) to the identity.
/// Each returned `(c, t)` means `A[t] ^= A[c]`, i.e. emit `CX(c, t)`; applying
/// them in order restores the register to the computational basis.
///
/// Patel-Markov-Hayes (arXiv:quant-ph/0302002): columns are processed in
/// sections of `m`, and rows sharing a sub-pattern within a section are
/// cancelled against each other once instead of eliminated one at a time —
/// that sharing is what turns the O(n²) of plain elimination into
/// O(n²/log n).  `m = 1` degenerates to ordinary Gauss-Jordan, which is why
/// the tests sweep `m` and compare against it.
///
/// Row *swaps* are never needed: over GF(2), GL(n) is generated by
/// transvections, so a zero pivot is repaired by adding a lower row into it.
std::vector<std::pair<int, int>> reduce_to_identity(BitMatrix A, int m) {
    const int n = static_cast<int>(A.size());
    std::vector<std::pair<int, int>> ops;
    if (n == 0) return ops;
    if (m < 1) m = 1;

    auto op = [&](int c, int t) {
        add_row(A, c, t);
        ops.emplace_back(c, t);
    };

    // Forward pass → upper triangular.
    for (int lo = 0; lo < n; lo += m) {
        const int hi = std::min(n, lo + m);

        // PMH sharing: collapse rows with identical sub-patterns in this
        // column section before eliminating.
        std::map<std::vector<bool>, int> seen;
        for (int row = lo; row < n; ++row) {
            std::vector<bool> sub(A[static_cast<std::size_t>(row)].begin() + lo,
                                  A[static_cast<std::size_t>(row)].begin() + hi);
            if (std::none_of(sub.begin(), sub.end(), [](bool b) { return b; })) continue;
            auto it = seen.find(sub);
            if (it == seen.end()) {
                seen.emplace(std::move(sub), row);
            } else {
                op(it->second, row);
            }
        }

        for (int col = lo; col < hi; ++col) {
            const auto ucol = static_cast<std::size_t>(col);
            if (!A[ucol][ucol]) {
                for (int row = col + 1; row < n; ++row) {
                    if (A[static_cast<std::size_t>(row)][ucol]) {
                        op(row, col);
                        break;
                    }
                }
            }
            if (!A[ucol][ucol]) {
                // Internal invariant, not caller input: surfaces to Python as
                // RuntimeError (logic_error), matching the sibling guards —
                // invalid_argument would misclassify a synthesis bug as user
                // error (ValueError).
                throw std::logic_error(
                    "reduce_to_identity: matrix is singular over GF(2)");
            }
            for (int row = col + 1; row < n; ++row) {
                if (A[static_cast<std::size_t>(row)][ucol]) op(col, row);
            }
        }
    }

    // Back-substitution → identity.  Upper triangular and invertible over
    // GF(2) means the diagonal is already all ones.
    for (int col = n - 1; col >= 0; --col) {
        for (int row = 0; row < col; ++row) {
            if (A[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)]) op(col, row);
        }
    }
    return ops;
}

/// Default PMH section width.  The log factor only pays off once n is large
/// enough for sub-patterns to repeat; below that the bookkeeping costs more
/// than it saves.
int default_section_width(int n) {
    int m = 1;
    while ((1 << (m + 1)) <= n) ++m;
    return std::max(1, m);
}

}  // anonymous namespace

void synthesise_diagonal_block_graysynth(
    Block& block,
    const std::vector<std::vector<bool>>& z_rows,
    const std::vector<bool>& signs,
    const std::vector<Param>& angles)
{
    check_diagonal_block_input(z_rows, signs, angles);
    if (z_rows.empty()) return;

    const int n = static_cast<int>(z_rows[0].size());
    if (n == 0) {
        for (std::size_t i = 0; i < z_rows.size(); ++i) {
            block.gphase(-effective_angle(angles[i], signs[i]) / Param(2.0));
        }
        return;
    }

    // Identity rows are pure global phase; peel them off so the parity
    // machinery only ever sees non-zero targets.
    std::vector<std::size_t> pending;
    pending.reserve(z_rows.size());
    for (std::size_t i = 0; i < z_rows.size(); ++i) {
        if (hamming(z_rows[i]) == 0) {
            block.gphase(-effective_angle(angles[i], signs[i]) / Param(2.0));
        } else {
            pending.push_back(i);
        }
    }
    if (pending.empty()) return;

    // `A[q]` is the parity qubit q currently holds, over the original inputs;
    // `B = (Aᵀ)⁻¹` turns a wanted parity into the set of qubits to combine.
    BitMatrix A = identity_matrix(n);
    BitMatrix B = identity_matrix(n);

    // Every Z-only exponential commutes with every other, so the rotations may
    // be emitted in any order — that is what licenses the greedy reordering.
    while (!pending.empty()) {
        std::size_t best_pos = 0;
        std::vector<int> best_support;
        bool first = true;

        for (std::size_t p = 0; p < pending.size(); ++p) {
            const std::vector<bool>& v = z_rows[pending[p]];
            // x = B·v: the (unique) set of current rows whose XOR is v.
            std::vector<int> support;
            for (int i = 0; i < n; ++i) {
                bool bit = false;
                for (int j = 0; j < n; ++j) {
                    if (B[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]
                        && v[static_cast<std::size_t>(j)]) {
                        bit = !bit;
                    }
                }
                if (bit) support.push_back(i);
            }
            if (first || support.size() < best_support.size()) {
                best_pos = p;
                best_support = std::move(support);
                first = false;
            }
            if (best_support.size() <= 1) break;  // cannot do better than free
        }

        const std::size_t row_index = pending[best_pos];
        pending.erase(pending.begin() + static_cast<std::ptrdiff_t>(best_pos));

        if (best_support.empty()) {
            throw std::logic_error("graysynth: non-zero parity resolved to an empty support");
        }

        // Prefer to overwrite an already-dirty qubit, so clean basis rows stay
        // available as single-CX building blocks for later parities.
        int target = best_support.front();
        int best_weight = hamming(A[static_cast<std::size_t>(target)]);
        for (int q : best_support) {
            const int w = hamming(A[static_cast<std::size_t>(q)]);
            if (w > best_weight) {
                best_weight = w;
                target = q;
            }
        }

        for (int q : best_support) {
            if (q == target) continue;
            block.cx(static_cast<uint32_t>(q), static_cast<uint32_t>(target));
            add_row(A, q, target);
            // B tracks (Aᵀ)⁻¹, which updates in the opposite direction.
            add_row(B, target, q);
        }

        block.rz(static_cast<uint32_t>(target),
                 effective_angle(angles[row_index], signs[row_index]));
    }

    // The rotations left the register in a permuted basis; restore it.
    for (const auto& [c, t] : reduce_to_identity(A, default_section_width(n))) {
        block.cx(static_cast<uint32_t>(c), static_cast<uint32_t>(t));
    }
}

void synthesise_diagonal_block_best(
    Block& block,
    const std::vector<std::vector<bool>>& z_rows,
    const std::vector<bool>& signs,
    const std::vector<Param>& angles)
{
    check_diagonal_block_input(z_rows, signs, angles);
    if (z_rows.empty()) return;

    const auto n_qubits = static_cast<uint32_t>(z_rows[0].size());

    auto count_cx = [](const SimpleBlock& b) {
        std::size_t n = 0;
        for (const auto& cmd : b.commands()) {
            if (cmd.gate == GateType::CX) ++n;
        }
        return n;
    };

    SimpleBlock naive(n_qubits, "naive");
    synthesise_diagonal_block_naive(naive, z_rows, signs, angles);

    SimpleBlock gray(n_qubits, "gray");
    synthesise_diagonal_block_graysynth(gray, z_rows, signs, angles);

    // Graysynth shares CX cascades but pays a Patel-Markov-Hayes restoration
    // the naive ladder avoids by uncomputing.  For small groups on many qubits
    // that tail is not amortised and naive wins — measured on ~10% of random
    // (n, k) configurations, concentrated at k ≤ 6.  Taking the cheaper of the
    // two makes "never worse than the naive ladder" a guarantee.
    const SimpleBlock& winner = count_cx(gray) <= count_cx(naive) ? gray : naive;

    for (const auto& cmd : winner.commands()) {
        switch (cmd.gate) {
            case GateType::CX:     block.cx(cmd.qubits[0], cmd.qubits[1]); break;
            case GateType::Rz:     block.rz(cmd.qubits[0], cmd.params[0]);  break;
            case GateType::GPhase: block.gphase(cmd.params[0]);             break;
            default:
                throw std::logic_error(
                    "synthesise_diagonal_block_best: unexpected gate in diagonal block");
        }
    }
}

}  // namespace qarpx::synthesis::detail

// ── Top-level public API ──────────────────────────────────────────────────

namespace qarpx::synthesis {

void pauli_exp(
    Block& block,
    const PauliString& pauli,
    const Param& angle)
{
    if (static_cast<uint32_t>(pauli.size()) != block.n_qubits) {
        throw std::invalid_argument(
            "pauli_exp: Pauli string length (" + std::to_string(pauli.size())
            + ") must equal block.n_qubits (" + std::to_string(block.n_qubits) + ")");
    }

    // Collect non-identity qubits with their Pauli letters.  Canonical
    // single-Pauli-exponential pattern — basis change → CX-ladder to last
    // non-I qubit → Rz(angle) → reverse-CX-ladder → reverse basis change.
    std::vector<std::pair<uint32_t, Pauli>> non_id;
    non_id.reserve(pauli.size());
    for (std::size_t q = 0; q < pauli.size(); ++q) {
        if (pauli[q] != Pauli::I) {
            non_id.emplace_back(static_cast<uint32_t>(q), pauli[q]);
        }
    }

    if (non_id.empty()) {
        // Identity Pauli: U = exp(-i·angle/2 · I) = e^{-i·angle/2}.
        block.gphase(-angle / Param(2.0));
        return;
    }

    // Forward basis change: X → H, Y → Sdg·H, Z → no-op.
    for (const auto& [q, p] : non_id) {
        if (p == Pauli::X) {
            block.h(q);
        } else if (p == Pauli::Y) {
            block.sdg(q);
            block.h(q);
        }
    }

    // CX-ladder onto the last non-identity qubit.
    const uint32_t target = non_id.back().first;
    for (std::size_t j = 0; j + 1 < non_id.size(); ++j) {
        block.cx(non_id[j].first, target);
    }

    // Rz on the parity-target qubit.  Rz(θ) = exp(-iθ/2·Z) gives the
    // multi-qubit exp(-i·angle/2·Z^v) once the CX-ladder has gathered
    // parity onto the target.
    block.rz(target, angle);

    // Reverse CX-ladder.
    for (std::size_t j = non_id.size() - 1; j > 0; --j) {
        block.cx(non_id[j - 1].first, target);
    }

    // Reverse basis change (Hermitian conjugate of the forward sweep).
    for (const auto& [q, p] : non_id) {
        if (p == Pauli::X) {
            block.h(q);
        } else if (p == Pauli::Y) {
            block.h(q);
            block.s(q);
        }
    }
}

void commuting_pauli_set_exp(
    Block& block,
    const std::vector<PauliString>& paulis,
    const std::vector<Param>& angles)
{
    if (paulis.size() != angles.size()) {
        throw std::invalid_argument(
            "commuting_pauli_set_exp: paulis (" + std::to_string(paulis.size())
            + ") and angles (" + std::to_string(angles.size())
            + ") must have the same length");
    }
    if (paulis.empty()) return;

    const int n_qubits = static_cast<int>(block.n_qubits);
    for (std::size_t i = 0; i < paulis.size(); ++i) {
        if (static_cast<int>(paulis[i].size()) != n_qubits) {
            throw std::invalid_argument(
                "commuting_pauli_set_exp: paulis[" + std::to_string(i)
                + "] has length " + std::to_string(paulis[i].size())
                + " but block.n_qubits is " + std::to_string(n_qubits));
        }
    }

    // Pairwise commutation is validated inside compute_basis_change_clifford;
    // the k = 1 fast path below has nothing to check.

    // k = 1 fast path — emit the canonical single-Pauli decomposition
    // (avoids the symplectic machinery for the common case).
    if (paulis.size() == 1) {
        pauli_exp(block, paulis[0], angles[0]);
        return;
    }

    // General case: basis-change Clifford → diagonal block → inverse Clifford.
    const auto bc = detail::compute_basis_change_clifford(paulis, n_qubits);

    // Forward Clifford: replay the emitted commands as actual gates.
    for (const auto& cmd : bc.clifford) {
        switch (cmd.gate) {
            case GateType::H:   block.h  (cmd.qubits[0]); break;
            case GateType::S:   block.s  (cmd.qubits[0]); break;
            case GateType::Sdg: block.sdg(cmd.qubits[0]); break;
            case GateType::CX:  block.cx (cmd.qubits[0], cmd.qubits[1]); break;
            default:
                throw std::runtime_error(
                    "commuting_pauli_set_exp: unexpected Clifford gate in basis change");
        }
    }

    // Diagonal block: cheaper of the graysynth cascade and the naive ladder.
    detail::synthesise_diagonal_block_best(block, bc.z_rows, bc.signs, angles);

    // Inverse Clifford: reverse order, each gate replaced by its dagger.
    //   H   → H    (self-inverse)
    //   S   → Sdg
    //   Sdg → S
    //   CX  → CX   (self-inverse)
    for (auto it = bc.clifford.rbegin(); it != bc.clifford.rend(); ++it) {
        const auto& cmd = *it;
        switch (cmd.gate) {
            case GateType::H:   block.h  (cmd.qubits[0]); break;
            case GateType::S:   block.sdg(cmd.qubits[0]); break;
            case GateType::Sdg: block.s  (cmd.qubits[0]); break;
            case GateType::CX:  block.cx (cmd.qubits[0], cmd.qubits[1]); break;
            default:
                throw std::runtime_error(
                    "commuting_pauli_set_exp: unexpected Clifford gate in inverse");
        }
    }
}

}  // namespace qarpx::synthesis
