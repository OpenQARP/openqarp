#pragma once

#include <bit>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "qarpx/core/pauli.h"
#include "qarpx/core/small_vector.h"

namespace qarpx::ops {

/// A multi-qubit Pauli string in binary-symplectic form (the PauliEngine
/// design, arXiv:2601.02233, reimplemented).  Qubits are grouped into
/// 64-qubit blocks stored as interleaved words: words[2k] holds the x-bits
/// and words[2k+1] the z-bits of qubits [64k, 64k+64).  Per-qubit encoding:
/// (x,z) = (0,0) → I, (1,0) → X, (1,1) → Y, (0,1) → Z, with Y meaning the
/// true Pauli Y (= i·XZ), not the XZ product.
///
/// Canonical form strips trailing all-zero block pairs, so the identity is
/// empty and equality / hashing are independent of the width the string was
/// built at — keys are self-contained and comparable across operators.
/// Qubit indices are LSB-first, matching core/pauli.h and the rest of the
/// repo (qarp_conventions.md §1).
struct PackedPauli {
    SmallVector<uint64_t, 2> words;  // one block (≤64 qubits) stays inline

    int n_blocks() const { return static_cast<int>(words.size() / 2); }
    bool is_identity() const { return words.empty(); }

    uint64_t x_word(int block) const {
        return block < n_blocks() ? words[2 * block] : 0;
    }
    uint64_t z_word(int block) const {
        return block < n_blocks() ? words[2 * block + 1] : 0;
    }

    /// The Pauli acting on `qubit` (I outside the stored width).
    Pauli pauli_at(uint32_t qubit) const;

    /// Overwrite the slot for `qubit`.  May leave trailing all-zero blocks;
    /// call canonicalize() once after a batch of edits.
    void set_pauli(uint32_t qubit, Pauli p);

    /// Highest qubit with a non-identity Pauli; -1 for the identity.
    int max_qubit() const;

    /// Number of non-identity qubits.
    int weight() const;

    /// Strip trailing all-zero block pairs (restores the canonical form
    /// equality and hash() rely on).
    void canonicalize();

    bool operator==(const PackedPauli& o) const { return words == o.words; }
    bool operator!=(const PackedPauli& o) const { return !(*this == o); }

    /// Content hash (canonical form assumed, as produced by all operations
    /// here).  splitmix64-mixed FNV over the words.
    uint64_t hash() const {
        uint64_t h = 0x9e3779b97f4a7c15ull ^ (words.size() * 0xff51afd7ed558ccdull);
        for (uint64_t w : words) {
            w += 0x9e3779b97f4a7c15ull;
            w = (w ^ (w >> 30)) * 0xbf58476d1ce4e5b9ull;
            w = (w ^ (w >> 27)) * 0x94d049bb133111ebull;
            w ^= w >> 31;
            h = (h ^ w) * 0x100000001b3ull;
        }
        return h;
    }

    /// True iff the two strings commute: the number of qubits where they
    /// individually anticommute is even.  O(n/64).
    bool commutes_with(const PackedPauli& o) const {
        const int nb = n_blocks() > o.n_blocks() ? n_blocks() : o.n_blocks();
        int acc = 0;
        for (int b = 0; b < nb; ++b) {
            acc += std::popcount(x_word(b) & o.z_word(b));
            acc += std::popcount(z_word(b) & o.x_word(b));
        }
        return (acc & 1) == 0;
    }
};

/// k ∈ {0,1,2,3} such that σ(a)·σ(b) = i^k · σ(a XOR b).
///
/// With σ(p) = i^{y(p)}·X^{x_p}·Z^{z_p} per qubit (y(p) = |x_p ∧ z_p|, the
/// number of Y factors), commuting Z^{z_a} past X^{x_b} gives
///     k = ( y(a) + y(b) − y(a⊕b) + 2·|z_a ∧ x_b| ) mod 4
/// — four popcounts per 64-qubit block.
inline int phase_exponent_mod4(const PackedPauli& a, const PackedPauli& b) {
    const int nb = a.n_blocks() > b.n_blocks() ? a.n_blocks() : b.n_blocks();
    int64_t k = 0;
    for (int blk = 0; blk < nb; ++blk) {
        const uint64_t xa = a.x_word(blk), za = a.z_word(blk);
        const uint64_t xb = b.x_word(blk), zb = b.z_word(blk);
        k += std::popcount(xa & za);                        // y(a)
        k += std::popcount(xb & zb);                        // y(b)
        k -= std::popcount((xa ^ xb) & (za ^ zb));          // y(a⊕b)
        k += 2 * std::popcount(za & xb);
    }
    return static_cast<int>(((k % 4) + 4) % 4);
}

/// The (phaseless) symplectic product σ(a⊕b), in canonical form.
inline PackedPauli pauli_xor(const PackedPauli& a, const PackedPauli& b) {
    PackedPauli out;
    const int nb = a.n_blocks() > b.n_blocks() ? a.n_blocks() : b.n_blocks();
    out.words.reserve(2 * static_cast<size_t>(nb));
    for (int blk = 0; blk < nb; ++blk) {
        out.words.push_back(a.x_word(blk) ^ b.x_word(blk));
        out.words.push_back(a.z_word(blk) ^ b.z_word(blk));
    }
    out.canonicalize();
    return out;
}

/// Single-qubit product a·b = i^k · p.  Reproduces openfermion's
/// _PAULI_OPERATOR_PRODUCTS table exactly.
std::pair<Pauli, int> single_pauli_product(Pauli a, Pauli b);

/// 'I' / 'X' / 'Y' / 'Z' for a Pauli enum value.
char pauli_letter(Pauli p);

/// Inverse of pauli_letter for the non-identity letters; throws
/// std::invalid_argument on anything outside {X, Y, Z}.
Pauli pauli_from_letter(char c);

/// Emit the non-identity factors as (qubit, letter) pairs in ascending qubit
/// order — the shape of an openfermion QubitOperator term key.
void packed_to_index_pairs(const PackedPauli& p,
                           std::vector<std::pair<uint32_t, char>>& out);

/// Bridges to the dense core/pauli.h representation (LSB-first).
PackedPauli packed_from_dense(const PauliString& p);
PauliString packed_to_dense(const PackedPauli& p, int n_qubits);

}  // namespace qarpx::ops
