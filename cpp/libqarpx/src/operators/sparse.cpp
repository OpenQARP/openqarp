#include "qarpx/operators/sparse.h"

#include <algorithm>
#include <bit>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "qarpx/operators/transforms.h"

namespace qarpx::ops {

namespace {

// Dense dimension cap: 2^kMaxQubits complex entries per term.  Dense-matrix
// call sites in qarp stay ≤ ~14 qubits; beyond ~30 the matrix could not be
// materialized anyway.
constexpr int kMaxQubits = 30;

// Output cap on the triplet count n_masks · dim (32 B each).
constexpr int64_t kMaxTriplets = int64_t{1} << 33;

std::complex<double> i_pow(int k) {
    switch (((k % 4) + 4) % 4) {
        case 0: return {1.0, 0.0};
        case 1: return {0.0, 1.0};
        case 2: return {-1.0, 0.0};
        default: return {0.0, -1.0};
    }
}

// One Pauli term as a phased permutation: row = col ^ flip, value =
// base · (−1)^{popcount(col & sign)}.
struct MaskedTerm {
    uint64_t sign_mask;
    std::complex<double> base;
};

}  // namespace

SparseCoo qubit_operator_coo(const QubitOperator& op, int n_qubits, BitOrder order) {
    const int required = op.count_qubits();
    if (n_qubits < 0) n_qubits = required;
    if (n_qubits < required)
        throw std::invalid_argument("Invalid number of qubits specified.");
    if (n_qubits > kMaxQubits)
        throw std::invalid_argument(
            "sparse_matrix: dense dimension 2^" + std::to_string(n_qubits) +
            " is too large to materialize.");

    SparseCoo out;
    out.dim = int64_t{1} << n_qubits;

    // Terms sharing a flip mask land on the same (row, col) cells, so they
    // are summed here on one dim-length accumulator per mask — the true nnz
    // — instead of one triplet per (term, col) left for scipy to reduce.
    std::map<uint64_t, std::vector<MaskedTerm>> by_flip;
    for (const auto& e : op.numeric().terms) {
        // Qubit-q bit position: kMsb ↔ bit n_qubits-1-q, kLsb ↔ bit q.
        uint64_t flip_mask = 0;  // X and Y flip the basis bit
        uint64_t sign_mask = 0;  // Z and Y contribute (−1)^bit
        int n_y = 0;
        for (int q = 0; q < n_qubits; ++q) {
            const uint64_t bit = order == BitOrder::kLsb
                                     ? uint64_t{1} << q
                                     : uint64_t{1} << (n_qubits - 1 - q);
            switch (e.key.pauli_at(static_cast<uint32_t>(q))) {
                case Pauli::I: break;
                case Pauli::X: flip_mask |= bit; break;
                case Pauli::Y:
                    flip_mask |= bit;
                    sign_mask |= bit;
                    ++n_y;
                    break;
                case Pauli::Z: sign_mask |= bit; break;
            }
        }
        by_flip[flip_mask].push_back({sign_mask, e.coeff * i_pow(n_y)});
    }

    const auto n_masks = static_cast<int64_t>(by_flip.size());
    if (n_masks > kMaxTriplets / out.dim)
        throw std::invalid_argument(
            "sparse_matrix: " + std::to_string(n_masks) + " distinct flip masks × dimension 2^" +
            std::to_string(n_qubits) + " exceeds the " + std::to_string(kMaxTriplets) +
            "-entry cap.");
    out.data.reserve(static_cast<size_t>(n_masks * out.dim));
    out.rows.reserve(static_cast<size_t>(n_masks * out.dim));
    out.cols.reserve(static_cast<size_t>(n_masks * out.dim));

    std::vector<std::complex<double>> acc(static_cast<size_t>(out.dim));
    for (const auto& [flip_mask, terms] : by_flip) {
        std::fill(acc.begin(), acc.end(), std::complex<double>{});
        for (const auto& t : terms) {
            for (int64_t col = 0; col < out.dim; ++col) {
                const bool negate =
                    (std::popcount(static_cast<uint64_t>(col) & t.sign_mask) & 1) != 0;
                acc[static_cast<size_t>(col)] += negate ? -t.base : t.base;
            }
        }
        for (int64_t col = 0; col < out.dim; ++col) {
            const auto v = acc[static_cast<size_t>(col)];
            if (v == std::complex<double>{}) continue;
            out.rows.push_back(col ^ static_cast<int64_t>(flip_mask));
            out.cols.push_back(col);
            out.data.push_back(v);
        }
    }
    return out;
}

SparseCoo fermion_operator_coo(const FermionOperator& op, int n_qubits, BitOrder order) {
    const int required = op.count_qubits();
    if (n_qubits < 0) n_qubits = required;
    if (n_qubits < required)
        throw std::invalid_argument("Invalid number of qubits specified.");
    return qubit_operator_coo(jordan_wigner(op), n_qubits, order);
}

}  // namespace qarpx::ops
