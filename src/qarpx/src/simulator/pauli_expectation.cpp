#include "qarpx/simulator/pauli_expectation.h"
#include "qarpx/parallel/thread_pool.h"

#include <algorithm>
#include <bit>
#include <stdexcept>
#include <string>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace qarpx {

namespace {

using cd = std::complex<double>;

// (-1)^popcount(v)
inline double parity_sign(uint64_t v) {
    return (std::popcount(v) & 1U) ? -1.0 : 1.0;
}

// Terms sharing a flip mask `x`: P|i⟩ = i^{n_Y} (-1)^{popcount(i ∧ z)} |i ⊕ x⟩,
// so the group contributes conj(bra[i ⊕ x]) · ket[i] · Σ_j c_j (-1)^{popcount(i ∧ z_j)}
// with i^{n_Y} folded into c_j.
struct Group {
    uint64_t         x = 0;
    std::vector<uint64_t> z;
    std::vector<double>   cr, ci;  // split so the lane loop is two real FMAs
};

std::vector<Group> build_groups(const PauliObservable& observable, int n_qubits) {
    std::unordered_map<uint64_t, std::size_t> index;
    std::vector<Group> groups;
    for (const auto& [pauli, coeff] : observable) {
        uint64_t x = 0, z = 0, seen = 0;
        int n_y = 0;
        for (const auto& [q, op] : pauli) {
            if (q >= static_cast<uint32_t>(n_qubits))
                throw std::invalid_argument(
                    "pauli_transition: observable qubit " + std::to_string(q) +
                    " out of range for " + std::to_string(n_qubits) + " qubits");
            const uint64_t bit = uint64_t{1} << q;
            if (seen & bit)
                throw std::invalid_argument(
                    "pauli_transition: qubit " + std::to_string(q) +
                    " repeated within one Pauli string");
            seen |= bit;
            switch (op) {
                case 'X': x |= bit; break;
                case 'Z': z |= bit; break;
                case 'Y': x |= bit; z |= bit; ++n_y; break;
                case 'I': break;
                default:
                    throw std::invalid_argument(
                        "pauli_transition: invalid Pauli '" + std::string(1, op) + "'");
            }
        }
        static const cd i_pow[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        const cd c = coeff * i_pow[n_y & 3];
        auto [it, inserted] = index.try_emplace(x, groups.size());
        if (inserted) groups.push_back(Group{x, {}, {}, {}});
        Group& g = groups[it->second];
        g.z.push_back(z);
        g.cr.push_back(c.real());
        g.ci.push_back(c.imag());
    }
    return groups;
}

// T[zl][lo] = (-1)^popcount(zl ∧ lo) for the low six bits: the in-block part
// of every term's parity, shared by all terms and L1-resident.
struct SignTable {
    alignas(64) double t[64][64];
    SignTable() {
        for (int zl = 0; zl < 64; ++zl)
            for (int lo = 0; lo < 64; ++lo)
                t[zl][lo] = parity_sign(static_cast<uint64_t>(zl & lo));
    }
};
const SignTable& sign_table() {
    static const SignTable tab;
    return tab;
}

// n_qubits < 6: fewer than one block — per-index popcount, no table.
cd transition_small(const cd* bra, const cd* ket, uint64_t dim,
                    const std::vector<Group>& groups) {
    cd total{0.0, 0.0};
    for (uint64_t i = 0; i < dim; ++i) {
        cd acc{0.0, 0.0};
        for (const Group& g : groups) {
            double wr = 0.0, wi = 0.0;
            for (std::size_t j = 0; j < g.z.size(); ++j) {
                const double s = parity_sign(i & g.z[j]);
                wr += s * g.cr[j];
                wi += s * g.ci[j];
            }
            acc += std::conj(bra[i ^ g.x]) * cd{wr, wi};
        }
        total += acc * ket[i];
    }
    return total;
}

constexpr uint64_t BLOCK = 64;

cd transition_blocked(const cd* bra, const cd* ket, uint64_t dim,
                      const std::vector<Group>& groups) {
    const SignTable& tab = sign_table();
    const int64_t n_blocks = static_cast<int64_t>(dim / BLOCK);
    double re = 0.0, im = 0.0;

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = std::max(1, std::min(static_cast<int>(configured_thread_count()),
                                     omp_get_max_threads()));
#endif
    // Below 2^10 amplitudes the team fork costs more than the sweep.
#pragma omp parallel for num_threads(n_threads) schedule(static) \
    reduction(+ : re, im) if (n_blocks >= 16)
    for (int64_t b = 0; b < n_blocks; ++b) {
        const uint64_t base = static_cast<uint64_t>(b) * BLOCK;
        alignas(64) double acc_r[BLOCK] = {}, acc_i[BLOCK] = {};
        for (const Group& g : groups) {
            alignas(64) double wr[BLOCK] = {}, wi[BLOCK] = {};
            for (std::size_t j = 0; j < g.z.size(); ++j) {
                // parity(i ∧ z) = parity(base ∧ z) · T[z ∧ 63][i ∧ 63]: one
                // popcount per (block, term), the rest a 64-lane FMA.
                const double  s  = parity_sign(base & g.z[j]);
                const double  cr = s * g.cr[j], ci = s * g.ci[j];
                const double* row = tab.t[g.z[j] & (BLOCK - 1)];
                for (uint64_t lo = 0; lo < BLOCK; ++lo) {
                    wr[lo] += cr * row[lo];
                    wi[lo] += ci * row[lo];
                }
            }
            const uint64_t bra_base = base ^ (g.x & ~(BLOCK - 1));
            const uint64_t xl       = g.x & (BLOCK - 1);
            for (uint64_t lo = 0; lo < BLOCK; ++lo) {
                const cd p = std::conj(bra[bra_base | (lo ^ xl)]);
                acc_r[lo] += p.real() * wr[lo] - p.imag() * wi[lo];
                acc_i[lo] += p.real() * wi[lo] + p.imag() * wr[lo];
            }
        }
        for (uint64_t lo = 0; lo < BLOCK; ++lo) {
            const cd v = cd{acc_r[lo], acc_i[lo]} * ket[base + lo];
            re += v.real();
            im += v.imag();
        }
    }
    return {re, im};
}

}  // namespace

cd pauli_transition(const cd* bra, const cd* ket, int n_qubits,
                    const PauliObservable& observable) {
    check_transition_width(n_qubits);
    const auto groups = build_groups(observable, n_qubits);
    if (groups.empty()) return {0.0, 0.0};
    const uint64_t dim = uint64_t{1} << n_qubits;
    return dim < BLOCK ? transition_small(bra, ket, dim, groups)
                       : transition_blocked(bra, ket, dim, groups);
}

}  // namespace qarpx
