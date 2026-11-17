#include "qarpx/operators/packed_pauli.h"

#include <stdexcept>

namespace qarpx::ops {

namespace {

// (x, z) bit pair for each Pauli enum value, indexed by the enum's integer.
constexpr uint8_t kXBit[4] = {0, 1, 1, 0};  // I X Y Z
constexpr uint8_t kZBit[4] = {0, 0, 1, 1};

Pauli pauli_from_bits(uint64_t x, uint64_t z) {
    // (0,0)→I (1,0)→X (1,1)→Y (0,1)→Z
    return x ? (z ? Pauli::Y : Pauli::X) : (z ? Pauli::Z : Pauli::I);
}

}  // namespace

Pauli PackedPauli::pauli_at(uint32_t qubit) const {
    const int blk = static_cast<int>(qubit / 64);
    if (blk >= n_blocks()) return Pauli::I;
    const uint32_t bit = qubit % 64;
    return pauli_from_bits((x_word(blk) >> bit) & 1, (z_word(blk) >> bit) & 1);
}

void PackedPauli::set_pauli(uint32_t qubit, Pauli p) {
    const size_t blk = qubit / 64;
    while (words.size() < 2 * (blk + 1)) words.push_back(0);
    const uint64_t mask = uint64_t{1} << (qubit % 64);
    const auto idx = static_cast<uint8_t>(p);
    if (kXBit[idx])
        words[2 * blk] |= mask;
    else
        words[2 * blk] &= ~mask;
    if (kZBit[idx])
        words[2 * blk + 1] |= mask;
    else
        words[2 * blk + 1] &= ~mask;
}

int PackedPauli::max_qubit() const {
    for (int blk = n_blocks() - 1; blk >= 0; --blk) {
        const uint64_t occ = x_word(blk) | z_word(blk);
        if (occ != 0) return blk * 64 + (63 - std::countl_zero(occ));
    }
    return -1;
}

int PackedPauli::weight() const {
    int w = 0;
    for (int blk = 0; blk < n_blocks(); ++blk)
        w += std::popcount(x_word(blk) | z_word(blk));
    return w;
}

void PackedPauli::canonicalize() {
    size_t nb = words.size() / 2;
    while (nb > 0 && words[2 * nb - 2] == 0 && words[2 * nb - 1] == 0) --nb;
    words.resize(2 * nb);
}

std::pair<Pauli, int> single_pauli_product(Pauli a, Pauli b) {
    PackedPauli pa, pb;
    if (a != Pauli::I) pa.set_pauli(0, a);
    if (b != Pauli::I) pb.set_pauli(0, b);
    pa.canonicalize();
    pb.canonicalize();
    const int k = phase_exponent_mod4(pa, pb);
    return {pauli_xor(pa, pb).pauli_at(0), k};
}

char pauli_letter(Pauli p) {
    switch (p) {
        case Pauli::I: return 'I';
        case Pauli::X: return 'X';
        case Pauli::Y: return 'Y';
        default: return 'Z';
    }
}

Pauli pauli_from_letter(char c) {
    switch (c) {
        case 'X': return Pauli::X;
        case 'Y': return Pauli::Y;
        case 'Z': return Pauli::Z;
        default:
            throw std::invalid_argument(std::string("Invalid Pauli letter '") +
                                        c + "'; expected X, Y or Z.");
    }
}

void packed_to_index_pairs(const PackedPauli& p,
                           std::vector<std::pair<uint32_t, char>>& out) {
    out.clear();
    for (int blk = 0; blk < p.n_blocks(); ++blk) {
        uint64_t occ = p.x_word(blk) | p.z_word(blk);
        while (occ != 0) {
            const int bit = std::countr_zero(occ);
            occ &= occ - 1;
            const auto qubit = static_cast<uint32_t>(blk * 64 + bit);
            out.emplace_back(qubit, pauli_letter(p.pauli_at(qubit)));
        }
    }
}

PackedPauli packed_from_dense(const PauliString& p) {
    PackedPauli out;
    for (size_t q = 0; q < p.size(); ++q)
        if (p[q] != Pauli::I) out.set_pauli(static_cast<uint32_t>(q), p[q]);
    out.canonicalize();
    return out;
}

PauliString packed_to_dense(const PackedPauli& p, int n_qubits) {
    PauliString out(static_cast<size_t>(n_qubits), Pauli::I);
    for (int q = 0; q < n_qubits; ++q)
        out[static_cast<size_t>(q)] = p.pauli_at(static_cast<uint32_t>(q));
    return out;
}

}  // namespace qarpx::ops
