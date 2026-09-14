#pragma once

#include <cstdint>

#include "qarpx/core/small_vector.h"

namespace qarpx::ops {

/// An openfermion FermionOperator term key: an ordered sequence of ladder
/// operators, each packed as (mode_index << 1) | action with action
/// 1 = creation (a†, "^") and 0 = annihilation (a).
///
/// Deliberately NO canonicalization: openfermion stores the sequence
/// verbatim (its FermionOperator sets different_indices_commute=False), so
/// multiplication concatenates and nothing is ever normal-ordered.
struct FermionKey {
    SmallVector<uint32_t, 4> ops;  // two-body terms (4 ladder ops) stay inline

    static constexpr uint32_t pack(uint32_t index, uint32_t action) {
        return (index << 1) | (action & 1u);
    }
    static constexpr uint32_t index_of(uint32_t packed) { return packed >> 1; }
    static constexpr uint32_t action_of(uint32_t packed) { return packed & 1u; }

    bool is_identity() const { return ops.empty(); }
    size_t size() const { return ops.size(); }

    void append(uint32_t index, uint32_t action) {
        ops.push_back(pack(index, action));
    }

    /// Multiplication of fermion terms = sequence concatenation.
    FermionKey concat(const FermionKey& o) const {
        FermionKey out;
        out.ops.reserve(ops.size() + o.ops.size());
        for (uint32_t v : ops) out.ops.push_back(v);
        for (uint32_t v : o.ops) out.ops.push_back(v);
        return out;
    }

    /// Hermitian conjugate: reverse the sequence and flip every action bit
    /// (openfermion: tuple(reversed(...)) with action → 1 - action).
    FermionKey dagger() const {
        FermionKey out;
        out.ops.reserve(ops.size());
        for (size_t i = ops.size(); i > 0; --i) out.ops.push_back(ops[i - 1] ^ 1u);
        return out;
    }

    /// Highest mode index touched; -1 for the identity term.
    int max_index() const {
        int m = -1;
        for (uint32_t v : ops) {
            const int idx = static_cast<int>(index_of(v));
            if (idx > m) m = idx;
        }
        return m;
    }

    bool operator==(const FermionKey& o) const { return ops == o.ops; }
    bool operator!=(const FermionKey& o) const { return !(*this == o); }

    uint64_t hash() const {
        uint64_t h = 0xcbf29ce484222325ull ^ (ops.size() * 0x9e3779b97f4a7c15ull);
        for (uint32_t v : ops) {
            uint64_t w = v + 0x9e3779b97f4a7c15ull;
            w = (w ^ (w >> 30)) * 0xbf58476d1ce4e5b9ull;
            w = (w ^ (w >> 27)) * 0x94d049bb133111ebull;
            w ^= w >> 31;
            h = (h ^ w) * 0x100000001b3ull;
        }
        return h;
    }
};

}  // namespace qarpx::ops
