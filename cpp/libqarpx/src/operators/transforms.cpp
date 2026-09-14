#include "qarpx/operators/transforms.h"

namespace qarpx::ops {

namespace detail {

// Verbatim ports of openfermion's Fenwick-tree set computations
// (bravyi_kitaev.py _update_set/_occupation_set/_parity_set), including the
// 1-based index dance.

std::set<uint32_t> bk_update_set(uint32_t index, uint32_t n_qubits) {
    std::set<uint32_t> indices;
    uint64_t i = static_cast<uint64_t>(index) + 1;
    i += i & (~i + 1);  // add least significant one bit
    while (i <= n_qubits) {
        indices.insert(static_cast<uint32_t>(i - 1));
        i += i & (~i + 1);
    }
    return indices;
}

std::set<uint32_t> bk_occupation_set(uint32_t index) {
    std::set<uint32_t> indices;
    uint64_t i = static_cast<uint64_t>(index) + 1;
    indices.insert(static_cast<uint32_t>(i - 1));
    const uint64_t parent = i & (i - 1);
    uint64_t j = i - 1;
    while (j != parent) {
        indices.insert(static_cast<uint32_t>(j - 1));
        j &= j - 1;  // remove least significant one bit
    }
    return indices;
}

std::set<uint32_t> bk_parity_set(uint32_t index) {
    std::set<uint32_t> indices;
    uint64_t i = index;
    while (i > 0) {
        indices.insert(static_cast<uint32_t>(i - 1));
        i &= i - 1;
    }
    return indices;
}

}  // namespace detail

QubitOperator jordan_wigner(const FermionOperator& op) {
    return op.visit([](const auto& eng) {
        return QubitOperator::from_engine(detail::jordan_wigner_impl(eng));
    });
}

QubitOperator bravyi_kitaev(const FermionOperator& op, int n_qubits) {
    return op.visit([n_qubits](const auto& eng) {
        return QubitOperator::from_engine(detail::bravyi_kitaev_impl(eng, n_qubits));
    });
}

QubitOperator parity_transform(const FermionOperator& op, int n_qubits) {
    return op.visit([n_qubits](const auto& eng) {
        return QubitOperator::from_engine(detail::parity_transform_impl(eng, n_qubits));
    });
}

}  // namespace qarpx::ops
