#pragma once

#include "qarpx/operators/fermion_operator.h"
#include "qarpx/operators/qubit_operator.h"

#include <set>
#include <stdexcept>
#include <unordered_map>

namespace qarpx::ops {

/// Fermion→qubit transforms, mirrored loop-for-loop from openfermion 1.7.1
/// (JW/BK) and qarp's previous hand-rolled Python parity mapping (now
/// wrapped by qarp/operators/mappings.py) so that the resulting `.terms`
/// insertion order matches the Python implementations verbatim — qarp's
/// golden mapping tests assert that order.
QubitOperator jordan_wigner(const FermionOperator& op);

/// openfermion bravyi_kitaev(operator, n_qubits=None): n_qubits < 0 means
/// "use count_qubits(op)"; anything smaller than that throws
/// std::invalid_argument("Invalid number of qubits specified.").
QubitOperator bravyi_kitaev(const FermionOperator& op, int n_qubits = -1);

/// qarp's parity mapping (hand-rolled in parity.py, ported verbatim).
/// Throws std::runtime_error with parity.py's exact message when a mode
/// index reaches n_qubits.
QubitOperator parity_transform(const FermionOperator& op, int n_qubits);

namespace detail {

// ── Bravyi-Kitaev Fenwick-tree index sets (verbatim ports) ────────────────

/// Bits that must be updated when mode `index` flips (all > index).
std::set<uint32_t> bk_update_set(uint32_t index, uint32_t n_qubits);
/// Bits whose parity stores the occupation of mode `index` (all ≤ index).
std::set<uint32_t> bk_occupation_set(uint32_t index);
/// Bits whose parity stores the parity of modes 0..index-1 (all < index).
std::set<uint32_t> bk_parity_set(uint32_t index);

// ── Templated engines (instantiated for the numeric backend now; the
//    SymEngine backend reuses them unchanged) ──────────────────────────────

/// openfermion _jordan_wigner_fermion_operator:
///   for term: seed = identity·coeff; seed *= cached ladder op (X-component
///   inserted before Y); out += seed.
template <typename Coeff>
QubitOperatorT<Coeff> jordan_wigner_impl(const FermionOperatorT<Coeff>& op) {
    using Traits = CoeffTraits<Coeff>;
    QubitOperatorT<Coeff> out;
    std::unordered_map<uint32_t, QubitOperatorT<Coeff>> ladder_cache;
    for (const auto& term : op.terms) {
        QubitOperatorT<Coeff> transformed;
        transformed.set_term(PackedPauli{}, term.coeff);
        for (const uint32_t ladder : term.key.ops) {
            auto it = ladder_cache.find(ladder);
            if (it == ladder_cache.end()) {
                const uint32_t index = FermionKey::index_of(ladder);
                const bool creation = FermionKey::action_of(ladder) == 1;
                PackedPauli x_key, y_key;
                for (uint32_t z = 0; z < index; ++z) {
                    x_key.set_pauli(z, Pauli::Z);
                    y_key.set_pauli(z, Pauli::Z);
                }
                x_key.set_pauli(index, Pauli::X);
                y_key.set_pauli(index, Pauli::Y);
                x_key.canonicalize();
                y_key.canonicalize();
                QubitOperatorT<Coeff> ladder_op;
                ladder_op.set_term(x_key, Coeff(std::complex<double>(0.5, 0.0)));
                ladder_op.set_term(
                    y_key, Coeff(std::complex<double>(0.0, creation ? -0.5 : 0.5)));
                it = ladder_cache.emplace(ladder, std::move(ladder_op)).first;
            }
            transformed = transformed.mul(it->second);
        }
        out.iadd(transformed);
    }
    (void)sizeof(Traits);
    // The ladder factors are exact halves, but their products over a float
    // coefficient leave residuals ~1e-18 that no longer vanish in iadd (which
    // erases exact cancellations only).  Truncate once, here, where the whole
    // image is in hand: openfermion gets the same effect for free from its
    // per-add tolerance, and leaving them in inflates every downstream term
    // count (H2/STO-3G JW: 27 terms instead of 15).
    out.compress();
    return out;
}

/// openfermion _transform_ladder_operator: X(update)·Z(parity) term with
/// coefficient 0.5, then ± the Majorana difference
/// Y(index)·X(update−{index})·Z((parity^occupation)−{index}) with ∓0.5j.
template <typename Coeff>
QubitOperatorT<Coeff> bk_ladder_operator(uint32_t index, bool creation, uint32_t n_qubits) {
    std::set<uint32_t> update = bk_update_set(index, n_qubits);
    update.insert(index);
    const std::set<uint32_t> occupation = bk_occupation_set(index);
    const std::set<uint32_t> parity = bk_parity_set(index);

    PackedPauli x_key;
    for (const uint32_t q : update) x_key.set_pauli(q, Pauli::X);
    for (const uint32_t q : parity) x_key.set_pauli(q, Pauli::Z);
    x_key.canonicalize();

    PackedPauli d_key;
    d_key.set_pauli(index, Pauli::Y);
    for (const uint32_t q : update)
        if (q != index) d_key.set_pauli(q, Pauli::X);
    for (const uint32_t q : parity)
        if (occupation.count(q) == 0 && q != index) d_key.set_pauli(q, Pauli::Z);
    for (const uint32_t q : occupation)
        if (parity.count(q) == 0 && q != index) d_key.set_pauli(q, Pauli::Z);
    d_key.canonicalize();

    QubitOperatorT<Coeff> out;
    out.set_term(x_key, Coeff(std::complex<double>(0.5, 0.0)));
    out.set_term(d_key, Coeff(std::complex<double>(0.0, creation ? -0.5 : 0.5)));
    return out;
}

/// openfermion _bravyi_kitaev_fermion_operator via inline_sum/inline_product.
template <typename Coeff>
QubitOperatorT<Coeff> bravyi_kitaev_impl(const FermionOperatorT<Coeff>& op, int n_qubits) {
    const int required = op.count_qubits();
    if (n_qubits < 0) n_qubits = required;
    if (n_qubits < required)
        throw std::invalid_argument("Invalid number of qubits specified.");

    QubitOperatorT<Coeff> out;
    std::unordered_map<uint32_t, QubitOperatorT<Coeff>> ladder_cache;
    for (const auto& term : op.terms) {
        QubitOperatorT<Coeff> transformed;
        transformed.set_term(PackedPauli{}, term.coeff);
        for (const uint32_t ladder : term.key.ops) {
            auto it = ladder_cache.find(ladder);
            if (it == ladder_cache.end()) {
                it = ladder_cache
                         .emplace(ladder,
                                  bk_ladder_operator<Coeff>(
                                      FermionKey::index_of(ladder),
                                      FermionKey::action_of(ladder) == 1,
                                      static_cast<uint32_t>(n_qubits)))
                         .first;
            }
            transformed = transformed.mul(it->second);
        }
        out.iadd(transformed);
    }
    // The ladder factors are exact halves, but their products over a float
    // coefficient leave residuals ~1e-18 that no longer vanish in iadd (which
    // erases exact cancellations only).  Truncate once, here, where the whole
    // image is in hand: openfermion gets the same effect for free from its
    // per-add tolerance, and leaving them in inflates every downstream term
    // count (H2/STO-3G JW: 27 terms instead of 15).
    out.compress();
    return out;
}

/// qarp parity.py _parity_creation/_parity_annihilation, ported verbatim:
///   identity·0.5, then X on every mode above `orbital`, then the edge factor
///   (Z(orb-1)·X(orb) ∓ i·Y(orb), or X(orb) ∓ i·Y(orb) at orbital 0).
template <typename Coeff>
QubitOperatorT<Coeff> parity_ladder_operator(uint32_t orbital, bool creation,
                                             uint32_t n_qubits) {
    QubitOperatorT<Coeff> qop;
    qop.set_term(PackedPauli{}, Coeff(std::complex<double>(0.5, 0.0)));
    for (uint32_t k = orbital + 1; k < n_qubits; ++k) {
        QubitOperatorT<Coeff> x_k;
        PackedPauli key;
        key.set_pauli(k, Pauli::X);
        key.canonicalize();
        x_k.set_term(key, Coeff(std::complex<double>(1.0, 0.0)));
        qop = qop.mul(x_k);
    }

    PackedPauli zx_key;
    if (orbital > 0) zx_key.set_pauli(orbital - 1, Pauli::Z);
    zx_key.set_pauli(orbital, Pauli::X);
    zx_key.canonicalize();
    PackedPauli y_key;
    y_key.set_pauli(orbital, Pauli::Y);
    y_key.canonicalize();

    QubitOperatorT<Coeff> edge;
    edge.set_term(zx_key, Coeff(std::complex<double>(1.0, 0.0)));
    edge.set_term(y_key, Coeff(std::complex<double>(0.0, creation ? -1.0 : 1.0)));
    return qop.mul(edge);
}

/// qarp parity.py Parity.encode inner loop, ported verbatim.
template <typename Coeff>
QubitOperatorT<Coeff> parity_transform_impl(const FermionOperatorT<Coeff>& op, int n_qubits) {
    QubitOperatorT<Coeff> out;
    for (const auto& term : op.terms) {
        QubitOperatorT<Coeff> qop_term;
        qop_term.set_term(PackedPauli{}, term.coeff);
        for (const uint32_t ladder : term.key.ops) {
            const uint32_t orbital = FermionKey::index_of(ladder);
            if (static_cast<int>(orbital) >= n_qubits)
                throw std::runtime_error(
                    "Orbital number is larger than number of qubits in parity.encode()");
            qop_term = qop_term.mul(parity_ladder_operator<Coeff>(
                orbital, FermionKey::action_of(ladder) == 1,
                static_cast<uint32_t>(n_qubits)));
        }
        out.iadd(qop_term);
    }
    // The ladder factors are exact halves, but their products over a float
    // coefficient leave residuals ~1e-18 that no longer vanish in iadd (which
    // erases exact cancellations only).  Truncate once, here, where the whole
    // image is in hand: openfermion gets the same effect for free from its
    // per-add tolerance, and leaving them in inflates every downstream term
    // count (H2/STO-3G JW: 27 terms instead of 15).
    out.compress();
    return out;
}

}  // namespace detail

}  // namespace qarpx::ops
