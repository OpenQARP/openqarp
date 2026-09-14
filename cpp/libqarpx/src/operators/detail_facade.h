// Internal helpers shared by the operator facade .cpp files (not installed):
// numeric/symbolic payload dispatch with automatic promotion.
#pragma once

#include <utility>
#include <vector>

#include "qarpx/operators/coeff.h"

#ifdef QARP_WITH_SYMENGINE
#include <algorithm>
#include <string>

#include <symengine/visitor.h>
#endif

namespace qarpx::ops::detail {

/// Run fn(engine_a, engine_b) with both operands on the same backend,
/// promoting to symbolic when either side is symbolic (openfermion/
/// PauliEngine behaviour: promotion happens before the hot loop).
template <typename Op, typename Fn>
Op& binary_inplace(Op& self, const Op& other, Fn&& fn) {
#ifdef QARP_WITH_SYMENGINE
    if (self.is_symbolic() || other.is_symbolic()) {
        self.promote_to_symbolic();
        if (other.is_symbolic()) {
            fn(self.symbolic(), other.symbolic());
        } else {
            Op tmp(other);
            tmp.promote_to_symbolic();
            fn(self.symbolic(), tmp.symbolic());
        }
        return self;
    }
#endif
    fn(self.numeric(), other.numeric());
    return self;
}

template <typename Op>
bool payload_isclose(const Op& a, const Op& b, double tol) {
#ifdef QARP_WITH_SYMENGINE
    if (a.is_symbolic() || b.is_symbolic()) {
        Op pa(a), pb(b);
        pa.promote_to_symbolic();
        pb.promote_to_symbolic();
        return pa.symbolic().isclose(pb.symbolic(), tol);
    }
#endif
    return a.numeric().isclose(b.numeric(), tol);
}

#ifdef QARP_WITH_SYMENGINE

/// Numeric engine → symbolic engine, term order preserved.
template <typename SymEngineT, typename NumEngineT>
SymEngineT promote_engine(const NumEngineT& n) {
    SymEngineT out;
    out.terms.reserve(n.terms.size());
    for (const auto& e : n.terms) out.terms.set(e.key, SymCoeff(e.coeff));
    return out;
}

/// Symbolic engine → numeric engine when every coefficient is numeric;
/// returns false (out untouched) otherwise.
template <typename NumEngineT, typename SymEngineT>
bool try_demote_engine(const SymEngineT& s, NumEngineT& out) {
    for (const auto& e : s.terms)
        if (!e.coeff.is_numeric()) return false;
    out = NumEngineT();
    out.terms.reserve(s.terms.size());
    for (const auto& e : s.terms)
        out.terms.set(e.key, CoeffTraits<SymCoeff>::to_complex(e.coeff));
    return true;
}

/// In-place substitution of named symbols with numeric values.
template <typename SymEngineT>
void substitute_engine(SymEngineT& eng,
                       const std::vector<std::pair<std::string, std::complex<double>>>& values) {
    SymEngine::map_basic_basic subs;
    for (const auto& [name, value] : values) {
        subs[SymEngine::symbol(name)] =
            value.imag() == 0.0
                ? SymEngine::RCP<const SymEngine::Basic>(SymEngine::real_double(value.real()))
                : SymEngine::RCP<const SymEngine::Basic>(SymEngine::complex_double(value));
    }
    eng.terms.for_each_coeff([&subs](const auto&, SymCoeff& c) {
        c = SymCoeff(c.expr.subs(subs));
    });
}

/// Sorted union of free-symbol names across all coefficients.
template <typename SymEngineT>
std::vector<std::string> engine_free_symbols(const SymEngineT& eng) {
    std::vector<std::string> names;
    for (const auto& e : eng.terms) {
        for (const auto& sym : SymEngine::free_symbols(*e.coeff.expr.get_basic())) {
            const auto* s = SymEngine::down_cast<const SymEngine::Symbol*>(sym.get());
            names.push_back(s->get_name());
        }
    }
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());
    return names;
}

#endif  // QARP_WITH_SYMENGINE

}  // namespace qarpx::ops::detail
