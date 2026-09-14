#pragma once

#include <cmath>
#include <complex>
#include <cstddef>

#include "qarpx/operators/coeff.h"
#include "qarpx/operators/term_map.h"

namespace qarpx::ops {

/// Shared engine for openfermion-style symbolic operators: a TermMap of
/// Key → Coeff with the exact openfermion mutation semantics (see
/// term_map.h).  QubitOperatorT / FermionOperatorT derive from this and add
/// the product rule and the key-specific queries.
///
/// Every loop below iterates in insertion order on purpose — reproducing
/// openfermion's dict order verbatim is a compatibility requirement, not an
/// implementation detail.
template <typename Key, typename Coeff>
class TermOperatorT {
public:
    using Traits = CoeffTraits<Coeff>;
    using Map = TermMap<Key, Coeff>;

    Map terms;

    size_t n_terms() const { return terms.size(); }

    /// Constructor path and direct `.terms[k] = v` semantics: keeps zeros.
    void set_term(const Key& key, Coeff c) { terms.set(key, std::move(c)); }

    /// Accumulate the addend's terms in its insertion order, erasing a term
    /// only on *exact* cancellation.
    ///
    /// Deviation from openfermion, which erases below EQ_TOLERANCE here.  That
    /// rule tests only the incoming terms, never the receiver's, so `a + b`
    /// and `b + a` disagreed whenever one side carried a term below the
    /// tolerance; and being absolute it deleted the terms of operators whose
    /// whole norm is small (a perturbation, a commutator that came out small)
    /// for no reason other than their units.  Truncation belongs to the
    /// explicit compress(), which the caller invokes with a scale it can see.
    void iadd(const TermOperatorT& o) {
        if (&o == this) {
            TermOperatorT copy(o);
            iadd(copy);
            return;
        }
        for (const auto& e : o.terms) terms.accumulate(e.key, e.coeff, Traits::is_zero);
    }

    void isub(const TermOperatorT& o) {
        if (&o == this) {
            TermOperatorT copy(o);
            isub(copy);
            return;
        }
        for (const auto& e : o.terms) terms.accumulate(e.key, -e.coeff, Traits::is_zero);
    }

    /// Scalar multiply in place — openfermion scales values with no
    /// compaction, so zeros survive.
    void imul_scalar(const Coeff& s) {
        terms.for_each_coeff([&s](const Key&, Coeff& c) { c = c * s; });
    }

    /// openfermion `op ± scalar` folds into the constant (identity) term by
    /// plain assignment — no small-erase.
    Coeff constant() const {
        const Coeff* c = terms.find(Key{});
        return c ? *c : Traits::zero();
    }
    void set_constant(Coeff c) { terms.set(Key{}, std::move(c)); }
    void iadd_constant(const Coeff& s) { set_constant(constant() + s); }

    /// openfermion SymbolicOperator.isclose: terms present on both sides
    /// compare with tolerance scaled by max(1, |a|, |b|) when both are
    /// numeric (symbolic pairs compare structurally, like the sympy branch);
    /// terms present on one side only must be numerically below the absolute
    /// tolerance.
    bool isclose(const TermOperatorT& o, double tol = kEqTolerance) const {
        for (const auto& a : terms) {
            if (const Coeff* b = o.terms.find(a.key)) {
                if (Traits::is_numeric(a.coeff) && Traits::is_numeric(*b)) {
                    const std::complex<double> ca = Traits::to_complex(a.coeff);
                    const std::complex<double> cb = Traits::to_complex(*b);
                    double scale = 1.0;
                    if (std::abs(ca) > scale) scale = std::abs(ca);
                    if (std::abs(cb) > scale) scale = std::abs(cb);
                    if (std::abs(ca - cb) > tol * scale) return false;
                } else if (!Traits::symbolically_equal(a.coeff, *b)) {
                    return false;
                }
            } else if (!Traits::is_numeric(a.coeff) ||
                       std::abs(Traits::to_complex(a.coeff)) > tol) {
                return false;
            }
        }
        for (const auto& b : o.terms) {
            if (terms.find(b.key)) continue;
            if (!Traits::is_numeric(b.coeff) ||
                std::abs(Traits::to_complex(b.coeff)) > tol)
                return false;
        }
        return true;
    }

    /// openfermion compress: coefficients with a negligible imaginary part
    /// become real; anything with magnitude ≤ abs_tol is dropped.  Survivor
    /// order is preserved (openfermion rebuilds the dict in iteration
    /// order).  Symbolic coefficients are kept verbatim.
    void compress(double abs_tol = kEqTolerance) {
        Map compressed;
        for (const auto& e : terms) {
            if (!Traits::is_numeric(e.coeff)) {
                compressed.set(e.key, e.coeff);
                continue;
            }
            std::complex<double> c = Traits::to_complex(e.coeff);
            if (std::abs(c.imag()) <= abs_tol) c = {c.real(), 0.0};
            if (std::abs(c) > abs_tol) compressed.set(e.key, Coeff(c));
        }
        terms = std::move(compressed);
    }
};

}  // namespace qarpx::ops
