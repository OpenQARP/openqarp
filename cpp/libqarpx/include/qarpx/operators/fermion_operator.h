#pragma once

#include <complex>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef QARP_WITH_SYMENGINE
#include <variant>
#endif

#include "qarpx/operators/fermion_key.h"
#include "qarpx/operators/term_operator.h"

namespace qarpx::ops {

/// Coefficient-templated FermionOperator engine: a weighted sum of ladder
/// operator sequences.  No canonicalization anywhere — openfermion never
/// normal-orders on construction or multiplication.
template <typename Coeff>
class FermionOperatorT : public TermOperatorT<FermionKey, Coeff> {
    using Base = TermOperatorT<FermionKey, Coeff>;

public:
    using Traits = typename Base::Traits;

    /// openfermion __mul__: nested loop, key concatenation, collisions
    /// accumulated WITHOUT the small-erase.
    FermionOperatorT mul(const FermionOperatorT& o) const {
        FermionOperatorT out;
        out.terms.reserve(this->terms.size() > o.terms.size() ? this->terms.size()
                                                              : o.terms.size());
        for (const auto& a : this->terms)
            for (const auto& b : o.terms)
                out.terms.accumulate_no_compact(a.key.concat(b.key), a.coeff * b.coeff);
        return out;
    }

    /// openfermion hermitian_conjugated: reversed sequence, flipped actions,
    /// conjugated coefficient.  dagger() is a key bijection, so inserting in
    /// iteration order can never collide — matching openfermion's fresh-dict
    /// direct assignment.
    FermionOperatorT conjugated() const {
        FermionOperatorT out;
        out.terms.reserve(this->terms.size());
        for (const auto& e : this->terms)
            out.terms.set(e.key.dagger(), Traits::conj(e.coeff));
        return out;
    }

    /// openfermion count_qubits: highest mode index + 1 (0 for constants).
    int count_qubits() const {
        int m = -1;
        for (const auto& e : this->terms) {
            const int idx = e.key.max_index();
            if (idx > m) m = idx;
        }
        return m + 1;
    }
};

/// The user-facing FermionOperator bound into Python — openfermion-
/// compatible for the surface qarp exercises.  Payload/promotion semantics
/// identical to QubitOperator (see its header).
class FermionOperator {
public:
    using Complex = std::complex<double>;
    using Numeric = FermionOperatorT<Complex>;
#ifdef QARP_WITH_SYMENGINE
    using Symbolic = FermionOperatorT<SymCoeff>;
#endif

    /// Additive zero (openfermion FermionOperator()); numeric payload.
    FermionOperator() = default;

    /// String form, e.g. ("2^ 1", -0.5).  "" is the identity term; the
    /// bracketed form "1.5 [2^ 3]" folds the prefix into the coefficient.
    /// The sequence is stored verbatim — no reordering.
    static FermionOperator from_term_string(std::string_view term, Complex coefficient = 1.0);

    /// Tuple form: ladder ops (mode_index, action) with action 1 = creation,
    /// 0 = annihilation, stored in the given order.
    static FermionOperator from_ladder_ops(std::span<const std::pair<uint32_t, uint32_t>> ops,
                                           Complex coefficient = 1.0);

    /// Single already-built term (.terms setter, converters).
    static FermionOperator from_key(FermionKey key, Complex coefficient);

    static FermionOperator identity(Complex c = 1.0);

#ifdef QARP_WITH_SYMENGINE
    static FermionOperator from_term_string(std::string_view term, SymCoeff coefficient);
    static FermionOperator from_ladder_ops(std::span<const std::pair<uint32_t, uint32_t>> ops,
                                           SymCoeff coefficient);
    static FermionOperator from_key(FermionKey key, SymCoeff coefficient);

    static FermionOperator from_engine(Symbolic engine);
#endif
    static FermionOperator from_engine(Numeric engine);

    // ── payload access ──
    bool is_symbolic() const;
    Numeric& numeric();
    const Numeric& numeric() const;
#ifdef QARP_WITH_SYMENGINE
    Symbolic& symbolic();
    const Symbolic& symbolic() const;
    void promote_to_symbolic();
#endif

    template <typename F>
    decltype(auto) visit(F&& f) {
#ifdef QARP_WITH_SYMENGINE
        return std::visit(std::forward<F>(f), impl_);
#else
        return std::forward<F>(f)(impl_);
#endif
    }
    template <typename F>
    decltype(auto) visit(F&& f) const {
#ifdef QARP_WITH_SYMENGINE
        return std::visit(std::forward<F>(f), impl_);
#else
        return std::forward<F>(f)(impl_);
#endif
    }

    // ── openfermion dunder surface ──
    FermionOperator& operator+=(const FermionOperator& o);
    FermionOperator& operator-=(const FermionOperator& o);
    FermionOperator& operator*=(const FermionOperator& o);
    FermionOperator& operator+=(Complex s);
    FermionOperator& operator-=(Complex s);
    FermionOperator& operator*=(Complex s);
    FermionOperator& operator/=(Complex s);
    FermionOperator operator-() const;

#ifdef QARP_WITH_SYMENGINE
    FermionOperator& iadd_scalar(const SymCoeff& s);
    FermionOperator& imul_scalar(const SymCoeff& s);
#endif

    friend FermionOperator operator+(FermionOperator a, const FermionOperator& b) { a += b; return a; }
    friend FermionOperator operator-(FermionOperator a, const FermionOperator& b) { a -= b; return a; }
    friend FermionOperator operator*(FermionOperator a, const FermionOperator& b) { a *= b; return a; }
    friend FermionOperator operator+(FermionOperator a, Complex s) { a += s; return a; }
    friend FermionOperator operator+(Complex s, FermionOperator a) { a += s; return a; }
    friend FermionOperator operator-(FermionOperator a, Complex s) { a -= s; return a; }
    friend FermionOperator operator-(Complex s, const FermionOperator& a) { return (-a) + s; }
    friend FermionOperator operator*(FermionOperator a, Complex s) { a *= s; return a; }
    friend FermionOperator operator*(Complex s, FermionOperator a) { a *= s; return a; }
    friend FermionOperator operator/(FermionOperator a, Complex s) { a /= s; return a; }

    FermionOperator pow(int exponent) const;

    bool isclose(const FermionOperator& o, double tol = kEqTolerance) const;
    bool equals(const FermionOperator& o) const { return isclose(o); }
    /// NOTE: tolerance-based, mirroring the Python class (openfermion __eq__
    /// is isclose, not exact dict equality).
    friend bool operator==(const FermionOperator& a, const FermionOperator& b) { return a.equals(b); }
    friend bool operator!=(const FermionOperator& a, const FermionOperator& b) { return !a.equals(b); }

    FermionOperator hermitian_conjugated() const;

    int count_qubits() const;
    Complex constant() const;
    void set_constant(Complex c);
    size_t n_terms() const;
    void compress(double abs_tol = kEqTolerance);

    std::vector<FermionOperator> get_operators() const;

    std::string str() const;

#ifdef QARP_WITH_SYMENGINE
    /// Substitute named symbols with numeric values; the result demotes to
    /// the numeric backend when no free symbols remain (qarpx extra).
    FermionOperator substituted(
        const std::vector<std::pair<std::string, Complex>>& values) const;
    /// Sorted names of the free symbols across all coefficients.
    std::vector<std::string> free_symbols() const;
#endif

private:
#ifdef QARP_WITH_SYMENGINE
    std::variant<Numeric, Symbolic> impl_;
#else
    Numeric impl_;
#endif
};

}  // namespace qarpx::ops
