#pragma once

#include <cmath>
#include <complex>

#ifdef QARP_WITH_SYMENGINE
#include <stdexcept>
#include <string>
#include <utility>

#include <symengine/basic.h>
#include <symengine/complex_double.h>
#include <symengine/constants.h>
#include <symengine/eval_double.h>
#include <symengine/expression.h>
#include <symengine/functions.h>
#include <symengine/integer.h>
#include <symengine/real_double.h>
#endif

namespace qarpx::ops {

/// openfermion's EQ_TOLERANCE: the default tolerance of operator equality
/// (isclose) and of the explicit compress().  Unlike openfermion it is *not*
/// applied by += / -=, which compact only on exact cancellation.
inline constexpr double kEqTolerance = 1e-8;

/// Coefficient abstraction for the operator term containers.  The numeric
/// backend is std::complex<double>; a SymEngine::Expression backend slots in
/// behind QARP_WITH_SYMENGINE with the same trait surface.  Arithmetic
/// (+, -, *, unary -) is used directly on the coefficient type; the traits
/// cover only what plain operators cannot express.
template <typename C>
struct CoeffTraits;

template <>
struct CoeffTraits<std::complex<double>> {
    using value_type = std::complex<double>;

    static value_type zero() { return {0.0, 0.0}; }
    static value_type one() { return {1.0, 0.0}; }
    static value_type conj(const value_type& c) { return std::conj(c); }

    /// Truncation predicate used by compress() and isclose() (openfermion:
    /// abs(coeff) < tol).  A symbolic backend returns true only for *numeric*
    /// constants below tol — symbolic expressions are never auto-deleted.
    static bool is_small(const value_type& c, double tol) {
        return std::abs(c) < tol;
    }

    /// Compaction predicate used by += / -=: exact cancellation only.  A
    /// tolerance here would be absolute while operator norms are not, so it
    /// would delete the terms of a legitimately small operator; truncation is
    /// compress()'s job, where the caller chooses the scale.
    static bool is_zero(const value_type& c) { return c == value_type{0.0, 0.0}; }

    static bool is_numeric(const value_type&) { return true; }
    static std::complex<double> to_complex(const value_type& c) { return c; }

    /// c · i^k — the phase produced by Pauli-string products (k mod 4).
    static value_type mul_i_pow(const value_type& c, int k) {
        switch (((k % 4) + 4) % 4) {
            case 0: return c;
            case 1: return {-c.imag(), c.real()};
            case 2: return -c;
            default: return {c.imag(), -c.real()};
        }
    }

    /// Structural equality for non-numeric coefficients — never reached in
    /// the numeric backend (everything is numeric); present so the shared
    /// engine templates compile.
    static bool symbolically_equal(const value_type& a, const value_type& b) {
        return a == b;
    }
};

#ifdef QARP_WITH_SYMENGINE

/// Symbolic coefficient: a thin value wrapper around SymEngine::Expression
/// with the arithmetic the engine templates use directly (+, -, *, unary -).
struct SymCoeff {
    SymEngine::Expression expr;

    SymCoeff() : expr(0) {}
    explicit SymCoeff(SymEngine::Expression e) : expr(std::move(e)) {}
    explicit SymCoeff(const std::complex<double>& c)
        : expr(c.imag() == 0.0
                   ? SymEngine::Expression(SymEngine::real_double(c.real()))
                   : SymEngine::Expression(SymEngine::complex_double(c))) {}

    friend SymCoeff operator+(const SymCoeff& a, const SymCoeff& b) {
        return SymCoeff(a.expr + b.expr);
    }
    friend SymCoeff operator-(const SymCoeff& a, const SymCoeff& b) {
        return SymCoeff(a.expr - b.expr);
    }
    friend SymCoeff operator*(const SymCoeff& a, const SymCoeff& b) {
        return SymCoeff(a.expr * b.expr);
    }
    SymCoeff operator-() const { return SymCoeff(-expr); }

    bool is_numeric() const {
        return SymEngine::is_a_Number(*expr.get_basic());
    }
};

template <>
struct CoeffTraits<SymCoeff> {
    using value_type = SymCoeff;

    static value_type zero() { return SymCoeff(); }
    static value_type one() { return SymCoeff(SymEngine::Expression(1)); }

    static value_type conj(const value_type& c) {
        return SymCoeff(SymEngine::Expression(
            SymEngine::conjugate(c.expr.get_basic())));
    }

    /// Truncation: only *numeric* constants below tol are deleted — a
    /// symbolic expression is never auto-erased (sympy parity).
    static bool is_small(const value_type& c, double tol) {
        if (!c.is_numeric()) return false;
        return std::abs(to_complex(c)) < tol;
    }

    /// Exact-cancellation predicate for += / -=; symbolic expressions are
    /// never auto-erased, so an unevaluated `x - x` survives.
    static bool is_zero(const value_type& c) {
        if (!c.is_numeric()) return false;
        return to_complex(c) == std::complex<double>{0.0, 0.0};
    }

    static bool is_numeric(const value_type& c) { return c.is_numeric(); }

    /// Numeric evaluation; throws std::invalid_argument for expressions
    /// with free symbols (callers guard on is_numeric first).
    static std::complex<double> to_complex(const value_type& c) {
        if (!c.is_numeric())
            throw std::invalid_argument(
                "operator coefficient is symbolic; substitute numeric values "
                "before numeric evaluation.");
        return SymEngine::eval_complex_double(*c.expr.get_basic());
    }

    static value_type mul_i_pow(const value_type& c, int k) {
        switch (((k % 4) + 4) % 4) {
            case 0: return c;
            case 1: return SymCoeff(c.expr * SymEngine::Expression(SymEngine::I));
            case 2: return SymCoeff(-c.expr);
            default:
                return SymCoeff(-(c.expr * SymEngine::Expression(SymEngine::I)));
        }
    }

    /// Structural equality via expand(a − b) == 0 (weaker than
    /// sympy.simplify on trig identities — documented deviation).
    static bool symbolically_equal(const value_type& a, const value_type& b) {
        const SymEngine::Expression diff = SymEngine::expand(a.expr - b.expr);
        return SymEngine::is_a<SymEngine::Integer>(*diff.get_basic()) &&
               diff.get_basic()->__eq__(*SymEngine::integer(0));
    }
};

/// Parse a (sympy-printed) expression string into a symbolic coefficient.
SymCoeff parse_symbolic(const std::string& text);

/// Canonical string form for the Python boundary (sympy.sympify-able).
std::string symbolic_to_string(const SymCoeff& c);

#endif  // QARP_WITH_SYMENGINE

}  // namespace qarpx::ops
