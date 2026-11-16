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

#include "qarpx/operators/packed_pauli.h"
#include "qarpx/operators/term_operator.h"

namespace qarpx::ops {

/// Coefficient-templated QubitOperator engine: a weighted sum of Pauli
/// strings over the packed symplectic keys.
template <typename Coeff>
class QubitOperatorT : public TermOperatorT<PackedPauli, Coeff> {
    using Base = TermOperatorT<PackedPauli, Coeff>;

public:
    using Traits = typename Base::Traits;

    /// openfermion __mul__: nested loop, self outer / other inner, XOR keys
    /// with i^k phase folded into the coefficient, collisions accumulated
    /// WITHOUT the small-erase (exact cancellations stay as zero terms).
    QubitOperatorT mul(const QubitOperatorT& o) const {
        QubitOperatorT out;
        out.terms.reserve(this->terms.size() > o.terms.size() ? this->terms.size()
                                                              : o.terms.size());
        for (const auto& a : this->terms) {
            for (const auto& b : o.terms) {
                const int k = phase_exponent_mod4(a.key, b.key);
                out.terms.accumulate_no_compact(
                    pauli_xor(a.key, b.key),
                    Traits::mul_i_pow(a.coeff * b.coeff, k));
            }
        }
        return out;
    }

    /// Hermitian conjugate: same keys (Pauli strings are Hermitian), each
    /// coefficient conjugated, order preserved.
    QubitOperatorT conjugated() const {
        QubitOperatorT out;
        out.terms.reserve(this->terms.size());
        for (const auto& e : this->terms) out.terms.set(e.key, Traits::conj(e.coeff));
        return out;
    }

    /// openfermion count_qubits: highest qubit index + 1 (0 for a constant
    /// or empty operator).
    int count_qubits() const {
        int m = -1;
        for (const auto& e : this->terms) {
            const int q = e.key.max_qubit();
            if (q > m) m = q;
        }
        return m + 1;
    }

    /// The simulator observable ABI (qarp_simulator.h batch_expectation /
    /// run_gradient): [(sparse [(qubit, 'X'|'Y'|'Z')], complex coeff)].
    /// Throws for non-numeric coefficients.
    void to_observable(
        std::vector<std::pair<std::vector<std::pair<uint32_t, char>>,
                              std::complex<double>>>& out) const {
        out.clear();
        out.reserve(this->terms.size());
        std::vector<std::pair<uint32_t, char>> factors;
        for (const auto& e : this->terms) {
            packed_to_index_pairs(e.key, factors);
            out.emplace_back(factors, Traits::to_complex(e.coeff));
        }
    }
};

/// The user-facing QubitOperator bound into Python — openfermion-compatible
/// for the API surface qarp exercises.  With QARP_WITH_SYMENGINE the payload
/// is a numeric/symbolic variant with automatic promotion: any binary
/// operation touching a symbolic operand promotes both sides before the hot
/// loop runs (two monomorphic code paths, never mixed-type inner loops).
class QubitOperator {
public:
    using Complex = std::complex<double>;
    using Numeric = QubitOperatorT<Complex>;
#ifdef QARP_WITH_SYMENGINE
    using Symbolic = QubitOperatorT<SymCoeff>;
#endif

    /// Additive zero (openfermion QubitOperator()); numeric payload.
    QubitOperator() = default;

    /// String form, e.g. ("X0 Z2 Y3", 0.5).  "" is the identity term; the
    /// bracketed form "1.5 [X0 Z1]" folds the prefix into the coefficient.
    /// Repeated qubits multiply out through the Pauli product table with the
    /// phase folded into the coefficient, exactly like openfermion's
    /// _simplify — so ("X0 X0", c) is the identity with coefficient c.
    static QubitOperator from_term_string(std::string_view term, Complex coefficient = 1.0);

    /// Tuple form: factors (qubit, 'X'|'Y'|'Z') in the given order, folded
    /// like the string form.  Throws std::invalid_argument on bad letters.
    static QubitOperator from_factors(std::span<const std::pair<uint32_t, char>> factors,
                                      Complex coefficient = 1.0);

    /// Single already-canonical term (transform kernels, .terms setter).
    static QubitOperator from_packed_term(PackedPauli term, Complex coefficient);

    static QubitOperator identity(Complex c = 1.0);

#ifdef QARP_WITH_SYMENGINE
    static QubitOperator from_term_string(std::string_view term, SymCoeff coefficient);
    static QubitOperator from_factors(std::span<const std::pair<uint32_t, char>> factors,
                                      SymCoeff coefficient);
    static QubitOperator from_packed_term(PackedPauli term, SymCoeff coefficient);

    static QubitOperator from_engine(Symbolic engine);
#endif
    static QubitOperator from_engine(Numeric engine);

    // ── payload access ──
    bool is_symbolic() const;

    /// The numeric engine; throws std::invalid_argument when the payload is
    /// symbolic (e.g. sparse/observable kernels need numeric coefficients).
    Numeric& numeric();
    const Numeric& numeric() const;

#ifdef QARP_WITH_SYMENGINE
    Symbolic& symbolic();
    const Symbolic& symbolic() const;
    /// Convert the payload to the symbolic backend in place (term order
    /// preserved); no-op if already symbolic.
    void promote_to_symbolic();
#endif

    /// Apply f to whichever engine holds the payload (f must accept both
    /// engine types and return the same type for each).
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
    QubitOperator& operator+=(const QubitOperator& o);
    QubitOperator& operator-=(const QubitOperator& o);
    QubitOperator& operator*=(const QubitOperator& o);
    QubitOperator& operator+=(Complex s);
    QubitOperator& operator-=(Complex s);
    QubitOperator& operator*=(Complex s);
    QubitOperator& operator/=(Complex s);
    QubitOperator operator-() const;

#ifdef QARP_WITH_SYMENGINE
    QubitOperator& iadd_scalar(const SymCoeff& s);
    QubitOperator& imul_scalar(const SymCoeff& s);
#endif

    friend QubitOperator operator+(QubitOperator a, const QubitOperator& b) { a += b; return a; }
    friend QubitOperator operator-(QubitOperator a, const QubitOperator& b) { a -= b; return a; }
    friend QubitOperator operator*(QubitOperator a, const QubitOperator& b) { a *= b; return a; }
    friend QubitOperator operator+(QubitOperator a, Complex s) { a += s; return a; }
    friend QubitOperator operator+(Complex s, QubitOperator a) { a += s; return a; }
    friend QubitOperator operator-(QubitOperator a, Complex s) { a -= s; return a; }
    friend QubitOperator operator-(Complex s, const QubitOperator& a) { return (-a) + s; }
    friend QubitOperator operator*(QubitOperator a, Complex s) { a *= s; return a; }
    friend QubitOperator operator*(Complex s, QubitOperator a) { a *= s; return a; }
    friend QubitOperator operator/(QubitOperator a, Complex s) { a /= s; return a; }

    /// exponent ≥ 0; throws std::invalid_argument otherwise (openfermion
    /// raises ValueError).
    QubitOperator pow(int exponent) const;

    /// Python __eq__ semantics (openfermion isclose at EQ_TOLERANCE).
    bool isclose(const QubitOperator& o, double tol = kEqTolerance) const;
    bool equals(const QubitOperator& o) const { return isclose(o); }
    /// NOTE: tolerance-based, mirroring the Python class (openfermion __eq__
    /// is isclose, not exact dict equality).
    friend bool operator==(const QubitOperator& a, const QubitOperator& b) { return a.equals(b); }
    friend bool operator!=(const QubitOperator& a, const QubitOperator& b) { return !a.equals(b); }

    QubitOperator hermitian_conjugated() const;
    bool is_hermitian(double tol = kEqTolerance) const {
        return isclose(hermitian_conjugated(), tol);
    }

    int count_qubits() const;
    /// Constant (identity) term as complex; throws for a symbolic constant.
    Complex constant() const;
    void set_constant(Complex c);
    size_t n_terms() const;
    void compress(double abs_tol = kEqTolerance);

    /// openfermion get_operators(): one single-term operator per term, in
    /// .terms order (zero coefficients included).
    std::vector<QubitOperator> get_operators() const;

    /// openfermion __str__ ("0" when empty; "coeff [X0 Y1] +\n..." rows).
    std::string str() const;

#ifdef QARP_WITH_SYMENGINE
    /// Substitute named symbols with numeric values; the result demotes to
    /// the numeric backend when no free symbols remain (qarpx extra).
    QubitOperator substituted(
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
