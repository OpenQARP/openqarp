#pragma once

#include <cmath>
#include <functional>
#include <memory>
#include <numbers>
#include <ostream>
#include <string>
#include <unordered_map>
#include <variant>

namespace qarpx {

/// Lightweight symbolic parameter for quantum gate arguments.
///
/// Stores either a concrete double value or a symbolic expression.
/// The symbolic representation supports the operations needed by quantum
/// circuits: negation, addition, multiplication by scalar, and substitution.
///
/// The expression representation (`Expr`) is private, so it can be replaced
/// without touching call sites.  Unrelated to `QARP_WITH_SYMENGINE`, which
/// backs operator *coefficients* (`operators/coeff.h`) and never gate params.
class Param {
public:
    /// Concrete numeric value.
    Param() : data_(0.0) {}
    Param(double v) : data_(v) {}  // NOLINT(implicit)

    /// Symbolic parameter identified by name.
    static Param symbol(const std::string& name);

    /// Symbolic expression: coeff * symbol + offset
    /// Covers the vast majority of quantum circuit parametrizations.
    static Param linear(double coeff, const std::string& symbol_name, double offset = 0.0);

    // Queries
    [[nodiscard]] bool   is_symbolic()    const;
    [[nodiscard]] bool   is_concrete()    const { return !is_symbolic(); }
    [[nodiscard]] double value()          const; // throws if symbolic
    [[nodiscard]] double value_or(double fallback) const;

    /// All symbol names appearing in this expression.
    [[nodiscard]] std::vector<std::string> free_symbols() const;

    /// Substitute symbol values. Returns a (possibly still symbolic) Param.
    [[nodiscard]] Param substitute(const std::unordered_map<std::string, double>& values) const;

    /// Rename symbols in this expression — preserves the `coeff·symbol+offset`
    /// structure where present.  Compound expressions (those built via
    /// `eval_fn`) stay evaluatable: the closure keeps resolving the old names
    /// behind a new->old translation, so the result binds under the new ones.
    [[nodiscard]] Param rename_symbols(
        const std::unordered_map<std::string, std::string>& mapping) const;

    /// Evaluate: substitute all symbols and return concrete value.
    /// Throws if any symbols remain unsubstituted.
    [[nodiscard]] double evaluate(const std::unordered_map<std::string, double>& values) const;

    // Arithmetic (returns new Param, no mutation)
    [[nodiscard]] Param operator-() const;
    [[nodiscard]] Param operator+(const Param& rhs) const;
    [[nodiscard]] Param operator-(const Param& rhs) const;
    [[nodiscard]] Param operator*(const Param& rhs) const;
    [[nodiscard]] Param operator/(const Param& rhs) const;

    // Comparison
    bool operator==(const Param& other) const;
    bool operator!=(const Param& other) const { return !(*this == other); }

    /// Tolerant comparison for "is this the same circuit" checks (e.g. after
    /// a round-trip through an SDK whose angle convention requires a
    /// division/multiplication by pi, which is not bit-exact).  Concrete
    /// values compare within `atol`.  Symbolic linear params (`coeff*sym +
    /// offset`) require the same symbol name and compare `coeff`/`offset`
    /// within `atol`.  Compound expressions (built via the `eval_fn` escape
    /// hatch) fall back to exact `to_string()` comparison since there's no
    /// symbol-value-independent way to compare them numerically.
    [[nodiscard]] bool approx_equal(const Param& other, double atol = 1e-9) const;

    // String representation
    [[nodiscard]] std::string to_string() const;
    friend std::ostream& operator<<(std::ostream& os, const Param& p);

private:
    /// Internal expression node.  Kept as a separate type so we can
    /// replace the representation later without touching the public API.
    struct Expr {
        double coeff  = 1.0;   // multiplicative coefficient
        std::string symbol;    // symbol name (empty if pure offset)
        double offset = 0.0;   // additive constant

        // For compound expressions that don't fit coeff*sym+offset,
        // we store a general expression string and an evaluator.
        // This is the escape hatch — used rarely.
        std::string repr;
        std::function<double(const std::unordered_map<std::string, double>&)> eval_fn;
        std::vector<std::string> symbols;
    };

    // Concrete value or symbolic expression
    std::variant<double, Expr> data_;

    explicit Param(Expr e) : data_(std::move(e)) {}

    const Expr& expr() const { return std::get<Expr>(data_); }
};

// Convenience factory functions (free-standing, matching spec API)
inline Param concrete(double v)                     { return Param(v); }
inline Param sym(const std::string& name)           { return Param::symbol(name); }

/// ``∂p/∂s`` for every free symbol ``s`` of ``p``, keyed by symbol name —
/// the single affine probe shared by the adjoint (gate_param_dependencies)
/// and the Python parameter-shift plan, so the two cannot drift.  Affinity is
/// checked at four probe points (0, 1, 2 per symbol and all-ones, tolerance
/// ``1e-9·max(1, |p|)``), not proved: a polynomial vanishing there passes.
/// Anything else throws ``capability_error`` — the shift rules and the adjoint
/// chain rule are exact only for affine angles.  A concrete ``p`` yields an empty map.
[[nodiscard]] std::unordered_map<std::string, double> affine_coefficients(const Param& p);

}  // namespace qarpx
