#include "qarpx/operators/qubit_operator.h"

#include <cctype>
#include <stdexcept>

#include "detail_facade.h"
#include "detail_parse.h"

namespace qarpx::ops {

namespace {

using Complex = QubitOperator::Complex;

/// Fold one (qubit, pauli) factor into an under-construction term, exactly
/// like openfermion's _simplify: repeated qubits multiply through the Pauli
/// product table; returns the i^k phase exponent contributed.
int fold_factor(PackedPauli& term, uint32_t qubit, Pauli p) {
    const Pauli existing = term.pauli_at(qubit);
    if (existing == Pauli::I) {
        term.set_pauli(qubit, p);
        return 0;
    }
    const auto [prod, k] = single_pauli_product(existing, p);
    term.set_pauli(qubit, prod);
    return k;
}

struct ParsedTerm {
    PackedPauli term;
    int phase = 0;  // accumulated i^phase from repeated-qubit folding
};

/// Parse "X0 Z2 Y3"-style factor lists (openfermion QubitOperator string
/// form: upper-case action letter followed by the qubit index).
ParsedTerm parse_factor_list(std::string_view s) {
    ParsedTerm out;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
        if (i >= s.size()) break;
        size_t j = i;
        while (j < s.size() && !std::isspace(static_cast<unsigned char>(s[j]))) ++j;
        const std::string_view factor = s.substr(i, j - i);
        i = j;

        if (factor.size() < 2)
            throw std::invalid_argument("Invalid factor '" + std::string(factor) + "'.");
        const char letter = factor[0];
        if (letter != 'X' && letter != 'Y' && letter != 'Z')
            throw std::invalid_argument("Invalid factor '" + std::string(factor) + "'.");
        uint32_t qubit = 0;
        for (size_t k = 1; k < factor.size(); ++k) {
            const char c = factor[k];
            if (c < '0' || c > '9')
                throw std::invalid_argument("Invalid factor '" + std::string(factor) + "'.");
            qubit = qubit * 10 + static_cast<uint32_t>(c - '0');
        }
        out.phase += fold_factor(out.term, qubit, pauli_from_letter(letter));
    }
    out.term.canonicalize();
    return out;
}

ParsedTerm parse_factor_pairs(std::span<const std::pair<uint32_t, char>> factors) {
    ParsedTerm parsed;
    for (const auto& [qubit, letter] : factors)
        parsed.phase += fold_factor(parsed.term, qubit, pauli_from_letter(letter));
    parsed.term.canonicalize();
    return parsed;
}

}  // namespace

// ── Construction (numeric) ────────────────────────────────────────────────

QubitOperator QubitOperator::from_term_string(std::string_view term, Complex coefficient) {
    const std::string_view factors = detail::split_bracket_form(term, coefficient);
    const ParsedTerm parsed = parse_factor_list(factors);
    return from_packed_term(parsed.term,
                            CoeffTraits<Complex>::mul_i_pow(coefficient, parsed.phase));
}

QubitOperator QubitOperator::from_factors(std::span<const std::pair<uint32_t, char>> factors,
                                          Complex coefficient) {
    const ParsedTerm parsed = parse_factor_pairs(factors);
    return from_packed_term(parsed.term,
                            CoeffTraits<Complex>::mul_i_pow(coefficient, parsed.phase));
}

QubitOperator QubitOperator::from_packed_term(PackedPauli term, Complex coefficient) {
    QubitOperator out;
    out.numeric().set_term(term, coefficient);
    return out;
}

QubitOperator QubitOperator::identity(Complex c) {
    return from_packed_term(PackedPauli{}, c);
}

QubitOperator QubitOperator::from_engine(Numeric engine) {
    QubitOperator out;
    out.impl_ = std::move(engine);
    return out;
}

// ── Payload access ────────────────────────────────────────────────────────

#ifdef QARP_WITH_SYMENGINE

QubitOperator QubitOperator::from_term_string(std::string_view term, SymCoeff coefficient) {
    // The bracketed prefix stays numeric (it comes from term-string syntax);
    // only fold it in when actually present so plain terms don't get their
    // expressions contaminated with a spurious ·1.0 float factor.
    Complex prefix(1.0, 0.0);
    const std::string_view factors = detail::split_bracket_form(term, prefix);
    const ParsedTerm parsed = parse_factor_list(factors);
    SymCoeff c = prefix == Complex(1.0, 0.0) ? std::move(coefficient)
                                             : coefficient * SymCoeff(prefix);
    return from_packed_term(parsed.term,
                            CoeffTraits<SymCoeff>::mul_i_pow(c, parsed.phase));
}

QubitOperator QubitOperator::from_factors(std::span<const std::pair<uint32_t, char>> factors,
                                          SymCoeff coefficient) {
    const ParsedTerm parsed = parse_factor_pairs(factors);
    return from_packed_term(parsed.term,
                            CoeffTraits<SymCoeff>::mul_i_pow(coefficient, parsed.phase));
}

QubitOperator QubitOperator::from_packed_term(PackedPauli term, SymCoeff coefficient) {
    QubitOperator out;
    Symbolic engine;
    engine.set_term(term, std::move(coefficient));
    out.impl_ = std::move(engine);
    return out;
}

QubitOperator QubitOperator::from_engine(Symbolic engine) {
    QubitOperator out;
    out.impl_ = std::move(engine);
    return out;
}

bool QubitOperator::is_symbolic() const {
    return std::holds_alternative<Symbolic>(impl_);
}

QubitOperator::Numeric& QubitOperator::numeric() {
    if (auto* n = std::get_if<Numeric>(&impl_)) return *n;
    throw std::invalid_argument(
        "operator has symbolic coefficients; substitute numeric values first.");
}
const QubitOperator::Numeric& QubitOperator::numeric() const {
    if (const auto* n = std::get_if<Numeric>(&impl_)) return *n;
    throw std::invalid_argument(
        "operator has symbolic coefficients; substitute numeric values first.");
}

QubitOperator::Symbolic& QubitOperator::symbolic() {
    if (auto* s = std::get_if<Symbolic>(&impl_)) return *s;
    throw std::invalid_argument("operator has numeric coefficients.");
}
const QubitOperator::Symbolic& QubitOperator::symbolic() const {
    if (const auto* s = std::get_if<Symbolic>(&impl_)) return *s;
    throw std::invalid_argument("operator has numeric coefficients.");
}

void QubitOperator::promote_to_symbolic() {
    if (is_symbolic()) return;
    impl_ = detail::promote_engine<Symbolic>(std::get<Numeric>(impl_));
}

QubitOperator& QubitOperator::iadd_scalar(const SymCoeff& s) {
    promote_to_symbolic();
    symbolic().iadd_constant(s);
    return *this;
}

QubitOperator& QubitOperator::imul_scalar(const SymCoeff& s) {
    promote_to_symbolic();
    symbolic().imul_scalar(s);
    return *this;
}

QubitOperator QubitOperator::substituted(
    const std::vector<std::pair<std::string, Complex>>& values) const {
    if (!is_symbolic()) return *this;
    Symbolic engine = symbolic();
    detail::substitute_engine(engine, values);
    Numeric demoted;
    if (detail::try_demote_engine(engine, demoted)) return from_engine(std::move(demoted));
    return from_engine(std::move(engine));
}

std::vector<std::string> QubitOperator::free_symbols() const {
    if (!is_symbolic()) return {};
    return detail::engine_free_symbols(symbolic());
}

#else  // !QARP_WITH_SYMENGINE

bool QubitOperator::is_symbolic() const { return false; }
QubitOperator::Numeric& QubitOperator::numeric() { return impl_; }
const QubitOperator::Numeric& QubitOperator::numeric() const { return impl_; }

#endif  // QARP_WITH_SYMENGINE

// ── Arithmetic ────────────────────────────────────────────────────────────

QubitOperator& QubitOperator::operator+=(const QubitOperator& o) {
    return detail::binary_inplace(*this, o, [](auto& a, const auto& b) { a.iadd(b); });
}
QubitOperator& QubitOperator::operator-=(const QubitOperator& o) {
    return detail::binary_inplace(*this, o, [](auto& a, const auto& b) { a.isub(b); });
}
QubitOperator& QubitOperator::operator*=(const QubitOperator& o) {
    return detail::binary_inplace(*this, o, [](auto& a, const auto& b) { a = a.mul(b); });
}

QubitOperator& QubitOperator::operator+=(Complex s) {
    visit([&](auto& eng) {
        using C = typename std::decay_t<decltype(eng)>::Traits::value_type;
        eng.iadd_constant(C(s));
    });
    return *this;
}
QubitOperator& QubitOperator::operator-=(Complex s) { return *this += -s; }
QubitOperator& QubitOperator::operator*=(Complex s) {
    visit([&](auto& eng) {
        using C = typename std::decay_t<decltype(eng)>::Traits::value_type;
        eng.imul_scalar(C(s));
    });
    return *this;
}
QubitOperator& QubitOperator::operator/=(Complex s) {
    return *this *= (Complex(1.0, 0.0) / s);
}

QubitOperator QubitOperator::operator-() const {
    QubitOperator out(*this);
    out *= Complex(-1.0, 0.0);
    return out;
}

QubitOperator QubitOperator::pow(int exponent) const {
    if (exponent < 0)
        throw std::invalid_argument(
            "exponent must be a non-negative int, but was " + std::to_string(exponent));
    QubitOperator out = identity(1.0);
    for (int i = 0; i < exponent; ++i) out *= *this;
    return out;
}

bool QubitOperator::isclose(const QubitOperator& o, double tol) const {
    return detail::payload_isclose(*this, o, tol);
}

QubitOperator QubitOperator::hermitian_conjugated() const {
    return visit([](const auto& eng) { return from_engine(eng.conjugated()); });
}

int QubitOperator::count_qubits() const {
    return visit([](const auto& eng) { return eng.count_qubits(); });
}

QubitOperator::Complex QubitOperator::constant() const {
    return visit([](const auto& eng) {
        using Eng = std::decay_t<decltype(eng)>;
        return Eng::Traits::to_complex(eng.constant());
    });
}

void QubitOperator::set_constant(Complex c) {
    visit([&](auto& eng) {
        using C = typename std::decay_t<decltype(eng)>::Traits::value_type;
        eng.set_constant(C(c));
    });
}

size_t QubitOperator::n_terms() const {
    return visit([](const auto& eng) { return eng.n_terms(); });
}

void QubitOperator::compress(double abs_tol) {
    visit([&](auto& eng) { eng.compress(abs_tol); });
}

std::vector<QubitOperator> QubitOperator::get_operators() const {
    return visit([](const auto& eng) {
        std::vector<QubitOperator> out;
        out.reserve(eng.n_terms());
        for (const auto& e : eng.terms) out.push_back(from_packed_term(e.key, e.coeff));
        return out;
    });
}

std::string QubitOperator::str() const {
    return visit([](const auto& eng) -> std::string {
        if (eng.n_terms() == 0) return "0";
        using Eng = std::decay_t<decltype(eng)>;
        using C = typename Eng::Traits::value_type;
        std::string s;
        std::vector<std::pair<uint32_t, char>> factors;
        bool first = true;
        for (const auto& e : eng.terms) {
            if (!first) s += " +\n";
            first = false;
            if constexpr (std::is_same_v<C, Complex>) {
                s += detail::format_python_complex(e.coeff);
            }
#ifdef QARP_WITH_SYMENGINE
            else {
                s += symbolic_to_string(e.coeff);
            }
#endif
            s += " [";
            packed_to_index_pairs(e.key, factors);
            for (size_t i = 0; i < factors.size(); ++i) {
                if (i > 0) s += ' ';
                s += factors[i].second;
                s += std::to_string(factors[i].first);
            }
            s += ']';
        }
        return s;
    });
}

}  // namespace qarpx::ops
