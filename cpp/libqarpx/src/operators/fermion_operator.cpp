#include "qarpx/operators/fermion_operator.h"

#include <cctype>
#include <stdexcept>

#include "detail_facade.h"
#include "detail_parse.h"

namespace qarpx::ops {

namespace {

using Complex = FermionOperator::Complex;

/// Parse "2^ 1"-style ladder factor lists (openfermion FermionOperator
/// string form: mode index with an optional trailing '^' for creation).
/// The sequence is preserved verbatim — no reordering, no folding.
FermionKey parse_ladder_list(std::string_view s) {
    FermionKey key;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
        if (i >= s.size()) break;
        size_t j = i;
        while (j < s.size() && !std::isspace(static_cast<unsigned char>(s[j]))) ++j;
        const std::string_view factor = s.substr(i, j - i);
        i = j;

        uint32_t action = 0;
        std::string_view digits = factor;
        if (!digits.empty() && digits.back() == '^') {
            action = 1;
            digits.remove_suffix(1);
        }
        if (digits.empty())
            throw std::invalid_argument("Invalid factor '" + std::string(factor) + "'.");
        uint32_t index = 0;
        for (const char c : digits) {
            if (c < '0' || c > '9')
                throw std::invalid_argument("Invalid factor '" + std::string(factor) + "'.");
            index = index * 10 + static_cast<uint32_t>(c - '0');
        }
        key.append(index, action);
    }
    return key;
}

FermionKey key_from_ladder_ops(std::span<const std::pair<uint32_t, uint32_t>> ops) {
    FermionKey key;
    key.ops.reserve(ops.size());
    for (const auto& [index, action] : ops) {
        if (action > 1)
            throw std::invalid_argument("Invalid action provided to term.");
        key.append(index, action);
    }
    return key;
}

}  // namespace

// ── Construction (numeric) ────────────────────────────────────────────────

FermionOperator FermionOperator::from_term_string(std::string_view term, Complex coefficient) {
    const std::string_view factors = detail::split_bracket_form(term, coefficient);
    return from_key(parse_ladder_list(factors), coefficient);
}

FermionOperator FermionOperator::from_ladder_ops(
    std::span<const std::pair<uint32_t, uint32_t>> ops, Complex coefficient) {
    return from_key(key_from_ladder_ops(ops), coefficient);
}

FermionOperator FermionOperator::from_key(FermionKey key, Complex coefficient) {
    FermionOperator out;
    out.numeric().set_term(key, coefficient);
    return out;
}

FermionOperator FermionOperator::identity(Complex c) {
    return from_key(FermionKey{}, c);
}

FermionOperator FermionOperator::from_engine(Numeric engine) {
    FermionOperator out;
    out.impl_ = std::move(engine);
    return out;
}

// ── Payload access ────────────────────────────────────────────────────────

#ifdef QARP_WITH_SYMENGINE

FermionOperator FermionOperator::from_term_string(std::string_view term, SymCoeff coefficient) {
    Complex prefix(1.0, 0.0);
    const std::string_view factors = detail::split_bracket_form(term, prefix);
    SymCoeff c = prefix == Complex(1.0, 0.0) ? std::move(coefficient)
                                             : coefficient * SymCoeff(prefix);
    return from_key(parse_ladder_list(factors), std::move(c));
}

FermionOperator FermionOperator::from_ladder_ops(
    std::span<const std::pair<uint32_t, uint32_t>> ops, SymCoeff coefficient) {
    return from_key(key_from_ladder_ops(ops), std::move(coefficient));
}

FermionOperator FermionOperator::from_key(FermionKey key, SymCoeff coefficient) {
    FermionOperator out;
    Symbolic engine;
    engine.set_term(key, std::move(coefficient));
    out.impl_ = std::move(engine);
    return out;
}

FermionOperator FermionOperator::from_engine(Symbolic engine) {
    FermionOperator out;
    out.impl_ = std::move(engine);
    return out;
}

bool FermionOperator::is_symbolic() const {
    return std::holds_alternative<Symbolic>(impl_);
}

FermionOperator::Numeric& FermionOperator::numeric() {
    if (auto* n = std::get_if<Numeric>(&impl_)) return *n;
    throw std::invalid_argument(
        "operator has symbolic coefficients; substitute numeric values first.");
}
const FermionOperator::Numeric& FermionOperator::numeric() const {
    if (const auto* n = std::get_if<Numeric>(&impl_)) return *n;
    throw std::invalid_argument(
        "operator has symbolic coefficients; substitute numeric values first.");
}

FermionOperator::Symbolic& FermionOperator::symbolic() {
    if (auto* s = std::get_if<Symbolic>(&impl_)) return *s;
    throw std::invalid_argument("operator has numeric coefficients.");
}
const FermionOperator::Symbolic& FermionOperator::symbolic() const {
    if (const auto* s = std::get_if<Symbolic>(&impl_)) return *s;
    throw std::invalid_argument("operator has numeric coefficients.");
}

void FermionOperator::promote_to_symbolic() {
    if (is_symbolic()) return;
    impl_ = detail::promote_engine<Symbolic>(std::get<Numeric>(impl_));
}

FermionOperator& FermionOperator::iadd_scalar(const SymCoeff& s) {
    promote_to_symbolic();
    symbolic().iadd_constant(s);
    return *this;
}

FermionOperator& FermionOperator::imul_scalar(const SymCoeff& s) {
    promote_to_symbolic();
    symbolic().imul_scalar(s);
    return *this;
}

FermionOperator FermionOperator::substituted(
    const std::vector<std::pair<std::string, Complex>>& values) const {
    if (!is_symbolic()) return *this;
    Symbolic engine = symbolic();
    detail::substitute_engine(engine, values);
    Numeric demoted;
    if (detail::try_demote_engine(engine, demoted)) return from_engine(std::move(demoted));
    return from_engine(std::move(engine));
}

std::vector<std::string> FermionOperator::free_symbols() const {
    if (!is_symbolic()) return {};
    return detail::engine_free_symbols(symbolic());
}

#else  // !QARP_WITH_SYMENGINE

bool FermionOperator::is_symbolic() const { return false; }
FermionOperator::Numeric& FermionOperator::numeric() { return impl_; }
const FermionOperator::Numeric& FermionOperator::numeric() const { return impl_; }

#endif  // QARP_WITH_SYMENGINE

// ── Arithmetic ────────────────────────────────────────────────────────────

FermionOperator& FermionOperator::operator+=(const FermionOperator& o) {
    return detail::binary_inplace(*this, o, [](auto& a, const auto& b) { a.iadd(b); });
}
FermionOperator& FermionOperator::operator-=(const FermionOperator& o) {
    return detail::binary_inplace(*this, o, [](auto& a, const auto& b) { a.isub(b); });
}
FermionOperator& FermionOperator::operator*=(const FermionOperator& o) {
    return detail::binary_inplace(*this, o, [](auto& a, const auto& b) { a = a.mul(b); });
}

FermionOperator& FermionOperator::operator+=(Complex s) {
    visit([&](auto& eng) {
        using C = typename std::decay_t<decltype(eng)>::Traits::value_type;
        eng.iadd_constant(C(s));
    });
    return *this;
}
FermionOperator& FermionOperator::operator-=(Complex s) { return *this += -s; }
FermionOperator& FermionOperator::operator*=(Complex s) {
    visit([&](auto& eng) {
        using C = typename std::decay_t<decltype(eng)>::Traits::value_type;
        eng.imul_scalar(C(s));
    });
    return *this;
}
FermionOperator& FermionOperator::operator/=(Complex s) {
    return *this *= (Complex(1.0, 0.0) / s);
}

FermionOperator FermionOperator::operator-() const {
    FermionOperator out(*this);
    out *= Complex(-1.0, 0.0);
    return out;
}

FermionOperator FermionOperator::pow(int exponent) const {
    if (exponent < 0)
        throw std::invalid_argument(
            "exponent must be a non-negative int, but was " + std::to_string(exponent));
    FermionOperator out = identity(1.0);
    for (int i = 0; i < exponent; ++i) out *= *this;
    return out;
}

bool FermionOperator::isclose(const FermionOperator& o, double tol) const {
    return detail::payload_isclose(*this, o, tol);
}

FermionOperator FermionOperator::hermitian_conjugated() const {
    return visit([](const auto& eng) { return from_engine(eng.conjugated()); });
}

int FermionOperator::count_qubits() const {
    return visit([](const auto& eng) { return eng.count_qubits(); });
}

FermionOperator::Complex FermionOperator::constant() const {
    return visit([](const auto& eng) {
        using Eng = std::decay_t<decltype(eng)>;
        return Eng::Traits::to_complex(eng.constant());
    });
}

void FermionOperator::set_constant(Complex c) {
    visit([&](auto& eng) {
        using C = typename std::decay_t<decltype(eng)>::Traits::value_type;
        eng.set_constant(C(c));
    });
}

size_t FermionOperator::n_terms() const {
    return visit([](const auto& eng) { return eng.n_terms(); });
}

void FermionOperator::compress(double abs_tol) {
    visit([&](auto& eng) { eng.compress(abs_tol); });
}

std::vector<FermionOperator> FermionOperator::get_operators() const {
    return visit([](const auto& eng) {
        std::vector<FermionOperator> out;
        out.reserve(eng.n_terms());
        for (const auto& e : eng.terms) out.push_back(from_key(e.key, e.coeff));
        return out;
    });
}

std::string FermionOperator::str() const {
    return visit([](const auto& eng) -> std::string {
        if (eng.n_terms() == 0) return "0";
        using Eng = std::decay_t<decltype(eng)>;
        using C = typename Eng::Traits::value_type;
        std::string s;
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
            for (size_t i = 0; i < e.key.ops.size(); ++i) {
                if (i > 0) s += ' ';
                s += std::to_string(FermionKey::index_of(e.key.ops[i]));
                if (FermionKey::action_of(e.key.ops[i]) == 1) s += '^';
            }
            s += ']';
        }
        return s;
    });
}

}  // namespace qarpx::ops
