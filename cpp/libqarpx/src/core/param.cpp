#include "qarpx/core/param.h"
#include "qarpx/core/errors.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace qarpx {

namespace {
/// Shortest decimal that round-trips back to the same double (std::to_chars,
/// as in operators/detail_parse.h). Default ostream precision (6 sig figs) is
/// lossy, so parameter text must not be formatted through it.
std::string fmt_double(double v) {
    char buf[32];
    auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), v);
    return std::string(buf, ptr);
}

/// Rewrite whole identifier tokens of a rendered expression.  Plain substring
/// replacement would corrupt names that contain one another ("p" inside "phi").
std::string rename_identifiers(
    const std::string& repr,
    const std::unordered_map<std::string, std::string>& mapping)
{
    auto is_head = [](char c) {
        return std::isalpha(static_cast<unsigned char>(c)) != 0 || c == '_';
    };
    auto is_tail = [&is_head](char c) {
        return is_head(c) || std::isdigit(static_cast<unsigned char>(c)) != 0;
    };

    std::string out;
    out.reserve(repr.size());
    for (std::size_t i = 0; i < repr.size();) {
        if (!is_head(repr[i])) {
            out += repr[i++];
            continue;
        }
        std::size_t j = i;
        while (j < repr.size() && is_tail(repr[j])) ++j;
        const std::string token = repr.substr(i, j - i);
        auto it = mapping.find(token);
        out += (it != mapping.end()) ? it->second : token;
        i = j;
    }
    return out;
}
}  // namespace

Param Param::symbol(const std::string& name) {
    Expr e;
    e.coeff = 1.0;
    e.symbol = name;
    e.offset = 0.0;
    e.symbols = {name};
    return Param(std::move(e));
}

Param Param::linear(double coeff, const std::string& symbol_name, double offset) {
    Expr e;
    e.coeff = coeff;
    e.symbol = symbol_name;
    e.offset = offset;
    e.symbols = {symbol_name};
    return Param(std::move(e));
}

bool Param::is_symbolic() const {
    return std::holds_alternative<Expr>(data_);
}

double Param::value() const {
    if (is_symbolic()) {
        throw std::runtime_error("Param::value() called on symbolic parameter: " + to_string());
    }
    return std::get<double>(data_);
}

double Param::value_or(double fallback) const {
    return is_concrete() ? std::get<double>(data_) : fallback;
}

std::vector<std::string> Param::free_symbols() const {
    if (is_concrete()) return {};
    const auto& e = expr();
    if (!e.symbols.empty()) return e.symbols;
    if (!e.symbol.empty()) return {e.symbol};
    return {};
}

Param Param::substitute(const std::unordered_map<std::string, double>& values) const {
    if (is_concrete()) return *this;

    const auto& e = expr();

    // General expression with custom evaluator
    if (e.eval_fn) {
        // Check if all symbols are provided
        bool all_resolved = true;
        for (const auto& s : e.symbols) {
            if (values.find(s) == values.end()) {
                all_resolved = false;
                break;
            }
        }
        if (all_resolved) {
            return Param(e.eval_fn(values));
        }
        return *this;  // can't partially substitute general expressions
    }

    // Linear expression: coeff * symbol + offset
    if (!e.symbol.empty()) {
        auto it = values.find(e.symbol);
        if (it != values.end()) {
            return Param(e.coeff * it->second + e.offset);
        }
        return *this;  // symbol not in values
    }

    // Pure offset (shouldn't be symbolic, but handle gracefully)
    return Param(e.offset);
}

Param Param::rename_symbols(
    const std::unordered_map<std::string, std::string>& mapping) const
{
    if (is_concrete()) return *this;
    const auto& e = expr();

    // Linear form: coeff * symbol + offset.  Rebuild via the public factory
    // so coeff/offset are preserved exactly.
    if (!e.eval_fn && !e.symbol.empty()) {
        auto it = mapping.find(e.symbol);
        const std::string& new_name = (it != mapping.end()) ? it->second : e.symbol;
        return Param::linear(e.coeff, new_name, e.offset);
    }

    // Compound expression.  The captured `eval_fn` closure resolves the *old*
    // names, so renaming the symbols vector alone yields a Param that no
    // mapping can bind: `substitute` gates on the new names, then the closure
    // looks up the old ones.  Translate old->new on the way into the closure
    // so the rename is evaluation-correct, not merely cosmetic.  Keyed by old
    // name so a collapsing rename (two symbols tied to one new name) feeds
    // both old slots from the shared new one.
    if (e.eval_fn) {
        Expr renamed = e;
        std::unordered_map<std::string, std::string> to_new;
        std::vector<std::string> new_symbols;
        for (const auto& s : renamed.symbols) {
            auto it = mapping.find(s);
            const std::string& name = (it != mapping.end()) ? it->second : s;
            if (it != mapping.end()) to_new.emplace(s, name);
            // Dedup while renaming: a collapse would otherwise report the
            // shared name once per old symbol in free_symbols().
            if (std::find(new_symbols.begin(), new_symbols.end(), name) == new_symbols.end()) {
                new_symbols.push_back(name);
            }
        }
        renamed.symbols = std::move(new_symbols);
        if (!to_new.empty()) {
            renamed.repr = rename_identifiers(e.repr, mapping);
            auto old_fn = e.eval_fn;
            renamed.eval_fn =
                [old_fn, to_new](const std::unordered_map<std::string, double>& values) {
                    // Copy first so untouched symbols still resolve; renamed
                    // ones then overwrite their old slot, which makes a
                    // simultaneous swap (a->b, b->a) come out right.
                    std::unordered_map<std::string, double> translated = values;
                    for (const auto& [old_name, new_name] : to_new) {
                        auto it = values.find(new_name);
                        if (it != values.end()) translated[old_name] = it->second;
                    }
                    return old_fn(translated);
                };
        }
        Param out{};
        out.data_ = std::move(renamed);
        return out;
    }

    return *this;
}

double Param::evaluate(const std::unordered_map<std::string, double>& values) const {
    if (is_concrete()) return std::get<double>(data_);

    Param result = substitute(values);
    if (result.is_symbolic()) {
        throw std::runtime_error(
            "Param::evaluate() — unresolved symbols remain in: " + result.to_string());
    }
    return result.value();
}

Param Param::operator-() const {
    if (is_concrete()) return Param(-std::get<double>(data_));

    const auto& e = expr();
    if (e.eval_fn) {
        // General expression: wrap the evaluator
        Expr ne;
        ne.repr = "-(" + e.repr + ")";
        ne.symbols = e.symbols;
        auto old_fn = e.eval_fn;
        ne.eval_fn = [old_fn](const auto& vals) { return -old_fn(vals); };
        return Param(std::move(ne));
    }

    Expr ne;
    ne.coeff = -e.coeff;
    ne.symbol = e.symbol;
    ne.offset = -e.offset;
    ne.symbols = e.symbols;
    return Param(std::move(ne));
}

Param Param::operator+(const Param& rhs) const {
    if (is_concrete() && rhs.is_concrete()) {
        return Param(std::get<double>(data_) + std::get<double>(rhs.data_));
    }

    // concrete + symbolic
    if (is_concrete()) {
        double lhs_val = std::get<double>(data_);
        const auto& re = rhs.expr();
        if (!re.eval_fn) {
            Expr ne;
            ne.coeff = re.coeff;
            ne.symbol = re.symbol;
            ne.offset = re.offset + lhs_val;
            ne.symbols = re.symbols;
            return Param(std::move(ne));
        }
    }

    // symbolic + concrete
    if (rhs.is_concrete()) {
        return rhs + *this;
    }

    // Same symbol: (a*x + b) + (c*x + d) = (a+c)*x + (b+d).  A concrete
    // operand reaches this point only next to a compound closure (the linear
    // branches above did not apply); it has no Expr, so it must take the
    // general path below, which evaluates both sides.
    if (!is_concrete() && !rhs.is_concrete()) {
        const auto& le = expr();
        const auto& re = rhs.expr();
        if (!le.eval_fn && !re.eval_fn && le.symbol == re.symbol) {
            Expr ne;
            ne.coeff = le.coeff + re.coeff;
            ne.symbol = le.symbol;
            ne.offset = le.offset + re.offset;
            ne.symbols = le.symbols;
            return Param(std::move(ne));
        }
    }

    // General case: create compound expression
    Expr ne;
    ne.repr = "(" + to_string() + " + " + rhs.to_string() + ")";
    // Merge symbol lists
    ne.symbols = free_symbols();
    for (const auto& s : rhs.free_symbols()) {
        bool found = false;
        for (const auto& existing : ne.symbols) {
            if (existing == s) { found = true; break; }
        }
        if (!found) ne.symbols.push_back(s);
    }
    // Capture by value for the evaluator
    Param lhs_copy = *this;
    Param rhs_copy = rhs;
    ne.eval_fn = [lhs_copy, rhs_copy](const auto& vals) {
        return lhs_copy.evaluate(vals) + rhs_copy.evaluate(vals);
    };
    return Param(std::move(ne));
}

Param Param::operator-(const Param& rhs) const {
    return *this + (-rhs);
}

Param Param::operator*(const Param& rhs) const {
    if (is_concrete() && rhs.is_concrete()) {
        return Param(std::get<double>(data_) * std::get<double>(rhs.data_));
    }

    // concrete * symbolic
    if (is_concrete()) {
        double lhs_val = std::get<double>(data_);
        const auto& re = rhs.expr();
        if (!re.eval_fn) {
            Expr ne;
            ne.coeff = re.coeff * lhs_val;
            ne.symbol = re.symbol;
            ne.offset = re.offset * lhs_val;
            ne.symbols = re.symbols;
            return Param(std::move(ne));
        }
    }

    // symbolic * concrete
    if (rhs.is_concrete()) {
        return rhs * *this;
    }

    // Both symbolic — general expression
    Expr ne;
    ne.repr = "(" + to_string() + " * " + rhs.to_string() + ")";
    ne.symbols = free_symbols();
    for (const auto& s : rhs.free_symbols()) {
        bool found = false;
        for (const auto& existing : ne.symbols) {
            if (existing == s) { found = true; break; }
        }
        if (!found) ne.symbols.push_back(s);
    }
    Param lhs_copy = *this;
    Param rhs_copy = rhs;
    ne.eval_fn = [lhs_copy, rhs_copy](const auto& vals) {
        return lhs_copy.evaluate(vals) * rhs_copy.evaluate(vals);
    };
    return Param(std::move(ne));
}

Param Param::operator/(const Param& rhs) const {
    if (rhs.is_concrete()) {
        double d = rhs.value();
        if (d == 0.0) throw std::runtime_error("Param: division by zero");
        return *this * Param(1.0 / d);
    }
    // General division
    Expr ne;
    ne.repr = "(" + to_string() + " / " + rhs.to_string() + ")";
    ne.symbols = free_symbols();
    for (const auto& s : rhs.free_symbols()) {
        bool found = false;
        for (const auto& existing : ne.symbols) {
            if (existing == s) { found = true; break; }
        }
        if (!found) ne.symbols.push_back(s);
    }
    Param lhs_copy = *this;
    Param rhs_copy = rhs;
    ne.eval_fn = [lhs_copy, rhs_copy](const auto& vals) {
        double d = rhs_copy.evaluate(vals);
        if (d == 0.0) throw std::runtime_error("Param: division by zero");
        return lhs_copy.evaluate(vals) / d;
    };
    return Param(std::move(ne));
}

bool Param::operator==(const Param& other) const {
    if (is_concrete() && other.is_concrete()) {
        return std::get<double>(data_) == std::get<double>(other.data_);
    }
    if (is_concrete() != other.is_concrete()) return false;
    // Both symbolic — compare string representations
    return to_string() == other.to_string();
}

bool Param::approx_equal(const Param& other, double atol) const {
    if (is_concrete() != other.is_concrete()) return false;
    if (is_concrete()) {
        return std::abs(std::get<double>(data_) - std::get<double>(other.data_)) <= atol;
    }
    const Expr& a = expr();
    const Expr& b = other.expr();
    // Compound (eval_fn) expressions have no symbol-independent numeric
    // comparison — fall back to exact string form.
    if (!a.repr.empty() || !b.repr.empty()) {
        return to_string() == other.to_string();
    }
    return a.symbol == b.symbol &&
           std::abs(a.coeff - b.coeff) <= atol &&
           std::abs(a.offset - b.offset) <= atol;
}

std::string Param::to_string() const {
    if (is_concrete()) {
        return fmt_double(std::get<double>(data_));
    }

    const auto& e = expr();

    // General expression
    if (!e.repr.empty()) return e.repr;

    // Linear: coeff * symbol + offset
    std::ostringstream oss;
    if (e.symbol.empty()) {
        oss << fmt_double(e.offset);
    } else if (e.offset == 0.0) {
        if (e.coeff == 1.0) {
            oss << e.symbol;
        } else if (e.coeff == -1.0) {
            oss << "-" << e.symbol;
        } else {
            oss << fmt_double(e.coeff) << "*" << e.symbol;
        }
    } else {
        if (e.coeff == 1.0) {
            oss << e.symbol;
        } else if (e.coeff == -1.0) {
            oss << "-" << e.symbol;
        } else {
            oss << fmt_double(e.coeff) << "*" << e.symbol;
        }
        if (e.offset > 0) oss << " + " << fmt_double(e.offset);
        else               oss << " - " << fmt_double(-e.offset);
    }
    return oss.str();
}

std::ostream& operator<<(std::ostream& os, const Param& p) {
    return os << p.to_string();
}

}  // namespace qarpx


namespace qarpx {

namespace {
// Private per-occurrence symbol names carry NUL separators; a message that
// reaches Python through what() is a C string and would be cut there.
std::string printable(std::string s) {
    for (auto& ch : s) if (ch == '\0') ch = '#';
    return s;
}
}  // namespace

namespace {
// Every comparison below is `abs(...) > tol`, which an inf/NaN probe value
// fails silently — it would then flow into a gradient as a number.  An
// evaluation that throws (Param's own division-by-zero) is the same refusal.
double probe(const Param& p, const std::unordered_map<std::string, double>& at) {
    double v;
    try {
        v = p.evaluate(at);
    } catch (const std::runtime_error& e) {
        throw capability_error(
            "parameter '" + printable(p.to_string()) + "' cannot be evaluated at a probe "
            "point (" + e.what() + "): gradient rules are exact only for affine gate angles "
            "(coeff*symbol + offset); use method='finite-diff'.");
    }
    if (!std::isfinite(v)) {
        throw capability_error(
            "parameter '" + printable(p.to_string()) + "' evaluates to a non-finite value "
            "at a probe point: gradient rules are exact only for affine gate angles "
            "(coeff*symbol + offset); use method='finite-diff'.");
    }
    return v;
}
}  // namespace

std::unordered_map<std::string, double> affine_coefficients(const Param& p) {
    std::unordered_map<std::string, double> out;
    if (p.is_concrete()) return out;
    const auto syms = p.free_symbols();
    std::unordered_map<std::string, double> at;
    for (const auto& s : syms) at[s] = 0.0;
    const double v0 = probe(p, at);
    double scale = std::max(1.0, std::abs(v0));
    double sum = 0.0;
    for (const auto& s : syms) {
        at[s] = 1.0;
        const double v1 = probe(p, at);
        at[s] = 2.0;
        const double v2 = probe(p, at);
        at[s] = 0.0;
        const double c = v1 - v0;
        scale = std::max({scale, std::abs(v1), std::abs(v2)});
        if (std::abs((v2 - v0) - 2.0 * c) > 1e-9 * scale) {
            throw capability_error(
                "parameter '" + printable(p.to_string()) + "' enters non-linearly in symbol '"
                + printable(s) + "': gradient rules are exact only for affine gate angles "
                  "(coeff*symbol + offset); use method='finite-diff'.");
        }
        out[s] = c;
        sum += c;
    }
    // Mixed probe: an affine function of several symbols is additive.
    for (const auto& s : syms) at[s] = 1.0;
    const double v_all = probe(p, at);
    if (std::abs((v_all - v0) - sum) > 1e-9 * std::max(scale, std::abs(v_all))) {
        throw capability_error(
            "parameter '" + printable(p.to_string()) + "' is not affine in its symbols: gradient "
            "rules are exact only for affine gate angles; use method='finite-diff'.");
    }
    return out;
}

}  // namespace qarpx
