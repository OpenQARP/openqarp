// Internal helpers shared by the operator facade .cpp files (not installed).
#pragma once

#include <cctype>
#include <charconv>
#include <complex>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace qarpx::ops::detail {

inline std::string_view trim(std::string_view s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.remove_suffix(1);
    return s;
}

/// Shortest round-trip formatting of a double — matches how CPython prints
/// the float components inside complex repr (integral values print without
/// a trailing ".0": str(1+0j) == "(1+0j)").
inline std::string format_double_repr(double v) {
    char buf[32];
    const auto res = std::to_chars(buf, buf + sizeof(buf), v);
    return std::string(buf, res.ptr);
}

/// Python str(complex) style: "(a+bj)" / "(a-bj)", or "bj" when the real
/// part is a positive zero.
inline std::string format_python_complex(std::complex<double> c) {
    const std::string im = format_double_repr(c.imag());
    if (c.real() == 0.0 && !std::signbit(c.real())) return im + "j";
    const std::string re = format_double_repr(c.real());
    if (!im.empty() && im.front() == '-') return "(" + re + im + "j)";
    return "(" + re + "+" + im + "j)";
}

/// Parse a Python-style complex literal: "1.5", "-2e-3", "1.5j", "(1+2j)",
/// "j", "-j".  Used for the coefficient prefix of the bracketed term form
/// ("1.5 [2^ 3]").  Throws std::invalid_argument on anything else.
inline std::complex<double> parse_complex_literal(std::string_view sv) {
    const std::string t(trim(sv));
    std::string s = t;
    if (s.size() >= 2 && s.front() == '(' && s.back() == ')')
        s = s.substr(1, s.size() - 2);
    if (s.empty()) throw std::invalid_argument("Invalid coefficient '" + t + "'.");

    if (s == "j" || s == "+j") return {0.0, 1.0};
    if (s == "-j") return {0.0, -1.0};

    const char* begin = s.c_str();
    char* end = nullptr;
    const double first = std::strtod(begin, &end);
    if (end == begin) throw std::invalid_argument("Invalid coefficient '" + t + "'.");
    size_t pos = static_cast<size_t>(end - begin);

    if (pos == s.size()) return {first, 0.0};                       // "1.5"
    if (s[pos] == 'j' && pos + 1 == s.size()) return {0.0, first};  // "1.5j"

    // "a+bj" / "a-bj" (strtod consumes the sign), with "a+j" / "a-j" specials.
    if (s[pos] != '+' && s[pos] != '-')
        throw std::invalid_argument("Invalid coefficient '" + t + "'.");
    const std::string_view rest(s.data() + pos, s.size() - pos);
    if (rest == "+j") return {first, 1.0};
    if (rest == "-j") return {first, -1.0};

    const char* begin2 = s.c_str() + pos;
    char* end2 = nullptr;
    const double second = std::strtod(begin2, &end2);
    if (end2 == begin2) throw std::invalid_argument("Invalid coefficient '" + t + "'.");
    const size_t pos2 = pos + static_cast<size_t>(end2 - begin2);
    if (pos2 + 1 != s.size() || s[pos2] != 'j')
        throw std::invalid_argument("Invalid coefficient '" + t + "'.");
    return {first, second};
}

/// Split the bracketed term form: on input "1.5 [X0 Z1]" multiplies the
/// prefix into `coefficient` and returns the factor list "X0 Z1"; input
/// without '[' passes through untouched.
inline std::string_view split_bracket_form(std::string_view term,
                                           std::complex<double>& coefficient) {
    const size_t l = term.find('[');
    if (l == std::string_view::npos) return term;
    const size_t r = term.find(']', l);
    if (r == std::string_view::npos)
        throw std::invalid_argument("Unbalanced brackets in term string.");
    const std::string_view prefix = trim(term.substr(0, l));
    if (!prefix.empty()) coefficient *= parse_complex_literal(prefix);
    return term.substr(l + 1, r - l - 1);
}

}  // namespace qarpx::ops::detail
