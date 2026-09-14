#pragma once

/// One home for the emit/absorb parameter bridge.  QARPx Params cross the
/// SDK boundary as linear forms `coeff*sym + offset`; both directions
/// recover the pair by probing the expression at 0 and 1, then confirm
/// linearity at a third, non-rational point (a polynomial with rational
/// roots cannot fool it).  The single-symbol-linear restriction is
/// enforced here, once, with one message per direction — call sites never
/// re-implement the check.  `Emitter::validate()` uses `is_linear` so
/// `can_emit_to()` answers without the SDK.

#include "../core/errors.h"
#include "../core/param.h"

#include <cmath>
#include <cstddef>
#include <string>
#include <utility>

namespace qarpx::param_bridge {

struct LinearForm {
    std::string symbol;
    double      coeff;
    double      offset;
};

/// Third probe point: irrational, so no polynomial with rational roots
/// vanishes at 0, 1 and here simultaneously.
inline constexpr double kLinearityProbe = 0.61803398874989484820;

/// True when `v_probe` agrees with the line through (0, offset), (1, coeff+offset).
inline bool linear_at_probe(double coeff, double offset, double v_probe) {
    const double expect = coeff * kLinearityProbe + offset;
    const double tol = 1e-9 * (1.0 + std::abs(coeff) + std::abs(offset));
    return std::abs(v_probe - expect) <= tol;
}

/// Emit-side declarative check: `c*x + d` in exactly one symbol.  Concrete
/// Params are trivially linear.
inline bool is_linear(const Param& p) {
    if (!p.is_symbolic()) return true;
    auto syms = p.free_symbols();
    if (syms.size() != 1) return false;
    const std::string name = *syms.begin();
    const double offset = p.evaluate({{name, 0.0}});
    const double coeff  = p.evaluate({{name, 1.0}}) - offset;
    return linear_at_probe(coeff, offset, p.evaluate({{name, kLinearityProbe}}));
}

/// Emit side: decompose a symbolic Param into its linear form.  The
/// multi-symbol and non-linear cases are rejected by `Emitter::validate()`
/// before emission, so these throws are the defensive backstop.
inline LinearForm linear_form(const Param& p, const std::string& target) {
    auto syms = p.free_symbols();
    if (syms.size() != 1) {
        throw capability_error(
            target + " emitter: symbolic parameters must be linear in a "
                     "single symbol");
    }
    const std::string name = *syms.begin();
    const double offset = p.evaluate({{name, 0.0}});
    const double coeff  = p.evaluate({{name, 1.0}}) - offset;
    if (!linear_at_probe(coeff, offset, p.evaluate({{name, kLinearityProbe}}))) {
        throw capability_error(
            target + " emitter: symbolic parameters must be linear in a "
                     "single symbol; '" + p.to_string() + "' is not");
    }
    return {name, coeff, offset};
}

/// Absorb side: the same restriction on the incoming SDK expression.
inline void require_single_symbol(std::size_t n_free, const std::string& source) {
    if (n_free != 1) {
        throw capability_error(
            source + " absorber: multi-symbol parameter expressions are not "
                     "supported; substitute parameters before absorbing");
    }
}

/// Absorb side: rebuild a Param from an SDK expression probed at 0 and 1,
/// confirmed linear at the third point.  `eval_at(x)` substitutes the
/// expression's single free symbol with x and returns the concrete value;
/// `scale` converts SDK units to radians; `source` names the absorber in
/// the rejection.
template <typename Probe>
Param absorb_linear(const std::string& symbol, Probe&& eval_at,
                    double scale, const std::string& source) {
    const double v0 = eval_at(0.0);
    const double v1 = eval_at(1.0);
    if (!linear_at_probe(v1 - v0, v0, eval_at(kLinearityProbe))) {
        throw capability_error(
            source + " absorber: parameter expression in '" + symbol +
            "' is not linear (c*" + symbol + " + d); substitute it before absorbing");
    }
    return Param::linear((v1 - v0) * scale, symbol, v0 * scale);
}

}  // namespace qarpx::param_bridge
