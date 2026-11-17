// Symbolic coefficient string boundary (sympy ↔ SymEngine): compiled to an
// empty TU unless QARP_WITH_SYMENGINE is enabled.

#ifdef QARP_WITH_SYMENGINE

#include "qarpx/operators/coeff.h"

#include <symengine/parser.h>
#include <symengine/printers.h>

namespace qarpx::ops {

SymCoeff parse_symbolic(const std::string& text) {
    return SymCoeff(SymEngine::Expression(SymEngine::parse(text)));
}

std::string symbolic_to_string(const SymCoeff& c) {
    return SymEngine::str(*c.expr.get_basic());
}

}  // namespace qarpx::ops

#endif  // QARP_WITH_SYMENGINE
