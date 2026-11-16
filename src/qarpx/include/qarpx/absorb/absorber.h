#pragma once

#include <string>

namespace qarpx {

/// Abstract base for backend absorbers.
///
/// An absorber parses an external representation (a QASM3 string,
/// a serialised QIR module, etc.) and produces a flat qarpx command
/// sequence that the rest of the library can simulate, transpile, or
/// re-emit.
class Absorber {
public:
    virtual ~Absorber() = default;

    /// Human-readable name of the source format (e.g. "qasm3").
    [[nodiscard]] virtual std::string source_name() const = 0;
};

}  // namespace qarpx
