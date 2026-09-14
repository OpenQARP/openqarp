#pragma once

#include "absorber.h"
#include "../core/command.h"

#include <cstdint>
#include <string>
#include <vector>

namespace qarpx {

/// Parses an OpenQASM 2.0 program string into a flat command sequence.
///
/// Deliberately accepts more than `QASM2Emitter` writes (§12.2): most of the
/// OpenQASM 2 in the world was produced by something else.
///   - ``OPENQASM 2.0;`` + ``include "…";`` header (ignored)
///   - Multiple ``qreg``/``creg`` registers, concatenated in declaration order
///     into qarp's single flat qubit / cbit index space
///   - The 23-gate spec ``qelib1.inc``, the later qiskit additions
///     (``swap cswap crx cry rxx rzz p cp u0 sx sxdg``), and the twelve
///     definitions `QASM2Emitter` writes (``ecr iswap iswapdg ryy cu`` …)
///   - ``measure q[i] -> c[j];`` and the whole-register ``measure q -> c;``
///   - ``reset``, ``barrier`` (both indexed and whole-register)
///   - ``if (<creg> == <int>) <qop>;`` — one guarded operation per `if`
///   - ``gate`` definitions: the body is skipped and the name resolved from
///     the table above, so the emitter's own prelude round-trips
///
/// Does NOT support (each raises `std::runtime_error` naming the construct):
///   - ``opaque`` declarations
///   - OpenQASM 3 syntax reaching this parser: ``input``, ``gphase``,
///     ``ctrl @``, ``inv @``
///   - ``gate`` definitions whose name is not in the table
///   - For/while loops (not OpenQASM 2 anyway)
///
/// `free_symbols` is always empty: OpenQASM 2 has no symbolic parameter.
///
/// Round-trip invariant (for programs `QASM2Emitter` produced):
///   @code
///   auto r = QASM2Absorber().absorb(QASM2Emitter().emit(cmds, n));
///   assert(QASM2Emitter().emit(r.commands, r.n_qubits) == original);
///   @endcode
class QASM2Absorber : public Absorber {
public:
    struct Result {
        std::vector<Command>     commands;
        uint32_t                 n_qubits    = 0;
        uint32_t                 n_cbits     = 0;
        std::vector<std::string> free_symbols;  ///< always empty (§12.2)
    };

    /// Parse *qasm2_source*.
    ///
    /// @throws std::runtime_error on lexer or parse errors, or on
    ///         constructs outside the supported subset.
    [[nodiscard]] Result absorb(const std::string& qasm2_source) const;

    [[nodiscard]] std::string source_name() const override { return "qasm2"; }
};

}  // namespace qarpx
