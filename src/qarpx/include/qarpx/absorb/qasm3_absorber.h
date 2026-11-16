#pragma once

#include "absorber.h"
#include "../core/command.h"

#include <cstdint>
#include <string>
#include <vector>

namespace qarpx {

/// Parses an OpenQASM 3.0 program string into a flat command sequence.
///
/// Supports the subset of QASM 3 produced by QASM3Emitter:
///   - ``OPENQASM 3.0;`` + ``include "stdgates.inc";`` header (ignored)
///   - ``qubit[N] q;`` quantum register
///   - ``bit[M] c;``   classical register
///   - ``input float[64] <name>;``  symbolic parameter declarations
///   - All gates from stdgates.inc that map to GateType entries
///   - ``c[i] = measure q[j];`` measurement
///   - ``reset q[i];`` reset
///   - ``if (cond) { … } else { … }`` classical conditionals
///     (condition = single ``c[i] == 0|1`` or AND-chain thereof)
///   - ``ctrl(n) @ z q[…];`` for MCZ
///   - ``inv @ iswap q[a], q[b];`` for iSWAPdg
///   - ``gphase(θ);`` global phase
///   - ``barrier q[…];``
///
/// Does NOT support:
///   - Gate/subroutine definitions
///   - For/while loops
///   - Box / delay modifiers
///   - Custom gate calls (raises std::runtime_error)
///
/// Round-trip invariant (no symbolic params):
///   @code
///   auto r = QASM3Absorber().absorb(QASM3Emitter().emit(cmds, n));
///   assert(QASM3Emitter().emit(r.commands, r.n_qubits) == original);
///   @endcode
class QASM3Absorber : public Absorber {
public:
    struct Result {
        std::vector<Command>     commands;
        uint32_t                 n_qubits    = 0;
        uint32_t                 n_cbits     = 0;
        std::vector<std::string> free_symbols;  ///< declared ``input float[64]``
    };

    /// Parse *qasm3_source*.
    ///
    /// @throws std::runtime_error on lexer or parse errors, or on
    ///         constructs outside the supported subset.
    [[nodiscard]] Result absorb(const std::string& qasm3_source) const;

    [[nodiscard]] std::string source_name() const override { return "qasm3"; }
};

}  // namespace qarpx
