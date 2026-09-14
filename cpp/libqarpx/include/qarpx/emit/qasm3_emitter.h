#pragma once

#include "emitter.h"

#include <cstdint>
#include <string>
#include <vector>

namespace qarpx {

/// Emits an OpenQASM 3.0 program string from a flattened command sequence.
///
/// Conventions match `qarp_conventions.md`:
///   - Qubit 0 is the LSB (matches OpenQASM 3 register layout).
///   - Rotations are exp(-iθ/2 P).
///   - `Custom` gates have no QASM 3 base-profile representation and are
///     rejected by `validate()` — fuse them away or transpile them out first.
///
/// Symbolic params declared via `Param::symbol(name)` become
/// `input float[64] <name>;` declarations at the top of the program.
/// Linear-form params (`coeff*sym + offset`) emit inline arithmetic in the
/// gate-arg expression.
class QASM3Emitter : public Emitter {
public:
    [[nodiscard]] GateSet gate_set() const override {
        GateSet gs = universal_gateset();
        gs.name = "qasm3";
        gs.allowed.erase(GateType::Custom);
        gs.allowed.erase(GateType::BranchBegin);
        gs.allowed.erase(GateType::BranchElse);
        gs.allowed.erase(GateType::BranchEnd);
        return gs;
    }

    [[nodiscard]] EmitterCapabilities capabilities() const override {
        return EmitterCapabilities{
            .symbolic_params     = true,   // input float[64] declarations
            .multi_symbol_params = true,   // Param::to_string renders the expr
            .distinct_cbits      = true,
            .conditionals        = true,
            .multibit_conditions = true,   // rendered as &&-chains
            .multibit_else       = true,
            .custom_unitary      = false,
            .cu_global_phase     = true,   // stdgates `cu` carries γ
        };
    }

    /// Emit OpenQASM 3.0 as a string.
    [[nodiscard]] std::string emit(
        const std::vector<Command>& commands,
        uint32_t n_qubits,
        const std::string& circuit_name = "circuit") const;

    /// Emit and write to a file.  `circuit_name` is recorded in a header comment.
    void emit_to_file(
        const std::vector<Command>& commands,
        uint32_t n_qubits,
        const std::string& path,
        const std::string& circuit_name = "circuit") const;

    [[nodiscard]] std::string target_name() const override { return "qasm3"; }
};

}  // namespace qarpx
