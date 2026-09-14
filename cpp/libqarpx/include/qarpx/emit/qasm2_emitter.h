#pragma once

#include "emitter.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace qarpx {

/// Emits an OpenQASM 2.0 program string from a flattened command sequence.
///
/// Conventions match `qarp_conventions.md` §12.2:
///   - Qubit 0 is the LSB (matches OpenQASM 2 register layout).
///   - Rotations are exp(-iθ/2 P).
///   - The spec `qelib1.inc` is 23 gates; everything outside it that this
///     emitter uses carries its own `gate` definition in the prelude, each
///     body a §11 phase-exact identity.
///
/// Three constructs qarp's IR carries have no OpenQASM 2 representation and
/// are rejected by `validate()` before emission:
///   - symbolic `Param` — the language has no `input` declaration
///   - `GPhase`         — no global-phase statement.  Deliberately rejected
///                        rather than dropped (§11's per-target phase-loss
///                        boundary is for simulators; a `.qasm` file is
///                        portable and may be controlled downstream).
///   - `MCZ`            — no `ctrl(n) @` modifier, and a `gate` body cannot
///                        be arity-generic.  `decompose_mcz` lowers it first.
///
/// Classical registers are content-dependent: a single `creg c[K];` when the
/// program has no conditionals, one `creg c<i>[1];` per bit when it does —
/// OpenQASM 2's `if` compares a whole register to an integer, so per-bit
/// registers are the only way to condition on a single bit.
class QASM2Emitter : public Emitter {
public:
    [[nodiscard]] GateSet gate_set() const override {
        GateSet gs = universal_gateset();
        gs.name = "qasm2";
        gs.allowed.erase(GateType::Custom);
        gs.allowed.erase(GateType::GPhase);
        gs.allowed.erase(GateType::MCZ);
        gs.allowed.erase(GateType::BranchBegin);
        gs.allowed.erase(GateType::BranchElse);
        gs.allowed.erase(GateType::BranchEnd);
        return gs;
    }

    [[nodiscard]] EmitterCapabilities capabilities() const override {
        return EmitterCapabilities{
            .symbolic_params     = false,  // no `input` declarations
            .multi_symbol_params = false,  // moot once symbolic_params is false
            .distinct_cbits      = true,   // measure q[i] -> c[j];
            .conditionals        = true,
            .multibit_conditions = false,  // `if` compares a whole creg
            .multibit_else       = true,   // by value negation, single-bit only
            .custom_unitary      = false,
            .cu_global_phase     = true,   // via the emitted `cu` definition
        };
    }

    /// `Emitter::validate` plus the two branch shapes OpenQASM 2's *grammar*
    /// forbids, which no `EmitterCapabilities` flag can express: `if` governs
    /// a single `qop`, so neither a nested `if` nor a guarded `barrier`
    /// (`barrier` is not a `qop`) is spellable.
    ///
    /// Deliberately *shadows* the non-virtual base member rather than adding a
    /// capability flag — a flag would mean editing the shared base and every
    /// other backend for a limit only this one has.  Safe because emitters are
    /// always held as their concrete type; the payoff is that
    /// `Block.can_emit_to("qasm2")` gives the same verdict `emit()` does.
    [[nodiscard]] std::optional<Incompatibility> validate(
        const std::vector<Command>& commands) const;

    /// Emit OpenQASM 2.0 as a string.
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

    [[nodiscard]] std::string target_name() const override { return "qasm2"; }
};

}  // namespace qarpx
