#pragma once

#include "../core/command.h"
#include "../core/errors.h"
#include "../core/gates.h"
#include "../transpiler/gateset.h"
#include "param_bridge.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace qarpx {

/// Non-gate-set limits of an emit backend.  The GateSet covers "which
/// gates"; this covers everything the GateSet cannot express.  Flags are
/// compile-time declarations — `Emitter::validate()` is implemented once in
/// terms of them, so a backend declares its limits and never re-implements
/// the checking.
struct EmitterCapabilities {
    bool symbolic_params     = false;  ///< symbolic Params survive emission
    bool multi_symbol_params = false;  ///< any expression shape; false = linear c*x + d in one symbol
    bool distinct_cbits      = false;  ///< Measure(q, c) with c != q is representable
    bool conditionals        = false;  ///< BranchBegin/Else/End are representable
    bool multibit_conditions = false;  ///< a branch condition may span >1 cbit
    bool multibit_else       = false;  ///< BranchElse on a multi-bit condition
    bool custom_unitary      = false;  ///< GateType::Custom is representable
    bool cu_global_phase     = false;  ///< CU(θ,φ,λ,γ) with γ != 0 is representable
};

/// Why a command cannot cross to this backend.
struct Incompatibility {
    Command     command;
    std::string reason;   ///< caller-facing; names the backend and the limit
};

/// Abstract base for backend emitters.
///
/// An emitter takes a flattened command sequence and produces a
/// backend-specific output: a qiskit.QuantumCircuit, a QASM 3 string, a
/// QIR module, etc.  Backends declare `gate_set()` (what they accept —
/// for SDK backends this can be broader than the transpiler rebase
/// target) and `capabilities()`; `validate()` is implemented once here in
/// terms of both and is called by every `emit()` before any SDK import.
class Emitter {
public:
    virtual ~Emitter() = default;

    /// Human-readable name of this emitter target (e.g., "qulacs", "qir").
    [[nodiscard]] virtual std::string target_name() const = 0;

    /// The set of GateTypes this emitter accepts.  Branch markers are not
    /// gates and are governed by `capabilities().conditionals` instead.
    [[nodiscard]] virtual GateSet gate_set() const = 0;

    /// The non-gate-set limits of this backend.
    [[nodiscard]] virtual EmitterCapabilities capabilities() const = 0;

    /// First command that cannot cross to this backend, or nullopt if the
    /// whole sequence can.  Purely declarative — no SDK import happens, so
    /// it answers on machines where the SDK is absent.  Well-formedness of
    /// branch-marker nesting is the flattener's contract, not a capability:
    /// a malformed marker sequence stays an emit-time internal error.
    [[nodiscard]] std::optional<Incompatibility> validate(
        const std::vector<Command>& commands) const {
        const GateSet gs = gate_set();
        const EmitterCapabilities caps = capabilities();
        const std::string tag = target_name() + " emitter: ";

        auto reject = [&](const Command& cmd, std::string why) {
            return Incompatibility{cmd, tag + std::move(why)};
        };

        // Condition width of each open branch frame, for the Else check.
        std::vector<size_t> frame_widths;

        for (const auto& cmd : commands) {
            switch (cmd.gate) {
                case GateType::BranchBegin:
                    if (!caps.conditionals)
                        return reject(cmd,
                            "classical conditionals (BranchBegin) are not supported");
                    if (cmd.condition_bits.size() > 1 && !caps.multibit_conditions)
                        return reject(cmd,
                            "multi-bit branch conditions are not supported");
                    frame_widths.push_back(cmd.condition_bits.size());
                    continue;
                case GateType::BranchElse:
                    if (!frame_widths.empty() && frame_widths.back() > 1 &&
                        !caps.multibit_else)
                        return reject(cmd,
                            "BranchElse on a multi-bit condition is not supported");
                    continue;
                case GateType::BranchEnd:
                    if (!frame_widths.empty()) frame_widths.pop_back();
                    continue;
                case GateType::Custom:
                    if (!gs.contains(GateType::Custom) && !caps.custom_unitary)
                        return reject(cmd,
                            "Custom unitaries are not representable");
                    break;
                default:
                    if (!gs.contains(cmd.gate))
                        return reject(cmd,
                            std::string("gate '") + std::string(gate_name(cmd.gate)) +
                            "' is not in the " + gs.name + " gate set");
                    break;
            }

            for (const auto& p : cmd.params) {
                if (!p.is_symbolic()) continue;
                if (!caps.symbolic_params)
                    return reject(cmd, "symbolic parameters are not supported");
                if (!caps.multi_symbol_params && !param_bridge::is_linear(p))
                    return reject(cmd,
                        "symbolic parameters must be linear in a single symbol"
                        " (c*x + d); '" + p.to_string() + "' is not");
            }

            if (cmd.gate == GateType::Measure && !caps.distinct_cbits &&
                !cmd.cbits.empty() && cmd.cbits[0] != cmd.qubits[0])
                return reject(cmd,
                    "measuring qubit " + std::to_string(cmd.qubits[0]) +
                    " into a different classical bit (" +
                    std::to_string(cmd.cbits[0]) +
                    ") is not representable — no classical register concept");

            if (cmd.gate == GateType::CU && !caps.cu_global_phase) {
                const Param& gamma = cmd.params[3];
                if (gamma.is_symbolic())
                    return reject(cmd,
                        "CU with symbolic global phase (gamma) is not representable");
                if (std::abs(gamma.value()) > 1e-12)
                    return reject(cmd,
                        "CU with non-zero global phase (gamma) is not representable");
            }
        }
        return std::nullopt;
    }

    /// `validate()` as a hard gate — every `emit()` calls this first.
    void throw_if_invalid(const std::vector<Command>& commands) const {
        if (auto inc = validate(commands)) {
            throw capability_error(inc->reason, inc->command);
        }
    }
};

}  // namespace qarpx
