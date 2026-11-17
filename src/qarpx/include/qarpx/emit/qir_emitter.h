#pragma once

#include "emitter.h"

#include <cstdint>
#include <string>
#include <vector>

namespace qarpx {

/// Emits QIR-compatible LLVM IR text (.ll format).
///
/// QIR is the quantum intermediate representation standard
/// (https://github.com/qir-alliance/qir-spec).  We emit text LLVM IR
/// conforming to the QIR Base Profile — no LLVM library dependency.
///
/// The output can be consumed by:
///   - cudaq (cudaq-translate --import-qir)
///   - Azure Quantum
///   - pyqir
///   - llc (LLVM compiler)
class QIREmitter : public Emitter {
public:
    [[nodiscard]] GateSet gate_set() const override {
        return GateSet{
            "qir",
            {GateType::H, GateType::X, GateType::Y, GateType::Z,
             GateType::S, GateType::Sdg, GateType::T, GateType::Tdg,
             GateType::Rx, GateType::Ry, GateType::Rz,
             GateType::CX, GateType::CZ, GateType::SWAP,
             GateType::Measure, GateType::Reset, GateType::Barrier}};
    }

    [[nodiscard]] EmitterCapabilities capabilities() const override {
        return EmitterCapabilities{
            .symbolic_params     = false,  // Base Profile: concrete doubles only
            .multi_symbol_params = false,
            .distinct_cbits      = true,
            .conditionals        = true,   // basic-block br control flow
            .multibit_conditions = true,   // AND-of-i1 reads
            .multibit_else       = true,
            .custom_unitary      = false,
            .cu_global_phase     = false,
        };
    }

    /// Emit QIR as an LLVM IR string.
    [[nodiscard]] std::string emit(
        const std::vector<Command>& commands,
        uint32_t n_qubits,
        const std::string& function_name = "circuit") const;

    /// Emit QIR and write to a file.
    void emit_to_file(
        const std::vector<Command>& commands,
        uint32_t n_qubits,
        const std::string& path,
        const std::string& function_name = "circuit") const;

    [[nodiscard]] std::string target_name() const override { return "qir"; }
};

}  // namespace qarpx
