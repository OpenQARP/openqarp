#include "qarpx/dag/commutation.h"

namespace qarpx::dag_passes {

WireBasis qubit_basis(const Command& cmd, std::size_t k) {
    switch (cmd.gate) {
        // Z-diagonal on their only qubit.
        case GateType::Z: case GateType::S: case GateType::Sdg:
        case GateType::T: case GateType::Tdg:
        case GateType::Rz: case GateType::P:
            return WireBasis::kZ;

        case GateType::X: case GateType::Rx:
        case GateType::SX: case GateType::SXdg:
            return WireBasis::kX;

        case GateType::Y: case GateType::Ry:
            return WireBasis::kY;

        // Fully diagonal two-qubit gates: Z on both wires.
        case GateType::CZ: case GateType::CP: case GateType::RZZ:
        case GateType::CS: case GateType::CSdg:
            return WireBasis::kZ;

        // Multi-controlled Z: diagonal on every wire.
        case GateType::MCZ:
            return WireBasis::kZ;

        // Controlled gates: control wire is always Z (block-diagonal in the
        // control's computational basis); target inherits the rotation axis.
        case GateType::CX:  return k == 0 ? WireBasis::kZ : WireBasis::kX;
        case GateType::CY:  return k == 0 ? WireBasis::kZ : WireBasis::kY;
        case GateType::CRx: return k == 0 ? WireBasis::kZ : WireBasis::kX;
        case GateType::CRy: return k == 0 ? WireBasis::kZ : WireBasis::kY;
        case GateType::CRz: return k == 0 ? WireBasis::kZ : WireBasis::kZ;
        case GateType::CSX: case GateType::CSXdg:
                            return k == 0 ? WireBasis::kZ : WireBasis::kX;
        case GateType::CH:  return k == 0 ? WireBasis::kZ : WireBasis::kNone;

        case GateType::RXX: return WireBasis::kX;
        case GateType::RYY: return WireBasis::kY;

        // CCX: two Z controls + X target.  CSWAP / CU: Z control only.
        case GateType::CCX:   return k <= 1 ? WireBasis::kZ : WireBasis::kX;
        case GateType::CSWAP: return k == 0 ? WireBasis::kZ : WireBasis::kNone;
        case GateType::CU:    return k == 0 ? WireBasis::kZ : WireBasis::kNone;

        // No certified single-wire basis (H, U, permutation-like 2q gates,
        // Custom payloads, meta ops).
        default:
            return WireBasis::kNone;
    }
}

bool commute(const Command& a, const Command& b) {
    auto is_opaque = [](const Command& c) {
        switch (c.gate) {
            case GateType::Barrier: case GateType::Measure: case GateType::Reset:
            case GateType::BranchBegin: case GateType::BranchElse:
            case GateType::BranchEnd: case GateType::Custom:
                return true;
            default:
                return false;
        }
    };
    if (is_opaque(a) || is_opaque(b)) return false;

    // Conservative total order on cbit wires: any shared classical bit —
    // read or write, either side — blocks commutation.
    auto touches_cbit = [](const Command& c, uint32_t bit) {
        for (auto x : c.cbits)          if (x == bit) return true;
        for (auto x : c.condition_bits) if (x == bit) return true;
        return false;
    };
    for (auto bit : a.cbits)          if (touches_cbit(b, bit)) return false;
    for (auto bit : a.condition_bits) if (touches_cbit(b, bit)) return false;

    // Shared qubits need matching, certified bases (see header: sufficient
    // condition — simultaneous block-diagonality in the shared product basis).
    for (std::size_t i = 0; i < a.qubits.size(); ++i) {
        for (std::size_t j = 0; j < b.qubits.size(); ++j) {
            if (a.qubits[i] != b.qubits[j]) continue;
            const WireBasis ba = qubit_basis(a, i);
            if (ba == WireBasis::kNone || ba != qubit_basis(b, j)) return false;
        }
    }
    return true;
}

}  // namespace qarpx::dag_passes
