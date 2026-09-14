#pragma once

#include <cstdint>
#include <string_view>

namespace qarpx {

enum class GateType : uint16_t {
    // Single-qubit, no params
    X, Y, Z, H, S, Sdg, T, Tdg,
    // sqrt-X pair (hardware basis gate; OpenQASM 3 `sx`) and the explicit
    // identity (OpenQASM `id`) — §2.6.
    SX, SXdg, Id,

    // Single-qubit, 1 param
    Rx, Ry, Rz, P,

    // Single-qubit, 3 params (OpenQASM 3 general single-qubit gate)
    U,

    // Two-qubit, no params
    CX, CY, CZ, SWAP, ECR, iSWAP, iSWAPdg,
    // Controlled Clifford singles — §3.1.
    CH, CS, CSdg, CSX, CSXdg,

    // Two-qubit, 1 param
    CRx, CRy, CRz, CP, RZZ, RXX, RYY,

    // Two-qubit, 4 params (OpenQASM 3 controlled-U with global-phase γ)
    CU,

    // Three-qubit, no params
    CCX, CSWAP,

    // Multi-qubit, no params
    MCZ,

    // Zero-qubit, 1 param
    GPhase,

    // Special
    Barrier, Measure, Reset,

    // Classical control flow markers (meta).  Emitted by ConditionalBlock to
    // delimit `if (cbits == values) { then } else { else } end` regions in
    // the flat command list.  Dispatched at the simulator's run() loop level,
    // not per-command.  BranchBegin carries the AND-of-condition tuple in its
    // condition_bits / condition_values fields; BranchElse and BranchEnd are
    // pure markers.
    BranchBegin, BranchElse, BranchEnd,

    // Extensibility
    Custom,

    // Sentinel
    NUM_GATE_TYPES
};

/// Human-readable gate name.
constexpr std::string_view gate_name(GateType g) {
    switch (g) {
        case GateType::X:       return "X";
        case GateType::Y:       return "Y";
        case GateType::Z:       return "Z";
        case GateType::H:       return "H";
        case GateType::S:       return "S";
        case GateType::Sdg:     return "Sdg";
        case GateType::T:       return "T";
        case GateType::Tdg:     return "Tdg";
        case GateType::SX:      return "SX";
        case GateType::SXdg:    return "SXdg";
        case GateType::Id:      return "Id";
        case GateType::Rx:      return "Rx";
        case GateType::Ry:      return "Ry";
        case GateType::Rz:      return "Rz";
        case GateType::P:       return "P";
        case GateType::U:       return "U";
        case GateType::CX:      return "CX";
        case GateType::CY:      return "CY";
        case GateType::CZ:      return "CZ";
        case GateType::SWAP:    return "SWAP";
        case GateType::ECR:     return "ECR";
        case GateType::iSWAP:   return "iSWAP";
        case GateType::iSWAPdg: return "iSWAPdg";
        case GateType::CH:      return "CH";
        case GateType::CS:      return "CS";
        case GateType::CSdg:    return "CSdg";
        case GateType::CSX:     return "CSX";
        case GateType::CSXdg:   return "CSXdg";
        case GateType::CRx:     return "CRx";
        case GateType::CRy:     return "CRy";
        case GateType::CRz:     return "CRz";
        case GateType::CP:      return "CP";
        case GateType::RZZ:     return "RZZ";
        case GateType::RXX:     return "RXX";
        case GateType::RYY:     return "RYY";
        case GateType::CU:      return "CU";
        case GateType::CCX:     return "CCX";
        case GateType::CSWAP:   return "CSWAP";
        case GateType::MCZ:     return "MCZ";
        case GateType::GPhase:  return "GPhase";
        case GateType::Barrier: return "Barrier";
        case GateType::Measure: return "Measure";
        case GateType::Reset:        return "Reset";
        case GateType::BranchBegin:  return "BranchBegin";
        case GateType::BranchElse:   return "BranchElse";
        case GateType::BranchEnd:    return "BranchEnd";
        case GateType::Custom:       return "Custom";
        default:                return "Unknown";
    }
}

/// Number of parameters expected by this gate type.
constexpr uint8_t gate_num_params(GateType g) {
    switch (g) {
        case GateType::GPhase:
        case GateType::Rx: case GateType::Ry: case GateType::Rz: case GateType::P:
        case GateType::CRx: case GateType::CRy: case GateType::CRz: case GateType::CP:
        case GateType::RZZ: case GateType::RXX: case GateType::RYY:
            return 1;
        case GateType::U:
            return 3;
        case GateType::CU:
            return 4;
        default:
            return 0;
    }
}

/// Minimum number of qubits for this gate type.
constexpr uint8_t gate_num_qubits(GateType g) {
    switch (g) {
        case GateType::CX: case GateType::CY: case GateType::CZ: case GateType::SWAP:
        case GateType::ECR: case GateType::iSWAP: case GateType::iSWAPdg:
        case GateType::CH: case GateType::CS: case GateType::CSdg:
        case GateType::CSX: case GateType::CSXdg:
        case GateType::CRx: case GateType::CRy: case GateType::CRz: case GateType::CP:
        case GateType::RZZ: case GateType::RXX: case GateType::RYY:
        case GateType::CU:
            return 2;
        case GateType::CCX: case GateType::CSWAP:
            return 3;
        case GateType::MCZ:
            return 2; // minimum 2 qubits (1 control + target)
        case GateType::GPhase:
        case GateType::BranchBegin:
        case GateType::BranchElse:
        case GateType::BranchEnd:
            return 0;
        default:
            return 1;
    }
}

/// True if this GateType applies a physical operation to the register — the
/// classification used by resource-counting queries (`n_nqb_gates`,
/// core/command.h). False for the non-unitary / zero-qubit specials
/// (`Barrier`, `Measure`, `Reset`, `GPhase`, §8) and the classical-control
/// markers (`BranchBegin`/`BranchElse`/`BranchEnd`, §16) — none of these
/// represent a gate applied to the qubits.
constexpr bool gate_is_physical(GateType g) {
    switch (g) {
        case GateType::Barrier:
        case GateType::Measure:
        case GateType::Reset:
        case GateType::GPhase:
        case GateType::BranchBegin:
        case GateType::BranchElse:
        case GateType::BranchEnd:
            return false;
        default:
            return true;
    }
}

/// True if the gate is self-adjoint (its own inverse).
constexpr bool gate_is_self_adjoint(GateType g) {
    switch (g) {
        case GateType::X: case GateType::Y: case GateType::Z: case GateType::H:
        case GateType::Id:
        case GateType::CX: case GateType::CY: case GateType::CZ: case GateType::SWAP:
        case GateType::CH:
        case GateType::CCX: case GateType::CSWAP:
        case GateType::MCZ:
        case GateType::ECR:                 // exactly self-adjoint: ECR^2 = I (§3.2, §9)
        case GateType::Barrier: case GateType::Reset:
            return true;
        default:
            return false;
    }
}

/// True when a two-qubit gate's unitary is unchanged by swapping its two
/// qubit arguments: `CZ` (§3.1), `SWAP` / `iSWAP` / `iSWAPdg` (§3.2) and the
/// symmetric rotations (§3.3).  `ECR` is deliberately absent — §3.2 states it
/// carries an implicit (control, target) structure and is *not*
/// argument-symmetric, even though it is self-adjoint.
///
/// Meaningful only for two-qubit gates.  `CCX`, `CSWAP` and `MCZ` have partial
/// symmetries (among controls, or among the swapped pair) that a single bool
/// cannot express, so they report false.
constexpr bool gate_is_qubit_symmetric(GateType g) {
    switch (g) {
        case GateType::CZ:
        case GateType::SWAP: case GateType::iSWAP: case GateType::iSWAPdg:
        case GateType::RXX:  case GateType::RYY:   case GateType::RZZ:
            return true;
        default:
            return false;
    }
}

/// True for the one-parameter gates whose family is closed under angle
/// addition, G(a)·G(b) = G(a+b) — exactly the row §9 daggers by negating the
/// single parameter: Rx, Ry, Rz, P, CRx, CRy, CRz, CP, RXX, RYY, RZZ, GPhase.
/// The optimizer folds adjacent same-gate pairs of this family into one gate;
/// U and CU are excluded because their dagger swaps (phi, lambda) and
/// U(a)·U(b) is not a U of summed parameters.
constexpr bool gate_is_additive_in_param(GateType g) {
    switch (g) {
        case GateType::Rx:  case GateType::Ry:  case GateType::Rz:
        case GateType::P:
        case GateType::CRx: case GateType::CRy: case GateType::CRz:
        case GateType::CP:
        case GateType::RXX: case GateType::RYY: case GateType::RZZ:
        case GateType::GPhase:
            return true;
        default:
            return false;
    }
}

/// The adjoint GateType of a named-inverse pair (S <-> Sdg, T <-> Tdg,
/// SX <-> SXdg, iSWAP <-> iSWAPdg, CS <-> CSdg, CSX <-> CSXdg).  Returns the
/// gate itself when it is self-adjoint or when its adjoint is a parameter
/// change rather than a type change (handled in Command::dagger()).
constexpr GateType gate_adjoint(GateType g) {
    switch (g) {
        case GateType::S:       return GateType::Sdg;
        case GateType::Sdg:     return GateType::S;
        case GateType::T:       return GateType::Tdg;
        case GateType::Tdg:     return GateType::T;
        case GateType::iSWAP:   return GateType::iSWAPdg;
        case GateType::iSWAPdg: return GateType::iSWAP;
        case GateType::SX:      return GateType::SXdg;
        case GateType::SXdg:    return GateType::SX;
        case GateType::CS:      return GateType::CSdg;
        case GateType::CSdg:    return GateType::CS;
        case GateType::CSX:     return GateType::CSXdg;
        case GateType::CSXdg:   return GateType::CSX;
        default:
            if (gate_is_self_adjoint(g)) return g;
            return g; // parametric gates: adjoint handled by Command::dagger()
    }
}

}  // namespace qarpx
