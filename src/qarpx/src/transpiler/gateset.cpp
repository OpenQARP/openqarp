#include "qarpx/transpiler/gateset.h"

#include <initializer_list>
#include <utility>

namespace qarpx {

GateSet qulacs_gateset() {
    return GateSet{
        .name = "qulacs",
        .allowed = {
            GateType::X, GateType::Y, GateType::Z,
            GateType::H,
            GateType::S, GateType::Sdg,
            GateType::T, GateType::Tdg,
            GateType::Rx, GateType::Ry, GateType::Rz,
            GateType::CX, GateType::CZ, GateType::SWAP,
            GateType::Measure, GateType::Barrier, GateType::GPhase,
        }
    };
}

GateSet qulacs_emitter_gateset() {
    // QulacsEmitter's accept set — the transpile target plus every gate the
    // emitter lifts via add_dense_matrix_gate.  Not a rebase target.
    GateSet gs = qulacs_gateset();
    gs.name = "qulacs_emitter";
    gs.allowed.insert({
        GateType::P, GateType::U,
        GateType::CY, GateType::ECR, GateType::iSWAP, GateType::iSWAPdg,
        GateType::CRx, GateType::CRy, GateType::CRz, GateType::CP,
        GateType::RZZ, GateType::RXX, GateType::RYY,
        GateType::CU,
        GateType::CCX, GateType::CSWAP,
        GateType::SX, GateType::SXdg, GateType::Id,
        GateType::CH, GateType::CS, GateType::CSdg, GateType::CSX, GateType::CSXdg,
    });
    return gs;
}

namespace {
/// Shared body for the SDK accept sets that span the whole enum: every
/// concrete gate plus the meta entries, minus branch markers (governed by
/// EmitterCapabilities::conditionals, not set membership) and any listed
/// exclusion.
GateSet full_enum_gateset(std::string name,
                          std::initializer_list<GateType> excluded) {
    GateSet gs;
    gs.name = std::move(name);
    for (uint16_t i = 0; i < static_cast<uint16_t>(GateType::NUM_GATE_TYPES); ++i) {
        gs.allowed.insert(static_cast<GateType>(i));
    }
    gs.allowed.erase(GateType::BranchBegin);
    gs.allowed.erase(GateType::BranchElse);
    gs.allowed.erase(GateType::BranchEnd);
    for (GateType g : excluded) gs.allowed.erase(g);
    return gs;
}
}  // namespace

GateSet qiskit_gateset() {
    return full_enum_gateset("qiskit", {GateType::Custom});
}

GateSet pytket_gateset() {
    return full_enum_gateset("pytket", {GateType::Custom});
}

GateSet pennylane_gateset() {
    return full_enum_gateset("pennylane", {GateType::Custom, GateType::Reset});
}

GateSet native_gateset() {
    // Superset of qulacs_gateset() — adds every gate that QarpSimulator
    // dispatches with a single csim kernel call.  Used as QarpEngine's
    // canonical target: anything in here passes through transpilation
    // unchanged and is executed natively by csim.
    GateSet gs = qulacs_gateset();
    gs.name = "native";
    gs.allowed.insert({
        // 1Q dense-matrix dispatch
        GateType::P, GateType::U,
        // 2Q controlled-1Q dispatch
        GateType::CY,
        GateType::CRx, GateType::CRy, GateType::CRz, GateType::CP,
        GateType::CU,
        // 2Q multi-Pauli rotation dispatch
        GateType::RZZ, GateType::RXX, GateType::RYY,
        // n-controlled dispatch
        GateType::CCX, GateType::MCZ,
        // 1Q constant-matrix dispatch (csim sqrtX kernels) and no-op Id
        GateType::SX, GateType::SXdg, GateType::Id,
        // 2Q controlled-1Q dense dispatch, as CY
        GateType::CH, GateType::CS, GateType::CSdg, GateType::CSX, GateType::CSXdg,
        // 1Q dense-matrix dispatch of the O1 fusion product.  Without it
        // `optimize_in_target` would (rightly) skip fusion on the default
        // engine target.
        GateType::Custom,
    });
    return gs;
}

GateSet cudaq_gateset() {
    return GateSet{
        .name = "cudaq",
        .allowed = {
            GateType::X, GateType::Y, GateType::Z,
            GateType::H,
            GateType::S, GateType::T,
            GateType::Rx, GateType::Ry, GateType::Rz,
            GateType::P,
            GateType::CX, GateType::CZ, GateType::SWAP,
            GateType::Measure, GateType::Barrier,
        }
    };
}

GateSet clifford_t_gateset() {
    return GateSet{
        .name = "clifford_t",
        .allowed = {
            GateType::H, GateType::S, GateType::T,
            GateType::CX,
            GateType::Measure, GateType::Barrier,
        }
    };
}

GateSet multi_control_basis_gateset() {
    // Target gate set for inner-block lowering at multi-control wrap time.
    // ``make_multi_controlled`` in ``controlled_block.cpp`` handles every gate
    // in this set directly.  Anything outside it (H, S, T, Sdg, Tdg, U, CX,
    // SWAP, CSWAP, CCX, CU, controlled rotations, RXX/RYY/RZZ, iSWAP/ECR, …)
    // is lowered via the standard decomposition table — phase-preserving for
    // the gates where the built-in lowering drops global phase
    // (see ``controlled_block.cpp::mc_basis_decompositions``).
    return GateSet{
        .name = "multi_control_basis",
        .allowed = {
            GateType::X, GateType::Y, GateType::Z,
            GateType::Rx, GateType::Ry, GateType::Rz, GateType::P,
            GateType::CX,
            // MCZ is basis-native here: controlling it only widens its qubit
            // tuple, so lowering it first would replace one gate with a
            // Barenco cascade for nothing.
            GateType::MCZ,
            GateType::GPhase, GateType::Barrier,
        }
    };
}

GateSet clifford_t_rz_gateset() {
    return GateSet{
        .name = "clifford_t_rz",
        .allowed = {
            // Clifford
            GateType::X, GateType::Y, GateType::Z,
            GateType::H,
            GateType::S, GateType::Sdg,
            GateType::CX, GateType::CZ,
            // T (magic-state cultivation)
            GateType::T, GateType::Tdg,
            // Analog rotation — the set's only non-Clifford+T rotation
            GateType::Rz,
            // Meta
            GateType::Measure, GateType::Barrier, GateType::GPhase,
        },
        .rules = "clifford_t_rz",
    };
}

GateSet universal_gateset() {
    GateSet gs;
    gs.name = "universal";
    for (uint16_t i = 0; i < static_cast<uint16_t>(GateType::NUM_GATE_TYPES); ++i) {
        gs.allowed.insert(static_cast<GateType>(i));
    }
    return gs;
}

GateSet full_gateset_1q() {
    return GateSet{
        .name = "full_1q",
        .allowed = {
            GateType::X, GateType::Y, GateType::Z,
            GateType::H,
            GateType::S, GateType::Sdg,
            GateType::T, GateType::Tdg,
            GateType::SX, GateType::SXdg, GateType::Id,
            GateType::Rx, GateType::Ry, GateType::Rz,
            GateType::P, GateType::U,
            // GPhase must be admitted or rebasing to this set silently drops
            // global phase, which ControlledBlock turns into a relative one.
            GateType::Measure, GateType::Barrier, GateType::GPhase,
        }
    };
}

GateSet full_gateset_2q() {
    return GateSet{
        .name = "full_2q",
        .allowed = {
            GateType::CX, GateType::CY, GateType::CZ,
            GateType::SWAP, GateType::ECR,
            GateType::iSWAP, GateType::iSWAPdg,
            GateType::CH, GateType::CS, GateType::CSdg, GateType::CSX, GateType::CSXdg,
            GateType::CRx, GateType::CRy, GateType::CRz, GateType::CP,
            GateType::RZZ, GateType::RXX, GateType::RYY,
            GateType::CU,
        }
    };
}

GateSet full_gateset_1q_2q() {
    GateSet gs = full_gateset_1q();
    for (auto g : full_gateset_2q().allowed) gs.allowed.insert(g);
    gs.name = "full_1q_2q";
    return gs;
}

GateSet routable_subset(const GateSet& gs) {
    GateSet out{gs.name + "_routable", {}, gs.rules};
    for (GateType g : gs.allowed) {
        // Static 3q entries (CCX, CSWAP) and variadic MCZ, whose table arity
        // is only a minimum.  Custom stays: its 1q form is the O1-fusion
        // output, and a multi-qubit Custom is already refused upstream.
        if (gate_num_qubits(g) > 2 || g == GateType::MCZ) continue;
        out.allowed.insert(g);
    }
    return out;
}

}  // namespace qarpx
