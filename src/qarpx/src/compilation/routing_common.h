// Internal helpers shared by the Lite and SABRE routers.  Not installed.
#pragma once

#include "qarpx/core/command.h"
#include "qarpx/core/errors.h"
#include "qarpx/device/architecture.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace qarpx::routing_detail {

/// 2q gates whose unitary changes under qubit exchange, so a directed edge
/// constrains their orientation.  ECR, CH, CSX, CSXdg are asymmetric too —
/// they used to be treated as direction-free and slipped through reversed.
constexpr bool is_asymmetric_2q(GateType g) {
    switch (g) {
        case GateType::CX:  case GateType::CY:
        case GateType::CRx: case GateType::CRy: case GateType::CRz:
        case GateType::CP:  case GateType::CU:
        case GateType::ECR: case GateType::CH:
        case GateType::CSX: case GateType::CSXdg:
            return true;
        default:
            return false;
    }
}

/// Resolution outcome for "is the current physical placement satisfiable
/// for this 2q gate?".
enum class Resolution {
    NEEDS_SWAPS,        // qubits are not adjacent — insert SWAPs.
    EMIT_AS_IS,         // emit with translated qubits.
    EMIT_H_CONJUGATED,  // emit H-conjugated CX (direction flip).
};

inline Resolution resolve(GateType g, uint32_t p0, uint32_t p1,
                          const Architecture& arch, bool directedness) {
    if (!arch.is_connected(p0, p1)) return Resolution::NEEDS_SWAPS;
    if (!directedness)              return Resolution::EMIT_AS_IS;
    if (!is_asymmetric_2q(g))       return Resolution::EMIT_AS_IS;
    if (arch.has_directed_edge(p0, p1)) return Resolution::EMIT_AS_IS;
    if (g == GateType::CX && arch.has_directed_edge(p1, p0))
        return Resolution::EMIT_H_CONJUGATED;
    // Non-CX asymmetric with wrong direction: not supported.
    throw capability_error(
        "Router: gate '" + std::string(gate_name(g))
        + "' is required in a direction not provided by the directed "
        "architecture, and the router does not H-conjugate non-CX "
        "asymmetric 2q gates.  Rebase to CX first (qx::Transpiler with a "
        "CX-only 2q gate set), then route.");
}

inline Command translated_copy(const Command& src,
                               const std::vector<uint32_t>& l2p) {
    Command out;
    out.gate = src.gate;
    for (auto q : src.qubits) out.qubits.push_back(l2p[q]);
    for (auto& p : src.params) out.params.push_back(p);
    for (auto c : src.cbits) out.cbits.push_back(c);
    for (auto b : src.condition_bits) out.condition_bits.push_back(b);
    for (auto v : src.condition_values) out.condition_values.push_back(v);
    out.unitary = src.unitary;
    return out;
}

inline void emit_h_conjugated_cx(const Command&        src,
                                 uint32_t              pa,
                                 uint32_t              pb,
                                 std::vector<Command>& out) {
    // CX(pa, pb) when only edge (pb, pa) exists, expanded as:
    //   H(pa), H(pb), CX(pb, pa), H(pa), H(pb).
    // Any classical condition on the CX must also gate the H bracketing.
    auto emit_h = [&](uint32_t q) {
        Command h_cmd;
        h_cmd.gate = GateType::H;
        h_cmd.qubits.push_back(q);
        for (auto b : src.condition_bits) h_cmd.condition_bits.push_back(b);
        for (auto v : src.condition_values) h_cmd.condition_values.push_back(v);
        out.push_back(h_cmd);
    };
    emit_h(pa);
    emit_h(pb);
    Command flipped;
    flipped.gate = GateType::CX;
    flipped.qubits.push_back(pb);
    flipped.qubits.push_back(pa);
    for (auto& p : src.params) flipped.params.push_back(p);
    for (auto c : src.cbits) flipped.cbits.push_back(c);
    for (auto b : src.condition_bits) flipped.condition_bits.push_back(b);
    for (auto v : src.condition_values) flipped.condition_values.push_back(v);
    out.push_back(flipped);
    emit_h(pa);
    emit_h(pb);
}

/// Validate + return the initial logical→physical mapping (identity when
/// absent) and its inverse.  Throws capability_error on size mismatch or
/// non-permutation.
inline void resolve_initial_mapping(
    const std::optional<std::vector<uint32_t>>& initial,
    uint32_t n,
    std::vector<uint32_t>& l2p,
    std::vector<uint32_t>& p2l) {
    l2p.resize(n);
    if (initial) {
        if (initial->size() != n) {
            throw capability_error(
                "Router: initial_mapping size (" + std::to_string(initial->size())
                + ") must equal arch.n_qubits (" + std::to_string(n) + ").");
        }
        l2p = *initial;
    } else {
        for (uint32_t i = 0; i < n; ++i) l2p[i] = i;
    }
    p2l.assign(n, n);  // sentinel = n (unset)
    for (uint32_t l = 0; l < n; ++l) {
        const uint32_t p = l2p[l];
        if (p >= n)
            throw capability_error(
                "Router: initial_mapping entry " + std::to_string(p)
                + " >= n_qubits.");
        if (p2l[p] != n)
            throw capability_error(
                "Router: initial_mapping is not a permutation (duplicate "
                "physical " + std::to_string(p) + ").");
        p2l[p] = l;
    }
}

}  // namespace qarpx::routing_detail
