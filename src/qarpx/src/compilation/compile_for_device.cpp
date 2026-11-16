#include "qarpx/compilation/compile_for_device.h"

#include "qarpx/compilation/router.h"
#include "qarpx/core/errors.h"
#include "qarpx/transpiler/transpiler.h"
#include "routing_common.h"

#include <string>

namespace qarpx {

namespace {

std::vector<uint32_t> identity_mapping(uint32_t n) {
    std::vector<uint32_t> id(n);
    for (uint32_t i = 0; i < n; ++i) id[i] = i;
    return id;
}

Command with_condition_of(const Command& src, GateType g, uint32_t q0,
                          std::optional<uint32_t> q1 = std::nullopt) {
    Command c;
    c.gate = g;
    c.qubits.push_back(q0);
    if (q1) c.qubits.push_back(*q1);
    for (auto b : src.condition_bits)   c.condition_bits.push_back(b);
    for (auto v : src.condition_values) c.condition_values.push_back(v);
    return c;
}

/// SWAP(a,b) as three CX all in the stored edge's orientation:
/// CX(a,b) · H(a)H(b) · CX(a,b) · H(a)H(b) · CX(a,b)  (the middle bracket is
/// CX(b,a)).  A SWAP on a pair that is no edge at all is a router bug, not a
/// capability of the circuit — surfaced as such.
void lower_swap_directed(const Command& swap, const Architecture& arch,
                         std::vector<Command>& out) {
    uint32_t a = swap.qubits[0], b = swap.qubits[1];
    if (!arch.has_directed_edge(a, b)) {
        if (!arch.has_directed_edge(b, a)) {
            throw capability_error(
                "compile_for_device: SWAP on physical qubits " + std::to_string(a) +
                ", " + std::to_string(b) + " which the architecture does not connect",
                swap);
        }
        std::swap(a, b);
    }
    auto cx = [&] { out.push_back(with_condition_of(swap, GateType::CX, a, b)); };
    auto hh = [&] {
        out.push_back(with_condition_of(swap, GateType::H, a));
        out.push_back(with_condition_of(swap, GateType::H, b));
    };
    cx(); hh(); cx(); hh(); cx();
}

std::vector<Command> lower_swaps_directed(const std::vector<Command>& commands,
                                          const Architecture&         arch) {
    std::vector<Command> out;
    out.reserve(commands.size());
    for (const auto& cmd : commands) {
        if (cmd.gate == GateType::SWAP) lower_swap_directed(cmd, arch, out);
        else                            out.push_back(cmd);
    }
    return out;
}

}  // namespace

void assert_directions(const std::vector<Command>& commands,
                       const Architecture&         arch) {
    for (const auto& cmd : commands) {
        if (cmd.qubits.size() != 2 || !routing_detail::is_asymmetric_2q(cmd.gate))
            continue;
        const uint32_t a = cmd.qubits[0], b = cmd.qubits[1];
        if (arch.has_directed_edge(a, b)) continue;
        throw capability_error(
            "compile_for_device: gate '" + std::string(gate_name(cmd.gate)) +
            "' on physical qubits (" + std::to_string(a) + ", " + std::to_string(b) +
            ") runs against the directed architecture" +
            (arch.has_directed_edge(b, a) ? " (only the reversed edge exists)"
                                           : " (no edge between them)") +
            "; the device cannot run it in this orientation",
            cmd);
    }
}

CompiledCircuit compile_for_device(const std::vector<Command>& commands,
                                   uint32_t                    n_qubits,
                                   const Device&               device,
                                   RouterKind                  router) {
    device.check_fits(n_qubits);

    CompiledCircuit out;
    out.commands                    = commands;
    out.initial_logical_to_physical = identity_mapping(device.n_qubits);
    out.final_logical_to_physical   = out.initial_logical_to_physical;

    // 1. Rebase to device gate set (if specified).  With an architecture the
    // router follows, and it accepts 0/1/2-qubit gates only (§14) — so this
    // pass targets the routable subset, dropping gates a device may legally
    // keep native when unrouted (MCZ, CCX, CSWAP).  Pass 3 restores the full
    // set.  Without a gate set there is nothing to rebase to: a wide gate
    // then reaches the router, which throws with its own actionable message.
    if (device.gate_set.has_value()) {
        const GateSet pre = device.architecture.has_value()
            ? routable_subset(*device.gate_set)
            : *device.gate_set;
        Transpiler t(pre);
        out.commands = t.transpile(out.commands);
    }

    // 2. Route to architecture (if specified).
    if (device.architecture.has_value()) {
        RoutingOptions ro;
        ro.arch         = *device.architecture;
        ro.directedness = device.directedness;
        ro.router       = router;
        auto routed = route(out.commands, ro);
        out.commands                    = std::move(routed.commands);
        out.initial_logical_to_physical = std::move(routed.initial_logical_to_physical);
        out.final_logical_to_physical   = std::move(routed.final_logical_to_physical);

        // Directed SWAP lowering before the rebase sees them: the generic
        // SWAP rule is CX·CX(reversed)·CX and would hand pass 3 a gate
        // against the edge.  A target that keeps SWAP native needs none.
        const bool target_has_swap =
            device.gate_set.has_value() && device.gate_set->contains(GateType::SWAP);
        if (device.directedness && device.gate_set.has_value() && !target_has_swap)
            out.commands = lower_swaps_directed(out.commands, *device.architecture);

        // 3. Second rebase pass — decomposes router-introduced SWAPs and
        // H-conjugated CX into the target gate set.  Only needed when
        // both a gate set AND an architecture were supplied.
        if (device.gate_set.has_value()) {
            Transpiler t(*device.gate_set);
            out.commands = t.transpile(out.commands);
        }

        if (device.directedness) assert_directions(out.commands, *device.architecture);
    }

    return out;
}

}  // namespace qarpx
