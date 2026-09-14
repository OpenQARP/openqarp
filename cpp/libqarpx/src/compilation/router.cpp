#include "qarpx/compilation/router.h"

#include "qarpx/compilation/sabre.h"
#include "routing_common.h"
#include "qarpx/core/errors.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace qarpx {

namespace {

using routing_detail::Resolution;
using routing_detail::emit_h_conjugated_cx;
using routing_detail::resolve;
using routing_detail::resolve_initial_mapping;
using routing_detail::translated_copy;

/// Move logical qubit at `p_moving` one step along the shortest path
/// toward `p_target` via an emitted SWAP.  Returns the new physical
/// position of the moving qubit (== path[1]) and emits a SWAP command,
/// updating both the forward and reverse mappings.
///
/// Pre: `p_moving != p_target`, and a path exists.
uint32_t step_swap(uint32_t                     p_moving,
                   uint32_t                     p_target,
                   const Architecture&          arch,
                   std::vector<uint32_t>&       l2p,
                   std::vector<uint32_t>&       p2l,
                   std::vector<Command>&        out_commands) {
    const auto path = arch.shortest_path(p_moving, p_target);
    if (path.size() < 2) {
        throw capability_error(
            "Router: physical qubits " + std::to_string(p_moving) + " and "
            + std::to_string(p_target) + " are disconnected — cannot route.");
    }
    const uint32_t a = path[0];   // == p_moving
    const uint32_t b = path[1];

    Command swap_cmd;
    swap_cmd.gate = GateType::SWAP;
    swap_cmd.qubits.push_back(a);
    swap_cmd.qubits.push_back(b);
    out_commands.push_back(swap_cmd);

    const uint32_t la = p2l[a];
    const uint32_t lb = p2l[b];
    l2p[la] = b;
    l2p[lb] = a;
    p2l[a] = lb;
    p2l[b] = la;

    return b;
}

RoutingResult route_lite(const std::vector<Command>& commands,
                         const RoutingOptions&        opts) {
    const uint32_t n = opts.arch.n_qubits;

    std::vector<uint32_t> l2p, p2l;
    resolve_initial_mapping(opts.initial_mapping, n, l2p, p2l);

    RoutingResult result;
    result.initial_logical_to_physical = l2p;  // before routing mutates it
    result.commands.reserve(commands.size());

    // Depth of BranchBegin…BranchEnd nesting.  SWAPs inside a region are
    // forbidden: the condition lives on the markers (no per-command
    // residue), so unconditional SWAPs inside the conditionally-executed
    // body would desynchronize the mapping when the branch is not taken.
    int branch_depth = 0;

    for (const auto& cmd : commands) {
        // Reject qubit indices out of range.
        for (auto q : cmd.qubits) {
            if (q >= n) {
                throw capability_error(
                    "Router: command on gate '" + std::string(gate_name(cmd.gate))
                    + "' references logical qubit " + std::to_string(q)
                    + " >= n_qubits (" + std::to_string(n) + ").");
            }
        }

        if (cmd.gate == GateType::BranchBegin)      ++branch_depth;
        else if (cmd.gate == GateType::BranchEnd)   --branch_depth;

        const size_t k = cmd.qubits.size();

        if (k == 0) {
            result.commands.push_back(cmd);
            continue;
        }

        if (k == 1) {
            result.commands.push_back(translated_copy(cmd, l2p));
            continue;
        }

        if (k == 2) {
            uint32_t p0 = l2p[cmd.qubits[0]];
            uint32_t p1 = l2p[cmd.qubits[1]];

            // Greedy SWAP insertion until resolution lands on something we
            // can emit.  Each iteration moves the first qubit one step along
            // the shortest path toward the second.
            while (true) {
                const auto r = resolve(cmd.gate, p0, p1, opts.arch, opts.directedness);
                if (r == Resolution::EMIT_AS_IS) {
                    Command out = translated_copy(cmd, l2p);
                    out.qubits[0] = p0;
                    out.qubits[1] = p1;
                    result.commands.push_back(std::move(out));
                    break;
                }
                if (r == Resolution::EMIT_H_CONJUGATED) {
                    emit_h_conjugated_cx(cmd, p0, p1, result.commands);
                    break;
                }
                // NEEDS_SWAPS: conditional 2q gates can't be routed safely.
                if (!cmd.condition_bits.empty()) {
                    throw capability_error(
                        "Router: 2-qubit gate '" + std::string(gate_name(cmd.gate))
                        + "' has a classical condition AND requires SWAPs.  "
                        "Unconditional SWAPs around a conditional gate would "
                        "corrupt state when the condition is false.  Either "
                        "place the qubits adjacently in the input, or rewrite "
                        "the conditional with explicit BranchBegin/Else/End "
                        "scoping the SWAPs.");
                }
                if (branch_depth > 0) {
                    throw capability_error(
                        "Router: 2-qubit gate '" + std::string(gate_name(cmd.gate))
                        + "' inside a BranchBegin/BranchEnd region requires "
                        "SWAPs.  The region executes conditionally, so "
                        "unconditional SWAPs inside it would desynchronize "
                        "the qubit mapping when the branch is not taken.  "
                        "Place the interacting qubits adjacently in the "
                        "input (or via initial_mapping).");
                }
                p0 = step_swap(p0, p1, opts.arch, l2p, p2l, result.commands);
            }
            continue;
        }

        throw capability_error(
            "Router: gate '" + std::string(gate_name(cmd.gate)) + "' has "
            + std::to_string(k) + " qubits; the router supports 0/1/2-qubit "
            "gates only.  Rebase to a 1q/2q gate set first (qx::Transpiler).");
    }

    result.final_logical_to_physical = std::move(l2p);
    return result;
}

}  // namespace

RoutingResult route(const std::vector<Command>& commands,
                    const RoutingOptions&        opts) {
    if (opts.arch.n_qubits == 0) {
        throw capability_error("Router: arch.n_qubits must be > 0");
    }
    if (opts.router == RouterKind::Sabre) {
        return sabre_route(commands, opts);
    }
    return route_lite(commands, opts);
}

SamplingResult reindex_sampling_result(
    const SamplingResult&            physical,
    const std::vector<uint32_t>&     final_logical_to_physical)
{
    SamplingResult logical;
    logical.n_qubits      = physical.n_qubits;
    logical.n_shots       = physical.n_shots;
    logical.n_cbits       = physical.n_cbits;
    logical.cbit_history  = physical.cbit_history;  // cbit space unchanged

    const size_t n = final_logical_to_physical.size();
    for (const auto& [phys_bits, count] : physical.counts) {
        uint64_t log_bits = 0;
        for (size_t l = 0; l < n; ++l) {
            const uint32_t p = final_logical_to_physical[l];
            if (phys_bits & (uint64_t{1} << p)) {
                log_bits |= (uint64_t{1} << l);
            }
        }
        logical.counts[log_bits] += count;
    }
    return logical;
}

}  // namespace qarpx
