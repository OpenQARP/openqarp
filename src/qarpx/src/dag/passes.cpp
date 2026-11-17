#include "qarpx/dag/passes.h"

#include "qarpx/dag/commutation.h"
#include "qarpx/transpiler/identities.h"

#include <cmath>
#include <deque>
#include <functional>
#include <vector>

namespace qarpx::dag_passes {

namespace {

using NodeId = CircuitDAG::NodeId;

bool is_zero_angle(const Param& p) {
    return p.is_concrete() && std::abs(p.value()) < 1e-12;
}

// A zero angle on any gate of the §9 single-parameter family is exactly the
// identity (P(0) = CP(0) = RZZ(0) = GPhase(0) = I, global phase included), so
// the command can go.  Was Rx/Ry/Rz only; a lone CP(0) used to survive.
bool is_droppable_zero_rotation(const Command& c) {
    return gate_is_additive_in_param(c.gate) &&
           c.params.size() == 1 && is_zero_angle(c.params[0]);
}

/// GPhase pairs merge like rotations; new at O1 (free on the global wire,
/// plan §4.2).  Condition-gated like every other rewrite.
bool can_merge_gphase(const Command& a, const Command& b) {
    return a.gate == GateType::GPhase && b.gate == GateType::GPhase &&
           a.params.size() == 1 && b.params.size() == 1 &&
           a.condition_bits == b.condition_bits &&
           a.condition_values == b.condition_values &&
           // Same linear-form constraint as the rotation merge.
           merged_param_stays_linear(a.params[0], b.params[0]);
}

/// Optimize each body of a `BranchBegin … [BranchElse …] BranchEnd` span as
/// an independent sub-stream (nested regions recurse via from_commands).
/// The region BOUNDARY stays a barrier: commands are combined only within a
/// body, never across one.  `optimizer` is the calling pass, applied to each
/// body's sub-DAG.  Returns the optimized span
/// and adds to `eliminated`.
std::vector<Command> optimize_region_span(
    const std::vector<Command>& span, std::size_t& eliminated,
    const std::function<std::size_t(CircuitDAG&)>& optimizer) {
    // Top-level BranchElse (depth 0 relative to the interior), if any.
    std::size_t else_idx = 0;
    bool has_else = false;
    int depth = 0;
    for (std::size_t i = 1; i + 1 < span.size(); ++i) {
        if (span[i].gate == GateType::BranchBegin)    ++depth;
        else if (span[i].gate == GateType::BranchEnd) --depth;
        else if (span[i].gate == GateType::BranchElse && depth == 0) {
            else_idx = i;
            has_else = true;
            break;
        }
    }

    auto optimize_body = [&](std::size_t first, std::size_t last) {
        std::vector<Command> body(span.begin() + static_cast<std::ptrdiff_t>(first),
                                  span.begin() + static_cast<std::ptrdiff_t>(last));
        auto dag = CircuitDAG::from_commands(body);
        eliminated += optimizer(dag);
        return dag.to_commands();
    };

    std::vector<Command> out;
    out.reserve(span.size());
    out.push_back(span.front());  // BranchBegin
    if (has_else) {
        auto then_body = optimize_body(1, else_idx);
        auto else_body = optimize_body(else_idx + 1, span.size() - 1);
        out.insert(out.end(), then_body.begin(), then_body.end());
        out.push_back(span[else_idx]);  // BranchElse
        out.insert(out.end(), else_body.begin(), else_body.end());
    } else {
        auto then_body = optimize_body(1, span.size() - 1);
        out.insert(out.end(), then_body.begin(), then_body.end());
    }
    out.push_back(span.back());  // BranchEnd
    return out;
}

}  // anonymous namespace

std::size_t cancel_wire_adjacent(CircuitDAG& dag) {
    std::size_t eliminated = 0;

    std::deque<NodeId> work;
    std::vector<char> queued(dag.n_slots(), 0);
    for (NodeId id = 0; id < dag.n_slots(); ++id) {
        if (!dag.is_removed(id)) {
            work.push_back(id);
            queued[id] = 1;
        }
    }

    auto enqueue = [&](NodeId id) {
        if (id == CircuitDAG::kNone || dag.is_removed(id) || queued[id]) return;
        queued[id] = 1;
        work.push_back(id);
    };

    // Wire neighbours whose adjacency changes when `id` is removed / merged.
    auto collect_neighbours = [&](NodeId id, std::vector<NodeId>& out) {
        for (auto w : dag.wires(id)) {
            out.push_back(dag.prev_on_wire(id, w));
            out.push_back(dag.next_on_wire(id, w));
        }
    };

    std::vector<NodeId> touched;
    while (!work.empty()) {
        const NodeId id = work.front();
        work.pop_front();
        queued[id] = 0;
        if (dag.is_removed(id)) continue;

        if (dag.is_region(id)) {
            // Boundary stays a barrier; the interior bodies are optimized
            // recursively.
            dag.replace_region_commands(
                id, optimize_region_span(
                        dag.region_commands(id), eliminated,
                        [](CircuitDAG& d) { return cancel_wire_adjacent(d); }));
            continue;
        }

        const Command& cmd = dag.command(id);

        if (is_droppable_zero_rotation(cmd)) {
            touched.clear();
            collect_neighbours(id, touched);
            dag.remove_node(id);
            ++eliminated;
            for (auto t : touched) enqueue(t);
            continue;
        }

        // Candidate partner: the successor on the first wire, provided it is
        // the immediate successor on EVERY wire and spans no extra wires —
        // the wire-adjacency generalization of textual adjacency.
        const auto& ws = dag.wires(id);
        if (ws.empty()) continue;
        const NodeId b = dag.next_on_wire(id, ws[0]);
        if (b == CircuitDAG::kNone || dag.is_region(b)) continue;

        bool adjacent = true;
        for (auto w : ws) {
            if (dag.next_on_wire(id, w) != b) { adjacent = false; break; }
        }
        if (!adjacent || dag.wires(b).size() != ws.size()) continue;

        const Command& bc = dag.command(b);

        if (are_inverse_pair(cmd, bc)) {
            touched.clear();
            collect_neighbours(id, touched);
            collect_neighbours(b, touched);
            dag.remove_node(b);
            dag.remove_node(id);
            eliminated += 2;
            for (auto t : touched) enqueue(t);
            continue;
        }

        if (can_merge_rotations(cmd, bc) || can_merge_gphase(cmd, bc)) {
            const Param merged = cmd.params[0] + bc.params[0];
            touched.clear();
            collect_neighbours(id, touched);
            collect_neighbours(b, touched);
            dag.remove_node(b);
            if (is_zero_angle(merged)) {
                dag.remove_node(id);
                eliminated += 2;
            } else {
                Command nc = cmd;
                nc.params[0] = merged;
                dag.replace_command(id, std::move(nc));
                ++eliminated;
                enqueue(id);  // may now combine with its new successor
            }
            for (auto t : touched) enqueue(t);
            continue;
        }
    }

    return eliminated;
}

std::size_t commute_and_cancel(CircuitDAG& dag, std::size_t window) {
    std::size_t eliminated = 0;

    std::deque<NodeId> work;
    std::vector<char> queued(dag.n_slots(), 0);
    for (NodeId id = 0; id < dag.n_slots(); ++id) {
        if (!dag.is_removed(id)) {
            work.push_back(id);
            queued[id] = 1;
        }
    }

    auto enqueue = [&](NodeId id) {
        if (id == CircuitDAG::kNone || dag.is_removed(id) || queued[id]) return;
        queued[id] = 1;
        work.push_back(id);
    };

    auto collect_neighbours = [&](NodeId id, std::vector<NodeId>& out) {
        for (auto w : dag.wires(id)) {
            out.push_back(dag.prev_on_wire(id, w));
            out.push_back(dag.next_on_wire(id, w));
        }
    };

    auto can_combine = [](const Command& a, const Command& b) {
        return are_inverse_pair(a, b) || can_merge_rotations(a, b) ||
               can_merge_gphase(a, b);
    };

    // Every node sharing a wire with A strictly between A and B must commute
    // with A — then A slides (virtually) up to B and the O1 rewrite applies.
    auto interveners_commute = [&](NodeId a, NodeId b) {
        const Command& ac = dag.command(a);
        for (auto w : dag.wires(a)) {
            NodeId cur = dag.next_on_wire(a, w);
            std::size_t steps = 0;
            while (cur != b) {
                if (cur == CircuitDAG::kNone || ++steps > window) return false;
                if (dag.is_region(cur) || !commute(ac, dag.command(cur)))
                    return false;
                cur = dag.next_on_wire(cur, w);
            }
        }
        return true;
    };

    std::vector<NodeId> touched;
    while (!work.empty()) {
        const NodeId id = work.front();
        work.pop_front();
        queued[id] = 0;
        if (dag.is_removed(id)) continue;

        if (dag.is_region(id)) {
            dag.replace_region_commands(
                id, optimize_region_span(
                        dag.region_commands(id), eliminated,
                        [window](CircuitDAG& d) {
                            return commute_and_cancel(d, window);
                        }));
            continue;
        }

        const Command& cmd = dag.command(id);
        const auto& ws = dag.wires(id);
        if (ws.empty()) continue;

        // Walk A's first wire looking for a combinable partner it can
        // (virtually) commute up to.  Non-commuting non-partner ⇒ stop.
        NodeId cur = dag.next_on_wire(id, ws[0]);
        for (std::size_t step = 0;
             step < window && cur != CircuitDAG::kNone; ++step) {
            if (dag.is_region(cur)) break;
            const Command& bc = dag.command(cur);

            if (can_combine(cmd, bc) && interveners_commute(id, cur)) {
                touched.clear();
                collect_neighbours(id, touched);
                collect_neighbours(cur, touched);

                if (are_inverse_pair(cmd, bc)) {
                    dag.remove_node(cur);
                    dag.remove_node(id);
                    eliminated += 2;
                } else {
                    const Param merged = cmd.params[0] + bc.params[0];
                    if (merged.is_concrete() &&
                        std::abs(merged.value()) < 1e-12) {
                        dag.remove_node(cur);
                        dag.remove_node(id);
                        eliminated += 2;
                    } else {
                        // Slide-right semantics: the merge lands at B.
                        Command nc = bc;
                        nc.params[0] = merged;
                        dag.remove_node(id);
                        dag.replace_command(cur, std::move(nc));
                        ++eliminated;
                        enqueue(cur);
                    }
                }
                for (auto t : touched) enqueue(t);
                break;
            }

            if (!commute(cmd, bc)) break;
            cur = dag.next_on_wire(cur, ws[0]);
        }
    }

    return eliminated;
}

}  // namespace qarpx::dag_passes
