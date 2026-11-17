#include "qarpx/dag/circuit_dag.h"

#include <algorithm>
#include <functional>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>

namespace qarpx {

namespace {

bool is_branch_marker(GateType g) {
    return g == GateType::BranchBegin || g == GateType::BranchElse ||
           g == GateType::BranchEnd;
}

}  // anonymous namespace

CircuitDAG::CircuitDAG(uint32_t n_qubits, uint32_t n_cbits)
    : n_qubits_(n_qubits)
    , n_cbits_(n_cbits)
    , first_on_wire_(n_qubits + n_cbits + 1, kNone)
    , last_on_wire_(n_qubits + n_cbits + 1, kNone)
{}

CircuitDAG CircuitDAG::from_commands(const std::vector<Command>& cmds) {
    uint32_t n_qubits = 0;
    uint32_t n_cbits = 0;
    for (const auto& cmd : cmds) {
        for (auto q : cmd.qubits)         n_qubits = std::max(n_qubits, q + 1);
        for (auto c : cmd.cbits)          n_cbits  = std::max(n_cbits, c + 1);
        for (auto c : cmd.condition_bits) n_cbits  = std::max(n_cbits, c + 1);
    }
    return from_commands(cmds, n_qubits, n_cbits);
}

CircuitDAG CircuitDAG::from_commands(const std::vector<Command>& cmds,
                                     uint32_t n_qubits, uint32_t n_cbits) {
    CircuitDAG dag(n_qubits, n_cbits);
    dag.nodes_.reserve(cmds.size());

    for (std::size_t i = 0; i < cmds.size(); ++i) {
        const Command& cmd = cmds[i];

        if (cmd.gate == GateType::BranchElse || cmd.gate == GateType::BranchEnd) {
            throw std::invalid_argument(
                "CircuitDAG::from_commands: stray " +
                std::string(gate_name(cmd.gate)) + " at index " + std::to_string(i) +
                " (no open BranchBegin)");
        }

        Node node;
        node.original_index = static_cast<uint32_t>(i);

        if (cmd.gate == GateType::BranchBegin) {
            // Collapse the balanced region — nested groups included — into one
            // opaque node spanning all wires (full barrier, plan §4.3).
            std::size_t end = i;
            int depth = 0;
            for (std::size_t j = i; j < cmds.size(); ++j) {
                if (cmds[j].gate == GateType::BranchBegin)      ++depth;
                else if (cmds[j].gate == GateType::BranchEnd)   --depth;
                if (depth == 0) { end = j; break; }
            }
            if (depth != 0) {
                throw std::invalid_argument(
                    "CircuitDAG::from_commands: unterminated BranchBegin at index " +
                    std::to_string(i));
            }
            node.cmd = cmd;
            node.region.assign(cmds.begin() + static_cast<std::ptrdiff_t>(i),
                               cmds.begin() + static_cast<std::ptrdiff_t>(end) + 1);
            for (const auto& rc : node.region) dag.check_indices(rc);
            for (uint32_t w = 0; w < dag.n_wires(); ++w) node.wires.push_back(w);
            i = end;
        } else {
            dag.check_indices(cmd);
            node.cmd = cmd;
            node.wires = dag.compute_wires(cmd);
        }

        dag.link_node(std::move(node));
    }

    return dag;
}

SmallVector<uint32_t, 4> CircuitDAG::compute_wires(const Command& cmd) const {
    SmallVector<uint32_t, 4> wires;
    auto add_wire = [&](uint32_t w) {
        for (auto existing : wires)
            if (existing == w) return;
        wires.push_back(w);
    };
    for (auto q : cmd.qubits)         add_wire(qubit_wire(q));
    for (auto c : cmd.cbits)          add_wire(cbit_wire(c));
    for (auto c : cmd.condition_bits) add_wire(cbit_wire(c));
    if (cmd.gate == GateType::GPhase) add_wire(global_wire());
    // A qubit-less Barrier is the OpenQASM `barrier;` — it fences the whole
    // register (matching fuse_single_qubit_gates), not nothing.
    if (cmd.gate == GateType::Barrier && cmd.qubits.empty()) {
        for (uint32_t q = 0; q < n_qubits_; ++q) add_wire(qubit_wire(q));
        add_wire(global_wire());
    }
    return wires;
}

CircuitDAG::NodeId CircuitDAG::link_node(Node node) {
    const NodeId id = static_cast<NodeId>(nodes_.size());
    node.prev.resize(node.wires.size(), kNone);
    node.next.resize(node.wires.size(), kNone);

    for (std::size_t k = 0; k < node.wires.size(); ++k) {
        const uint32_t w = node.wires[k];
        const NodeId tail = last_on_wire_[w];
        node.prev[k] = tail;
        if (tail == kNone) {
            first_on_wire_[w] = id;
        } else {
            Node& t = nodes_[tail];
            t.next[wire_slot(tail, w)] = id;
        }
        last_on_wire_[w] = id;
    }

    nodes_.push_back(std::move(node));
    ++n_live_;
    return id;
}

std::vector<Command> CircuitDAG::to_commands() const {
    std::vector<Command> out;
    out.reserve(nodes_.size());

    // Kahn with a min-heap on original_index: ties among ready nodes resolve
    // to the earliest input position (RT-1).
    std::vector<uint32_t> indegree(nodes_.size(), 0);
    using Entry = std::pair<uint32_t, NodeId>;  // (original_index, id)
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> ready;

    for (NodeId id = 0; id < nodes_.size(); ++id) {
        const Node& n = nodes_[id];
        if (n.removed) continue;
        uint32_t deg = 0;
        for (auto p : n.prev)
            if (p != kNone) ++deg;
        indegree[id] = deg;
        if (deg == 0) ready.emplace(n.original_index, id);
    }

    while (!ready.empty()) {
        const NodeId id = ready.top().second;
        ready.pop();
        const Node& n = nodes_[id];

        if (!n.region.empty()) {
            out.insert(out.end(), n.region.begin(), n.region.end());
        } else {
            out.push_back(n.cmd);
        }

        for (auto s : n.next) {
            if (s == kNone) continue;
            if (--indegree[s] == 0) ready.emplace(nodes_[s].original_index, s);
        }
    }

    return out;
}

const Command& CircuitDAG::command(NodeId id) const {
    check_id(id);
    return nodes_[id].cmd;
}

bool CircuitDAG::is_region(NodeId id) const {
    check_id(id);
    return !nodes_[id].region.empty();
}

const std::vector<Command>& CircuitDAG::region_commands(NodeId id) const {
    check_id(id);
    if (nodes_[id].region.empty())
        throw std::invalid_argument(
            "CircuitDAG::region_commands: node " + std::to_string(id) +
            " is not a region node");
    return nodes_[id].region;
}

bool CircuitDAG::is_removed(NodeId id) const {
    if (id >= nodes_.size())
        throw std::invalid_argument(
            "CircuitDAG: node id " + std::to_string(id) + " out of range");
    return nodes_[id].removed;
}

const SmallVector<uint32_t, 4>& CircuitDAG::wires(NodeId id) const {
    check_id(id);
    return nodes_[id].wires;
}

CircuitDAG::NodeId CircuitDAG::wire_front(uint32_t wire) const {
    if (wire >= n_wires())
        throw std::invalid_argument("CircuitDAG::wire_front: wire out of range");
    return first_on_wire_[wire];
}

CircuitDAG::NodeId CircuitDAG::wire_back(uint32_t wire) const {
    if (wire >= n_wires())
        throw std::invalid_argument("CircuitDAG::wire_back: wire out of range");
    return last_on_wire_[wire];
}

CircuitDAG::NodeId CircuitDAG::next_on_wire(NodeId id, uint32_t wire) const {
    check_id(id);
    return nodes_[id].next[wire_slot(id, wire)];
}

CircuitDAG::NodeId CircuitDAG::prev_on_wire(NodeId id, uint32_t wire) const {
    check_id(id);
    return nodes_[id].prev[wire_slot(id, wire)];
}

std::size_t CircuitDAG::depth() const {
    // Edges always point from lower to higher original_index (construction
    // appends after wire tails; remove_node only splices), so a forward scan
    // over node ids is a valid topological order.
    std::vector<std::size_t> longest(nodes_.size(), 0);
    std::size_t result = 0;

    for (NodeId id = 0; id < nodes_.size(); ++id) {
        const Node& n = nodes_[id];
        if (n.removed) continue;

        std::size_t at = 0;
        for (auto p : n.prev)
            if (p != kNone) at = std::max(at, longest[p]);

        const bool weightless = n.region.empty() &&
            (n.cmd.gate == GateType::Barrier || n.cmd.gate == GateType::GPhase);
        at += weightless ? 0 : 1;

        longest[id] = at;
        result = std::max(result, at);
    }
    return result;
}

std::vector<std::vector<CircuitDAG::NodeId>> CircuitDAG::layers() const {
    // Forward scan is a valid topological order (edges point from lower to
    // higher original_index — see depth()).
    std::vector<std::size_t> level(nodes_.size(), 0);
    std::vector<std::vector<NodeId>> out;

    for (NodeId id = 0; id < nodes_.size(); ++id) {
        const Node& n = nodes_[id];
        if (n.removed) continue;

        std::size_t at = 0;
        for (auto p : n.prev)
            if (p != kNone) at = std::max(at, level[p] + 1);
        level[id] = at;

        if (at >= out.size()) out.resize(at + 1);
        out[at].push_back(id);
    }
    return out;
}

std::unordered_map<std::string, std::size_t> CircuitDAG::count_ops() const {
    std::unordered_map<std::string, std::size_t> counts;
    for (NodeId id = 0; id < nodes_.size(); ++id) {
        const Node& n = nodes_[id];
        if (n.removed) continue;
        if (!n.region.empty()) {
            for (const auto& rc : n.region)
                ++counts[std::string(gate_name(rc.gate))];
        } else {
            ++counts[std::string(gate_name(n.cmd.gate))];
        }
    }
    return counts;
}

std::vector<CircuitDAG::NodeId> CircuitDAG::front_layer() const {
    std::vector<NodeId> out;
    for (NodeId id = 0; id < nodes_.size(); ++id) {
        const Node& n = nodes_[id];
        if (n.removed) continue;
        bool ready = true;
        for (auto p : n.prev)
            if (p != kNone) { ready = false; break; }
        if (ready) out.push_back(id);
    }
    // Node ids are assigned in input order, so `out` is already sorted by
    // original_index.
    return out;
}

void CircuitDAG::remove_node(NodeId id) {
    check_id(id);
    Node& n = nodes_[id];

    for (std::size_t k = 0; k < n.wires.size(); ++k) {
        const uint32_t w = n.wires[k];
        const NodeId p = n.prev[k];
        const NodeId s = n.next[k];

        if (p != kNone) nodes_[p].next[wire_slot(p, w)] = s;
        else            first_on_wire_[w] = s;

        if (s != kNone) nodes_[s].prev[wire_slot(s, w)] = p;
        else            last_on_wire_[w] = p;

        n.prev[k] = kNone;
        n.next[k] = kNone;
    }

    n.removed = true;
    --n_live_;
}

void CircuitDAG::replace_command(NodeId id, Command cmd) {
    check_id(id);
    Node& n = nodes_[id];
    if (!n.region.empty())
        throw std::invalid_argument(
            "CircuitDAG::replace_command: node " + std::to_string(id) +
            " is a region node");

    // Same wire set (order-insensitive) — links stay valid without re-linking.
    auto new_wires = compute_wires(cmd);
    bool same = new_wires.size() == n.wires.size();
    if (same) {
        for (auto w : new_wires) {
            bool found = false;
            for (auto existing : n.wires)
                if (existing == w) { found = true; break; }
            if (!found) { same = false; break; }
        }
    }
    if (!same)
        throw std::invalid_argument(
            "CircuitDAG::replace_command: replacement for node " +
            std::to_string(id) + " touches a different wire set");

    n.cmd = std::move(cmd);
}

void CircuitDAG::replace_region_commands(NodeId id, std::vector<Command> span) {
    check_id(id);
    Node& n = nodes_[id];
    if (n.region.empty())
        throw std::invalid_argument(
            "CircuitDAG::replace_region_commands: node " + std::to_string(id) +
            " is not a region node");
    if (span.size() < 2 || span.front().gate != GateType::BranchBegin ||
        span.back().gate != GateType::BranchEnd)
        throw std::invalid_argument(
            "CircuitDAG::replace_region_commands: span must be "
            "BranchBegin…BranchEnd delimited");
    for (const auto& cmd : span) check_indices(cmd);
    n.region = std::move(span);
}

void CircuitDAG::check_indices(const Command& cmd) const {
    for (auto q : cmd.qubits) {
        if (q >= n_qubits_)
            throw std::invalid_argument(
                "CircuitDAG: qubit " + std::to_string(q) +
                " out of range for n_qubits=" + std::to_string(n_qubits_));
    }
    for (auto c : cmd.cbits) {
        if (c >= n_cbits_)
            throw std::invalid_argument(
                "CircuitDAG: cbit " + std::to_string(c) +
                " out of range for n_cbits=" + std::to_string(n_cbits_));
    }
    for (auto c : cmd.condition_bits) {
        if (c >= n_cbits_)
            throw std::invalid_argument(
                "CircuitDAG: condition bit " + std::to_string(c) +
                " out of range for n_cbits=" + std::to_string(n_cbits_));
    }
}

std::size_t CircuitDAG::wire_slot(NodeId id, uint32_t wire) const {
    const Node& n = nodes_[id];
    for (std::size_t k = 0; k < n.wires.size(); ++k)
        if (n.wires[k] == wire) return k;
    throw std::invalid_argument(
        "CircuitDAG: node " + std::to_string(id) + " does not touch wire " +
        std::to_string(wire));
}

/// Bounds + liveness check: every accessor except `is_removed` rejects
/// tombstoned nodes.
void CircuitDAG::check_id(NodeId id) const {
    if (id >= nodes_.size())
        throw std::invalid_argument(
            "CircuitDAG: node id " + std::to_string(id) + " out of range");
    if (nodes_[id].removed)
        throw std::invalid_argument(
            "CircuitDAG: node " + std::to_string(id) + " has been removed");
}

}  // namespace qarpx
