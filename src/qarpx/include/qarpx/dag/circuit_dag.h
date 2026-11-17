#pragma once

#include "../core/command.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace qarpx {

/// Wire-dependency DAG over a flat command stream.
///
/// Built from a `std::vector<Command>` and re-linearized back to one; the flat
/// vector remains the exchange format everywhere else.
/// Nodes are commands; edges are per-wire ordering dependencies.  Wire id
/// space, in one flat index range:
///
///   [0, n_qubits)                       qubit wires
///   [n_qubits, n_qubits + n_cbits)      cbit wires (conservatively totally
///                                       ordered: every read or write of a
///                                       cbit depends on its previous touch)
///   n_qubits + n_cbits                  one global-phase wire (GPhase only)
///
/// A `BranchBegin … [BranchElse] … BranchEnd` group — including any nested
/// groups — collapses into ONE opaque region node spanning ALL wires (full
/// barrier) whose raw sub-vector is re-emitted verbatim by `to_commands()`.
///
/// Contract RT-1 (round-trip exactness): `to_commands(from_commands(v)) == v`
/// element-for-element.  `to_commands()` is a stable topological sort with
/// minimum-original-index tie-breaking; the input order is itself a valid
/// linearization, so with no mutations the sort reproduces it exactly, and
/// after `remove_node` calls it yields the survivors in original order.
class CircuitDAG {
public:
    using NodeId = uint32_t;

    /// Sentinel: no node (wire boundary / removed link).
    static constexpr NodeId kNone = std::numeric_limits<NodeId>::max();

    /// Build from a flat stream.  Throws std::invalid_argument on a qubit /
    /// cbit index out of range, on a stray BranchElse / BranchEnd, or on an
    /// unterminated BranchBegin.
    static CircuitDAG from_commands(const std::vector<Command>& cmds,
                                    uint32_t n_qubits, uint32_t n_cbits);

    /// Convenience overload: infers n_qubits / n_cbits from the largest
    /// qubit / cbit / condition-bit index in the stream.
    static CircuitDAG from_commands(const std::vector<Command>& cmds);

    /// Deterministic linearization (see RT-1 above).
    [[nodiscard]] std::vector<Command> to_commands() const;

    // ── Shape ──

    [[nodiscard]] uint32_t n_qubits() const { return n_qubits_; }
    [[nodiscard]] uint32_t n_cbits()  const { return n_cbits_; }
    [[nodiscard]] uint32_t n_wires()  const { return n_qubits_ + n_cbits_ + 1; }

    /// Live (non-removed) node count.
    [[nodiscard]] std::size_t n_nodes() const { return n_live_; }

    /// Total allocated node slots, including tombstones.  Valid NodeIds are
    /// [0, n_slots); check `is_removed` when iterating.
    [[nodiscard]] std::size_t n_slots() const { return nodes_.size(); }

    // Wire-id helpers.
    [[nodiscard]] uint32_t qubit_wire(uint32_t q) const { return q; }
    [[nodiscard]] uint32_t cbit_wire(uint32_t c)  const { return n_qubits_ + c; }
    [[nodiscard]] uint32_t global_wire()          const { return n_qubits_ + n_cbits_; }

    // ── Node access ──

    /// The node's command.  For a region node this is the BranchBegin marker
    /// (condition tuple in its condition_bits / condition_values); the full
    /// span lives in `region_commands`.
    [[nodiscard]] const Command& command(NodeId id) const;

    /// True if this node is a collapsed BranchBegin…BranchEnd region.
    [[nodiscard]] bool is_region(NodeId id) const;

    /// The region's raw sub-vector (markers included), verbatim.
    /// Throws std::invalid_argument if the node is not a region.
    [[nodiscard]] const std::vector<Command>& region_commands(NodeId id) const;

    [[nodiscard]] bool is_removed(NodeId id) const;

    /// Wires this node touches, deduplicated, in first-appearance order.
    [[nodiscard]] const SmallVector<uint32_t, 4>& wires(NodeId id) const;

    // ── Per-wire iteration ──

    /// First / last live node on a wire; kNone if the wire is untouched.
    [[nodiscard]] NodeId wire_front(uint32_t wire) const;
    [[nodiscard]] NodeId wire_back(uint32_t wire) const;

    /// Neighbour of `id` along `wire`; kNone at the boundary.
    /// Throws std::invalid_argument if `id` does not touch `wire`.
    [[nodiscard]] NodeId next_on_wire(NodeId id, uint32_t wire) const;
    [[nodiscard]] NodeId prev_on_wire(NodeId id, uint32_t wire) const;

    // ── Analysis ──

    /// Longest dependency path.  Every node weighs 1 except Barrier and
    /// GPhase (weight 0: synchronization / phase bookkeeping, not an
    /// operation in time).  Region nodes weigh 1.
    [[nodiscard]] std::size_t depth() const;

    /// Live nodes with no predecessor on any wire, in original-index order.
    [[nodiscard]] std::vector<NodeId> front_layer() const;

    /// ASAP moments: layer k holds the live nodes whose longest predecessor
    /// chain has k nodes.  Every node (Barrier/GPhase/regions included)
    /// occupies exactly one layer, so `layers().size() >= depth()` is
    /// possible — `depth()` weighs Barrier/GPhase at 0, layers do not.
    /// Nodes within a layer are in original-index order.
    [[nodiscard]] std::vector<std::vector<NodeId>> layers() const;

    /// Gate-name → occurrence count over the stream `to_commands()` would
    /// emit: region interiors (markers included) are counted, tombstones
    /// are not.
    [[nodiscard]] std::unordered_map<std::string, std::size_t> count_ops() const;

    // ── Mutation ──

    /// Tombstone the node and splice its per-wire links in O(wires touched).
    /// Preserves the forward-edge invariant, so `to_commands()` afterwards
    /// yields the survivors in original relative order.
    /// Throws std::invalid_argument if already removed or out of range.
    void remove_node(NodeId id);

    /// Swap the node's command for one spanning the SAME wire set (e.g. a
    /// merged rotation with a new angle) — links stay valid, no re-linking.
    /// Throws std::invalid_argument on a region node or a wire-set mismatch.
    void replace_command(NodeId id, Command cmd);

    /// Swap a region node's raw span (e.g. after optimizing its interior).
    /// The node keeps its all-wires barrier span.  Throws
    /// std::invalid_argument if the node is not a region, if the new span is
    /// not BranchBegin…BranchEnd delimited, or on out-of-range indices.
    void replace_region_commands(NodeId id, std::vector<Command> span);

private:
    struct Node {
        Command cmd;
        std::vector<Command> region;      // non-empty iff region node
        SmallVector<uint32_t, 4> wires;   // deduped, first-appearance order
        SmallVector<NodeId, 4>   prev;    // parallel to wires
        SmallVector<NodeId, 4>   next;    // parallel to wires
        uint32_t original_index = 0;
        bool removed = false;
    };

    CircuitDAG(uint32_t n_qubits, uint32_t n_cbits);

    /// Append a node and link it after the current tail of each of its wires.
    NodeId link_node(Node node);

    /// Wires a (non-region) command touches: qubits, cbit writes, condition
    /// reads, global-phase wire for GPhase.  Deduped, first-appearance order.
    [[nodiscard]] SmallVector<uint32_t, 4> compute_wires(const Command& cmd) const;

    /// Slot of `wire` within nodes_[id].wires.  Throws if absent.
    [[nodiscard]] std::size_t wire_slot(NodeId id, uint32_t wire) const;

    void check_id(NodeId id) const;
    void check_indices(const Command& cmd) const;

    uint32_t n_qubits_ = 0;
    uint32_t n_cbits_  = 0;
    std::size_t n_live_ = 0;
    std::vector<Node> nodes_;
    std::vector<NodeId> first_on_wire_;
    std::vector<NodeId> last_on_wire_;
};

}  // namespace qarpx
