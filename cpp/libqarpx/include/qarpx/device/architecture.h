#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace qarpx {

/// Coupling-map description for a quantum device.
///
/// Owns an undirected edge list over physical qubit indices and answers
/// connectivity / shortest-path queries used by the router.  Directedness
/// (one-way 2q gate edges) is a *Device*-level flag, not an architecture
/// property — an architecture says "which pairs of qubits can interact",
/// and the router decides at compile time whether the recorded direction
/// satisfies the device constraint or whether an H-conjugation is needed.
struct Architecture {
    std::string                                    name;
    uint32_t                                       n_qubits = 0;

    Architecture() = default;
    Architecture(uint32_t n_qubits,
                 std::vector<std::pair<uint32_t, uint32_t>> edges,
                 std::string name = "");

    /// The edge list as stored (order and orientation preserved — the
    /// router's SWAP tie-breaks and `has_directed_edge` depend on both).
    [[nodiscard]] const std::vector<std::pair<uint32_t, uint32_t>>& edges() const {
        return edges_;
    }

    /// Replace the edge list; every endpoint must be `< n_qubits`
    /// (std::invalid_argument).  Rebuilds the adjacency and directed-edge
    /// indices the queries below read.
    void set_edges(std::vector<std::pair<uint32_t, uint32_t>> edges);

    /// Undirected adjacency: `a == b`, or an edge in either orientation.  O(1).
    [[nodiscard]] bool is_connected(uint32_t a, uint32_t b) const;

    /// Order-sensitive edge lookup.  Returns true iff an edge was stored
    /// with `first == a && second == b`.  Used by the router under
    /// `Device::directedness=true` to decide whether a 2-qubit gate's
    /// recorded orientation is allowed on the hardware (or needs an
    /// H-conjugation flip for CX, or routing to another oriented edge).
    /// Under undirected use, prefer `is_connected`.  O(1).
    [[nodiscard]] bool has_directed_edge(uint32_t a, uint32_t b) const;

    /// Sorted, duplicate-free neighbours of `q`; empty when out of range.
    [[nodiscard]] const std::vector<uint32_t>& neighbours(uint32_t q) const;

    /// BFS shortest path from `a` to `b` (inclusive).  Returns an empty
    /// vector if the graph contains no path (e.g. disconnected components
    /// or out-of-range index).  For `a == b` returns `{a}`.
    [[nodiscard]] std::vector<uint32_t> shortest_path(uint32_t a, uint32_t b) const;

private:
    static uint64_t directed_key(uint32_t a, uint32_t b) {
        return (static_cast<uint64_t>(a) << 32) | b;
    }
    void rebuild_indices_();

    std::vector<std::pair<uint32_t, uint32_t>>     edges_;
    // Per-qubit sorted neighbour lists and the set of stored orientations;
    // both derived from edges_ and refreshed by set_edges().
    std::vector<std::vector<uint32_t>>             adjacency_;
    std::unordered_set<uint64_t>                   directed_;
};

/// Nearest-neighbour grid connectivity (undirected).
Architecture nearest_neighbour_architecture(uint32_t xdim, uint32_t ydim);

/// All-to-all connectivity (undirected).
Architecture all_to_all_architecture(uint32_t n_qubits);

}  // namespace qarpx
