#include "qarpx/device/architecture.h"

#include <algorithm>
#include <queue>
#include <stdexcept>

namespace qarpx {

Architecture::Architecture(uint32_t n_qubits_,
                           std::vector<std::pair<uint32_t, uint32_t>> edges_in,
                           std::string name_)
    : name(std::move(name_)), n_qubits(n_qubits_) {
    set_edges(std::move(edges_in));
}

void Architecture::set_edges(std::vector<std::pair<uint32_t, uint32_t>> edges_in) {
    for (const auto& [a, b] : edges_in) {
        if (a >= n_qubits || b >= n_qubits) {
            throw std::invalid_argument(
                "Architecture: edge endpoint out of range for n_qubits");
        }
    }
    edges_ = std::move(edges_in);
    rebuild_indices_();
}

void Architecture::rebuild_indices_() {
    adjacency_.assign(n_qubits, {});
    directed_.clear();
    directed_.reserve(edges_.size());
    for (const auto& [u, v] : edges_) {
        directed_.insert(directed_key(u, v));
        if (u == v) {
            adjacency_[u].push_back(v);
            continue;
        }
        adjacency_[u].push_back(v);
        adjacency_[v].push_back(u);
    }
    for (auto& nbrs : adjacency_) {
        std::sort(nbrs.begin(), nbrs.end());
        nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
    }
}

bool Architecture::is_connected(uint32_t a, uint32_t b) const {
    if (a == b) return true;
    return directed_.count(directed_key(a, b)) != 0 || directed_.count(directed_key(b, a)) != 0;
}

bool Architecture::has_directed_edge(uint32_t a, uint32_t b) const {
    return directed_.count(directed_key(a, b)) != 0;
}

const std::vector<uint32_t>& Architecture::neighbours(uint32_t q) const {
    static const std::vector<uint32_t> none;
    return q < adjacency_.size() ? adjacency_[q] : none;
}

std::vector<uint32_t> Architecture::shortest_path(uint32_t a, uint32_t b) const {
    if (a >= n_qubits || b >= n_qubits) return {};
    if (a == b) return {a};

    std::vector<int64_t> parent(n_qubits, -1);
    std::vector<uint8_t> visited(n_qubits, 0);
    std::queue<uint32_t> q;
    q.push(a);
    visited[a] = 1;

    while (!q.empty()) {
        uint32_t cur = q.front();
        q.pop();
        if (cur == b) {
            std::vector<uint32_t> path;
            for (int64_t node = b; node != -1; node = parent[node]) {
                path.push_back(static_cast<uint32_t>(node));
            }
            std::reverse(path.begin(), path.end());
            return path;
        }
        for (uint32_t nbr : neighbours(cur)) {
            if (!visited[nbr]) {
                visited[nbr] = 1;
                parent[nbr] = cur;
                q.push(nbr);
            }
        }
    }
    return {};
}

Architecture nearest_neighbour_architecture(uint32_t xdim, uint32_t ydim) {
    const uint32_t n = xdim * ydim;
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    auto idx = [xdim](uint32_t x, uint32_t y) { return y * xdim + x; };
    for (uint32_t y = 0; y < ydim; ++y) {
        for (uint32_t x = 0; x < xdim; ++x) {
            const uint32_t q = idx(x, y);
            if (x + 1 < xdim) edges.emplace_back(q, idx(x + 1, y));
            if (y + 1 < ydim) edges.emplace_back(q, idx(x, y + 1));
        }
    }
    return Architecture(n, std::move(edges), "nearest_neighbour");
}

Architecture all_to_all_architecture(uint32_t n_qubits) {
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    for (uint32_t a = 0; a < n_qubits; ++a) {
        for (uint32_t b = a + 1; b < n_qubits; ++b) {
            edges.emplace_back(a, b);
        }
    }
    return Architecture(n_qubits, std::move(edges), "all_to_all");
}

}  // namespace qarpx
