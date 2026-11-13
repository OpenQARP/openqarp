#include "qarpx/compilation/perfect_layout.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <queue>

namespace qarpx {

namespace {

constexpr uint32_t kUnmapped = static_cast<uint32_t>(-1);

/// Undirected adjacency, deduplicated, with a bitset for O(1) edge tests.
struct Graph {
    uint32_t                            n = 0;
    std::vector<std::vector<uint32_t>>  adj;
    std::vector<bool>                   matrix;

    explicit Graph(uint32_t size) : n(size), adj(size), matrix(size * size, false) {}

    void add(uint32_t a, uint32_t b) {
        if (a == b || a >= n || b >= n || matrix[a * n + b]) return;
        matrix[a * n + b] = matrix[b * n + a] = true;
        adj[a].push_back(b);
        adj[b].push_back(a);
    }

    [[nodiscard]] bool connected(uint32_t a, uint32_t b) const { return matrix[a * n + b]; }
    [[nodiscard]] std::size_t degree(uint32_t q) const { return adj[q].size(); }
};

Graph interaction_graph(const std::vector<Command>& commands, uint32_t n) {
    Graph g(n);
    for (const auto& cmd : commands) {
        if (cmd.qubits.size() == 2) g.add(cmd.qubits[0], cmd.qubits[1]);
    }
    return g;
}

Graph coupling_graph(const Architecture& arch) {
    Graph g(arch.n_qubits);
    for (const auto& e : arch.edges()) g.add(e.first, e.second);
    return g;
}

/// Map the highest-degree vertices first, and prefer ones already adjacent to
/// something mapped: both prune the search tree hard, and the second keeps the
/// partial mapping connected so neighbourhood constraints bite immediately.
std::vector<uint32_t> search_order(const Graph& g) {
    std::vector<uint32_t> active;
    for (uint32_t q = 0; q < g.n; ++q) {
        if (g.degree(q) > 0) active.push_back(q);
    }
    std::sort(active.begin(), active.end(), [&](uint32_t a, uint32_t b) {
        if (g.degree(a) != g.degree(b)) return g.degree(a) > g.degree(b);
        return a < b;  // deterministic
    });

    std::vector<uint32_t> order;
    std::vector<bool> taken(g.n, false), touched(g.n, false);
    order.reserve(active.size());
    while (order.size() < active.size()) {
        uint32_t next = kUnmapped;
        for (uint32_t q : active) {
            if (taken[q]) continue;
            if (touched[q]) { next = q; break; }
            if (next == kUnmapped) next = q;
        }
        taken[next] = true;
        order.push_back(next);
        for (uint32_t nb : g.adj[next]) touched[nb] = true;
    }
    return order;
}

struct Searcher {
    const Graph&                 logical;
    const Graph&                 physical;
    const std::vector<uint32_t>& order;
    std::size_t                  budget;

    std::vector<uint32_t> l2p;
    std::vector<bool>     used;

    bool descend(std::size_t depth) {
        if (depth == order.size()) return true;
        if (budget == 0) return false;

        const uint32_t lq = order[depth];
        for (uint32_t pq = 0; pq < physical.n; ++pq) {
            if (budget == 0) return false;
            if (used[pq]) continue;
            if (physical.degree(pq) < logical.degree(lq)) continue;

            bool ok = true;
            for (uint32_t nb : logical.adj[lq]) {
                if (l2p[nb] == kUnmapped) continue;
                if (!physical.connected(pq, l2p[nb])) { ok = false; break; }
            }
            if (!ok) continue;

            --budget;
            l2p[lq] = pq;
            used[pq] = true;
            if (descend(depth + 1)) return true;
            l2p[lq] = kUnmapped;
            used[pq] = false;
        }
        return false;
    }
};

}  // namespace

std::optional<std::vector<uint32_t>> find_perfect_layout(const std::vector<Command>& commands,
                                                        const Architecture&         arch,
                                                        std::size_t                 node_budget) {
    const uint32_t n = arch.n_qubits;
    if (n == 0) return std::nullopt;

    const Graph logical = interaction_graph(commands, n);
    const Graph physical = coupling_graph(arch);

    // Cheap impossibility test before any backtracking: a logical qubit that
    // interacts with more partners than any hardware qubit has neighbours can
    // never be placed (an all-to-all circuit on a grid dies here).
    std::size_t max_physical_degree = 0;
    for (uint32_t q = 0; q < n; ++q) {
        max_physical_degree = std::max(max_physical_degree, physical.degree(q));
    }
    for (uint32_t q = 0; q < n; ++q) {
        if (logical.degree(q) > max_physical_degree) return std::nullopt;
    }

    const std::vector<uint32_t> order = search_order(logical);
    if (order.empty()) {  // no two-qubit gates at all: identity is perfect
        std::vector<uint32_t> identity(n);
        std::iota(identity.begin(), identity.end(), 0u);
        return identity;
    }

    Searcher searcher{logical, physical, order, node_budget,
                      std::vector<uint32_t>(n, kUnmapped), std::vector<bool>(n, false)};
    if (!searcher.descend(0)) return std::nullopt;

    // Idle qubits take the leftover physical wires, lowest first, so the
    // mapping is a permutation and the result stays deterministic.
    std::vector<uint32_t> spare;
    for (uint32_t pq = 0; pq < n; ++pq) {
        if (!searcher.used[pq]) spare.push_back(pq);
    }
    std::size_t next = 0;
    for (uint32_t lq = 0; lq < n; ++lq) {
        if (searcher.l2p[lq] == kUnmapped) searcher.l2p[lq] = spare[next++];
    }
    return searcher.l2p;
}

std::vector<uint32_t> greedy_weighted_layout(const std::vector<Command>& commands,
                                             const Architecture&         arch) {
    const uint32_t n = arch.n_qubits;
    std::vector<uint32_t> l2p(n, kUnmapped);
    if (n == 0) return l2p;

    // Interaction weights: how many 2q gates each logical pair shares.
    std::vector<uint32_t> weight(static_cast<std::size_t>(n) * n, 0);
    auto w = [&](uint32_t a, uint32_t b) -> uint32_t& { return weight[a * n + b]; };
    for (const auto& cmd : commands) {
        if (cmd.qubits.size() != 2) continue;
        const uint32_t a = cmd.qubits[0], b = cmd.qubits[1];
        if (a >= n || b >= n || a == b) continue;
        ++w(a, b);
        ++w(b, a);
    }

    // BFS distances from every physical qubit (small n; the router builds the
    // same matrix — recomputing here keeps this function self-contained).
    constexpr uint32_t kFar = std::numeric_limits<uint32_t>::max();
    std::vector<std::vector<uint32_t>> dist(n, std::vector<uint32_t>(n, kFar));
    for (uint32_t s = 0; s < n; ++s) {
        dist[s][s] = 0;
        std::queue<uint32_t> bfs;
        bfs.push(s);
        while (!bfs.empty()) {
            const uint32_t u = bfs.front();
            bfs.pop();
            for (auto v : arch.neighbours(u)) {
                if (dist[s][v] != kFar) continue;
                dist[s][v] = dist[s][u] + 1;
                bfs.push(v);
            }
        }
    }

    std::vector<bool> placed_l(n, false), used_p(n, false);

    // Anchor: heaviest logical pair on a max-degree-sum physical edge, so the
    // hot core of the circuit sits in the best-connected part of the device.
    uint32_t best_a = kUnmapped, best_b = kUnmapped, best_w = 0;
    for (uint32_t a = 0; a < n; ++a)
        for (uint32_t b = a + 1; b < n; ++b)
            if (w(a, b) > best_w) { best_w = w(a, b); best_a = a; best_b = b; }

    if (best_w > 0 && !arch.edges().empty()) {
        std::size_t best_edge = 0, best_deg = 0;
        for (std::size_t e = 0; e < arch.edges().size(); ++e) {
            const auto& [pa, pb] = arch.edges()[e];
            if (pa >= n || pb >= n) continue;
            const std::size_t deg = arch.neighbours(pa).size() + arch.neighbours(pb).size();
            if (deg > best_deg) { best_deg = deg; best_edge = e; }
        }
        const auto& [pa, pb] = arch.edges()[best_edge];
        l2p[best_a] = pa;
        l2p[best_b] = pb;
        placed_l[best_a] = placed_l[best_b] = true;
        used_p[pa] = used_p[pb] = true;
    }

    // Attach the rest: strongest connection to the placed set first, each on
    // the free physical qubit minimizing weighted distance to its placed
    // neighbours.  Ties break on lowest index for determinism.
    for (;;) {
        uint32_t next_l = kUnmapped;
        uint64_t next_w = 0;
        for (uint32_t l = 0; l < n; ++l) {
            if (placed_l[l]) continue;
            uint64_t attach = 0;
            for (uint32_t m = 0; m < n; ++m)
                if (placed_l[m]) attach += w(l, m);
            if (attach > next_w || (attach > 0 && next_l == kUnmapped)) {
                next_w = attach;
                next_l = l;
            }
        }
        if (next_l == kUnmapped) break;  // nothing left with a placed neighbour

        uint64_t best_cost = std::numeric_limits<uint64_t>::max();
        uint32_t best_p = kUnmapped;
        for (uint32_t p = 0; p < n; ++p) {
            if (used_p[p]) continue;
            uint64_t cost = 0;
            bool reachable = true;
            for (uint32_t m = 0; m < n; ++m) {
                if (!placed_l[m] || w(next_l, m) == 0) continue;
                if (dist[p][l2p[m]] == kFar) { reachable = false; break; }
                cost += static_cast<uint64_t>(w(next_l, m)) * dist[p][l2p[m]];
            }
            if (reachable && cost < best_cost) { best_cost = cost; best_p = p; }
        }
        if (best_p == kUnmapped) break;  // disconnected leftovers: fill below
        l2p[next_l] = best_p;
        placed_l[next_l] = true;
        used_p[best_p] = true;
    }

    // Idle / disconnected qubits fill the leftover wires in index order.
    std::vector<uint32_t> spare;
    for (uint32_t p = 0; p < n; ++p)
        if (!used_p[p]) spare.push_back(p);
    std::size_t next = 0;
    for (uint32_t l = 0; l < n; ++l)
        if (l2p[l] == kUnmapped) l2p[l] = spare[next++];
    return l2p;
}

}  // namespace qarpx
