#include <gtest/gtest.h>
#include "qarpx/qarpx.h"

#include <algorithm>
#include <random>

using namespace qarpx;

TEST(Architecture, IsConnectedUndirected) {
    Architecture a(3, {{0, 1}, {1, 2}});
    EXPECT_TRUE(a.is_connected(0, 1));
    EXPECT_TRUE(a.is_connected(1, 0));  // undirected
    EXPECT_TRUE(a.is_connected(1, 2));
    EXPECT_FALSE(a.is_connected(0, 2));
    EXPECT_TRUE(a.is_connected(0, 0));  // self
}

TEST(Architecture, NeighboursDedupedAndSorted) {
    Architecture a(4, {{0, 1}, {1, 2}, {2, 1}, {1, 3}});
    auto n = a.neighbours(1);
    ASSERT_EQ(n.size(), 3u);
    EXPECT_EQ(n[0], 0u);
    EXPECT_EQ(n[1], 2u);
    EXPECT_EQ(n[2], 3u);
}

TEST(Architecture, ShortestPathTrivial) {
    Architecture a(3, {{0, 1}, {1, 2}});
    auto path = a.shortest_path(2, 2);
    ASSERT_EQ(path.size(), 1u);
    EXPECT_EQ(path[0], 2u);
}

TEST(Architecture, ShortestPathLinearChain) {
    Architecture a(4, {{0, 1}, {1, 2}, {2, 3}});
    auto path = a.shortest_path(0, 3);
    ASSERT_EQ(path.size(), 4u);
    EXPECT_EQ(path.front(), 0u);
    EXPECT_EQ(path.back(), 3u);
}

TEST(Architecture, ShortestPathPicksShorter) {
    // 0 - 1 - 2
    //  \-----/
    Architecture a(3, {{0, 1}, {1, 2}, {0, 2}});
    auto path = a.shortest_path(0, 2);
    ASSERT_EQ(path.size(), 2u);
    EXPECT_EQ(path[0], 0u);
    EXPECT_EQ(path[1], 2u);
}

TEST(Architecture, ShortestPathDisconnected) {
    Architecture a(4, {{0, 1}, {2, 3}});
    EXPECT_TRUE(a.shortest_path(0, 3).empty());
}

TEST(Architecture, ShortestPathOutOfRange) {
    Architecture a(2, {{0, 1}});
    EXPECT_TRUE(a.shortest_path(0, 5).empty());
    EXPECT_TRUE(a.shortest_path(5, 0).empty());
}

TEST(Architecture, ConstructorRejectsOutOfRangeEdge) {
    EXPECT_THROW(Architecture(2, {{0, 5}}), std::invalid_argument);
}

TEST(Architecture, NearestNeighbour2x2Grid) {
    auto a = nearest_neighbour_architecture(2, 2);
    EXPECT_EQ(a.n_qubits, 4u);
    // Expected edges (row-major):  (0-1), (0-2), (1-3), (2-3)
    EXPECT_EQ(a.edges().size(), 4u);
    EXPECT_TRUE(a.is_connected(0, 1));
    EXPECT_TRUE(a.is_connected(0, 2));
    EXPECT_TRUE(a.is_connected(1, 3));
    EXPECT_TRUE(a.is_connected(2, 3));
    EXPECT_FALSE(a.is_connected(0, 3));  // diagonal not connected
}

TEST(Architecture, NearestNeighbourLinearChain) {
    auto a = nearest_neighbour_architecture(4, 1);
    EXPECT_EQ(a.n_qubits, 4u);
    EXPECT_EQ(a.edges().size(), 3u);
    for (uint32_t i = 0; i < 3; ++i) {
        EXPECT_TRUE(a.is_connected(i, i + 1));
    }
}

TEST(Architecture, AllToAll4Qubits) {
    auto a = all_to_all_architecture(4);
    EXPECT_EQ(a.n_qubits, 4u);
    EXPECT_EQ(a.edges().size(), 6u);  // C(4,2)
    for (uint32_t i = 0; i < 4; ++i) {
        for (uint32_t j = i + 1; j < 4; ++j) {
            EXPECT_TRUE(a.is_connected(i, j));
        }
    }
}

// ── Gateset helpers ──

// ── P2.9: the cached indices agree with a naive edge scan ──────────────────

namespace {

// The pre-P2.9 implementations, kept as the oracle: every query answered by
// scanning the stored edge list.
struct NaiveArch {
    uint32_t n;
    std::vector<std::pair<uint32_t, uint32_t>> edges;

    bool is_connected(uint32_t a, uint32_t b) const {
        if (a == b) return true;
        for (const auto& [u, v] : edges)
            if ((u == a && v == b) || (u == b && v == a)) return true;
        return false;
    }
    bool has_directed_edge(uint32_t a, uint32_t b) const {
        for (const auto& [u, v] : edges)
            if (u == a && v == b) return true;
        return false;
    }
    std::vector<uint32_t> neighbours(uint32_t q) const {
        std::vector<uint32_t> out;
        if (q >= n) return out;
        for (const auto& [u, v] : edges) {
            if (u == q) out.push_back(v);
            else if (v == q) out.push_back(u);
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }
    // BFS depth only: a shortest path is not unique, its length is.
    int64_t distance(uint32_t a, uint32_t b) const {
        if (a >= n || b >= n) return -1;
        std::vector<int64_t> dist(n, -1);
        std::vector<uint32_t> frontier{a};
        dist[a] = 0;
        while (!frontier.empty()) {
            std::vector<uint32_t> next;
            for (auto u : frontier)
                for (auto v : neighbours(u))
                    if (dist[v] < 0) { dist[v] = dist[u] + 1; next.push_back(v); }
            frontier = std::move(next);
        }
        return dist[b];
    }
};

std::vector<std::pair<uint32_t, uint32_t>> random_edges(uint32_t n, std::size_t m,
                                                        uint32_t seed) {
    // Duplicates, both orientations and self-loops on purpose.
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> pick(0, n - 1);
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    for (std::size_t i = 0; i < m; ++i) edges.emplace_back(pick(rng), pick(rng));
    return edges;
}

}  // namespace

TEST(Architecture, CachedQueriesMatchNaiveScanOnRandomGraphs) {
    for (uint32_t seed = 0; seed < 12; ++seed) {
        const uint32_t n = 4 + (seed * 5) % 61;  // 4..64
        const auto edges = random_edges(n, static_cast<std::size_t>(n) * 2, seed);
        const Architecture arch(n, edges);
        const NaiveArch naive{n, edges};
        for (uint32_t a = 0; a < n; ++a) {
            EXPECT_EQ(arch.neighbours(a), naive.neighbours(a)) << "seed " << seed;
            for (uint32_t b = 0; b < n; ++b) {
                EXPECT_EQ(arch.is_connected(a, b), naive.is_connected(a, b));
                EXPECT_EQ(arch.has_directed_edge(a, b), naive.has_directed_edge(a, b));
                const auto path = arch.shortest_path(a, b);
                const int64_t d = naive.distance(a, b);
                if (d < 0) {
                    EXPECT_TRUE(path.empty());
                } else {
                    ASSERT_EQ(static_cast<int64_t>(path.size()), d + 1);
                    EXPECT_EQ(path.front(), a);
                    EXPECT_EQ(path.back(), b);
                    for (std::size_t i = 1; i < path.size(); ++i)
                        EXPECT_TRUE(naive.is_connected(path[i - 1], path[i]));
                }
            }
        }
        // Out of range: empty neighbours, not connected, no path.
        EXPECT_TRUE(arch.neighbours(n + 3).empty());
        EXPECT_FALSE(arch.is_connected(n + 3, 0));
        EXPECT_TRUE(arch.shortest_path(0, n + 3).empty());
    }
}

TEST(Architecture, SetEdgesRebuildsIndices) {
    Architecture a(4, {{0, 1}});
    EXPECT_TRUE(a.is_connected(0, 1));
    EXPECT_TRUE(a.has_directed_edge(0, 1));
    EXPECT_FALSE(a.has_directed_edge(1, 0));

    a.set_edges({{2, 3}, {3, 1}});
    EXPECT_FALSE(a.is_connected(0, 1));
    EXPECT_FALSE(a.has_directed_edge(0, 1));
    EXPECT_TRUE(a.is_connected(1, 3));
    EXPECT_TRUE(a.has_directed_edge(3, 1));
    EXPECT_EQ(a.neighbours(3), (std::vector<uint32_t>{1, 2}));
    EXPECT_EQ(a.shortest_path(2, 1), (std::vector<uint32_t>{2, 3, 1}));

    // A rejected edge list leaves the previous one in place.
    EXPECT_THROW(a.set_edges({{0, 4}}), std::invalid_argument);
    EXPECT_EQ(a.edges().size(), 2u);
    EXPECT_TRUE(a.is_connected(2, 3));
}

TEST(GateSet, Full1qContainsExpected) {
    auto gs = full_gateset_1q();
    EXPECT_TRUE(gs.contains(GateType::Rx));
    EXPECT_TRUE(gs.contains(GateType::H));
    EXPECT_TRUE(gs.contains(GateType::Measure));
    EXPECT_FALSE(gs.contains(GateType::CX));
    EXPECT_FALSE(gs.contains(GateType::CCX));
}

TEST(GateSet, Full2qContainsExpected) {
    auto gs = full_gateset_2q();
    EXPECT_TRUE(gs.contains(GateType::CX));
    EXPECT_TRUE(gs.contains(GateType::SWAP));
    EXPECT_TRUE(gs.contains(GateType::CU));
    EXPECT_FALSE(gs.contains(GateType::H));
    EXPECT_FALSE(gs.contains(GateType::CCX));  // 3q
}

TEST(GateSet, Full1q2qUnion) {
    auto gs = full_gateset_1q_2q();
    EXPECT_TRUE(gs.contains(GateType::H));
    EXPECT_TRUE(gs.contains(GateType::CX));
    EXPECT_FALSE(gs.contains(GateType::CCX));
}
