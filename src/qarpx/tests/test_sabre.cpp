// Tests for the SABRE router and the
// branch-region routing guard shared with the Lite router.
//
// Correctness contract (both routers): every emitted 2q gate satisfies the
// architecture (and direction), the returned initial and final
// logical_to_physical maps are valid permutations, and the routed circuit's
// statevector equals the original's under those permutations — checked from
// an asymmetric input, since |0...0> is blind to the initial placement.

#include "dag_test_helpers.h"

#include "qarpx/compilation/router.h"
#include "qarpx/core/errors.h"
#include "qarpx/device/architecture.h"
#include "qarpx/simulator/qarp_simulator.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <random>
#include <vector>

namespace qarpx::test {

namespace {

Architecture line_arch(uint32_t n) {
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    for (uint32_t i = 0; i + 1 < n; ++i) edges.emplace_back(i, i + 1);
    return Architecture(n, std::move(edges), "line");
}

RoutingOptions sabre_opts(Architecture arch, bool directedness = false) {
    RoutingOptions o;
    o.arch = std::move(arch);
    o.directedness = directedness;
    o.router = RouterKind::Sabre;
    return o;
}

bool is_permutation(const std::vector<uint32_t>& v) {
    std::vector<uint32_t> s = v;
    std::sort(s.begin(), s.end());
    for (uint32_t i = 0; i < s.size(); ++i)
        if (s[i] != i) return false;
    return true;
}

/// Every emitted 2q gate must sit on an architecture edge (and, under
/// directedness, on a correctly-oriented one for asymmetric gates).
void expect_all_satisfied(const std::vector<Command>& routed,
                          const Architecture& arch, bool directedness) {
    for (const auto& cmd : routed) {
        if (cmd.qubits.size() != 2) continue;
        EXPECT_TRUE(arch.is_connected(cmd.qubits[0], cmd.qubits[1]))
            << gate_name(cmd.gate) << " on (" << cmd.qubits[0] << ","
            << cmd.qubits[1] << ")";
        if (directedness && cmd.gate == GateType::CX) {
            EXPECT_TRUE(arch.has_directed_edge(cmd.qubits[0], cmd.qubits[1]));
        }
    }
}

/// |ψ_routed⟩ must equal |ψ_logical⟩ with qubits permuted by l2p:
/// amplitude_phys[b'] == amplitude_logical[b] where bit l of b maps to bit
/// l2p[l] of b'.
void expect_state_equivalent(const std::vector<Command>& logical,
                             const std::vector<Command>& routed,
                             const std::vector<uint32_t>& l2p,
                             uint32_t n) {
    QarpSimulator sim;
    const auto psi_l = sim.statevector(logical, static_cast<int>(n));
    const auto psi_p = sim.statevector(routed, static_cast<int>(n));
    const std::size_t dim = std::size_t{1} << n;
    for (std::size_t b = 0; b < dim; ++b) {
        std::size_t bp = 0;
        for (uint32_t l = 0; l < n; ++l)
            if (b & (std::size_t{1} << l)) bp |= (std::size_t{1} << l2p[l]);
        const auto diff = psi_l[b] - psi_p[bp];
        ASSERT_LT(std::abs(diff), 1e-10) << "amplitude mismatch at " << b;
    }
}


/// The same equivalence from a per-qubit *asymmetric* product input: the
/// input layer (distinct Ry angles) hits logical qubit l on physical wire
/// init[l], so a wrong initial map cannot pass.  |0...0> is fixed by every
/// permutation and therefore says nothing about placement.
void expect_state_equivalent_asymmetric(const std::vector<Command>& logical,
                                        const std::vector<Command>& routed,
                                        const std::vector<uint32_t>& init,
                                        const std::vector<uint32_t>& fin,
                                        uint32_t n) {
    ASSERT_EQ(init.size(), n);
    std::vector<Command> l, r;
    for (uint32_t q = 0; q < n; ++q) {
        const double theta = 0.3 + 0.37 * q;
        l.emplace_back(GateType::Ry, q, Param(theta));
        r.emplace_back(GateType::Ry, init[q], Param(theta));
    }
    l.insert(l.end(), logical.begin(), logical.end());
    r.insert(r.end(), routed.begin(), routed.end());
    expect_state_equivalent(l, r, fin, n);
}

std::size_t count_swaps(const std::vector<Command>& cmds) {
    std::size_t n = 0;
    for (const auto& c : cmds)
        if (c.gate == GateType::SWAP) ++n;
    return n;
}

/// Random unitary-only circuit generator biased toward 2q gates (routing
/// stress); logical indices over [0, n).
std::vector<Command> random_2q_heavy(uint32_t seed, uint32_t n, std::size_t len) {
    std::mt19937 rng(seed);
    auto pick = [&](uint32_t k) {
        return std::uniform_int_distribution<uint32_t>(0, k - 1)(rng);
    };
    std::vector<Command> out;
    while (out.size() < len) {
        const uint32_t a = pick(n);
        uint32_t b = pick(n);
        while (b == a) b = pick(n);
        switch (pick(4)) {
            case 0: out.emplace_back(GateType::H, a); break;
            case 1: out.emplace_back(GateType::Rz, a,
                        Param(0.1 + 0.13 * static_cast<double>(out.size()))); break;
            default: out.emplace_back(GateType::CX, a, b); break;
        }
    }
    return out;
}

}  // anonymous namespace

// ── Correctness on random circuits ──────────────────────────────────────────

TEST(SabreRouter, RandomCircuitsRouteCorrectlyOnALine) {
    const uint32_t n = 5;
    for (uint32_t seed = 0; seed < 40; ++seed) {
        const auto cmds = random_2q_heavy(seed, n, 30);
        const auto res = route(cmds, sabre_opts(line_arch(n)));
        ASSERT_TRUE(is_permutation(res.initial_logical_to_physical));
        ASSERT_TRUE(is_permutation(res.final_logical_to_physical));
        expect_all_satisfied(res.commands, line_arch(n), false);
        expect_state_equivalent_asymmetric(cmds, res.commands, res.initial_logical_to_physical,
                                           res.final_logical_to_physical, n);
    }
}

TEST(SabreRouter, DirectednessHConjugatesCx) {
    // Directed line 0→1→2: CX(1, 0) needs the flip.
    Architecture arch(3, {{0, 1}, {1, 2}}, "directed-line");
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::H, 0u);
    cmds.emplace_back(GateType::CX, 1u, 0u);

    const auto res = route(cmds, sabre_opts(arch, /*directedness=*/true));
    expect_all_satisfied(res.commands, arch, true);
    expect_state_equivalent_asymmetric(cmds, res.commands, res.initial_logical_to_physical,
                                       res.final_logical_to_physical, 3);
}

// ── The initial-mapping search win ──────────────────────────────────────────

TEST(SabreRouter, InitialMappingSearchAvoidsSwapsForDistantPair) {
    // CX(0, 7) on an 8-qubit line: Lite (identity mapping) needs 6 SWAPs;
    // Sabre's reverse-traversal search places the pair adjacently — 0 SWAPs.
    const uint32_t n = 8;
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::H, 0u);
    cmds.emplace_back(GateType::CX, 0u, 7u);
    cmds.emplace_back(GateType::Rz, 7u, Param(0.4));

    RoutingOptions lite;
    lite.router = RouterKind::Lite;
    lite.arch = line_arch(n);
    const auto lite_res = route(cmds, lite);
    EXPECT_EQ(count_swaps(lite_res.commands), 6u);

    const auto sabre_res = route(cmds, sabre_opts(line_arch(n)));
    EXPECT_EQ(count_swaps(sabre_res.commands), 0u);
    expect_state_equivalent_asymmetric(cmds, sabre_res.commands,
                                       sabre_res.initial_logical_to_physical,
                                       sabre_res.final_logical_to_physical, n);
}

TEST(SabreRouter, LookaheadBeatsLiteOnStructuredCircuit) {
    // All-pairs ZZ-style interaction layer on a line — the canonical case
    // where front-layer scoring + lookahead reduce SWAP count.
    const uint32_t n = 6;
    std::vector<Command> cmds;
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = i + 1; j < n; ++j)
            cmds.emplace_back(GateType::CX, i, j);

    RoutingOptions lite;
    lite.router = RouterKind::Lite;
    lite.arch = line_arch(n);
    const auto lite_res = route(cmds, lite);
    const auto sabre_res = route(cmds, sabre_opts(line_arch(n)));

    EXPECT_LE(count_swaps(sabre_res.commands), count_swaps(lite_res.commands));
    expect_all_satisfied(sabre_res.commands, line_arch(n), false);
    expect_state_equivalent_asymmetric(cmds, sabre_res.commands,
                                       sabre_res.initial_logical_to_physical,
                                       sabre_res.final_logical_to_physical, n);
}

// ── Explicit initial mapping is honoured (search skipped) ───────────────────

TEST(SabreRouter, ExplicitInitialMappingHonoured) {
    const uint32_t n = 4;
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::CX, 0u, 3u);

    auto opts = sabre_opts(line_arch(n));
    opts.initial_mapping = std::vector<uint32_t>{0, 2, 3, 1};  // 0 and 3 adjacent
    const auto res = route(cmds, opts);
    EXPECT_EQ(count_swaps(res.commands), 0u);
    // The pin is reported back verbatim; with no SWAPs the final map equals it.
    EXPECT_EQ(res.initial_logical_to_physical, (std::vector<uint32_t>{0, 2, 3, 1}));
    EXPECT_EQ(res.final_logical_to_physical, res.initial_logical_to_physical);
    ASSERT_EQ(res.commands.size(), 1u);
    EXPECT_EQ(res.commands[0].qubits[0], 0u);
    EXPECT_EQ(res.commands[0].qubits[1], 1u);
}

// ── Branch regions ──────────────────────────────────────────────────────────

TEST(SabreRouter, RegionInteriorTranslatedAndMappingPinned) {
    // Adjacent-in-mapping interior gates route fine; the mapping is pinned
    // across the region.
    const uint32_t n = 3;
    std::vector<Command> cmds;
    cmds.push_back(make_measure(0, 0));
    cmds.push_back(make_branch_begin({0}, {true}));
    cmds.emplace_back(GateType::X, 1u);
    cmds.emplace_back(GateType::CX, 1u, 2u);  // adjacent on the line
    cmds.push_back(make_marker(GateType::BranchEnd));
    cmds.emplace_back(GateType::H, 0u);

    const auto res = route(cmds, sabre_opts(line_arch(n)));
    ASSERT_TRUE(is_permutation(res.initial_logical_to_physical));
    ASSERT_TRUE(is_permutation(res.final_logical_to_physical));
    // Marker structure survives verbatim.
    std::size_t begins = 0, ends = 0;
    for (const auto& c : res.commands) {
        if (c.gate == GateType::BranchBegin) ++begins;
        if (c.gate == GateType::BranchEnd) ++ends;
    }
    EXPECT_EQ(begins, 1u);
    EXPECT_EQ(ends, 1u);
}

TEST(BothRouters, ThrowOnSwapsInsideBranchRegion) {
    // CX(0, 2) inside a region on a 3-line needs a SWAP — both routers must
    // refuse (unconditional SWAPs inside a conditionally-executed body
    // desynchronize the mapping when the branch is not taken).
    const uint32_t n = 3;
    std::vector<Command> cmds;
    cmds.push_back(make_measure(0, 0));
    cmds.push_back(make_branch_begin({0}, {true}));
    cmds.emplace_back(GateType::CX, 0u, 2u);
    cmds.push_back(make_marker(GateType::BranchEnd));

    // Both mappings are pinned to the identity so the fixture keeps needing a
    // SWAP: Sabre's perfect-layout pre-pass would otherwise place 0 and 2 on
    // adjacent wires, and the circuit would compile — correctly, but then the
    // test would no longer exercise the guard it exists for.
    const std::vector<uint32_t> identity{0, 1, 2};

    RoutingOptions lite;
    lite.router = RouterKind::Lite;
    lite.arch = line_arch(n);
    lite.initial_mapping = identity;
    EXPECT_THROW((void)route(cmds, lite), std::runtime_error);

    auto sabre = sabre_opts(line_arch(n));
    sabre.initial_mapping = identity;
    EXPECT_THROW((void)route(cmds, sabre), std::runtime_error);
}

// ── Shared guard rails ──────────────────────────────────────────────────────

TEST(SabreRouter, SharedThrowConditions) {
    const auto opts = sabre_opts(line_arch(4));

    // 3+-qubit gate.
    std::vector<Command> ccx;
    ccx.emplace_back(GateType::CCX, SmallVector<uint32_t, 2>{0u, 1u, 2u});
    EXPECT_THROW((void)route(ccx, opts), std::runtime_error);

    // Conditional 2q gate that needs SWAPs.  The mapping is pinned so the
    // fixture stays about the *throw contract*: left free, the perfect-layout
    // pre-pass would place this pair adjacently and the circuit would compile
    // (see ConditionalTwoQubitGateCompilesWhenPlaceable), which says nothing
    // about what happens when SWAPs really are unavoidable.
    auto cond_opts = sabre_opts(line_arch(4));
    cond_opts.initial_mapping = std::vector<uint32_t>{0, 1, 2, 3};
    std::vector<Command> cond;
    cond.push_back(make_measure(1, 0));
    cond.push_back(make_conditional(Command(GateType::CX, 0u, 3u), {0}, {true}));
    EXPECT_THROW((void)route(cond, cond_opts), std::runtime_error);

    // Out-of-range logical qubit.
    std::vector<Command> oor;
    oor.emplace_back(GateType::H, 9u);
    EXPECT_THROW((void)route(oor, opts), qarpx::capability_error);
}

namespace {

Architecture grid_arch(uint32_t rows, uint32_t cols) {
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            const uint32_t q = r * cols + c;
            if (c + 1 < cols) edges.emplace_back(q, q + 1);
            if (r + 1 < rows) edges.emplace_back(q, q + cols);
        }
    }
    return Architecture(rows * cols, std::move(edges), "grid");
}

/// Ring entangler: CX(q, q+1 mod n) — the interaction graph is an n-cycle,
/// which embeds in any grid with an even number of cells.
std::vector<Command> ring_entangler(uint32_t n) {
    std::vector<Command> cmds;
    for (uint32_t q = 0; q < n; ++q) cmds.emplace_back(GateType::CX, q, (q + 1) % n);
    return cmds;
}

std::vector<Command> relabel(const std::vector<Command>& cmds,
                             const std::vector<uint32_t>& perm) {
    std::vector<Command> out;
    out.reserve(cmds.size());
    for (const auto& c : cmds) {
        Command copy = c;
        for (auto& q : copy.qubits) q = perm[q];
        out.push_back(std::move(copy));
    }
    return out;
}

}  // namespace

// A ring that fits the device exactly must route with no SWAPs at all.  The
// reverse-traversal mapping search alone could not find these: it refines the
// identity locally, so on 3x6 and 4x5 it never reached the zero-SWAP
// embedding that exists.
TEST(SabreRouter, RingOnGridNeedsNoSwaps) {
    for (const auto& dims : {std::pair<uint32_t, uint32_t>{2, 4},
                             {3, 4},
                             {2, 7},
                             {4, 4},
                             {3, 6},
                             {4, 5}}) {
        const uint32_t n = dims.first * dims.second;
        const auto res = route(ring_entangler(n), sabre_opts(grid_arch(dims.first, dims.second)));
        EXPECT_EQ(count_swaps(res.commands), 0u)
            << "ring on " << dims.first << "x" << dims.second << " grid";
    }
}

// The property that would have caught the bug: renaming logical qubits is not
// a change to the problem, so it must not change the routing cost.  A local
// search seeded from the identity fails this — the same 12-qubit ring on a
// 3x4 grid moved between 0 and 10 SWAPs under relabelling alone.
TEST(SabreRouter, RoutingCostIsInvariantUnderRelabelling) {
    const uint32_t n = 12;
    const auto cmds = ring_entangler(n);
    const auto arch = grid_arch(3, 4);
    const std::size_t baseline = count_swaps(route(cmds, sabre_opts(arch)).commands);

    std::mt19937 rng(20260824);
    std::vector<uint32_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0u);
    for (int trial = 0; trial < 16; ++trial) {
        std::shuffle(perm.begin(), perm.end(), rng);
        const auto res = route(relabel(cmds, perm), sabre_opts(arch));
        EXPECT_EQ(count_swaps(res.commands), baseline) << "relabelling trial " << trial;
    }
}

// Behaviour change worth recording: a conditional 2q gate used to be refused
// whenever the *identity* mapping left its qubits apart, and the error told
// the caller to "place the interacting qubits adjacently in the input (or via
// initial_mapping)".  The pre-pass now does exactly that on its own, so the
// circuit compiles — with the gate on a hardware edge and no SWAPs, which is
// what made it unsafe inside a branch region in the first place.
TEST(SabreRouter, ConditionalTwoQubitGateCompilesWhenPlaceable) {
    std::vector<Command> cmds;
    cmds.push_back(make_measure(1, 0));
    cmds.push_back(make_conditional(Command(GateType::CX, 0u, 3u), {0}, {true}));

    const auto arch = line_arch(4);
    const auto res = route(cmds, sabre_opts(arch));
    EXPECT_EQ(count_swaps(res.commands), 0u);
    for (const auto& c : res.commands) {
        if (c.qubits.size() == 2) {
            EXPECT_TRUE(arch.is_connected(c.qubits[0], c.qubits[1]));
        }
    }
}

// The perfect-layout pre-pass must not fire when no embedding exists: an
// all-to-all circuit cannot fit a grid, and the search has to give up cheaply
// and leave the heuristic router to do its job.
TEST(SabreRouter, AllToAllStillRoutesOnGrid) {
    const uint32_t n = 9;
    std::vector<Command> cmds;
    for (uint32_t a = 0; a < n; ++a)
        for (uint32_t b = a + 1; b < n; ++b) cmds.emplace_back(GateType::CX, a, b);

    const auto res = route(cmds, sabre_opts(grid_arch(3, 3)));
    EXPECT_GT(count_swaps(res.commands), 0u);  // genuinely needs routing
    EXPECT_EQ(res.final_logical_to_physical.size(), n);
    std::vector<uint32_t> seen = res.final_logical_to_physical;
    std::sort(seen.begin(), seen.end());
    for (uint32_t q = 0; q < n; ++q) EXPECT_EQ(seen[q], q);  // still a permutation
}

// Portfolio trials must preserve the correctness contract on grids exactly as
// the single shot did on lines: full statevector equivalence under the
// returned permutation, every gate on a coupling edge.
TEST(SabreRouter, RandomCircuitsRouteCorrectlyOnAGrid) {
    const uint32_t n = 6;
    const auto arch = grid_arch(2, 3);
    for (uint32_t seed = 0; seed < 40; ++seed) {
        const auto cmds = random_2q_heavy(seed, n, 30);
        const auto res = route(cmds, sabre_opts(arch));
        ASSERT_TRUE(is_permutation(res.initial_logical_to_physical));
        ASSERT_TRUE(is_permutation(res.final_logical_to_physical));
        expect_all_satisfied(res.commands, arch, false);
        expect_state_equivalent_asymmetric(cmds, res.commands, res.initial_logical_to_physical,
                                           res.final_logical_to_physical, n);
    }
}

namespace {

/// The compilation benchmark's fixture LCG (benchmarks/compilation/inputs.py),
/// ported bit-exactly so the quality bounds below track the published rows.
struct FixtureLcg {
    uint64_t s;
    uint64_t next() {
        s = 6364136223846793005ull * s + 1442695040888963407ull;
        return s >> 11;
    }
    void shuffle(std::vector<uint32_t>& v) {
        for (std::size_t i = v.size() - 1; i > 0; --i)
            std::swap(v[i], v[next() % (i + 1)]);
    }
};
constexpr uint64_t kFixtureSeed = 20260824;

std::vector<Command> trotter_fixture(uint32_t n) {
    std::vector<Command> ops;
    for (uint32_t a = 0; a < n; ++a)
        for (uint32_t b = a + 1; b < n; ++b) {
            ops.emplace_back(GateType::CX, a, b);
            ops.emplace_back(GateType::Rz, b);
            ops.emplace_back(GateType::CX, a, b);
        }
    for (uint32_t q = 0; q < n; ++q) ops.emplace_back(GateType::Rx, q);
    return ops;
}

std::vector<Command> qv_fixture(uint32_t n) {
    FixtureLcg rng{kFixtureSeed + 13ull * n};
    std::vector<Command> ops;
    for (int layer = 0; layer < 4; ++layer) {
        std::vector<uint32_t> order(n);
        std::iota(order.begin(), order.end(), 0u);
        rng.shuffle(order);
        for (uint32_t k = 0; k + 1 < n; k += 2) {
            const uint32_t a = order[k], b = order[k + 1];
            rng.next();  // the Python fixture draws two Ry angles here
            rng.next();
            ops.emplace_back(GateType::Ry, a);
            ops.emplace_back(GateType::Ry, b);
            ops.emplace_back(GateType::CX, a, b);
            ops.emplace_back(GateType::Rz, b);
            rng.next();  // rz angle
            ops.emplace_back(GateType::CX, a, b);
        }
    }
    return ops;
}

}  // namespace

// Quality regression guard for the portfolio search.  Bounds are the values
// measured 2026-08-24 (trials=32, refine rounds=3) plus two SWAPs of slack;
// the pre-portfolio single shot sat far above every one of them
// (trotter/line/14: 123, trotter/grid/14: 65, qv/grid/14: 17, qv/line/12: 26),
// so a regression back toward single-shot quality fails loudly while small
// legitimate heuristic adjustments do not.
TEST(SabreRouter, PortfolioQualityBounds) {
    struct Case {
        std::vector<Command> ops;
        Architecture         arch;
        std::size_t          max_swaps;
    };
    const Case cases[] = {
        {trotter_fixture(14), line_arch(14), 86},  // measured 84
        {trotter_fixture(14), grid_arch(2, 7), 46},  // measured 44
        {qv_fixture(14), grid_arch(2, 7), 14},  // measured 12
        {qv_fixture(12), line_arch(12), 25},  // measured 23
    };
    for (const auto& c : cases) {
        const auto res = route(c.ops, sabre_opts(c.arch));
        expect_all_satisfied(res.commands, c.arch, false);
        EXPECT_LE(count_swaps(res.commands), c.max_swaps);
    }
}

TEST(SabreRouter, Deterministic) {
    const uint32_t n = 5;
    const auto cmds = random_2q_heavy(123, n, 40);
    const auto a = route(cmds, sabre_opts(line_arch(n)));
    const auto b = route(cmds, sabre_opts(line_arch(n)));
    EXPECT_TRUE(streams_equal(a.commands, b.commands));
    EXPECT_EQ(a.initial_logical_to_physical, b.initial_logical_to_physical);
    EXPECT_EQ(a.final_logical_to_physical, b.final_logical_to_physical);
}

}  // namespace qarpx::test
