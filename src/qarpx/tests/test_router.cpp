#include <gtest/gtest.h>
#include "qarpx/qarpx.h"
#include "qarpx/compilation/compile_for_device.h"
#include "qarpx/core/errors.h"

#include <algorithm>

#include "reference_unitary.h"

using namespace qarpx;

namespace {

Command make_cmd(GateType g, std::vector<uint32_t> qubits = {}) {
    Command c;
    c.gate = g;
    for (auto q : qubits) c.qubits.push_back(q);
    return c;
}

int count_gates(const std::vector<Command>& cmds, GateType g) {
    return static_cast<int>(std::count_if(cmds.begin(), cmds.end(),
        [g](const Command& c) { return c.gate == g; }));
}

}  // namespace

// ── Architecture extension: directed-edge lookup ────────────────────────────

TEST(Architecture, DirectedEdgeRespectsOrder) {
    Architecture a(3, {{0, 1}, {1, 2}});
    EXPECT_TRUE(a.has_directed_edge(0, 1));
    EXPECT_FALSE(a.has_directed_edge(1, 0));
    EXPECT_TRUE(a.has_directed_edge(1, 2));
    EXPECT_FALSE(a.has_directed_edge(2, 1));
    // Bidirectional sanity:
    EXPECT_TRUE(a.is_connected(1, 0));
    EXPECT_TRUE(a.is_connected(0, 1));
}

// ── Routing: identity case when no SWAPs needed ─────────────────────────────

TEST(Router, NoSwapsNeededOnConnectedPair) {
    // 0 — 1 — 2 chain.  CX(0,1) and CX(1,2) are both adjacent.
    RoutingOptions opts;
    opts.router = RouterKind::Lite;  // this file tests the Lite implementation
    opts.arch = Architecture(3, {{0, 1}, {1, 2}});

    std::vector<Command> cmds = {
        make_cmd(GateType::CX, {0, 1}),
        make_cmd(GateType::CX, {1, 2}),
    };
    auto result = route(cmds, opts);

    EXPECT_EQ(result.commands.size(), 2u);
    EXPECT_EQ(count_gates(result.commands, GateType::SWAP), 0);
    EXPECT_EQ(count_gates(result.commands, GateType::CX), 2);
    // Identity mapping preserved, before and after.
    EXPECT_EQ(result.initial_logical_to_physical, (std::vector<uint32_t>{0, 1, 2}));
    EXPECT_EQ(result.final_logical_to_physical, (std::vector<uint32_t>{0, 1, 2}));
}

TEST(Router, OneQubitGatesTranslated) {
    RoutingOptions opts;
    opts.router = RouterKind::Lite;  // this file tests the Lite implementation
    opts.arch = Architecture(3, {{0, 1}});
    opts.initial_mapping = std::vector<uint32_t>{2, 0, 1};

    auto result = route({make_cmd(GateType::H, {0}),
                         make_cmd(GateType::X, {2})}, opts);
    ASSERT_EQ(result.commands.size(), 2u);
    EXPECT_EQ(result.initial_logical_to_physical, (std::vector<uint32_t>{2, 0, 1}));
    EXPECT_EQ(result.commands[0].qubits[0], 2u);  // logical 0 → physical 2
    EXPECT_EQ(result.commands[1].qubits[0], 1u);  // logical 2 → physical 1
}

TEST(Router, SwapInsertedOnDisconnectedPair) {
    // 0 — 1 — 2 chain.  CX(0, 2) is not adjacent → router inserts SWAPs.
    RoutingOptions opts;
    opts.router = RouterKind::Lite;  // this file tests the Lite implementation
    opts.arch = Architecture(3, {{0, 1}, {1, 2}});

    std::vector<Command> cmds = { make_cmd(GateType::CX, {0, 2}) };
    auto result = route(cmds, opts);

    EXPECT_GT(count_gates(result.commands, GateType::SWAP), 0);
    EXPECT_EQ(count_gates(result.commands, GateType::CX), 1);
}

// ── Directedness: H-conjugation flips a wrong-direction CX ─────────────────

TEST(Router, DirectednessFlipsCxViaHConjugation) {
    // 0 -> 1 directed edge only.  CX(1, 0) requires H-conjugation.
    RoutingOptions opts;
    opts.router = RouterKind::Lite;  // this file tests the Lite implementation
    opts.arch = Architecture(2, {{0, 1}});
    opts.directedness = true;

    std::vector<Command> cmds = { make_cmd(GateType::CX, {1, 0}) };
    auto result = route(cmds, opts);

    // H(1) H(0) CX(0,1) H(1) H(0)
    EXPECT_EQ(result.commands.size(), 5u);
    EXPECT_EQ(count_gates(result.commands, GateType::H),  4);
    EXPECT_EQ(count_gates(result.commands, GateType::CX), 1);
    EXPECT_EQ(count_gates(result.commands, GateType::SWAP), 0);

    // Inner CX runs in the architecture's recorded direction.
    const Command* cx = nullptr;
    for (const auto& c : result.commands)
        if (c.gate == GateType::CX) { cx = &c; break; }
    ASSERT_NE(cx, nullptr);
    EXPECT_EQ(cx->qubits[0], 0u);
    EXPECT_EQ(cx->qubits[1], 1u);
}

TEST(Router, DirectednessHonouredWhenEdgeMatches) {
    // 0 -> 1 directed.  CX(0, 1) just fires as-is.
    RoutingOptions opts;
    opts.router = RouterKind::Lite;  // this file tests the Lite implementation
    opts.arch = Architecture(2, {{0, 1}});
    opts.directedness = true;

    auto result = route({make_cmd(GateType::CX, {0, 1})}, opts);
    EXPECT_EQ(result.commands.size(), 1u);
    EXPECT_EQ(result.commands[0].gate, GateType::CX);
    EXPECT_EQ(result.commands[0].qubits[0], 0u);
    EXPECT_EQ(result.commands[0].qubits[1], 1u);
}

TEST(Router, DirectednessNonCxAsymmetricThrows) {
    RoutingOptions opts;
    opts.router = RouterKind::Lite;  // this file tests the Lite implementation
    opts.arch = Architecture(2, {{0, 1}});
    opts.directedness = true;
    // CY in the wrong orientation — router refuses to H-conjugate non-CX.
    EXPECT_THROW((void)route({make_cmd(GateType::CY, {1, 0})}, opts),
                 capability_error);
}

TEST(Router, DirectednessEcrChCsxAreAsymmetric) {
    // ECR/CH/CSX/CSXdg change under qubit exchange; they used to be treated
    // as direction-free and slipped through reversed (P1.18).
    RoutingOptions opts;
    opts.router = RouterKind::Lite;
    opts.arch = Architecture(2, {{0, 1}});
    opts.directedness = true;
    for (GateType g : {GateType::ECR, GateType::CH, GateType::CSX, GateType::CSXdg}) {
        EXPECT_THROW((void)route({make_cmd(g, {1, 0})}, opts), capability_error)
            << gate_name(g);
        auto ok = route({make_cmd(g, {0, 1})}, opts);
        EXPECT_EQ(ok.commands.size(), 1u) << gate_name(g);
    }
}

TEST(AssertDirections, FlagsReversedAsymmetricOnly) {
    Architecture arch(2, {{0, 1}});
    EXPECT_NO_THROW(assert_directions({make_cmd(GateType::CX, {0, 1})}, arch));
    EXPECT_NO_THROW(assert_directions({make_cmd(GateType::CZ, {1, 0})}, arch));   // symmetric
    EXPECT_NO_THROW(assert_directions({make_cmd(GateType::SWAP, {1, 0})}, arch)); // symmetric
    for (GateType g : {GateType::CX, GateType::CY, GateType::CRz, GateType::CP,
                       GateType::ECR, GateType::CH, GateType::CSX, GateType::CSXdg}) {
        EXPECT_THROW(assert_directions({make_cmd(g, {1, 0})}, arch), capability_error)
            << gate_name(g);
    }
    // No edge at all is also a violation, not a pass.
    EXPECT_THROW(assert_directions({make_cmd(GateType::CX, {0, 1})},
                                   Architecture(2, {})), capability_error);
}

TEST(CompileForDevice, DirectedSwapLowersOnStoredEdge) {
    // 0 -> 1 -> 2 directed line, CX-only 2q target: CX(0, 2) needs a SWAP,
    // which the generic rule would lower with a reversed middle CX.  Every
    // CX in the output must sit on a stored edge.
    Architecture arch(3, {{0, 1}, {1, 2}});
    GateSet target;
    target.name = "cx_only";
    target.allowed = {GateType::H, GateType::Rz, GateType::Rx, GateType::Ry, GateType::X,
                      GateType::CX, GateType::GPhase, GateType::Measure, GateType::Barrier};
    Device dev(3, arch, std::nullopt, target, /*directedness=*/true);
    const std::vector<Command> logical = {make_cmd(GateType::CX, {0, 2})};
    auto out = compile_for_device(logical, 3, dev, RouterKind::Lite);
    EXPECT_NO_THROW(assert_directions(out.commands, arch));
    EXPECT_GT(count_gates(out.commands, GateType::CX), 1);
    EXPECT_EQ(count_gates(out.commands, GateType::SWAP), 0);

    // Unitary oracle (reference_unitary.h, no csim): unpinned Lite places
    // nothing, so the initial map is the identity and the physical unitary is
    // P·U_logical with P relocating logical bit l to bit final_l2p[l].
    // A dropped H pair in the lowered SWAP would fail this, not the counts.
    EXPECT_EQ(out.initial_logical_to_physical, (std::vector<uint32_t>{0, 1, 2}));
    const std::size_t dim = 8;
    Eigen::MatrixXcd perm = Eigen::MatrixXcd::Zero(dim, dim);
    for (std::size_t j = 0; j < dim; ++j) {
        std::size_t image = 0;
        for (uint32_t l = 0; l < 3; ++l)
            image |= ((j >> l) & 1u) << out.final_logical_to_physical[l];
        perm(static_cast<Eigen::Index>(image), static_cast<Eigen::Index>(j)) = 1.0;
    }
    const Eigen::MatrixXcd expected = perm * qarpx::test::reference_unitary(logical, 3);
    const Eigen::MatrixXcd actual = qarpx::test::reference_unitary(out.commands, 3);
    EXPECT_LT((actual - expected).norm(), 1e-10);
}

TEST(Router, SymmetricTwoQubitGateIgnoresDirection) {
    // CZ is symmetric; directedness shouldn't matter.
    RoutingOptions opts;
    opts.router = RouterKind::Lite;  // this file tests the Lite implementation
    opts.arch = Architecture(2, {{0, 1}});
    opts.directedness = true;

    auto result = route({make_cmd(GateType::CZ, {1, 0})}, opts);
    EXPECT_EQ(result.commands.size(), 1u);
    EXPECT_EQ(result.commands[0].gate, GateType::CZ);
}

// ── Error paths ─────────────────────────────────────────────────────────────

TEST(Router, ThrowsOnOutOfRangeQubit) {
    RoutingOptions opts;
    opts.router = RouterKind::Lite;  // this file tests the Lite implementation
    opts.arch = Architecture(2, {{0, 1}});
    EXPECT_THROW((void)route({make_cmd(GateType::H, {5})}, opts), capability_error);
}

TEST(Router, ThrowsOnThreeQubitGate) {
    RoutingOptions opts;
    opts.router = RouterKind::Lite;  // this file tests the Lite implementation
    opts.arch = all_to_all_architecture(3);
    EXPECT_THROW((void)route({make_cmd(GateType::CCX, {0, 1, 2})}, opts),
                 std::runtime_error);
}

TEST(Router, ThrowsOnDisconnectedComponents) {
    // Two disconnected single-edge components.  CX(0, 2) has no route.
    RoutingOptions opts;
    opts.router = RouterKind::Lite;  // this file tests the Lite implementation
    opts.arch = Architecture(4, {{0, 1}, {2, 3}});
    EXPECT_THROW((void)route({make_cmd(GateType::CX, {0, 2})}, opts),
                 std::runtime_error);
}

TEST(Router, ThrowsOnConditionalGateNeedingSwaps) {
    RoutingOptions opts;
    opts.router = RouterKind::Lite;  // this file tests the Lite implementation
    opts.arch = Architecture(3, {{0, 1}, {1, 2}});  // 0-1-2 chain

    Command conditional_cx = make_cmd(GateType::CX, {0, 2});
    conditional_cx.condition_bits.push_back(0);
    conditional_cx.condition_values.push_back(true);

    EXPECT_THROW((void)route({conditional_cx}, opts), std::runtime_error);
}

TEST(Router, BadInitialMappingThrows) {
    RoutingOptions opts;
    opts.router = RouterKind::Lite;  // this file tests the Lite implementation
    opts.arch = Architecture(2, {{0, 1}});
    opts.initial_mapping = std::vector<uint32_t>{0, 0};  // not a permutation
    EXPECT_THROW((void)route({make_cmd(GateType::H, {0})}, opts),
                 capability_error);
}

// ── Reindex SamplingResult ──────────────────────────────────────────────────

TEST(RouterReindex, IdentityMappingIsIdentity) {
    SamplingResult sr;
    sr.n_qubits = 3;
    sr.n_shots = 100;
    sr.counts[0b011] = 60;
    sr.counts[0b101] = 40;
    auto out = reindex_sampling_result(sr, {0, 1, 2});
    EXPECT_EQ(out.counts.at(0b011), 60);
    EXPECT_EQ(out.counts.at(0b101), 40);
}

TEST(RouterReindex, PermutedMappingRelabelsBits) {
    // final_logical_to_physical = {2, 0, 1} →
    //   logical bit 0 = physical bit 2
    //   logical bit 1 = physical bit 0
    //   logical bit 2 = physical bit 1
    SamplingResult phys;
    phys.n_qubits = 3;
    phys.n_shots = 100;
    // physical 0b001  (qubit 0 = 1, others 0)
    //   → logical bit 1 = 1, others 0  → 0b010
    phys.counts[0b001] = 100;

    auto out = reindex_sampling_result(phys, {2, 0, 1});
    EXPECT_EQ(out.counts.at(0b010), 100);
    EXPECT_EQ(out.counts.size(), 1u);
}


/// Statevector equivalence from a per-qubit asymmetric product input (distinct
/// Ry angles), applied to logical qubit l on physical wire init[l]: a wrong
/// initial map cannot pass, whereas |0...0> is fixed by every permutation.
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
    QarpSimulator sim;
    const auto psi_l = sim.statevector(l, static_cast<int>(n));
    const auto psi_p = sim.statevector(r, static_cast<int>(n));
    const std::size_t dim = std::size_t{1} << n;
    for (std::size_t b = 0; b < dim; ++b) {
        std::size_t bp = 0;
        for (uint32_t q = 0; q < n; ++q)
            if (b & (std::size_t{1} << q)) bp |= (std::size_t{1} << fin[q]);
        ASSERT_LT(std::abs(psi_l[b] - psi_p[bp]), 1e-10) << "amplitude mismatch at " << b;
    }
}

TEST(Router, PinnedPlacementIsReportedAndFinalMapMovesWithSwaps) {
    // Pin logical 0 to the far end of the line so CX(0,2) must ferry across.
    RoutingOptions opts;
    opts.router = RouterKind::Lite;
    opts.arch = Architecture(3, {{0, 1}, {1, 2}});
    opts.initial_mapping = std::vector<uint32_t>{2, 1, 0};
    std::vector<Command> cmds = {make_cmd(GateType::H, {0}), make_cmd(GateType::CX, {0, 2})};

    auto routed = route(cmds, opts);
    ASSERT_GT(count_gates(routed.commands, GateType::SWAP), 0);
    EXPECT_EQ(routed.initial_logical_to_physical, (std::vector<uint32_t>{2, 1, 0}));
    EXPECT_NE(routed.final_logical_to_physical, routed.initial_logical_to_physical);
    expect_state_equivalent_asymmetric(cmds, routed.commands, routed.initial_logical_to_physical,
                                       routed.final_logical_to_physical, 3);
}

TEST(Router, UnpinnedLiteReportsIdentityPlacement) {
    RoutingOptions opts;
    opts.router = RouterKind::Lite;
    opts.arch = Architecture(3, {{0, 1}, {1, 2}});
    auto routed = route({make_cmd(GateType::H, {0}), make_cmd(GateType::CX, {0, 2})}, opts);
    EXPECT_EQ(routed.initial_logical_to_physical, (std::vector<uint32_t>{0, 1, 2}));
}

// ── End-to-end via QarpSimulator: routed CX reproduces the unrouted result ─

TEST(Router, RoutedCxMatchesUnroutedDistribution) {
    // Unrouted: H on q0, then CX(q0, q2) on an all-to-all simulator.
    // Routed:   linear chain 0-1-2, router inserts SWAPs.  After reindexing,
    //          the resulting distribution must match the unrouted one.
    QarpSimulator sim;

    std::vector<Command> base_cmds = {
        make_cmd(GateType::H,  {0}),
        make_cmd(GateType::CX, {0, 2}),
    };

    RoutingOptions opts;
    opts.router = RouterKind::Lite;  // this file tests the Lite implementation
    opts.arch = Architecture(3, {{0, 1}, {1, 2}});
    auto routed = route(base_cmds, opts);
    ASSERT_GT(count_gates(routed.commands, GateType::SWAP), 0);

    auto unrouted_sr = sim.run(base_cmds,        3, 8000, 7u);
    auto routed_phys = sim.run(routed.commands,  3, 8000, 7u);
    auto routed_log  = reindex_sampling_result(routed_phys, routed.final_logical_to_physical);

    // The unrouted produces a Bell-like {000, 101} (modulo qubit order)
    // distribution.  After reindexing, the routed result should match.
    for (const auto& [outcome, count] : unrouted_sr.counts) {
        const int routed_count = routed_log.counts.count(outcome)
                                   ? routed_log.counts.at(outcome) : 0;
        EXPECT_NEAR(static_cast<double>(routed_count) / 8000.0,
                    static_cast<double>(count) / 8000.0,
                    0.05);
    }
}
