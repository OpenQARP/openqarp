#include <gtest/gtest.h>
#include "qarpx/qarpx.h"

#include <algorithm>

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

// ── Device ──────────────────────────────────────────────────────────────────

TEST(Device, DefaultIsNoArchNoNoiseNoGateset) {
    Device d(4);
    EXPECT_EQ(d.n_qubits, 4u);
    EXPECT_FALSE(d.architecture.has_value());
    EXPECT_FALSE(d.noise_model.has_value());
    EXPECT_FALSE(d.gate_set.has_value());
    EXPECT_FALSE(d.directedness);
}

TEST(Device, CheckFitsAcceptsBelowLimitAndThrowsAbove) {
    Device d(3);
    EXPECT_NO_THROW(d.check_fits(0));
    EXPECT_NO_THROW(d.check_fits(3));
    EXPECT_THROW(d.check_fits(4), std::runtime_error);
}

TEST(Device, ArchitectureQubitMismatchThrows) {
    // Routing would produce a final_logical_to_physical of size 4, but the simulator
    // would run at width 10 — undersized permutation silently mis-reindexes
    // sampling results.  Reject at construction.
    Architecture nn22(4, {{0, 1}, {1, 2}, {2, 3}}, "chain4");
    EXPECT_THROW(Device(10, nn22, std::nullopt, std::nullopt, false),
                 std::runtime_error);
}

TEST(Device, ArchitectureQubitMatchAccepted) {
    Architecture chain4(4, {{0, 1}, {1, 2}, {2, 3}}, "chain4");
    EXPECT_NO_THROW(Device(4, chain4, std::nullopt, std::nullopt, false));
}

// ── compile_for_device: pass-through cases ──────────────────────────────────

TEST(CompileForDevice, NoArchNoGatesetIsIdentity) {
    std::vector<Command> cmds = {
        make_cmd(GateType::H,  {0}),
        make_cmd(GateType::CX, {0, 2}),
    };
    Device d(3);
    auto out = compile_for_device(cmds, 3, d);
    EXPECT_EQ(out.commands.size(), 2u);
    EXPECT_EQ(out.commands[1].gate, GateType::CX);
    EXPECT_EQ(out.initial_logical_to_physical, (std::vector<uint32_t>{0, 1, 2}));
    EXPECT_EQ(out.final_logical_to_physical, (std::vector<uint32_t>{0, 1, 2}));
}

TEST(CompileForDevice, CheckFitsRaisesWhenCircuitTooBig) {
    Device d(2);
    auto cmds = std::vector<Command>{ make_cmd(GateType::H, {0}) };
    EXPECT_THROW((void)compile_for_device(cmds, 3, d), std::runtime_error);
}

// ── compile_for_device: rebase only ────────────────────────────────────────

TEST(CompileForDevice, RebaseRunsWhenGatesetSet) {
    // U gate is not in the qulacs gateset → rebases through to Rz·Ry·Rz path.
    std::vector<Command> cmds = { make_cmd(GateType::U, {0}) };
    // Build a U with three params so the transpiler can decompose it.
    cmds[0].params.push_back(Param(0.3));
    cmds[0].params.push_back(Param(0.7));
    cmds[0].params.push_back(Param(0.1));

    Device d(1);
    d.gate_set = qulacs_gateset();

    auto out = compile_for_device(cmds, 1, d);
    // After rebase the output contains no U gate.
    EXPECT_EQ(count_gates(out.commands, GateType::U), 0);
    EXPECT_GT(out.commands.size(), 0u);
}

// ── compile_for_device: routing only ────────────────────────────────────────

TEST(CompileForDevice, RoutingOnlyInsertsSwapsOnDisconnectedPair) {
    std::vector<Command> cmds = {
        make_cmd(GateType::H,  {0}),
        make_cmd(GateType::CX, {0, 2}),
    };
    Device d(3);
    d.architecture = Architecture(3, {{0, 1}, {1, 2}}, "chain");

    // Pin Lite: this test observes the SWAP-insertion mechanic itself, which
    // Sabre's initial-mapping search legitimately avoids on this fixture.
    auto out = compile_for_device(cmds, 3, d, RouterKind::Lite);
    EXPECT_GT(count_gates(out.commands, GateType::SWAP), 0);
    EXPECT_EQ(count_gates(out.commands, GateType::CX),   1);

    // The default (Sabre) must satisfy the architecture with at most as
    // many SWAPs — here, none: the mapping search places the pair adjacent.
    auto sabre_out = compile_for_device(cmds, 3, d);
    EXPECT_EQ(count_gates(sabre_out.commands, GateType::SWAP), 0);
}

// ── compile_for_device: rebase + route + re-rebase ─────────────────────────

TEST(CompileForDevice, FullPipelineDecomposesRouterIntroducedSwaps) {
    // Architecture forces routing; gate set forbids SWAP, so the second
    // rebase pass MUST decompose SWAPs (typically into 3 CX).
    std::vector<Command> cmds = {
        make_cmd(GateType::H,  {0}),
        make_cmd(GateType::CX, {0, 2}),
    };
    Device d(3);
    d.architecture = Architecture(3, {{0, 1}, {1, 2}}, "chain");

    // Build a gate set that contains CX + H + 1q gates but NOT SWAP.
    GateSet gs;
    gs.name = "no_swap";
    gs.allowed = {
        GateType::X, GateType::Y, GateType::Z, GateType::H,
        GateType::S, GateType::Sdg, GateType::T, GateType::Tdg,
        GateType::Rx, GateType::Ry, GateType::Rz,
        GateType::CX,
        GateType::Measure, GateType::Barrier, GateType::GPhase,
    };
    d.gate_set = gs;

    // Pin Lite: the point is the second-rebase-decomposes-SWAPs mechanic,
    // and Sabre's mapping search avoids the SWAP on this fixture entirely.
    auto out = compile_for_device(cmds, 3, d, RouterKind::Lite);
    // No SWAP survives the second rebase pass.
    EXPECT_EQ(count_gates(out.commands, GateType::SWAP), 0);
    // Multiple CX must be present: original + the 3-CX SWAP expansion.
    EXPECT_GE(count_gates(out.commands, GateType::CX), 4);
}

// ── compile_for_device end-to-end with QarpSimulator ───────────────────────

TEST(CompileForDevice, EndToEndDistributionMatchesUnroutedAfterReindex) {
    std::vector<Command> base = {
        make_cmd(GateType::H,  {0}),
        make_cmd(GateType::CX, {0, 2}),
    };

    Device d(3);
    d.architecture = Architecture(3, {{0, 1}, {1, 2}});
    auto compiled = compile_for_device(base, 3, d);

    QarpSimulator sim;
    auto unrouted = sim.run(base,             3, 6000, 17u);
    auto phys     = sim.run(compiled.commands, 3, 6000, 17u);
    auto routed   = reindex_sampling_result(phys, compiled.final_logical_to_physical);

    for (const auto& [outcome, count] : unrouted.counts) {
        const int r = routed.counts.count(outcome) ? routed.counts.at(outcome) : 0;
        EXPECT_NEAR(static_cast<double>(r) / 6000.0,
                    static_cast<double>(count) / 6000.0,
                    0.05);
    }
}
