// Tests for qarpx::CircuitDAG.
//
// The load-bearing contract is RT-1: to_commands(from_commands(v)) == v,
// element-for-element, for every valid stream — including branch regions,
// conditions, Custom, GPhase, measures, resets, and barriers.  With
// remove_node the linearization must yield the survivors in original
// relative order.

#include <gtest/gtest.h>

#include "dag_test_helpers.h"

#include "qarpx/block/block.h"
#include "qarpx/block/composite_block.h"
#include "qarpx/block/conditional_block.h"
#include "qarpx/block/controlled_block.h"

#include <algorithm>
#include <random>
#include <vector>

namespace qarpx::test {

namespace {
using NodeId = CircuitDAG::NodeId;
}  // anonymous namespace

// ── RT-1 round-trip exactness ───────────────────────────────────────────────

TEST(CircuitDAG, RoundTripEmpty) {
    expect_round_trip({});
    auto dag = CircuitDAG::from_commands({});
    EXPECT_EQ(dag.n_nodes(), 0u);
    EXPECT_EQ(dag.depth(), 0u);
    EXPECT_TRUE(dag.front_layer().empty());
}

TEST(CircuitDAG, RoundTripSimpleSequence) {
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::H, 0u);
    cmds.emplace_back(GateType::CX, 0u, 1u);
    cmds.emplace_back(GateType::Rz, 1u, Param::symbol("theta"));
    expect_round_trip(cmds);
}

TEST(CircuitDAG, RoundTripPreservesInterleavingOnDisjointWires) {
    // H(0) X(1) H(0) X(1): wire-independent interleaving must come back
    // in exactly the input order, not grouped per wire.
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::H, 0u);
    cmds.emplace_back(GateType::X, 1u);
    cmds.emplace_back(GateType::H, 0u);
    cmds.emplace_back(GateType::X, 1u);
    expect_round_trip(cmds);
}

TEST(CircuitDAG, RoundTripFullGrammar) {
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::H, 0u);
    cmds.push_back(make_measure(0, 0));
    cmds.emplace_back(GateType::Reset, 0u);
    cmds.push_back(Command(GateType::Barrier, {0u, 2u}));
    Command gp;
    gp.gate = GateType::GPhase;
    gp.params.push_back(Param(0.25));
    cmds.push_back(gp);
    cmds.push_back(make_custom_1q(1));
    cmds.push_back(make_conditional(Command(GateType::X, 1u), {0}, {true}));
    cmds.emplace_back(GateType::CCX, SmallVector<uint32_t, 2>{0u, 1u, 2u});
    expect_round_trip(cmds);
}

TEST(CircuitDAG, RoundTripBranchRegionVerbatim) {
    std::vector<Command> cmds;
    cmds.push_back(make_measure(0, 0));
    cmds.push_back(make_branch_begin({0}, {true}));
    cmds.emplace_back(GateType::X, 1u);
    cmds.push_back(make_marker(GateType::BranchElse));
    cmds.emplace_back(GateType::Z, 1u);
    cmds.push_back(make_marker(GateType::BranchEnd));
    cmds.emplace_back(GateType::H, 2u);
    expect_round_trip(cmds);

    auto dag = CircuitDAG::from_commands(cmds);
    // Measure + region + H = 3 nodes; the 5-command region is one node.
    EXPECT_EQ(dag.n_nodes(), 3u);
}

TEST(CircuitDAG, RoundTripNestedBranchRegions) {
    std::vector<Command> cmds;
    cmds.push_back(make_measure(0, 0));
    cmds.push_back(make_measure(1, 1));
    cmds.push_back(make_branch_begin({0}, {true}));
    cmds.emplace_back(GateType::X, 0u);
    cmds.push_back(make_branch_begin({1}, {false}));
    cmds.emplace_back(GateType::Y, 1u);
    cmds.push_back(make_marker(GateType::BranchEnd));
    cmds.push_back(make_marker(GateType::BranchEnd));
    expect_round_trip(cmds);

    auto dag = CircuitDAG::from_commands(cmds);
    // Two measures + one outer region (inner region stays inside it).
    EXPECT_EQ(dag.n_nodes(), 3u);
}

// ── Malformed input ─────────────────────────────────────────────────────────

TEST(CircuitDAG, ThrowsOnStrayElse) {
    std::vector<Command> cmds{make_marker(GateType::BranchElse)};
    EXPECT_THROW(CircuitDAG::from_commands(cmds), std::invalid_argument);
}

TEST(CircuitDAG, ThrowsOnStrayEnd) {
    std::vector<Command> cmds{make_marker(GateType::BranchEnd)};
    EXPECT_THROW(CircuitDAG::from_commands(cmds), std::invalid_argument);
}

TEST(CircuitDAG, ThrowsOnUnterminatedBegin) {
    std::vector<Command> cmds;
    cmds.push_back(make_branch_begin({0}, {true}));
    cmds.emplace_back(GateType::X, 0u);
    EXPECT_THROW(CircuitDAG::from_commands(cmds), std::invalid_argument);
}

TEST(CircuitDAG, ThrowsOnOutOfRangeIndices) {
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::H, 3u);
    EXPECT_THROW(CircuitDAG::from_commands(cmds, 2, 0), std::invalid_argument);

    std::vector<Command> cmds2{make_measure(0, 5)};
    EXPECT_THROW(CircuitDAG::from_commands(cmds2, 1, 2), std::invalid_argument);

    std::vector<Command> cmds3{
        make_conditional(Command(GateType::X, 0u), {7}, {true})};
    EXPECT_THROW(CircuitDAG::from_commands(cmds3, 1, 2), std::invalid_argument);
}

// ── Wire structure ──────────────────────────────────────────────────────────

TEST(CircuitDAG, WireAdjacency) {
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::H, 0u);   // node 0
    cmds.emplace_back(GateType::CX, 0u, 1u);  // node 1
    cmds.emplace_back(GateType::H, 1u);   // node 2
    auto dag = CircuitDAG::from_commands(cmds);

    const uint32_t w0 = dag.qubit_wire(0);
    const uint32_t w1 = dag.qubit_wire(1);

    EXPECT_EQ(dag.wire_front(w0), 0u);
    EXPECT_EQ(dag.wire_back(w0), 1u);
    EXPECT_EQ(dag.wire_front(w1), 1u);
    EXPECT_EQ(dag.wire_back(w1), 2u);

    EXPECT_EQ(dag.next_on_wire(0, w0), 1u);
    EXPECT_EQ(dag.prev_on_wire(1, w0), 0u);
    EXPECT_EQ(dag.prev_on_wire(1, w1), CircuitDAG::kNone);
    EXPECT_EQ(dag.next_on_wire(1, w1), 2u);
    EXPECT_EQ(dag.next_on_wire(2, w1), CircuitDAG::kNone);

    // Node 2 does not touch qubit 0's wire.
    EXPECT_THROW((void)dag.next_on_wire(2, w0), std::invalid_argument);
}

TEST(CircuitDAG, ConditionReadOrdersAfterMeasureWrite) {
    std::vector<Command> cmds;
    cmds.push_back(make_measure(0, 0));                                  // node 0
    cmds.push_back(make_conditional(Command(GateType::X, 1u), {0}, {true}));  // node 1
    auto dag = CircuitDAG::from_commands(cmds);

    // The conditional X reads cbit 0 → ordered after the Measure write,
    // despite disjoint qubits.
    const uint32_t cw = dag.cbit_wire(0);
    EXPECT_EQ(dag.prev_on_wire(1, cw), 0u);
    EXPECT_EQ(dag.next_on_wire(0, cw), 1u);
    EXPECT_EQ(dag.front_layer(), std::vector<NodeId>({0}));
}

TEST(CircuitDAG, GlobalBarrierTouchesEveryQubitWireAndGlobal) {
    // A qubit-less Barrier is OpenQASM's `barrier;` — it fences the whole
    // register, like fuse_single_qubit_gates flushes every accumulator.
    std::vector<Command> cmds;
    cmds.push_back(Command(GateType::H, 0u));
    Command global_barrier;
    global_barrier.gate = GateType::Barrier;
    cmds.push_back(global_barrier);
    cmds.push_back(Command(GateType::H, 1u));
    auto dag = CircuitDAG::from_commands(cmds, 3, 1);
    std::vector<uint32_t> wires(dag.wires(1).begin(), dag.wires(1).end());
    std::sort(wires.begin(), wires.end());
    EXPECT_EQ(wires, (std::vector<uint32_t>{0, 1, 2, dag.global_wire()}));
    // A listed barrier still fences only its qubits.
    std::vector<Command> listed;
    listed.push_back(Command(GateType::Barrier, {0u}));
    auto d2 = CircuitDAG::from_commands(listed, 3, 0);
    EXPECT_EQ(d2.wires(0).size(), 1u);
}

TEST(CircuitDAG, RegionNodeIsFullBarrier) {
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::H, 0u);                 // node 0
    cmds.push_back(make_branch_begin({0}, {true}));      // node 1 (region)
    cmds.emplace_back(GateType::X, 1u);
    cmds.push_back(make_marker(GateType::BranchEnd));
    cmds.emplace_back(GateType::H, 2u);                 // node 2
    auto dag = CircuitDAG::from_commands(cmds);

    ASSERT_EQ(dag.n_nodes(), 3u);
    EXPECT_TRUE(dag.is_region(1));
    EXPECT_EQ(dag.region_commands(1).size(), 3u);
    EXPECT_FALSE(dag.is_region(0));
    EXPECT_THROW((void)dag.region_commands(0), std::invalid_argument);

    // Region spans every wire: H(2) is ordered after it even though qubit 2
    // is untouched inside the region.
    EXPECT_EQ(dag.prev_on_wire(2, dag.qubit_wire(2)), 1u);
    // And the region is ordered after H(0) on qubit 0's wire.
    EXPECT_EQ(dag.next_on_wire(0, dag.qubit_wire(0)), 1u);
}

// ── Analysis ────────────────────────────────────────────────────────────────

TEST(CircuitDAG, FrontLayerBasic) {
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::H, 0u);       // node 0: ready
    cmds.emplace_back(GateType::H, 1u);       // node 1: ready
    cmds.emplace_back(GateType::CX, 0u, 1u);  // node 2: blocked by 0 and 1
    auto dag = CircuitDAG::from_commands(cmds);
    EXPECT_EQ(dag.front_layer(), std::vector<NodeId>({0, 1}));
}

TEST(CircuitDAG, DepthCountsLongestWirePath) {
    // H(0) H(1) CX(0,1) H(0): depth 3 (H → CX → H), parallel H(1) absorbed.
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::H, 0u);
    cmds.emplace_back(GateType::H, 1u);
    cmds.emplace_back(GateType::CX, 0u, 1u);
    cmds.emplace_back(GateType::H, 0u);
    auto dag = CircuitDAG::from_commands(cmds);
    EXPECT_EQ(dag.depth(), 3u);
}

TEST(CircuitDAG, DepthIgnoresBarrierAndGPhase) {
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::H, 0u);
    cmds.push_back(Command(GateType::Barrier, {0u, 1u}));
    Command gp;
    gp.gate = GateType::GPhase;
    gp.params.push_back(Param(0.5));
    cmds.push_back(gp);
    cmds.emplace_back(GateType::X, 0u);
    auto dag = CircuitDAG::from_commands(cmds);
    // H → Barrier(0) → X on qubit 0: weight 1 + 0 + 1.
    EXPECT_EQ(dag.depth(), 2u);
}

TEST(CircuitDAG, LayersAreAsapMoments) {
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::H, 0u);       // L0
    cmds.emplace_back(GateType::H, 1u);       // L0
    cmds.emplace_back(GateType::CX, 0u, 1u);  // L1
    cmds.emplace_back(GateType::H, 0u);       // L2
    auto dag = CircuitDAG::from_commands(cmds);

    const auto ls = dag.layers();
    ASSERT_EQ(ls.size(), 3u);
    EXPECT_EQ(ls[0], (std::vector<NodeId>{0, 1}));
    EXPECT_EQ(ls[1], (std::vector<NodeId>{2}));
    EXPECT_EQ(ls[2], (std::vector<NodeId>{3}));

    // Every live node appears exactly once.
    std::size_t total = 0;
    for (const auto& l : ls) total += l.size();
    EXPECT_EQ(total, dag.n_nodes());
}

TEST(CircuitDAG, LayersAreAntichains) {
    for (uint32_t seed = 0; seed < 40; ++seed) {
        RandomCircuit gen(seed + 3000, /*n_qubits=*/4, /*n_cbits=*/2);
        auto dag = CircuitDAG::from_commands(gen.generate(40));
        const auto ls = dag.layers();

        std::vector<std::size_t> level(dag.n_slots(), 0);
        for (std::size_t k = 0; k < ls.size(); ++k)
            for (auto id : ls[k]) level[id] = k;

        // Each node sits strictly after every wire predecessor's layer.
        for (NodeId id = 0; id < dag.n_slots(); ++id) {
            if (dag.is_removed(id)) continue;
            for (auto w : dag.wires(id)) {
                const NodeId p = dag.prev_on_wire(id, w);
                if (p != CircuitDAG::kNone)
                    EXPECT_LT(level[p], level[id]) << "seed " << seed;
            }
        }
    }
}

TEST(CircuitDAG, CountOpsIncludesRegionInteriors) {
    std::vector<Command> cmds;
    cmds.push_back(make_measure(0, 0));
    cmds.emplace_back(GateType::H, 1u);
    cmds.push_back(make_branch_begin({0}, {true}));
    cmds.emplace_back(GateType::X, 1u);
    cmds.emplace_back(GateType::X, 1u);
    cmds.push_back(make_marker(GateType::BranchEnd));
    auto dag = CircuitDAG::from_commands(cmds);

    const auto ops = dag.count_ops();
    EXPECT_EQ(ops.at("Measure"), 1u);
    EXPECT_EQ(ops.at("H"), 1u);
    EXPECT_EQ(ops.at("X"), 2u);
    EXPECT_EQ(ops.at("BranchBegin"), 1u);
    EXPECT_EQ(ops.at("BranchEnd"), 1u);
}

// ── Mutation ────────────────────────────────────────────────────────────────

TEST(CircuitDAG, RemoveNodeSplicesWireAndRelinearizes) {
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::H, 0u);   // node 0
    cmds.emplace_back(GateType::X, 0u);   // node 1
    cmds.emplace_back(GateType::H, 0u);   // node 2
    auto dag = CircuitDAG::from_commands(cmds);

    dag.remove_node(1);
    EXPECT_EQ(dag.n_nodes(), 2u);
    EXPECT_TRUE(dag.is_removed(1));
    EXPECT_THROW(dag.remove_node(1), std::invalid_argument);
    EXPECT_THROW((void)dag.command(1), std::invalid_argument);

    const uint32_t w0 = dag.qubit_wire(0);
    EXPECT_EQ(dag.next_on_wire(0, w0), 2u);
    EXPECT_EQ(dag.prev_on_wire(2, w0), 0u);

    auto back = dag.to_commands();
    ASSERT_EQ(back.size(), 2u);
    EXPECT_EQ(back[0].gate, GateType::H);
    EXPECT_EQ(back[1].gate, GateType::H);
    EXPECT_EQ(dag.depth(), 2u);
}

TEST(CircuitDAG, RemoveHeadAndTailUpdateWireBoundaries) {
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::H, 0u);
    cmds.emplace_back(GateType::X, 0u);
    cmds.emplace_back(GateType::Y, 0u);
    auto dag = CircuitDAG::from_commands(cmds);
    const uint32_t w0 = dag.qubit_wire(0);

    dag.remove_node(0);
    EXPECT_EQ(dag.wire_front(w0), 1u);
    dag.remove_node(2);
    EXPECT_EQ(dag.wire_back(w0), 1u);
    dag.remove_node(1);
    EXPECT_EQ(dag.wire_front(w0), CircuitDAG::kNone);
    EXPECT_EQ(dag.wire_back(w0), CircuitDAG::kNone);
    EXPECT_TRUE(dag.to_commands().empty());
    EXPECT_EQ(dag.depth(), 0u);
}

// ── Replay of real block flatten() outputs (Phase 0 exit criterion) ─────────

TEST(CircuitDAG, RoundTripRealBlockFlattenOutputs) {
    // Synthesis builders produce the longest / densest realistic streams.
    ref<Block> prep(new SimpleBlock(2, "prep"));
    prep->state_preparation({0.5, {0.0, 0.5}, {0.5, 0.0}, -0.5});
    prep->h(0).rz(1, Param::symbol("theta")).cx(0, 1);
    prep->build();
    expect_round_trip(prep->flatten());

    // Controlled wrapper: control-prepended commands, MC basis lowering.
    ref<Block> inner(new SimpleBlock(2, "inner"));
    inner->h(0).cx(0, 1).rz(1, Param(0.7));
    inner->build();
    ref<Block> ctrl(new ControlledBlock(inner, 1, std::vector<bool>{true}));
    ctrl->build();
    expect_round_trip(ctrl->flatten());

    // Conditional block: emits a BranchBegin…BranchEnd region.
    ref<Block> then_body(new SimpleBlock(1, "then"));
    then_body->x(0);
    then_body->build();
    ref<Block> else_body(new SimpleBlock(1, "else"));
    else_body->z(0);
    else_body->build();
    ref<Block> cond(new ConditionalBlock(
        std::vector<uint32_t>{0}, std::vector<bool>{true},
        then_body, else_body, "cond"));
    cond->build();

    // Composite: measure feeding the conditional, cbit offsetting exercised.
    ref<Block> meas(new SimpleBlock(1, "meas"));
    meas->h(0).measure(0u, 0u);
    meas->build();
    ref<Block> comp(new CompositeBlock(
        std::vector<ref<Block>>{meas, cond}, 1));
    comp->build();
    expect_round_trip(comp->flatten());
}

// ── Randomized property tests ───────────────────────────────────────────────
// (RandomCircuit lives in dag_test_helpers.h, shared with test_dag_passes.cpp)

TEST(CircuitDAGProperty, RoundTripOnRandomCircuits) {
    for (uint32_t seed = 0; seed < 200; ++seed) {
        RandomCircuit gen(seed, /*n_qubits=*/5, /*n_cbits=*/3);
        auto cmds = gen.generate(60);
        auto dag = CircuitDAG::from_commands(cmds);
        auto back = dag.to_commands();
        ASSERT_TRUE(streams_equal(back, cmds)) << "RT-1 violated at seed " << seed;
    }
}

TEST(CircuitDAGProperty, RemovalYieldsSurvivorsInOriginalOrder) {
    for (uint32_t seed = 0; seed < 100; ++seed) {
        RandomCircuit gen(seed + 1000, /*n_qubits=*/4, /*n_cbits=*/2);
        auto cmds = gen.generate(40);
        auto dag = CircuitDAG::from_commands(cmds);

        std::mt19937 rng(seed);
        for (NodeId id = 0; id < dag.n_slots(); ++id) {
            if (std::uniform_int_distribution<int>(0, 3)(rng) == 0)
                dag.remove_node(id);
        }

        // Expected: surviving nodes in id order (== original order), regions
        // expanded verbatim.
        std::vector<Command> expected;
        for (NodeId id = 0; id < dag.n_slots(); ++id) {
            if (dag.is_removed(id)) continue;
            if (dag.is_region(id)) {
                const auto& r = dag.region_commands(id);
                expected.insert(expected.end(), r.begin(), r.end());
            } else {
                expected.push_back(dag.command(id));
            }
        }
        ASSERT_TRUE(streams_equal(dag.to_commands(), expected))
            << "survivor order violated at seed " << seed;
    }
}

TEST(CircuitDAGProperty, DepthNeverExceedsLiveOpCount) {
    for (uint32_t seed = 0; seed < 50; ++seed) {
        RandomCircuit gen(seed + 2000, /*n_qubits=*/4, /*n_cbits=*/2);
        auto cmds = gen.generate(30);
        auto dag = CircuitDAG::from_commands(cmds);
        EXPECT_LE(dag.depth(), dag.n_nodes());
    }
}

}  // namespace qarpx::test
