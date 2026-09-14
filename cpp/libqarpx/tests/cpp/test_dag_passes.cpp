// Tests for dag_passes::cancel_wire_adjacent and the OptLevel-parameterized
// Transpiler::transpile_and_optimize.
//
// Contracts under test:
//   - Same rewrite rules as eliminate_identities, on WIRE adjacency —
//     strictly more eliminations (differential superset), never fewer.
//   - EQ-2: exact unitary equivalence (global phase included) on unitary
//     streams.
//   - Blocked exactly where the plan says: touched-qubit barriers, branch
//     regions, mismatched condition tuples, intervening cbit writes.

#include "dag_test_helpers.h"

#include "qarpx/dag/passes.h"
#include "qarpx/simulator/qarp_simulator.h"
#include "qarpx/transpiler/gateset.h"
#include "qarpx/transpiler/identities.h"
#include "qarpx/transpiler/transpiler.h"

#include <cmath>
#include <functional>
#include <unordered_map>
#include <vector>

namespace qarpx::test {

namespace {

/// Run the DAG pass over a stream; returns (output, eliminated count).
std::pair<std::vector<Command>, std::size_t> dag_cancel(
    const std::vector<Command>& cmds) {
    auto dag = CircuitDAG::from_commands(cmds);
    const std::size_t n = dag_passes::cancel_wire_adjacent(dag);
    return {dag.to_commands(), n};
}

bool unitaries_match(const std::vector<Command>& a,
                     const std::vector<Command>& b, int n_qubits) {
    QarpSimulator sim;
    const auto ua = sim.unitary_matrix(a, n_qubits);
    const auto ub = sim.unitary_matrix(b, n_qubits);
    return (ua - ub).norm() < 1e-10;
}

}  // anonymous namespace

// ── Wire adjacency beats textual adjacency ──────────────────────────────────

TEST(DagCancel, CancelsAcrossDisjointWireInterleaving) {
    // H(0) X(1) H(0): the marquee case the linear pass misses.
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::H, 0u);
    cmds.emplace_back(GateType::X, 1u);
    cmds.emplace_back(GateType::H, 0u);

    auto linear = cmds;
    // The retired textually-adjacent pass was blind to it (frozen reference).
    EXPECT_EQ(frozen_eliminate_identities_reference(linear), 0u);

    auto [out, n] = dag_cancel(cmds);
    EXPECT_EQ(n, 2u);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].gate, GateType::X);
    EXPECT_TRUE(unitaries_match(cmds, out, 2));
}

TEST(DagCancel, MergesRotationsAcrossDisjointWires) {
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::Rz, 0u, Param(0.3));
    cmds.emplace_back(GateType::CX, 1u, 2u);
    cmds.emplace_back(GateType::Rz, 0u, Param(0.4));

    auto [out, n] = dag_cancel(cmds);
    EXPECT_EQ(n, 1u);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].gate, GateType::Rz);
    EXPECT_NEAR(out[0].params[0].value(), 0.7, 1e-12);
    EXPECT_TRUE(unitaries_match(cmds, out, 3));
}

TEST(DagCancel, CascadeExposesOuterPair) {
    // CX(0,1) H(0) H(0) CX(0,1) → inner HH cancels, then the CX pair becomes
    // wire-adjacent on both wires and cancels too: fixpoint to empty.
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::CX, 0u, 1u);
    cmds.emplace_back(GateType::H, 0u);
    cmds.emplace_back(GateType::H, 0u);
    cmds.emplace_back(GateType::CX, 0u, 1u);

    auto [out, n] = dag_cancel(cmds);
    EXPECT_TRUE(out.empty());
    EXPECT_EQ(n, 4u);
}

TEST(DagCancel, TwoQubitPairNeedsAdjacencyOnBothWires) {
    // CX(0,1) X(1) CX(0,1): X intervenes on the target wire — no cancel.
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::CX, 0u, 1u);
    cmds.emplace_back(GateType::X, 1u);
    cmds.emplace_back(GateType::CX, 0u, 1u);

    auto [out, n] = dag_cancel(cmds);
    EXPECT_EQ(n, 0u);
    EXPECT_EQ(out.size(), 3u);
}

TEST(DagCancel, UnorderedCzSwapCancel) {
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::CZ, 0u, 1u);
    cmds.emplace_back(GateType::CZ, 1u, 0u);
    cmds.emplace_back(GateType::SWAP, 2u, 3u);
    cmds.emplace_back(GateType::SWAP, 3u, 2u);

    auto [out, n] = dag_cancel(cmds);
    EXPECT_TRUE(out.empty());
    EXPECT_EQ(n, 4u);
}

TEST(DagCancel, ZeroRotationDropAndZeroMerge) {
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::Rx, 0u, Param(0.0));           // dropped: +1
    cmds.emplace_back(GateType::Rz, 1u, Param(0.5));           // merged to zero
    cmds.emplace_back(GateType::H, 2u);
    cmds.emplace_back(GateType::Rz, 1u, Param(-0.5));          // with this: +2

    auto [out, n] = dag_cancel(cmds);
    EXPECT_EQ(n, 3u);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].gate, GateType::H);
}

TEST(DagCancel, GPhaseMergesAcrossGates) {
    // GPhase rides its own global wire: gates in between are transparent.
    std::vector<Command> cmds;
    Command g1; g1.gate = GateType::GPhase; g1.params.push_back(Param(0.2));
    Command g2; g2.gate = GateType::GPhase; g2.params.push_back(Param(0.3));
    cmds.push_back(g1);
    cmds.emplace_back(GateType::H, 0u);
    cmds.push_back(g2);

    auto [out, n] = dag_cancel(cmds);
    EXPECT_EQ(n, 1u);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].gate, GateType::GPhase);
    EXPECT_NEAR(out[0].params[0].value(), 0.5, 1e-12);
    EXPECT_TRUE(unitaries_match(cmds, out, 1));
}

TEST(DagCancel, SymbolicRotationsMerge) {
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::Rz, 0u, Param::symbol("theta"));
    cmds.emplace_back(GateType::X, 1u);
    cmds.emplace_back(GateType::Rz, 0u, Param::symbol("theta"));

    auto [out, n] = dag_cancel(cmds);
    EXPECT_EQ(n, 1u);
    ASSERT_EQ(out.size(), 2u);
    // (1·θ) + (1·θ) = 2·θ, still the linear form.
    EXPECT_EQ(out[0].params[0], Param::linear(2.0, "theta"));
}

// ── Blocking conditions ─────────────────────────────────────────────────────

TEST(DagCancel, GlobalBarrierBlocksCancellationAtEveryLevel) {
    // H; barrier; H — the qubit-less barrier fences qubit 0 too, so the pair
    // must not cancel at O1 or O2 (P1.19).
    std::vector<Command> cmds;
    cmds.push_back(Command(GateType::H, 0u));
    Command global_barrier;
    global_barrier.gate = GateType::Barrier;
    cmds.push_back(global_barrier);
    cmds.push_back(Command(GateType::H, 0u));
    Transpiler t(native_gateset());
    for (OptLevel lvl : {OptLevel::O1, OptLevel::O2}) {
        auto out = t.transpile_and_optimize(cmds, lvl);
        int n_h = 0;
        for (const auto& c : out) if (c.gate == GateType::H) ++n_h;
        EXPECT_EQ(n_h, 2) << "level " << static_cast<int>(lvl);
        EXPECT_EQ(out.size(), 3u);
    }
}

TEST(DagCancel, BlockedByTouchedBarrier_NotByUntouchedBarrier) {
    // Barrier on qubit 0 blocks the H(0) pair...
    std::vector<Command> blocked;
    blocked.emplace_back(GateType::H, 0u);
    blocked.push_back(Command(GateType::Barrier, {0u}));
    blocked.emplace_back(GateType::H, 0u);
    EXPECT_EQ(dag_cancel(blocked).second, 0u);

    // ...but a barrier on qubit 1 only does not (fusion already scopes
    // barriers to their listed qubits — same semantics here).
    std::vector<Command> transparent;
    transparent.emplace_back(GateType::H, 0u);
    transparent.push_back(Command(GateType::Barrier, {1u}));
    transparent.emplace_back(GateType::H, 0u);
    auto [out, n] = dag_cancel(transparent);
    EXPECT_EQ(n, 2u);
    EXPECT_EQ(out.size(), 1u);
}

TEST(DagCancel, BlockedByBranchRegion) {
    // Region nodes span all wires: nothing combines across them.
    std::vector<Command> cmds;
    cmds.push_back(make_measure(0, 0));
    cmds.emplace_back(GateType::H, 1u);
    cmds.push_back(make_branch_begin({0}, {true}));
    cmds.emplace_back(GateType::X, 2u);
    cmds.push_back(make_marker(GateType::BranchEnd));
    cmds.emplace_back(GateType::H, 1u);

    auto [out, n] = dag_cancel(cmds);
    EXPECT_EQ(n, 0u);
    EXPECT_EQ(out.size(), cmds.size());
}

TEST(DagCancel, OptimizesInsideRegionBodies_NotAcrossBoundary) {
    // Parity with the frozen linear oracle: commands combine WITHIN a branch body;
    // the boundary stays a barrier.  Then-body HH cancels, else-body Rz pair
    // merges; the outer H(1) pair straddling the region does not cancel.
    std::vector<Command> cmds;
    cmds.push_back(make_measure(0, 0));
    cmds.emplace_back(GateType::H, 1u);
    cmds.push_back(make_branch_begin({0}, {true}));
    cmds.emplace_back(GateType::H, 2u);
    cmds.emplace_back(GateType::H, 2u);          // cancels inside then-body
    cmds.emplace_back(GateType::X, 2u);
    cmds.push_back(make_marker(GateType::BranchElse));
    cmds.emplace_back(GateType::Rz, 2u, Param(0.2));
    cmds.emplace_back(GateType::Rz, 2u, Param(0.3));  // merges inside else-body
    cmds.push_back(make_marker(GateType::BranchEnd));
    cmds.emplace_back(GateType::H, 1u);

    auto [out, n] = dag_cancel(cmds);
    EXPECT_EQ(n, 3u);  // HH pair (+2) + Rz merge (+1)
    // measure, H(1), Begin, X, Else, Rz(0.5), End, H(1) = 8 commands.
    ASSERT_EQ(out.size(), 8u);
    EXPECT_EQ(out[1].gate, GateType::H);   // outer pair survives
    EXPECT_EQ(out[7].gate, GateType::H);
    EXPECT_EQ(out[3].gate, GateType::X);   // then-body reduced to X
    EXPECT_NEAR(out[5].params[0].value(), 0.5, 1e-12);  // merged else-body Rz
}

TEST(DagCancel, ConditionTuplesMustMatchExactly) {
    // Unconditional H never cancels against conditional H.
    std::vector<Command> mismatched;
    mismatched.push_back(make_measure(0, 0));
    mismatched.emplace_back(GateType::H, 1u);
    mismatched.push_back(make_conditional(Command(GateType::H, 1u), {0}, {true}));
    EXPECT_EQ(dag_cancel(mismatched).second, 0u);

    // Identical condition tuples do cancel (same_condition parity).
    std::vector<Command> matched;
    matched.push_back(make_measure(0, 0));
    matched.push_back(make_conditional(Command(GateType::H, 1u), {0}, {true}));
    matched.push_back(make_conditional(Command(GateType::H, 1u), {0}, {true}));
    auto [out, n] = dag_cancel(matched);
    EXPECT_EQ(n, 2u);
    EXPECT_EQ(out.size(), 1u);
}

TEST(DagCancel, ConditionalPairBlockedByInterveningCbitWrite) {
    // A Measure writing the read cbit between two conditional gates keeps
    // them apart on the cbit wire — cancelling would change semantics.
    std::vector<Command> cmds;
    cmds.push_back(make_measure(0, 0));
    cmds.push_back(make_conditional(Command(GateType::H, 1u), {0}, {true}));
    cmds.push_back(make_measure(0, 0));
    cmds.push_back(make_conditional(Command(GateType::H, 1u), {0}, {true}));
    EXPECT_EQ(dag_cancel(cmds).second, 0u);
}

// ── Differential vs the frozen linear oracle ────────────────────────────────

TEST(DagCancelProperty, EliminatesSupersetOfLinearPass) {
    for (uint32_t seed = 0; seed < 150; ++seed) {
        RandomCircuit gen(seed, /*n_qubits=*/4, /*n_cbits=*/2);
        auto cmds = gen.generate(50);

        auto linear = cmds;
        const std::size_t linear_n = frozen_eliminate_identities_reference(linear);
        const std::size_t dag_n = dag_cancel(cmds).second;

        EXPECT_GE(dag_n, linear_n) << "seed " << seed;
    }
}

TEST(DagCancelProperty, PreservesUnitaryExactly) {
    for (uint32_t seed = 0; seed < 150; ++seed) {
        RandomCircuit gen(seed + 5000, /*n_qubits=*/3, /*n_cbits=*/1,
                          CircuitGrammar::kUnitaryConcrete);
        auto cmds = gen.generate(40);
        auto [out, n] = dag_cancel(cmds);
        (void)n;
        ASSERT_TRUE(unitaries_match(cmds, out, 3)) << "EQ-2 violated at seed " << seed;
    }
}

TEST(DagCancelProperty, OutputIsFixpoint) {
    // Re-running the pass on its own output eliminates nothing further.
    for (uint32_t seed = 0; seed < 50; ++seed) {
        RandomCircuit gen(seed + 7000, /*n_qubits=*/4, /*n_cbits=*/2);
        auto [out, n] = dag_cancel(gen.generate(50));
        (void)n;
        EXPECT_EQ(dag_cancel(out).second, 0u) << "seed " << seed;
    }
}

// ── DAG-pass dominance over the frozen linear oracle ────────────────────────
//
// The DAG pass must (a) eliminate a superset of what the frozen linear oracle
// eliminates on the FULL grammar including nested regions, and (b) preserve
// measurement-outcome distributions on non-unitary streams — the coverage the
// unitary-only EQ-2 property tests can't give.

namespace {

/// Dispatchable MCM-heavy generator: plain csim-native gates + measure +
/// reset + conditional-1q + (nested) branch regions.  Concrete params only.
std::vector<Command> random_mcm_circuit(uint32_t seed, uint32_t n_qubits,
                                        uint32_t n_cbits, std::size_t len) {
    std::mt19937 rng(seed);
    auto pick = [&](uint32_t k) {
        return std::uniform_int_distribution<uint32_t>(0, k - 1)(rng);
    };
    auto q = [&] { return pick(n_qubits); };
    auto c = [&] { return pick(n_cbits); };
    auto angle = [&] {
        return Param(std::uniform_real_distribution<double>(-3.0, 3.0)(rng));
    };

    std::vector<Command> out;
    std::function<void(int)> emit = [&](int region_budget) {
        switch (pick(region_budget > 0 ? 9 : 8)) {
            case 0: out.emplace_back(GateType::H, q()); break;
            case 1: out.emplace_back(GateType::X, q()); break;
            case 2: out.emplace_back(GateType::Rz, q(), angle()); break;
            case 3: {
                const uint32_t a = q();
                uint32_t b = q();
                while (b == a) b = q();
                out.emplace_back(GateType::CX, a, b);
                break;
            }
            case 4: out.push_back(make_measure(q(), c())); break;
            case 5: out.emplace_back(GateType::Reset, q()); break;
            case 6: out.push_back(make_conditional(
                        Command(GateType::X, q()), {c()}, {pick(2) == 0}));
                    break;
            case 7: {  // adjacent pair — guaranteed elimination material
                const uint32_t t = q();
                out.emplace_back(GateType::H, t);
                out.emplace_back(GateType::H, t);
                break;
            }
            case 8: {
                out.push_back(make_branch_begin({c()}, {pick(2) == 0}));
                const std::size_t body = 1 + pick(3);
                for (std::size_t i = 0; i < body; ++i) emit(region_budget - 1);
                if (pick(2) == 0) {
                    out.push_back(make_marker(GateType::BranchElse));
                    out.emplace_back(GateType::Z, q());
                }
                out.push_back(make_marker(GateType::BranchEnd));
                break;
            }
        }
    };
    while (out.size() < len) emit(/*region_budget=*/2);
    return out;
}

double total_variation(const SamplingResult& a, const SamplingResult& b) {
    double tv = 0.0;
    const double na = static_cast<double>(a.n_shots);
    const double nb = static_cast<double>(b.n_shots);
    std::unordered_map<uint64_t, double> diff;
    for (const auto& [bits, count] : a.counts)
        diff[bits] += static_cast<double>(count) / na;
    for (const auto& [bits, count] : b.counts)
        diff[bits] -= static_cast<double>(count) / nb;
    for (const auto& [bits, d] : diff) tv += std::abs(d);
    return tv / 2.0;
}

}  // anonymous namespace

TEST(FrozenOracleDominance, SupersetHoldsOnNestedRegionGrammar) {
    // Differential vs the FROZEN reference (the production linear pass is
    // gone) — full grammar with nesting depth 2.
    for (uint32_t seed = 0; seed < 200; ++seed) {
        RandomCircuit gen(seed + 20000, /*n_qubits=*/4, /*n_cbits=*/2);
        auto cmds = gen.generate(60, /*max_region_depth=*/2);

        auto linear = cmds;
        const std::size_t linear_n = frozen_eliminate_identities_reference(linear);
        auto dag = CircuitDAG::from_commands(cmds);
        const std::size_t dag_n = dag_passes::cancel_wire_adjacent(dag);

        EXPECT_GE(dag_n, linear_n) << "seed " << seed;
    }
}

TEST(FrozenOracleDominance, McmDistributionsPreserved) {
    // On measure/reset/condition/region streams, both the frozen linear
    // output and the DAG output must sample the same distribution as the
    // unoptimized input.  Covers the non-unitary semantics the EQ-2
    // unitary-equality property tests cannot reach.
    QarpSimulator sim;
    const int n_qubits = 3;
    const int n_shots = 16384;

    for (uint32_t seed = 0; seed < 20; ++seed) {
        auto cmds = random_mcm_circuit(seed + 30000, 3, 2, 35);

        auto linear = cmds;
        frozen_eliminate_identities_reference(linear);
        auto dag = CircuitDAG::from_commands(cmds);
        dag_passes::cancel_wire_adjacent(dag);
        const auto dag_out = dag.to_commands();

        const auto r_in  = sim.run(cmds,    n_qubits, n_shots, seed);
        const auto r_lin = sim.run(linear,  n_qubits, n_shots, seed + 1);
        const auto r_dag = sim.run(dag_out, n_qubits, n_shots, seed + 2);

        EXPECT_LT(total_variation(r_in, r_dag), 0.05) << "seed " << seed;
        EXPECT_LT(total_variation(r_lin, r_dag), 0.05) << "seed " << seed;
    }
}

TEST(FrozenOracleDominance, WrapperMatchesDagPassExactly) {
    // The production `eliminate_identities` (now DAG-backed) must agree with
    // an explicit from_commands → cancel → to_commands pipeline, count and
    // stream alike.
    for (uint32_t seed = 0; seed < 100; ++seed) {
        RandomCircuit gen(seed + 40000, /*n_qubits=*/4, /*n_cbits=*/2);
        const auto cmds = gen.generate(50, /*max_region_depth=*/2);

        auto via_wrapper = cmds;
        const std::size_t n_wrapper = eliminate_identities(via_wrapper);

        auto dag = CircuitDAG::from_commands(cmds);
        const std::size_t n_direct = dag_passes::cancel_wire_adjacent(dag);

        EXPECT_EQ(n_wrapper, n_direct) << "seed " << seed;
        EXPECT_TRUE(streams_equal(via_wrapper, dag.to_commands()))
            << "seed " << seed;
    }
}

// ── transpile_and_optimize integration ──────────────────────────────────────

TEST(TranspileAndOptimize, OptLevelsBehave) {
    // CY decomposes to Sdg·CX·S; surround with a cancellable H pair on a
    // spectator qubit that only O1 removes.
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::H, 2u);
    cmds.emplace_back(GateType::CY, 0u, 1u);
    cmds.emplace_back(GateType::H, 2u);

    Transpiler t(native_gateset());

    auto o0 = t.transpile_and_optimize(cmds, OptLevel::O0);
    EXPECT_EQ(o0.size(), t.transpile(cmds).size());

    auto o1 = t.transpile_and_optimize(cmds);  // default O1
    // The spectator H pair is gone; only the CY decomposition remains.
    for (const auto& c : o1)
        EXPECT_NE(c.qubits.empty() ? 99u : c.qubits[0], 2u);

    QarpSimulator sim;
    EXPECT_LT((sim.unitary_matrix(o1, 3) - sim.unitary_matrix(cmds, 3)).norm(), 1e-10);

    // O2 (Phase 2): at least as strong as O1, unitary-exact.
    auto o2 = t.transpile_and_optimize(cmds, OptLevel::O2);
    EXPECT_LE(o2.size(), o1.size());
    EXPECT_LT((sim.unitary_matrix(o2, 3) - sim.unitary_matrix(cmds, 3)).norm(), 1e-10);
}

}  // namespace qarpx::test
