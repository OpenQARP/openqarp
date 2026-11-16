// Tests for dag_passes::commute (matrix-verified) and
// dag_passes::commute_and_cancel (OptLevel::O2).
//
// The commutation table is CONSERVATIVE: commute()==true must imply the
// operators commute exactly (verified against kron-product unitaries here,
// per plan §7.4); commute()==false certifies nothing.

#include "dag_test_helpers.h"

#include "qarpx/dag/commutation.h"
#include "qarpx/dag/passes.h"
#include "qarpx/simulator/qarp_simulator.h"
#include "qarpx/transpiler/gateset.h"
#include "qarpx/transpiler/transpiler.h"

#include <cmath>
#include <vector>

namespace qarpx::test {

namespace {

using dag_passes::commute;

std::pair<std::vector<Command>, std::size_t> o2_cancel(
    const std::vector<Command>& cmds, std::size_t window = 8) {
    auto dag = CircuitDAG::from_commands(cmds);
    const std::size_t n = dag_passes::commute_and_cancel(dag, window);
    return {dag.to_commands(), n};
}

bool unitaries_match(const std::vector<Command>& a,
                     const std::vector<Command>& b, int n_qubits) {
    QarpSimulator sim;
    return (sim.unitary_matrix(a, n_qubits) - sim.unitary_matrix(b, n_qubits))
               .norm() < 1e-10;
}

/// A gate instance with concrete params on a given qubit tuple.
Command instantiate(GateType g, const std::vector<uint32_t>& qs, double seed_angle) {
    SmallVector<uint32_t, 2> qubits;
    for (auto q : qs) qubits.push_back(q);
    SmallVector<Param, 1> params;
    for (uint8_t i = 0; i < gate_num_params(g); ++i)
        params.push_back(Param(seed_angle + 0.211 * i));
    return Command(g, std::move(qubits), std::move(params));
}

}  // anonymous namespace

// ── Matrix-verified soundness of commute() ──────────────────────────────────

TEST(DagCommutation, TableIsSoundAgainstExplicitUnitaries) {
    // Every gate type the table classifies, instantiated on overlapping qubit
    // patterns over 4 qubits.  Whenever commute() says yes, [A,B] must vanish
    // on the explicit 16×16 unitaries.  ~30k pairs, all cheap.
    const std::vector<GateType> kGates = {
        GateType::X,   GateType::Y,   GateType::Z,    GateType::H,
        GateType::S,   GateType::Sdg, GateType::T,    GateType::Tdg,
        GateType::Rx,  GateType::Ry,  GateType::Rz,   GateType::P,
        GateType::U,   GateType::CX,  GateType::CY,   GateType::CZ,
        GateType::SWAP, GateType::iSWAP, GateType::iSWAPdg, GateType::ECR,
        GateType::CRx, GateType::CRy, GateType::CRz,  GateType::CP,
        GateType::RZZ, GateType::RXX, GateType::RYY,  GateType::CU,
        GateType::CCX, GateType::CSWAP, GateType::MCZ,
        GateType::SX,  GateType::SXdg, GateType::Id,
        GateType::CH,  GateType::CS,   GateType::CSdg, GateType::CSX, GateType::CSXdg,
    };
    // Qubit tuples per arity, chosen to exercise every overlap pattern
    // (same-position, crossed, partial, disjoint) within 4 qubits.
    auto tuples_for = [](GateType g) -> std::vector<std::vector<uint32_t>> {
        switch (gate_num_qubits(g)) {
            case 1:  return {{0}, {1}, {2}};
            case 2:  return {{0, 1}, {1, 0}, {1, 2}, {2, 3}, {0, 2}};
            default: return {{0, 1, 2}, {1, 2, 3}, {2, 0, 3}};
        }
    };

    QarpSimulator sim;
    // Exact rebase to the qulacs set guarantees every pair is dispatchable
    // by unitary_matrix without changing the unitary.
    Transpiler rebase(qulacs_gateset());
    auto unitary_of = [&](const Command& x, const Command& y) {
        return sim.unitary_matrix(rebase.transpile({x, y}), 4);
    };

    std::size_t verified = 0;
    for (auto ga : kGates) {
        for (const auto& qa : tuples_for(ga)) {
            const Command a = instantiate(ga, qa, 0.37);
            for (auto gb : kGates) {
                for (const auto& qb : tuples_for(gb)) {
                    const Command b = instantiate(gb, qb, 0.83);
                    if (!commute(a, b)) continue;
                    ASSERT_LT((unitary_of(a, b) - unitary_of(b, a)).norm(), 1e-10)
                        << gate_name(ga) << " / " << gate_name(gb)
                        << " falsely certified as commuting";
                    ++verified;
                }
            }
        }
    }
    // Sanity: the table certifies a substantial set, not just disjoint pairs.
    EXPECT_GT(verified, 1000u);
}

TEST(DagCommutation, ExpectedClassifications) {
    const Command rz0(GateType::Rz, 0u, Param(0.3));
    const Command x1(GateType::X, 1u);
    const Command cx01(GateType::CX, 0u, 1u);
    const Command cz02(GateType::CZ, 0u, 2u);
    const Command h0(GateType::H, 0u);

    EXPECT_TRUE(commute(rz0, cx01));   // Z-diag on shared control wire
    EXPECT_TRUE(commute(x1, cx01));    // X on shared target wire
    EXPECT_TRUE(commute(cx01, cz02));  // shared control, both Z there
    EXPECT_FALSE(commute(h0, cx01));   // H has no certified basis
    EXPECT_FALSE(commute(rz0, h0));
    EXPECT_TRUE(commute(rz0, x1));     // disjoint supports

    // CX(0,1) vs CX(1,0): crossed control/target — X vs Z on both wires.
    EXPECT_FALSE(commute(cx01, Command(GateType::CX, 1u, 0u)));
}

TEST(DagCommutation, MetaAndClassicalBitsBlock) {
    const Command rz0(GateType::Rz, 0u, Param(0.3));
    EXPECT_FALSE(commute(rz0, make_measure(1, 0)));
    EXPECT_FALSE(commute(rz0, Command(GateType::Barrier, {1u})));
    EXPECT_FALSE(commute(rz0, make_custom_1q(1)));

    // Shared cbit (read/read and read/write) blocks even on disjoint qubits.
    const auto cond_a = make_conditional(Command(GateType::Z, 0u), {0}, {true});
    const auto cond_b = make_conditional(Command(GateType::Z, 1u), {0}, {true});
    EXPECT_FALSE(commute(cond_a, cond_b));
    EXPECT_FALSE(commute(cond_a, make_measure(2, 0)));
    // Distinct cbits + matching qubit bases: fine.
    const auto cond_c = make_conditional(Command(GateType::Z, 0u), {1}, {true});
    EXPECT_TRUE(commute(cond_a, cond_c));
}

// ── commute_and_cancel rewrites ─────────────────────────────────────────────

TEST(DagCommutePass, RzMergesThroughCxControl) {
    // The marquee O2 case from identities.h:26.
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::Rz, 0u, Param(0.3));
    cmds.emplace_back(GateType::CX, 0u, 1u);
    cmds.emplace_back(GateType::Rz, 0u, Param(0.4));

    auto [out, n] = o2_cancel(cmds);
    EXPECT_EQ(n, 1u);
    ASSERT_EQ(out.size(), 2u);
    // Slide-right: the merge lands at B's position, after the CX.
    EXPECT_EQ(out[0].gate, GateType::CX);
    EXPECT_EQ(out[1].gate, GateType::Rz);
    EXPECT_NEAR(out[1].params[0].value(), 0.7, 1e-12);
    EXPECT_TRUE(unitaries_match(cmds, out, 2));
}

TEST(DagCommutePass, CxPairCancelsThroughSharedControlCz) {
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::CX, 0u, 1u);
    cmds.emplace_back(GateType::CZ, 0u, 2u);
    cmds.emplace_back(GateType::X, 1u);   // X on CX target: also commutes
    cmds.emplace_back(GateType::CX, 0u, 1u);

    auto [out, n] = o2_cancel(cmds);
    EXPECT_EQ(n, 2u);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].gate, GateType::CZ);
    EXPECT_EQ(out[1].gate, GateType::X);
    EXPECT_TRUE(unitaries_match(cmds, out, 3));
}

TEST(DagCommutePass, BlockedByNonCommutingIntervener) {
    // Z(1) on the CX target wire anti-commutes with the X-class there.
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::CX, 0u, 1u);
    cmds.emplace_back(GateType::Z, 1u);
    cmds.emplace_back(GateType::CX, 0u, 1u);
    EXPECT_EQ(o2_cancel(cmds).second, 0u);

    // H blocks rotation merges.
    std::vector<Command> cmds2;
    cmds2.emplace_back(GateType::Rz, 0u, Param(0.3));
    cmds2.emplace_back(GateType::H, 0u);
    cmds2.emplace_back(GateType::Rz, 0u, Param(0.4));
    EXPECT_EQ(o2_cancel(cmds2).second, 0u);
}

TEST(DagCommutePass, WindowBoundsTheSearch) {
    // 9 commuting, non-combinable T interveners on the wire: the Rz pair is
    // found with window 16, not with window 4.  (T·T neither cancels nor
    // merges, so the interveners themselves are inert.)
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::Rz, 0u, Param(0.3));
    for (uint32_t i = 0; i < 9; ++i)
        cmds.emplace_back(GateType::T, 0u);
    cmds.emplace_back(GateType::Rz, 0u, Param(0.4));

    EXPECT_EQ(o2_cancel(cmds, /*window=*/4).second, 0u);
    auto [out, n] = o2_cancel(cmds, /*window=*/16);
    EXPECT_EQ(n, 1u);
    EXPECT_TRUE(unitaries_match(cmds, out, 1));
}

TEST(DagCommutePass, ConditionalPairMergesAcrossCommutingGate) {
    // Same condition tuple; the intervening CX shares only the qubit wire
    // (Z/Z on the control) — the cbit wire chain is direct.
    std::vector<Command> cmds;
    cmds.push_back(make_measure(2, 0));
    cmds.push_back(make_conditional(Command(GateType::Rz, 0u, Param(0.2)), {0}, {true}));
    cmds.emplace_back(GateType::CX, 0u, 1u);
    cmds.push_back(make_conditional(Command(GateType::Rz, 0u, Param(0.3)), {0}, {true}));

    auto [out, n] = o2_cancel(cmds);
    EXPECT_EQ(n, 1u);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_NEAR(out[2].params[0].value(), 0.5, 1e-12);
}

TEST(DagCommutePass, RegionInteriorsGetO2) {
    std::vector<Command> cmds;
    cmds.push_back(make_measure(0, 0));
    cmds.push_back(make_branch_begin({0}, {true}));
    cmds.emplace_back(GateType::Rz, 1u, Param(0.2));
    cmds.emplace_back(GateType::CX, 1u, 2u);
    cmds.emplace_back(GateType::Rz, 1u, Param(0.3));
    cmds.push_back(make_marker(GateType::BranchEnd));

    auto [out, n] = o2_cancel(cmds);
    EXPECT_EQ(n, 1u);
    // measure, Begin, CX, Rz(0.5), End.
    ASSERT_EQ(out.size(), 5u);
    EXPECT_EQ(out[2].gate, GateType::CX);
    EXPECT_NEAR(out[3].params[0].value(), 0.5, 1e-12);
}

TEST(DagCommutePass, SubsumesWireAdjacency) {
    // Adjacent pairs are the zero-intervener case.
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::H, 0u);
    cmds.emplace_back(GateType::X, 1u);
    cmds.emplace_back(GateType::H, 0u);
    auto [out, n] = o2_cancel(cmds);
    EXPECT_EQ(n, 2u);
    EXPECT_EQ(out.size(), 1u);
}

// ── Properties at O2 ────────────────────────────────────────────────────────

TEST(DagCommutePassProperty, PreservesUnitaryExactly) {
    for (uint32_t seed = 0; seed < 150; ++seed) {
        RandomCircuit gen(seed + 9000, /*n_qubits=*/3, /*n_cbits=*/1,
                          CircuitGrammar::kUnitaryConcrete);
        auto cmds = gen.generate(40);
        auto dag = CircuitDAG::from_commands(cmds);
        dag_passes::cancel_wire_adjacent(dag);
        dag_passes::commute_and_cancel(dag);
        ASSERT_TRUE(unitaries_match(cmds, dag.to_commands(), 3))
            << "EQ-2 violated at seed " << seed;
    }
}

TEST(DagCommutePassProperty, FixpointAndSurvivorOrder) {
    for (uint32_t seed = 0; seed < 60; ++seed) {
        RandomCircuit gen(seed + 11000, /*n_qubits=*/4, /*n_cbits=*/2);
        auto cmds = gen.generate(50);
        auto [out, n] = o2_cancel(cmds);
        (void)n;
        // Idempotent on its own output.
        EXPECT_EQ(o2_cancel(out).second, 0u) << "seed " << seed;
        // Round-trips exactly (well-formed stream, incl. region re-emission).
        expect_round_trip(out);
    }
}

// ── Integration: OptLevel::O2 through the Transpiler ────────────────────────

TEST(TranspileAndOptimizeO2, MoreReductionThanO1AndUnitaryExact) {
    // Rz·CX·Rz on the control needs commutation; O1 can't touch it.
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::Rz, 0u, Param(0.3));
    cmds.emplace_back(GateType::CX, 0u, 1u);
    cmds.emplace_back(GateType::Rz, 0u, Param(0.4));

    Transpiler t(native_gateset());
    auto o1 = t.transpile_and_optimize(cmds, OptLevel::O1);
    auto o2 = t.transpile_and_optimize(cmds, OptLevel::O2);
    EXPECT_LT(o2.size(), o1.size());

    QarpSimulator sim;
    EXPECT_LT((sim.unitary_matrix(o2, 2) - sim.unitary_matrix(cmds, 2)).norm(),
              1e-10);
}

}  // namespace qarpx::test
