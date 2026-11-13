// Trajectory dispatch for mid-circuit-measurement support
// (apply_command_trajectory) and per-shot trajectory mode in run().
//
// Tests follow the project's existing patterns:
//   - test_gate_unitaries style: forced-outcome parity by pinning the seed.
//   - test_dagger_round_trip style: structural invariants (here, that
//     condition_bits round-trip through dagger and substitute).
//   - test_fusion_consistency style: confirm that the fast path is unchanged
//     for circuits with no MCM ops.

#include <gtest/gtest.h>
#include "qarpx/qarpx.h"

#include <complex>
#include <vector>

using namespace qarpx;

namespace {

Command measure_to(uint32_t q, uint32_t c) {
    Command m;
    m.gate = GateType::Measure;
    m.qubits.push_back(q);
    m.cbits.push_back(c);
    return m;
}

Command reset_q(uint32_t q) {
    Command r;
    r.gate = GateType::Reset;
    r.qubits.push_back(q);
    return r;
}

Command conditional(Command body,
                    std::vector<uint32_t> bits,
                    std::vector<bool> values) {
    for (std::size_t i = 0; i < bits.size(); ++i) {
        body.condition_bits.push_back(bits[i]);
        body.condition_values.push_back(values[i]);
    }
    return body;
}

}  // namespace

// ── Command-level invariants (P1.A) ───────────────────────────────────────────

TEST(Trajectory_Command, ConditionRoundTripsThroughDagger) {
    Command c = conditional(Command(GateType::Rx, 0u, Param(0.7)), {0}, {true});
    Command d = c.dagger();
    EXPECT_EQ(d.condition_bits,   c.condition_bits);
    EXPECT_EQ(d.condition_values, c.condition_values);
    // Dagger negates the param; condition is data-flow metadata, not affected.
    EXPECT_NE(d.params[0], c.params[0]);
}

TEST(Trajectory_Command, ConditionRoundTripsThroughSubstitute) {
    Command c = conditional(
        Command(GateType::Rz, 0u, Param::symbol("theta")), {2}, {false});
    Command s = c.substitute({{"theta", 1.5}});
    EXPECT_EQ(s.condition_bits,   c.condition_bits);
    EXPECT_EQ(s.condition_values, c.condition_values);
    EXPECT_TRUE(s.params[0].is_concrete());
}

TEST(Trajectory_Command, ConditionRoundTripsThroughRemap) {
    Command c = conditional(Command(GateType::CX, 0u, 1u), {3}, {true});
    Command r = c.remap_qubits({5, 7});
    EXPECT_EQ(r.condition_bits,   c.condition_bits);
    EXPECT_EQ(r.condition_values, c.condition_values);
    EXPECT_EQ(r.qubits[0], 5u);
    EXPECT_EQ(r.qubits[1], 7u);
}

TEST(Trajectory_Command, EqualityComparesConditions) {
    Command a(GateType::H, 0u);
    Command b(GateType::H, 0u);
    EXPECT_EQ(a, b);

    a = conditional(a, {0}, {true});
    EXPECT_NE(a, b);

    b = conditional(b, {0}, {true});
    EXPECT_EQ(a, b);

    b = conditional(Command(GateType::H, 0u), {0}, {false});
    EXPECT_NE(a, b);
}

// ── Optimisation passes treat conditions as a barrier (P1.C) ──────────────────

TEST(Trajectory_Identities, DoesNotCancelConditionalHHWithDifferentConditions) {
    std::vector<Command> cmds = {
        conditional(Command(GateType::H, 0u), {0}, {true}),
        conditional(Command(GateType::H, 0u), {0}, {false}),
    };
    auto orig = cmds;
    eliminate_identities(cmds);
    EXPECT_EQ(cmds.size(), 2u);  // not cancelled
    EXPECT_EQ(cmds[0], orig[0]);
    EXPECT_EQ(cmds[1], orig[1]);
}

TEST(Trajectory_Identities, CancelsConditionalHHWithSameCondition) {
    std::vector<Command> cmds = {
        conditional(Command(GateType::H, 0u), {0}, {true}),
        conditional(Command(GateType::H, 0u), {0}, {true}),
    };
    auto removed = eliminate_identities(cmds);
    EXPECT_EQ(removed, 2u);
    EXPECT_TRUE(cmds.empty());
}

TEST(Trajectory_Identities, DoesNotMergeConditionalRzWithDifferentConditions) {
    std::vector<Command> cmds = {
        conditional(Command(GateType::Rz, 0u, Param(0.3)), {0}, {true}),
        conditional(Command(GateType::Rz, 0u, Param(0.4)), {1}, {true}),
    };
    eliminate_identities(cmds);
    EXPECT_EQ(cmds.size(), 2u);
}

TEST(Trajectory_Fusion, FlushesAccumulatorOnConditionChange) {
    // Two H gates on qubit 0 with the same condition fuse; an unconditional
    // H breaks the run.
    std::vector<Command> cmds = {
        conditional(Command(GateType::H, 0u), {0}, {true}),
        conditional(Command(GateType::H, 0u), {0}, {true}),
        Command(GateType::H, 0u),                          // unconditional
        Command(GateType::H, 0u),                          // unconditional
    };
    auto out = fuse_single_qubit_gates(cmds);
    // Conditional pair: H·H = I, so the accumulator suppresses on flush.
    // Unconditional pair: same — also suppressed.  Net: empty.
    EXPECT_TRUE(out.empty());
}

TEST(Trajectory_Fusion, ConditionalFusionInheritsCondition) {
    // Two unfusible (in the cancellation sense) gates — H · X — with the same
    // condition produce a single conditional Custom gate.
    std::vector<Command> cmds = {
        conditional(Command(GateType::H, 0u), {3}, {true}),
        conditional(Command(GateType::X, 0u), {3}, {true}),
    };
    auto out = fuse_single_qubit_gates(cmds);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].gate, GateType::Custom);
    ASSERT_EQ(out[0].condition_bits.size(), 1u);
    EXPECT_EQ(out[0].condition_bits[0], 3u);
    EXPECT_TRUE(out[0].condition_values[0]);
}

// ── Transpiler propagates condition through decomposition (P1.C) ──────────────

TEST(Trajectory_Transpile, RuleOutputInheritsSourceCondition) {
    // RZZ → CX · Rz · CX in qulacs target.  Each produced sub-command should
    // carry the source's condition.
    Transpiler t(qulacs_gateset());
    std::vector<Command> input = {
        conditional(Command(GateType::RZZ, 0u, 1u, Param(0.5)), {7}, {true}),
    };
    auto out = t.transpile(input);
    ASSERT_FALSE(out.empty());
    for (const auto& cmd : out) {
        ASSERT_EQ(cmd.condition_bits.size(),   1u) << "missing on " << gate_name(cmd.gate);
        EXPECT_EQ(cmd.condition_bits[0],   7u);
        EXPECT_TRUE(cmd.condition_values[0]);
    }
}

// ── Forced-outcome trajectory dispatch (P1.D, gate_unitaries style) ───────────

TEST(Trajectory_Dispatch, MeasurePlusSplitsHalfHalfAcrossSeeds) {
    // Prepare |+⟩, measure into cbit 0.  Over many seeds the outcome
    // distribution should be ~50/50 within statistical tolerance.
    QarpSimulator sim;
    std::vector<Command> cmds = {
        Command(GateType::H, 0u),
        measure_to(0u, 0u),
    };

    int n0 = 0, n1 = 0;
    constexpr int N = 4000;
    for (uint32_t seed = 0; seed < N; ++seed) {
        auto r = sim.run(cmds, /*n_qubits=*/1, /*n_shots=*/1, seed);
        if (r.cbit_history[0][0]) ++n1; else ++n0;
    }
    // 4σ band on a Bernoulli(0.5, 4000): ~126.  Generous tolerance.
    EXPECT_GT(n0, N/2 - 200);
    EXPECT_LT(n0, N/2 + 200);
}

TEST(Trajectory_Dispatch, ResetCollapsesAnyStateToZero) {
    QarpSimulator sim;
    // Prepare an arbitrary single-qubit state via H · Rz(0.7), reset, then
    // observe — outcome must be 0 every shot.
    std::vector<Command> cmds = {
        Command(GateType::H, 0u),
        Command(GateType::Rz, 0u, Param(0.7)),
        reset_q(0u),
        measure_to(0u, 0u),
    };

    auto r = sim.run(cmds, /*n_qubits=*/1, /*n_shots=*/200, /*seed=*/42u);
    for (const auto& shot : r.cbit_history)
        EXPECT_FALSE(shot[0]);
}

TEST(Trajectory_Dispatch, ConditionalGateFiresOnlyWhenBitMatches) {
    // Prepare |+⟩, measure into cbit 0, then apply X conditioned on cbit==1.
    //   outcome=1 → post-measurement |1⟩, X fires, ends |0⟩.
    //   outcome=0 → post-measurement |0⟩, X skipped, stays |0⟩.
    // Net: every shot ends in |0⟩.
    QarpSimulator sim;
    Command cx; cx.gate = GateType::X;
    cx.qubits.push_back(0u);
    cx.condition_bits.push_back(0u);
    cx.condition_values.push_back(true);

    std::vector<Command> cmds = {
        Command(GateType::H, 0u),
        measure_to(0u, 0u),
        cx,
    };
    auto r = sim.run(cmds, /*n_qubits=*/1, /*n_shots=*/500, /*seed=*/7u);
    EXPECT_EQ(r.counts[0], 500);
    EXPECT_EQ(r.counts.count(1), 0u);
}

TEST(Trajectory_Dispatch, ConditionalSkippedWhenBitDoesNotMatch) {
    // Same shape but condition on cbit==0 — fires on the OPPOSITE half of
    // trajectories.
    //   outcome=1 → post-measurement |1⟩, X skipped, stays |1⟩.
    //   outcome=0 → post-measurement |0⟩, X fires, ends |1⟩.
    // Net: every shot ends in |1⟩.  Confirms the AND-of-equality check looks
    // at *both* the bit and the value, not just the bit.
    QarpSimulator sim;
    Command cx; cx.gate = GateType::X;
    cx.qubits.push_back(0u);
    cx.condition_bits.push_back(0u);
    cx.condition_values.push_back(false);

    std::vector<Command> cmds = {
        Command(GateType::H, 0u),
        measure_to(0u, 0u),
        cx,
    };
    auto r = sim.run(cmds, /*n_qubits=*/1, /*n_shots=*/500, /*seed=*/7u);
    EXPECT_EQ(r.counts[1], 500);
    EXPECT_EQ(r.counts.count(0), 0u);
}

// ── Fast path is unchanged (P1.E, fusion_consistency style) ───────────────────

TEST(Trajectory_FastPath, NoMCMOpsKeepsFastPath) {
    // A circuit without Measure-with-cbit, Reset, or condition_bits goes
    // through the deterministic statevector + marginal-sample path — no
    // cbit_history, n_cbits == 0.
    QarpSimulator sim;
    std::vector<Command> cmds = {
        Command(GateType::H, 0u),
        Command(GateType::CX, 0u, 1u),
    };
    auto r = sim.run(cmds, /*n_qubits=*/2, /*n_shots=*/1024, /*seed=*/1u);
    EXPECT_EQ(r.n_cbits, 0);
    EXPECT_TRUE(r.cbit_history.empty());

    // Bell distribution: only |00⟩ and |11⟩ have non-zero probability.
    EXPECT_GT(r.counts[0], 400);
    EXPECT_GT(r.counts[3], 400);
    EXPECT_EQ(r.counts.count(1), 0u);
    EXPECT_EQ(r.counts.count(2), 0u);
}

TEST(Trajectory_FastPath, MeasureWithoutCbitStaysOnFastPath) {
    // A trailing Measure with no cbit recorded is the "end-of-circuit sample"
    // pattern, which is routed through the fast path.
    QarpSimulator sim;
    Command bare_meas;
    bare_meas.gate = GateType::Measure;
    bare_meas.qubits.push_back(0u);  // no cbits

    std::vector<Command> cmds = {
        Command(GateType::H, 0u),
        bare_meas,
    };
    auto r = sim.run(cmds, /*n_qubits=*/1, /*n_shots=*/1024, /*seed=*/3u);
    EXPECT_EQ(r.n_cbits, 0);
    EXPECT_TRUE(r.cbit_history.empty());
}

TEST(Trajectory_FastPath, StatevectorRejectsTrajectoryCircuit) {
    QarpSimulator sim;
    std::vector<Command> cmds = {
        Command(GateType::H, 0u),
        measure_to(0u, 0u),
    };
    EXPECT_THROW((void)sim.statevector(cmds, 1), std::runtime_error);
}

TEST(Trajectory_FastPath, UnitaryMatrixRejectsTrajectoryCircuit) {
    QarpSimulator sim;
    std::vector<Command> cmds = {
        Command(GateType::H, 0u),
        reset_q(0u),
    };
    EXPECT_THROW((void)sim.unitary_matrix(cmds, 1), std::runtime_error);
}

// ── Prefix/suffix split (P1.E) ────────────────────────────────────────────────

TEST(Trajectory_Prefix, PrefixRunsOnceAndSharesAcrossShots) {
    // The deterministic prefix (H · CX) must produce the same Bell state
    // every shot before the measurement diverges trajectories.  Verify by
    // measuring qubit 0 mid-circuit; cbit_history outcomes should exhibit
    // perfect correlation with the post-shot qubit-1 state.
    QarpSimulator sim;
    std::vector<Command> cmds = {
        Command(GateType::H, 0u),
        Command(GateType::CX, 0u, 1u),
        measure_to(0u, 0u),
    };
    auto r = sim.run(cmds, /*n_qubits=*/2, /*n_shots=*/2000, /*seed=*/11u);
    // Bell correlation: cbit (= qubit 0 outcome) equals the final qubit 1
    // value, so counts should only have keys 0 (=|00⟩) and 3 (=|11⟩).
    EXPECT_EQ(r.counts.count(1), 0u);
    EXPECT_EQ(r.counts.count(2), 0u);
    EXPECT_GT(r.counts[0], 800);
    EXPECT_GT(r.counts[3], 800);
}
