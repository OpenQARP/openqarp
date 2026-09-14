// Direct tests of the shared gate-count helpers (`n_nqb_gates`,
// `n_gates_of_type`, core/command.h) — the resource-counting queries behind
// `Block.n_1q_gates()` / `n_2q_gates()` / `n_nqb_gates()` / `n_gates_of_type()`.
//
// Oracle throughout: hand-computed counts over explicitly-constructed
// command streams.

#include <gtest/gtest.h>
#include "qarpx/qarpx.h"

#include <vector>

using namespace qarpx;

namespace {

Command gate1(GateType g, uint32_t q) {
    Command cmd;
    cmd.gate = g;
    cmd.qubits = {q};
    return cmd;
}

Command gate2(GateType g, uint32_t q0, uint32_t q1) {
    Command cmd;
    cmd.gate = g;
    cmd.qubits = {q0, q1};
    return cmd;
}

Command ccx(uint32_t c0, uint32_t c1, uint32_t t) {
    Command cmd;
    cmd.gate = GateType::CCX;
    cmd.qubits = {c0, c1, t};
    return cmd;
}

Command measure(uint32_t qubit, uint32_t cbit) {
    Command cmd;
    cmd.gate = GateType::Measure;
    cmd.qubits = {qubit};
    cmd.cbits = {cbit};
    return cmd;
}

Command reset(uint32_t qubit) {
    Command cmd;
    cmd.gate = GateType::Reset;
    cmd.qubits = {qubit};
    return cmd;
}

Command barrier(std::initializer_list<uint32_t> qs) {
    Command cmd;
    cmd.gate = GateType::Barrier;
    for (auto q : qs) cmd.qubits.push_back(q);
    return cmd;
}

Command gphase(double theta) {
    Command cmd;
    cmd.gate = GateType::GPhase;
    cmd.params = {Param(theta)};
    return cmd;
}

Command branch_begin(std::initializer_list<uint32_t> cbits) {
    Command cmd;
    cmd.gate = GateType::BranchBegin;
    for (auto c : cbits) {
        cmd.condition_bits.push_back(c);
        cmd.condition_values.push_back(true);
    }
    return cmd;
}

Command branch_end() {
    Command cmd;
    cmd.gate = GateType::BranchEnd;
    return cmd;
}

}  // namespace

TEST(GateIsPhysical, ExcludesNonGateSpecials) {
    EXPECT_FALSE(gate_is_physical(GateType::Barrier));
    EXPECT_FALSE(gate_is_physical(GateType::Measure));
    EXPECT_FALSE(gate_is_physical(GateType::Reset));
    EXPECT_FALSE(gate_is_physical(GateType::GPhase));
    EXPECT_FALSE(gate_is_physical(GateType::BranchBegin));
    EXPECT_FALSE(gate_is_physical(GateType::BranchElse));
    EXPECT_FALSE(gate_is_physical(GateType::BranchEnd));
}

TEST(GateIsPhysical, IncludesOrdinaryGates) {
    EXPECT_TRUE(gate_is_physical(GateType::H));
    EXPECT_TRUE(gate_is_physical(GateType::CX));
    EXPECT_TRUE(gate_is_physical(GateType::CCX));
    EXPECT_TRUE(gate_is_physical(GateType::Custom));
}

TEST(NPhysicalGates, TotalsAllAritiesWithSameExclusions) {
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::H, 0);
    cmds.emplace_back(GateType::CX, 0, 1);
    Command ccx(GateType::CCX, 0);
    ccx.qubits = {0, 1, 2};
    cmds.push_back(ccx);
    Command barrier(GateType::Barrier, 0);
    cmds.push_back(barrier);
    Command meas(GateType::Measure, 0, 0u);
    cmds.push_back(meas);
    EXPECT_EQ(n_physical_gates(cmds),
              n_nqb_gates(cmds, 1) + n_nqb_gates(cmds, 2) + n_nqb_gates(cmds, 3));
    EXPECT_EQ(n_physical_gates(cmds), 3u);
}

TEST(NNqbGates, EmptyStreamIsZero) {
    EXPECT_EQ(n_nqb_gates({}, 1), 0u);
    EXPECT_EQ(n_nqb_gates({}, 2), 0u);
}

TEST(NNqbGates, CountsByActualQubitFootprint) {
    const std::vector<Command> cmds = {
        gate1(GateType::H, 0), gate1(GateType::H, 1),
        gate2(GateType::CX, 0, 1),
        gate1(GateType::Rz, 0),
    };
    EXPECT_EQ(n_nqb_gates(cmds, 1), 3u);  // H, H, Rz
    EXPECT_EQ(n_nqb_gates(cmds, 2), 1u);  // CX
    EXPECT_EQ(n_nqb_gates(cmds, 3), 0u);
}

TEST(NNqbGates, ThreeQubitGateLandsInItsOwnBucket) {
    const std::vector<Command> cmds = {ccx(0, 1, 2)};
    EXPECT_EQ(n_nqb_gates(cmds, 1), 0u);
    EXPECT_EQ(n_nqb_gates(cmds, 2), 0u);
    EXPECT_EQ(n_nqb_gates(cmds, 3), 1u);
}

TEST(NNqbGates, ExcludesBarrierMeasureResetGPhaseAndBranchMarkers) {
    // A 1-qubit Barrier could be miscounted as a "1-qubit gate" by qubit
    // count alone; gate_is_physical is what rules it out (matches the
    // fusion.cpp precedent for the same misclassification risk).
    const std::vector<Command> cmds = {
        gate1(GateType::H, 0), gate1(GateType::H, 1),
        gate2(GateType::CX, 0, 1),
        measure(0, 0),
        reset(1),
        gphase(0.5),
        barrier({0}),
        branch_begin({0}),
        branch_end(),
    };
    EXPECT_EQ(n_nqb_gates(cmds, 1), 2u);  // H, H only
    EXPECT_EQ(n_nqb_gates(cmds, 2), 1u);  // CX
}

TEST(NGatesOfType, EmptyStreamIsZero) {
    EXPECT_EQ(n_gates_of_type({}, GateType::H), 0u);
}

TEST(NGatesOfType, CountsExactTypeMatches) {
    const std::vector<Command> cmds = {
        gate1(GateType::H, 0), gate1(GateType::H, 1),
        gate2(GateType::CX, 0, 1),
        gate1(GateType::Rz, 0),
    };
    EXPECT_EQ(n_gates_of_type(cmds, GateType::H), 2u);
    EXPECT_EQ(n_gates_of_type(cmds, GateType::CX), 1u);
    EXPECT_EQ(n_gates_of_type(cmds, GateType::Rz), 1u);
    EXPECT_EQ(n_gates_of_type(cmds, GateType::Y), 0u);
}

TEST(NGatesOfType, UnfilteredUnlikeNNqbGates) {
    // Barrier/Measure/Reset/GPhase/branch markers are ordinary GateTypes
    // here — the exclusion is n_nqb_gates-specific, not global.
    const std::vector<Command> cmds = {
        measure(0, 0), reset(1), gphase(0.5), barrier({0}),
        branch_begin({0}), branch_end(),
    };
    EXPECT_EQ(n_gates_of_type(cmds, GateType::Measure), 1u);
    EXPECT_EQ(n_gates_of_type(cmds, GateType::Reset), 1u);
    EXPECT_EQ(n_gates_of_type(cmds, GateType::GPhase), 1u);
    EXPECT_EQ(n_gates_of_type(cmds, GateType::Barrier), 1u);
    EXPECT_EQ(n_gates_of_type(cmds, GateType::BranchBegin), 1u);
    EXPECT_EQ(n_gates_of_type(cmds, GateType::BranchEnd), 1u);
}
