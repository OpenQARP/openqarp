// Direct tests of the shared classical-register width helper
// (`cbit_register_width`, core/command.h) — the single definition the
// simulator, the OpenQASM 3 emitter and `SimpleBlock::build()` all use.
//
// Oracle throughout: hand-computed widths (1 + largest referenced cbit index).

#include <gtest/gtest.h>
#include "qarpx/qarpx.h"

#include <vector>

using namespace qarpx;

namespace {

Command measure(uint32_t qubit, uint32_t cbit) {
    Command cmd;
    cmd.gate = GateType::Measure;
    cmd.qubits = {qubit};
    cmd.cbits = {cbit};
    return cmd;
}

/// A Measure carrying no classical target — malformed per §8.
Command measure_no_cbit(uint32_t qubit) {
    Command cmd;
    cmd.gate = GateType::Measure;
    cmd.qubits = {qubit};
    return cmd;
}

Command branch_begin(std::initializer_list<uint32_t> cbits) {
    Command cmd;
    cmd.gate = GateType::BranchBegin;
    // condition_bits / condition_values are SmallVector, not std::vector.
    for (auto c : cbits) {
        cmd.condition_bits.push_back(c);
        cmd.condition_values.push_back(true);
    }
    return cmd;
}

Command gate_x(uint32_t qubit) {
    Command cmd;
    cmd.gate = GateType::X;
    cmd.qubits = {qubit};
    return cmd;
}

}  // namespace

TEST(CbitRegisterWidth, EmptyStreamIsZero) {
    EXPECT_EQ(cbit_register_width({}), 0u);
}

TEST(CbitRegisterWidth, GateOnlyStreamIsZero) {
    EXPECT_EQ(cbit_register_width({gate_x(0), gate_x(3)}), 0u);
}

TEST(CbitRegisterWidth, ContiguousMeasuresCountToMaxPlusOne) {
    const std::vector<Command> cmds = {measure(0, 0), measure(1, 1), measure(2, 2)};
    EXPECT_EQ(cbit_register_width(cmds), 3u);
}

TEST(CbitRegisterWidth, PermutedAndSparseCbitsUseTheLargestIndex) {
    // Writes c3 then c0: width is 4, not 2 (count of measurements).
    const std::vector<Command> cmds = {measure(0, 3), measure(1, 0)};
    EXPECT_EQ(cbit_register_width(cmds), 4u);
}

TEST(CbitRegisterWidth, ConditionOnlyStreamCountsConditionBits) {
    // No Measure at all — a branch reading c2 still needs a 3-bit register.
    const std::vector<Command> cmds = {branch_begin({2})};
    EXPECT_EQ(cbit_register_width(cmds), 3u);
}

TEST(CbitRegisterWidth, ConditionIndexAboveAnyMeasuredCbitWins) {
    const std::vector<Command> cmds = {measure(0, 0), branch_begin({4})};
    EXPECT_EQ(cbit_register_width(cmds), 5u);
}

TEST(CbitRegisterWidth, RepeatedWritesToOneCbitDoNotDoubleCount) {
    const std::vector<Command> cmds = {measure(0, 1), measure(1, 1), measure(2, 1)};
    EXPECT_EQ(cbit_register_width(cmds), 2u);
}

TEST(CbitRegisterWidth, MeasureWithEmptyCbitsContributesNothing) {
    // Decided semantics: the cbit is part of the Measure contract (§8), so a
    // Measure without one is malformed and does NOT widen the register — in
    // particular it is not treated as a qubit-indexed default.
    EXPECT_EQ(cbit_register_width({measure_no_cbit(3)}), 0u);

    // ...and it does not disturb a width established by well-formed commands.
    const std::vector<Command> mixed = {measure(0, 0), measure_no_cbit(7)};
    EXPECT_EQ(cbit_register_width(mixed), 1u);
}
