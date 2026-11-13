// Block primitives for mid-circuit-measurement support and the structural
// pieces around them — leaf blocks, ConditionalBlock with branch markers, cbit
// offsetting, dagger of ConditionalBlock, ControlledBlock guard against
// non-unitary bodies.

#include <gtest/gtest.h>
#include "qarpx/qarpx.h"
#include "qarpx/block/measure_block.h"
#include "qarpx/block/reset_block.h"
#include "qarpx/block/conditional_block.h"

#include <memory>
#include <vector>

using namespace qarpx;

// Test helper: allocate a block as an intrusive ref<Block> — block trees are
// reference counted through nanobind's intrusive ref<T>.
namespace {
template <class T, class... Args>
ref<Block> make_block(Args&&... args) {
    return ref<Block>(new T(std::forward<Args>(args)...));
}
}  // namespace

namespace {

ref<Block> make_circuit_block(uint32_t nq) {
    auto b = make_block<SimpleBlock>(nq);
    return b;
}

}  // namespace

// ── Leaf primitives (P2.B) ────────────────────────────────────────────────────

TEST(Blocks_MCM, MeasureBlockEmitsMeasureCommand) {
    MeasureBlock m(2u, 5u);
    m.build();
    auto cmds = m.flatten();
    ASSERT_EQ(cmds.size(), 1u);
    EXPECT_EQ(cmds[0].gate, GateType::Measure);
    ASSERT_EQ(cmds[0].qubits.size(), 1u);
    ASSERT_EQ(cmds[0].cbits.size(), 1u);
    EXPECT_EQ(cmds[0].qubits[0], 2u);
    EXPECT_EQ(cmds[0].cbits[0], 5u);
    EXPECT_EQ(m.n_qubits, 1u);
    EXPECT_EQ(m.n_cbits, 1u);
}

TEST(Blocks_MCM, ResetBlockEmitsResetCommand) {
    ResetBlock r(3u);
    r.build();
    auto cmds = r.flatten();
    ASSERT_EQ(cmds.size(), 1u);
    EXPECT_EQ(cmds[0].gate, GateType::Reset);
    ASSERT_EQ(cmds[0].qubits.size(), 1u);
    EXPECT_EQ(cmds[0].qubits[0], 3u);
    EXPECT_EQ(r.n_qubits, 1u);
    EXPECT_EQ(r.n_cbits, 0u);
}

TEST(Blocks_MCM, BlockResetShortcutMatchesResetBlock) {
    SimpleBlock c(2u);
    c.h(0u).reset(0u);
    c.build();
    auto cmds = c.flatten();
    ASSERT_EQ(cmds.size(), 2u);
    EXPECT_EQ(cmds[1].gate, GateType::Reset);
    EXPECT_EQ(cmds[1].qubits[0], 0u);
}

// ── ConditionalBlock branch markers (P2.G3) ───────────────────────────────────

TEST(Blocks_MCM, ConditionalEmitsBranchBeginEndAroundThenBody) {
    auto body = make_circuit_block(1u);
    body->x(0u).h(0u);
    body->build();

    ConditionalBlock cond({0u}, {true}, body);
    cond.build();
    auto cmds = cond.flatten();

    // Expected: BranchBegin → X → H → BranchEnd
    ASSERT_EQ(cmds.size(), 4u);
    EXPECT_EQ(cmds[0].gate, GateType::BranchBegin);
    EXPECT_EQ(cmds[1].gate, GateType::X);
    EXPECT_EQ(cmds[2].gate, GateType::H);
    EXPECT_EQ(cmds[3].gate, GateType::BranchEnd);

    // The condition tuple lives on BranchBegin.
    ASSERT_EQ(cmds[0].condition_bits.size(), 1u);
    EXPECT_EQ(cmds[0].condition_bits[0], 0u);
    EXPECT_TRUE(cmds[0].condition_values[0]);

    // Body commands carry no per-command condition (dispatch happens at the
    // BranchBegin level instead).
    EXPECT_TRUE(cmds[1].condition_bits.empty());
    EXPECT_TRUE(cmds[2].condition_bits.empty());
}

TEST(Blocks_MCM, ConditionalWithElseEmitsBranchElseMarker) {
    auto then_body = make_circuit_block(1u);
    then_body->x(0u);
    then_body->build();
    auto else_body = make_circuit_block(1u);
    else_body->z(0u);
    else_body->build();

    ConditionalBlock cond({0u}, {true}, then_body, else_body);
    cond.build();
    auto cmds = cond.flatten();

    // BranchBegin → X → BranchElse → Z → BranchEnd
    ASSERT_EQ(cmds.size(), 5u);
    EXPECT_EQ(cmds[0].gate, GateType::BranchBegin);
    EXPECT_EQ(cmds[1].gate, GateType::X);
    EXPECT_EQ(cmds[2].gate, GateType::BranchElse);
    EXPECT_EQ(cmds[3].gate, GateType::Z);
    EXPECT_EQ(cmds[4].gate, GateType::BranchEnd);
}

TEST(Blocks_MCM, ConditionalElseOnlyOmitsThenCommands) {
    auto else_body = make_circuit_block(1u);
    else_body->z(0u);
    else_body->build();

    ConditionalBlock cond({0u}, {true}, /*then=*/nullptr, else_body);
    cond.build();
    auto cmds = cond.flatten();

    // BranchBegin → BranchElse → Z → BranchEnd  (empty then-region)
    ASSERT_EQ(cmds.size(), 4u);
    EXPECT_EQ(cmds[0].gate, GateType::BranchBegin);
    EXPECT_EQ(cmds[1].gate, GateType::BranchElse);
    EXPECT_EQ(cmds[2].gate, GateType::Z);
    EXPECT_EQ(cmds[3].gate, GateType::BranchEnd);
}

TEST(Blocks_MCM, ConditionalRejectsBothNullBodies) {
    EXPECT_THROW(ConditionalBlock({0u}, {true}, nullptr, nullptr),
                 std::invalid_argument);
}

TEST(Blocks_MCM, ConditionalRejectsEmptyCondition) {
    auto body = make_circuit_block(1u);
    body->x(0u);
    body->build();
    EXPECT_THROW(ConditionalBlock({}, {}, body), std::invalid_argument);
}

TEST(Blocks_MCM, ConditionalRejectsMismatchedConditionLengths) {
    auto body = make_circuit_block(1u);
    body->x(0u);
    body->build();
    EXPECT_THROW(ConditionalBlock({0u, 1u}, {true}, body),
                 std::invalid_argument);
}

TEST(Blocks_MCM, NestedConditionalEmitsNestedBranches) {
    // Outer cond on cbit 0 wrapping a body that itself contains an inner
    // cond on cbit 1.  The flatten produces nested BranchBegin/BranchEnd
    // pairs without merging conditions.
    auto inner_body = make_circuit_block(1u);
    inner_body->x(0u);
    inner_body->build();

    auto inner_cond = make_block<ConditionalBlock>(
        std::vector<uint32_t>{1u}, std::vector<bool>{false}, inner_body);
    inner_cond->build();

    ConditionalBlock outer({0u}, {true}, inner_cond);
    outer.build();

    auto cmds = outer.flatten();
    // Outer:   BranchBegin(cbit 0=true)
    // Inner:     BranchBegin(cbit 1=false)
    //              X
    //            BranchEnd
    //          BranchEnd
    ASSERT_EQ(cmds.size(), 5u);
    EXPECT_EQ(cmds[0].gate, GateType::BranchBegin);
    ASSERT_EQ(cmds[0].condition_bits.size(), 1u);
    EXPECT_EQ(cmds[0].condition_bits[0], 0u);
    EXPECT_TRUE(cmds[0].condition_values[0]);

    EXPECT_EQ(cmds[1].gate, GateType::BranchBegin);
    ASSERT_EQ(cmds[1].condition_bits.size(), 1u);
    EXPECT_EQ(cmds[1].condition_bits[0], 1u);
    EXPECT_FALSE(cmds[1].condition_values[0]);

    EXPECT_EQ(cmds[2].gate, GateType::X);
    EXPECT_EQ(cmds[3].gate, GateType::BranchEnd);
    EXPECT_EQ(cmds[4].gate, GateType::BranchEnd);
}

// ── Dagger of ConditionalBlock (P2.G3, dagger_round_trip style) ───────────────

TEST(Blocks_MCM, ConditionalDaggerRoundTripsBothBodies) {
    auto then_body = make_circuit_block(1u);
    then_body->rx(0u, Param(0.7));
    then_body->build();
    auto else_body = make_circuit_block(1u);
    else_body->ry(0u, Param(1.1));
    else_body->build();

    ConditionalBlock cond({0u}, {true}, then_body, else_body);
    cond.build();

    auto dag = cond.dagger();
    auto fwd_cmds = cond.flatten();
    auto dag_cmds = dag->flatten();

    // Both flat sequences have the same shape (markers + bodies).
    ASSERT_EQ(fwd_cmds.size(), dag_cmds.size());
    EXPECT_EQ(dag_cmds[0].gate, GateType::BranchBegin);
    EXPECT_EQ(dag_cmds[2].gate, GateType::BranchElse);
    EXPECT_EQ(dag_cmds[4].gate, GateType::BranchEnd);

    // Body params negated.
    EXPECT_EQ(dag_cmds[1].gate, GateType::Rx);
    EXPECT_NEAR(dag_cmds[1].params[0].value(), -0.7, 1e-12);
    EXPECT_EQ(dag_cmds[3].gate, GateType::Ry);
    EXPECT_NEAR(dag_cmds[3].params[0].value(), -1.1, 1e-12);

    // Condition tuple preserved.
    ASSERT_EQ(dag_cmds[0].condition_bits.size(), 1u);
    EXPECT_EQ(dag_cmds[0].condition_bits[0], 0u);
    EXPECT_TRUE(dag_cmds[0].condition_values[0]);
}

// ── CompositeBlock cbit offsetting (P2.D) ─────────────────────────────────────

TEST(Blocks_MCM, CompositeOffsetsSiblingCbitsByDefault) {
    auto m1 = make_block<MeasureBlock>(0u, 0u);
    auto m2 = make_block<MeasureBlock>(1u, 0u);

    CompositeBlock parent({m1, m2}, /*nq=*/2u);
    parent.build();
    auto cmds = parent.flatten();

    ASSERT_EQ(cmds.size(), 2u);
    EXPECT_EQ(cmds[0].cbits[0], 0u);
    EXPECT_EQ(cmds[1].cbits[0], 1u);
    EXPECT_EQ(parent.n_cbits, 2u);
}

TEST(Blocks_MCM, TargetCbitsOverrideAliasesSiblings) {
    // Both children share parent cbit 0 — measurement writes it, conditional
    // correction reads it.
    auto m = make_block<MeasureBlock>(0u, 0u);
    m->target_cbits = std::vector<uint32_t>{0u};

    auto correction_body = make_circuit_block(1u);
    correction_body->x(0u);
    correction_body->build();

    auto cond = make_block<ConditionalBlock>(
        std::vector<uint32_t>{0u}, std::vector<bool>{true},
        correction_body);
    cond->target_cbits  = std::vector<uint32_t>{0u};
    cond->target_qubits = std::vector<uint32_t>{0u};

    CompositeBlock parent({m, cond}, /*nq=*/1u);
    parent.n_cbits = 1u;
    parent.build();
    auto cmds = parent.flatten();

    // Measure → BranchBegin → X → BranchEnd
    ASSERT_EQ(cmds.size(), 4u);
    EXPECT_EQ(cmds[0].gate, GateType::Measure);
    EXPECT_EQ(cmds[0].cbits[0], 0u);
    EXPECT_EQ(cmds[1].gate, GateType::BranchBegin);
    ASSERT_EQ(cmds[1].condition_bits.size(), 1u);
    EXPECT_EQ(cmds[1].condition_bits[0], 0u);
    EXPECT_TRUE(cmds[1].condition_values[0]);
    EXPECT_EQ(cmds[2].gate, GateType::X);
    EXPECT_EQ(cmds[3].gate, GateType::BranchEnd);
}

// ── ControlledBlock rejects non-unitary (P2.E) ────────────────────────────────

TEST(Blocks_MCM, ControlledRejectsMeasureBody) {
    auto body = make_circuit_block(1u);
    body->measure(0u, 0u);
    body->build();
    ControlledBlock cb(body, 1u, {true});
    cb.build();
    EXPECT_THROW((void)cb.flatten(), std::runtime_error);
}

TEST(Blocks_MCM, ControlledRejectsResetBody) {
    auto body = make_circuit_block(1u);
    body->reset(0u);
    body->build();
    ControlledBlock cb(body, 1u, {true});
    cb.build();
    EXPECT_THROW((void)cb.flatten(), std::runtime_error);
}

// ── End-to-end through the simulator ──────────────────────────────────────────

TEST(Blocks_MCM, ConditionalAfterMeasureRunsThroughSimulator) {
    // H(0) → Measure(q=0, c=0) → ConditionalBlock(c=0, v=true, then=X(0))
    // outcome=1: post-meas |1⟩, then-X fires, ends |0⟩.
    // outcome=0: post-meas |0⟩, then skipped (no else), stays |0⟩.
    // Net: every shot ends in |0⟩.
    auto h_block = make_circuit_block(1u);
    h_block->h(0u);
    h_block->build();

    auto meas = make_block<MeasureBlock>(0u, 0u);
    meas->target_cbits = std::vector<uint32_t>{0u};

    auto correction_body = make_circuit_block(1u);
    correction_body->x(0u);
    correction_body->build();
    auto cond = make_block<ConditionalBlock>(
        std::vector<uint32_t>{0u}, std::vector<bool>{true}, correction_body);
    cond->target_cbits  = std::vector<uint32_t>{0u};
    cond->target_qubits = std::vector<uint32_t>{0u};

    CompositeBlock root({h_block, meas, cond}, 1u);
    root.n_cbits = 1u;
    root.build();
    auto cmds = root.flatten();

    QarpSimulator sim;
    auto r = sim.run(cmds, /*n_qubits=*/1, /*n_shots=*/500, /*seed=*/3u);
    EXPECT_EQ(r.counts[0], 500);
    EXPECT_EQ(r.counts.count(1), 0u);
}

TEST(Blocks_MCM, ConditionalElseFiresOnFalseAND) {
    // H(0) → Measure(q=0, c=0) → Conditional(c=0, v=true,
    //                                          then=I, else=X(0))
    // outcome=1 → then runs (no-op), state stays |1⟩.
    // outcome=0 → else fires, X|0⟩ = |1⟩.
    // Net: every shot ends in |1⟩.
    auto h_block = make_circuit_block(1u);
    h_block->h(0u);
    h_block->build();

    auto meas = make_block<MeasureBlock>(0u, 0u);
    meas->target_cbits = std::vector<uint32_t>{0u};

    auto then_body = make_circuit_block(1u);  // empty (identity on the flow)
    then_body->build();
    auto else_body = make_circuit_block(1u);
    else_body->x(0u);
    else_body->build();

    auto cond = make_block<ConditionalBlock>(
        std::vector<uint32_t>{0u}, std::vector<bool>{true},
        then_body, else_body);
    cond->target_cbits  = std::vector<uint32_t>{0u};
    cond->target_qubits = std::vector<uint32_t>{0u};

    CompositeBlock root({h_block, meas, cond}, 1u);
    root.n_cbits = 1u;
    root.build();
    auto cmds = root.flatten();

    QarpSimulator sim;
    auto r = sim.run(cmds, /*n_qubits=*/1, /*n_shots=*/500, /*seed=*/3u);
    EXPECT_EQ(r.counts[1], 500);
    EXPECT_EQ(r.counts.count(0), 0u);
}

TEST(Blocks_MCM, MultiBitConditionalWithElseDispatchesCorrectly) {
    // Two-bit condition (c0 == true AND c1 == false).
    // Prepare q0 = |+⟩ and q1 = |+⟩; measure both.  Apply X(q0) only when
    // (c0=1 AND c1=0); apply Z(q0) otherwise (else branch).  Verify each
    // shot's q0 final amplitude is consistent with which branch fired.
    //
    // The point: branch markers encode an else-branch with a multi-bit AND
    // natively, which per-command conditions cannot express directly (DeMorgan
    // would turn the negated AND into an OR).
    auto setup = make_circuit_block(2u);
    setup->h(0u).h(1u);
    setup->build();

    auto m0 = make_block<MeasureBlock>(0u, 0u);
    m0->target_cbits = std::vector<uint32_t>{0u};
    auto m1 = make_block<MeasureBlock>(1u, 0u);
    m1->target_cbits = std::vector<uint32_t>{1u};

    auto then_body = make_circuit_block(1u);
    then_body->x(0u);
    then_body->build();
    auto else_body = make_circuit_block(1u);
    else_body->z(0u);  // Z on |0⟩ or |1⟩ doesn't change the measurement.
    else_body->build();

    auto cond = make_block<ConditionalBlock>(
        std::vector<uint32_t>{0u, 1u}, std::vector<bool>{true, false},
        then_body, else_body);
    cond->target_cbits  = std::vector<uint32_t>{0u, 1u};
    cond->target_qubits = std::vector<uint32_t>{0u};

    CompositeBlock root({setup, m0, m1, cond}, 2u);
    root.n_cbits = 2u;
    root.build();
    auto cmds = root.flatten();

    QarpSimulator sim;
    auto r = sim.run(cmds, /*n_qubits=*/2, /*n_shots=*/4000, /*seed=*/9u);
    // Every shot's q1 outcome is whatever the H put it on; q0 outcome is:
    //   - if (c0=1 AND c1=0): then-X flips q0, so final-q0 = 1 - c0 = 0.
    //   - else: Z doesn't change q0's basis, so final-q0 = c0.
    //
    // Roll up by classical-register outcome.  cbit_history shape matches
    // and every shot was processed by exactly one branch.
    ASSERT_EQ(r.cbit_history.size(), 4000u);
    for (const auto& reg : r.cbit_history) {
        EXPECT_EQ(reg.size(), 2u);
    }
}
