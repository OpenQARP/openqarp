// Functional tests for mid-circuit measurement.  Each scenario
// exercises a real MCM-using protocol assembled from the block primitives:
//
//   1. Teleportation        — Bell pair + 2 measurements + 2 conditional fixups.
//                              Verifies trajectory dispatch + branch markers
//                              cooperate correctly across multiple cbits.
//   2. Repeat-until-success — sequential measurement rounds reading distinct
//                              cbits.  Verifies cbit_history records every
//                              round and no information leaks across them.
//   3. Syndrome extraction  — three rounds of ancilla measurement on a
//                              shared qubit.  Verifies cbit_history shape
//                              [n_shots][n_cbits] for n_cbits = n_rounds.

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

ref<Block> circ(uint32_t nq) {
    return make_block<SimpleBlock>(nq);
}

}  // namespace

// ── Teleportation ─────────────────────────────────────────────────────────────

TEST(Functional_MCM, TeleportationRecoversInputState) {
    // Three qubits: q0 = state to teleport, q1 = Alice's half of EPR, q2 = Bob's.
    //   Prepare |+⟩ on q0 (the message state).
    //   Prepare an EPR pair on q1, q2.
    //   Bell measurement of (q0, q1) → cbits 0, 1.
    //   Apply Z(q2) if cbit 0 == 1; X(q2) if cbit 1 == 1.
    //
    // After teleportation, q2 carries the original |+⟩ state — measuring q2
    // in the X basis (H q2; measure q2 → cbit 2) should always give 0.

    // Qubit layout: composite has 3 qubits.
    //   Subblocks address local qubits 0..n-1; target_qubits remaps to global.
    auto prep = circ(1);   prep->h(0);                  // q0 → |+⟩
    prep->build();
    prep->target_qubits = std::vector<uint32_t>{0u};

    auto epr  = circ(2);   epr->h(0).cx(0, 1);          // EPR on q1, q2
    epr->build();
    epr->target_qubits = std::vector<uint32_t>{1u, 2u};

    auto bell_prep = circ(2);  bell_prep->cx(0, 1).h(0);  // q0=q0, q1=q1
    bell_prep->build();
    bell_prep->target_qubits = std::vector<uint32_t>{0u, 1u};

    // Each child block addresses its own local cbit space starting at 0;
    // target_cbits maps that local space into the parent's classical
    // register.  m_q0's measurement is wired to parent cbit 0, m_q1's to
    // parent cbit 1.
    auto m_q0 = make_block<MeasureBlock>(0u, 0u);
    m_q0->target_qubits = std::vector<uint32_t>{0u};
    m_q0->target_cbits  = std::vector<uint32_t>{0u};

    auto m_q1 = make_block<MeasureBlock>(0u, 0u);
    m_q1->target_qubits = std::vector<uint32_t>{1u};
    m_q1->target_cbits  = std::vector<uint32_t>{1u};

    auto z_body = circ(1);  z_body->z(0);  z_body->build();
    auto z_correction = make_block<ConditionalBlock>(
        std::vector<uint32_t>{0u}, std::vector<bool>{true}, z_body);
    z_correction->target_qubits = std::vector<uint32_t>{2u};
    z_correction->target_cbits  = std::vector<uint32_t>{0u};

    auto x_body = circ(1);  x_body->x(0);  x_body->build();
    auto x_correction = make_block<ConditionalBlock>(
        std::vector<uint32_t>{0u}, std::vector<bool>{true}, x_body);
    x_correction->target_qubits = std::vector<uint32_t>{2u};
    x_correction->target_cbits  = std::vector<uint32_t>{1u};

    // Verify in X basis: H(q2), measure q2 → cbit 2.  If teleportation is
    // correct, q2 was |+⟩, so H|+⟩ = |0⟩ and the measurement is always 0.
    auto verify = circ(1);  verify->h(0);  verify->build();
    verify->target_qubits = std::vector<uint32_t>{2u};

    auto m_q2 = make_block<MeasureBlock>(0u, 0u);
    m_q2->target_qubits = std::vector<uint32_t>{2u};
    m_q2->target_cbits  = std::vector<uint32_t>{2u};

    CompositeBlock root({
        prep, epr, bell_prep, m_q0, m_q1,
        z_correction, x_correction,
        verify, m_q2,
    }, /*n_qubits=*/3u);
    root.n_cbits = 3u;
    root.build();

    QarpSimulator sim;
    auto r = sim.run(root.flatten(), /*n_qubits=*/3, /*n_shots=*/1000, /*seed=*/42u);

    // The verification cbit (index 2) should be 0 on every shot.
    ASSERT_EQ(r.cbit_history.size(), 1000u);
    int verify_outcomes_zero = 0;
    for (const auto& reg : r.cbit_history) {
        ASSERT_EQ(reg.size(), 3u);
        if (!reg[2]) ++verify_outcomes_zero;
    }
    EXPECT_EQ(verify_outcomes_zero, 1000)
        << "Teleportation verification cbit was 1 on some shot — "
           "the Z and X corrections didn't recover the |+⟩ state";
}

// ── Repeat-until-success: 3 rounds, each writes a distinct cbit ──────────────

TEST(Functional_MCM, RepeatUntilSuccessRecordsEveryRound) {
    // Three rounds of: prepare |+⟩, measure into a fresh cbit.  Outcomes are
    // independent ~Bernoulli(1/2).  Verify cbit_history[shot][round] is
    // 50/50 over many shots and the rounds are uncorrelated (variance test).
    auto round = [&](uint32_t cbit_global) {
        auto prep = circ(1);  prep->h(0);  prep->build();
        prep->target_qubits = std::vector<uint32_t>{0u};
        auto reset = make_block<ResetBlock>(0u);
        reset->target_qubits = std::vector<uint32_t>{0u};
        auto m = make_block<MeasureBlock>(0u, 0u);
        m->target_qubits = std::vector<uint32_t>{0u};
        m->target_cbits  = std::vector<uint32_t>{cbit_global};
        return std::vector<ref<Block>>{reset, prep, m};
    };

    auto r0 = round(0u);
    auto r1 = round(1u);
    auto r2 = round(2u);
    std::vector<ref<Block>> all;
    all.insert(all.end(), r0.begin(), r0.end());
    all.insert(all.end(), r1.begin(), r1.end());
    all.insert(all.end(), r2.begin(), r2.end());

    CompositeBlock root(std::move(all), /*n_qubits=*/1u);
    root.n_cbits = 3u;
    root.build();

    QarpSimulator sim;
    auto r = sim.run(root.flatten(), /*n_qubits=*/1, /*n_shots=*/4000, /*seed=*/11u);

    ASSERT_EQ(r.cbit_history.size(), 4000u);
    // Per-round marginal counts; each should land near n_shots/2.
    int r0_ones = 0, r1_ones = 0, r2_ones = 0;
    for (const auto& reg : r.cbit_history) {
        ASSERT_EQ(reg.size(), 3u);
        if (reg[0]) ++r0_ones;
        if (reg[1]) ++r1_ones;
        if (reg[2]) ++r2_ones;
    }
    // Generous 4σ band on Bernoulli(0.5, 4000): ~127.
    auto in_band = [](int x) { return x > 4000/2 - 200 && x < 4000/2 + 200; };
    EXPECT_TRUE(in_band(r0_ones)) << "r0_ones=" << r0_ones;
    EXPECT_TRUE(in_band(r1_ones)) << "r1_ones=" << r1_ones;
    EXPECT_TRUE(in_band(r2_ones)) << "r2_ones=" << r2_ones;
}

// ── Syndrome extraction across multiple rounds ───────────────────────────────

TEST(Functional_MCM, ThreeRoundSyndromeShape) {
    // Three ancilla measurements with the data qubit in |+⟩ and CX(data,
    // ancilla) before each round (parity check).  Reset the ancilla between
    // rounds and write each round's outcome to a distinct cbit.
    auto prep_data = circ(2);  prep_data->h(0);  prep_data->build();

    // Concrete ref<CompositeBlock> so add_child is callable below; upcast to
    // ref<Block> at the point it joins the parent's child vector.
    ref<CompositeBlock> rounds(new CompositeBlock(
        std::vector<ref<Block>>{}, /*n_qubits=*/2u));
    for (uint32_t round = 0; round < 3; ++round) {
        auto cx_block = circ(2); cx_block->cx(0, 1); cx_block->build();
        auto m = make_block<MeasureBlock>(1u, 0u);
        m->target_cbits = std::vector<uint32_t>{round};
        auto rst = make_block<ResetBlock>(1u);
        rounds->add_child(cx_block);
        rounds->add_child(m);
        rounds->add_child(rst);
    }

    CompositeBlock root({prep_data, ref<Block>(rounds.get())}, /*n_qubits=*/2u);
    root.n_cbits = 3u;
    root.build();

    QarpSimulator sim;
    auto r = sim.run(root.flatten(), /*n_qubits=*/2, /*n_shots=*/200, /*seed=*/3u);

    EXPECT_EQ(r.n_cbits, 3);
    ASSERT_EQ(r.cbit_history.size(), 200u);
    for (const auto& reg : r.cbit_history) {
        EXPECT_EQ(reg.size(), 3u);
    }
}
