// Caller-supplied initial states on QarpSimulator::statevector / run.
// Oracles are
// hand-computed analytic amplitudes in the LSB convention (§1): qubit q is
// bit q of the amplitude index.

#include <gtest/gtest.h>
#include "qarpx/qarpx.h"

#include <complex>
#include <optional>
#include <stdexcept>
#include <vector>

using namespace qarpx;

namespace {

using cd = std::complex<double>;

constexpr double kInvSqrt2 = 0.70710678118654752440;

std::vector<cd> basis_state(std::size_t dim, std::size_t idx) {
    std::vector<cd> psi(dim, cd{0.0, 0.0});
    psi[idx] = {1.0, 0.0};
    return psi;
}

Command measure_to(uint32_t q, uint32_t c) {
    Command m;
    m.gate = GateType::Measure;
    m.qubits.push_back(q);
    m.cbits.push_back(c);
    return m;
}

}  // namespace

// ── statevector: identity oracle ─────────────────────────────────────────────

TEST(InitialState_Statevector, EmptyCircuitReturnsInputExactly) {
    QarpSimulator sim;
    // Non-trivial amplitudes with phases; norm 1 by construction.
    std::vector<cd> psi = {{0.5, 0.0}, {0.0, 0.5}, {-0.5, 0.0}, {0.0, -0.5}};
    const auto sv = sim.statevector({}, 2, psi);
    ASSERT_EQ(sv.size(), psi.size());
    for (std::size_t i = 0; i < psi.size(); ++i) {
        EXPECT_EQ(sv[i], psi[i]) << "amplitude " << i;
    }
}

TEST(InitialState_Statevector, OmittedInitialStateStillZeroState) {
    QarpSimulator sim;
    const auto sv = sim.statevector({}, 2);
    EXPECT_EQ(sv[0], (cd{1.0, 0.0}));
    for (std::size_t i = 1; i < sv.size(); ++i) EXPECT_EQ(sv[i], (cd{0.0, 0.0}));
}

// ── statevector: analytic oracles ────────────────────────────────────────────

TEST(InitialState_Statevector, HOnSeededBasisState) {
    // |q1=1, q0=0⟩ = e_2; H on qubit 1 → (e_0 − e_2)/√2.
    QarpSimulator sim;
    const auto sv = sim.statevector({Command(GateType::H, 1u)}, 2, basis_state(4, 2));
    EXPECT_NEAR(sv[0].real(),  kInvSqrt2, 1e-12);
    EXPECT_NEAR(sv[2].real(), -kInvSqrt2, 1e-12);
    EXPECT_NEAR(std::abs(sv[1]), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(sv[3]), 0.0, 1e-12);
}

TEST(InitialState_Statevector, CXOnSeededPlusStateGivesBell) {
    // (e_0 + e_1)/√2 = (|0⟩+|1⟩)_q0/√2 ⊗ |0⟩_q1; CX(control 0, target 1)
    // → (e_0 + e_3)/√2.
    QarpSimulator sim;
    std::vector<cd> plus = {{kInvSqrt2, 0.0}, {kInvSqrt2, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    const auto sv = sim.statevector({Command(GateType::CX, 0u, 1u)}, 2, plus);
    EXPECT_NEAR(sv[0].real(), kInvSqrt2, 1e-12);
    EXPECT_NEAR(sv[3].real(), kInvSqrt2, 1e-12);
    EXPECT_NEAR(std::abs(sv[1]), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(sv[2]), 0.0, 1e-12);
}

// ── run: analytic (deterministic) ────────────────────────────────────────────

TEST(InitialState_Run, SeededBasisStateIsDeterministic) {
    // Seed e_1 (q0=1), no gates: every shot samples outcome 1; the recorded
    // measure writes cbit 0 = true on every shot.
    QarpSimulator sim;
    const auto res = sim.run({measure_to(0, 0)}, 1, 64, 7u, basis_state(2, 1));
    ASSERT_EQ(res.counts.size(), 1u);
    EXPECT_EQ(res.counts.at(1), 64);
    ASSERT_EQ(res.cbit_history.size(), 64u);
    for (const auto& reg : res.cbit_history) EXPECT_TRUE(reg.at(0));
}

TEST(InitialState_Run, TrajectoryPathSeesSeededState) {
    // Reset forces the per-shot trajectory path.  From (e_0+e_1)/√2, Reset
    // q0 always ends in |0⟩ regardless of the sampled branch.
    QarpSimulator sim;
    std::vector<cd> plus = {{kInvSqrt2, 0.0}, {kInvSqrt2, 0.0}};
    Command reset;
    reset.gate = GateType::Reset;
    reset.qubits.push_back(0);
    const auto res = sim.run({reset}, 1, 32, 3u, plus);
    ASSERT_EQ(res.counts.size(), 1u);
    EXPECT_EQ(res.counts.at(0), 32);
}

// ── validation: contract rows ────────────────────────────────────────────────

TEST(InitialState_Validation, WrongLengthThrows) {
    QarpSimulator sim;
    EXPECT_THROW((void)sim.statevector({}, 2, basis_state(2, 0)), std::invalid_argument);
    EXPECT_THROW((void)sim.run({}, 2, 8, 0u, basis_state(8, 0)), std::invalid_argument);
}

TEST(InitialState_Validation, NonUnitNormThrows) {
    QarpSimulator sim;
    std::vector<cd> unnormalised = {{0.5, 0.0}, {0.5, 0.0}};
    EXPECT_THROW((void)sim.statevector({}, 1, unnormalised), std::invalid_argument);
    EXPECT_THROW((void)sim.run({}, 1, 8, 0u, unnormalised), std::invalid_argument);
}

TEST(InitialState_Validation, MessageNamesTheCheck) {
    QarpSimulator sim;
    std::vector<cd> unnormalised = {{0.5, 0.0}, {0.5, 0.0}};
    try {
        (void)sim.statevector({}, 1, unnormalised);
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("normalised"), std::string::npos);
    }
}

TEST(InitialState_Validation, InputIsCopiedNotMutated) {
    QarpSimulator sim;
    std::vector<cd> plus = {{kInvSqrt2, 0.0}, {kInvSqrt2, 0.0}};
    const auto snapshot = plus;
    (void)sim.statevector({Command(GateType::H, 0u)}, 1, plus);
    EXPECT_EQ(plus, snapshot);
}
