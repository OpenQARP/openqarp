// Tests for qarpx::synthesis::state_preparation.
//
// Strategy: build a SimpleBlock, call state_preparation with a target amplitude
// vector, simulate the unitary, and check that its first column equals the
// (normalized) target amplitudes.  Covers n = 1..4 with structured states,
// random states, sparsity edge cases, and a uniform superposition.

#include <gtest/gtest.h>

#include "qarpx/qarpx.h"
#include "qarpx/synthesis/state_preparation.h"
#include "qarpx/synthesis/uniformly_controlled.h"

#include <Eigen/Dense>

#include <complex>
#include <random>
#include <vector>

#include "gate_test_helpers.h"

namespace qarpx::test {

namespace {

using cd = std::complex<double>;

/// First column of `block`'s unitary, after build + transpile + simulate.
Eigen::VectorXcd prepared_state(SimpleBlock& block) {
    block.build();
    Transpiler t(native_gateset());
    QarpSimulator sim;
    auto U = sim.unitary_matrix(t.transpile(block.flatten()), block.n_qubits);
    return U.col(0);
}

void normalize(std::vector<cd>& v) {
    double norm_sq = 0.0;
    for (const auto& z : v) norm_sq += std::norm(z);
    const double inv = 1.0 / std::sqrt(norm_sq);
    for (auto& z : v) z *= inv;
}

/// Compare prepared statevector against the (normalized) target amplitudes.
::testing::AssertionResult expect_state_close(
    const Eigen::VectorXcd& actual,
    const std::vector<cd>& expected,
    double tol = 1e-10)
{
    if (static_cast<size_t>(actual.size()) != expected.size()) {
        return ::testing::AssertionFailure()
            << "size mismatch: actual " << actual.size()
            << " vs expected " << expected.size();
    }
    auto exp = expected;
    normalize(exp);
    double max_diff = 0.0;
    for (Eigen::Index i = 0; i < actual.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(actual[i] - exp[i]));
    }
    if (max_diff < tol) return ::testing::AssertionSuccess();
    std::ostringstream oss;
    oss << "max diff " << max_diff << " > tol " << tol << "\n";
    for (Eigen::Index i = 0; i < actual.size(); ++i) {
        oss << "  i=" << i << "  actual=" << actual[i]
            << "  expected=" << exp[i] << "\n";
    }
    return ::testing::AssertionFailure() << oss.str();
}

}  // anonymous namespace

// ── Single-qubit edge cases ────────────────────────────────────────────────

TEST(StatePreparation, SingleQubit_AllZero_RaisesOnZeroNorm) {
    SimpleBlock b(1, "prep");
    EXPECT_THROW(b.state_preparation({cd{0,0}, cd{0,0}}),
                 std::runtime_error);
}

TEST(StatePreparation, SingleQubit_EqualSuperposition) {
    SimpleBlock b(1, "prep");
    const cd a{1.0/std::sqrt(2.0), 0.0};
    b.state_preparation({a, a});
    EXPECT_TRUE(expect_state_close(prepared_state(b), {a, a}));
}

TEST(StatePreparation, SingleQubit_Phased) {
    SimpleBlock b(1, "prep");
    // |ψ⟩ = 0.6 |0⟩ + 0.8 e^{iπ/3} |1⟩
    std::vector<cd> psi = {cd{0.6, 0.0}, std::polar(0.8, PI/3.0)};
    b.state_preparation(psi);
    EXPECT_TRUE(expect_state_close(prepared_state(b), psi));
}

TEST(StatePreparation, SingleQubit_OnlyZeroAmpl) {
    SimpleBlock b(1, "prep");
    std::vector<cd> psi = {cd{0.0, 0.0}, std::polar(1.0, 0.7)};  // |1⟩ with phase
    b.state_preparation(psi);
    EXPECT_TRUE(expect_state_close(prepared_state(b), psi));
}

TEST(StatePreparation, SingleQubit_OnlyOneAmpl) {
    SimpleBlock b(1, "prep");
    std::vector<cd> psi = {std::polar(1.0, 0.7), cd{0.0, 0.0}};  // |0⟩ with phase
    b.state_preparation(psi);
    EXPECT_TRUE(expect_state_close(prepared_state(b), psi));
}

// ── 2..4 qubit structured states ───────────────────────────────────────────

TEST(StatePreparation, TwoQubit_BellState) {
    SimpleBlock b(2, "prep");
    // |Φ+⟩ = (|00⟩ + |11⟩)/√2.  In LSB indexing: i=0 → q0=0,q1=0; i=3 → q0=1,q1=1.
    std::vector<cd> psi = {cd{1,0}, cd{0,0}, cd{0,0}, cd{1,0}};
    b.state_preparation(psi);
    EXPECT_TRUE(expect_state_close(prepared_state(b), psi));
}

TEST(StatePreparation, TwoQubit_W_like) {
    SimpleBlock b(2, "prep");
    // (|01⟩ + |10⟩) / √2
    std::vector<cd> psi = {cd{0,0}, cd{1,0}, cd{1,0}, cd{0,0}};
    b.state_preparation(psi);
    EXPECT_TRUE(expect_state_close(prepared_state(b), psi));
}

TEST(StatePreparation, TwoQubit_PhasedNonsymmetric) {
    SimpleBlock b(2, "prep");
    std::vector<cd> psi = {
        std::polar(0.5, 0.1),
        std::polar(0.6, 0.7),
        std::polar(0.4, -0.3),
        std::polar(0.3, 1.7),
    };
    b.state_preparation(psi);
    EXPECT_TRUE(expect_state_close(prepared_state(b), psi));
}

TEST(StatePreparation, ThreeQubit_UniformSuperposition) {
    SimpleBlock b(3, "prep");
    std::vector<cd> psi(8, cd{1.0 / std::sqrt(8.0), 0.0});
    b.state_preparation(psi);
    EXPECT_TRUE(expect_state_close(prepared_state(b), psi));
}

TEST(StatePreparation, ThreeQubit_SparseAmplitudes) {
    SimpleBlock b(3, "prep");
    // Only basis states |001⟩ and |110⟩ have amplitude.  Indices: |001⟩=1, |110⟩=6.
    std::vector<cd> psi(8, cd{0,0});
    psi[1] = std::polar(1.0, 0.4);
    psi[6] = std::polar(1.0, -0.8);
    b.state_preparation(psi);
    EXPECT_TRUE(expect_state_close(prepared_state(b), psi));
}

TEST(StatePreparation, FourQubit_RandomComplex) {
    SimpleBlock b(4, "prep");
    std::mt19937 rng(0xC0FFEE);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<cd> psi(16);
    for (auto& z : psi) z = cd{u(rng), u(rng)};
    b.state_preparation(psi);
    EXPECT_TRUE(expect_state_close(prepared_state(b), psi));
}

TEST(StatePreparation, FourQubit_RandomReal_NoPhase) {
    SimpleBlock b(4, "prep");
    std::mt19937 rng(0xBADF00D);
    std::uniform_real_distribution<double> u(0.1, 1.0);
    std::vector<cd> psi(16);
    for (auto& z : psi) z = cd{u(rng), 0.0};
    b.state_preparation(psi);
    EXPECT_TRUE(expect_state_close(prepared_state(b), psi));
}

// ── Input validation ───────────────────────────────────────────────────────

TEST(StatePreparation, RejectsNonPowerOfTwoSize) {
    SimpleBlock b(2, "prep");
    EXPECT_THROW(b.state_preparation({cd{1,0}, cd{1,0}, cd{1,0}}),
                 std::runtime_error);
}

TEST(StatePreparation, RejectsZeroNorm) {
    SimpleBlock b(2, "prep");
    EXPECT_THROW(b.state_preparation({cd{0,0}, cd{0,0}, cd{0,0}, cd{0,0}}),
                 std::runtime_error);
}

// ── Helper-level smoke: UC-Ry/Rz with no controls is a plain rotation ──────

TEST(UniformlyControlled, EmptyControls_PlainRotation) {
    SimpleBlock b(1, "smoke");
    qarpx::synthesis::apply_uc_ry(b, /*target=*/0, /*controls=*/{}, /*angles=*/{0.7});
    qarpx::synthesis::apply_uc_rz(b, /*target=*/0, /*controls=*/{}, /*angles=*/{1.3});
    b.build();
    auto cmds = b.flatten();
    ASSERT_EQ(cmds.size(), 2u);
    EXPECT_EQ(cmds[0].gate, GateType::Ry);
    EXPECT_EQ(cmds[1].gate, GateType::Rz);
}

}  // namespace qarpx::test
