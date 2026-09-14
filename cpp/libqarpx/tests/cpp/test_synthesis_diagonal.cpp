// Tests for qarpx::synthesis::diagonal_unitary.
//
// Build the synthesised circuit's full unitary (via QarpSimulator::unitary_matrix)
// and compare element-wise against the target diag(d_0, ..., d_{2^n-1}).

#include <gtest/gtest.h>

#include "qarpx/qarpx.h"
#include "qarpx/synthesis/diagonal.h"

#include <Eigen/Dense>

#include <complex>
#include <random>
#include <vector>

#include "gate_test_helpers.h"

namespace qarpx::test {

namespace {

using cd = std::complex<double>;

Eigen::MatrixXcd built_unitary(SimpleBlock& block) {
    block.build();
    Transpiler t(native_gateset());
    QarpSimulator sim;
    return sim.unitary_matrix(t.transpile(block.flatten()), block.n_qubits);
}

::testing::AssertionResult expect_diagonal_close(
    const Eigen::MatrixXcd& U,
    const std::vector<cd>& expected_diag,
    double tol = 1e-10)
{
    if (static_cast<size_t>(U.rows()) != expected_diag.size() ||
        static_cast<size_t>(U.cols()) != expected_diag.size()) {
        return ::testing::AssertionFailure() << "shape mismatch";
    }
    Eigen::MatrixXcd D(U.rows(), U.cols());
    D.setZero();
    for (Eigen::Index i = 0; i < U.rows(); ++i) D(i, i) = expected_diag[i];
    const double diff = (U - D).cwiseAbs().maxCoeff();
    if (diff < tol) return ::testing::AssertionSuccess();
    std::ostringstream oss;
    oss << "max element-wise diff " << diff << " > tol " << tol << "\n";
    oss << "actual:\n" << U << "\nexpected:\n" << D;
    return ::testing::AssertionFailure() << oss.str();
}

}  // anonymous namespace

// ── Trivial / structured cases ────────────────────────────────────────────

TEST(DiagonalUnitary, OneQubit_Identity) {
    SimpleBlock b(1, "diag");
    b.diagonal_unitary({cd{1, 0}, cd{1, 0}});
    EXPECT_TRUE(expect_diagonal_close(built_unitary(b), {cd{1, 0}, cd{1, 0}}));
}

TEST(DiagonalUnitary, OneQubit_Z) {
    SimpleBlock b(1, "diag");
    // Z = diag(1, -1)
    b.diagonal_unitary({cd{1, 0}, cd{-1, 0}});
    EXPECT_TRUE(expect_diagonal_close(built_unitary(b), {cd{1, 0}, cd{-1, 0}}));
}

TEST(DiagonalUnitary, OneQubit_S) {
    SimpleBlock b(1, "diag");
    // S = diag(1, i)
    b.diagonal_unitary({cd{1, 0}, cd{0, 1}});
    EXPECT_TRUE(expect_diagonal_close(built_unitary(b), {cd{1, 0}, cd{0, 1}}));
}

TEST(DiagonalUnitary, OneQubit_T) {
    SimpleBlock b(1, "diag");
    const cd t{std::cos(PI/4.0), std::sin(PI/4.0)};
    b.diagonal_unitary({cd{1, 0}, t});
    EXPECT_TRUE(expect_diagonal_close(built_unitary(b), {cd{1, 0}, t}));
}

TEST(DiagonalUnitary, TwoQubit_CZ) {
    SimpleBlock b(2, "diag");
    // CZ = diag(1, 1, 1, -1) on (q0, q1) with LSB convention: index 3 is q0=1,q1=1.
    std::vector<cd> diag = {cd{1, 0}, cd{1, 0}, cd{1, 0}, cd{-1, 0}};
    b.diagonal_unitary(diag);
    EXPECT_TRUE(expect_diagonal_close(built_unitary(b), diag));
}

TEST(DiagonalUnitary, TwoQubit_PhaseChain) {
    SimpleBlock b(2, "diag");
    std::vector<cd> diag = {
        std::polar(1.0,  0.1),
        std::polar(1.0, -0.4),
        std::polar(1.0,  0.7),
        std::polar(1.0,  1.3),
    };
    b.diagonal_unitary(diag);
    EXPECT_TRUE(expect_diagonal_close(built_unitary(b), diag));
}

TEST(DiagonalUnitary, ThreeQubit_RandomPhases) {
    SimpleBlock b(3, "diag");
    std::mt19937 rng(0xD1A6);
    std::uniform_real_distribution<double> u(-PI, PI);
    std::vector<cd> diag(8);
    for (auto& z : diag) z = std::polar(1.0, u(rng));
    b.diagonal_unitary(diag);
    EXPECT_TRUE(expect_diagonal_close(built_unitary(b), diag));
}

TEST(DiagonalUnitary, FourQubit_RandomPhases) {
    SimpleBlock b(4, "diag");
    std::mt19937 rng(0xFEED);
    std::uniform_real_distribution<double> u(-PI, PI);
    std::vector<cd> diag(16);
    for (auto& z : diag) z = std::polar(1.0, u(rng));
    b.diagonal_unitary(diag);
    EXPECT_TRUE(expect_diagonal_close(built_unitary(b), diag));
}

TEST(DiagonalUnitary, GlobalPhaseOnly) {
    SimpleBlock b(2, "diag");
    // diag(e^{iθ}, e^{iθ}, e^{iθ}, e^{iθ}) — pure global phase.
    const cd ph = std::polar(1.0, 0.7);
    std::vector<cd> diag(4, ph);
    b.diagonal_unitary(diag);
    EXPECT_TRUE(expect_diagonal_close(built_unitary(b), diag));
}

// ── Input validation ─────────────────────────────────────────────────────

TEST(DiagonalUnitary, RejectsNonPowerOfTwoSize) {
    SimpleBlock b(2, "diag");
    EXPECT_THROW(
        b.diagonal_unitary({cd{1, 0}, cd{1, 0}, cd{1, 0}}),
        std::runtime_error);
}

TEST(DiagonalUnitary, RejectsNonUnitModulus) {
    SimpleBlock b(1, "diag");
    EXPECT_THROW(
        b.diagonal_unitary({cd{1, 0}, cd{0.5, 0}}),
        std::runtime_error);
}

}  // namespace qarpx::test
