// Tests for qarpx::synthesis::unitary_synthesis (Quantum Shannon Decomposition).
//
// Strategy: synthesise the circuit for a target unitary, simulate its full
// matrix, and compare against the target EXACTLY — global phase included.
// Covers n=1 (ZYZ base case), n=2 (single recursion), and n=3 (deeper
// recursion).
//
// The comparison must not divide out the global phase: a caller that wraps a
// synthesized block in a control turns that phase into a relative one, so
// an up-to-phase assertion here is blind to the eigenphase shift it causes
// downstream (QPE, QMEGS, Hadamard tests).

#include <gtest/gtest.h>

#include "qarpx/qarpx.h"
#include "qarpx/synthesis/unitary.h"

#include <Eigen/Dense>
#include <Eigen/QR>
#include <unsupported/Eigen/MatrixFunctions>

#include <array>
#include <complex>
#include <random>
#include <vector>

#include "gate_test_helpers.h"

namespace qarpx::test {

namespace {

using cd  = std::complex<double>;
using Mat = Eigen::MatrixXcd;

Mat built_unitary(SimpleBlock& block) {
    block.build();
    Transpiler t(native_gateset());
    QarpSimulator sim;
    return sim.unitary_matrix(t.transpile(block.flatten()), block.n_qubits);
}

/// Build a Haar-random unitary of size N×N via QR of a complex Gaussian matrix.
Mat haar_random(Eigen::Index N, std::mt19937& rng) {
    std::normal_distribution<double> g(0.0, 1.0);
    Mat A(N, N);
    for (Eigen::Index i = 0; i < N; ++i)
        for (Eigen::Index j = 0; j < N; ++j)
            A(i, j) = cd{g(rng), g(rng)};
    Eigen::HouseholderQR<Mat> qr(A);
    Mat Q = qr.householderQ() * Mat::Identity(N, N);
    Mat R = qr.matrixQR().triangularView<Eigen::Upper>();
    // Standard QR-based Haar trick: rescale Q by phase of R diagonal.
    for (Eigen::Index k = 0; k < N; ++k) {
        const cd r = R(k, k);
        if (std::abs(r) > 1e-15) Q.col(k) *= r / std::abs(r);
    }
    return Q;
}

// Exact element-wise comparison comes from `expect_unitary_close` in
// gate_test_helpers.h — deliberately not a phase-insensitive variant.

}  // anonymous namespace

// ── 1-qubit base case (ZYZ) ───────────────────────────────────────────────

TEST(UnitarySynthesis, OneQubit_Identity) {
    SimpleBlock b(1, "u");
    Mat U(2, 2);
    U << 1, 0,
         0, 1;
    b.unitary_synthesis(U);
    EXPECT_TRUE(expect_unitary_close(built_unitary(b), U));
}

TEST(UnitarySynthesis, OneQubit_PauliX) {
    SimpleBlock b(1, "u");
    Mat U(2, 2);
    U << 0, 1,
         1, 0;
    b.unitary_synthesis(U);
    EXPECT_TRUE(expect_unitary_close(built_unitary(b), U));
}

TEST(UnitarySynthesis, OneQubit_PauliY) {
    SimpleBlock b(1, "u");
    Mat U(2, 2);
    U << cd{0, 0}, cd{0, -1},
         cd{0, 1}, cd{0, 0};
    b.unitary_synthesis(U);
    EXPECT_TRUE(expect_unitary_close(built_unitary(b), U));
}

TEST(UnitarySynthesis, OneQubit_PauliZ) {
    SimpleBlock b(1, "u");
    Mat U(2, 2);
    U << 1, 0,
         0, -1;
    b.unitary_synthesis(U);
    EXPECT_TRUE(expect_unitary_close(built_unitary(b), U));
}

TEST(UnitarySynthesis, OneQubit_Hadamard) {
    SimpleBlock b(1, "u");
    const double r = 1.0 / std::sqrt(2.0);
    Mat U(2, 2);
    U << r,  r,
         r, -r;
    b.unitary_synthesis(U);
    EXPECT_TRUE(expect_unitary_close(built_unitary(b), U));
}

TEST(UnitarySynthesis, OneQubit_HaarRandom) {
    std::mt19937 rng(42);
    for (int trial = 0; trial < 5; ++trial) {
        SimpleBlock b(1, "u");
        Mat U = haar_random(2, rng);
        b.unitary_synthesis(U);
        EXPECT_TRUE(expect_unitary_close(built_unitary(b), U))
            << "trial " << trial;
    }
}

// ── ZYZ edge branches (γ ≈ 0 and γ ≈ π) ───────────────────────────────────
//
// `apply_zyz` splits into three branches on γ = 2·atan2(|u10|, |u00|).  The
// generic branch is exercised by the Haar cases above; a diagonal target hits
// γ ≈ 0 and an anti-diagonal one hits γ ≈ π.  Both edge branches once emitted
// a wrong global phase, which only structured operators reach — a Haar target
// lands on the generic branch with probability 1, so these cases pin the two
// branches directly.

TEST(UnitarySynthesis, OneQubit_DiagonalGenericPhases) {
    // γ ≈ 0 branch.  Phases chosen so no two agree and neither is 0.
    SimpleBlock b(1, "u");
    Mat U = Mat::Zero(2, 2);
    U(0, 0) = std::exp(cd{0.0,  0.7});
    U(1, 1) = std::exp(cd{0.0, -1.3});
    b.unitary_synthesis(U);
    EXPECT_TRUE(expect_unitary_close(built_unitary(b), U));
}

TEST(UnitarySynthesis, OneQubit_AntiDiagonalGenericPhases) {
    // γ ≈ π branch.
    SimpleBlock b(1, "u");
    Mat U = Mat::Zero(2, 2);
    U(0, 1) = std::exp(cd{0.0, 0.4});
    U(1, 0) = std::exp(cd{0.0, 2.1});
    b.unitary_synthesis(U);
    EXPECT_TRUE(expect_unitary_close(built_unitary(b), U));
}

TEST(UnitarySynthesis, OneQubit_PhaseGates_S_T_Sdg) {
    // Diagonal Clifford+T gates: the γ ≈ 0 branch at its most common inputs.
    const std::array<cd, 3> phases = {cd{0.0, 1.0},                    // S
                                      std::exp(cd{0.0, M_PI / 4.0}),   // T
                                      cd{0.0, -1.0}};                  // S†
    for (const cd& ph : phases) {
        SimpleBlock b(1, "u");
        Mat U = Mat::Zero(2, 2);
        U(0, 0) = 1;
        U(1, 1) = ph;
        b.unitary_synthesis(U);
        EXPECT_TRUE(expect_unitary_close(built_unitary(b), U))
            << "phase " << ph;
    }
}

TEST(UnitarySynthesis, OneQubit_GlobalPhaseIsReproduced) {
    // e^{iφ}·I is pure global phase: nothing but the ZYZ α term carries it.
    for (const double phi : {0.3, 1.0, -2.4, M_PI}) {
        SimpleBlock b(1, "u");
        Mat U = std::exp(cd{0.0, phi}) * Mat::Identity(2, 2);
        b.unitary_synthesis(U);
        EXPECT_TRUE(expect_unitary_close(built_unitary(b), U)) << "phi " << phi;
    }
}

// ── 2-qubit recursion ─────────────────────────────────────────────────────

TEST(UnitarySynthesis, TwoQubit_CNOT) {
    SimpleBlock b(2, "u");
    Mat U = Mat::Zero(4, 4);
    // CX(0, 1) in LSB convention: q0 control, q1 target.
    // |00⟩ → |00⟩  (i=0)
    // |01⟩ → |11⟩  (q0=1, q1=0 → q0=1, q1=1; i=1 → i=3)
    // |10⟩ → |10⟩  (i=2)
    // |11⟩ → |01⟩  (i=3 → i=1)
    U(0, 0) = 1;
    U(3, 1) = 1;
    U(2, 2) = 1;
    U(1, 3) = 1;
    b.unitary_synthesis(U);
    EXPECT_TRUE(expect_unitary_close(built_unitary(b), U));
}

TEST(UnitarySynthesis, TwoQubit_SWAP) {
    SimpleBlock b(2, "u");
    Mat U = Mat::Zero(4, 4);
    U(0, 0) = 1;  // |00⟩ → |00⟩
    U(2, 1) = 1;  // |01⟩ → |10⟩
    U(1, 2) = 1;  // |10⟩ → |01⟩
    U(3, 3) = 1;  // |11⟩ → |11⟩
    b.unitary_synthesis(U);
    EXPECT_TRUE(expect_unitary_close(built_unitary(b), U));
}

TEST(UnitarySynthesis, TwoQubit_HaarRandom) {
    std::mt19937 rng(1234);
    for (int trial = 0; trial < 5; ++trial) {
        SimpleBlock b(2, "u");
        Mat U = haar_random(4, rng);
        b.unitary_synthesis(U);
        EXPECT_TRUE(expect_unitary_close(built_unitary(b), U))
            << "trial " << trial;
    }
}

// ── 3-qubit recursion ─────────────────────────────────────────────────────

TEST(UnitarySynthesis, ThreeQubit_HaarRandom) {
    std::mt19937 rng(0xBEEF);
    for (int trial = 0; trial < 3; ++trial) {
        SimpleBlock b(3, "u");
        Mat U = haar_random(8, rng);
        b.unitary_synthesis(U);
        EXPECT_TRUE(expect_unitary_close(built_unitary(b), U))
            << "trial " << trial;
    }
}

// ── Degenerate-spectrum regression tests ─────────────────────────────────
//
// These exercise input shapes that stress the CSD path at three points:
// (a) JacobiSVD's basis indeterminacy on a degenerate σ spectrum,
// (b) ComplexSchur's basis indeterminacy on repeated eigenvalues, and
// (c) acos(c) numerical noise when c is mathematically 1.  Together these can
// make diagonal `exp(-i·Z⊗Z·t)` and several Kronecker shapes synthesize to
// ≈ -I on some toolchains (e.g. macOS/arm64) while passing on others
// (Linux/x86).  Cover all three so any toolchain regression shows up here first.

TEST(UnitarySynthesis, Degenerate_DiagonalZZ_Evolution) {
    SimpleBlock b(2, "u");
    const double t = 0.42;
    const cd p = std::exp(cd{0.0, -t});
    const cd m = std::exp(cd{0.0,  t});
    Mat U = Mat::Zero(4, 4);
    U(0, 0) = p;
    U(1, 1) = m;
    U(2, 2) = m;
    U(3, 3) = p;
    b.unitary_synthesis(U);
    EXPECT_TRUE(expect_unitary_close(built_unitary(b), U));
}

TEST(UnitarySynthesis, Degenerate_DiagonalZZZ_Evolution_3Q) {
    SimpleBlock b(3, "u");
    const double t = 0.42;
    // Diagonal of exp(-i Z⊗Z⊗Z · t): index k has sign = (-1)^popcount(k).
    Mat U = Mat::Zero(8, 8);
    for (Eigen::Index k = 0; k < 8; ++k) {
        int parity = (((k >> 0) ^ (k >> 1) ^ (k >> 2)) & 1);
        U(k, k) = std::exp(cd{0.0, (parity ? +t : -t)});
    }
    b.unitary_synthesis(U);
    EXPECT_TRUE(expect_unitary_close(built_unitary(b), U));
}

TEST(UnitarySynthesis, Degenerate_RecursiveDiagonal_HtensorExpZZ) {
    // H ⊗ expZZ: non-diagonal at the top level, but its CSD recursion lands on
    // a diagonal sub-problem one level deep — the case that motivated adding
    // the recursive diagonal fast-path / Schur canonicalization.
    SimpleBlock b(3, "u");
    const double t = 0.42;
    const double r = 1.0 / std::sqrt(2.0);
    Mat expZZ = Mat::Zero(4, 4);
    const cd p = std::exp(cd{0.0, -t});
    const cd m = std::exp(cd{0.0,  t});
    expZZ(0, 0) = p;
    expZZ(1, 1) = m;
    expZZ(2, 2) = m;
    expZZ(3, 3) = p;
    Mat U(8, 8);
    U.topLeftCorner(4, 4)     =  r * expZZ;
    U.topRightCorner(4, 4)    =  r * expZZ;
    U.bottomLeftCorner(4, 4)  =  r * expZZ;
    U.bottomRightCorner(4, 4) = -r * expZZ;
    b.unitary_synthesis(U);
    EXPECT_TRUE(expect_unitary_close(built_unitary(b), U));
}

TEST(UnitarySynthesis, Degenerate_AntiDiagonalBlocks_XtensorExpZZ) {
    SimpleBlock b(3, "u");
    const double t = 0.42;
    Mat expZZ = Mat::Zero(4, 4);
    const cd p = std::exp(cd{0.0, -t});
    const cd m = std::exp(cd{0.0,  t});
    expZZ(0, 0) = p;
    expZZ(1, 1) = m;
    expZZ(2, 2) = m;
    expZZ(3, 3) = p;
    Mat U = Mat::Zero(8, 8);
    U.topRightCorner(4, 4)   = expZZ;  // X⊗expZZ: anti-block-diagonal layout
    U.bottomLeftCorner(4, 4) = expZZ;
    b.unitary_synthesis(U);
    EXPECT_TRUE(expect_unitary_close(built_unitary(b), U));
}

TEST(UnitarySynthesis, Degenerate_Heisenberg_4Site_StructuralSpread) {
    // 4-site Heisenberg `exp(-i (X⊗X+Y⊗Y+Z⊗Z) bonds · t)` at small/medium t.
    // The Sz-conservation symmetry produces clusters of σ values in the CSD
    // recursion that are spread by ~1e-5 — too tight to be "actually equal"
    // for a tolerance-based degenerate-σ detector but tight enough that the
    // previous Eigen-SVD-plus-orthonormal-completion path miscomposed.  The
    // LAPACK CSD primitive handles this structural near-degeneracy cleanly.
    SimpleBlock b(4, "u");
    const double t = 0.35;
    // Build H = sum over bonds (i, i+1) and Paulis (X, Y, Z) of P_i ⊗ P_{i+1}.
    Mat I2(2, 2); I2.setIdentity();
    Mat X(2, 2); X << 0, 1, 1, 0;
    Mat Y(2, 2); Y << cd{0,0}, cd{0,-1}, cd{0,1}, cd{0,0};
    Mat Z(2, 2); Z << 1, 0, 0, -1;
    std::array<Mat, 3> paulis = {X, Y, Z};
    auto kron = [](const Mat& a, const Mat& b) -> Mat {
        Mat r(a.rows()*b.rows(), a.cols()*b.cols());
        for (Eigen::Index i = 0; i < a.rows(); ++i)
            for (Eigen::Index j = 0; j < a.cols(); ++j)
                r.block(i*b.rows(), j*b.cols(), b.rows(), b.cols()) = a(i,j) * b;
        return r;
    };
    Mat H = Mat::Zero(16, 16);
    for (int site = 0; site < 3; ++site) {
        for (const Mat& P : paulis) {
            Mat term = (site == 0) ? P : I2;
            for (int k = 1; k < 4; ++k) {
                if (k == site)      term = kron(term, P);
                else if (k == site+1) term = kron(term, P);
                else                term = kron(term, I2);
            }
            H += term;
        }
    }
    // U = exp(-i·H·t) via Taylor series approximation is unstable here;
    // we use Eigen's matrix exponential.
    Eigen::MatrixXcd minusiHt = cd{0.0, -t} * H;
    Mat U = minusiHt.exp();
    b.unitary_synthesis(U);
#ifdef QARP_USE_LAPACK
    const double tol = 1e-10;
#else
    // The pure-Eigen CSD fallback resolves this Sz-symmetric structural
    // near-degeneracy (a σ that sits ~2e-10 from a σ=1 block) to ~1.5e-10 —
    // correct to 10 significant figures, but a touch above the floor LAPACK's
    // `zuncsd` reaches (~1e-13).
    const double tol = 1e-9;
#endif
    EXPECT_TRUE(expect_unitary_close(built_unitary(b), U, tol));
}

TEST(UnitarySynthesis, Degenerate_FourQubit_HaarRandom) {
    // 4-qubit Haar — exercises the deeper recursion where degenerate Schur
    // eigenvalues are more likely to appear in intermediate W/V sub-problems.
    std::mt19937 rng(0xDEC0DE);
    for (int trial = 0; trial < 3; ++trial) {
        SimpleBlock b(4, "u");
        Mat U = haar_random(16, rng);
        b.unitary_synthesis(U);
        EXPECT_TRUE(expect_unitary_close(built_unitary(b), U, 1e-8))
            << "trial " << trial;
    }
}

// ── Input validation ──────────────────────────────────────────────────────

TEST(UnitarySynthesis, RejectsNonSquare) {
    SimpleBlock b(2, "u");
    Mat U(4, 3);
    U.setZero();
    EXPECT_THROW(b.unitary_synthesis(U), std::runtime_error);
}

TEST(UnitarySynthesis, RejectsNonPowerOfTwoSize) {
    SimpleBlock b(2, "u");
    Mat U(3, 3);
    U.setIdentity();
    EXPECT_THROW(b.unitary_synthesis(U), std::runtime_error);
}

TEST(UnitarySynthesis, RejectsNonUnitary) {
    SimpleBlock b(1, "u");
    Mat U(2, 2);
    U << 1, 0,
         0, 0.5;
    EXPECT_THROW(b.unitary_synthesis(U), std::runtime_error);
}

}  // namespace qarpx::test
