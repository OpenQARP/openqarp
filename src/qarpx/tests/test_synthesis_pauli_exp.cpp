// End-to-end tests for the public API:
//   qarpx::synthesis::commuting_pauli_set_exp
//   qarpx::synthesis::pauli_exp
//
// Numerical contract covered (§ numbers are this file's own grouping):
//   §1  Unitary equivalence: exp(-i/2 · Σ θᵢ Pᵢ) on small n.
//   §2  Single-Pauli match with the canonical Pauli-exponential pattern.
//   §3  Symbolic Param round-trip via set_symbols → flatten.
//   §4  Dagger round-trip: block · dagger(block) ≈ I.
//   §5  Validation rejects: lengths, mismatch, non-commuting.

#include <gtest/gtest.h>

#include "qarpx/synthesis/pauli_exponential.h"

#include "gate_test_helpers.h"

#include <unsupported/Eigen/MatrixFunctions>

#include <stdexcept>
#include <vector>

namespace qarpx::test {

using qarpx::synthesis::commuting_pauli_set_exp;
using qarpx::synthesis::pauli_exp;

namespace {

/// Build the matrix of a single Pauli string on n qubits in LSB-first basis.
Mat pauli_matrix(const PauliString& p, int n) {
    Mat result = Mat::Ones(1, 1);
    for (int q = 0; q < n; ++q) {
        Mat m;
        switch (p[q]) {
            case Pauli::I: m = Mat::Identity(2, 2); break;
            case Pauli::X: m = analytic_x();        break;
            case Pauli::Y: m = analytic_y();        break;
            case Pauli::Z: m = analytic_z();        break;
        }
        result = kron(m, result);
    }
    return result;
}

/// Reference unitary: U = exp(-i/2 · Σᵢ θᵢ Pᵢ) computed via direct matrix
/// exponential.  Valid for any input (commuting or not, but our synthesis
/// is only correct on commuting input).
Mat reference_commuting_unitary(
    const std::vector<PauliString>& paulis,
    const std::vector<double>& angles,
    int n)
{
    const int dim = 1 << n;
    Mat H = Mat::Zero(dim, dim);
    for (std::size_t i = 0; i < paulis.size(); ++i) {
        H += angles[i] * pauli_matrix(paulis[i], n);
    }
    return (cd{0, -0.5} * H).exp();
}

/// Build a single-block circuit for a commuting Pauli set and return its
/// flattened unitary (after transpiling to the native gate set).
Mat build_commuting_set_unitary(
    const std::vector<PauliString>& paulis,
    const std::vector<double>& angles,
    int n)
{
    SimpleBlock block(static_cast<uint32_t>(n), "test");
    std::vector<Param> param_angles;
    param_angles.reserve(angles.size());
    for (double a : angles) param_angles.emplace_back(a);
    commuting_pauli_set_exp(block, paulis, param_angles);
    block.build();
    return build_unitary(block.flatten(), n);
}

Mat build_single_pauli_unitary(
    const PauliString& p, double angle, int n)
{
    SimpleBlock block(static_cast<uint32_t>(n), "test");
    pauli_exp(block, p, Param(angle));
    block.build();
    return build_unitary(block.flatten(), n);
}

}  // anonymous namespace

// ── §1 Unitary equivalence ──────────────────────────────────────────────

TEST(CommutingPauliSetExp_Unitary, EmptyInputIsNoOp) {
    SimpleBlock block(2, "t");
    commuting_pauli_set_exp(block, {}, {});
    EXPECT_TRUE(block.commands().empty());
}

TEST(CommutingPauliSetExp_Unitary, SingleQubitZ) {
    const int n = 1;
    const std::vector<PauliString> P = {parse_pauli_string("Z")};
    const std::vector<double> A = {0.7};
    Mat actual = build_commuting_set_unitary(P, A, n);
    Mat ref = reference_commuting_unitary(P, A, n);
    EXPECT_TRUE(expect_unitary_close(actual, ref));
}

TEST(CommutingPauliSetExp_Unitary, SingleQubitX) {
    Mat actual = build_commuting_set_unitary(
        {parse_pauli_string("X")}, {0.5}, 1);
    Mat ref = reference_commuting_unitary(
        {parse_pauli_string("X")}, {0.5}, 1);
    EXPECT_TRUE(expect_unitary_close(actual, ref));
}

TEST(CommutingPauliSetExp_Unitary, SingleQubitY) {
    Mat actual = build_commuting_set_unitary(
        {parse_pauli_string("Y")}, {0.3}, 1);
    Mat ref = reference_commuting_unitary(
        {parse_pauli_string("Y")}, {0.3}, 1);
    EXPECT_TRUE(expect_unitary_close(actual, ref));
}

TEST(CommutingPauliSetExp_Unitary, IdentityRowGlobalPhase) {
    // Single all-I Pauli: should produce gphase(-angle/2).
    Mat actual = build_commuting_set_unitary(
        {parse_pauli_string("II")}, {0.6}, 2);
    Mat ref = reference_commuting_unitary(
        {parse_pauli_string("II")}, {0.6}, 2);
    EXPECT_TRUE(expect_unitary_close(actual, ref));
}

TEST(CommutingPauliSetExp_Unitary, BellStateStabilisers_n2) {
    // {XX, YY, ZZ} — the canonical 2-qubit commuting set.
    const std::vector<PauliString> P = {
        parse_pauli_string("XX"),
        parse_pauli_string("YY"),
        parse_pauli_string("ZZ"),
    };
    const std::vector<double> A = {0.4, -0.2, 0.7};
    Mat actual = build_commuting_set_unitary(P, A, 2);
    Mat ref = reference_commuting_unitary(P, A, 2);
    EXPECT_TRUE(expect_unitary_close(actual, ref));
}

TEST(CommutingPauliSetExp_Unitary, ZParityFamily_n3) {
    const std::vector<PauliString> P = {
        parse_pauli_string("ZZI"),
        parse_pauli_string("IZZ"),
        parse_pauli_string("ZIZ"),
    };
    const std::vector<double> A = {0.3, 0.5, 0.1};
    Mat actual = build_commuting_set_unitary(P, A, 3);
    Mat ref = reference_commuting_unitary(P, A, 3);
    EXPECT_TRUE(expect_unitary_close(actual, ref));
}

TEST(CommutingPauliSetExp_Unitary, MixedXYZ_n3) {
    const std::vector<PauliString> P = {
        parse_pauli_string("XYZ"),
        parse_pauli_string("YXZ"),
        parse_pauli_string("ZZI"),
    };
    const std::vector<double> A = {0.4, -0.6, 1.1};
    Mat actual = build_commuting_set_unitary(P, A, 3);
    Mat ref = reference_commuting_unitary(P, A, 3);
    EXPECT_TRUE(expect_unitary_close(actual, ref));
}

TEST(CommutingPauliSetExp_Unitary, FourQubitHeisenberg) {
    const std::vector<PauliString> P = {
        parse_pauli_string("ZZII"),
        parse_pauli_string("IZZI"),
        parse_pauli_string("IIZZ"),
        parse_pauli_string("XXXX"),
    };
    const std::vector<double> A = {0.2, -0.3, 0.4, 0.5};
    Mat actual = build_commuting_set_unitary(P, A, 4);
    Mat ref = reference_commuting_unitary(P, A, 4);
    EXPECT_TRUE(expect_unitary_close(actual, ref));
}

TEST(CommutingPauliSetExp_Unitary, IdentityPlusZZ) {
    // Identity row coexisting with a real Pauli.
    const std::vector<PauliString> P = {
        parse_pauli_string("II"),
        parse_pauli_string("ZZ"),
    };
    const std::vector<double> A = {0.5, 1.0};
    Mat actual = build_commuting_set_unitary(P, A, 2);
    Mat ref = reference_commuting_unitary(P, A, 2);
    EXPECT_TRUE(expect_unitary_close(actual, ref));
}

// ── §2 Single-Pauli (pauli_exp) ─────────────────────────────────────────

TEST(PauliExp_Unitary, SingleZ) {
    Mat actual = build_single_pauli_unitary(parse_pauli_string("Z"), 0.5, 1);
    Mat ref = reference_commuting_unitary({parse_pauli_string("Z")}, {0.5}, 1);
    EXPECT_TRUE(expect_unitary_close(actual, ref));
}

TEST(PauliExp_Unitary, SingleX) {
    Mat actual = build_single_pauli_unitary(parse_pauli_string("X"), 0.5, 1);
    Mat ref = reference_commuting_unitary({parse_pauli_string("X")}, {0.5}, 1);
    EXPECT_TRUE(expect_unitary_close(actual, ref));
}

TEST(PauliExp_Unitary, SingleY) {
    Mat actual = build_single_pauli_unitary(parse_pauli_string("Y"), 0.3, 1);
    Mat ref = reference_commuting_unitary({parse_pauli_string("Y")}, {0.3}, 1);
    EXPECT_TRUE(expect_unitary_close(actual, ref));
}

TEST(PauliExp_Unitary, MultiQubit_XYZ) {
    Mat actual = build_single_pauli_unitary(parse_pauli_string("XYZ"), 0.7, 3);
    Mat ref = reference_commuting_unitary({parse_pauli_string("XYZ")}, {0.7}, 3);
    EXPECT_TRUE(expect_unitary_close(actual, ref));
}

TEST(PauliExp_Unitary, IdentityPauliEmitsGlobalPhase) {
    Mat actual = build_single_pauli_unitary(parse_pauli_string("II"), 1.0, 2);
    Mat ref = reference_commuting_unitary({parse_pauli_string("II")}, {1.0}, 2);
    EXPECT_TRUE(expect_unitary_close(actual, ref));
}

// k=1 fast path in commuting_pauli_set_exp should match pauli_exp byte-for-byte
// (same command sequence emitted).
TEST(CommutingPauliSetExp, KEqualsOneMatchesPauliExp) {
    SimpleBlock b1(3, "t1");
    SimpleBlock b2(3, "t2");
    commuting_pauli_set_exp(b1, {parse_pauli_string("XYZ")}, {Param(0.5)});
    pauli_exp(b2, parse_pauli_string("XYZ"), Param(0.5));
    ASSERT_EQ(b1.commands().size(), b2.commands().size());
    for (std::size_t i = 0; i < b1.commands().size(); ++i) {
        EXPECT_EQ(b1.commands()[i].gate, b2.commands()[i].gate);
    }
}

// ── §3 Symbolic Param round-trip ────────────────────────────────────────

TEST(CommutingPauliSetExp_Symbolic, SubstituteThenSimulate) {
    const std::vector<PauliString> P = {parse_pauli_string("ZZ"), parse_pauli_string("XX")};
    const double theta_val = 0.4, phi_val = 0.7;

    SimpleBlock sym_block(2, "sym");
    commuting_pauli_set_exp(
        sym_block, P, {Param::symbol("theta"), Param::symbol("phi")});
    sym_block.build();

    // Substitute symbols, then simulate.
    auto cmds = qarpx::substitute_all(
        sym_block.flatten(), {{"theta", theta_val}, {"phi", phi_val}});
    QarpSimulator sim;
    Transpiler t(native_gateset());
    Mat actual = sim.unitary_matrix(t.transpile(cmds), 2);

    Mat ref = reference_commuting_unitary(P, {theta_val, phi_val}, 2);
    EXPECT_TRUE(expect_unitary_close(actual, ref));
}

TEST(PauliExp_Symbolic, SubstituteThenSimulate) {
    SimpleBlock sym_block(3, "sym");
    pauli_exp(sym_block, parse_pauli_string("XYZ"), Param::symbol("alpha"));
    sym_block.build();

    auto cmds = qarpx::substitute_all(sym_block.flatten(), {{"alpha", 1.234}});
    QarpSimulator sim;
    Transpiler t(native_gateset());
    Mat actual = sim.unitary_matrix(t.transpile(cmds), 3);

    Mat ref = reference_commuting_unitary({parse_pauli_string("XYZ")}, {1.234}, 3);
    EXPECT_TRUE(expect_unitary_close(actual, ref));
}

// ── §4 Dagger round-trip ────────────────────────────────────────────────

TEST(CommutingPauliSetExp, DaggerRoundTripIsIdentity) {
    // Build U then dagger by emitting commuting_pauli_set_exp with negated angles.
    const std::vector<PauliString> P = {
        parse_pauli_string("XX"),
        parse_pauli_string("YY"),
        parse_pauli_string("ZZ"),
    };
    const std::vector<double> A_forward = {0.5, -0.3, 0.7};
    const std::vector<Param> P_forward = {Param(0.5), Param(-0.3), Param(0.7)};
    const std::vector<Param> P_reverse = {Param(-0.5), Param(0.3), Param(-0.7)};

    SimpleBlock block(2, "fwd_then_rev");
    commuting_pauli_set_exp(block, P, P_forward);
    commuting_pauli_set_exp(block, P, P_reverse);
    block.build();

    Mat composite = build_unitary(block.flatten(), 2);
    Mat I = Mat::Identity(4, 4);
    EXPECT_TRUE(expect_unitary_close(composite, I));
}

// ── §5 Validation ──────────────────────────────────────────────────────

TEST(CommutingPauliSetExp_Validation, LengthMismatchThrows) {
    SimpleBlock block(2, "t");
    EXPECT_THROW(
        commuting_pauli_set_exp(block, {parse_pauli_string("XX")}, {}),
        std::invalid_argument);
    EXPECT_THROW(
        commuting_pauli_set_exp(
            block, {parse_pauli_string("XX")},
            {Param(0.1), Param(0.2)}),
        std::invalid_argument);
}

TEST(CommutingPauliSetExp_Validation, PauliRowLengthMismatchThrows) {
    SimpleBlock block(2, "t");
    EXPECT_THROW(
        commuting_pauli_set_exp(
            block, {parse_pauli_string("XXX")}, {Param(0.1)}),
        std::invalid_argument);
}

TEST(CommutingPauliSetExp_Validation, NonCommutingThrows) {
    SimpleBlock block(1, "t");
    EXPECT_THROW(
        commuting_pauli_set_exp(
            block,
            {parse_pauli_string("X"), parse_pauli_string("Y")},
            {Param(0.1), Param(0.2)}),
        std::invalid_argument);
    SimpleBlock block2(2, "u");
    EXPECT_THROW(
        commuting_pauli_set_exp(
            block2,
            {parse_pauli_string("XX"), parse_pauli_string("XY")},
            {Param(0.1), Param(0.2)}),
        std::invalid_argument);
}

TEST(PauliExp_Validation, RowLengthMismatchThrows) {
    SimpleBlock block(2, "t");
    EXPECT_THROW(
        pauli_exp(block, parse_pauli_string("XXX"), Param(0.1)),
        std::invalid_argument);
}

}  // namespace qarpx::test
