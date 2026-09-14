// Tests for qarpx::synthesis::detail: the symplectic tableau
// machinery and the basis-change Clifford that diagonalises a commuting
// Pauli set to Z-only form.
//
// Two layers of coverage:
//   1. Individual Clifford tableau updates (apply_h / s / sdg / cx) against
//      hand-computed expected outputs, including phase tracking.
//   2. The full algorithm: feed a commuting set, take the emitted Clifford
//      gates, build C as an actual unitary via QarpSimulator, and verify
//      C · P_i · C† equals ±Z_only_i for each input row.

#include <gtest/gtest.h>

#include "qarpx/synthesis/pauli_exponential.h"

#include "gate_test_helpers.h"

#include <stdexcept>
#include <vector>

namespace qarpx::test {

using qarpx::synthesis::detail::apply_cx;
using qarpx::synthesis::detail::apply_h;
using qarpx::synthesis::detail::apply_s;
using qarpx::synthesis::detail::apply_sdg;
using qarpx::synthesis::detail::compute_basis_change_clifford;
using qarpx::synthesis::detail::pauli_strings_to_tableau;

namespace {

// Build the matrix of a Pauli string on n qubits in the qarpx LSB-first basis
// (qubit 0 innermost, matching qarp_conventions.md §1).
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
        result = kron(m, result);  // outer dims grow MSB-first → qubit 0 innermost
    }
    return result;
}

// Build the matrix of a (possibly negated) Z-only Pauli string from a
// per-qubit z-bit vector.
Mat z_only_matrix(const std::vector<bool>& z_row, int n, bool negate) {
    Mat result = Mat::Ones(1, 1);
    for (int q = 0; q < n; ++q) {
        Mat m = z_row[q] ? analytic_z() : Mat::Identity(2, 2);
        result = kron(m, result);
    }
    return negate ? Mat(-result) : result;
}

// Run the full integration check on a commuting Pauli set.
void verify_basis_change_unitary(
    const std::vector<PauliString>& paulis, int n_qubits, double tol = 1e-10)
{
    auto result = compute_basis_change_clifford(paulis, n_qubits);

    // Build C as an actual unitary by simulating the emitted Clifford.
    // build_unitary applies an empty command list correctly (returns I).
    Mat C = build_unitary(result.clifford, n_qubits);
    Mat C_dag = C.adjoint();

    for (std::size_t i = 0; i < paulis.size(); ++i) {
        const Mat P_i = pauli_matrix(paulis[i], n_qubits);
        const Mat Z_i = z_only_matrix(result.z_rows[i], n_qubits, result.signs[i]);
        const Mat conjugated = C * P_i * C_dag;

        std::ostringstream tag;
        tag << "row " << i << " (" << to_string(paulis[i]) << "): C·P·C† should equal "
            << (result.signs[i] ? "-" : "+") << "Z_only";
        EXPECT_TRUE(expect_unitary_close(conjugated, Z_i, tol, tag.str().c_str()));
    }
}

}  // anonymous namespace

// ── Tableau Clifford updates ──────────────────────────────────────────────

TEST(BasisChangeClifford_Tableau, HSwapsXAndZ) {
    auto T = pauli_strings_to_tableau({parse_pauli_string("X")}, 1);
    apply_h(T, 0);
    EXPECT_FALSE(T.x[0][0]);  // X → Z
    EXPECT_TRUE(T.z[0][0]);
    EXPECT_FALSE(T.sign[0]);

    auto T2 = pauli_strings_to_tableau({parse_pauli_string("Z")}, 1);
    apply_h(T2, 0);
    EXPECT_TRUE(T2.x[0][0]);   // Z → X
    EXPECT_FALSE(T2.z[0][0]);
    EXPECT_FALSE(T2.sign[0]);
}

TEST(BasisChangeClifford_Tableau, HFlipsSignOnY) {
    // H Y H = -Y.
    auto T = pauli_strings_to_tableau({parse_pauli_string("Y")}, 1);
    apply_h(T, 0);
    EXPECT_TRUE(T.x[0][0]);    // Y stays Y
    EXPECT_TRUE(T.z[0][0]);
    EXPECT_TRUE(T.sign[0]);    // sign flipped
}

TEST(BasisChangeClifford_Tableau, SMapsXToYNoSign) {
    // S X S† = +Y.
    auto T = pauli_strings_to_tableau({parse_pauli_string("X")}, 1);
    apply_s(T, 0);
    EXPECT_TRUE(T.x[0][0]);
    EXPECT_TRUE(T.z[0][0]);
    EXPECT_FALSE(T.sign[0]);
}

TEST(BasisChangeClifford_Tableau, SFlipsSignOnY) {
    // S Y S† = -X.
    auto T = pauli_strings_to_tableau({parse_pauli_string("Y")}, 1);
    apply_s(T, 0);
    EXPECT_TRUE(T.x[0][0]);
    EXPECT_FALSE(T.z[0][0]);
    EXPECT_TRUE(T.sign[0]);
}

TEST(BasisChangeClifford_Tableau, SdgMapsYToXNoSign) {
    // Sdg Y Sdg† = +X — the phase-clean Y → Z basis change uses Sdg, not S.
    auto T = pauli_strings_to_tableau({parse_pauli_string("Y")}, 1);
    apply_sdg(T, 0);
    EXPECT_TRUE(T.x[0][0]);
    EXPECT_FALSE(T.z[0][0]);
    EXPECT_FALSE(T.sign[0]);
}

TEST(BasisChangeClifford_Tableau, SdgFlipsSignOnX) {
    // Sdg X Sdg† = -Y.
    auto T = pauli_strings_to_tableau({parse_pauli_string("X")}, 1);
    apply_sdg(T, 0);
    EXPECT_TRUE(T.x[0][0]);
    EXPECT_TRUE(T.z[0][0]);
    EXPECT_TRUE(T.sign[0]);
}

TEST(BasisChangeClifford_Tableau, CXMapsXcToXcXt) {
    // CX X_c CX = X_c X_t.
    auto T = pauli_strings_to_tableau({parse_pauli_string("XI")}, 2);
    apply_cx(T, 0, 1);
    EXPECT_TRUE(T.x[0][0]);   // X stays at control
    EXPECT_TRUE(T.x[0][1]);   // and propagates to target
    EXPECT_FALSE(T.z[0][0]);
    EXPECT_FALSE(T.z[0][1]);
    EXPECT_FALSE(T.sign[0]);
}

TEST(BasisChangeClifford_Tableau, CXMapsZtToZcZt) {
    // CX Z_t CX = Z_c Z_t.
    auto T = pauli_strings_to_tableau({parse_pauli_string("IZ")}, 2);
    apply_cx(T, 0, 1);
    EXPECT_FALSE(T.x[0][0]);
    EXPECT_FALSE(T.x[0][1]);
    EXPECT_TRUE(T.z[0][0]);   // Z propagates to control
    EXPECT_TRUE(T.z[0][1]);   // and stays on target
    EXPECT_FALSE(T.sign[0]);
}

TEST(BasisChangeClifford_Tableau, CXFlipsSignOnXcZt) {
    // CX X_c Z_t CX = -Y_c Y_t.
    auto T = pauli_strings_to_tableau({parse_pauli_string("XZ")}, 2);
    apply_cx(T, 0, 1);
    EXPECT_TRUE(T.x[0][0]);   // Y_c
    EXPECT_TRUE(T.z[0][0]);
    EXPECT_TRUE(T.x[0][1]);   // Y_t
    EXPECT_TRUE(T.z[0][1]);
    EXPECT_TRUE(T.sign[0]);   // sign flipped
}

TEST(BasisChangeClifford_Tableau, CXSelfInverse) {
    // (CX)² = I as a tableau action.  Random-ish input.
    auto T = pauli_strings_to_tableau({
        parse_pauli_string("XY"),
        parse_pauli_string("YZ"),
        parse_pauli_string("ZX"),
    }, 2);
    auto T2 = T;  // copy
    apply_cx(T, 0, 1);
    apply_cx(T, 0, 1);
    EXPECT_EQ(T.x, T2.x);
    EXPECT_EQ(T.z, T2.z);
    EXPECT_EQ(T.sign, T2.sign);
}

// ── compute_basis_change_clifford: structural cases ───────────────────────

TEST(BasisChangeClifford_Algorithm, EmptyInput) {
    auto result = compute_basis_change_clifford({}, 2);
    EXPECT_TRUE(result.clifford.empty());
    EXPECT_TRUE(result.z_rows.empty());
    EXPECT_TRUE(result.signs.empty());
}

// Regression: pre-check, {X, Z} returned z_rows [[1],[0]] / signs [+,+] with
// no error — the Z row collapsed to identity, read as expectation +1.
TEST(BasisChangeClifford_Algorithm, NonCommutingInputThrows) {
    EXPECT_THROW(
        compute_basis_change_clifford(
            {parse_pauli_string("X"), parse_pauli_string("Z")}, 1),
        std::invalid_argument);
    EXPECT_THROW(
        compute_basis_change_clifford(
            {parse_pauli_string("XX"), parse_pauli_string("XY")}, 2),
        std::invalid_argument);
}

TEST(BasisChangeClifford_Algorithm, AllIdentityRows) {
    auto result = compute_basis_change_clifford(
        {parse_pauli_string("II"), parse_pauli_string("II")}, 2);
    EXPECT_TRUE(result.clifford.empty());
    ASSERT_EQ(result.z_rows.size(), 2u);
    for (const auto& row : result.z_rows) {
        for (bool z : row) EXPECT_FALSE(z);
    }
}

TEST(BasisChangeClifford_Algorithm, SingleZ_NoClifford) {
    auto result = compute_basis_change_clifford(
        {parse_pauli_string("ZII")}, 3);
    EXPECT_TRUE(result.clifford.empty());
    ASSERT_EQ(result.z_rows.size(), 1u);
    EXPECT_TRUE (result.z_rows[0][0]);
    EXPECT_FALSE(result.z_rows[0][1]);
    EXPECT_FALSE(result.z_rows[0][2]);
    EXPECT_FALSE(result.signs[0]);
}

TEST(BasisChangeClifford_Algorithm, SingleX_EmitsH) {
    auto result = compute_basis_change_clifford(
        {parse_pauli_string("XII")}, 3);
    ASSERT_EQ(result.clifford.size(), 1u);
    EXPECT_EQ(result.clifford[0].gate, GateType::H);
    EXPECT_EQ(result.clifford[0].qubits[0], 0u);
    EXPECT_TRUE(result.z_rows[0][0]);
    EXPECT_FALSE(result.signs[0]);
}

TEST(BasisChangeClifford_Algorithm, SingleY_EmitsSdgThenH_PhaseClean) {
    auto result = compute_basis_change_clifford(
        {parse_pauli_string("YII")}, 3);
    ASSERT_EQ(result.clifford.size(), 2u);
    EXPECT_EQ(result.clifford[0].gate, GateType::Sdg);
    EXPECT_EQ(result.clifford[1].gate, GateType::H);
    EXPECT_TRUE (result.z_rows[0][0]);
    EXPECT_FALSE(result.z_rows[0][1]);
    EXPECT_FALSE(result.z_rows[0][2]);
    EXPECT_FALSE(result.signs[0]);  // Sdg + H is phase-clean for Y → Z
}

TEST(BasisChangeClifford_Algorithm, MultiQubitX_AllRowsBecomeZOnly) {
    // After processing, every row must have x = false everywhere on the
    // post-Clifford tableau.  Check via the unitary integration test.
    verify_basis_change_unitary({parse_pauli_string("XXX")}, 3);
}

// ── Integration: C · P · C† == ±Z_only for every input row ───────────────

TEST(BasisChangeClifford_Algorithm, BellStateStabilisers) {
    // {XX, YY, ZZ} is the Bell-state stabiliser family — pairwise commuting.
    verify_basis_change_unitary({
        parse_pauli_string("XX"),
        parse_pauli_string("YY"),
        parse_pauli_string("ZZ"),
    }, 2);
}

TEST(BasisChangeClifford_Algorithm, ZParityFamily_n3) {
    // Z-only set: nothing to do, but the algorithm should produce a no-op.
    verify_basis_change_unitary({
        parse_pauli_string("ZZI"),
        parse_pauli_string("IZZ"),
        parse_pauli_string("ZIZ"),
    }, 3);
}

TEST(BasisChangeClifford_Algorithm, MixedXZ_n3) {
    // {XXI, IXX} — pairwise commuting (each pair shares two X's, both anticommute → 2 → commute).
    verify_basis_change_unitary({
        parse_pauli_string("XXI"),
        parse_pauli_string("IXX"),
    }, 3);
}

TEST(BasisChangeClifford_Algorithm, FullPauliMix_n3) {
    // {XYZ, YXZ, ZZI} — pairwise commuting:
    //   XYZ·YXZ: X·Y antic, Y·X antic, Z·Z commute → 2 antic → commute.
    //   XYZ·ZZI: X·Z antic, Y·Z antic, Z·I commute → 2 antic → commute.
    //   YXZ·ZZI: Y·Z antic, X·Z antic, Z·I commute → 2 antic → commute.
    auto a = parse_pauli_string("XYZ");
    auto b = parse_pauli_string("YXZ");
    auto c = parse_pauli_string("ZZI");
    ASSERT_TRUE(commutes(a, b));
    ASSERT_TRUE(commutes(a, c));
    ASSERT_TRUE(commutes(b, c));
    verify_basis_change_unitary({a, b, c}, 3);
}

TEST(BasisChangeClifford_Algorithm, FourQubitChain) {
    // Heisenberg-style commuting set.
    auto a = parse_pauli_string("ZZII");
    auto b = parse_pauli_string("IZZI");
    auto c = parse_pauli_string("IIZZ");
    auto d = parse_pauli_string("XXXX");
    ASSERT_TRUE(commutes(a, b));
    ASSERT_TRUE(commutes(a, c));
    ASSERT_TRUE(commutes(a, d));
    ASSERT_TRUE(commutes(b, c));
    ASSERT_TRUE(commutes(b, d));
    ASSERT_TRUE(commutes(c, d));
    verify_basis_change_unitary({a, b, c, d}, 4);
}

TEST(BasisChangeClifford_Algorithm, PreservesNonRedundantCount) {
    // After diagonalisation, the algorithm should emit one pivot per
    // linearly-independent generator.  Sanity: 2 rank-2 generators → 2 pivots.
    auto result = compute_basis_change_clifford(
        {parse_pauli_string("XX"), parse_pauli_string("YY")}, 2);
    int nonzero_rows = 0;
    for (const auto& row : result.z_rows) {
        for (bool b : row) if (b) { ++nonzero_rows; break; }
    }
    EXPECT_EQ(nonzero_rows, 2);
}

}  // namespace qarpx::test
