// Tests for the C++ QubitOperator facade + engine: openfermion-compatible
// construction (string/tuple/_simplify folding), arithmetic semantics
// (+= erases only exact cancellations, * keeps exact zeros, scalar folding),
// isclose tolerance rules, conjugation, compress, formatting.

#include <gtest/gtest.h>

#include "qarpx/operators/qubit_operator.h"

#include <complex>
#include <utility>
#include <vector>

namespace qarpx::ops::test {

namespace {

using Complex = std::complex<double>;
using Factors = std::vector<std::pair<uint32_t, char>>;

// (term key as (qubit, letter) pairs, coefficient) in .terms order.
std::vector<std::pair<Factors, Complex>> terms_of(const QubitOperator& op) {
    std::vector<std::pair<Factors, Complex>> out;
    Factors factors;
    for (const auto& e : op.numeric().terms) {
        packed_to_index_pairs(e.key, factors);
        out.emplace_back(factors, e.coeff);
    }
    return out;
}

QubitOperator qop(std::string_view s, Complex c = 1.0) {
    return QubitOperator::from_term_string(s, c);
}

}  // namespace

// ── Construction ──────────────────────────────────────────────────────────

TEST(OperatorsQubitOperator, DefaultIsAdditiveZero) {
    QubitOperator zero;
    EXPECT_EQ(zero.n_terms(), 0u);
    QubitOperator a = qop("X0", 0.5);
    EXPECT_TRUE((a + zero).equals(a));
}

TEST(OperatorsQubitOperator, CtorKeepsZeroCoefficient) {
    const QubitOperator op = qop("X0", 0.0);
    const auto terms = terms_of(op);
    ASSERT_EQ(terms.size(), 1u);
    EXPECT_EQ(terms[0].first, Factors({{0, 'X'}}));
    EXPECT_EQ(terms[0].second, Complex(0.0, 0.0));
}

TEST(OperatorsQubitOperator, StringParsingBasics) {
    const auto terms = terms_of(qop("X0 Z2 Y3", 0.5));
    ASSERT_EQ(terms.size(), 1u);
    EXPECT_EQ(terms[0].first, Factors({{0, 'X'}, {2, 'Z'}, {3, 'Y'}}));
    EXPECT_EQ(terms[0].second, Complex(0.5, 0.0));

    // Empty string is the identity (constant) term.
    const auto id_terms = terms_of(qop("", 2.0));
    ASSERT_EQ(id_terms.size(), 1u);
    EXPECT_TRUE(id_terms[0].first.empty());

    // Factor order in the input doesn't matter (openfermion sorts by qubit).
    EXPECT_TRUE(qop("Z2 X0").equals(qop("X0 Z2")));
}

TEST(OperatorsQubitOperator, RepeatedQubitFoldsLikeSimplify) {
    // The "X0 X0" identity-carrier idiom used across qarp.
    const auto id_terms = terms_of(qop("X0 X0", 0.5));
    ASSERT_EQ(id_terms.size(), 1u);
    EXPECT_TRUE(id_terms[0].first.empty());
    EXPECT_EQ(id_terms[0].second, Complex(0.5, 0.0));

    // X0·Y0 = iZ0 — phase folds into the coefficient.
    const auto z_terms = terms_of(qop("X0 Y0", 1.0));
    ASSERT_EQ(z_terms.size(), 1u);
    EXPECT_EQ(z_terms[0].first, Factors({{0, 'Z'}}));
    EXPECT_EQ(z_terms[0].second, Complex(0.0, 1.0));
}

TEST(OperatorsQubitOperator, TupleFormMatchesStringForm) {
    const Factors factors = {{3, 'Y'}, {0, 'X'}};
    const QubitOperator from_tuple = QubitOperator::from_factors(factors, Complex(0.0, 1.0));
    EXPECT_TRUE(from_tuple.equals(qop("X0 Y3", Complex(0.0, 1.0))));
}

TEST(OperatorsQubitOperator, BracketFormFoldsPrefixCoefficient) {
    EXPECT_TRUE(qop("1.5 [X0 Z1]", 2.0).equals(qop("X0 Z1", 3.0)));
    EXPECT_TRUE(qop("[X0]").equals(qop("X0")));
    EXPECT_TRUE(qop("0.5j [X0]").equals(qop("X0", Complex(0.0, 0.5))));
}

TEST(OperatorsQubitOperator, InvalidFactorsThrow) {
    EXPECT_THROW((void)qop("A0"), std::invalid_argument);
    EXPECT_THROW((void)qop("X"), std::invalid_argument);
    EXPECT_THROW((void)qop("X0Y1"), std::invalid_argument);
    EXPECT_THROW((void)qop("I0"), std::invalid_argument);
    const Factors bad = {{0, 'I'}};
    EXPECT_THROW((void)QubitOperator::from_factors(bad), std::invalid_argument);
}

// ── Addition / subtraction ────────────────────────────────────────────────

TEST(OperatorsQubitOperator, AdditionAccumulatesInAddendOrder) {
    QubitOperator a = qop("X0");
    a += qop("Y1", 2.0);
    a += qop("X0", 0.5);

    const auto terms = terms_of(a);
    ASSERT_EQ(terms.size(), 2u);
    EXPECT_EQ(terms[0].first, Factors({{0, 'X'}}));
    EXPECT_EQ(terms[0].second, Complex(1.5, 0.0));
    EXPECT_EQ(terms[1].first, Factors({{1, 'Y'}}));
}

TEST(OperatorsQubitOperator, AdditionCancellationErasesTerm) {
    QubitOperator a = qop("X0");
    a -= qop("X0");
    EXPECT_EQ(a.n_terms(), 0u);
}

// The three tests below pin the deviation from openfermion, which erases at
// EQ_TOLERANCE here.  Near-cancellation is a *residual*, not nothing, and only
// the caller knows whether it is negligible — compress() is where that is said.
TEST(OperatorsQubitOperator, AdditionKeepsNearCancellationResidual) {
    QubitOperator b = qop("X0");
    b += qop("X0", -1.0 + 5e-9);
    ASSERT_EQ(b.n_terms(), 1u);
    // 1.0 + (-1.0 + 5e-9) is 4.999999969e-9 — the residual is what float
    // subtraction leaves, not the literal, so compare relatively.
    EXPECT_NEAR(std::abs(terms_of(b)[0].second), 5e-9, 5e-16);

    b.compress();  // opting in to truncation does erase it
    EXPECT_EQ(b.n_terms(), 0u);
}

TEST(OperatorsQubitOperator, AdditionKeepsNewSmallTerm) {
    QubitOperator a;
    a += qop("X0", 5e-9);
    ASSERT_EQ(a.n_terms(), 1u);
    EXPECT_EQ(terms_of(a)[0].second, Complex(5e-9, 0.0));
}

// The bug the deviation exists for: the old rule tested the addend's terms and
// never the receiver's, so the same two operands gave different results
// depending on which side they were written.
TEST(OperatorsQubitOperator, AdditionIsCommutative) {
    const QubitOperator big = qop("Z0", 1.0);
    const QubitOperator tiny = qop("X0", 1e-9);

    QubitOperator ab = big;
    ab += tiny;
    QubitOperator ba = tiny;
    ba += big;

    ASSERT_EQ(ab.n_terms(), 2u);
    ASSERT_EQ(ba.n_terms(), 2u);
    EXPECT_TRUE(ab.equals(ba));
}

// An absolute cutoff annihilates an operator whose whole norm is small, for no
// reason other than its units.
TEST(OperatorsQubitOperator, AdditionKeepsAllTermsOfASmallNormOperator) {
    QubitOperator small = qop("Z0", 1e-9);
    small += qop("X0", -1e-9);
    ASSERT_EQ(small.n_terms(), 2u);

    small *= Complex(1e9, 0.0);  // rescaling recovers an ordinary operator
    ASSERT_EQ(small.n_terms(), 2u);
    EXPECT_TRUE(small.equals(qop("Z0", 1.0) - qop("X0", 1.0)));
}

TEST(OperatorsQubitOperator, SelfAdditionIsSafe) {
    QubitOperator a = qop("X0", 0.5);
    a += a;
    const auto terms = terms_of(a);
    ASSERT_EQ(terms.size(), 1u);
    EXPECT_EQ(terms[0].second, Complex(1.0, 0.0));
}

// ── Multiplication ────────────────────────────────────────────────────────

TEST(OperatorsQubitOperator, ProductAppliesPauliAlgebra) {
    // (X0)(Y0) = iZ0 ; cross-qubit factors merge.
    EXPECT_TRUE((qop("X0") * qop("Y0")).equals(qop("Z0", Complex(0.0, 1.0))));
    EXPECT_TRUE((qop("X0", 2.0) * qop("Y1", 3.0)).equals(qop("X0 Y1", 6.0)));
}

TEST(OperatorsQubitOperator, ProductKeepsExactZeroCollisions) {
    // (X0 + Y0)(X0 - Y0): identity terms cancel exactly but stay as an
    // explicit zero (openfermion * never compacts); Z0 collects -2i.
    const QubitOperator a = qop("X0") + qop("Y0");
    const QubitOperator b = qop("X0") + qop("Y0", -1.0);
    const QubitOperator prod = a * b;

    const auto terms = terms_of(prod);
    ASSERT_EQ(terms.size(), 2u);
    EXPECT_TRUE(terms[0].first.empty());  // identity inserted first
    EXPECT_EQ(terms[0].second, Complex(0.0, 0.0));
    EXPECT_EQ(terms[1].first, Factors({{0, 'Z'}}));
    EXPECT_EQ(terms[1].second, Complex(0.0, -2.0));
}

TEST(OperatorsQubitOperator, InPlaceMultiplyMatchesBinary) {
    QubitOperator a = qop("X0") + qop("Z1", 0.5);
    const QubitOperator b = qop("Y0", 2.0);
    QubitOperator c = a;
    c *= b;
    EXPECT_TRUE(c.equals(a * b));
}

// ── Scalar operations ─────────────────────────────────────────────────────

TEST(OperatorsQubitOperator, ScalarMultiplyDivideNegate) {
    const QubitOperator a = qop("X0", 0.5) + qop("Y1", -2.0);
    EXPECT_TRUE((a * Complex(2.0, 0.0)).equals(qop("X0") + qop("Y1", -4.0)));
    EXPECT_TRUE((Complex(2.0, 0.0) * a).equals(a * Complex(2.0, 0.0)));
    EXPECT_TRUE((a / Complex(2.0, 0.0)).equals(qop("X0", 0.25) + qop("Y1", -1.0)));
    EXPECT_TRUE((-a).equals(qop("X0", -0.5) + qop("Y1", 2.0)));
}

TEST(OperatorsQubitOperator, ScalarMultiplyKeepsSmallCoefficients) {
    // Scalar scaling never compacts (openfermion scales dict values).
    QubitOperator a = qop("X0");
    a *= Complex(1e-12, 0.0);
    EXPECT_EQ(a.n_terms(), 1u);
}

TEST(OperatorsQubitOperator, ScalarAddFoldsIntoConstantByAssignment) {
    QubitOperator a = qop("X0");
    a += Complex(2.0, 0.0);
    EXPECT_EQ(a.constant(), Complex(2.0, 0.0));
    const auto terms = terms_of(a);
    ASSERT_EQ(terms.size(), 2u);
    EXPECT_TRUE(terms[1].first.empty());  // constant appended at the end

    // Subtracting it back leaves an explicit zero constant (assignment
    // semantics — no small-erase on the constant path).
    a -= Complex(2.0, 0.0);
    EXPECT_EQ(a.n_terms(), 2u);
    EXPECT_EQ(a.constant(), Complex(0.0, 0.0));
}

TEST(OperatorsQubitOperator, PowMatchesRepeatedProduct) {
    const QubitOperator a = qop("X0", 0.5) + qop("Z1");
    EXPECT_TRUE(a.pow(0).equals(QubitOperator::identity(1.0)));
    EXPECT_TRUE(a.pow(2).equals(a * a));
    EXPECT_THROW((void)a.pow(-1), std::invalid_argument);
}

// ── Equality (openfermion isclose) ────────────────────────────────────────

TEST(OperatorsQubitOperator, IscloseToleranceRules) {
    EXPECT_TRUE(qop("X0").equals(qop("X0", 1.0 + 5e-9)));
    EXPECT_FALSE(qop("X0").equals(qop("X0", 1.0 + 2e-8)));

    // Tolerance scales with max(1, |a|, |b|).
    EXPECT_TRUE(qop("X0", 1e6).equals(qop("X0", 1e6 + 1e-3)));

    // One-sided terms below the absolute tolerance are ignored.
    EXPECT_TRUE((qop("X0") + qop("Y1", 5e-9)).equals(qop("X0")));
    EXPECT_FALSE((qop("X0") + qop("Y1", 1e-3)).equals(qop("X0")));

    // Zero-coefficient one-sided terms compare equal to nothing at all.
    EXPECT_TRUE(qop("X0", 0.0).equals(QubitOperator()));
}

// ── Conjugation & friends ─────────────────────────────────────────────────

TEST(OperatorsQubitOperator, HermitianConjugatedConjugatesCoefficients) {
    const QubitOperator a = qop("X0", Complex(1.0, 2.0)) + qop("Z1", 0.5);
    const auto terms = terms_of(a.hermitian_conjugated());
    ASSERT_EQ(terms.size(), 2u);
    EXPECT_EQ(terms[0].first, Factors({{0, 'X'}}));
    EXPECT_EQ(terms[0].second, Complex(1.0, -2.0));
    EXPECT_EQ(terms[1].second, Complex(0.5, 0.0));
}

TEST(OperatorsQubitOperator, IsHermitian) {
    EXPECT_TRUE((qop("X0", 0.5) + qop("Z0 Z1", -1.0)).is_hermitian());
    EXPECT_FALSE(qop("X0", Complex(0.0, 1.0)).is_hermitian());
}

TEST(OperatorsQubitOperator, CountQubits) {
    EXPECT_EQ(QubitOperator().count_qubits(), 0);
    EXPECT_EQ(QubitOperator::identity(2.0).count_qubits(), 0);
    EXPECT_EQ(qop("X3").count_qubits(), 4);
    EXPECT_EQ((qop("X3") + qop("Z1 Y70")).count_qubits(), 71);
}

TEST(OperatorsQubitOperator, CompressDropsSmallAndRealifies) {
    QubitOperator a = qop("X0", Complex(1.0, 1e-10)) + qop("Y1", 1e-10);
    a.compress();
    const auto terms = terms_of(a);
    ASSERT_EQ(terms.size(), 1u);
    EXPECT_EQ(terms[0].second, Complex(1.0, 0.0));
}

TEST(OperatorsQubitOperator, GetOperatorsPreservesOrderAndZeros) {
    QubitOperator a = qop("X0", 0.5) + qop("Y1", 2.0);
    a += Complex(0.0, 0.0);  // explicit zero constant at the end
    const auto ops = a.get_operators();
    ASSERT_EQ(ops.size(), 3u);
    EXPECT_TRUE(ops[0].equals(qop("X0", 0.5)));
    EXPECT_TRUE(ops[1].equals(qop("Y1", 2.0)));
    EXPECT_EQ(ops[2].n_terms(), 1u);
    EXPECT_EQ(ops[2].constant(), Complex(0.0, 0.0));
}

// ── Formatting ────────────────────────────────────────────────────────────

TEST(OperatorsQubitOperator, StrMatchesOpenfermionShape) {
    EXPECT_EQ(QubitOperator().str(), "0");
    EXPECT_EQ(qop("X0 Y1", 0.5).str(), "(0.5+0j) [X0 Y1]");
    EXPECT_EQ((qop("X0", 0.5) + qop("Z1", Complex(0.0, -1.0))).str(),
              "(0.5+0j) [X0] +\n-1j [Z1]");
    EXPECT_EQ(QubitOperator::identity(2.0).str(), "(2+0j) []");
}

}  // namespace qarpx::ops::test
