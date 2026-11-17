// Tests for the C++ FermionOperator facade + engine: ladder-string parsing,
// verbatim (never normal-ordered) term keys, concatenating products, dagger,
// and the shared openfermion arithmetic semantics.

#include <gtest/gtest.h>

#include "qarpx/operators/fermion_operator.h"

#include <complex>
#include <utility>
#include <vector>

namespace qarpx::ops::test {

namespace {

using Complex = std::complex<double>;
using Ladder = std::vector<std::pair<uint32_t, uint32_t>>;  // (index, action)

std::vector<std::pair<Ladder, Complex>> terms_of(const FermionOperator& op) {
    std::vector<std::pair<Ladder, Complex>> out;
    for (const auto& e : op.numeric().terms) {
        Ladder ladder;
        for (const uint32_t packed : e.key.ops)
            ladder.emplace_back(FermionKey::index_of(packed), FermionKey::action_of(packed));
        out.emplace_back(std::move(ladder), e.coeff);
    }
    return out;
}

FermionOperator fop(std::string_view s, Complex c = 1.0) {
    return FermionOperator::from_term_string(s, c);
}

}  // namespace

// ── Construction ──────────────────────────────────────────────────────────

TEST(OperatorsFermionOperator, StringParsingBasics) {
    const auto terms = terms_of(fop("2^ 1", -0.5));
    ASSERT_EQ(terms.size(), 1u);
    EXPECT_EQ(terms[0].first, Ladder({{2, 1}, {1, 0}}));
    EXPECT_EQ(terms[0].second, Complex(-0.5, 0.0));

    EXPECT_EQ(terms_of(fop("0^"))[0].first, Ladder({{0, 1}}));
    EXPECT_TRUE(terms_of(fop("", 2.0))[0].first.empty());  // identity term

    // Trailing whitespace tolerated (used in qarp tests: "5^ 5^ ").
    EXPECT_EQ(terms_of(fop("5^ 5^ "))[0].first, Ladder({{5, 1}, {5, 1}}));
}

TEST(OperatorsFermionOperator, TupleFormMatchesStringForm) {
    const Ladder ladder = {{2, 1}, {0, 0}};
    EXPECT_TRUE(FermionOperator::from_ladder_ops(ladder, -1.0).equals(fop("2^ 0", -1.0)));
    const Ladder bad = {{2, 3}};
    EXPECT_THROW((void)FermionOperator::from_ladder_ops(bad), std::invalid_argument);
}

TEST(OperatorsFermionOperator, BracketFormFoldsPrefixCoefficient) {
    EXPECT_TRUE(fop("1.5 [2^ 3]", 2.0).equals(fop("2^ 3", 3.0)));
    EXPECT_TRUE(fop("[2^]").equals(fop("2^")));
}

TEST(OperatorsFermionOperator, InvalidFactorsThrow) {
    EXPECT_THROW((void)fop("a"), std::invalid_argument);
    EXPECT_THROW((void)fop("^"), std::invalid_argument);
    EXPECT_THROW((void)fop("2^^"), std::invalid_argument);
    EXPECT_THROW((void)fop("2x"), std::invalid_argument);
}

TEST(OperatorsFermionOperator, CtorKeepsZeroCoefficient) {
    const FermionOperator op = fop("0^", 0.0);
    EXPECT_EQ(op.n_terms(), 1u);
    EXPECT_EQ(terms_of(op)[0].second, Complex(0.0, 0.0));
}

// ── Term keys are verbatim: no canonicalization ever ──────────────────────

TEST(OperatorsFermionOperator, NoNormalOrdering) {
    // "1 2^" stays exactly as written — annihilation before creation.
    EXPECT_EQ(terms_of(fop("1 2^"))[0].first, Ladder({{1, 0}, {2, 1}}));
    // "0 0" and "0^ 0^" are legitimate distinct keys (they vanish
    // physically, but openfermion stores them verbatim).
    EXPECT_EQ(terms_of(fop("0 0"))[0].first, Ladder({{0, 0}, {0, 0}}));
    EXPECT_FALSE(fop("0 0").equals(fop("0^ 0^")));
}

TEST(OperatorsFermionOperator, ProductConcatenatesSequences) {
    const FermionOperator prod = fop("2^ 1", 2.0) * fop("3^ 0", 0.5);
    const auto terms = terms_of(prod);
    ASSERT_EQ(terms.size(), 1u);
    EXPECT_EQ(terms[0].first, Ladder({{2, 1}, {1, 0}, {3, 1}, {0, 0}}));
    EXPECT_EQ(terms[0].second, Complex(1.0, 0.0));
}

TEST(OperatorsFermionOperator, ProductKeepsExactZeroCollisions) {
    // Distinct term pairs whose concatenations collide: "2^"·"1 3^" and
    // "2^ 1"·"3^" both produce (2^ 1 3^); with coefficients 1·1 and 1·(-1)
    // they cancel exactly — and * never compacts, so the zero survives.
    const FermionOperator a = fop("2^") + fop("2^ 1");
    const FermionOperator b = fop("1 3^") + fop("3^", -1.0);
    const auto terms = terms_of(a * b);
    ASSERT_EQ(terms.size(), 3u);
    EXPECT_EQ(terms[0].first, Ladder({{2, 1}, {1, 0}, {3, 1}}));
    EXPECT_EQ(terms[0].second, Complex(0.0, 0.0));
    EXPECT_EQ(terms[1].first, Ladder({{2, 1}, {3, 1}}));
    EXPECT_EQ(terms[1].second, Complex(-1.0, 0.0));
    EXPECT_EQ(terms[2].first, Ladder({{2, 1}, {1, 0}, {1, 0}, {3, 1}}));
    EXPECT_EQ(terms[2].second, Complex(1.0, 0.0));
}

// ── Dagger ────────────────────────────────────────────────────────────────

TEST(OperatorsFermionOperator, HermitianConjugatedReversesAndFlips) {
    const auto terms = terms_of(fop("2^ 1", Complex(1.0, 2.0)).hermitian_conjugated());
    ASSERT_EQ(terms.size(), 1u);
    EXPECT_EQ(terms[0].first, Ladder({{1, 1}, {2, 0}}));  // reversed, flipped
    EXPECT_EQ(terms[0].second, Complex(1.0, -2.0));
}

TEST(OperatorsFermionOperator, DaggerRoundTrip) {
    const FermionOperator op = fop("3^ 1 2^", Complex(0.5, -0.25)) + fop("0^", 2.0);
    EXPECT_TRUE(op.hermitian_conjugated().hermitian_conjugated().equals(op));
}

// ── Shared arithmetic semantics ───────────────────────────────────────────

TEST(OperatorsFermionOperator, AdditionAccumulatesAndCancels) {
    FermionOperator a = fop("2^ 1");
    a += fop("2^ 1", -1.0);
    EXPECT_EQ(a.n_terms(), 0u);

    FermionOperator b = fop("2^ 1");
    b -= fop("1 2^");  // different key — no cancellation
    EXPECT_EQ(b.n_terms(), 2u);
}

TEST(OperatorsFermionOperator, ScalarOpsAndPow) {
    const FermionOperator a = fop("2^ 1", 0.5);
    EXPECT_TRUE((a * Complex(2.0, 0.0)).equals(fop("2^ 1")));
    EXPECT_TRUE((a / Complex(0.5, 0.0)).equals(fop("2^ 1")));
    EXPECT_TRUE((-a).equals(fop("2^ 1", -0.5)));
    EXPECT_TRUE(a.pow(2).equals(a * a));
    EXPECT_THROW((void)a.pow(-2), std::invalid_argument);
}

TEST(OperatorsFermionOperator, CountQubits) {
    EXPECT_EQ(FermionOperator().count_qubits(), 0);
    EXPECT_EQ(FermionOperator::identity(3.0).count_qubits(), 0);
    EXPECT_EQ(fop("2^ 1").count_qubits(), 3);
    EXPECT_EQ(fop("0^ 17").count_qubits(), 18);
}

TEST(OperatorsFermionOperator, GetOperatorsPreservesOrder) {
    const FermionOperator op = fop("2^ 1", 0.5) + fop("0^", -1.0);
    const auto ops = op.get_operators();
    ASSERT_EQ(ops.size(), 2u);
    EXPECT_TRUE(ops[0].equals(fop("2^ 1", 0.5)));
    EXPECT_TRUE(ops[1].equals(fop("0^", -1.0)));
}

// ── Formatting ────────────────────────────────────────────────────────────

TEST(OperatorsFermionOperator, StrMatchesOpenfermionShape) {
    EXPECT_EQ(FermionOperator().str(), "0");
    EXPECT_EQ(fop("2^ 1").str(), "(1+0j) [2^ 1]");
    EXPECT_EQ((fop("2^ 1", 0.5) + fop("0", -2.0)).str(),
              "(0.5+0j) [2^ 1] +\n(-2+0j) [0]");
}

}  // namespace qarpx::ops::test
