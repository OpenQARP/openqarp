// Tests for the SymEngine symbolic coefficient backend — compiled only when
// QARP_WITH_SYMENGINE is enabled (empty TU otherwise).

#ifdef QARP_WITH_SYMENGINE

#include <gtest/gtest.h>

#include "qarpx/operators/transforms.h"

#include <complex>

namespace qarpx::ops::test {

namespace {

using Complex = std::complex<double>;

SymCoeff sym(const std::string& text) { return parse_symbolic(text); }

QubitOperator qop(std::string_view s, Complex c = 1.0) {
    return QubitOperator::from_term_string(s, c);
}
QubitOperator qop_sym(std::string_view s, const std::string& expr) {
    return QubitOperator::from_term_string(s, sym(expr));
}

}  // namespace

TEST(OperatorsSymbolic, ParsePrintRoundTrip) {
    EXPECT_EQ(symbolic_to_string(sym("2*a")), "2*a");
    EXPECT_EQ(symbolic_to_string(sym("a + b")), symbolic_to_string(sym("b + a")));
    EXPECT_TRUE(sym("3").is_numeric());
    EXPECT_FALSE(sym("a").is_numeric());
}

TEST(OperatorsSymbolic, PromotionOnMixedArithmetic) {
    QubitOperator numeric = qop("X0", 0.5);
    const QubitOperator symbolic = qop_sym("Y1", "a");
    EXPECT_FALSE(numeric.is_symbolic());
    EXPECT_TRUE(symbolic.is_symbolic());

    numeric += symbolic;  // promotes in place
    EXPECT_TRUE(numeric.is_symbolic());
    EXPECT_EQ(numeric.n_terms(), 2u);

    // Products promote too, on either side.
    EXPECT_TRUE((qop("X0") * symbolic).is_symbolic());
    EXPECT_TRUE((symbolic * qop("X0")).is_symbolic());
}

TEST(OperatorsSymbolic, SymbolicCoefficientsAreNeverAutoErased) {
    // A tiny numeric multiple of a symbol is not "small".
    QubitOperator a = qop_sym("X0", "a");
    a += QubitOperator::from_term_string("X0", Complex(1e-12, 0.0));
    EXPECT_EQ(a.n_terms(), 1u);

    // …but an exact symbolic cancellation simplifies to numeric zero and IS
    // erased by the += rule.
    QubitOperator b = qop_sym("X0", "a");
    b -= qop_sym("X0", "a");
    EXPECT_EQ(b.n_terms(), 0u);
}

TEST(OperatorsSymbolic, ConjugationConjugatesSymbols) {
    // Build the expected coefficients through the trait (the SymEngine
    // string parser may model "conjugate(a)" as a generic function symbol,
    // which is not structurally the conjugate node).
    const SymCoeff conj_a = CoeffTraits<SymCoeff>::conj(sym("a"));

    const QubitOperator h = qop_sym("X0", "a").hermitian_conjugated();
    EXPECT_TRUE(h.equals(QubitOperator::from_term_string("X0", conj_a)));

    // Fermion side: dagger reverses/flips the key as well.
    const FermionOperator f =
        FermionOperator::from_term_string("2^ 1", sym("a")).hermitian_conjugated();
    EXPECT_TRUE(f.equals(FermionOperator::from_term_string("1^ 2", conj_a)));
}

TEST(OperatorsSymbolic, IscloseMixesToleranceAndStructure) {
    // Numeric constants inside a symbolic payload still compare with the
    // openfermion tolerance rules.
    QubitOperator a = qop_sym("X0", "a") + qop("Z1", 1.0);
    QubitOperator b = qop_sym("X0", "a") + qop("Z1", 1.0 + 5e-9);
    EXPECT_TRUE(a.equals(b));

    // Structurally different symbols are not equal.
    EXPECT_FALSE(qop_sym("X0", "a").equals(qop_sym("X0", "b")));
    // Same expression written differently is (expand-level) equal.
    EXPECT_TRUE(qop_sym("X0", "a*(1+b)").equals(qop_sym("X0", "a + a*b")));
}

TEST(OperatorsSymbolic, SubstituteDemotesWhenFullyNumeric) {
    const QubitOperator h = qop_sym("X0", "a") + qop_sym("Y1", "2*b") + qop("Z2", 0.5);
    EXPECT_EQ(h.free_symbols(), (std::vector<std::string>{"a", "b"}));

    const QubitOperator partial = h.substituted({{"a", Complex(0.25, 0.0)}});
    EXPECT_TRUE(partial.is_symbolic());
    EXPECT_EQ(partial.free_symbols(), (std::vector<std::string>{"b"}));

    const QubitOperator full = partial.substituted({{"b", Complex(1.5, 0.0)}});
    EXPECT_FALSE(full.is_symbolic());
    const QubitOperator expected = qop("X0", 0.25) + qop("Y1", 3.0) + qop("Z2", 0.5);
    EXPECT_TRUE(full.equals(expected));
}

TEST(OperatorsSymbolic, TransformsCarrySymbolicCoefficients) {
    const FermionOperator f = FermionOperator::from_term_string("2^ 1", sym("a"));
    const QubitOperator jw_sym = jordan_wigner(f);
    EXPECT_TRUE(jw_sym.is_symbolic());
    EXPECT_EQ(jw_sym.n_terms(), 4u);

    // JW(a·op) == a·JW(op): substitute and compare against the numeric path.
    const Complex value(0.75, -0.5);
    const QubitOperator numeric_jw =
        jordan_wigner(FermionOperator::from_term_string("2^ 1", value));
    EXPECT_TRUE(jw_sym.substituted({{"a", value}}).equals(numeric_jw));

    EXPECT_TRUE(bravyi_kitaev(f).is_symbolic());
    EXPECT_TRUE(parity_transform(f, 3).is_symbolic());
}

TEST(OperatorsSymbolic, NumericAccessorsGuard) {
    const QubitOperator h = qop_sym("X0", "a");
    EXPECT_THROW((void)h.numeric(), std::invalid_argument);
    // Absent constant term evaluates to numeric zero even on the symbolic
    // payload; a genuinely symbolic constant throws.
    EXPECT_EQ(h.constant(), Complex(0.0, 0.0));
    const QubitOperator c = QubitOperator::from_packed_term(PackedPauli{}, sym("a"));
    EXPECT_THROW((void)c.constant(), std::invalid_argument);
}

TEST(OperatorsSymbolic, StrUsesExpressionText) {
    EXPECT_EQ(qop_sym("X0", "2*a").str(), "2*a [X0]");
}

}  // namespace qarpx::ops::test

#endif  // QARP_WITH_SYMENGINE
