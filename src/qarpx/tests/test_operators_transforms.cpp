// Golden tests for the C++ fermion→qubit transforms.  The expected term
// lists (keys, coefficients AND order) are transcribed verbatim from
// tests/test_fermionic/test_mappings.py, which pins openfermion's dict
// insertion order — the compatibility bar for the mapping rewrite.

#include <gtest/gtest.h>

#include "qarpx/operators/transforms.h"

#include <complex>
#include <utility>
#include <vector>

namespace qarpx::ops::test {

namespace {

using Complex = std::complex<double>;
using F = std::vector<std::pair<uint32_t, char>>;

struct GoldenTerm {
    F factors;
    Complex coeff;
};

FermionOperator fop(std::string_view s, Complex c = 1.0) {
    return FermionOperator::from_term_string(s, c);
}

void expect_golden(const QubitOperator& op, const std::vector<GoldenTerm>& expected) {
    std::vector<std::pair<F, Complex>> actual;
    F factors;
    for (const auto& e : op.numeric().terms) {
        packed_to_index_pairs(e.key, factors);
        actual.emplace_back(factors, e.coeff);
    }
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(actual[i].first, expected[i].factors) << "term index " << i;
        EXPECT_LT(std::abs(actual[i].second - expected[i].coeff), 1e-16)
            << "term index " << i;
    }
}

constexpr Complex J(double re, double im) { return {re, im}; }

}  // namespace

// ── Jordan-Wigner ─────────────────────────────────────────────────────────

TEST(OperatorsTransforms, JWEncodeOneElectron) {
    expect_golden(jordan_wigner(fop("2^ 1")),
                  {{{{1, 'Y'}, {2, 'X'}}, J(0, 0.25)},
                   {{{1, 'X'}, {2, 'X'}}, J(0.25, 0)},
                   {{{1, 'Y'}, {2, 'Y'}}, J(0.25, 0)},
                   {{{1, 'X'}, {2, 'Y'}}, J(0, -0.25)}});

    expect_golden(jordan_wigner(fop("1^ 5")),
                  {{{{1, 'Y'}, {2, 'Z'}, {3, 'Z'}, {4, 'Z'}, {5, 'X'}}, J(0, -0.25)},
                   {{{1, 'Y'}, {2, 'Z'}, {3, 'Z'}, {4, 'Z'}, {5, 'Y'}}, J(0.25, 0)},
                   {{{1, 'X'}, {2, 'Z'}, {3, 'Z'}, {4, 'Z'}, {5, 'X'}}, J(0.25, 0)},
                   {{{1, 'X'}, {2, 'Z'}, {3, 'Z'}, {4, 'Z'}, {5, 'Y'}}, J(0, 0.25)}});
}

TEST(OperatorsTransforms, JWEncodeTwoElectron) {
    expect_golden(
        jordan_wigner(fop("2^ 1 5^ 3")),
        {{{{1, 'Y'}, {2, 'X'}, {3, 'Y'}, {4, 'Z'}, {5, 'X'}}, J(-0.0625, 0)},
         {{{1, 'Y'}, {2, 'X'}, {3, 'X'}, {4, 'Z'}, {5, 'X'}}, J(0, 0.0625)},
         {{{1, 'Y'}, {2, 'X'}, {3, 'Y'}, {4, 'Z'}, {5, 'Y'}}, J(0, 0.0625)},
         {{{1, 'Y'}, {2, 'X'}, {3, 'X'}, {4, 'Z'}, {5, 'Y'}}, J(0.0625, 0)},
         {{{1, 'X'}, {2, 'X'}, {3, 'Y'}, {4, 'Z'}, {5, 'X'}}, J(0, 0.0625)},
         {{{1, 'X'}, {2, 'X'}, {3, 'X'}, {4, 'Z'}, {5, 'X'}}, J(0.0625, 0)},
         {{{1, 'X'}, {2, 'X'}, {3, 'Y'}, {4, 'Z'}, {5, 'Y'}}, J(0.0625, 0)},
         {{{1, 'X'}, {2, 'X'}, {3, 'X'}, {4, 'Z'}, {5, 'Y'}}, J(0, -0.0625)},
         {{{1, 'Y'}, {2, 'Y'}, {3, 'Y'}, {4, 'Z'}, {5, 'X'}}, J(0, 0.0625)},
         {{{1, 'Y'}, {2, 'Y'}, {3, 'X'}, {4, 'Z'}, {5, 'X'}}, J(0.0625, 0)},
         {{{1, 'Y'}, {2, 'Y'}, {3, 'Y'}, {4, 'Z'}, {5, 'Y'}}, J(0.0625, 0)},
         {{{1, 'Y'}, {2, 'Y'}, {3, 'X'}, {4, 'Z'}, {5, 'Y'}}, J(0, -0.0625)},
         {{{1, 'X'}, {2, 'Y'}, {3, 'Y'}, {4, 'Z'}, {5, 'X'}}, J(0.0625, 0)},
         {{{1, 'X'}, {2, 'Y'}, {3, 'X'}, {4, 'Z'}, {5, 'X'}}, J(0, -0.0625)},
         {{{1, 'X'}, {2, 'Y'}, {3, 'Y'}, {4, 'Z'}, {5, 'Y'}}, J(0, -0.0625)},
         {{{1, 'X'}, {2, 'Y'}, {3, 'X'}, {4, 'Z'}, {5, 'Y'}}, J(-0.0625, 0)}});
}

TEST(OperatorsTransforms, JWProperties) {
    const uint32_t a = 7;  // fixed stand-in for the Python test's random index
    // a^ → (X_a - iY_a)/2 with Z string; a=0 has no Z string.
    QubitOperator x0 = QubitOperator::from_term_string("X0", 0.5);
    QubitOperator y0 = QubitOperator::from_term_string("Y0", Complex(0, 0.5));
    EXPECT_TRUE(jordan_wigner(fop("0^")).equals(x0 - y0));
    EXPECT_TRUE(jordan_wigner(fop("0")).equals(x0 + y0));

    // a^ a^ and a a vanish (all coefficients cancel to zero exactly).
    EXPECT_TRUE(jordan_wigner(fop("7^ 7^")).equals(QubitOperator()));
    EXPECT_TRUE(jordan_wigner(fop("7 7")).equals(QubitOperator()));

    // a^ a = (X-iY)/2 · (X+iY)/2 on the same mode.
    QubitOperator xa = QubitOperator::from_factors(F{{a, 'X'}}, 0.5);
    QubitOperator ya = QubitOperator::from_factors(F{{a, 'Y'}}, Complex(0, 0.5));
    EXPECT_TRUE(jordan_wigner(fop("7^ 7")).equals((xa - ya) * (xa + ya)));
}

// ── Bravyi-Kitaev ─────────────────────────────────────────────────────────

TEST(OperatorsTransforms, BKEncodeOneElectron) {
    expect_golden(bravyi_kitaev(fop("2^ 1")),
                  {{{{0, 'Z'}, {1, 'Y'}, {2, 'X'}}, J(0, 0.25)},
                   {{{1, 'X'}, {2, 'X'}}, J(0.25, 0)},
                   {{{0, 'Z'}, {1, 'Y'}, {2, 'Y'}}, J(0.25, 0)},
                   {{{1, 'X'}, {2, 'Y'}}, J(0, -0.25)}});

    expect_golden(bravyi_kitaev(fop("1^ 5")),
                  {{{{0, 'Z'}, {1, 'X'}, {3, 'Y'}, {4, 'Z'}, {5, 'X'}}, J(0, -0.25)},
                   {{{0, 'Z'}, {1, 'X'}, {3, 'Y'}, {5, 'Y'}}, J(0.25, 0)},
                   {{{1, 'Y'}, {3, 'Y'}, {4, 'Z'}, {5, 'X'}}, J(-0.25, 0)},
                   {{{1, 'Y'}, {3, 'Y'}, {5, 'Y'}}, J(0, -0.25)}});
}

TEST(OperatorsTransforms, BKEncodeTwoElectron) {
    expect_golden(
        bravyi_kitaev(fop("2^ 1 5^ 3")),
        {{{{0, 'Z'}, {1, 'X'}, {2, 'Y'}, {3, 'Y'}, {4, 'Z'}, {5, 'X'}}, J(-0.0625, 0)},
         {{{0, 'Z'}, {1, 'Y'}, {2, 'X'}, {3, 'X'}, {4, 'Z'}, {5, 'X'}}, J(0, 0.0625)},
         {{{0, 'Z'}, {1, 'X'}, {2, 'Y'}, {3, 'Y'}, {5, 'Y'}}, J(0, 0.0625)},
         {{{0, 'Z'}, {1, 'Y'}, {2, 'X'}, {3, 'X'}, {5, 'Y'}}, J(0.0625, 0)},
         {{{1, 'Y'}, {2, 'Y'}, {3, 'Y'}, {4, 'Z'}, {5, 'X'}}, J(0, -0.0625)},
         {{{1, 'X'}, {2, 'X'}, {3, 'X'}, {4, 'Z'}, {5, 'X'}}, J(0.0625, 0)},
         {{{1, 'Y'}, {2, 'Y'}, {3, 'Y'}, {5, 'Y'}}, J(-0.0625, 0)},
         {{{1, 'X'}, {2, 'X'}, {3, 'X'}, {5, 'Y'}}, J(0, -0.0625)},
         {{{0, 'Z'}, {1, 'X'}, {2, 'X'}, {3, 'Y'}, {4, 'Z'}, {5, 'X'}}, J(0, -0.0625)},
         {{{0, 'Z'}, {1, 'Y'}, {2, 'Y'}, {3, 'X'}, {4, 'Z'}, {5, 'X'}}, J(0.0625, 0)},
         {{{0, 'Z'}, {1, 'X'}, {2, 'X'}, {3, 'Y'}, {5, 'Y'}}, J(-0.0625, 0)},
         {{{0, 'Z'}, {1, 'Y'}, {2, 'Y'}, {3, 'X'}, {5, 'Y'}}, J(0, -0.0625)},
         {{{1, 'Y'}, {2, 'X'}, {3, 'Y'}, {4, 'Z'}, {5, 'X'}}, J(0.0625, 0)},
         {{{1, 'X'}, {2, 'Y'}, {3, 'X'}, {4, 'Z'}, {5, 'X'}}, J(0, -0.0625)},
         {{{1, 'Y'}, {2, 'X'}, {3, 'Y'}, {5, 'Y'}}, J(0, -0.0625)},
         {{{1, 'X'}, {2, 'Y'}, {3, 'X'}, {5, 'Y'}}, J(-0.0625, 0)}});
}

TEST(OperatorsTransforms, BKEncodeSecondTwoElectron) {
    expect_golden(bravyi_kitaev(fop("2^ 0 3^ 1")),
                  {{{{0, 'Y'}, {2, 'Y'}}, J(0.0625, 0)},
                   {{{0, 'X'}, {1, 'Z'}, {2, 'Y'}}, J(0, -0.0625)},
                   {{{0, 'Y'}, {1, 'Z'}, {2, 'X'}, {3, 'Z'}}, J(0, 0.0625)},
                   {{{0, 'X'}, {2, 'X'}, {3, 'Z'}}, J(0.0625, 0)},
                   {{{0, 'X'}, {2, 'Y'}}, J(0, -0.0625)},
                   {{{0, 'Y'}, {1, 'Z'}, {2, 'Y'}}, J(0.0625, 0)},
                   {{{0, 'X'}, {1, 'Z'}, {2, 'X'}, {3, 'Z'}}, J(0.0625, 0)},
                   {{{0, 'Y'}, {2, 'X'}, {3, 'Z'}}, J(0, 0.0625)},
                   {{{0, 'Y'}, {2, 'X'}}, J(0, 0.0625)},
                   {{{0, 'X'}, {1, 'Z'}, {2, 'X'}}, J(0.0625, 0)},
                   {{{0, 'Y'}, {1, 'Z'}, {2, 'Y'}, {3, 'Z'}}, J(0.0625, 0)},
                   {{{0, 'X'}, {2, 'Y'}, {3, 'Z'}}, J(0, -0.0625)},
                   {{{0, 'X'}, {2, 'X'}}, J(0.0625, 0)},
                   {{{0, 'Y'}, {1, 'Z'}, {2, 'X'}}, J(0, 0.0625)},
                   {{{0, 'X'}, {1, 'Z'}, {2, 'Y'}, {3, 'Z'}}, J(0, -0.0625)},
                   {{{0, 'Y'}, {2, 'Y'}, {3, 'Z'}}, J(0.0625, 0)}});
}

TEST(OperatorsTransforms, BKProperties) {
    QubitOperator x0 = QubitOperator::from_term_string("X0", 0.5);
    QubitOperator y0 = QubitOperator::from_term_string("Y0", Complex(0, 0.5));
    EXPECT_TRUE(bravyi_kitaev(fop("0^")).equals(x0 - y0));
    EXPECT_TRUE(bravyi_kitaev(fop("0")).equals(x0 + y0));
    EXPECT_TRUE(bravyi_kitaev(fop("6^ 6^")).equals(QubitOperator()));
    EXPECT_TRUE(bravyi_kitaev(fop("6 6")).equals(QubitOperator()));

    // Even mode index: a^ a → I/2 − Z_a/2 (test_BK_properties tail).
    const QubitOperator expected =
        QubitOperator::identity(0.5) -
        QubitOperator::from_term_string("Z4", 0.5);
    EXPECT_TRUE(bravyi_kitaev(fop("4^ 4")).equals(expected));
}

TEST(OperatorsTransforms, BKWideningAndValidation) {
    // n_qubits may widen the register (changes update sets) …
    const QubitOperator narrow = bravyi_kitaev(fop("0^ 0"));
    const QubitOperator wide = bravyi_kitaev(fop("0^ 0"), 4);
    EXPECT_TRUE(narrow.equals(wide));  // number operator is update-set free

    // … but must not shrink it below count_qubits.
    EXPECT_THROW((void)bravyi_kitaev(fop("2^ 1"), 2), std::invalid_argument);
}

// ── Parity ────────────────────────────────────────────────────────────────

TEST(OperatorsTransforms, ParityEncodeOneElectron) {
    expect_golden(parity_transform(fop("2^ 1"), 3),
                  {{{{0, 'Z'}, {1, 'Y'}}, J(0, 0.25)},
                   {{{1, 'X'}}, J(0.25, 0)},
                   {{{0, 'Z'}, {1, 'X'}, {2, 'Z'}}, J(-0.25, 0)},
                   {{{1, 'Y'}, {2, 'Z'}}, J(0, -0.25)}});

    expect_golden(
        parity_transform(fop("1^ 5"), 6),
        {{{{0, 'Z'}, {1, 'X'}, {2, 'X'}, {3, 'X'}, {4, 'Y'}}, J(0, -0.25)},
         {{{0, 'Z'}, {1, 'X'}, {2, 'X'}, {3, 'X'}, {4, 'X'}, {5, 'Z'}}, J(-0.25, 0)},
         {{{1, 'Y'}, {2, 'X'}, {3, 'X'}, {4, 'Y'}}, J(-0.25, 0)},
         {{{1, 'Y'}, {2, 'X'}, {3, 'X'}, {4, 'X'}, {5, 'Z'}}, J(0, 0.25)}});
}

TEST(OperatorsTransforms, ParityEncodeTwoElectron) {
    expect_golden(
        parity_transform(fop("2^ 1 5^ 3"), 6),
        {{{{0, 'Z'}, {1, 'Y'}, {2, 'Z'}, {3, 'X'}, {4, 'Y'}}, J(-0.0625, 0)},
         {{{0, 'Z'}, {1, 'Y'}, {3, 'Y'}, {4, 'Y'}}, J(0, -0.0625)},
         {{{0, 'Z'}, {1, 'Y'}, {2, 'Z'}, {3, 'X'}, {4, 'X'}, {5, 'Z'}}, J(0, -0.0625)},
         {{{0, 'Z'}, {1, 'Y'}, {3, 'Y'}, {4, 'X'}, {5, 'Z'}}, J(0.0625, 0)},
         {{{1, 'X'}, {2, 'Z'}, {3, 'X'}, {4, 'Y'}}, J(0, 0.0625)},
         {{{1, 'X'}, {3, 'Y'}, {4, 'Y'}}, J(-0.0625, 0)},
         {{{1, 'X'}, {2, 'Z'}, {3, 'X'}, {4, 'X'}, {5, 'Z'}}, J(-0.0625, 0)},
         {{{1, 'X'}, {3, 'Y'}, {4, 'X'}, {5, 'Z'}}, J(0, -0.0625)},
         {{{0, 'Z'}, {1, 'X'}, {3, 'X'}, {4, 'Y'}}, J(0, -0.0625)},
         {{{0, 'Z'}, {1, 'X'}, {2, 'Z'}, {3, 'Y'}, {4, 'Y'}}, J(0.0625, 0)},
         {{{0, 'Z'}, {1, 'X'}, {3, 'X'}, {4, 'X'}, {5, 'Z'}}, J(0.0625, 0)},
         {{{0, 'Z'}, {1, 'X'}, {2, 'Z'}, {3, 'Y'}, {4, 'X'}, {5, 'Z'}}, J(0, 0.0625)},
         {{{1, 'Y'}, {3, 'X'}, {4, 'Y'}}, J(0.0625, 0)},
         {{{1, 'Y'}, {2, 'Z'}, {3, 'Y'}, {4, 'Y'}}, J(0, 0.0625)},
         {{{1, 'Y'}, {3, 'X'}, {4, 'X'}, {5, 'Z'}}, J(0, 0.0625)},
         {{{1, 'Y'}, {2, 'Z'}, {3, 'Y'}, {4, 'X'}, {5, 'Z'}}, J(-0.0625, 0)}});
}

TEST(OperatorsTransforms, ParityEncodeSecondTwoElectron) {
    expect_golden(parity_transform(fop("2^ 0 3^ 1"), 6),
                  {{{{0, 'Y'}, {1, 'Z'}, {2, 'Y'}}, J(0.0625, 0)},
                   {{{0, 'X'}, {2, 'Y'}}, J(0, -0.0625)},
                   {{{0, 'Y'}, {1, 'Z'}, {2, 'X'}, {3, 'Z'}}, J(0, 0.0625)},
                   {{{0, 'X'}, {2, 'X'}, {3, 'Z'}}, J(0.0625, 0)},
                   {{{0, 'X'}, {1, 'Z'}, {2, 'Y'}}, J(0, -0.0625)},
                   {{{0, 'Y'}, {2, 'Y'}}, J(0.0625, 0)},
                   {{{0, 'X'}, {1, 'Z'}, {2, 'X'}, {3, 'Z'}}, J(0.0625, 0)},
                   {{{0, 'Y'}, {2, 'X'}, {3, 'Z'}}, J(0, 0.0625)},
                   {{{0, 'Y'}, {2, 'X'}}, J(0, 0.0625)},
                   {{{0, 'X'}, {1, 'Z'}, {2, 'X'}}, J(0.0625, 0)},
                   {{{0, 'Y'}, {2, 'Y'}, {3, 'Z'}}, J(0.0625, 0)},
                   {{{0, 'X'}, {1, 'Z'}, {2, 'Y'}, {3, 'Z'}}, J(0, -0.0625)},
                   {{{0, 'X'}, {2, 'X'}}, J(0.0625, 0)},
                   {{{0, 'Y'}, {1, 'Z'}, {2, 'X'}}, J(0, 0.0625)},
                   {{{0, 'X'}, {2, 'Y'}, {3, 'Z'}}, J(0, -0.0625)},
                   {{{0, 'Y'}, {1, 'Z'}, {2, 'Y'}, {3, 'Z'}}, J(0.0625, 0)}});
}

TEST(OperatorsTransforms, ParityValidation) {
    try {
        (void)parity_transform(fop("2^ 1"), 2);
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_STREQ(e.what(),
                     "Orbital number is larger than number of qubits in parity.encode()");
    }
}

// ── Coefficient / zero-operator propagation ───────────────────────────────

TEST(OperatorsTransforms, CoefficientsPropagate) {
    // Transform of c·op == c·(transform of op).
    const Complex c(0.75, -0.5);
    EXPECT_TRUE(jordan_wigner(fop("2^ 1", c)).equals(jordan_wigner(fop("2^ 1")) * c));
    EXPECT_TRUE(bravyi_kitaev(fop("2^ 1", c)).equals(bravyi_kitaev(fop("2^ 1")) * c));
    EXPECT_TRUE(parity_transform(fop("2^ 1", c), 3)
                    .equals(parity_transform(fop("2^ 1"), 3) * c));

    // Multi-term operators accumulate in term order.
    const FermionOperator combo = fop("2^ 1", 0.5) + fop("0^", -1.0);
    EXPECT_TRUE(jordan_wigner(combo).equals(jordan_wigner(fop("2^ 1", 0.5)) +
                                            jordan_wigner(fop("0^", -1.0))));

    EXPECT_TRUE(jordan_wigner(FermionOperator()).equals(QubitOperator()));
    EXPECT_TRUE(bravyi_kitaev(FermionOperator()).equals(QubitOperator()));
}

}  // namespace qarpx::ops::test
