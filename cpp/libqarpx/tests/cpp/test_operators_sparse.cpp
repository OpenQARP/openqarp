// Tests for the COO sparse kernel: LSB default (qubit q ↔ bit q, §1), the
// explicit MSB openfermion layout, agreement with dense kron construction,
// per-flip-mask accumulation (no duplicate triplets), and n_qubits validation.

#include <gtest/gtest.h>

#include "qarpx/operators/sparse.h"

#include <complex>
#include <random>
#include <vector>

namespace qarpx::ops::test {

namespace {

using Complex = std::complex<double>;

// Dense row-major matrix helpers (same shape as test_operators_packed_pauli).
struct Mat {
    size_t d = 1;
    std::vector<Complex> v{Complex(1.0, 0.0)};
};

Mat pauli_matrix(Pauli p) {
    const Complex i(0.0, 1.0);
    switch (p) {
        case Pauli::I: return {2, {1, 0, 0, 1}};
        case Pauli::X: return {2, {0, 1, 1, 0}};
        case Pauli::Y: return {2, {0, -i, i, 0}};
        default: return {2, {1, 0, 0, -1}};
    }
}

Mat kron(const Mat& a, const Mat& b) {
    Mat out;
    out.d = a.d * b.d;
    out.v.assign(out.d * out.d, Complex(0.0, 0.0));
    for (size_t ar = 0; ar < a.d; ++ar)
        for (size_t ac = 0; ac < a.d; ++ac)
            for (size_t br = 0; br < b.d; ++br)
                for (size_t bc = 0; bc < b.d; ++bc)
                    out.v[(ar * b.d + br) * out.d + (ac * b.d + bc)] =
                        a.v[ar * a.d + ac] * b.v[br * b.d + bc];
    return out;
}

// Dense matrix of an operator by explicit kron: kLsb puts qubit 0 as the
// LAST factor (P_{n-1} ⊗ … ⊗ P_0), kMsb as the FIRST (openfermion).
Mat dense_of(const QubitOperator& op, int n_qubits, BitOrder order = BitOrder::kLsb) {
    Mat out;
    out.d = size_t{1} << n_qubits;
    out.v.assign(out.d * out.d, Complex(0.0, 0.0));
    for (const auto& e : op.numeric().terms) {
        Mat term;
        for (int k = 0; k < n_qubits; ++k) {
            const int q = order == BitOrder::kLsb ? n_qubits - 1 - k : k;
            term = kron(term, pauli_matrix(e.key.pauli_at(static_cast<uint32_t>(q))));
        }
        for (size_t i = 0; i < term.v.size(); ++i) out.v[i] += e.coeff * term.v[i];
    }
    return out;
}

// Assemble COO triplets into a dense matrix.  Summing is what scipy's CSC
// conversion does; the kernel emits no duplicates, which
// AccumulatesPerFlipMask pins separately.
Mat dense_of(const SparseCoo& coo) {
    Mat out;
    out.d = static_cast<size_t>(coo.dim);
    out.v.assign(out.d * out.d, Complex(0.0, 0.0));
    for (size_t i = 0; i < coo.data.size(); ++i)
        out.v[static_cast<size_t>(coo.rows[i]) * out.d +
              static_cast<size_t>(coo.cols[i])] += coo.data[i];
    return out;
}

void expect_close(const Mat& a, const Mat& b, double tol = 1e-12) {
    ASSERT_EQ(a.d, b.d);
    for (size_t i = 0; i < a.v.size(); ++i)
        EXPECT_LT(std::abs(a.v[i] - b.v[i]), tol) << "flat index " << i;
}

QubitOperator qop(std::string_view s, Complex c = 1.0) {
    return QubitOperator::from_term_string(s, c);
}

}  // namespace

TEST(OperatorsSparse, DefaultOrderingZ0IsLsb) {
    // Z0 with qubit 0 least significant: diag(1, -1, 1, -1) at n=2.
    const SparseCoo coo = qubit_operator_coo(qop("Z0"), 2);
    const Mat m = dense_of(coo);
    const std::vector<Complex> expected = {1, 0, 0, 0, 0, -1, 0, 0,
                                           0, 0, 1, 0, 0, 0, 0, -1};
    ASSERT_EQ(m.v.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) EXPECT_EQ(m.v[i], expected[i]);
}

TEST(OperatorsSparse, DefaultOrderingX0FlipsLowBit) {
    // X0 at n=2: kLsb (default) flips the low bit (columns 0↔1), kMsb the
    // high bit (0↔2) — an asymmetric pin that tells the two conventions apart.
    const Mat m = dense_of(qubit_operator_coo(qop("X0"), 2));
    EXPECT_EQ(m.v[0 * 4 + 1], Complex(1, 0));
    EXPECT_EQ(m.v[1 * 4 + 0], Complex(1, 0));
    EXPECT_EQ(m.v[2 * 4 + 3], Complex(1, 0));
    EXPECT_EQ(m.v[3 * 4 + 2], Complex(1, 0));
    EXPECT_EQ(m.v[0 * 4 + 2], Complex(0, 0));
}

TEST(OperatorsSparse, MsbOrderingZ0) {
    // openfermion layout, qubit 0 most significant: diag(1, 1, -1, -1).
    const Mat m = dense_of(qubit_operator_coo(qop("Z0"), 2, BitOrder::kMsb));
    const std::vector<Complex> expected = {1, 0, 0, 0, 0, 1, 0, 0,
                                           0, 0, -1, 0, 0, 0, 0, -1};
    ASSERT_EQ(m.v.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) EXPECT_EQ(m.v[i], expected[i]);
}

TEST(OperatorsSparse, MsbOrderingX0FlipsHighBit) {
    // |00⟩ ↔ |10⟩ i.e. columns 0↔2 and 1↔3.
    const Mat m = dense_of(qubit_operator_coo(qop("X0"), 2, BitOrder::kMsb));
    EXPECT_EQ(m.v[2 * 4 + 0], Complex(1, 0));
    EXPECT_EQ(m.v[0 * 4 + 2], Complex(1, 0));
    EXPECT_EQ(m.v[3 * 4 + 1], Complex(1, 0));
    EXPECT_EQ(m.v[0 * 4 + 0], Complex(0, 0));
    EXPECT_EQ(m.v[0 * 4 + 1], Complex(0, 0));
}

TEST(OperatorsSparse, LsbIsBitReversedMsb) {
    const QubitOperator op = qop("X0 Z1", 0.7) + qop("Y2", Complex(0.0, 1.5)) +
                             qop("Z0 Y1 X2", -0.3);
    const int n = 3;
    const Mat msb = dense_of(qubit_operator_coo(op, n, BitOrder::kMsb));
    const Mat lsb = dense_of(qubit_operator_coo(op, n));
    const auto rev = [n](size_t i) {
        size_t r = 0;
        for (int b = 0; b < n; ++b) r |= ((i >> b) & 1) << (n - 1 - b);
        return r;
    };
    for (size_t r = 0; r < msb.d; ++r)
        for (size_t c = 0; c < msb.d; ++c)
            EXPECT_EQ(lsb.v[r * lsb.d + c], msb.v[rev(r) * msb.d + rev(c)]);
}

TEST(OperatorsSparse, RandomOperatorsMatchDense) {
    std::mt19937 rng(11);
    std::uniform_int_distribution<int> pauli_dist(0, 3);
    std::uniform_real_distribution<double> coeff_dist(-1.0, 1.0);
    const char letters[4] = {'I', 'X', 'Y', 'Z'};

    for (int n = 1; n <= 5; ++n) {
        for (int trial = 0; trial < 10; ++trial) {
            QubitOperator op;
            for (int t = 0; t < 6; ++t) {
                std::string s;
                for (int q = 0; q < n; ++q) {
                    const char letter = letters[pauli_dist(rng)];
                    if (letter != 'I') {
                        if (!s.empty()) s += ' ';
                        s += letter;
                        s += std::to_string(q);
                    }
                }
                op += qop(s, Complex(coeff_dist(rng), coeff_dist(rng)));
            }
            expect_close(dense_of(qubit_operator_coo(op, n)), dense_of(op, n));
            expect_close(dense_of(qubit_operator_coo(op, n, BitOrder::kMsb)),
                         dense_of(op, n, BitOrder::kMsb));
        }
    }
}

TEST(OperatorsSparse, AccumulatesPerFlipMask) {
    // Terms sharing a flip mask land on the same cells and are summed in
    // the kernel: one triplet per non-zero, never one per (term, col).
    const QubitOperator op = qop("Z0", 1.0) + qop("", 0.5);
    const SparseCoo coo = qubit_operator_coo(op, 1);
    EXPECT_EQ(coo.data.size(), 2u);  // 1 mask × dim 2, was 2 terms × dim 2
    expect_close(dense_of(coo), dense_of(op, 1));

    // Z0 + Z1 at n=2: diag(2, 0, 0, −2) — the exact cancellations are
    // dropped, so the two flip-mask-0 terms yield two triplets, not four.
    const SparseCoo diag = qubit_operator_coo(qop("Z0") + qop("Z1"), 2);
    EXPECT_EQ(diag.data.size(), 2u);
    expect_close(dense_of(diag), dense_of(qop("Z0") + qop("Z1"), 2));

    // Distinct flip masks stay distinct: X0 and X1 are 2 masks × dim 4.
    const SparseCoo two = qubit_operator_coo(qop("X0") + qop("X1"), 2);
    EXPECT_EQ(two.data.size(), 8u);
    for (size_t i = 0; i + 1 < two.data.size(); ++i)
        for (size_t j = i + 1; j < two.data.size(); ++j)
            EXPECT_FALSE(two.rows[i] == two.rows[j] && two.cols[i] == two.cols[j]);
}

TEST(OperatorsSparse, FermionOperatorViaJordanWigner) {
    // a†_0 at n=1 in the occupation basis: |1⟩⟨0| (row 1, col 0).
    const Mat creation =
        dense_of(fermion_operator_coo(FermionOperator::from_term_string("0^"), -1));
    ASSERT_EQ(creation.d, 2u);
    EXPECT_EQ(creation.v[1 * 2 + 0], Complex(1, 0));
    EXPECT_EQ(creation.v[0 * 2 + 1], Complex(0, 0));

    // Number operator a†a on mode 1 at n=2: diag(0, 0, 1, 1) under the LSB
    // default (mode 1 ↔ bit 1); openfermion's MSB layout gives diag(0, 1, 0, 1).
    const Mat number = dense_of(
        fermion_operator_coo(FermionOperator::from_term_string("1^ 1"), -1));
    ASSERT_EQ(number.d, 4u);
    EXPECT_EQ(number.v[0 * 4 + 0], Complex(0, 0));
    EXPECT_EQ(number.v[1 * 4 + 1], Complex(0, 0));
    EXPECT_EQ(number.v[2 * 4 + 2], Complex(1, 0));
    EXPECT_EQ(number.v[3 * 4 + 3], Complex(1, 0));
    const Mat number_msb = dense_of(fermion_operator_coo(
        FermionOperator::from_term_string("1^ 1"), -1, BitOrder::kMsb));
    EXPECT_EQ(number_msb.v[1 * 4 + 1], Complex(1, 0));
    EXPECT_EQ(number_msb.v[2 * 4 + 2], Complex(0, 0));

    // The fermionic register width governs the default dimension even when
    // the JW image is narrower (e.g. the zero operator "3 3").
    EXPECT_EQ(fermion_operator_coo(FermionOperator::from_term_string("3 3"), -1).dim,
              16);
}

TEST(OperatorsSparse, WideningAndValidation) {
    const SparseCoo wide = qubit_operator_coo(qop("Z0"), 3);
    EXPECT_EQ(wide.dim, 8);
    expect_close(dense_of(wide), dense_of(qop("Z0"), 3));

    EXPECT_THROW((void)qubit_operator_coo(qop("Z3"), 2), std::invalid_argument);
    EXPECT_THROW((void)fermion_operator_coo(
                     FermionOperator::from_term_string("3^ 3"), 2),
                 std::invalid_argument);

    // Empty and identity-only operators are 1×1.
    EXPECT_EQ(qubit_operator_coo(QubitOperator(), -1).dim, 1);
    const SparseCoo ident = qubit_operator_coo(QubitOperator::identity(2.5), -1);
    EXPECT_EQ(ident.dim, 1);
    ASSERT_EQ(ident.data.size(), 1u);
    EXPECT_EQ(ident.data[0], Complex(2.5, 0));
}

}  // namespace qarpx::ops::test
