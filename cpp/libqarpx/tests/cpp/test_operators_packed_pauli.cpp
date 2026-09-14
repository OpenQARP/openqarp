// Tests for the packed binary-symplectic Pauli key: the single-qubit product
// table (openfermion's _PAULI_OPERATOR_PRODUCTS), random multi-qubit products
// and commutators cross-checked against dense matrices, canonical form,
// hashing and conversions.

#include <gtest/gtest.h>

#include "qarpx/operators/packed_pauli.h"

#include <complex>
#include <random>
#include <vector>

namespace qarpx::ops::test {

namespace {

using Complex = std::complex<double>;

// Minimal dense complex matrices (row-major) — enough to validate the
// symplectic phase law without pulling in a linear algebra dependency.
struct Mat {
    size_t d = 1;
    std::vector<Complex> v{Complex(1.0, 0.0)};  // 1x1 identity scalar
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

Mat matmul(const Mat& a, const Mat& b) {
    Mat out;
    out.d = a.d;
    out.v.assign(out.d * out.d, Complex(0.0, 0.0));
    for (size_t r = 0; r < a.d; ++r)
        for (size_t k = 0; k < a.d; ++k) {
            const Complex av = a.v[r * a.d + k];
            if (av == Complex(0.0, 0.0)) continue;
            for (size_t c = 0; c < a.d; ++c) out.v[r * a.d + c] += av * b.v[k * a.d + c];
        }
    return out;
}

// Dense matrix of a packed Pauli over n qubits, qubit 0 as the first
// (most significant) kron factor.
Mat dense_of(const PackedPauli& p, int n_qubits) {
    Mat out;
    for (int q = 0; q < n_qubits; ++q) out = kron(out, pauli_matrix(p.pauli_at(static_cast<uint32_t>(q))));
    return out;
}

bool approx_equal(const Mat& a, const Mat& b, double tol = 1e-12) {
    if (a.d != b.d) return false;
    for (size_t i = 0; i < a.v.size(); ++i)
        if (std::abs(a.v[i] - b.v[i]) > tol) return false;
    return true;
}

Mat scale(Mat m, Complex s) {
    for (auto& x : m.v) x *= s;
    return m;
}

Complex i_pow(int k) {
    switch (((k % 4) + 4) % 4) {
        case 0: return {1, 0};
        case 1: return {0, 1};
        case 2: return {-1, 0};
        default: return {0, -1};
    }
}

PackedPauli random_packed(std::mt19937& rng, int n_qubits) {
    std::uniform_int_distribution<int> dist(0, 3);
    PackedPauli p;
    for (int q = 0; q < n_qubits; ++q) {
        const auto pauli = static_cast<Pauli>(dist(rng));
        if (pauli != Pauli::I) p.set_pauli(static_cast<uint32_t>(q), pauli);
    }
    p.canonicalize();
    return p;
}

}  // namespace

// ── Single-qubit product table (openfermion _PAULI_OPERATOR_PRODUCTS) ─────

TEST(OperatorsPackedPauli, SingleQubitProductTable) {
    struct Row {
        Pauli a, b, prod;
        int phase;
    };
    // i^phase · prod == a·b
    const Row table[] = {
        {Pauli::I, Pauli::I, Pauli::I, 0}, {Pauli::I, Pauli::X, Pauli::X, 0},
        {Pauli::I, Pauli::Y, Pauli::Y, 0}, {Pauli::I, Pauli::Z, Pauli::Z, 0},
        {Pauli::X, Pauli::I, Pauli::X, 0}, {Pauli::X, Pauli::X, Pauli::I, 0},
        {Pauli::X, Pauli::Y, Pauli::Z, 1}, {Pauli::X, Pauli::Z, Pauli::Y, 3},
        {Pauli::Y, Pauli::I, Pauli::Y, 0}, {Pauli::Y, Pauli::X, Pauli::Z, 3},
        {Pauli::Y, Pauli::Y, Pauli::I, 0}, {Pauli::Y, Pauli::Z, Pauli::X, 1},
        {Pauli::Z, Pauli::I, Pauli::Z, 0}, {Pauli::Z, Pauli::X, Pauli::Y, 1},
        {Pauli::Z, Pauli::Y, Pauli::X, 3}, {Pauli::Z, Pauli::Z, Pauli::I, 0},
    };
    for (const Row& row : table) {
        const auto [prod, phase] = single_pauli_product(row.a, row.b);
        EXPECT_EQ(prod, row.prod) << pauli_letter(row.a) << "·" << pauli_letter(row.b);
        EXPECT_EQ(phase, row.phase) << pauli_letter(row.a) << "·" << pauli_letter(row.b);
    }
}

// ── Random products & commutation vs dense matrices ───────────────────────

TEST(OperatorsPackedPauli, RandomProductsMatchDenseMatrices) {
    std::mt19937 rng(42);
    for (int n = 1; n <= 6; ++n) {
        for (int trial = 0; trial < 50; ++trial) {
            const PackedPauli a = random_packed(rng, n);
            const PackedPauli b = random_packed(rng, n);
            const int k = phase_exponent_mod4(a, b);
            const PackedPauli c = pauli_xor(a, b);

            const Mat lhs = matmul(dense_of(a, n), dense_of(b, n));
            const Mat rhs = scale(dense_of(c, n), i_pow(k));
            EXPECT_TRUE(approx_equal(lhs, rhs))
                << "n=" << n << " trial=" << trial << " k=" << k;
        }
    }
}

TEST(OperatorsPackedPauli, CommutesMatchesCorePredicate) {
    std::mt19937 rng(7);
    for (int n = 1; n <= 8; ++n) {
        for (int trial = 0; trial < 50; ++trial) {
            const PackedPauli a = random_packed(rng, n);
            const PackedPauli b = random_packed(rng, n);
            EXPECT_EQ(a.commutes_with(b),
                      qarpx::commutes(packed_to_dense(a, n), packed_to_dense(b, n)));
        }
    }
}

// ── Canonical form, equality, hashing ─────────────────────────────────────

TEST(OperatorsPackedPauli, CanonicalFormStripsTrailingIdentity) {
    PackedPauli p;
    p.set_pauli(70, Pauli::X);  // second block
    EXPECT_EQ(p.n_blocks(), 2);
    p.set_pauli(70, Pauli::I);
    p.canonicalize();
    EXPECT_TRUE(p.is_identity());
    EXPECT_EQ(p, PackedPauli{});
    EXPECT_EQ(p.hash(), PackedPauli{}.hash());
}

TEST(OperatorsPackedPauli, EqualityIndependentOfConstructionWidth) {
    PackedPauli a;
    a.set_pauli(3, Pauli::Z);
    a.canonicalize();

    PackedPauli b;
    b.set_pauli(3, Pauli::Z);
    b.set_pauli(100, Pauli::Y);  // widen, then clear
    b.set_pauli(100, Pauli::I);
    b.canonicalize();

    EXPECT_EQ(a, b);
    EXPECT_EQ(a.hash(), b.hash());
}

TEST(OperatorsPackedPauli, MaxQubitAndWeight) {
    EXPECT_EQ(PackedPauli{}.max_qubit(), -1);
    EXPECT_EQ(PackedPauli{}.weight(), 0);

    PackedPauli p;
    p.set_pauli(0, Pauli::X);
    p.set_pauli(65, Pauli::Y);
    p.set_pauli(2, Pauli::Z);
    p.canonicalize();
    EXPECT_EQ(p.max_qubit(), 65);
    EXPECT_EQ(p.weight(), 3);
    EXPECT_EQ(p.pauli_at(65), Pauli::Y);
    EXPECT_EQ(p.pauli_at(64), Pauli::I);
}

TEST(OperatorsPackedPauli, ToIndexPairsAscending) {
    PackedPauli p;
    p.set_pauli(5, Pauli::Y);
    p.set_pauli(1, Pauli::X);
    p.set_pauli(70, Pauli::Z);
    p.canonicalize();

    std::vector<std::pair<uint32_t, char>> pairs;
    packed_to_index_pairs(p, pairs);
    const std::vector<std::pair<uint32_t, char>> expected = {
        {1, 'X'}, {5, 'Y'}, {70, 'Z'}};
    EXPECT_EQ(pairs, expected);
}

TEST(OperatorsPackedPauli, DenseRoundTrip) {
    const PauliString dense = parse_pauli_string("IXYZIZ");
    const PackedPauli packed = packed_from_dense(dense);
    EXPECT_EQ(packed_to_dense(packed, 6), dense);
    EXPECT_EQ(packed.weight(), 4);
    EXPECT_EQ(packed.max_qubit(), 5);
}

}  // namespace qarpx::ops::test
