// Tests for qarpx::synthesis::detail::synthesise_diagonal_block_naive and
// synthesise_diagonal_block_graysynth.
//
// Both functions emit gates implementing
//
//     Π_i exp( -i · angle_eff_i / 2 · Z^{z_rows[i]} )
//
// where `angle_eff_i = signs[i] ? -angles[i] : angles[i]`.  Since all Z-only
// Paulis commute, the product equals a single exponential of the sum, which
// is diagonal in the computational basis — we build the reference unitary
// from that closed form and compare element-wise.

#include <gtest/gtest.h>

#include "qarpx/synthesis/pauli_exponential.h"

#include "gate_test_helpers.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace qarpx::test {

using qarpx::synthesis::detail::synthesise_diagonal_block_best;
using qarpx::synthesis::detail::synthesise_diagonal_block_graysynth;
using qarpx::synthesis::detail::synthesise_diagonal_block_naive;

namespace {

/// Build the reference diagonal unitary of
///     Π_i exp(-i · angle_eff_i/2 · Z^{z_rows[i]}).
/// Identity rows (all-zero z_rows) contribute a global phase exp(-i·angle/2).
Mat reference_diagonal_block(
    const std::vector<std::vector<bool>>& z_rows,
    const std::vector<bool>& signs,
    const std::vector<double>& angles,
    int n_qubits)
{
    const int dim = 1 << n_qubits;
    Mat U = Mat::Zero(dim, dim);
    for (int s = 0; s < dim; ++s) {
        double phase = 0.0;
        for (std::size_t i = 0; i < z_rows.size(); ++i) {
            const double a = signs[i] ? -angles[i] : angles[i];
            int parity = 0;
            for (std::size_t q = 0; q < z_rows[i].size(); ++q) {
                if (z_rows[i][q] && ((s >> q) & 1)) {
                    parity ^= 1;
                }
            }
            // Eigenvalue of Z^v on |s⟩ is (-1)^parity.
            phase += -0.5 * a * (parity ? -1.0 : 1.0);
        }
        U(s, s) = ei(phase);
    }
    return U;
}

/// Build the unitary produced by a synthesis function `fn` on the given
/// inputs, by running it through QarpSimulator after transpiling to the
/// native gate set.
template <typename Fn>
Mat synthesised_unitary(
    Fn fn,
    const std::vector<std::vector<bool>>& z_rows,
    const std::vector<bool>& signs,
    const std::vector<double>& angles,
    int n_qubits)
{
    SimpleBlock block(static_cast<uint32_t>(n_qubits), "dblock");
    std::vector<Param> param_angles;
    param_angles.reserve(angles.size());
    for (double a : angles) param_angles.emplace_back(a);
    fn(block, z_rows, signs, param_angles);
    block.build();
    return build_unitary(block.flatten(), n_qubits);
}

/// Count CX gates in the block's flattened command list (post-transpile).
int count_cx(const SimpleBlock& block) {
    int n = 0;
    for (const auto& cmd : block.commands()) {
        if (cmd.gate == GateType::CX) ++n;
    }
    return n;
}

}  // anonymous namespace

// ── Naive: structural tests ───────────────────────────────────────────────

TEST(DiagonalBlockNaive, EmptyInputEmitsNoCommands) {
    SimpleBlock b(2, "t");
    synthesise_diagonal_block_naive(b, {}, {}, {});
    EXPECT_EQ(b.commands().size(), 0u);
}

TEST(DiagonalBlockNaive, SingleZ_EmitsJustRz) {
    SimpleBlock b(3, "t");
    synthesise_diagonal_block_naive(
        b, {{true, false, false}}, {false}, {Param(0.7)});
    ASSERT_EQ(b.commands().size(), 1u);
    EXPECT_EQ(b.commands()[0].gate, GateType::Rz);
    EXPECT_EQ(b.commands()[0].qubits[0], 0u);
}

TEST(DiagonalBlockNaive, AllIdentityRow_EmitsGlobalPhase) {
    SimpleBlock b(3, "t");
    synthesise_diagonal_block_naive(
        b, {{false, false, false}}, {false}, {Param(0.7)});
    ASSERT_EQ(b.commands().size(), 1u);
    EXPECT_EQ(b.commands()[0].gate, GateType::GPhase);
}

TEST(DiagonalBlockNaive, MultiQubitZ_EmitsCxLadderPlusRzPlusReverseLadder) {
    // Z on qubits {0, 1, 2}: expect CX(0,2), CX(1,2), Rz(2), CX(1,2), CX(0,2).
    SimpleBlock b(3, "t");
    synthesise_diagonal_block_naive(
        b, {{true, true, true}}, {false}, {Param(0.5)});
    ASSERT_EQ(b.commands().size(), 5u);
    EXPECT_EQ(b.commands()[0].gate, GateType::CX);
    EXPECT_EQ(b.commands()[0].qubits[0], 0u);
    EXPECT_EQ(b.commands()[0].qubits[1], 2u);
    EXPECT_EQ(b.commands()[1].gate, GateType::CX);
    EXPECT_EQ(b.commands()[1].qubits[0], 1u);
    EXPECT_EQ(b.commands()[1].qubits[1], 2u);
    EXPECT_EQ(b.commands()[2].gate, GateType::Rz);
    EXPECT_EQ(b.commands()[2].qubits[0], 2u);
    EXPECT_EQ(b.commands()[3].gate, GateType::CX);
    EXPECT_EQ(b.commands()[3].qubits[0], 1u);
    EXPECT_EQ(b.commands()[3].qubits[1], 2u);
    EXPECT_EQ(b.commands()[4].gate, GateType::CX);
    EXPECT_EQ(b.commands()[4].qubits[0], 0u);
    EXPECT_EQ(b.commands()[4].qubits[1], 2u);
}

TEST(DiagonalBlockNaive, InputSizeMismatchThrows) {
    SimpleBlock b(2, "t");
    EXPECT_THROW(
        synthesise_diagonal_block_naive(
            b, {{true, false}}, {false, false}, {Param(0.1)}),
        std::invalid_argument);
    EXPECT_THROW(
        synthesise_diagonal_block_naive(
            b, {{true, false}, {false}}, {false, false},
            {Param(0.1), Param(0.2)}),
        std::invalid_argument);
}

// ── Naive: unitary equivalence ────────────────────────────────────────────

TEST(DiagonalBlockNaive_Unitary, SingleQubitZ) {
    const int n = 1;
    const std::vector<std::vector<bool>> z = {{true}};
    const std::vector<bool> signs = {false};
    const std::vector<double> angles = {0.5};
    Mat U = synthesised_unitary(synthesise_diagonal_block_naive, z, signs, angles, n);
    Mat U_ref = reference_diagonal_block(z, signs, angles, n);
    EXPECT_TRUE(expect_unitary_close(U, U_ref));
}

TEST(DiagonalBlockNaive_Unitary, TwoQubitZZ) {
    const int n = 2;
    const std::vector<std::vector<bool>> z = {{true, true}};
    const std::vector<bool> signs = {false};
    const std::vector<double> angles = {0.7};
    Mat U = synthesised_unitary(synthesise_diagonal_block_naive, z, signs, angles, n);
    Mat U_ref = reference_diagonal_block(z, signs, angles, n);
    EXPECT_TRUE(expect_unitary_close(U, U_ref));
}

TEST(DiagonalBlockNaive_Unitary, ThreeQubitParitySum) {
    // exp(-i·θ₁/2·Z₀Z₁) · exp(-i·θ₂/2·Z₁Z₂) · exp(-i·θ₃/2·Z₀Z₁Z₂)
    const int n = 3;
    const std::vector<std::vector<bool>> z = {
        {true,  true,  false},
        {false, true,  true },
        {true,  true,  true },
    };
    const std::vector<bool> signs = {false, false, false};
    const std::vector<double> angles = {0.7, -0.3, 1.234};
    Mat U = synthesised_unitary(synthesise_diagonal_block_naive, z, signs, angles, n);
    Mat U_ref = reference_diagonal_block(z, signs, angles, n);
    EXPECT_TRUE(expect_unitary_close(U, U_ref));
}

TEST(DiagonalBlockNaive_Unitary, SignFlipsNegateAngle) {
    // Test that signs[i] = true produces the same unitary as negated angles.
    const int n = 2;
    const std::vector<std::vector<bool>> z = {{true, false}, {false, true}};
    const std::vector<bool> signs_flipped = {true, false};
    const std::vector<bool> signs_clean   = {false, false};
    const std::vector<double> angles      = {0.5, 0.3};
    const std::vector<double> angles_neg  = {-0.5, 0.3};

    Mat U_flipped = synthesised_unitary(
        synthesise_diagonal_block_naive, z, signs_flipped, angles, n);
    Mat U_clean = synthesised_unitary(
        synthesise_diagonal_block_naive, z, signs_clean, angles_neg, n);

    EXPECT_TRUE(expect_unitary_close(U_flipped, U_clean));
}

TEST(DiagonalBlockNaive_Unitary, IdentityRowGivesGlobalPhase) {
    const int n = 2;
    const std::vector<std::vector<bool>> z = {{false, false}};
    const std::vector<bool> signs = {false};
    const std::vector<double> angles = {0.8};
    Mat U = synthesised_unitary(synthesise_diagonal_block_naive, z, signs, angles, n);
    Mat U_ref = reference_diagonal_block(z, signs, angles, n);
    EXPECT_TRUE(expect_unitary_close(U, U_ref));
    // It's a global phase exp(-i·0.8/2) = exp(-i·0.4), so all diagonal entries
    // should be equal and unit-modulus.
    EXPECT_NEAR(std::abs(U(0, 0)), 1.0, 1e-12);
    for (int s = 1; s < (1 << n); ++s) {
        EXPECT_LE(std::abs(U(s, s) - U(0, 0)), 1e-12);
    }
}

TEST(DiagonalBlockNaive_Unitary, MixedZAndIdentityRows) {
    const int n = 3;
    const std::vector<std::vector<bool>> z = {
        {false, false, false},     // identity → global phase
        {true,  false, false},     // single Z
        {false, true,  true },     // ZZ
        {true,  true,  true },     // ZZZ
    };
    const std::vector<bool> signs = {false, true, false, true};
    const std::vector<double> angles = {0.6, 0.3, -0.4, 1.1};
    Mat U = synthesised_unitary(synthesise_diagonal_block_naive, z, signs, angles, n);
    Mat U_ref = reference_diagonal_block(z, signs, angles, n);
    EXPECT_TRUE(expect_unitary_close(U, U_ref));
}

// ── Naive: CX-count anchor ────────────────────────────────────────────────

TEST(DiagonalBlockNaive, CxCountMatchesNaiveLadderFormula) {
    // For each row, the naive ladder emits 2·(hamming(v_i) − 1) CXs.
    // Verify on a structured input where the prediction is unambiguous.
    SimpleBlock b(4, "t");
    const std::vector<std::vector<bool>> z = {
        {true, true, false, false},          // ZZII: 2·(2−1) = 2 CX
        {true, true, true,  false},          // ZZZI: 2·(3−1) = 4 CX
        {true, true, true,  true },          // ZZZZ: 2·(4−1) = 6 CX
    };
    synthesise_diagonal_block_naive(
        b, z, {false, false, false},
        {Param(0.1), Param(0.2), Param(0.3)});
    EXPECT_EQ(count_cx(b), 2 + 4 + 6);
}

// ── Graysynth ─────────────────────────────────────────────────────────────
//
// The oracle throughout is `reference_diagonal_block`, built from the closed
// form.  It is doing double duty: graysynth reorders the rotations and leaves
// the register in a permuted basis mid-circuit, so a unitary that matches a
// *diagonal* reference is simultaneous proof that (a) every parity was
// realised on the right qubit and (b) the linear state was restored to the
// identity at the end.  A missed restoration would leave a non-diagonal
// unitary and fail here.

TEST(DiagonalBlockGraysynth, EmptyInputEmitsNoCommands) {
    SimpleBlock b(2, "t");
    synthesise_diagonal_block_graysynth(b, {}, {}, {});
    EXPECT_TRUE(b.commands().empty());
}

TEST(DiagonalBlockGraysynth, SingleZ_EmitsJustRz) {
    SimpleBlock b(2, "t");
    synthesise_diagonal_block_graysynth(b, {{true, false}}, {false}, {Param(0.3)});
    EXPECT_EQ(count_cx(b), 0);
    ASSERT_EQ(b.commands().size(), 1u);
    EXPECT_EQ(b.commands()[0].gate, GateType::Rz);
}

TEST(DiagonalBlockGraysynth, AllIdentityRow_EmitsGlobalPhase) {
    SimpleBlock b(2, "t");
    synthesise_diagonal_block_graysynth(b, {{false, false}}, {false}, {Param(0.4)});
    ASSERT_EQ(b.commands().size(), 1u);
    EXPECT_EQ(b.commands()[0].gate, GateType::GPhase);
}

TEST(DiagonalBlockGraysynth, InputSizeMismatchThrows) {
    SimpleBlock b(2, "t");
    EXPECT_THROW(
        synthesise_diagonal_block_graysynth(b, {{true, false}}, {}, {Param(0.1)}),
        std::invalid_argument);
}

// ── Graysynth: unitary equivalence ────────────────────────────────────────

namespace {

/// Assert graysynth reproduces the closed-form reference, and that it agrees
/// with the naive ladder gate-for-unitary.
void expect_graysynth_correct(
    const std::vector<std::vector<bool>>& z,
    const std::vector<bool>& signs,
    const std::vector<double>& angles,
    int n)
{
    const Mat expected = reference_diagonal_block(z, signs, angles, n);
    const Mat gray = synthesised_unitary(synthesise_diagonal_block_graysynth, z, signs, angles, n);
    const Mat naive = synthesised_unitary(synthesise_diagonal_block_naive, z, signs, angles, n);
    EXPECT_TRUE(expect_unitary_close(gray, expected)) << "graysynth != closed-form reference";
    EXPECT_TRUE(expect_unitary_close(gray, naive)) << "graysynth != naive ladder";
}

}  // namespace

TEST(DiagonalBlockGraysynth_Unitary, TwoQubitZZ) {
    expect_graysynth_correct({{true, true}}, {false}, {0.7}, 2);
}

TEST(DiagonalBlockGraysynth_Unitary, ThreeQubitParitySum) {
    expect_graysynth_correct(
        {{true, true, false}, {false, true, true}, {true, false, true}},
        {false, false, false}, {0.3, -0.8, 1.1}, 3);
}

TEST(DiagonalBlockGraysynth_Unitary, SignFlipsNegateAngle) {
    expect_graysynth_correct(
        {{true, true, false}, {false, true, true}},
        {true, false}, {0.45, 0.9}, 3);
}

TEST(DiagonalBlockGraysynth_Unitary, MixedZAndIdentityRows) {
    expect_graysynth_correct(
        {{true, true, false}, {false, false, false}, {true, true, true}},
        {false, true, false}, {0.25, 0.5, -0.35}, 3);
}

TEST(DiagonalBlockGraysynth_Unitary, RepeatedParitiesAreIndependent) {
    // The same parity twice must add its angles, not collapse to one.
    expect_graysynth_correct(
        {{true, true, false}, {true, true, false}},
        {false, false}, {0.3, 0.4}, 3);
}

TEST(DiagonalBlockGraysynth_Unitary, DenseUccLikeGroup) {
    // Shape of a UCCSD double-excitation generator after diagonalisation:
    // many parities, few qubits — where the sharing is supposed to pay off.
    expect_graysynth_correct(
        {
            {true, false, false, false}, {true, true, false, false},
            {true, true, true, false},   {true, true, true, true},
            {false, true, true, true},   {false, false, true, true},
            {false, false, false, true}, {true, false, true, false},
        },
        std::vector<bool>(8, false),
        {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, 4);
}

TEST(DiagonalBlockGraysynth_Unitary, ExhaustiveThreeQubitParitySets) {
    // Every non-empty subset of the 7 non-zero parities on 3 qubits.  This is
    // the test that would catch a basis-restoration bug on some odd shape.
    for (int mask = 1; mask < (1 << 7); ++mask) {
        std::vector<std::vector<bool>> z;
        std::vector<bool> signs;
        std::vector<double> angles;
        for (int p = 1; p <= 7; ++p) {
            if (!((mask >> (p - 1)) & 1)) continue;
            z.push_back({(p & 1) != 0, (p & 2) != 0, (p & 4) != 0});
            signs.push_back((p & 1) != 0);
            angles.push_back(0.1 * p + 0.05);
        }
        const Mat expected = reference_diagonal_block(z, signs, angles, 3);
        const Mat gray =
            synthesised_unitary(synthesise_diagonal_block_graysynth, z, signs, angles, 3);
        ASSERT_TRUE(expect_unitary_close(gray, expected)) << "failed for parity mask " << mask;
    }
}

TEST(DiagonalBlockGraysynth_Unitary, RandomisedAgainstNaive) {
    // Deterministic LCG — no <random> so the sequence is identical everywhere.
    uint64_t seed = 0x9E3779B97F4A7C15ull;
    auto next = [&seed]() {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<uint32_t>(seed >> 33);
    };

    for (int trial = 0; trial < 200; ++trial) {
        const int n = 2 + static_cast<int>(next() % 4);       // 2..5 qubits
        const int k = 1 + static_cast<int>(next() % 8);       // 1..8 parities
        std::vector<std::vector<bool>> z;
        std::vector<bool> signs;
        std::vector<double> angles;
        for (int i = 0; i < k; ++i) {
            std::vector<bool> row(static_cast<std::size_t>(n));
            for (int q = 0; q < n; ++q) row[static_cast<std::size_t>(q)] = (next() % 2) == 0;
            z.push_back(std::move(row));
            signs.push_back((next() % 2) == 0);
            angles.push_back(0.05 + 0.01 * static_cast<double>(next() % 100));
        }
        const Mat expected = reference_diagonal_block(z, signs, angles, n);
        const Mat gray =
            synthesised_unitary(synthesise_diagonal_block_graysynth, z, signs, angles, n);
        ASSERT_TRUE(expect_unitary_close(gray, expected))
            << "trial " << trial << " (n=" << n << ", k=" << k << ")";
    }
}

// ── Graysynth: gate count ─────────────────────────────────────────────────

// ── `best`: the min(naive, graysynth) selector ────────────────────────────

TEST(DiagonalBlockBest, NeverWorseThanEitherAlternative) {
    // Graysynth wins on dense groups but pays a Patel-Markov-Hayes tail the
    // naive ladder avoids by uncomputing, so neither dominates.  `best` must
    // be <= both on every input, which is what makes switching the default
    // safe for callers who were getting the naive ladder before.
    uint64_t seed = 0xD1B54A32D192ED03ull;
    auto next = [&seed]() {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<uint32_t>(seed >> 33);
    };

    int gray_strictly_better = 0, naive_strictly_better = 0;
    for (int trial = 0; trial < 600; ++trial) {
        // Sweep well into the regime where graysynth's Patel-Markov-Hayes tail
        // is least amortised: many qubits, very few parities.
        const int n = 2 + static_cast<int>(next() % 15);  // 2..16 qubits
        const int k = 1 + static_cast<int>(next() % 6);   // 1..6 parities
        std::vector<std::vector<bool>> z;
        std::vector<bool> signs;
        std::vector<Param> angles;
        for (int i = 0; i < k; ++i) {
            std::vector<bool> row(static_cast<std::size_t>(n));
            for (int q = 0; q < n; ++q) row[static_cast<std::size_t>(q)] = (next() % 2) == 0;
            z.push_back(std::move(row));
            signs.push_back((next() % 2) == 0);
            angles.emplace_back(0.05 + 0.01 * static_cast<double>(next() % 100));
        }

        SimpleBlock bn(static_cast<uint32_t>(n), "n");
        synthesise_diagonal_block_naive(bn, z, signs, angles);
        SimpleBlock bg(static_cast<uint32_t>(n), "g");
        synthesise_diagonal_block_graysynth(bg, z, signs, angles);
        SimpleBlock bb(static_cast<uint32_t>(n), "b");
        synthesise_diagonal_block_best(bb, z, signs, angles);

        const int cn = count_cx(bn), cg = count_cx(bg), cb = count_cx(bb);
        ASSERT_LE(cb, cn) << "trial " << trial << ": best worse than naive";
        ASSERT_LE(cb, cg) << "trial " << trial << ": best worse than graysynth";
        ASSERT_EQ(cb, std::min(cn, cg)) << "trial " << trial;
        if (cg < cn) ++gray_strictly_better;
        if (cn < cg) ++naive_strictly_better;
    }
    // Both directions must actually occur, or the selector is dead code and
    // one of the two algorithms could simply be deleted.  Neither dominates:
    // graysynth wins on dense groups, naive on few parities spread over many
    // qubits (where the Patel-Markov-Hayes tail is not amortised).
    RecordProperty("graysynth_strictly_better", gray_strictly_better);
    RecordProperty("naive_strictly_better", naive_strictly_better);
    EXPECT_GT(gray_strictly_better, 0) << "graysynth never won — selector pointless";
    EXPECT_GT(naive_strictly_better, 0) << "naive never won — could drop the selector";
}

TEST(DiagonalBlockBest, MatchesReferenceUnitary) {
    const std::vector<std::vector<bool>> z = {
        {true, true, false}, {false, true, true}, {true, false, true}};
    const std::vector<bool> signs = {false, true, false};
    const std::vector<double> angles = {0.3, -0.8, 1.1};
    const Mat u = synthesised_unitary(synthesise_diagonal_block_best, z, signs, angles, 3);
    EXPECT_TRUE(expect_unitary_close(u, reference_diagonal_block(z, signs, angles, 3)));
}

TEST(DiagonalBlockGraysynth, NeverWorseThanNaiveOnNestedParities) {
    // Nested parities are the best case for sharing: each differs from the
    // previous by one qubit, so every rotation after the first costs 1 CX.
    const std::vector<std::vector<bool>> z = {
        {true, false, false, false},
        {true, true,  false, false},
        {true, true,  true,  false},
        {true, true,  true,  true },
    };
    const std::vector<bool> signs(4, false);
    const std::vector<Param> angles = {Param(0.1), Param(0.2), Param(0.3), Param(0.4)};

    SimpleBlock gray(4, "g");
    synthesise_diagonal_block_graysynth(gray, z, signs, angles);
    SimpleBlock naive(4, "n");
    synthesise_diagonal_block_naive(naive, z, signs, angles);

    // Naive pays 2·(1−1) + 2·(2−1) + 2·(3−1) + 2·(4−1) = 12.
    EXPECT_EQ(count_cx(naive), 12);
    EXPECT_LT(count_cx(gray), count_cx(naive));
}

}  // namespace qarpx::test
