// ── Decomposition-table equivalence ─────────────────────────────────────────
//
// Walks the registered decomposition rules and asserts that each rule
// reproduces its gate's unitary EXACTLY — global phase included — on a
// register wider than the gate and at non-trivial operand placements.
//
// The oracle is `reference_unitary.h`: analytic gate matrices built straight
// from qarp_conventions.md §1–§7, sharing no code with csim / QarpSimulator,
// so a kernel bug cannot cancel against a rule bug.  Neither side of the
// comparison runs the simulator.
//
// Why this file exists: a rule is only covered by the rest of the suite if
// some test happens to target a gate set that excludes its gate.  `MCZ` is in
// `native_gateset()`, so `TEST(GateUnitary, MCZ_4q)` exercised the simulator
// *kernel* and never `decompose_mcz` — which silently dropped every control
// past the first two for width >= 4.
//
// Exactness is the contract, not a nicety: §16 EQ-2 requires passes to
// preserve the unitary including global phase, because a rule that is right
// only up to e^{iγ} becomes a *relative* phase the moment its block sits under
// a control (§13).

#include <gtest/gtest.h>

#include "qarpx/core/command.h"
#include "qarpx/core/gates.h"
#include "qarpx/transpiler/decompositions.h"

#include <Eigen/Dense>

#include <array>
#include <complex>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "gate_test_helpers.h"
#include "reference_unitary.h"

namespace qarpx::test {

namespace {

using cd  = std::complex<double>;
using Mat = Eigen::MatrixXcd;

/// One instantiation of a rule: the gate, the register it is evaluated on, its
/// operands, and its parameters.  Operands are deliberately not `0,1,2,…` so a
/// rule that hard-codes or transposes operand order is caught.
struct RuleCase {
    GateType gate;
    uint32_t n_qubits;
    std::vector<uint32_t> qubits;
    std::vector<double> params;
    std::string label;
};

Command make_command(const RuleCase& rc) {
    Command cmd;
    cmd.gate = rc.gate;
    for (uint32_t q : rc.qubits) cmd.qubits.push_back(q);
    for (double p : rc.params) cmd.params.push_back(Param(p));
    return cmd;
}

/// Apply the registered rule for `rc.gate` and compare the expansion's unitary
/// against the gate's own analytic unitary, both via `reference_unitary`.
::testing::AssertionResult rule_is_exact(const RuleCase& rc, double tol = 1e-10) {
    const auto table = builtin_decompositions();
    const auto it = table.find(rc.gate);
    if (it == table.end()) {
        return ::testing::AssertionFailure()
            << rc.label << ": no rule registered for " << gate_name(rc.gate);
    }

    const Command cmd = make_command(rc);
    const std::vector<Command> expansion = it->second(cmd);

    const Mat expected = reference_unitary({cmd}, rc.n_qubits);
    const Mat actual   = reference_unitary(expansion, rc.n_qubits);

    const double diff = (actual - expected).cwiseAbs().maxCoeff();
    if (diff < tol) return ::testing::AssertionSuccess();

    // Separate a phase-only regression from a genuine decomposition error —
    // they have very different causes and fixes.
    const cd overlap = (expected.adjoint() * actual).trace();
    const double gamma = std::arg(overlap);
    const double diff_mod_phase =
        (actual - std::exp(cd{0.0, gamma}) * expected).cwiseAbs().maxCoeff();

    return ::testing::AssertionFailure()
        << rc.label << " (" << gate_name(rc.gate) << "): max diff " << diff
        << " > tol " << tol
        << "\n  residual global phase " << gamma
        << " (diff after removing it: " << diff_mod_phase << ")"
        << (diff_mod_phase < tol ? "  <-- PHASE-ONLY regression" : "")
        << "\n  expansion has " << expansion.size() << " commands";
}

/// Angles chosen to include the values where closed forms degenerate:
/// 0 (identity), ±π/2, π (anti-diagonal / sign flip), plus a generic value.
const std::vector<double> kAngles = {0.0, 0.3, PI / 2, -PI / 2, PI, 2.4};

}  // namespace

// ── Parameterless / fixed-shape rules ───────────────────────────────────────

TEST(DecompositionTable, ParameterlessRulesAreExact) {
    const std::vector<RuleCase> cases = {
        // Operands are permuted and offset so a transposed-operand rule fails.
        {GateType::CY,      3, {2, 0}, {}, "CY(q2->q0) on 3q"},
        {GateType::CY,      2, {0, 1}, {}, "CY(q0->q1) on 2q"},
        {GateType::CZ,      3, {2, 0}, {}, "CZ(q2,q0) on 3q"},
        {GateType::CZ,      2, {0, 1}, {}, "CZ(q0,q1) on 2q"},
        {GateType::CCX,     4, {3, 1, 0}, {}, "CCX(c=q3,q1;t=q0) on 4q"},
        {GateType::CCX,     3, {0, 1, 2}, {}, "CCX on 3q"},
        {GateType::CSWAP,   4, {0, 3, 1}, {}, "CSWAP(c=q0;q3,q1) on 4q"},
        {GateType::CSWAP,   3, {0, 1, 2}, {}, "CSWAP on 3q"},
        {GateType::SWAP,    3, {2, 0}, {}, "SWAP(q2,q0) on 3q"},
        {GateType::ECR,     3, {2, 0}, {}, "ECR(q2,q0) on 3q"},
        {GateType::ECR,     2, {0, 1}, {}, "ECR(q0,q1) on 2q"},
        {GateType::iSWAP,   3, {2, 0}, {}, "iSWAP(q2,q0) on 3q"},
        {GateType::iSWAPdg, 3, {2, 0}, {}, "iSWAPdg(q2,q0) on 3q"},
        {GateType::Sdg,     2, {1}, {}, "Sdg(q1) on 2q"},
        {GateType::Tdg,     2, {1}, {}, "Tdg(q1) on 2q"},
        // Gateset expansion (§2.6, §3.1): every rule exact, phase included.
        {GateType::SX,      2, {1}, {}, "SX(q1) on 2q"},
        {GateType::SXdg,    2, {1}, {}, "SXdg(q1) on 2q"},
        {GateType::Id,      2, {1}, {}, "Id(q1) on 2q"},
        {GateType::CH,      3, {2, 0}, {}, "CH(q2->q0) on 3q"},
        {GateType::CS,      3, {2, 0}, {}, "CS(q2->q0) on 3q"},
        {GateType::CSdg,    3, {2, 0}, {}, "CSdg(q2->q0) on 3q"},
        {GateType::CSX,     3, {2, 0}, {}, "CSX(q2->q0) on 3q"},
        {GateType::CSXdg,   3, {2, 0}, {}, "CSXdg(q2->q0) on 3q"},
    };
    for (const auto& rc : cases) EXPECT_TRUE(rule_is_exact(rc));
}

// ── Single-angle rules, swept over degenerate and generic angles ────────────

TEST(DecompositionTable, SingleAngleRulesAreExact) {
    const std::vector<std::pair<GateType, std::vector<uint32_t>>> gates = {
        {GateType::CRx, {2, 0}},
        {GateType::CRy, {2, 0}},
        {GateType::CRz, {2, 0}},
        {GateType::CP,  {2, 0}},
        {GateType::RZZ, {2, 0}},
        {GateType::RXX, {2, 0}},
        {GateType::RYY, {2, 0}},
    };
    for (const auto& [gate, qubits] : gates) {
        for (double th : kAngles) {
            RuleCase rc{gate, 3, qubits, {th},
                        std::string(gate_name(gate)) + " θ=" + std::to_string(th)};
            EXPECT_TRUE(rule_is_exact(rc));
        }
    }
}

TEST(DecompositionTable, PhaseGateRuleIsExact) {
    // decompose_p emits GPhase(θ/2)·Rz(θ).  P and Rz differ by exactly that
    // global phase (§2.3), so dropping it here would be invisible standalone
    // and wrong under control.
    for (double th : kAngles) {
        RuleCase rc{GateType::P, 2, {1}, {th}, "P θ=" + std::to_string(th)};
        EXPECT_TRUE(rule_is_exact(rc));
    }
}

// ── Multi-angle rules ───────────────────────────────────────────────────────

TEST(DecompositionTable, URuleIsExact) {
    const std::vector<std::array<double, 3>> params = {
        {0.0, 0.0, 0.0}, {0.3, 0.5, 0.9}, {PI, 0.0, 0.0},
        {PI / 2, PI / 2, -PI / 2}, {0.0, 1.1, -0.4},
    };
    for (const auto& p : params) {
        RuleCase rc{GateType::U, 2, {1}, {p[0], p[1], p[2]},
                    "U(" + std::to_string(p[0]) + "," + std::to_string(p[1]) +
                        "," + std::to_string(p[2]) + ")"};
        EXPECT_TRUE(rule_is_exact(rc));
    }
}

TEST(DecompositionTable, CURuleIsExact) {
    // γ (4th param) is a phase on the controlled branch — a rule that drops it
    // is wrong in exactly the way this file exists to catch.
    const std::vector<std::array<double, 4>> params = {
        {0.0, 0.0, 0.0, 0.0}, {0.3, 0.5, 0.9, 0.2},
        {PI, 0.0, 0.0, PI / 2}, {0.7, -0.2, 1.3, -0.8},
    };
    for (const auto& p : params) {
        RuleCase rc{GateType::CU, 3, {2, 0}, {p[0], p[1], p[2], p[3]},
                    "CU γ=" + std::to_string(p[3])};
        EXPECT_TRUE(rule_is_exact(rc));
    }
}

// ── MCZ across widths ───────────────────────────────────────────────────────

TEST(DecompositionTable, MCZRuleIsExactAcrossWidths) {
    for (uint32_t width = 2; width <= 8; ++width) {
        std::vector<uint32_t> qubits(width);
        for (uint32_t k = 0; k < width; ++k) qubits[k] = k;
        RuleCase rc{GateType::MCZ, width, qubits, {},
                    "MCZ width=" + std::to_string(width)};
        EXPECT_TRUE(rule_is_exact(rc));
    }
}

TEST(DecompositionTable, MCZRuleIsExactAtPermutedOperands) {
    // MCZ is symmetric in its tuple, so a rule that special-cases "the last
    // qubit is the target" must still be right under any operand order.
    const std::vector<std::vector<uint32_t>> orders = {
        {3, 1, 0, 2}, {2, 0, 3, 1}, {0, 3, 2, 1},
    };
    for (const auto& qs : orders) {
        RuleCase rc{GateType::MCZ, 5, qs, {}, "MCZ permuted width=4 on 5q"};
        EXPECT_TRUE(rule_is_exact(rc));
    }
}

// Regression pins for the exact states the broken rule corrupted: it collapsed
// MCZ to CCZ(q0, q1, target), so every state with q0=q1=target=1 but some
// middle control 0 picked up a spurious -1.
TEST(DecompositionTable, MCZRuleDoesNotFlipMiddleControlZeroStates) {
    const auto table = builtin_decompositions();
    const auto rule = table.at(GateType::MCZ);

    struct Pin { uint32_t width; std::vector<int> states; };
    const std::vector<Pin> pins = {
        {4, {11}},           // q0q1q2=1, q3(target)=1, middle q2=0 -> 0b1011
        {5, {23, 27, 19}},   // middle controls not all 1
    };
    for (const auto& pin : pins) {
        std::vector<uint32_t> qs(pin.width);
        for (uint32_t k = 0; k < pin.width; ++k) qs[k] = k;
        Command cmd;
        cmd.gate = GateType::MCZ;
        for (uint32_t q : qs) cmd.qubits.push_back(q);
        const Mat u = reference_unitary(rule(cmd), pin.width);
        for (int s : pin.states) {
            EXPECT_NEAR(std::real(u(s, s)), 1.0, 1e-10)
                << "width " << pin.width << " state " << s
                << " must be untouched (not all controls are 1)";
        }
        const int all_ones = (1 << pin.width) - 1;
        EXPECT_NEAR(std::real(u(all_ones, all_ones)), -1.0, 1e-10)
            << "width " << pin.width << ": all-ones state must pick up -1";
    }
}

// ── MCZ cascade size ────────────────────────────────────────────────────────

TEST(DecompositionTable, MCZCascadeIsQuadraticInWidth) {
    // With the (7.2) v-chain engaged each `mc_phase` level costs O(width) and
    // there are O(width) levels, so the expansion is O(width^2); the split-only
    // predecessor was O(width^3).  The bound has to be checked where the two
    // actually separate: the quadratic constant is still settling below width
    // ~32 (measured gates/w^2 = 1.7, 3.9, 5.3, 6.1, 6.6 at w = 8, 16, 32, 64,
    // 128), and at width 16 a cubic cascade still fits under any constant loose
    // enough to pass a quadratic one.  At width 64 they are 6.1 w^2 against
    // 41 w^2 — so that is where the assertion earns its keep.
    const auto rule = builtin_decompositions().at(GateType::MCZ);

    auto expansion_size = [&rule](uint32_t width) {
        Command cmd;
        cmd.gate = GateType::MCZ;
        for (uint32_t k = 0; k < width; ++k) cmd.qubits.push_back(k);
        return rule(cmd).size();
    };

    std::size_t prev = 0;
    for (uint32_t width = 4; width <= 16; ++width) {
        const std::size_t n = expansion_size(width);
        EXPECT_GT(n, prev) << "width " << width << ": expansion must grow with width";
        prev = n;
    }

    // 10 w^2 sits above the quadratic curve's limit (~7) and far below the
    // cubic one, which reaches 41 w^2 by this width.
    const uint32_t wide = 64;
    const std::size_t n_wide = expansion_size(wide);
    EXPECT_LE(n_wide, 10u * wide * wide)
        << "width " << wide << ": " << n_wide
        << " gates — cascade regressed to the cubic (split-only) construction";

    // Doubling test: quadratic -> ~4x, cubic -> ~9x.  Measured 4.6x here.
    const std::size_t n_half = expansion_size(wide / 2);
    EXPECT_LT(static_cast<double>(n_wide) / static_cast<double>(n_half), 6.0)
        << "width " << wide / 2 << " -> " << wide << ": growth "
        << static_cast<double>(n_wide) / static_cast<double>(n_half)
        << "x is cubic, not quadratic";
}

// The v-chain's exactness (and its ancilla restoration) is covered by
// `MCZRuleIsExactAcrossWidths` above: widths 4-5 take it for the top-level
// ladder directly, and from width 6 both halves of the (7.3) split reach it.
// A dedicated wider sweep was tried and removed — `reference_unitary` is a
// dense 2^w x 2^w product per command, so width 10 alone ran ~18 min and
// added no path the width 2-8 sweep does not already hit.

// ── Custom (1-qubit ZYZ re-opening) ─────────────────────────────────────────

TEST(DecompositionTable, CustomRuleIsExact) {
    // decompose_custom re-opens a fused 1-qubit Custom via phase-exact ZYZ
    // (§6.2 rebase totality).  Haar-random targets land on the generic ZYZ
    // branch; diagonal and anti-diagonal ones hit the edge branches that were
    // wrong before 2026-08-07.
    std::mt19937 rng(20260807);
    std::vector<Mat> targets;
    {
        std::normal_distribution<double> g(0.0, 1.0);
        for (int trial = 0; trial < 4; ++trial) {
            Mat a(2, 2);
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j) a(i, j) = cd{g(rng), g(rng)};
            Eigen::HouseholderQR<Mat> qr(a);
            Mat q = qr.householderQ() * Mat::Identity(2, 2);
            Mat r = qr.matrixQR().triangularView<Eigen::Upper>();
            for (int k = 0; k < 2; ++k)
                if (std::abs(r(k, k)) > 1e-15) q.col(k) *= r(k, k) / std::abs(r(k, k));
            targets.push_back(q);
        }
    }
    {   // diagonal (γ ≈ 0 branch)
        Mat d = Mat::Zero(2, 2);
        d(0, 0) = std::exp(cd{0.0, 0.7});
        d(1, 1) = std::exp(cd{0.0, -1.3});
        targets.push_back(d);
    }
    {   // anti-diagonal (γ ≈ π branch)
        Mat a = Mat::Zero(2, 2);
        a(0, 1) = std::exp(cd{0.0, 0.4});
        a(1, 0) = std::exp(cd{0.0, 2.1});
        targets.push_back(a);
    }

    const auto table = builtin_decompositions();
    const auto rule = table.at(GateType::Custom);
    for (size_t i = 0; i < targets.size(); ++i) {
        Command cmd;
        cmd.gate = GateType::Custom;
        cmd.qubits.push_back(1u);
        cmd.unitary = std::make_shared<const Mat>(targets[i]);
        const Mat expected = reference_unitary({cmd}, 2);
        const Mat actual   = reference_unitary(rule(cmd), 2);
        EXPECT_TRUE(expect_unitary_close(actual, expected))
            << "Custom target #" << i;
    }
}

// ── The one deliberately lossy rule ─────────────────────────────────────────

TEST(DecompositionTable, GPhaseRuleDropsExactlyTheGlobalPhase) {
    // `decompose_gphase` returns {} on purpose: it is the per-target boundary
    // for phase loss, for targets (cudaq) with no global-phase primitive.  It
    // is therefore the sole rule exempt from the exactness sweep above, and
    // that exemption is asserted rather than assumed — the expansion must be
    // the identity, i.e. drop the phase and nothing else.
    const auto table = builtin_decompositions();
    const auto rule = table.at(GateType::GPhase);
    for (double th : kAngles) {
        Command cmd;
        cmd.gate = GateType::GPhase;
        cmd.params.push_back(Param(th));
        const std::vector<Command> expansion = rule(cmd);
        EXPECT_TRUE(expansion.empty()) << "GPhase rule must drop, not rewrite";
        const Mat actual = reference_unitary(expansion, 1);
        EXPECT_TRUE(expect_unitary_close(actual, Mat::Identity(2, 2)));
    }
}

// ── Coverage guard ──────────────────────────────────────────────────────────

TEST(DecompositionTable, EveryRegisteredRuleIsCoveredBySomeTest) {
    // A new table entry with no case above would otherwise land untested — the
    // exact hole that let the MCZ bug through.  Update this set together with
    // the cases when adding a rule.
    const std::set<GateType> covered = {
        GateType::CY,      GateType::MCZ,     GateType::CCX,   GateType::CSWAP,
        GateType::CRx,     GateType::CRy,     GateType::CRz,   GateType::CP,
        GateType::RZZ,     GateType::RXX,     GateType::RYY,   GateType::ECR,
        GateType::iSWAP,   GateType::iSWAPdg, GateType::SWAP,  GateType::U,
        GateType::P,       GateType::Sdg,     GateType::Tdg,   GateType::GPhase,
        GateType::CU,      GateType::Custom,  GateType::CZ,
        GateType::SX,      GateType::SXdg,    GateType::Id,
        GateType::CH,      GateType::CS,      GateType::CSdg,  GateType::CSX,
        GateType::CSXdg,
    };
    for (const auto& [gate, fn] : builtin_decompositions()) {
        EXPECT_TRUE(covered.count(gate) > 0)
            << "decomposition rule for '" << gate_name(gate)
            << "' has no case in test_decomposition_table.cpp";
    }
}

// ── Rz-only-target overrides ────────────────────────────────────────────────

TEST(DecompositionTable, RzOnlyOverrideRulesAreExact) {
    // `clifford_t_rz_decompositions()` is the built-in table plus the Rx/Ry rules; the
    // shared entries are already swept above, so only those two are exercised
    // here.  They must meet the same exactness bar (§16 EQ-2) — a phase slip
    // here reaches QPE through the Clifford+T+Rz pipeline.
    const auto rz_only = clifford_t_rz_decompositions();
    const auto builtin = builtin_decompositions();

    for (GateType gate : {GateType::Rx, GateType::Ry}) {
        const auto rule = rz_only.at(gate);
        for (double th : kAngles) {
            Command cmd;
            cmd.gate = gate;
            cmd.qubits.push_back(1u);
            cmd.params.push_back(Param(th));
            const Mat expected = reference_unitary({cmd}, 3);
            const Mat actual   = reference_unitary(rule(cmd), 3);
            EXPECT_TRUE(expect_unitary_close(actual, expected))
                << "Rz-only rule " << gate_name(gate) << " θ=" << th;
        }
    }

    // The Rx/Ry rules must be genuine additions — a built-in rule appearing
    // for either means the Rz-only override is silently shadowed.
    EXPECT_EQ(builtin.count(GateType::Rx), 0u)
        << "built-in table gained an Rx rule — re-check the Rz-only override";
    EXPECT_EQ(builtin.count(GateType::Ry), 0u)
        << "built-in table gained an Ry rule — re-check the Rz-only override";
}

}  // namespace qarpx::test
