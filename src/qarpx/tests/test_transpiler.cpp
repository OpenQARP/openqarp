#include <gtest/gtest.h>
#include "qarpx/qarpx.h"

#include <cmath>
#include <complex>

#include "reference_unitary.h"

using namespace qarpx;

TEST(Transpiler, NativeGatesPassThrough) {
    Transpiler t(qulacs_gateset());
    std::vector<Command> input = {
        Command(GateType::H, 0),
        Command(GateType::CX, 0, 1),
        Command(GateType::Rx, 0, Param(1.57)),
    };
    auto result = t.transpile(input);
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0].gate, GateType::H);
    EXPECT_EQ(result[1].gate, GateType::CX);
    EXPECT_EQ(result[2].gate, GateType::Rx);
}

TEST(Transpiler, DecomposeCY) {
    Transpiler t(qulacs_gateset());
    std::vector<Command> input = {
        Command(GateType::CY, 0, 1),
    };
    auto result = t.transpile(input);
    // CY -> Sdg CX S
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0].gate, GateType::Sdg);
    EXPECT_EQ(result[1].gate, GateType::CX);
    EXPECT_EQ(result[2].gate, GateType::S);
}

TEST(Transpiler, DecomposeRZZ) {
    Transpiler t(qulacs_gateset());
    Param theta(0.5);
    std::vector<Command> input = {
        Command(GateType::RZZ, 0, 1, theta),
    };
    auto result = t.transpile(input);
    // RZZ -> CX Rz CX
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0].gate, GateType::CX);
    EXPECT_EQ(result[1].gate, GateType::Rz);
    EXPECT_EQ(result[2].gate, GateType::CX);
}

TEST(Transpiler, CascadingDecomposition) {
    // RXX -> H H RZZ H H -> H H CX Rz CX H H
    Transpiler t(qulacs_gateset());
    std::vector<Command> input = {
        Command(GateType::RXX, 0, 1, Param(0.5)),
    };
    auto result = t.transpile(input);
    // Should be fully decomposed to qulacs-native gates
    for (const auto& cmd : result) {
        EXPECT_TRUE(qulacs_gateset().contains(cmd.gate))
            << "Gate " << gate_name(cmd.gate) << " not in qulacs gate set";
    }
    // H H CX Rz CX H H = 7 gates
    EXPECT_EQ(result.size(), 7u);
}

TEST(Transpiler, SymbolicParametersPreserved) {
    Transpiler t(qulacs_gateset());
    Param theta = Param::symbol("theta");
    std::vector<Command> input = {
        Command(GateType::CRz, 0, 1, theta),
    };
    auto result = t.transpile(input);
    // CRz -> Rz(theta/2) CX Rz(-theta/2) CX
    ASSERT_EQ(result.size(), 4u);
    // Check that symbolic params survived decomposition
    EXPECT_TRUE(result[0].is_parametric());
}

TEST(Transpiler, TranspileParallel) {
    Transpiler t(qulacs_gateset());
    std::vector<std::vector<Command>> blocks = {
        {Command(GateType::CY, 0, 1)},
        {Command(GateType::CY, 2, 3)},
        {Command(GateType::RZZ, 0, 1, Param(0.5))},
    };

    auto results = t.transpile_parallel(blocks);
    ASSERT_EQ(results.size(), 3u);
    EXPECT_EQ(results[0].size(), 3u);  // CY -> 3 gates
    EXPECT_EQ(results[1].size(), 3u);
    EXPECT_EQ(results[2].size(), 3u);  // RZZ -> 3 gates
}

TEST(Transpiler, CustomDecomposition) {
    Transpiler t(qulacs_gateset());

    // Register a custom (silly) decomposition for CY that uses X instead
    t.register_decomposition(GateType::CY,
        [](const Command& cmd) -> std::vector<Command> {
            return {Command(GateType::X, cmd.qubits[1])};
        });

    std::vector<Command> input = {Command(GateType::CY, 0, 1)};
    auto result = t.transpile(input);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].gate, GateType::X);
}

TEST(Transpiler, TranspileAndOptimize) {
    Transpiler t(qulacs_gateset());
    // H H should cancel after decomposition and optimization
    std::vector<Command> input = {
        Command(GateType::H, 0),
        Command(GateType::H, 0),
        Command(GateType::CX, 0, 1),
    };
    auto result = t.transpile_and_optimize(input);
    // H H should be eliminated, leaving just CX
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].gate, GateType::CX);
}

// ── Wide MCZ cascades to completion ────────────────────────────────────────
//
// `decompose_mcz` recurses *inside the rule*, so a width-w MCZ leaves the rule
// already expanded to CP/CX/CCX/P.  Cascade depth is therefore constant in w
// rather than proportional to it, and no width bumps into
// `kMaxDecomposeDepth`.  Guards against a future rewrite that recurses across
// transpiler passes instead, which would silently cap the usable width.

TEST(Transpiler, WideMCZFullyLowersWithinDepthBound) {
    GateSet target{
        .name = "clifford_t_rz",
        .allowed = {
            GateType::H, GateType::S, GateType::Sdg, GateType::T, GateType::Tdg,
            GateType::X, GateType::Y, GateType::Z, GateType::CX,
            GateType::Rx, GateType::Ry, GateType::Rz, GateType::P,
            GateType::GPhase, GateType::Barrier,
        }
    };
    Transpiler t(target);

    for (uint32_t width = 2; width <= 12; ++width) {
        SmallVector<uint32_t, 2> qubits;
        for (uint32_t k = 0; k < width; ++k) qubits.push_back(k);
        Command mcz;
        mcz.gate = GateType::MCZ;
        mcz.qubits = qubits;

        // transpile() throws if any residual out-of-target gate survives, so
        // reaching the assertions at all is the convergence check.
        std::vector<Command> result;
        ASSERT_NO_THROW(result = t.transpile({mcz})) << "width " << width;
        for (const auto& cmd : result) {
            EXPECT_TRUE(target.contains(cmd.gate))
                << "width " << width << ": residual " << gate_name(cmd.gate);
            EXPECT_NE(cmd.gate, GateType::MCZ)
                << "width " << width << ": rule re-emitted MCZ (3^n cascade risk)";
        }
    }
}

// ── Wide MCZ on a GPhase-less target: the loss is exactly one known phase ───
//
// `decompose_gphase` returns {} for targets with no global-phase primitive
// (cudaq).  Inside a wide MCZ the CP(+/-θ/2) conjugation pairs cancel their
// GPhase(θ/4), but the recursion's base CP(π/2^(m-1)) does not — so exactly
// e^{-iπ/2^(m+1)} is dropped, m = width-1 controls.  Everything else stays
// exact.  This is the documented per-target loss boundary (§5): asserted
// here rather than assumed, so a rule change that loses a *different* phase
// is a test failure and not a silent QPE bug.

TEST(Transpiler, WideMCZOnGPhaseLessTargetLosesOnlyTheKnownPhase) {
    GateSet target = cudaq_gateset();
    ASSERT_FALSE(target.contains(GateType::GPhase)) << "premise of this test";
    Transpiler t(target);

    for (uint32_t width = 3; width <= 6; ++width) {
        SmallVector<uint32_t, 2> qubits;
        for (uint32_t k = 0; k < width; ++k) qubits.push_back(k);
        Command mcz;
        mcz.gate = GateType::MCZ;
        mcz.qubits = qubits;

        const auto lowered = t.transpile({mcz});
        const Eigen::MatrixXcd got = qarpx::test::reference_unitary(lowered, width);
        const Eigen::MatrixXcd mcz_exact = qarpx::test::reference_unitary({mcz}, width);

        // Widths <= 3 take the Clifford special cases and stay phase-exact.
        const uint32_t m = width - 1;
        const double expected_phase =
            (width <= 3) ? 0.0 : -std::numbers::pi / std::pow(2.0, m + 1);
        const std::complex<double> factor{std::cos(expected_phase),
                                          std::sin(expected_phase)};

        EXPECT_LT((got - factor * mcz_exact).cwiseAbs().maxCoeff(), 1e-9)
            << "width " << width << ": lowered unitary is not MCZ times the "
            << "predicted phase " << expected_phase
            << " (actual arg " << std::arg(got(0, 0)) << ")";
    }
}

// ── optimize_in_target is total at every level (P1.20, §16) ────────────────

namespace {

// An O1-fused Custom from a Custom-admitting target: the one input that used
// to slip through `require_in_target_` because is_meta_gate lists Custom.
std::vector<Command> fused_custom_input() {
    std::vector<Command> input = {
        Command(GateType::H, 0),
        Command(GateType::T, 0),
        Command(GateType::Rz, 0, Param(0.7)),
    };
    auto fused = Transpiler(native_gateset()).transpile_and_optimize(input, OptLevel::O1);
    EXPECT_TRUE(std::any_of(fused.begin(), fused.end(),
        [](const Command& c) { return c.gate == GateType::Custom; }));
    return fused;
}

}  // namespace

TEST(Transpiler, OptimizeInTargetRefusesFusedCustomOnCustomFreeTarget) {
    const auto fused = fused_custom_input();
    for (const GateSet& gs : {qiskit_gateset(), pytket_gateset(), clifford_t_rz_gateset()}) {
        ASSERT_FALSE(gs.contains(GateType::Custom)) << gs.name;
        for (OptLevel level : {OptLevel::O0, OptLevel::O1, OptLevel::O2}) {
            EXPECT_THROW((void)Transpiler(gs).optimize_in_target(fused, level), capability_error)
                << gs.name << " O" << static_cast<int>(level);
        }
    }
    // transpile re-opens the same Custom (ZYZ) — the rebase stays total.
    auto reopened = Transpiler(qiskit_gateset()).transpile(fused);
    EXPECT_TRUE(std::none_of(reopened.begin(), reopened.end(),
        [](const Command& c) { return c.gate == GateType::Custom; }));
}

TEST(Transpiler, OptimizeInTargetVerifiesAtO0) {
    GateSet gs;
    gs.name = "rx_rz";
    gs.allowed = {GateType::Rx, GateType::Rz};
    const std::vector<Command> off_target = {Command(GateType::H, 0)};
    EXPECT_THROW((void)Transpiler(gs).optimize_in_target(off_target, OptLevel::O0),
                 capability_error);
    const std::vector<Command> in_target = {Command(GateType::Rx, 0, Param(0.3))};
    auto out = Transpiler(gs).optimize_in_target(in_target, OptLevel::O0);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].gate, GateType::Rx);
}
