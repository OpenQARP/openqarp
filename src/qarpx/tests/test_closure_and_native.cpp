#include <gtest/gtest.h>
#include "qarpx/qarpx.h"

#include <complex>
#include <vector>

using namespace qarpx;

namespace {

double max_abs_diff(const std::vector<std::complex<double>>& a,
                   const std::vector<std::complex<double>>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

double max_prob_diff(const std::vector<std::complex<double>>& a,
                     const std::vector<std::complex<double>>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::abs(std::norm(a[i]) - std::norm(b[i])));
    return m;
}

}  // namespace

// ── Closure verification ──────────────────────────────────────────────────────

TEST(Closure, NativeGateSetIsClosed) {
    Transpiler t(native_gateset());
    EXPECT_TRUE(t.unreachable_gates().empty());
    EXPECT_NO_THROW(t.verify_closure());
}

TEST(Closure, QulacsGateSetIsClosed) {
    Transpiler t(qulacs_gateset());
    EXPECT_TRUE(t.unreachable_gates().empty());
    EXPECT_NO_THROW(t.verify_closure());
}

TEST(Closure, CliffordTRzGateSetIsClosedWithoutAnInstallCall) {
    // Per-target rules travel with the target: a Transpiler built from
    // clifford_t_rz_gateset() carries the Rx/Ry overrides from construction.
    Transpiler t(clifford_t_rz_gateset());
    EXPECT_TRUE(t.unreachable_gates().empty());
    EXPECT_NO_THROW(t.verify_closure());

    // Kept for source compatibility; a second install changes nothing.
    t.install_clifford_t_rz_decompositions();
    EXPECT_TRUE(t.unreachable_gates().empty());
    auto out = t.transpile({Command(GateType::Rx, 0u, Param(0.3))});
    ASSERT_EQ(out.size(), 3u);
    EXPECT_EQ(out[0].gate, GateType::H);
    EXPECT_EQ(out[1].gate, GateType::Rz);
    EXPECT_EQ(out[2].gate, GateType::H);
}

TEST(Closure, PerTargetRulesFollowTheRulesTagNotTheName) {
    // The overrides are chosen by GateSet::rules, which derived sets carry
    // (routable_subset) and a rename does not disturb.
    GateSet renamed = clifford_t_rz_gateset();
    renamed.name = "not_clifford_t_rz";
    EXPECT_TRUE(Transpiler(renamed).unreachable_gates().empty());

    EXPECT_EQ(routable_subset(clifford_t_rz_gateset()).rules, "clifford_t_rz");
    EXPECT_TRUE(Transpiler(routable_subset(clifford_t_rz_gateset())).unreachable_gates().empty());

    GateSet untagged = clifford_t_rz_gateset();
    untagged.rules.clear();
    EXPECT_FALSE(Transpiler(untagged).unreachable_gates().empty());

    GateSet bogus = clifford_t_rz_gateset();
    bogus.rules = "no_such_rules";
    EXPECT_THROW(Transpiler{bogus}, std::invalid_argument);
}

TEST(Closure, NoDecompositionErrorListsEveryUnreachableGate) {
    // One failure should name every gap of the target, not just the gate hit.
    GateSet only_h{"only_h", {GateType::H, GateType::Measure, GateType::Barrier}};
    Transpiler t(only_h);
    try {
        (void)t.transpile({Command(GateType::Rx, 0u, Param(0.3))});
        FAIL() << "expected a runtime_error";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("Rx"), std::string::npos) << msg;
        EXPECT_NE(msg.find("only_h"), std::string::npos) << msg;
        // Gates never mentioned by the input, but equally unreachable.
        EXPECT_NE(msg.find("CX"), std::string::npos) << msg;
        EXPECT_NE(msg.find("Rz"), std::string::npos) << msg;
    }
}

TEST(Closure, CudaqGateSetIsClosed) {
    // cudaq lacks Sdg/Tdg/GPhase, but the rule table provides:
    //   Sdg → P(-π/2)   (P is in cudaq)
    //   Tdg → P(-π/4)
    //   GPhase → []     (unobservable; dropped)
    // so the closure check passes.
    Transpiler t(cudaq_gateset());
    EXPECT_TRUE(t.unreachable_gates().empty());
    EXPECT_NO_THROW(t.verify_closure());
}

TEST(Closure, CudaqDecomposesSdgIntoSingleP) {
    Transpiler t(cudaq_gateset());
    auto out = t.transpile({Command(GateType::Sdg, 0u)});
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].gate, GateType::P);
}

TEST(Closure, CudaqDropsGPhase) {
    Transpiler t(cudaq_gateset());
    Command gphase(GateType::GPhase,
        SmallVector<uint32_t, 2>{},
        SmallVector<Param, 1>{Param(0.5)});
    auto out = t.transpile({
        Command(GateType::H, 0u),
        gphase,
        Command(GateType::H, 0u),
    });
    // GPhase rewritten away; only the two H gates remain.
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].gate, GateType::H);
    EXPECT_EQ(out[1].gate, GateType::H);
}

TEST(Closure, CliffordTHasGapsWithoutSynthesis) {
    Transpiler t(clifford_t_gateset());
    // Clifford+T cannot be reached for arbitrary rotations without
    // Solovay-Kitaev-style ε-synthesis, which the rule table does not provide.
    EXPECT_FALSE(t.unreachable_gates().empty());
}

TEST(Closure, VerifyClosureThrowsOnUnreachableTarget) {
    GateSet only_h{"only_h", {GateType::H, GateType::Measure, GateType::Barrier}};
    Transpiler t(only_h);
    EXPECT_THROW(t.verify_closure(), std::runtime_error);
}

// ── Output validation post-cascade ────────────────────────────────────────────

TEST(OutputValidation, ThrowsOnResidualNonTargetGate) {
    Transpiler t(qulacs_gateset());
    EXPECT_NO_THROW((void)t.transpile({Command(GateType::CY, 0u, 1u)}));

    // If a registered rule returns the source gate itself (e.g., a user
    // mistake), the cascade hits kMaxDecomposeDepth and the post-pass
    // validator catches the residual gate instead of leaking it silently.
    t.register_decomposition(GateType::CY, [](const Command& cmd) {
        return std::vector<Command>{cmd};  // pathological self-loop
    });
    EXPECT_THROW((void)t.transpile({Command(GateType::CY, 0u, 1u)}),
                 std::runtime_error);
}

TEST(OutputValidation, MetaGatesPassThroughRegardlessOfTarget) {
    GateSet minimal{"minimal", {GateType::H, GateType::CX}};
    // Need to register a rule for everything else — but Barrier/Measure/Reset/
    // Custom should pass through without needing rules.
    // Just check that meta gates are accepted without registered rules.
    Transpiler t(minimal);
    auto out = t.transpile({
        Command(GateType::H, 0u),
        Command(GateType::Barrier, 0u),
        Command(GateType::Measure, 0u),
        Command(GateType::H, 0u),
    });
    EXPECT_EQ(out.size(), 4u);
    EXPECT_EQ(out[1].gate, GateType::Barrier);
    EXPECT_EQ(out[2].gate, GateType::Measure);
}

// ── Native-dispatch correctness vs. decomposed reference ──────────────────────

TEST(NativeDispatch, RZZMatchesDecomposedSV) {
    QarpSimulator sim;
    std::vector<Command> cmds = {
        Command(GateType::H, 0u), Command(GateType::H, 1u),
        Command(GateType::RZZ, 0u, 1u, Param(0.7)),
    };
    auto sv_native = sim.statevector(cmds, 2);

    Transpiler t(qulacs_gateset());
    auto sv_decomp = sim.statevector(t.transpile(cmds), 2);

    EXPECT_LT(max_abs_diff(sv_native, sv_decomp), 1e-10);
}

TEST(NativeDispatch, RXXMatchesDecomposedSV) {
    QarpSimulator sim;
    std::vector<Command> cmds = {
        Command(GateType::H, 0u),
        Command(GateType::RXX, 0u, 1u, Param(1.3)),
    };
    auto sv_native = sim.statevector(cmds, 2);
    Transpiler t(qulacs_gateset());
    auto sv_decomp = sim.statevector(t.transpile(cmds), 2);
    EXPECT_LT(max_abs_diff(sv_native, sv_decomp), 1e-10);
}

TEST(NativeDispatch, CRzMatchesDecomposedSV) {
    QarpSimulator sim;
    std::vector<Command> cmds = {
        Command(GateType::H, 0u),
        Command(GateType::CRz, 0u, 1u, Param(0.9)),
    };
    auto sv_native = sim.statevector(cmds, 2);
    Transpiler t(qulacs_gateset());
    auto sv_decomp = sim.statevector(t.transpile(cmds), 2);
    EXPECT_LT(max_abs_diff(sv_native, sv_decomp), 1e-10);
}

TEST(NativeDispatch, CYMatchesDecomposedSV) {
    QarpSimulator sim;
    std::vector<Command> cmds = {
        Command(GateType::H, 0u),
        Command(GateType::CY, 0u, 1u),
    };
    auto sv_native = sim.statevector(cmds, 2);
    Transpiler t(qulacs_gateset());
    auto sv_decomp = sim.statevector(t.transpile(cmds), 2);
    EXPECT_LT(max_abs_diff(sv_native, sv_decomp), 1e-10);
}

TEST(NativeDispatch, CCXMatchesDecomposedSV) {
    QarpSimulator sim;
    std::vector<Command> cmds = {
        Command(GateType::H, 0u), Command(GateType::H, 1u),
        Command(GateType::CCX, SmallVector<uint32_t,2>{0u, 1u, 2u}),
    };
    auto sv_native = sim.statevector(cmds, 3);
    Transpiler t(qulacs_gateset());
    auto sv_decomp = sim.statevector(t.transpile(cmds), 3);
    EXPECT_LT(max_abs_diff(sv_native, sv_decomp), 1e-10);
}

TEST(NativeDispatch, PMatchesDecomposedProbabilities) {
    // P(θ) = diag(1, e^{iθ}); decomposes to Rz(θ) which differs by a global
    // phase factor of e^{iθ/2}.  Native dispatch is the exact P; verify they
    // agree on probabilities (the only physically observable quantity).
    QarpSimulator sim;
    std::vector<Command> cmds = {
        Command(GateType::H, 0u),
        Command(GateType::P, 0u, Param(1.1)),
    };
    auto sv_native = sim.statevector(cmds, 1);
    Transpiler t(qulacs_gateset());
    auto sv_decomp = sim.statevector(t.transpile(cmds), 1);
    EXPECT_LT(max_prob_diff(sv_native, sv_decomp), 1e-10);
}

TEST(NativeDispatch, MCZ3ControlsFlipsOnly1111) {
    // Independent-of-decomposition reference: MCZ on |++++⟩ should flip the
    // sign of |1111⟩ only; all other 15 amplitudes unchanged.
    QarpSimulator sim;
    std::vector<Command> setup = {
        Command(GateType::H, 0u), Command(GateType::H, 1u),
        Command(GateType::H, 2u), Command(GateType::H, 3u),
    };
    auto with_mcz = setup;
    with_mcz.push_back(Command(GateType::MCZ,
        SmallVector<uint32_t,2>{0u, 1u, 2u, 3u}));

    auto sv_no = sim.statevector(setup, 4);
    auto sv_mcz = sim.statevector(with_mcz, 4);

    for (size_t i = 0; i < sv_no.size(); ++i) {
        if (i == 15) {
            EXPECT_NEAR(std::real(sv_mcz[i] / sv_no[i]), -1.0, 1e-10);
        } else {
            EXPECT_NEAR(std::real(sv_mcz[i] / sv_no[i]), 1.0, 1e-10);
        }
    }
}

TEST(NativeDispatch, NativeGateSetSkipsDecomposition) {
    // With the native gateset, gates like RZZ/CCX/CRz pass through transpile()
    // unchanged.  This is the speedup: no decomposition cost, no extra csim
    // sweeps from decomposed sub-gates.
    Transpiler t(native_gateset());
    std::vector<Command> input = {
        Command(GateType::RZZ, 0u, 1u, Param(0.5)),
        Command(GateType::CCX, SmallVector<uint32_t,2>{0u, 1u, 2u}),
        Command(GateType::CRz, 0u, 1u, Param(0.3)),
        Command(GateType::P, 0u, Param(0.7)),
    };
    auto out = t.transpile(input);
    ASSERT_EQ(out.size(), input.size());
    for (size_t i = 0; i < input.size(); ++i)
        EXPECT_EQ(out[i].gate, input[i].gate);
}
