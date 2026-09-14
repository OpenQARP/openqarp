#include <gtest/gtest.h>
#include "qarpx/qarpx.h"

#include "dag_test_helpers.h"

using namespace qarpx;

TEST(Identities, HHCancellation) {
    std::vector<Command> cmds = {
        Command(GateType::H, 0),
        Command(GateType::H, 0),
    };
    auto eliminated = eliminate_identities(cmds);
    EXPECT_EQ(eliminated, 2u);
    EXPECT_TRUE(cmds.empty());
}

TEST(Identities, XXCancellation) {
    std::vector<Command> cmds = {
        Command(GateType::X, 0),
        Command(GateType::X, 0),
    };
    auto eliminated = eliminate_identities(cmds);
    EXPECT_EQ(eliminated, 2u);
    EXPECT_TRUE(cmds.empty());
}

TEST(Identities, CXCXCancellation) {
    std::vector<Command> cmds = {
        Command(GateType::CX, 0, 1),
        Command(GateType::CX, 0, 1),
    };
    auto eliminated = eliminate_identities(cmds);
    EXPECT_EQ(eliminated, 2u);
    EXPECT_TRUE(cmds.empty());
}

TEST(Identities, CXDifferentQubitsNoCancel) {
    std::vector<Command> cmds = {
        Command(GateType::CX, 0, 1),
        Command(GateType::CX, 1, 0),  // reversed qubits — NOT an identity
    };
    auto eliminated = eliminate_identities(cmds);
    EXPECT_EQ(eliminated, 0u);
    EXPECT_EQ(cmds.size(), 2u);
}

TEST(Identities, STdgCancellation) {
    std::vector<Command> cmds = {
        Command(GateType::S, 0),
        Command(GateType::Sdg, 0),
    };
    auto eliminated = eliminate_identities(cmds);
    EXPECT_EQ(eliminated, 2u);
    EXPECT_TRUE(cmds.empty());
}

TEST(Identities, TTdgCancellation) {
    std::vector<Command> cmds = {
        Command(GateType::Tdg, 0),
        Command(GateType::T, 0),
    };
    auto eliminated = eliminate_identities(cmds);
    EXPECT_EQ(eliminated, 2u);
    EXPECT_TRUE(cmds.empty());
}

TEST(Identities, RotationMerge) {
    std::vector<Command> cmds = {
        Command(GateType::Rz, 0, Param(0.3)),
        Command(GateType::Rz, 0, Param(0.7)),
    };
    auto eliminated = eliminate_identities(cmds);
    EXPECT_EQ(eliminated, 1u);
    ASSERT_EQ(cmds.size(), 1u);
    EXPECT_DOUBLE_EQ(cmds[0].params[0].value(), 1.0);
}

TEST(Identities, RotationMergeToZero) {
    std::vector<Command> cmds = {
        Command(GateType::Rx, 0, Param(0.5)),
        Command(GateType::Rx, 0, Param(-0.5)),
    };
    auto eliminated = eliminate_identities(cmds);
    EXPECT_EQ(eliminated, 2u);
    EXPECT_TRUE(cmds.empty());
}

TEST(Identities, ZeroRotationRemoved) {
    std::vector<Command> cmds = {
        Command(GateType::Rz, 0, Param(0.0)),
        Command(GateType::H, 1),
    };
    auto eliminated = eliminate_identities(cmds);
    EXPECT_EQ(eliminated, 1u);
    ASSERT_EQ(cmds.size(), 1u);
    EXPECT_EQ(cmds[0].gate, GateType::H);
}

TEST(Identities, MixedCircuitPartialElimination) {
    std::vector<Command> cmds = {
        Command(GateType::H, 0),
        Command(GateType::H, 0),        // cancel with previous
        Command(GateType::CX, 0, 1),    // survives
        Command(GateType::Rz, 1, Param(0.5)),
        Command(GateType::Rz, 1, Param(0.5)),  // merge -> Rz(1.0)
    };
    auto eliminated = eliminate_identities(cmds);
    EXPECT_EQ(eliminated, 3u);  // 2 from H+H, 1 from Rz merge
    ASSERT_EQ(cmds.size(), 2u);
    EXPECT_EQ(cmds[0].gate, GateType::CX);
    EXPECT_EQ(cmds[1].gate, GateType::Rz);
    EXPECT_DOUBLE_EQ(cmds[1].params[0].value(), 1.0);
}

TEST(Identities, CascadingCancellation) {
    // A B B A -> after B B cancel, A A cancel
    std::vector<Command> cmds = {
        Command(GateType::H, 0),
        Command(GateType::X, 0),
        Command(GateType::X, 0),  // X X cancel
        Command(GateType::H, 0),  // H H cancel (now adjacent)
    };
    auto eliminated = eliminate_identities(cmds);
    EXPECT_EQ(eliminated, 4u);
    EXPECT_TRUE(cmds.empty());
}

TEST(Identities, DifferentQubitsDontCancel) {
    std::vector<Command> cmds = {
        Command(GateType::H, 0),
        Command(GateType::H, 1),  // different qubit — no cancel
    };
    auto eliminated = eliminate_identities(cmds);
    EXPECT_EQ(eliminated, 0u);
    EXPECT_EQ(cmds.size(), 2u);
}

// ── Symbolic rotation merges must stay differentiable ──────────────────────
//
// `Param` carries the linear form α·sym + β and the adjoint-gradient path
// differentiates one symbol per gate.  Merging two *distinct* symbols into one
// angle is unitary-exact but leaves a gate no gradient path can handle, so the
// merge must decline and keep both gates.

TEST(Identities, DistinctSymbolicRotationsDoNotMerge) {
    std::vector<Command> cmds = {
        Command(GateType::Ry, 0, Param::symbol("a")),
        Command(GateType::Ry, 0, Param::symbol("b")),
    };
    auto eliminated = eliminate_identities(cmds);
    EXPECT_EQ(eliminated, 0u);
    ASSERT_EQ(cmds.size(), 2u);
    EXPECT_EQ(cmds[0].params[0].free_symbols().size(), 1u);
    EXPECT_EQ(cmds[1].params[0].free_symbols().size(), 1u);
}

TEST(Identities, SameSymbolRotationsStillMerge) {
    // 2·a stays in linear form — one symbol, still differentiable.
    std::vector<Command> cmds = {
        Command(GateType::Ry, 0, Param::symbol("a")),
        Command(GateType::Ry, 0, Param::symbol("a")),
    };
    eliminate_identities(cmds);
    ASSERT_EQ(cmds.size(), 1u);
    ASSERT_EQ(cmds[0].params[0].free_symbols().size(), 1u);
    std::unordered_map<std::string, double> at_one{{"a", 1.0}};
    EXPECT_DOUBLE_EQ(cmds[0].params[0].evaluate(at_one), 2.0);
}

TEST(Identities, SymbolPlusConcreteRotationsStillMerge) {
    // α·sym + β is the supported linear form.
    std::vector<Command> cmds = {
        Command(GateType::Ry, 0, Param::symbol("a")),
        Command(GateType::Ry, 0, Param(0.25)),
    };
    eliminate_identities(cmds);
    ASSERT_EQ(cmds.size(), 1u);
    ASSERT_EQ(cmds[0].params[0].free_symbols().size(), 1u);
    std::unordered_map<std::string, double> at_zero{{"a", 0.0}};
    EXPECT_DOUBLE_EQ(cmds[0].params[0].evaluate(at_zero), 0.25);
}

TEST(Identities, DistinctSymbolicGPhasesDoNotMerge) {
    auto gphase = [](const char* sym) {
        Command c;
        c.gate = GateType::GPhase;
        c.params.push_back(Param::symbol(sym));
        return c;
    };
    std::vector<Command> cmds = {gphase("a"), gphase("b")};
    auto eliminated = eliminate_identities(cmds);
    EXPECT_EQ(eliminated, 0u);
    EXPECT_EQ(cmds.size(), 2u);
}


// ── §9 via Command::dagger(): named-inverse and parametric cancellation ──
//
// are_inverse_pair used to spell out its own pair list (S/Sdg, T/Tdg only) and
// excluded parametric gates entirely.  It now asks Command::dagger(), so §9 has
// one implementation.  These pin the pairs that list was missing, and the
// negatives that a gate-type-only match would wrongly collapse.

TEST(Identities, NamedInversePairsFromTheContract) {
    struct Case { GateType a, b; bool two_qubit; };
    const Case cases[] = {
        {GateType::SX,    GateType::SXdg,    false},
        {GateType::SXdg,  GateType::SX,      false},
        {GateType::CS,    GateType::CSdg,    true},
        {GateType::CSdg,  GateType::CS,      true},
        {GateType::CSX,   GateType::CSXdg,   true},
        {GateType::CSXdg, GateType::CSX,     true},
        {GateType::iSWAP, GateType::iSWAPdg, true},
    };
    for (const auto& c : cases) {
        std::vector<Command> cmds =
            c.two_qubit ? std::vector<Command>{Command(c.a, 0, 1), Command(c.b, 0, 1)}
                        : std::vector<Command>{Command(c.a, 0),    Command(c.b, 0)};
        EXPECT_EQ(eliminate_identities(cmds), 2u) << gate_name(c.a) << "/" << gate_name(c.b);
        EXPECT_TRUE(cmds.empty())                 << gate_name(c.a) << "/" << gate_name(c.b);
    }
}

TEST(Identities, EcrIsSelfInverseButArgumentOrdered) {
    // §3.2: ECR^2 = I, but ECR is *not* argument-symmetric.
    std::vector<Command> same = {Command(GateType::ECR, 0, 1), Command(GateType::ECR, 0, 1)};
    EXPECT_EQ(eliminate_identities(same), 2u);
    EXPECT_TRUE(same.empty());

    std::vector<Command> swapped = {Command(GateType::ECR, 0, 1), Command(GateType::ECR, 1, 0)};
    EXPECT_EQ(eliminate_identities(swapped), 0u);
    EXPECT_EQ(swapped.size(), 2u);
}

TEST(Identities, SymmetricGateCancelsWithSwappedArguments) {
    // §3.2: iSWAP is argument-symmetric, so the swapped dagger still cancels.
    std::vector<Command> cmds = {Command(GateType::iSWAP, 0, 1),
                                 Command(GateType::iSWAPdg, 1, 0)};
    EXPECT_EQ(eliminate_identities(cmds), 2u);
    EXPECT_TRUE(cmds.empty());
}

TEST(Identities, ParametricInverseCancels) {
    // The angle is negated by Command::dagger(), concrete or symbolic.
    std::vector<Command> concrete = {Command(GateType::CP, 0, 1, {Param(0.3)}),
                                     Command(GateType::CP, 0, 1, {Param(-0.3)})};
    EXPECT_EQ(eliminate_identities(concrete), 2u);
    EXPECT_TRUE(concrete.empty());

    const Param t = Param::symbol("t");
    std::vector<Command> symbolic = {Command(GateType::Rx, 0, {t}),
                                     Command(GateType::Rx, 0, {-t})};
    EXPECT_EQ(eliminate_identities(symbolic), 2u)
        << "Rx(t)Rx(-t) used to merge into a dead Rx(0*t) that is_zero_angle refused";
    EXPECT_TRUE(symbolic.empty());
}

TEST(Identities, SymmetricParametricStillComparesItsAngle) {
    // RZZ is argument-symmetric AND parametric: matching on gate type and
    // qubit set alone would collapse two unrelated angles.
    std::vector<Command> inverse = {Command(GateType::RZZ, 0, 1, {Param(0.3)}),
                                    Command(GateType::RZZ, 1, 0, {Param(-0.3)})};
    EXPECT_EQ(eliminate_identities(inverse), 2u);

    // Not an inverse pair — but RZZ is additive too (§9, §16), so the two now
    // FOLD into RZZ(0,1, 1.0) rather than being left alone.  The invariant this
    // test guards is that the angle is compared, never ignored: matching on
    // gate and qubit set alone would cancel them to nothing (size 0), whereas
    // the correct fold keeps one gate carrying the summed angle.
    std::vector<Command> unrelated = {Command(GateType::RZZ, 0, 1, {Param(0.3)}),
                                      Command(GateType::RZZ, 1, 0, {Param(0.7)})};
    EXPECT_EQ(eliminate_identities(unrelated), 1u);
    ASSERT_EQ(unrelated.size(), 1u);
    EXPECT_EQ(unrelated[0].gate, GateType::RZZ);
    EXPECT_NEAR(unrelated[0].params[0].value(), 1.0, 1e-12);
}

TEST(Identities, RegionMarkersNeverCancel) {
    // §9 leaves Branch* "unchanged as commands", so a.dagger() == a for them.
    // A *nested* region puts two identical markers next to each other, which is
    // where that bites: cancelling them would corrupt the region grammar.  The
    // gate_is_physical guard is what prevents it — dropping that guard makes
    // this test, and FrozenOracleDominance, fail.
    std::vector<Command> cmds = {
        qarpx::test::make_branch_begin({0}, {true}),
        qarpx::test::make_branch_begin({1}, {true}),   // adjacent BranchBegin pair
        Command(GateType::X, 0),
        qarpx::test::make_marker(GateType::BranchEnd),
        qarpx::test::make_marker(GateType::BranchEnd),  // adjacent BranchEnd pair
    };
    const auto before = cmds.size();
    EXPECT_EQ(eliminate_identities(cmds), 0u);
    EXPECT_EQ(cmds.size(), before);
}

// ── §16 O1 merge over the whole §9 single-parameter family ────────────────
//
// can_merge_rotations was hard-limited to Rx/Ry/Rz.  The family closed under
// angle addition is exactly the §9 "negate the single parameter" row, so every
// member folds the same way; U/CU do not belong and must not fold.

TEST(Identities, AdditiveFamilyFoldsToOneGate) {
    struct Case { GateType g; bool two_qubit; };
    for (const Case c : {Case{GateType::P, false},   Case{GateType::CP, true},
                         Case{GateType::CRz, true},  Case{GateType::CRx, true},
                         Case{GateType::RZZ, true},  Case{GateType::RXX, true}}) {
        std::vector<Command> cmds =
            c.two_qubit ? std::vector<Command>{Command(c.g, 0, 1, {Param(0.3)}),
                                               Command(c.g, 0, 1, {Param(0.1)})}
                        : std::vector<Command>{Command(c.g, 0, {Param(0.3)}),
                                               Command(c.g, 0, {Param(0.1)})};
        EXPECT_EQ(eliminate_identities(cmds), 1u) << gate_name(c.g);
        ASSERT_EQ(cmds.size(), 1u) << gate_name(c.g);
        EXPECT_EQ(cmds[0].gate, c.g);
        EXPECT_NEAR(cmds[0].params[0].value(), 0.4, 1e-12) << gate_name(c.g);
    }
}

TEST(Identities, SymbolicSameSymbolFolds) {
    // CRz(t) · CRz(0.5 t + 0.2) -> CRz(1.5 t + 0.2): one symbol, stays linear.
    const Param t = Param::symbol("t");
    std::vector<Command> cmds = {Command(GateType::CRz, 0, 1, {t}),
                                 Command(GateType::CRz, 0, 1, {Param::linear(0.5, "t", 0.2)})};
    EXPECT_EQ(eliminate_identities(cmds), 1u);
    ASSERT_EQ(cmds.size(), 1u);
    EXPECT_EQ(cmds[0].params[0], Param::linear(1.5, "t", 0.2));
}

TEST(Identities, TwoSymbolsRefuseToFold) {
    // §17: a fold that would leave a compound two-symbol angle is refused.
    std::vector<Command> cmds = {Command(GateType::CP, 0, 1, {Param::symbol("a")}),
                                 Command(GateType::CP, 0, 1, {Param::symbol("b")})};
    EXPECT_EQ(eliminate_identities(cmds), 0u);
    EXPECT_EQ(cmds.size(), 2u);
}

TEST(Identities, SymmetricAdditiveFoldsAcrossSwappedArguments) {
    std::vector<Command> cmds = {Command(GateType::RZZ, 0, 1, {Param(0.3)}),
                                 Command(GateType::RZZ, 1, 0, {Param(0.1)})};
    EXPECT_EQ(eliminate_identities(cmds), 1u);
    ASSERT_EQ(cmds.size(), 1u);
    EXPECT_NEAR(cmds[0].params[0].value(), 0.4, 1e-12);
}

TEST(Identities, LoneZeroAngleOfTheFamilyIsDropped) {
    std::vector<Command> cmds = {Command(GateType::CP, 0, 1, {Param(0.0)}),
                                 Command(GateType::H, 2)};
    EXPECT_EQ(eliminate_identities(cmds), 1u);
    ASSERT_EQ(cmds.size(), 1u);
    EXPECT_EQ(cmds[0].gate, GateType::H);
}

TEST(Identities, UDoesNotFold) {
    // U(a)·U(b) is not U(a+b): the family is not additive, so no merge.
    auto make_u = [](double th, double ph, double la) {
        Command u(GateType::U, 0);
        u.params.push_back(Param(th)); u.params.push_back(Param(ph)); u.params.push_back(Param(la));
        return u;
    };
    std::vector<Command> cmds = {make_u(0.3, 0.2, 0.1), make_u(0.4, 0.5, 0.6)};
    EXPECT_EQ(eliminate_identities(cmds), 0u);
    EXPECT_EQ(cmds.size(), 2u);
}

