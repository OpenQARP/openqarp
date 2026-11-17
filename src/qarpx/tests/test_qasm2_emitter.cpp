// ── OpenQASM 2 emitter tests ──
//
// String-shape assertions: each test emits a small program and checks the
// expected qelib1 name, syntactic form, or register layout appears.  Semantics
// (does the emitted `gate` definition evaluate to the gate qarp means?) are a
// Python-side concern — qiskit.qasm2.loads is the independent oracle, in
// tests/test_emit/test_qasm2_external_parse.py.

#include <gtest/gtest.h>
#include "qarpx/qarpx.h"

#include <string>
#include <vector>

using namespace qarpx;

namespace {

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

std::string emit(const std::vector<Command>& cmds, uint32_t n_qubits) {
    return QASM2Emitter().emit(cmds, n_qubits);
}

/// A BranchBegin guarding one command on a single cbit.
std::vector<Command> guarded(uint32_t cbit, bool value, Command body) {
    Command begin;
    begin.gate = GateType::BranchBegin;
    begin.condition_bits.push_back(cbit);
    begin.condition_values.push_back(value);
    Command end;
    end.gate = GateType::BranchEnd;
    return {begin, std::move(body), end};
}

}  // namespace

// ── Header & registers ───────────────────────────────────────────────────────

TEST(QASM2, EmitsHeaderAndQelib1) {
    const std::string s = emit({Command(GateType::H, 0u)}, 1);
    EXPECT_TRUE(contains(s, "OPENQASM 2.0;"));
    EXPECT_TRUE(contains(s, "include \"qelib1.inc\";"));
    EXPECT_TRUE(contains(s, "qreg q[1];"));
}

TEST(QASM2, EmitsSingleClassicalRegisterWhenNoConditionals) {
    Command meas(GateType::Measure, 0u);
    meas.cbits = {3};
    const std::string s = emit({Command(GateType::H, 0u), meas}, 1);
    // Register sized to fit the largest cbit index (3 → creg c[4];)
    EXPECT_TRUE(contains(s, "creg c[4];"));
    EXPECT_TRUE(contains(s, "measure q[0] -> c[3];"));
}

TEST(QASM2, NoClassicalRegisterWhenNoMeasure) {
    const std::string s = emit({Command(GateType::H, 0u)}, 1);
    EXPECT_FALSE(contains(s, "creg"));
}

TEST(QASM2, MeasureWithoutCbitWidensTheLocalRegister) {
    // Emitter-local concession (§8/§12.2): a Measure carrying no cbit has no
    // classical target to name, so it goes to c[qubit] and the declaration is
    // widened to keep the program valid.  Not part of the register contract.
    Command meas(GateType::Measure, 2u);
    const std::string s = emit({meas}, 3);
    EXPECT_TRUE(contains(s, "creg c[3];"));
    EXPECT_TRUE(contains(s, "measure q[2] -> c[2];"));
}

// ── 1Q gates ─────────────────────────────────────────────────────────────────

TEST(QASM2, SingleQubitGateNames) {
    const std::vector<Command> cmds = {
        Command(GateType::X,   0u),
        Command(GateType::Y,   0u),
        Command(GateType::Z,   0u),
        Command(GateType::H,   0u),
        Command(GateType::S,   0u),
        Command(GateType::Sdg, 0u),
        Command(GateType::T,   0u),
        Command(GateType::Tdg, 0u),
    };
    const std::string s = emit(cmds, 1);
    for (const char* g : {"x q[0];", "y q[0];", "z q[0];", "h q[0];",
                          "s q[0];", "sdg q[0];", "t q[0];", "tdg q[0];"}) {
        EXPECT_TRUE(contains(s, g)) << g;
    }
}

TEST(QASM2, PhaseGateIsU1AndGeneralGateIsU3) {
    // `p` and `U` are OpenQASM 3 spellings; qelib1 calls them u1 and u3.
    const std::string s = emit({
        Command(GateType::P, 0u, Param(0.25)),
        Command(GateType::U, {0u}, {Param(0.7), Param(1.1), Param(-0.4)}),
    }, 1);
    EXPECT_TRUE(contains(s, "u1(0.25) q[0];"));
    EXPECT_TRUE(contains(s, "u3(0.7, 1.1, -0.4) q[0];"));
    EXPECT_FALSE(contains(s, "p("));
}

TEST(QASM2, RotationNames) {
    const std::string s = emit({
        Command(GateType::Rx, 0u, Param(0.5)),
        Command(GateType::Ry, 0u, Param(0.5)),
        Command(GateType::Rz, 0u, Param(0.5)),
    }, 1);
    EXPECT_TRUE(contains(s, "rx(0.5) q[0];"));
    EXPECT_TRUE(contains(s, "ry(0.5) q[0];"));
    EXPECT_TRUE(contains(s, "rz(0.5) q[0];"));
}

// ── 2Q / 3Q gates ────────────────────────────────────────────────────────────

TEST(QASM2, TwoQubitGateNames) {
    const std::string s = emit({
        Command(GateType::CX,   0u, 1u),
        Command(GateType::CY,   0u, 1u),
        Command(GateType::CZ,   0u, 1u),
        Command(GateType::SWAP, 0u, 1u),
    }, 2);
    EXPECT_TRUE(contains(s, "cx q[0], q[1];"));
    EXPECT_TRUE(contains(s, "cy q[0], q[1];"));
    EXPECT_TRUE(contains(s, "cz q[0], q[1];"));
    EXPECT_TRUE(contains(s, "swap q[0], q[1];"));
}

TEST(QASM2, ControlledPhaseIsCu1) {
    const std::string s = emit({Command(GateType::CP, 0u, 1u, Param(0.5))}, 2);
    EXPECT_TRUE(contains(s, "cu1(0.5) q[0], q[1];"));
    EXPECT_FALSE(contains(s, "cp("));
}

TEST(QASM2, ThreeQubitGateNames) {
    const std::string s = emit({
        Command(GateType::CCX,   {0u, 1u, 2u}),
        Command(GateType::CSWAP, {0u, 1u, 2u}),
    }, 3);
    EXPECT_TRUE(contains(s, "ccx q[0], q[1], q[2];"));
    EXPECT_TRUE(contains(s, "cswap q[0], q[1], q[2];"));
}

TEST(QASM2, InverseISwapHasARealDefinitionNotAModifier) {
    // OpenQASM 2 has no `inv @`, so iSWAPdg needs its own gate body.
    const std::string s = emit({Command(GateType::iSWAPdg, 0u, 1u)}, 2);
    EXPECT_TRUE(contains(s, "gate iswapdg a, b {"));
    EXPECT_TRUE(contains(s, "iswapdg q[0], q[1];"));
    EXPECT_FALSE(contains(s, "inv @"));
}

// ── Emitted gate definitions ─────────────────────────────────────────────────

TEST(QASM2, NoDefinitionsWhenProgramUsesOnlySpecQelib1) {
    const std::string s = emit({
        Command(GateType::H, 0u),
        Command(GateType::CX, 0u, 1u),
        Command(GateType::CRz, 0u, 1u, Param(0.5)),
    }, 2);
    EXPECT_FALSE(contains(s, "gate "));
}

TEST(QASM2, SwapCarriesADefinitionBecauseSpecQelib1LacksIt) {
    // swap/cswap/crx/cry/rxx/rzz are later qiskit additions, not language.
    const std::string s = emit({Command(GateType::SWAP, 0u, 1u)}, 2);
    EXPECT_TRUE(contains(s, "gate swap a, b { cx a, b; cx b, a; cx a, b; }"));
}

TEST(QASM2, RzzDefinitionUsesRzNotU1) {
    // `cx; u1(θ); cx` is RZZ only up to e^{-iθ/2} — §11 demands exactness.
    const std::string s = emit({Command(GateType::RZZ, 0u, 1u, Param(0.4))}, 2);
    EXPECT_TRUE(contains(s, "gate rzz(theta) a, b { cx a, b; rz(theta) b; cx a, b; }"));
}

TEST(QASM2, RxxAndRyyPullInTheirRzzDependency) {
    const std::string rxx = emit({Command(GateType::RXX, 0u, 1u, Param(0.4))}, 2);
    EXPECT_TRUE(contains(rxx, "gate rzz(theta)"));
    EXPECT_LT(rxx.find("gate rzz(theta)"), rxx.find("gate rxx(theta)"));

    const std::string ryy = emit({Command(GateType::RYY, 0u, 1u, Param(0.4))}, 2);
    EXPECT_TRUE(contains(ryy, "gate rzz(theta)"));
    EXPECT_LT(ryy.find("gate rzz(theta)"), ryy.find("gate ryy(theta)"));
}

TEST(QASM2, EcrPullsInItsRzxDependency) {
    const std::string s = emit({Command(GateType::ECR, 0u, 1u)}, 2);
    EXPECT_TRUE(contains(s, "gate rzx(theta)"));
    EXPECT_LT(s.find("gate rzx(theta)"), s.find("gate ecr a, b"));
}

// id and ch are in the spec qelib1.inc: direct calls, no definition.
TEST(QASM2, IdAndChAreSpecQelib1Names) {
    const std::string s = emit({Command(GateType::Id, 0u), Command(GateType::CH, 0u, 1u)}, 2);
    EXPECT_TRUE(contains(s, "id q[0];"));
    EXPECT_TRUE(contains(s, "ch q[0], q[1];"));
    EXPECT_FALSE(contains(s, "gate "));
}

// sx is a later qiskit addition: emitted definition, phase-exact
// (H·P(±π/2)·H = e^{±iπ/4}·Rx(±π/2), §2.6).
TEST(QASM2, SqrtXPairCarryDefinitions) {
    const std::string s = emit({Command(GateType::SX, 0u), Command(GateType::SXdg, 0u)}, 1);
    EXPECT_TRUE(contains(s, "gate sx a { h a; s a; h a; }"));
    EXPECT_TRUE(contains(s, "gate sxdg a { h a; sdg a; h a; }"));
    EXPECT_TRUE(contains(s, "sx q[0];"));
    EXPECT_TRUE(contains(s, "sxdg q[0];"));
}

TEST(QASM2, ControlledCliffordSinglesCarryDefinitions) {
    const std::vector<Command> cmds = {
        Command(GateType::CS,    0u, 1u),
        Command(GateType::CSdg,  0u, 1u),
        Command(GateType::CSX,   0u, 1u),
        Command(GateType::CSXdg, 0u, 1u),
    };
    const std::string s = emit(cmds, 2);
    EXPECT_TRUE(contains(s, "gate cs a, b { cu1(pi/2) a, b; }"));
    EXPECT_TRUE(contains(s, "gate csdg a, b { cu1(-pi/2) a, b; }"));
    EXPECT_TRUE(contains(s, "gate csx a, b { u1(pi/4) a; crx(pi/2) a, b; }"));
    EXPECT_TRUE(contains(s, "gate csxdg a, b { u1(-pi/4) a; crx(-pi/2) a, b; }"));
    // csx/csxdg bodies call crx, so its definition must precede them.
    EXPECT_TRUE(contains(s, "gate crx(theta) a, b"));
    EXPECT_LT(s.find("gate crx(theta)"), s.find("gate csx a, b"));
    EXPECT_TRUE(contains(s, "cs q[0], q[1];"));
    EXPECT_TRUE(contains(s, "csxdg q[0], q[1];"));
}

TEST(QASM2, CuIsDefinedOverU1AndCu3) {
    const std::string s = emit({
        Command(GateType::CU, {0u, 1u},
                {Param(0.7), Param(1.1), Param(-0.4), Param(0.25)}),
    }, 2);
    EXPECT_TRUE(contains(s, "gate cu(theta,phi,lam,gam) c, t {"));
    EXPECT_TRUE(contains(s, "cu(0.7, 1.1, -0.4, 0.25) q[0], q[1];"));
}

// ── Conditionals ─────────────────────────────────────────────────────────────

TEST(QASM2, ConditionalsSwitchTheRegisterLayoutToPerBit) {
    // OpenQASM 2's `if` compares a whole creg to an integer, so a single-bit
    // condition needs that bit to be its own register.
    auto cmds = guarded(0, true, Command(GateType::X, 1u));
    const std::string s = emit(cmds, 2);
    EXPECT_TRUE(contains(s, "creg c0[1];"));
    EXPECT_FALSE(contains(s, "creg c["));
    EXPECT_TRUE(contains(s, "if (c0 == 1) x q[1];"));
}

TEST(QASM2, ElseIsRenderedByNegatingTheComparedValue) {
    Command begin;
    begin.gate = GateType::BranchBegin;
    begin.condition_bits.push_back(0);
    begin.condition_values.push_back(true);
    Command els;
    els.gate = GateType::BranchElse;
    Command end;
    end.gate = GateType::BranchEnd;
    const std::string s = emit(
        {begin, Command(GateType::X, 1u), els, Command(GateType::H, 1u), end}, 2);
    EXPECT_TRUE(contains(s, "if (c0 == 1) x q[1];"));
    EXPECT_TRUE(contains(s, "if (c0 == 0) h q[1];"));
    EXPECT_FALSE(contains(s, "else"));
}

TEST(QASM2, EveryCommandInABranchBodyCarriesItsOwnGuard) {
    // `if` governs one statement — there is no braced block to share.
    Command begin;
    begin.gate = GateType::BranchBegin;
    begin.condition_bits.push_back(1);
    begin.condition_values.push_back(true);
    Command end;
    end.gate = GateType::BranchEnd;
    const std::string s = emit(
        {begin, Command(GateType::X, 0u), Command(GateType::H, 0u), end}, 1);
    EXPECT_TRUE(contains(s, "if (c1 == 1) x q[0];"));
    EXPECT_TRUE(contains(s, "if (c1 == 1) h q[0];"));
}

TEST(QASM2, ConditionBitsWidenTheRegisterEvenWithoutAMeasure) {
    // §8: cbit_register_width counts condition_bits too.
    auto cmds = guarded(2, true, Command(GateType::X, 0u));
    const std::string s = emit(cmds, 1);
    EXPECT_TRUE(contains(s, "creg c0[1];"));
    EXPECT_TRUE(contains(s, "creg c2[1];"));
    EXPECT_TRUE(contains(s, "if (c2 == 1) x q[0];"));
}

// ── Rejections ───────────────────────────────────────────────────────────────

TEST(QASM2, ValidateRejectsGPhase) {
    // Rejected, not dropped: a .qasm file is portable and may be controlled
    // downstream, where a lost global phase becomes a relative one (§12.2).
    const auto inc = QASM2Emitter().validate({Command(GateType::GPhase, {}, {Param(0.5)})});
    ASSERT_TRUE(inc.has_value());
    EXPECT_NE(inc->reason.find("GPhase"), std::string::npos);
}

TEST(QASM2, ValidateRejectsMcz) {
    const auto inc = QASM2Emitter().validate({Command(GateType::MCZ, {0u, 1u, 2u})});
    ASSERT_TRUE(inc.has_value());
    EXPECT_NE(inc->reason.find("MCZ"), std::string::npos);
}

TEST(QASM2, ValidateRejectsSymbolicParameters) {
    const auto inc = QASM2Emitter().validate(
        {Command(GateType::Rx, 0u, Param::symbol("theta"))});
    ASSERT_TRUE(inc.has_value());
    EXPECT_NE(inc->reason.find("symbolic"), std::string::npos);
}

TEST(QASM2, ValidateRejectsCustom) {
    Command custom;
    custom.gate = GateType::Custom;
    custom.qubits.push_back(0);
    EXPECT_TRUE(QASM2Emitter().validate({custom}).has_value());
}

TEST(QASM2, ValidateRejectsMultiBitConditions) {
    Command begin;
    begin.gate = GateType::BranchBegin;
    begin.condition_bits = {0, 1};
    begin.condition_values = {true, true};
    const auto inc = QASM2Emitter().validate({begin});
    ASSERT_TRUE(inc.has_value());
    EXPECT_NE(inc->reason.find("multi-bit"), std::string::npos);
}

TEST(QASM2, ValidateRejectsNestedConditionals) {
    // `if` governs a single quantum operation, which cannot itself be an `if`.
    // No EmitterCapabilities flag covers this, so QASM2Emitter::validate adds
    // the check — the pre-flight must give the same verdict emit() does.
    Command outer, inner, end;
    outer.gate = GateType::BranchBegin;
    outer.condition_bits.push_back(0);
    outer.condition_values.push_back(true);
    inner.gate = GateType::BranchBegin;
    inner.condition_bits.push_back(1);
    inner.condition_values.push_back(true);
    end.gate = GateType::BranchEnd;
    const std::vector<Command> cmds = {outer, inner, Command(GateType::X, 0u), end, end};
    const auto inc = QASM2Emitter().validate(cmds);
    ASSERT_TRUE(inc.has_value());
    EXPECT_NE(inc->reason.find("nested classical conditionals"), std::string::npos);
    EXPECT_THROW(emit(cmds, 1), capability_error);
}

TEST(QASM2, ValidateRejectsGuardedBarrier) {
    // `barrier` is not a `qop`, so OpenQASM 2's `if` cannot govern one.
    auto cmds = guarded(0, true, Command(GateType::Barrier, 0u));
    ASSERT_TRUE(QASM2Emitter().validate(cmds).has_value());
    EXPECT_THROW(emit(cmds, 1), capability_error);
}

TEST(QASM2, ValidateAcceptsAnUnnestedGuardedGate) {
    // The shape check must not fire on the ordinary single-frame case.
    auto cmds = guarded(0, true, Command(GateType::X, 0u));
    EXPECT_FALSE(QASM2Emitter().validate(cmds).has_value());
}

TEST(QASM2, ValidateAcceptsAllRepresentableGates) {
    const std::vector<Command> cmds = {
        Command(GateType::H, 0u),
        Command(GateType::CX, 0u, 1u),
        Command(GateType::Rz, 0u, Param(0.5)),
        Command(GateType::iSWAPdg, 0u, 1u),
        Command(GateType::CSWAP, {0u, 1u, 2u}),
    };
    EXPECT_FALSE(QASM2Emitter().validate(cmds).has_value());
}

// ── Special commands ─────────────────────────────────────────────────────────

TEST(QASM2, EmitsBarrierAndReset) {
    const std::string s = emit({
        Command(GateType::Barrier, {0u, 1u}),
        Command(GateType::Reset, 0u),
    }, 2);
    EXPECT_TRUE(contains(s, "barrier q[0], q[1];"));
    EXPECT_TRUE(contains(s, "reset q[0];"));
}

TEST(QASM2, MeasureUsesTheArrowForm) {
    Command meas(GateType::Measure, 1u);
    meas.cbits = {0};
    const std::string s = emit({meas}, 2);
    EXPECT_TRUE(contains(s, "measure q[1] -> c[0];"));
}

// ── target_name() ────────────────────────────────────────────────────────────

TEST(QASM2, TargetName) {
    EXPECT_EQ(QASM2Emitter().target_name(), "qasm2");
}

TEST(QASM2, GateSetIsNamedAndExcludesTheUnrepresentable) {
    const GateSet gs = QASM2Emitter().gate_set();
    EXPECT_EQ(gs.name, "qasm2");
    EXPECT_FALSE(gs.contains(GateType::GPhase));
    EXPECT_FALSE(gs.contains(GateType::MCZ));
    EXPECT_FALSE(gs.contains(GateType::Custom));
    EXPECT_TRUE(gs.contains(GateType::CU));
    EXPECT_TRUE(gs.contains(GateType::iSWAPdg));
}

// ── Full small-circuit smoke ─────────────────────────────────────────────────

TEST(QASM2, FullPipelineFromBlock) {
    SimpleBlock blk(2, "bell_with_measurement");
    blk.h(0).cx(0, 1).measure(0, 0).measure(1, 1);
    blk.build();

    auto cmds = blk.flatten();
    QASM2Emitter em;
    EXPECT_FALSE(em.validate(cmds).has_value());
    const std::string s = em.emit(cmds, 2, "bell_with_measurement");

    EXPECT_TRUE(contains(s, "// bell_with_measurement"));
    EXPECT_TRUE(contains(s, "qreg q[2];"));
    EXPECT_TRUE(contains(s, "creg c[2];"));
    EXPECT_TRUE(contains(s, "h q[0];"));
    EXPECT_TRUE(contains(s, "cx q[0], q[1];"));
    EXPECT_TRUE(contains(s, "measure q[0] -> c[0];"));
    EXPECT_TRUE(contains(s, "measure q[1] -> c[1];"));
}
