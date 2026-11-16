// ── OpenQASM 3 emitter tests ──
//
// String-shape assertions: each test emits a small program and checks that
// the expected stdgates name, syntactic form, or special-case modifier
// appears in the output.  Round-trip (emit → re-parse → unitary) is a
// Python-side concern (qiskit.qasm3.loads) and lives in
// tests/test_emit/test_qasm3_external_parse.py.

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
    return QASM3Emitter().emit(cmds, n_qubits);
}

}  // namespace

// ── Header & registers ───────────────────────────────────────────────────────

TEST(QASM3, EmitsHeaderAndStdgates) {
    const std::string s = emit({Command(GateType::H, 0u)}, 1);
    EXPECT_TRUE(contains(s, "OPENQASM 3.0;"));
    EXPECT_TRUE(contains(s, "include \"stdgates.inc\";"));
    EXPECT_TRUE(contains(s, "qubit[1] q;"));
}

TEST(QASM3, EmitsClassicalRegisterWhenMeasurePresent) {
    Command meas(GateType::Measure, 0u);
    meas.cbits = {3};
    const std::string s = emit({Command(GateType::H, 0u), meas}, 1);
    // Register sized to fit the largest cbit index (3 → bit[4] c;)
    EXPECT_TRUE(contains(s, "bit[4] c;"));
    EXPECT_TRUE(contains(s, "c[3] = measure q[0];"));
}

TEST(QASM3, NoClassicalRegisterWhenNoMeasure) {
    const std::string s = emit({Command(GateType::H, 0u)}, 1);
    // "bit[" alone matches "qubit[" too — anchor on the cbit-register form.
    EXPECT_FALSE(contains(s, "\nbit["));
    EXPECT_FALSE(contains(s, "] c;"));
}

// ── 1Q gates ─────────────────────────────────────────────────────────────────

TEST(QASM3, SingleQubitGateNames) {
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
    EXPECT_TRUE(contains(s, "x q[0];"));
    EXPECT_TRUE(contains(s, "y q[0];"));
    EXPECT_TRUE(contains(s, "z q[0];"));
    EXPECT_TRUE(contains(s, "h q[0];"));
    EXPECT_TRUE(contains(s, "s q[0];"));
    EXPECT_TRUE(contains(s, "sdg q[0];"));
    EXPECT_TRUE(contains(s, "t q[0];"));
    EXPECT_TRUE(contains(s, "tdg q[0];"));
}

TEST(QASM3, ParametricSingleQubitGates) {
    const std::vector<Command> cmds = {
        Command(GateType::Rx, 0u, Param(0.5)),
        Command(GateType::Ry, 0u, Param(0.7)),
        Command(GateType::Rz, 0u, Param(0.9)),
        Command(GateType::P,  0u, Param(1.1)),
    };
    const std::string s = emit(cmds, 1);
    EXPECT_TRUE(contains(s, "rx(0.5) q[0];"));
    EXPECT_TRUE(contains(s, "ry(0.7) q[0];"));
    EXPECT_TRUE(contains(s, "rz(0.9) q[0];"));
    EXPECT_TRUE(contains(s, "p(1.1) q[0];"));
}

// §2.6 / §12.1: sx and id are stdgates.inc symbols; sxdg is not and rides
// on the `inv @` modifier.
TEST(QASM3, SqrtXAndIdMapToStdgates) {
    const std::string s = emit({Command(GateType::SX, 0u), Command(GateType::Id, 0u)}, 1);
    EXPECT_TRUE(contains(s, "sx q[0];"));
    EXPECT_TRUE(contains(s, "id q[0];"));
    EXPECT_FALSE(contains(s, "gate "));
}

TEST(QASM3, SXdgUsesInvModifier) {
    const std::string s = emit({Command(GateType::SXdg, 0u)}, 1);
    EXPECT_TRUE(contains(s, "inv @ sx q[0];"));
}

// General U uses the language built-in (capital U), not stdgates.
TEST(QASM3, GeneralU_UsesLanguageBuiltin) {
    Command u;
    u.gate = GateType::U;
    u.qubits.push_back(0);
    u.params.push_back(Param(0.31));
    u.params.push_back(Param(0.42));
    u.params.push_back(Param(0.53));
    const std::string s = emit({u}, 1);
    EXPECT_TRUE(contains(s, "U(0.31, 0.42, 0.53) q[0];"));
}

// ── 2Q & 3Q gates ────────────────────────────────────────────────────────────

TEST(QASM3, TwoQubitGateNames) {
    const std::vector<Command> cmds = {
        Command(GateType::CX,   0u, 1u),
        Command(GateType::CY,   0u, 1u),
        Command(GateType::CZ,   0u, 1u),
        Command(GateType::SWAP, 0u, 1u),
    };
    const std::string s = emit(cmds, 2);
    EXPECT_TRUE(contains(s, "cx q[0], q[1];"));
    EXPECT_TRUE(contains(s, "cy q[0], q[1];"));
    EXPECT_TRUE(contains(s, "cz q[0], q[1];"));
    EXPECT_TRUE(contains(s, "swap q[0], q[1];"));
}

// ecr is not a stdgates.inc symbol: the call must ride on an emitted `gate`
// definition (via the rzx helper) or an external parser rejects the program.
TEST(QASM3, ECR_EmitsCallAndDefinition) {
    const std::string s = emit({Command(GateType::ECR, 0u, 1u)}, 2);
    EXPECT_TRUE(contains(s, "ecr q[0], q[1];"));
    EXPECT_TRUE(contains(s, "gate ecr a, b"));
    EXPECT_TRUE(contains(s, "gate rzx(theta) a, b"));
}

TEST(QASM3, iSWAP_EmitsCallAndDefinition) {
    const std::string s = emit({Command(GateType::iSWAP, 0u, 1u)}, 2);
    EXPECT_TRUE(contains(s, "iswap q[0], q[1];"));
    EXPECT_TRUE(contains(s, "gate iswap a, b"));
}

// iSWAPdg additionally has no gate name of its own — emit via the OpenQASM
// `inv @` modifier, which needs the iswap definition in scope.
TEST(QASM3, iSWAPdgUsesInvModifierAndPullsIswapDefinition) {
    const std::string s = emit({Command(GateType::iSWAPdg, 0u, 1u)}, 2);
    EXPECT_TRUE(contains(s, "inv @ iswap q[0], q[1];"));
    EXPECT_TRUE(contains(s, "gate iswap a, b"));
}

// §3.1 / §12.1: ch is a stdgates.inc symbol; the other controlled Clifford
// singles compose via `ctrl @` on their 1Q base, as stdgates.inc itself does
// for ch (`ctrl @ h`).
TEST(QASM3, CHMapsToStdgatesCh) {
    const std::string s = emit({Command(GateType::CH, 0u, 1u)}, 2);
    EXPECT_TRUE(contains(s, "ch q[0], q[1];"));
    EXPECT_FALSE(contains(s, "gate "));
}

TEST(QASM3, ControlledCliffordSinglesUseCtrlModifier) {
    const std::vector<Command> cmds = {
        Command(GateType::CS,    0u, 1u),
        Command(GateType::CSdg,  0u, 1u),
        Command(GateType::CSX,   0u, 1u),
        Command(GateType::CSXdg, 0u, 1u),
    };
    const std::string s = emit(cmds, 2);
    EXPECT_TRUE(contains(s, "ctrl @ s q[0], q[1];"));
    EXPECT_TRUE(contains(s, "ctrl @ sdg q[0], q[1];"));
    EXPECT_TRUE(contains(s, "ctrl @ sx q[0], q[1];"));
    EXPECT_TRUE(contains(s, "ctrl @ inv @ sx q[0], q[1];"));
    EXPECT_FALSE(contains(s, "gate "));
}

// Programs touching none of the defined gates must stay definition-free.
TEST(QASM3, NoGateDefinitionsWhenUnused) {
    const std::string s = emit(
        {Command(GateType::H, 0u), Command(GateType::CX, 0u, 1u)}, 2);
    EXPECT_FALSE(contains(s, "gate "));
}

TEST(QASM3, TwoQubitParametricGates) {
    const std::vector<Command> cmds = {
        Command(GateType::CRx, 0u, 1u, Param(0.5)),
        Command(GateType::CRy, 0u, 1u, Param(0.6)),
        Command(GateType::CRz, 0u, 1u, Param(0.7)),
        Command(GateType::CP,  0u, 1u, Param(0.8)),
    };
    const std::string s = emit(cmds, 2);
    EXPECT_TRUE(contains(s, "crx(0.5) q[0], q[1];"));
    EXPECT_TRUE(contains(s, "cry(0.6) q[0], q[1];"));
    EXPECT_TRUE(contains(s, "crz(0.7) q[0], q[1];"));
    EXPECT_TRUE(contains(s, "cp(0.8) q[0], q[1];"));
}

// rzz/rxx/ryy are not stdgates.inc symbols either — rxx and ryy define
// themselves through the shared rzz definition, mirroring §11.
TEST(QASM3, RZZ_RXX_RYY_EmitCallsAndDefinitions) {
    const std::vector<Command> cmds = {
        Command(GateType::RZZ, 0u, 1u, Param(0.7)),
        Command(GateType::RXX, 0u, 1u, Param(0.7)),
        Command(GateType::RYY, 0u, 1u, Param(0.7)),
    };
    const std::string s = emit(cmds, 2);
    EXPECT_TRUE(contains(s, "rzz(0.7) q[0], q[1];"));
    EXPECT_TRUE(contains(s, "rxx(0.7) q[0], q[1];"));
    EXPECT_TRUE(contains(s, "ryy(0.7) q[0], q[1];"));
    EXPECT_TRUE(contains(s, "gate rzz(theta) a, b"));
    EXPECT_TRUE(contains(s, "gate rxx(theta) a, b"));
    EXPECT_TRUE(contains(s, "gate ryy(theta) a, b"));
}

// CU has 4 params (θ, φ, λ, γ) per qarp_conventions.md §3.1.
TEST(QASM3, CU_EmitsAllFourParams) {
    Command cu;
    cu.gate = GateType::CU;
    cu.qubits.push_back(0);
    cu.qubits.push_back(1);
    cu.params.push_back(Param(0.31));
    cu.params.push_back(Param(0.42));
    cu.params.push_back(Param(0.53));
    cu.params.push_back(Param(0.64));
    const std::string s = emit({cu}, 2);
    EXPECT_TRUE(contains(s, "cu(0.31, 0.42, 0.53, 0.64) q[0], q[1];"));
}

TEST(QASM3, CCXandCSWAP) {
    Command ccx;
    ccx.gate = GateType::CCX;
    for (uint32_t q = 0; q < 3; ++q) ccx.qubits.push_back(q);

    Command cswap;
    cswap.gate = GateType::CSWAP;
    for (uint32_t q = 0; q < 3; ++q) cswap.qubits.push_back(q);

    const std::string s = emit({ccx, cswap}, 3);
    EXPECT_TRUE(contains(s, "ccx q[0], q[1], q[2];"));
    EXPECT_TRUE(contains(s, "cswap q[0], q[1], q[2];"));
}

// ── MCZ → ctrl(n) @ z ────────────────────────────────────────────────────────

TEST(QASM3, MCZ_3q_UsesCtrlModifier) {
    Command mcz;
    mcz.gate = GateType::MCZ;
    for (uint32_t q = 0; q < 3; ++q) mcz.qubits.push_back(q);
    const std::string s = emit({mcz}, 3);
    // 3 qubits = 2 controls + 1 target.
    EXPECT_TRUE(contains(s, "ctrl(2) @ z q[0], q[1], q[2];"));
}

TEST(QASM3, MCZ_4q_UsesCtrlModifier) {
    Command mcz;
    mcz.gate = GateType::MCZ;
    for (uint32_t q = 0; q < 4; ++q) mcz.qubits.push_back(q);
    const std::string s = emit({mcz}, 4);
    EXPECT_TRUE(contains(s, "ctrl(3) @ z q[0], q[1], q[2], q[3];"));
}

// ── GPhase, Barrier, Reset ───────────────────────────────────────────────────

TEST(QASM3, GPhase) {
    Command g;
    g.gate = GateType::GPhase;
    g.params.push_back(Param(0.5));
    const std::string s = emit({g}, 1);
    EXPECT_TRUE(contains(s, "gphase(0.5);"));
}

TEST(QASM3, Barrier) {
    Command b(GateType::Barrier, 0u);
    const std::string s = emit({b}, 1);
    EXPECT_TRUE(contains(s, "barrier q[0];"));
}

TEST(QASM3, Reset) {
    const std::string s = emit({Command(GateType::Reset, 0u)}, 1);
    EXPECT_TRUE(contains(s, "reset q[0];"));
}

// ── Symbolic params → input float[64] declarations ───────────────────────────

TEST(QASM3, SymbolicParamEmitsInputDecl) {
    Command rz;
    rz.gate = GateType::Rz;
    rz.qubits.push_back(0);
    rz.params.push_back(Param::symbol("theta"));
    const std::string s = emit({rz}, 1);
    EXPECT_TRUE(contains(s, "input float[64] theta;"));
    EXPECT_TRUE(contains(s, "rz(theta) q[0];"));
}

TEST(QASM3, MultipleSymbolicParamsAreSortedAndUnique) {
    Command a, b, c;
    a.gate = GateType::Rx; a.qubits.push_back(0); a.params.push_back(Param::symbol("phi"));
    b.gate = GateType::Ry; b.qubits.push_back(0); b.params.push_back(Param::symbol("theta"));
    c.gate = GateType::Rz; c.qubits.push_back(0); c.params.push_back(Param::symbol("theta"));   // duplicate
    const std::string s = emit({a, b, c}, 1);
    // Both unique symbols declared, no duplicate.
    EXPECT_TRUE(contains(s, "input float[64] phi;"));
    EXPECT_TRUE(contains(s, "input float[64] theta;"));
    // Determinism: phi (alpha-sorted before theta) comes first.
    EXPECT_LT(s.find("input float[64] phi;"), s.find("input float[64] theta;"));
    // No spurious duplicate.
    const std::string needle = "input float[64] theta;";
    const auto first = s.find(needle);
    const auto second = s.find(needle, first + 1);
    EXPECT_EQ(second, std::string::npos);
}

TEST(QASM3, LinearParamEmitsInlineArithmetic) {
    Command rx;
    rx.gate = GateType::Rx;
    rx.qubits.push_back(0);
    rx.params.push_back(Param::linear(2.0, "theta", 0.5));
    const std::string s = emit({rx}, 1);
    // Param::to_string() renders this as "2*theta + 0.5"; appears verbatim.
    EXPECT_TRUE(contains(s, "rx(2*theta + 0.5) q[0];"));
}

// ── validate() ───────────────────────────────────────────────────────────────

TEST(QASM3, ValidateRejectsCustom) {
    Command custom;
    custom.gate = GateType::Custom;
    custom.qubits.push_back(0);
    EXPECT_TRUE(QASM3Emitter().validate({custom}).has_value());
}

TEST(QASM3, ValidateAcceptsAllUnitaryGates) {
    const std::vector<Command> cmds = {
        Command(GateType::H, 0u),
        Command(GateType::CX, 0u, 1u),
        Command(GateType::Rz, 0u, Param(0.5)),
        Command(GateType::iSWAPdg, 0u, 1u),
    };
    EXPECT_FALSE(QASM3Emitter().validate(cmds).has_value());
}

// ── target_name() ────────────────────────────────────────────────────────────

TEST(QASM3, TargetName) {
    EXPECT_EQ(QASM3Emitter().target_name(), "qasm3");
}

// ── Full small-circuit smoke ─────────────────────────────────────────────────

TEST(QASM3, FullPipelineFromBlock) {
    // Build → flatten → emit (no transpilation/optimization needed; all gates
    // are already QASM3-emittable).
    SimpleBlock blk(2, "bell_with_measurement");
    blk.h(0).cx(0, 1).measure(0, 0).measure(1, 1);
    blk.build();

    auto cmds = blk.flatten();
    QASM3Emitter em;
    EXPECT_FALSE(em.validate(cmds).has_value());
    const std::string s = em.emit(cmds, 2, "bell_with_measurement");

    EXPECT_TRUE(contains(s, "// bell_with_measurement"));
    EXPECT_TRUE(contains(s, "qubit[2] q;"));
    EXPECT_TRUE(contains(s, "bit[2] c;"));
    EXPECT_TRUE(contains(s, "h q[0];"));
    EXPECT_TRUE(contains(s, "cx q[0], q[1];"));
    EXPECT_TRUE(contains(s, "c[0] = measure q[0];"));
    EXPECT_TRUE(contains(s, "c[1] = measure q[1];"));
}

// ── Reset and branch markers ──────────────────────────────────────────────────

TEST(QASM3, EmitsResetCall) {
    const std::string s = emit({
        Command(GateType::H, 0u),
        Command(GateType::Reset, SmallVector<uint32_t,2>{0u}),
    }, 1);
    EXPECT_TRUE(contains(s, "reset q[0];"));
}

TEST(QASM3, EmitsConditionalIfBlockSingleBit) {
    // Build a flat command list with branch markers manually so the test
    // exercises QASM3Emitter directly without depending on ConditionalBlock.
    Command begin;
    begin.gate = GateType::BranchBegin;
    begin.condition_bits.push_back(0u);
    begin.condition_values.push_back(true);
    Command end;
    end.gate = GateType::BranchEnd;

    Command meas;
    meas.gate = GateType::Measure;
    meas.qubits.push_back(0u);
    meas.cbits.push_back(0u);

    std::vector<Command> cmds = {
        Command(GateType::H, 0u),
        meas,
        begin,
        Command(GateType::X, 0u),
        end,
    };
    const std::string s = emit(cmds, 1);
    EXPECT_TRUE(contains(s, "bit[1] c;"));
    EXPECT_TRUE(contains(s, "c[0] = measure q[0];"));
    EXPECT_TRUE(contains(s, "if (c[0] == true) {"));
    EXPECT_TRUE(contains(s, "    x q[0];"));   // body indented one level deeper
    EXPECT_TRUE(contains(s, "  }"));
}

TEST(QASM3, EmitsConditionalWithElseAndAndedMultiBit) {
    Command begin;
    begin.gate = GateType::BranchBegin;
    begin.condition_bits.push_back(0u);
    begin.condition_bits.push_back(1u);
    begin.condition_values.push_back(true);
    begin.condition_values.push_back(false);
    Command br_else;  br_else.gate = GateType::BranchElse;
    Command br_end;   br_end.gate  = GateType::BranchEnd;

    std::vector<Command> cmds = {
        Command(GateType::H, 0u),
        Command(GateType::H, 1u),
        // Two measurements (so the cbit register is sized to 2).
        [&]() { Command m; m.gate = GateType::Measure;
            m.qubits.push_back(0u); m.cbits.push_back(0u); return m; }(),
        [&]() { Command m; m.gate = GateType::Measure;
            m.qubits.push_back(1u); m.cbits.push_back(1u); return m; }(),
        begin,
        Command(GateType::X, 0u),
        br_else,
        Command(GateType::Z, 0u),
        br_end,
    };
    const std::string s = emit(cmds, 2);
    EXPECT_TRUE(contains(s, "bit[2] c;"));
    EXPECT_TRUE(contains(s, "if (c[0] == true && c[1] == false) {"));
    EXPECT_TRUE(contains(s, "} else {"));
    EXPECT_TRUE(contains(s, "    x q[0];"));
    EXPECT_TRUE(contains(s, "    z q[0];"));
}
