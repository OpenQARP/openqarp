#include <gtest/gtest.h>
#include "qarpx/qarpx.h"

using namespace qarpx;

TEST(QIREmitter, BellCircuit) {
    std::vector<Command> cmds = {
        Command(GateType::H, 0),
        Command(GateType::CX, 0, 1),
    };

    QIREmitter emitter;
    EXPECT_FALSE(emitter.validate(cmds).has_value());

    std::string qir = emitter.emit(cmds, 2, "bell");

    // Check key parts of the output
    EXPECT_NE(qir.find("define void @bell()"), std::string::npos);
    EXPECT_NE(qir.find("__quantum__qis__h__body"), std::string::npos);
    EXPECT_NE(qir.find("__quantum__qis__cnot__body"), std::string::npos);
    EXPECT_NE(qir.find("required_num_qubits\"=\"2\""), std::string::npos);
}

TEST(QIREmitter, ParametricGateWithConcreteValue) {
    std::vector<Command> cmds = {
        Command(GateType::Rx, 0, Param(1.5707963)),
    };

    QIREmitter emitter;
    EXPECT_FALSE(emitter.validate(cmds).has_value());

    std::string qir = emitter.emit(cmds, 1);
    EXPECT_NE(qir.find("__quantum__qis__rx__body"), std::string::npos);
    // The value is present but formatting may vary — check for rx call
    EXPECT_NE(qir.find("rx__body(double"), std::string::npos);
}

TEST(QIREmitter, SymbolicParamRejectsValidation) {
    std::vector<Command> cmds = {
        Command(GateType::Rx, 0, Param::symbol("theta")),
    };

    QIREmitter emitter;
    EXPECT_TRUE(emitter.validate(cmds).has_value());
}

TEST(QIREmitter, MeasureGate) {
    Command meas;
    meas.gate = GateType::Measure;
    meas.qubits = {0};
    meas.cbits = {0};

    std::vector<Command> cmds = {
        Command(GateType::H, 0),
        meas,
    };

    QIREmitter emitter;
    std::string qir = emitter.emit(cmds, 1);
    EXPECT_NE(qir.find("__quantum__qis__mz__body"), std::string::npos);
    EXPECT_NE(qir.find("required_num_results\"=\"1\""), std::string::npos);
}

TEST(QIREmitter, FullPipeline) {
    // Build -> Flatten -> Transpile -> Optimize -> Emit QIR
    SimpleBlock block(2, "test");
    block.h(0).cx(0, 1).rz(1, Param(0.5));
    block.build();

    auto flat = block.flatten();

    Transpiler transpiler(qulacs_gateset());
    auto transpiled = transpiler.transpile_and_optimize(flat);

    QIREmitter emitter;
    EXPECT_FALSE(emitter.validate(transpiled).has_value());
    std::string qir = emitter.emit(transpiled, 2, "full_pipeline");
    EXPECT_NE(qir.find("define void @full_pipeline()"), std::string::npos);
}

// ── Reset + branch markers ────────────────────────────────────────────────────

TEST(QIREmitter, EmitsResetIntrinsic) {
    Command reset;  reset.gate = GateType::Reset;  reset.qubits.push_back(0u);
    QIREmitter em;
    auto s = em.emit({reset}, 1);
    EXPECT_NE(s.find("__quantum__qis__reset__body"), std::string::npos);
}

TEST(QIREmitter, EmitsConditionalAsLLVMBranch) {
    Command begin;
    begin.gate = GateType::BranchBegin;
    begin.condition_bits.push_back(0u);
    begin.condition_values.push_back(true);
    Command end;  end.gate = GateType::BranchEnd;
    Command meas; meas.gate = GateType::Measure;
    meas.qubits.push_back(0u); meas.cbits.push_back(0u);

    std::vector<qarpx::Command> cmds = {
        qarpx::Command(qarpx::GateType::H, 0u),
        meas, begin,
        qarpx::Command(qarpx::GateType::X, 0u),
        end,
    };
    QIREmitter em;
    auto s = em.emit(cmds, 1);
    EXPECT_NE(s.find("__quantum__rt__read_result__body"), std::string::npos);
    EXPECT_NE(s.find("br i1"), std::string::npos);
    EXPECT_NE(s.find("then0:"), std::string::npos);
    EXPECT_NE(s.find("end0:"), std::string::npos);
}

TEST(QIREmitter, EmitsConditionalWithElseAndMultiBit) {
    Command begin;
    begin.gate = GateType::BranchBegin;
    begin.condition_bits.push_back(0u);
    begin.condition_bits.push_back(1u);
    begin.condition_values.push_back(true);
    begin.condition_values.push_back(false);
    Command br_else;  br_else.gate = GateType::BranchElse;
    Command br_end;   br_end.gate  = GateType::BranchEnd;

    std::vector<qarpx::Command> cmds = {
        begin,
        qarpx::Command(qarpx::GateType::X, 0u),
        br_else,
        qarpx::Command(qarpx::GateType::Z, 0u),
        br_end,
    };
    QIREmitter em;
    auto s = em.emit(cmds, 2);
    EXPECT_NE(s.find("xor i1"), std::string::npos)        // cbit value=0 → negate
        << s;
    EXPECT_NE(s.find("and i1"), std::string::npos)        // multi-bit AND
        << s;
    EXPECT_NE(s.find("else0:"), std::string::npos);
    EXPECT_NE(s.find("then0:"), std::string::npos);
    EXPECT_NE(s.find("end0:"),  std::string::npos);
}
