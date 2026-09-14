#include <gtest/gtest.h>
#include "qarpx/qarpx.h"

using namespace qarpx;

// Test helper: allocate a block as an intrusive ref<Block> — block trees are
// reference counted through nanobind's intrusive ref<T>.
namespace {
template <class T, class... Args>
ref<Block> make_block(Args&&... args) {
    return ref<Block>(new T(std::forward<Args>(args)...));
}
}  // namespace

// ── SmallVector ──

TEST(SmallVector, InlineStorage) {
    SmallVector<uint32_t, 2> sv;
    sv.push_back(0);
    sv.push_back(1);
    EXPECT_EQ(sv.size(), 2u);
    EXPECT_EQ(sv[0], 0u);
    EXPECT_EQ(sv[1], 1u);
}

TEST(SmallVector, HeapFallback) {
    SmallVector<uint32_t, 2> sv;
    sv.push_back(0);
    sv.push_back(1);
    sv.push_back(2);  // triggers heap allocation
    EXPECT_EQ(sv.size(), 3u);
    EXPECT_EQ(sv[2], 2u);
}

TEST(SmallVector, InitializerList) {
    SmallVector<uint32_t, 2> sv = {10, 20, 30};
    EXPECT_EQ(sv.size(), 3u);
    EXPECT_EQ(sv[0], 10u);
    EXPECT_EQ(sv[2], 30u);
}

TEST(SmallVector, CopyAndMove) {
    SmallVector<uint32_t, 2> a = {1, 2};
    SmallVector<uint32_t, 2> b = a;  // copy
    EXPECT_EQ(b.size(), 2u);
    EXPECT_EQ(b[0], 1u);

    SmallVector<uint32_t, 2> c = std::move(a);  // move
    EXPECT_EQ(c.size(), 2u);
    EXPECT_EQ(c[0], 1u);
}

// ── Param ──

TEST(Param, Concrete) {
    Param p(3.14);
    EXPECT_FALSE(p.is_symbolic());
    EXPECT_DOUBLE_EQ(p.value(), 3.14);
}

TEST(Param, Symbol) {
    Param p = Param::symbol("theta");
    EXPECT_TRUE(p.is_symbolic());
    auto syms = p.free_symbols();
    ASSERT_EQ(syms.size(), 1u);
    EXPECT_EQ(syms[0], "theta");
}

TEST(Param, Substitute) {
    Param p = Param::symbol("theta");
    Param result = p.substitute({{"theta", 1.57}});
    EXPECT_FALSE(result.is_symbolic());
    EXPECT_DOUBLE_EQ(result.value(), 1.57);
}

TEST(Param, LinearExpression) {
    Param p = Param::linear(2.0, "t", 1.0);  // 2*t + 1
    Param result = p.substitute({{"t", 3.0}});
    EXPECT_DOUBLE_EQ(result.value(), 7.0);
}

TEST(Param, ConcreteWithCompoundClosure) {
    // A compound (closure-backed) Param next to a concrete one used to reach
    // the "both symbolic" merge and std::get an Expr from the concrete side.
    // decompose_cu → decompose_u builds exactly this: 0.0 + ((φ+λ)/2)·(−1).
    const Param comp = Param::symbol("a") + Param::symbol("b");
    const std::unordered_map<std::string, double> at{{"a", 1.0}, {"b", 3.0}};
    EXPECT_NEAR((Param(0.0) + comp).evaluate(at), 4.0, 1e-12);
    EXPECT_NEAR((comp + Param(0.5)).evaluate(at), 4.5, 1e-12);
    EXPECT_NEAR((comp - Param(1.0)).evaluate(at), 3.0, 1e-12);
    EXPECT_NEAR((Param(1.0) - comp).evaluate(at), -3.0, 1e-12);
}

TEST(Param, Arithmetic) {
    Param a(2.0);
    Param b(3.0);
    EXPECT_DOUBLE_EQ((a + b).value(), 5.0);
    EXPECT_DOUBLE_EQ((a - b).value(), -1.0);
    EXPECT_DOUBLE_EQ((a * b).value(), 6.0);
    EXPECT_DOUBLE_EQ((a / b).value(), 2.0 / 3.0);
    EXPECT_DOUBLE_EQ((-a).value(), -2.0);
}

TEST(Param, SymbolicArithmetic) {
    Param theta = Param::symbol("theta");
    Param half_theta = theta * Param(0.5);
    EXPECT_TRUE(half_theta.is_symbolic());

    Param result = half_theta.substitute({{"theta", 2.0}});
    EXPECT_DOUBLE_EQ(result.value(), 1.0);
}

TEST(Param, Negation) {
    Param theta = Param::symbol("theta");
    Param neg = -theta;
    Param result = neg.substitute({{"theta", 3.0}});
    EXPECT_DOUBLE_EQ(result.value(), -3.0);
}

// ── Command ──

TEST(Command, SingleQubitGate) {
    Command cmd(GateType::H, 0);
    EXPECT_EQ(cmd.gate, GateType::H);
    EXPECT_EQ(cmd.qubits.size(), 1u);
    EXPECT_EQ(cmd.qubits[0], 0u);
    EXPECT_FALSE(cmd.is_parametric());
}

TEST(Command, ParametricGate) {
    Command cmd(GateType::Rx, 0, Param::symbol("theta"));
    EXPECT_TRUE(cmd.is_parametric());
}

TEST(Command, DaggerSelfAdjoint) {
    Command h(GateType::H, 0);
    auto h_dag = h.dagger();
    EXPECT_EQ(h_dag.gate, GateType::H);  // H is self-adjoint
}

TEST(Command, DaggerNamedPair) {
    Command s(GateType::S, 0);
    auto sdg = s.dagger();
    EXPECT_EQ(sdg.gate, GateType::Sdg);
}

TEST(Command, DaggerParametric) {
    Command rx(GateType::Rx, 0, Param(1.57));
    auto rx_dag = rx.dagger();
    EXPECT_DOUBLE_EQ(rx_dag.params[0].value(), -1.57);
}

// ── Conventions §9: SX ↔ SXdg, CS ↔ CSdg, CSX ↔ CSXdg named-inverse pairs ──
TEST(Command, DaggerSqrtXAndControlledCliffordNamedPairs) {
    EXPECT_EQ(Command(GateType::SX,    0u).dagger().gate,     GateType::SXdg);
    EXPECT_EQ(Command(GateType::SXdg,  0u).dagger().gate,     GateType::SX);
    EXPECT_EQ(Command(GateType::CS,    0u, 1u).dagger().gate, GateType::CSdg);
    EXPECT_EQ(Command(GateType::CSdg,  0u, 1u).dagger().gate, GateType::CS);
    EXPECT_EQ(Command(GateType::CSX,   0u, 1u).dagger().gate, GateType::CSXdg);
    EXPECT_EQ(Command(GateType::CSXdg, 0u, 1u).dagger().gate, GateType::CSX);
    // Self-adjoint: unchanged, qubit order preserved.
    auto ch = Command(GateType::CH, 0u, 1u).dagger();
    EXPECT_EQ(ch.gate, GateType::CH);
    EXPECT_EQ(ch.qubits[0], 0u);
    EXPECT_EQ(ch.qubits[1], 1u);
    EXPECT_EQ(Command(GateType::Id, 0u).dagger().gate, GateType::Id);
}

TEST(GateHelpers, NewGateTypesReportArityNameAndParams) {
    for (GateType g : {GateType::SX, GateType::SXdg, GateType::Id}) {
        EXPECT_EQ(gate_num_qubits(g), 1u) << gate_name(g);
        EXPECT_EQ(gate_num_params(g), 0u) << gate_name(g);
        EXPECT_TRUE(gate_is_physical(g)) << gate_name(g);
    }
    for (GateType g : {GateType::CH, GateType::CS, GateType::CSdg, GateType::CSX, GateType::CSXdg}) {
        EXPECT_EQ(gate_num_qubits(g), 2u) << gate_name(g);
        EXPECT_EQ(gate_num_params(g), 0u) << gate_name(g);
        EXPECT_TRUE(gate_is_physical(g)) << gate_name(g);
    }
    EXPECT_EQ(gate_name(GateType::SX),    "SX");
    EXPECT_EQ(gate_name(GateType::SXdg),  "SXdg");
    EXPECT_EQ(gate_name(GateType::Id),    "Id");
    EXPECT_EQ(gate_name(GateType::CH),    "CH");
    EXPECT_EQ(gate_name(GateType::CS),    "CS");
    EXPECT_EQ(gate_name(GateType::CSdg),  "CSdg");
    EXPECT_EQ(gate_name(GateType::CSX),   "CSX");
    EXPECT_EQ(gate_name(GateType::CSXdg), "CSXdg");
    EXPECT_TRUE(gate_is_self_adjoint(GateType::Id));
    EXPECT_TRUE(gate_is_self_adjoint(GateType::CH));
    EXPECT_TRUE(gate_is_self_adjoint(GateType::ECR));   // §3.2/§9: ECR^2 = I
    EXPECT_FALSE(gate_is_self_adjoint(GateType::SX));
    EXPECT_FALSE(gate_is_self_adjoint(GateType::CS));
}

// ── Conventions §9: the single-parameter (additive) family ──
TEST(Gates, AdditiveFamilyMatchesTheContract) {
    for (auto g : {GateType::Rx, GateType::Ry, GateType::Rz, GateType::P,
                   GateType::CRx, GateType::CRy, GateType::CRz, GateType::CP,
                   GateType::RXX, GateType::RYY, GateType::RZZ, GateType::GPhase})
        EXPECT_TRUE(gate_is_additive_in_param(g)) << gate_name(g);
    // U/CU dagger by a (phi, lambda) swap, not a negation; the rest carry no angle.
    for (auto g : {GateType::U, GateType::CU, GateType::H, GateType::CX,
                   GateType::SX, GateType::iSWAP, GateType::ECR, GateType::Custom})
        EXPECT_FALSE(gate_is_additive_in_param(g)) << gate_name(g);
}

// ── Conventions §3.1/§3.2/§3.3: argument symmetry ──
// Mirrors the contract directly so the predicate cannot drift behind it the
// way the hand-rolled list inside are_inverse_pair did.
TEST(Gates, QubitSymmetryMatchesTheContract) {
    // §3.1 CZ; §3.2 SWAP / iSWAP / iSWAPdg; §3.3 the symmetric rotations.
    EXPECT_TRUE(gate_is_qubit_symmetric(GateType::CZ));
    EXPECT_TRUE(gate_is_qubit_symmetric(GateType::SWAP));
    EXPECT_TRUE(gate_is_qubit_symmetric(GateType::iSWAP));
    EXPECT_TRUE(gate_is_qubit_symmetric(GateType::iSWAPdg));
    EXPECT_TRUE(gate_is_qubit_symmetric(GateType::RXX));
    EXPECT_TRUE(gate_is_qubit_symmetric(GateType::RYY));
    EXPECT_TRUE(gate_is_qubit_symmetric(GateType::RZZ));
    // §3.2 says ECR is NOT argument-symmetric, though it is self-adjoint.
    EXPECT_FALSE(gate_is_qubit_symmetric(GateType::ECR));
    // Controlled gates carry a (control, target) structure.
    EXPECT_FALSE(gate_is_qubit_symmetric(GateType::CX));
    EXPECT_FALSE(gate_is_qubit_symmetric(GateType::CY));
    EXPECT_FALSE(gate_is_qubit_symmetric(GateType::CS));
    EXPECT_FALSE(gate_is_qubit_symmetric(GateType::CSX));
}

// ── Conventions §8: iSWAP ↔ iSWAPdg named-inverse pair ──
TEST(Command, DaggerISwapNamedPair) {
    Command iswap(GateType::iSWAP, 0, 1);
    auto iswap_dag = iswap.dagger();
    EXPECT_EQ(iswap_dag.gate, GateType::iSWAPdg);
    EXPECT_EQ(iswap_dag.qubits[0], 0u);
    EXPECT_EQ(iswap_dag.qubits[1], 1u);

    auto iswap_again = iswap_dag.dagger();
    EXPECT_EQ(iswap_again.gate, GateType::iSWAP);
}

// ── Conventions §8: U(θ, φ, λ)† = U(-θ, -λ, -φ)  (note (φ, λ) swap) ──
TEST(Command, DaggerUSwapsPhiLambda) {
    Command u;
    u.gate = GateType::U;
    u.qubits.push_back(0);
    u.params.push_back(Param(0.3));   // θ
    u.params.push_back(Param(0.4));   // φ
    u.params.push_back(Param(0.5));   // λ

    auto udg = u.dagger();
    EXPECT_EQ(udg.gate, GateType::U);
    EXPECT_DOUBLE_EQ(udg.params[0].value(), -0.3);  // -θ
    EXPECT_DOUBLE_EQ(udg.params[1].value(), -0.5);  // -λ at index 1
    EXPECT_DOUBLE_EQ(udg.params[2].value(), -0.4);  // -φ at index 2

    // Involutive
    auto u_again = udg.dagger();
    EXPECT_DOUBLE_EQ(u_again.params[0].value(), 0.3);
    EXPECT_DOUBLE_EQ(u_again.params[1].value(), 0.4);
    EXPECT_DOUBLE_EQ(u_again.params[2].value(), 0.5);
}

// ── Conventions §8: CU(θ, φ, λ, γ)† = CU(-θ, -λ, -φ, -γ) ──
TEST(Command, DaggerCUSwapsPhiLambdaAndNegatesGamma) {
    Command cu;
    cu.gate = GateType::CU;
    cu.qubits.push_back(0);
    cu.qubits.push_back(1);
    cu.params.push_back(Param(0.3));   // θ
    cu.params.push_back(Param(0.4));   // φ
    cu.params.push_back(Param(0.5));   // λ
    cu.params.push_back(Param(0.6));   // γ

    auto cudg = cu.dagger();
    EXPECT_EQ(cudg.gate, GateType::CU);
    EXPECT_DOUBLE_EQ(cudg.params[0].value(), -0.3);  // -θ
    EXPECT_DOUBLE_EQ(cudg.params[1].value(), -0.5);  // -λ
    EXPECT_DOUBLE_EQ(cudg.params[2].value(), -0.4);  // -φ
    EXPECT_DOUBLE_EQ(cudg.params[3].value(), -0.6);  // -γ

    // Involutive
    auto cu_again = cudg.dagger();
    EXPECT_DOUBLE_EQ(cu_again.params[0].value(), 0.3);
    EXPECT_DOUBLE_EQ(cu_again.params[1].value(), 0.4);
    EXPECT_DOUBLE_EQ(cu_again.params[2].value(), 0.5);
    EXPECT_DOUBLE_EQ(cu_again.params[3].value(), 0.6);
}

// ── Round-trip: applying G followed by G† leaves the state untouched. ──
//
// Tests the full chain (decomposition → simulator) for the gates with
// non-trivial dagger rules:  iSWAP, U, CU.  Catches any mistake in either
// Command::dagger() or in the iSWAPdg / U / CU decompositions.
TEST(Command, DaggerRoundTripUnitary) {
    QarpSimulator sim;
    Transpiler tp(native_gateset());

    auto run = [&](const std::vector<Command>& body) {
        // Prepare a non-trivial state |ψ⟩ on qubits [0, 1] then apply body.
        std::vector<Command> prog = {
            Command(GateType::H,  0u),
            Command(GateType::Rx, 1u, Param(0.7)),
            Command(GateType::CX, 0u, 1u),
        };
        std::vector<Command> ref_prog = prog;  // |ψ⟩ alone
        prog.insert(prog.end(), body.begin(), body.end());

        auto sv_after = sim.statevector(tp.transpile(prog),     2);
        auto sv_ref   = sim.statevector(tp.transpile(ref_prog), 2);

        ASSERT_EQ(sv_after.size(), sv_ref.size());
        double max = 0;
        for (size_t i = 0; i < sv_after.size(); ++i)
            max = std::max(max, std::abs(sv_after[i] - sv_ref[i]));
        EXPECT_LT(max, 1e-10);
    };

    // iSWAP · iSWAPdg = I
    run({Command(GateType::iSWAP,   0u, 1u),
         Command(GateType::iSWAPdg, 0u, 1u)});

    // iSWAPdg · iSWAP = I
    run({Command(GateType::iSWAPdg, 0u, 1u),
         Command(GateType::iSWAP,   0u, 1u)});

    // U(θ,φ,λ) · U(θ,φ,λ)† = I
    Command u;
    u.gate = GateType::U;
    u.qubits.push_back(0);
    u.params.push_back(Param(0.31));
    u.params.push_back(Param(0.42));
    u.params.push_back(Param(0.53));
    run({u, u.dagger()});

    // CU(θ,φ,λ,γ) · CU(θ,φ,λ,γ)† = I  (γ ≠ 0 exercises the global-phase term)
    Command cu;
    cu.gate = GateType::CU;
    cu.qubits.push_back(0);
    cu.qubits.push_back(1);
    cu.params.push_back(Param(0.31));
    cu.params.push_back(Param(0.42));
    cu.params.push_back(Param(0.53));
    cu.params.push_back(Param(0.64));
    run({cu, cu.dagger()});
}

TEST(Command, RemapQubits) {
    Command cx(GateType::CX, 0, 1);
    auto remapped = cx.remap_qubits({2, 3});  // 0->2, 1->3
    EXPECT_EQ(remapped.qubits[0], 2u);
    EXPECT_EQ(remapped.qubits[1], 3u);
}

TEST(Command, ToString) {
    Command rx(GateType::Rx, 0, Param(1.57));
    std::string s = rx.to_string();
    EXPECT_NE(s.find("Rx"), std::string::npos);
    EXPECT_NE(s.find("q0"), std::string::npos);
}

// ── Block ──

TEST(SimpleBlock, BuildAndFlatten) {
    SimpleBlock bell(2, "Bell");
    bell.h(0).cx(0, 1);
    bell.build();

    EXPECT_TRUE(bell.is_built());
    auto cmds = bell.flatten();
    ASSERT_EQ(cmds.size(), 2u);
    EXPECT_EQ(cmds[0].gate, GateType::H);
    EXPECT_EQ(cmds[1].gate, GateType::CX);
}

TEST(SimpleBlock, Dagger) {
    SimpleBlock block(2, "test");
    block.h(0).cx(0, 1).rz(1, Param(0.5));
    block.build();

    auto dag = block.dagger();
    auto cmds = dag->flatten();
    ASSERT_EQ(cmds.size(), 3u);

    // Reversed order
    EXPECT_EQ(cmds[0].gate, GateType::Rz);     // Rz(0.5)† = Rz(-0.5)
    EXPECT_DOUBLE_EQ(cmds[0].params[0].value(), -0.5);
    EXPECT_EQ(cmds[1].gate, GateType::CX);      // CX† = CX
    EXPECT_EQ(cmds[2].gate, GateType::H);       // H† = H
}

TEST(SimpleBlock, SetSymbols) {
    SimpleBlock block(1, "test");
    block.rx(0, Param::symbol("theta"));
    block.build();

    auto resolved = block.set_symbols({{"theta", 1.57}});
    auto cmds = resolved->flatten();
    EXPECT_FALSE(cmds[0].is_parametric());
    EXPECT_DOUBLE_EQ(cmds[0].params[0].value(), 1.57);
}

// ── CompositeBlock ──

TEST(CompositeBlock, FlattenChildren) {
    auto b1 = make_block<SimpleBlock>(2, "b1");
    b1->h(0).cx(0, 1);
    b1->target_qubits = std::vector<uint32_t>{0, 1};

    auto b2 = make_block<SimpleBlock>(2, "b2");
    b2->h(0).cx(0, 1);
    b2->target_qubits = std::vector<uint32_t>{2, 3};

    CompositeBlock comp({b1, b2}, 4, "composite");
    comp.build();

    auto cmds = comp.flatten();
    ASSERT_EQ(cmds.size(), 4u);

    // b1's gates on qubits 0,1
    EXPECT_EQ(cmds[0].qubits[0], 0u);
    EXPECT_EQ(cmds[1].qubits[0], 0u);
    EXPECT_EQ(cmds[1].qubits[1], 1u);

    // b2's gates on qubits 2,3
    EXPECT_EQ(cmds[2].qubits[0], 2u);
    EXPECT_EQ(cmds[3].qubits[0], 2u);
    EXPECT_EQ(cmds[3].qubits[1], 3u);
}

TEST(CompositeBlock, SetSymbolsRecursesIntoChildren) {
    // Without the override, the base Block::set_symbols would substitute
    // against the composite's empty commands_ buffer and drop every child.
    auto child = make_block<SimpleBlock>(1, "child");
    child->rz(0, Param::symbol("alpha"));

    CompositeBlock comp({child}, 1, "comp");
    comp.build();

    auto resolved = comp.set_symbols({{"alpha", 0.75}});
    auto cmds = resolved->flatten();
    ASSERT_EQ(cmds.size(), 1u);
    EXPECT_EQ(cmds[0].gate, GateType::Rz);
    EXPECT_FALSE(cmds[0].is_parametric());
    EXPECT_DOUBLE_EQ(cmds[0].params[0].value(), 0.75);
}

TEST(CompositeBlock, ReplaceSymbolsRecursesIntoChildren) {
    auto child = make_block<SimpleBlock>(1, "child");
    child->rz(0, Param::symbol("alpha"));

    CompositeBlock comp({child}, 1, "comp");
    comp.build();

    auto renamed = comp.replace_symbols({{"alpha", "beta"}});
    auto cmds = renamed->flatten();
    ASSERT_EQ(cmds.size(), 1u);
    auto syms = cmds[0].params[0].free_symbols();
    ASSERT_EQ(syms.size(), 1u);
    EXPECT_EQ(syms[0], "beta");
}

TEST(ControlledBlock, SetSymbolsPropagatesToInner) {
    auto inner = make_block<SimpleBlock>(1, "inner");
    inner->rz(0, Param::symbol("theta"));
    inner->build();

    ControlledBlock ctrl(inner, /*num_controls=*/1, /*ctrl_state=*/{true}, "wrapper");
    ctrl.build();

    auto resolved = ctrl.set_symbols({{"theta", 0.4}});
    auto cmds = resolved->flatten();
    // Inner Rz becomes CRz on (control, target) — single command with the
    // resolved float angle.
    ASSERT_EQ(cmds.size(), 1u);
    EXPECT_EQ(cmds[0].gate, GateType::CRz);
    EXPECT_FALSE(cmds[0].is_parametric());
    EXPECT_DOUBLE_EQ(cmds[0].params[0].value(), 0.4);
}

// ── Recursive dagger overrides for composite/controlled wrappers ──
// The base ``Block::dagger`` operates on the empty top-level ``commands_``
// buffer for composite/controlled wrappers, so without these overrides the
// dagger would silently return an empty block.

TEST(CompositeBlock, DaggerRecursesIntoChildren) {
    auto child0 = make_block<SimpleBlock>(1, "h0");
    child0->h(0);
    child0->build();
    auto child1 = make_block<SimpleBlock>(1, "s0");
    child1->s(0);
    child1->build();

    CompositeBlock comp({child0, child1}, 1, "hs");
    comp.build();
    auto dagged = comp.dagger();
    auto cmds = dagged->flatten();

    // Reverse order + each gate daggered: Sdg, then H (H is self-adjoint).
    ASSERT_EQ(cmds.size(), 2u);
    EXPECT_EQ(cmds[0].gate, GateType::Sdg);
    EXPECT_EQ(cmds[1].gate, GateType::H);
}

TEST(ControlledBlock, DaggerDaggersInnerAndPreservesControls) {
    auto inner = make_block<SimpleBlock>(1, "rz_pi_4");
    inner->rz(0, Param(0.7));
    inner->build();

    ControlledBlock ctrl(inner, /*num_controls=*/1, /*ctrl_state=*/{true}, "crz");
    ctrl.build();
    auto dagged = ctrl.dagger();
    auto cmds = dagged->flatten();

    // CRz with negated angle (since Rz(θ).dagger() = Rz(-θ)).
    ASSERT_EQ(cmds.size(), 1u);
    EXPECT_EQ(cmds[0].gate, GateType::CRz);
    EXPECT_FALSE(cmds[0].is_parametric());
    EXPECT_DOUBLE_EQ(cmds[0].params[0].value(), -0.7);
}

// ── Multi-control lowering for Pauli inner gates ──

TEST(ControlledBlock, TwoControlPauliXLowersToHMczH) {
    auto inner = make_block<SimpleBlock>(1, "inner");
    inner->x(0);

    ControlledBlock ctrl(inner, /*num_controls=*/2, /*ctrl_state=*/{true, true}, "ccX");
    ctrl.build();

    auto cmds = ctrl.flatten();
    ASSERT_EQ(cmds.size(), 3u);
    EXPECT_EQ(cmds[0].gate, GateType::H);
    EXPECT_EQ(cmds[0].qubits[0], 2u);
    EXPECT_EQ(cmds[1].gate, GateType::MCZ);
    ASSERT_EQ(cmds[1].qubits.size(), 3u);
    EXPECT_EQ(cmds[1].qubits[0], 0u);
    EXPECT_EQ(cmds[1].qubits[1], 1u);
    EXPECT_EQ(cmds[1].qubits[2], 2u);
    EXPECT_EQ(cmds[2].gate, GateType::H);
}

TEST(ControlledBlock, ThreeControlPauliZLowersToMcz) {
    auto inner = make_block<SimpleBlock>(1, "inner");
    inner->z(0);

    ControlledBlock ctrl(inner, /*num_controls=*/3, /*ctrl_state=*/{true, true, true}, "cccZ");
    ctrl.build();

    auto cmds = ctrl.flatten();
    ASSERT_EQ(cmds.size(), 1u);
    EXPECT_EQ(cmds[0].gate, GateType::MCZ);
    ASSERT_EQ(cmds[0].qubits.size(), 4u);  // 3 controls + 1 target
}

TEST(ControlledBlock, MultiControlPauliYLowersToSdgHMczHS) {
    auto inner = make_block<SimpleBlock>(1, "inner");
    inner->y(0);

    ControlledBlock ctrl(inner, /*num_controls=*/2, /*ctrl_state=*/{true, true}, "ccY");
    ctrl.build();

    auto cmds = ctrl.flatten();
    ASSERT_EQ(cmds.size(), 5u);
    EXPECT_EQ(cmds[0].gate, GateType::Sdg);
    EXPECT_EQ(cmds[1].gate, GateType::H);
    EXPECT_EQ(cmds[2].gate, GateType::MCZ);
    EXPECT_EQ(cmds[3].gate, GateType::H);
    EXPECT_EQ(cmds[4].gate, GateType::S);
}

TEST(ControlledBlock, MultiControlControlOnZeroEmitsXFlips) {
    auto inner = make_block<SimpleBlock>(1, "inner");
    inner->x(0);

    // ctrl_state = {false, true} → X-flip on control 0 around the body.
    ControlledBlock ctrl(inner, /*num_controls=*/2, /*ctrl_state=*/{false, true}, "ccX_01");
    ctrl.build();

    auto cmds = ctrl.flatten();
    // X(0) [pre-flip] + H + MCZ + H + X(0) [post-flip] = 5 commands
    ASSERT_EQ(cmds.size(), 5u);
    EXPECT_EQ(cmds[0].gate, GateType::X);
    EXPECT_EQ(cmds[0].qubits[0], 0u);
    EXPECT_EQ(cmds[4].gate, GateType::X);
    EXPECT_EQ(cmds[4].qubits[0], 0u);
}

namespace {
/// Test helper: SimpleBlock that exposes ``add_command`` so we can plant
/// a Custom gate (the public API has no ``custom()`` constructor by design —
/// ``Custom`` is the output of fusion / unitary synthesis, not a hand-written
/// gate).
struct CustomCarrierBlock : public SimpleBlock {
    CustomCarrierBlock(uint32_t n, const Eigen::MatrixXcd& U)
        : SimpleBlock(n, "custom_inner") {
        Command custom;
        custom.gate = GateType::Custom;
        custom.qubits.push_back(0);
        custom.unitary = std::make_shared<Eigen::MatrixXcd>(U);
        add_command(std::move(custom));
    }
};
}  // namespace

TEST(ControlledBlock, CustomGateRejectedWithActionableMessage) {
    // ``Custom`` is a meta gate (opaque dense matrix) — the multi-control
    // lowering can't auto-decompose it without first running unitary synthesis
    // *and* recovering the global phase QSD drops.  Reject early with a
    // message pointing at ``Block::unitary_synthesis(U)`` (single-control
    // also rejects with the same guidance).
    Eigen::MatrixXcd U(2, 2);
    U << 1, 0,
         0, std::complex<double>(0, 1);  // = P(π/2) but routed via Custom.

    auto inner = make_block<CustomCarrierBlock>(1, U);
    inner->build();

    // Single-control: should fail at flatten with a message naming Custom
    // and pointing at unitary_synthesis.
    {
        ControlledBlock ctrl1(inner, /*num_controls=*/1, /*ctrl_state=*/{true}, "c_custom");
        ctrl1.build();
        try {
            std::vector<Command> ignore = ctrl1.flatten();
            (void)ignore;
            FAIL() << "expected runtime_error for Custom-in-single-control";
        } catch (const std::runtime_error& e) {
            std::string msg = e.what();
            EXPECT_NE(msg.find("Custom"), std::string::npos);
            EXPECT_NE(msg.find("unitary_synthesis"), std::string::npos);
        }
    }

    // Multi-control: same — rejected upfront in flatten() before lowering.
    {
        ControlledBlock ctrl2(inner, /*num_controls=*/2, /*ctrl_state=*/{true, true}, "cc_custom");
        ctrl2.build();
        try {
            std::vector<Command> ignore = ctrl2.flatten();
            (void)ignore;
            FAIL() << "expected runtime_error for Custom-in-multi-control";
        } catch (const std::runtime_error& e) {
            std::string msg = e.what();
            EXPECT_NE(msg.find("Custom"), std::string::npos);
            EXPECT_NE(msg.find("unitary_synthesis"), std::string::npos);
        }
    }
}

TEST(ControlledBlock, MultiControlHLowersThroughBasis) {
    // ``ControlledBlock::flatten`` auto-transpiles the inner block to the
    // multi-control basis first (qarp_conventions.md §6), so any unitary inner
    // block works even when its gates fall outside the raw multi-control basis
    // {X, Y, Z, Rx, Ry, Rz, P, GPhase, Barrier}.  Unitary correctness for every
    // gate type is pinned in ``test_controlled_block_unitaries.cpp``; here we
    // just verify that an H inner block flattens without throwing.
    auto inner = make_block<SimpleBlock>(1, "inner");
    inner->h(0);

    ControlledBlock ctrl(inner, /*num_controls=*/2, /*ctrl_state=*/{true, true}, "ccH");
    ctrl.build();

    EXPECT_NO_THROW((void)ctrl.flatten());
}

// ── Multi-controlled rotations + GPhase via Barenco recursion (§3.3 #9) ──

TEST(ControlledBlock, TwoControlRzExpandsAndPreservesGateCount) {
    auto inner = make_block<SimpleBlock>(1, "inner");
    inner->rz(0, Param(0.7));
    inner->build();

    ControlledBlock ctrl(inner, /*num_controls=*/2, /*ctrl_state=*/{true, true}, "ccRz");
    ctrl.build();
    auto cmds = ctrl.flatten();

    // Barenco recursion at n=2 emits: C(V), C^1(X), C(V†), C^1(X), C^1(V).
    // C^1(V) = CRz(θ/2), and C^1(X) = CX (single command each).  Total: 5.
    ASSERT_EQ(cmds.size(), 5u);
    EXPECT_EQ(cmds[0].gate, GateType::CRz);
    EXPECT_EQ(cmds[1].gate, GateType::CX);
    EXPECT_EQ(cmds[2].gate, GateType::CRz);
    EXPECT_EQ(cmds[3].gate, GateType::CX);
    EXPECT_EQ(cmds[4].gate, GateType::CRz);
}

TEST(ControlledBlock, TwoControlGPhaseLowersToCP) {
    auto inner = make_block<SimpleBlock>(1, "inner");
    inner->gphase(Param(0.7));
    inner->build();

    ControlledBlock ctrl(inner, /*num_controls=*/2, /*ctrl_state=*/{true, true}, "ccGPhase");
    ctrl.build();
    auto cmds = ctrl.flatten();

    // C^2(GPhase(θ)) reduces to C^1(P(θ)) on the two control qubits, which
    // is a single CP gate.  No target qubit ops at all.
    ASSERT_EQ(cmds.size(), 1u);
    EXPECT_EQ(cmds[0].gate, GateType::CP);
}

TEST(ControlledBlock, OneControlGPhaseLowersToP) {
    auto inner = make_block<SimpleBlock>(1, "inner");
    inner->gphase(Param(0.7));
    inner->build();

    ControlledBlock ctrl(inner, /*num_controls=*/1, /*ctrl_state=*/{true}, "cGPhase");
    ctrl.build();
    auto cmds = ctrl.flatten();

    // C^1(GPhase(θ)) = P(θ) on the control.
    ASSERT_EQ(cmds.size(), 1u);
    EXPECT_EQ(cmds[0].gate, GateType::P);
}
