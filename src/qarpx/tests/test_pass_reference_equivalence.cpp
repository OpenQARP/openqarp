// Pass equivalence against an INDEPENDENT mathematical reference.
//
// reference_unitary.h assembles circuit unitaries from the analytic gate
// definitions in qarp_conventions.md, sharing no code with csim.  Two layers
// of protection:
//   1. Reference ↔ csim cross-validation on wide-vocabulary random circuits
//      — a bug in either implementation (kernel dispatch, LSB embedding,
//      control conventions) surfaces as a disagreement between them.
//   2. Every optimization entry point must preserve the REFERENCE unitary —
//      so a pass bug can no longer hide behind a matching csim bug.

#include "dag_test_helpers.h"
#include "reference_unitary.h"

#include "qarpx/dag/passes.h"
#include "qarpx/simulator/qarp_simulator.h"
#include "qarpx/transpiler/gateset.h"
#include "qarpx/transpiler/identities.h"
#include "qarpx/transpiler/transpiler.h"

#include <vector>

namespace qarpx::test {

namespace {

constexpr uint32_t kQubits = 3;
constexpr std::size_t kLen = 30;

bool reference_equal(const std::vector<Command>& a,
                     const std::vector<Command>& b) {
    return (reference_unitary(a, kQubits) - reference_unitary(b, kQubits))
               .norm() < 1e-9;
}


std::vector<Command> wide_circuit(uint32_t seed) {
    RandomCircuit gen(seed, kQubits, /*n_cbits=*/1,
                      CircuitGrammar::kUnitaryWide);
    return gen.generate(kLen);
}

}  // anonymous namespace

// ── Layer 1: the reference and csim corroborate each other ──────────────────

TEST(ReferenceUnitary, AgreesWithCsimOnWideVocabulary) {
    QarpSimulator sim;
    // native_gateset dispatches the wide vocabulary directly — no
    // decomposition in the loop, so the comparison is phase-EXACT and
    // cross-validates csim's kernels + embedding against the analytic
    // reference, gate for gate.
    Transpiler rebase(native_gateset());
    for (uint32_t seed = 0; seed < 100; ++seed) {
        const auto cmds = wide_circuit(seed);
        const auto ref = reference_unitary(cmds, kQubits);
        const auto csim = sim.unitary_matrix(rebase.transpile(cmds), kQubits);
        ASSERT_LT((ref - csim).norm(), 1e-9)
            << "reference/csim disagreement at seed " << seed;
    }
}

TEST(ReferenceUnitary, RebaseIsPhaseExact) {
    // Since 2026-07-10 the phase-family decompositions (P/U/CP, and CU
    // transitively) emit their compensating GPhase, so rebasing to a gate
    // set without those gates preserves the unitary EXACTLY — global phase
    // included — whenever the target can express GPhase.  (Targets that
    // cannot, e.g. cudaq, drop it via decompose_gphase: the documented
    // per-target boundary.)
    QarpSimulator sim;
    Transpiler rebase(qulacs_gateset());
    for (uint32_t seed = 0; seed < 60; ++seed) {
        const auto cmds = wide_circuit(seed + 500);
        const auto ref = reference_unitary(cmds, kQubits);
        const auto csim = sim.unitary_matrix(rebase.transpile(cmds), kQubits);
        ASSERT_LT((ref - csim).norm(), 1e-9)
            << "rebase broke phase exactness at seed " << seed;
    }
}

TEST(ReferenceUnitary, RebasePlusOptimizeIsPhaseExact) {
    // The order-of-operations footgun that motivated the fix: rebase to a
    // P-less gate set FIRST, optimize after — the eigenphases a QPE-style
    // consumer would measure must survive unchanged.
    QarpSimulator sim;
    Transpiler t(qulacs_gateset());
    for (uint32_t seed = 0; seed < 60; ++seed) {
        const auto cmds = wide_circuit(seed + 1500);
        const auto ref = reference_unitary(cmds, kQubits);
        for (auto level : {OptLevel::O1, OptLevel::O2}) {
            const auto out = t.transpile_and_optimize(cmds, level);
            ASSERT_LT((ref - sim.unitary_matrix(out, kQubits)).norm(), 1e-9)
                << "level " << static_cast<int>(level) << ", seed " << seed;
        }
    }
}

TEST(ReferenceUnitary, DetectsTampering) {
    // Sanity: the check has teeth — dropping one non-identity gate must be
    // caught.
    auto cmds = wide_circuit(7);
    auto tampered = cmds;
    // Drop the first fixed non-identity gate; seed 7 happens to draw neither
    // a CX nor an H, so a two-gate whitelist left the circuit untouched.
    for (std::size_t i = 0; i < tampered.size(); ++i) {
        switch (tampered[i].gate) {
            case GateType::X:  case GateType::Y:  case GateType::Z:  case GateType::H:
            case GateType::S:  case GateType::T:  case GateType::CX: case GateType::CZ:
            case GateType::SWAP: case GateType::CCX:
                tampered.erase(tampered.begin() + static_cast<std::ptrdiff_t>(i));
                i = tampered.size();  // done
                break;
            default:
                break;
        }
    }
    ASSERT_NE(tampered.size(), cmds.size());
    EXPECT_FALSE(reference_equal(cmds, tampered));
}

// ── Layer 2: every optimization entry point vs the reference ────────────────

TEST(PassReferenceEquivalence, WireAdjacentCancel) {
    for (uint32_t seed = 0; seed < 100; ++seed) {
        const auto cmds = wide_circuit(seed + 1000);
        auto dag = CircuitDAG::from_commands(cmds, kQubits, 1);
        dag_passes::cancel_wire_adjacent(dag);
        ASSERT_TRUE(reference_equal(cmds, dag.to_commands()))
            << "O1 pass broke the reference unitary at seed " << seed;
    }
}

TEST(PassReferenceEquivalence, CommuteAndCancel) {
    for (uint32_t seed = 0; seed < 100; ++seed) {
        const auto cmds = wide_circuit(seed + 2000);
        auto dag = CircuitDAG::from_commands(cmds, kQubits, 1);
        dag_passes::cancel_wire_adjacent(dag);
        dag_passes::commute_and_cancel(dag);
        ASSERT_TRUE(reference_equal(cmds, dag.to_commands()))
            << "O2 pass broke the reference unitary at seed " << seed;
    }
}

TEST(PassReferenceEquivalence, EliminateIdentitiesWrapper) {
    for (uint32_t seed = 0; seed < 60; ++seed) {
        const auto cmds = wide_circuit(seed + 3000);
        auto optimized = cmds;
        eliminate_identities(optimized);
        ASSERT_TRUE(reference_equal(cmds, optimized)) << "seed " << seed;
    }
}

TEST(PassReferenceEquivalence, TranspileAndOptimizeAllLevels) {
    // Full production pipeline (decompose + DAG passes + fusion) vs the
    // reference built from the UN-transpiled input: covers the decomposition
    // tables, both DAG passes, and single-qubit fusion in one assertion.
    Transpiler t(native_gateset());
    for (uint32_t seed = 0; seed < 60; ++seed) {
        const auto cmds = wide_circuit(seed + 4000);
        const auto ref = reference_unitary(cmds, kQubits);
        for (auto level : {OptLevel::O0, OptLevel::O1, OptLevel::O2}) {
            const auto out = t.transpile_and_optimize(cmds, level);
            ASSERT_LT((ref - reference_unitary(out, kQubits)).norm(), 1e-9)
                << "level " << static_cast<int>(level) << ", seed " << seed;
        }
    }
}

// ── Layer 3: the pairs are_inverse_pair newly recognises, each vs the reference ──
//
// Layer 2 draws angles at random and picks 2q gates 1-in-12 per slot, so it
// essentially never places an exact inverse pair adjacently: it proves the
// pass makes no *wrong* cancellation far better than it proves the *right*
// ones are identities.  Splice each newly recognised pair into a random wide
// context and require both that it is removed (non-vacuity) and that the
// reference unitary is untouched (§16 EQ-2, global phase included).

TEST(PassReferenceEquivalence, NewlyCancellablePairsPreserveTheReference) {
    struct Pair { Command a, b; };
    // U(theta, phi, lambda): no three-parameter convenience constructor.
    const auto make_u = [](uint32_t q, double th, double ph, double la) {
        Command u(GateType::U, q);
        u.params.push_back(Param(th));
        u.params.push_back(Param(ph));
        u.params.push_back(Param(la));
        return u;
    };
    const Param th(0.37);
    const std::vector<Pair> pairs = {
        // named-inverse pairs the hand-rolled list was missing
        {Command(GateType::SX, 0),       Command(GateType::SXdg, 0)},
        {Command(GateType::CS, 0, 1),    Command(GateType::CSdg, 0, 1)},
        {Command(GateType::CSX, 1, 2),   Command(GateType::CSXdg, 1, 2)},
        {Command(GateType::iSWAP, 0, 2), Command(GateType::iSWAPdg, 0, 2)},
        // §3.2 argument symmetry: iSWAP cancels with its qubits swapped
        {Command(GateType::iSWAP, 0, 1), Command(GateType::iSWAPdg, 1, 0)},
        // §3.2/§9: ECR is self-adjoint, argument-ordered
        {Command(GateType::ECR, 1, 2),   Command(GateType::ECR, 1, 2)},
        // parametric: the angle is negated by Command::dagger()
        {Command(GateType::CP,  0, 1, {th}), Command(GateType::CP,  0, 1, {-th})},
        {Command(GateType::CRz, 2, 0, {th}), Command(GateType::CRz, 2, 0, {-th})},
        {Command(GateType::Rx,  1,    {th}), Command(GateType::Rx,  1,    {-th})},
        // symmetric AND parametric: swapped qubits, angle still checked
        {Command(GateType::RZZ, 0, 1, {th}), Command(GateType::RZZ, 1, 0, {-th})},
        // U's dagger swaps phi and lambda (§9), not a per-parameter negation
        {make_u(0, 0.3, 0.2, 0.1), make_u(0, -0.3, -0.1, -0.2)},
    };
    for (uint32_t seed = 0; seed < 20; ++seed) {
        for (std::size_t i = 0; i < pairs.size(); ++i) {
            auto cmds = wide_circuit(seed + 5000);
            const auto at = static_cast<std::ptrdiff_t>(1 + (seed * 7 + i) % (cmds.size() - 1));
            cmds.insert(cmds.begin() + at, {pairs[i].a, pairs[i].b});
            auto optimized = cmds;
            const std::size_t removed = eliminate_identities(optimized);
            ASSERT_GE(removed, 2u) << "pair " << i << " was not cancelled, seed " << seed;
            ASSERT_TRUE(reference_equal(cmds, optimized))
                << "cancelling pair " << i << " changed the reference unitary, seed " << seed;
        }
    }
}

// ── Layer 3b: additive folds vs the reference ────────────────────────────────
//
// Random wide circuits rarely put two same-gate parametric commands adjacently
// on the same qubits.  Splice one such pair per family member into a random
// context and require the fold to happen and the reference unitary to survive.

TEST(PassReferenceEquivalence, AdditiveFoldsPreserveTheReference) {
    struct Pair { Command a, b; };
    const std::vector<Pair> pairs = {
        {Command(GateType::P,   0,    {Param(0.31)}), Command(GateType::P,   0,    {Param(0.17)})},
        {Command(GateType::Rz,  1,    {Param(0.31)}), Command(GateType::Rz,  1,    {Param(0.17)})},
        {Command(GateType::CP,  0, 1, {Param(0.31)}), Command(GateType::CP,  0, 1, {Param(0.17)})},
        {Command(GateType::CRx, 2, 0, {Param(0.31)}), Command(GateType::CRx, 2, 0, {Param(0.17)})},
        {Command(GateType::CRz, 1, 2, {Param(0.31)}), Command(GateType::CRz, 1, 2, {Param(0.17)})},
        {Command(GateType::RXX, 0, 2, {Param(0.31)}), Command(GateType::RXX, 0, 2, {Param(0.17)})},
        {Command(GateType::RZZ, 0, 1, {Param(0.31)}), Command(GateType::RZZ, 1, 0, {Param(0.17)})},
    };
    for (uint32_t seed = 0; seed < 20; ++seed) {
        for (std::size_t i = 0; i < pairs.size(); ++i) {
            auto cmds = wide_circuit(seed + 6000);
            const auto at = static_cast<std::ptrdiff_t>(1 + (seed * 5 + i) % (cmds.size() - 1));
            cmds.insert(cmds.begin() + at, {pairs[i].a, pairs[i].b});
            auto optimized = cmds;
            const std::size_t removed = eliminate_identities(optimized);
            ASSERT_GE(removed, 1u) << "pair " << i << " did not fold, seed " << seed;
            ASSERT_TRUE(reference_equal(cmds, optimized))
                << "folding pair " << i << " changed the reference unitary, seed " << seed;
        }
    }
}

}  // namespace qarpx::test
