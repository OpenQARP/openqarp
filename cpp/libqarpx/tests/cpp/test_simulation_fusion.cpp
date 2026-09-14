// ── Simulation fusion (fuse_for_simulation + k-qubit Custom dispatch) ──
//
// Oracle: tests/cpp/reference_unitary.h — the analytic gate definitions of
// qarp_conventions.md assembled by explicit embedding, sharing no code with
// csim, the fusion pass or the dense kernel.  Every numerical assertion here
// is against that reference (or against the unfused trajectory path, whose
// own oracle tests live in test_trajectory.cpp / test_dag_passes.cpp).

#include "dag_test_helpers.h"
#include "reference_unitary.h"

#include "qarpx/simulator/dense_kernel.h"
#include "qarpx/simulator/fusion.h"
#include "qarpx/simulator/qarp_simulator.h"
#include "qarpx/transpiler/fusion.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace qarpx::test {

namespace {

using Mat = Eigen::MatrixXcd;
using Vec = Eigen::VectorXcd;

constexpr double kTol = 1e-10;

Vec to_vec(const std::vector<std::complex<double>>& sv) {
    Vec v(static_cast<Eigen::Index>(sv.size()));
    for (std::size_t i = 0; i < sv.size(); ++i) v[static_cast<Eigen::Index>(i)] = sv[i];
    return v;
}

Vec random_state(uint32_t seed, uint32_t n_qubits) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> g(0.0, 1.0);
    Vec v(Eigen::Index{1} << n_qubits);
    for (Eigen::Index i = 0; i < v.size(); ++i) v[i] = {g(rng), g(rng)};
    return v / v.norm();
}

std::vector<std::complex<double>> to_std(const Vec& v) {
    return std::vector<std::complex<double>>(v.data(), v.data() + v.size());
}

/// Wide-vocabulary unitary circuit: every §2–§7 gate incl. GPhase, 3-qubit
/// gates, MCZ and 1-qubit Custom; concrete parameters.
std::vector<Command> wide_circuit(uint32_t seed, uint32_t n_qubits, std::size_t len) {
    RandomCircuit gen(seed, n_qubits, /*n_cbits=*/1, CircuitGrammar::kUnitaryWide);
    return gen.generate(len);
}

/// Sparse Pauli-free "no-frills" dispatchable trajectory circuit: measure,
/// reset, barrier (listed and register-wide), per-command conditions on 1q
/// and 2q gates, branch regions, 1q Custom.  Concrete parameters.
std::vector<Command> trajectory_circuit(uint32_t seed, uint32_t n_qubits,
                                        uint32_t n_cbits, std::size_t len) {
    std::mt19937 rng(seed);
    auto pick = [&](uint32_t k) { return std::uniform_int_distribution<uint32_t>(0, k - 1)(rng); };
    auto q = [&] { return pick(n_qubits); };
    auto c = [&] { return pick(n_cbits); };
    auto angle = [&] { return Param(std::uniform_real_distribution<double>(-3.0, 3.0)(rng)); };
    auto pair = [&] {
        const uint32_t a = q();
        uint32_t b = q();
        while (b == a) b = q();
        return std::pair{a, b};
    };

    std::vector<Command> out;
    std::function<void(int)> emit = [&](int region_budget) {
        switch (pick(region_budget > 0 ? 13 : 12)) {
            case 0: out.emplace_back(GateType::H, q()); break;
            case 1: out.emplace_back(GateType::Ry, q(), angle()); break;
            case 2: out.emplace_back(GateType::Rz, q(), angle()); break;
            case 3: { auto [a, b] = pair(); out.emplace_back(GateType::CX, a, b); break; }
            case 4: { auto [a, b] = pair(); out.emplace_back(GateType::RZZ, a, b, angle()); break; }
            case 5: out.push_back(make_measure(q(), c())); break;
            case 6: out.emplace_back(GateType::Reset, q()); break;
            case 7: {
                Command barrier;
                barrier.gate = GateType::Barrier;
                if (pick(3) != 0)
                    for (uint32_t i = 0; i < n_qubits; ++i)
                        if (pick(2) == 0) barrier.qubits.push_back(i);
                out.push_back(std::move(barrier));
                break;
            }
            case 8: out.push_back(make_conditional(
                        Command(GateType::Rx, q(), angle()), {c()}, {pick(2) == 0}));
                    break;
            case 9: {
                auto [a, b] = pair();
                out.push_back(make_conditional(Command(GateType::CX, a, b), {c()}, {pick(2) == 0}));
                break;
            }
            case 10: out.push_back(make_custom_1q(q())); break;
            case 11: {
                Command gp;
                gp.gate = GateType::GPhase;
                gp.params.push_back(angle());
                out.push_back(std::move(gp));
                break;
            }
            case 12: {
                out.push_back(make_branch_begin({c()}, {pick(2) == 0}));
                const std::size_t body = 1 + pick(3);
                for (std::size_t i = 0; i < body; ++i) emit(region_budget - 1);
                if (pick(2) == 0) {
                    out.push_back(make_marker(GateType::BranchElse));
                    out.emplace_back(GateType::Z, q());
                }
                out.push_back(make_marker(GateType::BranchEnd));
                break;
            }
        }
    };
    while (out.size() < len) emit(/*region_budget=*/2);
    return out;
}

LocalUnitary local_unitary_of(const QarpSimulator& sim) {
    return [&sim](const Command& c) { return sim.local_unitary(c); };
}

bool same_command(const Command& a, const Command& b) {
    if (a.gate != b.gate || !(a.qubits == b.qubits) || !(a.params == b.params)) return false;
    if (!(a.condition_bits == b.condition_bits) || !(a.condition_values == b.condition_values))
        return false;
    if (a.gate != GateType::Custom) return true;
    if (!a.unitary || !b.unitary || a.unitary->rows() != b.unitary->rows()) return false;
    return (*a.unitary - *b.unitary).norm() < 1e-12;
}

}  // namespace

// ── Statevector against the analytic reference ───────────────────────────────

TEST(SimulationFusion, FusedStatevectorMatchesReference) {
    // U·|0…0⟩ from reference_unitary.h, exact including global phase, for
    // every block width the knob accepts on wide-vocabulary circuits.
    for (uint32_t n_qubits : {3u, 4u, 6u}) {
        for (std::size_t k : {2u, 3u, 4u, 5u}) {
            QarpSimulator sim;
            sim.set_fusion_max_qubits(k);
            sim.set_fusion_min_qubits(0);  // small registers: reach the dense path
            for (uint32_t seed = 0; seed < 40; ++seed) {
                const auto cmds = wide_circuit(seed + 100 * n_qubits, n_qubits, 40);
                const Vec ref = reference_unitary(cmds, n_qubits).col(0);
                const Vec got = to_vec(sim.statevector(cmds, static_cast<int>(n_qubits)));
                ASSERT_LT((ref - got).norm(), kTol)
                    << "n=" << n_qubits << " k=" << k << " seed=" << seed;
            }
        }
    }
}

TEST(SimulationFusion, FusedStatevectorWithInitialStateMatchesReference) {
    constexpr uint32_t n = 5;
    QarpSimulator sim;
    sim.set_fusion_max_qubits(3);
    sim.set_fusion_min_qubits(0);  // small registers: reach the dense path
    for (uint32_t seed = 0; seed < 30; ++seed) {
        const auto cmds = wide_circuit(seed + 7000, n, 40);
        const Vec psi0 = random_state(seed, n);
        const Vec ref = reference_unitary(cmds, n) * psi0;
        const Vec got = to_vec(sim.statevector(cmds, n, to_std(psi0)));
        ASSERT_LT((ref - got).norm(), kTol) << "seed " << seed;
    }
}

TEST(SimulationFusion, SymbolicGatesAreLeftForSubstitution) {
    // A symbolic gate flushes its wires and passes through unchanged; after
    // substitution the fused stream must still reproduce the reference of the
    // substituted input.
    constexpr uint32_t n = 4;
    QarpSimulator sim;
    sim.set_fusion_max_qubits(3);
    sim.set_fusion_min_qubits(0);  // small registers: reach the dense path
    for (uint32_t seed = 0; seed < 20; ++seed) {
        auto cmds = wide_circuit(seed + 9000, n, 30);
        cmds.insert(cmds.begin() + 10, Command(GateType::Rx, 1u, Param::symbol("a")));
        cmds.insert(cmds.begin() + 20, Command(GateType::CRz, 2u, 0u, Param::symbol("b")));

        const auto fused = fuse_for_simulation(cmds, 3, local_unitary_of(sim));
        std::size_t symbolic = 0;
        for (const auto& c : fused) {
            if (!c.is_parametric()) continue;
            ++symbolic;
            EXPECT_TRUE(c.gate == GateType::Rx || c.gate == GateType::CRz);
        }
        EXPECT_EQ(symbolic, 2u) << "seed " << seed;

        const std::unordered_map<std::string, double> values{{"a", 0.7}, {"b", -1.9}};
        const auto concrete = substitute_all(cmds, values);
        const Vec ref = reference_unitary(concrete, n).col(0);
        const Vec got = to_vec(sim.statevector(substitute_all(fused, values), n));
        ASSERT_LT((ref - got).norm(), kTol) << "seed " << seed;
    }
}

// ── k-qubit Custom dispatch: ordering pinned on non-sorted, non-adjacent targets ──

TEST(SimulationFusion, CustomDispatchOrderingMatchesReference) {
    QarpSimulator sim;
    constexpr uint32_t n = 5;

    // Local bit b ↔ qubits[b]: CCX on [3, 0, 2] is controls 3 and 0, target 2.
    Command ccx(GateType::CCX, SmallVector<uint32_t, 2>{3u, 0u, 2u});
    Command custom3(GateType::Custom, SmallVector<uint32_t, 2>{3u, 0u, 2u});
    custom3.unitary = std::make_shared<const Mat>(refu_detail::gate_matrix_of(ccx));
    EXPECT_LT((sim.unitary_matrix({custom3}, n) - reference_unitary({ccx}, n)).norm(), kTol);

    Command cx(GateType::CX, 2u, 0u);
    Command custom2(GateType::Custom, SmallVector<uint32_t, 2>{2u, 0u});
    custom2.unitary = std::make_shared<const Mat>(refu_detail::gate_matrix_of(cx));
    EXPECT_LT((sim.unitary_matrix({custom2}, n) - reference_unitary({cx}, n)).norm(), kTol);

    // A random dense 3-qubit unitary on [4, 1, 2] against the reference's own
    // embedding — the two conventions must agree bit for bit in meaning.
    for (uint32_t seed = 0; seed < 5; ++seed) {
        std::mt19937 rng(seed);
        std::normal_distribution<double> g;
        Mat a(8, 8);
        for (Eigen::Index i = 0; i < 8; ++i)
            for (Eigen::Index j = 0; j < 8; ++j) a(i, j) = {g(rng), g(rng)};
        const Mat u = Eigen::HouseholderQR<Mat>(a).householderQ();
        Command custom(GateType::Custom, SmallVector<uint32_t, 2>{4u, 1u, 2u});
        custom.unitary = std::make_shared<const Mat>(u);
        EXPECT_LT((sim.unitary_matrix({custom}, n) - reference_unitary({custom}, n)).norm(), kTol)
            << "seed " << seed;
    }
}

TEST(SimulationFusion, DenseKernelEveryWidthMatchesReference) {
    // apply_dense_block for k = 1..kMaxDenseBlockQubits on a random state,
    // random targets, against the reference embedding.
    constexpr uint32_t n = 9;
    for (std::size_t k = 1; k <= kMaxDenseBlockQubits; ++k) {
        std::mt19937 rng(static_cast<uint32_t>(k));
        std::vector<uint32_t> all(n);
        for (uint32_t i = 0; i < n; ++i) all[i] = i;
        std::shuffle(all.begin(), all.end(), rng);
        Command custom;
        custom.gate = GateType::Custom;
        for (std::size_t b = 0; b < k; ++b) custom.qubits.push_back(all[b]);
        const Eigen::Index d = Eigen::Index{1} << k;
        std::normal_distribution<double> g;
        Mat a(d, d);
        for (Eigen::Index i = 0; i < d; ++i)
            for (Eigen::Index j = 0; j < d; ++j) a(i, j) = {g(rng), g(rng)};
        custom.unitary = std::make_shared<const Mat>(Eigen::HouseholderQR<Mat>(a).householderQ());

        const Vec psi0 = random_state(static_cast<uint32_t>(k) + 50, n);
        const Vec ref = reference_unitary({custom}, n) * psi0;
        auto state = to_std(psi0);
        apply_dense_block(custom.qubits.begin(), k, *custom.unitary, state.data(),
                          uint64_t{1} << n);
        ASSERT_LT((ref - to_vec(state)).norm(), kTol) << "k=" << k;
    }
}

TEST(SimulationFusion, CustomDispatchRejectsMalformedPayload) {
    QarpSimulator sim;
    EXPECT_EQ(sim.fusion_max_qubits(), QarpSimulator::kDefaultFusionQubits);
    EXPECT_EQ(sim.fusion_min_qubits(), QarpSimulator::kDefaultFusionMinQubits);
    std::vector<std::complex<double>> state(4, 0.0);
    state[0] = 1.0;
    Command wrong(GateType::Custom, SmallVector<uint32_t, 2>{0u, 1u});
    wrong.unitary = std::make_shared<const Mat>(Mat::Identity(2, 2));
    EXPECT_THROW(sim.apply_command(wrong, state.data(), 4), std::runtime_error);
    Command none(GateType::Custom, 0u);
    EXPECT_THROW(sim.apply_command(none, state.data(), 4), std::runtime_error);
    EXPECT_THROW(sim.set_fusion_max_qubits(QarpSimulator::kMaxFusionQubits + 1),
                 std::invalid_argument);
}

// ── local_unitary against the analytic gate matrices ─────────────────────────

TEST(SimulationFusion, LocalUnitaryMatchesAnalyticGateMatrices) {
    // Every gate the wide grammar emits, on non-sorted qubits so the
    // local-bit ↔ qubits[b] convention is exercised, plus the 1q family the
    // O1 table covers (test_fusion_consistency.cpp) — the two paths cannot
    // diverge.
    QarpSimulator sim;
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> ang(-3.0, 3.0);
    std::vector<Command> gates;
    for (auto g : {GateType::X, GateType::Y, GateType::Z, GateType::H, GateType::S,
                   GateType::Sdg, GateType::T, GateType::Tdg, GateType::SX, GateType::SXdg,
                   GateType::Id})
        gates.emplace_back(g, 3u);
    for (auto g : {GateType::Rx, GateType::Ry, GateType::Rz, GateType::P})
        gates.emplace_back(g, 2u, Param(ang(rng)));
    {
        Command u(GateType::U, 1u);
        for (int i = 0; i < 3; ++i) u.params.push_back(Param(ang(rng)));
        gates.push_back(u);
    }
    for (auto g : {GateType::CX, GateType::CY, GateType::CZ, GateType::SWAP, GateType::iSWAP,
                   GateType::iSWAPdg, GateType::ECR, GateType::CH, GateType::CS, GateType::CSdg,
                   GateType::CSX, GateType::CSXdg})
        gates.emplace_back(g, 2u, 0u);
    for (auto g : {GateType::CRx, GateType::CRy, GateType::CRz, GateType::CP, GateType::RZZ,
                   GateType::RXX, GateType::RYY})
        gates.emplace_back(g, 3u, 1u, Param(ang(rng)));
    {
        Command cu(GateType::CU, 1u, 0u);
        for (int i = 0; i < 4; ++i) cu.params.push_back(Param(ang(rng)));
        gates.push_back(cu);
    }
    gates.emplace_back(GateType::CCX, SmallVector<uint32_t, 2>{2u, 0u, 1u});
    gates.emplace_back(GateType::CSWAP, SmallVector<uint32_t, 2>{1u, 2u, 0u});
    gates.emplace_back(GateType::MCZ, SmallVector<uint32_t, 2>{2u, 0u, 1u});
    gates.push_back(make_custom_1q(0u));

    for (const auto& cmd : gates) {
        const Mat ref = refu_detail::gate_matrix_of(cmd);
        const Mat got = sim.local_unitary(cmd);
        ASSERT_EQ(got.rows(), ref.rows()) << gate_name(cmd.gate);
        EXPECT_LT((ref - got).norm(), kTol) << gate_name(cmd.gate);
    }
}

// ── Pass invariants ──────────────────────────────────────────────────────────

TEST(SimulationFusion, BlocksNeverExceedMaxQubits) {
    QarpSimulator sim;
    for (std::size_t k : {1u, 2u, 3u, 4u}) {
        for (uint32_t seed = 0; seed < 40; ++seed) {
            const auto cmds = wide_circuit(seed + 3000, 6, 60);
            const auto fused = fuse_for_simulation(cmds, k, local_unitary_of(sim));
            for (const auto& c : fused) {
                if (c.gate != GateType::Custom) continue;
                ASSERT_LE(c.qubits.size(), k) << "k=" << k << " seed=" << seed;
                ASSERT_TRUE(c.unitary);
                ASSERT_EQ(c.unitary->rows(), Eigen::Index{1} << c.qubits.size());
            }
        }
    }
}

TEST(SimulationFusion, NonUnitaryCommandsSurviveInOrder) {
    // Measure / Reset / Barrier / branch markers / GPhase are never fused
    // and keep their relative order.
    QarpSimulator sim;
    auto skeleton = [](const std::vector<Command>& cmds) {
        std::vector<Command> out;
        for (const auto& c : cmds)
            if (c.gate == GateType::Measure || c.gate == GateType::Reset
                    || c.gate == GateType::Barrier || c.gate == GateType::GPhase
                    || c.gate == GateType::BranchBegin || c.gate == GateType::BranchElse
                    || c.gate == GateType::BranchEnd)
                out.push_back(c);
        return out;
    };
    for (uint32_t seed = 0; seed < 40; ++seed) {
        const auto cmds = trajectory_circuit(seed + 4000, 4, 2, 60);
        const auto fused = fuse_for_simulation(cmds, 3, local_unitary_of(sim));
        const auto a = skeleton(cmds), b = skeleton(fused);
        ASSERT_EQ(a.size(), b.size()) << "seed " << seed;
        for (std::size_t i = 0; i < a.size(); ++i)
            EXPECT_TRUE(same_command(a[i], b[i])) << "seed " << seed << " at " << i;
    }
}

TEST(SimulationFusion, WidthOneIsAtLeastAsTightAsSingleQubitPass) {
    // At k = 1 the pass is a superset of fuse_single_qubit_gates (it also
    // folds 1-qubit Custom payloads), so it emits only 1-qubit Customs,
    // never more commands, and the same unitary.
    QarpSimulator sim;
    for (uint32_t seed = 0; seed < 40; ++seed) {
        const auto cmds = wide_circuit(seed + 5000, 4, 40);
        const auto ours   = fuse_for_simulation(cmds, 1, local_unitary_of(sim));
        const auto theirs = fuse_single_qubit_gates(cmds);
        EXPECT_LE(ours.size(), theirs.size()) << "seed " << seed;
        for (const auto& c : ours)
            if (c.gate == GateType::Custom) ASSERT_EQ(c.qubits.size(), 1u) << "seed " << seed;
        EXPECT_LT((reference_unitary(ours, 4) - reference_unitary(cmds, 4)).norm(), kTol)
            << "seed " << seed;
    }
}

TEST(SimulationFusion, ZeroWidthIsIdentityPass) {
    QarpSimulator sim;
    const auto cmds = wide_circuit(1, 4, 30);
    EXPECT_TRUE(streams_equal(fuse_for_simulation(cmds, 0, local_unitary_of(sim)), cmds));
}

TEST(SimulationFusion, ConditionalGateNeverCrossesItsMeasurement) {
    // A conditional gate on wire 1 that reads cbit 0 *before* the Measure on
    // wire 0 writes it must stay before that Measure (it reads the
    // zero-initialised register and is skipped); fusing it past the write
    // would flip the outcome.  Deterministic circuit: the pinned oracle is
    // the analytic outcome.
    QarpSimulator sim;
    sim.set_fusion_max_qubits(3);
    sim.set_fusion_min_qubits(0);  // small registers: reach the dense path
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::X, 0u);
    cmds.emplace_back(GateType::H, 1u);
    cmds.push_back(make_conditional(Command(GateType::X, 1u), {0u}, {true}));
    cmds.emplace_back(GateType::H, 1u);
    cmds.push_back(make_measure(0u, 0u));
    cmds.push_back(make_measure(1u, 1u));
    // Wire 1: H·H = I → |0⟩ iff the conditional X was skipped.
    const auto r = sim.run(cmds, 2, 64, 5u);
    ASSERT_EQ(r.counts.size(), 1u);
    EXPECT_EQ(r.counts.begin()->first, 1u);  // qubit 0 = 1, qubit 1 = 0

    // The incident (2026-09-13): with nothing else on wire 1, the
    // single-qubit pass held the conditional X in its accumulator past the
    // Measure that writes cbit 0 and re-emitted it after — outcome 3, not 1.
    // Every width, the old pass (k = 1) included, must keep it before.
    std::vector<Command> bare;
    bare.emplace_back(GateType::X, 0u);
    bare.push_back(make_conditional(Command(GateType::X, 1u), {0u}, {true}));
    bare.push_back(make_measure(0u, 0u));
    bare.push_back(make_measure(1u, 1u));
    for (std::size_t k : {0u, 1u, 3u}) {
        QarpSimulator s;
        s.set_fusion_max_qubits(k);
        s.set_fusion_min_qubits(0);  // small registers: reach the dense path
        const auto rb = s.run(bare, 2, 16, 5u);
        ASSERT_EQ(rb.counts.size(), 1u) << "k=" << k;
        EXPECT_EQ(rb.counts.begin()->first, 1u) << "k=" << k;
    }

    const auto fused = fuse_for_simulation(cmds, 3, local_unitary_of(sim));
    std::size_t measure_at = fused.size(), conditional_at = fused.size();
    for (std::size_t i = 0; i < fused.size(); ++i) {
        if (fused[i].gate == GateType::Measure && fused[i].qubits[0] == 0u) measure_at = i;
        if (!fused[i].condition_bits.empty()) conditional_at = i;
    }
    ASSERT_LT(conditional_at, fused.size());
    EXPECT_LT(conditional_at, measure_at);
}

// ── Trajectory path: fused vs unfused, same seed ─────────────────────────────

TEST(SimulationFusion, TrajectoryRunIsIdenticalToUnfused) {
    // Per-shot RNG streams are seeded identically, so a fused suffix that
    // preserves every measurement's position and distribution reproduces
    // the unfused counts and cbit history exactly (amplitudes differ at
    // 1e-16, far below any draw boundary).  The unfused path is pinned to
    // analytic outcomes in test_trajectory.cpp.
    QarpSimulator plain;
    plain.set_fusion_max_qubits(0);
    plain.set_fusion_min_qubits(0);  // small registers: reach the dense path
    QarpSimulator fused;
    fused.set_fusion_max_qubits(3);
    fused.set_fusion_min_qubits(0);  // small registers: reach the dense path
    for (uint32_t seed = 0; seed < 40; ++seed) {
        const auto cmds = trajectory_circuit(seed + 6000, 4, 2, 50);
        const auto a = plain.run(cmds, 4, 48, seed);
        const auto b = fused.run(cmds, 4, 48, seed);
        EXPECT_EQ(a.counts, b.counts) << "seed " << seed;
        EXPECT_EQ(a.n_cbits, b.n_cbits) << "seed " << seed;
        EXPECT_EQ(a.cbit_history, b.cbit_history) << "seed " << seed;
    }
}

TEST(SimulationFusion, TerminalMeasurementFastPathMatchesReference) {
    // run()'s fast path applies the fused stream with terminal Measures as
    // no-ops; the sampled distribution must be |U|0⟩|².  16k shots, TV < 0.03.
    constexpr uint32_t n = 4;
    QarpSimulator sim;
    sim.set_fusion_max_qubits(3);
    sim.set_fusion_min_qubits(0);  // small registers: reach the dense path
    for (uint32_t seed = 0; seed < 5; ++seed) {
        auto cmds = wide_circuit(seed + 8000, n, 30);
        for (uint32_t q = 0; q < n; ++q) cmds.push_back(make_measure(q, q));
        const Vec ref = reference_unitary(
            std::vector<Command>(cmds.begin(), cmds.end() - n), n).col(0);
        const int shots = 16384;
        const auto r = sim.run(cmds, n, shots, seed);
        double tv = 0.0;
        for (Eigen::Index i = 0; i < ref.size(); ++i)
            tv += std::abs(std::norm(ref[i]) - r.probability(static_cast<uint64_t>(i)));
        EXPECT_LT(tv / 2.0, 0.03) << "seed " << seed;
        EXPECT_EQ(r.n_cbits, static_cast<int>(n));
    }
}

}  // namespace qarpx::test
