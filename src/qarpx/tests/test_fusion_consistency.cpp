// ── Fusion-consistency tests ──
//
// `fusion.cpp::gate_matrix()` carries its own analytic 2×2 matrices for every
// fusible single-qubit gate.  These must agree with `QarpSimulator::apply_command`
// or the two execution paths drift silently:
//
//   QarpSimulator::statevector(...)     →  uses fusion (fuses ≥2 fusible 1Q
//                                          gates into a Custom matrix)
//   QarpSimulator::unitary_matrix(...)  →  no fusion; calls apply_command directly
//
// Each test below applies ≥2 fusible single-qubit gates back-to-back so that
// fusion definitely fires, and compares the resulting amplitudes against
// `unitary_matrix` applied to |0⟩.  This was the regression path that hid the
// csim-sign bug in Rx/Ry/Rz before the conventions tests were added.

#include <gtest/gtest.h>
#include "qarpx/qarpx.h"

#include <cmath>
#include <complex>
#include <numbers>
#include <sstream>
#include <vector>

using namespace qarpx;

namespace {

using cd = std::complex<double>;
constexpr double TOL = 1e-10;
constexpr double PI  = std::numbers::pi;

::testing::AssertionResult fusion_matches_dispatch(
        const std::vector<Command>& cmds, int n_qubits, double tol = TOL) {
    QarpSimulator sim;
    sim.set_fusion_max_qubits(1);                       // pin the 1q pass, whatever the default
    auto sv_fused = sim.statevector(cmds, n_qubits);    // uses fuse_single_qubit_gates
    auto U        = sim.unitary_matrix(cmds, n_qubits); // no fusion

    // Reference statevector = U · |0⟩ = U.col(0).
    if (sv_fused.size() != static_cast<size_t>(U.rows())) {
        return ::testing::AssertionFailure() << "shape mismatch";
    }
    double max_diff = 0.0;
    for (size_t i = 0; i < sv_fused.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(sv_fused[i] - U(i, 0)));
    }
    if (max_diff < tol) return ::testing::AssertionSuccess();

    std::ostringstream oss;
    oss << "fusion vs apply_command max amplitude diff " << max_diff << " > tol " << tol;
    return ::testing::AssertionFailure() << oss.str();
}

}  // namespace

// ── 1Q no-param gates (paired with H to force fusion) ────────────────────────

TEST(FusionConsistency, X_then_H) {
    EXPECT_TRUE(fusion_matches_dispatch(
        {Command(GateType::X, 0u), Command(GateType::H, 0u)}, 1));
}
TEST(FusionConsistency, Y_then_H) {
    EXPECT_TRUE(fusion_matches_dispatch(
        {Command(GateType::Y, 0u), Command(GateType::H, 0u)}, 1));
}
TEST(FusionConsistency, Z_then_H) {
    EXPECT_TRUE(fusion_matches_dispatch(
        {Command(GateType::Z, 0u), Command(GateType::H, 0u)}, 1));
}
TEST(FusionConsistency, S_then_H) {
    EXPECT_TRUE(fusion_matches_dispatch(
        {Command(GateType::S, 0u), Command(GateType::H, 0u)}, 1));
}
TEST(FusionConsistency, Sdg_then_H) {
    EXPECT_TRUE(fusion_matches_dispatch(
        {Command(GateType::Sdg, 0u), Command(GateType::H, 0u)}, 1));
}
TEST(FusionConsistency, T_then_H) {
    EXPECT_TRUE(fusion_matches_dispatch(
        {Command(GateType::T, 0u), Command(GateType::H, 0u)}, 1));
}
TEST(FusionConsistency, Tdg_then_H) {
    EXPECT_TRUE(fusion_matches_dispatch(
        {Command(GateType::Tdg, 0u), Command(GateType::H, 0u)}, 1));
}

// ── 1Q parametric (the path that hid the csim sign bug) ──────────────────────

TEST(FusionConsistency, Rx_then_H) {
    EXPECT_TRUE(fusion_matches_dispatch(
        {Command(GateType::Rx, 0u, Param(0.7)), Command(GateType::H, 0u)}, 1));
}
TEST(FusionConsistency, Ry_then_H) {
    EXPECT_TRUE(fusion_matches_dispatch(
        {Command(GateType::Ry, 0u, Param(0.7)), Command(GateType::H, 0u)}, 1));
}
TEST(FusionConsistency, Rz_then_H) {
    EXPECT_TRUE(fusion_matches_dispatch(
        {Command(GateType::Rz, 0u, Param(0.7)), Command(GateType::H, 0u)}, 1));
}
TEST(FusionConsistency, P_then_H) {
    EXPECT_TRUE(fusion_matches_dispatch(
        {Command(GateType::P, 0u, Param(0.7)), Command(GateType::H, 0u)}, 1));
}

// ── U(θ, φ, λ) — fusible at concrete params ──────────────────────────────────

TEST(FusionConsistency, U_then_H) {
    Command u;
    u.gate = GateType::U;
    u.qubits.push_back(0);
    u.params.push_back(Param(0.31));
    u.params.push_back(Param(0.42));
    u.params.push_back(Param(0.53));
    EXPECT_TRUE(fusion_matches_dispatch({u, Command(GateType::H, 0u)}, 1));
}

// ── Multi-gate fusion run (Hadamard sandwich + rotations) ────────────────────

TEST(FusionConsistency, ChainOfManyFusibleGates) {
    // Hadamard sandwich + a few rotations.  Verifies that the accumulated
    // matrix product matches the per-gate unitary chain.
    EXPECT_TRUE(fusion_matches_dispatch({
        Command(GateType::H,  0u),
        Command(GateType::Rx, 0u, Param( 0.31)),
        Command(GateType::Rz, 0u, Param(-0.42)),
        Command(GateType::Ry, 0u, Param( 0.53)),
        Command(GateType::H,  0u),
        Command(GateType::S,  0u),
    }, 1));
}

// ── Fusion across a barrier must NOT happen (barrier is a fence) ─────────────
//
// We can't directly assert "fusion didn't fuse" via statevector alone, but we
// can check that the result still agrees with the analytic matrix product.
TEST(FusionConsistency, BarrierDoesNotCorruptResult) {
    EXPECT_TRUE(fusion_matches_dispatch({
        Command(GateType::H, 0u),
        Command(GateType::Rx, 0u, Param(0.7)),
        Command(GateType::Barrier, 0u),
        Command(GateType::Rz, 0u, Param(0.4)),
        Command(GateType::H, 0u),
    }, 1));
}

// The structural half the semantic check above cannot see: a barrier is a
// no-op to the statevector, so gates fusing across it (and the fused Custom
// landing on the far side) corrupts nothing numerically — only the fence
// contract (§16: barriers fence exactly their listed qubits).

TEST(FusionConsistency, BarrierFencesItsListedQubit) {
    // T(0) T(1) Barrier(0) T(0) T(1): q0's pair must NOT fuse; q1's must.
    const auto out = fuse_single_qubit_gates({
        Command(GateType::T, 0u),
        Command(GateType::T, 1u),
        Command(GateType::Barrier, {0u}),
        Command(GateType::T, 0u),
        Command(GateType::T, 1u),
    });

    std::vector<GateType> q0, q1;
    for (const auto& c : out) {
        if (c.gate == GateType::Barrier) continue;
        (c.qubits[0] == 0 ? q0 : q1).push_back(c.gate);
    }
    EXPECT_EQ(q0, (std::vector<GateType>{GateType::T, GateType::T}))
        << "q0's pair fused across its own barrier";
    EXPECT_EQ(q1, (std::vector<GateType>{GateType::Custom}))
        << "q1 is not fenced and must fuse";

    // The barrier sits strictly between q0's two commands.
    std::vector<GateType> wire0;
    for (const auto& c : out)
        if (c.qubits.empty() || c.qubits[0] == 0) wire0.push_back(c.gate);
    EXPECT_EQ(wire0, (std::vector<GateType>{
        GateType::T, GateType::Barrier, GateType::T}));
}

TEST(FusionConsistency, EmptyBarrierFencesTheWholeRegister) {
    // OpenQASM `barrier;` — no listed qubits — fences every wire.
    Command bare;
    bare.gate = GateType::Barrier;
    const auto out = fuse_single_qubit_gates({
        Command(GateType::T, 0u),
        Command(GateType::T, 1u),
        bare,
        Command(GateType::T, 0u),
        Command(GateType::T, 1u),
    });
    for (const auto& c : out) {
        EXPECT_NE(c.gate, GateType::Custom)
            << "a pair fused across an all-register barrier";
    }
}
