// ── Dagger round-trip tests against qarp_conventions.md §9 ──
//
// For every unitary GateType, build [G, G.dagger()] and verify the resulting
// unitary equals identity within tolerance.  Helpers in gate_test_helpers.h.

#include "gate_test_helpers.h"

using namespace qarpx;
using namespace qarpx::test;

namespace {

// Verify that [cmd, cmd.dagger()] composes to the identity unitary.
::testing::AssertionResult round_trips_to_identity(
        const Command& cmd, int n_qubits, double tol = DEFAULT_TOL) {
    Mat M = build_unitary({cmd, cmd.dagger()}, n_qubits);
    return expect_unitary_close(M, Mat::Identity(M.rows(), M.cols()), tol);
}

// Convenience builder for parametric n-qubit gates.
Command make(GateType g, std::initializer_list<uint32_t> qubits,
             std::initializer_list<double> params = {}) {
    Command c;
    c.gate = g;
    for (auto q : qubits) c.qubits.push_back(q);
    for (auto p : params) c.params.push_back(Param(p));
    return c;
}

}  // namespace

// ── Self-adjoint single-qubit gates (G† = G) ─────────────────────────────────

TEST(DaggerRoundTrip, X) { EXPECT_TRUE(round_trips_to_identity(Command(GateType::X, 0u), 1)); }
TEST(DaggerRoundTrip, Y) { EXPECT_TRUE(round_trips_to_identity(Command(GateType::Y, 0u), 1)); }
TEST(DaggerRoundTrip, Z) { EXPECT_TRUE(round_trips_to_identity(Command(GateType::Z, 0u), 1)); }
TEST(DaggerRoundTrip, H) { EXPECT_TRUE(round_trips_to_identity(Command(GateType::H, 0u), 1)); }
TEST(DaggerRoundTrip, Id) { EXPECT_TRUE(round_trips_to_identity(Command(GateType::Id, 0u), 1)); }

// ── Named-inverse pairs ──────────────────────────────────────────────────────

TEST(DaggerRoundTrip, S)   { EXPECT_TRUE(round_trips_to_identity(Command(GateType::S,   0u), 1)); }
TEST(DaggerRoundTrip, Sdg) { EXPECT_TRUE(round_trips_to_identity(Command(GateType::Sdg, 0u), 1)); }
TEST(DaggerRoundTrip, T)   { EXPECT_TRUE(round_trips_to_identity(Command(GateType::T,   0u), 1)); }
TEST(DaggerRoundTrip, Tdg) { EXPECT_TRUE(round_trips_to_identity(Command(GateType::Tdg, 0u), 1)); }
TEST(DaggerRoundTrip, SX)   { EXPECT_TRUE(round_trips_to_identity(Command(GateType::SX,   0u), 1)); }
TEST(DaggerRoundTrip, SXdg) { EXPECT_TRUE(round_trips_to_identity(Command(GateType::SXdg, 0u), 1)); }

// ── Single-qubit parametric (per-param negation) ─────────────────────────────

TEST(DaggerRoundTrip, Rx) {
    for (double th : ANGLE_SAMPLES)
        EXPECT_TRUE(round_trips_to_identity(make(GateType::Rx, {0}, {th}), 1)) << "θ=" << th;
}
TEST(DaggerRoundTrip, Ry) {
    for (double th : ANGLE_SAMPLES)
        EXPECT_TRUE(round_trips_to_identity(make(GateType::Ry, {0}, {th}), 1)) << "θ=" << th;
}
TEST(DaggerRoundTrip, Rz) {
    for (double th : ANGLE_SAMPLES)
        EXPECT_TRUE(round_trips_to_identity(make(GateType::Rz, {0}, {th}), 1)) << "θ=" << th;
}
TEST(DaggerRoundTrip, P) {
    for (double th : ANGLE_SAMPLES)
        EXPECT_TRUE(round_trips_to_identity(make(GateType::P, {0}, {th}), 1)) << "θ=" << th;
}

// ── §2.5 General U(θ, φ, λ): dagger = U(-θ, -λ, -φ) ──────────────────────────

TEST(DaggerRoundTrip, U) {
    const double sets[][3] = {
        {0.0, 0.0, 0.0},
        {PI/2, 0.0, 0.0},
        {0.31, 0.42, 0.53},
        {PI, PI/3, -PI/4},
        {-0.7, 1.234, 2.0},
    };
    for (const auto& a : sets)
        EXPECT_TRUE(round_trips_to_identity(make(GateType::U, {0}, {a[0], a[1], a[2]}), 1))
            << "θ=" << a[0] << " φ=" << a[1] << " λ=" << a[2];
}

// ── §3.1 Self-adjoint two-qubit ──────────────────────────────────────────────

TEST(DaggerRoundTrip, CX)   { EXPECT_TRUE(round_trips_to_identity(Command(GateType::CX,   0u, 1u), 2)); }
TEST(DaggerRoundTrip, CY)   { EXPECT_TRUE(round_trips_to_identity(Command(GateType::CY,   0u, 1u), 2)); }
TEST(DaggerRoundTrip, CZ)   { EXPECT_TRUE(round_trips_to_identity(Command(GateType::CZ,   0u, 1u), 2)); }
TEST(DaggerRoundTrip, SWAP) { EXPECT_TRUE(round_trips_to_identity(Command(GateType::SWAP, 0u, 1u), 2)); }
TEST(DaggerRoundTrip, ECR)  { EXPECT_TRUE(round_trips_to_identity(Command(GateType::ECR,  0u, 1u), 2)); }
TEST(DaggerRoundTrip, CH)   { EXPECT_TRUE(round_trips_to_identity(Command(GateType::CH,   0u, 1u), 2)); }

// ── §3.2 iSWAP ↔ iSWAPdg named pair ──────────────────────────────────────────

TEST(DaggerRoundTrip, iSWAP)   { EXPECT_TRUE(round_trips_to_identity(Command(GateType::iSWAP,   0u, 1u), 2)); }
TEST(DaggerRoundTrip, iSWAPdg) { EXPECT_TRUE(round_trips_to_identity(Command(GateType::iSWAPdg, 0u, 1u), 2)); }

// §9 named-inverse pairs among the controlled Clifford singles.
TEST(DaggerRoundTrip, CS)    { EXPECT_TRUE(round_trips_to_identity(Command(GateType::CS,    0u, 1u), 2)); }
TEST(DaggerRoundTrip, CSdg)  { EXPECT_TRUE(round_trips_to_identity(Command(GateType::CSdg,  0u, 1u), 2)); }
TEST(DaggerRoundTrip, CSX)   { EXPECT_TRUE(round_trips_to_identity(Command(GateType::CSX,   0u, 1u), 2)); }
TEST(DaggerRoundTrip, CSXdg) { EXPECT_TRUE(round_trips_to_identity(Command(GateType::CSXdg, 0u, 1u), 2)); }

// ── §3.1 Two-qubit parametric (per-param negation) ───────────────────────────

TEST(DaggerRoundTrip, CRx) {
    for (double th : ANGLE_SAMPLES)
        EXPECT_TRUE(round_trips_to_identity(make(GateType::CRx, {0, 1}, {th}), 2)) << "θ=" << th;
}
TEST(DaggerRoundTrip, CRy) {
    for (double th : ANGLE_SAMPLES)
        EXPECT_TRUE(round_trips_to_identity(make(GateType::CRy, {0, 1}, {th}), 2)) << "θ=" << th;
}
TEST(DaggerRoundTrip, CRz) {
    for (double th : ANGLE_SAMPLES)
        EXPECT_TRUE(round_trips_to_identity(make(GateType::CRz, {0, 1}, {th}), 2)) << "θ=" << th;
}
TEST(DaggerRoundTrip, CP) {
    for (double th : ANGLE_SAMPLES)
        EXPECT_TRUE(round_trips_to_identity(make(GateType::CP, {0, 1}, {th}), 2)) << "θ=" << th;
}

// ── §3.3 Symmetric two-qubit Pauli rotations ─────────────────────────────────

TEST(DaggerRoundTrip, RZZ) {
    for (double th : ANGLE_SAMPLES)
        EXPECT_TRUE(round_trips_to_identity(make(GateType::RZZ, {0, 1}, {th}), 2)) << "θ=" << th;
}
TEST(DaggerRoundTrip, RXX) {
    for (double th : ANGLE_SAMPLES)
        EXPECT_TRUE(round_trips_to_identity(make(GateType::RXX, {0, 1}, {th}), 2)) << "θ=" << th;
}
TEST(DaggerRoundTrip, RYY) {
    for (double th : ANGLE_SAMPLES)
        EXPECT_TRUE(round_trips_to_identity(make(GateType::RYY, {0, 1}, {th}), 2)) << "θ=" << th;
}

// ── CU(θ, φ, λ, γ): dagger = CU(-θ, -λ, -φ, -γ) ──────────────────────────────

TEST(DaggerRoundTrip, CU) {
    const double sets[][4] = {
        {0.0,  0.0, 0.0, 0.0},
        {0.31, 0.42, 0.53, 0.0},
        {0.31, 0.42, 0.53, 0.64},
        {PI,   PI/3, -PI/4, -PI/5},
        {-0.7, 1.234, 2.0, 0.5},
    };
    for (const auto& a : sets)
        EXPECT_TRUE(round_trips_to_identity(
            make(GateType::CU, {0, 1}, {a[0], a[1], a[2], a[3]}), 2))
            << "θ=" << a[0] << " φ=" << a[1] << " λ=" << a[2] << " γ=" << a[3];
}

// ── §4 Three-qubit self-adjoint ──────────────────────────────────────────────

TEST(DaggerRoundTrip, CCX) {
    Command c;
    c.gate = GateType::CCX;
    c.qubits.push_back(0);
    c.qubits.push_back(1);
    c.qubits.push_back(2);
    EXPECT_TRUE(round_trips_to_identity(c, 3));
}

TEST(DaggerRoundTrip, CSWAP) {
    Command c;
    c.gate = GateType::CSWAP;
    c.qubits.push_back(0);
    c.qubits.push_back(1);
    c.qubits.push_back(2);
    EXPECT_TRUE(round_trips_to_identity(c, 3));
}

// ── §5 MCZ (self-adjoint) ────────────────────────────────────────────────────

TEST(DaggerRoundTrip, MCZ_2q) {
    Command c; c.gate = GateType::MCZ;
    for (uint32_t q = 0; q < 2; ++q) c.qubits.push_back(q);
    EXPECT_TRUE(round_trips_to_identity(c, 2));
}
TEST(DaggerRoundTrip, MCZ_3q) {
    Command c; c.gate = GateType::MCZ;
    for (uint32_t q = 0; q < 3; ++q) c.qubits.push_back(q);
    EXPECT_TRUE(round_trips_to_identity(c, 3));
}
TEST(DaggerRoundTrip, MCZ_4q) {
    Command c; c.gate = GateType::MCZ;
    for (uint32_t q = 0; q < 4; ++q) c.qubits.push_back(q);
    EXPECT_TRUE(round_trips_to_identity(c, 4));
}

// ── §6 GPhase: dagger negates the angle ──────────────────────────────────────

TEST(DaggerRoundTrip, GPhase) {
    for (double th : ANGLE_SAMPLES) {
        Command g;
        g.gate = GateType::GPhase;
        g.params.push_back(Param(th));
        EXPECT_TRUE(round_trips_to_identity(g, 1)) << "θ=" << th;
    }
}
