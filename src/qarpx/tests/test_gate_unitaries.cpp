// ── Per-gate unitary tests against qarp_conventions.md ──
//
// One TEST per GateType.  Each builds the gate, transpiles through
// native_gateset() so that any non-natively-dispatched gate (iSWAP, iSWAPdg,
// ECR, CSWAP, …) reduces to gates the simulator can execute, then compares
// the resulting 2^n × 2^n unitary against the analytic matrix specified by
// the conventions document.
//
// Helpers live in gate_test_helpers.h.

#include "gate_test_helpers.h"

using namespace qarpx;
using namespace qarpx::test;

// ── §2.1 Pauli + Hadamard ────────────────────────────────────────────────────

TEST(GateUnitary, X) { EXPECT_TRUE(expect_unitary_close(build_unitary(Command(GateType::X, 0u), 1), analytic_x())); }
TEST(GateUnitary, Y) { EXPECT_TRUE(expect_unitary_close(build_unitary(Command(GateType::Y, 0u), 1), analytic_y())); }
TEST(GateUnitary, Z) { EXPECT_TRUE(expect_unitary_close(build_unitary(Command(GateType::Z, 0u), 1), analytic_z())); }
TEST(GateUnitary, H) { EXPECT_TRUE(expect_unitary_close(build_unitary(Command(GateType::H, 0u), 1), analytic_h())); }

// ── §2.2 Discrete phase gates ────────────────────────────────────────────────

TEST(GateUnitary, S)   { EXPECT_TRUE(expect_unitary_close(build_unitary(Command(GateType::S,   0u), 1), analytic_s())); }
TEST(GateUnitary, Sdg) { EXPECT_TRUE(expect_unitary_close(build_unitary(Command(GateType::Sdg, 0u), 1), analytic_sdg())); }
TEST(GateUnitary, T)   { EXPECT_TRUE(expect_unitary_close(build_unitary(Command(GateType::T,   0u), 1), analytic_t())); }
TEST(GateUnitary, Tdg) { EXPECT_TRUE(expect_unitary_close(build_unitary(Command(GateType::Tdg, 0u), 1), analytic_tdg())); }

// ── §2.6 sqrt-X pair and identity ────────────────────────────────────────────

TEST(GateUnitary, SX)   { EXPECT_TRUE(expect_unitary_close(build_unitary(Command(GateType::SX,   0u), 1), analytic_sx())); }
TEST(GateUnitary, SXdg) { EXPECT_TRUE(expect_unitary_close(build_unitary(Command(GateType::SXdg, 0u), 1), analytic_sxdg())); }
TEST(GateUnitary, Id)   { EXPECT_TRUE(expect_unitary_close(build_unitary(Command(GateType::Id,   0u), 1), Mat::Identity(2, 2))); }

// SX is the principal square root of X, not X itself.
TEST(GateUnitary, SX_squared_equals_X) {
    std::vector<Command> cmds{Command(GateType::SX, 0u), Command(GateType::SX, 0u)};
    EXPECT_TRUE(expect_unitary_close(build_unitary(cmds, 1), analytic_x()));
}

// All three are in native_gateset(): the simulator must dispatch them with no
// lowering at all (a transpile-first path would hide a missing kernel arm).
TEST(GateUnitary, SqrtXPairAndIdDispatchNatively) {
    QarpSimulator sim;
    EXPECT_TRUE(expect_unitary_close(sim.unitary_matrix({Command(GateType::SX,   0u)}, 1), analytic_sx()));
    EXPECT_TRUE(expect_unitary_close(sim.unitary_matrix({Command(GateType::SXdg, 0u)}, 1), analytic_sxdg()));
    EXPECT_TRUE(expect_unitary_close(sim.unitary_matrix({Command(GateType::Id,   0u)}, 1), Mat::Identity(2, 2)));
}

// §2.2 also asserts S ≡ P(π/2) and T ≡ P(π/4) exactly.
TEST(GateUnitary, S_equals_P_pi_over_2) {
    EXPECT_TRUE(expect_unitary_close(
        build_unitary(Command(GateType::P, 0u, Param( PI / 2.0)), 1), analytic_s()));
}
TEST(GateUnitary, Sdg_equals_P_minus_pi_over_2) {
    EXPECT_TRUE(expect_unitary_close(
        build_unitary(Command(GateType::P, 0u, Param(-PI / 2.0)), 1), analytic_sdg()));
}
TEST(GateUnitary, T_equals_P_pi_over_4) {
    EXPECT_TRUE(expect_unitary_close(
        build_unitary(Command(GateType::P, 0u, Param( PI / 4.0)), 1), analytic_t()));
}

// ── §2.3 Rotations: exp(-iθ/2 P) ─────────────────────────────────────────────

TEST(GateUnitary, Rx_at_various_angles) {
    for (double th : ANGLE_SAMPLES)
        EXPECT_TRUE(expect_unitary_close(
            build_unitary(Command(GateType::Rx, 0u, Param(th)), 1), analytic_rx(th))) << "θ=" << th;
}
TEST(GateUnitary, Ry_at_various_angles) {
    for (double th : ANGLE_SAMPLES)
        EXPECT_TRUE(expect_unitary_close(
            build_unitary(Command(GateType::Ry, 0u, Param(th)), 1), analytic_ry(th))) << "θ=" << th;
}
TEST(GateUnitary, Rz_at_various_angles) {
    for (double th : ANGLE_SAMPLES)
        EXPECT_TRUE(expect_unitary_close(
            build_unitary(Command(GateType::Rz, 0u, Param(th)), 1), analytic_rz(th))) << "θ=" << th;
}

// §2.3 also asserts `Rz(θ) = e^{-iθ/2} · P(θ)` (up to a global phase).
TEST(GateUnitary, Rz_equals_PhaseTimesP) {
    const double th = 1.234;
    EXPECT_TRUE(expect_unitary_close(
        build_unitary(Command(GateType::Rz, 0u, Param(th)), 1),
        ei(-th / 2.0) * analytic_p(th)));
}

// ── §2.4 Phase gate P(θ) ─────────────────────────────────────────────────────

TEST(GateUnitary, P_at_various_angles) {
    for (double th : ANGLE_SAMPLES)
        EXPECT_TRUE(expect_unitary_close(
            build_unitary(Command(GateType::P, 0u, Param(th)), 1), analytic_p(th))) << "θ=" << th;
}

// ── §2.5 General single-qubit U(θ, φ, λ) ─────────────────────────────────────

TEST(GateUnitary, U_at_various_params) {
    const double angles[][3] = {
        {0.0, 0.0, 0.0},
        {PI/2, 0.0, 0.0},
        {0.31, 0.42, 0.53},
        {PI, PI/3, -PI/4},
        {-0.7, 1.234, 2.0},
    };
    for (const auto& a : angles) {
        Command c;
        c.gate = GateType::U;
        c.qubits.push_back(0);
        c.params.push_back(Param(a[0]));
        c.params.push_back(Param(a[1]));
        c.params.push_back(Param(a[2]));
        EXPECT_TRUE(expect_unitary_close(build_unitary(c, 1), analytic_u(a[0], a[1], a[2])))
            << "θ=" << a[0] << " φ=" << a[1] << " λ=" << a[2];
    }
}

// ── §3.1 Two-qubit controlled gates (q0 = control, q1 = target) ──────────────

TEST(GateUnitary, CX) {
    EXPECT_TRUE(expect_unitary_close(
        build_unitary(Command(GateType::CX, 0u, 1u), 2),
        permutation_unitary(2, [](int i) { return bit(i,0) ? flip(i,1) : i; })));
}
TEST(GateUnitary, CY) {
    EXPECT_TRUE(expect_unitary_close(
        build_unitary(Command(GateType::CY, 0u, 1u), 2),
        controlled_1q(analytic_y(), 0, 1, 2)));
}
TEST(GateUnitary, CZ) {
    EXPECT_TRUE(expect_unitary_close(
        build_unitary(Command(GateType::CZ, 0u, 1u), 2),
        diag_unitary({1, 1, 1, -1})));
}
TEST(GateUnitary, CRx) {
    const double th = 0.7;
    EXPECT_TRUE(expect_unitary_close(
        build_unitary(Command(GateType::CRx, 0u, 1u, Param(th)), 2),
        controlled_1q(analytic_rx(th), 0, 1, 2)));
}
TEST(GateUnitary, CRy) {
    const double th = 0.7;
    EXPECT_TRUE(expect_unitary_close(
        build_unitary(Command(GateType::CRy, 0u, 1u, Param(th)), 2),
        controlled_1q(analytic_ry(th), 0, 1, 2)));
}
TEST(GateUnitary, CRz) {
    const double th = 0.7;
    EXPECT_TRUE(expect_unitary_close(
        build_unitary(Command(GateType::CRz, 0u, 1u, Param(th)), 2),
        controlled_1q(analytic_rz(th), 0, 1, 2)));
}
TEST(GateUnitary, CP) {
    const double th = 0.7;
    EXPECT_TRUE(expect_unitary_close(
        build_unitary(Command(GateType::CP, 0u, 1u, Param(th)), 2),
        controlled_1q(analytic_p(th), 0, 1, 2)));
}

// CU(c, t, θ, φ, λ, γ) applies e^{iγ} · U(θ, φ, λ) on the |1⟩_c subspace.
TEST(GateUnitary, CU) {
    const double th = 0.31, phi = 0.42, lam = 0.53, gamma = 0.64;
    Command c;
    c.gate = GateType::CU;
    c.qubits.push_back(0);
    c.qubits.push_back(1);
    c.params.push_back(Param(th));
    c.params.push_back(Param(phi));
    c.params.push_back(Param(lam));
    c.params.push_back(Param(gamma));
    EXPECT_TRUE(expect_unitary_close(
        build_unitary(c, 2),
        controlled_1q(ei(gamma) * analytic_u(th, phi, lam), 0, 1, 2)));
}

// ── §3.1 Controlled Clifford singles (CH, CS, CSdg, CSX, CSXdg) ─────────────

TEST(GateUnitary, CH) {
    EXPECT_TRUE(expect_unitary_close(
        build_unitary(Command(GateType::CH, 0u, 1u), 2),
        controlled_1q(analytic_h(), 0, 1, 2)));
}
TEST(GateUnitary, CS) {
    EXPECT_TRUE(expect_unitary_close(
        build_unitary(Command(GateType::CS, 0u, 1u), 2),
        controlled_1q(analytic_s(), 0, 1, 2)));
}
TEST(GateUnitary, CSdg) {
    EXPECT_TRUE(expect_unitary_close(
        build_unitary(Command(GateType::CSdg, 0u, 1u), 2),
        controlled_1q(analytic_sdg(), 0, 1, 2)));
}
TEST(GateUnitary, CSX) {
    EXPECT_TRUE(expect_unitary_close(
        build_unitary(Command(GateType::CSX, 0u, 1u), 2),
        controlled_1q(analytic_sx(), 0, 1, 2)));
}
TEST(GateUnitary, CSXdg) {
    EXPECT_TRUE(expect_unitary_close(
        build_unitary(Command(GateType::CSXdg, 0u, 1u), 2),
        controlled_1q(analytic_sxdg(), 0, 1, 2)));
}
// Control on the higher qubit: catches a transposed-operand kernel arm.
TEST(GateUnitary, CSX_ControlOnQ1) {
    EXPECT_TRUE(expect_unitary_close(
        build_unitary(Command(GateType::CSX, 1u, 0u), 2),
        controlled_1q(analytic_sx(), 1, 0, 2)));
}
TEST(GateUnitary, CSX_squared_equals_CX) {
    std::vector<Command> cmds{Command(GateType::CSX, 0u, 1u), Command(GateType::CSX, 0u, 1u)};
    EXPECT_TRUE(expect_unitary_close(build_unitary(cmds, 2), controlled_1q(analytic_x(), 0, 1, 2)));
}
TEST(GateUnitary, ControlledCliffordSinglesDispatchNatively) {
    QarpSimulator sim;
    EXPECT_TRUE(expect_unitary_close(sim.unitary_matrix({Command(GateType::CH,    0u, 1u)}, 2), controlled_1q(analytic_h(),    0, 1, 2)));
    EXPECT_TRUE(expect_unitary_close(sim.unitary_matrix({Command(GateType::CS,    0u, 1u)}, 2), controlled_1q(analytic_s(),    0, 1, 2)));
    EXPECT_TRUE(expect_unitary_close(sim.unitary_matrix({Command(GateType::CSdg,  0u, 1u)}, 2), controlled_1q(analytic_sdg(),  0, 1, 2)));
    EXPECT_TRUE(expect_unitary_close(sim.unitary_matrix({Command(GateType::CSX,   0u, 1u)}, 2), controlled_1q(analytic_sx(),   0, 1, 2)));
    EXPECT_TRUE(expect_unitary_close(sim.unitary_matrix({Command(GateType::CSXdg, 0u, 1u)}, 2), controlled_1q(analytic_sxdg(), 0, 1, 2)));
}

// ── §3.2 Symmetric two-qubit gates ───────────────────────────────────────────

TEST(GateUnitary, SWAP) {
    EXPECT_TRUE(expect_unitary_close(
        build_unitary(Command(GateType::SWAP, 0u, 1u), 2),
        permutation_unitary(2, [](int i) {
            const int a = bit(i, 0), b = bit(i, 1);
            return (b << 0) | (a << 1);
        })));
}

TEST(GateUnitary, iSWAP) {
    Mat expected(4, 4);
    expected << 1, 0,        0,        0,
                0, 0,        cd{0, 1}, 0,
                0, cd{0, 1}, 0,        0,
                0, 0,        0,        1;
    EXPECT_TRUE(expect_unitary_close(build_unitary(Command(GateType::iSWAP, 0u, 1u), 2), expected));
}

TEST(GateUnitary, iSWAPdg) {
    Mat expected(4, 4);
    expected << 1, 0,         0,         0,
                0, 0,         cd{0,-1},  0,
                0, cd{0,-1},  0,         0,
                0, 0,         0,         1;
    EXPECT_TRUE(expect_unitary_close(build_unitary(Command(GateType::iSWAPdg, 0u, 1u), 2), expected));
}

// ECR canonical matrix in qarpx LSB-first basis (q0 is the LSB).  Equivalent
// to Qiskit's ECRGate matrix (also LSB-first).
TEST(GateUnitary, ECR) {
    const double s = 1.0 / std::sqrt(2.0);
    Mat expected(4, 4);
    expected <<
        cd{ 0, 0},  cd{ s, 0},  cd{ 0, 0},  cd{ 0, s},
        cd{ s, 0},  cd{ 0, 0},  cd{ 0,-s},  cd{ 0, 0},
        cd{ 0, 0},  cd{ 0, s},  cd{ 0, 0},  cd{ s, 0},
        cd{ 0,-s},  cd{ 0, 0},  cd{ s, 0},  cd{ 0, 0};
    EXPECT_TRUE(expect_unitary_close(build_unitary(Command(GateType::ECR, 0u, 1u), 2), expected));
}

// ── §3.3 Symmetric two-qubit rotations: exp(-iθ/2 P⊗P) ───────────────────────

TEST(GateUnitary, RZZ) {
    const double th = 0.7;
    EXPECT_TRUE(expect_unitary_close(
        build_unitary(Command(GateType::RZZ, 0u, 1u, Param(th)), 2),
        diag_unitary({ei(-th/2), ei(th/2), ei(th/2), ei(-th/2)})));
}

TEST(GateUnitary, RXX) {
    const double th = 0.7;
    Mat HH  = kron(analytic_h(), analytic_h());
    Mat rzz = diag_unitary({ei(-th/2), ei(th/2), ei(th/2), ei(-th/2)});
    EXPECT_TRUE(expect_unitary_close(
        build_unitary(Command(GateType::RXX, 0u, 1u, Param(th)), 2),
        HH * rzz * HH));
}

TEST(GateUnitary, RYY) {
    const double th = 0.7;
    Mat RxP = kron(analytic_rx( PI / 2.0), analytic_rx( PI / 2.0));
    Mat RxN = kron(analytic_rx(-PI / 2.0), analytic_rx(-PI / 2.0));
    Mat rzz = diag_unitary({ei(-th/2), ei(th/2), ei(th/2), ei(-th/2)});
    EXPECT_TRUE(expect_unitary_close(
        build_unitary(Command(GateType::RYY, 0u, 1u, Param(th)), 2),
        RxP * rzz * RxN));
}

// ── §4 Three-qubit gates ─────────────────────────────────────────────────────

TEST(GateUnitary, CCX) {
    Command ccx;
    ccx.gate = GateType::CCX;
    ccx.qubits.push_back(0);
    ccx.qubits.push_back(1);
    ccx.qubits.push_back(2);
    EXPECT_TRUE(expect_unitary_close(
        build_unitary(ccx, 3),
        permutation_unitary(3, [](int i) {
            return (bit(i, 0) && bit(i, 1)) ? flip(i, 2) : i;
        })));
}

TEST(GateUnitary, CSWAP) {
    Command cswap;
    cswap.gate = GateType::CSWAP;
    cswap.qubits.push_back(0);
    cswap.qubits.push_back(1);
    cswap.qubits.push_back(2);
    EXPECT_TRUE(expect_unitary_close(
        build_unitary(cswap, 3),
        permutation_unitary(3, [](int i) {
            if (!bit(i, 0)) return i;
            const int b1 = bit(i, 1), b2 = bit(i, 2);
            return (i & ~0b110) | (b2 << 1) | (b1 << 2);
        })));
}

// ── §5 Multi-controlled MCZ ──────────────────────────────────────────────────

TEST(GateUnitary, MCZ_2q) {
    Command mcz;
    mcz.gate = GateType::MCZ;
    mcz.qubits.push_back(0);
    mcz.qubits.push_back(1);
    EXPECT_TRUE(expect_unitary_close(
        build_unitary(mcz, 2),
        diag_unitary({1, 1, 1, -1})));
}

TEST(GateUnitary, MCZ_3q) {
    Command mcz;
    mcz.gate = GateType::MCZ;
    mcz.qubits.push_back(0);
    mcz.qubits.push_back(1);
    mcz.qubits.push_back(2);
    std::vector<cd> d(8, cd{1, 0});
    d[7] = -1;
    EXPECT_TRUE(expect_unitary_close(build_unitary(mcz, 3), diag_unitary(d)));
}

TEST(GateUnitary, MCZ_4q) {
    Command mcz;
    mcz.gate = GateType::MCZ;
    for (uint32_t q = 0; q < 4; ++q) mcz.qubits.push_back(q);
    std::vector<cd> d(16, cd{1, 0});
    d[15] = -1;
    EXPECT_TRUE(expect_unitary_close(build_unitary(mcz, 4), diag_unitary(d)));
}

// ── §6 Zero-qubit GPhase ─────────────────────────────────────────────────────

TEST(GateUnitary, GPhase) {
    const double th = 0.7;
    Command g;
    g.gate = GateType::GPhase;
    g.params.push_back(Param(th));
    EXPECT_TRUE(expect_unitary_close(build_unitary(g, 1), ei(th) * Mat::Identity(2, 2)));
}

// ── §7 Special / non-unitary commands ────────────────────────────────────────

TEST(GateUnitary, Barrier_is_noop) {
    EXPECT_TRUE(expect_unitary_close(
        build_unitary(Command(GateType::Barrier, 0u), 1),
        Mat::Identity(2, 2)));
}
