// ── ControlledBlock unitary contract tests ──────────────────────────────────
//
// qarp_conventions.md §4 specifies that ``ControlledBlock(U, num_controls=n,
// ctrl_state=s)`` produces the unitary that applies ``U`` to the inner qubits
// when every control matches ``s``, and identity otherwise.  Equivalently, in
// the qarpx LSB convention with controls at q0..q_{n-1} and inner targets at
// q_n..q_{n+inner.n_qubits-1}, the matrix is block-diagonal:
//
//   M(s)[i, j] = δ(ctrl_bits(i) = s) · δ(ctrl_bits(j) = s) · U[inner_idx(i), inner_idx(j)]
//              + δ(ctrl_bits(i) ≠ s) · δ(i = j)
//
// These tests pin that contract for every supported single-control case
// (X, Y, Z, Rx, Ry, Rz, P, U, CX, SWAP, GPhase, H, S, Sdg, T, Tdg) and the
// multi-control basis set (X, Y, Z, Rx, Ry, Rz, P, GPhase) at n_ctrls ∈ {2, 3}.
//
// They guard against sign errors in the controlled decompositions — e.g. a
// C-H that produces ``(X − Z)/√2`` instead of ``H = (X + Z)/√2`` from a
// sign-swap in the Ry-sandwich decomposition.

#include "gate_test_helpers.h"
#include "qarpx/qarpx.h"
#include "qarpx/block/controlled_block.h"

#include <memory>

using namespace qarpx;

// Test helper: allocate a block as an intrusive ref<Block> — block trees are
// reference counted through nanobind's intrusive ref<T>.
namespace {
template <class T, class... Args>
ref<Block> make_block(Args&&... args) {
    return ref<Block>(new T(std::forward<Args>(args)...));
}
}  // namespace
using namespace qarpx::test;

namespace {

/// Build a single-gate inner ``SimpleBlock``, wrap it in ``ControlledBlock``,
/// transpile-then-simulate to get the controlled unitary.
Mat controlled_unitary(
    std::function<void(SimpleBlock&)> emit_inner,
    uint32_t inner_n_qubits,
    uint32_t num_controls,
    std::vector<bool> ctrl_state)
{
    // Concrete ref<SimpleBlock> so emit_inner (takes SimpleBlock&) binds; the
    // ControlledBlock ctor takes ref<Block>, hence the explicit upcast.
    ref<SimpleBlock> inner(new SimpleBlock(inner_n_qubits, "inner"));
    emit_inner(*inner);
    inner->build();

    ControlledBlock ctrl(ref<Block>(inner.get()), num_controls, ctrl_state, "ctrl");
    ctrl.build();

    return build_unitary(ctrl.flatten(),
                         static_cast<int>(num_controls + inner_n_qubits));
}

/// Reference matrix for the controlled contract: apply ``inner`` to the
/// inner-qubit register when the control register matches ``ctrl_state``,
/// identity otherwise.  Controls are at q0..q_{nc-1} (LSB first), inner
/// qubits at q_nc..q_{nc+ni-1}.
Mat controlled_reference(
    const Mat& inner,
    uint32_t num_controls,
    std::vector<bool> ctrl_state,
    uint32_t inner_n_qubits)
{
    const int total_q = static_cast<int>(num_controls + inner_n_qubits);
    const int dim = 1 << total_q;
    const int inner_dim = 1 << inner_n_qubits;
    const int ctrl_mask = (1 << num_controls) - 1;

    int target_ctrl_bits = 0;
    for (uint32_t k = 0; k < num_controls; ++k)
        if (ctrl_state[k]) target_ctrl_bits |= (1 << k);

    Mat M = Mat::Zero(dim, dim);
    for (int j = 0; j < dim; ++j) {
        const int ctrl_bits_j = j & ctrl_mask;
        const int inner_j = j >> num_controls;
        if (ctrl_bits_j != target_ctrl_bits) {
            M(j, j) = 1;  // identity on the non-matching control sector
            continue;
        }
        for (int inner_i = 0; inner_i < inner_dim; ++inner_i) {
            const int i = (inner_i << num_controls) | target_ctrl_bits;
            M(i, j) = inner(inner_i, inner_j);
        }
    }
    return M;
}

}  // namespace

// ── Single-control: parameterless gates ─────────────────────────────────────

#define TEST_SINGLE_CTRL_1Q_GATE(NAME, EMIT, ANALYTIC)                                 \
    TEST(ControlledBlockUnitary, NAME ## _CtrlTrue) {                                  \
        EXPECT_TRUE(expect_unitary_close(                                              \
            controlled_unitary([](SimpleBlock& b) { EMIT; }, 1, 1, {true}),            \
            controlled_reference((ANALYTIC), 1, {true}, 1)));                          \
    }                                                                                  \
    TEST(ControlledBlockUnitary, NAME ## _CtrlFalse) {                                 \
        EXPECT_TRUE(expect_unitary_close(                                              \
            controlled_unitary([](SimpleBlock& b) { EMIT; }, 1, 1, {false}),           \
            controlled_reference((ANALYTIC), 1, {false}, 1)));                         \
    }

TEST_SINGLE_CTRL_1Q_GATE(X,   b.x(0),   analytic_x())
TEST_SINGLE_CTRL_1Q_GATE(Y,   b.y(0),   analytic_y())
TEST_SINGLE_CTRL_1Q_GATE(Z,   b.z(0),   analytic_z())
TEST_SINGLE_CTRL_1Q_GATE(H,   b.h(0),   analytic_h())
TEST_SINGLE_CTRL_1Q_GATE(S,   b.s(0),   analytic_s())
TEST_SINGLE_CTRL_1Q_GATE(Sdg, b.sdg(0), analytic_sdg())
TEST_SINGLE_CTRL_1Q_GATE(T,   b.t(0),   analytic_t())
TEST_SINGLE_CTRL_1Q_GATE(Tdg, b.tdg(0), analytic_tdg())
TEST_SINGLE_CTRL_1Q_GATE(SX,   b.sx(0),   analytic_sx())
TEST_SINGLE_CTRL_1Q_GATE(SXdg, b.sxdg(0), analytic_sxdg())
TEST_SINGLE_CTRL_1Q_GATE(Id,   b.id(0),   Mat::Identity(2, 2))

// Controlling a controlled Clifford single goes through the generic
// lower-and-recurse fallback: C(CH) etc. must stay exact.
TEST(ControlledBlockUnitary, CH_AsInnerGate_CtrlTrue) {
    EXPECT_TRUE(expect_unitary_close(
        controlled_unitary([](SimpleBlock& b) { b.ch(0, 1); }, 2, 1, {true}),
        controlled_reference(controlled_1q(analytic_h(), 0, 1, 2), 1, {true}, 2)));
}
TEST(ControlledBlockUnitary, CSX_AsInnerGate_CtrlTrue) {
    EXPECT_TRUE(expect_unitary_close(
        controlled_unitary([](SimpleBlock& b) { b.csx(0, 1); }, 2, 1, {true}),
        controlled_reference(controlled_1q(analytic_sx(), 0, 1, 2), 1, {true}, 2)));
}

// Declared observable change: the n = 1 path emits the controlled gate itself,
// one command, the way X -> CX already does.
TEST(ControlledBlockUnitary, SingleControlledCliffordSinglesFlattenToOneGate) {
    struct Case { std::function<void(SimpleBlock&)> emit; GateType expect; };
    const std::vector<Case> cases = {
        {[](SimpleBlock& b) { b.h(0); },    GateType::CH},
        {[](SimpleBlock& b) { b.s(0); },    GateType::CS},
        {[](SimpleBlock& b) { b.sdg(0); },  GateType::CSdg},
        {[](SimpleBlock& b) { b.sx(0); },   GateType::CSX},
        {[](SimpleBlock& b) { b.sxdg(0); }, GateType::CSXdg},
    };
    for (const auto& c : cases) {
        ref<SimpleBlock> inner(new SimpleBlock(1, "inner"));
        c.emit(*inner);
        inner->build();
        ControlledBlock ctrl(ref<Block>(inner.get()), 1, {true}, "ctrl");
        ctrl.build();
        const auto cmds = ctrl.flatten();
        ASSERT_EQ(cmds.size(), 1u) << gate_name(c.expect);
        EXPECT_EQ(cmds[0].gate, c.expect);
        EXPECT_EQ(cmds[0].qubits[0], 0u);  // control
        EXPECT_EQ(cmds[0].qubits[1], 1u);  // target
    }
    // C-Id is a no-op regardless of the control state.
    ref<SimpleBlock> inner(new SimpleBlock(1, "inner"));
    inner->id(0);
    inner->build();
    ControlledBlock ctrl(ref<Block>(inner.get()), 1, {true}, "ctrl");
    ctrl.build();
    EXPECT_TRUE(ctrl.flatten().empty());
}

// ── Single-control: parametric rotations ────────────────────────────────────

#define TEST_SINGLE_CTRL_ROTATION(NAME, EMIT, ANALYTIC)                                 \
    TEST(ControlledBlockUnitary, NAME ## _AtVariousAngles) {                            \
        for (double th : ANGLE_SAMPLES) {                                               \
            EXPECT_TRUE(expect_unitary_close(                                           \
                controlled_unitary([th](SimpleBlock& b) { EMIT; }, 1, 1, {true}),       \
                controlled_reference((ANALYTIC), 1, {true}, 1))) << "θ=" << th;         \
            EXPECT_TRUE(expect_unitary_close(                                           \
                controlled_unitary([th](SimpleBlock& b) { EMIT; }, 1, 1, {false}),      \
                controlled_reference((ANALYTIC), 1, {false}, 1))) << "θ=" << th;        \
        }                                                                               \
    }

TEST_SINGLE_CTRL_ROTATION(Rx, b.rx(0, Param(th)), analytic_rx(th))
TEST_SINGLE_CTRL_ROTATION(Ry, b.ry(0, Param(th)), analytic_ry(th))
TEST_SINGLE_CTRL_ROTATION(Rz, b.rz(0, Param(th)), analytic_rz(th))
TEST_SINGLE_CTRL_ROTATION(P,  b.p (0, Param(th)), analytic_p(th))

TEST(ControlledBlockUnitary, U_AtVariousParams) {
    const std::array<std::array<double, 3>, 3> params = {{
        {{0.0,        0.0,       0.0      }},
        {{PI / 3.0,  -0.4,       1.1      }},
        {{1.234,     -2.345,     0.987    }},
    }};
    for (const auto& p : params) {
        double th = p[0], phi = p[1], lam = p[2];
        EXPECT_TRUE(expect_unitary_close(
            controlled_unitary(
                [th, phi, lam](SimpleBlock& b) { b.u(0, Param(th), Param(phi), Param(lam)); },
                1, 1, {true}),
            controlled_reference(analytic_u(th, phi, lam), 1, {true}, 1)))
            << "θ=" << th << " φ=" << phi << " λ=" << lam;
    }
}

// ── Single-control: GPhase ──────────────────────────────────────────────────
//
// ``GPhase(θ)`` is a 0-qubit global-phase scalar; controlled-GPhase is a
// single-qubit phase gate ``P(θ)`` on the control qubit (no inner target).

TEST(ControlledBlockUnitary, GPhase_AtVariousAngles) {
    for (double th : ANGLE_SAMPLES) {
        auto inner = make_block<SimpleBlock>(0, "inner");
        inner->gphase(Param(th));
        inner->build();
        ControlledBlock ctrl(inner, 1, {true}, "ctrl");
        ctrl.build();
        // Controlled GPhase = P(θ) on the control qubit.
        EXPECT_TRUE(expect_unitary_close(
            build_unitary(ctrl.flatten(), 1), analytic_p(th))) << "θ=" << th;
    }
}

// Multi-control GPhase reduces to C^{n-1}(P(θ)) on the controls themselves —
// the inner block has no target qubit.  This is the path that turns a block's
// global phase into a physical relative phase, so it is pinned at every n.
TEST(ControlledBlockUnitary, GPhase_MultiControl) {
    for (uint32_t n_ctrls : {2u, 3u}) {
        for (double th : ANGLE_SAMPLES) {
            auto inner = make_block<SimpleBlock>(0, "inner");
            inner->gphase(Param(th));
            inner->build();
            ControlledBlock ctrl(inner, n_ctrls, std::vector<bool>(n_ctrls, true), "ctrl");
            ctrl.build();
            // diag(1, …, 1, e^{iθ}): the phase lands only on the all-ones control state.
            const Eigen::Index dim = Eigen::Index{1} << n_ctrls;
            Mat expected = Mat::Identity(dim, dim);
            expected(dim - 1, dim - 1) = std::exp(cd{0.0, th});
            EXPECT_TRUE(expect_unitary_close(build_unitary(ctrl.flatten(), n_ctrls), expected))
                << "n_ctrls=" << n_ctrls << " θ=" << th;
        }
    }
}

// ── Single-control: MCZ inner ─────────────────────────────────────────────
//
// A block carrying MCZ (e.g. ReflectionBlock) must be controllable at n=1.
// It previously threw at flatten() while n>=2 worked, because only the
// multi-control path pre-lowered to the MC basis.

TEST(ControlledBlockUnitary, MCZ_SingleControl) {
    for (uint32_t n_inner : {2u, 3u}) {
        // Inner MCZ over all inner qubits: diag(-1) on the all-ones state.
        const Eigen::Index dim = Eigen::Index{1} << n_inner;
        Mat mcz = Mat::Identity(dim, dim);
        mcz(dim - 1, dim - 1) = -1.0;
        EXPECT_TRUE(expect_unitary_close(
            controlled_unitary(
                [n_inner](SimpleBlock& b) {
                    std::vector<uint32_t> qs(n_inner);
                    for (uint32_t k = 0; k < n_inner; ++k) qs[k] = k;
                    b.mcz(qs);
                },
                n_inner, 1, {true}),
            controlled_reference(mcz, 1, {true}, n_inner)))
            << "n_inner=" << n_inner;
    }
}

// Multi-control MCZ.  Before MCZ joined `multi_control_basis_gateset`, n >= 2
// pre-lowered it into a Barenco cascade (322 gates at n=2 over a width-4 MCZ)
// *and* did so via the broken width->=4 rule; now it stays one gate.
TEST(ControlledBlockUnitary, MCZ_MultiControl) {
    for (uint32_t n_inner = 2; n_inner <= 5; ++n_inner) {
        for (uint32_t n_ctrls : {1u, 2u, 3u}) {
            const Eigen::Index dim = Eigen::Index{1} << n_inner;
            Mat mcz = Mat::Identity(dim, dim);
            mcz(dim - 1, dim - 1) = -1.0;
            EXPECT_TRUE(expect_unitary_close(
                controlled_unitary(
                    [n_inner](SimpleBlock& b) {
                        std::vector<uint32_t> qs(n_inner);
                        for (uint32_t k = 0; k < n_inner; ++k) qs[k] = k;
                        b.mcz(qs);
                    },
                    n_inner, n_ctrls, std::vector<bool>(n_ctrls, true)),
                controlled_reference(mcz, n_ctrls, std::vector<bool>(n_ctrls, true), n_inner)))
                << "n_inner=" << n_inner << " n_ctrls=" << n_ctrls;
        }
    }
}

// C^n(MCZ) must widen the tuple rather than expand: one MCZ, nothing else.
TEST(ControlledBlockUnitary, MCZ_ControlWideningEmitsOneGate) {
    for (uint32_t n_inner : {2u, 4u}) {
        for (uint32_t n_ctrls : {1u, 2u, 3u}) {
            ref<SimpleBlock> inner(new SimpleBlock(n_inner, "mcz"));
            std::vector<uint32_t> qs(n_inner);
            for (uint32_t k = 0; k < n_inner; ++k) qs[k] = k;
            inner->mcz(qs);
            inner->build();

            ControlledBlock ctrl(ref<Block>(inner.get()), n_ctrls,
                                 std::vector<bool>(n_ctrls, true), "ctrl");
            ctrl.build();
            const auto cmds = ctrl.flatten();

            ASSERT_EQ(cmds.size(), 1u)
                << "n_inner=" << n_inner << " n_ctrls=" << n_ctrls;
            EXPECT_EQ(cmds[0].gate, GateType::MCZ);
            EXPECT_EQ(cmds[0].qubits.size(), n_inner + n_ctrls);
        }
    }
}

// ── Single-control: 2q inner — CX (→ Toffoli) and SWAP (→ Fredkin) ─────────
//
// Inner has 2 qubits, wrapped with 1 control → 3 total qubits.  Reference
// reuses ``controlled_reference`` with the inner 2-qubit unitary.

TEST(ControlledBlockUnitary, CX_BecomesToffoli) {
    // Inner CX: control=q0, target=q1, both inner qubits.
    Mat cx_inner = controlled_1q(analytic_x(), 0, 1, 2);
    EXPECT_TRUE(expect_unitary_close(
        controlled_unitary([](SimpleBlock& b) { b.cx(0, 1); }, 2, 1, {true}),
        controlled_reference(cx_inner, 1, {true}, 2)));
}

TEST(ControlledBlockUnitary, SWAP_BecomesFredkin) {
    // Inner SWAP on (q0, q1): a permutation 00↔00, 01↔10, 10↔01, 11↔11.
    Mat swap_inner = permutation_unitary(2, [](int i) {
        return ((i & 1) << 1) | ((i & 2) >> 1);
    });
    EXPECT_TRUE(expect_unitary_close(
        controlled_unitary([](SimpleBlock& b) { b.swap(0, 1); }, 2, 1, {true}),
        controlled_reference(swap_inner, 1, {true}, 2)));
}

// ── Multi-control: Barenco recursive construction (n_ctrls ∈ {2, 3}) ───────
//
// Only the basis-set primitives are supported for n_ctrls >= 2: X, Y, Z, Rx,
// Ry, Rz, P, GPhase.  Other gates raise at build time (see
// plan_qarpx_multi_control_basis.md).  These tests pin the unitary of each
// supported case.

#define TEST_MULTI_CTRL_PAULI(NAME, EMIT, ANALYTIC, NCTRLS)                                                  \
    TEST(ControlledBlockUnitary, NAME ## _NCtrl ## NCTRLS) {                                                 \
        std::vector<bool> all_true(NCTRLS, true);                                                            \
        EXPECT_TRUE(expect_unitary_close(                                                                    \
            controlled_unitary([](SimpleBlock& b) { EMIT; }, 1, NCTRLS, all_true),                           \
            controlled_reference((ANALYTIC), NCTRLS, all_true, 1)));                                         \
        std::vector<bool> mixed(NCTRLS, true);                                                               \
        mixed[0] = false;                                                                                    \
        EXPECT_TRUE(expect_unitary_close(                                                                    \
            controlled_unitary([](SimpleBlock& b) { EMIT; }, 1, NCTRLS, mixed),                              \
            controlled_reference((ANALYTIC), NCTRLS, mixed, 1)));                                            \
    }

TEST_MULTI_CTRL_PAULI(X_n2, b.x(0), analytic_x(), 2)
TEST_MULTI_CTRL_PAULI(Y_n2, b.y(0), analytic_y(), 2)
TEST_MULTI_CTRL_PAULI(Z_n2, b.z(0), analytic_z(), 2)
TEST_MULTI_CTRL_PAULI(X_n3, b.x(0), analytic_x(), 3)
TEST_MULTI_CTRL_PAULI(Z_n3, b.z(0), analytic_z(), 3)
// SX/SXdg at n = 2 exercise the auto-lowering to {GPhase, Rx}; the built-in
// rule is phase-exact, so no mc-basis override is needed.
TEST_MULTI_CTRL_PAULI(SX_n2,   b.sx(0),   analytic_sx(),   2)
TEST_MULTI_CTRL_PAULI(SXdg_n2, b.sxdg(0), analytic_sxdg(), 2)
TEST_MULTI_CTRL_PAULI(Id_n2,   b.id(0),   Mat::Identity(2, 2), 2)

#define TEST_MULTI_CTRL_ROTATION(NAME, EMIT, ANALYTIC, NCTRLS)                                               \
    TEST(ControlledBlockUnitary, NAME ## _NCtrl ## NCTRLS ## _AtVariousAngles) {                             \
        std::vector<bool> all_true(NCTRLS, true);                                                            \
        for (double th : ANGLE_SAMPLES) {                                                                    \
            EXPECT_TRUE(expect_unitary_close(                                                                \
                controlled_unitary([th](SimpleBlock& b) { EMIT; }, 1, NCTRLS, all_true),                     \
                controlled_reference((ANALYTIC), NCTRLS, all_true, 1))) << "θ=" << th;                       \
        }                                                                                                    \
    }

TEST_MULTI_CTRL_ROTATION(Rx_n2, b.rx(0, Param(th)), analytic_rx(th), 2)
TEST_MULTI_CTRL_ROTATION(Ry_n2, b.ry(0, Param(th)), analytic_ry(th), 2)
TEST_MULTI_CTRL_ROTATION(Rz_n2, b.rz(0, Param(th)), analytic_rz(th), 2)
TEST_MULTI_CTRL_ROTATION(P_n2,  b.p (0, Param(th)), analytic_p(th),  2)
TEST_MULTI_CTRL_ROTATION(Rz_n3, b.rz(0, Param(th)), analytic_rz(th), 3)

// ── Multi-control via auto-transpile (qarp_conventions.md §6) ──────────────
//
// For n_controls >= 2, ``ControlledBlock::flatten`` first lowers the inner
// block to the multi-control basis ``{X, Y, Z, Rx, Ry, Rz, P, CX, GPhase,
// Barrier}`` via ``lower_to_mc_basis`` (controlled_block.cpp).  Tier 1
// unblocks Trotter: H, S, Sdg, CX.  Tier 2 extends to Clifford+T.  Tier 3
// covers the general single-qubit U.  Tier 4 covers two-qubit inner gates.

// Tier 1: H, S, Sdg, CX (n_ctrls=2,3)
TEST_MULTI_CTRL_PAULI(H_n2,   b.h(0),   analytic_h(),   2)
TEST_MULTI_CTRL_PAULI(H_n3,   b.h(0),   analytic_h(),   3)
TEST_MULTI_CTRL_PAULI(S_n2,   b.s(0),   analytic_s(),   2)
TEST_MULTI_CTRL_PAULI(Sdg_n2, b.sdg(0), analytic_sdg(), 2)

// Tier 2: T, Tdg (Clifford+T completion)
TEST_MULTI_CTRL_PAULI(T_n2,   b.t(0),   analytic_t(),   2)
TEST_MULTI_CTRL_PAULI(Tdg_n2, b.tdg(0), analytic_tdg(), 2)

// Tier 3: general single-qubit U
TEST(ControlledBlockUnitary, U_NCtrl2_AtVariousParams) {
    const std::array<std::array<double, 3>, 3> params = {{
        {{0.0,        0.0,       0.0      }},
        {{PI / 3.0,  -0.4,       1.1      }},
        {{1.234,     -2.345,     0.987    }},
    }};
    for (const auto& p : params) {
        double th = p[0], phi = p[1], lam = p[2];
        EXPECT_TRUE(expect_unitary_close(
            controlled_unitary(
                [th, phi, lam](SimpleBlock& b) { b.u(0, Param(th), Param(phi), Param(lam)); },
                1, 2, {true, true}),
            controlled_reference(analytic_u(th, phi, lam), 2, {true, true}, 1)))
            << "θ=" << th << " φ=" << phi << " λ=" << lam;
    }
}

// Tier 4: CX, SWAP, CCX as inner — multi-qubit inner with n_controls=2.

TEST(ControlledBlockUnitary, CX_NCtrl2) {
    // Inner CX on (q0, q1) → wrapped with 2 controls = C^3(X) (4 qubits total).
    Mat cx_inner = controlled_1q(analytic_x(), 0, 1, 2);
    EXPECT_TRUE(expect_unitary_close(
        controlled_unitary([](SimpleBlock& b) { b.cx(0, 1); }, 2, 2, {true, true}),
        controlled_reference(cx_inner, 2, {true, true}, 2)));
}

TEST(ControlledBlockUnitary, SWAP_NCtrl2) {
    Mat swap_inner = permutation_unitary(2, [](int i) {
        return ((i & 1) << 1) | ((i & 2) >> 1);
    });
    EXPECT_TRUE(expect_unitary_close(
        controlled_unitary([](SimpleBlock& b) { b.swap(0, 1); }, 2, 2, {true, true}),
        controlled_reference(swap_inner, 2, {true, true}, 2)));
}

TEST(ControlledBlockUnitary, CCX_NCtrl1) {
    // Inner CCX on (q0, q1, q2), wrapped with 1 control = C^4(X)... actually
    // C(CCX) = C^3(X) on (outer_ctrl, q0, q1, q2).  Reference: build CCX as
    // a permutation, then lift via controlled_reference.
    Mat ccx_inner = permutation_unitary(3, [](int i) {
        // CCX(c0=q0, c1=q1, t=q2): flip q2 iff q0==1 && q1==1.
        bool c0 = (i >> 0) & 1, c1 = (i >> 1) & 1;
        return (c0 && c1) ? (i ^ (1 << 2)) : i;
    });
    EXPECT_TRUE(expect_unitary_close(
        controlled_unitary([](SimpleBlock& b) { b.ccx(0, 1, 2); }, 3, 1, {true}),
        controlled_reference(ccx_inner, 1, {true}, 3)));
}

TEST(ControlledBlockUnitary, CSWAP_NCtrl1) {
    // Inner CSWAP: c=q0, swaps q1 and q2.
    Mat cswap_inner = permutation_unitary(3, [](int i) {
        bool c = (i >> 0) & 1;
        if (!c) return i;
        // Swap q1 and q2 bits.
        int b1 = (i >> 1) & 1, b2 = (i >> 2) & 1;
        int rest = i & ~((1 << 1) | (1 << 2));
        return rest | (b2 << 1) | (b1 << 2);
    });
    EXPECT_TRUE(expect_unitary_close(
        controlled_unitary([](SimpleBlock& b) { b.cswap(0, 1, 2); }, 3, 1, {true}),
        controlled_reference(cswap_inner, 1, {true}, 3)));
}

// ── Width contract ──────────────────────────────────────────────────────────
// ``n_qubits`` is assigned in the constructor as well as in ``build()``: a
// parent sizing itself from an unbuilt child (CompositeBlock::resolve_qubits,
// and the Python CompositeBlock's own inference) must not read 0.

TEST(ControlledBlockWidth, KnownBeforeBuild) {
    ref<SimpleBlock> inner(new SimpleBlock(2, "inner"));
    inner->h(0);
    inner->cx(0, 1);
    inner->build();

    ControlledBlock ctrl(ref<Block>(inner.get()), 2, {true, true}, "C2-inner");
    EXPECT_EQ(ctrl.n_qubits, 4u);
    ctrl.build();
    EXPECT_EQ(ctrl.n_qubits, 4u);
}

TEST(ControlledBlockWidth, NestedUnbuiltResolvesBottomUp) {
    ref<SimpleBlock> inner(new SimpleBlock(1, "inner"));
    inner->x(0);
    inner->build();

    ref<ControlledBlock> single(
        new ControlledBlock(ref<Block>(inner.get()), 1, {true}, "C-inner"));
    ControlledBlock outer(ref<Block>(single.get()), 1, {true}, "CC-inner");
    EXPECT_EQ(single->n_qubits, 2u);
    EXPECT_EQ(outer.n_qubits, 3u);
}
