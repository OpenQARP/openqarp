#include "qarpx/transpiler/decompositions.h"
#include "qarpx/core/errors.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>

#include <Eigen/Dense>

namespace qarpx {

namespace {
constexpr double PI   = std::numbers::pi;
constexpr double PI_2 = std::numbers::pi / 2.0;
constexpr double PI_4 = std::numbers::pi / 4.0;
}

// CY -> Sdg(t) CX(c,t) S(t)
std::vector<Command> decompose_cy(const Command& cmd) {
    uint32_t c = cmd.qubits[0], t = cmd.qubits[1];
    return {
        Command(GateType::Sdg, t),
        Command(GateType::CX, c, t),
        Command(GateType::S, t),
    };
}

// CZ -> H(t) CX(c,t) H(t).  Without this rule a target set carrying CX but not
// CZ has no route for CZ and rebase totality (§6.2) fails on it.
std::vector<Command> decompose_cz(const Command& cmd) {
    uint32_t c = cmd.qubits[0], t = cmd.qubits[1];
    return {
        Command(GateType::H, t),
        Command(GateType::CX, c, t),
        Command(GateType::H, t),
    };
}

// ── Multi-controlled X / Z, ancilla-free (Barenco et al. 1995) ───────────────
//
// A decomposition rule is a *local* rewrite: it sees only its own operands, so
// it can neither allocate scratch nor learn about idle qubits elsewhere in the
// register.  Everything below is built from qubits already named in the
// command.  Three identities carry the construction.
//
// (7.5) Peel one control off a controlled phase, halving the angle:
//
//   C^n(P(θ)) = CP(θ/2)_{c_n,t} · MCX(c_<n → c_n) · CP(-θ/2)_{c_n,t}
//               · MCX(c_<n → c_n) · C^{n-1}(P(θ/2))_{c_<n, t}
//
//   With p = AND(c_<n), the exponent accumulated on t is
//   c_n − (c_n ⊕ p) + p, which is 0 unless every control is 1, where it is 2
//   — hence the half angle.  The two MCX never touch t, which is what makes
//   t available as the dirty ancilla the MCX below requires.
//
// (7.3) Split a multi-controlled X around one dirty ancilla `a`:
//
//   MCX(c → τ) = A · B · A · B,   A = MCX(lo → a),   B = MCX(hi ∪ {a} → τ)
//
//   With p = AND(lo), q = AND(hi) and `a` starting in an arbitrary state α:
//   τ ^= q·(α⊕p) ⊕ q·α = q·p, and `a` ends back at α.  `a` may therefore be
//   any qubit in any state, provided neither half also uses it.
//
// (7.2) Chain the same trick across m-2 dirty ancillas at once: 4(m-2)
//   Toffolis, linear in m.  (7.3) only splits — it is the fallback for the
//   levels where too few qubits are borrowable, and taking (7.2) whenever it
//   applies is what keeps the whole cascade quadratic rather than cubic.
//
// Neither routine re-emits MCZ.  That matters: expressing MCZ through a
// helper that builds its X-ladders *from MCZ* turns this cascade from O(n²)
// into 3^n, and the transpiler's depth guard would reject it.

namespace {

/// `free` minus `a`, preserving order.
std::vector<uint32_t> without(const std::vector<uint32_t>& free, uint32_t a) {
    std::vector<uint32_t> out;
    out.reserve(free.size());
    for (uint32_t q : free)
        if (q != a) out.push_back(q);
    return out;
}

/// (7.2) C^m(X) as a linear v-chain of 4(m-2) Toffolis around m-2 dirty
/// ancillas.  Two identical sweeps: the first threads the AND up the chain
/// onto `target`, the second unwinds every ancilla back to its entry state
/// (each appears an even number of times, and CCX is self-inverse).
/// Requires m >= 3 and `anc.size() >= m - 2`; `anc` is disjoint from
/// `controls` and `target`.
std::vector<Command> mcx_vchain(const std::vector<uint32_t>& controls,
                                uint32_t target,
                                const std::vector<uint32_t>& anc) {
    const std::size_t m = controls.size();
    std::vector<Command> out;
    out.reserve(4 * (m - 2));
    auto toffoli = [&out](uint32_t c0, uint32_t c1, uint32_t t) {
        out.emplace_back(GateType::CCX, SmallVector<uint32_t, 2>{c0, c1, t});
    };
    for (int rep = 0; rep < 2; ++rep) {
        toffoli(controls[m - 1], anc[m - 3], target);
        for (std::size_t i = m - 3; i >= 1; --i)
            toffoli(controls[i + 1], anc[i - 1], anc[i]);
        toffoli(controls[0], controls[1], anc[0]);
        for (std::size_t i = 1; i <= m - 3; ++i)
            toffoli(controls[i + 1], anc[i - 1], anc[i]);
    }
    return out;
}

/// C^m(X) on `controls` → `target`, borrowing any of `free` as a dirty
/// ancilla.  `free` must be disjoint from `controls` and from `target`; its
/// qubits are left exactly as they were found.
std::vector<Command> mcx_dirty(const std::vector<uint32_t>& controls,
                               uint32_t target,
                               const std::vector<uint32_t>& free) {
    const std::size_t m = controls.size();
    if (m == 0) return {Command(GateType::X, target)};
    if (m == 1) return {Command(GateType::CX, controls[0], target)};
    if (m == 2) {
        return {Command(GateType::CCX,
            SmallVector<uint32_t, 2>{controls[0], controls[1], target})};
    }
    if (free.empty()) {
        throw std::runtime_error(
            "mcx_dirty: >=3 controls need a borrowable qubit, none supplied");
    }
    // Enough borrowable qubits for the linear chain: take it.  Without this
    // the halving split below recurses even when (7.2) applies, and the
    // per-level Theta(m) becomes Theta(m^2) — cubic once `mc_phase` peels a
    // ladder per control.
    if (free.size() >= m - 2) return mcx_vchain(controls, target, free);

    const uint32_t a = free[0];
    // 2 <= k <= m-1 keeps *both* halves strictly smaller than m: the lower
    // half has k controls, the upper has (m-k)+1 including the ancilla.
    std::size_t k = std::max<std::size_t>(2, (m + 1) / 2);
    if (k > m - 1) k = m - 1;

    const std::vector<uint32_t> lo(controls.begin(), controls.begin() + static_cast<long>(k));
    const std::vector<uint32_t> hi(controls.begin() + static_cast<long>(k), controls.end());
    std::vector<uint32_t> hi_with_ancilla = hi;
    hi_with_ancilla.push_back(a);

    // A targets `a` and never touches `target`; B never touches `lo`.
    std::vector<uint32_t> free_a = without(free, a);
    free_a.insert(free_a.end(), hi.begin(), hi.end());
    free_a.push_back(target);

    std::vector<uint32_t> free_b = without(free, a);
    free_b.insert(free_b.end(), lo.begin(), lo.end());

    const auto A = mcx_dirty(lo, a, free_a);
    const auto B = mcx_dirty(hi_with_ancilla, target, free_b);

    std::vector<Command> out;
    out.reserve(2 * (A.size() + B.size()));
    for (int rep = 0; rep < 2; ++rep) {
        out.insert(out.end(), A.begin(), A.end());
        out.insert(out.end(), B.begin(), B.end());
    }
    return out;
}

/// C^n(P(θ)) on `controls` → `target`, ancilla-free.  `free` carries the
/// controls already peeled off by outer levels, which the MCX ladders may
/// borrow alongside `target`.
std::vector<Command> mc_phase(const std::vector<uint32_t>& controls,
                              uint32_t target,
                              double theta,
                              const std::vector<uint32_t>& free) {
    const std::size_t n = controls.size();
    if (n == 0) return {Command(GateType::P, target, Param(theta))};
    if (n == 1) return {Command(GateType::CP, controls[0], target, Param(theta))};

    const uint32_t c_last = controls.back();
    const std::vector<uint32_t> upper(controls.begin(), controls.end() - 1);

    // The ladder acts only on `upper ∪ {c_last}`, so `target` — and anything
    // an outer level already peeled — is borrowable.
    std::vector<uint32_t> ladder_free = free;
    ladder_free.push_back(target);
    const auto ladder = mcx_dirty(upper, c_last, ladder_free);

    std::vector<Command> out;
    out.emplace_back(GateType::CP, c_last, target, Param(theta * 0.5));
    out.insert(out.end(), ladder.begin(), ladder.end());
    out.emplace_back(GateType::CP, c_last, target, Param(theta * -0.5));
    out.insert(out.end(), ladder.begin(), ladder.end());

    // Recurse with c_last peeled: it is now idle and joins the free set.
    std::vector<uint32_t> inner_free = free;
    inner_free.push_back(c_last);
    const auto rest = mc_phase(upper, target, theta * 0.5, inner_free);
    out.insert(out.end(), rest.begin(), rest.end());
    return out;
}

}  // namespace

std::vector<Command> decompose_mcz(const Command& cmd) {
    if (cmd.qubits.empty()) {
        throw std::invalid_argument("decompose_mcz: empty MCZ qubit tuple");
    }
    // MCZ is diag(-1) on the all-ones state and symmetric in its tuple, so
    // singling out the last qubit as "target" loses no generality.
    const uint32_t target = cmd.qubits.back();
    const std::vector<uint32_t> controls(cmd.qubits.begin(), cmd.qubits.end() - 1);
    const std::size_t m = controls.size();

    // Small widths have exact Clifford / Clifford+T forms.  Prefer them: the
    // generic construction below halves the angle at every level, and
    // CP(π/2^k) leaves the Clifford+T set once k >= 3 — routing widths that
    // don't need Rz through Rz would force gridsynth on them for nothing.
    if (m == 0) return {Command(GateType::Z, target)};
    if (m == 1) return {Command(GateType::CZ, controls[0], target)};
    if (m == 2) {
        return {
            Command(GateType::H, target),
            Command(GateType::CCX,
                    SmallVector<uint32_t, 2>{controls[0], controls[1], target}),
            Command(GateType::H, target),
        };
    }

    // m >= 3: every operand is either a control or the target, so there is
    // nothing to borrow and an ancilla-free construction has to go through
    // halved phases (Z = P(π), so MCZ = C^m(P(π))).  Rz consequently appears
    // downstream of CP; a Clifford+T target must synthesise it (gridsynth).
    // This is a property of ancilla-free C^m(Z), not of this rule — the
    // pure-Toffoli v-chain needs m-2 borrowable qubits that a local rewrite
    // cannot conjure.
    return mc_phase(controls, target, PI, /*free=*/{});
}

// CCX (Toffoli) -> standard 6-CX decomposition
std::vector<Command> decompose_ccx(const Command& cmd) {
    uint32_t c0 = cmd.qubits[0], c1 = cmd.qubits[1], t = cmd.qubits[2];
    return {
        Command(GateType::H, t),
        Command(GateType::CX, c1, t),
        Command(GateType::Tdg, t),
        Command(GateType::CX, c0, t),
        Command(GateType::T, t),
        Command(GateType::CX, c1, t),
        Command(GateType::Tdg, t),
        Command(GateType::CX, c0, t),
        Command(GateType::T, c1),
        Command(GateType::T, t),
        Command(GateType::H, t),
        Command(GateType::CX, c0, c1),
        Command(GateType::T, c0),
        Command(GateType::Tdg, c1),
        Command(GateType::CX, c0, c1),
    };
}

// CSWAP (Fredkin) -> CX + CCX based
std::vector<Command> decompose_cswap(const Command& cmd) {
    uint32_t c = cmd.qubits[0], t0 = cmd.qubits[1], t1 = cmd.qubits[2];
    // CSWAP = CX(t1,t0) CCX(c,t0,t1) CX(t1,t0)
    return {
        Command(GateType::CX, t1, t0),
        Command(GateType::CCX,
            SmallVector<uint32_t, 2>{c, t0, t1}),
        Command(GateType::CX, t1, t0),
    };
}

// CRz(θ) -> Rz(θ/2,t) CX(c,t) Rz(-θ/2,t) CX(c,t)
std::vector<Command> decompose_crz(const Command& cmd) {
    uint32_t c = cmd.qubits[0], t = cmd.qubits[1];
    Param theta = cmd.params[0];
    Param half = theta * Param(0.5);
    return {
        Command(GateType::Rz, t, half),
        Command(GateType::CX, c, t),
        Command(GateType::Rz, t, -half),
        Command(GateType::CX, c, t),
    };
}

// CRx(θ) -> H(t) CRz(θ) H(t)  (unrolled)
std::vector<Command> decompose_crx(const Command& cmd) {
    uint32_t c = cmd.qubits[0], t = cmd.qubits[1];
    Param theta = cmd.params[0];
    Param half = theta * Param(0.5);
    return {
        Command(GateType::H, t),
        Command(GateType::Rz, t, half),
        Command(GateType::CX, c, t),
        Command(GateType::Rz, t, -half),
        Command(GateType::CX, c, t),
        Command(GateType::H, t),
    };
}

// CRy(θ) -> Ry(θ/2,t) CX(c,t) Ry(-θ/2,t) CX(c,t)
std::vector<Command> decompose_cry(const Command& cmd) {
    uint32_t c = cmd.qubits[0], t = cmd.qubits[1];
    Param theta = cmd.params[0];
    Param half = theta * Param(0.5);
    return {
        Command(GateType::Ry, t, half),
        Command(GateType::CX, c, t),
        Command(GateType::Ry, t, -half),
        Command(GateType::CX, c, t),
    };
}

// GPhase(θ) as a command — the compensating term that keeps the phase-family
// decompositions exactly unitary-preserving (not just up-to-global-phase).
// Targets that cannot express a global phase drop it via decompose_gphase,
// which is the per-target boundary for the loss.
static Command make_gphase(Param theta) {
    Command g;
    g.gate = GateType::GPhase;
    g.params.push_back(std::move(theta));
    return g;
}

// CP(θ) -> GPhase(θ/4) Rz(θ/2,t) CX(c,t) Rz(-θ/2,t) CX(c,t) Rz(θ/2,c)
// The Rz/CX product equals e^{-iθ/4}·CP(θ); the GPhase makes it exact.
std::vector<Command> decompose_cp(const Command& cmd) {
    uint32_t c = cmd.qubits[0], t = cmd.qubits[1];
    Param theta = cmd.params[0];
    Param half = theta * Param(0.5);
    return {
        make_gphase(theta * Param(0.25)),
        Command(GateType::Rz, t, half),
        Command(GateType::CX, c, t),
        Command(GateType::Rz, t, -half),
        Command(GateType::CX, c, t),
        Command(GateType::Rz, c, half),
    };
}

// RZZ(θ) -> CX(0,1) Rz(θ,1) CX(0,1)
std::vector<Command> decompose_rzz(const Command& cmd) {
    uint32_t q0 = cmd.qubits[0], q1 = cmd.qubits[1];
    Param theta = cmd.params[0];
    return {
        Command(GateType::CX, q0, q1),
        Command(GateType::Rz, q1, theta),
        Command(GateType::CX, q0, q1),
    };
}

// RXX(θ) -> H(0) H(1) RZZ(θ) H(0) H(1)  — RZZ will be further decomposed
std::vector<Command> decompose_rxx(const Command& cmd) {
    uint32_t q0 = cmd.qubits[0], q1 = cmd.qubits[1];
    Param theta = cmd.params[0];
    return {
        Command(GateType::H, q0),
        Command(GateType::H, q1),
        Command(GateType::RZZ, q0, q1, theta),
        Command(GateType::H, q0),
        Command(GateType::H, q1),
    };
}

// RYY(θ) -> Rx(π/2,0) Rx(π/2,1) RZZ(θ) Rx(-π/2,0) Rx(-π/2,1)
std::vector<Command> decompose_ryy(const Command& cmd) {
    uint32_t q0 = cmd.qubits[0], q1 = cmd.qubits[1];
    Param theta = cmd.params[0];
    return {
        Command(GateType::Rx, q0, Param(PI_2)),
        Command(GateType::Rx, q1, Param(PI_2)),
        Command(GateType::RZZ, q0, q1, theta),
        Command(GateType::Rx, q0, Param(-PI_2)),
        Command(GateType::Rx, q1, Param(-PI_2)),
    };
}

// ECR(q0, q1) — OpenQASM 3 stdgates definition:
//
//   gate ecr a, b { rzx(pi/4) a, b; x a; rzx(-pi/4) a, b; }
//   gate rzx(theta) a, b { h b; cx a, b; rz(theta) b; cx a, b; h b; }
//
// Inlined and with the adjacent H·X·H pair on q1 collapsed to X on q0
// (they commute: q1's H gates are spectators relative to X on q0):
//
//   h(q1)
//   cx(q0,q1)  rz(π/4, q1)  cx(q0,q1)
//   x(q0)
//   cx(q0,q1)  rz(-π/4, q1) cx(q0,q1)
//   h(q1)
//
// Yields a self-adjoint unitary (ECR² = I) per qarp_conventions.md §9.
std::vector<Command> decompose_ecr(const Command& cmd) {
    uint32_t q0 = cmd.qubits[0], q1 = cmd.qubits[1];
    return {
        Command(GateType::H,  q1),
        Command(GateType::CX, q0, q1),
        Command(GateType::Rz, q1, Param( PI_4)),
        Command(GateType::CX, q0, q1),
        Command(GateType::X,  q0),
        Command(GateType::CX, q0, q1),
        Command(GateType::Rz, q1, Param(-PI_4)),
        Command(GateType::CX, q0, q1),
        Command(GateType::H,  q1),
    };
}

// iSWAP -> S(0) S(1) H(0) CX(0,1) CX(1,0) H(1)
std::vector<Command> decompose_iswap(const Command& cmd) {
    uint32_t q0 = cmd.qubits[0], q1 = cmd.qubits[1];
    return {
        Command(GateType::S, q0),
        Command(GateType::S, q1),
        Command(GateType::H, q0),
        Command(GateType::CX, q0, q1),
        Command(GateType::CX, q1, q0),
        Command(GateType::H, q1),
    };
}

// iSWAPdg = iSWAP† — reverse the iSWAP sequence and dagger each step:
// H(1) CX(1,0) CX(0,1) H(0) Sdg(1) Sdg(0)   (in circuit/time order)
std::vector<Command> decompose_iswapdg(const Command& cmd) {
    uint32_t q0 = cmd.qubits[0], q1 = cmd.qubits[1];
    return {
        Command(GateType::H, q1),
        Command(GateType::CX, q1, q0),
        Command(GateType::CX, q0, q1),
        Command(GateType::H, q0),
        Command(GateType::Sdg, q1),
        Command(GateType::Sdg, q0),
    };
}

// U(θ,φ,λ) -> GPhase((φ+λ)/2) Rz(λ) Ry(θ) Rz(φ)
// OpenQASM U = e^{i(φ+λ)/2}·Rz(φ)·Ry(θ)·Rz(λ); the GPhase makes it exact.
std::vector<Command> decompose_u(const Command& cmd) {
    uint32_t q = cmd.qubits[0];
    Param theta  = cmd.params[0];
    Param phi    = cmd.params[1];
    Param lambda = cmd.params[2];
    return {
        make_gphase((phi + lambda) * Param(0.5)),
        Command(GateType::Rz, q, lambda),
        Command(GateType::Ry, q, theta),
        Command(GateType::Rz, q, phi),
    };
}

// P(θ) -> GPhase(θ/2) Rz(θ):  P(θ) = e^{iθ/2}·Rz(θ), exactly.
std::vector<Command> decompose_p(const Command& cmd) {
    uint32_t q = cmd.qubits[0];
    return {
        make_gphase(cmd.params[0] * Param(0.5)),
        Command(GateType::Rz, q, cmd.params[0]),
    };
}

// Sdg -> P(-π/2):  P(-π/2) = diag(1, e^{-iπ/2}) = diag(1, -i) = Sdg.
std::vector<Command> decompose_sdg(const Command& cmd) {
    uint32_t q = cmd.qubits[0];
    return {Command(GateType::P, q, Param(-PI_2))};
}

// Tdg -> P(-π/4):  P(-π/4) = diag(1, e^{-iπ/4}) = Tdg.
std::vector<Command> decompose_tdg(const Command& cmd) {
    uint32_t q = cmd.qubits[0];
    return {Command(GateType::P, q, Param(-PI_4))};
}

// GPhase -> []:  global phase is unobservable.  Targets that don't expose it
// as a primitive (e.g. cudaq) drop it; targets that do (qulacs, native) keep
// it because the cascade only invokes the rule for non-target gates.
std::vector<Command> decompose_gphase(const Command& /*cmd*/) {
    return {};
}

// SX -> GPhase(π/4) Rx(π/2): SX = e^{iπ/4}·Rx(π/2) exactly (§2.6).
std::vector<Command> decompose_sx(const Command& cmd) {
    return {make_gphase(Param(PI_4)), Command(GateType::Rx, cmd.qubits[0], Param(PI_2))};
}

std::vector<Command> decompose_sxdg(const Command& cmd) {
    return {make_gphase(Param(-PI_4)), Command(GateType::Rx, cmd.qubits[0], Param(-PI_2))};
}

std::vector<Command> decompose_id(const Command& /*cmd*/) {
    return {};
}

// CH -> Ry(π/4,t) CX(c,t) Ry(-π/4,t) in circuit order.  Ry(θ) = exp(-iθ/2·Y)
// (§2.3) and Ry(-π/4)·X·Ry(+π/4) = H as a matrix product, so the first gate
// applied is Ry(+π/4).  Single home of this sandwich: ControlledBlock's
// n = 1 path emits CH itself and lowers through this rule.
std::vector<Command> decompose_ch(const Command& cmd) {
    uint32_t c = cmd.qubits[0], t = cmd.qubits[1];
    return {
        Command(GateType::Ry, t, Param(PI_4)),
        Command(GateType::CX, c, t),
        Command(GateType::Ry, t, Param(-PI_4)),
    };
}

// CS = CP(π/2) and CSdg = CP(-π/2) exactly; CP's own rule carries the phase.
std::vector<Command> decompose_cs(const Command& cmd) {
    return {Command(GateType::CP, cmd.qubits[0], cmd.qubits[1], Param(PI_2))};
}

std::vector<Command> decompose_csdg(const Command& cmd) {
    return {Command(GateType::CP, cmd.qubits[0], cmd.qubits[1], Param(-PI_2))};
}

// CSX -> P(π/4,c) CRx(π/2,c,t).  SX = e^{iπ/4}·Rx(π/2); both factors are
// block-diagonal in the control (P puts e^{iπ/4} on |c=1>, CRx rotates only
// there), so the product is exact with no compensating GPhase — unlike CP/CU.
std::vector<Command> decompose_csx(const Command& cmd) {
    uint32_t c = cmd.qubits[0], t = cmd.qubits[1];
    return {
        Command(GateType::P, c, Param(PI_4)),
        Command(GateType::CRx, c, t, Param(PI_2)),
    };
}

std::vector<Command> decompose_csxdg(const Command& cmd) {
    uint32_t c = cmd.qubits[0], t = cmd.qubits[1];
    return {
        Command(GateType::P, c, Param(-PI_4)),
        Command(GateType::CRx, c, t, Param(-PI_2)),
    };
}

// SWAP(a, b) -> CX(a, b) CX(b, a) CX(a, b).  Standard textbook identity.
// CX-only output, so every target carrying CX is served by
// this one rule.
std::vector<Command> decompose_swap(const Command& cmd) {
    uint32_t q0 = cmd.qubits[0], q1 = cmd.qubits[1];
    return {
        Command(GateType::CX, q0, q1),
        Command(GateType::CX, q1, q0),
        Command(GateType::CX, q0, q1),
    };
}

// CU(θ,φ,λ,γ, c, t) — OpenQASM 3 cu definition:
//
//   P(γ,         c)                ← global-phase factor on |1⟩_c
//   P((φ+λ)/2,   c)
//   P((λ-φ)/2,   t)
//   CX(c, t)
//   U(-θ/2, 0, -(φ+λ)/2, t)
//   CX(c, t)
//   U(θ/2, φ, 0, t)
//
// P and U are further decomposed by the transpiler if not in the target gateset.
std::vector<Command> decompose_cu(const Command& cmd) {
    uint32_t c = cmd.qubits[0], t = cmd.qubits[1];
    Param theta  = cmd.params[0];
    Param phi    = cmd.params[1];
    Param lambda = cmd.params[2];
    Param gamma  = cmd.params[3];

    Param half      = Param(0.5);
    Param neg_half  = Param(-0.5);
    Param zero      = Param(0.0);

    Param phi_plus_lambda_half     = (phi + lambda) * half;
    Param lambda_minus_phi_half    = (lambda - phi) * half;
    Param neg_theta_half           = theta * neg_half;
    Param neg_phi_plus_lambda_half = phi_plus_lambda_half * Param(-1.0);
    Param theta_half               = theta * half;

    auto make_u = [&](uint32_t q, Param th, Param ph, Param la) {
        Command cmd_u;
        cmd_u.gate = GateType::U;
        cmd_u.qubits.push_back(q);
        cmd_u.params.push_back(std::move(th));
        cmd_u.params.push_back(std::move(ph));
        cmd_u.params.push_back(std::move(la));
        return cmd_u;
    };

    return {
        Command(GateType::P, c, gamma),
        Command(GateType::P, c, phi_plus_lambda_half),
        Command(GateType::P, t, lambda_minus_phi_half),
        Command(GateType::CX, c, t),
        make_u(t, neg_theta_half, zero, neg_phi_plus_lambda_half),
        Command(GateType::CX, c, t),
        make_u(t, theta_half, phi, zero),
    };
}



// Custom(U₂ₓ₂) -> GPhase(α) Rz(δ) Ry(γ) Rz(β):  U = e^{iα}·Rz(β)·Ry(γ)·Rz(δ)
// (ZYZ Euler angles, convention exp(-iθP/2)).  Makes the rebase *total* for
// targets without Custom — 1q gates fused by the optimizer re-open into
// rotations the target gate set (and any per-target cost model) can see.  A
// multi-qubit or matrixless Custom cannot be re-opened: raise loudly rather
// than let it flow through silently into a gate set that rejects it.
std::vector<Command> decompose_custom(const Command& cmd) {
    if (cmd.qubits.size() != 1 || !cmd.unitary) {
        throw capability_error(
            "Transpiler: cannot rebase a Custom gate without a stored 2x2 "
            "matrix (or acting on more than one qubit) — the target gate set "
            "does not admit Custom and no ZYZ re-opening is possible");
    }
    const Eigen::MatrixXcd& U = *cmd.unitary;
    const uint32_t q = cmd.qubits[0];

    const std::complex<double> det = U(0, 0) * U(1, 1) - U(0, 1) * U(1, 0);
    const double alpha = 0.5 * std::arg(det);
    const std::complex<double> undo(std::cos(alpha), -std::sin(alpha));
    const std::complex<double> v00 = undo * U(0, 0);
    const std::complex<double> v10 = undo * U(1, 0);

    const double gamma = 2.0 * std::atan2(std::abs(v10), std::abs(v00));
    double beta, delta;
    if (std::abs(v10) < 1e-12) {         // γ ≈ 0: only β+δ is defined
        beta  = -2.0 * std::arg(v00);
        delta = 0.0;
    } else if (std::abs(v00) < 1e-12) {  // γ ≈ π: only β−δ is defined
        beta  = 2.0 * std::arg(v10);
        delta = 0.0;
    } else {
        beta  = std::arg(v10) - std::arg(v00);
        delta = -std::arg(v10) - std::arg(v00);
    }
    return {
        make_gphase(Param(alpha)),
        Command(GateType::Rz, q, Param(delta)),
        Command(GateType::Ry, q, Param(gamma)),
        Command(GateType::Rz, q, Param(beta)),
    };
}

std::unordered_map<GateType, DecompositionFn> builtin_decompositions() {
    return {
        {GateType::CY,      decompose_cy},
        {GateType::CZ,      decompose_cz},
        {GateType::MCZ,     decompose_mcz},
        {GateType::CCX,     decompose_ccx},
        {GateType::CSWAP,   decompose_cswap},
        {GateType::CRx,     decompose_crx},
        {GateType::CRy,     decompose_cry},
        {GateType::CRz,     decompose_crz},
        {GateType::CP,      decompose_cp},
        {GateType::RZZ,     decompose_rzz},
        {GateType::RXX,     decompose_rxx},
        {GateType::RYY,     decompose_ryy},
        {GateType::ECR,     decompose_ecr},
        {GateType::iSWAP,   decompose_iswap},
        {GateType::iSWAPdg, decompose_iswapdg},
        {GateType::SWAP,    decompose_swap},
        {GateType::U,       decompose_u},
        {GateType::P,       decompose_p},
        {GateType::Sdg,     decompose_sdg},
        {GateType::Tdg,     decompose_tdg},
        {GateType::GPhase,  decompose_gphase},
        {GateType::CU,      decompose_cu},
        {GateType::Custom,  decompose_custom},
        {GateType::SX,      decompose_sx},
        {GateType::SXdg,    decompose_sxdg},
        {GateType::Id,      decompose_id},
        {GateType::CH,      decompose_ch},
        {GateType::CS,      decompose_cs},
        {GateType::CSdg,    decompose_csdg},
        {GateType::CSX,     decompose_csx},
        {GateType::CSXdg,   decompose_csxdg},
    };
}

std::unordered_map<GateType, DecompositionFn> decompositions_for(const GateSet& target) {
    if (target.rules.empty()) return builtin_decompositions();
    if (target.rules == "clifford_t_rz") return clifford_t_rz_decompositions();
    throw std::invalid_argument(
        "GateSet '" + target.name + "' names an unknown decomposition rule set '" +
        target.rules + "' (known: \"\", \"clifford_t_rz\")");
}

// ── Rz-only-target decompositions ─────────────────────────────────────────────

std::vector<Command> decompose_rx_rz_only(const Command& cmd) {
    uint32_t q = cmd.qubits[0];
    Param theta = cmd.params[0];
    return {
        Command(GateType::H, q),
        Command(GateType::Rz, q, theta),
        Command(GateType::H, q),
    };
}

std::vector<Command> decompose_ry_rz_only(const Command& cmd) {
    uint32_t q = cmd.qubits[0];
    Param theta = cmd.params[0];
    // Operator:  Ry(θ) = S · H · Rz(θ) · H · Sdg
    // Circuit order (earliest first): Sdg, H, Rz(θ), H, S
    return {
        Command(GateType::Sdg, q),
        Command(GateType::H,   q),
        Command(GateType::Rz,  q, theta),
        Command(GateType::H,   q),
        Command(GateType::S,   q),
    };
}

std::unordered_map<GateType, DecompositionFn> clifford_t_rz_decompositions() {
    // Start from built-ins and add Rx/Ry rewrites — the built-in table has no
    // rule for either (they are native everywhere else).  The transpiler
    // cascades: U → Rz·Ry·Rz (built-in) → Rz·(Sdg·H·Rz·H·S)·Rz
    auto table = builtin_decompositions();
    table[GateType::Rx] = decompose_rx_rz_only;
    table[GateType::Ry] = decompose_ry_rz_only;
    return table;
}

}  // namespace qarpx
