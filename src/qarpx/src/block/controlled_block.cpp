#include "qarpx/block/controlled_block.h"

#include "qarpx/transpiler/decompositions.h"
#include "qarpx/transpiler/gateset.h"
#include "qarpx/transpiler/transpiler.h"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace qarpx {

namespace {
constexpr double PI   = std::numbers::pi;
constexpr double PI_2 = std::numbers::pi / 2.0;
constexpr double PI_4 = std::numbers::pi / 4.0;

// ── Multi-control basis lowering ────────────────────────────────────────────
//
// ``ControlledBlock(num_controls >= 2)`` lowers its inner block to the
// multi-control basis ``{X, Y, Z, Rx, Ry, Rz, P, CX, MCZ, GPhase, Barrier}``
// before wrapping.  The built-in decomposition table handles most gates exactly, but
// a few drop a global phase that ``ControlledBlock`` would otherwise observe
// (the phase becomes a controlled-phase on the wrapping controls).  Override
// the lossy decompositions with phase-preserving variants below.

/// Phase-preserving ``H`` decomposition.
///
/// ``H = e^{iπ/2} · Ry(π/2) · Rz(π)`` (matrix product, time flows right→left).
/// Emit in circuit order (first emitted = first applied = rightmost in the
/// matrix product): ``Rz(π)``, ``Ry(π/2)``, ``GPhase(π/2)``.
std::vector<Command> decompose_h_phase_correct(const Command& cmd) {
    uint32_t q = cmd.qubits[0];
    Command gphase;
    gphase.gate = GateType::GPhase;
    gphase.params.push_back(Param(PI_2));
    return {
        Command(GateType::Rz, q, Param(PI)),
        Command(GateType::Ry, q, Param(PI_2)),
        std::move(gphase),
    };
}

/// ``S = P(π/2)`` exactly.
std::vector<Command> decompose_s_to_p(const Command& cmd) {
    return {Command(GateType::P, cmd.qubits[0], Param(PI_2))};
}

/// ``T = P(π/4)`` exactly.
std::vector<Command> decompose_t_to_p(const Command& cmd) {
    return {Command(GateType::P, cmd.qubits[0], Param(PI_4))};
}

/// Phase-preserving ``U(θ, φ, λ)`` decomposition.
///
/// ``U(θ, φ, λ) = e^{i(φ+λ)/2} · Rz(φ) · Ry(θ) · Rz(λ)`` (matrix product).
/// Emit in circuit order: ``Rz(λ)``, ``Ry(θ)``, ``Rz(φ)``, ``GPhase((φ+λ)/2)``.
/// Like the built-in ``decompose_u`` but also emits the global phase that
/// ``decompose_u`` drops, so the unitary is preserved exactly (matters under
/// controlling).
std::vector<Command> decompose_u_phase_correct(const Command& cmd) {
    uint32_t q = cmd.qubits[0];
    Param theta  = cmd.params[0];
    Param phi    = cmd.params[1];
    Param lambda = cmd.params[2];
    Param half_phase = (phi + lambda) * Param(0.5);
    Command gphase;
    gphase.gate = GateType::GPhase;
    gphase.params.push_back(std::move(half_phase));
    return {
        Command(GateType::Rz, q, lambda),
        Command(GateType::Ry, q, theta),
        Command(GateType::Rz, q, phi),
        std::move(gphase),
    };
}

/// The ``Transpiler`` for the multi-control basis, with the phase-preserving
/// overrides for ``H``, ``S``, ``T``, ``U``.  ``Sdg`` and ``Tdg`` already
/// lower exactly via the built-in ``decompose_sdg``/``_tdg`` (``Sdg =
/// P(-π/2)``, ``Tdg = P(-π/4)``); ``P`` is in the target basis and is not
/// lowered.  Built once: the single-control path lowers one command at a
/// time, and `transpile()` mutates nothing while the cache is off.
const Transpiler& mc_basis_transpiler() {
    static const Transpiler t = [] {
        Transpiler t(multi_control_basis_gateset());
        t.register_decomposition(GateType::H, decompose_h_phase_correct);
        t.register_decomposition(GateType::S, decompose_s_to_p);
        t.register_decomposition(GateType::T, decompose_t_to_p);
        t.register_decomposition(GateType::U, decompose_u_phase_correct);
        return t;
    }();
    return t;
}

/// Lower a command sequence to the multi-control basis.
std::vector<Command> lower_to_mc_basis(const std::vector<Command>& cmds) {
    return mc_basis_transpiler().transpile(cmds);
}
}  // namespace

ControlledBlock::ControlledBlock(
    ref<Block> inner,
    uint32_t num_controls,
    std::vector<bool> ctrl_state,
    const std::string& nm)
    : inner_(std::move(inner))
    , num_controls_(num_controls)
    , ctrl_state_(std::move(ctrl_state))
{
    if (num_controls_ == 0) {
        throw std::invalid_argument("ControlledBlock requires at least 1 control qubit");
    }
    if (ctrl_state_.size() != num_controls_) {
        throw std::invalid_argument("ctrl_state length must match num_controls");
    }
    name = nm;
    // Set here as well as in build(): a parent sizing itself from an unbuilt
    // child must not read 0.  Inner blocks are constructed before their
    // wrapper, so a nested chain resolves bottom-up; build() recomputes it
    // anyway once the inner is built and authoritative.
    n_qubits = inner_->n_qubits + num_controls_;
}

void ControlledBlock::build() {
    if (!inner_->is_built()) {
        inner_->build();
    }
    n_qubits = inner_->n_qubits + num_controls_;
    built_ = true;
}

/// Produce the controlled version of `cmd` with `ctrl` as the control qubit,
/// expanding to a short sequence where a single controlled gate does not exist
/// (e.g. C-H, C-S).  Used by ``make_multi_controlled`` for the n_controls=1
/// fast path.
static std::vector<Command> make_single_controlled(const Command& cmd, uint32_t ctrl) {
    std::vector<Command> out;
    switch (cmd.gate) {
        case GateType::X:
            out.emplace_back(GateType::CX, SmallVector<uint32_t,2>{ctrl, cmd.qubits[0]},
                             cmd.params, cmd.cbits);
            break;
        case GateType::Y:
            out.emplace_back(GateType::CY, SmallVector<uint32_t,2>{ctrl, cmd.qubits[0]},
                             cmd.params, cmd.cbits);
            break;
        case GateType::Z:
            out.emplace_back(GateType::CZ, SmallVector<uint32_t,2>{ctrl, cmd.qubits[0]},
                             cmd.params, cmd.cbits);
            break;
        case GateType::Rx:
            out.emplace_back(GateType::CRx, SmallVector<uint32_t,2>{ctrl, cmd.qubits[0]},
                             cmd.params, cmd.cbits);
            break;
        case GateType::Ry:
            out.emplace_back(GateType::CRy, SmallVector<uint32_t,2>{ctrl, cmd.qubits[0]},
                             cmd.params, cmd.cbits);
            break;
        case GateType::Rz:
            out.emplace_back(GateType::CRz, SmallVector<uint32_t,2>{ctrl, cmd.qubits[0]},
                             cmd.params, cmd.cbits);
            break;
        case GateType::P:
            out.emplace_back(GateType::CP, SmallVector<uint32_t,2>{ctrl, cmd.qubits[0]},
                             cmd.params, cmd.cbits);
            break;
        case GateType::U: {
            // Controlling U(θ, φ, λ) produces CU(θ, φ, λ, γ=0).
            SmallVector<Param, 1> cu_params;
            for (const auto& p : cmd.params) cu_params.push_back(p);
            cu_params.push_back(Param(0.0));
            out.emplace_back(GateType::CU,
                             SmallVector<uint32_t,2>{ctrl, cmd.qubits[0]},
                             cu_params, cmd.cbits);
            break;
        }
        case GateType::CX:
            out.emplace_back(GateType::CCX,
                SmallVector<uint32_t,2>{ctrl, cmd.qubits[0], cmd.qubits[1]},
                cmd.params, cmd.cbits);
            break;
        case GateType::SWAP:
            out.emplace_back(GateType::CSWAP,
                SmallVector<uint32_t,2>{ctrl, cmd.qubits[0], cmd.qubits[1]},
                cmd.params, cmd.cbits);
            break;
        case GateType::GPhase: {
            // Controlled global phase GPhase(θ) on the inner block must act
            // as diag(1, e^{+iθ}) on the control qubit (identity when ctrl=0,
            // apply the global phase when ctrl=1).  That is exactly the
            // single-qubit phase gate ``P(θ)`` on the control.
            out.emplace_back(GateType::P, ctrl, cmd.params[0]);
            break;
        }
        case GateType::Barrier: {
            Command b = cmd;
            b.qubits.push_back(ctrl);
            out.push_back(std::move(b));
            break;
        }
        case GateType::MCZ: {
            // MCZ is diag(-1) on the all-ones state of its qubit tuple, so
            // controlling it just extends the tuple: C(MCZ(qs)) = MCZ(qs ∪ {ctrl}).
            // One gate, exact — the lowering fallback below would emit a cascade.
            Command m = cmd;
            m.qubits.push_back(ctrl);
            out.push_back(std::move(m));
            break;
        }
        // ── Clifford singles with a dedicated controlled GateType (§3.1) ──
        // One command each, as X -> CX; the sandwich identities live in
        // decompositions.cpp and apply when the controlled gate is lowered.
        case GateType::H:
            out.emplace_back(GateType::CH, ctrl, cmd.qubits[0]);
            break;
        case GateType::S:
            out.emplace_back(GateType::CS, ctrl, cmd.qubits[0]);
            break;
        case GateType::Sdg:
            out.emplace_back(GateType::CSdg, ctrl, cmd.qubits[0]);
            break;
        case GateType::SX:
            out.emplace_back(GateType::CSX, ctrl, cmd.qubits[0]);
            break;
        case GateType::SXdg:
            out.emplace_back(GateType::CSXdg, ctrl, cmd.qubits[0]);
            break;
        case GateType::Id:
            // C-Id is a no-op regardless of the control state.
            break;
        case GateType::T: {
            // CT = CP(π/4)
            out.emplace_back(GateType::CP, ctrl, cmd.qubits[0], Param(PI_4));
            break;
        }
        case GateType::Tdg: {
            // CTdg = CP(-π/4)
            out.emplace_back(GateType::CP, ctrl, cmd.qubits[0], Param(-PI_4));
            break;
        }
        // ── 3-qubit inner gates: decompose first, then single-control each ──
        // C(CCX) and C(CSWAP) have no direct primitive; the decompositions in
        // ``decompose_ccx`` and ``decompose_cswap`` produce only gates this
        // switch handles directly, so recursing through them is closed.  Same
        // pattern for CU via ``decompose_cu`` (which emits P/CX/U only).
        case GateType::CCX: {
            for (const auto& sub : decompose_ccx(cmd)) {
                auto cs = make_single_controlled(sub, ctrl);
                out.insert(out.end(), cs.begin(), cs.end());
            }
            break;
        }
        case GateType::CSWAP: {
            for (const auto& sub : decompose_cswap(cmd)) {
                auto cs = make_single_controlled(sub, ctrl);
                out.insert(out.end(), cs.begin(), cs.end());
            }
            break;
        }
        case GateType::CU: {
            for (const auto& sub : decompose_cu(cmd)) {
                auto cs = make_single_controlled(sub, ctrl);
                out.insert(out.end(), cs.begin(), cs.end());
            }
            break;
        }
        default: {
            if (cmd.gate == GateType::Custom) {
                throw std::runtime_error(
                    "Cannot create controlled version of a 'Custom' gate "
                    "(opaque dense matrix).  Lower the matrix to native gates "
                    "first via ``Block::unitary_synthesis(U)`` (Quantum Shannon "
                    "Decomposition), then wrap with ControlledBlock.  See "
                    "qarp_conventions.md §6.2.");
            }
            // Outside the direct table: lower this one command to the
            // multi-control basis and control the pieces.  Keeps n=1 as capable
            // as n>1 (which pre-lowers the whole block) without giving up the
            // direct one-gate paths above — CU would otherwise cost 4 gates.
            const auto lowered = lower_to_mc_basis({cmd});
            const bool made_progress =
                !(lowered.size() == 1 && lowered[0].gate == cmd.gate);
            if (!made_progress) {
                throw std::runtime_error(
                    "Cannot create controlled version of gate '" +
                    std::string(gate_name(cmd.gate)) +
                    "'. The transpiler has no decomposition rule for it "
                    "(qarp_conventions.md §6.2).");
            }
            for (const auto& sub : lowered) {
                auto cs = make_single_controlled(sub, ctrl);
                out.insert(out.end(), cs.begin(), cs.end());
            }
            break;
        }
    }
    return out;
}

/// Forward declaration: ``make_multi_controlled_via_sqrt`` calls back into
/// ``make_multi_controlled`` to build the C^{n-1}(X) ladders that connect
/// successive halved-rotation pieces in the recursive Barenco construction.
static std::vector<Command> make_multi_controlled(
    const Command& cmd, const std::vector<uint32_t>& ctrls);

/// Recursive Barenco-style multi-controlled decomposition for single-qubit
/// gates whose half-rotation is itself an instance of the same gate
/// (``Rx`` / ``Ry`` / ``Rz`` / ``P``).  Bottoms out at ``n_ctrls == 1``
/// (delegating to ``make_single_controlled``) and uses the textbook identity
///
///   C^n(U) = (I⊗V)·C^{n-1}(X on c_n)·(I⊗V†)·C^{n-1}(X on c_n)·C^{n-1}(V)
///
/// where V² = U.  For half-angle rotations V is the same gate type with the
/// parameter halved and the C(V) reduces to a primitive controlled rotation
/// (CRx / CRy / CRz / CP) via ``make_single_controlled``.
///
/// Cost is O(n²) gates, no ancillas — see Barenco et al. (1995), Lemma 7.5.
static std::vector<Command> make_multi_controlled_rotation(
    GateType gate,
    Param theta,
    uint32_t target,
    const std::vector<uint32_t>& ctrls)
{
    if (ctrls.empty()) {
        // Bare uncontrolled rotation — emit it directly.
        std::vector<Command> out;
        out.emplace_back(gate, target, theta);
        return out;
    }
    if (ctrls.size() == 1) {
        Command bare(gate, target, theta);
        return make_single_controlled(bare, ctrls[0]);
    }

    Param half = theta * Param(0.5);
    Param neg_half = -half;

    std::vector<Command> out;
    uint32_t last_ctrl = ctrls.back();
    std::vector<uint32_t> upper_ctrls(ctrls.begin(), ctrls.end() - 1);

    // 1. C(V) on (last_ctrl, target) — single-controlled half-angle rotation.
    Command v_bare(gate, target, half);
    auto cv = make_single_controlled(v_bare, last_ctrl);
    out.insert(out.end(), cv.begin(), cv.end());

    // 2. C^{n-1}(X) on (upper_ctrls, last_ctrl).
    Command x_bare(GateType::X, last_ctrl);
    auto cnx = make_multi_controlled(x_bare, upper_ctrls);
    out.insert(out.end(), cnx.begin(), cnx.end());

    // 3. C(V†) on (last_ctrl, target) — V† = same gate with negated angle for
    //    Rx / Ry / Rz / P.  All four satisfy ``gate(-θ) = gate(θ)†``.
    Command vdg_bare(gate, target, neg_half);
    auto cvdg = make_single_controlled(vdg_bare, last_ctrl);
    out.insert(out.end(), cvdg.begin(), cvdg.end());

    // 4. Repeat C^{n-1}(X).
    out.insert(out.end(), cnx.begin(), cnx.end());

    // 5. C^{n-1}(V) on (upper_ctrls, target) — recurse on a smaller control set.
    auto cnv = make_multi_controlled_rotation(gate, half, target, upper_ctrls);
    out.insert(out.end(), cnv.begin(), cnv.end());

    return out;
}

/// Produce the n-controlled version of ``cmd`` with controls ``ctrls``.
/// For ``ctrls.size() == 1`` this is just ``make_single_controlled``.
///
/// For n > 1:
///   * Paulis (X / Y / Z): direct lowering via the variadic ``MCZ`` gate plus
///     ``H``/``S`` basis-change pair on the target.
///   * Single-qubit rotations (Rx / Ry / Rz / P): recursive Barenco via
///     ``make_multi_controlled_rotation`` (O(n²) gates, no ancillas).
///   * GPhase(θ): equivalent to a multi-controlled phase on the controls
///     themselves (the inner block has no target qubit), reduces to
///     ``C^{n-1}(P(θ))``.
///   * Barrier: just absorbs the controls into its qubit list.
///
/// Other gates (H / S / T / Sdg / Tdg / CX / SWAP / U / CU) still throw with
/// a clear message — they need a separate decomposition pass (or pre-transpile
/// the inner block to the supported basis above).
static std::vector<Command> make_multi_controlled(
    const Command& cmd, const std::vector<uint32_t>& ctrls)
{
    if (ctrls.size() == 1) {
        return make_single_controlled(cmd, ctrls[0]);
    }

    auto build_mcz_qubits = [&](uint32_t target) {
        SmallVector<uint32_t, 2> qs;
        qs.reserve(ctrls.size() + 1);
        for (auto c : ctrls) qs.push_back(c);
        qs.push_back(target);
        return qs;
    };

    std::vector<Command> out;
    switch (cmd.gate) {
        case GateType::X: {
            // C^n-X = H(t) · MCZ([ctrls, t]) · H(t).
            uint32_t t = cmd.qubits[0];
            out.emplace_back(GateType::H, t);
            out.emplace_back(GateType::MCZ, build_mcz_qubits(t));
            out.emplace_back(GateType::H, t);
            break;
        }
        case GateType::Y: {
            // Y = S · X · Sdg, so C^n-Y = S(t) · C^n-X · Sdg(t).
            // Expanding: C^n-Y = S(t) · H(t) · MCZ · H(t) · Sdg(t).
            uint32_t t = cmd.qubits[0];
            out.emplace_back(GateType::Sdg, t);
            out.emplace_back(GateType::H, t);
            out.emplace_back(GateType::MCZ, build_mcz_qubits(t));
            out.emplace_back(GateType::H, t);
            out.emplace_back(GateType::S, t);
            break;
        }
        case GateType::Z: {
            // C^n-Z = MCZ([ctrls, t]).
            uint32_t t = cmd.qubits[0];
            out.emplace_back(GateType::MCZ, build_mcz_qubits(t));
            break;
        }
        case GateType::Rx:
        case GateType::Ry:
        case GateType::Rz:
        case GateType::P: {
            uint32_t t = cmd.qubits[0];
            return make_multi_controlled_rotation(cmd.gate, cmd.params[0], t, ctrls);
        }
        case GateType::GPhase: {
            // Controlled global phase: when all n controls are |1⟩, multiply the
            // state by e^{iθ}.  Equivalent to a multi-controlled phase gate on
            // the controls — single out the last control as the "target" of a
            // P(θ) and reduce to C^{n-1}(P(θ)) on the remaining controls.  The
            // identity uses the symmetry of the controlled phase gate in its
            // operands.
            uint32_t target_for_p = ctrls.back();
            std::vector<uint32_t> remaining_ctrls(ctrls.begin(), ctrls.end() - 1);
            auto p_cmds = make_multi_controlled_rotation(
                GateType::P, cmd.params[0], target_for_p, remaining_ctrls);
            out.insert(out.end(),
                std::make_move_iterator(p_cmds.begin()),
                std::make_move_iterator(p_cmds.end()));
            break;
        }
        case GateType::MCZ: {
            // Same identity as the single-control case: MCZ is diag(-1) on the
            // all-ones state of its tuple, so controlling it appends the
            // controls to that tuple.  One gate at any n.
            Command m = cmd;
            for (auto c : ctrls) m.qubits.push_back(c);
            out.push_back(std::move(m));
            break;
        }
        case GateType::Barrier: {
            Command b = cmd;
            for (auto c : ctrls) b.qubits.push_back(c);
            out.push_back(std::move(b));
            break;
        }
        case GateType::CX: {
            // C^n(CX(c_inner, t)) = C^{n+1}(X(t)) with c_inner appended to the
            // outer control list.  Recurse through the X path (handled above
            // via the MCZ + H sandwich for any number of controls).
            uint32_t c_inner = cmd.qubits[0];
            uint32_t t = cmd.qubits[1];
            std::vector<uint32_t> all_ctrls(ctrls);
            all_ctrls.push_back(c_inner);
            Command x_bare(GateType::X, t);
            auto mcx = make_multi_controlled(x_bare, all_ctrls);
            out.insert(out.end(), mcx.begin(), mcx.end());
            break;
        }
        default:
            // After ``lower_to_mc_basis`` runs, every reachable inner gate
            // should be in ``multi_control_basis_gateset`` (handled by one of
            // the cases above) or rejected upstream in ``flatten`` (Measure,
            // Reset, Custom).  Anything else here is a transpiler-table gap.
            throw std::runtime_error(
                "Multi-controlled (n=" + std::to_string(ctrls.size()) +
                ") version of gate '" + std::string(gate_name(cmd.gate)) +
                "' reached make_multi_controlled but is not in the "
                "multi-control basis (see qarp_conventions.md §6.2).  Either "
                "the lowering pass missed a case, or this is a new GateType "
                "without a decomposition rule in the transpiler.");
    }
    return out;
}

std::vector<Command> ControlledBlock::flatten() const {
    if (!built_) {
        throw std::runtime_error("ControlledBlock::flatten() called before build()");
    }

    auto inner_cmds = inner_->flatten();

    // Quantum-controlling a measurement or reset is non-physical: a
    // measurement projects the state, and projection in a superposition of
    // "did the projection happen" is not unitary.  Reject before we attempt
    // to expand the inner commands.  ``Custom`` gates (opaque dense matrices)
    // are also rejected — the multi-control lowering would need an explicit
    // unitary synthesis pass (Quantum Shannon Decomposition) plus global-phase
    // recovery to preserve the §6.1 contract; callers must lower the matrix
    // themselves (``inner.unitary_synthesis(U)``) before wrapping.
    for (const auto& cmd : inner_cmds) {
        if (cmd.gate == GateType::Measure || cmd.gate == GateType::Reset) {
            throw std::runtime_error(
                "ControlledBlock: cannot quantum-control a '"
                + std::string(gate_name(cmd.gate))
                + "' (non-unitary).  Move the measurement outside the "
                  "controlled block, or use classical control "
                  "(ConditionalBlock) instead.");
        }
        if (cmd.gate == GateType::Custom) {
            throw std::runtime_error(
                "ControlledBlock: cannot quantum-control a 'Custom' gate "
                "(opaque dense matrix).  Lower the matrix to native gates "
                "first via ``Block::unitary_synthesis(U)`` (Quantum Shannon "
                "Decomposition), then wrap the resulting block in "
                "ControlledBlock.  ``unitary_synthesis`` is phase-exact, so no "
                "separate global-phase compensation is needed.");
        }
    }

    // For n >= 2 controls, lower the inner block to the multi-control basis
    // before wrapping.  ``make_multi_controlled`` dispatches each gate type
    // directly for the basis ``{X, Y, Z, Rx, Ry, Rz, P, CX, MCZ, GPhase,
    // Barrier}``;
    // anything outside the basis (H, S, T, U, controlled rotations, multi-qubit
    // gates, …) is decomposed via the standard table here with phase-preserving
    // overrides so the wrapped unitary matches ``ControlledBlock``'s contract
    // (qarp_conventions.md §6).  Single-control (n == 1) goes through
    // ``make_single_controlled`` directly — it has its own per-gate case table
    // and doesn't need the pre-transpile.
    if (num_controls_ > 1) {
        inner_cmds = lower_to_mc_basis(inner_cmds);
    }

    // Shift inner qubit indices to make room for the control qubits at indices [0..num_controls).
    std::vector<uint32_t> shift_map(inner_->n_qubits);
    for (uint32_t i = 0; i < inner_->n_qubits; ++i) {
        shift_map[i] = i + num_controls_;
    }
    for (auto& cmd : inner_cmds) {
        cmd = cmd.remap_qubits(shift_map);
    }

    std::vector<uint32_t> ctrls(num_controls_);
    for (uint32_t i = 0; i < num_controls_; ++i) ctrls[i] = i;

    std::vector<Command> result;
    result.reserve(inner_cmds.size() + 2 * num_controls_);

    // X gates on control qubits where ctrl_state is false (control-on-zero)
    for (uint32_t i = 0; i < num_controls_; ++i) {
        if (!ctrl_state_[i]) {
            result.emplace_back(GateType::X, i);
        }
    }

    // n-controlled version of each inner command; may expand to several
    for (const auto& cmd : inner_cmds) {
        auto expanded = make_multi_controlled(cmd, ctrls);
        result.insert(result.end(),
            std::make_move_iterator(expanded.begin()),
            std::make_move_iterator(expanded.end()));
    }

    // Un-apply X gates for control-on-zero
    for (uint32_t i = 0; i < num_controls_; ++i) {
        if (!ctrl_state_[i]) {
            result.emplace_back(GateType::X, i);
        }
    }

    // Apply target_qubits remapping if set
    if (target_qubits.has_value()) {
        const auto& mapping = target_qubits.value();
        for (auto& cmd : result) {
            cmd = cmd.remap_qubits(mapping);
        }
    }

    return result;
}

ref<Block> ControlledBlock::set_symbols(
    const std::unordered_map<std::string, double>& values) const
{
    if (!built_)
        throw std::runtime_error("ControlledBlock::set_symbols() called before build()");
    ref<Block> new_inner = inner_->set_symbols(values);
    ref<Block> result(new ControlledBlock(
        std::move(new_inner), num_controls_, ctrl_state_, name));
    result->target_qubits = target_qubits;
    result->target_cbits  = target_cbits;
    result->n_controls    = n_controls;
    result->control_state = control_state;
    result->build();
    return result;
}

ref<Block> ControlledBlock::replace_symbols(
    const std::unordered_map<std::string, std::string>& mapping) const
{
    if (!built_)
        throw std::runtime_error("ControlledBlock::replace_symbols() called before build()");
    ref<Block> new_inner = inner_->replace_symbols(mapping);
    ref<Block> result(new ControlledBlock(
        std::move(new_inner), num_controls_, ctrl_state_, name));
    result->target_qubits = target_qubits;
    result->target_cbits  = target_cbits;
    result->n_controls    = n_controls;
    result->control_state = control_state;
    result->build();
    return result;
}

ref<Block> ControlledBlock::dagger() const {
    if (!built_)
        throw std::runtime_error("ControlledBlock::dagger() called before build()");
    ref<Block> new_inner = inner_->dagger();
    ref<Block> result(new ControlledBlock(
        std::move(new_inner), num_controls_, ctrl_state_, name + "_dag"));
    result->target_qubits = target_qubits;
    result->target_cbits  = target_cbits;
    result->n_controls    = n_controls;
    result->control_state = control_state;
    result->build();
    return result;
}

}  // namespace qarpx
