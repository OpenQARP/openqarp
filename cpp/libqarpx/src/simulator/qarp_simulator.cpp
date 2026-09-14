#include "qarpx/simulator/qarp_simulator.h"
#include "qarpx/core/errors.h"
#include "qarpx/parallel/thread_pool.h"
#include "qarpx/transpiler/decompositions.h"
#include "qarpx/simulator/dense_kernel.h"
#include "qarpx/simulator/fusion.h"
#include "qarpx/transpiler/fusion.h"

// csim C API
#include <csim/memory_ops.hpp>
#include <csim/init_ops.hpp>
#include <csim/stat_ops.hpp>
#include <csim/update_ops.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <exception>
#include <functional>
#include <numbers>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace qarpx {

namespace {

void reject_idle_channel(const NoiseModel& nm) {
    // Per-cycle idle-qubit channels need a notion of circuit cycles which the
    // Block IR doesn't expose; the doc on `NoiseModel::idle` reserves the
    // field and promises this throw on use.
    if (nm.idle.has_value())
        throw std::runtime_error(
            "QarpSimulator: NoiseModel::idle is not supported in v1.  Per-cycle "
            "idle-qubit channels require a notion of circuit cycles which the "
            "Block IR does not expose.  Leave `idle` empty.");
}

// QARP_FUSION_MAX_QUBITS, read once per process like QARP_NUM_THREADS; a
// value that does not parse or exceeds kMaxFusionQubits is ignored, not an
// error (same policy as read_thread_count).
std::size_t default_fusion_max_qubits() {
    static const std::size_t k = [] {
        if (const char* env = std::getenv("QARP_FUSION_MAX_QUBITS")) {
            char* end = nullptr;
            const long n = std::strtol(env, &end, 10);
            if (end != env && *end == '\0' && n >= 0
                    && static_cast<std::size_t>(n) <= QarpSimulator::kMaxFusionQubits)
                return static_cast<std::size_t>(n);
        }
        return QarpSimulator::kDefaultFusionQubits;
    }();
    return k;
}

}  // namespace

QarpSimulator::QarpSimulator()
    : fusion_max_qubits_(default_fusion_max_qubits()) {
    init_threading();
}

QarpSimulator::QarpSimulator(NoiseModel noise)
    : noise_model_(std::move(noise)),
      fusion_max_qubits_(default_fusion_max_qubits()) {
    init_threading();
    reject_idle_channel(noise_model_);
}

void QarpSimulator::set_noise_model(NoiseModel n) {
    reject_idle_channel(n);
    noise_model_ = std::move(n);
}

void QarpSimulator::set_fusion_max_qubits(std::size_t k) {
    if (k > kMaxFusionQubits)
        throw std::invalid_argument(
            "QarpSimulator::set_fusion_max_qubits: " + std::to_string(k)
            + " exceeds the maximum block width " + std::to_string(kMaxFusionQubits));
    fusion_max_qubits_ = k;
}

Eigen::MatrixXcd QarpSimulator::local_unitary(const Command& cmd) const {
    const std::size_t k = cmd.qubits.size();
    if (k == 0 || k > kMaxCustomQubits)
        throw std::invalid_argument(
            "QarpSimulator::local_unitary: '" + std::string(gate_name(cmd.gate))
            + "' acts on " + std::to_string(k) + " qubits");
    if (cmd.gate == GateType::Custom && cmd.unitary) return *cmd.unitary;

    // Same gate on qubits 0..k-1 in list order, so local bit b ↔ qubits[b];
    // the condition is dropped because apply_command never reads it.
    Command local = cmd;
    for (std::size_t b = 0; b < k; ++b) local.qubits[b] = static_cast<uint32_t>(b);
    local.condition_bits.clear();
    local.condition_values.clear();

    const uint64_t dim = uint64_t{1} << k;
    Eigen::MatrixXcd U(static_cast<Eigen::Index>(dim), static_cast<Eigen::Index>(dim));
    std::vector<std::complex<double>> col(dim);
    for (uint64_t j = 0; j < dim; ++j) {
        std::fill(col.begin(), col.end(), std::complex<double>{});
        col[j] = 1.0;
        apply_command(local, col.data(), dim);
        for (uint64_t i = 0; i < dim; ++i)
            U(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = col[i];
    }
    return U;
}

std::vector<Command> QarpSimulator::fuse_for_dispatch(
    const std::vector<Command>& commands, int n_qubits) const {
    if (fusion_max_qubits_ == 0) return commands;
    if (fusion_max_qubits_ == 1
            || static_cast<std::size_t>(std::max(n_qubits, 0)) < fusion_min_qubits_)
        return fuse_single_qubit_gates(commands);
    return fuse_for_simulation(
        commands, fusion_max_qubits_,
        [this](const Command& c) { return local_unitary(c); });
}

namespace {

// Sdg matrix: [[1,0],[0,-i]]
constexpr CTYPE SDG_MAT[4] = {
    {1.0, 0.0}, {0.0,  0.0},
    {0.0, 0.0}, {0.0, -1.0}
};
// Tdg matrix: [[1,0],[0,exp(-i*pi/4)]]
const CTYPE TDG_MAT[4] = {
    {1.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {std::cos(-std::numbers::pi / 4.0), std::sin(-std::numbers::pi / 4.0)}
};
// H, S and the sqrt-X pair (§2.6): controlled-1Q targets for CH/CS/CSdg/CSX/CSXdg.
constexpr CTYPE H_MAT[4] = {
    {1.0 / std::numbers::sqrt2, 0.0}, { 1.0 / std::numbers::sqrt2, 0.0},
    {1.0 / std::numbers::sqrt2, 0.0}, {-1.0 / std::numbers::sqrt2, 0.0}
};
constexpr CTYPE S_MAT[4] = {
    {1.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {0.0, 1.0}
};
constexpr CTYPE SX_MAT[4] = {
    {0.5,  0.5}, {0.5, -0.5},
    {0.5, -0.5}, {0.5,  0.5}
};
constexpr CTYPE SXDG_MAT[4] = {
    {0.5, -0.5}, {0.5,  0.5},
    {0.5,  0.5}, {0.5, -0.5}
};
// Pauli matrices (used as controlled-1Q targets for CY/CCX/MCZ).
constexpr CTYPE X_MAT[4] = {
    {0.0, 0.0}, {1.0, 0.0},
    {1.0, 0.0}, {0.0, 0.0}
};
constexpr CTYPE Y_MAT[4] = {
    {0.0, 0.0}, {0.0, -1.0},
    {0.0, 1.0}, {0.0,  0.0}
};
constexpr CTYPE Z_MAT[4] = {
    {1.0, 0.0}, {0.0,  0.0},
    {0.0, 0.0}, {-1.0, 0.0}
};

double param_val(const Command& cmd, size_t idx) {
    if (cmd.params.size() <= idx)
        throw std::runtime_error("gate '" + std::string(gate_name(cmd.gate))
                                 + "' missing param[" + std::to_string(idx) + ']');
    if (cmd.params[idx].is_symbolic())
        throw std::runtime_error("gate '" + std::string(gate_name(cmd.gate))
                                 + "' has unresolved symbolic parameter");
    return cmd.params[idx].value();
}

// 2×2 rotation matrices in row-major order (csim's convention).
//
// Rx(θ) = [[ cos(θ/2), -i sin(θ/2)],
//          [-i sin(θ/2),  cos(θ/2)]]
inline std::array<CTYPE, 4> rx_matrix(double theta) {
    const double c = std::cos(theta / 2.0);
    const double s = std::sin(theta / 2.0);
    return {{ {c, 0.0}, {0.0, -s},
              {0.0, -s}, {c, 0.0} }};
}

// Ry(θ) = [[cos(θ/2), -sin(θ/2)],
//          [sin(θ/2),  cos(θ/2)]]
inline std::array<CTYPE, 4> ry_matrix(double theta) {
    const double c = std::cos(theta / 2.0);
    const double s = std::sin(theta / 2.0);
    return {{ {c, 0.0}, {-s, 0.0},
              {s, 0.0}, {c, 0.0} }};
}

// Rz(θ) = diag(e^{-iθ/2}, e^{+iθ/2})
inline std::array<CTYPE, 4> rz_matrix(double theta) {
    const double c = std::cos(theta / 2.0);
    const double s = std::sin(theta / 2.0);
    return {{ {c, -s}, {0.0, 0.0},
              {0.0, 0.0}, {c, s} }};
}

// P(θ) = diag(1, e^{iθ})
inline std::array<CTYPE, 4> p_matrix(double theta) {
    return {{ {1.0, 0.0}, {0.0, 0.0},
              {0.0, 0.0}, {std::cos(theta), std::sin(theta)} }};
}

// ── Trajectory detection ─────────────────────────────────────────────────────

/// True if the command requires per-shot trajectory mode: a stateful Measure
/// (records a cbit), a Reset (non-deterministic), a classically-conditional
/// gate (reads a cbit), or a branch marker emitted by ConditionalBlock.
inline bool needs_trajectory(const Command& c) {
    if (c.gate == GateType::Reset) return true;
    if (c.gate == GateType::Measure && !c.cbits.empty()) return true;
    if (!c.condition_bits.empty()) return true;
    if (c.gate == GateType::BranchBegin) return true;
    return false;
}

inline bool any_needs_trajectory(const std::vector<Command>& cmds) {
    for (const auto& c : cmds) if (needs_trajectory(c)) return true;
    return false;
}

/// True if every trajectory-forcing command is a terminal Measure: no Reset,
/// no classical condition, no branch markers, and no recorded-measured qubit
/// is ever acted on again after its measurement (gates on *other* qubits may
/// follow freely — measurements commute with them).  Such measurements cannot
/// influence later evolution, so the per-shot trajectory loop can be replaced
/// by sampling the final statevector distribution once and reading each
/// shot's classical bits out of its sampled outcome (the measured bits are
/// perfectly correlated with the sampled basis state).
inline bool measurements_are_terminal(const std::vector<Command>& cmds) {
    uint64_t measured_mask = 0;  // qubits whose recorded measurement happened
    for (const auto& c : cmds) {
        if (c.gate == GateType::Reset) return false;
        if (!c.condition_bits.empty()) return false;
        if (c.gate == GateType::BranchBegin
                || c.gate == GateType::BranchElse
                || c.gate == GateType::BranchEnd) return false;
        if (c.gate == GateType::Barrier) continue;
        if (c.gate == GateType::Measure) {
            // Re-measuring an untouched qubit reproduces the same outcome,
            // which reading the same bit of the sampled basis state matches.
            if (!c.cbits.empty()) {
                if (c.qubits.empty() || c.qubits[0] >= 64) return false;
                measured_mask |= uint64_t{1} << c.qubits[0];
            }
            continue;
        }
        for (auto q : c.qubits) {
            if (q >= 64) return false;  // beyond mask width: be conservative
            if ((measured_mask >> q) & 1) return false;
        }
    }
    return true;
}

/// Index of the first command that requires trajectory mode; cmds.size() if
/// the circuit is fully unitary.  Used to split prefix (run once, shared
/// across shots) from suffix (run per-shot).
inline std::size_t first_trajectory_index(const std::vector<Command>& cmds) {
    for (std::size_t i = 0; i < cmds.size(); ++i)
        if (needs_trajectory(cmds[i])) return i;
    return cmds.size();
}

/// Width of the per-shot classical register.  Delegates to the shared
/// `cbit_register_width` (core/command.h) so the simulator, the OpenQASM 3
/// emitter and `Block::n_cbits` cannot drift apart.
inline int classical_register_width(const std::vector<Command>& cmds) {
    return static_cast<int>(cbit_register_width(cmds));
}

/// For each BranchBegin index, the indices of its matching BranchElse and
/// BranchEnd (BranchElse may be SIZE_MAX if absent).  Indexed sparsely by
/// command position via an unordered_map for cheap lookup at dispatch time.
struct BranchTable {
    static constexpr std::size_t kNoElse = static_cast<std::size_t>(-1);
    struct Entry { std::size_t else_idx; std::size_t end_idx; };
    std::unordered_map<std::size_t, Entry> by_begin;
};

/// Walk the command list once, matching every BranchBegin to its BranchEnd
/// (and optional BranchElse) using a depth stack.  Throws on unbalanced or
/// stray markers — those would only arise from a malformed flatten() output.
BranchTable build_branch_table(const std::vector<Command>& cmds) {
    BranchTable table;
    std::vector<std::size_t> begin_stack;
    std::vector<std::size_t> else_at;  // parallel to begin_stack: kNoElse or position

    for (std::size_t i = 0; i < cmds.size(); ++i) {
        switch (cmds[i].gate) {
            case GateType::BranchBegin:
                begin_stack.push_back(i);
                else_at.push_back(BranchTable::kNoElse);
                break;
            case GateType::BranchElse:
                if (begin_stack.empty())
                    throw std::runtime_error(
                        "branch table: BranchElse without matching BranchBegin");
                if (else_at.back() != BranchTable::kNoElse)
                    throw std::runtime_error(
                        "branch table: duplicate BranchElse for the same BranchBegin");
                else_at.back() = i;
                break;
            case GateType::BranchEnd: {
                if (begin_stack.empty())
                    throw std::runtime_error(
                        "branch table: BranchEnd without matching BranchBegin");
                std::size_t begin = begin_stack.back();
                std::size_t else_i = else_at.back();
                begin_stack.pop_back();
                else_at.pop_back();
                table.by_begin[begin] = {else_i, i};
                break;
            }
            default:
                break;
        }
    }
    if (!begin_stack.empty())
        throw std::runtime_error(
            "branch table: unterminated BranchBegin (missing BranchEnd)");
    return table;
}

/// True if every cbit in `bits[i]` of the register equals `values[i]`.
inline bool branch_condition_holds(const Command& begin_cmd,
                                   const std::vector<bool>& reg)
{
    const auto& bits   = begin_cmd.condition_bits;
    const auto& values = begin_cmd.condition_values;
    for (std::size_t k = 0; k < bits.size(); ++k) {
        const uint32_t b = bits[k];
        if (b >= reg.size() || reg[b] != values[k]) return false;
    }
    return true;
}

}  // namespace

// ── apply_command ─────────────────────────────────────────────────────────────

void QarpSimulator::apply_command(const Command&        cmd,
                                    std::complex<double>* state,
                                    uint64_t              dim) const {
    auto* s = reinterpret_cast<CTYPE*>(state);
    ITYPE  d = static_cast<ITYPE>(dim);

    const uint32_t q0 = cmd.qubits.empty() ? 0 : cmd.qubits[0];
    const uint32_t q1 = cmd.qubits.size() > 1 ? cmd.qubits[1] : 0;

    switch (cmd.gate) {
        // ── 1-qubit, no params ──
        case GateType::X:    X_gate(q0, s, d);    break;
        case GateType::Y:    Y_gate(q0, s, d);    break;
        case GateType::Z:    Z_gate(q0, s, d);    break;
        case GateType::H:    H_gate(q0, s, d);    break;
        case GateType::S:    S_gate(q0, s, d);    break;
        case GateType::T:    T_gate(q0, s, d);    break;
        case GateType::Sdg: single_qubit_dense_matrix_gate(q0, SDG_MAT, s, d); break;
        case GateType::Tdg: single_qubit_dense_matrix_gate(q0, TDG_MAT, s, d); break;
        // csim's SQRT_X_GATE_MATRIX is ½[[1+i,1-i],[1-i,1+i]] = SX exactly (§2.6).
        case GateType::SX:   sqrtX_gate(q0, s, d);    break;
        case GateType::SXdg: sqrtXdag_gate(q0, s, d); break;
        case GateType::Id:   break;

        // ── 1-qubit, 1 param ──
        // csim's RX/RY/RZ_gate apply exp(+i θ/2 P); the qarpx convention is
        // exp(-i θ/2 P) (qarp_conventions.md §2.3), so pass α = -θ.  Same fix
        // pattern as the multi-qubit RZZ/RXX/RYY block below.
        case GateType::Rx:   RX_gate(q0, -param_val(cmd, 0), s, d);  break;
        case GateType::Ry:   RY_gate(q0, -param_val(cmd, 0), s, d);  break;
        case GateType::Rz:   RZ_gate(q0, -param_val(cmd, 0), s, d);  break;
        case GateType::P: {
            const double theta = param_val(cmd, 0);
            const CTYPE phase{std::cos(theta), std::sin(theta)};
            single_qubit_phase_gate(q0, phase, s, d);
            break;
        }

        // ── 2-qubit, no params ──
        case GateType::CX:   CNOT_gate(q0, q1, s, d);  break;
        case GateType::CZ:   CZ_gate(q0, q1, s, d);    break;
        case GateType::SWAP: SWAP_gate(q0, q1, s, d);  break;

        // ── 2-qubit, no params, lowered on the fly ──
        // iSWAP/iSWAPdg/ECR have no csim kernel.  Rather than carry a second
        // copy of their matrices here, apply the canonical §11 decomposition
        // the transpiler already uses: it targets gates handled above and is
        // phase-exact, so the global phase this simulator preserves is right.
        case GateType::iSWAP:
        case GateType::iSWAPdg:
        case GateType::ECR: {
            const std::vector<Command> lowered =
                  cmd.gate == GateType::ECR   ? decompose_ecr(cmd)
                : cmd.gate == GateType::iSWAP ? decompose_iswap(cmd)
                                              : decompose_iswapdg(cmd);
            for (const Command& sub : lowered)
                apply_command(sub, state, dim);
            break;
        }
        case GateType::CY:
            single_qubit_control_single_qubit_dense_matrix_gate(q0, 1, q1, Y_MAT, s, d);
            break;
        case GateType::CH:
            single_qubit_control_single_qubit_dense_matrix_gate(q0, 1, q1, H_MAT, s, d);
            break;
        case GateType::CS:
            single_qubit_control_single_qubit_dense_matrix_gate(q0, 1, q1, S_MAT, s, d);
            break;
        case GateType::CSdg:
            single_qubit_control_single_qubit_dense_matrix_gate(q0, 1, q1, SDG_MAT, s, d);
            break;
        case GateType::CSX:
            single_qubit_control_single_qubit_dense_matrix_gate(q0, 1, q1, SX_MAT, s, d);
            break;
        case GateType::CSXdg:
            single_qubit_control_single_qubit_dense_matrix_gate(q0, 1, q1, SXDG_MAT, s, d);
            break;

        // ── 2-qubit, 1 param (controlled rotations) ──────────────────────
        case GateType::CRx: {
            auto m = rx_matrix(param_val(cmd, 0));
            single_qubit_control_single_qubit_dense_matrix_gate(q0, 1, q1, m.data(), s, d);
            break;
        }
        case GateType::CRy: {
            auto m = ry_matrix(param_val(cmd, 0));
            single_qubit_control_single_qubit_dense_matrix_gate(q0, 1, q1, m.data(), s, d);
            break;
        }
        case GateType::CRz: {
            auto m = rz_matrix(param_val(cmd, 0));
            single_qubit_control_single_qubit_dense_matrix_gate(q0, 1, q1, m.data(), s, d);
            break;
        }
        case GateType::CP: {
            auto m = p_matrix(param_val(cmd, 0));
            single_qubit_control_single_qubit_dense_matrix_gate(q0, 1, q1, m.data(), s, d);
            break;
        }

        // ── 2-qubit Pauli rotations ──────────────────────────────────────
        // csim's multi_qubit_Pauli_rotation_gate applies exp(+i α/2 P);
        // our convention is RZZ(θ) = exp(-i θ/2 Z⊗Z), so pass α = -θ.
        case GateType::RZZ:
        case GateType::RXX:
        case GateType::RYY: {
            const UINT pauli =
                cmd.gate == GateType::RZZ ? 3 :
                cmd.gate == GateType::RXX ? 1 : 2;
            const UINT targets[2] = {q0, q1};
            const UINT paulis[2]  = {pauli, pauli};
            const double angle    = -param_val(cmd, 0);
            multi_qubit_Pauli_rotation_gate_partial_list(
                targets, paulis, /*count=*/2, angle, s, d);
            break;
        }

        // ── 3-qubit, no params ───────────────────────────────────────────
        case GateType::CCX: {
            const UINT controls[2]  = {q0, q1};
            const UINT ctrl_vals[2] = {1, 1};
            const UINT target       = cmd.qubits.size() > 2 ? cmd.qubits[2] : 0;
            multi_qubit_control_single_qubit_dense_matrix_gate(
                controls, ctrl_vals, /*n_controls=*/2, target, X_MAT, s, d);
            break;
        }
        case GateType::CSWAP: {
            // CSWAP(c, t0, t1) = CX(t1, t0) · CCX(c, t0, t1) · CX(t1, t0).
            // Matches the transpiler's CSWAP decomposition; keeps the
            // simulator self-contained (no transpile pass required).
            if (cmd.qubits.size() < 3)
                throw std::runtime_error("CSWAP requires 3 qubits");
            const UINT c  = q0;
            const UINT t0 = q1;
            const UINT t1 = cmd.qubits[2];
            CNOT_gate(t1, t0, s, d);
            const UINT controls[2]  = {c, t0};
            const UINT ctrl_vals[2] = {1, 1};
            multi_qubit_control_single_qubit_dense_matrix_gate(
                controls, ctrl_vals, /*n_controls=*/2, t1, X_MAT, s, d);
            CNOT_gate(t1, t0, s, d);
            break;
        }

        // ── n-qubit, no params (variadic) ────────────────────────────────
        // MCZ takes [c0, c1, …, c_{n-1}, target] in `qubits`.
        case GateType::MCZ: {
            if (cmd.qubits.size() < 2)
                throw std::runtime_error("MCZ requires ≥2 qubits");
            const UINT n_ctrl = static_cast<UINT>(cmd.qubits.size() - 1);
            std::vector<UINT> controls(n_ctrl);
            std::vector<UINT> ctrl_vals(n_ctrl, 1);
            for (UINT i = 0; i < n_ctrl; ++i)
                controls[i] = cmd.qubits[i];
            const UINT target = cmd.qubits[n_ctrl];
            multi_qubit_control_single_qubit_dense_matrix_gate(
                controls.data(), ctrl_vals.data(), n_ctrl, target, Z_MAT, s, d);
            break;
        }

        // ── 1-qubit, 3 params (OpenQASM 3 general U) ──
        case GateType::U: {
            double theta_  = param_val(cmd, 0);
            double phi_    = param_val(cmd, 1);
            double lambda_ = param_val(cmd, 2);
            CTYPE mat[4] = {
                { std::cos(theta_ / 2.0), 0.0},
                {-std::cos(lambda_) * std::sin(theta_ / 2.0), -std::sin(lambda_) * std::sin(theta_ / 2.0)},
                { std::cos(phi_)    * std::sin(theta_ / 2.0),  std::sin(phi_)    * std::sin(theta_ / 2.0)},
                { std::cos(phi_ + lambda_) * std::cos(theta_ / 2.0), std::sin(phi_ + lambda_) * std::cos(theta_ / 2.0)}
            };
            single_qubit_dense_matrix_gate(q0, mat, s, d);
            break;
        }

        // ── 2-qubit, 4 params (OpenQASM 3 cu: e^{iγ}·U on the |1⟩_c subspace) ──
        case GateType::CU: {
            double theta_  = param_val(cmd, 0);
            double phi_    = param_val(cmd, 1);
            double lambda_ = param_val(cmd, 2);
            double gamma_  = param_val(cmd, 3);
            const double cg = std::cos(gamma_), sg = std::sin(gamma_);
            const CTYPE phase{cg, sg};
            CTYPE mat[4] = {
                { std::cos(theta_ / 2.0), 0.0},
                {-std::cos(lambda_) * std::sin(theta_ / 2.0), -std::sin(lambda_) * std::sin(theta_ / 2.0)},
                { std::cos(phi_)    * std::sin(theta_ / 2.0),  std::sin(phi_)    * std::sin(theta_ / 2.0)},
                { std::cos(phi_ + lambda_) * std::cos(theta_ / 2.0), std::sin(phi_ + lambda_) * std::cos(theta_ / 2.0)}
            };
            for (auto& m : mat) m = m * phase;
            single_qubit_control_single_qubit_dense_matrix_gate(q0, 1, q1, mat, s, d);
            break;
        }

        // ── Zero-qubit, 1 param ──
        case GateType::GPhase: {
            double theta = param_val(cmd, 0);
            CTYPE phase(std::cos(theta), std::sin(theta));
            for (ITYPE i = 0; i < d; i++) s[i] *= phase;
            break;
        }

        // ── Dense Custom gate: 1 qubit from O1 fusion / user gates, k qubits
        //    from fuse_for_simulation (local bit b ↔ qubits[b]).
        case GateType::Custom: {
            const std::size_t k = cmd.qubits.size();
            if (!cmd.unitary || k == 0 || k > kMaxCustomQubits)
                throw std::runtime_error(
                    "QarpSimulator: Custom gate must carry a unitary and "
                    "1.." + std::to_string(kMaxCustomQubits) + " qubit targets");
            const Eigen::Index want = Eigen::Index{1} << k;
            if (cmd.unitary->rows() != want || cmd.unitary->cols() != want)
                throw std::runtime_error(
                    "QarpSimulator: Custom unitary is "
                    + std::to_string(cmd.unitary->rows()) + "x"
                    + std::to_string(cmd.unitary->cols()) + " for "
                    + std::to_string(k) + " qubit(s)");
            if (k == 1) {
                CTYPE mat[4] = {
                    (*cmd.unitary)(0,0), (*cmd.unitary)(0,1),
                    (*cmd.unitary)(1,0), (*cmd.unitary)(1,1),
                };
                single_qubit_dense_matrix_gate(q0, mat, s, d);
                break;
            }
            apply_dense_block(cmd.qubits.begin(), k, *cmd.unitary, state, dim);
            break;
        }

        // ── Non-unitary / meta ──
        case GateType::Barrier:
        case GateType::Measure:
        case GateType::Reset:
            break;  // no-op in statevector simulation

        default:
            throw std::runtime_error(
                "QarpSimulator: unsupported gate '"
                + std::string(gate_name(cmd.gate)) + '\'');
    }
}

// ── apply_command_trajectory ──────────────────────────────────────────────────

bool QarpSimulator::apply_command_trajectory(
    const Command&        cmd,
    std::complex<double>* state,
    uint64_t              dim,
    std::vector<bool>&    cbit_register,
    std::mt19937&         rng) const
{
    // Branch markers must be dispatched at the run() loop level, not here —
    // they affect *which next command runs*, not the state of the current one.
    if (cmd.gate == GateType::BranchBegin
            || cmd.gate == GateType::BranchElse
            || cmd.gate == GateType::BranchEnd) {
        throw std::runtime_error(
            "QarpSimulator::apply_command_trajectory: stray "
            + std::string(gate_name(cmd.gate))
            + " — branch dispatch must happen in the per-shot run loop.");
    }

    // Classical condition check: AND of (cbit == value).  Empty = unconditional.
    if (!cmd.condition_bits.empty()) {
        for (std::size_t i = 0; i < cmd.condition_bits.size(); ++i) {
            const uint32_t b = cmd.condition_bits[i];
            if (b >= cbit_register.size())
                throw std::runtime_error(
                    "trajectory: condition references cbit "
                    + std::to_string(b) + " outside register of size "
                    + std::to_string(cbit_register.size()));
            if (cbit_register[b] != cmd.condition_values[i]) return false;
        }
    }

    auto* s = reinterpret_cast<CTYPE*>(state);
    ITYPE  d = static_cast<ITYPE>(dim);

    // Mid-circuit Measure: sample outcome from M0_prob, project, normalise,
    // record outcome in the per-shot classical register.
    if (cmd.gate == GateType::Measure) {
        if (cmd.qubits.empty())
            throw std::runtime_error("trajectory: Measure requires a qubit");
        const uint32_t q = cmd.qubits[0];
        const double p0  = M0_prob(q, s, d);

        // Numerical clamp — accumulated floating-point error can put p0
        // slightly outside [0, 1].
        std::uniform_real_distribution<double> u(0.0, 1.0);
        const double r = u(rng);
        const bool outcome_one = (r >= p0);

        if (outcome_one) {
            P1_gate(q, s, d);
            const double p1 = std::max(1.0 - p0, 0.0);
            normalize(p1, s, d);
        } else {
            P0_gate(q, s, d);
            normalize(std::max(p0, 0.0), s, d);
        }

        if (!cmd.cbits.empty()) {
            const uint32_t c = cmd.cbits[0];
            if (c >= cbit_register.size())
                throw std::runtime_error(
                    "trajectory: Measure writes to cbit "
                    + std::to_string(c) + " outside register of size "
                    + std::to_string(cbit_register.size()));
            cbit_register[c] = outcome_one;
        }
        return true;
    }

    // Reset: identical to Measure but discard the outcome and apply X if 1
    // so the qubit ends in |0⟩.
    if (cmd.gate == GateType::Reset) {
        if (cmd.qubits.empty())
            throw std::runtime_error("trajectory: Reset requires a qubit");
        const uint32_t q = cmd.qubits[0];
        const double p0  = M0_prob(q, s, d);

        std::uniform_real_distribution<double> u(0.0, 1.0);
        const double r = u(rng);
        const bool outcome_one = (r >= p0);

        if (outcome_one) {
            P1_gate(q, s, d);
            normalize(std::max(1.0 - p0, 0.0), s, d);
            X_gate(q, s, d);
        } else {
            P0_gate(q, s, d);
            normalize(std::max(p0, 0.0), s, d);
        }
        return true;
    }

    // Anything else: plain unitary dispatch (or Barrier/Custom no-op).
    apply_command(cmd, state, dim);
    return true;
}

// ── apply_post_gate_noise ─────────────────────────────────────────────────────

void QarpSimulator::apply_post_gate_noise(const Command&        cmd,
                                          std::complex<double>* state,
                                          uint64_t              dim,
                                          std::mt19937&         rng) const {
    if (!noise_model_.enabled) return;

    // Non-evolving / non-physical-gate ops: no error injected.  Measure
    // and Reset model their own stochastic behaviour; Barrier/GPhase have
    // no qubits to perturb.
    switch (cmd.gate) {
        case GateType::Measure:
        case GateType::Reset:
        case GateType::Barrier:
        case GateType::GPhase:
        case GateType::BranchBegin:
        case GateType::BranchElse:
        case GateType::BranchEnd:
            return;
        default: break;
    }
    const auto& chfn = noise_model_.per_gate[static_cast<size_t>(cmd.gate)];
    if (!chfn) return;

    Channel ch = chfn(cmd);
    std::vector<uint32_t> qubits(cmd.qubits.begin(), cmd.qubits.end());
    apply_channel(ch, qubits, state, dim, rng);
}

namespace {

// Caller-supplied initial states must match the register exactly and be
// normalised — reject at 1e-10, no silent renormalisation.
void validate_initial_state(const std::vector<std::complex<double>>& psi,
                            uint64_t dim, const char* where) {
    if (psi.size() != dim)
        throw std::invalid_argument(
            "QarpSimulator::" + std::string(where) + ": initial_state length "
            + std::to_string(psi.size()) + " != 2^n_qubits = "
            + std::to_string(dim));
    double norm2 = 0.0;
    for (const auto& a : psi) norm2 += std::norm(a);
    if (std::abs(std::sqrt(norm2) - 1.0) > 1e-10)
        throw std::invalid_argument(
            "QarpSimulator::" + std::string(where) + ": initial_state is not "
            "normalised: |psi| = " + std::to_string(std::sqrt(norm2))
            + " differs from 1 by more than 1e-10.  Normalise the input "
            "yourself — no silent renormalisation.");
}

}  // namespace

// ── statevector ───────────────────────────────────────────────────────────────

std::vector<std::complex<double>>
QarpSimulator::statevector(const std::vector<Command>& commands,
                             int                          n_qubits,
                             const std::optional<std::vector<std::complex<double>>>&
                                                          initial_state) const {
    if (any_needs_trajectory(commands))
        throw std::runtime_error(
            "QarpSimulator::statevector: circuit contains mid-circuit measurement, "
            "Reset, or classical condition — statevector is not deterministic. "
            "Use run() for trajectory simulation.");
    if (noise_active())
        throw std::runtime_error(
            "QarpSimulator::statevector: noise model is active — the noisy "
            "ensemble is not representable as a single statevector. "
            "Either disable the noise model (noise_model.enabled=False) or "
            "use run() for trajectory simulation.");

    const ITYPE dim = ITYPE{1} << n_qubits;

    // Every kernel call is a pass over 2^n amplitudes; fusion trades gates
    // for passes (fusion_max_qubits).
    const auto fused = fuse_for_dispatch(commands, n_qubits);

    CTYPE* raw = allocate_quantum_state(dim);
    if (initial_state) {
        validate_initial_state(*initial_state, static_cast<uint64_t>(dim),
                               "statevector");
        std::copy(initial_state->begin(), initial_state->end(),
                  reinterpret_cast<std::complex<double>*>(raw));
    } else {
        initialize_quantum_state(raw, dim);
    }

    for (const auto& cmd : fused)
        apply_command(cmd, reinterpret_cast<std::complex<double>*>(raw), static_cast<uint64_t>(dim));

    std::vector<std::complex<double>> sv(
        reinterpret_cast<std::complex<double>*>(raw),
        reinterpret_cast<std::complex<double>*>(raw) + dim);

    release_quantum_state(raw);
    return sv;
}

// ── expectation / transition / batch_expectation ─────────────────────────────

double QarpSimulator::expectation(
    const std::vector<Command>& commands,
    int                          n_qubits,
    const PauliObservable&       observable,
    const std::optional<std::vector<std::complex<double>>>& initial_state) const {
    const auto psi = statevector(commands, n_qubits, initial_state);
    return pauli_transition(psi.data(), psi.data(), n_qubits, observable).real();
}

std::complex<double> QarpSimulator::transition(
    const std::vector<std::complex<double>>& bra,
    const std::vector<std::complex<double>>& ket,
    int                                      n_qubits,
    const PauliObservable&                   observable) const {
    check_transition_width(n_qubits);
    const uint64_t dim = uint64_t{1} << n_qubits;
    if (bra.size() != dim || ket.size() != dim)
        throw std::invalid_argument(
            "QarpSimulator::transition: bra/ket lengths " + std::to_string(bra.size()) +
            "/" + std::to_string(ket.size()) + " do not match 2^" +
            std::to_string(n_qubits) + " = " + std::to_string(dim));
    return pauli_transition(bra.data(), ket.data(), n_qubits, observable);
}

std::vector<double> QarpSimulator::batch_expectation(
    const std::vector<Command>&                                 commands,
    int                                                         n_qubits,
    const PauliObservable&                                      observable,
    const std::vector<std::unordered_map<std::string, double>>& param_sets) const {
    std::vector<double> out;
    out.reserve(param_sets.size());
    for (const auto& params : param_sets) {
        std::vector<Command> concrete = commands;
        for (auto& cmd : concrete) cmd = cmd.substitute(params);
        out.push_back(expectation(concrete, n_qubits, observable));
    }
    return out;
}

// ── run (shot-based) ──────────────────────────────────────────────────────────

SamplingResult QarpSimulator::run(const std::vector<Command>& commands,
                                    int                          n_qubits,
                                    int                          n_shots,
                                    std::optional<uint32_t>      seed,
                                    const std::optional<std::vector<std::complex<double>>>&
                                                                 initial_state) const {
    // NOTE: a condition reading a cbit no Measure wrote is *defined* here —
    // the register is zero-initialised, so the condition reads false and the
    // guarded body is skipped.  The simulator deliberately does not reject it:
    // randomized property tests and deliberately-dead branches are legitimate.
    // The footgun (a ConditionalBlock composed as a sibling of the measurement
    // feeding it) is surfaced one layer up, where the program is user-authored
    // — see Engine.build and qx.uninitialised_condition_cbits.
    const uint64_t dim    = uint64_t{1} << n_qubits;
    const uint32_t base_s = seed.has_value() ? *seed : std::random_device{}();

    if (initial_state)
        validate_initial_state(*initial_state, dim, "run");

    SamplingResult result;
    result.n_qubits = n_qubits;
    result.n_shots  = n_shots;

    // ── Fast path: no active noise model and every measurement (if any) is
    // terminal.  Run the circuit once to a final statevector and sample the
    // distribution n_shots times; per-shot classical bits are read straight
    // out of each sampled outcome.  Active noise forces the per-shot
    // trajectory path below (stochastic noise can't be represented as a
    // single statevector).
    if (!noise_active() && measurements_are_terminal(commands)) {
        // Inline statevector simulation: statevector() would reject recorded
        // Measure commands, but here they are provably terminal (and a no-op
        // for the state itself in apply_command).
        const auto fused = fuse_for_dispatch(commands, n_qubits);
        std::vector<CTYPE> sv(dim, CTYPE{0., 0.});
        if (initial_state)
            std::copy(initial_state->begin(), initial_state->end(),
                      reinterpret_cast<std::complex<double>*>(sv.data()));
        else
            sv[0] = {1., 0.};
        for (const auto& cmd : fused)
            apply_command(cmd,
                          reinterpret_cast<std::complex<double>*>(sv.data()),
                          dim);

        std::vector<double> probs(dim);
        for (uint64_t i = 0; i < dim; ++i) probs[i] = std::norm(sv[i]);

        // (qubit, cbit) pairs of recorded measurements, in program order.
        std::vector<std::pair<uint32_t, uint32_t>> recorded;
        for (const auto& c : commands)
            if (c.gate == GateType::Measure && !c.cbits.empty())
                recorded.emplace_back(c.qubits[0], c.cbits[0]);
        result.n_cbits = classical_register_width(commands);

        std::mt19937 rng(base_s);
        std::discrete_distribution<uint64_t> dist(probs.begin(), probs.end());
        if (recorded.empty()) {
            for (int i = 0; i < n_shots; ++i)
                result.counts[dist(rng)]++;
        } else {
            result.cbit_history.reserve(n_shots);
            for (int i = 0; i < n_shots; ++i) {
                const uint64_t outcome = dist(rng);
                result.counts[outcome]++;
                std::vector<bool> reg(result.n_cbits, false);
                for (const auto& [q, c] : recorded)
                    reg[c] = (outcome >> q) & 1;
                result.cbit_history.push_back(std::move(reg));
            }
        }
        return result;
    }

    // ── Trajectory path: shots diverge at the first non-deterministic op,
    // *or* at the first gate when noise is active (noise is stochastic per
    // shot, so the prefix can't be shared).
    //
    // Run the unitary prefix once into shared_state; copy into a per-shot
    // buffer and run the suffix via apply_command_trajectory.  Each shot
    // gets a distinct RNG seeded from base_s + shot index for reproducibility.
    const auto split_idx = noise_active() ? std::size_t{0}
                                          : first_trajectory_index(commands);
    const auto fused_prefix = (split_idx > 0)
        ? fuse_for_dispatch(
              std::vector<Command>(commands.begin(), commands.begin() + split_idx), n_qubits)
        : std::vector<Command>{};

    // Pre-fuse the suffix once so the per-shot loop dispatches fewer kernels.
    // Fusion treats Measure / Reset / Branch* as flush barriers, so semantics
    // are preserved.  Branch-table indices are computed against
    // the fused suffix because that's what run_range walks.
    //
    // When noise is active we disable suffix fusion: each gate type carries
    // its own NoiseChannel, and a fused multi-gate kernel would only receive
    // one channel application instead of one per source gate.  Correctness
    // first; a fusion-aware "barrier on differing noise channels" pass is a
    // separate optimisation.
    const std::vector<Command> fused_suffix = noise_active()
        ? std::vector<Command>(commands.begin() + split_idx, commands.end())
        : fuse_for_dispatch(
              std::vector<Command>(commands.begin() + split_idx, commands.end()), n_qubits);

    std::vector<CTYPE> shared_state(dim);
    {
        CTYPE* raw = shared_state.data();
        // initialize_quantum_state expects a malloc'd buffer; do it manually
        // so we don't have to allocate twice.
        if (initial_state) {
            std::copy(initial_state->begin(), initial_state->end(),
                      reinterpret_cast<std::complex<double>*>(raw));
        } else {
            std::fill(raw, raw + dim, CTYPE{0., 0.});
            raw[0] = {1., 0.};
        }
        for (const auto& cmd : fused_prefix)
            apply_command(cmd,
                          reinterpret_cast<std::complex<double>*>(raw),
                          static_cast<uint64_t>(dim));
    }

    const int n_cbits = classical_register_width(commands);
    result.n_cbits = n_cbits;
    result.cbit_history.reserve(n_shots);

    // Build the BranchBegin → (BranchElse, BranchEnd) jump table once over
    // the fused suffix (indices match the suffix, not the original input).
    const auto branch_table = build_branch_table(fused_suffix);

    // Recursive dispatch over a half-open command range [start, end) within
    // the fused suffix.  When BranchBegin is hit, evaluate its AND-of-
    // condition and recurse into the active branch only; the inactive branch
    // (if any) is skipped wholesale.  Recursion depth = nesting depth of
    // conditional blocks; usually shallow.
    std::function<void(std::size_t, std::size_t,
                       std::complex<double>*,
                       std::vector<bool>&,
                       std::mt19937&)> run_range;
    run_range = [&](std::size_t start, std::size_t end,
                    std::complex<double>* shot_buf,
                    std::vector<bool>&    reg,
                    std::mt19937&         rng)
    {
        std::size_t i = start;
        while (i < end) {
            const auto& cmd = fused_suffix[i];
            if (cmd.gate == GateType::BranchBegin) {
                const auto entry = branch_table.by_begin.at(i);
                const bool matched = branch_condition_holds(cmd, reg);
                if (matched) {
                    const std::size_t then_end =
                        (entry.else_idx == BranchTable::kNoElse)
                            ? entry.end_idx : entry.else_idx;
                    run_range(i + 1, then_end, shot_buf, reg, rng);
                } else if (entry.else_idx != BranchTable::kNoElse) {
                    run_range(entry.else_idx + 1, entry.end_idx,
                              shot_buf, reg, rng);
                }
                i = entry.end_idx + 1;
            } else if (cmd.gate == GateType::BranchElse
                    || cmd.gate == GateType::BranchEnd) {
                throw std::runtime_error(
                    "QarpSimulator::run: stray "
                    + std::string(gate_name(cmd.gate)));
            } else {
                const bool applied = apply_command_trajectory(
                    cmd, shot_buf, static_cast<uint64_t>(dim), reg, rng);
                if (applied) {
                    apply_post_gate_noise(cmd, shot_buf,
                                          static_cast<uint64_t>(dim), rng);
                }
                ++i;
            }
        }
    };

    // Per-shot trajectory worker.  Runs shots [shot_start, shot_end) into a
    // local result bundle; merged at the end.  Each shot's RNG is seeded
    // from base_s + shot index so the threading is deterministic.
    struct ShotRange {
        std::unordered_map<uint64_t, int>   counts;
        std::vector<std::vector<bool>>      cbit_history;
    };

    auto run_shots = [&](int shot_start, int shot_end) -> ShotRange {
        ShotRange local;
        local.cbit_history.reserve(shot_end - shot_start);

        std::vector<CTYPE>  shot_state(dim);
        std::vector<double> probs(dim);
        std::uniform_real_distribution<double> u01(0.0, 1.0);

        for (int shot = shot_start; shot < shot_end; ++shot) {
            std::mt19937 rng(base_s + static_cast<uint32_t>(shot));
            std::copy(shared_state.begin(), shared_state.end(),
                      shot_state.begin());
            std::vector<bool> cbit_register(n_cbits, false);

            run_range(0, fused_suffix.size(),
                      reinterpret_cast<std::complex<double>*>(shot_state.data()),
                      cbit_register, rng);

            // Sample marginal (inline CDF walk against reused `probs`).
            double total = 0.0;
            for (uint64_t i = 0; i < dim; ++i) {
                probs[i] = std::norm(shot_state[i]);
                total += probs[i];
            }
            const double threshold = u01(rng) * total;
            double accum = 0.0;
            uint64_t outcome = dim - 1;
            for (uint64_t i = 0; i < dim; ++i) {
                accum += probs[i];
                if (accum >= threshold) { outcome = i; break; }
            }
            local.counts[outcome]++;
            local.cbit_history.push_back(std::move(cbit_register));
        }
        return local;
    };

    // Threading threshold: the per-shot work has to be substantial enough to
    // amortise thread-pool dispatch (~10-100 µs).  At fewer than 16 shots,
    // sequential is faster.
    constexpr int kSerialThreshold = 16;
    if (n_shots < kSerialThreshold) {
        auto local = run_shots(0, n_shots);
        result.counts       = std::move(local.counts);
        result.cbit_history = std::move(local.cbit_history);
        return result;
    }

    // Partition shots into contiguous chunks, one per worker.  Each worker
    // keeps its own ShotRange (no synchronization in the hot loop); the
    // chunks are merged in index order so cbit_history stays in shot order.
    // The partition and the per-shot seeds do not depend on how the chunks
    // are scheduled, so every thread count gives the same result.
    const std::size_t n_threads = std::min<std::size_t>(
        configured_thread_count(), std::max<std::size_t>(1, n_shots / 4));
    const int per_thread = (n_shots + static_cast<int>(n_threads) - 1)
                         / static_cast<int>(n_threads);
    const int n_chunks = (n_shots + per_thread - 1) / per_thread;
    std::vector<ShotRange> locals(static_cast<std::size_t>(n_chunks));

#ifdef _OPENMP
    // The chunks run as one OpenMP team.  csim's per-gate `omp parallel`
    // regions are then nested and serialise (init_threading caps the active
    // levels at 1) — a shot worker never spawns a kernel team.  With a
    // separate pool, each worker's kernel regions contended with the OpenMP
    // runtime's own workers: 1.3 ms/shot at 12 qubits for 0.02 ms of work.
    std::exception_ptr failure;
#pragma omp parallel for num_threads(static_cast<int>(n_threads)) schedule(static, 1)
    for (int t = 0; t < n_chunks; ++t) {
        try {
            const int s0 = t * per_thread;
            locals[static_cast<std::size_t>(t)] = run_shots(s0, std::min(s0 + per_thread, n_shots));
        } catch (...) {
#pragma omp critical(qarpx_shot_failure)
            if (!failure) failure = std::current_exception();
        }
    }
    if (failure) std::rethrow_exception(failure);
#else
    auto& pool = global_thread_pool();
    std::vector<std::future<ShotRange>> futures;
    futures.reserve(static_cast<std::size_t>(n_chunks));
    for (int t = 0; t < n_chunks; ++t) {
        const int s0 = t * per_thread;
        futures.push_back(pool.submit(run_shots, s0, std::min(s0 + per_thread, n_shots)));
    }
    for (int t = 0; t < n_chunks; ++t) locals[static_cast<std::size_t>(t)] = futures[t].get();
#endif

    for (auto& local : locals) {
        for (const auto& [k, v] : local.counts) result.counts[k] += v;
        for (auto& reg : local.cbit_history)
            result.cbit_history.push_back(std::move(reg));
    }

    return result;
}

// ── batch_run ─────────────────────────────────────────────────────────────────

std::vector<SamplingResult>
QarpSimulator::batch_run(
    const std::vector<Command>&                                  commands,
    int                                                           n_qubits,
    int                                                           n_shots,
    const std::vector<std::unordered_map<std::string, double>>&  param_sets,
    std::optional<uint32_t>                                       seed) const {
    std::vector<SamplingResult> results;
    results.reserve(param_sets.size());
    for (size_t i = 0; i < param_sets.size(); ++i) {
        std::vector<Command> concrete = commands;
        for (auto& cmd : concrete)
            cmd = cmd.substitute(param_sets[i]);
        auto s = seed.has_value()
            ? std::optional<uint32_t>(*seed + static_cast<uint32_t>(i))
            : std::optional<uint32_t>{};
        results.push_back(run(concrete, n_qubits, n_shots, s));
    }
    return results;
}

// ── unitary_matrix ────────────────────────────────────────────────────────────

Eigen::MatrixXcd
QarpSimulator::unitary_matrix(const std::vector<Command>& commands,
                               int                          n_qubits) const {
    if (any_needs_trajectory(commands))
        throw std::runtime_error(
            "QarpSimulator::unitary_matrix: circuit contains mid-circuit "
            "measurement, Reset, or classical condition — output is not unitary.");
    if (noise_active())
        throw std::runtime_error(
            "QarpSimulator::unitary_matrix: noise model is active — noisy "
            "evolution is not unitary.  Disable noise (enabled=False) or use "
            "run() for stochastic sampling.");

    const ITYPE dim = ITYPE{1} << n_qubits;
    Eigen::MatrixXcd U(dim, dim);

    for (ITYPE col = 0; col < dim; ++col) {
        // Allocate and initialise to |col⟩
        CTYPE* raw = allocate_quantum_state(dim);
        std::fill(raw, raw + dim, CTYPE{0., 0.});
        raw[col] = {1., 0.};

        for (const auto& cmd : commands)
            apply_command(cmd,
                          reinterpret_cast<std::complex<double>*>(raw),
                          static_cast<uint64_t>(dim));

        for (ITYPE row = 0; row < dim; ++row)
            U(static_cast<int>(row), static_cast<int>(col)) = raw[row];

        release_quantum_state(raw);
    }
    return U;
}

// ── helpers shared by QPE and DOS-QPE ────────────────────────────────────────

namespace {

/// Compute U, U^2, U^4, ..., U^(2^(n_ancilla-1)) by repeated squaring.
std::vector<Eigen::MatrixXcd>
compute_u_powers(const Eigen::MatrixXcd& U, int n_ancilla) {
    std::vector<Eigen::MatrixXcd> pows(n_ancilla);
    pows[0] = U;
    for (int k = 1; k < n_ancilla; ++k)
        pows[k] = pows[k - 1] * pows[k - 1];
    return pows;
}

/// Apply controlled-U^(2^k) to the state vector sv.
///
/// Layout  sv[a | (s << n_a) | (env << (n_a + n_s))]
///   a   = ancilla index         (dim_a values)
///   s   = state-register index  (dim_s values)
///   env = extra-register index  (n_env_extra values; 1 for QPE, dim_s for DOSQPE)
///
/// For each (a, env) where bit k of a is 1, applies U to the state sub-vector
/// (stride = dim_a, base = a + env * dim_a * dim_s).
void apply_controlled_u_power(
    std::vector<CTYPE>&          sv,
    const Eigen::MatrixXcd&      Uk,
    int                          k,            // control qubit bit-position
    int64_t                      dim_a,
    int64_t                      dim_s,
    int64_t                      n_env_extra)  // 1 for QPE, dim_s for DOSQPE
{
    const int64_t stride     = dim_a;
    const int64_t block_size = dim_a * dim_s;  // stride between env blocks

    Eigen::VectorXcd col(dim_s), result(dim_s);

    for (int64_t env = 0; env < n_env_extra; ++env) {
        const int64_t env_offset = env * block_size;
        for (int64_t a = 0; a < dim_a; ++a) {
            if (!((a >> k) & 1)) continue;      // control qubit k not set
            // Extract strided state sub-vector
            for (int64_t s = 0; s < dim_s; ++s)
                col[s] = sv[env_offset + a + s * stride];
            result.noalias() = Uk * col;
            for (int64_t s = 0; s < dim_s; ++s)
                sv[env_offset + a + s * stride] = result[s];
        }
    }
}

/// Sample the ancilla register from sv (marginalise over all other qubits).
SamplingResult
sample_ancilla(const std::vector<CTYPE>& sv,
               int64_t                   dim_total,
               int64_t                   dim_a,
               int                       n_ancilla,
               int                       n_shots,
               std::optional<uint32_t>   seed)
{
    std::vector<double> probs_a(dim_a, 0.0);
    for (int64_t i = 0; i < dim_total; ++i)
        probs_a[i & (dim_a - 1)] += std::norm(sv[i]);

    std::mt19937 rng(seed.has_value() ? *seed : std::random_device{}());
    std::discrete_distribution<uint64_t> dist(probs_a.begin(), probs_a.end());

    SamplingResult result;
    result.n_qubits = n_ancilla;
    result.n_shots  = n_shots;
    for (int i = 0; i < n_shots; ++i)
        result.counts[dist(rng)]++;
    return result;
}

}  // anonymous namespace

// ── simulate_qpe_structured ───────────────────────────────────────────────────

SamplingResult
QarpSimulator::simulate_qpe_structured(
    const std::vector<Command>& u_cmds,
    const std::vector<Command>& state_prep,
    const std::vector<Command>& iqft_cmds,
    int                          n_state,
    int                          n_ancilla,
    int                          n_shots,
    std::optional<uint32_t>      seed) const
{
    const int     n_total  = n_ancilla + n_state;
    const int64_t dim_a    = INT64_C(1) << n_ancilla;
    const int64_t dim_s    = INT64_C(1) << n_state;
    const int64_t dim_total= INT64_C(1) << n_total;

    // 1. U matrix and powers
    auto U      = unitary_matrix(u_cmds, n_state);
    auto U_pows = compute_u_powers(U, n_ancilla);

    // 2. Initialise full state to |0⟩
    std::vector<CTYPE> sv(dim_total, {0., 0.});
    sv[0] = {1., 0.};

    // 3. State prep on state register (qubits 0..n_s-1 → n_a..n_a+n_s-1)
    {
        std::vector<uint32_t> remap(static_cast<size_t>(n_state));
        for (int j = 0; j < n_state; ++j)
            remap[static_cast<size_t>(j)] = static_cast<uint32_t>(j + n_ancilla);
        for (const auto& cmd : state_prep)
            apply_command(cmd.remap_qubits(remap),
                          reinterpret_cast<std::complex<double>*>(sv.data()),
                          static_cast<uint64_t>(dim_total));
    }

    // 4. H on each ancilla qubit (qubits 0..n_a-1)
    for (int k = 0; k < n_ancilla; ++k) {
        Command hcmd(GateType::H, static_cast<uint32_t>(k));
        apply_command(hcmd,
                      reinterpret_cast<std::complex<double>*>(sv.data()),
                      static_cast<uint64_t>(dim_total));
    }

    // 5. Controlled-U^(2^k) ladder (no extra register → n_env_extra = 1)
    for (int k = 0; k < n_ancilla; ++k)
        apply_controlled_u_power(sv, U_pows[k], k, dim_a, dim_s, 1);

    // 6. IQFT on ancilla (qubits 0..n_a-1, no remap needed)
    for (const auto& cmd : iqft_cmds)
        apply_command(cmd,
                      reinterpret_cast<std::complex<double>*>(sv.data()),
                      static_cast<uint64_t>(dim_total));

    // 7. Sample ancilla register
    return sample_ancilla(sv, dim_total, dim_a, n_ancilla, n_shots, seed);
}

// ── simulate_dosqpe_structured ────────────────────────────────────────────────

SamplingResult
QarpSimulator::simulate_dosqpe_structured(
    const std::vector<Command>& u_cmds,
    const std::vector<Command>& state_prep,
    const std::vector<Command>& iqft_cmds,
    int                          n_state,
    int                          n_ancilla,
    int                          n_shots,
    std::optional<uint32_t>      seed) const
{
    const int     n_total  = n_ancilla + 2 * n_state;
    const int64_t dim_a    = INT64_C(1) << n_ancilla;
    const int64_t dim_s    = INT64_C(1) << n_state;
    const int64_t dim_total= INT64_C(1) << n_total;

    // 1. U matrix and powers
    auto U      = unitary_matrix(u_cmds, n_state);
    auto U_pows = compute_u_powers(U, n_ancilla);

    // 2. Initialise full state to |0⟩
    std::vector<CTYPE> sv(dim_total, {0., 0.});
    sv[0] = {1., 0.};

    // 3. State prep on state register (qubits 0..n_s-1 → n_a..n_a+n_s-1)
    {
        std::vector<uint32_t> remap(static_cast<size_t>(n_state));
        for (int j = 0; j < n_state; ++j)
            remap[static_cast<size_t>(j)] = static_cast<uint32_t>(j + n_ancilla);
        for (const auto& cmd : state_prep)
            apply_command(cmd.remap_qubits(remap),
                          reinterpret_cast<std::complex<double>*>(sv.data()),
                          static_cast<uint64_t>(dim_total));
    }

    // 4. CNOT entanglement: state[j] → purif[j]  (CX(n_a+j, n_a+n_s+j))
    for (int j = 0; j < n_state; ++j) {
        Command cx(GateType::CX,
                   static_cast<uint32_t>(n_ancilla + j),
                   static_cast<uint32_t>(n_ancilla + n_state + j));
        apply_command(cx,
                      reinterpret_cast<std::complex<double>*>(sv.data()),
                      static_cast<uint64_t>(dim_total));
    }

    // 5. H on each ancilla qubit
    for (int k = 0; k < n_ancilla; ++k) {
        Command hcmd(GateType::H, static_cast<uint32_t>(k));
        apply_command(hcmd,
                      reinterpret_cast<std::complex<double>*>(sv.data()),
                      static_cast<uint64_t>(dim_total));
    }

    // 6. Controlled-U^(2^k) ladder
    // Extra register = purification (dim_s values) → n_env_extra = dim_s
    for (int k = 0; k < n_ancilla; ++k)
        apply_controlled_u_power(sv, U_pows[k], k, dim_a, dim_s, dim_s);

    // 7. IQFT on ancilla
    for (const auto& cmd : iqft_cmds)
        apply_command(cmd,
                      reinterpret_cast<std::complex<double>*>(sv.data()),
                      static_cast<uint64_t>(dim_total));

    // 8. Sample ancilla register
    return sample_ancilla(sv, dim_total, dim_a, n_ancilla, n_shots, seed);
}

// ── run_gradient (adjoint backpropagation) ───────────────────────────────────
//
// Computes ∂⟨H⟩/∂θ_k for every parameter in `param_order` via reverse-mode
// (adjoint-state) differentiation.  Cost: ~2 statevector simulations
// (one forward, one backward), independent of the parameter count.
//
// For a parametric gate g_j with angle θ_actual = α_j · θ_k + β_j, the
// gradient contribution is
//
//     ∂⟨H⟩/∂θ_k from g_j = α_j · 2·Re(⟨φ_j| M_g |ψ_j⟩)
//
// where |ψ_j⟩ is the state AFTER gate j, |φ_j⟩ is the back-propagated
// adjoint state (H|ψ_n⟩ pulled back through g_n†, …, g_{j+1}†), and M_g is
// the gate's "left multiplier" (∂g/∂θ_actual · g⁻¹).  See the algorithm
// description in qarp_simulator.h.
//
// M_g per gate type:
//   Rx(θ)            — M = -i/2 · X_t
//   Ry(θ)            — M = -i/2 · Y_t
//   Rz(θ)            — M = -i/2 · Z_t
//   P(θ)             — M = i · |1⟩⟨1|_t
//   GPhase(θ)        — M = i · I
//   CRx/CRy/CRz(θ)   — M = -i/2 · |1⟩⟨1|_c ⊗ {X|Y|Z}_t
//   CP(θ)            — M = i · |11⟩⟨11|
//   Rxx/Ryy/Rzz(θ)   — M = -i/2 · {X⊗X|Y⊗Y|Z⊗Z}
//
// The adjoint of every parametric gate equals the same gate type with the
// angle negated (``Rx(-θ)`` etc.) — used to step both ψ and φ backward.

namespace {

// Apply ``M_g · |ψ⟩`` in-place for the gate ``cmd``.  Caller is
// responsible for restoring or discarding ``state`` afterwards (it is
// destroyed by this routine).  Returns the prefactor ``c`` such that the
// raw inner product ``⟨φ| ψ_after ⟩`` represents the dressed quantity
// ``c · ⟨φ| G |ψ⟩``.  The gradient contribution is then
// ``α · 2·Re(c · ⟨φ| ψ_after⟩)``.
//
// We split (G, c) rather than folding c into ψ to keep the per-gate
// arithmetic close to the existing csim primitives.

struct GeneratorAction {
    std::complex<double> prefactor;  // c such that 2·Re(c · <φ|ψ_after>) = ∂<H>/∂θ at α=1.
    bool                 active;     // false ⇒ gate not parametric
};

GeneratorAction apply_generator(const Command& cmd,
                                std::complex<double>* state,
                                uint64_t dim) {
    auto* s = reinterpret_cast<CTYPE*>(state);
    ITYPE  d = static_cast<ITYPE>(dim);
    const uint32_t q0 = cmd.qubits.empty() ? 0 : cmd.qubits[0];
    const uint32_t q1 = cmd.qubits.size() > 1 ? cmd.qubits[1] : 0;

    switch (cmd.gate) {
        case GateType::Rx: {
            X_gate(q0, s, d);
            return {std::complex<double>{0.0, -0.5}, true};  // -i/2
        }
        case GateType::Ry: {
            Y_gate(q0, s, d);
            return {std::complex<double>{0.0, -0.5}, true};
        }
        case GateType::Rz: {
            Z_gate(q0, s, d);
            return {std::complex<double>{0.0, -0.5}, true};
        }
        case GateType::P: {
            // M = i · |1⟩⟨1|_t.  Project ψ onto |1⟩_t (zero amplitudes
            // where bit q0 = 0).
            const ITYPE bit = ITYPE{1} << q0;
            for (ITYPE i = 0; i < d; ++i)
                if ((i & bit) == 0) state[i] = std::complex<double>{0.0, 0.0};
            return {std::complex<double>{0.0, 1.0}, true};  // +i
        }
        case GateType::GPhase: {
            // M = i · I — leave ψ unchanged.
            return {std::complex<double>{0.0, 1.0}, true};
        }
        case GateType::CRx:
        case GateType::CRy:
        case GateType::CRz: {
            // M = -i/2 · |1⟩⟨1|_c ⊗ G_t.  Project on control = |1⟩, then apply G_t.
            const ITYPE ctrl_bit = ITYPE{1} << q0;
            for (ITYPE i = 0; i < d; ++i)
                if ((i & ctrl_bit) == 0) state[i] = std::complex<double>{0.0, 0.0};
            if (cmd.gate == GateType::CRx) X_gate(q1, s, d);
            else if (cmd.gate == GateType::CRy) Y_gate(q1, s, d);
            else                                Z_gate(q1, s, d);
            return {std::complex<double>{0.0, -0.5}, true};
        }
        case GateType::CP: {
            // M = i · |11⟩⟨11|.  Project on (control = 1) AND (target = 1).
            const ITYPE c_bit = ITYPE{1} << q0;
            const ITYPE t_bit = ITYPE{1} << q1;
            const ITYPE both  = c_bit | t_bit;
            for (ITYPE i = 0; i < d; ++i)
                if ((i & both) != both) state[i] = std::complex<double>{0.0, 0.0};
            return {std::complex<double>{0.0, 1.0}, true};
        }
        case GateType::RXX:
        case GateType::RYY:
        case GateType::RZZ: {
            // M = -i/2 · G⊗G on (q0, q1).
            if (cmd.gate == GateType::RXX) { X_gate(q0, s, d); X_gate(q1, s, d); }
            else if (cmd.gate == GateType::RYY) { Y_gate(q0, s, d); Y_gate(q1, s, d); }
            else                                { Z_gate(q0, s, d); Z_gate(q1, s, d); }
            return {std::complex<double>{0.0, -0.5}, true};
        }
        default:
            return {std::complex<double>{0.0, 0.0}, false};
    }
}

// Apply g_j† to a state in place.
//
// §9 is implemented once, in Command::dagger() (table-driven via
// gate_adjoint / gate_is_self_adjoint).  This used to carry its own
// switch, which drifted: it named S↔Sdg and T↔Tdg but let the other four
// named-inverse pairs (SX↔SXdg, CS↔CSdg, CSX↔CSXdg, iSWAP↔iSWAPdg) fall into
// a default labelled "self-adjoint", so the backward sweep un-applied
// them with the wrong sign and every gradient through such a circuit was
// silently wrong.  Delegate instead — one source of truth for §9.
void apply_command_dagger(const QarpSimulator& sim,
                          const Command& cmd,
                          std::complex<double>* state,
                          uint64_t dim) {
    sim.apply_command(cmd.dagger(), state, dim);
}

// Apply a Pauli string (list of (qubit, 'X'|'Y'|'Z')) to a state in place.
void apply_pauli_string(const std::vector<std::pair<uint32_t, char>>& pauli,
                        std::complex<double>* state,
                        uint64_t dim) {
    auto* s = reinterpret_cast<CTYPE*>(state);
    ITYPE  d = static_cast<ITYPE>(dim);
    for (const auto& [q, op] : pauli) {
        if (q >= 64 || (d >> q) <= 1) {
            throw std::invalid_argument(
                "run_gradient: observable qubit " + std::to_string(q) +
                " out of range for a state of dimension " + std::to_string(d));
        }
        if      (op == 'X') X_gate(q, s, d);
        else if (op == 'Y') Y_gate(q, s, d);
        else if (op == 'Z') Z_gate(q, s, d);
        else if (op == 'I') { /* no-op */ }
        else throw std::runtime_error(
            "run_gradient: invalid Pauli '" + std::string(1, op) + "'");
    }
}

// Compute |φ⟩ = H |ψ⟩ in `out`, where H = Σ c_t · P_t.  ``ψ`` is consumed
// (modified by intermediate Pauli applications) and restored at exit.
void apply_observable(
    const std::vector<std::pair<std::vector<std::pair<uint32_t, char>>,
                                std::complex<double>>>& observable,
    const std::complex<double>* psi,
    std::complex<double>*       out,
    uint64_t dim)
{
    // Zero the accumulator.
    for (uint64_t i = 0; i < dim; ++i) out[i] = std::complex<double>{0.0, 0.0};

    // Scratch buffer that will receive each Pauli-applied copy of ψ.
    std::vector<std::complex<double>> scratch(dim);

    for (const auto& [pauli, coeff] : observable) {
        // Copy ψ into scratch and apply the Pauli string in place.
        for (uint64_t i = 0; i < dim; ++i) scratch[i] = psi[i];
        apply_pauli_string(pauli, scratch.data(), dim);
        // Accumulate: out += coeff · (P · ψ).
        for (uint64_t i = 0; i < dim; ++i) out[i] += coeff * scratch[i];
    }
}

// ``(parameter index, ∂angle/∂symbol)`` pairs for every tracked symbol any
// of the gate's parameters depends on, via the shared affine probe
// (``affine_coefficients``): a symbol appearing in several gates sums across
// them, a compound-affine angle such as ``(φ+λ)/2`` contributes to both, and
// a non-affine angle throws ``capability_error``.  Empty when the gate is
// concrete or none of its symbols is tracked.
std::vector<std::pair<size_t, double>> gate_param_dependencies(
    const Command& cmd,
    const std::unordered_map<std::string, size_t>& param_index)
{
    std::vector<std::pair<size_t, double>> deps;
    for (const Param& p : cmd.params) {
        if (!p.is_symbolic()) continue;
        std::unordered_map<std::string, double> coeffs;
        try {
            coeffs = affine_coefficients(p);
        } catch (const capability_error& e) {
            throw capability_error(std::string("run_gradient: ") + e.what(), cmd);
        }
        for (const auto& [name, alpha] : coeffs) {
            auto it = param_index.find(name);
            if (it == param_index.end()) continue;  // not in tracked params
            deps.emplace_back(it->second, alpha);
        }
    }
    return deps;
}

}  // anonymous namespace

// Shared backbone for both run_gradient overloads: forward-simulate to
// |ψ_n⟩, build |φ⟩ via ``phi_builder``, then sweep backward accumulating
// 2·Re(c·⟨φ|G|ψ⟩)·α at each parametric gate.
//
// ``phi_builder(psi)`` receives the freshly-computed |ψ_n⟩ and must populate
// the returned vector with the desired |φ⟩.  Splitting this out lets the
// observable-based and caller-supplied-|φ⟩ entry points share the expensive
// per-gate backward sweep without code duplication.
template <typename PhiBuilder>
static std::vector<double> run_gradient_core(
    const QarpSimulator&                              sim,
    const std::vector<Command>&                       commands,
    int                                               n_qubits,
    const std::unordered_map<std::string, double>&    params,
    const std::vector<std::string>&                   param_order,
    const std::optional<std::vector<std::complex<double>>>& initial_state,
    PhiBuilder&&                                      phi_builder)
{
    if (any_needs_trajectory(commands))
        throw std::runtime_error(
            "QarpSimulator::run_gradient: trajectory commands (Measure / "
            "Reset / classical condition) are not differentiable.");

    const ITYPE dim = ITYPE{1} << n_qubits;
    const uint64_t dim_u = static_cast<uint64_t>(dim);

    std::unordered_map<std::string, size_t> param_index;
    for (size_t i = 0; i < param_order.size(); ++i)
        param_index[param_order[i]] = i;

    // Substitute parameter values into a working copy of the command stream
    // (concrete) — we keep ``commands`` as the source of symbolic info so the
    // backward sweep can read each parametric gate's linear coefficient.
    std::vector<Command> concrete = commands;
    for (auto& cmd : concrete) {
        for (auto& p : cmd.params) {
            if (p.is_symbolic()) p = Param(p.evaluate(params));
        }
    }

    // 1. Forward pass: |ψ_n⟩ = U|ψ₀⟩ (|0…0⟩ unless seeded).
    CTYPE* psi_raw = allocate_quantum_state(dim);
    initialize_quantum_state(psi_raw, dim);
    auto* psi = reinterpret_cast<std::complex<double>*>(psi_raw);
    if (initial_state) {
        validate_initial_state(*initial_state, dim_u, "run_gradient");
        std::copy(initial_state->begin(), initial_state->end(), psi);
    }
    for (const auto& cmd : concrete)
        sim.apply_command(cmd, psi, dim_u);

    // 2. |φ⟩ — supplied by caller via phi_builder.
    std::vector<std::complex<double>> phi = phi_builder(psi, dim_u);
    if (phi.size() != dim_u)
        throw std::runtime_error(
            "run_gradient: |φ⟩ size mismatch (expected 2^n_qubits)");

    // 3. Backward sweep.
    std::vector<double> grad(param_order.size(), 0.0);
    std::vector<std::complex<double>> psi_scratch(dim_u);

    for (size_t idx = concrete.size(); idx-- > 0;) {
        const Command& cmd_concrete = concrete[idx];
        const Command& cmd_symbolic = commands[idx];

        auto deps = gate_param_dependencies(cmd_symbolic, param_index);
        if (!deps.empty()) {
            for (uint64_t i = 0; i < dim_u; ++i) psi_scratch[i] = psi[i];
            auto action = apply_generator(
                cmd_concrete, psi_scratch.data(), dim_u);
            if (!action.active) {
                // A tracked symbol on a gate without a generator rule must be
                // loud: contributing zero here was the silent U/CU bug.
                throw capability_error(
                    "run_gradient: gate '" + std::string(gate_name(cmd_symbolic.gate))
                    + "' carries a symbolic parameter but has no adjoint generator "
                      "rule (Rx/Ry/Rz/P/GPhase/CRx/CRy/CRz/CP/RXX/RYY/RZZ); rebase it "
                      "to those gates before differentiating.",
                    cmd_symbolic);
            }
            std::complex<double> ip{0.0, 0.0};
            for (uint64_t i = 0; i < dim_u; ++i)
                ip += std::conj(phi[i]) * psi_scratch[i];
            const double base = 2.0 * std::real(action.prefactor * ip);
            for (const auto& [p_idx, alpha] : deps)
                grad[p_idx] += alpha * base;
        }

        apply_command_dagger(sim, cmd_concrete, psi, dim_u);
        apply_command_dagger(sim, cmd_concrete, phi.data(), dim_u);
    }

    release_quantum_state(psi_raw);
    return grad;
}

std::vector<double> QarpSimulator::run_gradient(
    const std::vector<Command>& commands,
    int                          n_qubits,
    const std::vector<
        std::pair<std::vector<std::pair<uint32_t, char>>,
                  std::complex<double>>>&             observable,
    const std::unordered_map<std::string, double>&    params,
    const std::vector<std::string>&                   param_order,
    const std::optional<std::vector<std::complex<double>>>& initial_state) const
{
    return run_gradient_core(
        *this, commands, n_qubits, params, param_order, initial_state,
        [&observable](const std::complex<double>* psi, uint64_t dim_u) {
            std::vector<std::complex<double>> phi(dim_u);
            apply_observable(observable, psi, phi.data(), dim_u);
            return phi;
        });
}

std::vector<double> QarpSimulator::run_gradient_phi(
    const std::vector<Command>&                       commands,
    int                                               n_qubits,
    const std::vector<std::complex<double>>&          phi,
    const std::unordered_map<std::string, double>&    params,
    const std::vector<std::string>&                   param_order,
    const std::optional<std::vector<std::complex<double>>>& initial_state) const
{
    return run_gradient_core(
        *this, commands, n_qubits, params, param_order, initial_state,
        [&phi](const std::complex<double>* /*psi*/, uint64_t /*dim_u*/) {
            return phi;  // copy
        });
}

}  // namespace qarpx
