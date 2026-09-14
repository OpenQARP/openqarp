#include "qarpx/transpiler/fusion.h"

#include <Eigen/Dense>
#include <csim/type.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>
#include <unordered_map>

namespace qarpx {

// ── Matrix helpers ────────────────────────────────────────────────────────────

// Internal 2×2 complex matrix stored row-major:  [M00, M01, M10, M11].
using Mat2 = std::array<CTYPE, 4>;

static constexpr CTYPE I1 = {1.0, 0.0};
static constexpr CTYPE I0 = {0.0, 0.0};

static Mat2 eye2() { return {I1, I0, I0, I1}; }

/// A = B * A  (in-place left-multiply: A gets the combined operation where
/// B is applied *after* A from the state-update perspective, but since we
/// accumulate left-to-right and apply the product at the end we compose as
/// A_new = B * A_old).
static void mat2_lmul(Mat2& A, const Mat2& B) {
    CTYPE c00 = B[0]*A[0] + B[1]*A[2];
    CTYPE c01 = B[0]*A[1] + B[1]*A[3];
    CTYPE c10 = B[2]*A[0] + B[3]*A[2];
    CTYPE c11 = B[2]*A[1] + B[3]*A[3];
    A[0] = c00; A[1] = c01; A[2] = c10; A[3] = c11;
}

static bool is_identity(const Mat2& m, double tol) {
    return std::abs(m[0] - 1.0) < tol
        && std::abs(m[1])       < tol
        && std::abs(m[2])       < tol
        && std::abs(m[3] - 1.0) < tol;
}

// ── Matrix extraction for every fusible gate type ────────────────────────────

/// Returns the 2×2 row-major unitary for a single-qubit gate with concrete
/// parameters.  Throws if the gate is not a known 1-qubit type or has
/// symbolic parameters.
static Mat2 gate_matrix(const Command& cmd) {
    const double pi = std::numbers::pi;
    auto val = [&](int i) -> double {
        return cmd.params[static_cast<std::size_t>(i)].value();
    };

    switch (cmd.gate) {
        case GateType::X:
            return {I0, I1, I1, I0};
        case GateType::Y:
            return {I0, CTYPE{0.,-1.}, CTYPE{0.,1.}, I0};
        case GateType::Z:
            return {I1, I0, I0, CTYPE{-1.,0.}};
        case GateType::H: {
            const double s = 1.0 / std::sqrt(2.0);
            return {CTYPE{s,0.}, CTYPE{s,0.}, CTYPE{s,0.}, CTYPE{-s,0.}};
        }
        case GateType::S:
            return {I1, I0, I0, CTYPE{0., 1.}};
        case GateType::Sdg:
            return {I1, I0, I0, CTYPE{0.,-1.}};
        case GateType::T: {
            const double c = std::cos(pi/4), s = std::sin(pi/4);
            return {I1, I0, I0, CTYPE{c, s}};
        }
        case GateType::Tdg: {
            const double c = std::cos(pi/4), s = std::sin(pi/4);
            return {I1, I0, I0, CTYPE{c,-s}};
        }
        case GateType::Rx: {
            double th = val(0);
            double c = std::cos(th/2), s = std::sin(th/2);
            return {CTYPE{c,0.}, CTYPE{0.,-s}, CTYPE{0.,-s}, CTYPE{c,0.}};
        }
        case GateType::Ry: {
            double th = val(0);
            double c = std::cos(th/2), s = std::sin(th/2);
            return {CTYPE{c,0.}, CTYPE{-s,0.}, CTYPE{s,0.}, CTYPE{c,0.}};
        }
        case GateType::Rz: {
            double th = val(0);
            return {CTYPE{std::cos(-th/2), std::sin(-th/2)}, I0,
                    I0, CTYPE{std::cos(th/2), std::sin(th/2)}};
        }
        case GateType::P: {
            double th = val(0);
            return {I1, I0, I0, CTYPE{std::cos(th), std::sin(th)}};
        }
        case GateType::U: {
            double th = val(0), phi = val(1), lam = val(2);
            double c = std::cos(th/2), s = std::sin(th/2);
            return {
                CTYPE{c, 0.},
                CTYPE{-std::cos(lam)*s, -std::sin(lam)*s},
                CTYPE{ std::cos(phi)*s,  std::sin(phi)*s},
                CTYPE{ std::cos(phi+lam)*c, std::sin(phi+lam)*c}
            };
        }
        default:
            throw std::runtime_error("gate_matrix: unsupported gate type");
    }
}

/// True if the gate type is a fusible single-qubit unitary and all its
/// parameters (if any) are concrete.
static bool is_fusible_1q(const Command& cmd) {
    // Must have exactly one qubit target
    if (cmd.qubits.size() != 1) return false;

    switch (cmd.gate) {
        // No-param 1-qubit unitaries
        case GateType::X: case GateType::Y: case GateType::Z: case GateType::H:
        case GateType::S: case GateType::Sdg: case GateType::T: case GateType::Tdg:
            return true;
        // Parametric 1-qubit unitaries — only if concrete
        case GateType::Rx: case GateType::Ry: case GateType::Rz: case GateType::P:
        case GateType::U:
            return !cmd.is_parametric();
        default:
            return false;
    }
}

// ── Accumulator state ─────────────────────────────────────────────────────────

struct Accumulator {
    Mat2        matrix   = eye2();
    uint32_t    n_gates  = 0;       // number of gates accumulated since last flush
    uint32_t    qubit    = 0;
    Command     first;              // valid when n_gates >= 1; re-emitted on lone flush

    // Classical condition shared by every gate currently in the accumulator.
    // A gate with a different condition tuple forces a flush, then starts a
    // fresh accumulator carrying its own condition.
    SmallVector<uint32_t, 1> condition_bits;
    SmallVector<bool, 1>     condition_values;
};

/// Two condition tuples are compatible if they match exactly.  Empty tuple
/// (unconditional) matches itself only — an unconditional gate cannot be
/// fused with a conditional one.
static bool condition_matches(const Accumulator& acc, const Command& cmd) {
    return acc.condition_bits   == cmd.condition_bits
        && acc.condition_values == cmd.condition_values;
}

// Emit the accumulator's contents and reset it.
//   n_gates == 0  → nothing to emit.
//   n_gates == 1  → re-emit the original Command verbatim.  Replacing one
//                   gate with one Custom would save no kernel calls and
//                   would obscure the gate type for downstream consumers
//                   that don't understand Custom (e.g. QIREmitter).
//   n_gates >= 2  → emit a single Custom carrying the fused 2×2 matrix
//                   (suppressing it if the product is the identity, e.g.
//                   H · H or Rz(θ) · Rz(-θ)).
static void flush(Accumulator& acc, std::vector<Command>& out, double tol) {
    if (acc.n_gates == 0) return;

    if (acc.n_gates == 1) {
        out.push_back(std::move(acc.first));
    } else if (!is_identity(acc.matrix, tol)) {
        auto eigen_mat = std::make_shared<Eigen::MatrixXcd>(2, 2);
        (*eigen_mat)(0,0) = acc.matrix[0];
        (*eigen_mat)(0,1) = acc.matrix[1];
        (*eigen_mat)(1,0) = acc.matrix[2];
        (*eigen_mat)(1,1) = acc.matrix[3];

        Command fused(GateType::Custom, acc.qubit);
        fused.unitary = std::move(eigen_mat);
        // Fused output inherits the accumulator's classical condition so the
        // simulator skips the entire fused 2×2 application when the condition
        // is false.
        fused.condition_bits   = acc.condition_bits;
        fused.condition_values = acc.condition_values;
        out.push_back(std::move(fused));
    }

    acc.matrix  = eye2();
    acc.n_gates = 0;
    acc.first   = Command{};
    acc.condition_bits.clear();
    acc.condition_values.clear();
}

// ── Public API ────────────────────────────────────────────────────────────────

std::vector<Command> fuse_single_qubit_gates(
    const std::vector<Command>& commands,
    double identity_tol)
{
    if (commands.empty()) return {};

    // Determine the maximum qubit index to size the accumulator table.
    uint32_t max_q = 0;
    for (const auto& cmd : commands)
        for (auto q : cmd.qubits)
            max_q = std::max(max_q, q);

    std::vector<Accumulator> accs(max_q + 1);
    for (uint32_t q = 0; q <= max_q; ++q)
        accs[q].qubit = q;

    std::vector<Command> out;
    out.reserve(commands.size());

    for (const auto& cmd : commands) {
        // ── Fusible single-qubit gate ─────────────────────────────────────
        if (is_fusible_1q(cmd)) {
            uint32_t q = cmd.qubits[0];
            // A condition mismatch flushes whatever was accumulated and starts
            // a fresh accumulator carrying this gate's condition.  Equal-
            // condition (including both empty) accumulates normally.
            if (accs[q].n_gates >= 1 && !condition_matches(accs[q], cmd)) {
                flush(accs[q], out, identity_tol);
            }
            if (accs[q].n_gates == 0) {
                accs[q].first            = cmd;
                accs[q].condition_bits   = cmd.condition_bits;
                accs[q].condition_values = cmd.condition_values;
            }
            mat2_lmul(accs[q].matrix, gate_matrix(cmd));
            ++accs[q].n_gates;
            continue;
        }

        // ── Barrier — fence exactly its listed qubits (§16); an empty list
        //    fences the whole register (OpenQASM `barrier;`).  Without the
        //    flush the accumulator stays open and its fused Custom is emitted
        //    *after* the barrier — gates hopping the fence.
        if (cmd.gate == GateType::Barrier) {
            if (cmd.qubits.empty()) {
                for (auto& acc : accs) flush(acc, out, identity_tol);
            } else {
                for (auto q : cmd.qubits) flush(accs[q], out, identity_tol);
            }
            out.push_back(cmd);
            continue;
        }

        // ── Single-qubit gate with symbolic params — flush then emit ──────
        if (cmd.qubits.size() == 1
                && cmd.gate != GateType::Measure && cmd.gate != GateType::Reset) {
            flush(accs[cmd.qubits[0]], out, identity_tol);
            out.push_back(cmd);
            continue;
        }

        // ── Multi-qubit gate — flush all touched qubits, then emit ────────
        if (cmd.qubits.size() >= 2) {
            for (auto q : cmd.qubits)
                flush(accs[q], out, identity_tol);
            out.push_back(cmd);
            continue;
        }

        // ── Measure / Reset / GPhase / Branch* / other ───────────────────
        // For Measure and Reset, flush the qubit first (observable outcome may
        // depend on the accumulated rotation).  Branch markers are barriers:
        // we cannot fuse across a conditional boundary because the dispatch
        // is decided at run time.
        if ((cmd.gate == GateType::Measure || cmd.gate == GateType::Reset)
                && !cmd.qubits.empty()) {
            flush(accs[cmd.qubits[0]], out, identity_tol);
        }
        // A Measure also fences every accumulator whose condition reads a
        // cbit it writes: emitted after the write, a conditional gate that
        // ran against the zero-initialised register would run against the
        // outcome instead (2026-09-13 regression, test_simulation_fusion).
        if (cmd.gate == GateType::Measure) {
            for (auto& acc : accs) {
                if (acc.n_gates == 0) continue;
                bool reads = false;
                for (auto c : cmd.cbits)
                    for (auto b : acc.condition_bits)
                        if (b == c) reads = true;
                if (reads) flush(acc, out, identity_tol);
            }
        }
        if (cmd.gate == GateType::BranchBegin
                || cmd.gate == GateType::BranchElse
                || cmd.gate == GateType::BranchEnd) {
            for (auto& acc : accs) flush(acc, out, identity_tol);
        }
        out.push_back(cmd);
    }

    // Flush any remaining accumulators
    for (auto& acc : accs)
        flush(acc, out, identity_tol);

    return out;
}

}  // namespace qarpx
