// Independent reference unitary for pass-equivalence testing.
//
// Builds the full 2^n × 2^n circuit unitary by explicit gate-matrix
// embedding, straight from the analytic definitions pinned in
// qarp_conventions.md (§1 LSB endianness, §2-§7 gate matrices, rotations
// exp(-iθP/2), q0 = control) — deliberately sharing NO code with csim /
// QarpSimulator, so a kernel or embedding bug cannot cancel out of an
// equivalence check.  Test-only; keep it dumb and literal.
#pragma once

#include "qarpx/core/command.h"
#include "qarpx/core/gates.h"

#include <Eigen/Dense>

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

namespace qarpx::test {

namespace refu_detail {

using Mat = Eigen::MatrixXcd;
using CD = std::complex<double>;
constexpr CD kI{0.0, 1.0};

inline Mat kron(const Mat& a, const Mat& b) {
    Mat out(a.rows() * b.rows(), a.cols() * b.cols());
    for (Eigen::Index i = 0; i < a.rows(); ++i)
        for (Eigen::Index j = 0; j < a.cols(); ++j)
            out.block(i * b.rows(), j * b.cols(), b.rows(), b.cols()) =
                a(i, j) * b;
    return out;
}

inline Mat mat_x() { Mat m(2, 2); m << 0, 1, 1, 0; return m; }
inline Mat mat_y() { Mat m(2, 2); m << 0, -kI, kI, 0; return m; }
inline Mat mat_z() { Mat m(2, 2); m << 1, 0, 0, -1; return m; }
inline Mat mat_h() {
    Mat m(2, 2);
    m << 1, 1, 1, -1;
    return m / std::sqrt(2.0);
}
inline Mat mat_p(double lam) {
    Mat m = Mat::Identity(2, 2);
    m(1, 1) = std::exp(kI * lam);
    return m;
}
inline Mat mat_rx(double t) {
    const double c = std::cos(t / 2), s = std::sin(t / 2);
    Mat m(2, 2);
    m << c, -kI * s, -kI * s, c;
    return m;
}
inline Mat mat_ry(double t) {
    const double c = std::cos(t / 2), s = std::sin(t / 2);
    Mat m(2, 2);
    m << c, -s, s, c;
    return m;
}
inline Mat mat_rz(double t) {
    Mat m = Mat::Zero(2, 2);
    m(0, 0) = std::exp(-kI * (t / 2));
    m(1, 1) = std::exp(kI * (t / 2));
    return m;
}
inline Mat mat_u(double theta, double phi, double lam) {
    const double c = std::cos(theta / 2), s = std::sin(theta / 2);
    Mat m(2, 2);
    m << c, -std::exp(kI * lam) * s,
         std::exp(kI * phi) * s, std::exp(kI * (phi + lam)) * c;
    return m;
}

// §2.6: SX = e^{iπ/4}·Rx(π/2); SXdg = e^{-iπ/4}·Rx(-π/2).
inline Mat mat_sx()   { return std::exp(kI * (M_PI / 4)) * mat_rx(M_PI / 2); }
inline Mat mat_sxdg() { return std::exp(-kI * (M_PI / 4)) * mat_rx(-M_PI / 2); }

/// Controlled 1q gate with control = local bit 0, target = local bit 1
/// (local index = c + 2t, matching bit b ↔ qubits[b]).
inline Mat controlled_1q(const Mat& u) {
    Mat m = Mat::Identity(4, 4);
    // c = 1 rows/cols: indices 1 (t=0) and 3 (t=1).
    m(1, 1) = u(0, 0); m(1, 3) = u(0, 1);
    m(3, 1) = u(1, 0); m(3, 3) = u(1, 1);
    return m;
}

/// exp(-i t/2 · P⊗P) for a 2q Pauli product (bit0 = qubits[0]).
inline Mat two_pauli_rotation(const Mat& p, double t) {
    const Mat pp = kron(p, p);  // kron LSB-last: bit0 ↔ qubits[0] either way (P⊗P symmetric)
    return std::cos(t / 2) * Mat::Identity(4, 4) - kI * std::sin(t / 2) * pp;
}

/// Gate-local unitary: 2^k × 2^k with local bit b ↔ cmd.qubits[b].
inline Mat gate_matrix_of(const Command& cmd) {
    auto par = [&](std::size_t i) { return cmd.params[i].value(); };
    switch (cmd.gate) {
        case GateType::X:   return mat_x();
        case GateType::Y:   return mat_y();
        case GateType::Z:   return mat_z();
        case GateType::H:   return mat_h();
        case GateType::S:   return mat_p(M_PI / 2);
        case GateType::Sdg: return mat_p(-M_PI / 2);
        case GateType::T:   return mat_p(M_PI / 4);
        case GateType::Tdg: return mat_p(-M_PI / 4);
        case GateType::SX:   return mat_sx();
        case GateType::SXdg: return mat_sxdg();
        case GateType::Id:   return Mat::Identity(2, 2);
        case GateType::Rx:  return mat_rx(par(0));
        case GateType::Ry:  return mat_ry(par(0));
        case GateType::Rz:  return mat_rz(par(0));
        case GateType::P:   return mat_p(par(0));
        case GateType::U:   return mat_u(par(0), par(1), par(2));

        case GateType::CX:  return controlled_1q(mat_x());
        case GateType::CY:  return controlled_1q(mat_y());
        case GateType::CZ:  return controlled_1q(mat_z());
        case GateType::CRx: return controlled_1q(mat_rx(par(0)));
        case GateType::CRy: return controlled_1q(mat_ry(par(0)));
        case GateType::CRz: return controlled_1q(mat_rz(par(0)));
        case GateType::CP:  return controlled_1q(mat_p(par(0)));
        case GateType::CU:  // CU(θ,φ,λ,γ): γ = global phase on the controlled block
            return controlled_1q(std::exp(kI * par(3)) *
                                 mat_u(par(0), par(1), par(2)));

        case GateType::CH:    return controlled_1q(mat_h());
        case GateType::CS:    return controlled_1q(mat_p(M_PI / 2));
        case GateType::CSdg:  return controlled_1q(mat_p(-M_PI / 2));
        case GateType::CSX:   return controlled_1q(mat_sx());
        case GateType::CSXdg: return controlled_1q(mat_sxdg());

        case GateType::RZZ: return two_pauli_rotation(mat_z(), par(0));
        case GateType::RXX: return two_pauli_rotation(mat_x(), par(0));
        case GateType::RYY: return two_pauli_rotation(mat_y(), par(0));

        case GateType::SWAP: {
            Mat m = Mat::Zero(4, 4);
            m(0, 0) = m(3, 3) = 1;
            m(1, 2) = m(2, 1) = 1;
            return m;
        }
        case GateType::iSWAP: {
            Mat m = Mat::Zero(4, 4);
            m(0, 0) = m(3, 3) = 1;
            m(1, 2) = m(2, 1) = kI;
            return m;
        }
        case GateType::iSWAPdg: {
            Mat m = Mat::Zero(4, 4);
            m(0, 0) = m(3, 3) = 1;
            m(1, 2) = m(2, 1) = -kI;
            return m;
        }

        case GateType::ECR: {  // §2.5 echoed cross-resonance, LSB-first (bit0 = qubits[0])
            const double s = 1.0 / std::sqrt(2.0);
            Mat m(4, 4);
            m << CD{0, 0}, CD{s, 0}, CD{0, 0},  CD{0, s},
                 CD{s, 0}, CD{0, 0}, CD{0, -s}, CD{0, 0},
                 CD{0, 0}, CD{0, s}, CD{0, 0},  CD{s, 0},
                 CD{0, -s}, CD{0, 0}, CD{s, 0}, CD{0, 0};
            return m;
        }

        case GateType::CCX: {  // controls = bits 0,1; target = bit 2
            Mat m = Mat::Identity(8, 8);
            m(3, 3) = m(7, 7) = 0;   // |c0=1,c1=1,t⟩: 3 = 011, 7 = 111
            m(3, 7) = m(7, 3) = 1;
            return m;
        }
        case GateType::CSWAP: {  // control = bit 0; swap bits 1,2
            Mat m = Mat::Identity(8, 8);
            // c=1: swap t1 (bit1) and t2 (bit2): 011 (3) ↔ 101 (5).
            m(3, 3) = m(5, 5) = 0;
            m(3, 5) = m(5, 3) = 1;
            return m;
        }
        case GateType::MCZ: {  // diag(-1) on all-ones over its qubit tuple
            const Eigen::Index dim =
                Eigen::Index{1} << cmd.qubits.size();
            Mat m = Mat::Identity(dim, dim);
            m(dim - 1, dim - 1) = -1;
            return m;
        }

        case GateType::Custom:
            if (!cmd.unitary)
                throw std::invalid_argument(
                    "reference_unitary: Custom gate without unitary payload");
            return *cmd.unitary;

        default:
            throw std::invalid_argument(
                "reference_unitary: unsupported gate '" +
                std::string(gate_name(cmd.gate)) + "'");
    }
}

/// Embed a 2^k gate-local unitary (local bit b ↔ qs[b]) into 2^n.
inline Mat embed(const Mat& g, const Command& cmd, uint32_t n) {
    const std::size_t dim = std::size_t{1} << n;
    const std::size_t k = cmd.qubits.size();
    const std::size_t gdim = std::size_t{1} << k;
    Mat out = Mat::Zero(static_cast<Eigen::Index>(dim),
                        static_cast<Eigen::Index>(dim));

    for (std::size_t j = 0; j < dim; ++j) {
        std::size_t lj = 0;
        std::size_t rest = j;
        for (std::size_t b = 0; b < k; ++b) {
            lj |= ((j >> cmd.qubits[b]) & 1u) << b;
            rest &= ~(std::size_t{1} << cmd.qubits[b]);
        }
        for (std::size_t li = 0; li < gdim; ++li) {
            std::size_t i = rest;
            for (std::size_t b = 0; b < k; ++b)
                if ((li >> b) & 1u) i |= std::size_t{1} << cmd.qubits[b];
            out(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
                g(static_cast<Eigen::Index>(li), static_cast<Eigen::Index>(lj));
        }
    }
    return out;
}

}  // namespace refu_detail

/// Full-circuit unitary from first principles.  Supports the unitary gate
/// vocabulary + GPhase + Barrier (identity) + Custom; throws on non-unitary
/// / symbolic commands.  LSB: qubit q ↔ bit q of the state index.
inline Eigen::MatrixXcd reference_unitary(const std::vector<Command>& cmds,
                                          uint32_t n_qubits) {
    using refu_detail::kI;
    const std::size_t dim = std::size_t{1} << n_qubits;
    Eigen::MatrixXcd u = Eigen::MatrixXcd::Identity(
        static_cast<Eigen::Index>(dim), static_cast<Eigen::Index>(dim));

    for (const auto& cmd : cmds) {
        for (const auto& p : cmd.params) {
            if (!p.is_concrete())
                throw std::invalid_argument(
                    "reference_unitary: symbolic parameter");
        }
        if (cmd.gate == GateType::Barrier) continue;
        if (cmd.gate == GateType::GPhase) {
            u *= std::exp(kI * cmd.params[0].value());
            continue;
        }
        u = refu_detail::embed(refu_detail::gate_matrix_of(cmd), cmd,
                               n_qubits) * u;
    }
    return u;
}

}  // namespace qarpx::test
