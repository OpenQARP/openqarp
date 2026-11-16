// Shared helpers for per-gate unitary and dagger round-trip tests.
//
// All helpers honour the qarpx LSB-first convention (qarp_conventions.md §1):
// amplitude index i = Σ_k 2^k · b_k where b_k is the value of qubit k.

#pragma once

#include <gtest/gtest.h>
#include "qarpx/qarpx.h"

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <numbers>
#include <sstream>
#include <vector>

namespace qarpx::test {

using cd  = std::complex<double>;
using Mat = Eigen::MatrixXcd;

inline constexpr double DEFAULT_TOL = 1e-10;
inline constexpr double PI          = std::numbers::pi;

/// Sample angles used to exercise parametric gates over a representative range.
inline constexpr std::array<double, 4> ANGLE_SAMPLES = {
    0.0, PI / 3.0, -0.7, 1.234,
};

// ── Utilities ───────────────────────────────────────────────────────────────

inline cd  ei(double theta)         { return cd{std::cos(theta), std::sin(theta)}; }
inline bool bit(int i, int k)       { return (i >> k) & 1; }
inline int  flip(int i, int k)      { return i ^ (1 << k); }

/// Build the full 2^n × 2^n unitary of a single command after transpilation
/// through native_gateset() (so that gates the simulator doesn't dispatch
/// directly are decomposed first).
inline Mat build_unitary(const std::vector<Command>& cmds, int n_qubits) {
    QarpSimulator sim;
    Transpiler t(native_gateset());
    return sim.unitary_matrix(t.transpile(cmds), n_qubits);
}

/// Convenience wrapper for a single-command unitary.
inline Mat build_unitary(const Command& cmd, int n_qubits) {
    return build_unitary(std::vector<Command>{cmd}, n_qubits);
}

/// gtest-friendly element-wise comparison.  Returns AssertionFailure with the
/// max diff and both matrices on mismatch, AssertionSuccess on match.
inline ::testing::AssertionResult expect_unitary_close(
        const Mat& actual, const Mat& expected,
        double tol = DEFAULT_TOL, const char* msg = "") {
    if (actual.rows() != expected.rows() || actual.cols() != expected.cols()) {
        return ::testing::AssertionFailure()
            << (msg && *msg ? std::string(msg) + ": " : "")
            << "shape mismatch: actual " << actual.rows() << "x" << actual.cols()
            << " vs expected " << expected.rows() << "x" << expected.cols();
    }
    const double diff = (actual - expected).cwiseAbs().maxCoeff();
    if (diff < tol) return ::testing::AssertionSuccess();
    std::ostringstream oss;
    if (msg && *msg) oss << msg << ": ";
    oss << "max element-wise diff " << diff << " > tol " << tol
        << "\nactual:\n"   << actual
        << "\nexpected:\n" << expected;
    return ::testing::AssertionFailure() << oss.str();
}

// ── Matrix builders for n-qubit gates from semantics ────────────────────────

/// Kronecker product (no Eigen unsupported-module dependency).
inline Mat kron(const Mat& A, const Mat& B) {
    const int ra = static_cast<int>(A.rows()), ca = static_cast<int>(A.cols());
    const int rb = static_cast<int>(B.rows()), cb = static_cast<int>(B.cols());
    Mat M(ra * rb, ca * cb);
    for (int i = 0; i < ra; ++i)
        for (int j = 0; j < ca; ++j)
            M.block(i * rb, j * cb, rb, cb) = A(i, j) * B;
    return M;
}

/// Permutation unitary on n qubits: column i of the matrix is basis state
/// `perm(i)`, optionally weighted by `phase(i)`.
inline Mat permutation_unitary(int n_qubits, std::function<int(int)> perm,
                               std::function<cd(int)> phase = nullptr) {
    const int dim = 1 << n_qubits;
    Mat M = Mat::Zero(dim, dim);
    for (int i = 0; i < dim; ++i)
        M(perm(i), i) = phase ? phase(i) : cd{1, 0};
    return M;
}

/// Diagonal unitary with the given diagonal entries.
inline Mat diag_unitary(const std::vector<cd>& d) {
    const int dim = static_cast<int>(d.size());
    Mat M = Mat::Zero(dim, dim);
    for (int i = 0; i < dim; ++i) M(i, i) = d[i];
    return M;
}

/// Embed a 2×2 single-qubit operator on qubit `q` of an n-qubit register
/// (identity on all other qubits).
inline Mat embed_1q(const Mat& m, int q, int n_qubits) {
    const int dim = 1 << n_qubits;
    Mat M = Mat::Zero(dim, dim);
    for (int j = 0; j < dim; ++j) {
        const int b  = bit(j, q);
        const int j0 = j & ~(1 << q);
        const int j1 = j |  (1 << q);
        M(j0, j) += m(0, b);
        M(j1, j) += m(1, b);
    }
    return M;
}

/// Controlled-1q gate: apply `m` to qubit `t` iff qubit `c` = |1⟩.
inline Mat controlled_1q(const Mat& m, int c, int t, int n_qubits) {
    const int dim = 1 << n_qubits;
    Mat M = Mat::Zero(dim, dim);
    for (int j = 0; j < dim; ++j) {
        if (!bit(j, c)) {
            M(j, j) = 1;
            continue;
        }
        const int b  = bit(j, t);
        const int j0 = j & ~(1 << t);
        const int j1 = j |  (1 << t);
        M(j0, j) += m(0, b);
        M(j1, j) += m(1, b);
    }
    return M;
}

// ── Analytic single-qubit matrices per qarp_conventions.md §2 ────────────────

inline Mat analytic_x() { Mat m(2,2); m << 0, 1, 1, 0; return m; }
inline Mat analytic_y() { Mat m(2,2); m << cd{0,0}, cd{0,-1}, cd{0,1}, cd{0,0}; return m; }
inline Mat analytic_z() { Mat m(2,2); m << 1, 0, 0, -1; return m; }
inline Mat analytic_h() {
    const double s = 1.0 / std::sqrt(2.0);
    Mat m(2,2); m << s, s, s, -s; return m;
}
inline Mat analytic_s()   { Mat m(2,2); m << 1, 0, 0, cd{0,  1}; return m; }
inline Mat analytic_sdg() { Mat m(2,2); m << 1, 0, 0, cd{0, -1}; return m; }
inline Mat analytic_t()   { Mat m(2,2); m << 1, 0, 0, ei( PI / 4.0); return m; }
inline Mat analytic_tdg() { Mat m(2,2); m << 1, 0, 0, ei(-PI / 4.0); return m; }

inline Mat analytic_rx(double th) {
    const double c = std::cos(th/2), s = std::sin(th/2);
    Mat m(2,2); m << cd{c,0}, cd{0,-s}, cd{0,-s}, cd{c,0}; return m;
}
inline Mat analytic_ry(double th) {
    const double c = std::cos(th/2), s = std::sin(th/2);
    Mat m(2,2); m << c, -s, s, c; return m;
}
inline Mat analytic_rz(double th) {
    Mat m(2,2); m << ei(-th/2), 0, 0, ei( th/2); return m;
}
inline Mat analytic_p(double th) {
    Mat m(2,2); m << 1, 0, 0, ei(th); return m;
}
inline Mat analytic_u(double th, double phi, double lam) {
    const double c = std::cos(th/2), s = std::sin(th/2);
    Mat m(2,2);
    m << cd{c, 0},          -ei(lam) * s,
         ei(phi) * s,        ei(phi + lam) * c;
    return m;
}

// §2.6: SX = e^{iπ/4}·Rx(π/2) (the principal √X), SXdg = e^{-iπ/4}·Rx(-π/2).
inline Mat analytic_sx()   { return ei( PI / 4.0) * analytic_rx( PI / 2.0); }
inline Mat analytic_sxdg() { return ei(-PI / 4.0) * analytic_rx(-PI / 2.0); }

}  // namespace qarpx::test
