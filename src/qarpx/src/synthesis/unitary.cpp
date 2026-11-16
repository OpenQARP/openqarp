#include "qarpx/synthesis/unitary.h"
#include "qarpx/synthesis/uniformly_controlled.h"

#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/SVD>

#ifdef QARP_USE_LAPACK
// LAPACK `zuncsd_` declaration.  On macOS we redirect to Apple Accelerate's
// `_zuncsd$NEWLAPACK` (the symbol scipy uses) — Homebrew's reference LAPACK
// 3.12 ships a buggy `zuncsd_` that produces wrong output for non-trivial
// inputs (verified independently via ctypes).  On Linux we trust the system
// LAPACK (OpenBLAS / reference) to provide a correct `zuncsd_`.
extern "C" {
    void zuncsd_(
        const char* jobu1, const char* jobu2,
        const char* jobv1t, const char* jobv2t,
        const char* trans, const char* signs,
        const int* m, const int* p, const int* q,
        std::complex<double>* x11, const int* ldx11,
        std::complex<double>* x12, const int* ldx12,
        std::complex<double>* x21, const int* ldx21,
        std::complex<double>* x22, const int* ldx22,
        double* theta,
        std::complex<double>* u1,  const int* ldu1,
        std::complex<double>* u2,  const int* ldu2,
        std::complex<double>* v1t, const int* ldv1t,
        std::complex<double>* v2t, const int* ldv2t,
        std::complex<double>* work, const int* lwork,
        double* rwork, const int* lrwork,
        int* iwork,
        int* info)
#ifdef __APPLE__
        __asm("_zuncsd$NEWLAPACK")
#endif
        ;
}
#endif  // QARP_USE_LAPACK

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

namespace qarpx::synthesis {

namespace {

using cd  = std::complex<double>;
using Mat = Eigen::MatrixXcd;
using Vec = Eigen::VectorXcd;

constexpr double kUnitaryTol     = 1e-9;
constexpr double kSingularEps    = 1e-10;
// Global phases below this are rounding of zero.  A skipped phase is lost,
// and under control it becomes a relative phase.
constexpr double kPhaseEps       = 1e-15;
// The diagonal fast path drops off-diagonal entries up to this size.
constexpr double kDiagonalTol    = 1e-12;
// Rounding splits a repeated eigenvalue of a unitary by ~N·ε; merging wider
// gaps rotates across distinct eigenvectors and costs accuracy of that size.
constexpr double kDegenerateTol  = 1e-12;

// ── Basis canonicalization for ComplexSchur in `demux_multiplexer` ─────────
//
// `demux_multiplexer` calls Eigen's `ComplexSchur` on the product `A·B^H` of
// the two unitaries it demultiplexes (`L1·L2^H` for the left multiplexer,
// `R1^H·R2` for the right).  When that product has repeated eigenvalues the
// basis is implementation-defined and the principal-branch `sqrt` below would
// otherwise pick mismatched values on different toolchains.  Sort by eigenvalue
// phase, then canonicalize V within each repeated-eigenvalue group via
// LQ + diagonal-phase fixup.
//
// (The corresponding SVD basis indeterminacy in the CSD step is resolved
// numerically but not canonically by the default pure-Eigen path; a
// QARP_USE_LAPACK=ON build delegates to LAPACK's `zuncsd`, which returns a
// deterministic (U1, U2, V1, V2, θ) — see `cosine_sine_decomposition` below.)

struct Group { Eigen::Index start; Eigen::Index size; };

template <typename Derived>
std::vector<Group> find_runs(const Eigen::DenseBase<Derived>& v, double tol) {
    std::vector<Group> out;
    const Eigen::Index N = v.size();
    Eigen::Index a = 0;
    while (a < N) {
        Eigen::Index b = a + 1;
        while (b < N && std::abs(v[b] - v[a]) < tol) ++b;
        out.push_back({a, b - a});
        a = b;
    }
    return out;
}

void canonicalize_unitary_schur(Mat& V, Mat& T) {
    const Eigen::Index N = V.rows();

    Eigen::VectorXd keys(N);
    for (Eigen::Index k = 0; k < N; ++k) keys(k) = std::arg(T(k, k));
    std::vector<Eigen::Index> perm(N);
    for (Eigen::Index k = 0; k < N; ++k) perm[k] = k;
    std::stable_sort(perm.begin(), perm.end(),
                     [&](Eigen::Index a, Eigen::Index b) { return keys(a) < keys(b); });

    Mat V_sorted(N, N);
    Mat T_sorted = Mat::Zero(N, N);
    for (Eigen::Index k = 0; k < N; ++k) {
        V_sorted.col(k) = V.col(perm[k]);
        T_sorted(k, k)  = T(perm[k], perm[k]);
    }
    V = std::move(V_sorted);
    T = std::move(T_sorted);

    Eigen::VectorXd sorted_keys(N);
    for (Eigen::Index k = 0; k < N; ++k) sorted_keys(k) = std::arg(T(k, k));
    const auto groups = find_runs(sorted_keys, kDegenerateTol);
    for (const auto& g : groups) {
        if (g.size <= 1) continue;
        const Mat Top = V.block(0, g.start, g.size, g.size);
        Eigen::HouseholderQR<Mat> qr(Top.adjoint());
        const Mat Q = qr.householderQ() * Mat::Identity(g.size, g.size);
        V.block(0, g.start, N, g.size) = V.block(0, g.start, N, g.size) * Q;
        for (Eigen::Index k = 0; k < g.size; ++k) {
            const cd d = V(g.start + k, g.start + k);
            if (std::abs(d) > kSingularEps) {
                const cd phase = d / std::abs(d);
                V.col(g.start + k) *= std::conj(phase);
            }
        }
    }
}

// ── Diagonal fast path ─────────────────────────────────────────────────────
//
// Performance shortcut: a diagonal U on n qubits compiles via Shende-Bullock-
// Markov to O(2^n) gates, while the generic QSD path emits O(4^n).  The
// LAPACK CSD path also handles diagonal U correctly; we just avoid the
// unnecessary 4× gate-count cost.

// Returns true if `U` is diagonal within `kDiagonalTol`.  When true, the
// diagonal entries are written into `out_diag`.
bool extract_if_diagonal(const Mat& U, Vec& out_diag) {
    const Eigen::Index N = U.rows();
    out_diag.resize(N);
    for (Eigen::Index i = 0; i < N; ++i) {
        for (Eigen::Index j = 0; j < N; ++j) {
            if (i == j) {
                out_diag(i) = U(i, j);
            } else if (std::abs(U(i, j)) > kDiagonalTol) {
                return false;
            }
        }
    }
    return true;
}

// SBM diagonal-unitary recursion adapted to operate on qubits `[base, base+n-1]`.
// At each level the highest qubit is peeled off via a uniformly-controlled `Rz`
// on it (controlled by all lower qubits); the recursion continues on the half-
// size "common-phase" diagonal `common[k] = (phases[k] + phases[N/2+k]) / 2`.
// Deepest level emits a `GPhase` to set the absolute global phase.
void apply_diagonal_phases(Block& block,
                           const std::vector<double>& phases,
                           uint32_t base,
                           uint32_t n) {
    if (n == 0) {
        if (std::abs(phases[0]) > kPhaseEps) block.gphase(Param(phases[0]));
        return;
    }

    const size_t half = static_cast<size_t>(1) << (n - 1);
    std::vector<double> common(half), delta(half);
    for (size_t k = 0; k < half; ++k) {
        common[k] = 0.5 * (phases[k] + phases[half + k]);
        delta[k]  = phases[half + k] - phases[k];
    }

    std::vector<uint32_t> controls;
    controls.reserve(n - 1);
    for (uint32_t k = 0; k < n - 1; ++k) controls.push_back(base + k);

    apply_uc_rz(block, base + n - 1, controls, delta);
    apply_diagonal_phases(block, common, base, n - 1);
}

// Convenience wrapper: synthesize a diagonal U on qubits `[base, base+n-1]`.
// Returns true iff U was handled by this fast path.
bool try_diagonal_fast_path(Block& block, const Mat& U, uint32_t base, uint32_t n) {
    Vec diag(0);
    if (!extract_if_diagonal(U, diag)) return false;
    const Eigen::Index N = diag.size();
    std::vector<double> phases(static_cast<size_t>(N));
    for (Eigen::Index k = 0; k < N; ++k) phases[static_cast<size_t>(k)] = std::arg(diag(k));
    apply_diagonal_phases(block, phases, base, n);
    return true;
}

// ── Single-qubit ZYZ decomposition ─────────────────────────────────────────
//
//   U = e^{iα} · Rz(β) · Ry(γ) · Rz(δ)
//
// with our convention Rz(θ) = diag(e^{-iθ/2}, e^{iθ/2}) and
// Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]].
//
// Closed form: if c := |u00| and s := |u10|, then γ = 2 atan2(s, c), and with
// a = arg u00, d = arg u11, b = arg u10, o = arg(−u01):
//   * c ≥ s:  α = (a + d) / 2,  β = b − a,  δ = d − b
//   * c < s:  α = (b + o) / 2,  β − δ = b − o,  β + δ = 2(α − a)
// The argument of an entry of size ε is known only to ~ε_mach/ε, so α always
// comes from the larger pair; the smaller pair's phases only scale entries of
// its own size.
//
// α is the absolute global phase, not a free parameter: a caller wrapping this
// block in a control reads it as a relative phase, so it must be reproduced
// exactly rather than up to e^{iα}.
void apply_zyz(Block& block, uint32_t target, const Mat& U) {
    const cd u00 = U(0, 0), u01 = U(0, 1), u10 = U(1, 0), u11 = U(1, 1);
    const double c = std::abs(u00), s = std::abs(u10);
    const double gamma = 2.0 * std::atan2(s, c);

    double alpha, beta, delta;
    if (c >= s) {
        alpha = 0.5 * (std::arg(u00) + std::arg(u11));
        beta  = std::arg(u10) - std::arg(u00);
        delta = std::arg(u11) - std::arg(u10);
    } else {
        const double o = std::arg(-u01);
        alpha = 0.5 * (std::arg(u10) + o);
        const double sum  = 2.0 * (alpha - std::arg(u00));  // β + δ
        const double diff = std::arg(u10) - o;              // β − δ
        beta  = 0.5 * (sum + diff);
        delta = 0.5 * (sum - diff);
    }

    block.rz(target, Param(delta));
    block.ry(target, Param(gamma));
    block.rz(target, Param(beta));
    if (std::abs(alpha) > kPhaseEps) block.gphase(Param(alpha));
}

// ── Multiplexer demultiplexer ──────────────────────────────────────────────
//
// Given (n-1)-qubit unitaries A and B, factor the block-diagonal
//
//   M = block_diag(A, B)
//
// (acting on n qubits with the highest qubit selecting the block) into
//
//   M = (V ⊕ V) · UC-Rz on q_high (controlled by lower qubits) · (W ⊕ W)
//
// using the eigendecomposition of `X = A · B^H = V · D² · V^H`.  Then
// `A = V · D · W` and `B = V · D^H · W` with `W = D · V^H · B`.  The UC-Rz
// angles are `θ_k = -2 · arg(D[k])` so that `Rz(θ_k) = diag(D[k], D[k]^*)`.
struct DemuxResult {
    Mat V;
    Mat W;
    std::vector<double> uc_rz_angles;  // size 2^(n-1)
};

DemuxResult demux_multiplexer(const Mat& A, const Mat& B) {
    const Eigen::Index N = A.rows();
    const Mat X = A * B.adjoint();

    // Schur decomposition: X = V · T · V^H with V unitary and T upper-triangular.
    // For unitary X, T is diagonal — the eigenvalues are on the diagonal of T.
    Eigen::ComplexSchur<Mat> schur(X);
    if (schur.info() != Eigen::Success) {
        throw std::runtime_error(
            "demux_multiplexer: Schur decomposition did not converge");
    }
    Mat V = schur.matrixU();
    Mat T = schur.matrixT();
    // Canonicalize within repeated-eigenvalue subspaces.  ComplexSchur's basis
    // is implementation-defined when eigenvalues coincide; the principal-
    // branch sqrt below would otherwise pick mismatched values across
    // toolchains.
    canonicalize_unitary_schur(V, T);

    // Principal square roots of the diagonal eigenvalues.
    Vec D_diag(N);
    for (Eigen::Index k = 0; k < N; ++k) {
        D_diag(k) = std::sqrt(T(k, k));  // principal branch — arg in (-π/2, π/2]
    }

    // W = diag(D) · V^H · B
    Mat W = V.adjoint() * B;
    for (Eigen::Index k = 0; k < N; ++k) {
        W.row(k) *= D_diag(k);
    }

    std::vector<double> angles(N);
    for (Eigen::Index k = 0; k < N; ++k) {
        angles[k] = -2.0 * std::arg(D_diag(k));
    }

    return DemuxResult{V, W, std::move(angles)};
}

// ── Cosine-sine decomposition ──────────────────────────────────────────────
//
// Given a `2N × 2N` unitary
//
//   U = [[U00, U01],
//        [U10, U11]]
//
// produce
//
//   U = (L1 ⊕ L2) · [[ C, -S ], [ S, C ]] · (R1 ⊕ R2)^H
//
// where `C = diag(cos θ_k)`, `S = diag(sin θ_k)`, and L1, L2, R1, R2 are
// `N × N` unitaries.  The middle matrix is exactly the multi-qubit form of a
// uniformly-controlled `Ry(2θ_k)` on the high qubit.
//
// Two implementations, selected at build time:
//   * QARP_USE_LAPACK=OFF (default) — pure Eigen (SVD with a per-column choice
//     of the well-conditioned sine or cosine channel; see the `#else` branch).
//     No BLAS/LAPACK link, so wheels bundle no OpenBLAS/libgfortran/
//     libquadmath.  Its degenerate-σ basis is resolved by an SVD rather than a
//     canonical rule, so decompositions are equivalent-but-not-bit-identical
//     across toolchains.
//   * QARP_USE_LAPACK=ON — delegates to LAPACK's `zuncsd`, which returns a
//     deterministic decomposition on a degenerate σ spectrum (no SVD basis
//     indeterminacy).  `zuncsd` convention: `V1T`/`V2T` outputs
//     hold `V1^H`/`V2^H`, so we adjoint them to recover R1 = V1, R2 = V2;
//     sign convention `signs = 'D'` matches our `[[C,-S],[S,C]]` block layout.
struct CSDResult {
    Mat L1, L2, R1, R2;
    std::vector<double> theta;
};

#ifdef QARP_USE_LAPACK

CSDResult cosine_sine_decomposition(const Mat& U) {
    const Eigen::Index twoN = U.rows();
    if (U.cols() != twoN || (twoN & 1)) {
        throw std::runtime_error("cosine_sine_decomposition: square 2N × 2N matrix required");
    }
    const int N = static_cast<int>(twoN / 2);

    // zuncsd modifies the X11..X22 blocks in place; pass copies.
    Mat X11 = U.topLeftCorner(N, N);
    Mat X12 = U.topRightCorner(N, N);
    Mat X21 = U.bottomLeftCorner(N, N);
    Mat X22 = U.bottomRightCorner(N, N);

    Mat U1(N, N), U2(N, N), V1T(N, N), V2T(N, N);
    std::vector<double> theta(static_cast<size_t>(N));

    const char y = 'Y', n = 'N', d = 'D';
    const int m = 2 * N;

    // Workspace query: lwork = lrwork = -1 returns the optimal sizes.
    std::complex<double> work_query;
    double rwork_query = 0.0;
    int lwork_query = -1, lrwork_query = -1;
    int info = 0;
    std::vector<int> iwork(static_cast<size_t>(m));
    zuncsd_(&y, &y, &y, &y, &n, &d,
            &m, &N, &N,
            X11.data(), &N, X12.data(), &N, X21.data(), &N, X22.data(), &N,
            theta.data(),
            U1.data(), &N, U2.data(), &N, V1T.data(), &N, V2T.data(), &N,
            &work_query, &lwork_query, &rwork_query, &lrwork_query,
            iwork.data(), &info);
    if (info != 0) {
        throw std::runtime_error(
            "cosine_sine_decomposition: zuncsd workspace query failed, info=" +
            std::to_string(info));
    }
    const int lwork  = static_cast<int>(std::real(work_query));
    const int lrwork = static_cast<int>(rwork_query);
    std::vector<std::complex<double>> work(static_cast<size_t>(lwork));
    std::vector<double> rwork(static_cast<size_t>(lrwork));

    // Re-copy X (query call clobbers it).
    X11 = U.topLeftCorner(N, N);
    X12 = U.topRightCorner(N, N);
    X21 = U.bottomLeftCorner(N, N);
    X22 = U.bottomRightCorner(N, N);

    zuncsd_(&y, &y, &y, &y, &n, &d,
            &m, &N, &N,
            X11.data(), &N, X12.data(), &N, X21.data(), &N, X22.data(), &N,
            theta.data(),
            U1.data(), &N, U2.data(), &N, V1T.data(), &N, V2T.data(), &N,
            work.data(), &lwork, rwork.data(), &lrwork,
            iwork.data(), &info);
    if (info != 0) {
        throw std::runtime_error(
            "cosine_sine_decomposition: zuncsd returned info=" +
            std::to_string(info));
    }

    return CSDResult{std::move(U1), std::move(U2),
                     V1T.adjoint(), V2T.adjoint(),
                     std::move(theta)};
}

#else  // !QARP_USE_LAPACK — pure-Eigen fallback

// Orthonormal basis (N × (N−r)) for the orthogonal complement of the column
// space of `set_cols` (N × r, assumed to have orthonormal columns).  Used to
// fill the L2 / R2 columns that the sine channel leaves undefined (the θ_k ≈ 0
// subspace, where U10 and U01 vanish).
Mat orthonormal_complement(const Mat& set_cols, Eigen::Index N) {
    const Eigen::Index r = set_cols.cols();
    if (r == 0) return Mat::Identity(N, N);
    Eigen::HouseholderQR<Mat> qr(set_cols);
    const Mat Q = qr.householderQ() * Mat::Identity(N, N);
    return Q.rightCols(N - r);
}

// Pure-Eigen cosine-sine decomposition — the default (`QARP_USE_LAPACK=OFF`)
// implementation; keeps the build free of any BLAS/LAPACK link.  Every factor
// column is taken from whichever of the sine or cosine channel is well
// conditioned (Van Loan, 1985), so no step divides by less than 1/√2:
//
//   1. SVD  U00 = L1 · C · R1^H  (cosines descending).  Columns with
//      c_k ≤ 1/√2 are sine-dominant (set J), the rest cosine-dominant (set K).
//   2. J:  L2·col(k) = (U10 · R1)·col(k) / s_k,  s_k = ‖(U10 · R1)·col(k)‖.
//   3. K:  the sines are small, so dividing by them would amplify rounding.
//      Instead take the SVD of the sine block in the orthogonal complement P of
//      L2|J:  P^H · U10 · R1|K = Ur · S_K · Vr^H, set L2|K = P · Ur,
//      R1|K ← R1|K · Vr, and recompute L1|K = U00 · R1|K / c_k from the
//      cosine channel (c_k = column norm ≥ 1/√2).
//   4. R2 from the better channel per column:
//        K:  R2·col(k) =  (U11^H · L2)·col(k) / c_k,
//        J:  R2·col(k) = −(U01^H · L1)·col(k) / s_k.
//
// Not bit-reproducible across toolchains where C or S is degenerate (the SVD
// basis is free there) — build with QARP_USE_LAPACK=ON where canonical
// decompositions matter.
CSDResult cosine_sine_decomposition(const Mat& U) {
    const Eigen::Index twoN = U.rows();
    if (U.cols() != twoN || (twoN & 1)) {
        throw std::runtime_error("cosine_sine_decomposition: square 2N × 2N matrix required");
    }
    const Eigen::Index N = twoN / 2;

    const Mat U00 = U.topLeftCorner(N, N);
    const Mat U01 = U.topRightCorner(N, N);
    const Mat U10 = U.bottomLeftCorner(N, N);
    const Mat U11 = U.bottomRightCorner(N, N);

    Eigen::JacobiSVD<Mat> svd(U00, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat L1 = svd.matrixU();
    Mat R1 = svd.matrixV();
    Eigen::VectorXd c = svd.singularValues();

    // Singular values come descending, so the cosine-dominant set K leads.
    const double split = std::sqrt(0.5);
    Eigen::Index nk = 0;
    while (nk < N && c(nk) > split) ++nk;
    const Eigen::Index nj = N - nk;

    const Mat W = U10 * R1;
    Mat L2 = Mat::Zero(N, N);
    Eigen::VectorXd s(N);
    for (Eigen::Index k = nk; k < N; ++k) {
        s(k) = W.col(k).norm();
        L2.col(k) = W.col(k) / s(k);
    }

    if (nk > 0) {
        const Mat P = orthonormal_complement(L2.rightCols(nj), N);  // N × nk
        Eigen::JacobiSVD<Mat> svdK(P.adjoint() * W.leftCols(nk),
                                   Eigen::ComputeFullU | Eigen::ComputeFullV);
        L2.leftCols(nk) = P * svdK.matrixU();
        R1.leftCols(nk) = (R1.leftCols(nk) * svdK.matrixV()).eval();
        s.head(nk) = svdK.singularValues();
        const Mat Y = U00 * R1.leftCols(nk);
        for (Eigen::Index k = 0; k < nk; ++k) {
            c(k) = Y.col(k).norm();
            L1.col(k) = Y.col(k) / c(k);
        }
    }

    const Mat Hc = U11.adjoint() * L2;      // col k = c_k · R2·col(k)
    const Mat Hs = -(U01.adjoint() * L1);   // col k = s_k · R2·col(k)
    Mat R2(N, N);
    std::vector<double> theta(static_cast<size_t>(N));
    for (Eigen::Index k = 0; k < N; ++k) {
        R2.col(k) = k < nk ? Eigen::VectorXcd(Hc.col(k) / c(k))
                           : Eigen::VectorXcd(Hs.col(k) / s(k));
        theta[static_cast<size_t>(k)] = std::atan2(s(k), c(k));
    }

    return CSDResult{std::move(L1), std::move(L2),
                     std::move(R1), std::move(R2),
                     std::move(theta)};
}

#endif  // QARP_USE_LAPACK

// ── Recursive QSD ──────────────────────────────────────────────────────────

void qsd_recursive(Block& block, const Mat& U, uint32_t base, uint32_t n) {
    if (n == 1) {
        apply_zyz(block, base, U);
        return;
    }

    // Diagonal sub-problems route directly through the SBM uniformly-
    // controlled-Rz cascade — not for correctness (the CSD + canonicalization
    // path also handles them correctly) but for circuit depth: SBM emits
    // O(2^n) gates for a 2^n-diagonal, the generic QSD path emits O(4^n).
    if (try_diagonal_fast_path(block, U, base, n)) return;

    const auto csd = cosine_sine_decomposition(U);

    // Lower qubits = [base, base + n - 2]; high qubit = base + n - 1.
    const uint32_t high = base + n - 1;
    std::vector<uint32_t> lower_controls;
    lower_controls.reserve(n - 1);
    for (uint32_t k = 0; k < n - 1; ++k) lower_controls.push_back(base + k);

    // Right-to-left order: emit gates as they appear in circuit time.
    //
    // U = (L1 ⊕ L2)   [demux into V_L · UC-Rz · W_L]
    //     · UC-Ry(2θ) [middle uniformly-controlled rotation on high]
    //     · (R1 ⊕ R2)^H = (R1^H ⊕ R2^H)
    //                  [demux into V_R · UC-Rz · W_R]
    //
    // Circuit time (apply right-most first):
    //   (R1^H ⊕ R2^H) → UC-Ry(2θ) → (L1 ⊕ L2)

    // ── Right multiplexer: (R1^H ⊕ R2^H) — equivalent to multiplexer of (R1^H, R2^H)
    auto demux_R = demux_multiplexer(csd.R1.adjoint(), csd.R2.adjoint());
    // Order: W_R first, then UC-Rz, then V_R.
    qsd_recursive(block, demux_R.W, base, n - 1);
    apply_uc_rz(block, high, lower_controls, demux_R.uc_rz_angles);
    qsd_recursive(block, demux_R.V, base, n - 1);

    // ── Middle: UC-Ry on high qubit with angles 2 · θ_k.
    std::vector<double> ry_angles(csd.theta.size());
    for (size_t k = 0; k < csd.theta.size(); ++k) ry_angles[k] = 2.0 * csd.theta[k];
    apply_uc_ry(block, high, lower_controls, ry_angles);

    // ── Left multiplexer: (L1 ⊕ L2)
    auto demux_L = demux_multiplexer(csd.L1, csd.L2);
    qsd_recursive(block, demux_L.W, base, n - 1);
    apply_uc_rz(block, high, lower_controls, demux_L.uc_rz_angles);
    qsd_recursive(block, demux_L.V, base, n - 1);
}

}  // anonymous namespace

void unitary_synthesis(Block& block, const Eigen::MatrixXcd& U) {
    const Eigen::Index N = U.rows();
    if (U.cols() != N) {
        throw std::runtime_error("unitary_synthesis: matrix must be square");
    }
    if (N <= 0 || (N & (N - 1)) != 0) {
        throw std::runtime_error("unitary_synthesis: dimension must be a power of 2");
    }
    const uint32_t n = static_cast<uint32_t>(std::countr_zero(static_cast<uint64_t>(N)));

    // Verify unitarity within tolerance.
    const Mat I = Mat::Identity(N, N);
    const Mat err = U.adjoint() * U - I;
    if (err.cwiseAbs().maxCoeff() > kUnitaryTol) {
        throw std::runtime_error(
            "unitary_synthesis: input matrix is not unitary within tolerance");
    }

    qsd_recursive(block, Mat(U), /*base=*/0, n);
}

}  // namespace qarpx::synthesis
