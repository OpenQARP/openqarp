#include "qarpx/simulator/dense_kernel.h"

#include <csim/utility.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace qarpx {

namespace {

// Same gate as csim's kernels: OMPutil forks a team only from
// QULACS_PARALLEL_NQUBIT_THRESHOLD (exported by init_threading), which
// overrides this csim-matching fallback.
constexpr unsigned kParallelThreshold = 13;

template <int K>
void apply_dense_k(const uint32_t*        qubits,
                   const double*          mr,     // row-major [y * D + x]
                   const double*          mi,
                   std::complex<double>*  state,
                   uint64_t               dim) {
    constexpr int D = 1 << K;

    uint32_t sorted[K];
    std::copy(qubits, qubits + K, sorted);
    std::sort(sorted, sorted + K);

    uint64_t masks[D];
    for (int l = 0; l < D; ++l) {
        uint64_t m = 0;
        for (int b = 0; b < K; ++b)
            if ((l >> b) & 1) m |= uint64_t{1} << qubits[b];
        masks[l] = m;
    }

    const int64_t outer = static_cast<int64_t>(dim >> K);

#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(static_cast<ITYPE>(dim), kParallelThreshold);
#pragma omp parallel for schedule(static)
#endif
    for (int64_t o = 0; o < outer; ++o) {
        // Insert a zero bit at every target position, ascending, so `base`
        // is the outer index with the block's qubits cleared.
        uint64_t base = static_cast<uint64_t>(o);
        for (int b = 0; b < K; ++b) {
            const uint64_t p   = sorted[b];
            const uint64_t low = base & ((uint64_t{1} << p) - 1);
            base = ((base >> p) << (p + 1)) | low;
        }

        // One dot product per output amplitude — no accumulator array, and
        // the x loop vectorises without fast-math.
        double inr[D], ini[D];
        for (int l = 0; l < D; ++l) {
            const auto a = state[base | masks[l]];
            inr[l] = a.real();
            ini[l] = a.imag();
        }
        for (int y = 0; y < D; ++y) {
            const double* rr = mr + y * D;
            const double* ii = mi + y * D;
            double sr = 0.0, si = 0.0;
            for (int x = 0; x < D; ++x) {
                sr += rr[x] * inr[x] - ii[x] * ini[x];
                si += rr[x] * ini[x] + ii[x] * inr[x];
            }
            state[base | masks[y]] = {sr, si};
        }
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
}

}  // namespace

void apply_dense_block(const uint32_t*          qubits,
                       std::size_t              k,
                       const Eigen::MatrixXcd&  U,
                       std::complex<double>*    state,
                       uint64_t                 dim) {
    if (k == 0 || k > kMaxDenseBlockQubits)
        throw std::invalid_argument(
            "apply_dense_block: block width " + std::to_string(k)
            + " outside 1.." + std::to_string(kMaxDenseBlockQubits));
    const Eigen::Index d = Eigen::Index{1} << k;
    if (U.rows() != d || U.cols() != d)
        throw std::invalid_argument(
            "apply_dense_block: unitary is " + std::to_string(U.rows()) + "x"
            + std::to_string(U.cols()) + " for " + std::to_string(k) + " qubit(s)");
    if (dim < static_cast<uint64_t>(d))
        throw std::invalid_argument("apply_dense_block: block wider than the register");

    // Split into real / imaginary row-major copies once per call.
    std::vector<double> mr(static_cast<std::size_t>(d * d)), mi(static_cast<std::size_t>(d * d));
    for (Eigen::Index y = 0; y < d; ++y)
        for (Eigen::Index x = 0; x < d; ++x) {
            mr[static_cast<std::size_t>(y * d + x)] = U(y, x).real();
            mi[static_cast<std::size_t>(y * d + x)] = U(y, x).imag();
        }

    switch (k) {
        case 1: apply_dense_k<1>(qubits, mr.data(), mi.data(), state, dim); break;
        case 2: apply_dense_k<2>(qubits, mr.data(), mi.data(), state, dim); break;
        case 3: apply_dense_k<3>(qubits, mr.data(), mi.data(), state, dim); break;
        case 4: apply_dense_k<4>(qubits, mr.data(), mi.data(), state, dim); break;
        case 5: apply_dense_k<5>(qubits, mr.data(), mi.data(), state, dim); break;
        case 6: apply_dense_k<6>(qubits, mr.data(), mi.data(), state, dim); break;
        case 7: apply_dense_k<7>(qubits, mr.data(), mi.data(), state, dim); break;
        case 8: apply_dense_k<8>(qubits, mr.data(), mi.data(), state, dim); break;
        default: break;  // unreachable: range checked above
    }
}

}  // namespace qarpx
