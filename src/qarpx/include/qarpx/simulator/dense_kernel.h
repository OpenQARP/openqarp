#pragma once

#include <Eigen/Dense>

#include <complex>
#include <cstddef>
#include <cstdint>

namespace qarpx {

/// Widest block `apply_dense_block` accepts (2^k local amplitudes live on
/// the stack per outer index).
inline constexpr std::size_t kMaxDenseBlockQubits = 8;

/// Apply the 2^k × 2^k dense unitary `U` — local bit b ↔ `qubits[b]`, LSB
/// (§1) — to `state` in place.  qarpx's own kernel for the fused blocks of
/// `fuse_for_simulation`: csim's generic dense kernels cost 3–20 named
/// gates per pass (2026-09-13, 20 qubits, arm64), so fusing into them could
/// never win.  One gather / matvec / scatter per outer index, real and
/// imaginary parts split so the inner loop vectorises without fast-math;
/// parallel over the outer index above csim's per-gate threshold (same
/// `OMPutil` gate as the csim kernels, so `QULACS_PARALLEL_NQUBIT_THRESHOLD`
/// governs it too).  `k` must be 1..kMaxDenseBlockQubits and `U` must be
/// 2^k square; `qubits` must be distinct and below log2(dim).
void apply_dense_block(const uint32_t*          qubits,
                       std::size_t              k,
                       const Eigen::MatrixXcd&  U,
                       std::complex<double>*    state,
                       uint64_t                 dim);

}  // namespace qarpx
