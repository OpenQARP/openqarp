#pragma once

#include "../block/block.h"

#include <Eigen/Dense>

namespace qarpx::synthesis {

/// Append gates to `block` that implement an arbitrary `2^n × 2^n` unitary `U` on
/// qubits `[0, n-1]` (LSB-first).
///
/// Algorithm: Quantum Shannon Decomposition (Shende-Bullock-Markov 2006).
///
///   * `n == 1`: ZYZ decomposition `U = e^{iα} · Rz(β) · Ry(γ) · Rz(δ)`.
///   * `n ≥ 2`: cosine-sine decomposition (via LAPACK `zuncsd`) splits `U`
///     into two block-diagonals sandwiching a uniformly-controlled `Ry` on
///     the highest qubit; each block-diagonal is then "demultiplexed" into
///     two `(n-1)`-qubit unitaries and a uniformly-controlled `Rz`,
///     recursing on the smaller unitaries.
///
/// `U` must be square `2^n × 2^n` and unitary within `1e-9`; both are validated.
///
/// Complexity: `O(4^n)` gates (no Möttönen sub-block optimisation in this
/// implementation — straightforward recursion).
///
/// No build requirement by default: the CSD step runs on a pure-Eigen
/// implementation (handles structurally near-degenerate σ, Heisenberg-like
/// Hamiltonians, via per-cluster refinement).  An optional
/// `QARP_USE_LAPACK=ON` build routes the CSD through LAPACK's `zuncsd`
/// instead, whose output is additionally deterministic across host
/// toolchains (canonical degenerate-σ basis).
void unitary_synthesis(Block& block, const Eigen::MatrixXcd& U);

}  // namespace qarpx::synthesis
