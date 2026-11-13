#pragma once

#include "../block/block.h"

#include <complex>
#include <vector>

namespace qarpx::synthesis {

/// Append gates to `block` that implement the diagonal unitary
///
///     U = diag(d_0, d_1, …, d_{2^n - 1})
///
/// on qubits `[0, n-1]` (LSB-first: index `i` carries bit `k` of `i` on qubit `q_k`).
///
/// `diagonal_elements` must have size `2^n` and each entry must be unit-modulus
/// (within `1e-9`).  Only the phase of each entry is used.
///
/// Algorithm: Shende-Bullock-Markov 2006 recursive decomposition.  At each level
/// the highest qubit is peeled off via a uniformly-controlled `Rz` on it
/// (controlled by all lower qubits); the recursion continues on the half-size
/// "common-phase" diagonal `V_common[k] = (arg d_k + arg d_{N/2+k}) / 2`.  The
/// deepest level emits a `GPhase` to set the absolute global phase.
///
/// Complexity: `2^n - 1` `CX` gates and `2^n - 1` `Rz` gates plus one `GPhase`.
void diagonal_unitary(
    Block& block,
    const std::vector<std::complex<double>>& diagonal_elements);

}  // namespace qarpx::synthesis
