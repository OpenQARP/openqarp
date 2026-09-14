#pragma once

#include "../block/block.h"

#include <complex>
#include <vector>

namespace qarpx::synthesis {

/// Append gates to `block` that prepare the state `|ψ⟩ ∝ Σ amplitudes[i] |i⟩` from
/// `|0…0⟩`.  Indices are LSB-first: `|i⟩` has qubit `q_k` carrying bit `k` of `i`.
///
/// `amplitudes` must have size `2^n` where `n` is the number of qubits used.  The
/// vector is normalized internally.  An all-zero amplitude vector is rejected with
/// `std::runtime_error`.
///
/// Algorithm: recursive Möttönen-style decomposition (arXiv:quant-ph/0407010).  At
/// each level the lowest qubit is prepared by a uniformly-controlled `Ry`
/// (magnitudes) followed by a uniformly-controlled `Rz` (relative phases), with the
/// remaining qubits prepared recursively from a "merged" amplitude vector that
/// pairs `(a_{2j}, a_{2j+1}) → √(|a_{2j}|² + |a_{2j+1}|²) · e^{i(arg a_{2j} + arg a_{2j+1})/2}`.
/// The deepest level emits a `GPhase` to set the absolute global phase.
///
/// Complexity: `O(2^n)` rotations and `O(2^n)` `CX` gates total.
void state_preparation(
    Block& block,
    const std::vector<std::complex<double>>& amplitudes);

}  // namespace qarpx::synthesis
