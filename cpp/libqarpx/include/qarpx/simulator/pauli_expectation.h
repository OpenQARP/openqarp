#pragma once

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qarpx {

/// The simulator observable ABI shared with `run_gradient` and
/// `CudaqSimulator::batch_expectation`: `[(sparse [(qubit, 'X'|'Y'|'Z'|'I')],
/// complex coeff)]`; the identity term has an empty Pauli string.
using PauliObservable =
    std::vector<std::pair<std::vector<std::pair<uint32_t, char>>, std::complex<double>>>;

/// The width `pauli_transition` accepts.  Callers that compute `1 << n_qubits`
/// for their own length check must call this first: the shift is undefined
/// outside this range, and on arm64 `n = 64` wraps to 1 and passes a
/// one-element vector through.
inline void check_transition_width(int n_qubits) {
    if (n_qubits < 0 || n_qubits > 62)
        throw std::invalid_argument(
            "pauli_transition: n_qubits must be in [0, 62], got " + std::to_string(n_qubits));
}

/// ⟨bra|H|ket⟩ over host statevectors of length 2^n_qubits, LSB-indexed
/// (bit q of the amplitude index is qubit q, §1).  H need not be Hermitian
/// and bra need not equal ket — the full complex amplitude is returned.
///
/// One pass over the two vectors per call, not one per term: terms are
/// grouped by their X/Y flip mask and each group's Z-parity sum is evaluated
/// in aligned 64-index blocks against a ±1 table, so the cost is dominated
/// by `n_terms · 2^n / 64` popcounts plus SIMD fused multiply-adds rather than
/// `n_terms` memory sweeps.  One OpenMP region per call, capped by both
/// `OMP_NUM_THREADS` and `QARP_NUM_THREADS`; the floating-point reduction
/// order follows the team size, so results agree across thread counts to
/// rounding, not bit for bit.
///
/// Throws `std::invalid_argument` for a qubit index ≥ n_qubits, a Pauli
/// letter outside "XYZI", or n_qubits outside [0, 62].
[[nodiscard]] std::complex<double> pauli_transition(
    const std::complex<double>* bra,
    const std::complex<double>* ket,
    int                         n_qubits,
    const PauliObservable&      observable);

}  // namespace qarpx
