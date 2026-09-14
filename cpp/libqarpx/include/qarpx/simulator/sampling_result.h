#pragma once

#include <complex>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace qarpx {

/// Outcome counts from a single circuit execution.
///
/// Keys are bitstring outcomes encoded as integers (bit 0 = qubit 0).
/// Values are the number of shots that produced that outcome.
struct SamplingResult {
    int n_qubits = 0;
    int n_shots  = 0;
    std::unordered_map<uint64_t, int> counts;

    /// Number of classical bits in the per-shot register.  Zero when no
    /// command records a cbit.
    int n_cbits = 0;

    /// Per-shot end-of-shot classical register, shape [n_shots][n_cbits].
    /// Empty when n_cbits == 0.  Populated whenever measures are recorded —
    /// by trajectory runs and by QarpSimulator's terminal fast path.  Row
    /// order is not guaranteed to be shot-chronological (a simulator may
    /// group shots by trajectory).
    std::vector<std::vector<bool>> cbit_history;

    /// Probability of outcome k: counts[k] / n_shots.
    [[nodiscard]] double probability(uint64_t outcome) const {
        auto it = counts.find(outcome);
        return (it != counts.end()) ? static_cast<double>(it->second) / n_shots : 0.0;
    }

    /// Full statevector (only populated when n_shots == 0, i.e. statevector mode).
    std::vector<std::complex<double>> statevector;
};

}  // namespace qarpx
