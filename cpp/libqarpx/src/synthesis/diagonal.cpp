#include "qarpx/synthesis/diagonal.h"
#include "qarpx/synthesis/uniformly_controlled.h"

#include <bit>
#include <cmath>
#include <stdexcept>

namespace qarpx::synthesis {

namespace {

constexpr double kModulusTol = 1e-9;
// Global phases below this are rounding of zero.  A skipped phase is lost,
// and under control it becomes a relative phase.
constexpr double kPhaseEps = 1e-15;

/// Recursive diagonal-unitary builder.  `phases` carries the per-index phase
/// profile for the current level; the recursion strips the highest qubit at
/// each step.
void diagonal_recursive(
    Block& block,
    const std::vector<double>& phases,
    uint32_t base,
    uint32_t n)
{
    if (n == 0) {
        if (std::abs(phases[0]) > kPhaseEps) {
            block.gphase(Param(phases[0]));
        }
        return;
    }

    const size_t half = static_cast<size_t>(1) << (n - 1);
    std::vector<double> common(half);
    std::vector<double> delta(half);
    for (size_t k = 0; k < half; ++k) {
        const double alpha = phases[k];          // q_{n-1} = 0
        const double beta  = phases[half + k];   // q_{n-1} = 1
        common[k] = 0.5 * (alpha + beta);
        delta[k]  = beta - alpha;
    }

    // UC-Rz on qubit (base + n - 1) controlled by all lower qubits.
    std::vector<uint32_t> controls;
    controls.reserve(n - 1);
    for (uint32_t k = 0; k < n - 1; ++k) controls.push_back(base + k);

    apply_uc_rz(block, base + n - 1, controls, delta);

    // Recurse on the common-phase profile over qubits [base, base + n - 2].
    diagonal_recursive(block, common, base, n - 1);
}

}  // anonymous namespace

void diagonal_unitary(
    Block& block,
    const std::vector<std::complex<double>>& diagonal_elements)
{
    const size_t N = diagonal_elements.size();
    if (N == 0 || (N & (N - 1)) != 0) {
        throw std::runtime_error(
            "diagonal_unitary: diagonal size must be a power of 2");
    }
    const uint32_t n = static_cast<uint32_t>(std::countr_zero(N));

    // Validate unit-modulus and extract phases.
    std::vector<double> phases(N);
    for (size_t i = 0; i < N; ++i) {
        const double mod = std::abs(diagonal_elements[i]);
        if (std::abs(mod - 1.0) > kModulusTol) {
            throw std::runtime_error(
                "diagonal_unitary: every diagonal entry must be unit-modulus");
        }
        phases[i] = std::arg(diagonal_elements[i]);
    }

    diagonal_recursive(block, phases, /*base=*/0, n);
}

}  // namespace qarpx::synthesis
