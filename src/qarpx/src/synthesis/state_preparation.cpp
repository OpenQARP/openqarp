#include "qarpx/synthesis/state_preparation.h"
#include "qarpx/synthesis/uniformly_controlled.h"

#include <bit>
#include <cmath>
#include <stdexcept>

namespace qarpx::synthesis {

namespace {

constexpr double kZeroEps = 1e-15;

/// Recursive forward state preparation.
///
/// Prepares the state `Σ amplitudes[i] |i⟩` on qubits `[base, base+n-1]` (LSB-first:
/// the lowest-numbered qubit is `base`).  Assumes the amplitudes are already
/// normalized.
void prepare_recursive(
    Block& block,
    const std::vector<std::complex<double>>& amplitudes,
    uint32_t base,
    uint32_t n)
{
    // Base case: a single complex value.  Set the absolute global phase.
    if (n == 0) {
        const double phi = std::arg(amplitudes[0]);
        if (std::abs(phi) > kZeroEps) {
            block.gphase(Param(phi));
        }
        return;
    }

    const size_t half = static_cast<size_t>(1) << (n - 1);

    // Compute UC-Ry / UC-Rz angles for the lowest qubit (`base`) and the merged
    // (n-1)-qubit amplitude vector for the recursive call.  The merged amplitude
    // for pair (α, β) is √(|α|² + |β|²) · e^{i(arg α + arg β)/2}; the singular
    // cases where one of the pair is zero are handled to keep `arg` well-defined.
    std::vector<double> ty(half), tz(half);
    std::vector<std::complex<double>> parent(half);

    for (size_t j = 0; j < half; ++j) {
        const auto alpha = amplitudes[2 * j];
        const auto beta  = amplitudes[2 * j + 1];
        const double r_a = std::abs(alpha);
        const double r_b = std::abs(beta);

        if (r_a < kZeroEps && r_b < kZeroEps) {
            ty[j] = 0.0;
            tz[j] = 0.0;
            parent[j] = std::complex<double>(0.0, 0.0);
        } else if (r_a < kZeroEps) {
            // α = 0: Ry(π) maps (0, β) → (β, 0), no phase rotation needed (tz=0).
            ty[j] = M_PI;
            tz[j] = 0.0;
            parent[j] = beta;
        } else if (r_b < kZeroEps) {
            // β = 0: nothing to rotate; pass α through as the parent amplitude.
            ty[j] = 0.0;
            tz[j] = 0.0;
            parent[j] = alpha;
        } else {
            const double phi_a = std::arg(alpha);
            const double phi_b = std::arg(beta);
            ty[j] = 2.0 * std::atan2(r_b, r_a);
            tz[j] = phi_b - phi_a;
            const double r_new = std::sqrt(r_a * r_a + r_b * r_b);
            const double phi_new = 0.5 * (phi_a + phi_b);
            parent[j] = std::polar(r_new, phi_new);
        }
    }

    // Build the upper-register state first (qubits [base+1, base+n-1]).
    prepare_recursive(block, parent, base + 1, n - 1);

    // Then split q_base off the upper register: UC-Ry (magnitudes) and UC-Rz
    // (relative phases), both controlled on the upper register.
    std::vector<uint32_t> controls;
    controls.reserve(n - 1);
    for (uint32_t k = 1; k < n; ++k) controls.push_back(base + k);

    apply_uc_ry(block, base, controls, ty);
    apply_uc_rz(block, base, controls, tz);
}

}  // anonymous namespace

void state_preparation(
    Block& block,
    const std::vector<std::complex<double>>& amplitudes)
{
    const size_t N = amplitudes.size();
    if (N == 0 || (N & (N - 1)) != 0) {
        throw std::runtime_error(
            "state_preparation: amplitude vector size must be a power of 2");
    }
    const uint32_t n = static_cast<uint32_t>(std::countr_zero(N));

    // Normalize.
    double norm_sq = 0.0;
    for (const auto& z : amplitudes) norm_sq += std::norm(z);
    if (norm_sq < kZeroEps) {
        throw std::runtime_error(
            "state_preparation: amplitude vector has zero norm");
    }
    const double inv_norm = 1.0 / std::sqrt(norm_sq);

    std::vector<std::complex<double>> a(amplitudes);
    for (auto& z : a) z *= inv_norm;

    prepare_recursive(block, a, /*base=*/0, n);
}

}  // namespace qarpx::synthesis
