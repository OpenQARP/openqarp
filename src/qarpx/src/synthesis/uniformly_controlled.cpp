#include "qarpx/synthesis/uniformly_controlled.h"

#include <bit>
#include <stdexcept>

namespace qarpx::synthesis {

namespace {

/// Möttönen Walsh-Hadamard / Gray-code transform.
///
/// Given the desired rotation angle `α[i]` for each binary control state `i`, returns
/// the angle vector `ω` to be plugged into the 2^c plain rotations of the decomposition:
///
///   ω[k] = (1/2^c) · Σ_i (-1)^{popcount(b_i ∧ g_k)} · α[i]
///
/// where `g_k = k XOR (k >> 1)` is the binary-reflected Gray code at position `k`.
std::vector<double> mottonen_transform(const std::vector<double>& angles) {
    const size_t N = angles.size();
    std::vector<double> out(N, 0.0);
    for (size_t k = 0; k < N; ++k) {
        const size_t g_k = k ^ (k >> 1);
        for (size_t i = 0; i < N; ++i) {
            const size_t parity = std::popcount(i & g_k) & 1u;
            out[k] += (parity ? -angles[i] : angles[i]);
        }
        out[k] /= static_cast<double>(N);
    }
    return out;
}

/// Index (within `controls`) of the CX control to use after the `k`-th rotation.
/// Equals `ctz(k+1)` for `k+1 ∈ [1, 2^c-1]`, and `c-1` for the wrap-around step
/// `k+1 == 2^c` (closing the Gray-code cycle).
uint32_t cnot_ctrl_index(size_t k, int num_controls) {
    const size_t kp1 = k + 1;
    if (kp1 == (1ull << num_controls)) return static_cast<uint32_t>(num_controls - 1);
    return static_cast<uint32_t>(std::countr_zero(kp1));
}

template <typename ApplyRotation>
void apply_uc_rotation(
    const std::vector<uint32_t>& controls,
    const std::vector<double>& angles,
    ApplyRotation apply)
{
    const size_t N = angles.size();
    const int c = static_cast<int>(controls.size());
    if ((1ull << c) != N) {
        throw std::runtime_error(
            "apply_uc_rotation: angles size must equal 2^|controls|");
    }
    if (c == 0) {
        // No controls: a single plain rotation.
        apply(angles[0], /*emit_cnot=*/false, /*ctrl_qubit=*/0u);
        return;
    }
    const auto omega = mottonen_transform(angles);
    for (size_t k = 0; k < N; ++k) {
        const uint32_t ctrl_idx = cnot_ctrl_index(k, c);
        apply(omega[k], /*emit_cnot=*/true, controls[ctrl_idx]);
    }
}

}  // anonymous namespace

void apply_uc_ry(Block& block,
                 uint32_t target,
                 const std::vector<uint32_t>& controls,
                 const std::vector<double>& angles) {
    apply_uc_rotation(controls, angles,
        [&](double angle, bool emit_cnot, uint32_t ctrl_qubit) {
            block.ry(target, Param(angle));
            if (emit_cnot) block.cx(ctrl_qubit, target);
        });
}

void apply_uc_rz(Block& block,
                 uint32_t target,
                 const std::vector<uint32_t>& controls,
                 const std::vector<double>& angles) {
    apply_uc_rotation(controls, angles,
        [&](double angle, bool emit_cnot, uint32_t ctrl_qubit) {
            block.rz(target, Param(angle));
            if (emit_cnot) block.cx(ctrl_qubit, target);
        });
}

}  // namespace qarpx::synthesis
