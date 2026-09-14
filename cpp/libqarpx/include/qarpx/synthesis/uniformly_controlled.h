#pragma once

#include "../block/block.h"

#include <cstdint>
#include <vector>

namespace qarpx::synthesis {

/// Apply a uniformly-controlled Ry to `block`.
///
/// For each control state `i ∈ {0, ..., 2^|controls|-1}`, the rotation `Ry(angles[i])`
/// is applied to `target` iff the control register reads `i` (LSB-first: control-state
/// `i` means qubit `controls[k]` carries bit `k` of `i`).
///
/// Decomposition (Möttönen et al. 2004): `2^c` plain `Ry` rotations interleaved with
/// `2^c` `CX` gates, where the rotation angles are the binary-reflected-Gray-code-
/// reordered Walsh-Hadamard transform of `angles`.  The CX control at step `k` is the
/// position of the single bit that differs between `Gray(k)` and `Gray(k+1)`, with the
/// final wrap-around step (`k = 2^c-1`) using the highest control.
///
/// `controls.empty()` is allowed and emits a single `Ry(angles[0])`.
void apply_uc_ry(
    Block& block,
    uint32_t target,
    const std::vector<uint32_t>& controls,
    const std::vector<double>& angles);

/// Apply a uniformly-controlled Rz to `block`.  Same shape as `apply_uc_ry`.
void apply_uc_rz(
    Block& block,
    uint32_t target,
    const std::vector<uint32_t>& controls,
    const std::vector<double>& angles);

}  // namespace qarpx::synthesis
