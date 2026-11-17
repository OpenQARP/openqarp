#pragma once

#include "../core/command.h"

#include <cstddef>
#include <vector>

namespace qarpx {

/// Fuse consecutive single-qubit gates on the same qubit into a single Custom
/// gate (2×2 dense matrix).
///
/// Algorithm — per-qubit matrix accumulation (linear scan):
///
///   An active accumulator (2×2 complex matrix) is kept for every qubit that
///   has seen at least one fusible gate since its last flush.
///
///   For each command in the input sequence:
///     • Single-qubit gate, concrete params:
///         Multiply its 2×2 unitary into accumulator[q].  Do not emit yet.
///     • Single-qubit gate, symbolic params:
///         Flush accumulator[q] first (emit a Custom command if non-identity),
///         then emit the symbolic gate as-is.
///     • Multi-qubit gate (2+ qubits):
///         Flush accumulators for every qubit it touches, then emit the gate.
///     • GPhase / Barrier / Measure / Reset:
///         Emit as-is without touching accumulators (GPhase has no qubit index
///         so no flush is needed; Barrier/Measure/Reset flush touched qubits
///         to preserve semantics).
///
///   At the end of the sequence all remaining accumulators are flushed.
///
/// A flushed accumulator produces one Custom gate whose `unitary` field holds a
/// 2×2 Eigen::MatrixXcd.  The gate is suppressed if the accumulated matrix is
/// within `identity_tol` of the 2×2 identity (global phases are kept so as not
/// to change observable results).
///
/// @param commands     Input command sequence (may contain symbolic params).
/// @param identity_tol Frobenius-norm tolerance for identity suppression.
///                     Default 1e-12 catches only exact no-ops.
/// @returns            Fused command sequence.
[[nodiscard]] std::vector<Command> fuse_single_qubit_gates(
    const std::vector<Command>& commands,
    double identity_tol = 1e-12);

}  // namespace qarpx
