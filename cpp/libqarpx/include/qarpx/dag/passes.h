#pragma once

#include "circuit_dag.h"

#include <cstddef>

namespace qarpx::dag_passes {

/// Wire-adjacent cancellation and merging.
///
/// The same rewrite rules as `eliminate_identities` — inverse-pair
/// cancellation (via `are_inverse_pair`, incl. unordered CZ/SWAP), Rx/Ry/Rz
/// rotation merge (via `can_merge_rotations`), zero-rotation drop — plus
/// GPhase merging, applied on WIRE adjacency (immediate successor on every
/// shared wire) instead of textual adjacency.  Strictly more eliminations:
/// `H(0)·X(1)·H(0)` cancels here but not in the linear pass.
///
/// Guarantees (plan §4.4): per-wire order of survivors preserved (EQ-1),
/// exact unitary / outcome-distribution equivalence (EQ-2).  Two commands
/// combine only when their condition tuples match exactly; region nodes are
/// opaque barriers.  Worklist to fixpoint, O(n) amortized.
///
/// Returns the number of commands eliminated, with `eliminate_identities`
/// counting parity: cancelled pair = 2, merge = 1, zero-drop = 1.
std::size_t cancel_wire_adjacent(CircuitDAG& dag);

/// Commutation-aware cancellation and merging (Phase 2, `OptLevel::O2`).
///
/// For each node A, walks up to `window` successors along A's wire chains
/// looking for a combinable partner B (same rules as `cancel_wire_adjacent`);
/// the rewrite fires when every intervening node sharing a wire with A
/// provably commutes with A (`dag_passes::commute` — matrix-verified,
/// conservative).  A is then slid *virtually*: the pair is removed, or the
/// merged rotation replaces B at B's position — node storage is NEVER
/// physically reordered, so the output remains a subsequence-with-
/// substitutions of the input (QASM text / plot layout stay stable modulo
/// removed gates).  Region interiors recurse; boundaries stay barriers.
/// EQ-2 preserved exactly.  Returns eliminated count (same parity as O1).
std::size_t commute_and_cancel(CircuitDAG& dag, std::size_t window = 8);

}  // namespace qarpx::dag_passes
