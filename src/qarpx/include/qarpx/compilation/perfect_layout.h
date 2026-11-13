#pragma once

#include "../core/command.h"
#include "../device/architecture.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace qarpx {

/// Search for a layout that needs no SWAPs at all.
///
/// If the circuit's two-qubit *interaction graph* embeds into the coupling
/// map — every interacting pair landing on a hardware edge — then routing
/// that layout inserts zero SWAPs, which is optimal by construction.  This
/// is a subgraph-isomorphism search (VF2-style backtracking with degree and
/// neighbourhood pruning), bounded by a node budget so an unembeddable
/// circuit costs little before falling back.
///
/// Why it exists: SABRE's reverse-traversal initial mapping is a *local*
/// refinement of one starting point (the identity), so its result depends on
/// how the caller happened to label its logical qubits.  On a 12-qubit ring
/// entangler over a 3x4 grid — which has a Hamiltonian cycle, so zero SWAPs
/// is achievable — relabelling the same circuit moved the SWAP count between
/// 0 and 10, and at 3x6 and 4x5 no relabelling found the optimum at all.
/// A perfect embedding, when one exists, removes that dependence entirely.
///
/// Returns the logical→physical mapping, or `nullopt` when no embedding was
/// found within the budget (either none exists, or the search gave up).
/// Qubits that take part in no two-qubit gate are placed on whatever
/// physical qubits remain.
[[nodiscard]] std::optional<std::vector<uint32_t>> find_perfect_layout(
    const std::vector<Command>& commands,
    const Architecture&         arch,
    std::size_t                 node_budget = 200000);

/// Greedy weighted placement — a *seed*, not a solution.
///
/// Logical pairs are weighted by how many two-qubit gates they share.  The
/// heaviest pair is placed on a maximum-degree physical edge; every further
/// qubit (picked by total weight to already-placed qubits) goes on the free
/// physical qubit minimizing Σ weight(l, placed-neighbour) × distance.  The
/// result starts SABRE's local refinement near a good basin instead of at the
/// identity — the dense-layout idea.  Always returns a full permutation;
/// deterministic (ties break on lowest index).
[[nodiscard]] std::vector<uint32_t> greedy_weighted_layout(
    const std::vector<Command>& commands,
    const Architecture&         arch);

}  // namespace qarpx
