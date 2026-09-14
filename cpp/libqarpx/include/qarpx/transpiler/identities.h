#pragma once

#include "../core/command.h"

#include <cstddef>
#include <vector>

namespace qarpx {

/// Eliminate identity patterns from a command sequence (in-place).
///
/// Removes/simplifies WIRE-adjacent pairs (immediate neighbours on every
/// shared wire — gates on other qubits in between are transparent):
///   - H H (same qubit)                       → remove both
///   - X X, Y Y, Z Z (same qubit)             → remove both
///   - CX CX (same qubits, same order)        → remove both
///   - CZ CZ (same qubits, either order)      → remove both
///   - SWAP SWAP (same qubits, either order)   → remove both
///   - S Sdg, Sdg S (same qubit)              → remove both
///   - T Tdg, Tdg T (same qubit)              → remove both
///   - Rz(a) Rz(b) (same qubit, wire-adjacent) → Rz(a+b)
///   - Rx(a) Rx(b) / Ry(a) Ry(b) likewise
///   - Rz(0), Rx(0), Ry(0)                    → remove
///   - GPhase(a) GPhase(b)                    → GPhase(a+b)
///
/// Routes through the `CircuitDAG` pass
/// `dag_passes::cancel_wire_adjacent`, cancelling on wire adjacency rather
/// than textual adjacency.  Surviving commands keep their relative order;
/// `BranchBegin…BranchEnd` interiors are optimized per body, boundaries are
/// barriers.  Throws std::invalid_argument on unbalanced branch markers.
///
/// Returns the number of commands eliminated.
std::size_t eliminate_identities(std::vector<Command>& commands);

/// Check if two rotation commands on the same qubit can be merged.
/// Returns true if both are the same rotation type on the same qubit.
bool can_merge_rotations(const Command& a, const Command& b);

/// True iff summing two gate angles keeps `Param`'s linear form α·sym + β,
/// i.e. their symbols union to at most one.  Merges that would produce a
/// compound angle are not differentiable by the adjoint-gradient path.
bool merged_param_stays_linear(const Param& a, const Param& b);

/// Check if two commands are inverse pairs (cancel to identity).
bool are_inverse_pair(const Command& a, const Command& b);

}  // namespace qarpx
