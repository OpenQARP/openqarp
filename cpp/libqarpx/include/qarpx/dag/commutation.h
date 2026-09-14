#pragma once

#include "../core/command.h"

#include <cstddef>
#include <cstdint>

namespace qarpx::dag_passes {

/// Single-qubit commutant basis of a gate on one of its wires: the gate
/// commutes with that Pauli on that wire (equivalently, it is block-diagonal
/// in that wire's Pauli eigenbasis).  `kNone` = no certified basis.
enum class WireBasis : uint8_t { kNone, kX, kY, kZ };

/// Commutant basis of `cmd` on its k-th qubit (index into cmd.qubits).
/// Table-driven and conservative: gates without an obvious per-wire basis
/// (H, U, SWAP, iSWAP, ECR, Custom, meta ops) return kNone everywhere.
/// Pinned against explicit matrix commutators in test_dag_commutation.cpp.
[[nodiscard]] WireBasis qubit_basis(const Command& cmd, std::size_t k);

/// True if `a` and `b` PROVABLY commute as operators:
///   - neither is Barrier / Measure / Reset / Branch marker / Custom, and
///   - they share no classical bits (cbits or condition_bits — conservative
///     total order on cbit wires, plan §4.2), and
///   - on every shared qubit their commutant bases are equal and not kNone.
/// Disjoint supports (and no cbit overlap) trivially commute.  Sufficient,
/// not necessary: a false result never certifies non-commutation.
[[nodiscard]] bool commute(const Command& a, const Command& b);

}  // namespace qarpx::dag_passes
