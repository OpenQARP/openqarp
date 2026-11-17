#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace qarpx {

/// One of the four single-qubit Pauli operators.
enum class Pauli : uint8_t {
    I = 0,
    X = 1,
    Y = 2,
    Z = 3,
};

/// A multi-qubit Pauli operator P = ⊗_q P_q with P_q acting on qubit q.
/// Index q is the qubit index (LSB-first per qarp_conventions.md §1).
using PauliString = std::vector<Pauli>;

/// Parse a string like "IXYZ" into a PauliString.  Accepts both upper and
/// lower case.  Throws `std::invalid_argument` on empty input or any
/// character outside `{I, X, Y, Z, i, x, y, z}`.
PauliString parse_pauli_string(std::string_view s);

/// Inverse of `parse_pauli_string`: format a PauliString as upper-case
/// letters in qubit-index order, e.g. `{I, X, Y, Z} → "IXYZ"`.
std::string to_string(const PauliString& p);

/// Pairwise commutation predicate for multi-qubit Paulis.
///
/// `a` and `b` commute iff the number of qubits at which they
/// individually anticommute is even (single-qubit X/Y/Z pairwise
/// anticommute; identity commutes with everything).  Throws
/// `std::invalid_argument` if `a.size() != b.size()`.
bool commutes(const PauliString& a, const PauliString& b);

}  // namespace qarpx
