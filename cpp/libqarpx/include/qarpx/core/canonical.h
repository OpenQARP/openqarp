#pragma once

#include "command.h"

#include <cstdint>
#include <string>
#include <vector>

namespace qarpx::canonical {

/// A flat command stream rewritten so that real qubits, classical bits, and
/// symbol names are replaced with positional dummies in order of first
/// appearance.  Two blocks that differ only by qubit relocation or symbol
/// renaming produce identical `commands` here (and identical `topology_hash`).
///
/// Bindings:
///   * `qubit_binding[i]`   = real qubit  for dummy `i`
///   * `cbit_binding[i]`    = real cbit   for dummy `i`
///   * `symbol_binding[i]`  = real symbol name for positional symbol `__sym_i`
struct CanonicalForm {
    std::vector<Command>     commands;
    std::vector<uint32_t>    qubit_binding;
    std::vector<uint32_t>    cbit_binding;
    std::vector<std::string> symbol_binding;
};

/// Walk `cmds` in order; assign the next dummy index/name to each previously-
/// unseen real qubit / cbit / symbol.  O(N) over the total command size.
CanonicalForm canonicalize(const std::vector<Command>& cmds);

/// Inverse of `canonicalize`: substitute the dummies in `cf.commands` back to
/// the real qubits / cbits / symbols recorded in `cf.qubit_binding`,
/// `cf.cbit_binding`, `cf.symbol_binding`.  O(N).
[[nodiscard]] std::vector<Command> rebind(const CanonicalForm& cf);

/// Hash that depends only on the command sequence in dummy form.  Two blocks
/// whose canonical commands match (modulo qubit relocation / symbol rename)
/// produce the same value.
[[nodiscard]] uint64_t topology_hash(const CanonicalForm& cf);

/// Hash that depends only on the dummy-to-real bindings.  Two blocks with
/// the same topology but different qubit / symbol assignments produce
/// different values here.
[[nodiscard]] uint64_t binding_hash(const CanonicalForm& cf);

}  // namespace qarpx::canonical
