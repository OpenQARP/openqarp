#pragma once

#include "../core/command.h"

#include <Eigen/Dense>

#include <cstddef>
#include <functional>
#include <vector>

namespace qarpx {

/// 2^|qubits| × 2^|qubits| unitary of one concrete command over its own
/// qubits — local bit b ↔ `cmd.qubits[b]` (§1 LSB), exactly as
/// `QarpSimulator::apply_command` would apply it.
using LocalUnitary = std::function<Eigen::MatrixXcd(const Command&)>;

/// Simulator-internal fusion of a command stream into dense `Custom` blocks
/// of at most `max_qubits` qubits.  The product never leaves the simulator:
/// the transpiler's O1 pass (`fuse_single_qubit_gates`) keeps its 1-qubit
/// `Custom` contract (§16 rebase totality).
///
/// Greedy wire-front folding, one linear scan.  Every open block holds a
/// qubit tuple and the accumulated unitary over it; a gate on wires W folds
/// into open block B iff the union stays within `max_qubits`, the gate
/// carries B's classical condition, and no block created after B has touched
/// any wire in W (so appending the gate to B preserves every per-wire order —
/// EQ-1).  Wires are qubits plus the cbits a command reads (`condition_bits`)
/// or writes (`Measure::cbits`), so a conditional gate never drifts across
/// the measurement feeding it.  A gate that cannot fold — too wide, symbolic,
/// non-unitary — is emitted in place after flushing the open blocks that must
/// precede it.  `GPhase` rides the global wire and passes through untouched;
/// a qubit-less `Barrier` and the branch markers flush everything (§16).
///
/// A flushed block with one gate re-emits that gate verbatim; a 1-qubit block
/// whose product is the identity within 1e-12 is dropped (matching
/// `fuse_single_qubit_gates`); anything else becomes one `Custom` carrying the
/// block's unitary and condition.  Exact including global phase (EQ-2).
///
/// `max_qubits == 0` returns the input unchanged.
[[nodiscard]] std::vector<Command> fuse_for_simulation(
    const std::vector<Command>& commands,
    std::size_t                 max_qubits,
    const LocalUnitary&         local_unitary);

}  // namespace qarpx
