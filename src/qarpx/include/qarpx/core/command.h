#pragma once

#include "gates.h"
#include "param.h"
#include "small_vector.h"

#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

// Forward declare Eigen matrix type to avoid heavy header include
namespace Eigen {
template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
class Matrix;
using MatrixXcd = Matrix<std::complex<double>, -1, -1, 0, -1, -1>;
}

namespace qarpx {

/// A single quantum gate operation: gate type + qubit targets + parameters.
///
/// Memory layout is optimized for the common case:
///   - SmallVector<uint32_t, 2> for qubits  (99.5% of gates are 1-2 qubit)
///   - SmallVector<Param, 1>    for params   (92% of parametric gates have 1 param)
///   - SmallVector<uint32_t, 1> for cbits    (only Measure uses 1 cbit)
///   - SmallVector<uint32_t, 1> for condition_bits / condition_values
///         (empty = unconditional; populated only by ConditionalBlock::flatten)
struct Command {
    GateType                    gate   = GateType::Barrier;
    SmallVector<uint32_t, 2>    qubits;
    SmallVector<Param, 1>       params;
    SmallVector<uint32_t, 1>    cbits;

    /// Classical condition: gate runs only when, for every i,
    /// classical_register[condition_bits[i]] == condition_values[i].
    /// Empty = unconditional (default for every Command unless a ConditionalBlock
    /// stamped it during flatten).  AND-only conjunction; richer Boolean
    /// conditions are out of scope (see plan §scope-decisions).
    SmallVector<uint32_t, 1>    condition_bits;
    SmallVector<bool,     1>    condition_values;

    /// Optional unitary matrix for Custom gates.
    std::shared_ptr<const Eigen::MatrixXcd> unitary = nullptr;

    // Default constructors
    Command() = default;

    /// Single-qubit gate, no params.
    Command(GateType g, uint32_t q)
        : gate(g), qubits{q} {}

    /// Single-qubit gate, 1 param.
    Command(GateType g, uint32_t q, Param p)
        : gate(g), qubits{q}, params{std::move(p)} {}

    /// Two-qubit gate, no params.
    Command(GateType g, uint32_t q0, uint32_t q1)
        : gate(g), qubits{q0, q1} {}

    /// Two-qubit gate, 1 param.
    Command(GateType g, uint32_t q0, uint32_t q1, Param p)
        : gate(g), qubits{q0, q1}, params{std::move(p)} {}

    /// General constructor.
    Command(GateType g,
            SmallVector<uint32_t, 2> qs,
            SmallVector<Param, 1> ps = {},
            SmallVector<uint32_t, 1> cs = {})
        : gate(g), qubits(std::move(qs)), params(std::move(ps)), cbits(std::move(cs)) {}

    /// True if any parameter is symbolic (not a concrete double).
    [[nodiscard]] bool is_parametric() const;

    /// Return the adjoint (dagger) of this command.
    /// - Self-adjoint gates (X, H, CX, ...) are returned unchanged.
    /// - S <-> Sdg, T <-> Tdg
    /// - Parametric gates: params are negated (Rx(θ)† = Rx(-θ))
    [[nodiscard]] Command dagger() const;

    /// Return a copy with qubits remapped: qubit[i] = mapping[qubit[i]].
    [[nodiscard]] Command remap_qubits(const std::vector<uint32_t>& mapping) const;

    /// Substitute symbolic parameters with concrete values.
    [[nodiscard]] Command substitute(const std::unordered_map<std::string, double>& values) const;

    /// Copy with every parameter's symbols renamed (``Param::rename_symbols``).
    [[nodiscard]] Command rename_symbols(
        const std::unordered_map<std::string, std::string>& mapping) const;

    /// Equality: same gate, same qubits, same params.
    bool operator==(const Command& other) const;
    bool operator!=(const Command& other) const { return !(*this == other); }

    /// Tolerant equality: same gate/qubits/cbits/conditions, params compare
    /// via `Param::approx_equal` (within `atol`) instead of bit-exact.  Used
    /// by `Block::equals()` since SDK round-trips (angle unit conversions,
    /// sign flips) don't generally reproduce the exact same double.
    [[nodiscard]] bool approx_equal(const Command& other, double atol = 1e-9) const;

    /// Human-readable string (for debugging).
    [[nodiscard]] std::string to_string() const;
};

/// Substitute symbolic parameters in a full command sequence.
/// Equivalent to `[cmd.substitute(values) for cmd in commands]` but the
/// iteration stays entirely in C++ — used on the QarpEngine.run() hot path.
[[nodiscard]] std::vector<Command> substitute_all(
    const std::vector<Command>& commands,
    const std::unordered_map<std::string, double>& values);

/// One renamed appearance of a symbol: which command and parameter slot, and
/// the private name it now carries.
struct RenamedOccurrence {
    std::size_t command;
    std::size_t param;
    std::string name;
};

/// Give every occurrence of ``symbol`` in ``commands`` its own private name,
/// ``prefix + "<command>\0<param>"``, so a per-occurrence parameter shift
/// can move one gate angle while every other appearance of the symbol stays
/// put.  Compound parameters keep evaluating (``Param::rename_symbols``).
/// The caller chooses ``prefix`` so names cannot collide with user symbols
/// (qarp uses NUL separators, which no sympy symbol carries).
[[nodiscard]] std::pair<std::vector<Command>, std::vector<RenamedOccurrence>>
rename_symbol_occurrences(
    const std::vector<Command>& commands,
    const std::string& symbol,
    const std::string& prefix);

/// Tolerant sequence equality: same length, and every command pairwise
/// `approx_equal` at index i.  Order-sensitive — this is "same circuit",
/// not "same multiset of equivalent commands" (absorbers that reorder
/// commands relative to the source aren't covered by this check).
[[nodiscard]] bool commands_equal(
    const std::vector<Command>& a,
    const std::vector<Command>& b,
    double atol = 1e-9);

/// In-place rewrite of every cbit reference in a flat command list.
///
/// `mapping[i]` is the new cbit index that replaces local cbit `i`.  Both
/// `Command::cbits` (the Measure write target) and `Command::condition_bits`
/// (the ConditionalBlock read targets) are remapped.  Used by CompositeBlock::flatten
/// to push child-local cbits into the parent's cbit space, and by
/// ConditionalBlock::flatten when the body has its own cbit scope.
void cbit_remap_in_place(
    std::vector<Command>& commands,
    const std::vector<uint32_t>& mapping);

/// Width of the classical register a command stream needs: 1 + the largest
/// cbit index referenced, or 0 if none is.  Counts every `Measure`'s write
/// targets (`Command::cbits`) and every command's `condition_bits`
/// (`BranchBegin` stores its AND-condition tuple in the same field).
///
/// The single definition of the cbit register width: the simulator sizes
/// `SamplingResult::n_cbits` with it and `SimpleBlock::build()` infers
/// `Block::n_cbits` from it, so the two cannot disagree.
///
/// A `Measure` with empty `cbits` contributes nothing — §8 makes the cbit part
/// of the `Measure` contract, so such a command is malformed rather than
/// meaningful.  The OpenQASM 3 emitter widens its *own* declaration to keep
/// such a program syntactically valid; that fallback is emitter-local and
/// deliberately not part of this contract.
[[nodiscard]] uint32_t cbit_register_width(const std::vector<Command>& commands);

/// Number of `k`-qubit gates in a command stream — see `gate_is_physical`
/// (core/gates.h) for what counts as a gate (`Barrier`/`Measure`/`Reset`/
/// `GPhase`/branch markers do not). A plain linear scan: gate counting is
/// not a graph property, so no `CircuitDAG` is built for it.
[[nodiscard]] uint32_t n_nqb_gates(const std::vector<Command>& commands, uint32_t k);

/// Total physical gate count over all arities — the resource vector's
/// headline `n_gates` (same `gate_is_physical` classification).
[[nodiscard]] uint32_t n_physical_gates(const std::vector<Command>& commands);

/// Number of commands of the given `GateType` — unfiltered. Unlike
/// `n_nqb_gates`, every command counts as itself (`Barrier`/`Measure`/
/// `Reset`/`GPhase`/branch markers included), mirroring
/// `CircuitDAG::count_ops()`.
[[nodiscard]] uint32_t n_gates_of_type(const std::vector<Command>& commands, GateType gate);

/// Cbits a command stream *reads* as a classical condition before any `Measure`
/// has written them, in ascending order; empty means every condition is
/// preceded by a write that could have set it.
///
/// Reading an unwritten cbit is not a range error — the register is
/// zero-initialised, so the condition silently evaluates false and the guarded
/// body never runs.  The usual cause is composing a `ConditionalBlock` as a
/// sibling of the measurement that feeds it: `CompositeBlock` gives children
/// disjoint cbit ranges, so the condition is offset past the write unless an
/// explicit `target_cbits` aliases them.
///
/// Order-sensitive and whole-circuit: evaluate it on a *complete* flattened
/// program.  A sub-block whose measurement lives in an enclosing block will
/// report its condition cbits here and is not necessarily faulty.
[[nodiscard]] std::vector<uint32_t> uninitialised_condition_cbits(
    const std::vector<Command>& commands);

}  // namespace qarpx
