#pragma once

#include "../core/command.h"

#include <Eigen/Dense>

#include <complex>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace qarpx {

/// |x⟩ → |table[x]⟩ on `qubits`: local bit b ↔ `qubits[b]` (LSB, §1).
struct PermutationKernel {
    std::vector<uint32_t> qubits;
    std::vector<uint64_t> table;
};

/// A 2^k × 2^k unitary on `qubits` (local bit b ↔ `qubits[b]`), k ≤ 8.
struct DenseKernel {
    std::vector<uint32_t> qubits;
    Eigen::MatrixXcd      matrix;
};

/// `matrix` raised to `exponents[j]` on `targets` wherever control
/// `controls[j]` is |1⟩; the powers of one matrix commute, so the controls
/// apply in any order.
struct ControlledPowersKernel {
    std::vector<uint32_t> controls;
    std::vector<uint64_t> exponents;
    std::vector<uint32_t> targets;
    Eigen::MatrixXcd      matrix;
};

/// A slice of the gate stream, run through the ordinary fused dispatch.
struct GatesKernel {
    std::vector<Command> commands;
};

using Kernel = std::variant<PermutationKernel, DenseKernel,
                            ControlledPowersKernel, GatesKernel>;

/// An execution-time lowering of one circuit into typed kernels (§14
/// *Structured execution*).  Kernels validate on insertion, so execution
/// never re-checks them.
class Program {
public:
    void add_gates(std::vector<Command> commands);
    /// Throws std::invalid_argument unless `table` is a bijection of
    /// [0, 2^|qubits|) and `qubits` are distinct.  Folds into a directly
    /// preceding permutation kernel while their qubit union stays within the
    /// table cap, so a run of permutations costs one pass.
    void add_permutation(std::vector<uint32_t> qubits, std::vector<uint64_t> table);
    /// Throws std::invalid_argument unless `matrix` is 2^k square, 1 ≤ k ≤ 8.
    void add_dense(std::vector<uint32_t> qubits, Eigen::MatrixXcd matrix);
    /// Throws std::invalid_argument unless `matrix` is 2^|targets| square,
    /// controls and targets are disjoint and distinct, and there is one
    /// exponent per control.
    void add_controlled_powers(std::vector<uint32_t>  controls,
                               std::vector<uint64_t>  exponents,
                               std::vector<uint32_t>  targets,
                               Eigen::MatrixXcd       matrix);

    [[nodiscard]] const std::vector<Kernel>& kernels() const { return kernels_; }
    /// "gates" / "permutation" / "dense" / "controlled_powers", in order.
    [[nodiscard]] std::vector<std::string> kinds() const;
    /// Highest qubit index any kernel touches, plus one (0 if none).
    [[nodiscard]] uint32_t min_register_width() const;
    /// The program with `params` substituted into every gates kernel.
    [[nodiscard]] Program substituted(
        const std::unordered_map<std::string, double>& params) const;
    /// The program without `Measure` and `Barrier` commands — the unitary
    /// prefix an exact (Born) readout evaluates.
    [[nodiscard]] Program without_measurements() const;

private:
    std::vector<Kernel> kernels_;
};

/// Apply a permutation kernel out of place: `scratch` must hold `dim`
/// amplitudes and is swapped with `state`.
void apply_permutation(const PermutationKernel&            k,
                       std::vector<std::complex<double>>&  state,
                       std::vector<std::complex<double>>&  scratch);

void apply_controlled_powers(const ControlledPowersKernel&        k,
                             std::vector<std::complex<double>>&   state);

/// The permutation `commands` apply to basis states of `qubits`, exact
/// including phase (within 1e-10), or nullopt: classical gates evaluate on
/// integers at any width; otherwise at most `max_rest` qubits may change and
/// the work must stay within 2^`max_work_log2` · |commands|.
[[nodiscard]] std::optional<std::vector<uint64_t>> permutation_table(
    const std::vector<Command>&  commands,
    const std::vector<uint32_t>& qubits,
    uint32_t                     max_rest,
    uint32_t                     max_work_log2);

/// The 2^k × 2^k unitary `commands` apply to `qubits` (local bit b ↔
/// `qubits[b]`), global phase included; k ≤ 12.
[[nodiscard]] Eigen::MatrixXcd local_unitary_of(
    const std::vector<Command>&  commands,
    const std::vector<uint32_t>& qubits);

/// A 64-bit digest of a command stream: gate, qubits, cbits, conditions and
/// parameters.  Equal streams digest equally; used as a cache key.
[[nodiscard]] uint64_t commands_digest(const std::vector<Command>& commands);

/// `commands_digest` of `commands` remapped onto [0, |qubits|) (local bit b
/// ↔ `qubits[b]`), so equal circuits placed on different qubits digest
/// equally; falls back to the unmapped digest when a command lies outside
/// `qubits`.
[[nodiscard]] uint64_t local_commands_digest(const std::vector<Command>&  commands,
                                             const std::vector<uint32_t>& qubits);

}  // namespace qarpx
