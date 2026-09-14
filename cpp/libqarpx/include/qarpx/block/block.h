#pragma once

#include "../core/command.h"
#include "../core/param.h"
#include "../core/pauli.h"

#include <nanobind/intrusive/ref.h>

#include <complex>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qarpx {

class Block;

/// Smart pointer used throughout the block tree.  Aliased to nanobind's
/// intrusive ``ref<T>`` (see ``Block``'s ``nanobind::intrusive_base`` base):
/// a block's C++ references share a single reference count with its Python
/// wrapper, so there is no C++/Python reference cycle between the holder and
/// the Python ``__dict__``.  In pure C++ (the test binaries) it degrades to
/// plain atomic reference counting.
template <typename T>
using ref = nanobind::ref<T>;

/// Base class for all quantum circuit blocks.
///
/// Mirrors OpenQARP's Python Block class. Each concrete block implements build()
/// to populate its internal command list. The block hierarchy is preserved
/// until flatten() is called, which recursively collects all commands with
/// qubit remapping.
///
/// Key design: blocks are value-like objects that can be shared via
/// ``ref<Block>`` (intrusive reference counting) in CompositeBlock trees.
/// Transformations (dagger, set_symbols, replace_symbols) return new blocks —
/// originals are never mutated.
class Block : public nanobind::intrusive_base {
public:
    virtual ~Block() = default;

    // Identity
    std::string name = "Block";
    uint32_t n_qubits = 0;

    /// Declared width of the classical register this block writes into
    /// (Measure) or reads from (ConditionalBlock conditions).  Mirrors
    /// n_qubits for the classical register.
    ///
    /// `0` means "unset": `build()` fills it in — `SimpleBlock` from its own
    /// command stream via `cbit_register_width` (core/command.h),
    /// `CompositeBlock` by aggregating children.  Assigning a value before
    /// build declares a register *wider* than the circuit uses and is never
    /// overwritten; the absorbers rely on this to round-trip a source
    /// `bit[n] c;` that is wider than the measurements it contains.
    ///
    /// Inference runs at the *first* build only, so rebuilding never shrinks
    /// an established width.  The Python `Block.build()` additionally returns
    /// early on an already-built block (so `build_vanilla()` cannot
    /// double-append commands), which means clearing `n_cbits` afterwards does
    /// not re-arm inference — build a fresh block instead.
    uint32_t n_cbits = 0;

    // Qubit placement within parent composite
    std::optional<std::vector<uint32_t>> target_qubits;

    /// Cbit placement within parent composite, mirroring target_qubits.
    /// Lets two sibling blocks share a parent cbit (e.g. measurement-block
    /// writes cbit 0 of the parent; correction-block reads cbit 0 of the
    /// parent) by both pointing their local cbit indices at the same parent
    /// cbit.  Without an explicit override, CompositeBlock offsets each
    /// child's cbits sequentially.
    std::optional<std::vector<uint32_t>> target_cbits;

    // Control qubits
    std::optional<uint32_t> n_controls;
    std::optional<std::vector<bool>> control_state;

    /// Build the block: populate commands_.  Must be called before flatten().
    /// Concrete blocks override this to emit their gate sequence.
    virtual void build() = 0;

    [[nodiscard]] bool is_built() const { return built_; }

    /// Access the block's local commands (not recursively flattened).
    [[nodiscard]] const std::vector<Command>& commands() const { return commands_; }

    /// Replace the local command buffer wholesale.  Used by Python's
    /// ``__deepcopy__`` to clone an ad-hoc-populated block whose state
    /// lives in ``commands_`` rather than in a ``build_vanilla`` override.
    /// Does NOT touch ``built_`` — pair with ``set_built`` if needed.
    void set_commands(std::vector<Command> cmds) { commands_ = std::move(cmds); }

    /// Force the ``built_`` flag.  Mirrors the source state when a Python
    /// ``__deepcopy__`` clones an already-built block.
    void set_built(bool b) { built_ = b; }

    /// Clone-internal (Python ``__deepcopy__``): copy every base-class field
    /// (identity, placement, controls, command buffer, built flag) via the
    /// compiler-generated assignment, so the copy stays complete as fields
    /// are added.  Tree edges (composite children, controlled inner,
    /// conditional bodies) live in subclasses and are not touched; the
    /// nanobind intrusive refcount is copy-stable by design.
    void copy_base_state_from(const Block& src) { *this = src; }

    /// Collect all free symbol names across this block's commands.
    [[nodiscard]] std::vector<std::string> free_symbols() const;

    /// Recursively flatten the block tree into a single command sequence.
    /// Applies target_qubit remapping at each level.
    /// The block must be built before calling this.
    [[nodiscard]] virtual std::vector<Command> flatten() const;

    /// "Same circuit" check: equal qubit count and pairwise-`approx_equal`
    /// flattened command sequences (same gates, qubits, cbits, conditions,
    /// and rotation angles/symbols within `atol`).  Order-sensitive — two
    /// blocks whose commands are a reordered-but-equivalent permutation of
    /// each other compare unequal.  Both blocks must be built (as required
    /// by `flatten()`).
    [[nodiscard]] bool equals(const Block& other, double atol = 1e-9) const;

    bool operator==(const Block& other) const { return equals(other); }
    bool operator!=(const Block& other) const { return !equals(other); }

    // ── Transformations (return new block, original unchanged) ──

    /// Return the adjoint of this block.
    [[nodiscard]] virtual ref<Block> dagger() const;

    /// Substitute symbolic parameters with concrete values.
    [[nodiscard]] virtual ref<Block> set_symbols(
        const std::unordered_map<std::string, double>& values) const;

    /// Rename symbols (symbol -> new_symbol).
    [[nodiscard]] virtual ref<Block> replace_symbols(
        const std::unordered_map<std::string, std::string>& mapping) const;

    // ── Builder API — add gates before calling build() ──

    Block& h(uint32_t q);
    Block& x(uint32_t q);
    Block& y(uint32_t q);
    Block& z(uint32_t q);
    Block& s(uint32_t q);
    Block& sdg(uint32_t q);
    Block& t(uint32_t q);
    Block& tdg(uint32_t q);
    Block& sx(uint32_t q);
    Block& sxdg(uint32_t q);
    Block& id(uint32_t q);
    Block& rx(uint32_t q, Param angle);
    Block& ry(uint32_t q, Param angle);
    Block& rz(uint32_t q, Param angle);
    Block& p(uint32_t q, Param angle);
    Block& cx(uint32_t control, uint32_t target);
    Block& cy(uint32_t control, uint32_t target);
    Block& cz(uint32_t control, uint32_t target);
    Block& swap(uint32_t q0, uint32_t q1);
    Block& ccx(uint32_t c0, uint32_t c1, uint32_t target);
    Block& cswap(uint32_t control, uint32_t q0, uint32_t q1);
    Block& mcz(std::vector<uint32_t> qubits);
    Block& rzz(uint32_t q0, uint32_t q1, Param angle);
    Block& rxx(uint32_t q0, uint32_t q1, Param angle);
    Block& ryy(uint32_t q0, uint32_t q1, Param angle);
    Block& crx(uint32_t control, uint32_t target, Param angle);
    Block& cry(uint32_t control, uint32_t target, Param angle);
    Block& crz(uint32_t control, uint32_t target, Param angle);
    Block& cp(uint32_t control, uint32_t target, Param angle);
    Block& ecr(uint32_t q0, uint32_t q1);
    Block& iswap(uint32_t q0, uint32_t q1);
    Block& iswapdg(uint32_t q0, uint32_t q1);
    Block& ch(uint32_t control, uint32_t target);
    Block& cs(uint32_t control, uint32_t target);
    Block& csdg(uint32_t control, uint32_t target);
    Block& csx(uint32_t control, uint32_t target);
    Block& csxdg(uint32_t control, uint32_t target);
    Block& gphase(Param angle);
    Block& u(uint32_t q, Param theta, Param phi, Param lambda);
    Block& cu(uint32_t control, uint32_t target,
              Param theta, Param phi, Param lambda, Param gamma);
    Block& measure(uint32_t qubit, uint32_t cbit);
    Block& reset(uint32_t qubit);
    Block& barrier(std::vector<uint32_t> qubits);

    // ── Variadic / bulk builder overloads ──
    //
    // Each accepts a list of "gate applications" and emits them all in a
    // single call.  Shape mirrors the scalar form: 1Q-no-param takes a list
    // of qubits; 1Q-param takes a list of (qubit, Param) pairs; 2Q-no-param
    // takes a list of (q0, q1) pairs; 2Q-param takes a list of (q0, q1, Param)
    // tuples.  The bulk path goes through `commands_.reserve` + tight loop
    // to amortise the C++ overhead and (more importantly) collapse the
    // Python→C++ crossings to a single nanobind dispatch per call.

    Block& h(const std::vector<uint32_t>& qubits);
    Block& x(const std::vector<uint32_t>& qubits);
    Block& y(const std::vector<uint32_t>& qubits);
    Block& z(const std::vector<uint32_t>& qubits);
    Block& s(const std::vector<uint32_t>& qubits);
    Block& sdg(const std::vector<uint32_t>& qubits);
    Block& t(const std::vector<uint32_t>& qubits);
    Block& tdg(const std::vector<uint32_t>& qubits);
    Block& sx(const std::vector<uint32_t>& qubits);
    Block& sxdg(const std::vector<uint32_t>& qubits);
    Block& id(const std::vector<uint32_t>& qubits);

    Block& rx(const std::vector<std::pair<uint32_t, Param>>& gates);
    Block& ry(const std::vector<std::pair<uint32_t, Param>>& gates);
    Block& rz(const std::vector<std::pair<uint32_t, Param>>& gates);
    Block& p(const std::vector<std::pair<uint32_t, Param>>& gates);

    Block& cx(const std::vector<std::pair<uint32_t, uint32_t>>& pairs);
    Block& cy(const std::vector<std::pair<uint32_t, uint32_t>>& pairs);
    Block& cz(const std::vector<std::pair<uint32_t, uint32_t>>& pairs);
    Block& swap(const std::vector<std::pair<uint32_t, uint32_t>>& pairs);
    Block& ecr(const std::vector<std::pair<uint32_t, uint32_t>>& pairs);
    Block& iswap(const std::vector<std::pair<uint32_t, uint32_t>>& pairs);
    Block& iswapdg(const std::vector<std::pair<uint32_t, uint32_t>>& pairs);
    Block& ch(const std::vector<std::pair<uint32_t, uint32_t>>& pairs);
    Block& cs(const std::vector<std::pair<uint32_t, uint32_t>>& pairs);
    Block& csdg(const std::vector<std::pair<uint32_t, uint32_t>>& pairs);
    Block& csx(const std::vector<std::pair<uint32_t, uint32_t>>& pairs);
    Block& csxdg(const std::vector<std::pair<uint32_t, uint32_t>>& pairs);

    Block& crx(const std::vector<std::tuple<uint32_t, uint32_t, Param>>& gates);
    Block& cry(const std::vector<std::tuple<uint32_t, uint32_t, Param>>& gates);
    Block& crz(const std::vector<std::tuple<uint32_t, uint32_t, Param>>& gates);
    Block& cp(const std::vector<std::tuple<uint32_t, uint32_t, Param>>& gates);
    Block& rzz(const std::vector<std::tuple<uint32_t, uint32_t, Param>>& gates);
    Block& rxx(const std::vector<std::tuple<uint32_t, uint32_t, Param>>& gates);
    Block& ryy(const std::vector<std::tuple<uint32_t, uint32_t, Param>>& gates);

    Block& ccx(const std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>& triples);
    Block& cswap(const std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>& triples);

    Block& measure(const std::vector<std::pair<uint32_t, uint32_t>>& gates);
    Block& reset(const std::vector<uint32_t>& qubits);

    // ── Synthesis builders — produce a gate sequence from a target object ──
    //
    // These delegate to qarpx::synthesis::* and append the synthesized gate
    // sequence to commands_.  See qarpx/synthesis/*.h for algorithm details.

    /// Prepare an arbitrary `2^n`-amplitude state from `|0…0⟩` (Möttönen).
    Block& state_preparation(
        const std::vector<std::complex<double>>& amplitudes);

    /// Implement an arbitrary `2^n × 2^n` diagonal unitary (Shende-Bullock-Markov).
    Block& diagonal_unitary(
        const std::vector<std::complex<double>>& diagonal_elements);

    /// Implement an arbitrary `2^n × 2^n` unitary via Quantum Shannon Decomposition.
    Block& unitary_synthesis(const Eigen::MatrixXcd& U);

    /// Append `exp(-i · angle / 2 · pauli)` for a single multi-qubit Pauli.
    /// `pauli.size()` must equal `n_qubits`.
    Block& pauli_exp(const PauliString& pauli, Param angle);

    /// Append `Π_i exp(-i · angles[i] / 2 · paulis[i])` for a set of mutually
    /// commuting Pauli strings.  Validation: throws `std::invalid_argument`
    /// on length mismatch, row-length mismatch, or non-commuting pairs.
    Block& commuting_pauli_set_exp(
        const std::vector<PauliString>& paulis,
        const std::vector<Param>& angles);

protected:
    std::vector<Command> commands_;
    bool built_ = false;

    /// Helper: append a command.
    void add_command(Command cmd);

    /// Convenience overloads for add_command.
    void add_gate(GateType g, uint32_t q);
    void add_gate(GateType g, uint32_t q, Param p);
    void add_gate(GateType g, uint32_t q0, uint32_t q1);
    void add_gate(GateType g, uint32_t q0, uint32_t q1, Param p);
};


/// A simple concrete block that holds a manually-specified list of commands.
/// Useful for building circuits directly from Python without a custom Block subclass.
class SimpleBlock : public Block {
public:
    explicit SimpleBlock(uint32_t n_qubits, const std::string& name = "Circuit");

    void build() override;

    /// Allocate the next `size` qubit indices for a named register.
    /// Returns the allocated indices [next, ..., next+size-1].
    /// Purely a bookkeeping aid — no enforcement against n_qubits.
    std::vector<uint32_t> add_register(const std::string& name, uint32_t size);

private:
    uint32_t next_qubit_ = 0;
};

}  // namespace qarpx
