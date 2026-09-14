#pragma once

#include "block.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace qarpx {

/// A block composed of a sequence of child blocks.
///
/// Mirrors OpenQARP's Python CompositeBlock. Children are stored as ref<Block>
/// (intrusive reference counting), enabling structural sharing (the same block
/// instance can appear in multiple composites without copying commands).
///
/// The block hierarchy is preserved until flatten() is called. flatten()
/// recursively collects commands from all children, applying target_qubit
/// remapping at each level.
class CompositeBlock : public Block {
public:
    CompositeBlock(
        std::vector<ref<Block>> children,
        uint32_t n_qubits = 0,
        const std::string& name = "CompositeBlock"
    );

    /// Build all children that haven't been built yet, then collect symbols.
    void build() override;

    /// Recursively flatten all children into a single command sequence.
    [[nodiscard]] std::vector<Command> flatten() const override;

    /// Substitute symbolic params recursively across children.  The base
    /// ``Block::set_symbols`` operates on ``commands_`` (empty for composites),
    /// so without this override a composite would discard all of its content.
    [[nodiscard]] ref<Block> set_symbols(
        const std::unordered_map<std::string, double>& values) const override;

    /// Symbol→symbol rename, recursive.  Same rationale as ``set_symbols``.
    [[nodiscard]] ref<Block> replace_symbols(
        const std::unordered_map<std::string, std::string>& mapping) const override;

    /// Recursive dagger: each child is daggered and the child order is
    /// reversed.  The base ``Block::dagger`` operates on ``commands_`` (empty
    /// for composites), so without this override a composite's dagger would
    /// silently return an empty block.  Surfaced via QSVT even-degree
    /// polynomial application — every BE† layer in the QSVT recipe was
    /// dropping to identity, collapsing the d=2 circuit into a d=1
    /// computation.  Tracked under §3.3 #10 of the pytket-removal plan.
    [[nodiscard]] ref<Block> dagger() const override;

    /// Access the child blocks.
    [[nodiscard]] const std::vector<ref<Block>>& children() const {
        return children_;
    }

    /// Add a child block.
    void add_child(ref<Block> child);

    /// Scan the flattened command sequence and return the cbit indices that
    /// any conditional command reads but no Measure ever writes earlier.
    /// Empty result means every condition has a preceding Measure that could
    /// have set it.  Useful as a structural sanity check before passing the
    /// sequence to the simulator.
    [[nodiscard]] std::vector<uint32_t> uninitialised_condition_cbits() const;

private:
    std::vector<ref<Block>> children_;

    /// Determine n_qubits from children if not explicitly set.
    void resolve_qubits();

    /// Determine n_cbits by summing children's n_cbits (respecting
    /// target_cbits overrides).  Only runs when n_cbits is unset (== 0)
    /// and at least one child uses cbits.
    void resolve_cbits();
};

}  // namespace qarpx
