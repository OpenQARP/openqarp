#pragma once

#include "block.h"

#include <memory>
#include <string>
#include <vector>

namespace qarpx {

/// A wrapper that adds control qubits to an existing block.
///
/// Does not copy the inner block's commands — stores a ref<Block> reference.
/// Control is applied during flatten(): each command from the inner block
/// gets additional control qubits prepended.
class ControlledBlock : public Block {
public:
    ControlledBlock(
        ref<Block> inner,
        uint32_t num_controls,
        std::vector<bool> ctrl_state,
        const std::string& name = "Controlled"
    );

    void build() override;

    [[nodiscard]] std::vector<Command> flatten() const override;

    /// Substitute symbolic params in the inner block; rebuild a controlled
    /// wrapper around the result.  The base ``Block::set_symbols`` operates
    /// on ``commands_`` (empty here — the wrapper produces commands lazily
    /// in ``flatten()``), so without this override controlled composites
    /// would lose their inner content.
    [[nodiscard]] ref<Block> set_symbols(
        const std::unordered_map<std::string, double>& values) const override;

    /// Symbol→symbol rename, propagated to the inner block.
    [[nodiscard]] ref<Block> replace_symbols(
        const std::unordered_map<std::string, std::string>& mapping) const override;

    /// Recursive dagger: dagger the inner block, re-wrap with the same
    /// controls.  Same rationale as ``set_symbols`` / ``replace_symbols`` —
    /// the base ``Block::dagger`` operates on the empty top-level commands
    /// buffer and would silently return an empty block.
    [[nodiscard]] ref<Block> dagger() const override;

    [[nodiscard]] const ref<Block>& inner() const { return inner_; }

    [[nodiscard]] uint32_t num_controls() const { return num_controls_; }
    [[nodiscard]] const std::vector<bool>& ctrl_state() const { return ctrl_state_; }

private:
    ref<Block> inner_;
    uint32_t num_controls_;
    std::vector<bool> ctrl_state_;
};

}  // namespace qarpx
