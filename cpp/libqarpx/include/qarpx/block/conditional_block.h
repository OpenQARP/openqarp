#pragma once

#include "block.h"

#include <memory>
#include <string>
#include <vector>

namespace qarpx {

/// Block-level classical conditional: optionally a then-body, optionally an
/// else-body, gated by an AND of (cbit == value) pairs.
///
/// Encoding: `flatten()` emits explicit branch markers in the flat command
/// list, so the simulator dispatches at the block level (one AND check per
/// shot) rather than per-command:
///
///     BranchBegin(condition_bits=cbits, condition_values=values)
///     <then_body's flattened commands>
///     BranchElse                                ; only if else_body != null
///     <else_body's flattened commands>
///     BranchEnd
///
/// This sidesteps the AND-only condition format's inability to express the
/// negation of a multi-bit AND (DeMorgan would need OR), and maps directly
/// to OpenQASM 3 `if (c == v) { ... } else { ... }` for the emitter.
///
/// Constraints checked at construction:
///   * `cbits.size() == values.size()` and non-empty.
///   * At least one of `then_body`, `else_body` is non-null.
class ConditionalBlock : public Block {
public:
    ConditionalBlock(
        std::vector<uint32_t>  cbits,
        std::vector<bool>      values,
        ref<Block>             then_body,
        ref<Block>             else_body = nullptr,
        const std::string&     name = "Conditional");

    void build() override;

    [[nodiscard]] std::vector<Command> flatten() const override;

    /// dagger(ConditionalBlock(c, v, then, else))
    ///   = ConditionalBlock(c, v, then.dagger(), else.dagger()).
    /// Bodies containing Measure/Reset cannot be daggered (non-unitary); the
    /// underlying block's dagger() is responsible for that.
    [[nodiscard]] ref<Block> dagger() const override;
    [[nodiscard]] ref<Block> set_symbols(
        const std::unordered_map<std::string, double>& values) const override;
    [[nodiscard]] ref<Block> replace_symbols(
        const std::unordered_map<std::string, std::string>& mapping) const override;

    [[nodiscard]] const ref<Block>& then_body() const { return then_; }
    [[nodiscard]] const ref<Block>& else_body() const { return else_; }
    [[nodiscard]] const std::vector<uint32_t>&  condition_cbits()  const { return cbits_; }
    [[nodiscard]] const std::vector<bool>&      condition_values() const { return values_; }

private:
    std::vector<uint32_t>  cbits_;
    std::vector<bool>      values_;
    ref<Block>             then_;
    ref<Block>             else_;
};

}  // namespace qarpx
