#include "qarpx/block/conditional_block.h"

#include <algorithm>
#include <stdexcept>

namespace qarpx {

namespace {

/// Build a synthetic BranchBegin command carrying the AND-of-condition tuple
/// in its condition_bits / condition_values fields.  No qubits, no params.
Command make_branch_begin(const std::vector<uint32_t>& cbits,
                          const std::vector<bool>&     values)
{
    Command c;
    c.gate = GateType::BranchBegin;
    for (auto b : cbits)  c.condition_bits.push_back(b);
    for (bool v : values) c.condition_values.push_back(v);
    return c;
}

Command make_branch_else() {
    Command c;
    c.gate = GateType::BranchElse;
    return c;
}

Command make_branch_end() {
    Command c;
    c.gate = GateType::BranchEnd;
    return c;
}

}  // namespace

ConditionalBlock::ConditionalBlock(std::vector<uint32_t>  cbits,
                                   std::vector<bool>      values,
                                   ref<Block>             then_body,
                                   ref<Block>             else_body,
                                   const std::string&     nm)
    : cbits_(std::move(cbits))
    , values_(std::move(values))
    , then_(std::move(then_body))
    , else_(std::move(else_body))
{
    if (cbits_.empty())
        throw std::invalid_argument(
            "ConditionalBlock: empty condition would be unconditional — "
            "drop the wrapper instead.");
    if (cbits_.size() != values_.size())
        throw std::invalid_argument(
            "ConditionalBlock: cbits and values must be the same length");
    if (!then_ && !else_)
        throw std::invalid_argument(
            "ConditionalBlock: at least one of then_body / else_body must be non-null");

    name = nm;

    const uint32_t then_nq = then_ ? then_->n_qubits : 0;
    const uint32_t else_nq = else_ ? else_->n_qubits : 0;
    n_qubits = std::max(then_nq, else_nq);

    // Width of the cbit space the bodies expect — distinct from the cbits we
    // condition on.  Take the larger of any body's n_cbits and any condition
    // index we read.
    const uint32_t then_nc = then_ ? then_->n_cbits : 0;
    const uint32_t else_nc = else_ ? else_->n_cbits : 0;
    uint32_t needed = std::max(then_nc, else_nc);
    for (auto c : cbits_) needed = std::max(needed, c + 1);
    n_cbits = needed;
}

void ConditionalBlock::build() {
    if (built_) return;
    if (then_ && !then_->is_built()) then_->build();
    if (else_ && !else_->is_built()) else_->build();
    built_ = true;
    // commands_ stays empty; flatten() emits the branch-marker frame around
    // the bodies.
}

std::vector<Command> ConditionalBlock::flatten() const {
    if (!built_)
        throw std::runtime_error("ConditionalBlock::flatten() called before build()");

    std::vector<Command> cmds;
    cmds.push_back(make_branch_begin(cbits_, values_));

    if (then_) {
        auto then_cmds = then_->flatten();
        cmds.insert(cmds.end(),
                    std::make_move_iterator(then_cmds.begin()),
                    std::make_move_iterator(then_cmds.end()));
    }
    if (else_) {
        cmds.push_back(make_branch_else());
        auto else_cmds = else_->flatten();
        cmds.insert(cmds.end(),
                    std::make_move_iterator(else_cmds.begin()),
                    std::make_move_iterator(else_cmds.end()));
    }
    cmds.push_back(make_branch_end());

    if (target_qubits.has_value()) {
        const auto& mapping = target_qubits.value();
        for (auto& cmd : cmds) cmd = cmd.remap_qubits(mapping);
    }
    if (target_cbits.has_value())
        cbit_remap_in_place(cmds, target_cbits.value());

    return cmds;
}

ref<Block> ConditionalBlock::dagger() const {
    if (!built_)
        throw std::runtime_error("ConditionalBlock::dagger() called before build()");

    ref<Block> dag_then = then_ ? then_->dagger() : ref<Block>();
    ref<Block> dag_else = else_ ? else_->dagger() : ref<Block>();
    ref<Block> result(new ConditionalBlock(
        cbits_, values_,
        std::move(dag_then), std::move(dag_else),
        name + "_dag"));
    result->target_qubits = target_qubits;
    result->target_cbits  = target_cbits;
    result->n_controls    = n_controls;
    result->control_state = control_state;
    result->build();
    return result;
}

ref<Block> ConditionalBlock::set_symbols(
    const std::unordered_map<std::string, double>& values) const
{
    if (!built_)
        throw std::runtime_error("ConditionalBlock::set_symbols() called before build()");
    ref<Block> new_then = then_ ? then_->set_symbols(values) : ref<Block>();
    ref<Block> new_else = else_ ? else_->set_symbols(values) : ref<Block>();
    ref<Block> result(new ConditionalBlock(
        cbits_, values_,
        std::move(new_then), std::move(new_else),
        name));
    result->target_qubits = target_qubits;
    result->target_cbits  = target_cbits;
    result->n_controls    = n_controls;
    result->control_state = control_state;
    result->build();
    return result;
}

ref<Block> ConditionalBlock::replace_symbols(
    const std::unordered_map<std::string, std::string>& mapping) const
{
    if (!built_)
        throw std::runtime_error(
            "ConditionalBlock::replace_symbols() called before build()");
    ref<Block> new_then = then_ ? then_->replace_symbols(mapping) : ref<Block>();
    ref<Block> new_else = else_ ? else_->replace_symbols(mapping) : ref<Block>();
    ref<Block> result(new ConditionalBlock(
        cbits_, values_,
        std::move(new_then), std::move(new_else),
        name));
    result->target_qubits = target_qubits;
    result->target_cbits  = target_cbits;
    result->n_controls    = n_controls;
    result->control_state = control_state;
    result->build();
    return result;
}

}  // namespace qarpx
