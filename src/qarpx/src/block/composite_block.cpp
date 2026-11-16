#include "qarpx/block/composite_block.h"

#include <algorithm>
#include <set>
#include <stdexcept>

namespace qarpx {

CompositeBlock::CompositeBlock(
    std::vector<ref<Block>> children,
    uint32_t nq,
    const std::string& nm)
    : children_(std::move(children))
{
    n_qubits = nq;
    name = nm;
}

void CompositeBlock::build() {
    // Build any children that haven't been built yet
    for (auto& child : children_) {
        if (!child->is_built()) {
            child->build();
        }
    }

    resolve_qubits();
    resolve_cbits();
    built_ = true;
}

void CompositeBlock::resolve_qubits() {
    if (n_qubits > 0) return;  // explicitly set

    uint32_t max_qubit = 0;
    for (const auto& child : children_) {
        if (child->target_qubits.has_value()) {
            for (auto q : child->target_qubits.value()) {
                max_qubit = std::max(max_qubit, q + 1);
            }
        } else {
            max_qubit = std::max(max_qubit, child->n_qubits);
        }
    }
    n_qubits = std::max(max_qubit, uint32_t(1));
}

void CompositeBlock::resolve_cbits() {
    if (n_cbits > 0) return;  // explicitly set

    // Sum sequentially-allocated child cbits, OR take the max of any
    // explicit target_cbits overrides.  Mirrors resolve_qubits().
    uint32_t total = 0;
    bool has_overrides = false;
    uint32_t max_override = 0;
    for (const auto& child : children_) {
        if (child->target_cbits.has_value()) {
            has_overrides = true;
            for (auto c : child->target_cbits.value())
                max_override = std::max(max_override, c + 1);
        } else {
            total += child->n_cbits;
        }
    }
    n_cbits = std::max(total, has_overrides ? max_override : 0u);
}

std::vector<Command> CompositeBlock::flatten() const {
    if (!built_) {
        throw std::runtime_error("CompositeBlock::flatten() called before build()");
    }

    std::vector<Command> result;

    // Pre-compute total size for a single allocation
    std::size_t total = 0;
    for (const auto& child : children_) {
        total += child->commands().size();
    }
    result.reserve(total);

    // Sequential cbit offset: each child without an explicit target_cbits
    // gets [cbit_offset .. cbit_offset + child.n_cbits) in the parent space.
    uint32_t cbit_offset = 0;

    for (const auto& child : children_) {
        auto child_cmds = child->flatten();

        // If the child didn't carry an explicit target_cbits override, push
        // its cbit indices into the parent's cbit space.  Without this, two
        // sibling MeasureBlocks both writing to local cbit 0 would alias.
        if (!child->target_cbits.has_value() && child->n_cbits > 0) {
            std::vector<uint32_t> mapping(child->n_cbits);
            for (uint32_t i = 0; i < child->n_cbits; ++i)
                mapping[i] = cbit_offset + i;
            cbit_remap_in_place(child_cmds, mapping);
            cbit_offset += child->n_cbits;
        }

        result.insert(result.end(),
            std::make_move_iterator(child_cmds.begin()),
            std::make_move_iterator(child_cmds.end()));
    }

    // If this composite itself has a target_qubits mapping, apply it
    if (target_qubits.has_value()) {
        const auto& mapping = target_qubits.value();
        for (auto& cmd : result) {
            cmd = cmd.remap_qubits(mapping);
        }
    }

    if (target_cbits.has_value())
        cbit_remap_in_place(result, target_cbits.value());

    return result;
}

std::vector<uint32_t> CompositeBlock::uninitialised_condition_cbits() const {
    if (!built_) {
        throw std::runtime_error(
            "CompositeBlock::uninitialised_condition_cbits called before build()");
    }
    return qarpx::uninitialised_condition_cbits(flatten());
}

void CompositeBlock::add_child(ref<Block> child) {
    children_.push_back(std::move(child));
    built_ = false;  // need to rebuild
}

ref<Block> CompositeBlock::set_symbols(
    const std::unordered_map<std::string, double>& values) const
{
    if (!built_)
        throw std::runtime_error("CompositeBlock::set_symbols() called before build()");
    std::vector<ref<Block>> new_children;
    new_children.reserve(children_.size());
    for (const auto& c : children_)
        new_children.push_back(c->set_symbols(values));
    ref<Block> result(new CompositeBlock(std::move(new_children), n_qubits, name));
    result->n_cbits = n_cbits;
    result->target_qubits = target_qubits;
    result->target_cbits  = target_cbits;
    result->n_controls    = n_controls;
    result->control_state = control_state;
    result->build();
    return result;
}

ref<Block> CompositeBlock::replace_symbols(
    const std::unordered_map<std::string, std::string>& mapping) const
{
    if (!built_)
        throw std::runtime_error("CompositeBlock::replace_symbols() called before build()");
    std::vector<ref<Block>> new_children;
    new_children.reserve(children_.size());
    for (const auto& c : children_)
        new_children.push_back(c->replace_symbols(mapping));
    ref<Block> result(new CompositeBlock(std::move(new_children), n_qubits, name));
    result->n_cbits = n_cbits;
    result->target_qubits = target_qubits;
    result->target_cbits  = target_cbits;
    result->n_controls    = n_controls;
    result->control_state = control_state;
    result->build();
    return result;
}

ref<Block> CompositeBlock::dagger() const {
    if (!built_)
        throw std::runtime_error("CompositeBlock::dagger() called before build()");
    std::vector<ref<Block>> new_children;
    new_children.reserve(children_.size());
    // Reverse child order and dagger each.
    for (auto it = children_.rbegin(); it != children_.rend(); ++it)
        new_children.push_back((*it)->dagger());
    ref<Block> result(new CompositeBlock(
        std::move(new_children), n_qubits, name + "_dag"));
    result->n_cbits       = n_cbits;
    result->target_qubits = target_qubits;
    result->target_cbits  = target_cbits;
    result->n_controls    = n_controls;
    result->control_state = control_state;
    result->build();
    return result;
}

}  // namespace qarpx
