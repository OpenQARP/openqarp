#include "qarpx/transpiler/transpiler.h"
#include "qarpx/transpiler/fusion.h"
#include "qarpx/parallel/thread_pool.h"
#include "qarpx/core/canonical.h"
#include "qarpx/dag/circuit_dag.h"
#include "qarpx/dag/passes.h"

#include "qarpx/core/errors.h"

#include <algorithm>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>

namespace qarpx {

bool is_meta_gate(GateType g) {
    return g == GateType::Barrier
        || g == GateType::Measure
        || g == GateType::Reset
        || g == GateType::Custom
        || g == GateType::BranchBegin
        || g == GateType::BranchElse
        || g == GateType::BranchEnd;
}

namespace {
/// Diagnostics list gates by name in a stable order (the sets are unordered).
std::string sorted_gate_names(const std::unordered_set<GateType>& gates) {
    std::set<std::string> names;
    for (GateType g : gates) names.insert(std::string(gate_name(g)));
    std::string out;
    for (const auto& n : names) {
        if (!out.empty()) out += ", ";
        out += n;
    }
    return out;
}
}  // namespace

Transpiler::Transpiler(GateSet target)
    : target_(std::move(target))
    , decompositions_(decompositions_for(target_))
{}

void Transpiler::clear_topology_cache() const {
    cache_.clear();
    cache_hits_ = 0;
    cache_misses_ = 0;
}

Transpiler::CacheStats Transpiler::topology_cache_stats() const {
    return {cache_hits_, cache_misses_, cache_.size()};
}

std::vector<Command> Transpiler::transpile(const std::vector<Command>& input) const {
    if (!cache_enabled_ || input.empty()) {
        return transpile_uncached_(input);
    }

    // Canonicalize the query: dummify qubits / cbits / symbols.
    auto cf = canonical::canonicalize(input);
    const uint64_t h = canonical::topology_hash(cf);

    if (auto it = cache_.find(h); it != cache_.end()) {
        ++cache_hits_;
        // Cached output is in canonical form; rebind to the query's bindings.
        canonical::CanonicalForm cached{
            it->second, cf.qubit_binding, cf.cbit_binding, cf.symbol_binding};
        return canonical::rebind(cached);
    }

    ++cache_misses_;
    // Transpile the dummified form so the cached output is also canonical.
    // Decomposition / fusion / identity passes are relabel-equivariant
    // (they don't peek at concrete qubit indices), so running them on
    // dummified commands gives the same shape as running them on real ones.
    auto transpiled_dummy = transpile_uncached_(cf.commands);
    cache_.emplace(h, transpiled_dummy);
    canonical::CanonicalForm out{
        std::move(transpiled_dummy),
        cf.qubit_binding, cf.cbit_binding, cf.symbol_binding};
    return canonical::rebind(out);
}

std::vector<Command> Transpiler::transpile_uncached_(const std::vector<Command>& input) const {
    std::vector<Command> result;
    result.reserve(input.size());

    // Greedy decomposition: iterate commands, decompose any that aren't
    // in the target gate set.  Repeat until stable (handles cascading
    // decompositions, e.g., RXX -> RZZ -> CX+Rz).
    std::vector<Command> current = input;

    bool converged = false;
    for (int depth = 0; depth < kMaxDecomposeDepth; ++depth) {
        std::vector<Command> next;
        next.reserve(current.size());
        bool changed = false;

        for (const auto& cmd : current) {
            // Meta gates (Barrier, Measure, Reset) flow through
            // transpilation unchanged — simulators handle them directly.
            // Custom flows through ONLY when the target admits it; otherwise
            // it falls to the ZYZ decomposition rule (a rebase is total).
            if (is_meta_gate(cmd.gate)
                && !(cmd.gate == GateType::Custom
                     && !target_.contains(GateType::Custom))) {
                next.push_back(cmd);
                continue;
            }

            if (target_.contains(cmd.gate)) {
                next.push_back(cmd);
                continue;
            }

            // Look up decomposition
            auto it = decompositions_.find(cmd.gate);
            if (it == decompositions_.end()) {
                // Name every gap of the target, not just the one hit, so a
                // caller fixes the rule table in one round.
                std::string msg = "Transpiler: no decomposition for gate " +
                    std::string(gate_name(cmd.gate)) +
                    " to target gate set '" + target_.name + "'";
                const auto missing = unreachable_gates();
                if (!missing.empty())
                    msg += "; unreachable gates for this target: " + sorted_gate_names(missing);
                throw capability_error(msg, cmd);
            }

            auto decomposed = it->second(cmd);
            // Decomposition rules don't know about classical conditions; we
            // propagate the source command's condition tuple onto every
            // produced sub-command so the cascade stays semantically correct
            // for conditional gates.  AND-merging with any condition the rule
            // itself emitted: if a rule ever needs to introduce its own
            // condition (none do today), the combined condition is the
            // conjunction.  For now rules emit unconditional commands, so
            // this is a straight assignment.
            if (!cmd.condition_bits.empty()) {
                for (auto& sub : decomposed) {
                    if (sub.condition_bits.empty()) {
                        sub.condition_bits   = cmd.condition_bits;
                        sub.condition_values = cmd.condition_values;
                    } else {
                        for (std::size_t i = 0; i < cmd.condition_bits.size(); ++i) {
                            sub.condition_bits.push_back(cmd.condition_bits[i]);
                            sub.condition_values.push_back(cmd.condition_values[i]);
                        }
                    }
                }
            }
            next.insert(next.end(), decomposed.begin(), decomposed.end());
            changed = true;
        }

        current = std::move(next);
        if (!changed) { converged = true; break; }
    }

    // Output validation: catch the silent-leak case where the cascade hit
    // kMaxDecomposeDepth without converging.  Also defensively catches a
    // decomposition that produces a residual non-target gate even after a
    // notionally-converged pass (e.g., a registered rule that returns a gate
    // outside the closure of the rule table).
    require_in_target_(current, converged
        ? "Transpiler: output contains gates outside target gate set"
        : "Transpiler: output contains gates outside target gate set (cascade "
          "did not converge within " + std::to_string(kMaxDecomposeDepth) + " passes)");

    return current;
}

void Transpiler::require_in_target_(const std::vector<Command>& commands,
                                    const std::string& context) const {
    std::unordered_set<GateType> residual;
    const Command* first_bad = nullptr;
    for (const auto& cmd : commands) {
        // is_meta_gate lists Custom, but Custom is meta only when the target
        // admits it (§16 totality): test that before the meta exemption.
        const bool custom_gap = cmd.gate == GateType::Custom
            && !target_.contains(GateType::Custom);
        if (!custom_gap && (is_meta_gate(cmd.gate) || target_.contains(cmd.gate))) continue;
        if (!first_bad) first_bad = &cmd;
        residual.insert(cmd.gate);
    }
    if (residual.empty()) return;
    throw capability_error(
        context + " '" + target_.name + "': " + sorted_gate_names(residual),
        *first_bad);
}

std::vector<Command> Transpiler::transpile_and_optimize(
    const std::vector<Command>& input, OptLevel level) const
{
    return optimize_in_target(transpile(input), level);
}

std::vector<Command> Transpiler::optimize_in_target(
    const std::vector<Command>& input, OptLevel level) const
{
    // O0 still verifies: the entry point promises an in-target output at
    // every level, and an out-of-target input is the caller's mistake.
    if (level == OptLevel::O0) {
        require_in_target_(input, "Transpiler: optimized output contains gates outside target gate set");
        return input;
    }

    auto dag = CircuitDAG::from_commands(input);
    dag_passes::cancel_wire_adjacent(dag);
    if (level >= OptLevel::O2)
        dag_passes::commute_and_cancel(dag);
    auto result = dag.to_commands();
    // Fusion's product is a dense Custom gate; only a target that admits
    // Custom can run one, so elsewhere the pass is skipped, not undone.
    if (target_.contains(GateType::Custom))
        result = fuse_single_qubit_gates(result);
    require_in_target_(result, "Transpiler: optimized output contains gates outside target gate set");
    return result;
}

std::vector<std::vector<Command>> Transpiler::transpile_parallel(
    const std::vector<std::vector<Command>>& blocks,
    std::size_t n_threads) const
{
    if (blocks.empty()) return {};
    if (blocks.size() == 1) return {transpile(blocks[0])};

    auto& pool = global_thread_pool();
    std::vector<std::future<std::vector<Command>>> futures;
    futures.reserve(blocks.size());

    for (const auto& block : blocks) {
        futures.push_back(pool.submit([this, &block]() {
            return transpile(block);
        }));
    }

    std::vector<std::vector<Command>> results;
    results.reserve(blocks.size());
    for (auto& f : futures) {
        results.push_back(f.get());
    }

    return results;
}

void Transpiler::register_decomposition(GateType gate, DecompositionFn fn) {
    decompositions_[gate] = std::move(fn);
}

void Transpiler::install_clifford_t_rz_decompositions() {
    for (auto& [gate, fn] : clifford_t_rz_decompositions()) {
        decompositions_[gate] = fn;
    }
}

// ── Closure verification ──────────────────────────────────────────────────────

namespace {

/// Build a probe Command for the given gate that exercises its decomposition.
/// Uses sequential qubit indices [0..n-1] (5 qubits for variadic MCZ: width
/// >= 4 engages the generic mc_phase/mcx_dirty cascade, not just the width-3
/// H·CCX·H special case) and symbolic placeholder parameters.
Command make_probe(GateType g) {
    const uint8_t n_qubits = (g == GateType::MCZ)
        ? 5                          // reach the Barenco cascade branch
        : gate_num_qubits(g);
    const uint8_t n_params = gate_num_params(g);

    Command probe;
    probe.gate = g;
    for (uint8_t i = 0; i < n_qubits; ++i)
        probe.qubits.push_back(i);
    for (uint8_t i = 0; i < n_params; ++i)
        probe.params.push_back(Param::symbol("_probe_" + std::to_string(i)));
    return probe;
}

}  // namespace

std::unordered_set<GateType> Transpiler::unreachable_gates() const {
    std::unordered_set<GateType> unreachable;
    std::unordered_set<GateType> visited;

    const auto n_types = static_cast<uint16_t>(GateType::NUM_GATE_TYPES);

    // BFS from each GateType.  We share a single visited set across roots:
    // once a gate has been proven reachable (or unreachable) from one BFS,
    // the answer is the same regardless of where the cascade started.
    std::queue<GateType> frontier;

    auto enqueue_if_unvisited = [&](GateType g) {
        if (visited.insert(g).second) frontier.push(g);
    };

    for (uint16_t i = 0; i < n_types; ++i) {
        const auto root = static_cast<GateType>(i);
        if (root == GateType::NUM_GATE_TYPES) continue;
        if (is_meta_gate(root)) continue;
        enqueue_if_unvisited(root);
    }

    while (!frontier.empty()) {
        const GateType g = frontier.front();
        frontier.pop();

        if (is_meta_gate(g)) continue;
        if (target_.contains(g)) continue;

        auto it = decompositions_.find(g);
        if (it == decompositions_.end()) {
            unreachable.insert(g);
            continue;
        }

        // Probe the rule.  If a rule throws on a generic probe, surface it
        // as a closure failure rather than letting the BFS itself blow up.
        std::vector<Command> produced;
        try {
            produced = it->second(make_probe(g));
        } catch (const std::exception&) {
            unreachable.insert(g);
            continue;
        }

        for (const auto& cmd : produced)
            enqueue_if_unvisited(cmd.gate);
    }

    return unreachable;
}

void Transpiler::verify_closure() const {
    auto missing = unreachable_gates();
    if (missing.empty()) return;
    throw capability_error(
        "Transpiler: target gate set '" + target_.name +
        "' is not reachable from the rule table for: " + sorted_gate_names(missing));
}

}  // namespace qarpx
