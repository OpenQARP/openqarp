#pragma once

#include "../block/block.h"
#include "decompositions.h"
#include "gateset.h"

#include <cstddef>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace qarpx {

/// Optimization level for `Transpiler::transpile_and_optimize`.
enum class OptLevel : uint8_t {
    O0,  ///< Transpile only (no optimization passes).
    O1,  ///< + wire-adjacent cancel/merge (CircuitDAG) + single-qubit fusion.
         ///< Default.  Cancels on wire adjacency rather than textual
         ///< adjacency, and never reorders surviving commands.
    O2,  ///< + commutation-aware cancel/merge (matrix-verified commutation
         ///< table, bounded lookahead window).  Opt-in.  Rewrites never
         ///< physically reorder surviving commands.
};

/// Transpiles command sequences to a target gate set.
///
/// Greedy single-pass decomposition: each command not in the target gate set
/// is replaced by its decomposition, applied iteratively until all commands
/// are in the target set (handles cascading decompositions).
class Transpiler {
public:
    /// Seeds the rule table from `decompositions_for(target)`: the built-in
    /// rules plus whatever `target.rules` names, so a target's overrides
    /// travel with it and need no separate install call.
    explicit Transpiler(GateSet target);

    /// Transpile a flat command sequence to the target gate set.
    [[nodiscard]] std::vector<Command> transpile(
        const std::vector<Command>& input) const;

    /// Transpile, then `optimize_in_target(level)`.
    [[nodiscard]] std::vector<Command> transpile_and_optimize(
        const std::vector<Command>& input,
        OptLevel level = OptLevel::O1) const;

    /// Optimize an already in-target sequence at `level` without leaving the
    /// target: wire-adjacent cancellation on the CircuitDAG (O1), commutation-
    /// aware cancellation (O2), then single-qubit fusion **only if the target
    /// admits `Custom`** — fusion emits dense `Custom` gates, which a target
    /// without them cannot run.  The output is verified in-target (§16 rebase
    /// totality) and `capability_error` names any residual — at every level:
    /// `O0` returns the input unchanged once it has passed the same check.
    [[nodiscard]] std::vector<Command> optimize_in_target(
        const std::vector<Command>& input,
        OptLevel level = OptLevel::O1) const;

    /// Transpile multiple independent command sequences in parallel.
    /// Uses a thread pool internally. n_threads=0 means hardware_concurrency.
    [[nodiscard]] std::vector<std::vector<Command>> transpile_parallel(
        const std::vector<std::vector<Command>>& blocks,
        std::size_t n_threads = 0) const;

    /// Register a custom decomposition for a gate type.
    /// Overrides the built-in decomposition if one exists.
    void register_decomposition(GateType gate, DecompositionFn fn);

    /// Install the Rz-only-target overrides (Rx/Ry via Rz+Clifford).  A
    /// Transpiler built from `clifford_t_rz_gateset()` already carries them; this is
    /// idempotent and kept for callers targeting such a set without the tag.
    void install_clifford_t_rz_decompositions();

    /// Access the target gate set.
    [[nodiscard]] const GateSet& target() const { return target_; }

    /// BFS the decomposition graph from every GateType and return the set of
    /// gates that cannot be reduced into the target.  Empty result means the
    /// rule table is closed under the target gate set.
    ///
    /// Meta gates (Barrier, Measure, Reset, Custom) are exempt — they pass
    /// through transpilation unchanged regardless of the target.
    [[nodiscard]] std::unordered_set<GateType> unreachable_gates() const;

    /// Throws capability_error if any GateType cannot be reduced to the
    /// target.  The error message lists every unreachable gate, sorted by
    /// name.  Not called
    /// at construction on purpose: a partial target (clifford_t) is valid for
    /// circuits already inside it, and rules may be registered after
    /// construction.  `transpile()` reports the same list when it fails.
    void verify_closure() const;

    // ── Topology cache ──
    //
    // Opt-in cache keyed on a canonical-form topology hash (qubit-relocation
    // and symbol-rename invariant).  When enabled, `transpile()` checks the
    // cache for a previously-transpiled canonical form; on hit it rebinds the
    // cached output back to the query's qubits/symbols.  Useful when many
    // similar sub-circuits (e.g., HEA layers, Trotter steps, BE inside both
    // Qubitization and QSVT) share a topology.
    //
    // Disabled by default — flip via `enable_topology_cache(true)`.  All
    // existing transpiler tests run with cache off; correctness with cache on
    // is locked in by `test_transpiler_topology_cache.cpp`.

    void enable_topology_cache(bool on) const { cache_enabled_ = on; }
    [[nodiscard]] bool is_topology_cache_enabled() const { return cache_enabled_; }

    /// Drop all cached entries.
    void clear_topology_cache() const;

    struct CacheStats {
        std::size_t hits = 0;
        std::size_t misses = 0;
        std::size_t size = 0;
    };
    [[nodiscard]] CacheStats topology_cache_stats() const;

private:
    /// Direct transpile without the topology cache layer — the original
    /// greedy-decomposition implementation.  Public `transpile()` routes
    /// through the cache when enabled, falling back to this otherwise.
    [[nodiscard]] std::vector<Command> transpile_uncached_(
        const std::vector<Command>& input) const;

    /// Throw capability_error naming (sorted) every non-meta gate of
    /// `commands` outside the target.  `context` prefixes the message.
    void require_in_target_(const std::vector<Command>& commands,
                            const std::string& context) const;

    GateSet target_;
    std::unordered_map<GateType, DecompositionFn> decompositions_;

    /// Maximum decomposition depth to prevent infinite loops.
    static constexpr int kMaxDecomposeDepth = 10;

    // ── Cache state — `mutable` so const `transpile()` can populate. ──
    mutable bool cache_enabled_ = false;
    mutable std::unordered_map<uint64_t, std::vector<Command>> cache_;
    mutable std::size_t cache_hits_ = 0;
    mutable std::size_t cache_misses_ = 0;
};

/// Meta gates that flow through transpilation untouched.  These are not
/// subject to gateset closure: simulators dispatch them directly (or as
/// no-ops) regardless of the rule table.
[[nodiscard]] bool is_meta_gate(GateType g);

}  // namespace qarpx
