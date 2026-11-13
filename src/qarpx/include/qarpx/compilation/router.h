#pragma once

#include "../core/command.h"
#include "../device/architecture.h"
#include "../simulator/sampling_result.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace qarpx {

/// Router implementation selector.
enum class RouterKind : uint8_t {
    Lite,   ///< Greedy shortest-path sweep.
    Sabre,  ///< DAG front-layer SABRE: lookahead + decay heuristic +
            ///< initial-mapping search — a zero-SWAP embedding of the
            ///< interaction graph when one exists, else a best-of-N trial
            ///< portfolio over seeds and tie-break orders (both skipped when
            ///< an explicit `initial_mapping` is supplied), with a portfolio
            ///< guarantee of never using more SWAPs than Lite.  Deterministic
            ///< (fixed seeds).  Default.
};

struct RoutingOptions {
    Architecture                          arch;
    bool                                  directedness = false;
    /// Optional initial logical→physical mapping; identity if absent.
    /// Must have size `arch.n_qubits` and be a permutation if supplied.
    std::optional<std::vector<uint32_t>>  initial_mapping;
    RouterKind                            router = RouterKind::Sabre;
};

struct RoutingResult {
    /// Routed commands acting on *physical* qubit indices (after the
    /// initial mapping, intervening SWAPs, and any H-conjugation for
    /// asymmetric-direction CX flips).
    std::vector<Command>   commands;

    /// Initial logical → physical placement: `initial_logical_to_physical[l]
    /// = p` means logical qubit l sits on physical wire p before the first
    /// gate — the pinned `initial_mapping`, or the search's choice; identity
    /// when the router places nothing, never omitted.  An injected initial
    /// state must be permuted by this map, not by the final one (§14).
    std::vector<uint32_t>  initial_logical_to_physical;

    /// Final logical → physical mapping: `final_logical_to_physical[l] = p`
    /// means logical qubit l ended up at physical qubit p.  Pass into
    /// `reindex_sampling_result` to convert per-physical sampling counts
    /// back into logical-qubit bit-string order.
    std::vector<uint32_t>  final_logical_to_physical;
};

/// Correctness-first routing entry point.  Dispatches on `opts.router`:
/// `Sabre` (the default) or `Lite` — see `RouterKind` above.
///
/// `Lite` sweeps forward over `commands`; for each 2-qubit gate:
///   • If the current mapping already satisfies the architecture (and
///     direction, when `directedness=true`), emit the gate as-is —
///     applying H-conjugation for CX when only the reversed edge is
///     available.
///   • Otherwise, insert SWAPs along the shortest path on the coupling
///     graph until the two logical qubits land on a satisfiable edge.
///
/// Behaviour & limitations:
///   • Supports 0-qubit, 1-qubit, and 2-qubit gates.  3+-qubit gates raise
///     — rebase to a 1q/2q gate set first via `qx::Transpiler`.
///   • Conditional 2-qubit gates (non-empty `condition_bits`) that *need*
///     routing raise: emitting unconditional SWAPs around a conditional
///     gate would corrupt state when the condition is false.
///   • Asymmetric 2q gates other than CX (CY, CRx, CRy, CRz, CP, CU) cannot
///     be H-conjugated by the router.  On a directed edge with the
///     wrong orientation they raise — rebase to CX first.
///   • Lite: initial mapping defaults to identity (no search).  Sabre
///     searches for one when none is supplied — perfect-layout embedding,
///     else a best-of-N trial portfolio (see `RouterKind`).
///   • SWAPs introduced by the router may not be in the target gate set;
///     `qx::compile_for_device` runs a second `Transpiler` pass to
///     decompose them.
///
/// Throws on:
///   • Qubit indices ≥ `arch.n_qubits`.
///   • Disconnected components (no path between two logical qubits' physical positions).
///   • Non-CX asymmetric 2q gate with wrong direction on a directed architecture.
///   • 3+-qubit gates.
///   • Conditional 2q gates that require SWAPs.
///   • 2q gates INSIDE a BranchBegin…BranchEnd region that require SWAPs —
///     the region's condition lives on the markers, not on per-command
///     residue, so unconditional SWAPs inside the conditionally-executed
///     body would desynchronize the mapping when the branch is not taken.
[[nodiscard]] RoutingResult route(const std::vector<Command>& commands,
                                  const RoutingOptions&        opts);

/// Reindex a SamplingResult from per-physical-qubit bit ordering back into
/// per-logical-qubit bit ordering using the mapping returned by `route`.
///
/// For each physical bit string `b_phys`, the corresponding logical bit
/// string `b_logical` has bit `l` set iff bit `final_logical_to_physical[l]` of
/// `b_phys` is set.  `cbit_history` is left untouched: classical bits are
/// a separate index space owned by the user.
[[nodiscard]] SamplingResult reindex_sampling_result(
    const SamplingResult&            physical,
    const std::vector<uint32_t>&     final_logical_to_physical);

}  // namespace qarpx
