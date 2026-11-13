#pragma once

#include "router.h"

#include <vector>

namespace qarpx {

/// SABRE router: DAG front-layer routing
/// with an extended lookahead set, per-qubit decay to spread SWAPs, and —
/// when `opts.initial_mapping` is absent — an initial-mapping search in two
/// stages:
///   1. a zero-SWAP embedding of the interaction graph
///      (`find_perfect_layout`, undirected architectures only) — optimal by
///      construction when it exists;
///   2. otherwise a **trial portfolio**: identity, greedy-weighted, and
///      fixed pseudo-random seeds, each refined by iterated Li-Ding-Xie
///      reverse traversal and routed with per-trial randomized tie-breaking;
///      the fewest-SWAPs stream wins.  Trial 0 reproduces the legacy single
///      shot, so the portfolio is never worse than the old behaviour.  All
///      seeds are fixed constants — identical input yields identical output.
///
/// Same contract as `route()` with `RouterKind::Lite` — physical-qubit
/// commands + final logical→physical mapping, identical throw conditions
/// (3+q gates, conditional-2q-needing-SWAPs, 2q-needing-SWAPs inside branch
/// regions, wrong-direction non-CX asymmetric gates) — plus:
///   • Branch regions are scheduling barriers: the mapping is pinned across
///     the region; interior commands are translated wholesale through the
///     entry mapping.
///   • Deterministic: candidate SWAPs scored in `arch.edges` order with
///     strict-improvement tie-breaking.
///   • Portfolio guarantee: the Lite result is also computed and returned
///     when it uses strictly fewer SWAPs — Sabre is never worse than Lite
///     by construction (some topologies favour the greedy sweep).
///
/// Prefer calling `route()` with `RoutingOptions::router = RouterKind::Sabre`;
/// this symbol is the implementation entry point.
[[nodiscard]] RoutingResult sabre_route(const std::vector<Command>& commands,
                                        const RoutingOptions&        opts);

}  // namespace qarpx
