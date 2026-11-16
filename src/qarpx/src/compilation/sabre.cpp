#include "qarpx/compilation/sabre.h"

#include "qarpx/compilation/perfect_layout.h"
#include "qarpx/dag/circuit_dag.h"
#include "routing_common.h"
#include "qarpx/core/errors.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

namespace qarpx {

namespace {

using routing_detail::Resolution;
using routing_detail::emit_h_conjugated_cx;
using routing_detail::resolve;
using routing_detail::resolve_initial_mapping;
using routing_detail::translated_copy;

using NodeId = CircuitDAG::NodeId;

// SABRE heuristic constants (Li, Ding, Xie 2019).  The extended set weights
// upcoming gates at half the front layer; decay discourages ping-ponging
// the same physical qubits.
constexpr double      kExtendedWeight  = 0.5;
constexpr std::size_t kExtendedWindow  = 20;
constexpr double      kDecayIncrement  = 0.001;

/// BFS all-pairs distance matrix.  kUnreachable for disconnected pairs.
constexpr uint32_t kUnreachable = std::numeric_limits<uint32_t>::max();

std::vector<std::vector<uint32_t>> distance_matrix(const Architecture& arch) {
    const uint32_t n = arch.n_qubits;
    std::vector<std::vector<uint32_t>> dist(
        n, std::vector<uint32_t>(n, kUnreachable));
    for (uint32_t s = 0; s < n; ++s) {
        dist[s][s] = 0;
        std::queue<uint32_t> bfs;
        bfs.push(s);
        while (!bfs.empty()) {
            const uint32_t u = bfs.front();
            bfs.pop();
            for (auto v : arch.neighbours(u)) {
                if (dist[s][v] != kUnreachable) continue;
                dist[s][v] = dist[s][u] + 1;
                bfs.push(v);
            }
        }
    }
    return dist;
}

/// Deterministic xorshift64 for tie-breaking and permutation seeds.  Not a
/// statistical RNG — it only has to decorrelate trials reproducibly.
struct TieRng {
    uint64_t state;
    uint64_t next() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }
};

/// The routing core: schedules the DAG onto the architecture from
/// `initial_l2p`, returning the final mapping.  When `out` is non-null the
/// physical command stream is emitted into it (the mapping-search passes
/// run with `out == nullptr`).
///
/// `tie_seed == 0` keeps the legacy tie-breaking (first minimum in
/// `arch.edges` order).  Nonzero seeds pick uniformly among candidate SWAPs
/// whose scores tie within epsilon — the same heuristic landscape explored
/// through a different, reproducible sample path (qiskit's `swap_trials`).
std::vector<uint32_t> sabre_core(
    const CircuitDAG&                          dag,
    const RoutingOptions&                      opts,
    const std::vector<std::vector<uint32_t>>&  dist,
    std::vector<uint32_t>                      l2p,
    std::vector<Command>*                      out,
    uint64_t                                   tie_seed = 0) {
    const uint32_t n = opts.arch.n_qubits;
    TieRng tie_rng{tie_seed ? tie_seed : 1};

    std::vector<uint32_t> p2l(n);
    for (uint32_t l = 0; l < n; ++l) p2l[l2p[l]] = l;

    // Kahn state over the DAG.
    std::vector<uint32_t> indegree(dag.n_slots(), 0);
    std::vector<NodeId> ready;
    for (NodeId id = 0; id < dag.n_slots(); ++id) {
        if (dag.is_removed(id)) continue;
        uint32_t deg = 0;
        for (auto w : dag.wires(id))
            if (dag.prev_on_wire(id, w) != CircuitDAG::kNone) ++deg;
        indegree[id] = deg;
        if (deg == 0) ready.push_back(id);
    }
    // Deterministic processing: smallest node id first.  `ready` is kept
    // sorted by inserting each newly ready node at its position (it is
    // small — the front layer plus pending 1q gates), never re-sorted.
    std::sort(ready.begin(), ready.end());
    auto push_ready = [&](NodeId s) {
        ready.insert(std::lower_bound(ready.begin(), ready.end(), s), s);
    };

    // Decay per physical qubit, reset to 1.0 whenever a gate is emitted.
    // A generation stamp makes the reset O(1): an entry whose stamp is stale
    // reads as 1.0.
    std::vector<double>   decay(n, 1.0);
    std::vector<uint32_t> decay_stamp(n, 0);
    uint32_t              decay_gen = 0;
    auto decay_of = [&](uint32_t p) { return decay_stamp[p] == decay_gen ? decay[p] : 1.0; };
    auto bump_decay = [&](uint32_t p) {
        if (decay_stamp[p] != decay_gen) { decay[p] = 1.0; decay_stamp[p] = decay_gen; }
        decay[p] += kDecayIncrement;
    };

    auto mark_done = [&](NodeId id) {
        for (auto w : dag.wires(id)) {
            const NodeId s = dag.next_on_wire(id, w);
            if (s != CircuitDAG::kNone && --indegree[s] == 0)
                push_ready(s);
        }
    };

    auto emit_region = [&](NodeId id) {
        // Mapping is pinned across the region: translate the interior
        // wholesale; a 2q gate not satisfiable under the entry mapping
        // cannot be fixed with SWAPs inside the conditional body.
        for (const auto& rc : dag.region_commands(id)) {
            if (rc.qubits.size() > 2) {
                throw capability_error(
                    "Router: gate '" + std::string(gate_name(rc.gate)) +
                    "' inside a branch region has " +
                    std::to_string(rc.qubits.size()) +
                    " qubits; rebase to a 1q/2q gate set first.");
            }
            if (rc.qubits.size() == 2) {
                const uint32_t p0 = l2p[rc.qubits[0]];
                const uint32_t p1 = l2p[rc.qubits[1]];
                const auto r = resolve(rc.gate, p0, p1, opts.arch,
                                       opts.directedness);
                if (r == Resolution::NEEDS_SWAPS) {
                    throw capability_error(
                        "Router: 2-qubit gate '" +
                        std::string(gate_name(rc.gate)) +
                        "' inside a BranchBegin/BranchEnd region requires "
                        "SWAPs.  The region executes conditionally, so "
                        "unconditional SWAPs inside it would desynchronize "
                        "the qubit mapping when the branch is not taken.  "
                        "Place the interacting qubits adjacently in the "
                        "input (or via initial_mapping).");
                }
                if (out) {
                    if (r == Resolution::EMIT_H_CONJUGATED) {
                        emit_h_conjugated_cx(rc, p0, p1, *out);
                    } else {
                        Command oc = translated_copy(rc, l2p);
                        out->push_back(std::move(oc));
                    }
                }
                continue;
            }
            if (out) out->push_back(translated_copy(rc, l2p));
        }
    };

    // Try to emit one ready node; returns true if something was emitted.
    // Stuck 2q gates stay in `ready` (they form the SABRE front layer).
    auto try_emit_one = [&]() -> bool {
        for (std::size_t i = 0; i < ready.size(); ++i) {
            const NodeId id = ready[i];

            if (dag.is_region(id)) {
                emit_region(id);
            } else {
                const Command& cmd = dag.command(id);
                for (auto q : cmd.qubits) {
                    if (q >= n)
                        throw capability_error(
                            "Router: command on gate '" +
                            std::string(gate_name(cmd.gate)) +
                            "' references logical qubit " + std::to_string(q) +
                            " >= n_qubits (" + std::to_string(n) + ").");
                }
                if (cmd.qubits.size() > 2) {
                    throw capability_error(
                        "Router: gate '" + std::string(gate_name(cmd.gate)) +
                        "' has " + std::to_string(cmd.qubits.size()) +
                        " qubits; the router supports 0/1/2-qubit gates only."
                        "  Rebase to a 1q/2q gate set first (qx::Transpiler).");
                }
                if (cmd.qubits.size() == 2) {
                    const uint32_t p0 = l2p[cmd.qubits[0]];
                    const uint32_t p1 = l2p[cmd.qubits[1]];
                    if (dist[p0][p1] == kUnreachable) {
                        throw capability_error(
                            "Router: physical qubits " + std::to_string(p0) +
                            " and " + std::to_string(p1) +
                            " are disconnected — cannot route.");
                    }
                    const auto r = resolve(cmd.gate, p0, p1, opts.arch,
                                           opts.directedness);
                    if (r == Resolution::NEEDS_SWAPS) continue;  // stuck
                    if (out) {
                        if (r == Resolution::EMIT_H_CONJUGATED) {
                            emit_h_conjugated_cx(cmd, p0, p1, *out);
                        } else {
                            Command oc = translated_copy(cmd, l2p);
                            out->push_back(std::move(oc));
                        }
                    }
                } else if (out) {
                    out->push_back(translated_copy(cmd, l2p));
                }
            }

            ready.erase(ready.begin() + static_cast<std::ptrdiff_t>(i));
            mark_done(id);
            // A gate was emitted: reset decay (standard SABRE round reset).
            ++decay_gen;
            return true;
        }
        return false;
    };

    // Collect up to kExtendedWindow upcoming 2q descendants of the front
    // layer (breadth-first over DAG successors) for the lookahead term.
    // Called once per SWAP: the visited set is a stamp buffer and the BFS
    // queue a reused vector with a head index, so no allocation and no
    // O(n_slots) clearing per call.  Visiting order is unchanged.
    std::vector<NodeId>   ext;
    std::vector<uint32_t> seen_stamp(dag.n_slots(), 0);
    uint32_t              seen_gen = 0;
    std::vector<NodeId>   bfs;
    auto extended_set = [&](const std::vector<NodeId>& front) -> const std::vector<NodeId>& {
        ext.clear();
        bfs.assign(front.begin(), front.end());
        ++seen_gen;
        for (std::size_t head = 0; head < bfs.size() && ext.size() < kExtendedWindow; ++head) {
            const NodeId u = bfs[head];
            for (auto w : dag.wires(u)) {
                const NodeId s = dag.next_on_wire(u, w);
                if (s == CircuitDAG::kNone || seen_stamp[s] == seen_gen) continue;
                seen_stamp[s] = seen_gen;
                if (!dag.is_region(s) && dag.command(s).qubits.size() == 2)
                    ext.push_back(s);
                bfs.push_back(s);
            }
        }
        return ext;
    };

    auto swap_and_update = [&](uint32_t pa, uint32_t pb) {
        if (out) {
            Command swap_cmd;
            swap_cmd.gate = GateType::SWAP;
            swap_cmd.qubits.push_back(pa);
            swap_cmd.qubits.push_back(pb);
            out->push_back(std::move(swap_cmd));
        }
        const uint32_t la = p2l[pa];
        const uint32_t lb = p2l[pb];
        std::swap(l2p[la], l2p[lb]);
        std::swap(p2l[pa], p2l[pb]);
        bump_decay(pa);
        bump_decay(pb);
    };

    std::size_t swaps_since_progress = 0;
    const std::size_t stall_bound = std::size_t{2} * n * n + 16;

    std::vector<NodeId>   front;
    std::vector<uint32_t> front_stamp(n, 0);  // physical qubits touched by the front
    uint32_t              front_gen = 0;

    // Incidence of the front ∪ extended-set gates on physical qubits, rebuilt
    // per SWAP; `touch_gen` invalidates a qubit's list without clearing it.
    struct Touch { uint32_t p0, p1; bool is_ext; };
    constexpr uint32_t kNoQubit = std::numeric_limits<uint32_t>::max();
    std::vector<std::vector<Touch>> touches(n);
    std::vector<uint32_t>           touch_stamp(n, 0);
    auto add_touch = [&](uint32_t p0, uint32_t p1, bool is_ext) {
        for (uint32_t p : {p0, p1}) {
            if (touch_stamp[p] != front_gen) { touches[p].clear(); touch_stamp[p] = front_gen; }
            touches[p].push_back({p0, p1, is_ext});
            if (p0 == p1) break;
        }
    };
    static const std::vector<Touch> kNoTouches;
    auto touches_of = [&](uint32_t p) -> const std::vector<Touch>& {
        return touch_stamp[p] == front_gen ? touches[p] : kNoTouches;
    };

    while (!ready.empty()) {
        if (try_emit_one()) {
            swaps_since_progress = 0;
            continue;
        }

        // Everything ready is a stuck 2q gate: the front layer.
        front = ready;

        // Guard rails the front shares with Lite.
        for (auto f : front) {
            const Command& cmd = dag.command(f);
            if (!cmd.condition_bits.empty()) {
                throw capability_error(
                    "Router: 2-qubit gate '" + std::string(gate_name(cmd.gate))
                    + "' has a classical condition AND requires SWAPs.  "
                    "Unconditional SWAPs around a conditional gate would "
                    "corrupt state when the condition is false.  Either "
                    "place the qubits adjacently in the input, or rewrite "
                    "the conditional with explicit BranchBegin/Else/End "
                    "scoping the SWAPs.");
            }
        }

        if (swaps_since_progress > stall_bound) {
            // Anti-livelock: force-route the first front gate greedily.
            const Command& cmd = dag.command(front[0]);
            uint32_t p0 = l2p[cmd.qubits[0]];
            const uint32_t p1 = l2p[cmd.qubits[1]];
            const auto path = opts.arch.shortest_path(p0, p1);
            if (path.size() < 2) {
                throw capability_error(
                    "Router: physical qubits " + std::to_string(p0) + " and "
                    + std::to_string(p1) + " are disconnected — cannot route.");
            }
            swap_and_update(path[0], path[1]);
            continue;
        }

        // Candidate SWAPs: architecture edges touching a front-layer qubit.
        const auto& ext = extended_set(front);
        double best_score = std::numeric_limits<double>::infinity();
        uint32_t best_a = 0, best_b = 0;
        bool found = false;

        // Score of a candidate SWAP (pa, pb) = mean distance over the front
        // plus kExtendedWeight × mean over the extended set, both under the
        // mapping with pa and pb exchanged.  Only gates touching pa or pb
        // change distance, so each sum is the unswapped base plus a delta
        // over those gates — every term is an integer-valued double, so this
        // equals the full re-summation bit for bit.
        ++front_gen;
        double front_base = 0.0, ext_base = 0.0;
        for (auto f : front) {
            const Command& c = dag.command(f);
            const uint32_t p0 = l2p[c.qubits[0]], p1 = l2p[c.qubits[1]];
            front_stamp[p0] = front_gen;
            front_stamp[p1] = front_gen;
            front_base += static_cast<double>(dist[p0][p1]);
            add_touch(p0, p1, /*is_ext=*/false);
        }
        for (auto g : ext) {
            const Command& c = dag.command(g);
            const uint32_t p0 = l2p[c.qubits[0]], p1 = l2p[c.qubits[1]];
            ext_base += static_cast<double>(dist[p0][p1]);
            add_touch(p0, p1, /*is_ext=*/true);
        }

        auto swapped_sums = [&](uint32_t pa, uint32_t pb, double& front_sum, double& ext_sum) {
            auto translate = [&](uint32_t p) {
                if (p == pa) return pb;
                if (p == pb) return pa;
                return p;
            };
            front_sum = front_base;
            ext_sum = ext_base;
            auto apply = [&](uint32_t p, uint32_t skip) {
                for (const auto& t : touches_of(p)) {
                    // A gate on both pa and pb is listed under each; count it once.
                    if (t.p0 == skip || t.p1 == skip) continue;
                    const double delta =
                        static_cast<double>(dist[translate(t.p0)][translate(t.p1)]) -
                        static_cast<double>(dist[t.p0][t.p1]);
                    (t.is_ext ? ext_sum : front_sum) += delta;
                }
            };
            apply(pa, kNoQubit);
            apply(pb, pa);
        };

        // Reservoir over epsilon-ties: with tie_seed == 0 only strict
        // improvement replaces the incumbent (legacy path, bit-for-bit);
        // otherwise each of k tied candidates survives with probability 1/k.
        std::size_t tie_count = 0;
        for (const auto& [ea, eb] : opts.arch.edges()) {
            if (ea >= n || eb >= n) continue;
            if (front_stamp[ea] != front_gen && front_stamp[eb] != front_gen) continue;
            double front_sum, ext_sum;
            swapped_sums(ea, eb, front_sum, ext_sum);
            double score = front_sum / static_cast<double>(front.size());
            if (!ext.empty()) {
                score += kExtendedWeight * ext_sum / static_cast<double>(ext.size());
            }
            score *= std::max(decay_of(ea), decay_of(eb));
            const double eps = 1e-9 * (1.0 + std::abs(best_score));
            if (score < best_score - eps || !found) {
                best_score = score;
                best_a = ea;
                best_b = eb;
                found = true;
                tie_count = 1;
            } else if (tie_seed != 0 && score < best_score + eps) {
                ++tie_count;
                if (tie_rng.next() % tie_count == 0) {
                    best_a = ea;
                    best_b = eb;
                }
            }
        }

        if (!found) {
            throw capability_error(
                "Router: no candidate SWAP available — architecture edge "
                "list does not touch the front layer (disconnected?).");
        }

        swap_and_update(best_a, best_b);
        ++swaps_since_progress;
    }

    return l2p;
}

/// Reverse the top-level items of a stream (single commands or whole
/// BranchBegin…BranchEnd spans, kept intact) — the geometry mirror used by
/// the initial-mapping search.  Gate parameters are irrelevant there.
std::vector<Command> reverse_stream_preserving_regions(
    const std::vector<Command>& cmds) {
    std::vector<std::pair<std::size_t, std::size_t>> items;  // [first, last]
    for (std::size_t i = 0; i < cmds.size(); ++i) {
        if (cmds[i].gate == GateType::BranchBegin) {
            int depth = 0;
            std::size_t end = i;
            for (std::size_t j = i; j < cmds.size(); ++j) {
                if (cmds[j].gate == GateType::BranchBegin)      ++depth;
                else if (cmds[j].gate == GateType::BranchEnd)   --depth;
                if (depth == 0) { end = j; break; }
            }
            items.emplace_back(i, end);
            i = end;
        } else {
            items.emplace_back(i, i);
        }
    }
    std::vector<Command> out;
    out.reserve(cmds.size());
    for (auto it = items.rbegin(); it != items.rend(); ++it) {
        for (std::size_t k = it->first; k <= it->second; ++k)
            out.push_back(cmds[k]);
    }
    return out;
}

}  // namespace

RoutingResult sabre_route(const std::vector<Command>& commands,
                          const RoutingOptions&        opts) {
    const uint32_t n = opts.arch.n_qubits;
    if (n == 0) {
        throw capability_error("Router: arch.n_qubits must be > 0");
    }

    std::vector<uint32_t> l2p, p2l;
    resolve_initial_mapping(opts.initial_mapping, n, l2p, p2l);

    // Range-check up front with the Lite router's exception contract
    // (CircuitDAG::from_commands would otherwise throw invalid_argument).
    for (const auto& cmd : commands) {
        for (auto q : cmd.qubits) {
            if (q >= n) {
                throw capability_error(
                    "Router: command on gate '" + std::string(gate_name(cmd.gate))
                    + "' references logical qubit " + std::to_string(q)
                    + " >= n_qubits (" + std::to_string(n) + ").");
            }
        }
    }

    const auto dist = distance_matrix(opts.arch);
    const auto dag = CircuitDAG::from_commands(commands, n, /*n_cbits=*/
        [&] {
            uint32_t nc = 0;
            for (const auto& c : commands) {
                for (auto b : c.cbits)          nc = std::max(nc, b + 1);
                for (auto b : c.condition_bits) nc = std::max(nc, b + 1);
            }
            return nc;
        }());

    // Initial-mapping search (only when the caller didn't pin one).
    //
    // First try for a layout that needs no SWAPs at all: if the interaction
    // graph embeds in the coupling map, that embedding is optimal and no
    // amount of local refinement can beat it.  The reverse traversal below
    // is a local search seeded from the identity, so without this its answer
    // depends on how the caller labelled its qubits — a 12-qubit ring on a
    // 3x4 grid moved between 0 and 10 SWAPs under relabelling alone, and on
    // 3x6 and 4x5 it never found the zero-SWAP embedding that exists.
    //
    // Then refine: forward pass improves the guess, reverse pass propagates
    // the final placement back to the start (Li-Ding-Xie reverse traversal).
    //
    // Directed architectures are left on the old path: the embedding search
    // is direction-blind, and `resolve` throws outright for a non-CX
    // asymmetric gate placed on a reversed edge, so a "perfect" layout could
    // turn a circuit that compiles today into one that does not.  Directed
    // devices therefore keep exactly their current behaviour until the search
    // learns about edge orientation.
    auto count_swaps = [](const std::vector<Command>& cmds) {
        std::size_t n_swaps = 0;
        for (const auto& c : cmds)
            if (c.gate == GateType::SWAP) ++n_swaps;
        return n_swaps;
    };

    RoutingResult result;

    std::optional<std::vector<uint32_t>> perfect;
    if (!opts.initial_mapping && !opts.directedness)
        perfect = find_perfect_layout(commands, opts.arch);

    if (opts.initial_mapping || perfect) {
        // Pinned or provably optimal mapping: single routed pass, legacy
        // tie-breaking, exactly the pre-portfolio behaviour.
        if (perfect) l2p = std::move(*perfect);
        result.initial_logical_to_physical = l2p;
        result.commands.reserve(commands.size());
        result.final_logical_to_physical =
            sabre_core(dag, opts, dist, std::move(l2p), &result.commands);
    } else {
        // Trial portfolio.  Best-of-N over
        // independent seeds with per-trial randomized tie-breaking: the same
        // Li-Ding-Xie heuristic explored along many reproducible sample
        // paths.  Trial 0 is bit-for-bit the old single shot, so the result
        // is never worse than before by construction.
        const auto reversed = reverse_stream_preserving_regions(commands);
        const auto rev_dag = CircuitDAG::from_commands(reversed);

        const std::size_t n_cmds = commands.size();
        const std::size_t n_trials = n_cmds <= 3000 ? 32 : (n_cmds <= 30000 ? 8 : 4);
        constexpr std::size_t kRefineRounds = 3;  // Li-Ding-Xie iterated traversal

        bool have_best = false;
        std::size_t best_swaps = 0;
        std::exception_ptr first_error;

        for (std::size_t trial = 0; trial < n_trials; ++trial) {
            std::vector<uint32_t> seed;
            uint64_t tie_seed = 0;
            if (trial == 0) {
                seed = l2p;  // identity; legacy tie-breaking
            } else if (trial == 1) {
                seed = greedy_weighted_layout(commands, opts.arch);
            } else {
                // Fisher-Yates from a fixed xorshift stream: deterministic
                // pseudo-random seeds, identical on every run and host.
                seed.resize(n);
                std::iota(seed.begin(), seed.end(), 0u);
                TieRng perm_rng{0x9E3779B97F4A7C15ull * (trial + 1)};
                for (uint32_t i = n; i > 1; --i)
                    std::swap(seed[i - 1], seed[perm_rng.next() % i]);
                tie_seed = 0xD1B54A32D192ED03ull * (trial + 1);
            }

            // A trial may throw where another succeeds (a conditional 2q
            // gate is routable only under a placement that makes its pair
            // adjacent).  Keep successes; rethrow the first error only if
            // every trial failed, preserving the throw contract for
            // genuinely unroutable circuits.
            try {
                // Route after every refinement round and keep the best:
                // measurement showed round 1 and round 2 each winning on
                // different fixtures, so both depths enter the portfolio.
                auto refined = std::move(seed);
                for (std::size_t round = 0; round < kRefineRounds; ++round) {
                    refined = sabre_core(dag, opts, dist, std::move(refined),
                                         /*out=*/nullptr, tie_seed);
                    refined = sabre_core(rev_dag, opts, dist, std::move(refined),
                                         /*out=*/nullptr, tie_seed);
                    std::vector<Command> stream;
                    stream.reserve(commands.size());
                    auto final_map = sabre_core(dag, opts, dist, refined,
                                                &stream, tie_seed);
                    const std::size_t swaps = count_swaps(stream);
                    if (!have_best || swaps < best_swaps ||
                        (swaps == best_swaps && stream.size() < result.commands.size())) {
                        best_swaps = swaps;
                        result.commands = std::move(stream);
                        // `refined` was passed by value above, so it is intact.
                        result.initial_logical_to_physical = refined;
                        result.final_logical_to_physical = std::move(final_map);
                        have_best = true;
                    }
                }
            } catch (...) {
                if (!first_error) first_error = std::current_exception();
            }
        }
        if (!have_best) std::rethrow_exception(first_error);
    }

    // Portfolio guarantee: never worse than Lite.  Some topologies favour
    // the greedy shortest-path sweep (e.g. a circular entangler on a line);
    // routing is a build()-time stage, so the extra pass is cheap.
    try {
        RoutingOptions lite_opts = opts;
        lite_opts.router = RouterKind::Lite;
        auto lite = route(commands, lite_opts);
        if (count_swaps(lite.commands) < count_swaps(result.commands))
            return lite;
    } catch (const std::exception&) {
        // Lite refused (e.g. conditional-2q under its mapping) — keep Sabre.
    }
    return result;
}

}  // namespace qarpx
