// Tests for the opt-in topology-hash cache on Transpiler::transpile.
//
// Contract: with the cache enabled, transpile(X) must produce the same flat
// command stream as without the cache, for any input.  Additionally, two
// inputs that share a topology (qubit relocation / symbol rename) must hit
// the same cache slot.

#include <gtest/gtest.h>

#include "qarpx/qarpx.h"
#include "qarpx/transpiler/transpiler.h"
#include "qarpx/transpiler/gateset.h"
#include "qarpx/core/canonical.h"

#include <vector>

namespace qarpx::test {

namespace {

bool commands_exactly_equal(const std::vector<Command>& a, const std::vector<Command>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].gate   != b[i].gate)   return false;
        if (a[i].qubits != b[i].qubits) return false;
        if (a[i].cbits  != b[i].cbits)  return false;
        if (a[i].params.size() != b[i].params.size()) return false;
        for (size_t k = 0; k < a[i].params.size(); ++k) {
            if (!(a[i].params[k] == b[i].params[k])) return false;
        }
    }
    return true;
}

// A synthetic "rich" command sequence with mixed gate types, parametric and
// concrete params, and multiple symbols — hits most decomposition paths.
std::vector<Command> sample_circuit_on(uint32_t q0, uint32_t q1, uint32_t q2,
                                       const std::string& sym_a,
                                       const std::string& sym_b) {
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::H, q0);
    cmds.emplace_back(GateType::Rz, q0, Param::symbol(sym_a));
    cmds.emplace_back(GateType::CX, q0, q1);
    cmds.emplace_back(GateType::Ry, q1, Param::linear(2.5, sym_b, 0.5));
    cmds.emplace_back(GateType::CX, q1, q2);
    cmds.emplace_back(GateType::H, q2);
    return cmds;
}

// A symbolic `U`, which the qulacs gate set (no U, has GPhase) lowers to
// Rz·Ry·Rz plus GPhase((phi+lam)/2) — one param over *two* symbols.
std::vector<Command> symbolic_u_on(const std::string& theta,
                                   const std::string& phi,
                                   const std::string& lam) {
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::U,
                      SmallVector<uint32_t, 2>{0u},
                      SmallVector<Param, 1>{Param::symbol(theta),
                                            Param::symbol(phi),
                                            Param::symbol(lam)});
    return cmds;
}

// The widest param in a stream — the compound one, when there is one.
Param widest_param(const std::vector<Command>& cmds) {
    Param widest{};
    size_t best = 0;
    for (const auto& c : cmds) {
        for (const auto& p : c.params) {
            const size_t n = p.free_symbols().size();
            if (n > best) { best = n; widest = p; }
        }
    }
    return widest;
}

}  // anonymous namespace

// ── Cache-on equals cache-off ──────────────────────────────────────────────

TEST(TopologyCache, CachedEqualsUncached_BasicSequence) {
    Transpiler t(native_gateset());
    auto cmds = sample_circuit_on(0, 1, 2, "alpha", "beta");

    auto without = t.transpile(cmds);

    t.enable_topology_cache(true);
    auto first  = t.transpile(cmds);
    auto second = t.transpile(cmds);  // cache hit

    EXPECT_TRUE(commands_exactly_equal(without, first));
    EXPECT_TRUE(commands_exactly_equal(without, second));
    EXPECT_EQ(t.topology_cache_stats().hits, 1u);
    EXPECT_EQ(t.topology_cache_stats().misses, 1u);
}

// ── Topology hit across qubit relocation ──────────────────────────────────

TEST(TopologyCache, RelocatedQubitsHitSameCacheSlot) {
    Transpiler t(native_gateset());
    t.enable_topology_cache(true);

    auto a = sample_circuit_on(0, 1, 2, "alpha", "beta");
    auto b = sample_circuit_on(7, 8, 9, "alpha", "beta");

    auto out_a = t.transpile(a);
    auto out_b = t.transpile(b);  // same topology → cache hit

    EXPECT_EQ(t.topology_cache_stats().hits, 1u);
    EXPECT_EQ(t.topology_cache_stats().misses, 1u);

    // Verify cached output (rebound) matches direct transpile.
    Transpiler reference(native_gateset());
    auto ref_b = reference.transpile(b);
    EXPECT_TRUE(commands_exactly_equal(out_b, ref_b));
}

// ── Topology hit across symbol rename ─────────────────────────────────────

TEST(TopologyCache, RenamedSymbolsHitSameCacheSlot) {
    Transpiler t(native_gateset());
    t.enable_topology_cache(true);

    auto a = sample_circuit_on(0, 1, 2, "alpha", "beta");
    auto b = sample_circuit_on(0, 1, 2, "gamma", "delta");

    auto out_a = t.transpile(a);
    auto out_b = t.transpile(b);

    EXPECT_EQ(t.topology_cache_stats().hits, 1u);
    EXPECT_EQ(t.topology_cache_stats().misses, 1u);

    // The rebound output for `b` should use `b`'s symbol names.
    Transpiler reference(native_gateset());
    auto ref_b = reference.transpile(b);
    EXPECT_TRUE(commands_exactly_equal(out_b, ref_b));
}

// ── Different topology gets a distinct cache slot ─────────────────────────

TEST(TopologyCache, DifferentTopologyMisses) {
    Transpiler t(native_gateset());
    t.enable_topology_cache(true);

    std::vector<Command> a;
    a.emplace_back(GateType::H, 0u);
    a.emplace_back(GateType::CX, 0u, 1u);

    std::vector<Command> b;
    b.emplace_back(GateType::H, 0u);
    b.emplace_back(GateType::CX, 1u, 0u);  // CX with control/target swapped

    (void)t.transpile(a);
    (void)t.transpile(b);
    EXPECT_EQ(t.topology_cache_stats().hits, 0u);
    EXPECT_EQ(t.topology_cache_stats().misses, 2u);
}

// ── Concrete params that differ → cache miss (no ULP coalescing) ──────────

TEST(TopologyCache, DifferentConcreteParamsMiss) {
    Transpiler t(native_gateset());
    t.enable_topology_cache(true);

    std::vector<Command> a;
    a.emplace_back(GateType::Rz, 0u, Param(0.5));

    std::vector<Command> b;
    b.emplace_back(GateType::Rz, 0u, Param(0.7));

    (void)t.transpile(a);
    (void)t.transpile(b);
    EXPECT_EQ(t.topology_cache_stats().hits, 0u);
    EXPECT_EQ(t.topology_cache_stats().misses, 2u);
}

// ── clear_topology_cache resets state ─────────────────────────────────────

TEST(TopologyCache, ClearResetsState) {
    Transpiler t(native_gateset());
    t.enable_topology_cache(true);

    auto cmds = sample_circuit_on(0, 1, 2, "alpha", "beta");
    (void)t.transpile(cmds);
    (void)t.transpile(cmds);
    EXPECT_GT(t.topology_cache_stats().hits, 0u);
    EXPECT_GT(t.topology_cache_stats().size, 0u);

    t.clear_topology_cache();
    EXPECT_EQ(t.topology_cache_stats().hits, 0u);
    EXPECT_EQ(t.topology_cache_stats().misses, 0u);
    EXPECT_EQ(t.topology_cache_stats().size, 0u);
}

// ── Cache disabled by default ─────────────────────────────────────────────

TEST(TopologyCache, DisabledByDefault) {
    Transpiler t(native_gateset());
    EXPECT_FALSE(t.is_topology_cache_enabled());

    auto cmds = sample_circuit_on(0, 1, 2, "alpha", "beta");
    (void)t.transpile(cmds);
    (void)t.transpile(cmds);
    EXPECT_EQ(t.topology_cache_stats().hits, 0u);
    EXPECT_EQ(t.topology_cache_stats().misses, 0u);
    EXPECT_EQ(t.topology_cache_stats().size, 0u);
}

// ── Compound params across a rebind ───────────────────────────────────────
//
// Canonicalisation renames symbols to `__sym_i`, so a decomposition running on
// the dummified stream builds its compound params over *those* names; rebind
// then renames them to whatever the query supplied.  A rename that only
// rewrote a param's symbol list would leave its evaluator resolving `__sym_i`,
// which no caller ever supplies — the failure is a wrong number rather than a
// throw whenever a stale name happens to be in the value map.

TEST(TopologyCache, SymbolicUReallyProducesACompoundParam) {
    // Guard: without a two-symbol param the tests below prove nothing.
    Transpiler t(qulacs_gateset());
    const Param widest = widest_param(t.transpile(symbolic_u_on("theta", "phi", "lam")));
    EXPECT_GT(widest.free_symbols().size(), 1u)
        << "no compound param produced; got " << widest.to_string();
}

TEST(TopologyCache, CompoundParamEvaluatesAfterRebind) {
    Transpiler t(qulacs_gateset());
    t.enable_topology_cache(true);

    // Warm the slot with one symbol set, then hit it with another.
    (void)t.transpile(symbolic_u_on("theta", "phi", "lam"));
    const auto out = t.transpile(symbolic_u_on("t2", "p2", "l2"));
    EXPECT_EQ(t.topology_cache_stats().hits, 1u);

    const Param compound = widest_param(out);
    ASSERT_GT(compound.free_symbols().size(), 1u);
    for (const auto& s : compound.free_symbols()) {
        EXPECT_NE(s.rfind("__sym_", 0), 0u)
            << "rebound param still carries the canonical name " << s;
    }

    // (phi + lam) / 2 under the *query's* names.
    const double got = compound.evaluate({{"t2", 0.3}, {"p2", 0.4}, {"l2", 0.5}});
    EXPECT_NEAR(got, 0.45, 1e-15);
}

TEST(TopologyCache, CompoundParamIgnoresStaleCanonicalNames) {
    // The silent-wrong-value case: every symbol resolves, so nothing throws —
    // the evaluator must read the query's names, not leftover canonical ones.
    Transpiler t(qulacs_gateset());
    t.enable_topology_cache(true);

    (void)t.transpile(symbolic_u_on("theta", "phi", "lam"));
    const Param compound = widest_param(t.transpile(symbolic_u_on("t2", "p2", "l2")));
    ASSERT_GT(compound.free_symbols().size(), 1u);

    const double got = compound.evaluate({
        {"t2", 0.3}, {"p2", 0.4}, {"l2", 0.5},
        {"__sym_0", 99.0}, {"__sym_1", 99.0}, {"__sym_2", 99.0},
        {"theta", 99.0}, {"phi", 99.0}, {"lam", 99.0},
    });
    EXPECT_NEAR(got, 0.45, 1e-15);
}

TEST(TopologyCache, CompoundParamCachedEqualsUncached) {
    auto query = symbolic_u_on("t2", "p2", "l2");

    Transpiler reference(qulacs_gateset());
    const auto uncached = reference.transpile(query);

    Transpiler t(qulacs_gateset());
    t.enable_topology_cache(true);
    (void)t.transpile(symbolic_u_on("theta", "phi", "lam"));
    const auto cached = t.transpile(query);

    EXPECT_EQ(t.topology_cache_stats().hits, 1u);
    EXPECT_TRUE(commands_exactly_equal(uncached, cached));
}

// ── Empty input is short-circuited, doesn't touch the cache ──────────────

TEST(TopologyCache, EmptyInputBypassesCache) {
    Transpiler t(native_gateset());
    t.enable_topology_cache(true);

    auto out = t.transpile({});
    EXPECT_TRUE(out.empty());
    EXPECT_EQ(t.topology_cache_stats().hits, 0u);
    EXPECT_EQ(t.topology_cache_stats().misses, 0u);
}

}  // namespace qarpx::test
