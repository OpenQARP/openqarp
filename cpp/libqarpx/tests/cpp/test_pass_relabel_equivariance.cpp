// Tests that the transpiler passes (decomposition cascades, fusion, identity
// elimination) commute with qubit relabeling and symbol renaming.  This is
// the load-bearing assumption of the topology cache.
//
// Contract per pass:
//   pass(remap(input)) == remap(pass(input))
//
// for any qubit relabeling / symbol rename.  If a pass were to peek at the
// concrete qubit index (e.g., a routing pass with a coupling map), it would
// fail this check — and would need to be excluded from the cache.

#include <gtest/gtest.h>

#include "qarpx/qarpx.h"
#include "qarpx/transpiler/transpiler.h"
#include "qarpx/transpiler/identities.h"
#include "qarpx/transpiler/fusion.h"
#include "qarpx/transpiler/gateset.h"
#include "qarpx/core/canonical.h"

#include <vector>
#include <unordered_map>

namespace qarpx::test {

namespace {

// Apply a qubit / symbol relabeling to a copy of `cmds`.
std::vector<Command> relabel(
    const std::vector<Command>& cmds,
    const std::unordered_map<uint32_t, uint32_t>& qubit_map,
    const std::unordered_map<std::string, std::string>& symbol_map = {})
{
    std::vector<Command> out;
    out.reserve(cmds.size());
    for (const auto& cmd : cmds) {
        Command nc = cmd;
        for (auto& q : nc.qubits) {
            auto it = qubit_map.find(q);
            if (it != qubit_map.end()) q = it->second;
        }
        for (auto& cb : nc.condition_bits) { (void)cb; /* no relabel */ }
        if (!symbol_map.empty()) {
            SmallVector<Param, 1> new_params;
            new_params.reserve(nc.params.size());
            for (const auto& p : nc.params) {
                if (p.is_concrete()) {
                    new_params.push_back(p);
                } else {
                    new_params.push_back(p.rename_symbols(symbol_map));
                }
            }
            nc.params = std::move(new_params);
        }
        out.push_back(std::move(nc));
    }
    return out;
}

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

// Standard relabel maps used across the tests.
const std::unordered_map<uint32_t, uint32_t> kShift10 = {
    {0, 10}, {1, 11}, {2, 12}, {3, 13}, {4, 14}, {5, 15}};

}  // anonymous namespace

// ── Decomposition pass: equivariant under qubit relabel ─────────────────────

TEST(PassEquivariance, Transpile_RXX_DecompositionEquivariant) {
    // RXX targets qulacs_gateset which excludes RXX → cascades to CX + Rz + H.
    Transpiler t(qulacs_gateset());

    std::vector<Command> in;
    in.emplace_back(GateType::RXX, 0u, 1u, Param(0.7));
    in.emplace_back(GateType::H, 0u);

    auto out_orig = t.transpile(in);
    auto out_relocated = t.transpile(relabel(in, kShift10));
    auto expected = relabel(out_orig, kShift10);

    EXPECT_TRUE(commands_exactly_equal(out_relocated, expected));
}

TEST(PassEquivariance, Transpile_RZZ_DecompositionEquivariant) {
    Transpiler t(qulacs_gateset());

    std::vector<Command> in;
    in.emplace_back(GateType::RZZ, 0u, 1u, Param(0.5));

    auto out_orig = t.transpile(in);
    auto out_relocated = t.transpile(relabel(in, kShift10));
    EXPECT_TRUE(commands_exactly_equal(out_relocated, relabel(out_orig, kShift10)));
}

TEST(PassEquivariance, Transpile_CCX_DecompositionEquivariant) {
    Transpiler t(qulacs_gateset());

    std::vector<Command> in;
    in.emplace_back(GateType::CCX, SmallVector<uint32_t, 2>{0u, 1u, 2u});

    auto out_orig = t.transpile(in);
    auto out_relocated = t.transpile(relabel(in, kShift10));
    EXPECT_TRUE(commands_exactly_equal(out_relocated, relabel(out_orig, kShift10)));
}

TEST(PassEquivariance, Transpile_CSWAP_DecompositionEquivariant) {
    Transpiler t(qulacs_gateset());

    std::vector<Command> in;
    in.emplace_back(GateType::CSWAP, SmallVector<uint32_t, 2>{0u, 1u, 2u});

    auto out_orig = t.transpile(in);
    auto out_relocated = t.transpile(relabel(in, kShift10));
    EXPECT_TRUE(commands_exactly_equal(out_relocated, relabel(out_orig, kShift10)));
}

TEST(PassEquivariance, Transpile_MCZ_DecompositionEquivariant) {
    // qulacs_gateset (excludes MCZ but includes Rz/CX) — MCZ cascades to
    // CCZ-like Toffoli-net with Rz / CX leaves.
    Transpiler t(qulacs_gateset());

    std::vector<Command> in;
    in.emplace_back(GateType::MCZ, SmallVector<uint32_t, 2>{0u, 1u, 2u});

    auto out_orig = t.transpile(in);
    auto out_relocated = t.transpile(relabel(in, kShift10));
    EXPECT_TRUE(commands_exactly_equal(out_relocated, relabel(out_orig, kShift10)));
}

// ── Identity-elimination pass: equivariant under qubit relabel ──────────────

TEST(PassEquivariance, IdentityElimination_HHCancelsEquivariant) {
    std::vector<Command> in;
    in.emplace_back(GateType::H, 0u);
    in.emplace_back(GateType::H, 0u);
    in.emplace_back(GateType::H, 1u);

    auto out_orig = in;
    eliminate_identities(out_orig);
    auto in_relocated = relabel(in, kShift10);
    auto out_relocated = in_relocated;
    eliminate_identities(out_relocated);

    EXPECT_TRUE(commands_exactly_equal(out_relocated, relabel(out_orig, kShift10)));
}

TEST(PassEquivariance, IdentityElimination_CXCancelsEquivariant) {
    std::vector<Command> in;
    in.emplace_back(GateType::CX, 0u, 1u);
    in.emplace_back(GateType::CX, 0u, 1u);
    in.emplace_back(GateType::H, 2u);

    auto out_orig = in; eliminate_identities(out_orig);
    auto out_relocated = relabel(in, kShift10); eliminate_identities(out_relocated);
    EXPECT_TRUE(commands_exactly_equal(out_relocated, relabel(out_orig, kShift10)));
}

TEST(PassEquivariance, IdentityElimination_RzMergeEquivariant) {
    std::vector<Command> in;
    in.emplace_back(GateType::Rz, 0u, Param(0.3));
    in.emplace_back(GateType::Rz, 0u, Param(0.4));
    in.emplace_back(GateType::H, 1u);

    auto out_orig = in; eliminate_identities(out_orig);
    auto out_relocated = relabel(in, kShift10); eliminate_identities(out_relocated);
    EXPECT_TRUE(commands_exactly_equal(out_relocated, relabel(out_orig, kShift10)));
}

TEST(PassEquivariance, IdentityElimination_SSdgCancelsEquivariant) {
    std::vector<Command> in;
    in.emplace_back(GateType::S,   0u);
    in.emplace_back(GateType::Sdg, 0u);
    in.emplace_back(GateType::H,   1u);

    auto out_orig = in; eliminate_identities(out_orig);
    auto out_relocated = relabel(in, kShift10); eliminate_identities(out_relocated);
    EXPECT_TRUE(commands_exactly_equal(out_relocated, relabel(out_orig, kShift10)));
}

// ── Fusion pass: equivariant under qubit relabel ────────────────────────────

TEST(PassEquivariance, Fusion_OneQubitChainEquivariant) {
    // H · S · T on qubit 0 fuses to a single Custom matrix.  The fused
    // command must land on whichever qubit the original chain occupies.
    std::vector<Command> in;
    in.emplace_back(GateType::H, 0u);
    in.emplace_back(GateType::S, 0u);
    in.emplace_back(GateType::T, 0u);
    in.emplace_back(GateType::CX, 0u, 1u);  // multi-qubit barrier — flushes
    in.emplace_back(GateType::H, 2u);
    in.emplace_back(GateType::S, 2u);

    auto out_orig = fuse_single_qubit_gates(in);
    auto out_relocated = fuse_single_qubit_gates(relabel(in, kShift10));

    auto expected = relabel(out_orig, kShift10);
    // We can't compare the Custom-gate `unitary` field directly via
    // commands_exactly_equal (it's a shared_ptr<Eigen::MatrixXcd>), so we compare
    // shape only: same length, same gate types, same qubits.
    ASSERT_EQ(out_relocated.size(), expected.size());
    for (size_t i = 0; i < out_relocated.size(); ++i) {
        EXPECT_EQ(out_relocated[i].gate,   expected[i].gate);
        EXPECT_EQ(out_relocated[i].qubits, expected[i].qubits);
    }
}

// ── Combined: decomposition + identity elimination + fusion ────────────────

TEST(PassEquivariance, TranspileAndOptimize_FullPipelineEquivariant) {
    Transpiler t(qulacs_gateset());

    std::vector<Command> in;
    in.emplace_back(GateType::RXX, 0u, 1u, Param(0.5));
    in.emplace_back(GateType::H, 0u);
    in.emplace_back(GateType::H, 0u);   // ID-cancels
    in.emplace_back(GateType::Rz, 1u, Param(0.3));
    in.emplace_back(GateType::Rz, 1u, Param(0.2));   // merges with above
    in.emplace_back(GateType::CCX, SmallVector<uint32_t, 2>{0u, 1u, 2u});

    auto out_orig = t.transpile_and_optimize(in);
    auto out_relocated = t.transpile_and_optimize(relabel(in, kShift10));

    // The optimized output may have Custom-gates from fusion; compare shape
    // only as in the fusion test.
    auto expected = relabel(out_orig, kShift10);
    ASSERT_EQ(out_relocated.size(), expected.size());
    for (size_t i = 0; i < out_relocated.size(); ++i) {
        EXPECT_EQ(out_relocated[i].gate,   expected[i].gate);
        EXPECT_EQ(out_relocated[i].qubits, expected[i].qubits);
        EXPECT_EQ(out_relocated[i].params.size(), expected[i].params.size());
        for (size_t k = 0; k < out_relocated[i].params.size(); ++k) {
            EXPECT_TRUE(out_relocated[i].params[k] == expected[i].params[k])
                << "param mismatch at cmd " << i << ", pos " << k;
        }
    }
}

// ── Symbol-rename equivariance ──────────────────────────────────────────────

TEST(PassEquivariance, Transpile_SymbolicParamsRenameEquivariant) {
    Transpiler t(qulacs_gateset());

    std::vector<Command> in;
    in.emplace_back(GateType::RXX, 0u, 1u, Param::symbol("theta"));
    in.emplace_back(GateType::Rz, 0u, Param::linear(2.0, "phi", 0.5));
    in.emplace_back(GateType::H, 1u);

    std::unordered_map<std::string, std::string> rename = {
        {"theta", "alpha"}, {"phi", "beta"}};

    auto out_orig = t.transpile(in);
    auto out_renamed = t.transpile(relabel(in, /*qubit_map=*/{}, rename));
    EXPECT_TRUE(commands_exactly_equal(out_renamed, relabel(out_orig, /*qubit_map=*/{}, rename)));
}

// ── Cache-on equivalence: combined decomposition + fusion + identity ───────
//
// This is the punch line: the topology cache (which assumes pass equivariance)
// must reproduce the uncached output exactly across qubit-relocated AND
// symbol-renamed inputs that share a topology.

TEST(PassEquivariance, TopologyCache_TranspileAndOptimize_SharedTopology) {
    auto build_circuit = [](uint32_t q0, uint32_t q1, uint32_t q2,
                            const std::string& sym_a) {
        std::vector<Command> c;
        c.emplace_back(GateType::RXX, q0, q1, Param::symbol(sym_a));
        c.emplace_back(GateType::H, q0);
        c.emplace_back(GateType::H, q0);   // cancels
        c.emplace_back(GateType::Rz, q1, Param::linear(0.5, sym_a, 0.0));
        c.emplace_back(GateType::CCX, SmallVector<uint32_t, 2>{q0, q1, q2});
        return c;
    };

    Transpiler t(qulacs_gateset());
    t.enable_topology_cache(true);

    // Prime the cache with one query.
    auto a = build_circuit(0, 1, 2, "alpha");
    auto out_a = t.transpile(a);

    // Different qubits + different symbol — must hit the cache.
    auto b = build_circuit(7, 8, 9, "gamma");
    auto out_b = t.transpile(b);
    EXPECT_EQ(t.topology_cache_stats().hits, 1u);
    EXPECT_EQ(t.topology_cache_stats().misses, 1u);

    // Compare against the uncached reference.
    Transpiler reference(qulacs_gateset());  // cache off
    auto ref_b = reference.transpile(b);
    EXPECT_TRUE(commands_exactly_equal(out_b, ref_b));
}

// ── DAG cancel pass: equivariant under qubit relabel + symbol rename ────────

TEST(PassEquivariance, DagCancelWireAdjacentEquivariant) {
    // Wire-adjacent cancellation, rotation merge (concrete + symbolic), and a
    // spectator gate keeping the pair textually non-adjacent.
    std::vector<Command> in;
    in.emplace_back(GateType::H, 0u);
    in.emplace_back(GateType::X, 1u);
    in.emplace_back(GateType::H, 0u);          // wire-adjacent cancel
    in.emplace_back(GateType::Rz, 2u, Param::symbol("alpha"));
    in.emplace_back(GateType::CX, 0u, 1u);
    in.emplace_back(GateType::Rz, 2u, Param::linear(0.5, "alpha", 0.1));
    in.emplace_back(GateType::Rz, 3u, Param(0.3));
    in.emplace_back(GateType::Rz, 3u, Param(0.4));

    auto pass = [](const std::vector<Command>& cmds) {
        auto dag = CircuitDAG::from_commands(cmds);
        dag_passes::cancel_wire_adjacent(dag);
        return dag.to_commands();
    };

    const std::unordered_map<std::string, std::string> sym_map = {{"alpha", "beta"}};
    auto out_relocated = pass(relabel(in, kShift10, sym_map));
    auto expected = relabel(pass(in), kShift10, sym_map);
    EXPECT_TRUE(commands_exactly_equal(out_relocated, expected));
}

}  // namespace qarpx::test
