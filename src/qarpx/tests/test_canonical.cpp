// Tests for qarpx::canonical (canonicalize / rebind / topology_hash / binding_hash).
//
// The canonical form replaces real qubits / cbits / symbols with positional
// dummies in order of first appearance.  Two blocks that differ only by
// qubit relocation or symbol rename produce identical canonical commands
// (and therefore identical topology hashes); the difference shows up only
// in the binding hash.

#include <gtest/gtest.h>

#include "qarpx/core/canonical.h"
#include "qarpx/core/command.h"
#include "qarpx/core/param.h"
#include "qarpx/core/gates.h"

#include <vector>

namespace qarpx::test {

namespace {

using qarpx::canonical::CanonicalForm;
using qarpx::canonical::canonicalize;
using qarpx::canonical::rebind;
using qarpx::canonical::topology_hash;
using qarpx::canonical::binding_hash;

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

}  // anonymous namespace

// ── canonicalize / rebind round-trip identity ──────────────────────────────

TEST(Canonical, RoundTripPreservesCommands_NoParam) {
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::H, 5u);
    cmds.emplace_back(GateType::CX, 5u, 6u);
    cmds.emplace_back(GateType::H, 6u);

    auto cf = canonicalize(cmds);
    auto back = rebind(cf);
    EXPECT_TRUE(commands_exactly_equal(cmds, back));
}

TEST(Canonical, RoundTripPreservesCommands_WithSymbols) {
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::Rz, 3u, Param::symbol("theta"));
    cmds.emplace_back(GateType::CX, 3u, 7u);
    cmds.emplace_back(GateType::Rz, 7u, Param::linear(2.0, "phi", 0.5));

    auto cf = canonicalize(cmds);
    auto back = rebind(cf);
    EXPECT_TRUE(commands_exactly_equal(cmds, back));
}

// ── First-appearance dummy assignment ───────────────────────────────────────

TEST(Canonical, QubitDummiesAssignedInOrderOfFirstAppearance) {
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::H, 5u);
    cmds.emplace_back(GateType::CX, 5u, 9u);
    cmds.emplace_back(GateType::H, 9u);

    auto cf = canonicalize(cmds);
    // qubit 5 first → dummy 0; qubit 9 second → dummy 1.
    EXPECT_EQ(cf.qubit_binding, std::vector<uint32_t>({5, 9}));
    // The canonical commands use [0, 1].
    EXPECT_EQ(cf.commands[0].qubits[0], 0u);  // H(5) → H(0)
    EXPECT_EQ(cf.commands[1].qubits[0], 0u);  // CX(5, 9) → CX(0, 1)
    EXPECT_EQ(cf.commands[1].qubits[1], 1u);
    EXPECT_EQ(cf.commands[2].qubits[0], 1u);  // H(9) → H(1)
}

TEST(Canonical, SymbolDummiesUseSymKNaming) {
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::Rz, 0u, Param::symbol("theta"));
    cmds.emplace_back(GateType::Rz, 1u, Param::symbol("phi"));
    cmds.emplace_back(GateType::Rz, 0u, Param::symbol("theta"));  // reuse

    auto cf = canonicalize(cmds);
    EXPECT_EQ(cf.symbol_binding, std::vector<std::string>({"theta", "phi"}));
    EXPECT_EQ(cf.commands[0].params[0].free_symbols(),
              std::vector<std::string>({"__sym_0"}));
    EXPECT_EQ(cf.commands[1].params[0].free_symbols(),
              std::vector<std::string>({"__sym_1"}));
    // Reused symbol should map to the same dummy.
    EXPECT_EQ(cf.commands[2].params[0].free_symbols(),
              std::vector<std::string>({"__sym_0"}));
}

// ── topology_hash invariance ────────────────────────────────────────────────

TEST(Canonical, TopologyHashEqualUnderQubitRelocation) {
    std::vector<Command> cmds_a;
    cmds_a.emplace_back(GateType::H, 0u);
    cmds_a.emplace_back(GateType::CX, 0u, 1u);
    cmds_a.emplace_back(GateType::Rz, 1u, Param(0.5));

    std::vector<Command> cmds_b;
    cmds_b.emplace_back(GateType::H, 5u);
    cmds_b.emplace_back(GateType::CX, 5u, 6u);
    cmds_b.emplace_back(GateType::Rz, 6u, Param(0.5));

    EXPECT_EQ(topology_hash(canonicalize(cmds_a)),
              topology_hash(canonicalize(cmds_b)));
    // But binding hash differs.
    EXPECT_NE(binding_hash(canonicalize(cmds_a)),
              binding_hash(canonicalize(cmds_b)));
}

TEST(Canonical, TopologyHashEqualUnderSymbolRename) {
    std::vector<Command> cmds_a;
    cmds_a.emplace_back(GateType::Rz, 0u, Param::symbol("theta"));
    cmds_a.emplace_back(GateType::Rz, 1u, Param::symbol("phi"));

    std::vector<Command> cmds_b;
    cmds_b.emplace_back(GateType::Rz, 0u, Param::symbol("alpha"));
    cmds_b.emplace_back(GateType::Rz, 1u, Param::symbol("beta"));

    EXPECT_EQ(topology_hash(canonicalize(cmds_a)),
              topology_hash(canonicalize(cmds_b)));
    EXPECT_NE(binding_hash(canonicalize(cmds_a)),
              binding_hash(canonicalize(cmds_b)));
}

TEST(Canonical, TopologyHashEqualUnderSymbolRename_LinearForm) {
    // coeff*sym+offset should canonicalise the symbol but preserve coeff/offset.
    std::vector<Command> cmds_a;
    cmds_a.emplace_back(GateType::Rz, 0u, Param::linear(2.5, "theta", 1.0));

    std::vector<Command> cmds_b;
    cmds_b.emplace_back(GateType::Rz, 0u, Param::linear(2.5, "alpha", 1.0));

    EXPECT_EQ(topology_hash(canonicalize(cmds_a)),
              topology_hash(canonicalize(cmds_b)));

    // Different coefficient → different topology.
    std::vector<Command> cmds_c;
    cmds_c.emplace_back(GateType::Rz, 0u, Param::linear(2.5, "alpha", 2.0));
    EXPECT_NE(topology_hash(canonicalize(cmds_a)),
              topology_hash(canonicalize(cmds_c)));

    std::vector<Command> cmds_d;
    cmds_d.emplace_back(GateType::Rz, 0u, Param::linear(3.0, "alpha", 1.0));
    EXPECT_NE(topology_hash(canonicalize(cmds_a)),
              topology_hash(canonicalize(cmds_d)));
}

TEST(Canonical, TopologyHashDistinguishesGateOrder) {
    // CX(0,1) and CX(1,0) are physically distinct (control/target swap) and
    // must hash to different topologies even after canonicalisation.
    std::vector<Command> cmds_a;
    cmds_a.emplace_back(GateType::H, 0u);
    cmds_a.emplace_back(GateType::CX, 0u, 1u);

    std::vector<Command> cmds_b;
    cmds_b.emplace_back(GateType::H, 0u);
    cmds_b.emplace_back(GateType::CX, 1u, 0u);

    EXPECT_NE(topology_hash(canonicalize(cmds_a)),
              topology_hash(canonicalize(cmds_b)));
}

TEST(Canonical, TopologyHashDistinguishesGateSequenceOrder) {
    std::vector<Command> cmds_a;
    cmds_a.emplace_back(GateType::H, 0u);
    cmds_a.emplace_back(GateType::CX, 0u, 1u);

    std::vector<Command> cmds_b;
    cmds_b.emplace_back(GateType::CX, 0u, 1u);
    cmds_b.emplace_back(GateType::H, 0u);

    EXPECT_NE(topology_hash(canonicalize(cmds_a)),
              topology_hash(canonicalize(cmds_b)));
}

TEST(Canonical, ConcreteParamsBitIdenticalOrDifferent) {
    std::vector<Command> cmds_a;
    cmds_a.emplace_back(GateType::Rz, 0u, Param(0.5));

    std::vector<Command> cmds_b;
    cmds_b.emplace_back(GateType::Rz, 0u, Param(0.5));

    std::vector<Command> cmds_c;
    cmds_c.emplace_back(GateType::Rz, 0u, Param(0.5 + 1e-15));

    EXPECT_EQ(topology_hash(canonicalize(cmds_a)),
              topology_hash(canonicalize(cmds_b)));
    // Bit-different doubles → different hash, no ULP coalescing.
    EXPECT_NE(topology_hash(canonicalize(cmds_a)),
              topology_hash(canonicalize(cmds_c)));
}

// ── Binding hash distinguishes qubit / symbol assignments ───────────────────

TEST(Canonical, BindingHashIsDistinctAcrossDifferentMappings) {
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::H, 5u);
    cmds.emplace_back(GateType::H, 6u);  // qubit_binding = [5, 6]

    std::vector<Command> cmds2;
    cmds2.emplace_back(GateType::H, 6u);
    cmds2.emplace_back(GateType::H, 5u);  // qubit_binding = [6, 5]

    auto cf1 = canonicalize(cmds);
    auto cf2 = canonicalize(cmds2);
    // Same topology (both are "two H gates on different qubits").
    EXPECT_EQ(topology_hash(cf1), topology_hash(cf2));
    // Different binding (qubits in different order).
    EXPECT_NE(binding_hash(cf1), binding_hash(cf2));
}

}  // namespace qarpx::test
