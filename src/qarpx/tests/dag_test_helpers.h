// Shared helpers for the CircuitDAG test suites (test_circuit_dag.cpp,
// test_dag_passes.cpp): full-fidelity command equality, command factories,
// and a seeded random-circuit generator covering the full IR grammar.
#pragma once

#include "qarpx/dag/circuit_dag.h"
#include "qarpx/core/command.h"
#include "qarpx/core/gates.h"
#include "qarpx/core/param.h"
#include "qarpx/transpiler/identities.h"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>
#include <complex>
#include <memory>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

namespace qarpx::test {

/// Full-fidelity equality: Command::operator== plus the Custom unitary
/// payload (round-trip copies share the shared_ptr, so handle equality).
inline bool full_equal(const Command& a, const Command& b) {
    return a == b && a.unitary == b.unitary;
}

inline bool streams_equal(const std::vector<Command>& a,
                          const std::vector<Command>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (!full_equal(a[i], b[i])) return false;
    return true;
}

inline void expect_round_trip(const std::vector<Command>& cmds) {
    auto dag = CircuitDAG::from_commands(cmds);
    auto back = dag.to_commands();
    ASSERT_EQ(back.size(), cmds.size());
    for (std::size_t i = 0; i < cmds.size(); ++i)
        EXPECT_TRUE(full_equal(back[i], cmds[i])) << "mismatch at index " << i;
}

inline Command make_branch_begin(std::vector<uint32_t> cbits,
                                 std::vector<bool> values) {
    Command c;
    c.gate = GateType::BranchBegin;
    for (auto b : cbits)  c.condition_bits.push_back(b);
    for (bool v : values) c.condition_values.push_back(v);
    return c;
}

inline Command make_marker(GateType g) {
    Command c;
    c.gate = g;
    return c;
}

inline Command make_measure(uint32_t q, uint32_t c) {
    Command m(GateType::Measure, q);
    m.cbits.push_back(c);
    return m;
}

inline Command make_conditional(Command cmd, std::vector<uint32_t> cbits,
                                std::vector<bool> values) {
    for (auto b : cbits)  cmd.condition_bits.push_back(b);
    for (bool v : values) cmd.condition_values.push_back(v);
    return cmd;
}

inline Command make_custom_1q(uint32_t q) {
    Command c(GateType::Custom, q);
    Eigen::MatrixXcd u(2, 2);
    u << std::complex<double>(0, 1), 0, 0, std::complex<double>(0, -1);
    c.unitary = std::make_shared<const Eigen::MatrixXcd>(std::move(u));
    return c;
}

/// FROZEN stack-based, textually-adjacent-only `eliminate_identities` — the
/// differential oracle for the CircuitDAG pass of the same name: an
/// independent implementation the DAG pass must agree with.  Uses the
/// still-public predicates so rule semantics stay pinned to production.
/// Do not "improve" this — its value is being frozen.
inline std::size_t frozen_eliminate_identities_reference(
    std::vector<Command>& commands) {
    auto is_zero_rotation = [](const Param& p) {
        return p.is_concrete() && std::abs(p.value()) < 1e-12;
    };

    if (commands.empty()) return 0;
    std::vector<Command> stack;
    stack.reserve(commands.size());
    std::size_t eliminated = 0;

    for (auto& cmd : commands) {
        if (gate_num_params(cmd.gate) == 1 && !cmd.params.empty() &&
            (cmd.gate == GateType::Rx || cmd.gate == GateType::Ry ||
             cmd.gate == GateType::Rz)) {
            if (is_zero_rotation(cmd.params[0])) {
                ++eliminated;
                continue;
            }
        }
        if (!stack.empty()) {
            auto& top = stack.back();
            if (are_inverse_pair(top, cmd)) {
                stack.pop_back();
                eliminated += 2;
                continue;
            }
            if (can_merge_rotations(top, cmd)) {
                Param merged = top.params[0] + cmd.params[0];
                if (is_zero_rotation(merged)) {
                    stack.pop_back();
                    eliminated += 2;
                } else {
                    top.params[0] = merged;
                    ++eliminated;
                }
                continue;
            }
        }
        stack.push_back(std::move(cmd));
    }
    commands = std::move(stack);
    return eliminated;
}

/// Which subset of the IR grammar the generator draws from.
enum class CircuitGrammar {
    kFull,             ///< everything: MCM, conditions, branch regions, symbols
    kUnitaryConcrete,  ///< concrete-parameter unitary streams only — safe for
                       ///< QarpSimulator::unitary_matrix comparisons
    kUnitaryWide,      ///< kUnitaryConcrete over (nearly) the whole unitary
                       ///< gate vocabulary — pairs with reference_unitary.h.
                       ///< Requires n_qubits >= 3 (3q gates).
};

/// Seeded random circuit generator.  kFull covers 1q/2q gates, concrete and
/// symbolic rotations (occasionally exact-zero angles), measure, reset,
/// barrier, GPhase, Custom, conditional commands, and balanced branch
/// regions.  kUnitaryConcrete restricts to concrete unitary gates + GPhase +
/// Custom (no barrier/measure/reset/conditions/symbols).
class RandomCircuit {
public:
    RandomCircuit(uint32_t seed, uint32_t n_qubits, uint32_t n_cbits,
                  CircuitGrammar grammar = CircuitGrammar::kFull)
        : rng_(seed), n_qubits_(n_qubits), n_cbits_(n_cbits), grammar_(grammar) {}

    std::vector<Command> generate(std::size_t n, int max_region_depth = 1) {
        std::vector<Command> out;
        out.reserve(n);
        while (out.size() < n) emit(out, max_region_depth);
        return out;
    }

private:
    void emit(std::vector<Command>& out, int region_budget) {
        if (grammar_ == CircuitGrammar::kUnitaryConcrete) {
            emit_unitary(out);
        } else if (grammar_ == CircuitGrammar::kUnitaryWide) {
            emit_unitary_wide(out);
        } else {
            emit_full(out, region_budget);
        }
    }

    void emit_unitary_wide(std::vector<Command>& out) {
        switch (pick(10)) {
            case 0: {  // any fixed 1q gate
                static constexpr GateType k1q[] = {
                    GateType::X, GateType::Y, GateType::Z, GateType::H,
                    GateType::S, GateType::Sdg, GateType::T, GateType::Tdg,
                    GateType::SX, GateType::SXdg, GateType::Id};
                out.emplace_back(k1q[pick(11)], qubit());
                break;
            }
            case 1: {  // any 1q rotation
                static constexpr GateType kRot[] = {GateType::Rx, GateType::Ry,
                                                    GateType::Rz, GateType::P};
                out.emplace_back(kRot[pick(4)], qubit(), angle());
                break;
            }
            case 2: {  // general U
                Command u(GateType::U, qubit());
                u.params.push_back(angle());
                u.params.push_back(angle());
                u.params.push_back(angle());
                out.push_back(std::move(u));
                break;
            }
            case 3: {  // fixed 2q
                static constexpr GateType k2q[] = {
                    GateType::CX, GateType::CY, GateType::CZ, GateType::SWAP,
                    GateType::iSWAP, GateType::iSWAPdg, GateType::ECR,
                    GateType::CH, GateType::CS, GateType::CSdg, GateType::CSX, GateType::CSXdg};
                auto [a, b] = qubit_pair();
                out.emplace_back(k2q[pick(12)], a, b);
                break;
            }
            case 4: {  // parametric 2q
                static constexpr GateType k2qp[] = {
                    GateType::CRx, GateType::CRy, GateType::CRz, GateType::CP,
                    GateType::RZZ, GateType::RXX, GateType::RYY};
                auto [a, b] = qubit_pair();
                out.emplace_back(k2qp[pick(7)], a, b, angle());
                break;
            }
            case 5: {  // CU(θ, φ, λ, γ)
                auto [a, b] = qubit_pair();
                Command cu(GateType::CU, a, b);
                for (int i = 0; i < 4; ++i) cu.params.push_back(angle());
                out.push_back(std::move(cu));
                break;
            }
            case 6: {  // 3q
                auto [a, b, c] = qubit_triple();
                out.emplace_back(pick(2) == 0 ? GateType::CCX : GateType::CSWAP,
                                 SmallVector<uint32_t, 2>{a, b, c});
                break;
            }
            case 7: {  // MCZ over 3 qubits
                auto [a, b, c] = qubit_triple();
                out.emplace_back(GateType::MCZ,
                                 SmallVector<uint32_t, 2>{a, b, c});
                break;
            }
            case 8: {
                Command gp;
                gp.gate = GateType::GPhase;
                gp.params.push_back(angle());
                out.push_back(std::move(gp));
                break;
            }
            case 9: out.push_back(make_custom_1q(qubit())); break;
        }
    }

    void emit_unitary(std::vector<Command>& out) {
        switch (pick(6)) {
            case 0: out.emplace_back(one_q_gate(), qubit()); break;
            case 1: out.emplace_back(rotation_gate(), qubit(), angle()); break;
            case 2: {
                auto [a, b] = qubit_pair();
                out.emplace_back(two_q_gate(), a, b);
                break;
            }
            case 3: {
                auto [a, b] = qubit_pair();
                out.emplace_back(GateType::RZZ, a, b, angle());
                break;
            }
            case 4: {
                Command gp;
                gp.gate = GateType::GPhase;
                gp.params.push_back(angle());
                out.push_back(std::move(gp));
                break;
            }
            case 5: out.push_back(make_custom_1q(qubit())); break;
        }
    }

    void emit_full(std::vector<Command>& out, int region_budget) {
        switch (pick(region_budget > 0 ? 12 : 11)) {
            case 0: out.emplace_back(one_q_gate(), qubit()); break;
            case 1: out.emplace_back(rotation_gate(), qubit(), angle()); break;
            case 2: out.emplace_back(GateType::Rx, qubit(),
                                     Param::symbol("s" + std::to_string(pick(3))));
                    break;
            case 3: {
                auto [a, b] = qubit_pair();
                out.emplace_back(two_q_gate(), a, b);
                break;
            }
            case 4: out.push_back(make_measure(qubit(), cbit())); break;
            case 5: out.emplace_back(GateType::Reset, qubit()); break;
            case 6: {
                Command barrier;
                barrier.gate = GateType::Barrier;
                for (uint32_t q = 0; q < n_qubits_; ++q)
                    if (pick(2) == 0) barrier.qubits.push_back(q);
                out.push_back(std::move(barrier));
                break;
            }
            case 7: {
                Command gp;
                gp.gate = GateType::GPhase;
                gp.params.push_back(angle());
                out.push_back(std::move(gp));
                break;
            }
            case 8: out.push_back(make_custom_1q(qubit())); break;
            case 9: out.push_back(make_conditional(
                        Command(one_q_gate(), qubit()), {cbit()}, {pick(2) == 0}));
                    break;
            case 10: {
                auto [a, b] = qubit_pair();
                out.emplace_back(GateType::RZZ, a, b, angle());
                break;
            }
            case 11: {  // branch region, possibly nested
                out.push_back(make_branch_begin({cbit()}, {pick(2) == 0}));
                const std::size_t body = 1 + pick(3);
                for (std::size_t i = 0; i < body; ++i)
                    emit_full(out, region_budget - 1);
                if (pick(2) == 0) {
                    out.push_back(make_marker(GateType::BranchElse));
                    out.emplace_back(one_q_gate(), qubit());
                }
                out.push_back(make_marker(GateType::BranchEnd));
                break;
            }
        }
    }

    uint32_t pick(uint32_t n) {
        return std::uniform_int_distribution<uint32_t>(0, n - 1)(rng_);
    }
    uint32_t qubit() { return pick(n_qubits_); }
    uint32_t cbit()  { return pick(n_cbits_); }
    std::pair<uint32_t, uint32_t> qubit_pair() {
        const uint32_t a = qubit();
        uint32_t b = qubit();
        while (b == a) b = qubit();
        return {a, b};
    }
    std::tuple<uint32_t, uint32_t, uint32_t> qubit_triple() {
        const auto [a, b] = qubit_pair();
        uint32_t c = qubit();
        while (c == a || c == b) c = qubit();
        return {a, b, c};
    }
    /// Occasionally exactly zero — exercises the zero-rotation drop rule.
    Param angle() {
        if (pick(6) == 0) return Param(0.0);
        return Param(std::uniform_real_distribution<double>(-3.14, 3.14)(rng_));
    }
    GateType one_q_gate() {
        static constexpr GateType kGates[] = {GateType::H, GateType::X, GateType::Y,
                                              GateType::Z, GateType::S, GateType::T};
        return kGates[pick(6)];
    }
    GateType rotation_gate() {
        static constexpr GateType kGates[] = {GateType::Rx, GateType::Ry, GateType::Rz};
        return kGates[pick(3)];
    }
    GateType two_q_gate() {
        static constexpr GateType kGates[] = {GateType::CX, GateType::CZ, GateType::SWAP};
        return kGates[pick(3)];
    }

    std::mt19937 rng_;
    uint32_t n_qubits_;
    uint32_t n_cbits_;
    CircuitGrammar grammar_;
};

}  // namespace qarpx::test
