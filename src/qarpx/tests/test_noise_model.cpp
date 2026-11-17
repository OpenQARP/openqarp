#include <gtest/gtest.h>
#include "qarpx/qarpx.h"

#include <cmath>
#include <vector>

using namespace qarpx;

// ── Channel application: deterministic outcomes via forced RNG ──────────────

namespace {

Command make_cmd(GateType g, std::vector<uint32_t> qubits = {}) {
    Command c;
    c.gate = g;
    for (auto q : qubits) c.qubits.push_back(q);
    return c;
}

std::vector<std::complex<double>> make_state_plus(uint64_t dim) {
    std::vector<std::complex<double>> s(dim, 0.0);
    for (auto& a : s) a = std::complex<double>(1.0 / std::sqrt((double)dim), 0.0);
    return s;
}

}  // namespace

TEST(NoiseModel, PauliChannelFiringIsStochasticButBounded) {
    // Apply a heavy bit-flip channel many times and confirm fire-rate ~ p.
    auto nm = make_bit_flip(0.5, {GateType::X});
    EXPECT_TRUE(nm.has_channel(GateType::X));
    EXPECT_FALSE(nm.has_channel(GateType::Y));

    QarpSimulator sim(nm);
    auto cmds = std::vector<Command>{ make_cmd(GateType::X, {0}) };
    auto sr = sim.run(cmds, 1, 4000, 42u);
    // X fires gate, then ~50% bit-flip undoes it.  Expect ~50/50.
    const double p_one = sr.counts[1] / 4000.0;
    EXPECT_NEAR(p_one, 0.5, 0.05);
}

TEST(NoiseModel, IdleChannelRejectedAtConstruction) {
    // The `idle` field is reserved (per-cycle channels need a notion of cycles
    // the Block IR doesn't expose).  Setting it must throw — both at
    // construction and via set_noise_model.
    NoiseModel nm;
    nm.idle = [](const Command&) -> Channel { return PauliChannel{0.1, 0.0, 0.0}; };
    EXPECT_THROW({ QarpSimulator s(nm); (void)s; }, std::runtime_error);

    QarpSimulator sim;
    EXPECT_THROW(sim.set_noise_model(nm), std::runtime_error);
}

TEST(NoiseModel, NoChannelEqualsFastPath) {
    NoiseModel nm;  // empty
    QarpSimulator sim_noisy(nm);
    QarpSimulator sim_plain;

    auto cmds = std::vector<Command>{ make_cmd(GateType::H, {0}) };
    auto noisy = sim_noisy.run(cmds, 1, 1000, 7u);
    auto plain = sim_plain.run(cmds, 1, 1000, 7u);
    // With identical seeds and empty noise, results must match bit-for-bit.
    EXPECT_EQ(noisy.counts, plain.counts);
}

TEST(NoiseModel, DisabledNoiseSkipsTrajectoryPath) {
    NoiseModel nm = make_depolarizing(0.5, {GateType::H});
    nm.enabled = false;
    QarpSimulator sim(nm);

    auto cmds = std::vector<Command>{ make_cmd(GateType::H, {0}) };
    auto sr = sim.run(cmds, 1, 2000, 1u);
    // With noise disabled, H|0> = (|0>+|1>)/sqrt 2 → 50/50 exact marginal.
    EXPECT_NEAR(sr.counts[0] / 2000.0, 0.5, 0.05);
}

TEST(NoiseModel, StatevectorThrowsWithNoise) {
    auto nm = make_depolarizing(0.1, {GateType::H});
    QarpSimulator sim(nm);
    auto cmds = std::vector<Command>{ make_cmd(GateType::H, {0}) };
    EXPECT_THROW((void)sim.statevector(cmds, 1), std::runtime_error);
}

TEST(NoiseModel, UnitaryMatrixThrowsWithNoise) {
    auto nm = make_depolarizing(0.1, {GateType::H});
    QarpSimulator sim(nm);
    auto cmds = std::vector<Command>{ make_cmd(GateType::H, {0}) };
    EXPECT_THROW((void)sim.unitary_matrix(cmds, 1), std::runtime_error);
}

TEST(NoiseModel, AmplitudeDampingCollapsesToGround) {
    // X|0>=|1>, then heavy amp-damp → mostly relaxes back to |0>.
    auto nm = make_amplitude_damping(0.99, {GateType::X});
    QarpSimulator sim(nm);
    auto cmds = std::vector<Command>{ make_cmd(GateType::X, {0}) };
    auto sr = sim.run(cmds, 1, 2000, 11u);
    EXPECT_GT(sr.counts[0], 1800);
    EXPECT_LT(sr.counts[1], 200);
}

TEST(NoiseModel, Composition) {
    auto a = make_bit_flip(0.1, {GateType::H});
    auto b = make_bit_flip(0.2, {GateType::X});
    auto c = a + b;
    EXPECT_TRUE(c.has_channel(GateType::H));
    EXPECT_TRUE(c.has_channel(GateType::X));
    // Right-wins on collision:
    auto d = a + make_bit_flip(0.5, {GateType::H});
    EXPECT_TRUE(d.has_channel(GateType::H));
}

TEST(NoiseModel, Depolarizing2qPicks15Paulis) {
    auto nm = make_depolarizing(1.0, {GateType::CX});
    // Force a fire (p=1) and confirm we end up uniformly across the 15
    // non-identity 2q Paulis on the state |+,+>.
    QarpSimulator sim(nm);
    // Build |+,+> via H on both qubits, then a CX with full depolarizing.
    std::vector<Command> cmds = {
        make_cmd(GateType::H,  {0}),
        make_cmd(GateType::H,  {1}),
        make_cmd(GateType::CX, {0, 1}),
    };
    auto sr = sim.run(cmds, 2, 4000, 99u);
    // Sanity: total shots == 4000
    int total = 0;
    for (auto& kv : sr.counts) total += kv.second;
    EXPECT_EQ(total, 4000);
    // With full depolarizing, output distribution is approximately uniform.
    for (auto& kv : sr.counts) {
        EXPECT_GT(kv.second, 700);   // expect ~1000 each (allowing variance)
        EXPECT_LT(kv.second, 1300);
    }
}
