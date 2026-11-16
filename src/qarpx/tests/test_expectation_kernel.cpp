// P3.1 grouped single-pass Pauli kernel (`pauli_transition`).  Oracle: the
// explicit per-index action of each Pauli string on amplitudes (bit q of the
// index is qubit q, §1) — scalar code that shares nothing with the kernel's
// mask grouping, blocking or sign table.

#include <gtest/gtest.h>
#include "qarpx/qarpx.h"
#include "qarpx/simulator/pauli_expectation.h"

#include <complex>
#include <random>
#include <string>
#include <vector>

using namespace qarpx;

namespace {

using cd = std::complex<double>;

// ⟨bra|P|ket⟩ by explicit action: X flips bit q, Z signs it, Y = i·X·Z.
cd explicit_transition(const std::vector<cd>& bra, const std::vector<cd>& ket,
                       const std::vector<std::pair<uint32_t, char>>& pauli) {
    cd acc{0.0, 0.0};
    for (std::size_t i = 0; i < ket.size(); ++i) {
        std::size_t j = i;
        cd phase{1.0, 0.0};
        for (const auto& [q, op] : pauli) {
            const bool bit = (i >> q) & 1U;
            if (op == 'X') {
                j ^= (std::size_t{1} << q);
            } else if (op == 'Y') {
                j ^= (std::size_t{1} << q);
                phase *= bit ? cd{0.0, -1.0} : cd{0.0, 1.0};
            } else if (op == 'Z') {
                if (bit) phase = -phase;
            }
        }
        acc += std::conj(bra[j]) * phase * ket[i];
    }
    return acc;
}

cd explicit_transition(const std::vector<cd>& bra, const std::vector<cd>& ket,
                       const PauliObservable& obs) {
    cd acc{0.0, 0.0};
    for (const auto& [pauli, coeff] : obs) acc += coeff * explicit_transition(bra, ket, pauli);
    return acc;
}

std::vector<cd> random_state(std::mt19937_64& rng, int n) {
    std::normal_distribution<double> nd;
    std::vector<cd> psi(std::size_t{1} << n);
    double norm = 0.0;
    for (auto& a : psi) { a = cd{nd(rng), nd(rng)}; norm += std::norm(a); }
    for (auto& a : psi) a /= std::sqrt(norm);
    return psi;
}

// Random ≤4-local strings; complex coefficients so nothing is Hermitian.
PauliObservable random_observable(std::mt19937_64& rng, int n, int n_terms) {
    std::normal_distribution<double> nd;
    std::uniform_int_distribution<int> loc(1, std::min(4, n));
    std::uniform_int_distribution<int> which(0, 2);
    std::uniform_int_distribution<int> qubit(0, n - 1);
    PauliObservable obs;
    for (int t = 0; t < n_terms; ++t) {
        std::vector<std::pair<uint32_t, char>> pauli;
        uint64_t used = 0;
        const int k = loc(rng);
        while (static_cast<int>(pauli.size()) < k) {
            const int q = qubit(rng);
            if (used & (uint64_t{1} << q)) continue;
            used |= uint64_t{1} << q;
            pauli.emplace_back(static_cast<uint32_t>(q), "XYZ"[which(rng)]);
        }
        obs.emplace_back(std::move(pauli), cd{nd(rng), nd(rng)});
    }
    return obs;
}

}  // namespace

TEST(PauliTransition, MatchesExplicitActionBelowAndAboveOneBlock) {
    std::mt19937_64 rng(1);
    for (int n : {0, 1, 2, 5, 6, 8, 11}) {
        const auto obs = n == 0 ? PauliObservable{{{}, cd{1.5, -0.5}}}
                                : random_observable(rng, n, 50);
        const auto bra = random_state(rng, n), ket = random_state(rng, n);
        const cd got  = pauli_transition(bra.data(), ket.data(), n, obs);
        const cd want = explicit_transition(bra, ket, obs);
        EXPECT_NEAR(std::abs(got - want), 0.0, 1e-12) << "n = " << n;
    }
}

TEST(PauliTransition, HermitianExpectationIsReal) {
    std::mt19937_64 rng(2);
    const int n = 7;
    auto obs = random_observable(rng, n, 40);
    for (auto& term : obs) term.second = cd{term.second.real(), 0.0};
    const auto psi = random_state(rng, n);
    const cd got = pauli_transition(psi.data(), psi.data(), n, obs);
    EXPECT_NEAR(got.imag(), 0.0, 1e-12);
    EXPECT_NEAR(got.real(), explicit_transition(psi, psi, obs).real(), 1e-12);
}

TEST(PauliTransition, MixedYParityWithinOneFlipGroup) {
    // Same flip mask {0,1}; one term has one Y, the other two — the i^{n_Y}
    // factor must be applied per term.
    std::mt19937_64 rng(3);
    const int n = 6;
    const PauliObservable obs = {{{{0, 'Y'}, {1, 'X'}}, cd{0.7, 0.0}},
                                 {{{0, 'Y'}, {1, 'Y'}}, cd{-1.3, 0.0}}};
    const auto bra = random_state(rng, n), ket = random_state(rng, n);
    EXPECT_NEAR(std::abs(pauli_transition(bra.data(), ket.data(), n, obs) -
                         explicit_transition(bra, ket, obs)),
                0.0, 1e-12);
}

TEST(PauliTransition, IdentityLetterAndEmptyObservable) {
    std::mt19937_64 rng(4);
    const int n = 6;
    const auto bra = random_state(rng, n), ket = random_state(rng, n);
    const PauliObservable with_i = {{{{2, 'I'}, {3, 'Z'}}, cd{1.0, 0.0}}};
    const PauliObservable without = {{{{3, 'Z'}}, cd{1.0, 0.0}}};
    EXPECT_NEAR(std::abs(pauli_transition(bra.data(), ket.data(), n, with_i) -
                         pauli_transition(bra.data(), ket.data(), n, without)),
                0.0, 1e-15);
    EXPECT_EQ(pauli_transition(bra.data(), ket.data(), n, {}), cd(0.0, 0.0));
}

TEST(PauliTransition, RejectsBadStrings) {
    std::vector<cd> psi(8, cd{0.0, 0.0});
    EXPECT_THROW((void)pauli_transition(psi.data(), psi.data(), 3, {{{{3, 'Z'}}, cd{1, 0}}}),
                 std::invalid_argument);
    EXPECT_THROW((void)pauli_transition(psi.data(), psi.data(), 3, {{{{0, 'X'}, {0, 'Z'}}, cd{1, 0}}}),
                 std::invalid_argument);
    EXPECT_THROW((void)pauli_transition(psi.data(), psi.data(), 3, {{{{0, 'Q'}}, cd{1, 0}}}),
                 std::invalid_argument);
    EXPECT_THROW((void)pauli_transition(psi.data(), psi.data(), -1, {}), std::invalid_argument);
}

// The simulator entry point forms 1 << n_qubits for its length check; the
// range must be rejected before that shift, not by it (UB outside [0, 62]).
TEST(QarpSimulatorExpectation, TransitionRejectsOutOfRangeWidth) {
    QarpSimulator sim;
    const std::vector<cd> one(1, cd{1.0, 0.0});
    const PauliObservable z0{{{{0, 'Z'}}, cd{1, 0}}};
    for (int n : {-1, 63, 64, 70}) {
        try {
            (void)sim.transition(one, one, n, z0);
            FAIL() << "n_qubits = " << n << " was accepted";
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("must be in [0, 62]"), std::string::npos)
                << "n_qubits = " << n << ": " << e.what();
        }
    }
}

TEST(QarpSimulatorExpectation, MatchesStatevectorContraction) {
    std::mt19937_64 rng(5);
    const int n = 6;
    const auto obs = random_observable(rng, n, 30);
    std::vector<Command> cmds;
    for (uint32_t q = 0; q < static_cast<uint32_t>(n); ++q) cmds.emplace_back(GateType::H, q);
    for (uint32_t q = 0; q + 1 < static_cast<uint32_t>(n); ++q) cmds.emplace_back(GateType::CX, q, q + 1);
    for (uint32_t q = 0; q < static_cast<uint32_t>(n); ++q)
        cmds.emplace_back(GateType::Rz, q, Param(0.3 * (q + 1)));
    QarpSimulator sim;
    const auto psi = sim.statevector(cmds, n);
    EXPECT_NEAR(sim.expectation(cmds, n, obs), explicit_transition(psi, psi, obs).real(), 1e-12);
    const auto both = sim.batch_expectation(cmds, n, obs, {{}, {}});
    ASSERT_EQ(both.size(), 2U);
    EXPECT_DOUBLE_EQ(both[0], both[1]);
    EXPECT_NEAR(both[0], sim.expectation(cmds, n, obs), 1e-12);
    EXPECT_THROW((void)sim.transition(psi, std::vector<cd>(4), n, obs), std::invalid_argument);
}
