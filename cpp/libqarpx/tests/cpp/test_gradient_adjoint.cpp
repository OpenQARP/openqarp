// Adjoint-gradient kernels: the shared affine probe, the compound-affine chain rule, and the
// never-silent guard.  Oracles are central finite differences of the same
// expectation value computed from `statevector` (LSB convention, §1) —
// an independent numeric path, never the kernel's own output.

#include <gtest/gtest.h>
#include "qarpx/qarpx.h"

#include <cmath>
#include <complex>
#include <string>
#include <unordered_map>
#include <vector>

using namespace qarpx;

namespace {

using cd = std::complex<double>;
using Observable = std::vector<std::pair<std::vector<std::pair<uint32_t, char>>, cd>>;

// ⟨ψ|P|ψ⟩ for one Pauli string, by explicit action on amplitudes (bit q of
// the index is qubit q).
cd pauli_expectation(const std::vector<cd>& psi,
                     const std::vector<std::pair<uint32_t, char>>& pauli) {
    cd acc{0.0, 0.0};
    const std::size_t dim = psi.size();
    for (std::size_t i = 0; i < dim; ++i) {
        std::size_t j = i;
        cd phase{1.0, 0.0};
        for (const auto& [q, op] : pauli) {
            const bool bit = (i >> q) & 1U;
            if (op == 'X') {
                j ^= (std::size_t{1} << q);
            } else if (op == 'Y') {
                j ^= (std::size_t{1} << q);
                phase *= bit ? cd{0.0, -1.0} : cd{0.0, 1.0};
            } else {  // 'Z'
                if (bit) phase = -phase;
            }
        }
        acc += std::conj(psi[j]) * phase * psi[i];
    }
    return acc;
}

double energy(const QarpSimulator& sim, const std::vector<Command>& cmds, int n,
              const Observable& obs, const std::unordered_map<std::string, double>& values) {
    const auto psi = sim.statevector(substitute_all(cmds, values), n);
    cd e{0.0, 0.0};
    for (const auto& [pauli, coeff] : obs) e += coeff * pauli_expectation(psi, pauli);
    return e.real();
}

std::vector<double> central_fd(const QarpSimulator& sim, const std::vector<Command>& cmds, int n,
                               const Observable& obs,
                               std::unordered_map<std::string, double> values,
                               const std::vector<std::string>& order, double h = 1e-6) {
    std::vector<double> g;
    for (const auto& s : order) {
        auto up = values, dn = values;
        up[s] += h;
        dn[s] -= h;
        g.push_back((energy(sim, cmds, n, obs, up) - energy(sim, cmds, n, obs, dn)) / (2 * h));
    }
    return g;
}

}  // namespace

// ── affine_coefficients ───────────────────────────────────────────────────────

TEST(AffineCoefficients, LinearAndCompoundAffine) {
    auto lin = affine_coefficients(Param::linear(-2.0, "x", 0.3));
    ASSERT_EQ(lin.size(), 1U);
    EXPECT_NEAR(lin.at("x"), -2.0, 1e-12);

    const Param half_sum = (Param::symbol("a") + Param::symbol("b")) * Param(0.5);
    auto comp = affine_coefficients(half_sum);
    ASSERT_EQ(comp.size(), 2U);
    EXPECT_NEAR(comp.at("a"), 0.5, 1e-12);
    EXPECT_NEAR(comp.at("b"), 0.5, 1e-12);

    EXPECT_TRUE(affine_coefficients(Param(1.25)).empty());
}

TEST(AffineCoefficients, NonAffineThrowsCapabilityError) {
    const Param prod = Param::symbol("a") * Param::symbol("b");
    EXPECT_THROW((void)affine_coefficients(prod), capability_error);
    const Param sq = Param::symbol("t") * Param::symbol("t");
    EXPECT_THROW((void)affine_coefficients(sq), capability_error);
}

TEST(AffineCoefficients, NonFiniteProbeThrowsCapabilityError) {
    // Every `abs(...) > tol` comparison is false for inf/NaN, so without an
    // explicit finiteness check an overflowing angle came back with an
    // infinite coefficient.
    const Param overflow = Param(1e200) * Param::symbol("x") * Param(1e200);
    EXPECT_THROW((void)affine_coefficients(overflow), capability_error);
    // 1/x cannot be evaluated at the x = 0 probe: Param's own division-by-zero
    // (a std::runtime_error) is retyped to the same refusal.
    const Param inv = Param(1.0) / Param::symbol("x");
    EXPECT_THROW((void)affine_coefficients(inv), capability_error);
}

// ── adjoint vs finite differences ─────────────────────────────────────────────

TEST(AdjointGradient, SharedAndCompoundAnglesMatchFiniteDifferences) {
    // H(0); Ry(1, 0.4); CRx(0,1; θ); Ry(0; (a+b)/2); Rz(1; -2θ + 0.3)
    // X0 on the control keeps CRx's half-frequency terms alive.
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::H, 0U);
    cmds.emplace_back(GateType::Ry, 1U, Param(0.4));
    cmds.emplace_back(GateType::CRx, 0U, 1U, Param::symbol("theta"));
    cmds.emplace_back(GateType::Ry, 0U, (Param::symbol("a") + Param::symbol("b")) * Param(0.5));
    cmds.emplace_back(GateType::Rz, 1U, Param::linear(-2.0, "theta", 0.3));
    cmds.emplace_back(GateType::CX, 0U, 1U);

    const Observable obs{{{{0U, 'X'}}, cd{1.0, 0.0}},
                         {{{1U, 'Z'}}, cd{0.7, 0.0}},
                         {{{0U, 'Y'}, {1U, 'Z'}}, cd{0.4, 0.0}}};
    const std::unordered_map<std::string, double> values{{"theta", 0.7}, {"a", 0.3}, {"b", -0.9}};
    const std::vector<std::string> order{"theta", "a", "b"};

    QarpSimulator sim;
    const auto grad = sim.run_gradient(cmds, 2, obs, values, order);
    const auto fd = central_fd(sim, cmds, 2, obs, values, order);
    ASSERT_EQ(grad.size(), 3U);
    for (std::size_t k = 0; k < 3; ++k) EXPECT_NEAR(grad[k], fd[k], 1e-7) << order[k];
    for (std::size_t k = 0; k < 3; ++k) EXPECT_GT(std::abs(fd[k]), 1e-3) << "degenerate oracle";
}

TEST(AdjointGradient, CompoundGPhaseAndPhaseMatchFiniteDifferences) {
    // The shape decompose_u / decompose_cu emit: GPhase((φ+λ)/2) and
    // P((φ+λ)/2) next to a CRx(θ), on three qubits.  The GPhase must be
    // probed as a compound param without throwing and contribute nothing;
    // the P carries the same compound angle and does contribute.
    const Param half = (Param::symbol("phi") + Param::symbol("lam")) * Param(0.5);
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::H, 0U);
    cmds.emplace_back(GateType::Ry, 1U, Param(0.4));
    cmds.emplace_back(GateType::Rx, 2U, Param(0.7));
    cmds.emplace_back(GateType::CRx, 0U, 1U, Param::symbol("theta"));
    Command gphase;
    gphase.gate = GateType::GPhase;
    gphase.params.push_back(half);
    cmds.push_back(gphase);
    cmds.emplace_back(GateType::P, 2U, half);
    cmds.emplace_back(GateType::CX, 1U, 2U);
    cmds.emplace_back(GateType::Rz, 0U, Param::symbol("lam"));
    cmds.emplace_back(GateType::H, 2U);

    const Observable obs{{{{0U, 'X'}}, cd{1.0, 0.0}},
                         {{{1U, 'Z'}, {2U, 'Y'}}, cd{0.7, 0.0}},
                         {{{0U, 'Y'}, {2U, 'X'}}, cd{0.4, 0.0}}};
    const std::unordered_map<std::string, double> values{
        {"theta", 0.7}, {"phi", 0.3}, {"lam", -0.9}};
    const std::vector<std::string> order{"theta", "phi", "lam"};

    QarpSimulator sim;
    const auto grad = sim.run_gradient(cmds, 3, obs, values, order);
    const auto fd = central_fd(sim, cmds, 3, obs, values, order);
    ASSERT_EQ(grad.size(), 3U);
    for (std::size_t k = 0; k < 3; ++k) EXPECT_NEAR(grad[k], fd[k], 1e-7) << order[k];
    for (std::size_t k = 0; k < 3; ++k) EXPECT_GT(std::abs(fd[k]), 1e-3) << "degenerate oracle";
}

TEST(AdjointGradient, GPhaseContributesNothingToExpectationValues) {
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::Ry, 0U, Param::symbol("t"));
    Command gphase;
    gphase.gate = GateType::GPhase;
    gphase.params.push_back(Param::symbol("t"));
    cmds.push_back(gphase);
    const Observable obs{{{{0U, 'Z'}}, cd{1.0, 0.0}}};
    QarpSimulator sim;
    const auto grad = sim.run_gradient(cmds, 1, obs, {{"t", 0.6}}, {"t"});
    EXPECT_NEAR(grad[0], -std::sin(0.6), 1e-12);  // ⟨Z⟩ = cos t, phase drops out
}

TEST(AdjointGradient, SymbolicGateWithoutGeneratorRuleIsLoudNeverZero) {
    // U(θ,φ,λ) reaches the sweep untouched (it is native); the old kernel
    // returned [0, 0, 0] here.
    std::vector<Command> cmds;
    cmds.emplace_back(GateType::Ry, 0U, Param(0.3));
    Command u;
    u.gate = GateType::U;
    u.qubits.push_back(0U);
    u.params.push_back(Param::symbol("th"));
    u.params.push_back(Param::symbol("ph"));
    u.params.push_back(Param::symbol("la"));
    cmds.push_back(u);
    const Observable obs{{{{0U, 'X'}}, cd{1.0, 0.0}}};
    QarpSimulator sim;
    EXPECT_THROW(
        (void)sim.run_gradient(cmds, 1, obs, {{"th", 0.4}, {"ph", 0.9}, {"la", -0.6}},
                               {"th", "ph", "la"}),
        capability_error);

    // A concrete U is fine: it is just a fixed gate in the sweep.
    Command uc = u;
    uc.params = {Param(0.4), Param(0.9), Param(-0.6)};
    std::vector<Command> ok{cmds[0], uc};
    ok.emplace_back(GateType::Rx, 0U, Param::symbol("t"));
    const auto grad = sim.run_gradient(ok, 1, obs, {{"t", 0.2}}, {"t"});
    const auto fd = central_fd(sim, ok, 1, obs, {{"t", 0.2}}, {"t"});
    EXPECT_NEAR(grad[0], fd[0], 1e-7);
}
