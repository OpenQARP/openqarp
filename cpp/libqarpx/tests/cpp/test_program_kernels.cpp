// ── Structured programs (program.h: kernels, derivation, executor) ──
//
// Oracles: explicit index arithmetic for the permutation kernels, direct
// matrix products on gathered sub-vectors for the dense and controlled-power
// kernels, and tests/cpp/reference_unitary.h (analytic gate definitions,
// sharing no code with csim) for the tables derived from gate streams.

#include "reference_unitary.h"

#include "qarpx/simulator/program.h"
#include "qarpx/simulator/qarp_simulator.h"

#include <gtest/gtest.h>

#include <complex>
#include <numeric>
#include <random>
#include <vector>

namespace qarpx::test {

namespace {

using cd  = std::complex<double>;
using Mat = Eigen::MatrixXcd;

constexpr double kTol = 1e-12;

std::vector<cd> random_state(int n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> g;
    std::vector<cd> psi(uint64_t{1} << n);
    double norm = 0.0;
    for (auto& a : psi) { a = {g(rng), g(rng)}; norm += std::norm(a); }
    for (auto& a : psi) a /= std::sqrt(norm);
    return psi;
}

Mat random_unitary(int k, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> g;
    const Eigen::Index d = Eigen::Index{1} << k;
    Mat a(d, d);
    for (Eigen::Index i = 0; i < d; ++i)
        for (Eigen::Index j = 0; j < d; ++j) a(i, j) = {g(rng), g(rng)};
    Eigen::HouseholderQR<Mat> qr(a);
    return qr.householderQ();
}

std::vector<uint64_t> random_table(int k, uint32_t seed) {
    std::vector<uint64_t> t(uint64_t{1} << k);
    std::iota(t.begin(), t.end(), uint64_t{0});
    std::shuffle(t.begin(), t.end(), std::mt19937(seed));
    return t;
}

uint64_t local_of(uint64_t i, const std::vector<uint32_t>& qs) {
    uint64_t l = 0;
    for (std::size_t b = 0; b < qs.size(); ++b) l |= ((i >> qs[b]) & 1) << b;
    return l;
}

uint64_t with_local(uint64_t i, uint64_t l, const std::vector<uint32_t>& qs) {
    for (std::size_t b = 0; b < qs.size(); ++b) {
        i &= ~(uint64_t{1} << qs[b]);
        i |= ((l >> b) & 1) << qs[b];
    }
    return i;
}

// out[with_local(i, t[local(i)])] = in[i]
std::vector<cd> permuted(const std::vector<cd>& in, const std::vector<uint32_t>& qs,
                         const std::vector<uint64_t>& t) {
    std::vector<cd> out(in.size());
    for (uint64_t i = 0; i < in.size(); ++i) out[with_local(i, t[local_of(i, qs)], qs)] = in[i];
    return out;
}

// M applied to the sub-vector on qs wherever `select(i)`.
std::vector<cd> applied(const std::vector<cd>& in, const Mat& M, const std::vector<uint32_t>& qs,
                        uint64_t select_mask = 0) {
    std::vector<cd> out(in);
    const uint64_t d = uint64_t{1} << qs.size();
    for (uint64_t i = 0; i < in.size(); ++i) {
        if (local_of(i, qs) != 0 || (i & select_mask) != select_mask) continue;
        for (uint64_t r = 0; r < d; ++r) {
            cd acc = 0.0;
            for (uint64_t c = 0; c < d; ++c)
                acc += M(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c))
                       * in[with_local(i, c, qs)];
            out[with_local(i, r, qs)] = acc;
        }
    }
    return out;
}

Command measure(uint32_t qubit, uint32_t cbit) {
    return Command(GateType::Measure, SmallVector<uint32_t, 2>{qubit}, SmallVector<Param, 1>{},
                   SmallVector<uint32_t, 1>{cbit});
}

double max_diff(const std::vector<cd>& a, const std::vector<cd>& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

// The permutation a unitary implements, or empty if it is not one (phase included).
std::vector<uint64_t> permutation_of(const Mat& U) {
    std::vector<uint64_t> t(static_cast<std::size_t>(U.cols()));
    for (Eigen::Index c = 0; c < U.cols(); ++c) {
        Eigen::Index r = 0;
        U.col(c).cwiseAbs().maxCoeff(&r);
        if (std::abs(U(r, c) - cd{1.0, 0.0}) > 1e-9) return {};
        t[static_cast<std::size_t>(c)] = static_cast<uint64_t>(r);
    }
    return t;
}

}  // namespace

TEST(ProgramKernels, PermutationOnQubitSubsetMatchesIndexArithmetic) {
    const int n = 14;  // above the csim fork threshold
    const auto psi = random_state(n, 1);
    const std::vector<uint32_t> qs = {3, 11, 0, 7};
    const auto t = random_table(4, 2);
    Program p;
    p.add_permutation(qs, t);
    const auto out = QarpSimulator().program_statevector(p, n, psi);
    EXPECT_EQ(max_diff(out, permuted(psi, qs, t)), 0.0);
}

TEST(ProgramKernels, WholeRegisterPermutationMatchesIndexArithmetic) {
    const int n = 10;
    const auto psi = random_state(n, 3);
    std::vector<uint32_t> qs(n);
    std::iota(qs.begin(), qs.end(), 0u);
    const auto t = random_table(n, 4);
    Program p;
    p.add_permutation(qs, t);
    EXPECT_EQ(max_diff(QarpSimulator().program_statevector(p, n, psi), permuted(psi, qs, t)), 0.0);
}

TEST(ProgramKernels, AdjacentPermutationsComposeIntoOneKernel) {
    const int n = 8;
    const auto psi = random_state(n, 5);
    const std::vector<uint32_t> q1 = {1, 4, 6}, q2 = {6, 0};
    const auto t1 = random_table(3, 6), t2 = random_table(2, 7);
    Program p;
    p.add_permutation(q1, t1);
    p.add_permutation(q2, t2);
    EXPECT_EQ(p.kinds(), std::vector<std::string>{"permutation"});
    const auto expected = permuted(permuted(psi, q1, t1), q2, t2);
    EXPECT_EQ(max_diff(QarpSimulator().program_statevector(p, n, psi), expected), 0.0);
}

TEST(ProgramKernels, DenseKernelMatchesMatrixProduct) {
    const int n = 9;
    const auto psi = random_state(n, 8);
    const std::vector<uint32_t> qs = {5, 2, 7};
    const Mat U = random_unitary(3, 9);
    Program p;
    p.add_dense(qs, U);
    EXPECT_LT(max_diff(QarpSimulator().program_statevector(p, n, psi), applied(psi, U, qs)), kTol);
}

TEST(ProgramKernels, ControlledPowersApplyEachExponentUnderItsControl) {
    const int n = 14;
    const auto psi = random_state(n, 10);
    const std::vector<uint32_t> targets = {4, 9};
    const Mat V = random_unitary(2, 11);
    Program p;
    p.add_controlled_powers({0, 13}, {1, 5}, targets, V);
    Mat V5 = Mat::Identity(4, 4);
    for (int i = 0; i < 5; ++i) V5 = V5 * V;
    auto expected = applied(psi, V, targets, uint64_t{1} << 0);
    expected = applied(expected, V5, targets, uint64_t{1} << 13);
    EXPECT_LT(max_diff(QarpSimulator().program_statevector(p, n, psi), expected), 1e-10);
}

TEST(ProgramKernels, ControlledPowersShareSquaresAcrossExponents) {
    // Powers of two, sums of them, a repeat and a zero: every control must
    // apply exactly its own power.
    const int n = 10;
    const auto psi = random_state(n, 30);
    const std::vector<uint32_t> targets = {7, 2};
    const std::vector<uint32_t> controls = {0, 1, 3, 4, 5, 6, 8, 9};
    const std::vector<uint64_t> exponents = {1, 8, 6, 0, 13, 6, 4, 2};
    const Mat V = random_unitary(2, 31);
    Program p;
    p.add_controlled_powers(controls, exponents, targets, V);
    auto expected = psi;
    for (std::size_t j = 0; j < controls.size(); ++j) {
        Mat power = Mat::Identity(4, 4);
        for (uint64_t i = 0; i < exponents[j]; ++i) power = power * V;
        expected = applied(expected, power, targets, uint64_t{1} << controls[j]);
    }
    EXPECT_LT(max_diff(QarpSimulator().program_statevector(p, n, psi), expected), 1e-10);
}

TEST(ProgramKernels, InvalidKernelsAreRejectedOnInsertion) {
    Program p;
    EXPECT_THROW(p.add_permutation({0, 1}, {0, 0, 1, 2}), std::invalid_argument);
    EXPECT_THROW(p.add_permutation({0, 0}, {0, 1, 2, 3}), std::invalid_argument);
    EXPECT_THROW(p.add_permutation({0}, {0, 1, 2}), std::invalid_argument);
    EXPECT_THROW(p.add_dense({0, 1}, Mat::Identity(2, 2)), std::invalid_argument);
    EXPECT_THROW(p.add_controlled_powers({0}, {1}, {0, 1}, Mat::Identity(4, 4)),
                 std::invalid_argument);
    EXPECT_THROW(p.add_controlled_powers({0, 1}, {1}, {2}, Mat::Identity(2, 2)),
                 std::invalid_argument);
}

TEST(ProgramKernels, ClassicalGatesGiveTheirReferencePermutation) {
    // X, CX, CCX, SWAP, CSWAP and the H·MCZ·H that `mcx` emits.
    std::vector<Command> cmds = {
        Command(GateType::X, 1),
        Command(GateType::CX, 1, 3),
        Command(GateType::CCX, {0, 3, 2}),
        Command(GateType::SWAP, 0, 2),
        Command(GateType::CSWAP, {3, 1, 0}),
        Command(GateType::H, 2),
        Command(GateType::MCZ, {0, 1, 3, 2}),
        Command(GateType::H, 2),
    };
    const std::vector<uint32_t> qs = {0, 1, 2, 3};
    const auto t = permutation_table(cmds, qs, 0, 0);
    ASSERT_TRUE(t.has_value());
    EXPECT_EQ(*t, permutation_of(reference_unitary(cmds, 4)));
}

TEST(ProgramKernels, QftSandwichAdderDerivesItsReferencePermutation) {
    // Draper adder x -> x + 3 mod 8: QFT, phases, inverse QFT — not
    // classical gate by gate, a permutation as a whole.  The QFT has no final
    // swaps, so the phase for bit j lands on qubit k-1-j.
    const int k = 3;
    std::vector<Command> qft;
    for (int j = k - 1; j >= 0; --j) {
        qft.emplace_back(GateType::H, static_cast<uint32_t>(j));
        for (int m = j - 1; m >= 0; --m)
            qft.emplace_back(GateType::CP, static_cast<uint32_t>(m), static_cast<uint32_t>(j),
                             Param(M_PI / static_cast<double>(1 << (j - m))));
    }
    std::vector<Command> cmds(qft);
    for (int j = 0; j < k; ++j)
        cmds.emplace_back(GateType::P, static_cast<uint32_t>(j),
                          Param(2.0 * M_PI * 3.0 * static_cast<double>(1 << (k - 1 - j)) / 8.0));
    for (auto it = qft.rbegin(); it != qft.rend(); ++it) cmds.push_back(it->dagger());

    const std::vector<uint32_t> qs = {0, 1, 2};
    const auto reference = permutation_of(reference_unitary(cmds, k));
    ASSERT_FALSE(reference.empty()) << "the fixture must be a permutation";
    EXPECT_FALSE(permutation_table(cmds, qs, 0, 0).has_value()) << "not classical gate by gate";
    const auto t = permutation_table(cmds, qs, 3, 64);
    ASSERT_TRUE(t.has_value());
    EXPECT_EQ(*t, reference);
}

TEST(ProgramKernels, RelativePhaseIsNeverAPermutation) {
    const std::vector<Command> xz = {Command(GateType::X, 0), Command(GateType::Z, 0)};
    const std::vector<Command> cz = {Command(GateType::CZ, 0, 1)};
    EXPECT_FALSE(permutation_table(xz, {0}, 8, 64).has_value());
    EXPECT_FALSE(permutation_table(cz, {0, 1}, 8, 64).has_value());
    EXPECT_TRUE(permutation_of(reference_unitary(xz, 1)).empty());
}

TEST(ProgramKernels, GatesOnlyProgramSamplesLikeRun) {
    const int n = 6;
    const std::vector<Command> cmds = {
        Command(GateType::H, 0), Command(GateType::CX, 0, 3), Command(GateType::Ry, 4, Param(0.7)),
        measure(3, 0), measure(4, 1),
    };
    Program p;
    p.add_gates(cmds);
    const QarpSimulator sim;
    const auto a = sim.run(cmds, n, 500, 42u);
    const auto b = sim.program_run(p, n, 500, 42u);
    ASSERT_EQ(a.n_cbits, 2u);
    EXPECT_EQ(a.counts, b.counts);
    EXPECT_EQ(a.cbit_history, b.cbit_history);
    EXPECT_EQ(a.n_cbits, b.n_cbits);
}

namespace {

// QFT on `reg` (reg[0] least significant) without the final swaps.
std::vector<Command> qft_on(const std::vector<uint32_t>& reg) {
    std::vector<Command> out;
    const int k = static_cast<int>(reg.size());
    for (int j = k - 1; j >= 0; --j) {
        out.emplace_back(GateType::H, reg[static_cast<std::size_t>(j)]);
        for (int m = j - 1; m >= 0; --m)
            out.emplace_back(GateType::CP, reg[static_cast<std::size_t>(m)],
                             reg[static_cast<std::size_t>(j)],
                             Param(M_PI / static_cast<double>(1 << (j - m))));
    }
    return out;
}

// x_reg -> x_reg + a mod 2^k wherever `ctrl` is |1> (a Draper adder whose
// phases are controlled); no control means unconditional.
std::vector<Command> controlled_add(const std::vector<uint32_t>& reg,
                                    std::optional<uint32_t> ctrl, int a) {
    const int k = static_cast<int>(reg.size());
    const auto q = qft_on(reg);
    std::vector<Command> out(q);
    for (int j = 0; j < k; ++j) {
        const double angle = 2.0 * M_PI * a * static_cast<double>(1 << (k - 1 - j))
                             / static_cast<double>(1 << k);
        const uint32_t target = reg[static_cast<std::size_t>(j)];
        if (ctrl) out.emplace_back(GateType::CP, *ctrl, target, Param(angle));
        else out.emplace_back(GateType::P, target, Param(angle));
    }
    for (auto it = q.rbegin(); it != q.rend(); ++it) out.push_back(it->dagger());
    return out;
}

}  // namespace

TEST(ProgramKernels, ControlledAddersDeriveWithOnlyTheRegisterSimulated) {
    // Three control qubits, each adding a different constant to a 3-qubit
    // register: six qubits touched, three changed.
    const std::vector<uint32_t> reg = {0, 1, 2};
    std::vector<Command> cmds;
    for (auto [ctrl, a] : {std::pair<uint32_t, int>{3, 1}, {4, 3}, {5, 6}}) {
        auto add = controlled_add(reg, ctrl, a);
        cmds.insert(cmds.end(), add.begin(), add.end());
    }
    const std::vector<uint32_t> qs = {0, 1, 2, 3, 4, 5};
    const auto reference = permutation_of(reference_unitary(cmds, 6));
    ASSERT_FALSE(reference.empty()) << "the fixture must be a permutation";
    for (uint64_t v = 0; v < 64; ++v) {
        const uint64_t shift = ((v >> 3) & 1) * 1 + ((v >> 4) & 1) * 3 + ((v >> 5) & 1) * 6;
        ASSERT_EQ(reference[v], (v & ~uint64_t{7}) | (((v & 7) + shift) % 8));
    }
    // 2^3 assignments x 4^3 columns: within a budget far below 4^6.
    const auto t = permutation_table(cmds, qs, 3, 9);
    ASSERT_TRUE(t.has_value());
    EXPECT_EQ(*t, reference);
}

TEST(ProgramKernels, RandomCircuitsDeriveExactlyTheirReferencePermutation) {
    // Adders, classical gates and phases that may or may not cancel: every
    // result must be the reference permutation, or nothing when the
    // reference is not one.
    std::mt19937 rng(20260928);
    const std::vector<uint32_t> qs = {0, 1, 2, 3, 4};
    int positives = 0, negatives = 0;
    for (int trial = 0; trial < 200; ++trial) {
        std::vector<Command> cmds;
        const int blocks = 2 + static_cast<int>(rng() % 3);
        for (int b = 0; b < blocks; ++b) {
            switch (rng() % 7) {
                case 0: {
                    auto add = controlled_add({0, 1, 2}, 3 + static_cast<uint32_t>(rng() % 2),
                                              1 + static_cast<int>(rng() % 7));
                    cmds.insert(cmds.end(), add.begin(), add.end());
                    break;
                }
                case 1:
                    cmds.emplace_back(GateType::CCX, SmallVector<uint32_t, 2>{3, 4, rng() % 3});
                    break;
                case 2:
                    cmds.emplace_back(GateType::SWAP, 3, 4);
                    break;
                case 3: {
                    // A multi-controlled X written with a CP, so the stream is
                    // not classical gate by gate and MCZ meets fixed controls.
                    const uint32_t t = rng() % 3;
                    cmds.emplace_back(GateType::H, t);
                    cmds.emplace_back(GateType::MCZ, SmallVector<uint32_t, 2>{3, 4, t});
                    cmds.emplace_back(GateType::H, t);
                    cmds.emplace_back(GateType::CP, 3, 4, Param(M_PI));
                    cmds.emplace_back(GateType::CP, 3, 4, Param(-M_PI));
                    break;
                }
                case 4: {
                    // A classical gate driven by a qubit the adders change.
                    auto add = controlled_add({0, 1, 2}, 3, 1 + static_cast<int>(rng() % 7));
                    cmds.insert(cmds.end(), add.begin(), add.end());
                    cmds.emplace_back(GateType::CX, rng() % 3, 4);
                    break;
                }
                case 5: {
                    const double theta = 0.4 + 0.1 * static_cast<double>(rng() % 5);
                    cmds.emplace_back(GateType::GPhase, SmallVector<uint32_t, 2>{},
                                      SmallVector<Param, 1>{Param(theta)});
                    auto add = controlled_add({0, 1, 2}, std::nullopt, 1 + static_cast<int>(rng() % 7));
                    cmds.insert(cmds.end(), add.begin(), add.end());
                    if (rng() % 2)
                        cmds.emplace_back(GateType::GPhase, SmallVector<uint32_t, 2>{},
                                          SmallVector<Param, 1>{Param(-theta)});
                    break;
                }
                default: {
                    const double theta = (rng() % 2) ? M_PI : 0.5;
                    cmds.emplace_back(GateType::CP, 3, 4, Param(theta));
                    cmds.emplace_back(GateType::CP, 3, 4, Param(-theta));
                    if (rng() % 3 == 0) cmds.emplace_back(GateType::Rz, rng() % 5, Param(0.3));
                    break;
                }
            }
        }
        const auto reference = permutation_of(reference_unitary(cmds, 5));
        const auto t = permutation_table(cmds, qs, 5, 64);
        if (reference.empty()) {
            ++negatives;
            EXPECT_FALSE(t.has_value()) << "trial " << trial;
        } else {
            ++positives;
            ASSERT_TRUE(t.has_value()) << "trial " << trial;
            EXPECT_EQ(*t, reference) << "trial " << trial;
        }
    }
    EXPECT_GT(positives, 30);
    EXPECT_GT(negatives, 30);
}

namespace {

// A CX written as H · CP(π) · H: a permutation that is not classical gate by gate.
std::vector<Command> cx_via_cp(uint32_t control, uint32_t target) {
    return {Command(GateType::H, target),
            Command(GateType::CP, control, target, Param(M_PI)),
            Command(GateType::H, target)};
}

void expect_reference_table(const std::vector<Command>& cmds, uint32_t n) {
    std::vector<uint32_t> qs(n);
    std::iota(qs.begin(), qs.end(), 0u);
    const auto reference = permutation_of(reference_unitary(cmds, n));
    ASSERT_FALSE(reference.empty()) << "the fixture must be a permutation";
    const auto t = permutation_table(cmds, qs, n, 64);
    ASSERT_TRUE(t.has_value());
    EXPECT_EQ(*t, reference);
}

void expect_no_table(const std::vector<Command>& cmds, uint32_t n) {
    std::vector<uint32_t> qs(n);
    std::iota(qs.begin(), qs.end(), 0u);
    ASSERT_TRUE(permutation_of(reference_unitary(cmds, n)).empty())
        << "the fixture must not be a permutation";
    EXPECT_FALSE(permutation_table(cmds, qs, n, 64).has_value());
}

}  // namespace

TEST(ProgramKernels, McxPatternNeedsItsClosingHadamardOnTheTarget) {
    // H(0) · MCZ(1, 0) · H(1) is not a multi-controlled X.
    expect_no_table({Command(GateType::H, 0), Command(GateType::MCZ, SmallVector<uint32_t, 2>{1, 0}),
                     Command(GateType::H, 1)}, 2);
    expect_reference_table({Command(GateType::H, 0),
                            Command(GateType::MCZ, SmallVector<uint32_t, 2>{1, 0}),
                            Command(GateType::H, 0)}, 2);
}

TEST(ProgramKernels, RestrictedMczFiresOnlyUnderItsFixedControls) {
    // Qubits 1 and 3 never change; the MCZ between the Hadamards must act
    // only where qubit 3 (and 1) read 1.
    auto cmds = cx_via_cp(1, 0);
    cmds.emplace_back(GateType::H, 2);
    cmds.emplace_back(GateType::MCZ, SmallVector<uint32_t, 2>{3, 1, 2});
    cmds.emplace_back(GateType::H, 2);
    expect_reference_table(cmds, 4);
}

TEST(ProgramKernels, ClassicalGateDrivenByAChangingQubitDerives) {
    // Qubit 0 changes (the CX written with a CP); the CX it then drives onto
    // qubit 2 makes qubit 2 change as well.
    auto cmds = cx_via_cp(1, 0);
    cmds.emplace_back(GateType::CX, 0, 2);
    cmds.emplace_back(GateType::CCX, SmallVector<uint32_t, 2>{2, 1, 3});
    expect_reference_table(cmds, 4);
}

TEST(ProgramKernels, FixedQubitsAreTrackedThroughClassicalGates) {
    // Qubits 1 and 2 only ever move classically; the CP-built CX must read
    // qubit 1's value after the swap.
    std::vector<Command> cmds = {Command(GateType::SWAP, 1, 2), Command(GateType::X, 2)};
    const auto cx = cx_via_cp(1, 0);
    cmds.insert(cmds.end(), cx.begin(), cx.end());
    cmds.emplace_back(GateType::CX, 2, 1);
    expect_reference_table(cmds, 3);
}

TEST(ProgramKernels, GlobalPhaseCountsInTheDerivation) {
    const auto gphase = [](double theta) {
        return Command(GateType::GPhase, SmallVector<uint32_t, 2>{},
                       SmallVector<Param, 1>{Param(theta)});
    };
    auto cmds = cx_via_cp(1, 0);
    cmds.push_back(gphase(0.3));
    expect_no_table(cmds, 2);
    cmds.push_back(gphase(-0.3));
    expect_reference_table(cmds, 2);
}

TEST(ProgramKernels, KernelsAboveTheParallelThresholdMatchTheirReferences) {
    // qarp forks a team from 16 qubits (QULACS_PARALLEL_NQUBIT_THRESHOLD).
    const int n = 17;
    const auto psi = random_state(n, 21);
    const QarpSimulator sim;
    {
        const std::vector<uint32_t> qs = {16, 3, 11, 0, 7};
        const auto t = random_table(5, 22);
        Program p;
        p.add_permutation(qs, t);
        EXPECT_EQ(max_diff(sim.program_statevector(p, n, psi), permuted(psi, qs, t)), 0.0);
    }
    {
        std::vector<uint32_t> qs(n);
        std::iota(qs.begin(), qs.end(), 0u);
        const auto t = random_table(n, 23);
        Program p;
        p.add_permutation(qs, t);
        EXPECT_EQ(max_diff(sim.program_statevector(p, n, psi), permuted(psi, qs, t)), 0.0);
    }
    {
        // Composition of two 15-qubit tables (the union spans all 17 qubits).
        std::vector<uint32_t> q1(15), q2(15);
        std::iota(q1.begin(), q1.end(), 0u);
        std::iota(q2.begin(), q2.end(), 2u);
        const auto t1 = random_table(15, 24), t2 = random_table(15, 25);
        Program p;
        p.add_permutation(q1, t1);
        p.add_permutation(q2, t2);
        EXPECT_EQ(p.kinds(), std::vector<std::string>{"permutation"});
        EXPECT_EQ(max_diff(sim.program_statevector(p, n, psi),
                           permuted(permuted(psi, q1, t1), q2, t2)), 0.0);
    }
    {
        const std::vector<uint32_t> targets = {4, 16, 9};
        const Mat V = random_unitary(3, 26);
        Program p;
        p.add_controlled_powers({0, 13}, {3, 2}, targets, V);
        auto expected = applied(psi, V * V * V, targets, uint64_t{1} << 0);
        expected = applied(expected, V * V, targets, uint64_t{1} << 13);
        EXPECT_LT(max_diff(sim.program_statevector(p, n, psi), expected), 1e-10);
    }
    {
        const std::vector<uint32_t> qs = {15, 2, 8};
        const Mat U = random_unitary(3, 27);
        Program p;
        p.add_dense(qs, U);
        EXPECT_LT(max_diff(sim.program_statevector(p, n, psi), applied(psi, U, qs)), kTol);
    }
}

TEST(ProgramKernels, NonTerminalMeasurementsAreRefused) {
    Program reused;
    reused.add_permutation({0, 1}, {1, 0, 3, 2});
    reused.add_gates({measure(0, 0), Command(GateType::X, 0)});
    EXPECT_THROW((void)QarpSimulator().program_run(reused, 2, 10, 1u), std::invalid_argument);

    Program recorded;
    recorded.add_gates({Command(GateType::H, 0), measure(0, 0)});
    EXPECT_THROW((void)QarpSimulator().program_statevector(recorded, 1), std::runtime_error);
}

TEST(ProgramKernels, MeasurementOutsideTheLastGatesKernelIsRejected) {
    Program p;
    p.add_gates({measure(0, 0)});
    p.add_permutation({0, 1}, {1, 0, 3, 2});
    EXPECT_THROW((void)QarpSimulator().program_run(p, 2, 10, 1u), std::invalid_argument);
}

// ── Digests ─────────────────────────────────────────────────────────────
//
// Oracle: the plain digest of the stream written out in full.

TEST(ProgramDigests, PartsDigestIsTheDigestOfTheRepeatedStream) {
    const std::vector<Command> a = {Command(GateType::H, 0), Command(GateType::CX, 0, 2),
                                    Command(GateType::Ry, 2, Param(0.37))};
    const std::vector<Command> b = {Command(GateType::X, 1), measure(1, 0)};
    std::vector<Command> written;
    for (int i = 0; i < 3; ++i) written.insert(written.end(), a.begin(), a.end());
    written.insert(written.end(), b.begin(), b.end());
    for (int i = 0; i < 2; ++i) written.insert(written.end(), a.begin(), a.end());
    EXPECT_EQ(parts_digest({{a, 3}, {b, 1}, {a, 2}}), commands_digest(written));
    EXPECT_NE(parts_digest({{a, 2}, {b, 1}, {a, 3}}), commands_digest(written));
    EXPECT_EQ(parts_digest({}), commands_digest({}));
}

TEST(ProgramDigests, LocalDigestIsTheDigestOfTheRemappedStream) {
    const std::vector<Command> placed = {Command(GateType::H, 5), Command(GateType::CX, 5, 2),
                                         Command(GateType::Rz, 7, Param(0.21)), measure(2, 3)};
    const std::vector<uint32_t> qubits = {2, 7, 5};
    uint32_t top = 0;
    for (auto q : qubits) top = std::max(top, q + 1);
    std::vector<uint32_t> map(top, UINT32_MAX);
    for (std::size_t b = 0; b < qubits.size(); ++b) map[qubits[b]] = static_cast<uint32_t>(b);
    std::vector<Command> local;
    for (const auto& c : placed) local.push_back(c.remap_qubits(map));
    EXPECT_EQ(local_commands_digest(placed, qubits), commands_digest(local));
    EXPECT_NE(local_commands_digest(placed, qubits), commands_digest(placed));
    // A command outside `qubits` falls back to the unmapped digest.
    EXPECT_EQ(local_commands_digest(placed, {2, 7}), commands_digest(placed));
}

}  // namespace qarpx::test
