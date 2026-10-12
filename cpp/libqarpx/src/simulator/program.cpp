#include "qarpx/simulator/program.h"

#include "qarpx/simulator/dense_kernel.h"
#include "qarpx/simulator/qarp_simulator.h"

#include <csim/utility.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace qarpx {

namespace {

// Same fork gate as the csim kernels and apply_dense_block: OMPutil forks a
// team only from QULACS_PARALLEL_NQUBIT_THRESHOLD.
constexpr unsigned kParallelThreshold = 13;

// A derived column is a basis state when its amplitude is within this of 1
// and every other entry within it of 0 (§14): the kernel then applies the
// exact permutation, so a span closer than this to one is snapped to it.
constexpr double kBasisTolerance = 1e-10;

// Widest register a table may span: 2^26 entries is 512 MB of uint64.
constexpr std::size_t kMaxTableQubits = 26;

// Widest local unitary: 4^12 amplitudes is 256 MB.
constexpr std::size_t kMaxUnitaryQubits = 12;

using cd = std::complex<double>;

void check_distinct(const std::vector<uint32_t>& qubits, const char* what) {
    std::vector<uint32_t> s(qubits);
    std::sort(s.begin(), s.end());
    if (std::adjacent_find(s.begin(), s.end()) != s.end())
        throw std::invalid_argument(std::string(what) + ": qubits must be distinct");
}

uint64_t qubit_mask(const std::vector<uint32_t>& qubits) {
    uint64_t m = 0;
    for (auto q : qubits) m |= uint64_t{1} << q;
    return m;
}

// Local index of `i` over `qubits` (local bit b ↔ qubits[b]).
inline uint64_t extract_bits(uint64_t i, const uint32_t* qubits, std::size_t k) {
    uint64_t l = 0;
    for (std::size_t b = 0; b < k; ++b) l |= ((i >> qubits[b]) & 1) << b;
    return l;
}

inline uint64_t deposit_bits(uint64_t l, const uint32_t* qubits, std::size_t k) {
    uint64_t i = 0;
    for (std::size_t b = 0; b < k; ++b) i |= ((l >> b) & 1) << qubits[b];
    return i;
}

// One classical step: flip `target`, or swap `target` with `other`, wherever
// every bit of `controls` is set.
struct ClassicalStep {
    uint64_t controls = 0;
    uint32_t target   = 0;
    uint32_t other    = 0;
    bool     swap     = false;
};

// The classical steps of `local`, or nullopt when a command is not a
// basis-state permutation with trivial phase.  Reads gate types only, so a
// stream that is not classical costs no table.
std::optional<std::vector<ClassicalStep>> classical_steps(const std::vector<Command>& local) {
    std::vector<ClassicalStep> steps;
    steps.reserve(local.size());
    for (std::size_t i = 0; i < local.size(); ++i) {
        const Command& c = local[i];
        switch (c.gate) {
            case GateType::Barrier:
            case GateType::Id:
                break;
            case GateType::X:
                steps.push_back({0, c.qubits[0], 0, false});
                break;
            case GateType::CX:
                steps.push_back({uint64_t{1} << c.qubits[0], c.qubits[1], 0, false});
                break;
            case GateType::CCX:
                steps.push_back({(uint64_t{1} << c.qubits[0]) | (uint64_t{1} << c.qubits[1]),
                                 c.qubits[2], 0, false});
                break;
            case GateType::SWAP:
                steps.push_back({0, c.qubits[0], c.qubits[1], true});
                break;
            case GateType::CSWAP:
                steps.push_back({uint64_t{1} << c.qubits[0], c.qubits[1], c.qubits[2], true});
                break;
            case GateType::H: {
                // `mcx` emits H(t) · MCZ(controls…, t) · H(t) — a multi-controlled X.
                if (i + 2 >= local.size()) return std::nullopt;
                const Command& mcz = local[i + 1];
                const Command& h2  = local[i + 2];
                const uint32_t t   = c.qubits[0];
                if (mcz.gate != GateType::MCZ || h2.gate != GateType::H
                        || h2.qubits[0] != t || mcz.qubits.empty()
                        || mcz.qubits[mcz.qubits.size() - 1] != t)
                    return std::nullopt;
                uint64_t ctrl = 0;
                for (std::size_t q = 0; q + 1 < mcz.qubits.size(); ++q)
                    ctrl |= uint64_t{1} << mcz.qubits[q];
                steps.push_back({ctrl, t, 0, false});
                i += 2;
                break;
            }
            default:
                return std::nullopt;
        }
    }
    return steps;
}

std::optional<std::vector<uint64_t>> classical_table(
    const std::vector<Command>& local, std::size_t k) {
    const auto steps = classical_steps(local);
    if (!steps) return std::nullopt;
    const uint64_t n = uint64_t{1} << k;
    std::vector<uint64_t> table(n);
    for (uint64_t x = 0; x < n; ++x) {
        uint64_t v = x;
        for (const auto& s : *steps) {
            if ((v & s.controls) != s.controls) continue;
            if (!s.swap) {
                v ^= uint64_t{1} << s.target;
            } else if (((v >> s.target) & 1) != ((v >> s.other) & 1)) {
                v ^= (uint64_t{1} << s.target) | (uint64_t{1} << s.other);
            }
        }
        table[x] = v;
    }
    return table;
}

// A command as the restriction needs it: its local unitary (absent for MCZ),
// the qubits it may change, and whether it is a classical permutation gate.
struct GateView {
    const Command*   cmd = nullptr;
    Eigen::MatrixXcd unitary;
    uint64_t         changes   = 0;  // bit b: couples states differing in qubits[b]
    bool             classical = false;
};

// Numerical tolerance for "this matrix entry is zero" when classifying qubits.
constexpr double kZero = 1e-14;

bool is_classical_gate(GateType g) {
    switch (g) {
        case GateType::X: case GateType::CX: case GateType::CCX:
        case GateType::SWAP: case GateType::CSWAP:
            return true;
        default:
            return false;
    }
}

// A classical gate applied to the integer `v` (bit q ↔ qubit q).
uint64_t apply_classical(const Command& c, uint64_t v) {
    auto bit = [&](uint32_t q) { return (v >> q) & 1; };
    auto swap_bits = [&](uint32_t a, uint32_t b) {
        if (bit(a) != bit(b)) v ^= (uint64_t{1} << a) | (uint64_t{1} << b);
    };
    const auto& q = c.qubits;
    switch (c.gate) {
        case GateType::X:     v ^= uint64_t{1} << q[0]; break;
        case GateType::CX:    if (bit(q[0])) v ^= uint64_t{1} << q[1]; break;
        case GateType::CCX:   if (bit(q[0]) && bit(q[1])) v ^= uint64_t{1} << q[2]; break;
        case GateType::SWAP:  swap_bits(q[0], q[1]); break;
        case GateType::CSWAP: if (bit(q[0])) swap_bits(q[1], q[2]); break;
        default: break;
    }
    return v;
}

// "Fixed" qubits (only ever a control, a phase partner, or moved by classical
// gates among themselves) are enumerated and tracked; every other command is
// sliced at their value and the rest simulated column by column.
std::optional<std::vector<uint64_t>> restricted_table(
    const std::vector<Command>& local, std::size_t k,
    uint32_t max_rest, uint32_t max_work_log2) {
    const QarpSimulator sim;
    std::vector<GateView> gates;
    cd                    global_phase{1.0, 0.0};
    for (const auto& c : local) {
        if (c.gate == GateType::Barrier || c.gate == GateType::Id) continue;
        if (c.gate == GateType::GPhase) {
            global_phase *= std::polar(1.0, c.params[0].value());
            continue;
        }
        GateView g;
        g.cmd       = &c;
        g.classical = is_classical_gate(c.gate);
        const std::size_t q = c.qubits.size();
        if (c.gate != GateType::MCZ) {
            if (q == 0 || q > kMaxDenseBlockQubits) return std::nullopt;
            g.unitary = sim.local_unitary(c);
            for (std::size_t b = 0; b < q; ++b) {
                bool couples = false;
                for (Eigen::Index r = 0; !couples && r < g.unitary.rows(); ++r)
                    for (Eigen::Index col = 0; col < g.unitary.cols(); ++col)
                        if ((((r ^ col) >> b) & 1) && std::abs(g.unitary(r, col)) > kZero) {
                            couples = true;
                            break;
                        }
                if (couples) g.changes |= uint64_t{1} << b;
            }
        }
        gates.push_back(std::move(g));
    }

    // Rest: changed by a non-classical gate, or by a classical gate that
    // also reads a rest qubit (closure).
    std::vector<uint8_t> is_rest(k, 0);
    for (const auto& g : gates)
        if (!g.classical)
            for (std::size_t b = 0; b < g.cmd->qubits.size(); ++b)
                if ((g.changes >> b) & 1) is_rest[g.cmd->qubits[b]] = 1;
    for (bool grew = true; grew;) {
        grew = false;
        for (const auto& g : gates) {
            if (!g.classical) continue;
            bool reads_rest = false;
            for (auto q : g.cmd->qubits) reads_rest |= is_rest[q] != 0;
            if (!reads_rest) continue;
            for (std::size_t b = 0; b < g.cmd->qubits.size(); ++b)
                if (((g.changes >> b) & 1) && !is_rest[g.cmd->qubits[b]]) {
                    is_rest[g.cmd->qubits[b]] = 1;
                    grew = true;
                }
        }
    }
    std::vector<uint32_t> fixed, rest;
    std::vector<uint32_t> rest_index(k, UINT32_MAX);
    for (uint32_t q = 0; q < k; ++q) {
        if (is_rest[q]) {
            rest_index[q] = static_cast<uint32_t>(rest.size());
            rest.push_back(q);
        } else {
            fixed.push_back(q);
        }
    }
    if (rest.size() > max_rest) return std::nullopt;
    const uint64_t R = uint64_t{1} << rest.size();

    // Commands touching no fixed qubit, remapped once; a run of them longer
    // than one dense pass on the rest costs is compressed into that pass.
    struct Op {
        bool                 dependent = false;
        std::size_t          gate      = 0;   // dependent: index into `gates`
        std::vector<Command> commands;        // independent: remapped run
    };
    std::vector<Op> ops;
    for (std::size_t i = 0; i < gates.size(); ++i) {
        bool dependent = false;
        for (auto q : gates[i].cmd->qubits) dependent |= !is_rest[q];
        if (dependent) {
            ops.push_back(Op{true, i, {}});
            continue;
        }
        if (ops.empty() || ops.back().dependent) ops.push_back(Op{false, 0, {}});
        ops.back().commands.push_back(gates[i].cmd->remap_qubits(rest_index));
    }
    std::vector<cd> column(R);
    double units = 0.0;  // per column: amplitude sweeps of length R
    for (auto& op : ops) {
        if (op.dependent) { units += 1.0; continue; }
        if (rest.size() <= kMaxDenseBlockQubits && static_cast<double>(op.commands.size()) > static_cast<double>(R)) {
            Eigen::MatrixXcd seg(static_cast<Eigen::Index>(R), static_cast<Eigen::Index>(R));
            for (uint64_t c = 0; c < R; ++c) {
                std::fill(column.begin(), column.end(), cd{0.0, 0.0});
                column[c] = {1.0, 0.0};
                for (const auto& cmd : op.commands) sim.apply_command(cmd, column.data(), R);
                for (uint64_t row = 0; row < R; ++row)
                    seg(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(c)) = column[row];
            }
            SmallVector<uint32_t, 2> all;
            for (uint32_t b = 0; b < rest.size(); ++b) all.push_back(b);
            Command dense(GateType::Custom, std::move(all));
            dense.unitary = std::make_shared<const Eigen::MatrixXcd>(std::move(seg));
            op.commands.assign(1, std::move(dense));
            units += static_cast<double>(R);
        } else {
            units += static_cast<double>(op.commands.size());
        }
    }
    // Work: 2^|fixed| assignments × R columns × `units` sweeps of R amplitudes.
    const double work = std::ldexp(units * static_cast<double>(R) * static_cast<double>(R),
                                   static_cast<int>(fixed.size()));
    if (work > std::ldexp(static_cast<double>(std::max<std::size_t>(local.size(), 1)),
                          static_cast<int>(max_work_log2)))
        return std::nullopt;

    const uint64_t        n_fixed = uint64_t{1} << fixed.size();
    std::vector<uint64_t> table(uint64_t{1} << k);
    std::vector<uint8_t>  hit(R);
    std::vector<Command>  reduced;
    std::vector<cd>       block(R * R);

    for (uint64_t a = 0; a < n_fixed; ++a) {
        const uint64_t start = deposit_bits(a, fixed.data(), fixed.size());
        uint64_t cur   = start;
        cd       phase = global_phase;
        reduced.clear();
        for (const auto& op : ops) {
            if (!op.dependent) {
                reduced.insert(reduced.end(), op.commands.begin(), op.commands.end());
                continue;
            }
            const GateView& g  = gates[op.gate];
            const auto&     qs = g.cmd->qubits;
            bool touches_rest = false;
            for (auto q : qs) touches_rest |= is_rest[q] != 0;
            if (g.classical && !touches_rest) {
                cur = apply_classical(*g.cmd, cur);
                continue;
            }
            SmallVector<uint32_t, 2> live;
            std::vector<std::size_t> live_pos;
            uint64_t base   = 0;
            bool     active = true;
            for (std::size_t b = 0; b < qs.size(); ++b) {
                if (!is_rest[qs[b]]) {
                    const uint64_t bit = (cur >> qs[b]) & 1;
                    base |= bit << b;
                    active &= bit != 0;
                } else {
                    live.push_back(rest_index[qs[b]]);
                    live_pos.push_back(b);
                }
            }
            if (g.cmd->gate == GateType::MCZ) {
                // −1 exactly where every qubit is 1: the fixed ones must read 1.
                if (!active) continue;
                if (live.empty()) phase = -phase;
                else if (live.size() == 1) reduced.emplace_back(GateType::Z, live[0]);
                else reduced.emplace_back(GateType::MCZ, std::move(live));
                continue;
            }
            const Eigen::Index d = Eigen::Index{1} << live_pos.size();
            Eigen::MatrixXcd sub(d, d);
            for (Eigen::Index row = 0; row < d; ++row)
                for (Eigen::Index c = 0; c < d; ++c) {
                    uint64_t rr = base, cc = base;
                    for (std::size_t j = 0; j < live_pos.size(); ++j) {
                        rr |= ((static_cast<uint64_t>(row) >> j) & 1) << live_pos[j];
                        cc |= ((static_cast<uint64_t>(c) >> j) & 1) << live_pos[j];
                    }
                    sub(row, c) = g.unitary(static_cast<Eigen::Index>(rr), static_cast<Eigen::Index>(cc));
                }
            if (live.empty()) {
                phase *= sub(0, 0);
                continue;
            }
            Command custom(GateType::Custom, std::move(live));
            custom.unitary = std::make_shared<const Eigen::MatrixXcd>(std::move(sub));
            reduced.push_back(std::move(custom));
        }

        // All R columns at once: entry (row, x) at row + R·x of a 2·|rest|-qubit
        // vector, so a gate on rest qubit j (< |rest|) acts on every column in
        // one dispatch.
        std::fill(block.begin(), block.end(), cd{0.0, 0.0});
        for (uint64_t x = 0; x < R; ++x) block[x + R * x] = {1.0, 0.0};
        for (const auto& c : reduced) sim.apply_command(c, block.data(), R * R);

        std::fill(hit.begin(), hit.end(), 0);
        for (uint64_t x = 0; x < R; ++x) {
            const cd* col  = block.data() + R * x;
            uint64_t  y    = 0;
            double    best = -1.0;
            for (uint64_t j = 0; j < R; ++j) {
                const double m = std::norm(col[j]);
                if (m > best) { best = m; y = j; }
            }
            if (std::abs(col[y] * phase - cd{1.0, 0.0}) > kBasisTolerance) return std::nullopt;
            for (uint64_t j = 0; j < R; ++j)
                if (j != y && std::abs(col[j]) > kBasisTolerance) return std::nullopt;
            if (hit[y]) return std::nullopt;
            hit[y] = 1;
            table[start | deposit_bits(x, rest.data(), rest.size())] =
                cur | deposit_bits(y, rest.data(), rest.size());
        }
    }
    return table;
}

// `commands` remapped onto [0, |qubits|), or nullopt if any command is
// parametric, conditioned, non-unitary, or touches a qubit outside `qubits`.
std::optional<std::vector<Command>> to_local(const std::vector<Command>&  commands,
                                             const std::vector<uint32_t>& qubits) {
    uint32_t top = 0;
    for (const auto& c : commands)
        for (auto q : c.qubits) top = std::max(top, q + 1);
    for (auto q : qubits) top = std::max(top, q + 1);
    constexpr uint32_t kAbsent = UINT32_MAX;
    std::vector<uint32_t> map(top, kAbsent);
    for (std::size_t b = 0; b < qubits.size(); ++b) map[qubits[b]] = static_cast<uint32_t>(b);

    std::vector<Command> local;
    local.reserve(commands.size());
    for (const auto& c : commands) {
        if (c.is_parametric() || !c.condition_bits.empty() || !c.cbits.empty())
            return std::nullopt;
        switch (c.gate) {
            case GateType::Measure: case GateType::Reset:
            case GateType::BranchBegin: case GateType::BranchElse: case GateType::BranchEnd:
                return std::nullopt;
            default:
                break;
        }
        for (auto q : c.qubits)
            if (map[q] == kAbsent) return std::nullopt;
        local.push_back(c.remap_qubits(map));
    }
    return local;
}

// `second ∘ first` on the union of their qubits (ascending), or nullopt
// when the union exceeds the table cap.
std::optional<PermutationKernel> compose(const PermutationKernel& first,
                                         const PermutationKernel& second) {
    std::vector<uint32_t> uni(first.qubits);
    uni.insert(uni.end(), second.qubits.begin(), second.qubits.end());
    std::sort(uni.begin(), uni.end());
    uni.erase(std::unique(uni.begin(), uni.end()), uni.end());
    if (uni.size() > kMaxTableQubits) return std::nullopt;

    auto local_positions = [&](const std::vector<uint32_t>& qs) {
        std::vector<uint32_t> pos(qs.size());
        for (std::size_t b = 0; b < qs.size(); ++b)
            pos[b] = static_cast<uint32_t>(
                std::lower_bound(uni.begin(), uni.end(), qs[b]) - uni.begin());
        return pos;
    };
    const auto p1 = local_positions(first.qubits);
    const auto p2 = local_positions(second.qubits);
    const uint64_t keep1 = ~qubit_mask(p1), keep2 = ~qubit_mask(p2);

    const uint64_t n = uint64_t{1} << uni.size();
    std::vector<uint64_t> table(n);
#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(static_cast<ITYPE>(n), kParallelThreshold);
#pragma omp parallel for schedule(static)
#endif
    for (int64_t s = 0; s < static_cast<int64_t>(n); ++s) {
        uint64_t x = static_cast<uint64_t>(s);
        x = (x & keep1) | deposit_bits(first.table[extract_bits(x, p1.data(), p1.size())],
                                       p1.data(), p1.size());
        x = (x & keep2) | deposit_bits(second.table[extract_bits(x, p2.data(), p2.size())],
                                       p2.data(), p2.size());
        table[static_cast<uint64_t>(s)] = x;
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
    return PermutationKernel{std::move(uni), std::move(table)};
}

inline void mix(uint64_t& h, uint64_t v) {
    // FNV-1a over the 8 bytes of v.
    for (int b = 0; b < 8; ++b) {
        h ^= (v >> (8 * b)) & 0xff;
        h *= 0x100000001b3ULL;
    }
}

}  // namespace

// ── Program ──────────────────────────────────────────────────────────────────

void Program::add_gates(std::vector<Command> commands) {
    kernels_.emplace_back(GatesKernel{std::move(commands)});
}

void Program::add_permutation(std::vector<uint32_t> qubits, std::vector<uint64_t> table) {
    check_distinct(qubits, "add_permutation");
    if (qubits.empty() || qubits.size() > kMaxTableQubits)
        throw std::invalid_argument("add_permutation: 1.." + std::to_string(kMaxTableQubits)
                                    + " qubits required");
    const uint64_t n = uint64_t{1} << qubits.size();
    if (table.size() != n)
        throw std::invalid_argument("add_permutation: table length " + std::to_string(table.size())
                                    + " for " + std::to_string(qubits.size()) + " qubit(s)");
    std::vector<uint8_t> seen(n, 0);
    for (auto v : table) {
        if (v >= n || seen[v])
            throw std::invalid_argument("add_permutation: table is not a bijection");
        seen[v] = 1;
    }
    PermutationKernel next{std::move(qubits), std::move(table)};
    auto* prev = kernels_.empty() ? nullptr : std::get_if<PermutationKernel>(&kernels_.back());
    if (prev) {
        if (auto merged = compose(*prev, next)) {
            *prev = std::move(*merged);
            return;
        }
    }
    kernels_.emplace_back(std::move(next));
}

void Program::add_dense(std::vector<uint32_t> qubits, Eigen::MatrixXcd matrix) {
    check_distinct(qubits, "add_dense");
    const std::size_t k = qubits.size();
    if (k == 0 || k > kMaxDenseBlockQubits)
        throw std::invalid_argument("add_dense: 1.." + std::to_string(kMaxDenseBlockQubits)
                                    + " qubits required");
    const Eigen::Index d = Eigen::Index{1} << k;
    if (matrix.rows() != d || matrix.cols() != d)
        throw std::invalid_argument("add_dense: matrix is not 2^k square");
    kernels_.emplace_back(DenseKernel{std::move(qubits), std::move(matrix)});
}

void Program::add_controlled_powers(std::vector<uint32_t> controls,
                                    std::vector<uint64_t> exponents,
                                    std::vector<uint32_t> targets,
                                    Eigen::MatrixXcd      matrix) {
    if (controls.empty() || targets.empty())
        throw std::invalid_argument("add_controlled_powers: controls and targets required");
    if (exponents.size() != controls.size())
        throw std::invalid_argument("add_controlled_powers: one exponent per control");
    std::vector<uint32_t> all(controls);
    all.insert(all.end(), targets.begin(), targets.end());
    check_distinct(all, "add_controlled_powers");
    const Eigen::Index d = Eigen::Index{1} << targets.size();
    if (targets.size() > kMaxTableQubits || matrix.rows() != d || matrix.cols() != d)
        throw std::invalid_argument("add_controlled_powers: matrix is not 2^|targets| square");
    kernels_.emplace_back(ControlledPowersKernel{std::move(controls), std::move(exponents),
                                                 std::move(targets), std::move(matrix)});
}

std::vector<std::string> Program::kinds() const {
    std::vector<std::string> out;
    out.reserve(kernels_.size());
    for (const auto& k : kernels_) {
        switch (k.index()) {
            case 0: out.emplace_back("permutation"); break;
            case 1: out.emplace_back("dense"); break;
            case 2: out.emplace_back("controlled_powers"); break;
            default: out.emplace_back("gates"); break;
        }
    }
    return out;
}

uint32_t Program::min_register_width() const {
    uint32_t top = 0;
    auto widen = [&](const auto& qs) { for (auto q : qs) top = std::max(top, q + 1); };
    for (const auto& k : kernels_) {
        std::visit([&](const auto& kk) {
            using T = std::decay_t<decltype(kk)>;
            if constexpr (std::is_same_v<T, GatesKernel>) {
                for (const auto& c : kk.commands) widen(c.qubits);
            } else if constexpr (std::is_same_v<T, ControlledPowersKernel>) {
                widen(kk.controls);
                widen(kk.targets);
            } else {
                widen(kk.qubits);
            }
        }, k);
    }
    return top;
}

Program Program::substituted(const std::unordered_map<std::string, double>& params) const {
    Program out;
    out.kernels_.reserve(kernels_.size());
    for (const auto& k : kernels_) {
        if (const auto* g = std::get_if<GatesKernel>(&k))
            out.kernels_.emplace_back(GatesKernel{substitute_all(g->commands, params)});
        else
            out.kernels_.push_back(k);
    }
    return out;
}

Program Program::without_measurements() const {
    Program out;
    out.kernels_.reserve(kernels_.size());
    for (const auto& k : kernels_) {
        if (const auto* g = std::get_if<GatesKernel>(&k)) {
            GatesKernel kept;
            for (const auto& c : g->commands)
                if (c.gate != GateType::Measure && c.gate != GateType::Barrier)
                    kept.commands.push_back(c);
            out.kernels_.emplace_back(std::move(kept));
        } else {
            out.kernels_.push_back(k);
        }
    }
    return out;
}

// ── Kernels ──────────────────────────────────────────────────────────────────

void apply_permutation(const PermutationKernel& k,
                       std::vector<cd>&         state,
                       std::vector<cd>&         scratch) {
    const uint64_t    dim = state.size();
    const std::size_t kq  = k.qubits.size();
    const uint64_t*   tab = k.table.data();
    const cd*         in  = state.data();
    cd*               out = scratch.data();

    bool whole = (uint64_t{1} << kq) == dim;
    for (std::size_t b = 0; whole && b < kq; ++b) whole = k.qubits[b] == b;

#ifdef _OPENMP
    OMPutil::get_inst().set_qulacs_num_threads(static_cast<ITYPE>(dim), kParallelThreshold);
#endif
    if (whole) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t i = 0; i < static_cast<int64_t>(dim); ++i)
            out[tab[i]] = in[i];
    } else {
        const uint64_t  keep = ~qubit_mask(k.qubits);
        const uint32_t* qs   = k.qubits.data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t s = 0; s < static_cast<int64_t>(dim); ++s) {
            const uint64_t i = static_cast<uint64_t>(s);
            out[(i & keep) | deposit_bits(tab[extract_bits(i, qs, kq)], qs, kq)] = in[i];
        }
    }
#ifdef _OPENMP
    OMPutil::get_inst().reset_qulacs_num_threads();
#endif
    state.swap(scratch);
}

void apply_controlled_powers(const ControlledPowersKernel& k, std::vector<cd>& state) {
    const uint64_t    dim = state.size();
    const std::size_t m   = k.targets.size();
    const uint64_t    d   = uint64_t{1} << m;

    std::vector<uint32_t> sorted(k.targets);
    std::sort(sorted.begin(), sorted.end());
    std::vector<uint64_t> offset(d);
    for (uint64_t l = 0; l < d; ++l) offset[l] = deposit_bits(l, k.targets.data(), m);

    // squares[b] = matrix^(2^b), one chain shared by every exponent; an
    // exponent with several set bits multiplies its squares once.
    uint64_t all = 0;
    for (auto e : k.exponents) all |= e;
    const int n_squares = std::bit_width(all);
    std::vector<Eigen::MatrixXcd> squares;
    squares.reserve(static_cast<std::size_t>(n_squares));
    if (n_squares > 0) squares.push_back(k.matrix);
    for (int b = 1; b < n_squares; ++b) squares.push_back(squares.back() * squares.back());
    std::unordered_map<uint64_t, Eigen::MatrixXcd> products;
    for (auto e : k.exponents) {
        if (std::popcount(e) < 2 || products.count(e)) continue;
        Eigen::MatrixXcd result = squares[static_cast<std::size_t>(std::countr_zero(e))];
        for (uint64_t x = e & (e - 1); x != 0; x &= x - 1)
            result = result * squares[static_cast<std::size_t>(std::countr_zero(x))];
        products.emplace(e, std::move(result));
    }

    const int64_t outer = static_cast<int64_t>(dim >> m);
    for (std::size_t j = 0; j < k.controls.size(); ++j) {
        const uint64_t e = k.exponents[j];
        if (e == 0) continue;
        const Eigen::MatrixXcd& P    = std::has_single_bit(e)
            ? squares[static_cast<std::size_t>(std::countr_zero(e))] : products.at(e);
        const uint64_t          cbit = uint64_t{1} << k.controls[j];
        cd*                     sv   = state.data();
#ifdef _OPENMP
        OMPutil::get_inst().set_qulacs_num_threads(static_cast<ITYPE>(dim), kParallelThreshold);
#pragma omp parallel
#endif
        {
            Eigen::VectorXcd col(static_cast<Eigen::Index>(d)), res(static_cast<Eigen::Index>(d));
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for (int64_t o = 0; o < outer; ++o) {
                uint64_t base = static_cast<uint64_t>(o);
                for (std::size_t b = 0; b < m; ++b) {
                    const uint64_t p   = sorted[b];
                    const uint64_t low = base & ((uint64_t{1} << p) - 1);
                    base = ((base >> p) << (p + 1)) | low;
                }
                if (!(base & cbit)) continue;
                for (uint64_t l = 0; l < d; ++l) col[static_cast<Eigen::Index>(l)] = sv[base | offset[l]];
                res.noalias() = P * col;
                for (uint64_t l = 0; l < d; ++l) sv[base | offset[l]] = res[static_cast<Eigen::Index>(l)];
            }
        }
#ifdef _OPENMP
        OMPutil::get_inst().reset_qulacs_num_threads();
#endif
    }
}

// ── Derivation helpers ───────────────────────────────────────────────────────

std::optional<std::vector<uint64_t>> permutation_table(const std::vector<Command>&  commands,
                                                       const std::vector<uint32_t>& qubits,
                                                       uint32_t                     max_rest,
                                                       uint32_t                     max_work_log2) {
    check_distinct(qubits, "permutation_table");
    if (qubits.empty() || qubits.size() > kMaxTableQubits) return std::nullopt;
    auto local = to_local(commands, qubits);
    if (!local) return std::nullopt;
    if (auto t = classical_table(*local, qubits.size())) return t;
    return restricted_table(*local, qubits.size(), max_rest, max_work_log2);
}

Eigen::MatrixXcd local_unitary_of(const std::vector<Command>&  commands,
                                  const std::vector<uint32_t>& qubits) {
    check_distinct(qubits, "local_unitary_of");
    if (qubits.empty() || qubits.size() > kMaxUnitaryQubits)
        throw std::invalid_argument("local_unitary_of: 1.." + std::to_string(kMaxUnitaryQubits)
                                    + " qubits required");
    auto local = to_local(commands, qubits);
    if (!local)
        throw std::invalid_argument(
            "local_unitary_of: commands must be concrete, unconditioned and unitary, "
            "on the listed qubits only");
    return QarpSimulator().unitary_matrix(*local, static_cast<int>(qubits.size()));
}

namespace {

constexpr uint64_t kDigestSeed = 0xcbf29ce484222325ULL;

// One command into the FNV-1a state, its qubits read through `map` when given.
void mix_command(uint64_t& h, const Command& c, const std::vector<uint32_t>* map) {
    mix(h, static_cast<uint64_t>(c.gate));
    mix(h, c.qubits.size());
    for (auto q : c.qubits) mix(h, map ? (*map)[q] : q);
    mix(h, c.cbits.size());
    for (auto b : c.cbits) mix(h, b);
    mix(h, c.condition_bits.size());
    for (std::size_t i = 0; i < c.condition_bits.size(); ++i) {
        mix(h, c.condition_bits[i]);
        mix(h, c.condition_values[i] ? 1 : 0);
    }
    mix(h, c.params.size());
    for (const auto& p : c.params) {
        if (p.is_concrete()) {
            mix(h, std::bit_cast<uint64_t>(p.value()));
        } else {
            for (char ch : p.to_string()) mix(h, static_cast<unsigned char>(ch));
        }
    }
    if (c.unitary) {
        const auto& u = *c.unitary;
        for (Eigen::Index r = 0; r < u.rows(); ++r)
            for (Eigen::Index col = 0; col < u.cols(); ++col) {
                mix(h, std::bit_cast<uint64_t>(u(r, col).real()));
                mix(h, std::bit_cast<uint64_t>(u(r, col).imag()));
            }
    }
}

}  // namespace

uint64_t local_commands_digest(const std::vector<Command>&  commands,
                               const std::vector<uint32_t>& qubits) {
    uint32_t top = 0;
    for (const auto& c : commands)
        for (auto q : c.qubits) top = std::max(top, q + 1);
    for (auto q : qubits) top = std::max(top, q + 1);
    std::vector<uint32_t> map(top, UINT32_MAX);
    for (std::size_t b = 0; b < qubits.size(); ++b) map[qubits[b]] = static_cast<uint32_t>(b);
    for (const auto& c : commands)
        for (auto q : c.qubits)
            if (map[q] == UINT32_MAX) return commands_digest(commands);
    uint64_t h = kDigestSeed;
    for (const auto& c : commands) mix_command(h, c, &map);
    return h;
}

uint64_t commands_digest(const std::vector<Command>& commands) {
    uint64_t h = kDigestSeed;
    for (const auto& c : commands) mix_command(h, c, nullptr);
    return h;
}

uint64_t parts_digest(const std::vector<std::pair<std::vector<Command>, std::size_t>>& parts) {
    uint64_t h = kDigestSeed;
    for (const auto& [commands, count] : parts)
        for (std::size_t i = 0; i < count; ++i)
            for (const auto& c : commands) mix_command(h, c, nullptr);
    return h;
}

}  // namespace qarpx
