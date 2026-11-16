#include "qarpx/simulator/fusion.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qarpx {

namespace {

using Mat = Eigen::MatrixXcd;

// Ids are creation order; 0 is "never touched" on the wire-front table, so
// real ids start at 1 and a `front[w] <= id` test needs no special case.
struct Block {
    std::size_t              id = 0;
    std::vector<uint32_t>    qubits;   // local bit b ↔ qubits[b]
    std::vector<uint32_t>    wires;    // qubits + cbit wires, for flush matching
    Mat                      unitary;  // empty until a second gate folds in
    std::size_t              n_gates = 0;
    Command                  first;    // re-emitted verbatim when n_gates == 1
    SmallVector<uint32_t, 1> condition_bits;
    SmallVector<bool, 1>     condition_values;
};

bool is_marker(GateType g) {
    return g == GateType::Barrier || g == GateType::Measure || g == GateType::Reset
        || g == GateType::BranchBegin || g == GateType::BranchElse
        || g == GateType::BranchEnd;
}

bool is_fusible(const Command& cmd, std::size_t k) {
    if (cmd.qubits.empty() || cmd.qubits.size() > k) return false;
    if (is_marker(cmd.gate) || cmd.gate == GateType::GPhase) return false;
    if (cmd.gate == GateType::Custom && !cmd.unitary) return false;
    return !cmd.is_parametric();
}

bool condition_matches(const Block& b, const Command& cmd) {
    return b.condition_bits == cmd.condition_bits
        && b.condition_values == cmd.condition_values;
}

bool contains(const std::vector<uint32_t>& v, uint32_t x) {
    return std::find(v.begin(), v.end(), x) != v.end();
}

// Widening past two qubits needs a shared qubit: merging disjoint gates
// saves a pass but multiplies the per-amplitude work by 2^k, and a 2-qubit
// pass costs about one named gate while a 3-qubit pass costs two
// (dense_kernel.h; 2026-09-13, brickwork 20q at k = 3: 79 ms with disjoint
// merging, 52 ms without).
bool can_join(const std::vector<uint32_t>& block, const SmallVector<uint32_t, 2>& gate,
              std::size_t max_qubits) {
    std::size_t shared = 0, fresh = 0;
    for (auto q : gate) (contains(block, q) ? shared : fresh)++;
    if (fresh == 0) return true;
    const std::size_t width = block.size() + fresh;
    if (width > max_qubits) return false;
    return shared > 0 || width <= 2;
}

// U ← (I ⊗ U): the new qubits become the high local bits, so the existing
// entries keep their index.
void extend(Block& b, const std::vector<uint32_t>& new_qubits) {
    const Eigen::Index old = b.unitary.rows();
    const Eigen::Index copies = Eigen::Index{1} << new_qubits.size();
    Mat out = Mat::Zero(old * copies, old * copies);
    for (Eigen::Index h = 0; h < copies; ++h)
        out.block(h * old, h * old, old, old) = b.unitary;
    b.unitary = std::move(out);
    b.qubits.insert(b.qubits.end(), new_qubits.begin(), new_qubits.end());
}

// U ← E(g) · U, where E embeds the gate-local `g` (local bit b ↔ block bit
// pos[b]) into the block's space.  Row form: (E U).row(io) = Σ_i E(io, i) U.row(i).
void left_apply(Mat& U, const Mat& g, const std::vector<std::size_t>& pos) {
    const Eigen::Index dim = U.rows();
    const std::size_t  m    = pos.size();
    const std::size_t  gdim = std::size_t{1} << m;
    Mat out = Mat::Zero(dim, dim);
    for (Eigen::Index i = 0; i < dim; ++i) {
        const auto  ui   = static_cast<std::size_t>(i);
        std::size_t li   = 0;
        std::size_t rest = ui;
        for (std::size_t b = 0; b < m; ++b) {
            li   |= ((ui >> pos[b]) & 1u) << b;
            rest &= ~(std::size_t{1} << pos[b]);
        }
        for (std::size_t lo = 0; lo < gdim; ++lo) {
            const auto coeff = g(static_cast<Eigen::Index>(lo),
                                 static_cast<Eigen::Index>(li));
            if (coeff == std::complex<double>{}) continue;
            std::size_t io = rest;
            for (std::size_t b = 0; b < m; ++b)
                if ((lo >> b) & 1u) io |= std::size_t{1} << pos[b];
            out.row(static_cast<Eigen::Index>(io)) += coeff * U.row(i);
        }
    }
    U = std::move(out);
}

Mat checked_local_unitary(const LocalUnitary& local_unitary, const Command& cmd) {
    Mat g = local_unitary(cmd);
    const Eigen::Index want = Eigen::Index{1} << cmd.qubits.size();
    if (g.rows() != want || g.cols() != want)
        throw std::runtime_error(
            "fuse_for_simulation: local unitary of '" + std::string(gate_name(cmd.gate))
            + "' is " + std::to_string(g.rows()) + "x" + std::to_string(g.cols())
            + ", expected " + std::to_string(want) + "x" + std::to_string(want));
    return g;
}

}  // namespace

std::vector<Command> fuse_for_simulation(
    const std::vector<Command>& commands,
    std::size_t                 max_qubits,
    const LocalUnitary&         local_unitary)
{
    if (max_qubits == 0 || commands.empty()) return commands;

    // Wire space: qubits first, then one wire per cbit (read or written).
    uint32_t max_q = 0, max_c = 0;
    bool any_c = false;
    for (const auto& cmd : commands) {
        for (auto q : cmd.qubits) max_q = std::max(max_q, q);
        for (auto c : cmd.cbits)          { max_c = std::max(max_c, c); any_c = true; }
        for (auto c : cmd.condition_bits) { max_c = std::max(max_c, c); any_c = true; }
    }
    const uint32_t n_qubit_wires = max_q + 1;
    const std::size_t n_wires = n_qubit_wires + (any_c ? max_c + 1 : 0);

    std::vector<std::size_t> front(n_wires, 0);
    std::deque<Block> open;
    std::size_t next_id = 1;
    std::vector<Command> out;
    out.reserve(commands.size());

    auto wires_of = [&](const Command& cmd) {
        std::vector<uint32_t> ws(cmd.qubits.begin(), cmd.qubits.end());
        for (auto c : cmd.condition_bits) ws.push_back(n_qubit_wires + c);
        for (auto c : cmd.cbits)          ws.push_back(n_qubit_wires + c);
        return ws;
    };

    auto emit = [&](Block& b) {
        if (b.n_gates == 1) {
            out.push_back(std::move(b.first));
            return;
        }
        if (b.qubits.size() == 1
                && (b.unitary - Mat::Identity(2, 2)).norm() < 1e-12)
            return;
        Command fused;
        fused.gate = GateType::Custom;
        for (auto q : b.qubits) fused.qubits.push_back(q);
        fused.unitary          = std::make_shared<const Mat>(std::move(b.unitary));
        fused.condition_bits   = b.condition_bits;
        fused.condition_values = b.condition_values;
        out.push_back(std::move(fused));
    };

    auto shares_wire = [](const Block& a, const Block& b) {
        for (auto w : a.wires) if (contains(b.wires, w)) return true;
        return false;
    };

    // Emit every open block touching `ws` and, transitively, every earlier
    // open block sharing a wire with one being emitted — the exact set that
    // must precede the next command on those wires — in creation order.
    auto flush_wires = [&](const std::vector<uint32_t>& ws) {
        if (open.empty()) return;
        std::vector<bool> mark(open.size(), false);
        for (std::size_t i = 0; i < open.size(); ++i)
            for (auto w : ws)
                if (contains(open[i].wires, w)) { mark[i] = true; break; }
        for (std::size_t i = open.size(); i-- > 0;)
            if (mark[i])
                for (std::size_t j = 0; j < i; ++j)
                    if (!mark[j] && shares_wire(open[j], open[i])) mark[j] = true;
        std::deque<Block> keep;
        for (std::size_t i = 0; i < open.size(); ++i) {
            if (mark[i]) emit(open[i]);
            else         keep.push_back(std::move(open[i]));
        }
        open.swap(keep);
    };

    auto flush_all = [&]() {
        for (auto& b : open) emit(b);
        open.clear();
    };

    for (const auto& cmd : commands) {
        // Scalar on the global wire: commutes with everything, stays in place.
        if (cmd.gate == GateType::GPhase) {
            out.push_back(cmd);
            continue;
        }
        if (cmd.gate == GateType::BranchBegin || cmd.gate == GateType::BranchElse
                || cmd.gate == GateType::BranchEnd
                || (cmd.gate == GateType::Barrier && cmd.qubits.empty())) {
            flush_all();
            out.push_back(cmd);
            continue;
        }

        const auto ws = wires_of(cmd);

        if (is_fusible(cmd, max_qubits)) {
            Block* target = nullptr;
            for (auto it = open.rbegin(); it != open.rend(); ++it) {
                if (!condition_matches(*it, cmd)) continue;
                if (!can_join(it->qubits, cmd.qubits, max_qubits)) continue;
                bool ok = true;
                for (auto w : ws) if (front[w] > it->id) { ok = false; break; }
                if (ok) { target = &*it; break; }
            }
            if (!target) {
                Block b;
                b.id = next_id++;
                b.qubits.assign(cmd.qubits.begin(), cmd.qubits.end());
                b.wires = ws;
                b.n_gates = 1;
                b.first = cmd;
                b.condition_bits   = cmd.condition_bits;
                b.condition_values = cmd.condition_values;
                for (auto w : ws) front[w] = b.id;
                open.push_back(std::move(b));
                continue;
            }
            Block& b = *target;
            if (b.n_gates == 1)
                b.unitary = checked_local_unitary(local_unitary, b.first);
            std::vector<uint32_t> fresh;
            for (auto q : cmd.qubits) if (!contains(b.qubits, q)) fresh.push_back(q);
            if (!fresh.empty()) extend(b, fresh);
            std::vector<std::size_t> pos;
            pos.reserve(cmd.qubits.size());
            for (auto q : cmd.qubits)
                pos.push_back(static_cast<std::size_t>(
                    std::find(b.qubits.begin(), b.qubits.end(), q) - b.qubits.begin()));
            left_apply(b.unitary, checked_local_unitary(local_unitary, cmd), pos);
            ++b.n_gates;
            for (auto w : ws) {
                if (!contains(b.wires, w)) b.wires.push_back(w);
                front[w] = b.id;
            }
            continue;
        }

        // Cannot fold: everything that must precede it goes out first, the
        // command itself stays in place, and its wires are stamped so no
        // later gate folds into a block created before it.
        flush_wires(ws);
        out.push_back(cmd);
        const std::size_t id = next_id++;
        for (auto w : ws) front[w] = id;
    }

    flush_all();
    return out;
}

}  // namespace qarpx
