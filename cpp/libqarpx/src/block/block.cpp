#include "qarpx/block/block.h"
#include "qarpx/synthesis/diagonal.h"
#include "qarpx/synthesis/pauli_exponential.h"
#include "qarpx/synthesis/state_preparation.h"
#include "qarpx/synthesis/unitary.h"

#include <algorithm>
#include <set>
#include <stdexcept>

namespace qarpx {

// ── Block base ──

std::vector<std::string> Block::free_symbols() const {
    std::set<std::string> seen;
    std::vector<std::string> result;
    for (const auto& cmd : commands_) {
        for (const auto& p : cmd.params) {
            for (const auto& s : p.free_symbols()) {
                if (seen.insert(s).second) {
                    result.push_back(s);
                }
            }
        }
    }
    return result;
}

std::vector<Command> Block::flatten() const {
    if (!built_) {
        throw std::runtime_error("Block::flatten() called before build()");
    }

    std::vector<Command> result;
    result.reserve(commands_.size());

    if (target_qubits.has_value()) {
        const auto& mapping = target_qubits.value();
        for (std::size_t i = 0; i < commands_.size(); ++i) {
            try {
                result.push_back(commands_[i].remap_qubits(mapping));
            } catch (const std::out_of_range& e) {
                // Locate the offending gate: the command-level message names
                // the gate, this names the block and the append order.
                throw std::out_of_range(
                    "Block '" + name + "' (n_qubits=" + std::to_string(n_qubits) +
                    "): command #" + std::to_string(i) + " of " +
                    std::to_string(commands_.size()) + " — " + e.what());
            }
        }
    } else {
        result = commands_;
    }

    if (target_cbits.has_value())
        cbit_remap_in_place(result, target_cbits.value());

    return result;
}

bool Block::equals(const Block& other, double atol) const {
    if (n_qubits != other.n_qubits) return false;
    return commands_equal(flatten(), other.flatten(), atol);
}

// ── Transformations ──

namespace {

/// Internal block that wraps an existing command set with a transformation.
class TransformedBlock : public Block {
public:
    TransformedBlock(std::vector<Command> cmds, uint32_t nq, const std::string& nm) {
        commands_ = std::move(cmds);
        n_qubits = nq;
        name = nm;
        built_ = true;
    }
    void build() override { /* already built */ }
};

}  // anonymous namespace

ref<Block> Block::dagger() const {
    if (!built_) {
        throw std::runtime_error("Block::dagger() called before build()");
    }

    std::vector<Command> dag_cmds;
    dag_cmds.reserve(commands_.size());

    // Reverse order and adjoint each command
    for (auto it = commands_.rbegin(); it != commands_.rend(); ++it) {
        dag_cmds.push_back(it->dagger());
    }

    ref<Block> result(new TransformedBlock(
        std::move(dag_cmds), n_qubits, name + "_dag"));
    result->n_cbits = n_cbits;
    result->target_qubits = target_qubits;
    result->target_cbits  = target_cbits;
    result->n_controls = n_controls;
    result->control_state = control_state;
    return result;
}

ref<Block> Block::set_symbols(
    const std::unordered_map<std::string, double>& values) const
{
    if (!built_) {
        throw std::runtime_error("Block::set_symbols() called before build()");
    }

    std::vector<Command> new_cmds;
    new_cmds.reserve(commands_.size());
    for (const auto& cmd : commands_) {
        new_cmds.push_back(cmd.substitute(values));
    }

    ref<Block> result(new TransformedBlock(
        std::move(new_cmds), n_qubits, name));
    result->n_cbits = n_cbits;
    result->target_qubits = target_qubits;
    result->target_cbits  = target_cbits;
    result->n_controls = n_controls;
    result->control_state = control_state;
    return result;
}

ref<Block> Block::replace_symbols(
    const std::unordered_map<std::string, std::string>& mapping) const
{
    if (!built_) {
        throw std::runtime_error("Block::replace_symbols() called before build()");
    }

    // For symbol renaming, we substitute with new symbolic params
    // This is a string-level rename, not a value substitution
    std::vector<Command> new_cmds;
    new_cmds.reserve(commands_.size());

    for (const auto& cmd : commands_) {
        Command new_cmd = cmd;
        SmallVector<Param, 1> new_params;
        for (const auto& p : cmd.params) {
            // Rename via Param::rename_symbols which preserves the linear
            // coefficient + offset on the (coeff·sym + offset) form.  A plain
            // ``Param::symbol(new_name)`` would drop both — turning
            // ``Rx(0.5·θ)`` into ``Rx(φ)`` — and break parametric ansätze whose
            // parameters carry a non-unity coefficient by design (e.g. UCC /
            // ``TrotterAnsatzBlock`` ``per_step_factor``).
            new_params.push_back(p.rename_symbols(mapping));
        }
        new_cmd.params = std::move(new_params);
        new_cmds.push_back(std::move(new_cmd));
    }

    ref<Block> result(new TransformedBlock(
        std::move(new_cmds), n_qubits, name));
    result->n_cbits = n_cbits;
    result->target_qubits = target_qubits;
    result->target_cbits  = target_cbits;
    result->n_controls = n_controls;
    result->control_state = control_state;
    return result;
}

// ── Helpers ──

void Block::add_command(Command cmd) {
    commands_.push_back(std::move(cmd));
}

void Block::add_gate(GateType g, uint32_t q) {
    commands_.emplace_back(g, q);
}

void Block::add_gate(GateType g, uint32_t q, Param p) {
    commands_.emplace_back(g, q, std::move(p));
}

void Block::add_gate(GateType g, uint32_t q0, uint32_t q1) {
    commands_.emplace_back(g, q0, q1);
}

void Block::add_gate(GateType g, uint32_t q0, uint32_t q1, Param p) {
    commands_.emplace_back(g, q0, q1, std::move(p));
}

// ── Builder API (on Block base) ──

Block& Block::h(uint32_t q)   { add_gate(GateType::H, q); return *this; }
Block& Block::x(uint32_t q)   { add_gate(GateType::X, q); return *this; }
Block& Block::y(uint32_t q)   { add_gate(GateType::Y, q); return *this; }
Block& Block::z(uint32_t q)   { add_gate(GateType::Z, q); return *this; }
Block& Block::s(uint32_t q)   { add_gate(GateType::S, q); return *this; }
Block& Block::sdg(uint32_t q) { add_gate(GateType::Sdg, q); return *this; }
Block& Block::t(uint32_t q)   { add_gate(GateType::T, q); return *this; }
Block& Block::tdg(uint32_t q) { add_gate(GateType::Tdg, q); return *this; }
Block& Block::sx(uint32_t q)   { add_gate(GateType::SX, q); return *this; }
Block& Block::sxdg(uint32_t q) { add_gate(GateType::SXdg, q); return *this; }
Block& Block::id(uint32_t q)   { add_gate(GateType::Id, q); return *this; }

Block& Block::rx(uint32_t q, Param angle) {
    add_gate(GateType::Rx, q, std::move(angle));
    return *this;
}

Block& Block::ry(uint32_t q, Param angle) {
    add_gate(GateType::Ry, q, std::move(angle));
    return *this;
}

Block& Block::rz(uint32_t q, Param angle) {
    add_gate(GateType::Rz, q, std::move(angle));
    return *this;
}

Block& Block::p(uint32_t q, Param angle) {
    add_gate(GateType::P, q, std::move(angle));
    return *this;
}

Block& Block::cx(uint32_t c, uint32_t t) {
    add_gate(GateType::CX, c, t);
    return *this;
}

Block& Block::cy(uint32_t c, uint32_t t) {
    add_gate(GateType::CY, c, t);
    return *this;
}

Block& Block::cz(uint32_t c, uint32_t t) {
    add_gate(GateType::CZ, c, t);
    return *this;
}

Block& Block::swap(uint32_t q0, uint32_t q1) {
    add_gate(GateType::SWAP, q0, q1);
    return *this;
}

Block& Block::ccx(uint32_t c0, uint32_t c1, uint32_t tgt) {
    commands_.emplace_back(GateType::CCX,
        SmallVector<uint32_t, 2>{c0, c1, tgt});
    return *this;
}

Block& Block::cswap(uint32_t control, uint32_t q0, uint32_t q1) {
    commands_.emplace_back(GateType::CSWAP,
        SmallVector<uint32_t, 2>{control, q0, q1});
    return *this;
}

Block& Block::mcz(std::vector<uint32_t> qubits) {
    SmallVector<uint32_t, 2> sv;
    sv.reserve(qubits.size());
    for (uint32_t q : qubits) sv.push_back(q);
    commands_.emplace_back(GateType::MCZ, std::move(sv));
    return *this;
}

Block& Block::rzz(uint32_t q0, uint32_t q1, Param angle) {
    add_gate(GateType::RZZ, q0, q1, std::move(angle));
    return *this;
}

Block& Block::rxx(uint32_t q0, uint32_t q1, Param angle) {
    add_gate(GateType::RXX, q0, q1, std::move(angle));
    return *this;
}

Block& Block::ryy(uint32_t q0, uint32_t q1, Param angle) {
    add_gate(GateType::RYY, q0, q1, std::move(angle));
    return *this;
}

Block& Block::crx(uint32_t c, uint32_t t, Param angle) {
    add_gate(GateType::CRx, c, t, std::move(angle));
    return *this;
}

Block& Block::cry(uint32_t c, uint32_t t, Param angle) {
    add_gate(GateType::CRy, c, t, std::move(angle));
    return *this;
}

Block& Block::crz(uint32_t c, uint32_t t, Param angle) {
    add_gate(GateType::CRz, c, t, std::move(angle));
    return *this;
}

Block& Block::cp(uint32_t c, uint32_t t, Param angle) {
    add_gate(GateType::CP, c, t, std::move(angle));
    return *this;
}

Block& Block::ecr(uint32_t q0, uint32_t q1) {
    add_gate(GateType::ECR, q0, q1);
    return *this;
}

Block& Block::iswap(uint32_t q0, uint32_t q1) {
    add_gate(GateType::iSWAP, q0, q1);
    return *this;
}

Block& Block::iswapdg(uint32_t q0, uint32_t q1) {
    add_gate(GateType::iSWAPdg, q0, q1);
    return *this;
}

Block& Block::ch(uint32_t c, uint32_t t) {
    add_gate(GateType::CH, c, t);
    return *this;
}
Block& Block::cs(uint32_t c, uint32_t t) {
    add_gate(GateType::CS, c, t);
    return *this;
}
Block& Block::csdg(uint32_t c, uint32_t t) {
    add_gate(GateType::CSdg, c, t);
    return *this;
}
Block& Block::csx(uint32_t c, uint32_t t) {
    add_gate(GateType::CSX, c, t);
    return *this;
}
Block& Block::csxdg(uint32_t c, uint32_t t) {
    add_gate(GateType::CSXdg, c, t);
    return *this;
}

Block& Block::gphase(Param angle) {
    Command cmd;
    cmd.gate = GateType::GPhase;
    cmd.params.push_back(std::move(angle));
    commands_.push_back(std::move(cmd));
    return *this;
}

Block& Block::u(uint32_t q, Param theta, Param phi, Param lambda) {
    Command cmd;
    cmd.gate = GateType::U;
    cmd.qubits.push_back(q);
    cmd.params.push_back(std::move(theta));
    cmd.params.push_back(std::move(phi));
    cmd.params.push_back(std::move(lambda));
    commands_.push_back(std::move(cmd));
    return *this;
}

Block& Block::cu(uint32_t c, uint32_t t,
                 Param theta, Param phi, Param lambda, Param gamma) {
    Command cmd;
    cmd.gate = GateType::CU;
    cmd.qubits.push_back(c);
    cmd.qubits.push_back(t);
    cmd.params.push_back(std::move(theta));
    cmd.params.push_back(std::move(phi));
    cmd.params.push_back(std::move(lambda));
    cmd.params.push_back(std::move(gamma));
    commands_.push_back(std::move(cmd));
    return *this;
}

Block& Block::measure(uint32_t qubit, uint32_t cbit) {
    Command cmd;
    cmd.gate = GateType::Measure;
    cmd.qubits = {qubit};
    cmd.cbits = {cbit};
    commands_.push_back(std::move(cmd));
    return *this;
}

Block& Block::reset(uint32_t qubit) {
    Command cmd;
    cmd.gate = GateType::Reset;
    cmd.qubits = {qubit};
    commands_.push_back(std::move(cmd));
    return *this;
}

Block& Block::barrier(std::vector<uint32_t> qs) {
    Command cmd;
    cmd.gate = GateType::Barrier;
    for (auto q : qs) cmd.qubits.push_back(q);
    commands_.push_back(std::move(cmd));
    return *this;
}

// ── Variadic / bulk builder overloads ──
//
// Each pre-reserves `commands_.reserve(commands_.size() + N)` and emplaces
// commands directly in a tight loop.  The Python tax (per-call nanobind
// crossing) collapses to a single dispatch regardless of N.

Block& Block::h(const std::vector<uint32_t>& qubits) {
    commands_.reserve(commands_.size() + qubits.size());
    for (uint32_t q : qubits) commands_.emplace_back(GateType::H, q);
    return *this;
}
Block& Block::x(const std::vector<uint32_t>& qubits) {
    commands_.reserve(commands_.size() + qubits.size());
    for (uint32_t q : qubits) commands_.emplace_back(GateType::X, q);
    return *this;
}
Block& Block::y(const std::vector<uint32_t>& qubits) {
    commands_.reserve(commands_.size() + qubits.size());
    for (uint32_t q : qubits) commands_.emplace_back(GateType::Y, q);
    return *this;
}
Block& Block::z(const std::vector<uint32_t>& qubits) {
    commands_.reserve(commands_.size() + qubits.size());
    for (uint32_t q : qubits) commands_.emplace_back(GateType::Z, q);
    return *this;
}
Block& Block::s(const std::vector<uint32_t>& qubits) {
    commands_.reserve(commands_.size() + qubits.size());
    for (uint32_t q : qubits) commands_.emplace_back(GateType::S, q);
    return *this;
}
Block& Block::sdg(const std::vector<uint32_t>& qubits) {
    commands_.reserve(commands_.size() + qubits.size());
    for (uint32_t q : qubits) commands_.emplace_back(GateType::Sdg, q);
    return *this;
}
Block& Block::t(const std::vector<uint32_t>& qubits) {
    commands_.reserve(commands_.size() + qubits.size());
    for (uint32_t q : qubits) commands_.emplace_back(GateType::T, q);
    return *this;
}
Block& Block::tdg(const std::vector<uint32_t>& qubits) {
    commands_.reserve(commands_.size() + qubits.size());
    for (uint32_t q : qubits) commands_.emplace_back(GateType::Tdg, q);
    return *this;
}
Block& Block::sx(const std::vector<uint32_t>& qubits) {
    commands_.reserve(commands_.size() + qubits.size());
    for (uint32_t q : qubits) commands_.emplace_back(GateType::SX, q);
    return *this;
}
Block& Block::sxdg(const std::vector<uint32_t>& qubits) {
    commands_.reserve(commands_.size() + qubits.size());
    for (uint32_t q : qubits) commands_.emplace_back(GateType::SXdg, q);
    return *this;
}
Block& Block::id(const std::vector<uint32_t>& qubits) {
    commands_.reserve(commands_.size() + qubits.size());
    for (uint32_t q : qubits) commands_.emplace_back(GateType::Id, q);
    return *this;
}

Block& Block::rx(const std::vector<std::pair<uint32_t, Param>>& gates) {
    commands_.reserve(commands_.size() + gates.size());
    for (const auto& [q, angle] : gates) commands_.emplace_back(GateType::Rx, q, angle);
    return *this;
}
Block& Block::ry(const std::vector<std::pair<uint32_t, Param>>& gates) {
    commands_.reserve(commands_.size() + gates.size());
    for (const auto& [q, angle] : gates) commands_.emplace_back(GateType::Ry, q, angle);
    return *this;
}
Block& Block::rz(const std::vector<std::pair<uint32_t, Param>>& gates) {
    commands_.reserve(commands_.size() + gates.size());
    for (const auto& [q, angle] : gates) commands_.emplace_back(GateType::Rz, q, angle);
    return *this;
}
Block& Block::p(const std::vector<std::pair<uint32_t, Param>>& gates) {
    commands_.reserve(commands_.size() + gates.size());
    for (const auto& [q, angle] : gates) commands_.emplace_back(GateType::P, q, angle);
    return *this;
}

Block& Block::cx(const std::vector<std::pair<uint32_t, uint32_t>>& pairs) {
    commands_.reserve(commands_.size() + pairs.size());
    for (const auto& [c, t] : pairs) commands_.emplace_back(GateType::CX, c, t);
    return *this;
}
Block& Block::cy(const std::vector<std::pair<uint32_t, uint32_t>>& pairs) {
    commands_.reserve(commands_.size() + pairs.size());
    for (const auto& [c, t] : pairs) commands_.emplace_back(GateType::CY, c, t);
    return *this;
}
Block& Block::cz(const std::vector<std::pair<uint32_t, uint32_t>>& pairs) {
    commands_.reserve(commands_.size() + pairs.size());
    for (const auto& [c, t] : pairs) commands_.emplace_back(GateType::CZ, c, t);
    return *this;
}
Block& Block::swap(const std::vector<std::pair<uint32_t, uint32_t>>& pairs) {
    commands_.reserve(commands_.size() + pairs.size());
    for (const auto& [a, b] : pairs) commands_.emplace_back(GateType::SWAP, a, b);
    return *this;
}
Block& Block::ecr(const std::vector<std::pair<uint32_t, uint32_t>>& pairs) {
    commands_.reserve(commands_.size() + pairs.size());
    for (const auto& [a, b] : pairs) commands_.emplace_back(GateType::ECR, a, b);
    return *this;
}
Block& Block::iswap(const std::vector<std::pair<uint32_t, uint32_t>>& pairs) {
    commands_.reserve(commands_.size() + pairs.size());
    for (const auto& [a, b] : pairs) commands_.emplace_back(GateType::iSWAP, a, b);
    return *this;
}
Block& Block::iswapdg(const std::vector<std::pair<uint32_t, uint32_t>>& pairs) {
    commands_.reserve(commands_.size() + pairs.size());
    for (const auto& [a, b] : pairs) commands_.emplace_back(GateType::iSWAPdg, a, b);
    return *this;
}
Block& Block::ch(const std::vector<std::pair<uint32_t, uint32_t>>& pairs) {
    commands_.reserve(commands_.size() + pairs.size());
    for (const auto& [c, t] : pairs) commands_.emplace_back(GateType::CH, c, t);
    return *this;
}
Block& Block::cs(const std::vector<std::pair<uint32_t, uint32_t>>& pairs) {
    commands_.reserve(commands_.size() + pairs.size());
    for (const auto& [c, t] : pairs) commands_.emplace_back(GateType::CS, c, t);
    return *this;
}
Block& Block::csdg(const std::vector<std::pair<uint32_t, uint32_t>>& pairs) {
    commands_.reserve(commands_.size() + pairs.size());
    for (const auto& [c, t] : pairs) commands_.emplace_back(GateType::CSdg, c, t);
    return *this;
}
Block& Block::csx(const std::vector<std::pair<uint32_t, uint32_t>>& pairs) {
    commands_.reserve(commands_.size() + pairs.size());
    for (const auto& [c, t] : pairs) commands_.emplace_back(GateType::CSX, c, t);
    return *this;
}
Block& Block::csxdg(const std::vector<std::pair<uint32_t, uint32_t>>& pairs) {
    commands_.reserve(commands_.size() + pairs.size());
    for (const auto& [c, t] : pairs) commands_.emplace_back(GateType::CSXdg, c, t);
    return *this;
}

Block& Block::crx(const std::vector<std::tuple<uint32_t, uint32_t, Param>>& gates) {
    commands_.reserve(commands_.size() + gates.size());
    for (const auto& [c, t, angle] : gates) commands_.emplace_back(GateType::CRx, c, t, angle);
    return *this;
}
Block& Block::cry(const std::vector<std::tuple<uint32_t, uint32_t, Param>>& gates) {
    commands_.reserve(commands_.size() + gates.size());
    for (const auto& [c, t, angle] : gates) commands_.emplace_back(GateType::CRy, c, t, angle);
    return *this;
}
Block& Block::crz(const std::vector<std::tuple<uint32_t, uint32_t, Param>>& gates) {
    commands_.reserve(commands_.size() + gates.size());
    for (const auto& [c, t, angle] : gates) commands_.emplace_back(GateType::CRz, c, t, angle);
    return *this;
}
Block& Block::cp(const std::vector<std::tuple<uint32_t, uint32_t, Param>>& gates) {
    commands_.reserve(commands_.size() + gates.size());
    for (const auto& [c, t, angle] : gates) commands_.emplace_back(GateType::CP, c, t, angle);
    return *this;
}
Block& Block::rzz(const std::vector<std::tuple<uint32_t, uint32_t, Param>>& gates) {
    commands_.reserve(commands_.size() + gates.size());
    for (const auto& [a, b, angle] : gates) commands_.emplace_back(GateType::RZZ, a, b, angle);
    return *this;
}
Block& Block::rxx(const std::vector<std::tuple<uint32_t, uint32_t, Param>>& gates) {
    commands_.reserve(commands_.size() + gates.size());
    for (const auto& [a, b, angle] : gates) commands_.emplace_back(GateType::RXX, a, b, angle);
    return *this;
}
Block& Block::ryy(const std::vector<std::tuple<uint32_t, uint32_t, Param>>& gates) {
    commands_.reserve(commands_.size() + gates.size());
    for (const auto& [a, b, angle] : gates) commands_.emplace_back(GateType::RYY, a, b, angle);
    return *this;
}

Block& Block::ccx(const std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>& triples) {
    commands_.reserve(commands_.size() + triples.size());
    for (const auto& [c0, c1, t] : triples) {
        commands_.emplace_back(GateType::CCX,
            SmallVector<uint32_t, 2>{c0, c1, t});
    }
    return *this;
}
Block& Block::cswap(const std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>& triples) {
    commands_.reserve(commands_.size() + triples.size());
    for (const auto& [c, q0, q1] : triples) {
        commands_.emplace_back(GateType::CSWAP,
            SmallVector<uint32_t, 2>{c, q0, q1});
    }
    return *this;
}

Block& Block::measure(const std::vector<std::pair<uint32_t, uint32_t>>& gates) {
    commands_.reserve(commands_.size() + gates.size());
    for (const auto& [q, c] : gates) {
        Command cmd;
        cmd.gate = GateType::Measure;
        cmd.qubits = {q};
        cmd.cbits = {c};
        commands_.push_back(std::move(cmd));
    }
    return *this;
}

Block& Block::reset(const std::vector<uint32_t>& qubits) {
    commands_.reserve(commands_.size() + qubits.size());
    for (uint32_t q : qubits) {
        Command cmd;
        cmd.gate = GateType::Reset;
        cmd.qubits = {q};
        commands_.push_back(std::move(cmd));
    }
    return *this;
}

// ── Synthesis builders ──

Block& Block::state_preparation(
    const std::vector<std::complex<double>>& amplitudes)
{
    synthesis::state_preparation(*this, amplitudes);
    return *this;
}

Block& Block::diagonal_unitary(
    const std::vector<std::complex<double>>& diagonal_elements)
{
    synthesis::diagonal_unitary(*this, diagonal_elements);
    return *this;
}

Block& Block::unitary_synthesis(const Eigen::MatrixXcd& U)
{
    synthesis::unitary_synthesis(*this, U);
    return *this;
}

Block& Block::pauli_exp(const PauliString& pauli, Param angle)
{
    synthesis::pauli_exp(*this, pauli, angle);
    return *this;
}

Block& Block::commuting_pauli_set_exp(
    const std::vector<PauliString>& paulis,
    const std::vector<Param>& angles)
{
    synthesis::commuting_pauli_set_exp(*this, paulis, angles);
    return *this;
}

// ── SimpleBlock ──

SimpleBlock::SimpleBlock(uint32_t nq, const std::string& nm) {
    n_qubits = nq;
    name = nm;
}

void SimpleBlock::build() {
    // Mirrors CompositeBlock::resolve_cbits: 0 means "unset", so an explicitly
    // declared wider register survives build and rebuilds.  It may only widen:
    // a narrower value would disagree with the simulator, both emitters and
    // cbit_register_width, which all size the register from the commands.
    n_cbits = std::max(n_cbits, cbit_register_width(commands_));
    built_ = true;
}

std::vector<uint32_t> SimpleBlock::add_register(const std::string& /*name*/, uint32_t size) {
    std::vector<uint32_t> indices(size);
    for (uint32_t i = 0; i < size; ++i)
        indices[i] = next_qubit_++;
    return indices;
}

}  // namespace qarpx
