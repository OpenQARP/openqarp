#include "qarpx/core/command.h"

#include <Eigen/Dense>

#include <algorithm>
#include <set>
#include <sstream>
#include <stdexcept>

namespace qarpx {

bool Command::is_parametric() const {
    for (const auto& p : params) {
        if (p.is_symbolic()) return true;
    }
    return false;
}

Command Command::dagger() const {
    Command result = *this;

    // Self-adjoint gates are their own inverse
    if (gate_is_self_adjoint(gate)) return result;

    // Named inverse pairs (S↔Sdg, T↔Tdg, iSWAP↔iSWAPdg)
    result.gate = gate_adjoint(gate);
    if (result.gate != gate) return result;

    // OpenQASM 3 general U / controlled-U: swap (φ, λ) and negate everything.
    //   U(θ, φ, λ)†       = U(-θ, -λ, -φ)
    //   CU(θ, φ, λ, γ)†   = CU(-θ, -λ, -φ, -γ)
    if (gate == GateType::U) {
        result.params[0] = -params[0];           // -θ
        result.params[1] = -params[2];           // -λ (was at index 2)
        result.params[2] = -params[1];           // -φ (was at index 1)
        return result;
    }
    if (gate == GateType::CU) {
        result.params[0] = -params[0];           // -θ
        result.params[1] = -params[2];           // -λ
        result.params[2] = -params[1];           // -φ
        result.params[3] = -params[3];           // -γ
        return result;
    }

    // Custom gates carry a unitary matrix; dagger = conjugate transpose.
    if (gate == GateType::Custom && unitary) {
        result.unitary = std::make_shared<const Eigen::MatrixXcd>(unitary->adjoint());
        return result;
    }

    // Parametric gates whose dagger is per-parameter negation
    // (Rx, Ry, Rz, P, CRx, CRy, CRz, CP, RZZ, RXX, RYY, GPhase).
    for (auto& p : result.params) {
        p = -p;
    }

    return result;
}

Command Command::remap_qubits(const std::vector<uint32_t>& mapping) const {
    Command result = *this;
    for (auto& q : result.qubits) {
        if (q >= mapping.size()) {
            // Name the gate and its full qubit list: the bad index alone does
            // not identify which call site produced the command.
            std::ostringstream oss;
            oss << "Command::remap_qubits: " << gate_name(gate)
                << " acts on qubit " << q << " (qubits [";
            for (std::size_t i = 0; i < qubits.size(); ++i)
                oss << (i ? ", " : "") << qubits[i];
            oss << "]) but the target mapping has only " << mapping.size()
                << (mapping.size() == 1 ? " entry" : " entries")
                << ", so qubit indices must be < " << mapping.size();
            throw std::out_of_range(oss.str());
        }
        q = mapping[q];
    }
    return result;
}

Command Command::substitute(const std::unordered_map<std::string, double>& values) const {
    Command result = *this;
    for (auto& p : result.params) {
        p = p.substitute(values);
    }
    return result;
}

Command Command::rename_symbols(
    const std::unordered_map<std::string, std::string>& mapping) const
{
    Command result = *this;
    for (auto& p : result.params) {
        p = p.rename_symbols(mapping);
    }
    return result;
}

std::pair<std::vector<Command>, std::vector<RenamedOccurrence>>
rename_symbol_occurrences(
    const std::vector<Command>& commands,
    const std::string& symbol,
    const std::string& prefix)
{
    std::vector<Command> out;
    out.reserve(commands.size());
    std::vector<RenamedOccurrence> renamed;
    for (std::size_t ci = 0; ci < commands.size(); ++ci) {
        Command cmd = commands[ci];
        for (std::size_t pi = 0; pi < cmd.params.size(); ++pi) {
            const Param& p = cmd.params[pi];
            if (!p.is_symbolic()) continue;
            const auto syms = p.free_symbols();
            if (std::find(syms.begin(), syms.end(), symbol) == syms.end()) continue;
            std::string name = prefix + std::to_string(ci);
            name.push_back('\0');
            name += std::to_string(pi);
            cmd.params[pi] = p.rename_symbols({{symbol, name}});
            renamed.push_back(RenamedOccurrence{ci, pi, name});
        }
        out.push_back(std::move(cmd));
    }
    return {std::move(out), std::move(renamed)};
}

bool Command::operator==(const Command& other) const {
    if (gate != other.gate) return false;
    if (qubits != other.qubits) return false;
    if (params.size() != other.params.size()) return false;
    for (std::size_t i = 0; i < params.size(); ++i) {
        if (params[i] != other.params[i]) return false;
    }
    if (cbits != other.cbits) return false;
    if (condition_bits   != other.condition_bits)   return false;
    if (condition_values != other.condition_values) return false;
    return true;
}

std::string Command::to_string() const {
    std::ostringstream oss;
    oss << gate_name(gate);

    if (!params.empty()) {
        oss << "(";
        for (std::size_t i = 0; i < params.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << params[i];
        }
        oss << ")";
    }

    oss << " [";
    for (std::size_t i = 0; i < qubits.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << "q" << qubits[i];
    }
    oss << "]";

    if (!cbits.empty()) {
        oss << " -> [";
        for (std::size_t i = 0; i < cbits.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << "c" << cbits[i];
        }
        oss << "]";
    }

    // Classical condition, so a flattened stream shows which cbit a branch
    // reads (a sibling-offset mistake is invisible otherwise).
    if (!condition_bits.empty()) {
        oss << " if ";
        for (std::size_t i = 0; i < condition_bits.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << "c" << condition_bits[i] << "=="
                << (i < condition_values.size() && condition_values[i] ? 1 : 0);
        }
    }

    return oss.str();
}

std::vector<Command> substitute_all(
    const std::vector<Command>& commands,
    const std::unordered_map<std::string, double>& values)
{
    std::vector<Command> out;
    out.reserve(commands.size());
    for (const auto& cmd : commands) {
        out.push_back(cmd.substitute(values));
    }
    return out;
}

bool Command::approx_equal(const Command& other, double atol) const {
    if (gate != other.gate) return false;
    if (qubits != other.qubits) return false;
    if (params.size() != other.params.size()) return false;
    for (std::size_t i = 0; i < params.size(); ++i) {
        if (!params[i].approx_equal(other.params[i], atol)) return false;
    }
    if (cbits != other.cbits) return false;
    if (condition_bits   != other.condition_bits)   return false;
    if (condition_values != other.condition_values) return false;
    return true;
}

bool commands_equal(
    const std::vector<Command>& a,
    const std::vector<Command>& b,
    double atol)
{
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!a[i].approx_equal(b[i], atol)) return false;
    }
    return true;
}

void cbit_remap_in_place(
    std::vector<Command>& commands,
    const std::vector<uint32_t>& mapping)
{
    auto remap_one = [&](uint32_t local) -> uint32_t {
        if (local >= mapping.size()) {
            throw std::out_of_range(
                "cbit_remap_in_place: cbit " + std::to_string(local) +
                " out of range for mapping of size " +
                std::to_string(mapping.size()));
        }
        return mapping[local];
    };

    for (auto& cmd : commands) {
        for (auto& c : cmd.cbits)          c = remap_one(c);
        for (auto& c : cmd.condition_bits) c = remap_one(c);
    }
}

std::vector<uint32_t> uninitialised_condition_cbits(const std::vector<Command>& commands) {
    std::vector<bool> written;  // true once cbit i has been written by a Measure
    std::set<uint32_t> bad;
    for (const auto& cmd : commands) {
        // Read side: every condition bit must already have a write.
        for (auto b : cmd.condition_bits) {
            if (b >= written.size() || !written[b]) bad.insert(b);
        }
        // Write side: a Measure's cbits grow the written mask.
        if (cmd.gate == GateType::Measure) {
            for (auto b : cmd.cbits) {
                if (b >= written.size()) written.resize(b + 1, false);
                written[b] = true;
            }
        }
    }
    return std::vector<uint32_t>(bad.begin(), bad.end());
}

uint32_t cbit_register_width(const std::vector<Command>& commands) {
    uint32_t width = 0;
    auto track = [&width](uint32_t cbit) { width = std::max(width, cbit + 1); };

    for (const auto& cmd : commands) {
        // An empty `cbits` on a Measure is malformed, not a qubit-indexed
        // default — it contributes nothing (see the header).
        if (cmd.gate == GateType::Measure)
            for (auto c : cmd.cbits) track(c);
        for (auto b : cmd.condition_bits) track(b);
    }
    return width;
}

uint32_t n_nqb_gates(const std::vector<Command>& commands, uint32_t k) {
    uint32_t count = 0;
    for (const auto& cmd : commands) {
        if (gate_is_physical(cmd.gate) && cmd.qubits.size() == k) ++count;
    }
    return count;
}

uint32_t n_physical_gates(const std::vector<Command>& commands) {
    uint32_t count = 0;
    for (const auto& cmd : commands) {
        if (gate_is_physical(cmd.gate)) ++count;
    }
    return count;
}

uint32_t n_gates_of_type(const std::vector<Command>& commands, GateType gate) {
    uint32_t count = 0;
    for (const auto& cmd : commands) {
        if (cmd.gate == gate) ++count;
    }
    return count;
}

}  // namespace qarpx
