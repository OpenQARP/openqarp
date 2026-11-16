#include "qarpx/emit/qasm3_emitter.h"

#include "qarpx/core/command.h"

#include <algorithm>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace qarpx {

namespace {

/// Render qubit `q` of register `q` as `q[k]`.  (Single-register layout.)
std::string qref(uint32_t q) {
    return "q[" + std::to_string(q) + "]";
}

/// Render classical bit `c` of register `c` as `c[k]`.
std::string cref(uint32_t c) {
    return "c[" + std::to_string(c) + "]";
}

/// Format a comma-separated qubit list `q[i], q[j], …`.
std::string qref_list(const SmallVector<uint32_t, 2>& qubits) {
    std::string out;
    for (size_t i = 0; i < qubits.size(); ++i) {
        if (i) out += ", ";
        out += qref(qubits[i]);
    }
    return out;
}

/// Format a parameter list `(p0, p1, …)` for parametric gates.
std::string param_list(const SmallVector<Param, 1>& params) {
    if (params.empty()) return "";
    std::string out = "(";
    for (size_t i = 0; i < params.size(); ++i) {
        if (i) out += ", ";
        out += params[i].to_string();
    }
    out += ")";
    return out;
}

/// Render a BranchBegin's AND-condition tuple as a QASM 3 boolean expression.
/// ``== true`` / ``== false`` rather than ``== 1`` / ``== 0``: strict external
/// parsers (qiskit's importer) demand a bool comparison for a bit operand.
std::string render_condition(const Command& begin) {
    std::string out;
    for (size_t i = 0; i < begin.condition_bits.size(); ++i) {
        if (i) out += " && ";
        out += cref(begin.condition_bits[i]);
        out += " == ";
        out += begin.condition_values[i] ? "true" : "false";
    }
    return out;
}

/// Render `circuit_name` as a header comment.  Block names may contain
/// newlines (a composite block's name is built from its children's), and a
/// bare `// <name>` would then emit lines that are not comments at all —
/// an invalid program, and one this emitter's own absorber cannot read.
std::string header_comment(const std::string& name) {
    std::string out = "//";
    for (size_t i = 0; i < name.size(); ++i) {
        if (name[i] == '\n') {
            out += "\n//";
        } else {
            if (out.back() == '/') out += ' ';
            out += name[i];
        }
    }
    out += "\n";
    return out;
}

/// Emit one Command into the output stream as a QASM 3 statement.
void emit_command(std::ostringstream& os, const Command& cmd, int depth = 0) {
    auto p = [&]() { return param_list(cmd.params); };
    auto q = [&]() { return qref_list(cmd.qubits); };
    // Each `if { ... }` nesting level adds two spaces on top of the base
    // two-space body indent.
    const std::string extra(static_cast<size_t>(2 * depth), ' ');
    os << extra;

    switch (cmd.gate) {
        // ── 1Q no-param ──
        case GateType::X:    os << "  x "    << q() << ";\n"; break;
        case GateType::Y:    os << "  y "    << q() << ";\n"; break;
        case GateType::Z:    os << "  z "    << q() << ";\n"; break;
        case GateType::H:    os << "  h "    << q() << ";\n"; break;
        case GateType::S:    os << "  s "    << q() << ";\n"; break;
        case GateType::Sdg:  os << "  sdg "  << q() << ";\n"; break;
        case GateType::T:    os << "  t "    << q() << ";\n"; break;
        case GateType::Tdg:  os << "  tdg "  << q() << ";\n"; break;
        case GateType::SX:   os << "  sx "   << q() << ";\n"; break;
        case GateType::Id:   os << "  id "   << q() << ";\n"; break;
        // No stdgates symbol: OpenQASM inverse modifier on sx.
        case GateType::SXdg: os << "  inv @ sx " << q() << ";\n"; break;

        // ── 1Q parametric ──
        case GateType::Rx:   os << "  rx" << p() << " " << q() << ";\n"; break;
        case GateType::Ry:   os << "  ry" << p() << " " << q() << ";\n"; break;
        case GateType::Rz:   os << "  rz" << p() << " " << q() << ";\n"; break;
        case GateType::P:    os << "  p"  << p() << " " << q() << ";\n"; break;

        // ── 1Q general U(θ, φ, λ) — language built-in (capital U) ──
        case GateType::U:    os << "  U" << p() << " " << q() << ";\n"; break;

        // ── 2Q no-param ──
        case GateType::CX:    os << "  cx "    << q() << ";\n"; break;
        case GateType::CY:    os << "  cy "    << q() << ";\n"; break;
        case GateType::CZ:    os << "  cz "    << q() << ";\n"; break;
        case GateType::SWAP:  os << "  swap "  << q() << ";\n"; break;
        case GateType::ECR:   os << "  ecr "   << q() << ";\n"; break;
        case GateType::iSWAP: os << "  iswap " << q() << ";\n"; break;
        case GateType::CH:    os << "  ch "    << q() << ";\n"; break;

        // ── 2Q named-inverse (no stdgates symbol) ──
        case GateType::iSWAPdg:
            os << "  inv @ iswap " << q() << ";\n";
            break;

        // Controlled Clifford singles without a stdgates symbol compose via
        // `ctrl @` on the 1Q base, as stdgates.inc itself defines ch.
        case GateType::CS:    os << "  ctrl @ s "        << q() << ";\n"; break;
        case GateType::CSdg:  os << "  ctrl @ sdg "      << q() << ";\n"; break;
        case GateType::CSX:   os << "  ctrl @ sx "       << q() << ";\n"; break;
        case GateType::CSXdg: os << "  ctrl @ inv @ sx " << q() << ";\n"; break;

        // ── 2Q parametric ──
        case GateType::CRx: os << "  crx" << p() << " " << q() << ";\n"; break;
        case GateType::CRy: os << "  cry" << p() << " " << q() << ";\n"; break;
        case GateType::CRz: os << "  crz" << p() << " " << q() << ";\n"; break;
        case GateType::CP:  os << "  cp"  << p() << " " << q() << ";\n"; break;
        case GateType::RZZ: os << "  rzz" << p() << " " << q() << ";\n"; break;
        case GateType::RXX: os << "  rxx" << p() << " " << q() << ";\n"; break;
        case GateType::RYY: os << "  ryy" << p() << " " << q() << ";\n"; break;

        // ── 2Q controlled-U with global phase γ ──
        case GateType::CU:  os << "  cu"  << p() << " " << q() << ";\n"; break;

        // ── 3Q no-param ──
        case GateType::CCX:    os << "  ccx "    << q() << ";\n"; break;
        case GateType::CSWAP:  os << "  cswap "  << q() << ";\n"; break;

        // ── n-controlled Z: ctrl(n) @ z q[c0], …, q[c_{n-1}], q[t]; ──
        case GateType::MCZ: {
            const auto n_ctrl = static_cast<uint32_t>(cmd.qubits.size() - 1);
            os << "  ctrl(" << n_ctrl << ") @ z " << q() << ";\n";
            break;
        }

        // ── 0Q global phase ──
        case GateType::GPhase:
            os << "  gphase" << p() << ";\n";
            break;

        // ── Special / non-unitary ──
        case GateType::Barrier:
            os << "  barrier " << q() << ";\n";
            break;
        case GateType::Measure: {
            const uint32_t cbit = cmd.cbits.empty() ? cmd.qubits[0] : cmd.cbits[0];
            os << "  " << cref(cbit) << " = measure " << qref(cmd.qubits[0]) << ";\n";
            break;
        }
        case GateType::Reset:
            os << "  reset " << qref(cmd.qubits[0]) << ";\n";
            break;

        // Branch markers are handled by the body-emission loop directly; they
        // shouldn't reach this dispatch.  Defend anyway with a clear error so
        // a malformed flatten doesn't silently corrupt the output.
        case GateType::BranchBegin:
        case GateType::BranchElse:
        case GateType::BranchEnd:
            throw std::runtime_error(
                "QASM3Emitter::emit_command: stray "
                + std::string(gate_name(cmd.gate))
                + " — branch markers must be dispatched by the outer loop.");

        case GateType::Custom:
            // Should be filtered out by validate() before reaching here; defend
            // anyway so a stray Custom doesn't silently corrupt the output.
            throw std::runtime_error(
                "QASM3Emitter: Custom gate has no OpenQASM 3 base-profile representation. "
                "Decompose it first.");

        default:
            throw std::runtime_error(
                "QASM3Emitter: unsupported gate '" +
                std::string(gate_name(cmd.gate)) + "'");
    }
}

/// Emit `gate` definitions for the gate names the program uses that
/// `stdgates.inc` does not define (ecr, iswap, rzz, rxx, ryy) — without them
/// an external parser rejects the program.  Each body is the §11 phase-exact
/// identity for its gate, so the defined gate is the emitted gate on the
/// nose, not up to phase.  `inv @ iswap` (iSWAPdg) needs the iswap symbol.
void emit_gate_definitions(std::ostringstream& os, const std::vector<Command>& cmds) {
    bool rzz = false, rxx = false, ryy = false, ecr = false, iswap = false;
    for (const auto& cmd : cmds) {
        switch (cmd.gate) {
            case GateType::RZZ:      rzz = true;   break;
            case GateType::RXX:      rxx = true;   break;
            case GateType::RYY:      ryy = true;   break;
            case GateType::ECR:      ecr = true;   break;
            case GateType::iSWAP:
            case GateType::iSWAPdg:  iswap = true; break;
            default: break;
        }
    }
    const bool any = rzz || rxx || ryy || ecr || iswap;
    if (!any) return;

    if (rzz || rxx || ryy)
        os << "gate rzz(theta) a, b { cx a, b; rz(theta) b; cx a, b; }\n";
    if (rxx)
        os << "gate rxx(theta) a, b { h a; h b; rzz(theta) a, b; h a; h b; }\n";
    if (ryy)
        os << "gate ryy(theta) a, b { rx(pi/2) a; rx(pi/2) b; rzz(theta) a, b; "
              "rx(-pi/2) a; rx(-pi/2) b; }\n";
    if (ecr) {
        os << "gate rzx(theta) a, b { h b; cx a, b; rz(theta) b; cx a, b; h b; }\n";
        os << "gate ecr a, b { rzx(pi/4) a, b; x a; rzx(-pi/4) a, b; }\n";
    }
    if (iswap)
        os << "gate iswap a, b { s a; s b; h a; cx a, b; cx b, a; h b; }\n";
    os << "\n";
}

/// Collect the full set of free symbols across all command params, in stable
/// (sorted) order so emitted output is deterministic.
std::vector<std::string> collect_free_symbols(const std::vector<Command>& cmds) {
    std::set<std::string> uniq;
    for (const auto& cmd : cmds) {
        for (const auto& p : cmd.params) {
            for (const auto& s : p.free_symbols()) uniq.insert(s);
        }
    }
    return std::vector<std::string>(uniq.begin(), uniq.end());
}

/// Size of the emitted `bit[K] c;` declaration.
///
/// Normally the shared contract width (`cbit_register_width`, core/command.h).
/// OpenQASM 3 cannot express a measurement with no classical target, so a
/// `Measure` with empty `cbits` is emitted against `c[qubit]` (see the Measure
/// case in the dispatch above).  That index is deliberately *not* part of the
/// register contract, so widen the local declaration to keep the emitted
/// program valid.  Emitter-local concession — do not propagate it.
uint32_t emitted_register_size(const std::vector<Command>& cmds) {
    uint32_t width = cbit_register_width(cmds);
    for (const auto& cmd : cmds) {
        if (cmd.gate == GateType::Measure && cmd.cbits.empty())
            width = std::max(width, cmd.qubits[0] + 1);
    }
    return width;
}

}  // namespace

std::string QASM3Emitter::emit(
    const std::vector<Command>& commands,
    uint32_t n_qubits,
    const std::string& circuit_name) const
{
    throw_if_invalid(commands);

    std::ostringstream os;

    // Header.  `circuit_name` is informational only — OpenQASM 3 has no
    // top-level program name keyword.
    os << header_comment(circuit_name);
    os << "OPENQASM 3.0;\n";
    os << "include \"stdgates.inc\";\n\n";

    // Definitions for gate names stdgates.inc lacks, when the program uses them.
    emit_gate_definitions(os, commands);

    // Symbolic param declarations (input floats), sorted for determinism.
    const auto symbols = collect_free_symbols(commands);
    for (const auto& name : symbols) {
        os << "input float[64] " << name << ";\n";
    }
    if (!symbols.empty()) os << "\n";

    // Quantum register.
    os << "qubit[" << n_qubits << "] q;\n";

    // Classical register sized to fit every Measure target; omitted if no
    // measurement appears in the program.
    const uint32_t n_cbits = emitted_register_size(commands);
    if (n_cbits > 0) {
        os << "bit[" << n_cbits << "] c;\n";
    }
    os << "\n";

    // Body — track classical-conditional nesting depth; BranchBegin/Else/End
    // open and close `if (cond) { ... } else { ... }` regions, body commands
    // inside indent one level deeper.
    int depth = 0;
    auto pad = [](int d) {
        return std::string(static_cast<size_t>(2 * (d + 1)), ' ');
    };
    for (const auto& cmd : commands) {
        switch (cmd.gate) {
            case GateType::BranchBegin:
                os << pad(depth) << "if (" << render_condition(cmd) << ") {\n";
                ++depth;
                break;
            case GateType::BranchElse:
                os << pad(depth - 1) << "} else {\n";
                break;
            case GateType::BranchEnd:
                --depth;
                os << pad(depth) << "}\n";
                break;
            default:
                emit_command(os, cmd, depth);
                break;
        }
    }

    return os.str();
}

void QASM3Emitter::emit_to_file(
    const std::vector<Command>& commands,
    uint32_t n_qubits,
    const std::string& path,
    const std::string& circuit_name) const
{
    std::string content = emit(commands, n_qubits, circuit_name);
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("QASM3Emitter: cannot open file " + path);
    }
    file << content;
}

}  // namespace qarpx
