#include "qarpx/emit/qasm2_emitter.h"

#include "qarpx/core/command.h"
#include "qarpx/core/errors.h"

#include <algorithm>
#include <fstream>
#include <optional>
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

/// How the classical register is spelled.  OpenQASM 2's `if` compares a whole
/// `creg` to an integer, so conditioning on a single bit requires that bit to
/// *be* the register — hence a per-bit layout whenever the program branches,
/// and the conventional single register when it does not (§12.2).
struct CregLayout {
    uint32_t width   = 0;
    bool     per_bit = false;

    [[nodiscard]] std::string ref(uint32_t c) const {
        return per_bit ? "c" + std::to_string(c) + "[0]"
                       : "c[" + std::to_string(c) + "]";
    }

    /// The register *name* (not an indexed bit) — what `if (…)` compares.
    [[nodiscard]] std::string reg_name(uint32_t c) const {
        return per_bit ? "c" + std::to_string(c) : "c";
    }
};

/// Which `gate` definitions the program needs.  Dependencies are folded in
/// here rather than at the call sites: `rxx`/`ryy` bodies call `rzz`, and
/// `ecr`'s calls `rzx`, so requesting one requests the other.
struct GateDefs {
    bool swap = false, cswap = false, crx = false, cry = false;
    bool rzz = false, rxx = false, ryy = false;
    bool rzx = false, ecr = false;
    bool iswap = false, iswapdg = false, cu = false;
    bool sx = false, sxdg = false, cs = false, csdg = false, csx = false, csxdg = false;

    [[nodiscard]] bool any() const {
        return swap || cswap || crx || cry || rzz || rxx || ryy || rzx ||
               ecr || iswap || iswapdg || cu ||
               sx || sxdg || cs || csdg || csx || csxdg;
    }
};

GateDefs collect_gate_defs(const std::vector<Command>& cmds) {
    GateDefs d;
    for (const auto& cmd : cmds) {
        switch (cmd.gate) {
            case GateType::SWAP:    d.swap    = true; break;
            case GateType::CSWAP:   d.cswap   = true; break;
            case GateType::CRx:     d.crx     = true; break;
            case GateType::CRy:     d.cry     = true; break;
            case GateType::RZZ:     d.rzz     = true; break;
            case GateType::RXX:     d.rxx     = true; break;
            case GateType::RYY:     d.ryy     = true; break;
            case GateType::ECR:     d.ecr     = true; break;
            case GateType::iSWAP:   d.iswap   = true; break;
            case GateType::iSWAPdg: d.iswapdg = true; break;
            case GateType::CU:      d.cu      = true; break;
            case GateType::SX:      d.sx      = true; break;
            case GateType::SXdg:    d.sxdg    = true; break;
            case GateType::CS:      d.cs      = true; break;
            case GateType::CSdg:    d.csdg    = true; break;
            case GateType::CSX:     d.csx     = true; break;
            case GateType::CSXdg:   d.csxdg   = true; break;
            default: break;
        }
    }
    if (d.rxx || d.ryy)   d.rzz = true;
    if (d.ecr)            d.rzx = true;
    if (d.csx || d.csxdg) d.crx = true;
    return d;
}

/// Write the needed `gate` definitions, in dependency order.  Each body is the
/// §11 phase-exact identity for its gate — a QASM 2 file may be read back and
/// controlled, where a global phase becomes a relative one.  Only gates outside
/// the 23-gate spec `qelib1.inc` appear here; `swap`, `cswap`, `crx`, `cry`,
/// `rxx` and `rzz` are later qiskit additions, not part of the language.
void emit_gate_definitions(std::ostringstream& os, const GateDefs& d) {
    if (!d.any()) return;

    if (d.swap)
        os << "gate swap a, b { cx a, b; cx b, a; cx a, b; }\n";
    if (d.cswap)
        os << "gate cswap a, b, c { cx c, b; ccx a, b, c; cx c, b; }\n";
    if (d.crx)
        os << "gate crx(theta) a, b { u1(pi/2) b; cx a, b; u3(-theta/2,0,0) b; "
              "cx a, b; u3(theta/2,-pi/2,0) b; }\n";
    if (d.cry)
        os << "gate cry(theta) a, b { u3(theta/2,0,0) b; cx a, b; "
              "u3(-theta/2,0,0) b; cx a, b; }\n";
    // §2.6: H·P(±π/2)·H = e^{±iπ/4}·Rx(±π/2) = SX / SXdg, phase included.
    if (d.sx)
        os << "gate sx a { h a; s a; h a; }\n";
    if (d.sxdg)
        os << "gate sxdg a { h a; sdg a; h a; }\n";
    // §3.1: CS = CP(±π/2); CSX = P(π/4, c)·CRx(π/2) (both block-diagonal in c).
    if (d.cs)
        os << "gate cs a, b { cu1(pi/2) a, b; }\n";
    if (d.csdg)
        os << "gate csdg a, b { cu1(-pi/2) a, b; }\n";
    if (d.csx)
        os << "gate csx a, b { u1(pi/4) a; crx(pi/2) a, b; }\n";
    if (d.csxdg)
        os << "gate csxdg a, b { u1(-pi/4) a; crx(-pi/2) a, b; }\n";
    // `rz`, never qiskit's later `u1`: cx;u1(θ);cx is RZZ only up to e^{-iθ/2}.
    if (d.rzz)
        os << "gate rzz(theta) a, b { cx a, b; rz(theta) b; cx a, b; }\n";
    if (d.rxx)
        os << "gate rxx(theta) a, b { h a; h b; rzz(theta) a, b; h a; h b; }\n";
    if (d.ryy)
        os << "gate ryy(theta) a, b { rx(pi/2) a; rx(pi/2) b; rzz(theta) a, b; "
              "rx(-pi/2) a; rx(-pi/2) b; }\n";
    if (d.rzx)
        os << "gate rzx(theta) a, b { h b; cx a, b; rz(theta) b; cx a, b; h b; }\n";
    if (d.ecr)
        os << "gate ecr a, b { rzx(pi/4) a, b; x a; rzx(-pi/4) a, b; }\n";
    if (d.iswap)
        os << "gate iswap a, b { s a; s b; h a; cx a, b; cx b, a; h b; }\n";
    // OpenQASM 2 has no `inv @` modifier, so the dagger needs a real body.
    if (d.iswapdg)
        os << "gate iswapdg a, b { h b; cx b, a; cx a, b; h a; sdg b; sdg a; }\n";
    if (d.cu)
        os << "gate cu(theta,phi,lam,gam) c, t { u1(gam) c; "
              "cu3(theta,phi,lam) c, t; }\n";
    os << "\n";
}

/// Size of the emitted classical register.
///
/// Normally the shared contract width (`cbit_register_width`, core/command.h).
/// A `Measure` with empty `cbits` has no classical target to name, so it is
/// emitted against bit `qubit`; that index is deliberately not part of the
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

/// Emit one Command as a QASM 2 statement, prefixed by `guard` (an
/// `if (…) ` string, or empty when unconditional).
void emit_command(std::ostringstream& os,
                  const Command& cmd,
                  const std::string& guard,
                  const CregLayout& creg) {
    auto p = [&]() { return param_list(cmd.params); };
    auto q = [&]() { return qref_list(cmd.qubits); };
    os << guard;

    switch (cmd.gate) {
        // ── 1Q no-param ──
        case GateType::X:    os << "x "    << q() << ";\n"; break;
        case GateType::Y:    os << "y "    << q() << ";\n"; break;
        case GateType::Z:    os << "z "    << q() << ";\n"; break;
        case GateType::H:    os << "h "    << q() << ";\n"; break;
        case GateType::S:    os << "s "    << q() << ";\n"; break;
        case GateType::Sdg:  os << "sdg "  << q() << ";\n"; break;
        case GateType::T:    os << "t "    << q() << ";\n"; break;
        case GateType::Tdg:  os << "tdg "  << q() << ";\n"; break;
        case GateType::Id:   os << "id "   << q() << ";\n"; break;
        case GateType::SX:   os << "sx "   << q() << ";\n"; break;
        case GateType::SXdg: os << "sxdg " << q() << ";\n"; break;

        // ── 1Q parametric.  P is u1: same matrix, and `p` is not a QASM 2 name.
        case GateType::Rx:   os << "rx" << p() << " " << q() << ";\n"; break;
        case GateType::Ry:   os << "ry" << p() << " " << q() << ";\n"; break;
        case GateType::Rz:   os << "rz" << p() << " " << q() << ";\n"; break;
        case GateType::P:    os << "u1" << p() << " " << q() << ";\n"; break;

        // ── 1Q general U(θ, φ, λ) — `u3` in qelib1 ──
        case GateType::U:    os << "u3" << p() << " " << q() << ";\n"; break;

        // ── 2Q no-param ──
        case GateType::CX:      os << "cx "      << q() << ";\n"; break;
        case GateType::CY:      os << "cy "      << q() << ";\n"; break;
        case GateType::CZ:      os << "cz "      << q() << ";\n"; break;
        case GateType::SWAP:    os << "swap "    << q() << ";\n"; break;
        case GateType::ECR:     os << "ecr "     << q() << ";\n"; break;
        case GateType::iSWAP:   os << "iswap "   << q() << ";\n"; break;
        case GateType::iSWAPdg: os << "iswapdg " << q() << ";\n"; break;
        case GateType::CH:      os << "ch "      << q() << ";\n"; break;
        case GateType::CS:      os << "cs "      << q() << ";\n"; break;
        case GateType::CSdg:    os << "csdg "    << q() << ";\n"; break;
        case GateType::CSX:     os << "csx "     << q() << ";\n"; break;
        case GateType::CSXdg:   os << "csxdg "   << q() << ";\n"; break;

        // ── 2Q parametric.  CP is cu1: same matrix, and `cp` is not a QASM 2 name.
        case GateType::CRx: os << "crx" << p() << " " << q() << ";\n"; break;
        case GateType::CRy: os << "cry" << p() << " " << q() << ";\n"; break;
        case GateType::CRz: os << "crz" << p() << " " << q() << ";\n"; break;
        case GateType::CP:  os << "cu1" << p() << " " << q() << ";\n"; break;
        case GateType::RZZ: os << "rzz" << p() << " " << q() << ";\n"; break;
        case GateType::RXX: os << "rxx" << p() << " " << q() << ";\n"; break;
        case GateType::RYY: os << "ryy" << p() << " " << q() << ";\n"; break;

        // ── 2Q controlled-U with global phase γ ──
        case GateType::CU:  os << "cu" << p() << " " << q() << ";\n"; break;

        // ── 3Q no-param ──
        case GateType::CCX:   os << "ccx "   << q() << ";\n"; break;
        case GateType::CSWAP: os << "cswap " << q() << ";\n"; break;

        // ── Special / non-unitary ──
        case GateType::Barrier:
            os << "barrier " << q() << ";\n";
            break;
        case GateType::Measure: {
            const uint32_t cbit = cmd.cbits.empty() ? cmd.qubits[0] : cmd.cbits[0];
            os << "measure " << qref(cmd.qubits[0]) << " -> " << creg.ref(cbit) << ";\n";
            break;
        }
        case GateType::Reset:
            os << "reset " << qref(cmd.qubits[0]) << ";\n";
            break;

        // Branch markers are handled by the body-emission loop directly; they
        // shouldn't reach this dispatch.  Defend anyway with a clear error so
        // a malformed flatten doesn't silently corrupt the output.
        case GateType::BranchBegin:
        case GateType::BranchElse:
        case GateType::BranchEnd:
            throw std::runtime_error(
                "QASM2Emitter::emit_command: stray "
                + std::string(gate_name(cmd.gate))
                + " — branch markers must be dispatched by the outer loop.");

        // GPhase, MCZ and Custom are erased from gate_set(), so validate()
        // rejects them before emission.  Defend anyway.
        default:
            throw std::runtime_error(
                "QASM2Emitter: unsupported gate '" +
                std::string(gate_name(cmd.gate)) + "'");
    }
}

/// The two branch shapes OpenQASM 2's grammar cannot express.  Properties of
/// the emitted *shape* rather than of any single command, which is why no
/// `EmitterCapabilities` flag covers them: `if` takes a single `qop`, so
/// neither a nested `if` nor a guarded `barrier` (not a `qop`) is spellable.
std::optional<Incompatibility> unrepresentable_branch(
    const std::vector<Command>& commands) {
    int depth = 0;
    for (const auto& cmd : commands) {
        switch (cmd.gate) {
            case GateType::BranchBegin:
                if (cmd.condition_bits.empty())
                    return Incompatibility{cmd,
                        "qasm2 emitter: BranchBegin with an empty condition is not "
                        "representable — OpenQASM 2 `if` requires a register comparison"};
                if (++depth > 1)
                    return Incompatibility{cmd,
                        "qasm2 emitter: nested classical conditionals are not "
                        "representable — OpenQASM 2 `if` governs a single quantum "
                        "operation, which cannot itself be an `if`"};
                break;
            case GateType::BranchEnd:
                --depth;
                break;
            case GateType::Barrier:
                if (depth > 0)
                    return Incompatibility{cmd,
                        "qasm2 emitter: a barrier inside a classical conditional is not "
                        "representable — OpenQASM 2 `if` governs a quantum operation, "
                        "and `barrier` is not one"};
                break;
            default:
                break;
        }
    }
    return std::nullopt;
}

/// True if any command opens a classical conditional — decides the creg layout.
bool has_conditionals(const std::vector<Command>& commands) {
    return std::any_of(commands.begin(), commands.end(), [](const Command& c) {
        return c.gate == GateType::BranchBegin;
    });
}

}  // namespace

std::optional<Incompatibility> QASM2Emitter::validate(
    const std::vector<Command>& commands) const
{
    if (auto inc = Emitter::validate(commands)) return inc;
    return unrepresentable_branch(commands);
}

std::string QASM2Emitter::emit(
    const std::vector<Command>& commands,
    uint32_t n_qubits,
    const std::string& circuit_name) const
{
    if (auto inc = validate(commands)) {
        throw capability_error(inc->reason, inc->command);
    }

    std::ostringstream os;

    // Header.  `circuit_name` is informational only — OpenQASM 2 has no
    // top-level program name keyword.
    os << header_comment(circuit_name);
    os << "OPENQASM 2.0;\n";
    os << "include \"qelib1.inc\";\n\n";

    emit_gate_definitions(os, collect_gate_defs(commands));

    // Quantum register.
    os << "qreg q[" << n_qubits << "];\n";

    // Classical register(s), omitted entirely if the program has no cbits.
    CregLayout creg;
    creg.width   = emitted_register_size(commands);
    creg.per_bit = has_conditionals(commands);
    if (creg.width > 0) {
        if (creg.per_bit) {
            for (uint32_t i = 0; i < creg.width; ++i)
                os << "creg " << creg.reg_name(i) << "[1];\n";
        } else {
            os << "creg c[" << creg.width << "];\n";
        }
    }
    os << "\n";

    // Body.  OpenQASM 2 has no `else` and no braced block: each command in a
    // branch body carries its own `if (…) ` guard, and BranchElse negates the
    // compared value.  Nesting is rejected above, so the stack holds at most
    // one frame.
    std::vector<std::pair<uint32_t, bool>> frames;
    auto guard = [&]() -> std::string {
        if (frames.empty()) return "";
        const auto& [bit, value] = frames.back();
        return "if (" + creg.reg_name(bit) + " == " + (value ? "1" : "0") + ") ";
    };

    for (const auto& cmd : commands) {
        switch (cmd.gate) {
            case GateType::BranchBegin:
                frames.emplace_back(cmd.condition_bits[0],
                                    static_cast<bool>(cmd.condition_values[0]));
                break;
            case GateType::BranchElse:
                if (frames.empty())
                    throw std::runtime_error(
                        "QASM2Emitter: BranchElse outside a branch frame.");
                frames.back().second = !frames.back().second;
                break;
            case GateType::BranchEnd:
                if (frames.empty())
                    throw std::runtime_error(
                        "QASM2Emitter: BranchEnd outside a branch frame.");
                frames.pop_back();
                break;
            default:
                emit_command(os, cmd, guard(), creg);
                break;
        }
    }

    return os.str();
}

void QASM2Emitter::emit_to_file(
    const std::vector<Command>& commands,
    uint32_t n_qubits,
    const std::string& path,
    const std::string& circuit_name) const
{
    std::string content = emit(commands, n_qubits, circuit_name);
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("QASM2Emitter: cannot open file " + path);
    }
    file << content;
}

}  // namespace qarpx
