#include "qarpx/emit/qir_emitter.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace qarpx {

namespace {

/// Map a qubit index to QIR's %Qubit* representation.
/// Qubit 0 = null, qubit N = inttoptr(i64 N to %Qubit*)
std::string qubit_ref(uint32_t q) {
    if (q == 0) return "%Qubit* null";
    return "%Qubit* inttoptr (i64 " + std::to_string(q) + " to %Qubit*)";
}

/// Map a cbit/result index to QIR's %Result* representation.
std::string result_ref(uint32_t r) {
    if (r == 0) return "%Result* null";
    return "%Result* inttoptr (i64 " + std::to_string(r) + " to %Result*)";
}

/// Emit a single command as QIR intrinsic calls.
void emit_command(std::ostringstream& os, const Command& cmd) {
    switch (cmd.gate) {
        case GateType::H:
            os << "  call void @__quantum__qis__h__body("
               << qubit_ref(cmd.qubits[0]) << ")\n";
            break;
        case GateType::X:
            os << "  call void @__quantum__qis__x__body("
               << qubit_ref(cmd.qubits[0]) << ")\n";
            break;
        case GateType::Y:
            os << "  call void @__quantum__qis__y__body("
               << qubit_ref(cmd.qubits[0]) << ")\n";
            break;
        case GateType::Z:
            os << "  call void @__quantum__qis__z__body("
               << qubit_ref(cmd.qubits[0]) << ")\n";
            break;
        case GateType::S:
            os << "  call void @__quantum__qis__s__body("
               << qubit_ref(cmd.qubits[0]) << ")\n";
            break;
        case GateType::Sdg:
            os << "  call void @__quantum__qis__s__adj("
               << qubit_ref(cmd.qubits[0]) << ")\n";
            break;
        case GateType::T:
            os << "  call void @__quantum__qis__t__body("
               << qubit_ref(cmd.qubits[0]) << ")\n";
            break;
        case GateType::Tdg:
            os << "  call void @__quantum__qis__t__adj("
               << qubit_ref(cmd.qubits[0]) << ")\n";
            break;
        case GateType::Rx:
            if (cmd.params[0].is_symbolic()) {
                throw std::runtime_error(
                    "QIREmitter: cannot emit symbolic param " +
                    cmd.params[0].to_string() + " — substitute first");
            }
            os << "  call void @__quantum__qis__rx__body(double "
               << cmd.params[0].value() << ", "
               << qubit_ref(cmd.qubits[0]) << ")\n";
            break;
        case GateType::Ry:
            if (cmd.params[0].is_symbolic())
                throw std::runtime_error("QIREmitter: symbolic param in Ry");
            os << "  call void @__quantum__qis__ry__body(double "
               << cmd.params[0].value() << ", "
               << qubit_ref(cmd.qubits[0]) << ")\n";
            break;
        case GateType::Rz:
            if (cmd.params[0].is_symbolic())
                throw std::runtime_error("QIREmitter: symbolic param in Rz");
            os << "  call void @__quantum__qis__rz__body(double "
               << cmd.params[0].value() << ", "
               << qubit_ref(cmd.qubits[0]) << ")\n";
            break;
        case GateType::CX:
            os << "  call void @__quantum__qis__cnot__body("
               << qubit_ref(cmd.qubits[0]) << ", "
               << qubit_ref(cmd.qubits[1]) << ")\n";
            break;
        case GateType::CZ:
            os << "  call void @__quantum__qis__cz__body("
               << qubit_ref(cmd.qubits[0]) << ", "
               << qubit_ref(cmd.qubits[1]) << ")\n";
            break;
        case GateType::SWAP:
            os << "  call void @__quantum__qis__swap__body("
               << qubit_ref(cmd.qubits[0]) << ", "
               << qubit_ref(cmd.qubits[1]) << ")\n";
            break;
        case GateType::Measure:
            os << "  call void @__quantum__qis__mz__body("
               << qubit_ref(cmd.qubits[0]) << ", "
               << result_ref(cmd.cbits.empty() ? cmd.qubits[0] : cmd.cbits[0]) << ")\n";
            break;
        case GateType::Reset:
            os << "  call void @__quantum__qis__reset__body("
               << qubit_ref(cmd.qubits[0]) << ")\n";
            break;
        case GateType::Barrier:
            // QIR doesn't have barriers; emit as a comment
            os << "  ; barrier\n";
            break;
        case GateType::BranchBegin:
        case GateType::BranchElse:
        case GateType::BranchEnd:
            throw std::runtime_error(
                "QIREmitter::emit_command: stray "
                + std::string(gate_name(cmd.gate))
                + " — branch markers must be dispatched by the outer loop.");
        default:
            throw std::runtime_error(
                "QIREmitter: unsupported gate " + std::string(gate_name(cmd.gate)) +
                " — decompose first");
    }
}

/// Look ahead from a BranchBegin position to determine whether its matching
/// BranchEnd is preceded by a BranchElse at the same nesting depth.  Returns
/// (has_else, end_index).  Throws on malformed marker sequences.
struct BranchSpan { bool has_else; std::size_t end_idx; };
BranchSpan find_branch_span(const std::vector<Command>& commands,
                            std::size_t begin_idx)
{
    int depth = 1;
    bool has_else = false;
    for (std::size_t k = begin_idx + 1; k < commands.size(); ++k) {
        switch (commands[k].gate) {
            case GateType::BranchBegin:
                ++depth;
                break;
            case GateType::BranchElse:
                if (depth == 1) has_else = true;
                break;
            case GateType::BranchEnd:
                if (--depth == 0) return {has_else, k};
                break;
            default: break;
        }
    }
    throw std::runtime_error(
        "QIREmitter: unterminated BranchBegin (missing BranchEnd)");
}

}  // anonymous namespace

std::string QIREmitter::emit(
    const std::vector<Command>& commands,
    uint32_t n_qubits,
    const std::string& function_name) const
{
    throw_if_invalid(commands);

    // Count results (measurement outputs)
    uint32_t n_results = 0;
    for (const auto& cmd : commands) {
        if (cmd.gate == GateType::Measure) ++n_results;
    }

    std::ostringstream os;

    // Module header
    os << "; ModuleID = 'qarp_circuit'\n";
    os << "%Qubit = type opaque\n";
    os << "%Result = type opaque\n\n";

    // Function definition
    os << "define void @" << function_name << "() #0 {\n";
    os << "entry:\n";

    // Body — handle branch markers as LLVM control flow.  Each conditional
    // region opens an `if/else/end` triple of basic blocks.
    int next_label = 0;
    int next_ssa   = 0;
    struct ActiveBranch {
        std::string then_label;
        std::string else_label;
        std::string end_label;
        bool has_else;
    };
    std::vector<ActiveBranch> branch_stack;

    for (std::size_t i = 0; i < commands.size(); ++i) {
        const auto& cmd = commands[i];

        if (cmd.gate == GateType::BranchBegin) {
            const auto span = find_branch_span(commands, i);
            const std::string id = std::to_string(next_label++);
            ActiveBranch b{
                "then" + id, "else" + id, "end" + id, span.has_else
            };
            branch_stack.push_back(b);

            // Compute the AND-of-condition into a single i1.  Each cbit read
            // gives one i1; if the expected value is 0 we XOR with 1 to flip.
            std::vector<std::string> regs;
            for (size_t k = 0; k < cmd.condition_bits.size(); ++k) {
                const std::string r = "%c" + std::to_string(next_ssa++);
                os << "  " << r << " = call i1 @__quantum__rt__read_result__body("
                   << result_ref(cmd.condition_bits[k]) << ")\n";
                std::string final_reg = r;
                if (!cmd.condition_values[k]) {
                    final_reg = "%n" + std::to_string(next_ssa++);
                    os << "  " << final_reg << " = xor i1 " << r << ", true\n";
                }
                regs.push_back(final_reg);
            }
            std::string cond_reg = regs.front();
            for (size_t k = 1; k < regs.size(); ++k) {
                const std::string a = "%a" + std::to_string(next_ssa++);
                os << "  " << a << " = and i1 " << cond_reg << ", " << regs[k] << "\n";
                cond_reg = a;
            }
            // The false branch goes to else if present, end otherwise.
            const std::string& false_label = b.has_else ? b.else_label : b.end_label;
            os << "  br i1 " << cond_reg
               << ", label %" << b.then_label
               << ", label %" << false_label << "\n";
            os << b.then_label << ":\n";
            continue;
        }

        if (cmd.gate == GateType::BranchElse) {
            const auto& b = branch_stack.back();
            os << "  br label %" << b.end_label << "\n";
            os << b.else_label << ":\n";
            continue;
        }

        if (cmd.gate == GateType::BranchEnd) {
            const auto b = branch_stack.back();
            branch_stack.pop_back();
            os << "  br label %" << b.end_label << "\n";
            os << b.end_label << ":\n";
            continue;
        }

        emit_command(os, cmd);
    }

    os << "  ret void\n";
    os << "}\n\n";

    // QIR intrinsic declarations
    os << "; QIR intrinsic declarations\n";
    os << "declare void @__quantum__qis__h__body(%Qubit*)\n";
    os << "declare void @__quantum__qis__x__body(%Qubit*)\n";
    os << "declare void @__quantum__qis__y__body(%Qubit*)\n";
    os << "declare void @__quantum__qis__z__body(%Qubit*)\n";
    os << "declare void @__quantum__qis__s__body(%Qubit*)\n";
    os << "declare void @__quantum__qis__s__adj(%Qubit*)\n";
    os << "declare void @__quantum__qis__t__body(%Qubit*)\n";
    os << "declare void @__quantum__qis__t__adj(%Qubit*)\n";
    os << "declare void @__quantum__qis__rx__body(double, %Qubit*)\n";
    os << "declare void @__quantum__qis__ry__body(double, %Qubit*)\n";
    os << "declare void @__quantum__qis__rz__body(double, %Qubit*)\n";
    os << "declare void @__quantum__qis__cnot__body(%Qubit*, %Qubit*)\n";
    os << "declare void @__quantum__qis__cz__body(%Qubit*, %Qubit*)\n";
    os << "declare void @__quantum__qis__swap__body(%Qubit*, %Qubit*)\n";
    os << "declare void @__quantum__qis__mz__body(%Qubit*, %Result*)\n";
    os << "declare void @__quantum__qis__reset__body(%Qubit*)\n";
    os << "declare i1 @__quantum__rt__read_result__body(%Result*)\n\n";

    // Function attributes
    os << "attributes #0 = { \"entry_point\" "
       << "\"required_num_qubits\"=\"" << n_qubits << "\" "
       << "\"required_num_results\"=\"" << n_results << "\" }\n";

    return os.str();
}

void QIREmitter::emit_to_file(
    const std::vector<Command>& commands,
    uint32_t n_qubits,
    const std::string& path,
    const std::string& function_name) const
{
    std::string content = emit(commands, n_qubits, function_name);
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("QIREmitter: cannot open file " + path);
    }
    file << content;
}

}  // namespace qarpx
