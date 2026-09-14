#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include "qarpx/core/command.h"
#include "qarpx/core/errors.h"
#include "qarpx/core/gates.h"
#include "qarpx/core/param.h"
#include "qarpx/block/block.h"
#include "qarpx/emit/emitter.h"
#include "qarpx/emit/param_bridge.h"
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace nb = nanobind;
using namespace nb::literals;
using namespace qarpx;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Scan commands for Measure targets and condition_bits to find total cbit count.
static uint32_t compute_n_cbits(const std::vector<Command>& commands) {
    uint32_t max_cbit = 0;
    bool found = false;
    for (const auto& cmd : commands) {
        for (uint32_t c : cmd.cbits) {
            if (!found || c + 1 > max_cbit) { max_cbit = c + 1; found = true; }
        }
        for (uint32_t c : cmd.condition_bits) {
            if (!found || c + 1 > max_cbit) { max_cbit = c + 1; found = true; }
        }
    }
    return found ? max_cbit : 0;
}

/// Convert a QARPx Param to a TKET angle (half-turns = θ_rad / π).
/// sign can be used to negate (currently unused for tket, kept for symmetry).
static nb::object to_tket_angle(
        const Param& p,
        const std::unordered_map<std::string, nb::object>& sym_map,
        double sign = 1.0) {
    constexpr double PI = 3.14159265358979323846;
    if (p.is_concrete()) {
        return nb::cast(sign * p.value() / PI);
    }
    // Symbolic: extract linear form coeff * sym + offset and build sympy expr.
    if (p.free_symbols().empty()) {
        // Symbolic struct but no free symbols — evaluate concretely.
        return nb::cast(sign * p.evaluate({}) / PI);
    }
    const auto lf = param_bridge::linear_form(p, "pytket");
    auto it = sym_map.find(lf.symbol);
    if (it == sym_map.end()) {
        throw std::runtime_error(
            "PytketEmitter: symbol not found in map: " + lf.symbol);
    }
    nb::object sym = it->second;
    auto sympy = nb::module_::import_("sympy");
    // angle = sign * (coeff * sym + offset) / sympy.pi
    nb::object expr = nb::cast(sign * lf.coeff) * sym + nb::cast(sign * lf.offset);
    return expr / sympy.attr("pi");
}

// ─────────────────────────────────────────────────────────────────────────────

class PytketEmitter : public Emitter {
public:
    [[nodiscard]] std::string target_name() const override { return "pytket"; }

    [[nodiscard]] GateSet gate_set() const override { return pytket_gateset(); }

    [[nodiscard]] EmitterCapabilities capabilities() const override {
        return EmitterCapabilities{
            .symbolic_params     = true,   // sympy symbols, half-turn scaled
            .multi_symbol_params = false,  // linear single-symbol forms only
            .distinct_cbits      = true,
            .conditionals        = true,
            .multibit_conditions = true,   // add_gate condition mask
            .multibit_else       = false,  // Else flips a single bit only
            .custom_unitary      = false,
            .cu_global_phase     = false,  // CU3 has no γ
        };
    }

    nb::object emit(const std::vector<Command>& commands,
                    uint32_t n_qubits,
                    const std::string& circuit_name = "circuit") const
    {
        throw_if_invalid(commands);

        // ── Import pytket ─────────────────────────────────────────────────
        nb::object pytket;
        try {
            pytket = nb::module_::import_("pytket");
        } catch (...) {
            throw sdk_missing_error(
                "pytket not installed. Install with: pip install pytket");
        }
        auto pytket_circuit_mod = nb::module_::import_("pytket.circuit");
        auto OpType = pytket_circuit_mod.attr("OpType");

        // ── Build circuit ─────────────────────────────────────────────────
        uint32_t n_cbits = compute_n_cbits(commands);
        auto Circuit = pytket.attr("Circuit");
        auto circ = Circuit(n_qubits, n_cbits, "name"_a = circuit_name);

        // ── Build symbol map ──────────────────────────────────────────────
        std::unordered_map<std::string, nb::object> sym_map;
        for (const auto& cmd : commands) {
            for (const auto& p : cmd.params) {
                if (p.is_symbolic()) {
                    for (const auto& sname : p.free_symbols()) {
                        if (sym_map.find(sname) == sym_map.end()) {
                            auto sympy = nb::module_::import_("sympy");
                            sym_map[sname] = sympy.attr("Symbol")(sname);
                        }
                    }
                }
            }
        }

        // ── Branch stack: each frame holds {condition_bits, condition_values} ─
        struct BranchFrame {
            std::vector<uint32_t> cond_bits;
            std::vector<bool>     cond_vals;
        };
        std::vector<BranchFrame> branch_stack;

        // ── add_gate helper ───────────────────────────────────────────────
        // Dispatches to circ.add_gate with optional classical condition.
        auto add_gate = [&](nb::object op_type,
                            nb::list   params_list,
                            nb::list   qubits_list,
                            const std::vector<uint32_t>* cond_bits = nullptr,
                            const std::vector<bool>*     cond_vals = nullptr)
        {
            bool has_cond = cond_bits && !cond_bits->empty();
            if (has_cond) {
                // Compute integer mask from condition_values.
                int mask = 0;
                for (size_t i = 0; i < cond_vals->size(); ++i)
                    if ((*cond_vals)[i]) mask |= (1 << static_cast<int>(i));
                nb::list cb;
                for (uint32_t b : *cond_bits) cb.append(nb::cast(static_cast<int>(b)));
                if (params_list.size() == 0) {
                    circ.attr("add_gate")(op_type, qubits_list,
                        "condition_bits"_a  = cb,
                        "condition_value"_a = nb::cast(mask));
                } else {
                    circ.attr("add_gate")(op_type, params_list, qubits_list,
                        "condition_bits"_a  = cb,
                        "condition_value"_a = nb::cast(mask));
                }
            } else {
                if (params_list.size() == 0) {
                    circ.attr("add_gate")(op_type, qubits_list);
                } else {
                    circ.attr("add_gate")(op_type, params_list, qubits_list);
                }
            }
        };

        // ── Gate dispatch ─────────────────────────────────────────────────
        for (const auto& cmd : commands) {
            // Grab current condition from top of branch stack (if any).
            const std::vector<uint32_t>* cb = nullptr;
            const std::vector<bool>*     cv = nullptr;
            if (!branch_stack.empty()) {
                cb = &branch_stack.back().cond_bits;
                cv = &branch_stack.back().cond_vals;
            }

            // Helper to build a single-element tket-angle param list.
            auto p_list1 = [&](int idx, double sign = 1.0) {
                nb::list lst;
                lst.append(to_tket_angle(cmd.params[static_cast<size_t>(idx)], sym_map, sign));
                return lst;
            };
            // Helper to build a qubit list from a slice of cmd.qubits.
            auto q_list = [&](std::initializer_list<size_t> indices) {
                nb::list lst;
                for (size_t i : indices)
                    lst.append(nb::cast(static_cast<int>(cmd.qubits[i])));
                return lst;
            };
            auto q_all = [&]() {
                nb::list lst;
                for (uint32_t q : cmd.qubits)
                    lst.append(nb::cast(static_cast<int>(q)));
                return lst;
            };
            nb::list no_params;

            switch (cmd.gate) {

                // ── 1Q no-param ──────────────────────────────────────────
                case GateType::X:
                    add_gate(OpType.attr("X"), no_params, q_list({0}), cb, cv);
                    break;
                case GateType::Y:
                    add_gate(OpType.attr("Y"), no_params, q_list({0}), cb, cv);
                    break;
                case GateType::Z:
                    add_gate(OpType.attr("Z"), no_params, q_list({0}), cb, cv);
                    break;
                case GateType::H:
                    add_gate(OpType.attr("H"), no_params, q_list({0}), cb, cv);
                    break;
                case GateType::S:
                    add_gate(OpType.attr("S"), no_params, q_list({0}), cb, cv);
                    break;
                case GateType::Sdg:
                    add_gate(OpType.attr("Sdg"), no_params, q_list({0}), cb, cv);
                    break;
                case GateType::T:
                    add_gate(OpType.attr("T"), no_params, q_list({0}), cb, cv);
                    break;
                case GateType::Tdg:
                    add_gate(OpType.attr("Tdg"), no_params, q_list({0}), cb, cv);
                    break;
                // OpType.SX carries the e^{iπ/4}; OpType.V is Rx(π/2) and would drop it.
                case GateType::SX:
                    add_gate(OpType.attr("SX"), no_params, q_list({0}), cb, cv);
                    break;
                case GateType::SXdg:
                    add_gate(OpType.attr("SXdg"), no_params, q_list({0}), cb, cv);
                    break;
                case GateType::Id:
                    add_gate(OpType.attr("noop"), no_params, q_list({0}), cb, cv);
                    break;

                // ── 1Q parametric ────────────────────────────────────────
                case GateType::Rx:
                    add_gate(OpType.attr("Rx"), p_list1(0), q_list({0}), cb, cv);
                    break;
                case GateType::Ry:
                    add_gate(OpType.attr("Ry"), p_list1(0), q_list({0}), cb, cv);
                    break;
                case GateType::Rz:
                    add_gate(OpType.attr("Rz"), p_list1(0), q_list({0}), cb, cv);
                    break;
                case GateType::P: {
                    // Phase gate → TKET U1
                    add_gate(OpType.attr("U1"), p_list1(0), q_list({0}), cb, cv);
                    break;
                }
                case GateType::U: {
                    nb::list ps;
                    ps.append(to_tket_angle(cmd.params[0], sym_map));
                    ps.append(to_tket_angle(cmd.params[1], sym_map));
                    ps.append(to_tket_angle(cmd.params[2], sym_map));
                    add_gate(OpType.attr("U3"), ps, q_list({0}), cb, cv);
                    break;
                }

                // ── 2Q no-param ──────────────────────────────────────────
                case GateType::CX:
                    add_gate(OpType.attr("CX"), no_params, q_list({0, 1}), cb, cv);
                    break;
                case GateType::CY:
                    add_gate(OpType.attr("CY"), no_params, q_list({0, 1}), cb, cv);
                    break;
                case GateType::CZ:
                    add_gate(OpType.attr("CZ"), no_params, q_list({0, 1}), cb, cv);
                    break;
                case GateType::SWAP:
                    add_gate(OpType.attr("SWAP"), no_params, q_list({0, 1}), cb, cv);
                    break;
                case GateType::ECR:
                    add_gate(OpType.attr("ECR"), no_params, q_list({0, 1}), cb, cv);
                    break;
                case GateType::iSWAP: {
                    nb::list ps; ps.append(nb::cast(1.0));
                    add_gate(OpType.attr("ISWAP"), ps, q_list({0, 1}), cb, cv);
                    break;
                }
                case GateType::iSWAPdg: {
                    nb::list ps; ps.append(nb::cast(3.0));
                    add_gate(OpType.attr("ISWAP"), ps, q_list({0, 1}), cb, cv);
                    break;
                }
                case GateType::CH:
                    add_gate(OpType.attr("CH"), no_params, q_list({0, 1}), cb, cv);
                    break;
                case GateType::CS:
                    add_gate(OpType.attr("CS"), no_params, q_list({0, 1}), cb, cv);
                    break;
                case GateType::CSdg:
                    add_gate(OpType.attr("CSdg"), no_params, q_list({0, 1}), cb, cv);
                    break;
                case GateType::CSX:
                    add_gate(OpType.attr("CSX"), no_params, q_list({0, 1}), cb, cv);
                    break;
                case GateType::CSXdg:
                    add_gate(OpType.attr("CSXdg"), no_params, q_list({0, 1}), cb, cv);
                    break;

                // ── 2Q parametric ────────────────────────────────────────
                case GateType::CRx:
                    add_gate(OpType.attr("CRx"), p_list1(0), q_list({0, 1}), cb, cv);
                    break;
                case GateType::CRy:
                    add_gate(OpType.attr("CRy"), p_list1(0), q_list({0, 1}), cb, cv);
                    break;
                case GateType::CRz:
                    add_gate(OpType.attr("CRz"), p_list1(0), q_list({0, 1}), cb, cv);
                    break;
                case GateType::CP:
                    add_gate(OpType.attr("CU1"), p_list1(0), q_list({0, 1}), cb, cv);
                    break;
                case GateType::RZZ:
                    // ZZPhase(t) = exp(-i·t·π/2·ZZ), RZZ(θ) = exp(-iθ/2·ZZ)
                    // so t = +θ/π — no sign flip needed.
                    add_gate(OpType.attr("ZZPhase"), p_list1(0), q_list({0, 1}), cb, cv);
                    break;
                case GateType::RXX:
                    add_gate(OpType.attr("XXPhase"), p_list1(0), q_list({0, 1}), cb, cv);
                    break;
                case GateType::RYY:
                    add_gate(OpType.attr("YYPhase"), p_list1(0), q_list({0, 1}), cb, cv);
                    break;
                case GateType::CU: {
                    // CU(θ, φ, λ, γ): γ != 0 is rejected by validate()
                    // (cu_global_phase = false) — only γ = 0 reaches here.
                    nb::list ps;
                    ps.append(to_tket_angle(cmd.params[0], sym_map));
                    ps.append(to_tket_angle(cmd.params[1], sym_map));
                    ps.append(to_tket_angle(cmd.params[2], sym_map));
                    add_gate(OpType.attr("CU3"), ps, q_list({0, 1}), cb, cv);
                    break;
                }

                // ── 3Q no-param ──────────────────────────────────────────
                case GateType::CCX:
                    add_gate(OpType.attr("CCX"), no_params, q_list({0, 1, 2}), cb, cv);
                    break;
                case GateType::CSWAP:
                    add_gate(OpType.attr("CSWAP"), no_params, q_list({0, 1, 2}), cb, cv);
                    break;

                // ── Multi-qubit ───────────────────────────────────────────
                case GateType::MCZ:
                    add_gate(OpType.attr("CnZ"), no_params, q_all(), cb, cv);
                    break;

                // ── Global phase ──────────────────────────────────────────
                case GateType::GPhase: {
                    constexpr double PI = 3.14159265358979323846;
                    // Inside a branch the phase is conditional, so it cannot
                    // be folded into the circuit-level `.phase`: emit a
                    // conditioned Phase op (the absorber reads it back).
                    if (cb && !cb->empty()) {
                        nb::list no_qubits;
                        add_gate(OpType.attr("Phase"), p_list1(0), no_qubits, cb, cv);
                        break;
                    }
                    if (cmd.params[0].is_concrete()) {
                        circ.attr("add_phase")(nb::cast(cmd.params[0].value() / PI));
                    } else {
                        // Symbolic: divide by sympy.pi
                        auto sympy = nb::module_::import_("sympy");
                        auto expr  = to_tket_angle(cmd.params[0], sym_map);
                        circ.attr("add_phase")(expr);
                    }
                    break;
                }

                // ── Barrier ───────────────────────────────────────────────
                case GateType::Barrier: {
                    nb::list ql = q_all();
                    circ.attr("add_barrier")(ql);
                    break;
                }

                // ── Measure ───────────────────────────────────────────────
                // Route through add_gate so a branch condition (cb/cv) is
                // forwarded — circ.Measure()/circ.Reset() cannot carry one.
                case GateType::Measure: {
                    uint32_t qubit = cmd.qubits[0];
                    uint32_t cbit  = cmd.cbits.empty() ? qubit : cmd.cbits[0];
                    nb::list mq;
                    mq.append(nb::cast(static_cast<int>(qubit)));
                    mq.append(nb::cast(static_cast<int>(cbit)));
                    add_gate(OpType.attr("Measure"), no_params, mq, cb, cv);
                    break;
                }

                // ── Reset ─────────────────────────────────────────────────
                case GateType::Reset:
                    add_gate(OpType.attr("Reset"), no_params, q_list({0}), cb, cv);
                    break;

                // ── Branch control flow ───────────────────────────────────
                case GateType::BranchBegin: {
                    BranchFrame frame;
                    for (uint32_t b : cmd.condition_bits)  frame.cond_bits.push_back(b);
                    for (bool     v : cmd.condition_values) frame.cond_vals.push_back(v);
                    branch_stack.push_back(std::move(frame));
                    break;
                }
                case GateType::BranchElse: {
                    if (branch_stack.empty()) {
                        throw std::runtime_error(
                            "PytketEmitter: BranchElse without matching BranchBegin");
                    }
                    auto& frame = branch_stack.back();
                    if (frame.cond_bits.size() == 1) {
                        // Flip the single condition value.
                        frame.cond_vals[0] = !frame.cond_vals[0];
                    } else {
                        throw std::runtime_error(
                            "PytketEmitter: BranchElse on multi-bit condition not supported");
                    }
                    break;
                }
                case GateType::BranchEnd:
                    if (!branch_stack.empty()) branch_stack.pop_back();
                    break;

                // Custom is rejected by validate() and lands in the default
                // arm's defensive throw if it ever slips through.
                default:
                    throw std::runtime_error(
                        std::string("PytketEmitter: unsupported gate: ") +
                        std::string(gate_name(cmd.gate)));
            }
        }

        return circ;
    }
};

void register_pytket_emitter(nb::module_& m) {
    nb::class_<PytketEmitter>(m, "PytketEmitter")
        .def(nb::init<>())
        .def("target_name", &PytketEmitter::target_name)
        .def("gate_set", &PytketEmitter::gate_set)
        .def("capabilities", &PytketEmitter::capabilities)
        .def("validate",
             [](const PytketEmitter& self, const std::vector<Command>& cmds) {
                 return self.validate(cmds);
             },
             "commands"_a,
             "First Incompatibility in the sequence, or None if the whole "
             "sequence can cross.  Needs no SDK installed.")
        .def("emit", &PytketEmitter::emit,
             "commands"_a, "n_qubits"_a, "circuit_name"_a = "circuit",
             "Translate a flat QARPx command sequence into a pytket.Circuit.");
}
