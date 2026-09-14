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
#include <unordered_set>
#include <tuple>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using namespace qarpx;

// ── Symbolic param helper ────────────────────────────────────────────────────

static nb::object to_qiskit_param(
    const Param& p,
    const std::unordered_map<std::string, nb::object>& pm)
{
    if (p.is_concrete()) return nb::cast(p.value());
    const auto lf = param_bridge::linear_form(p, "qiskit");
    auto sym = pm.at(lf.symbol);
    nb::object result = nb::cast(lf.coeff) * sym;
    if (std::abs(lf.offset) > 1e-15) result = result + nb::cast(lf.offset);
    return result;
}

// ── Branch-stack types ────────────────────────────────────────────────────────

struct OpTriple {
    nb::object op;      // None signals a non-gate operation (measure/reset sentinel)
    nb::list   qubits;
    nb::list   clbits;
    // Sentinel encoding for measure / reset:
    //   sentinel_kind: 0=none, 1=measure, 2=reset
    int        sentinel_kind = 0;
    int        sentinel_q    = 0;
    int        sentinel_c    = 0;
};

struct BranchFrame {
    std::vector<uint32_t> condition_bits;
    std::vector<bool>     condition_values;
    std::vector<OpTriple> then_cmds;
    std::vector<OpTriple> else_cmds;
    bool in_else = false;
};

// ── Forward declarations ──────────────────────────────────────────────────────

static void flush_if_else(nb::object& qc,
                          const BranchFrame& frame,
                          uint32_t n_qubits,
                          const nb::object& cr,
                          nb::object& qiskit_circ_mod);

// ─────────────────────────────────────────────────────────────────────────────

class QiskitEmitter : public Emitter {
public:
    [[nodiscard]] std::string target_name() const override { return "qiskit"; }

    [[nodiscard]] GateSet gate_set() const override { return qiskit_gateset(); }

    [[nodiscard]] EmitterCapabilities capabilities() const override {
        return EmitterCapabilities{
            .symbolic_params     = true,   // qiskit.circuit.Parameter
            .multi_symbol_params = false,  // linear single-symbol forms only
            .distinct_cbits      = true,
            .conditionals        = true,   // single-bit IfElseOp conditions
            .multibit_conditions = false,
            .multibit_else       = false,
            .custom_unitary      = false,
            .cu_global_phase     = true,   // CUGate carries γ
        };
    }

    nb::object emit(const std::vector<Command>& commands,
                    uint32_t n_qubits,
                    const std::string& circuit_name = "circuit") const
    {
        throw_if_invalid(commands);

        // ── Import ────────────────────────────────────────────────────────
        nb::object qiskit_circ;
        nb::object qiskit_lib;
        try {
            qiskit_circ = nb::module_::import_("qiskit.circuit");
            qiskit_lib  = nb::module_::import_("qiskit.circuit.library");
        } catch (...) {
            throw sdk_missing_error(
                "Qiskit not installed. Install with: pip install qiskit>=1.0");
        }

        // ── Scan for n_cbits ──────────────────────────────────────────────
        uint32_t n_cbits = 0;
        for (const auto& cmd : commands) {
            for (auto c : cmd.cbits) {
                if (c + 1 > n_cbits) n_cbits = c + 1;
            }
            for (auto b : cmd.condition_bits) {
                if (b + 1 > n_cbits) n_cbits = b + 1;
            }
        }

        // ── Collect free symbols ──────────────────────────────────────────
        std::unordered_set<std::string> sym_names;
        for (const auto& cmd : commands) {
            for (const auto& p : cmd.params) {
                if (p.is_symbolic()) {
                    for (auto& s : p.free_symbols()) sym_names.insert(s);
                }
            }
        }

        // Build param_map: name → qiskit.circuit.Parameter(name)
        std::unordered_map<std::string, nb::object> param_map;
        nb::object ParameterCls = qiskit_circ.attr("Parameter");
        for (const auto& name : sym_names) {
            param_map[name] = ParameterCls(nb::cast(name));
        }

        // ── Build QuantumCircuit ──────────────────────────────────────────
        nb::object QuantumCircuit = qiskit_circ.attr("QuantumCircuit");
        nb::object qr = qiskit_circ.attr("QuantumRegister")(
            nb::cast(n_qubits), nb::cast(std::string("q")));
        nb::object cr = nb::none();
        nb::object qc;

        if (n_cbits > 0) {
            cr = qiskit_circ.attr("ClassicalRegister")(
                nb::cast(n_cbits), nb::cast(std::string("c")));
            qc = QuantumCircuit(qr, cr);
        } else {
            qc = QuantumCircuit(qr);
        }
        qc.attr("name") = nb::cast(circuit_name);

        // ── Branch stack ──────────────────────────────────────────────────
        std::vector<BranchFrame> branch_stack;

        // Helper: shorthand param converter
        auto qp = [&](const Param& p) -> nb::object {
            return to_qiskit_param(p, param_map);
        };

        // Helper: append triple to current frame or qc
        auto append_op = [&](nb::object op, nb::list qs, nb::list cs = nb::list()) {
            if (!branch_stack.empty()) {
                OpTriple t;
                t.op = std::move(op);
                t.qubits = std::move(qs);
                t.clbits = std::move(cs);
                if (branch_stack.back().in_else)
                    branch_stack.back().else_cmds.push_back(std::move(t));
                else
                    branch_stack.back().then_cmds.push_back(std::move(t));
            } else {
                qc.attr("append")(op, qs, cs);
            }
        };

        auto append_measure = [&](int q, int c) {
            if (!branch_stack.empty()) {
                OpTriple t;
                t.op = nb::none();
                t.sentinel_kind = 1;
                t.sentinel_q = q;
                t.sentinel_c = c;
                if (branch_stack.back().in_else)
                    branch_stack.back().else_cmds.push_back(std::move(t));
                else
                    branch_stack.back().then_cmds.push_back(std::move(t));
            } else {
                qc.attr("measure")(qr[nb::int_(q)], cr[nb::int_(c)]);
            }
        };

        auto append_reset = [&](int q) {
            if (!branch_stack.empty()) {
                OpTriple t;
                t.op = nb::none();
                t.sentinel_kind = 2;
                t.sentinel_q = q;
                if (branch_stack.back().in_else)
                    branch_stack.back().else_cmds.push_back(std::move(t));
                else
                    branch_stack.back().then_cmds.push_back(std::move(t));
            } else {
                qc.attr("reset")(qr[nb::int_(q)]);
            }
        };

        // ── Gate dispatch ─────────────────────────────────────────────────
        for (const auto& cmd : commands) {
            switch (cmd.gate) {

                // ── 1Q no-param ───────────────────────────────────────────
                case GateType::X: {
                    nb::list qs; qs.append(qr[nb::int_(cmd.qubits[0])]);
                    append_op(qiskit_lib.attr("XGate")(), qs);
                    break;
                }
                case GateType::Y: {
                    nb::list qs; qs.append(qr[nb::int_(cmd.qubits[0])]);
                    append_op(qiskit_lib.attr("YGate")(), qs);
                    break;
                }
                case GateType::Z: {
                    nb::list qs; qs.append(qr[nb::int_(cmd.qubits[0])]);
                    append_op(qiskit_lib.attr("ZGate")(), qs);
                    break;
                }
                case GateType::H: {
                    nb::list qs; qs.append(qr[nb::int_(cmd.qubits[0])]);
                    append_op(qiskit_lib.attr("HGate")(), qs);
                    break;
                }
                case GateType::S: {
                    nb::list qs; qs.append(qr[nb::int_(cmd.qubits[0])]);
                    append_op(qiskit_lib.attr("SGate")(), qs);
                    break;
                }
                case GateType::Sdg: {
                    nb::list qs; qs.append(qr[nb::int_(cmd.qubits[0])]);
                    append_op(qiskit_lib.attr("SdgGate")(), qs);
                    break;
                }
                case GateType::T: {
                    nb::list qs; qs.append(qr[nb::int_(cmd.qubits[0])]);
                    append_op(qiskit_lib.attr("TGate")(), qs);
                    break;
                }
                case GateType::Tdg: {
                    nb::list qs; qs.append(qr[nb::int_(cmd.qubits[0])]);
                    append_op(qiskit_lib.attr("TdgGate")(), qs);
                    break;
                }
                case GateType::SX: {
                    nb::list qs; qs.append(qr[nb::int_(cmd.qubits[0])]);
                    append_op(qiskit_lib.attr("SXGate")(), qs);
                    break;
                }
                case GateType::SXdg: {
                    nb::list qs; qs.append(qr[nb::int_(cmd.qubits[0])]);
                    append_op(qiskit_lib.attr("SXdgGate")(), qs);
                    break;
                }
                case GateType::Id: {
                    nb::list qs; qs.append(qr[nb::int_(cmd.qubits[0])]);
                    append_op(qiskit_lib.attr("IGate")(), qs);
                    break;
                }

                // ── 1Q parametric ─────────────────────────────────────────
                case GateType::Rx: {
                    nb::list qs; qs.append(qr[nb::int_(cmd.qubits[0])]);
                    append_op(qiskit_lib.attr("RXGate")(qp(cmd.params[0])), qs);
                    break;
                }
                case GateType::Ry: {
                    nb::list qs; qs.append(qr[nb::int_(cmd.qubits[0])]);
                    append_op(qiskit_lib.attr("RYGate")(qp(cmd.params[0])), qs);
                    break;
                }
                case GateType::Rz: {
                    nb::list qs; qs.append(qr[nb::int_(cmd.qubits[0])]);
                    append_op(qiskit_lib.attr("RZGate")(qp(cmd.params[0])), qs);
                    break;
                }
                case GateType::P: {
                    nb::list qs; qs.append(qr[nb::int_(cmd.qubits[0])]);
                    append_op(qiskit_lib.attr("PhaseGate")(qp(cmd.params[0])), qs);
                    break;
                }
                case GateType::U: {
                    nb::list qs; qs.append(qr[nb::int_(cmd.qubits[0])]);
                    append_op(qiskit_lib.attr("UGate")(
                        qp(cmd.params[0]), qp(cmd.params[1]), qp(cmd.params[2])), qs);
                    break;
                }

                // ── 2Q no-param ───────────────────────────────────────────
                case GateType::CX: {
                    nb::list qs;
                    qs.append(qr[nb::int_(cmd.qubits[0])]);
                    qs.append(qr[nb::int_(cmd.qubits[1])]);
                    append_op(qiskit_lib.attr("CXGate")(), qs);
                    break;
                }
                case GateType::CY: {
                    nb::list qs;
                    qs.append(qr[nb::int_(cmd.qubits[0])]);
                    qs.append(qr[nb::int_(cmd.qubits[1])]);
                    append_op(qiskit_lib.attr("CYGate")(), qs);
                    break;
                }
                case GateType::CZ: {
                    nb::list qs;
                    qs.append(qr[nb::int_(cmd.qubits[0])]);
                    qs.append(qr[nb::int_(cmd.qubits[1])]);
                    append_op(qiskit_lib.attr("CZGate")(), qs);
                    break;
                }
                case GateType::SWAP: {
                    nb::list qs;
                    qs.append(qr[nb::int_(cmd.qubits[0])]);
                    qs.append(qr[nb::int_(cmd.qubits[1])]);
                    append_op(qiskit_lib.attr("SwapGate")(), qs);
                    break;
                }
                case GateType::ECR: {
                    nb::list qs;
                    qs.append(qr[nb::int_(cmd.qubits[0])]);
                    qs.append(qr[nb::int_(cmd.qubits[1])]);
                    append_op(qiskit_lib.attr("ECRGate")(), qs);
                    break;
                }
                case GateType::iSWAP: {
                    nb::list qs;
                    qs.append(qr[nb::int_(cmd.qubits[0])]);
                    qs.append(qr[nb::int_(cmd.qubits[1])]);
                    append_op(qiskit_lib.attr("iSwapGate")(), qs);
                    break;
                }
                case GateType::iSWAPdg: {
                    nb::list qs;
                    qs.append(qr[nb::int_(cmd.qubits[0])]);
                    qs.append(qr[nb::int_(cmd.qubits[1])]);
                    nb::object gate = qiskit_lib.attr("iSwapGate")();
                    append_op(gate.attr("inverse")(), qs);
                    break;
                }
                case GateType::CH: {
                    nb::list qs;
                    qs.append(qr[nb::int_(cmd.qubits[0])]);
                    qs.append(qr[nb::int_(cmd.qubits[1])]);
                    append_op(qiskit_lib.attr("CHGate")(), qs);
                    break;
                }
                case GateType::CS: {
                    nb::list qs;
                    qs.append(qr[nb::int_(cmd.qubits[0])]);
                    qs.append(qr[nb::int_(cmd.qubits[1])]);
                    append_op(qiskit_lib.attr("CSGate")(), qs);
                    break;
                }
                case GateType::CSdg: {
                    nb::list qs;
                    qs.append(qr[nb::int_(cmd.qubits[0])]);
                    qs.append(qr[nb::int_(cmd.qubits[1])]);
                    append_op(qiskit_lib.attr("CSdgGate")(), qs);
                    break;
                }
                case GateType::CSX: {
                    nb::list qs;
                    qs.append(qr[nb::int_(cmd.qubits[0])]);
                    qs.append(qr[nb::int_(cmd.qubits[1])]);
                    append_op(qiskit_lib.attr("CSXGate")(), qs);
                    break;
                }
                case GateType::CSXdg: {
                    nb::list qs;
                    qs.append(qr[nb::int_(cmd.qubits[0])]);
                    qs.append(qr[nb::int_(cmd.qubits[1])]);
                    append_op(qiskit_lib.attr("CSXGate")().attr("inverse")(), qs);
                    break;
                }

                // ── 2Q parametric ─────────────────────────────────────────
                case GateType::CRx: {
                    nb::list qs;
                    qs.append(qr[nb::int_(cmd.qubits[0])]);
                    qs.append(qr[nb::int_(cmd.qubits[1])]);
                    append_op(qiskit_lib.attr("CRXGate")(qp(cmd.params[0])), qs);
                    break;
                }
                case GateType::CRy: {
                    nb::list qs;
                    qs.append(qr[nb::int_(cmd.qubits[0])]);
                    qs.append(qr[nb::int_(cmd.qubits[1])]);
                    append_op(qiskit_lib.attr("CRYGate")(qp(cmd.params[0])), qs);
                    break;
                }
                case GateType::CRz: {
                    nb::list qs;
                    qs.append(qr[nb::int_(cmd.qubits[0])]);
                    qs.append(qr[nb::int_(cmd.qubits[1])]);
                    append_op(qiskit_lib.attr("CRZGate")(qp(cmd.params[0])), qs);
                    break;
                }
                case GateType::CP: {
                    nb::list qs;
                    qs.append(qr[nb::int_(cmd.qubits[0])]);
                    qs.append(qr[nb::int_(cmd.qubits[1])]);
                    append_op(qiskit_lib.attr("CPhaseGate")(qp(cmd.params[0])), qs);
                    break;
                }
                case GateType::RZZ: {
                    nb::list qs;
                    qs.append(qr[nb::int_(cmd.qubits[0])]);
                    qs.append(qr[nb::int_(cmd.qubits[1])]);
                    append_op(qiskit_lib.attr("RZZGate")(qp(cmd.params[0])), qs);
                    break;
                }
                case GateType::RXX: {
                    nb::list qs;
                    qs.append(qr[nb::int_(cmd.qubits[0])]);
                    qs.append(qr[nb::int_(cmd.qubits[1])]);
                    append_op(qiskit_lib.attr("RXXGate")(qp(cmd.params[0])), qs);
                    break;
                }
                case GateType::RYY: {
                    nb::list qs;
                    qs.append(qr[nb::int_(cmd.qubits[0])]);
                    qs.append(qr[nb::int_(cmd.qubits[1])]);
                    append_op(qiskit_lib.attr("RYYGate")(qp(cmd.params[0])), qs);
                    break;
                }
                case GateType::CU: {
                    nb::list qs;
                    qs.append(qr[nb::int_(cmd.qubits[0])]);
                    qs.append(qr[nb::int_(cmd.qubits[1])]);
                    append_op(qiskit_lib.attr("CUGate")(
                        qp(cmd.params[0]), qp(cmd.params[1]),
                        qp(cmd.params[2]), qp(cmd.params[3])), qs);
                    break;
                }

                // ── 3Q no-param ───────────────────────────────────────────
                case GateType::CCX: {
                    nb::list qs;
                    qs.append(qr[nb::int_(cmd.qubits[0])]);
                    qs.append(qr[nb::int_(cmd.qubits[1])]);
                    qs.append(qr[nb::int_(cmd.qubits[2])]);
                    append_op(qiskit_lib.attr("CCXGate")(), qs);
                    break;
                }
                case GateType::CSWAP: {
                    nb::list qs;
                    qs.append(qr[nb::int_(cmd.qubits[0])]);
                    qs.append(qr[nb::int_(cmd.qubits[1])]);
                    qs.append(qr[nb::int_(cmd.qubits[2])]);
                    append_op(qiskit_lib.attr("CSwapGate")(), qs);
                    break;
                }

                // ── MCZ: H(target) + MCX + H(target) ─────────────────────
                case GateType::MCZ: {
                    // Last qubit is target; all preceding are controls.
                    uint32_t target = cmd.qubits.back();
                    int n_ctrl = static_cast<int>(cmd.qubits.size()) - 1;

                    nb::list all_qs;
                    for (auto q : cmd.qubits)
                        all_qs.append(qr[nb::int_(q)]);

                    nb::list tq; tq.append(qr[nb::int_(target)]);

                    // H before
                    append_op(qiskit_lib.attr("HGate")(), tq);

                    // MCX (all qubits, controls first then target)
                    append_op(qiskit_lib.attr("MCXGate")(nb::cast(n_ctrl)), all_qs);

                    // H after
                    append_op(qiskit_lib.attr("HGate")(), tq);
                    break;
                }

                // ── GPhase ────────────────────────────────────────────────
                case GateType::GPhase: {
                    nb::list empty_qs;
                    append_op(qiskit_lib.attr("GlobalPhaseGate")(qp(cmd.params[0])), empty_qs);
                    break;
                }

                // ── Barrier ───────────────────────────────────────────────
                case GateType::Barrier: {
                    nb::list qs;
                    for (auto q : cmd.qubits)
                        qs.append(qr[nb::int_(q)]);
                    if (!branch_stack.empty()) {
                        // Push a sentinel barrier triple
                        OpTriple t;
                        t.op = nb::none();
                        t.sentinel_kind = 3; // barrier sentinel
                        t.qubits = qs;
                        if (branch_stack.back().in_else)
                            branch_stack.back().else_cmds.push_back(std::move(t));
                        else
                            branch_stack.back().then_cmds.push_back(std::move(t));
                    } else {
                        qc.attr("barrier")(qs);
                    }
                    break;
                }

                // ── Measure ───────────────────────────────────────────────
                case GateType::Measure: {
                    int q = static_cast<int>(cmd.qubits[0]);
                    int c = cmd.cbits.empty()
                                ? static_cast<int>(cmd.qubits[0])
                                : static_cast<int>(cmd.cbits[0]);
                    append_measure(q, c);
                    break;
                }

                // ── Reset ─────────────────────────────────────────────────
                case GateType::Reset: {
                    append_reset(static_cast<int>(cmd.qubits[0]));
                    break;
                }

                // ── Branch control flow ───────────────────────────────────
                case GateType::BranchBegin: {
                    BranchFrame frame;
                    for (auto b : cmd.condition_bits)
                        frame.condition_bits.push_back(b);
                    for (auto v : cmd.condition_values)
                        frame.condition_values.push_back(v);
                    branch_stack.push_back(std::move(frame));
                    break;
                }
                case GateType::BranchElse: {
                    if (!branch_stack.empty())
                        branch_stack.back().in_else = true;
                    break;
                }
                case GateType::BranchEnd: {
                    if (!branch_stack.empty()) {
                        BranchFrame frame = std::move(branch_stack.back());
                        branch_stack.pop_back();
                        flush_if_else(qc, frame, n_qubits, cr, qiskit_circ);
                    }
                    break;
                }

                // Custom is rejected by validate() and lands in the default
                // arm's defensive throw if it ever slips through.
                default:
                    throw std::runtime_error(
                        std::string("QiskitEmitter: unrecognized gate: ") +
                        std::string(gate_name(cmd.gate)));
            }
        }

        return qc;
    }
};

// ── flush_if_else ─────────────────────────────────────────────────────────────

static void flush_if_else(nb::object& qc,
                          const BranchFrame& frame,
                          uint32_t n_qubits,
                          const nb::object& cr,
                          nb::object& qiskit_circ_mod)
{
    const auto& bits = frame.condition_bits;
    const auto& vals = frame.condition_values;
    bool has_else = !frame.else_cmds.empty();

    // ── Build condition ───────────────────────────────────────────────────
    // IfElseOp takes a (Clbit, int) 2-tuple. A single Clbit lifts to a Bool()
    // classical expression, which cannot be compared to an integer value, so
    // the expr builder path does not apply to bit-wise conditions.
    nb::object condition;
    if (!cr.is_none()) {
        if (bits.size() == 1) {
            condition = nb::make_tuple(cr[nb::int_(bits[0])], nb::cast((int)vals[0]));
        } else {
            throw std::runtime_error(
                "QiskitEmitter: multi-bit branch conditions are not supported");
        }
    }

    // ── Sub-circuit builder lambda ────────────────────────────────────────
    auto build_body = [&](const std::vector<OpTriple>& ops) -> nb::object {
        nb::object QuantumCircuit = qiskit_circ_mod.attr("QuantumCircuit");
        nb::object body_qr = qiskit_circ_mod.attr("QuantumRegister")(
            nb::cast(n_qubits), nb::cast(std::string("q")));
        nb::object body_cr = nb::none();
        nb::object body;
        if (!cr.is_none()) {
            int n_cbits = nb::cast<int>(cr.attr("size"));
            body_cr = qiskit_circ_mod.attr("ClassicalRegister")(
                nb::cast(n_cbits), nb::cast(std::string("c")));
            body = QuantumCircuit(body_qr, body_cr);
        } else {
            body = QuantumCircuit(body_qr);
        }

        for (const auto& t : ops) {
            if (t.sentinel_kind == 1) {
                // measure
                body.attr("measure")(body_qr[nb::int_(t.sentinel_q)],
                                     body_cr[nb::int_(t.sentinel_c)]);
            } else if (t.sentinel_kind == 2) {
                // reset
                body.attr("reset")(body_qr[nb::int_(t.sentinel_q)]);
            } else if (t.sentinel_kind == 3) {
                // barrier
                body.attr("barrier")(t.qubits);
            } else {
                // Regular gate: remap qubit objects to body_qr
                // t.qubits contains objects from the outer qr; we need their indices.
                // We stored the outer circuit qubit objects; we need positions.
                // Instead, re-index via the position in the outer qc.qubits list.
                nb::list mapped_qs;
                for (auto q : t.qubits) {
                    int idx = nb::cast<int>(qc.attr("find_bit")(q).attr("index"));
                    mapped_qs.append(body_qr[nb::int_(idx)]);
                }
                nb::list mapped_cs;
                for (auto c : t.clbits) {
                    int idx = nb::cast<int>(qc.attr("find_bit")(c).attr("index"));
                    if (!body_cr.is_none())
                        mapped_cs.append(body_cr[nb::int_(idx)]);
                }
                body.attr("append")(t.op, mapped_qs, mapped_cs);
            }
        }
        return body;
    };

    nb::object true_body  = build_body(frame.then_cmds);
    nb::object false_body = has_else ? build_body(frame.else_cmds) : nb::none();

    // ── Build IfElseOp and append ─────────────────────────────────────────
    nb::object IfElseOp = qiskit_circ_mod.attr("IfElseOp");

    nb::list all_qubits;
    for (uint32_t i = 0; i < n_qubits; ++i)
        all_qubits.append(qc.attr("qubits")[nb::int_(i)]);

    nb::list all_clbits;
    if (!cr.is_none()) {
        // A ClassicalRegister is iterable over its Clbits (qiskit dropped the
        // ``.bits`` attribute).
        for (auto b : cr)
            all_clbits.append(b);
    }

    nb::object op;
    if (has_else)
        op = IfElseOp(condition, true_body, false_body);
    else
        op = IfElseOp(condition, true_body);

    qc.attr("append")(op, all_qubits, all_clbits);
}

// ── Registration ──────────────────────────────────────────────────────────────

void register_qiskit_emitter(nb::module_& m) {
    nb::class_<QiskitEmitter>(m, "QiskitEmitter")
        .def(nb::init<>())
        .def("target_name", &QiskitEmitter::target_name)
        .def("gate_set", &QiskitEmitter::gate_set)
        .def("capabilities", &QiskitEmitter::capabilities)
        .def("validate",
             [](const QiskitEmitter& self, const std::vector<Command>& cmds) {
                 return self.validate(cmds);
             },
             "commands"_a,
             "First Incompatibility in the sequence, or None if the whole "
             "sequence can cross.  Needs no SDK installed.")
        .def("emit", &QiskitEmitter::emit,
             "commands"_a, "n_qubits"_a, "circuit_name"_a = "circuit",
             "Translate a flat QARPx command sequence into a "
             "qiskit.QuantumCircuit.");
}
