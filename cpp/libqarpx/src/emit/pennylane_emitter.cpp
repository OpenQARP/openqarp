#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/optional.h>
#include "qarpx/core/command.h"
#include "qarpx/core/errors.h"
#include "qarpx/core/gates.h"
#include "qarpx/core/param.h"
#include "qarpx/block/block.h"
#include "qarpx/emit/emitter.h"
#include <cmath>
#include <complex>
#include <stdexcept>
#include <unordered_map>

namespace nb = nanobind;
using namespace nb::literals;
using namespace qarpx;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Extract a concrete double from a Param.  Symbolic params are rejected
/// earlier (before the gate switch), so this always succeeds at call site.
static double pf(const Param& p) {
    return p.value();
}

/// CU(θ,φ,λ,γ) matrix = e^{iγ} · U3(θ,φ,λ).  PennyLane has no qml.CU gate;
/// the controlled-U-with-global-phase must be built as an explicit 2×2
/// matrix and applied via ControlledQubitUnitary.
static nb::object cu_matrix(double theta, double phi, double lambda_,
                             double gamma, nb::object& np) {
    double c = std::cos(theta / 2.0);
    double s = std::sin(theta / 2.0);
    using cd = std::complex<double>;
    cd ph    = cd(std::cos(gamma), std::sin(gamma));
    cd e_il  = cd(std::cos(lambda_),  std::sin(lambda_));
    cd e_ip  = cd(std::cos(phi),      std::sin(phi));
    cd e_ipl = cd(std::cos(phi + lambda_), std::sin(phi + lambda_));
    return np.attr("array")(
        nb::make_tuple(
            nb::make_tuple(nb::cast(ph * cd(c, 0.0)), nb::cast(ph * -e_il * s)),
            nb::make_tuple(nb::cast(ph * e_ip * s),   nb::cast(ph * e_ipl * c))),
        "dtype"_a = np.attr("complex128"));
}

// ─────────────────────────────────────────────────────────────────────────────

class PennylaneEmitter : public Emitter {
public:
    [[nodiscard]] std::string target_name() const override { return "pennylane"; }

    [[nodiscard]] GateSet gate_set() const override {
        return pennylane_gateset();
    }

    [[nodiscard]] EmitterCapabilities capabilities() const override {
        return EmitterCapabilities{
            .symbolic_params     = false,
            .multi_symbol_params = false,
            .distinct_cbits      = false,  // no classical register concept
            .conditionals        = false,
            .multibit_conditions = false,
            .multibit_else       = false,
            .custom_unitary      = false,
            .cu_global_phase     = true,   // ControlledQubitUnitary carries γ
        };
    }

    nb::object emit(const std::vector<Command>& commands,
                    uint32_t /*n_qubits*/,
                    const std::string& /*circuit_name*/ = "circuit") const
    {
        throw_if_invalid(commands);

        // ── Import check ──────────────────────────────────────────────────
        nb::object qml;
        try {
            qml = nb::module_::import_("pennylane");
        } catch (...) {
            throw sdk_missing_error(
                "PennyLane not installed. Install with: pip install pennylane>=0.36");
        }

        nb::object qml_tape   = nb::module_::import_("pennylane.tape");
        nb::object qml_ops    = nb::module_::import_("pennylane.ops");
        (void)qml_ops; // imported for side-effect; ops accessed via qml attribute
        nb::object np         = nb::module_::import_("numpy");

        // ── Build ops list ────────────────────────────────────────────────
        nb::list ops_list;
        // Map: cbit index → PennyLane MeasurementValue (qml.measure return)
        std::unordered_map<uint32_t, nb::object> mv_map;

        for (const auto& cmd : commands) {
            switch (cmd.gate) {

                // ── 1Q no-param ──────────────────────────────────────────
                case GateType::X: {
                    nb::object op = qml.attr("PauliX")(
                        "wires"_a = static_cast<int>(cmd.qubits[0]));
                    ops_list.append(op);
                    break;
                }
                case GateType::Y: {
                    nb::object op = qml.attr("PauliY")(
                        "wires"_a = static_cast<int>(cmd.qubits[0]));
                    ops_list.append(op);
                    break;
                }
                case GateType::Z: {
                    nb::object op = qml.attr("PauliZ")(
                        "wires"_a = static_cast<int>(cmd.qubits[0]));
                    ops_list.append(op);
                    break;
                }
                case GateType::H: {
                    nb::object op = qml.attr("Hadamard")(
                        "wires"_a = static_cast<int>(cmd.qubits[0]));
                    ops_list.append(op);
                    break;
                }
                case GateType::S: {
                    nb::object op = qml.attr("S")(
                        "wires"_a = static_cast<int>(cmd.qubits[0]));
                    ops_list.append(op);
                    break;
                }
                case GateType::T: {
                    nb::object op = qml.attr("T")(
                        "wires"_a = static_cast<int>(cmd.qubits[0]));
                    ops_list.append(op);
                    break;
                }
                case GateType::Sdg: {
                    // qml.adjoint(qml.S)(wires=q)
                    nb::object adj_s = qml.attr("adjoint")(qml.attr("S"));
                    nb::object op = adj_s(
                        "wires"_a = static_cast<int>(cmd.qubits[0]));
                    ops_list.append(op);
                    break;
                }
                case GateType::Tdg: {
                    nb::object adj_t = qml.attr("adjoint")(qml.attr("T"));
                    nb::object op = adj_t(
                        "wires"_a = static_cast<int>(cmd.qubits[0]));
                    ops_list.append(op);
                    break;
                }
                case GateType::SX: {
                    nb::object op = qml.attr("SX")(
                        "wires"_a = static_cast<int>(cmd.qubits[0]));
                    ops_list.append(op);
                    break;
                }
                case GateType::SXdg: {
                    nb::object op = qml.attr("adjoint")(qml.attr("SX"))(
                        "wires"_a = static_cast<int>(cmd.qubits[0]));
                    ops_list.append(op);
                    break;
                }
                case GateType::Id: {
                    nb::object op = qml.attr("Identity")(
                        "wires"_a = static_cast<int>(cmd.qubits[0]));
                    ops_list.append(op);
                    break;
                }

                // ── 1Q parametric ────────────────────────────────────────
                case GateType::Rx: {
                    nb::object op = qml.attr("RX")(
                        pf(cmd.params[0]),
                        "wires"_a = static_cast<int>(cmd.qubits[0]));
                    ops_list.append(op);
                    break;
                }
                case GateType::Ry: {
                    nb::object op = qml.attr("RY")(
                        pf(cmd.params[0]),
                        "wires"_a = static_cast<int>(cmd.qubits[0]));
                    ops_list.append(op);
                    break;
                }
                case GateType::Rz: {
                    nb::object op = qml.attr("RZ")(
                        pf(cmd.params[0]),
                        "wires"_a = static_cast<int>(cmd.qubits[0]));
                    ops_list.append(op);
                    break;
                }
                case GateType::P: {
                    nb::object op = qml.attr("PhaseShift")(
                        pf(cmd.params[0]),
                        "wires"_a = static_cast<int>(cmd.qubits[0]));
                    ops_list.append(op);
                    break;
                }
                case GateType::U: {
                    nb::object op = qml.attr("U3")(
                        pf(cmd.params[0]),
                        pf(cmd.params[1]),
                        pf(cmd.params[2]),
                        "wires"_a = static_cast<int>(cmd.qubits[0]));
                    ops_list.append(op);
                    break;
                }

                // ── 2Q no-param ──────────────────────────────────────────
                case GateType::CX: {
                    nb::list wires;
                    wires.append(static_cast<int>(cmd.qubits[0]));
                    wires.append(static_cast<int>(cmd.qubits[1]));
                    nb::object op = qml.attr("CNOT")("wires"_a = wires);
                    ops_list.append(op);
                    break;
                }
                case GateType::CY: {
                    nb::list wires;
                    wires.append(static_cast<int>(cmd.qubits[0]));
                    wires.append(static_cast<int>(cmd.qubits[1]));
                    nb::object op = qml.attr("CY")("wires"_a = wires);
                    ops_list.append(op);
                    break;
                }
                case GateType::CZ: {
                    nb::list wires;
                    wires.append(static_cast<int>(cmd.qubits[0]));
                    wires.append(static_cast<int>(cmd.qubits[1]));
                    nb::object op = qml.attr("CZ")("wires"_a = wires);
                    ops_list.append(op);
                    break;
                }
                case GateType::SWAP: {
                    nb::list wires;
                    wires.append(static_cast<int>(cmd.qubits[0]));
                    wires.append(static_cast<int>(cmd.qubits[1]));
                    nb::object op = qml.attr("SWAP")("wires"_a = wires);
                    ops_list.append(op);
                    break;
                }
                case GateType::ECR: {
                    nb::list wires;
                    wires.append(static_cast<int>(cmd.qubits[0]));
                    wires.append(static_cast<int>(cmd.qubits[1]));
                    nb::object op = qml.attr("ECR")("wires"_a = wires);
                    ops_list.append(op);
                    break;
                }
                case GateType::iSWAP: {
                    nb::list wires;
                    wires.append(static_cast<int>(cmd.qubits[0]));
                    wires.append(static_cast<int>(cmd.qubits[1]));
                    nb::object op = qml.attr("ISWAP")("wires"_a = wires);
                    ops_list.append(op);
                    break;
                }
                case GateType::iSWAPdg: {
                    nb::list wires;
                    wires.append(static_cast<int>(cmd.qubits[0]));
                    wires.append(static_cast<int>(cmd.qubits[1]));
                    nb::object adj_iswap = qml.attr("adjoint")(qml.attr("ISWAP"));
                    nb::object op = adj_iswap("wires"_a = wires);
                    ops_list.append(op);
                    break;
                }
                case GateType::CH: {
                    nb::list wires;
                    wires.append(static_cast<int>(cmd.qubits[0]));
                    wires.append(static_cast<int>(cmd.qubits[1]));
                    nb::object op = qml.attr("CH")("wires"_a = wires);
                    ops_list.append(op);
                    break;
                }
                // No CS/CSX classes in PennyLane: qml.ctrl on the 1Q base (control first).
                case GateType::CS: {
                    nb::object op = qml.attr("ctrl")(
                        qml.attr("S")("wires"_a = static_cast<int>(cmd.qubits[1])),
                        "control"_a = static_cast<int>(cmd.qubits[0]));
                    ops_list.append(op);
                    break;
                }
                case GateType::CSdg: {
                    nb::object op = qml.attr("ctrl")(
                        qml.attr("adjoint")(qml.attr("S"))("wires"_a = static_cast<int>(cmd.qubits[1])),
                        "control"_a = static_cast<int>(cmd.qubits[0]));
                    ops_list.append(op);
                    break;
                }
                case GateType::CSX: {
                    nb::object op = qml.attr("ctrl")(
                        qml.attr("SX")("wires"_a = static_cast<int>(cmd.qubits[1])),
                        "control"_a = static_cast<int>(cmd.qubits[0]));
                    ops_list.append(op);
                    break;
                }
                case GateType::CSXdg: {
                    nb::object op = qml.attr("ctrl")(
                        qml.attr("adjoint")(qml.attr("SX"))("wires"_a = static_cast<int>(cmd.qubits[1])),
                        "control"_a = static_cast<int>(cmd.qubits[0]));
                    ops_list.append(op);
                    break;
                }

                // ── 2Q parametric ────────────────────────────────────────
                case GateType::CRx: {
                    nb::list wires;
                    wires.append(static_cast<int>(cmd.qubits[0]));
                    wires.append(static_cast<int>(cmd.qubits[1]));
                    nb::object op = qml.attr("CRX")(
                        pf(cmd.params[0]), "wires"_a = wires);
                    ops_list.append(op);
                    break;
                }
                case GateType::CRy: {
                    nb::list wires;
                    wires.append(static_cast<int>(cmd.qubits[0]));
                    wires.append(static_cast<int>(cmd.qubits[1]));
                    nb::object op = qml.attr("CRY")(
                        pf(cmd.params[0]), "wires"_a = wires);
                    ops_list.append(op);
                    break;
                }
                case GateType::CRz: {
                    nb::list wires;
                    wires.append(static_cast<int>(cmd.qubits[0]));
                    wires.append(static_cast<int>(cmd.qubits[1]));
                    nb::object op = qml.attr("CRZ")(
                        pf(cmd.params[0]), "wires"_a = wires);
                    ops_list.append(op);
                    break;
                }
                case GateType::CP: {
                    nb::list wires;
                    wires.append(static_cast<int>(cmd.qubits[0]));
                    wires.append(static_cast<int>(cmd.qubits[1]));
                    nb::object op = qml.attr("ControlledPhaseShift")(
                        pf(cmd.params[0]), "wires"_a = wires);
                    ops_list.append(op);
                    break;
                }
                case GateType::RZZ: {
                    nb::list wires;
                    wires.append(static_cast<int>(cmd.qubits[0]));
                    wires.append(static_cast<int>(cmd.qubits[1]));
                    nb::object op = qml.attr("IsingZZ")(
                        pf(cmd.params[0]), "wires"_a = wires);
                    ops_list.append(op);
                    break;
                }
                case GateType::RXX: {
                    nb::list wires;
                    wires.append(static_cast<int>(cmd.qubits[0]));
                    wires.append(static_cast<int>(cmd.qubits[1]));
                    nb::object op = qml.attr("IsingXX")(
                        pf(cmd.params[0]), "wires"_a = wires);
                    ops_list.append(op);
                    break;
                }
                case GateType::RYY: {
                    nb::list wires;
                    wires.append(static_cast<int>(cmd.qubits[0]));
                    wires.append(static_cast<int>(cmd.qubits[1]));
                    nb::object op = qml.attr("IsingYY")(
                        pf(cmd.params[0]), "wires"_a = wires);
                    ops_list.append(op);
                    break;
                }
                case GateType::CU: {
                    // PennyLane has no qml.CU gate — build the explicit
                    // e^{iγ}·U3(θ,φ,λ) matrix and apply it as a controlled
                    // arbitrary unitary (control wire first, then target).
                    nb::object mat = cu_matrix(pf(cmd.params[0]), pf(cmd.params[1]),
                                                pf(cmd.params[2]), pf(cmd.params[3]), np);
                    nb::list wires;
                    wires.append(static_cast<int>(cmd.qubits[0]));
                    wires.append(static_cast<int>(cmd.qubits[1]));
                    nb::object op = qml.attr("ControlledQubitUnitary")(mat, "wires"_a = wires);
                    ops_list.append(op);
                    break;
                }

                // ── 3Q no-param ──────────────────────────────────────────
                case GateType::CCX: {
                    nb::list wires;
                    wires.append(static_cast<int>(cmd.qubits[0]));
                    wires.append(static_cast<int>(cmd.qubits[1]));
                    wires.append(static_cast<int>(cmd.qubits[2]));
                    nb::object op = qml.attr("Toffoli")("wires"_a = wires);
                    ops_list.append(op);
                    break;
                }
                case GateType::CSWAP: {
                    nb::list wires;
                    wires.append(static_cast<int>(cmd.qubits[0]));
                    wires.append(static_cast<int>(cmd.qubits[1]));
                    wires.append(static_cast<int>(cmd.qubits[2]));
                    nb::object op = qml.attr("CSWAP")("wires"_a = wires);
                    ops_list.append(op);
                    break;
                }

                // ── Multi-qubit: MCZ ─────────────────────────────────────
                case GateType::MCZ: {
                    // Last qubit is the target; all preceding are controls.
                    nb::list controls;
                    for (size_t i = 0; i + 1 < cmd.qubits.size(); ++i) {
                        controls.append(static_cast<int>(cmd.qubits[i]));
                    }
                    int target = static_cast<int>(cmd.qubits.back());
                    // Control an *instantiated* PauliZ(wires=target); the bare
                    // class produces an empty op. Gives CZ / CCZ / Controlled by
                    // control count.
                    nb::object pz = qml.attr("PauliZ")("wires"_a = target);
                    nb::object op = qml.attr("ctrl")(pz, "control"_a = controls);
                    ops_list.append(op);
                    break;
                }

                // ── GPhase ───────────────────────────────────────────────
                case GateType::GPhase: {
                    // qml.GlobalPhase(φ) = e^{-iφ}, QARPx GPhase(θ) = e^{+iθ} (§7):
                    // negate so the emitted global phase matches.
                    nb::object op = qml.attr("GlobalPhase")(-pf(cmd.params[0]));
                    ops_list.append(op);
                    break;
                }

                // ── Measure ──────────────────────────────────────────────
                case GateType::Measure: {
                    // cbit != qubit is rejected by validate() (distinct_cbits =
                    // false): qml.measure(wire) has no separate classical
                    // register, so only the trivial assignment reaches here.
                    uint32_t qubit = cmd.qubits[0];
                    uint32_t cbit  = cmd.cbits.empty() ? qubit : cmd.cbits[0];
                    nb::object mv  = qml.attr("measure")(static_cast<int>(qubit));
                    mv_map[cbit]   = mv;
                    // qml.measure() returns a MeasurementValue (used for qml.cond),
                    // not an Operation — the queueable mid-circuit-measurement op
                    // itself is mv.measurements[0].
                    ops_list.append(mv.attr("measurements")[0]);
                    break;
                }

                // ── Silently skipped ─────────────────────────────────────
                case GateType::Barrier:
                case GateType::BranchElse:
                case GateType::BranchEnd:
                    break;

                // ── Already rejected by validate() ────────────────────────
                case GateType::Reset:
                case GateType::BranchBegin:
                case GateType::Custom:
                    // Unreachable: rejected by throw_if_invalid().
                    break;

                default:
                    throw std::runtime_error(
                        std::string("PennylaneEmitter: unsupported gate: ") +
                        std::string(gate_name(cmd.gate)));
            }
        }

        // ── Construct QuantumScript ───────────────────────────────────────
        nb::object qs_cls    = qml_tape.attr("QuantumScript");
        nb::object tape      = qs_cls(ops_list);
        return tape;
    }
};

void register_pennylane_emitter(nb::module_& m) {
    nb::class_<PennylaneEmitter>(m, "PennylaneEmitter")
        .def(nb::init<>())
        .def("target_name", &PennylaneEmitter::target_name)
        .def("gate_set", &PennylaneEmitter::gate_set)
        .def("capabilities", &PennylaneEmitter::capabilities)
        .def("validate",
             [](const PennylaneEmitter& self, const std::vector<Command>& cmds) {
                 return self.validate(cmds);
             },
             "commands"_a,
             "First Incompatibility in the sequence, or None if the whole "
             "sequence can cross.  Needs no SDK installed.")
        .def("emit", &PennylaneEmitter::emit,
             "commands"_a, "n_qubits"_a, "circuit_name"_a = "circuit",
             "Translate a flat QARPx command sequence into a "
             "qml.tape.QuantumScript.");
}
