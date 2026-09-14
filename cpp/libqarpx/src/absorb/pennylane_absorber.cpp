#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/complex.h>
#include <nanobind/intrusive/ref.h>
#include "qarpx/absorb/absorber.h"
#include "qarpx/core/command.h"
#include "qarpx/core/gates.h"
#include "qarpx/core/param.h"
#include "qarpx/block/block.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <unordered_map>

namespace nb = nanobind;
using namespace nb::literals;
using namespace qarpx;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Collect integer wire indices from a PennyLane Wires object.
static std::vector<int> get_wires(nb::object op) {
    std::vector<int> ws;
    for (auto w : op.attr("wires")) {
        ws.push_back(nb::cast<int>(w));
    }
    return ws;
}

/// Get the Python class name of an object.
static std::string class_name(nb::object obj) {
    return nb::cast<std::string>(obj.attr("__class__").attr("__name__"));
}

/// Get double parameter at index i from op.data.
static double get_param(nb::object op, size_t i) {
    nb::object data = op.attr("data");
    return nb::cast<double>(data[nb::cast(static_cast<int>(i))]);
}

/// Decompose an arbitrary 2x2 unitary matrix M into QARPx's
/// (θ,φ,λ,γ) such that M == e^{iγ} · U3(θ,φ,λ), with
/// U3 = [[cos(θ/2), -e^{iλ}sin(θ/2)], [e^{iφ}sin(θ/2), e^{i(φ+λ)}cos(θ/2)]].
/// γ is taken as the phase of M[0][0] (assumes cos(θ/2) != 0; θ ≈ π is not
/// hit by any angle this codebase emits, so the degenerate branch is unused
/// in practice but still returns a valid — if arbitrary — phi/lambda split).
static std::array<double, 4> decompose_u3_with_phase(
        const std::complex<double>& m00, const std::complex<double>& m01,
        const std::complex<double>& m10, const std::complex<double>& m11) {
    using cd = std::complex<double>;
    double gamma;
    cd a, b, c, d;
    if (std::abs(m00) > 1e-9) {
        cd phase = m00 / std::abs(m00);
        gamma = std::atan2(phase.imag(), phase.real());
        a = m00 / phase; b = m01 / phase; c = m10 / phase; d = m11 / phase;
    } else {
        // cos(θ/2) ≈ 0 (θ ≈ π): anchor the phase on m10 = e^{iφ}sin(θ/2) instead.
        cd phase = m10 / std::abs(m10);
        gamma = std::atan2(phase.imag(), phase.real());
        a = m00 / phase; b = m01 / phase; c = m10 / phase; d = m11 / phase;
    }
    double theta = 2.0 * std::acos(std::clamp(a.real(), -1.0, 1.0));
    double phi = 0.0, lambda_ = 0.0;
    if (std::abs(c) > 1e-9 || std::abs(b) > 1e-9) {
        phi     = std::atan2(c.imag(), c.real());
        lambda_ = std::atan2((-b).imag(), (-b).real());
    }
    (void)d;
    return {theta, phi, lambda_, gamma};
}

// ─────────────────────────────────────────────────────────────────────────────

class PennylaneAbsorber : public Absorber {
public:
    [[nodiscard]] std::string source_name() const override { return "pennylane"; }

    nb::object absorb(nb::object tape) const {
        // ── Determine n_qubits from tape.wires ────────────────────────────
        nb::object py_sorted = nb::module_::import_("builtins").attr("sorted");
        nb::object wire_obj  = tape.attr("wires");
        nb::list   sorted_ws = nb::cast<nb::list>(py_sorted(wire_obj));

        uint32_t n_qubits = 0;
        if (sorted_ws.size() > 0) {
            int max_wire = nb::cast<int>(sorted_ws[sorted_ws.size() - 1]);
            n_qubits     = static_cast<uint32_t>(max_wire + 1);
        }

        // ── Create the SimpleBlock ────────────────────────────────────────
        ref<SimpleBlock> block(new SimpleBlock(n_qubits));

        // ── Iterate operations ────────────────────────────────────────────
        nb::object operations = tape.attr("operations");

        for (auto op_handle : operations) {
            nb::object op   = nb::cast<nb::object>(op_handle);
            std::string name = class_name(op);

            // ── Adjoint detection ─────────────────────────────────────────
            // PennyLane wraps Adjoint(S), Adjoint(T), Adjoint(ISWAP) with a
            // `.base` attribute; the wrapper class is named "Adjoint" for a
            // generic Operator base and "AdjointOperation" when the base is
            // an Operation subclass (PennyLane's internal dispatch mixin).
            if (name == "Adjoint" || name == "AdjointOperation") {
                nb::object base    = op.attr("base");
                std::string bname  = class_name(base);
                auto ws            = get_wires(op);

                if (bname == "S") {
                    block->sdg(static_cast<uint32_t>(ws[0]));
                } else if (bname == "T") {
                    block->tdg(static_cast<uint32_t>(ws[0]));
                } else if (bname == "ISWAP") {
                    block->iswapdg(static_cast<uint32_t>(ws[0]),
                                   static_cast<uint32_t>(ws[1]));
                } else if (bname == "SX") {
                    block->sxdg(static_cast<uint32_t>(ws[0]));
                } else {
                    throw std::runtime_error(
                        "PennylaneAbsorber: unsupported Adjoint base: " + bname);
                }
                continue;
            }

            // ── Standard gate dispatch ────────────────────────────────────
            auto ws = get_wires(op);

            if (name == "PauliX") {
                block->x(static_cast<uint32_t>(ws[0]));

            } else if (name == "PauliY") {
                block->y(static_cast<uint32_t>(ws[0]));

            } else if (name == "PauliZ") {
                block->z(static_cast<uint32_t>(ws[0]));

            } else if (name == "Hadamard") {
                block->h(static_cast<uint32_t>(ws[0]));

            } else if (name == "SX") {
                block->sx(static_cast<uint32_t>(ws[0]));

            } else if (name == "Identity") {
                block->id(static_cast<uint32_t>(ws[0]));

            } else if (name == "CH") {
                block->ch(static_cast<uint32_t>(ws[0]), static_cast<uint32_t>(ws[1]));

            // qml.ctrl on S / SX (or their adjoints): control wire first.
            } else if ((name == "Controlled" || name == "ControlledOp") &&
                       class_name(op.attr("base")) != "PauliZ") {
                nb::object base = op.attr("base");
                std::string bname = class_name(base);
                bool adj = false;
                if (bname == "Adjoint" || bname == "AdjointOperation") {
                    adj = true;
                    base = base.attr("base");
                    bname = class_name(base);
                }
                if (ws.size() != 2)
                    throw std::runtime_error(
                        "PennylaneAbsorber: controlled " + bname + " with " +
                        std::to_string(ws.size()) + " wires is not a 2-qubit gate");
                const auto c = static_cast<uint32_t>(ws[0]);
                const auto t = static_cast<uint32_t>(ws[1]);
                if (bname == "S")            { adj ? block->csdg(c, t)  : block->cs(c, t); }
                else if (bname == "SX")      { adj ? block->csxdg(c, t) : block->csx(c, t); }
                else if (bname == "Hadamard" && !adj) { block->ch(c, t); }
                else
                    throw std::runtime_error(
                        "PennylaneAbsorber: unsupported controlled base: " + bname);

            } else if (name == "S") {
                block->s(static_cast<uint32_t>(ws[0]));

            } else if (name == "T") {
                block->t(static_cast<uint32_t>(ws[0]));

            } else if (name == "RX") {
                block->rx(static_cast<uint32_t>(ws[0]),
                          Param(get_param(op, 0)));

            } else if (name == "RY") {
                block->ry(static_cast<uint32_t>(ws[0]),
                          Param(get_param(op, 0)));

            } else if (name == "RZ") {
                block->rz(static_cast<uint32_t>(ws[0]),
                          Param(get_param(op, 0)));

            } else if (name == "PhaseShift") {
                block->p(static_cast<uint32_t>(ws[0]),
                         Param(get_param(op, 0)));

            } else if (name == "U3") {
                block->u(static_cast<uint32_t>(ws[0]),
                         Param(get_param(op, 0)),
                         Param(get_param(op, 1)),
                         Param(get_param(op, 2)));

            } else if (name == "CNOT") {
                block->cx(static_cast<uint32_t>(ws[0]),
                          static_cast<uint32_t>(ws[1]));

            } else if (name == "CY") {
                block->cy(static_cast<uint32_t>(ws[0]),
                          static_cast<uint32_t>(ws[1]));

            } else if (name == "CZ") {
                block->cz(static_cast<uint32_t>(ws[0]),
                          static_cast<uint32_t>(ws[1]));

            } else if (name == "CCZ" ||
                       ((name == "Controlled" || name == "ControlledOp") &&
                        class_name(op.attr("base")) == "PauliZ")) {
                // Multi-controlled Z (qml.ctrl(PauliZ, ...) → CCZ for 2 controls,
                // Controlled/ControlledOp for ≥3 — the Operator/Operation split,
                // as with Adjoint/AdjointOperation above). MCZ is symmetric (phase
                // −1 on all-ones), so the full wire list maps directly.
                std::vector<uint32_t> mcz_ws;
                for (auto w : ws) mcz_ws.push_back(static_cast<uint32_t>(w));
                block->mcz(mcz_ws);

            } else if (name == "SWAP") {
                block->swap(static_cast<uint32_t>(ws[0]),
                            static_cast<uint32_t>(ws[1]));

            } else if (name == "ECR") {
                block->ecr(static_cast<uint32_t>(ws[0]),
                           static_cast<uint32_t>(ws[1]));

            } else if (name == "ISWAP") {
                block->iswap(static_cast<uint32_t>(ws[0]),
                             static_cast<uint32_t>(ws[1]));

            } else if (name == "CRX") {
                block->crx(static_cast<uint32_t>(ws[0]),
                           static_cast<uint32_t>(ws[1]),
                           Param(get_param(op, 0)));

            } else if (name == "CRY") {
                block->cry(static_cast<uint32_t>(ws[0]),
                           static_cast<uint32_t>(ws[1]),
                           Param(get_param(op, 0)));

            } else if (name == "CRZ") {
                block->crz(static_cast<uint32_t>(ws[0]),
                           static_cast<uint32_t>(ws[1]),
                           Param(get_param(op, 0)));

            } else if (name == "ControlledPhaseShift") {
                block->cp(static_cast<uint32_t>(ws[0]),
                          static_cast<uint32_t>(ws[1]),
                          Param(get_param(op, 0)));

            } else if (name == "IsingZZ") {
                block->rzz(static_cast<uint32_t>(ws[0]),
                           static_cast<uint32_t>(ws[1]),
                           Param(get_param(op, 0)));

            } else if (name == "IsingXX") {
                block->rxx(static_cast<uint32_t>(ws[0]),
                           static_cast<uint32_t>(ws[1]),
                           Param(get_param(op, 0)));

            } else if (name == "IsingYY") {
                block->ryy(static_cast<uint32_t>(ws[0]),
                           static_cast<uint32_t>(ws[1]),
                           Param(get_param(op, 0)));

            } else if (name == "ControlledQubitUnitary") {
                // PennyLane has no qml.CU gate; the emitter builds these as a
                // controlled arbitrary unitary. Recover (θ,φ,λ,γ) from the
                // base operator's 2x2 matrix.
                std::vector<int> ctrl_ws;
                for (auto w : op.attr("control_wires")) ctrl_ws.push_back(nb::cast<int>(w));
                if (ctrl_ws.size() != 1) {
                    throw std::runtime_error(
                        "PennylaneAbsorber: ControlledQubitUnitary with != 1 "
                        "control wire not supported");
                }
                nb::object mat = op.attr("base").attr("matrix")();
                auto elem = [&](int r, int c) {
                    return nb::cast<std::complex<double>>(mat.attr("item")(r, c));
                };
                auto [theta, phi, lambda_, gamma] = decompose_u3_with_phase(
                    elem(0, 0), elem(0, 1), elem(1, 0), elem(1, 1));
                block->cu(static_cast<uint32_t>(ctrl_ws[0]),
                          static_cast<uint32_t>(ws.back()),
                          Param(theta), Param(phi), Param(lambda_), Param(gamma));

            } else if (name == "Toffoli") {
                block->ccx(static_cast<uint32_t>(ws[0]),
                           static_cast<uint32_t>(ws[1]),
                           static_cast<uint32_t>(ws[2]));

            } else if (name == "CSWAP") {
                block->cswap(static_cast<uint32_t>(ws[0]),
                             static_cast<uint32_t>(ws[1]),
                             static_cast<uint32_t>(ws[2]));

            } else if (name == "GlobalPhase") {
                // qml.GlobalPhase(φ) = e^{-iφ}, QARPx gphase(θ) = e^{+iθ} (§7): negate.
                block->gphase(Param(-get_param(op, 0)));

            } else if (name == "MidMeasureMP" || name == "MidMeasure") {
                // Use the wire index as both qubit and cbit.
                uint32_t wire = static_cast<uint32_t>(ws[0]);
                block->measure(wire, wire);

            } else if (name == "Barrier" || name == "WireCut") {
                // Silently skip.

            } else {
                throw std::runtime_error(
                    "PennylaneAbsorber: unsupported operation: " + name);
            }
        }

        return nb::make_tuple(block->commands(), block->n_qubits, block->n_cbits);
    }
};

void register_pennylane_absorber(nb::module_& m) {
    nb::class_<PennylaneAbsorber>(m, "PennylaneAbsorber")
        .def(nb::init<>())
        .def("source_name", &PennylaneAbsorber::source_name)
        .def("absorb", &PennylaneAbsorber::absorb,
             "tape"_a,
             "Parse a qml.tape.QuantumScript into a QARPx SimpleBlock.");
}
