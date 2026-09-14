/// qulacs_emitter.cpp
///
/// Converts a QARPx flat command sequence into a qulacs.QuantumCircuit
/// Python object via nanobind.
///
/// Angle convention:
///   QARPx uses exp(-iθ/2 P); qulacs uses exp(+iθ/2 P).
///   All rotation angles are negated on emission: add_RX_gate(q, -θ).

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

namespace nb = nanobind;
using namespace nb::literals;
using namespace qarpx;

// ── Matrix helper functions ──────────────────────────────────────────────────

static nb::object u3_matrix(double theta, double phi, double lambda_,
                             nb::object& np) {
    double c = std::cos(theta / 2.0);
    double s = std::sin(theta / 2.0);
    using cd = std::complex<double>;
    cd e_il  = cd(std::cos(lambda_),  std::sin(lambda_));
    cd e_ip  = cd(std::cos(phi),      std::sin(phi));
    cd e_ipl = cd(std::cos(phi + lambda_), std::sin(phi + lambda_));
    return np.attr("array")(
        nb::make_tuple(
            nb::make_tuple(nb::cast(cd(c, 0.0)), nb::cast(-e_il * s)),
            nb::make_tuple(nb::cast(e_ip * s),   nb::cast(e_ipl * c))),
        "dtype"_a = np.attr("complex128"));
}

static nb::object rx_matrix(double theta, nb::object& np) {
    double c = std::cos(theta / 2.0);
    double s = std::sin(theta / 2.0);
    using cd = std::complex<double>;
    return np.attr("array")(
        nb::make_tuple(
            nb::make_tuple(nb::cast(cd(c, 0.0)),  nb::cast(cd(0.0, -s))),
            nb::make_tuple(nb::cast(cd(0.0, -s)), nb::cast(cd(c, 0.0)))),
        "dtype"_a = np.attr("complex128"));
}

static nb::object ry_matrix(double theta, nb::object& np) {
    double c = std::cos(theta / 2.0);
    double s = std::sin(theta / 2.0);
    using cd = std::complex<double>;
    return np.attr("array")(
        nb::make_tuple(
            nb::make_tuple(nb::cast(cd(c, 0.0)),  nb::cast(cd(-s, 0.0))),
            nb::make_tuple(nb::cast(cd(s, 0.0)),  nb::cast(cd(c, 0.0)))),
        "dtype"_a = np.attr("complex128"));
}

static nb::object rz_matrix(double theta, nb::object& np) {
    using cd = std::complex<double>;
    cd e_neg = cd(std::cos(theta / 2.0), -std::sin(theta / 2.0));
    cd e_pos = cd(std::cos(theta / 2.0),  std::sin(theta / 2.0));
    return np.attr("array")(
        nb::make_tuple(
            nb::make_tuple(nb::cast(e_neg), nb::cast(cd(0.0, 0.0))),
            nb::make_tuple(nb::cast(cd(0.0, 0.0)), nb::cast(e_pos))),
        "dtype"_a = np.attr("complex128"));
}

static nb::object y_matrix(nb::object& np) {
    using cd = std::complex<double>;
    return np.attr("array")(
        nb::make_tuple(
            nb::make_tuple(nb::cast(cd(0.0, 0.0)), nb::cast(cd(0.0, -1.0))),
            nb::make_tuple(nb::cast(cd(0.0, 1.0)), nb::cast(cd(0.0, 0.0)))),
        "dtype"_a = np.attr("complex128"));
}

// §2.2 / §2.6 constant 2×2 matrices for the controlled Clifford singles.
static nb::object h_matrix(nb::object& np) {
    using cd = std::complex<double>;
    return np.attr("array")(
        nb::make_tuple(
            nb::make_tuple(nb::cast(cd(M_SQRT1_2, 0.0)), nb::cast(cd(M_SQRT1_2, 0.0))),
            nb::make_tuple(nb::cast(cd(M_SQRT1_2, 0.0)), nb::cast(cd(-M_SQRT1_2, 0.0)))),
        "dtype"_a = np.attr("complex128"));
}

static nb::object s_matrix(nb::object& np) {
    using cd = std::complex<double>;
    return np.attr("array")(
        nb::make_tuple(
            nb::make_tuple(nb::cast(cd(1.0, 0.0)), nb::cast(cd(0.0, 0.0))),
            nb::make_tuple(nb::cast(cd(0.0, 0.0)), nb::cast(cd(0.0, 1.0)))),
        "dtype"_a = np.attr("complex128"));
}

static nb::object sdg_matrix(nb::object& np) {
    using cd = std::complex<double>;
    return np.attr("array")(
        nb::make_tuple(
            nb::make_tuple(nb::cast(cd(1.0, 0.0)), nb::cast(cd(0.0, 0.0))),
            nb::make_tuple(nb::cast(cd(0.0, 0.0)), nb::cast(cd(0.0, -1.0)))),
        "dtype"_a = np.attr("complex128"));
}

static nb::object sx_matrix(nb::object& np) {
    using cd = std::complex<double>;
    return np.attr("array")(
        nb::make_tuple(
            nb::make_tuple(nb::cast(cd(0.5, 0.5)), nb::cast(cd(0.5, -0.5))),
            nb::make_tuple(nb::cast(cd(0.5, -0.5)), nb::cast(cd(0.5, 0.5)))),
        "dtype"_a = np.attr("complex128"));
}

static nb::object sxdg_matrix(nb::object& np) {
    using cd = std::complex<double>;
    return np.attr("array")(
        nb::make_tuple(
            nb::make_tuple(nb::cast(cd(0.5, -0.5)), nb::cast(cd(0.5, 0.5))),
            nb::make_tuple(nb::cast(cd(0.5, 0.5)), nb::cast(cd(0.5, -0.5)))),
        "dtype"_a = np.attr("complex128"));
}

/// 4×4 controlled-U: eye(4) with bottom-right 2×2 replaced by u.
static nb::object controlled_gate(nb::object u, nb::object& np) {
    auto mat = np.attr("eye")(4, "dtype"_a = np.attr("complex128"));
    nb::object sl = nb::module_::import_("builtins").attr("slice")(2, 4);
    mat.attr("__setitem__")(nb::make_tuple(sl, sl), u);
    return mat;
}

static nb::object iswap_matrix(nb::object& np) {
    using cd = std::complex<double>;
    return np.attr("array")(
        nb::make_tuple(
            nb::make_tuple(nb::cast(cd(1,0)), nb::cast(cd(0,0)), nb::cast(cd(0,0)), nb::cast(cd(0,0))),
            nb::make_tuple(nb::cast(cd(0,0)), nb::cast(cd(0,0)), nb::cast(cd(0,1)), nb::cast(cd(0,0))),
            nb::make_tuple(nb::cast(cd(0,0)), nb::cast(cd(0,1)), nb::cast(cd(0,0)), nb::cast(cd(0,0))),
            nb::make_tuple(nb::cast(cd(0,0)), nb::cast(cd(0,0)), nb::cast(cd(0,0)), nb::cast(cd(1,0)))),
        "dtype"_a = np.attr("complex128"));
}

static nb::object ecr_matrix(nb::object& np) {
    using cd = std::complex<double>;
    double s = 1.0 / std::sqrt(2.0);
    return np.attr("array")(
        nb::make_tuple(
            nb::make_tuple(nb::cast(cd(0,0)),   nb::cast(cd(0,0)),   nb::cast(cd(s,0)),  nb::cast(cd(0,s))),
            nb::make_tuple(nb::cast(cd(0,0)),   nb::cast(cd(0,0)),   nb::cast(cd(0,s)),  nb::cast(cd(s,0))),
            nb::make_tuple(nb::cast(cd(s,0)),   nb::cast(cd(0,-s)),  nb::cast(cd(0,0)),  nb::cast(cd(0,0))),
            nb::make_tuple(nb::cast(cd(0,-s)),  nb::cast(cd(s,0)),   nb::cast(cd(0,0)),  nb::cast(cd(0,0)))),
        "dtype"_a = np.attr("complex128"));
}

static nb::object rzz_matrix(double theta, nb::object& np) {
    using cd = std::complex<double>;
    cd e_neg = cd(std::cos(theta / 2.0), -std::sin(theta / 2.0));
    cd e_pos = cd(std::cos(theta / 2.0),  std::sin(theta / 2.0));
    cd z     = cd(0.0, 0.0);
    return np.attr("array")(
        nb::make_tuple(
            nb::make_tuple(nb::cast(e_neg), nb::cast(z),     nb::cast(z),     nb::cast(z)),
            nb::make_tuple(nb::cast(z),     nb::cast(e_pos), nb::cast(z),     nb::cast(z)),
            nb::make_tuple(nb::cast(z),     nb::cast(z),     nb::cast(e_pos), nb::cast(z)),
            nb::make_tuple(nb::cast(z),     nb::cast(z),     nb::cast(z),     nb::cast(e_neg))),
        "dtype"_a = np.attr("complex128"));
}

static nb::object rxx_matrix(double theta, nb::object& np) {
    using cd = std::complex<double>;
    double c = std::cos(theta / 2.0);
    double s = std::sin(theta / 2.0);
    cd z = cd(0.0, 0.0);
    return np.attr("array")(
        nb::make_tuple(
            nb::make_tuple(nb::cast(cd(c,0)),    nb::cast(z),         nb::cast(z),         nb::cast(cd(0,-s))),
            nb::make_tuple(nb::cast(z),          nb::cast(cd(c,0)),   nb::cast(cd(0,-s)),  nb::cast(z)),
            nb::make_tuple(nb::cast(z),          nb::cast(cd(0,-s)),  nb::cast(cd(c,0)),   nb::cast(z)),
            nb::make_tuple(nb::cast(cd(0,-s)),   nb::cast(z),         nb::cast(z),         nb::cast(cd(c,0)))),
        "dtype"_a = np.attr("complex128"));
}

static nb::object ryy_matrix(double theta, nb::object& np) {
    using cd = std::complex<double>;
    double c = std::cos(theta / 2.0);
    double s = std::sin(theta / 2.0);
    cd z = cd(0.0, 0.0);
    return np.attr("array")(
        nb::make_tuple(
            nb::make_tuple(nb::cast(cd(c,0)),   nb::cast(z),          nb::cast(z),         nb::cast(cd(0,s))),
            nb::make_tuple(nb::cast(z),         nb::cast(cd(c,0)),    nb::cast(cd(0,-s)),  nb::cast(z)),
            nb::make_tuple(nb::cast(z),         nb::cast(cd(0,-s)),   nb::cast(cd(c,0)),   nb::cast(z)),
            nb::make_tuple(nb::cast(cd(0,s)),   nb::cast(z),          nb::cast(z),         nb::cast(cd(c,0)))),
        "dtype"_a = np.attr("complex128"));
}

/// 8×8 Toffoli: eye(8) with [6,7]×[6,7] swapped.
static nb::object toffoli_matrix(nb::object& np) {
    auto mat = np.attr("eye")(8, "dtype"_a = np.attr("complex128"));
    // Swap rows/cols 6 and 7
    using cd = std::complex<double>;
    mat.attr("__setitem__")(nb::make_tuple(6, 6), nb::cast(cd(0.0, 0.0)));
    mat.attr("__setitem__")(nb::make_tuple(7, 7), nb::cast(cd(0.0, 0.0)));
    mat.attr("__setitem__")(nb::make_tuple(6, 7), nb::cast(cd(1.0, 0.0)));
    mat.attr("__setitem__")(nb::make_tuple(7, 6), nb::cast(cd(1.0, 0.0)));
    return mat;
}

/// 8×8 CSWAP (Fredkin): eye(8) with [5,6]×[5,6] swapped.
static nb::object cswap_matrix(nb::object& np) {
    auto mat = np.attr("eye")(8, "dtype"_a = np.attr("complex128"));
    using cd = std::complex<double>;
    mat.attr("__setitem__")(nb::make_tuple(5, 5), nb::cast(cd(0.0, 0.0)));
    mat.attr("__setitem__")(nb::make_tuple(6, 6), nb::cast(cd(0.0, 0.0)));
    mat.attr("__setitem__")(nb::make_tuple(5, 6), nb::cast(cd(1.0, 0.0)));
    mat.attr("__setitem__")(nb::make_tuple(6, 5), nb::cast(cd(1.0, 0.0)));
    return mat;
}

// ── QulacsEmitter ────────────────────────────────────────────────────────────

class QulacsEmitter : public Emitter {
public:
    [[nodiscard]] std::string target_name() const override { return "qulacs"; }

    [[nodiscard]] GateSet gate_set() const override {
        return qulacs_emitter_gateset();
    }

    [[nodiscard]] EmitterCapabilities capabilities() const override {
        return EmitterCapabilities{
            .symbolic_params     = false,
            .multi_symbol_params = false,
            .distinct_cbits      = true,   // qulacs.gate.Measurement(q, c)
            .conditionals        = false,
            .multibit_conditions = false,
            .multibit_else       = false,
            .custom_unitary      = false,
            .cu_global_phase     = true,   // dense matrix carries γ
        };
    }

    nb::object emit(const std::vector<Command>& commands,
                    uint32_t n_qubits,
                    const std::string& /*circuit_name*/ = "circuit") const {
        throw_if_invalid(commands);

        // ── Import qulacs ────────────────────────────────────────────────────
        nb::object qulacs;
        try {
            qulacs = nb::module_::import_("qulacs");
        } catch (const nb::python_error&) {
            throw sdk_missing_error(
                "QulacsEmitter: qulacs not installed. "
                "Install it with: pip install qulacs");
        }

        // ── Import numpy ─────────────────────────────────────────────────────
        auto np = nb::module_::import_("numpy");
        auto qulacs_gate = nb::module_::import_("qulacs.gate");

        // ── Build circuit ────────────────────────────────────────────────────
        auto circuit = qulacs.attr("QuantumCircuit")(n_qubits);

        // ── Gate dispatch ────────────────────────────────────────────────────
        for (const auto& cmd : commands) {
            const auto& qs = cmd.qubits;

            // Helper: build a Python list of qubit indices
            auto make_qlist = [&](std::initializer_list<uint32_t> idxs) {
                auto lst = nb::list();
                for (auto q : idxs) lst.append(nb::cast(q));
                return lst;
            };

            switch (cmd.gate) {
                // ── 1Q no-param ──
                case GateType::X:
                    circuit.attr("add_X_gate")(qs[0]);
                    break;
                case GateType::Y:
                    circuit.attr("add_Y_gate")(qs[0]);
                    break;
                case GateType::Z:
                    circuit.attr("add_Z_gate")(qs[0]);
                    break;
                case GateType::H:
                    circuit.attr("add_H_gate")(qs[0]);
                    break;
                case GateType::S:
                    circuit.attr("add_S_gate")(qs[0]);
                    break;
                case GateType::Sdg:
                    circuit.attr("add_Sdag_gate")(qs[0]);
                    break;
                case GateType::T:
                    circuit.attr("add_T_gate")(qs[0]);
                    break;
                case GateType::Tdg:
                    circuit.attr("add_Tdag_gate")(qs[0]);
                    break;
                // qulacs' sqrtX is ½[[1+i,1-i],[1-i,1+i]] = SX (§2.6).
                case GateType::SX:
                    circuit.attr("add_sqrtX_gate")(qs[0]);
                    break;
                case GateType::SXdg:
                    circuit.attr("add_sqrtXdag_gate")(qs[0]);
                    break;
                case GateType::Id:
                    circuit.attr("add_gate")(qulacs_gate.attr("Identity")(qs[0]));
                    break;

                // ── 1Q parametric (negate angles for convention conversion) ──
                case GateType::Rx:
                    circuit.attr("add_RX_gate")(qs[0], -cmd.params[0].value());
                    break;
                case GateType::Ry:
                    circuit.attr("add_RY_gate")(qs[0], -cmd.params[0].value());
                    break;
                case GateType::Rz:
                    circuit.attr("add_RZ_gate")(qs[0], -cmd.params[0].value());
                    break;

                // ── P(θ) = [[1,0],[0,e^{iθ}]] ──
                case GateType::P: {
                    double theta = cmd.params[0].value();
                    using cd = std::complex<double>;
                    auto mat = np.attr("array")(
                        nb::make_tuple(
                            nb::make_tuple(nb::cast(cd(1,0)), nb::cast(cd(0,0))),
                            nb::make_tuple(nb::cast(cd(0,0)),
                                           nb::cast(cd(std::cos(theta), std::sin(theta))))),
                        "dtype"_a = np.attr("complex128"));
                    circuit.attr("add_dense_matrix_gate")(make_qlist({qs[0]}), mat);
                    break;
                }

                // ── U(θ,φ,λ) ──
                case GateType::U: {
                    double theta   = cmd.params[0].value();
                    double phi     = cmd.params[1].value();
                    double lambda_ = cmd.params[2].value();
                    auto mat = u3_matrix(theta, phi, lambda_, np);
                    circuit.attr("add_dense_matrix_gate")(make_qlist({qs[0]}), mat);
                    break;
                }

                // ── 2Q no-param ──
                case GateType::CX:
                    circuit.attr("add_CNOT_gate")(qs[0], qs[1]);
                    break;
                case GateType::CZ:
                    circuit.attr("add_CZ_gate")(qs[0], qs[1]);
                    break;
                case GateType::SWAP:
                    circuit.attr("add_SWAP_gate")(qs[0], qs[1]);
                    break;

                case GateType::CY: {
                    auto mat = controlled_gate(y_matrix(np), np);
                    // controlled_gate() puts the control on the local MSB; qulacs
                    // treats qubit_list[0] as the LSB, so pass control (qs[0]) last
                    // (mirrors CRx/CRy/CRz/CU below). ECR's matrix is likewise
                    // written MSB-first and needs the same reversal.
                    circuit.attr("add_dense_matrix_gate")(make_qlist({qs[1], qs[0]}), mat);
                    break;
                }
                case GateType::ECR: {
                    auto mat = ecr_matrix(np);
                    circuit.attr("add_dense_matrix_gate")(make_qlist({qs[1], qs[0]}), mat);
                    break;
                }
                case GateType::iSWAP: {
                    auto mat = iswap_matrix(np);
                    circuit.attr("add_dense_matrix_gate")(make_qlist({qs[0], qs[1]}), mat);
                    break;
                }
                case GateType::iSWAPdg: {
                    // iSWAP is symmetric, so its dagger is the plain conjugate.
                    // (A numpy `.T` view here is non-contiguous and segfaults
                    // qulacs' add_dense_matrix_gate.)
                    auto mat = iswap_matrix(np).attr("conj")();
                    circuit.attr("add_dense_matrix_gate")(make_qlist({qs[0], qs[1]}), mat);
                    break;
                }
                // Controlled Clifford singles: same dense path as CRx (control passed last).
                case GateType::CH: {
                    auto mat = controlled_gate(h_matrix(np), np);
                    circuit.attr("add_dense_matrix_gate")(make_qlist({qs[1], qs[0]}), mat);
                    break;
                }
                case GateType::CS: {
                    auto mat = controlled_gate(s_matrix(np), np);
                    circuit.attr("add_dense_matrix_gate")(make_qlist({qs[1], qs[0]}), mat);
                    break;
                }
                case GateType::CSdg: {
                    auto mat = controlled_gate(sdg_matrix(np), np);
                    circuit.attr("add_dense_matrix_gate")(make_qlist({qs[1], qs[0]}), mat);
                    break;
                }
                case GateType::CSX: {
                    auto mat = controlled_gate(sx_matrix(np), np);
                    circuit.attr("add_dense_matrix_gate")(make_qlist({qs[1], qs[0]}), mat);
                    break;
                }
                case GateType::CSXdg: {
                    auto mat = controlled_gate(sxdg_matrix(np), np);
                    circuit.attr("add_dense_matrix_gate")(make_qlist({qs[1], qs[0]}), mat);
                    break;
                }

                // ── 2Q parametric ──
                case GateType::CP: {
                    double theta = cmd.params[0].value();
                    using cd = std::complex<double>;
                    cd z(0.0, 0.0);
                    auto mat = np.attr("array")(
                        nb::make_tuple(
                            nb::make_tuple(nb::cast(cd(1,0)), nb::cast(z),     nb::cast(z),     nb::cast(z)),
                            nb::make_tuple(nb::cast(z),       nb::cast(cd(1,0)), nb::cast(z),   nb::cast(z)),
                            nb::make_tuple(nb::cast(z),       nb::cast(z),     nb::cast(cd(1,0)), nb::cast(z)),
                            nb::make_tuple(nb::cast(z),       nb::cast(z),     nb::cast(z),
                                           nb::cast(cd(std::cos(theta), std::sin(theta))))),
                        "dtype"_a = np.attr("complex128"));
                    circuit.attr("add_dense_matrix_gate")(make_qlist({qs[0], qs[1]}), mat);
                    break;
                }
                case GateType::CRx: {
                    double theta = cmd.params[0].value();
                    auto mat = controlled_gate(rx_matrix(theta, np), np);
                    // controlled_gate() puts U in the upper half (local MSB = control);
                    // qulacs' add_dense_matrix_gate treats qubit_list[0] as the LSB, so
                    // the control must be passed last to land on the MSB.
                    circuit.attr("add_dense_matrix_gate")(make_qlist({qs[1], qs[0]}), mat);
                    break;
                }
                case GateType::CRy: {
                    double theta = cmd.params[0].value();
                    auto mat = controlled_gate(ry_matrix(theta, np), np);
                    circuit.attr("add_dense_matrix_gate")(make_qlist({qs[1], qs[0]}), mat);
                    break;
                }
                case GateType::CRz: {
                    double theta = cmd.params[0].value();
                    auto mat = controlled_gate(rz_matrix(theta, np), np);
                    circuit.attr("add_dense_matrix_gate")(make_qlist({qs[1], qs[0]}), mat);
                    break;
                }
                case GateType::RZZ: {
                    double theta = cmd.params[0].value();
                    auto mat = rzz_matrix(theta, np);
                    circuit.attr("add_dense_matrix_gate")(make_qlist({qs[0], qs[1]}), mat);
                    break;
                }
                case GateType::RXX: {
                    double theta = cmd.params[0].value();
                    auto mat = rxx_matrix(theta, np);
                    circuit.attr("add_dense_matrix_gate")(make_qlist({qs[0], qs[1]}), mat);
                    break;
                }
                case GateType::RYY: {
                    double theta = cmd.params[0].value();
                    auto mat = ryy_matrix(theta, np);
                    circuit.attr("add_dense_matrix_gate")(make_qlist({qs[0], qs[1]}), mat);
                    break;
                }

                // ── CU(θ,φ,λ,γ) ──
                case GateType::CU: {
                    double theta   = cmd.params[0].value();
                    double phi     = cmd.params[1].value();
                    double lambda_ = cmd.params[2].value();
                    double gamma   = cmd.params[3].value();
                    using cd = std::complex<double>;
                    cd phase = cd(std::cos(gamma), std::sin(gamma));
                    auto u3  = u3_matrix(theta, phi, lambda_, np);
                    auto u3_phased = np.attr("multiply")(nb::cast(phase), u3);
                    auto mat = controlled_gate(u3_phased, np);
                    circuit.attr("add_dense_matrix_gate")(make_qlist({qs[1], qs[0]}), mat);
                    break;
                }

                // ── 3Q ──
                case GateType::CCX: {
                    // toffoli_matrix() swaps indices 6,7 (both controls=1); with
                    // qubit_list[0]=LSB, that requires qs[0],qs[1] (controls) on
                    // the high bits and qs[2] (target) on the LSB.
                    auto mat = toffoli_matrix(np);
                    circuit.attr("add_dense_matrix_gate")(make_qlist({qs[2], qs[1], qs[0]}), mat);
                    break;
                }
                case GateType::CSWAP: {
                    auto mat = cswap_matrix(np);
                    circuit.attr("add_dense_matrix_gate")(make_qlist({qs[2], qs[1], qs[0]}), mat);
                    break;
                }

                // MCZ / Reset / Custom are rejected by validate() and land in
                // the default arm's defensive throw if they ever slip through.
                case GateType::Measure: {
                    // QuantumCircuit has no add_measurement(); the gate must be
                    // constructed via qulacs.gate.Measurement(qubit, cbit) and
                    // added like any other gate.
                    uint32_t cbit = cmd.cbits.empty() ? qs[0] : cmd.cbits[0];
                    circuit.attr("add_gate")(qulacs_gate.attr("Measurement")(qs[0], cbit));
                    break;
                }

                // ── Silently skipped ──
                case GateType::GPhase:
                case GateType::Barrier:
                case GateType::BranchElse:
                case GateType::BranchEnd:
                    break;

                default:
                    throw std::runtime_error(
                        std::string("QulacsEmitter: unhandled gate type: ") +
                        std::string(gate_name(cmd.gate)));
            }
        }

        return circuit;
    }
};

void register_qulacs_emitter(nb::module_& m) {
    nb::class_<QulacsEmitter>(m, "QulacsEmitter")
        .def(nb::init<>())
        .def("target_name", &QulacsEmitter::target_name)
        .def("gate_set", &QulacsEmitter::gate_set)
        .def("capabilities", &QulacsEmitter::capabilities)
        .def("validate",
             [](const QulacsEmitter& self, const std::vector<Command>& cmds) {
                 return self.validate(cmds);
             },
             "commands"_a,
             "First Incompatibility in the sequence, or None if the whole "
             "sequence can cross.  Needs no SDK installed.")
        .def("emit", &QulacsEmitter::emit,
             "commands"_a, "n_qubits"_a, "circuit_name"_a = "circuit",
             "Convert a QARPx command sequence to a qulacs.QuantumCircuit.");
}
