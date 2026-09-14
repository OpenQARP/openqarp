/// qulacs_absorber.cpp
///
/// Converts a qulacs.QuantumCircuit Python object back into a QARPx
/// SimpleBlock (built and ready to flatten).
///
/// Angle convention:
///   Qulacs stores rotations as exp(+iθ/2 P).  When we read back a stored
///   angle `stored`, we must negate it to recover the QARPx angle θ:
///     block.rx(q, Param(-stored))
///   For Y-rotation we read the angle differently (via the real off-diagonal)
///   but the sign convention still applies.

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/complex.h>
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
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using namespace qarpx;

// ── Matrix helper functions (same definitions as in the emitter) ─────────────

static nb::object abs_u3_matrix(double theta, double phi, double lambda_,
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

static nb::object abs_rx_matrix(double theta, nb::object& np) {
    double c = std::cos(theta / 2.0);
    double s = std::sin(theta / 2.0);
    using cd = std::complex<double>;
    return np.attr("array")(
        nb::make_tuple(
            nb::make_tuple(nb::cast(cd(c, 0.0)),  nb::cast(cd(0.0, -s))),
            nb::make_tuple(nb::cast(cd(0.0, -s)), nb::cast(cd(c, 0.0)))),
        "dtype"_a = np.attr("complex128"));
}

static nb::object abs_ry_matrix(double theta, nb::object& np) {
    double c = std::cos(theta / 2.0);
    double s = std::sin(theta / 2.0);
    using cd = std::complex<double>;
    return np.attr("array")(
        nb::make_tuple(
            nb::make_tuple(nb::cast(cd(c, 0.0)),  nb::cast(cd(-s, 0.0))),
            nb::make_tuple(nb::cast(cd(s, 0.0)),  nb::cast(cd(c, 0.0)))),
        "dtype"_a = np.attr("complex128"));
}

static nb::object abs_rz_matrix(double theta, nb::object& np) {
    using cd = std::complex<double>;
    cd e_neg = cd(std::cos(theta / 2.0), -std::sin(theta / 2.0));
    cd e_pos = cd(std::cos(theta / 2.0),  std::sin(theta / 2.0));
    return np.attr("array")(
        nb::make_tuple(
            nb::make_tuple(nb::cast(e_neg), nb::cast(cd(0.0, 0.0))),
            nb::make_tuple(nb::cast(cd(0.0, 0.0)), nb::cast(e_pos))),
        "dtype"_a = np.attr("complex128"));
}

static nb::object abs_y_matrix(nb::object& np) {
    using cd = std::complex<double>;
    return np.attr("array")(
        nb::make_tuple(
            nb::make_tuple(nb::cast(cd(0.0, 0.0)), nb::cast(cd(0.0, -1.0))),
            nb::make_tuple(nb::cast(cd(0.0, 1.0)), nb::cast(cd(0.0, 0.0)))),
        "dtype"_a = np.attr("complex128"));
}

static nb::object abs_controlled_gate(nb::object u, nb::object& np) {
    auto mat = np.attr("eye")(4, "dtype"_a = np.attr("complex128"));
    nb::object sl = nb::module_::import_("builtins").attr("slice")(2, 4);
    mat.attr("__setitem__")(nb::make_tuple(sl, sl), u);
    return mat;
}

static nb::object abs_iswap_matrix(nb::object& np) {
    using cd = std::complex<double>;
    return np.attr("array")(
        nb::make_tuple(
            nb::make_tuple(nb::cast(cd(1,0)), nb::cast(cd(0,0)), nb::cast(cd(0,0)), nb::cast(cd(0,0))),
            nb::make_tuple(nb::cast(cd(0,0)), nb::cast(cd(0,0)), nb::cast(cd(0,1)), nb::cast(cd(0,0))),
            nb::make_tuple(nb::cast(cd(0,0)), nb::cast(cd(0,1)), nb::cast(cd(0,0)), nb::cast(cd(0,0))),
            nb::make_tuple(nb::cast(cd(0,0)), nb::cast(cd(0,0)), nb::cast(cd(0,0)), nb::cast(cd(1,0)))),
        "dtype"_a = np.attr("complex128"));
}

static nb::object abs_ecr_matrix(nb::object& np) {
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

static nb::object abs_rzz_matrix(double theta, nb::object& np) {
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

static nb::object abs_rxx_matrix(double theta, nb::object& np) {
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

static nb::object abs_ryy_matrix(double theta, nb::object& np) {
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

static nb::object abs_toffoli_matrix(nb::object& np) {
    auto mat = np.attr("eye")(8, "dtype"_a = np.attr("complex128"));
    using cd = std::complex<double>;
    mat.attr("__setitem__")(nb::make_tuple(6, 6), nb::cast(cd(0.0, 0.0)));
    mat.attr("__setitem__")(nb::make_tuple(7, 7), nb::cast(cd(0.0, 0.0)));
    mat.attr("__setitem__")(nb::make_tuple(6, 7), nb::cast(cd(1.0, 0.0)));
    mat.attr("__setitem__")(nb::make_tuple(7, 6), nb::cast(cd(1.0, 0.0)));
    return mat;
}

static nb::object abs_cswap_matrix(nb::object& np) {
    auto mat = np.attr("eye")(8, "dtype"_a = np.attr("complex128"));
    using cd = std::complex<double>;
    mat.attr("__setitem__")(nb::make_tuple(5, 5), nb::cast(cd(0.0, 0.0)));
    mat.attr("__setitem__")(nb::make_tuple(6, 6), nb::cast(cd(0.0, 0.0)));
    mat.attr("__setitem__")(nb::make_tuple(5, 6), nb::cast(cd(1.0, 0.0)));
    mat.attr("__setitem__")(nb::make_tuple(6, 5), nb::cast(cd(1.0, 0.0)));
    return mat;
}

// ── Utility: extract complex element from numpy array ──────────────────────

static std::complex<double> mat_elem(nb::object& mat, int row, int col) {
    return nb::cast<std::complex<double>>(mat.attr("item")(row, col));
}

/// Decompose an arbitrary 2x2 unitary matrix M into QARPx's (θ,φ,λ,γ) such
/// that M == e^{iγ} · U3(θ,φ,λ), with U3 = [[cos(θ/2), -e^{iλ}sin(θ/2)],
/// [e^{iφ}sin(θ/2), e^{i(φ+λ)}cos(θ/2)]]. γ is taken as the phase of M[0][0]
/// (θ ≈ π, where cos(θ/2) ≈ 0, is not hit by any angle this codebase emits).
static std::array<double, 4> decompose_u3_with_phase(
        const std::complex<double>& m00, const std::complex<double>& m01,
        const std::complex<double>& m10, const std::complex<double>& m11) {
    using cd = std::complex<double>;
    cd anchor = std::abs(m00) > 1e-9 ? m00 : m10;
    cd phase  = anchor / std::abs(anchor);
    double gamma = std::atan2(phase.imag(), phase.real());
    cd a = m00 / phase, b = m01 / phase, c = m10 / phase, d = m11 / phase;
    (void)d;
    double theta = 2.0 * std::acos(std::clamp(a.real(), -1.0, 1.0));
    double phi = 0.0, lambda_ = 0.0;
    if (std::abs(c) > 1e-9 || std::abs(b) > 1e-9) {
        phi     = std::atan2(c.imag(), c.real());
        lambda_ = std::atan2((-b).imag(), (-b).real());
    }
    return {theta, phi, lambda_, gamma};
}

// ── QulacsAbsorber ───────────────────────────────────────────────────────────

class QulacsAbsorber : public Absorber {
public:
    [[nodiscard]] std::string source_name() const override { return "qulacs"; }

    nb::object absorb(nb::object circuit) const {
        auto np = nb::module_::import_("numpy");

        uint32_t n_qubits = nb::cast<uint32_t>(circuit.attr("get_qubit_count")());
        int gate_count    = nb::cast<int>(circuit.attr("get_gate_count")());

        nb::ref<SimpleBlock> block(new SimpleBlock(n_qubits));

        for (int i = 0; i < gate_count; ++i) {
            auto gate = circuit.attr("get_gate")(i);
            std::string name = nb::cast<std::string>(gate.attr("get_name")());

            // Collect target and control qubit index lists
            auto target_obj = gate.attr("get_target_index_list")();
            auto ctrl_obj   = gate.attr("get_control_index_list")();

            std::vector<uint32_t> target_qubits;
            std::vector<uint32_t> ctrl_qubits;
            for (auto h : target_obj) target_qubits.push_back(nb::cast<uint32_t>(h));
            for (auto h : ctrl_obj)   ctrl_qubits.push_back(nb::cast<uint32_t>(h));

            // ── Named gate dispatch ──────────────────────────────────────────
            if (name == "X") {
                block->x(target_qubits[0]);
            } else if (name == "Y") {
                block->y(target_qubits[0]);
            } else if (name == "Z") {
                block->z(target_qubits[0]);
            } else if (name == "H") {
                block->h(target_qubits[0]);
            } else if (name == "S") {
                block->s(target_qubits[0]);
            } else if (name == "Sdag") {
                block->sdg(target_qubits[0]);
            } else if (name == "T") {
                block->t(target_qubits[0]);
            } else if (name == "Tdag") {
                block->tdg(target_qubits[0]);
            } else if (name == "sqrtX") {
                block->sx(target_qubits[0]);
            } else if (name == "sqrtXdag") {
                block->sxdg(target_qubits[0]);
            } else if (name == "I") {
                block->id(target_qubits[0]);
            } else if (name == "CNOT") {
                uint32_t ctrl   = ctrl_qubits.empty() ? target_qubits[0] : ctrl_qubits[0];
                uint32_t tgt    = target_qubits.back();
                block->cx(ctrl, tgt);
            } else if (name == "CZ") {
                // Like CNOT, add_CZ_gate(control, target) splits the qubits
                // across target_index_list (1 elem) and control_index_list
                // (1 elem) — target_qubits[1] would read out of bounds.
                uint32_t ctrl = ctrl_qubits.empty() ? target_qubits[0] : ctrl_qubits[0];
                uint32_t tgt  = target_qubits.back();
                block->cz(ctrl, tgt);
            } else if (name == "SWAP") {
                block->swap(target_qubits[0], target_qubits[1]);

            // ── Rotation gates: recover angle from matrix ──────────────────
            } else if (name == "X-rotation") {
                // Qulacs RX(alpha) = exp(+i*alpha/2*X): mat[0,1] = i*sin(alpha/2)
                // Emitter sends alpha = -theta, so mat[0,1].imag = -sin(theta/2)
                // theta = 2*asin(-imag(mat[0,1]))
                nb::object mat = gate.attr("get_matrix")();
                auto elem01 = mat_elem(mat, 0, 1);
                double theta = 2.0 * std::asin(-elem01.imag());
                block->rx(target_qubits[0], Param(theta));
            } else if (name == "Y-rotation") {
                // Qulacs RY(alpha) = exp(+i*alpha/2*Y): mat[1,0] = -sin(alpha/2)
                // Emitter sends alpha = -theta, so mat[1,0].real = sin(theta/2)
                // theta = 2*asin(real(mat[1,0]))
                nb::object mat = gate.attr("get_matrix")();
                auto elem10 = mat_elem(mat, 1, 0);
                double theta = 2.0 * std::asin(elem10.real());
                block->ry(target_qubits[0], Param(theta));
            } else if (name == "Z-rotation") {
                // Qulacs RZ(alpha) = exp(+i*alpha/2*Z): mat[1,1] = e^{-i*alpha/2}
                // Emitter sends alpha = -theta, so mat[1,1] = e^{+i*theta/2}
                // theta = 2*arg(mat[1,1])
                nb::object mat = gate.attr("get_matrix")();
                auto elem11 = mat_elem(mat, 1, 1);
                double theta = 2.0 * std::atan2(elem11.imag(), elem11.real());
                block->rz(target_qubits[0], Param(theta));

            } else if (name == "Adaptive") {
                throw std::runtime_error(
                    "QulacsAbsorber: Adaptive gates not supported");

            // ── DenseMatrix gate: identify by matrix comparison ────────────
            } else if (name == "DenseMatrix" || name == "") {
                // Combine target + ctrl into the full qubit list
                std::vector<uint32_t> all_qs;
                all_qs.insert(all_qs.end(), target_qubits.begin(), target_qubits.end());
                all_qs.insert(all_qs.end(), ctrl_qubits.begin(), ctrl_qubits.end());

                auto mat = gate.attr("get_matrix")();
                int n_gate_qubits = static_cast<int>(all_qs.size());

                if (n_gate_qubits == 1) {
                    // ── 1-qubit DenseMatrix ────────────────────────────────
                    uint32_t q = all_qs[0];
                    auto m00 = mat_elem(mat, 0, 0);
                    auto m01 = mat_elem(mat, 0, 1);
                    auto m10 = mat_elem(mat, 1, 0);
                    auto m11 = mat_elem(mat, 1, 1);

                    // P(θ): [[1,0],[0,e^{iθ}]]
                    if (std::abs(m00 - std::complex<double>(1,0)) < 1e-9 &&
                        std::abs(m01) < 1e-9 &&
                        std::abs(m10) < 1e-9) {
                        double theta = std::atan2(m11.imag(), m11.real());
                        block->p(q, Param(theta));
                    } else {
                        // General single-qubit gate: recover U(θ,φ,λ) + global
                        // phase γ (same helper as the 2-qubit CU fallback) rather
                        // than rejecting it.
                        auto [theta, phi, lambda_, gamma] =
                            decompose_u3_with_phase(m00, m01, m10, m11);
                        block->u(q, Param(theta), Param(phi), Param(lambda_));
                        if (std::abs(gamma) > 1e-9) {
                            block->gphase(Param(gamma));
                        }
                    }

                } else if (n_gate_qubits == 2) {
                    // ── 2-qubit DenseMatrix ────────────────────────────────
                    uint32_t q0 = all_qs[0];
                    uint32_t q1 = all_qs[1];

                    auto allclose = [&](nb::object& a, nb::object& b) -> bool {
                        return nb::cast<bool>(np.attr("allclose")(a, b,
                            "atol"_a = 1e-9));
                    };

                    // CY: controlled-Y. The ref puts the control on the local MSB
                    // (qubit_list[1] = q1), target on the LSB (q0) — so cy(control,
                    // target) = cy(q1, q0). ECR is written MSB-first likewise.
                    {
                        auto ref = abs_controlled_gate(abs_y_matrix(np), np);
                        if (allclose(mat, ref)) { block->cy(q1, q0); goto next_gate; }
                    }
                    // iSWAP
                    {
                        auto ref = abs_iswap_matrix(np);
                        if (allclose(mat, ref)) { block->iswap(q0, q1); goto next_gate; }
                    }
                    // iSWAPdg
                    {
                        nb::object ref = abs_iswap_matrix(np).attr("conj")().attr("T");
                        if (allclose(mat, ref)) { block->iswapdg(q0, q1); goto next_gate; }
                    }
                    // ECR (non-symmetric; MSB-first matrix → ecr(q1, q0))
                    {
                        auto ref = abs_ecr_matrix(np);
                        if (allclose(mat, ref)) { block->ecr(q1, q0); goto next_gate; }
                    }
                    // CP(θ): diagonal, mat[3,3] = e^{iθ}, mat[0..2,0..2] == I for those diag
                    {
                        auto m00e = mat_elem(mat, 0, 0);
                        auto m11e = mat_elem(mat, 1, 1);
                        auto m22e = mat_elem(mat, 2, 2);
                        auto m33e = mat_elem(mat, 3, 3);
                        // Check if it looks diagonal and first three are ~1
                        bool is_diag_like =
                            std::abs(m00e - std::complex<double>(1,0)) < 1e-9 &&
                            std::abs(m11e - std::complex<double>(1,0)) < 1e-9 &&
                            std::abs(m22e - std::complex<double>(1,0)) < 1e-9 &&
                            std::abs(mat_elem(mat, 0, 1)) < 1e-9 &&
                            std::abs(mat_elem(mat, 0, 2)) < 1e-9 &&
                            std::abs(mat_elem(mat, 0, 3)) < 1e-9 &&
                            std::abs(mat_elem(mat, 1, 0)) < 1e-9 &&
                            std::abs(mat_elem(mat, 1, 2)) < 1e-9 &&
                            std::abs(mat_elem(mat, 1, 3)) < 1e-9 &&
                            std::abs(mat_elem(mat, 2, 0)) < 1e-9 &&
                            std::abs(mat_elem(mat, 2, 1)) < 1e-9 &&
                            std::abs(mat_elem(mat, 2, 3)) < 1e-9 &&
                            std::abs(mat_elem(mat, 3, 0)) < 1e-9 &&
                            std::abs(mat_elem(mat, 3, 1)) < 1e-9 &&
                            std::abs(mat_elem(mat, 3, 2)) < 1e-9;
                        if (is_diag_like) {
                            double theta_cp = std::atan2(m33e.imag(), m33e.real());
                            block->cp(q0, q1, Param(theta_cp));
                            goto next_gate;
                        }
                    }
                    // RZZ(θ): diagonal — e^{-iθ/2}, e^{+iθ/2}, e^{+iθ/2}, e^{-iθ/2}
                    {
                        auto m00e = mat_elem(mat, 0, 0);
                        auto m11e = mat_elem(mat, 1, 1);
                        // Check diagonal pattern: m11 = e^{+iθ/2} => θ = 2*arg(m11)
                        // and verify against full rzz matrix
                        double theta_rzz = -2.0 * std::atan2(m00e.imag(), m00e.real());
                        auto ref = abs_rzz_matrix(theta_rzz, np);
                        if (allclose(mat, ref)) {
                            block->rzz(q0, q1, Param(theta_rzz));
                            goto next_gate;
                        }
                        (void)m11e; // suppress unused warning
                    }
                    // RXX(θ): try to recover θ from |mat[0,0]|
                    {
                        auto m00e = mat_elem(mat, 0, 0);
                        double theta_pos = 2.0 * std::acos(std::abs(m00e.real()));
                        for (double th : {theta_pos, -theta_pos}) {
                            auto ref = abs_rxx_matrix(th, np);
                            if (allclose(mat, ref)) {
                                block->rxx(q0, q1, Param(th));
                                goto next_gate;
                            }
                        }
                    }
                    // RYY(θ): similar
                    {
                        auto m00e = mat_elem(mat, 0, 0);
                        double theta_pos = 2.0 * std::acos(std::abs(m00e.real()));
                        for (double th : {theta_pos, -theta_pos}) {
                            auto ref = abs_ryy_matrix(th, np);
                            if (allclose(mat, ref)) {
                                block->ryy(q0, q1, Param(th));
                                goto next_gate;
                            }
                        }
                    }
                    // CRx/CRy/CRz — controlled single-qubit rotations
                    // Try to read off-diagonal block (rows/cols 2,3) and match
                    {
                        // Read 2x2 submatrix at [2:4, 2:4]
                        auto s00 = mat_elem(mat, 2, 2);
                        auto s01 = mat_elem(mat, 2, 3);
                        auto s10 = mat_elem(mat, 3, 2);
                        auto s11 = mat_elem(mat, 3, 3);
                        // Check top-left 2x2 is identity
                        bool top_id =
                            std::abs(mat_elem(mat, 0,0) - std::complex<double>(1,0)) < 1e-9 &&
                            std::abs(mat_elem(mat, 0,1)) < 1e-9 &&
                            std::abs(mat_elem(mat, 1,0)) < 1e-9 &&
                            std::abs(mat_elem(mat, 1,1) - std::complex<double>(1,0)) < 1e-9;
                        // And off-block is zero
                        bool off_zero =
                            std::abs(mat_elem(mat, 0,2)) < 1e-9 &&
                            std::abs(mat_elem(mat, 0,3)) < 1e-9 &&
                            std::abs(mat_elem(mat, 1,2)) < 1e-9 &&
                            std::abs(mat_elem(mat, 1,3)) < 1e-9 &&
                            std::abs(mat_elem(mat, 2,0)) < 1e-9 &&
                            std::abs(mat_elem(mat, 2,1)) < 1e-9 &&
                            std::abs(mat_elem(mat, 3,0)) < 1e-9 &&
                            std::abs(mat_elem(mat, 3,1)) < 1e-9;
                        if (top_id && off_zero) {
                            // sub = [[s00,s01],[s10,s11]] — identify the 1Q gate
                            // Rx: s00=c, s01=-is, s10=-is, s11=c  (real diag, imag off-diag)
                            bool rx_like =
                                std::abs(s00.imag()) < 1e-9 &&
                                std::abs(s11.imag()) < 1e-9 &&
                                std::abs(s01.real()) < 1e-9 &&
                                std::abs(s10.real()) < 1e-9 &&
                                std::abs(s01 - s10) < 1e-9;
                            if (rx_like) {
                                double c = s00.real();
                                double s = -s01.imag();  // -i*s => s = -imag
                                double th = 2.0 * std::atan2(s, c);
                                // Emitter passes qubit_list = {target, control} (LSB-first),
                                // so all_qs[0]=target=q0, all_qs[1]=control=q1.
                                block->crx(q1, q0, Param(th));
                                goto next_gate;
                            }
                            // Ry: s00=c, s01=-s, s10=s, s11=c  (all real)
                            bool ry_like =
                                std::abs(s00.imag()) < 1e-9 &&
                                std::abs(s11.imag()) < 1e-9 &&
                                std::abs(s01.imag()) < 1e-9 &&
                                std::abs(s10.imag()) < 1e-9 &&
                                std::abs(s10 + s01) < 1e-9;  // s10 = -s01
                            if (ry_like) {
                                double c = s00.real();
                                double s = s10.real();
                                double th = 2.0 * std::atan2(s, c);
                                block->cry(q1, q0, Param(th));
                                goto next_gate;
                            }
                            // Rz: diagonal 2x2
                            bool rz_like =
                                std::abs(s01) < 1e-9 &&
                                std::abs(s10) < 1e-9;
                            if (rz_like) {
                                // s11 = e^{+iθ/2} => θ = 2*arg(s11)
                                double th = 2.0 * std::atan2(s11.imag(), s11.real());
                                block->crz(q1, q0, Param(th));
                                goto next_gate;
                            }
                            // Fallback: general controlled-U (covers CU(θ,φ,λ,γ)
                            // with γ≠0, which none of the Rx/Ry/Rz patterns match).
                            auto [theta, phi, lambda_, gamma] =
                                decompose_u3_with_phase(s00, s01, s10, s11);
                            block->cu(q1, q0, Param(theta), Param(phi),
                                      Param(lambda_), Param(gamma));
                            goto next_gate;
                        }
                    }
                    throw std::runtime_error(
                        "QulacsAbsorber: unrecognized 2-qubit DenseMatrix gate");

                } else if (n_gate_qubits == 3) {
                    // ── 3-qubit DenseMatrix ────────────────────────────────
                    uint32_t q0 = all_qs[0];
                    uint32_t q1 = all_qs[1];
                    uint32_t q2 = all_qs[2];

                    auto allclose = [&](nb::object& a, nb::object& b) -> bool {
                        return nb::cast<bool>(np.attr("allclose")(a, b,
                            "atol"_a = 1e-9));
                    };

                    {
                        // Emitter passes qubit_list = {target, c2, c1} (LSB-first):
                        // all_qs[0]=target=q0, all_qs[1]=c2=q1, all_qs[2]=c1=q2.
                        auto ref = abs_toffoli_matrix(np);
                        if (allclose(mat, ref)) { block->ccx(q2, q1, q0); goto next_gate; }
                    }
                    {
                        auto ref = abs_cswap_matrix(np);
                        if (allclose(mat, ref)) { block->cswap(q2, q1, q0); goto next_gate; }
                    }
                    throw std::runtime_error(
                        "QulacsAbsorber: unrecognized 3-qubit DenseMatrix gate");

                } else {
                    throw std::runtime_error(
                        "QulacsAbsorber: DenseMatrix gate with more than 3 qubits "
                        "not supported");
                }

            } else if (name == "CPTP") {
                // Qulacs has no get_*() accessor for the classical register
                // address baked into a Measurement gate — it's only exposed
                // via to_json(). Parse that to recover the actual cbit index
                // (do NOT assume cbit == qubit).
                auto json_mod = nb::module_::import_("json");
                nb::object info = json_mod.attr("loads")(gate.attr("to_json")());
                // is_instrument / classical_register_address are JSON strings
                // (qulacs serializes them as "true" / "5", not native bool/int).
                nb::object is_instr = info.attr("get")("is_instrument", nb::str(""));
                if (nb::cast<std::string>(nb::str(is_instr)) != "true") {
                    throw std::runtime_error(
                        "QulacsAbsorber: unrecognized CPTP gate (not a Measurement)");
                }
                nb::object addr = info.attr("get")("classical_register_address");
                uint32_t cbit = static_cast<uint32_t>(
                    std::stoul(nb::cast<std::string>(nb::str(addr))));
                block->measure(target_qubits[0], cbit);
            } else {
                // Unknown gate name — skip with a warning embedded in a runtime error
                throw std::runtime_error(
                    std::string("QulacsAbsorber: unknown gate name: ") + name);
            }

            next_gate:;
        }

        return nb::make_tuple(block->commands(), block->n_qubits, block->n_cbits);
    }
};

void register_qulacs_absorber(nb::module_& m) {
    nb::class_<QulacsAbsorber>(m, "QulacsAbsorber")
        .def(nb::init<>())
        .def("source_name", &QulacsAbsorber::source_name)
        .def("absorb", &QulacsAbsorber::absorb,
             "circuit"_a,
             "Convert a qulacs.QuantumCircuit to a QARPx SimpleBlock.");
}
