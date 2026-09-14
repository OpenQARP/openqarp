/// pytket_absorber.cpp
///
/// Converts a pytket.Circuit into a QARPx SimpleBlock.  Angles arrive in
/// half-turns and leave in radians; symbolic angles must be linear in one
/// sympy symbol (param_bridge).  pytket conditions are per-command
/// (`Conditional(op, width, value)`), so runs of identically-conditioned
/// commands fold into one BranchBegin … BranchEnd frame, and a run on the
/// complemented single-bit condition becomes the BranchElse arm — the exact
/// inverse of what PytketEmitter writes.  Barriers are kept.

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/intrusive/ref.h>
#include "qarpx/absorb/absorber.h"
#include "qarpx/core/command.h"
#include "qarpx/core/errors.h"
#include "qarpx/core/gates.h"
#include "qarpx/core/param.h"
#include "qarpx/block/block.h"
#include "qarpx/emit/param_bridge.h"
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using namespace qarpx;

namespace {

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Convert a TKET half-turn angle (possibly a sympy Expr) to a QARPx Param
/// in radians.  sign may be used to negate (kept for API symmetry).
Param tket_angle_to_qarp(nb::object t, double sign = 1.0) {
    constexpr double PI = 3.14159265358979323846;
    try {
        auto sympy = nb::module_::import_("sympy");
        if (nb::cast<bool>(nb::isinstance(t, sympy.attr("Expr")))) {
            // Check if the expression has free symbols.
            nb::object free = t.attr("free_symbols");
            if (nb::len(free) == 0) {
                // Concrete sympy number — evaluate it.
                return Param(sign * nb::cast<double>(t) * PI);
            }
            param_bridge::require_single_symbol(nb::len(free), "pytket");
            nb::list sym_list(free);
            nb::object sym = nb::cast<nb::object>(sym_list[0]);
            std::string sym_name = nb::cast<std::string>(nb::str(sym));
            // Probe the symbol through sympy's own .subs();
            // half-turns → radians via scale = π (sign folded in).
            return param_bridge::absorb_linear(sym_name, [&](double x) {
                return nb::cast<double>(t.attr("subs")(sym, nb::cast(x)));
            }, PI * sign, "pytket");
        }
    } catch (const capability_error&) {
        throw;
    } catch (...) {}
    // Concrete float / int from pytket.
    return Param(sign * nb::cast<double>(t) * PI);
}

/// Check two Python objects for equality using Python's == operator.
bool py_eq(nb::object a, nb::object b) {
    return nb::cast<bool>(a.attr("__eq__")(b));
}

/// Block::add_command is protected; branch markers have no builder method.
struct AbsorbBlock : SimpleBlock {
    using SimpleBlock::SimpleBlock;
    using Block::add_command;
};

// ── Condition folding ────────────────────────────────────────────────────────

/// The open run of identically-conditioned commands.
struct CondFrame {
    std::vector<uint32_t> bits;
    std::vector<bool>     vals;
    bool in_else = false;
};

void push_marker(AbsorbBlock& block, GateType g) {
    Command c;
    c.gate = g;
    block.add_command(std::move(c));
}

void close_frame(AbsorbBlock& block, std::optional<CondFrame>& frame) {
    if (!frame) return;
    push_marker(block, GateType::BranchEnd);
    frame.reset();
}

void open_frame(AbsorbBlock& block, std::optional<CondFrame>& frame,
                const std::vector<uint32_t>& bits, const std::vector<bool>& vals) {
    Command begin;
    begin.gate = GateType::BranchBegin;
    for (auto b : bits) begin.condition_bits.push_back(b);
    for (bool v : vals) begin.condition_values.push_back(v);
    block.add_command(std::move(begin));
    frame = CondFrame{bits, vals, false};
}

/// Route one command's condition: continue the open frame, turn it into the
/// else arm (single bit, complemented value, no else yet), or close it and
/// open a fresh one.  An unconditioned command closes any open frame.
void route_condition(AbsorbBlock& block, std::optional<CondFrame>& frame,
                     const std::vector<uint32_t>& bits,
                     const std::vector<bool>& vals) {
    if (bits.empty()) { close_frame(block, frame); return; }
    if (frame && frame->bits == bits) {
        if (frame->vals == vals) return;
        if (bits.size() == 1 && !frame->in_else) {
            push_marker(block, GateType::BranchElse);
            frame->in_else = true;
            frame->vals = vals;
            return;
        }
    }
    close_frame(block, frame);
    open_frame(block, frame, bits, vals);
}

// ── Gate dispatch ────────────────────────────────────────────────────────────

/// Append one (unwrapped) pytket op to `block`.  Conditions are the caller's
/// business; `op.params` are TKET half-turns.
void dispatch(nb::object op, const std::vector<uint32_t>& qs,
              const std::vector<uint32_t>& cs, AbsorbBlock& block,
              nb::object& OpType) {
    nb::object ot = op.attr("type");

    // `op.params` raises "Bad operation type" on param-less ops such as
    // Barrier, so it is read lazily by the arms that need it.
    auto rad = [&](int i, double sign = 1.0) -> Param {
        return tket_angle_to_qarp(
            nb::cast<nb::object>(op.attr("params")[nb::int_(i)]), sign);
    };

    if (py_eq(ot, OpType.attr("X"))) {
        block.x(qs[0]);

    } else if (py_eq(ot, OpType.attr("Y"))) {
        block.y(qs[0]);

    } else if (py_eq(ot, OpType.attr("Z"))) {
        block.z(qs[0]);

    } else if (py_eq(ot, OpType.attr("H"))) {
        block.h(qs[0]);

    } else if (py_eq(ot, OpType.attr("S"))) {
        block.s(qs[0]);

    } else if (py_eq(ot, OpType.attr("Sdg"))) {
        block.sdg(qs[0]);

    } else if (py_eq(ot, OpType.attr("T"))) {
        block.t(qs[0]);

    } else if (py_eq(ot, OpType.attr("Tdg"))) {
        block.tdg(qs[0]);

    } else if (py_eq(ot, OpType.attr("SX"))) {
        block.sx(qs[0]);

    } else if (py_eq(ot, OpType.attr("SXdg"))) {
        block.sxdg(qs[0]);

    // V = Rx(π/2) exactly (no e^{iπ/4}): not SX.
    } else if (py_eq(ot, OpType.attr("V"))) {
        block.rx(qs[0], Param(M_PI / 2));

    } else if (py_eq(ot, OpType.attr("Vdg"))) {
        block.rx(qs[0], Param(-M_PI / 2));

    } else if (py_eq(ot, OpType.attr("Rx"))) {
        block.rx(qs[0], rad(0));

    } else if (py_eq(ot, OpType.attr("Ry"))) {
        block.ry(qs[0], rad(0));

    } else if (py_eq(ot, OpType.attr("Rz"))) {
        block.rz(qs[0], rad(0));

    } else if (py_eq(ot, OpType.attr("U1"))) {
        block.p(qs[0], rad(0));

    } else if (py_eq(ot, OpType.attr("U3"))) {
        block.u(qs[0], rad(0), rad(1), rad(2));

    } else if (py_eq(ot, OpType.attr("CX"))) {
        block.cx(qs[0], qs[1]);

    } else if (py_eq(ot, OpType.attr("CH"))) {
        block.ch(qs[0], qs[1]);

    } else if (py_eq(ot, OpType.attr("CS"))) {
        block.cs(qs[0], qs[1]);

    } else if (py_eq(ot, OpType.attr("CSdg"))) {
        block.csdg(qs[0], qs[1]);

    } else if (py_eq(ot, OpType.attr("CSX"))) {
        block.csx(qs[0], qs[1]);

    } else if (py_eq(ot, OpType.attr("CSXdg"))) {
        block.csxdg(qs[0], qs[1]);

    } else if (py_eq(ot, OpType.attr("CY"))) {
        block.cy(qs[0], qs[1]);

    } else if (py_eq(ot, OpType.attr("CZ"))) {
        block.cz(qs[0], qs[1]);

    } else if (py_eq(ot, OpType.attr("SWAP"))) {
        block.swap(qs[0], qs[1]);

    } else if (py_eq(ot, OpType.attr("ECR"))) {
        block.ecr(qs[0], qs[1]);

    } else if (py_eq(ot, OpType.attr("ISWAP"))) {
        // ISWAP(1.0) = iSWAP, ISWAP(3.0) = iSWAPdg
        double t_val = 1.0;
        try {
            t_val = nb::cast<double>(
                nb::cast<nb::object>(op.attr("params")[nb::int_(0)]));
        } catch (...) {}
        if (std::abs(t_val - 3.0) < 0.1) {
            block.iswapdg(qs[0], qs[1]);
        } else {
            block.iswap(qs[0], qs[1]);
        }

    } else if (py_eq(ot, OpType.attr("ISWAPMax"))) {
        block.iswap(qs[0], qs[1]);

    } else if (py_eq(ot, OpType.attr("CRx"))) {
        block.crx(qs[0], qs[1], rad(0));

    } else if (py_eq(ot, OpType.attr("CRy"))) {
        block.cry(qs[0], qs[1], rad(0));

    } else if (py_eq(ot, OpType.attr("CRz"))) {
        block.crz(qs[0], qs[1], rad(0));

    } else if (py_eq(ot, OpType.attr("CU1"))) {
        block.cp(qs[0], qs[1], rad(0));

    } else if (py_eq(ot, OpType.attr("ZZPhase"))) {
        // ZZPhase(t) = exp(-i·t·π/2·ZZ) = RZZ(t·π), t = +θ/π
        block.rzz(qs[0], qs[1], rad(0));

    } else if (py_eq(ot, OpType.attr("XXPhase"))) {
        block.rxx(qs[0], qs[1], rad(0));

    } else if (py_eq(ot, OpType.attr("YYPhase"))) {
        block.ryy(qs[0], qs[1], rad(0));

    } else if (py_eq(ot, OpType.attr("CU3"))) {
        block.cu(qs[0], qs[1], rad(0), rad(1), rad(2), Param(0.0));

    } else if (py_eq(ot, OpType.attr("CCX"))) {
        block.ccx(qs[0], qs[1], qs[2]);

    } else if (py_eq(ot, OpType.attr("CSWAP"))) {
        block.cswap(qs[0], qs[1], qs[2]);

    } else if (py_eq(ot, OpType.attr("CnZ"))) {
        block.mcz(std::vector<uint32_t>(qs.begin(), qs.end()));

    } else if (py_eq(ot, OpType.attr("Measure"))) {
        uint32_t cbit = cs.empty() ? qs[0] : cs[0];
        block.measure(qs[0], cbit);

    } else if (py_eq(ot, OpType.attr("Reset"))) {
        block.reset(qs[0]);

    } else if (py_eq(ot, OpType.attr("Barrier"))) {
        block.barrier(qs);

    } else if (py_eq(ot, OpType.attr("noop"))) {
        block.id(qs[0]);  // explicit identity (§2.6)

    } else if (py_eq(ot, OpType.attr("Phase"))) {
        // Global phase gate.
        int n_params = 0;
        try {
            nb::object params = op.attr("params");
            n_params = static_cast<int>(nb::len(params));
        } catch (...) {}
        if (n_params > 0) {
            block.gphase(rad(0));
        }

    } else if (py_eq(ot, OpType.attr("Conditional"))) {
        throw capability_error(
            "PytketAbsorber: nested Conditional ops are not supported");

    } else {
        // Try to get a human-readable name for the error message.
        std::string type_name = "unknown";
        try {
            type_name = nb::cast<std::string>(nb::str(ot));
        } catch (...) {}
        throw std::runtime_error(
            "PytketAbsorber: unrecognized OpType: " + type_name);
    }
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────

class PytketAbsorber : public Absorber {
public:
    [[nodiscard]] std::string source_name() const override { return "pytket"; }

    nb::object absorb(nb::object circuit) const {
        uint32_t n_qubits = nb::cast<uint32_t>(circuit.attr("n_qubits"));
        ref<AbsorbBlock> block(new AbsorbBlock(n_qubits));

        nb::object pytket_circuit_mod;
        try {
            pytket_circuit_mod = nb::module_::import_("pytket.circuit");
        } catch (...) {
            throw sdk_missing_error(
                "PytketAbsorber: pytket not installed. "
                "Install with: pip install pytket");
        }
        nb::object OpType = pytket_circuit_mod.attr("OpType");

        std::optional<CondFrame> frame;
        for (auto cmd_handle : circuit.attr("get_commands")()) {
            nb::object cmd = nb::cast<nb::object>(cmd_handle);
            nb::object op  = cmd.attr("op");

            // q.index[0] for q in cmd.qubits — the inner op's qubits.
            std::vector<uint32_t> qs;
            for (auto q : cmd.attr("qubits")) {
                nb::object q_obj = nb::cast<nb::object>(q);
                qs.push_back(nb::cast<uint32_t>(
                    q_obj.attr("index")[nb::int_(0)]));
            }

            // c.index[0] for c in cmd.bits — the inner op's own bits; a
            // Conditional's condition bits are not listed here.
            std::vector<uint32_t> cs;
            for (auto c : cmd.attr("bits")) {
                nb::object c_obj = nb::cast<nb::object>(c);
                cs.push_back(nb::cast<uint32_t>(
                    c_obj.attr("index")[nb::int_(0)]));
            }

            // Conditional(op, width, value): cmd.args lists the `width`
            // condition bits first; `value` is a little-endian mask over
            // them (bit i ↔ bits[i]) — the encoding PytketEmitter writes.
            std::vector<uint32_t> cond_bits;
            std::vector<bool>     cond_vals;
            if (py_eq(op.attr("type"), OpType.attr("Conditional"))) {
                const int width = nb::cast<int>(op.attr("width"));
                const int value = nb::cast<int>(op.attr("value"));
                nb::list args(cmd.attr("args"));
                for (int i = 0; i < width; ++i) {
                    nb::object b = nb::cast<nb::object>(args[i]);
                    cond_bits.push_back(nb::cast<uint32_t>(
                        b.attr("index")[nb::int_(0)]));
                    cond_vals.push_back(((value >> i) & 1) != 0);
                }
                op = op.attr("op");
            }
            route_condition(*block, frame, cond_bits, cond_vals);
            dispatch(op, qs, cs, *block, OpType);
        }
        close_frame(*block, frame);

        // pytket carries a circuit's global phase on the `.phase` attribute (in
        // half-turns), not as a command — recover it (QARPx gphase is radians, §7).
        double phase_ht = nb::cast<double>(circuit.attr("phase"));
        if (std::abs(phase_ht) > 1e-12) {
            constexpr double PI = 3.14159265358979323846;
            block->gphase(Param(phase_ht * PI));
        }

        return nb::make_tuple(block->commands(), block->n_qubits, block->n_cbits);
    }
};

void register_pytket_absorber(nb::module_& m) {
    nb::class_<PytketAbsorber>(m, "PytketAbsorber")
        .def(nb::init<>())
        .def("source_name", &PytketAbsorber::source_name)
        .def("absorb", &PytketAbsorber::absorb,
             "circuit"_a,
             "Parse a pytket.Circuit into a QARPx SimpleBlock.");
}
