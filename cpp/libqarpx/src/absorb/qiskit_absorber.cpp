/// qiskit_absorber.cpp
///
/// Converts a qiskit.QuantumCircuit Python object into a QARPx SimpleBlock.
///
/// Gate dispatch is by op.name (string), which is stable across Qiskit versions.
/// Symbolic parameters (qiskit.circuit.Parameter / ParameterExpression) are
/// converted to QARPx Param::symbol / Param::linear; anything not linear in
/// one symbol raises capability_error (param_bridge).
///
/// `circuit.global_phase` becomes a leading GPhase.  `IfElseOp` with a
/// single-bit condition becomes BranchBegin/Else/End the way QASM3Absorber
/// writes them; a wider ClassicalRegister condition or a classical
/// expression raises capability_error.  Barriers are kept.

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
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using namespace qarpx;

namespace {

constexpr double PI = 3.14159265358979323846;

/// Block::add_command is protected; branch markers have no builder method.
struct AbsorbBlock : SimpleBlock {
    using SimpleBlock::SimpleBlock;
    using Block::add_command;
};

// ── Parameter conversion ─────────────────────────────────────────────────────

/// Convert a Qiskit parameter value (float, Parameter, or ParameterExpression)
/// to a QARPx Param.  float_fn is builtins.float for concrete evaluation.
Param qiskit_param_to_qarp(nb::object p, nb::object& float_fn) {
    // Try direct numeric cast first
    try { return Param(nb::cast<double>(p)); } catch (...) {}

    // Try Python's float() conversion (handles concrete ParameterExpression)
    try { return Param(nb::cast<double>(float_fn(p))); } catch (...) {}

    // Symbolic: inspect .parameters to find free symbols
    nb::object params_set = p.attr("parameters");
    nb::list   params_list(params_set);
    size_t n_free = nb::len(params_list);

    if (n_free == 0) {
        throw std::runtime_error(
            "QiskitAbsorber: ParameterExpression has no free symbols "
            "but cannot be evaluated as float");
    }
    param_bridge::require_single_symbol(n_free, "qiskit");

    nb::object sym  = nb::cast<nb::object>(params_list[0]);
    std::string name = nb::cast<std::string>(sym.attr("name"));

    // Probe the symbol through qiskit's own .bind().
    return param_bridge::absorb_linear(name, [&](double x) {
        nb::dict d; d[sym] = nb::cast(x);
        return nb::cast<double>(float_fn(p.attr("bind")(d)));
    }, 1.0, "qiskit");
}

// ── Recursive circuit walk ───────────────────────────────────────────────────

/// Append `circuit`'s instructions to `block`.  `qmap` / `cmap` translate the
/// circuit's own bit indices to the top-level block's: an IfElseOp body is a
/// circuit over only the bits the op touches, positionally aligned with the
/// enclosing instruction's `qubits` / `clbits`.
void absorb_into(nb::object circuit, AbsorbBlock& block,
                 const std::vector<uint32_t>& qmap,
                 const std::vector<uint32_t>& cmap,
                 nb::object& float_fn)
{
    // A body circuit may carry its own phase — inside a branch it is a
    // conditional phase, which GPhase-inside-the-frame expresses exactly.
    {
        Param gp = qiskit_param_to_qarp(circuit.attr("global_phase"), float_fn);
        if (gp.is_symbolic() || std::abs(gp.value()) > 1e-12) block.gphase(gp);
    }

    for (auto inst_handle : circuit.attr("data")) {
        nb::object inst  = nb::cast<nb::object>(inst_handle);
        nb::object op    = inst.attr("operation");
        std::string gname = nb::cast<std::string>(op.attr("name"));

        std::vector<uint32_t> qs;
        for (auto q_h : inst.attr("qubits")) {
            nb::object q = nb::cast<nb::object>(q_h);
            qs.push_back(qmap.at(nb::cast<uint32_t>(
                circuit.attr("find_bit")(q).attr("index"))));
        }
        std::vector<uint32_t> cs;
        for (auto c_h : inst.attr("clbits")) {
            nb::object c = nb::cast<nb::object>(c_h);
            cs.push_back(cmap.at(nb::cast<uint32_t>(
                circuit.attr("find_bit")(c).attr("index"))));
        }

        nb::object params = op.attr("params");
        auto p = [&](int i) -> Param {
            return qiskit_param_to_qarp(
                nb::cast<nb::object>(params[nb::int_(i)]), float_fn);
        };

        // ── 1Q no-param ──────────────────────────────────────────────────
        if      (gname == "x")   block.x(qs[0]);
        else if (gname == "y")   block.y(qs[0]);
        else if (gname == "z")   block.z(qs[0]);
        else if (gname == "h")   block.h(qs[0]);
        else if (gname == "s")   block.s(qs[0]);
        else if (gname == "sdg") block.sdg(qs[0]);
        else if (gname == "t")   block.t(qs[0]);
        else if (gname == "tdg") block.tdg(qs[0]);
        else if (gname == "sx")   block.sx(qs[0]);
        else if (gname == "sxdg") block.sxdg(qs[0]);
        else if (gname == "id")   block.id(qs[0]);

        // ── 1Q parametric ─────────────────────────────────────────────────
        else if (gname == "rx") block.rx(qs[0], p(0));
        else if (gname == "ry") block.ry(qs[0], p(0));
        else if (gname == "rz") block.rz(qs[0], p(0));
        else if (gname == "p")  block.p(qs[0], p(0));
        else if (gname == "u")  block.u(qs[0], p(0), p(1), p(2));
        // Legacy OpenQASM 2 singles: u1(λ) = P(λ); u2(φ,λ) = U(π/2,φ,λ);
        // u3 = U; r(θ,φ) = U(θ, φ−π/2, π/2−φ) — all exact, no phase.
        else if (gname == "u1") block.p(qs[0], p(0));
        else if (gname == "u2") block.u(qs[0], Param(PI / 2), p(0), p(1));
        else if (gname == "u3") block.u(qs[0], p(0), p(1), p(2));
        else if (gname == "r") {
            Param phi = p(1);
            block.u(qs[0], p(0), phi - Param(PI / 2), Param(PI / 2) - phi);
        }

        // ── 2Q no-param ───────────────────────────────────────────────────
        else if (gname == "cx" || gname == "cnot")
            block.cx(qs[0], qs[1]);
        else if (gname == "cy")   block.cy(qs[0], qs[1]);
        else if (gname == "cz")   block.cz(qs[0], qs[1]);
        else if (gname == "swap") block.swap(qs[0], qs[1]);
        else if (gname == "ecr")  block.ecr(qs[0], qs[1]);
        else if (gname == "iswap") block.iswap(qs[0], qs[1]);
        else if (gname == "iswap_dg" || gname == "iswapdg")
            block.iswapdg(qs[0], qs[1]);

        // ── 2Q parametric ─────────────────────────────────────────────────
        else if (gname == "ch")    block.ch(qs[0], qs[1]);
        else if (gname == "cs")    block.cs(qs[0], qs[1]);
        else if (gname == "csdg")  block.csdg(qs[0], qs[1]);
        else if (gname == "csx")   block.csx(qs[0], qs[1]);
        else if (gname == "csxdg") block.csxdg(qs[0], qs[1]);
        else if (gname == "crx") block.crx(qs[0], qs[1], p(0));
        else if (gname == "cry") block.cry(qs[0], qs[1], p(0));
        else if (gname == "crz") block.crz(qs[0], qs[1], p(0));
        else if (gname == "cp")  block.cp(qs[0], qs[1], p(0));
        else if (gname == "rzz") block.rzz(qs[0], qs[1], p(0));
        else if (gname == "rxx") block.rxx(qs[0], qs[1], p(0));
        else if (gname == "ryy") block.ryy(qs[0], qs[1], p(0));
        else if (gname == "cu")  block.cu(qs[0], qs[1], p(0), p(1), p(2), p(3));

        // ── 3Q no-param ───────────────────────────────────────────────────
        else if (gname == "ccx")   block.ccx(qs[0], qs[1], qs[2]);
        else if (gname == "cswap") block.cswap(qs[0], qs[1], qs[2]);
        else if (gname == "mcx") {
            // Qiskit MCX (≥3 controls; 2 controls is "ccx" above). QARPx has
            // no MCX gate — build it as H(target)·MCZ(all)·H(target). qiskit
            // lists qubits controls-first, target last.
            uint32_t target = qs.back();
            block.h(target);
            block.mcz(qs);
            block.h(target);
        }

        // ── Classical / special ───────────────────────────────────────────
        else if (gname == "measure") {
            uint32_t cbit = cs.empty() ? qs[0] : cs[0];
            block.measure(qs[0], cbit);
        }
        else if (gname == "reset") {
            block.reset(qs[0]);
        }
        else if (gname == "global_phase") {
            block.gphase(p(0));
        }
        else if (gname == "barrier") {
            block.barrier(qs);
        }
        else if (gname == "if_else") {
            // Condition: (Clbit | ClassicalRegister, int).  Only a single
            // bit is representable as a branch frame here; a wider register
            // compares an integer against several bits at once and a
            // classical `expr` has no command-stream form.
            nb::object cond = op.attr("condition");
            if (!nb::isinstance<nb::tuple>(cond)) {
                throw capability_error(
                    "QiskitAbsorber: IfElseOp conditions must be a (bit, value) "
                    "pair; classical expressions are not supported");
            }
            nb::object target = cond[nb::int_(0)];
            int value = nb::cast<int>(cond[nb::int_(1)]);
            if (nb::hasattr(target, "size")) {
                const int width = nb::cast<int>(target.attr("size"));
                if (width != 1) {
                    throw capability_error(
                        "QiskitAbsorber: IfElseOp on a " + std::to_string(width) +
                        "-bit ClassicalRegister is not supported; condition on a "
                        "single Clbit");
                }
                target = target[nb::int_(0)];
            }
            const uint32_t cbit = cmap.at(nb::cast<uint32_t>(
                circuit.attr("find_bit")(target).attr("index")));

            Command begin;
            begin.gate = GateType::BranchBegin;
            begin.condition_bits.push_back(cbit);
            begin.condition_values.push_back((value & 1) != 0);
            block.add_command(std::move(begin));

            nb::tuple blocks(op.attr("blocks"));
            const size_t n_bodies = nb::len(blocks);
            for (size_t bi = 0; bi < n_bodies; ++bi) {
                if (bi == 1) {
                    Command els;
                    els.gate = GateType::BranchElse;
                    block.add_command(std::move(els));
                }
                // Bodies are circuits over exactly the op's bits, in order.
                absorb_into(nb::cast<nb::object>(blocks[bi]), block, qs, cs, float_fn);
            }
            Command end;
            end.gate = GateType::BranchEnd;
            block.add_command(std::move(end));
        }
        else {
            throw std::runtime_error(
                "QiskitAbsorber: unrecognized gate '" + gname + "'");
        }
    }
}

}  // namespace

// ── QiskitAbsorber ───────────────────────────────────────────────────────────

class QiskitAbsorber : public Absorber {
public:
    [[nodiscard]] std::string source_name() const override { return "qiskit"; }

    nb::object absorb(nb::object circuit) const {
        try {
            nb::module_::import_("qiskit");
        } catch (...) {
            throw sdk_missing_error(
                "QiskitAbsorber: Qiskit not installed. "
                "Install with: pip install qiskit>=1.0");
        }

        nb::object float_fn = nb::module_::import_("builtins").attr("float");

        uint32_t n_qubits = nb::cast<uint32_t>(circuit.attr("num_qubits"));
        uint32_t n_clbits = nb::cast<uint32_t>(circuit.attr("num_clbits"));
        ref<AbsorbBlock> block(new AbsorbBlock(n_qubits));

        std::vector<uint32_t> qmap(n_qubits), cmap(n_clbits);
        for (uint32_t i = 0; i < n_qubits; ++i) qmap[i] = i;
        for (uint32_t i = 0; i < n_clbits; ++i) cmap[i] = i;
        absorb_into(circuit, *block, qmap, cmap, float_fn);

        return nb::make_tuple(block->commands(), block->n_qubits, block->n_cbits);
    }
};

void register_qiskit_absorber(nb::module_& m) {
    nb::class_<QiskitAbsorber>(m, "QiskitAbsorber")
        .def(nb::init<>())
        .def("source_name", &QiskitAbsorber::source_name)
        .def("absorb", &QiskitAbsorber::absorb,
             "circuit"_a,
             "Parse a qiskit.QuantumCircuit into (commands, n_qubits, n_cbits).");
}
