#include <nanobind/nanobind.h>
#include "qarpx/parallel/cpu_budget.h"
#include "qarpx/parallel/thread_pool.h"
#include <nanobind/operators.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/eigen/dense.h>
// Pulls in the ref<T> type caster (active because it follows nanobind.h) so
// functions can take/return ref<Block>, plus nb::intrusive_init's declaration.
#include <nanobind/intrusive/ref.h>

#include "qarpx/qarpx.h"
#include "qarpx/block/measure_block.h"
#include "qarpx/block/reset_block.h"
#include "qarpx/block/conditional_block.h"
#include "qarpx/core/errors.h"
#include "qarpx/emit/emitter.h"
#include "qarpx/synthesis/pauli_exponential.h"

#include <tuple>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;
using namespace qarpx;

// Forward declarations for SDK emitter / absorber registration functions.
void register_pennylane_emitter(nb::module_& m);
void register_pennylane_absorber(nb::module_& m);
void register_pytket_emitter(nb::module_& m);
void register_pytket_absorber(nb::module_& m);
void register_qulacs_emitter(nb::module_& m);
void register_qulacs_absorber(nb::module_& m);
void register_qiskit_emitter(nb::module_& m);
void register_qiskit_absorber(nb::module_& m);

// Defined in bindings_operators.cpp — the openfermion-compatible
// QubitOperator/FermionOperator classes and their free functions.
namespace qarpx::python {
void register_operator_bindings(nb::module_& m);
}

// Move a statevector into an owning 1-D complex128 numpy array (zero copy:
// the array views the vector's buffer; a capsule deletes it with the array).
// Without this the stl caster converts 2^n amplitudes to a Python list —
// measured at ~40% of the simulation time itself at 20 qubits.
static nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<1>>
statevector_to_numpy(std::vector<std::complex<double>>&& v) {
    auto* heap = new std::vector<std::complex<double>>(std::move(v));
    nb::capsule owner(heap, [](void* p) noexcept {
        delete static_cast<std::vector<std::complex<double>>*>(p);
    });
    const size_t shape[1] = {heap->size()};
    return nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<1>>(
        heap->data(), 1, shape, owner);
}

// Caller-supplied initial state: a 1-D contiguous complex128 ndarray, taken
// via nb::ndarray rather than the stl caster — sequence conversion would cost
// O(2^n) Python calls per injection.
using InitialStateArray =
    nb::ndarray<const std::complex<double>, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

static std::optional<std::vector<std::complex<double>>>
initial_state_to_vector(const std::optional<InitialStateArray>& arr) {
    if (!arr) return std::nullopt;
    const auto* p = arr->data();
    return std::vector<std::complex<double>>(p, p + arr->shape(0));
}

// __deepcopy__ for raw qarpx blocks, defined here (not monkeypatched from
// qarp.blocks) so `copy.deepcopy` works on C++-created blocks regardless of
// Python import order.  Children/inner/bodies recurse through Python's
// copy.deepcopy so memo aliasing survives; qarp's mixin classes attach their
// own __deepcopy__ directly, which shadows this one.
static nb::object block_py_deepcopy(nb::handle self, nb::dict memo) {
    nb::object py_deepcopy = nb::module_::import_("copy").attr("deepcopy");
    nb::object qx = nb::module_::import_("qarpx");
    nb::object cls = nb::borrow<nb::object>(self.type());
    // C++-internal kinds (TransformedBlock) surface as ctor-less qx.Block —
    // a built SimpleBlock carrying the same state is behaviourally identical.
    if (cls.is(qx.attr("Block")))
        cls = qx.attr("SimpleBlock");
    nb::object result = cls.attr("__new__")(cls);
    memo[nb::cast(reinterpret_cast<std::uintptr_t>(self.ptr()))] = result;

    nb::object src = nb::borrow<nb::object>(self);
    if (nb::isinstance(self, qx.attr("CompositeBlock"))) {
        nb::list children;
        for (nb::handle c : src.attr("children")())
            children.append(py_deepcopy(c, memo));
        qx.attr("CompositeBlock").attr("__init__")(
            result, children, src.attr("n_qubits"), src.attr("name"));
    } else if (nb::isinstance(self, qx.attr("ControlledBlock"))) {
        // Attributes pass through as-is: nb::list(accessor) resolved to a
        // plain borrow on GCC/Clang and is an ambiguous overload on MSVC.
        nb::object inner = py_deepcopy(src.attr("inner")(), memo);
        qx.attr("ControlledBlock").attr("__init__")(
            result, inner, src.attr("num_controls"),
            src.attr("ctrl_state"), src.attr("name"));
    } else if (nb::isinstance(self, qx.attr("MeasureBlock"))) {
        qx.attr("MeasureBlock").attr("__init__")(
            result, src.attr("qubit_index"), src.attr("cbit_index"),
            src.attr("name"));
    } else if (nb::isinstance(self, qx.attr("ResetBlock"))) {
        qx.attr("ResetBlock").attr("__init__")(
            result, src.attr("qubit_index"), src.attr("name"));
    } else if (nb::isinstance(self, qx.attr("ConditionalBlock"))) {
        nb::object then_body = src.attr("then_body");
        nb::object else_body = src.attr("else_body");
        if (!then_body.is_none()) then_body = py_deepcopy(then_body, memo);
        if (!else_body.is_none()) else_body = py_deepcopy(else_body, memo);
        qx.attr("ConditionalBlock").attr("__init__")(
            result, src.attr("condition_cbits"),
            src.attr("condition_values"), then_body, else_body,
            src.attr("name"));
    } else {
        qx.attr("SimpleBlock").attr("__init__")(
            result, src.attr("n_qubits"), src.attr("name"));
    }
    result.attr("_copy_base_state_from")(src);
    if (nb::hasattr(self, "__dict__")) {
        for (auto item : nb::cast<nb::dict>(self.attr("__dict__")))
            result.attr("__setattr__")(item.first, py_deepcopy(item.second, memo));
    }
    return result;
}

// Bump together with EXPECTED_QARPX_ABI in qarp/_abi.py — same commit —
// whenever a binding signature, enum, or class shape changes (§15).
#define QARPX_ABI_VERSION 9

NB_MODULE(qarpx, m) {
    qarpx::init_threading();
    m.doc() = "OpenQARP C++ Circuit IR — Python bindings";

    m.def("_cpu_budget", [] {
        const qarpx::detail::CpuBudget b = qarpx::detail::read_cpu_budget();
        nb::dict d;
        d["logical"] = b.logical;
        d["physical"] = b.physical;
        d["limit"] = b.limit;
        d["default_thread_count"] = qarpx::detail::default_thread_count(b);
        return d;
    }, "Internal: this process's CPU budget as read now (0 = unknown, or no "
       "limit) and the default thread count it gives.");

    // ── Exception taxonomy at the Python boundary ──
    // capability_error → qarp.errors.CapabilityError (ValueError fallback if
    // qarp is not importable), with the offending Command attached as the
    // exception's `command` attribute.  sdk_missing_error → ImportError.
    // Anything else falls through to nanobind's default translation
    // (std::runtime_error → RuntimeError), so parse/internal errors keep
    // their type.
    nb::register_exception_translator(
        [](const std::exception_ptr& p, void* /*payload*/) {
            try {
                std::rethrow_exception(p);
            } catch (const qarpx::sdk_missing_error& e) {
                PyErr_SetString(PyExc_ImportError, e.what());
            } catch (const qarpx::capability_error& e) {
                nb::object cls;
                try {
                    cls = nb::module_::import_("qarp.errors")
                              .attr("CapabilityError");
                } catch (...) {
                    PyErr_SetString(PyExc_ValueError, e.what());
                    return;
                }
                nb::object exc = cls(e.what());
                try {
                    exc.attr("command") = nb::cast(e.command);
                } catch (...) {
                    // Best-effort attribute; the message stands alone.
                }
                PyErr_SetObject(cls.ptr(), exc.ptr());
            }
        });

    // Stamps for qarp/_abi.py's fail-fast import guard: a hand-bumped ABI
    // counter (stale build) and the configuring checkout (wrong-checkout
    // build; QARPX_SOURCE_DIR is baked in by python/CMakeLists.txt).
    m.attr("__abi_version__") = QARPX_ABI_VERSION;
    m.attr("__source_dir__") = QARPX_SOURCE_DIR;

    // Install the Python reference-counting handlers for Block's intrusive
    // counter (Block : nb::intrusive_base).  Once a block is handed to Python,
    // its C++ inc_ref/dec_ref delegate to Py_INCREF/Py_DECREF on the wrapper,
    // so C++ and Python share a single reference count — no shutdown leaks.
    // The hooks can run on threads Python did not create and during
    // interpreter shutdown, where the acquire fails and must not be followed
    // by a Python API call (nanobind 3, PEP 788).
    nb::intrusive_init(
        [](PyObject *o) noexcept {
            nb::gil_scoped_acquire guard;
            if (guard.is_valid())
                Py_INCREF(o);
        },
        [](PyObject *o) noexcept {
            nb::gil_scoped_acquire guard;
            if (guard.is_valid())
                Py_DECREF(o);
        });

    // openfermion-compatible QubitOperator / FermionOperator (separate TU).
    qarpx::python::register_operator_bindings(m);

    // ── GateType enum ──
    nb::enum_<GateType>(m, "GateType")
        .value("X", GateType::X)
        .value("Y", GateType::Y)
        .value("Z", GateType::Z)
        .value("H", GateType::H)
        .value("S", GateType::S)
        .value("Sdg", GateType::Sdg)
        .value("T", GateType::T)
        .value("Tdg", GateType::Tdg)
        .value("SX", GateType::SX)
        .value("SXdg", GateType::SXdg)
        .value("Id", GateType::Id)
        .value("Rx", GateType::Rx)
        .value("Ry", GateType::Ry)
        .value("Rz", GateType::Rz)
        .value("P", GateType::P)
        .value("U", GateType::U)
        .value("CX", GateType::CX)
        .value("CY", GateType::CY)
        .value("CZ", GateType::CZ)
        .value("SWAP", GateType::SWAP)
        .value("ECR", GateType::ECR)
        .value("iSWAP", GateType::iSWAP)
        .value("iSWAPdg", GateType::iSWAPdg)
        .value("CH", GateType::CH)
        .value("CS", GateType::CS)
        .value("CSdg", GateType::CSdg)
        .value("CSX", GateType::CSX)
        .value("CSXdg", GateType::CSXdg)
        .value("CRx", GateType::CRx)
        .value("CRy", GateType::CRy)
        .value("CRz", GateType::CRz)
        .value("CP", GateType::CP)
        .value("RZZ", GateType::RZZ)
        .value("RXX", GateType::RXX)
        .value("RYY", GateType::RYY)
        .value("CU", GateType::CU)
        .value("CCX", GateType::CCX)
        .value("CSWAP", GateType::CSWAP)
        .value("MCZ", GateType::MCZ)
        .value("GPhase", GateType::GPhase)
        .value("Barrier", GateType::Barrier)
        .value("Measure", GateType::Measure)
        .value("Reset", GateType::Reset)
        .value("BranchBegin", GateType::BranchBegin)
        .value("BranchElse", GateType::BranchElse)
        .value("BranchEnd", GateType::BranchEnd)
        .value("Custom", GateType::Custom);

    // ── Pauli enum ──
    //
    // Bound for ergonomic Python access (e.g. `qx.Pauli.X`).  Python callers
    // typically pass Paulis as strings ("XYZI") — the Block methods below
    // accept both.
    nb::enum_<Pauli>(m, "Pauli")
        .value("I", Pauli::I)
        .value("X", Pauli::X)
        .value("Y", Pauli::Y)
        .value("Z", Pauli::Z);

    m.def("parse_pauli_string",
          [](const std::string& s) {
              return parse_pauli_string(s);
          },
          "s"_a,
          "Parse a string like 'IXYZ' into a list[Pauli].  "
          "Accepts upper/lower case; throws on empty or invalid characters.");

    m.def("commutes",
          [](const PauliString& a, const PauliString& b) {
              return commutes(a, b);
          },
          "a"_a, "b"_a,
          "Pairwise commutation predicate for multi-qubit Paulis.");

    m.def("compute_basis_change_clifford",
          [](const std::vector<PauliString>& paulis, int n_qubits) {
              auto r = qarpx::synthesis::detail::compute_basis_change_clifford(
                  paulis, n_qubits);
              return std::make_tuple(std::move(r.clifford),
                                     std::move(r.z_rows),
                                     std::move(r.signs));
          },
          "paulis"_a, "n_qubits"_a,
          "Diagonalise a set of mutually-commuting multi-qubit Paulis to a "
          "Z-only basis.  Returns (clifford_commands, z_rows, signs): apply "
          "clifford_commands (in order) to the state, then measure every "
          "qubit; Pauli r then equals (signs[r] ? -1 : +1) * (tensor of Z "
          "over the qubits where z_rows[r][q] is true).  Raises ValueError "
          "if the input is not pairwise commuting.");

    // ── Param ──
    nb::class_<Param>(m, "Param")
        .def(nb::init<double>())
        .def_static("symbol", &Param::symbol)
        .def_static("linear", &Param::linear,
            "coeff"_a, "symbol_name"_a, "offset"_a = 0.0)
        .def("is_symbolic", &Param::is_symbolic)
        .def("is_concrete", &Param::is_concrete)
        .def("value", &Param::value)
        .def("free_symbols", &Param::free_symbols)
        .def("substitute", &Param::substitute)
        .def("rename_symbols", &Param::rename_symbols, "mapping"_a)
        .def("evaluate", &Param::evaluate)
        .def("approx_equal", &Param::approx_equal, "other"_a, "atol"_a = 1e-9)
        .def("__deepcopy__", [](const Param& p, nb::handle) { return Param(p); },
             "memo"_a)
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def("__neg__", [](const Param& p) { return -p; })
        .def("__add__", [](const Param& a, const Param& b) { return a + b; })
        .def("__sub__", [](const Param& a, const Param& b) { return a - b; })
        .def("__mul__", [](const Param& a, const Param& b) { return a * b; })
        .def("__truediv__", [](const Param& a, const Param& b) { return a / b; })
        .def("__repr__", &Param::to_string);

    // Allow Python float / int to be passed wherever a Param is expected.
    // Without this nanobind's overload resolution gets confused once we have
    // both scalar-Param and bulk-list-of-(qubit, Param) overloads on Block.
    nb::implicitly_convertible<double, Param>();
    nb::implicitly_convertible<int, Param>();

    // Convenience factory
    m.def("symbol", &Param::symbol, "Create a symbolic parameter");
    m.def("affine_coefficients", &affine_coefficients, "param"_a,
        "{symbol: d(param)/d(symbol)} for every free symbol, verified affine; "
        "raises CapabilityError otherwise.");
    m.def("concrete", [](double v) { return Param(v); }, "Create a concrete parameter");

    // ── Command ──
    nb::class_<Command>(m, "Command")
        .def(nb::init<>())
        .def(nb::init<GateType, uint32_t>())
        .def(nb::init<GateType, uint32_t, Param>())
        .def(nb::init<GateType, uint32_t, uint32_t>())
        .def(nb::init<GateType, uint32_t, uint32_t, Param>())
        .def_rw("gate", &Command::gate)
        // Structural equality (gate, qubits, params, cbits, conditions) —
        // without this Python `==` falls back to object identity.
        .def("__eq__", [](const Command& a, const Command& b) { return a == b; })
        .def("__ne__", [](const Command& a, const Command& b) { return a != b; })
        .def_prop_ro("qubits", [](const Command& c) {
            return std::vector<uint32_t>(c.qubits.begin(), c.qubits.end());
        })
        .def_prop_ro("params", [](const Command& c) {
            return std::vector<Param>(c.params.begin(), c.params.end());
        })
        .def_prop_ro("cbits", [](const Command& c) {
            return std::vector<uint32_t>(c.cbits.begin(), c.cbits.end());
        })
        .def_prop_ro("condition_bits", [](const Command& c) {
            return std::vector<uint32_t>(c.condition_bits.begin(), c.condition_bits.end());
        })
        .def_prop_ro("condition_values", [](const Command& c) {
            return std::vector<bool>(c.condition_values.begin(), c.condition_values.end());
        })
        .def("is_parametric", &Command::is_parametric)
        .def("dagger", &Command::dagger)
        .def("remap_qubits", &Command::remap_qubits)
        .def("substitute", &Command::substitute)
        .def("rename_symbols", &Command::rename_symbols, "mapping"_a)
        .def("approx_equal", &Command::approx_equal, "other"_a, "atol"_a = 1e-9)
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def("__repr__", &Command::to_string);

    // Batch substitution — keeps the per-command loop in C++.
    m.def("substitute_all", &substitute_all,
        "commands"_a, "values"_a,
        "Return a new command sequence with all symbolic parameters substituted.");
    m.def("rename_symbol_occurrences",
        [](const std::vector<Command>& commands, const std::string& symbol,
           const std::string& prefix) {
            auto [out, renamed] = rename_symbol_occurrences(commands, symbol, prefix);
            std::vector<std::tuple<std::size_t, std::size_t, std::string>> occ;
            occ.reserve(renamed.size());
            for (const auto& r : renamed) occ.emplace_back(r.command, r.param, r.name);
            return std::make_pair(std::move(out), std::move(occ));
        },
        "commands"_a, "symbol"_a, "prefix"_a,
        "(renamed commands, [(command, param, private_name), ...]): every "
        "occurrence of `symbol` gets its own private name for per-occurrence "
        "parameter shifts.");

    // Tolerant sequence comparison — used by Block.equals()/__eq__ and by the
    // Python block wrapper (qarp/blocks/block.py) to compare flattened
    // command lists after applying pending Python-level transforms.
    m.def("commands_equal", &commands_equal,
        "a"_a, "b"_a, "atol"_a = 1e-9,
        "Order-sensitive tolerant equality between two command sequences.");

    m.def("cbit_register_width", &cbit_register_width,
        "commands"_a,
        "Width of the classical register a command stream needs: 1 + the "
        "largest cbit index referenced by a Measure write or a condition read, "
        "or 0 if none is.");

    m.def("gate_is_physical", &gate_is_physical, "gate"_a,
        "True iff this GateType applies a physical operation to the register "
        "(False for Barrier/Measure/Reset/GPhase/branch markers) — the single "
        "classification behind n_nqb_gates and qarp.resources counting.");

    m.def("n_nqb_gates", &n_nqb_gates,
        "commands"_a, "k"_a,
        "Number of k-qubit gates in a command stream (excludes Barrier/"
        "Measure/Reset/GPhase/branch markers — see gate_is_physical).");

    m.def("n_physical_gates", &n_physical_gates,
        "commands"_a,
        "Total physical gate count over all arities (same exclusions as "
        "n_nqb_gates) — the resource vector's headline n_gates.");

    m.def("n_gates_of_type", &n_gates_of_type,
        "commands"_a, "gate"_a,
        "Number of commands of the given GateType — unfiltered.");

    m.def("uninitialised_condition_cbits", &uninitialised_condition_cbits,
        "commands"_a,
        "Cbits read as a classical condition before any Measure wrote them. "
        "Empty means every condition has a preceding write. Evaluate on a "
        "complete flattened program — a sub-block whose measurement lives in an "
        "enclosing block will report its condition cbits here.");

    // ── Block (base) ──
    //
    // nb::dynamic_attr() lets Python subclasses attach attributes — required
    // by the Python wrapper layer, which adds sympy state, deferred
    // substitutions, etc. on top of the C++ block.
    //
    // nb::intrusive_ptr wires Block's intrusive counter to nanobind: when a
    // block crosses into Python, set_self_py() hands lifetime ownership to the
    // wrapper so the C++ ref<Block> count and the Python refcount are one and
    // the same.  A block is then released as soon as both sides drop it, with
    // no reference cycle keeping it alive.  Subclasses inherit this annotation
    // automatically.
    nb::class_<Block>(m, "Block", nb::dynamic_attr(),
        nb::intrusive_ptr<Block>(
            [](Block *o, PyObject *po) noexcept { o->set_self_py(po); }))
        .def("build", &Block::build)
        .def("is_built", &Block::is_built)
        .def("flatten", &Block::flatten)
        .def("equals", &Block::equals, "other"_a, "atol"_a = 1e-9,
             "Tolerant 'same circuit' check — see Block::equals in block.h.")
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def("commands", &Block::commands, nb::rv_policy::reference_internal)
        .def("set_commands", &Block::set_commands,
             "Replace the local command buffer.  Used by Python __deepcopy__ "
             "to clone ad-hoc-populated blocks.")
        .def("set_built", &Block::set_built,
             "Force the C++ built_ flag.  Used by Python __deepcopy__ to mirror "
             "the source block's built state.")
        .def("_copy_base_state_from", &Block::copy_base_state_from,
             "Clone-internal: compiler-complete copy of all base Block fields "
             "(commands, built flag, name, sizes, targets, controls).  Tree "
             "edges live in subclasses and are untouched.")
        .def("__deepcopy__", &block_py_deepcopy, "memo"_a,
             "Deep clone for raw C++-created blocks (subclasses inherit; "
             "qarp's Python mixin __deepcopy__ shadows this).")
        .def("free_symbols", &Block::free_symbols)
        .def("dagger", &Block::dagger)
        .def("set_symbols", &Block::set_symbols)
        .def("replace_symbols", &Block::replace_symbols)
        .def_rw("name", &Block::name)
        .def_rw("n_qubits", &Block::n_qubits)
        .def_rw("n_cbits", &Block::n_cbits)
        .def_rw("target_qubits", &Block::target_qubits)
        .def_rw("target_cbits",  &Block::target_cbits)
        // Builder API — scalar overloads
        .def("h",   nb::overload_cast<uint32_t>(&Block::h),   nb::rv_policy::reference)
        .def("x",   nb::overload_cast<uint32_t>(&Block::x),   nb::rv_policy::reference)
        .def("y",   nb::overload_cast<uint32_t>(&Block::y),   nb::rv_policy::reference)
        .def("z",   nb::overload_cast<uint32_t>(&Block::z),   nb::rv_policy::reference)
        .def("s",   nb::overload_cast<uint32_t>(&Block::s),   nb::rv_policy::reference)
        .def("sdg", nb::overload_cast<uint32_t>(&Block::sdg), nb::rv_policy::reference)
        .def("t",   nb::overload_cast<uint32_t>(&Block::t),   nb::rv_policy::reference)
        .def("tdg", nb::overload_cast<uint32_t>(&Block::tdg), nb::rv_policy::reference)
        .def("sx",   nb::overload_cast<uint32_t>(&Block::sx),   nb::rv_policy::reference)
        .def("sxdg", nb::overload_cast<uint32_t>(&Block::sxdg), nb::rv_policy::reference)
        .def("id",   nb::overload_cast<uint32_t>(&Block::id),   nb::rv_policy::reference)
        .def("rx",  nb::overload_cast<uint32_t, Param>(&Block::rx), nb::rv_policy::reference)
        .def("ry",  nb::overload_cast<uint32_t, Param>(&Block::ry), nb::rv_policy::reference)
        .def("rz",  nb::overload_cast<uint32_t, Param>(&Block::rz), nb::rv_policy::reference)
        .def("p",   nb::overload_cast<uint32_t, Param>(&Block::p),  nb::rv_policy::reference)
        .def("cx",      nb::overload_cast<uint32_t, uint32_t>(&Block::cx),      nb::rv_policy::reference)
        .def("cy",      nb::overload_cast<uint32_t, uint32_t>(&Block::cy),      nb::rv_policy::reference)
        .def("cz",      nb::overload_cast<uint32_t, uint32_t>(&Block::cz),      nb::rv_policy::reference)
        .def("swap",    nb::overload_cast<uint32_t, uint32_t>(&Block::swap),    nb::rv_policy::reference)
        .def("ccx",     nb::overload_cast<uint32_t, uint32_t, uint32_t>(&Block::ccx),     nb::rv_policy::reference)
        .def("cswap",   nb::overload_cast<uint32_t, uint32_t, uint32_t>(&Block::cswap),   nb::rv_policy::reference)
        .def("mcz", &Block::mcz, nb::rv_policy::reference)
        .def("rzz", nb::overload_cast<uint32_t, uint32_t, Param>(&Block::rzz), nb::rv_policy::reference)
        .def("rxx", nb::overload_cast<uint32_t, uint32_t, Param>(&Block::rxx), nb::rv_policy::reference)
        .def("ryy", nb::overload_cast<uint32_t, uint32_t, Param>(&Block::ryy), nb::rv_policy::reference)
        .def("crx", nb::overload_cast<uint32_t, uint32_t, Param>(&Block::crx), nb::rv_policy::reference)
        .def("cry", nb::overload_cast<uint32_t, uint32_t, Param>(&Block::cry), nb::rv_policy::reference)
        .def("crz", nb::overload_cast<uint32_t, uint32_t, Param>(&Block::crz), nb::rv_policy::reference)
        .def("cp",  nb::overload_cast<uint32_t, uint32_t, Param>(&Block::cp),  nb::rv_policy::reference)
        .def("ecr",     nb::overload_cast<uint32_t, uint32_t>(&Block::ecr),     nb::rv_policy::reference)
        .def("iswap",   nb::overload_cast<uint32_t, uint32_t>(&Block::iswap),   nb::rv_policy::reference)
        .def("iswapdg", nb::overload_cast<uint32_t, uint32_t>(&Block::iswapdg), nb::rv_policy::reference)
        .def("ch",      nb::overload_cast<uint32_t, uint32_t>(&Block::ch),      nb::rv_policy::reference)
        .def("cs",      nb::overload_cast<uint32_t, uint32_t>(&Block::cs),      nb::rv_policy::reference)
        .def("csdg",    nb::overload_cast<uint32_t, uint32_t>(&Block::csdg),    nb::rv_policy::reference)
        .def("csx",     nb::overload_cast<uint32_t, uint32_t>(&Block::csx),     nb::rv_policy::reference)
        .def("csxdg",   nb::overload_cast<uint32_t, uint32_t>(&Block::csxdg),   nb::rv_policy::reference)
        .def("gphase", &Block::gphase, nb::rv_policy::reference)
        .def("u",  &Block::u,  nb::rv_policy::reference)
        .def("cu", &Block::cu, nb::rv_policy::reference)
        .def("measure", nb::overload_cast<uint32_t, uint32_t>(&Block::measure), nb::rv_policy::reference)
        .def("reset",   nb::overload_cast<uint32_t>(&Block::reset),             nb::rv_policy::reference)
        // ── Variadic / bulk builder overloads ──
        // Each accepts a list of qubit specs (or (qubit, param) / (q0, q1, …) tuples)
        // and emits all gates in one nanobind dispatch — collapses the per-gate
        // Python tax to ~one crossing per call regardless of N.
        .def("h",   nb::overload_cast<const std::vector<uint32_t>&>(&Block::h),   nb::rv_policy::reference)
        .def("x",   nb::overload_cast<const std::vector<uint32_t>&>(&Block::x),   nb::rv_policy::reference)
        .def("y",   nb::overload_cast<const std::vector<uint32_t>&>(&Block::y),   nb::rv_policy::reference)
        .def("z",   nb::overload_cast<const std::vector<uint32_t>&>(&Block::z),   nb::rv_policy::reference)
        .def("s",   nb::overload_cast<const std::vector<uint32_t>&>(&Block::s),   nb::rv_policy::reference)
        .def("sdg", nb::overload_cast<const std::vector<uint32_t>&>(&Block::sdg), nb::rv_policy::reference)
        .def("t",   nb::overload_cast<const std::vector<uint32_t>&>(&Block::t),   nb::rv_policy::reference)
        .def("tdg", nb::overload_cast<const std::vector<uint32_t>&>(&Block::tdg), nb::rv_policy::reference)
        .def("sx",   nb::overload_cast<const std::vector<uint32_t>&>(&Block::sx),   nb::rv_policy::reference)
        .def("sxdg", nb::overload_cast<const std::vector<uint32_t>&>(&Block::sxdg), nb::rv_policy::reference)
        .def("id",   nb::overload_cast<const std::vector<uint32_t>&>(&Block::id),   nb::rv_policy::reference)
        .def("rx",  nb::overload_cast<const std::vector<std::pair<uint32_t, Param>>&>(&Block::rx), nb::rv_policy::reference)
        .def("ry",  nb::overload_cast<const std::vector<std::pair<uint32_t, Param>>&>(&Block::ry), nb::rv_policy::reference)
        .def("rz",  nb::overload_cast<const std::vector<std::pair<uint32_t, Param>>&>(&Block::rz), nb::rv_policy::reference)
        .def("p",   nb::overload_cast<const std::vector<std::pair<uint32_t, Param>>&>(&Block::p),  nb::rv_policy::reference)
        .def("cx",      nb::overload_cast<const std::vector<std::pair<uint32_t, uint32_t>>&>(&Block::cx),      nb::rv_policy::reference)
        .def("cy",      nb::overload_cast<const std::vector<std::pair<uint32_t, uint32_t>>&>(&Block::cy),      nb::rv_policy::reference)
        .def("cz",      nb::overload_cast<const std::vector<std::pair<uint32_t, uint32_t>>&>(&Block::cz),      nb::rv_policy::reference)
        .def("swap",    nb::overload_cast<const std::vector<std::pair<uint32_t, uint32_t>>&>(&Block::swap),    nb::rv_policy::reference)
        .def("ecr",     nb::overload_cast<const std::vector<std::pair<uint32_t, uint32_t>>&>(&Block::ecr),     nb::rv_policy::reference)
        .def("iswap",   nb::overload_cast<const std::vector<std::pair<uint32_t, uint32_t>>&>(&Block::iswap),   nb::rv_policy::reference)
        .def("iswapdg", nb::overload_cast<const std::vector<std::pair<uint32_t, uint32_t>>&>(&Block::iswapdg), nb::rv_policy::reference)
        .def("ch",      nb::overload_cast<const std::vector<std::pair<uint32_t, uint32_t>>&>(&Block::ch),      nb::rv_policy::reference)
        .def("cs",      nb::overload_cast<const std::vector<std::pair<uint32_t, uint32_t>>&>(&Block::cs),      nb::rv_policy::reference)
        .def("csdg",    nb::overload_cast<const std::vector<std::pair<uint32_t, uint32_t>>&>(&Block::csdg),    nb::rv_policy::reference)
        .def("csx",     nb::overload_cast<const std::vector<std::pair<uint32_t, uint32_t>>&>(&Block::csx),     nb::rv_policy::reference)
        .def("csxdg",   nb::overload_cast<const std::vector<std::pair<uint32_t, uint32_t>>&>(&Block::csxdg),   nb::rv_policy::reference)
        .def("crx", nb::overload_cast<const std::vector<std::tuple<uint32_t, uint32_t, Param>>&>(&Block::crx), nb::rv_policy::reference)
        .def("cry", nb::overload_cast<const std::vector<std::tuple<uint32_t, uint32_t, Param>>&>(&Block::cry), nb::rv_policy::reference)
        .def("crz", nb::overload_cast<const std::vector<std::tuple<uint32_t, uint32_t, Param>>&>(&Block::crz), nb::rv_policy::reference)
        .def("cp",  nb::overload_cast<const std::vector<std::tuple<uint32_t, uint32_t, Param>>&>(&Block::cp),  nb::rv_policy::reference)
        .def("rzz", nb::overload_cast<const std::vector<std::tuple<uint32_t, uint32_t, Param>>&>(&Block::rzz), nb::rv_policy::reference)
        .def("rxx", nb::overload_cast<const std::vector<std::tuple<uint32_t, uint32_t, Param>>&>(&Block::rxx), nb::rv_policy::reference)
        .def("ryy", nb::overload_cast<const std::vector<std::tuple<uint32_t, uint32_t, Param>>&>(&Block::ryy), nb::rv_policy::reference)
        .def("ccx",   nb::overload_cast<const std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>&>(&Block::ccx),   nb::rv_policy::reference)
        .def("cswap", nb::overload_cast<const std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>&>(&Block::cswap), nb::rv_policy::reference)
        .def("measure", nb::overload_cast<const std::vector<std::pair<uint32_t, uint32_t>>&>(&Block::measure), nb::rv_policy::reference)
        .def("reset",   nb::overload_cast<const std::vector<uint32_t>&>(&Block::reset),                       nb::rv_policy::reference)
        // Synthesis builders
        .def("state_preparation", &Block::state_preparation,
             "amplitudes"_a, nb::rv_policy::reference,
             "Prepare an arbitrary 2^n-amplitude state from |0…0⟩ (Möttönen).")
        .def("diagonal_unitary", &Block::diagonal_unitary,
             "diagonal_elements"_a, nb::rv_policy::reference,
             "Implement an arbitrary 2^n × 2^n diagonal unitary "
             "(Shende-Bullock-Markov).")
        .def("unitary_synthesis", &Block::unitary_synthesis,
             "U"_a, nb::rv_policy::reference,
             "Implement an arbitrary 2^n × 2^n unitary via Quantum Shannon "
             "Decomposition.")
        // Pauli-exponential builders.  Python accepts either a string ("XYZI")
        // or a list[Pauli]; the lambdas parse strings on the way in.
        .def("pauli_exp",
             [](Block& b, const std::string& pauli, Param angle) -> Block& {
                 b.pauli_exp(parse_pauli_string(pauli), angle);
                 return b;
             },
             "pauli"_a, "angle"_a, nb::rv_policy::reference,
             "Append exp(-i · angle / 2 · pauli) for a single multi-qubit Pauli.")
        .def("pauli_exp",
             [](Block& b, const PauliString& pauli, Param angle) -> Block& {
                 b.pauli_exp(pauli, angle);
                 return b;
             },
             "pauli"_a, "angle"_a, nb::rv_policy::reference)
        .def("commuting_pauli_set_exp",
             [](Block& b,
                const std::vector<std::string>& paulis,
                const std::vector<Param>& angles) -> Block& {
                 std::vector<PauliString> parsed;
                 parsed.reserve(paulis.size());
                 for (const auto& s : paulis) parsed.push_back(parse_pauli_string(s));
                 b.commuting_pauli_set_exp(parsed, angles);
                 return b;
             },
             "paulis"_a, "angles"_a, nb::rv_policy::reference,
             "Append Π_i exp(-i · angles[i] / 2 · paulis[i]) for a set of "
             "mutually commuting Pauli strings.  Validates lengths and "
             "pairwise commutation.")
        .def("commuting_pauli_set_exp",
             [](Block& b,
                const std::vector<PauliString>& paulis,
                const std::vector<Param>& angles) -> Block& {
                 b.commuting_pauli_set_exp(paulis, angles);
                 return b;
             },
             "paulis"_a, "angles"_a, nb::rv_policy::reference);

    // ── SimpleBlock ──
    nb::class_<SimpleBlock, Block>(m, "SimpleBlock", nb::dynamic_attr())
        .def(nb::init<uint32_t, const std::string&>(),
            "n_qubits"_a, "name"_a = "Circuit")
        .def("add_register", &SimpleBlock::add_register,
            "name"_a, "size"_a,
            "Allocate next `size` qubit indices for a named register. "
            "Returns list of qubit indices [next, ..., next+size-1].");


    // ── CompositeBlock ──
    //
    // CompositeBlock, ControlledBlock and ConditionalBlock hold child blocks
    // via ref<Block> (intrusive reference counting).  Because the intrusive
    // counter is shared with the Python wrapper, there is no
    // C++-holder ↔ Python-__dict__ reference cycle — child blocks are freed by
    // ordinary reference counting when their parent goes away.
    nb::class_<CompositeBlock, Block>(m, "CompositeBlock", nb::dynamic_attr())
        .def(nb::init<std::vector<ref<Block>>, uint32_t, const std::string&>(),
            "children"_a, "n_qubits"_a = 0, "name"_a = "CompositeBlock")
        .def("add_child", &CompositeBlock::add_child)
        .def("children", &CompositeBlock::children, nb::rv_policy::reference_internal)
        .def("uninitialised_condition_cbits",
             &CompositeBlock::uninitialised_condition_cbits,
             "Cbits this block's flattened program reads as a classical "
             "condition before any Measure wrote them. Empty means every "
             "condition has a preceding write. See "
             "qx.uninitialised_condition_cbits for the free-function form.");

    // ── ControlledBlock ──
    nb::class_<ControlledBlock, Block>(m, "ControlledBlock", nb::dynamic_attr())
        .def(nb::init<ref<Block>, uint32_t, std::vector<bool>, const std::string&>(),
            "inner"_a, "num_controls"_a, "ctrl_state"_a, "name"_a = "Controlled")
        .def("inner", &ControlledBlock::inner)
        .def_prop_ro("num_controls", &ControlledBlock::num_controls)
        .def_prop_ro("ctrl_state",   &ControlledBlock::ctrl_state);

    // ── MeasureBlock ──
    nb::class_<MeasureBlock, Block>(m, "MeasureBlock", nb::dynamic_attr())
        .def(nb::init<uint32_t, uint32_t, const std::string&>(),
            "qubit"_a, "cbit"_a, "name"_a = "Measure")
        .def_prop_ro("qubit_index", &MeasureBlock::qubit_index)
        .def_prop_ro("cbit_index",  &MeasureBlock::cbit_index);

    // ── ResetBlock ──
    nb::class_<ResetBlock, Block>(m, "ResetBlock", nb::dynamic_attr())
        .def(nb::init<uint32_t, const std::string&>(),
            "qubit"_a, "name"_a = "Reset")
        .def_prop_ro("qubit_index", &ResetBlock::qubit_index);

    // ── ConditionalBlock ──
    nb::class_<ConditionalBlock, Block>(m, "ConditionalBlock", nb::dynamic_attr())
        .def(nb::init<std::vector<uint32_t>,
                       std::vector<bool>,
                       ref<Block>,
                       ref<Block>,
                       const std::string&>(),
            "cbits"_a, "values"_a,
            "then_body"_a,
            "else_body"_a = nb::none(),
            "name"_a = "Conditional")
        .def_prop_ro("then_body",        &ConditionalBlock::then_body)
        .def_prop_ro("else_body",        &ConditionalBlock::else_body)
        .def_prop_ro("condition_cbits",  &ConditionalBlock::condition_cbits)
        .def_prop_ro("condition_values", &ConditionalBlock::condition_values);

    // Callable-based factory for ConditionalBlock — the ergonomic Python
    // entrypoint.  Each branch callback receives a fresh SimpleBlock that it
    // populates with gates; the wrapper builds it and folds it into the
    // ConditionalBlock.  Either branch may be None (omitted).
    //
    //   def then_body(b): b.x(0)
    //   cond = qx.conditional([0], [True], then=then_body, n_qubits=1)
    m.def("conditional", [](std::vector<uint32_t> cbits,
                             std::vector<bool>     values,
                             nb::object            then_fn,
                             nb::object            else_fn,
                             uint32_t              n_qubits,
                             const std::string&    name)
        -> ref<ConditionalBlock>
    {
        auto build_branch = [&](nb::object fn) -> ref<Block> {
            if (fn.is_none()) return ref<Block>();
            ref<SimpleBlock> block(new SimpleBlock(n_qubits));
            nb::cast<nb::callable>(fn)(block);
            block->build();
            return ref<Block>(block.get());
        };
        ref<ConditionalBlock> cond(new ConditionalBlock(
            std::move(cbits), std::move(values),
            build_branch(then_fn), build_branch(else_fn),
            name));
        cond->build();
        return cond;
    },
        // kw_only: `n_qubits` is required but declared after two defaulted
        // arguments, which no Python signature can express positionally — the
        // advertised signature in __doc__ was not parseable.
        "cbits"_a, "values"_a, nb::kw_only(),
        "then"_a   = nb::none(),
        "else_"_a  = nb::none(),
        "n_qubits"_a,
        "name"_a   = "Conditional",
        "Build a ConditionalBlock from `then` and/or `else_` callables, "
        "each of which receives a fresh SimpleBlock to populate.");

    // ── GateSet ──
    nb::class_<GateSet>(m, "GateSet")
        .def(nb::init<>())
        .def_rw("name", &GateSet::name)
        .def_rw("allowed", &GateSet::allowed)
        .def_rw("rules", &GateSet::rules)
        .def("contains", &GateSet::contains);

    m.def("native_gateset", &native_gateset);
    m.def("qulacs_gateset", &qulacs_gateset);
    m.def("qulacs_emitter_gateset", &qulacs_emitter_gateset);
    m.def("qiskit_gateset", &qiskit_gateset);
    m.def("pytket_gateset", &pytket_gateset);
    m.def("pennylane_gateset", &pennylane_gateset);
    m.def("cudaq_gateset", &cudaq_gateset);

    // ── Emitter capability contract ──
    nb::class_<EmitterCapabilities>(m, "EmitterCapabilities")
        .def(nb::init<>())
        .def_rw("symbolic_params", &EmitterCapabilities::symbolic_params)
        .def_rw("multi_symbol_params", &EmitterCapabilities::multi_symbol_params)
        .def_rw("distinct_cbits", &EmitterCapabilities::distinct_cbits)
        .def_rw("conditionals", &EmitterCapabilities::conditionals)
        .def_rw("multibit_conditions", &EmitterCapabilities::multibit_conditions)
        .def_rw("multibit_else", &EmitterCapabilities::multibit_else)
        .def_rw("custom_unitary", &EmitterCapabilities::custom_unitary)
        .def_rw("cu_global_phase", &EmitterCapabilities::cu_global_phase);

    nb::class_<Incompatibility>(m, "Incompatibility")
        .def_ro("command", &Incompatibility::command)
        .def_ro("reason", &Incompatibility::reason);
    m.def("clifford_t_gateset", &clifford_t_gateset);
    m.def("clifford_t_rz_gateset", &clifford_t_rz_gateset);
    m.def("full_gateset_1q", &full_gateset_1q);
    m.def("full_gateset_2q", &full_gateset_2q);
    m.def("full_gateset_1q_2q", &full_gateset_1q_2q);
    m.def("universal_gateset", &universal_gateset);
    m.def("routable_subset", &routable_subset, "gate_set"_a);

    // ── Architecture ──
    nb::class_<Architecture>(m, "Architecture")
        .def(nb::init<>())
        .def(nb::init<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>, std::string>(),
             "n_qubits"_a, "edges"_a, "name"_a = "")
        .def_rw("name", &Architecture::name)
        .def_rw("n_qubits", &Architecture::n_qubits)
        .def_prop_rw("edges",
                     [](const Architecture& a) { return a.edges(); },
                     &Architecture::set_edges,
                     "Edge list; assigning rebuilds the adjacency indices.")
        .def("is_connected", &Architecture::is_connected, "a"_a, "b"_a)
        .def("has_directed_edge", &Architecture::has_directed_edge, "a"_a, "b"_a)
        .def("neighbours", [](const Architecture& a, uint32_t q) { return a.neighbours(q); }, "q"_a)
        .def("shortest_path", &Architecture::shortest_path, "a"_a, "b"_a);

    m.def("nearest_neighbour_architecture", &nearest_neighbour_architecture,
          "xdim"_a, "ydim"_a);
    m.def("all_to_all_architecture", &all_to_all_architecture, "n_qubits"_a);

    // ── NoiseModel ──
    //
    // The C++-side NoiseChannel `std::function<Channel(const Command&)>` is
    // not exposed to Python: Python closures in a per-shot hot loop would
    // erase the perf benefit of the unified kernel.  Python users compose
    // channels via the named builders below.
    nb::class_<NoiseModel>(m, "NoiseModel")
        .def(nb::init<>())
        .def_rw("enabled", &NoiseModel::enabled)
        .def("has_channel", &NoiseModel::has_channel, "gate"_a)
        .def("has_any_channel", &NoiseModel::has_any_channel)
        .def("__iadd__", [](NoiseModel& self, const NoiseModel& other) -> NoiseModel& {
            self += other;
            return self;
        }, nb::rv_policy::reference_internal)
        .def("__add__", [](const NoiseModel& a, const NoiseModel& b) {
            return a + b;
        });

    m.def("make_depolarizing", &make_depolarizing, "p"_a, "gates"_a);
    m.def("make_pauli", &make_pauli, "p_x"_a, "p_y"_a, "p_z"_a, "gates"_a);
    m.def("make_bit_flip", &make_bit_flip, "p"_a, "gates"_a);
    m.def("make_amplitude_damping", &make_amplitude_damping, "p"_a, "gates"_a);

    // ── Device ──
    nb::class_<Device>(m, "Device")
        .def(nb::init<>())
        .def(nb::init<uint32_t>(), "n_qubits"_a)
        .def(nb::init<uint32_t,
                      std::optional<Architecture>,
                      std::optional<NoiseModel>,
                      std::optional<GateSet>,
                      bool>(),
             "n_qubits"_a,
             "architecture"_a  = nb::none(),
             "noise_model"_a   = nb::none(),
             "gate_set"_a      = nb::none(),
             "directedness"_a  = false)
        .def_rw("n_qubits",     &Device::n_qubits)
        .def_rw("architecture", &Device::architecture)
        .def_rw("noise_model",  &Device::noise_model)
        .def_rw("gate_set",     &Device::gate_set)
        .def_rw("directedness", &Device::directedness)
        .def("check_fits",      &Device::check_fits, "needed"_a);

    // ── CompiledCircuit + compile_for_device ──
    nb::class_<CompiledCircuit>(m, "CompiledCircuit")
        .def_ro("commands",            &CompiledCircuit::commands)
        .def_ro("initial_logical_to_physical", &CompiledCircuit::initial_logical_to_physical)
        .def_ro("final_logical_to_physical",   &CompiledCircuit::final_logical_to_physical);

    // RouterKind must be registered BEFORE any binding that uses it as a
    // default argument (nanobind converts defaults at registration time).
    nb::enum_<RouterKind>(m, "RouterKind")
        .value("Lite",  RouterKind::Lite)
        .value("Sabre", RouterKind::Sabre);

    m.def("compile_for_device", &compile_for_device,
          "commands"_a, "n_qubits"_a, "device"_a,
          "router"_a = RouterKind::Sabre,
          "Run check_fits → rebase → route → (directed SWAP lowering) → rebase "
          "→ assert_directions against `device`.");
    m.def("assert_directions", &assert_directions, "commands"_a, "arch"_a,
          "Raise CapabilityError if an asymmetric 2-qubit gate sits on a pair "
          "that is not a stored directed edge of `arch`.");

    // ── Circuit DAG (read-only introspection; the mutation API stays C++-only) ──
    nb::class_<CircuitDAG>(m, "CircuitDAG")
        .def_static("from_commands",
            nb::overload_cast<const std::vector<Command>&>(
                &CircuitDAG::from_commands),
            "commands"_a,
            "Build the wire-dependency DAG; qubit/cbit counts are inferred.")
        .def_static("from_commands",
            nb::overload_cast<const std::vector<Command>&, uint32_t, uint32_t>(
                &CircuitDAG::from_commands),
            "commands"_a, "n_qubits"_a, "n_cbits"_a)
        .def("to_commands", &CircuitDAG::to_commands,
            "Deterministic linearization (RT-1: exact round-trip).")
        .def("depth", &CircuitDAG::depth,
            "Longest dependency path (Barrier/GPhase weigh 0).")
        .def("layers", &CircuitDAG::layers,
            "ASAP moments: lists of node ids, dependency-levelled.")
        .def("front_layer", &CircuitDAG::front_layer)
        .def("count_ops", &CircuitDAG::count_ops,
            "Gate-name → count over the emitted stream (region interiors "
            "included).")
        .def("command", &CircuitDAG::command, "node"_a)
        .def("is_region", &CircuitDAG::is_region, "node"_a)
        .def("region_commands", &CircuitDAG::region_commands, "node"_a)
        .def("is_removed", &CircuitDAG::is_removed, "node"_a)
        .def("wires", [](const CircuitDAG& d, CircuitDAG::NodeId id) {
                const auto& w = d.wires(id);
                return std::vector<uint32_t>(w.begin(), w.end());
            }, "node"_a)
        .def_prop_ro("n_qubits", &CircuitDAG::n_qubits)
        .def_prop_ro("n_cbits", &CircuitDAG::n_cbits)
        .def_prop_ro("n_nodes", &CircuitDAG::n_nodes)
        .def_prop_ro("n_slots", &CircuitDAG::n_slots);

    // ── Router ──
    nb::class_<RoutingOptions>(m, "RoutingOptions")
        .def(nb::init<>())
        .def_rw("arch",            &RoutingOptions::arch)
        .def_rw("directedness",    &RoutingOptions::directedness)
        .def_rw("initial_mapping", &RoutingOptions::initial_mapping)
        .def_rw("router",          &RoutingOptions::router);

    nb::class_<RoutingResult>(m, "RoutingResult")
        .def_ro("commands",             &RoutingResult::commands)
        .def_ro("initial_logical_to_physical", &RoutingResult::initial_logical_to_physical)
        .def_ro("final_logical_to_physical",   &RoutingResult::final_logical_to_physical);

    m.def("route", &route, "commands"_a, "options"_a,
          "Route onto options.arch.  options.router picks Sabre (default: "
          "lookahead heuristic plus initial-mapping search) or Lite (greedy "
          "shortest-path sweep); both H-conjugate CX for direction flips on "
          "directed architectures.");
    m.def("reindex_sampling_result", &reindex_sampling_result,
          "physical"_a, "final_logical_to_physical"_a,
          "Permute a SamplingResult from per-physical-qubit bit ordering "
          "back to per-logical-qubit bit ordering.");

    // ── Transpiler ──
    nb::enum_<OptLevel>(m, "OptLevel")
        .value("O0", OptLevel::O0)
        .value("O1", OptLevel::O1)
        .value("O2", OptLevel::O2);

    nb::class_<Transpiler>(m, "Transpiler")
        .def(nb::init<GateSet>())
        .def("transpile", &Transpiler::transpile)
        .def("transpile_and_optimize", &Transpiler::transpile_and_optimize,
            "input"_a, "level"_a = OptLevel::O1)
        .def("optimize_in_target", &Transpiler::optimize_in_target,
            "input"_a, "level"_a = OptLevel::O1,
            "Optimize an in-target sequence without leaving the target: DAG "
            "passes, then single-qubit fusion only if the target admits Custom.")
        .def("transpile_parallel", &Transpiler::transpile_parallel,
            "blocks"_a, "n_threads"_a = 0)
        .def("register_decomposition", &Transpiler::register_decomposition,
             "gate"_a, "fn"_a,
             "Register (or override) the rule lowering `gate`: fn(Command) -> list[Command].")
        .def("install_clifford_t_rz_decompositions", &Transpiler::install_clifford_t_rz_decompositions,
            "Install the Rx/Ry/SWAP → Rz+Clifford decompositions on top of the built-in table.")
        .def("verify_closure", &Transpiler::verify_closure,
            "Throw if any GateType cannot be reduced to the target gate set.")
        .def("unreachable_gates", [](const Transpiler& t) {
            std::vector<GateType> result;
            for (auto g : t.unreachable_gates()) result.push_back(g);
            std::sort(result.begin(), result.end(), [](GateType a, GateType b) {
                return std::string(gate_name(a)) < std::string(gate_name(b));
            });
            return result;
        }, "Return GateTypes that the rule table cannot reduce into the target, "
           "sorted by name.");

    // ── Identities ──
    m.def("eliminate_identities", [](std::vector<Command> cmds) {
        auto n = eliminate_identities(cmds);
        return std::make_pair(cmds, n);
    }, "Eliminate identity patterns. Returns (optimized_commands, num_eliminated).");

    // ── Gate metadata ──
    m.def("gate_name", [](GateType g) { return std::string(gate_name(g)); });
    m.def("gate_num_params", &gate_num_params);
    m.def("gate_num_qubits", &gate_num_qubits);

    // ── QIR Emitter ──
    nb::class_<QIREmitter>(m, "QIREmitter")
        .def(nb::init<>())
        .def("target_name", &QIREmitter::target_name)
        .def("gate_set", &QIREmitter::gate_set)
        .def("capabilities", &QIREmitter::capabilities)
        .def("emit", &QIREmitter::emit,
            "commands"_a, "n_qubits"_a, "function_name"_a = "circuit")
        .def("emit_to_file", &QIREmitter::emit_to_file,
            "commands"_a, "n_qubits"_a, "path"_a,
            "function_name"_a = "circuit")
        .def("validate",
             [](const QIREmitter& self, const std::vector<Command>& cmds) {
                 return self.validate(cmds);
             },
             "commands"_a,
            "First Incompatibility in the sequence, or None if the whole "
            "sequence can cross.");

    // ── OpenQASM 3 Emitter ──
    nb::class_<QASM3Emitter>(m, "QASM3Emitter")
        .def(nb::init<>())
        .def("target_name", &QASM3Emitter::target_name)
        .def("gate_set", &QASM3Emitter::gate_set)
        .def("capabilities", &QASM3Emitter::capabilities)
        .def("emit", &QASM3Emitter::emit,
            "commands"_a, "n_qubits"_a, "circuit_name"_a = "circuit")
        .def("emit_to_file", &QASM3Emitter::emit_to_file,
            "commands"_a, "n_qubits"_a, "path"_a,
            "circuit_name"_a = "circuit")
        .def("validate",
             [](const QASM3Emitter& self, const std::vector<Command>& cmds) {
                 return self.validate(cmds);
             },
             "commands"_a,
            "First Incompatibility in the sequence, or None if the whole "
            "sequence can cross.");

    // ── OpenQASM 2 Emitter ──
    nb::class_<QASM2Emitter>(m, "QASM2Emitter")
        .def(nb::init<>())
        .def("target_name", &QASM2Emitter::target_name)
        .def("gate_set", &QASM2Emitter::gate_set)
        .def("capabilities", &QASM2Emitter::capabilities)
        .def("emit", &QASM2Emitter::emit,
            "commands"_a, "n_qubits"_a, "circuit_name"_a = "circuit")
        .def("emit_to_file", &QASM2Emitter::emit_to_file,
            "commands"_a, "n_qubits"_a, "path"_a,
            "circuit_name"_a = "circuit")
        .def("validate",
             [](const QASM2Emitter& self, const std::vector<Command>& cmds) {
                 return self.validate(cmds);
             },
             "commands"_a,
            "First Incompatibility in the sequence, or None if the whole "
            "sequence can cross.");

    // ── OpenQASM 3 Absorber ──
    nb::class_<QASM3Absorber>(m, "QASM3Absorber")
        .def(nb::init<>())
        .def("absorb",
            [](const QASM3Absorber& a, const std::string& src) {
                auto r = a.absorb(src);
                std::vector<Command>     cmds(std::move(r.commands));
                std::vector<std::string> syms(std::move(r.free_symbols));
                return std::make_tuple(std::move(cmds),
                                       r.n_qubits,
                                       r.n_cbits,
                                       std::move(syms));
            },
            "qasm3_source"_a,
            "Parse an OpenQASM 3.0 string into (commands, n_qubits, n_cbits, free_symbols). "
            "Supports the subset produced by QASM3Emitter.")
        .def("source_name", &QASM3Absorber::source_name);

    // ── OpenQASM 2 Absorber ──
    nb::class_<QASM2Absorber>(m, "QASM2Absorber")
        .def(nb::init<>())
        .def("absorb",
            [](const QASM2Absorber& a, const std::string& src) {
                auto r = a.absorb(src);
                std::vector<Command>     cmds(std::move(r.commands));
                std::vector<std::string> syms(std::move(r.free_symbols));
                return std::make_tuple(std::move(cmds),
                                       r.n_qubits,
                                       r.n_cbits,
                                       std::move(syms));
            },
            "qasm2_source"_a,
            "Parse an OpenQASM 2.0 string into (commands, n_qubits, n_cbits, free_symbols). "
            "free_symbols is always empty — OpenQASM 2 has no symbolic parameter.")
        .def("source_name", &QASM2Absorber::source_name);

    // ── SamplingResult ──
    nb::class_<SamplingResult>(m, "SamplingResult")
        .def_ro("n_qubits",     &SamplingResult::n_qubits)
        .def_ro("n_shots",      &SamplingResult::n_shots)
        .def_ro("n_cbits",      &SamplingResult::n_cbits)
        .def_ro("counts",       &SamplingResult::counts)
        .def_ro("cbit_history", &SamplingResult::cbit_history)
        .def_ro("statevector",  &SamplingResult::statevector)
        .def("probability", &SamplingResult::probability, "outcome"_a);

    // ── QarpSimulator ──
    nb::class_<QarpSimulator>(m, "QarpSimulator")
        .def(nb::init<>())
        .def(nb::init<NoiseModel>(), "noise_model"_a)
        .def_prop_rw("noise_model",
                     [](const QarpSimulator& s) { return s.noise_model(); },
                     [](QarpSimulator& s, NoiseModel n) { s.set_noise_model(std::move(n)); })
        .def_prop_rw("fusion_max_qubits",
                     [](const QarpSimulator& s) { return s.fusion_max_qubits(); },
                     [](QarpSimulator& s, std::size_t k) { s.set_fusion_max_qubits(k); },
                     "Gate fusion before statevector/run dispatch: 0 = none, 1 = "
                     "consecutive single-qubit gates only, k >= 2 = dense blocks of up "
                     "to k qubits.  Default from QARP_FUSION_MAX_QUBITS (read once per "
                     "process) else the built-in default; at most MAX_FUSION_QUBITS.")
        .def_prop_rw("fusion_min_qubits",
                     [](const QarpSimulator& s) { return s.fusion_min_qubits(); },
                     [](QarpSimulator& s, std::size_t n) { s.set_fusion_min_qubits(n); },
                     "Registers narrower than this get single-qubit fusion only "
                     "(the dense pass costs more than it saves there); 0 = always fuse.")
        .def_ro_static("MAX_FUSION_QUBITS", &QarpSimulator::kMaxFusionQubits)
        .def_ro_static("DEFAULT_FUSION_QUBITS", &QarpSimulator::kDefaultFusionQubits)
        .def_ro_static("DEFAULT_FUSION_MIN_QUBITS", &QarpSimulator::kDefaultFusionMinQubits)
        .def("run", [](const QarpSimulator& sim,
                       const std::vector<Command>& cmds,
                       int n_qubits, int n_shots,
                       std::optional<uint32_t> seed,
                       std::optional<InitialStateArray> initial_state) {
            auto psi = initial_state_to_vector(initial_state);
            nb::gil_scoped_release nogil;
            return sim.run(cmds, n_qubits, n_shots, seed, std::move(psi));
        }, "commands"_a, "n_qubits"_a, "n_shots"_a, "seed"_a = nb::none(),
           nb::kw_only(), "initial_state"_a = nb::none(),
           "initial_state seeds the register with LSB-indexed amplitudes "
           "(length 2^n_qubits, unit norm within 1e-10) instead of |0...0>.")
        .def("batch_run", [](const QarpSimulator& sim,
                             const std::vector<Command>& cmds,
                             int n_qubits, int n_shots,
                             const std::vector<std::unordered_map<std::string, double>>& param_sets,
                             std::optional<uint32_t> seed) {
            nb::gil_scoped_release nogil;
            return sim.batch_run(cmds, n_qubits, n_shots, param_sets, seed);
        }, "commands"_a, "n_qubits"_a, "n_shots"_a, "param_sets"_a, "seed"_a = nb::none())
        .def("statevector", [](const QarpSimulator& sim,
                               const std::vector<Command>& cmds,
                               int n_qubits,
                               std::optional<InitialStateArray> initial_state) {
            auto psi = initial_state_to_vector(initial_state);
            std::vector<std::complex<double>> sv;
            {
                nb::gil_scoped_release nogil;
                sv = sim.statevector(cmds, n_qubits, std::move(psi));
            }
            return statevector_to_numpy(std::move(sv));
        }, "commands"_a, "n_qubits"_a,
           nb::kw_only(), "initial_state"_a = nb::none(),
           "Full statevector as a zero-copy 1-D complex128 numpy array.  "
           "initial_state: same contract as run().")
        .def("expectation", [](const QarpSimulator& sim,
                               const std::vector<Command>& cmds,
                               int n_qubits,
                               const PauliObservable& observable,
                               std::optional<InitialStateArray> initial_state) {
            auto psi = initial_state_to_vector(initial_state);
            nb::gil_scoped_release nogil;
            return sim.expectation(cmds, n_qubits, observable, std::move(psi));
        }, "commands"_a, "n_qubits"_a, "observable"_a,
           nb::kw_only(), "initial_state"_a = nb::none(),
           "Re <psi|H|psi> for psi = circuit(initial_state or |0...0>), "
           "contracted in C++ without returning the statevector.  Observable "
           "format as run_gradient; initial_state: same contract as run().")
        .def("transition", [](const QarpSimulator& /*sim*/,
                              InitialStateArray bra,
                              InitialStateArray ket,
                              int n_qubits,
                              const PauliObservable& observable) {
            // Read straight from the ndarray buffers: copying would cost the
            // same O(2^n) as the sweep itself.
            check_transition_width(n_qubits);
            const uint64_t dim = uint64_t{1} << n_qubits;
            if (bra.shape(0) != dim || ket.shape(0) != dim)
                throw std::invalid_argument(
                    "transition: bra/ket lengths " + std::to_string(bra.shape(0)) + "/" +
                    std::to_string(ket.shape(0)) + " do not match 2^" +
                    std::to_string(n_qubits) + " = " + std::to_string(dim));
            nb::gil_scoped_release nogil;
            return pauli_transition(bra.data(), ket.data(), n_qubits, observable);
        }, "bra"_a, "ket"_a, "n_qubits"_a, "observable"_a,
           "<bra|H|ket> over two host statevectors (contiguous complex128, "
           "length 2**n_qubits, LSB-indexed); H need not be Hermitian.")
        .def("batch_expectation", [](const QarpSimulator& sim,
                                     const std::vector<Command>& cmds,
                                     int n_qubits,
                                     const PauliObservable& observable,
                                     const std::vector<std::unordered_map<std::string, double>>&
                                         param_sets) {
            nb::gil_scoped_release nogil;
            return sim.batch_expectation(cmds, n_qubits, observable, param_sets);
        }, "commands"_a, "n_qubits"_a, "observable"_a, "param_sets"_a,
           "expectation() swept over parameter sets — one <H> per set.  Same "
           "signature as CudaqSimulator.batch_expectation.")
        .def("unitary_matrix", [](const QarpSimulator& sim,
                                  const std::vector<Command>& cmds,
                                  int n_qubits) {
            // Return as list-of-lists for easy Python consumption.
            Eigen::MatrixXcd M;
            {
                nb::gil_scoped_release nogil;
                M = sim.unitary_matrix(cmds, n_qubits);
            }
            const int dim = static_cast<int>(M.rows());
            std::vector<std::vector<std::complex<double>>> out(dim, std::vector<std::complex<double>>(dim));
            for (int r = 0; r < dim; ++r)
                for (int c = 0; c < dim; ++c)
                    out[r][c] = M(r, c);
            return out;
        }, "commands"_a, "n_qubits"_a,
           "Compute the 2^n × 2^n unitary matrix. Returns a list-of-rows.")
        .def("simulate_qpe_structured", &QarpSimulator::simulate_qpe_structured,
            "u_cmds"_a, "state_prep"_a, "iqft_cmds"_a,
            "n_state"_a, "n_ancilla"_a, "n_shots"_a, "seed"_a = nb::none(),
            "Fast QPE via matrix exponentiation. Returns SamplingResult over ancilla.")
        .def("simulate_dosqpe_structured", &QarpSimulator::simulate_dosqpe_structured,
            "u_cmds"_a, "state_prep"_a, "iqft_cmds"_a,
            "n_state"_a, "n_ancilla"_a, "n_shots"_a, "seed"_a = nb::none(),
            "Fast DOS-QPE via matrix exponentiation. Returns SamplingResult over ancilla.")
        .def("run_gradient", [](const QarpSimulator& sim,
                                const std::vector<Command>& cmds,
                                int n_qubits,
                                const std::vector<
                                    std::pair<std::vector<std::pair<uint32_t, char>>,
                                              std::complex<double>>>& observable,
                                const std::unordered_map<std::string, double>& params,
                                const std::vector<std::string>& param_order,
                                std::optional<InitialStateArray> initial_state) {
            auto psi = initial_state_to_vector(initial_state);
            nb::gil_scoped_release nogil;
            return sim.run_gradient(cmds, n_qubits, observable, params, param_order,
                                    std::move(psi));
        }, "commands"_a, "n_qubits"_a, "observable"_a,
            "params"_a, "param_order"_a,
            nb::kw_only(), "initial_state"_a = nb::none(),
            "Adjoint-backprop gradient of <H> w.r.t. each parameter.  "
            "Cost: ~2 statevector simulations regardless of parameter count.  "
            "Exact for gates whose angle is a linear function of the supplied "
            "parameters (Rx/Ry/Rz/P/GPhase/CRx/CRy/CRz/CP/Rxx/Ryy/Rzz).  "
            "initial_state seeds the forward-sweep root (same contract as run()).")
        .def("run_gradient_phi", [](const QarpSimulator& sim,
                                    const std::vector<Command>& cmds,
                                    int n_qubits,
                                    const std::vector<std::complex<double>>& phi,
                                    const std::unordered_map<std::string, double>& params,
                                    const std::vector<std::string>& param_order,
                                    std::optional<InitialStateArray> initial_state) {
            auto psi = initial_state_to_vector(initial_state);
            nb::gil_scoped_release nogil;
            return sim.run_gradient_phi(cmds, n_qubits, phi, params, param_order,
                                        std::move(psi));
        }, "commands"_a, "n_qubits"_a, "phi"_a,
            "params"_a, "param_order"_a,
            nb::kw_only(), "initial_state"_a = nb::none(),
            "Adjoint-backprop gradient with caller-supplied |φ⟩.  Returns "
            "∂(2·Re ⟨φ|U(θ)|ψ₀⟩)/∂θ_k with ψ₀ = initial_state or |0…0⟩.  "
            "Set φ = ⟨bra|ψ_n⟩·|bra⟩ to obtain "
            "∂|⟨bra|ψ⟩|²/∂θ — the natural form for VQD-style overlap "
            "penalties.");

    // ── SDK emitters / absorbers ──
    register_pennylane_emitter(m);
    register_pennylane_absorber(m);
    register_pytket_emitter(m);
    register_pytket_absorber(m);
    register_qulacs_emitter(m);
    register_qulacs_absorber(m);
    register_qiskit_emitter(m);
    register_qiskit_absorber(m);

    // ── CUDA-Q / GPU backend ──
    //
    // Registered unconditionally so `qx.CudaqSimulator.available()` is always
    // callable (it returns False on CPU-only builds — see cudaq_simulator.cpp).
    // The enums/config are likewise always present; the Python CudaqEngine
    // gates on `available()` before constructing a simulator.
    nb::enum_<CudaqBackend>(m, "CudaqBackend")
        .value("StateVector",     CudaqBackend::StateVector)
        .value("StateVectorMGPU", CudaqBackend::StateVectorMGPU)
        .value("TensorNet",       CudaqBackend::TensorNet)
        .value("TensorNetMPS",    CudaqBackend::TensorNetMPS);

    nb::enum_<CudaqPrecision>(m, "CudaqPrecision")
        .value("FP32", CudaqPrecision::FP32)
        .value("FP64", CudaqPrecision::FP64);

    nb::class_<CudaqConfig>(m, "CudaqConfig")
        .def(nb::init<>())
        .def_rw("backend",      &CudaqConfig::backend)
        .def_rw("precision",    &CudaqConfig::precision)
        .def_rw("target",       &CudaqConfig::target)
        .def_rw("max_bond_dim", &CudaqConfig::max_bond_dim);

    nb::class_<CudaqSimulator>(m, "CudaqSimulator")
        .def(nb::init<>())
        .def(nb::init<CudaqConfig>(), "config"_a)
        .def_static("available", &CudaqSimulator::available)
        .def_prop_rw("config",
                     [](const CudaqSimulator& s) { return s.config(); },
                     [](CudaqSimulator& s, CudaqConfig c) { s.set_config(std::move(c)); })
        .def("run", &CudaqSimulator::run,
            "commands"_a, "n_qubits"_a, "n_shots"_a, "seed"_a = nb::none())
        .def("batch_run", &CudaqSimulator::batch_run,
            "commands"_a, "n_qubits"_a, "n_shots"_a, "param_sets"_a, "seed"_a = nb::none())
        .def("statevector", [](const CudaqSimulator& sim,
                               const std::vector<Command>& cmds,
                               int n_qubits) {
            return statevector_to_numpy(sim.statevector(cmds, n_qubits));
        }, "commands"_a, "n_qubits"_a,
           "Host statevector as a zero-copy 1-D complex128 numpy array.")
        .def("batch_expectation", &CudaqSimulator::batch_expectation,
            "commands"_a, "n_qubits"_a, "observable"_a, "param_sets"_a,
            "Sampling-free <H> swept over parameter sets (CUDA-Q observe). "
            "One expectation value per parameter set.");
}
