// Bindings for the openfermion-compatible operator classes
// (qarpx::ops::QubitOperator / FermionOperator) and their free functions.
// Separate TU from bindings.cpp to keep both manageable; registered from
// NB_MODULE via register_operator_bindings(m).
//
// Deviations from openfermion (all documented in qarp/operators/*.py):
//   - .terms returns a version-cached read-only MappingProxyType, not the
//     live dict: reads are identical and cheap; unsupported in-place
//     mutation fails loudly with TypeError.  Full reassignment
//     (`op.terms = {...}`) and `dict(op.terms)` are the escape hatches.
//   - .terms coefficients are Python complex (openfermion keeps whatever
//     numeric type was stored); symbolic coefficients come back as sympy
//     expressions.
//   - sympy *numeric* expressions are accepted as coefficients and demoted
//     to complex; symbolic expressions engage the SymEngine backend
//     (QARP_WITH_SYMENGINE) or raise when built without it.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "qarpx/operators/fermion_operator.h"
#include "qarpx/operators/qubit_operator.h"
#include "qarpx/operators/sparse.h"
#include "qarpx/operators/transforms.h"

#include <complex>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using qarpx::ops::FermionKey;
using qarpx::ops::FermionOperator;
using qarpx::ops::PackedPauli;
using qarpx::ops::QubitOperator;
using Complex = std::complex<double>;

namespace {

// Move a vector into an owning 1-D numpy array (zero copy; a capsule frees
// the buffer with the array) — same pattern as bindings.cpp's
// statevector_to_numpy.
template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> vec_to_ndarray(std::vector<T>&& v) {
    auto* heap = new std::vector<T>(std::move(v));
    nb::capsule owner(heap, [](void* p) noexcept {
        delete static_cast<std::vector<T>*>(p);
    });
    const size_t shape[1] = {heap->size()};
    return nb::ndarray<nb::numpy, T, nb::ndim<1>>(heap->data(), 1, shape, owner);
}

nb::tuple coo_to_python(qarpx::ops::SparseCoo&& coo) {
    return nb::make_tuple(vec_to_ndarray(std::move(coo.data)),
                          vec_to_ndarray(std::move(coo.rows)),
                          vec_to_ndarray(std::move(coo.cols)), coo.dim);
}

// ── Coefficient ingestion ─────────────────────────────────────────────────

/// A Python coefficient: numeric (int/float/complex/numpy scalar/sympy
/// number) or, with the SymEngine backend, a symbolic sympy expression.
struct PyCoeff {
    Complex value{1.0, 0.0};
#ifdef QARP_WITH_SYMENGINE
    std::optional<qarpx::ops::SymCoeff> symbolic;
    bool is_symbolic() const { return symbolic.has_value(); }
#else
    static constexpr bool is_symbolic() { return false; }
#endif
};

PyCoeff parse_py_coefficient(nb::handle c) {
    PyCoeff out;
    if (nb::try_cast<Complex>(c, out.value)) return out;
    // sympy expression: numeric ones demote to complex; symbolic ones need
    // the SymEngine backend.
    if (nb::hasattr(c, "free_symbols") && nb::hasattr(c, "is_number")) {
        if (nb::cast<bool>(nb::getattr(c, "is_number"))) {
            nb::object as_complex =
                nb::module_::import_("builtins").attr("complex")(c);
            out.value = nb::cast<Complex>(as_complex);
            return out;
        }
#ifdef QARP_WITH_SYMENGINE
        out.symbolic = qarpx::ops::parse_symbolic(nb::cast<std::string>(nb::str(c)));
        return out;
#else
        throw std::runtime_error(
            "qarpx built without SymEngine; symbolic coefficients unavailable");
#endif
    }
    throw nb::value_error("Coefficient must be a numeric type.");
}

Complex coefficient_to_complex(nb::handle c) {
    const PyCoeff pc = parse_py_coefficient(c);
    if (pc.is_symbolic())
        throw nb::value_error("Coefficient must be numeric here, not symbolic.");
    return pc.value;
}

// ── Coefficient emission ──────────────────────────────────────────────────

nb::object coeff_to_python(const Complex& c) { return nb::cast(c); }

#ifdef QARP_WITH_SYMENGINE
nb::object coeff_to_python(const qarpx::ops::SymCoeff& c) {
    if (c.is_numeric())
        return nb::cast(qarpx::ops::CoeffTraits<qarpx::ops::SymCoeff>::to_complex(c));
    return nb::module_::import_("sympy").attr("sympify")(
        qarpx::ops::symbolic_to_string(c));
}
#endif

// ── Term-key ingestion (openfermion tuple forms) ──────────────────────────

// ((0,'X'), (1,'Y')) or the single-factor shorthand (0,'X').
std::vector<std::pair<uint32_t, char>> qubit_factors_from(nb::handle term) {
    nb::sequence seq = nb::cast<nb::sequence>(term);
    const size_t n = nb::len(seq);
    nb::list flat;
    const bool single_factor =
        n == 2 && nb::isinstance<nb::int_>(nb::object(seq[0]));
    if (single_factor) {
        flat.append(term);
    } else {
        for (size_t i = 0; i < n; ++i) flat.append(seq[i]);
    }

    std::vector<std::pair<uint32_t, char>> factors;
    factors.reserve(nb::len(flat));
    for (nb::handle f : flat) {
        nb::sequence fs;
        if (!nb::try_cast<nb::sequence>(f, fs) || nb::len(fs) != 2)
            throw nb::value_error("term specified incorrectly.");
        int64_t qubit = 0;
        if (!nb::try_cast<int64_t>(nb::object(fs[0]), qubit) || qubit < 0)
            throw nb::value_error("Invalid qubit index in term.");
        std::string action;
        if (!nb::try_cast<std::string>(nb::object(fs[1]), action) || action.size() != 1)
            throw nb::value_error("Invalid action provided to term.");
        factors.emplace_back(static_cast<uint32_t>(qubit), action[0]);
    }
    return factors;
}

// ((2,1), (1,0)) or the single-factor shorthand (2,1); action ∈ {0, 1}.
std::vector<std::pair<uint32_t, uint32_t>> ladder_ops_from(nb::handle term) {
    nb::sequence seq = nb::cast<nb::sequence>(term);
    const size_t n = nb::len(seq);
    nb::list flat;
    const bool single_factor =
        n == 2 && nb::isinstance<nb::int_>(nb::object(seq[0]));
    if (single_factor) {
        flat.append(term);
    } else {
        for (size_t i = 0; i < n; ++i) flat.append(seq[i]);
    }

    std::vector<std::pair<uint32_t, uint32_t>> ops;
    ops.reserve(nb::len(flat));
    for (nb::handle f : flat) {
        nb::sequence fs;
        if (!nb::try_cast<nb::sequence>(f, fs) || nb::len(fs) != 2)
            throw nb::value_error("term specified incorrectly.");
        int64_t index = 0, action = 0;
        if (!nb::try_cast<int64_t>(nb::object(fs[0]), index) || index < 0)
            throw nb::value_error("Invalid mode index in term.");
        if (!nb::try_cast<int64_t>(nb::object(fs[1]), action) ||
            (action != 0 && action != 1))
            throw nb::value_error("Invalid action provided to term.");
        ops.emplace_back(static_cast<uint32_t>(index), static_cast<uint32_t>(action));
    }
    return ops;
}

QubitOperator make_qubit_operator(nb::handle term, nb::handle coefficient) {
    const PyCoeff coeff = parse_py_coefficient(coefficient);
#ifdef QARP_WITH_SYMENGINE
    if (coeff.is_symbolic()) {
        if (term.is_none()) return QubitOperator();
        if (nb::isinstance<nb::str>(term))
            return QubitOperator::from_term_string(nb::cast<std::string>(term),
                                                   *coeff.symbolic);
        if (nb::isinstance<nb::tuple>(term) || nb::isinstance<nb::list>(term))
            return QubitOperator::from_factors(qubit_factors_from(term),
                                               *coeff.symbolic);
        throw nb::value_error("term specified incorrectly.");
    }
#endif
    if (term.is_none()) return QubitOperator();
    if (nb::isinstance<nb::str>(term))
        return QubitOperator::from_term_string(nb::cast<std::string>(term), coeff.value);
    if (nb::isinstance<nb::tuple>(term) || nb::isinstance<nb::list>(term))
        return QubitOperator::from_factors(qubit_factors_from(term), coeff.value);
    throw nb::value_error("term specified incorrectly.");
}

FermionOperator make_fermion_operator(nb::handle term, nb::handle coefficient) {
    const PyCoeff coeff = parse_py_coefficient(coefficient);
#ifdef QARP_WITH_SYMENGINE
    if (coeff.is_symbolic()) {
        if (term.is_none()) return FermionOperator();
        if (nb::isinstance<nb::str>(term))
            return FermionOperator::from_term_string(nb::cast<std::string>(term),
                                                     *coeff.symbolic);
        if (nb::isinstance<nb::tuple>(term) || nb::isinstance<nb::list>(term))
            return FermionOperator::from_ladder_ops(ladder_ops_from(term),
                                                    *coeff.symbolic);
        throw nb::value_error("term specified incorrectly.");
    }
#endif
    if (term.is_none()) return FermionOperator();
    if (nb::isinstance<nb::str>(term))
        return FermionOperator::from_term_string(nb::cast<std::string>(term), coeff.value);
    if (nb::isinstance<nb::tuple>(term) || nb::isinstance<nb::list>(term))
        return FermionOperator::from_ladder_ops(ladder_ops_from(term), coeff.value);
    throw nb::value_error("term specified incorrectly.");
}

// ── Term-key emission (openfermion .terms dict keys) ──────────────────────

nb::tuple qubit_key_tuple(const PackedPauli& key) {
    std::vector<std::pair<uint32_t, char>> factors;
    qarpx::ops::packed_to_index_pairs(key, factors);
    nb::list items;
    for (const auto& [qubit, letter] : factors)
        items.append(nb::make_tuple(qubit, std::string(1, letter)));
    return nb::tuple(items);
}

nb::tuple fermion_key_tuple(const FermionKey& key) {
    nb::list items;
    for (const uint32_t packed : key.ops)
        items.append(nb::make_tuple(FermionKey::index_of(packed),
                                    FermionKey::action_of(packed)));
    return nb::tuple(items);
}

// ── Shared binding machinery ──────────────────────────────────────────────

// Per-class glue used by the generic bind_operator_common below.
struct QubitGlue {
    using Op = QubitOperator;
    static nb::tuple key_tuple(const PackedPauli& key) { return qubit_key_tuple(key); }
    // Canonical single-term operator for a .terms key (folds/canonicalizes
    // exactly like the constructor, so assigned keys normalize).
    static Op single_term(nb::handle key, const PyCoeff& coeff) {
#ifdef QARP_WITH_SYMENGINE
        if (coeff.is_symbolic()) {
            if (nb::len(nb::cast<nb::sequence>(key)) == 0)
                return Op::from_packed_term(PackedPauli{}, *coeff.symbolic);
            return Op::from_factors(qubit_factors_from(key), *coeff.symbolic);
        }
#endif
        if (nb::len(nb::cast<nb::sequence>(key)) == 0) return Op::identity(coeff.value);
        return Op::from_factors(qubit_factors_from(key), coeff.value);
    }
};

struct FermionGlue {
    using Op = FermionOperator;
    static nb::tuple key_tuple(const FermionKey& key) { return fermion_key_tuple(key); }
    static Op single_term(nb::handle key, const PyCoeff& coeff) {
#ifdef QARP_WITH_SYMENGINE
        if (coeff.is_symbolic()) {
            if (nb::len(nb::cast<nb::sequence>(key)) == 0)
                return Op::from_key(FermionKey{}, *coeff.symbolic);
            return Op::from_ladder_ops(ladder_ops_from(key), *coeff.symbolic);
        }
#endif
        if (nb::len(nb::cast<nb::sequence>(key)) == 0) return Op::identity(coeff.value);
        return Op::from_ladder_ops(ladder_ops_from(key), coeff.value);
    }
};

template <typename Glue>
nb::dict terms_dict(const typename Glue::Op& op) {
    nb::dict d;
    op.visit([&d](const auto& eng) {
        for (const auto& e : eng.terms)
            d[Glue::key_tuple(e.key)] = coeff_to_python(e.coeff);
    });
    return d;
}

/// Merge one single-term operator's payload into an accumulating operator
/// via dict-assignment semantics (set: keeps zeros, appends new keys).
template <typename Op>
void set_single_term(Op& target, const Op& single) {
#ifdef QARP_WITH_SYMENGINE
    if (single.is_symbolic() && !target.is_symbolic()) target.promote_to_symbolic();
    if (target.is_symbolic()) {
        Op promoted(single);
        promoted.promote_to_symbolic();
        for (const auto& e : promoted.symbolic().terms)
            target.symbolic().terms.set(e.key, e.coeff);
        return;
    }
#endif
    for (const auto& e : single.numeric().terms)
        target.numeric().terms.set(e.key, e.coeff);
}

template <typename Glue>
void assign_terms(typename Glue::Op& op, nb::handle mapping) {
    using Op = typename Glue::Op;
    // Validate + normalize into a fresh operator first so a bad mapping
    // can't leave `op` half-assigned.
    Op fresh;
    nb::object items = nb::getattr(mapping, "items")();
    for (nb::handle item : items) {
        nb::sequence kv = nb::cast<nb::sequence>(item);
        const PyCoeff coeff = parse_py_coefficient(nb::object(kv[1]));
        set_single_term(fresh, Glue::single_term(nb::object(kv[0]), coeff));
    }
    op = std::move(fresh);
}

/// Scalar dunder dispatch for arbitrary Python scalars (sympy expressions
/// engage the symbolic backend; junk raises TypeError like openfermion).
template <typename Op>
PyCoeff scalar_or_type_error(nb::handle s) {
    try {
        return parse_py_coefficient(s);
    } catch (const nb::builtin_exception&) {
        throw nb::type_error("Cannot combine operator with a non-numeric object.");
    }
}

template <typename Op>
Op scalar_mul(const Op& a, nb::handle s) {
    const PyCoeff c = scalar_or_type_error<Op>(s);
    Op out(a);
#ifdef QARP_WITH_SYMENGINE
    if (c.is_symbolic()) {
        out.imul_scalar(*c.symbolic);
        return out;
    }
#endif
    out *= c.value;
    return out;
}

template <typename Op>
Op scalar_add(const Op& a, nb::handle s, bool negate_operator, bool negate_scalar) {
    const PyCoeff c = scalar_or_type_error<Op>(s);
    Op out = negate_operator ? -a : Op(a);
#ifdef QARP_WITH_SYMENGINE
    if (c.is_symbolic()) {
        out.iadd_scalar(negate_scalar ? qarpx::ops::SymCoeff(-c.symbolic->expr)
                                      : *c.symbolic);
        return out;
    }
#endif
    out += negate_scalar ? -c.value : c.value;
    return out;
}

template <typename Glue>
void bind_operator_common(nb::class_<typename Glue::Op>& cls) {
    using Op = typename Glue::Op;

    cls.def(
        "__init__",
        [](Op* self, nb::handle term, nb::handle coefficient) {
            if constexpr (std::is_same_v<Op, QubitOperator>)
                new (self) Op(make_qubit_operator(term, coefficient));
            else
                new (self) Op(make_fermion_operator(term, coefficient));
        },
        "term"_a = nb::none(), "coefficient"_a = 1.0,
        "openfermion-compatible constructor: term is None (zero operator), a "
        "term string, or a factor tuple; coefficient is any numeric type or "
        "a sympy expression (SymEngine backend).");

    // .terms — version-cached read-only view (see file header).
    cls.def_prop_rw(
        "terms",
        [](nb::handle_t<Op> self_h) -> nb::object {
            Op& self = nb::cast<Op&>(self_h);
            const uint64_t version =
                self.visit([](const auto& eng) { return eng.terms.version(); });
            nb::object cache = nb::getattr(self_h, "_terms_cache", nb::none());
            if (!cache.is_none()) {
                nb::tuple t = nb::cast<nb::tuple>(cache);
                if (nb::cast<uint64_t>(t[0]) == version &&
                    nb::cast<bool>(t[2]) == self.is_symbolic())
                    return t[1];
            }
            nb::object proxy = nb::module_::import_("types").attr(
                "MappingProxyType")(terms_dict<Glue>(self));
            nb::setattr(self_h, "_terms_cache",
                        nb::make_tuple(version, proxy, self.is_symbolic()));
            return proxy;
        },
        [](Op& self, nb::handle mapping) { assign_terms<Glue>(self, mapping); },
        "Term dictionary (read-only view; assign a full dict to replace).");

    cls.def(
        "coefficient",
        [](const Op& self, nb::handle term) -> nb::object {
            const Op single = Glue::single_term(term, PyCoeff{});
            nb::object out;
            bool found = false;
            self.visit([&](const auto& eng) {
                for (const auto& e : single.numeric().terms) {
                    if (const auto* c = eng.terms.find(e.key)) {
                        out = coeff_to_python(*c);
                        found = true;
                    }
                    break;
                }
            });
            if (!found) throw nb::key_error(nb::repr(term).c_str());
            return out;
        },
        "term"_a,
        "Fast single-term coefficient lookup (qarpx extra, not openfermion).");

    // Arithmetic dunders — operator·operator and numeric scalars (both
    // sides) via nb::self; arbitrary scalars (sympy) via handle overloads
    // that nanobind tries after these.
    cls.def(nb::self + nb::self)
        .def(nb::self - nb::self)
        .def(nb::self * nb::self)
        .def(nb::self += nb::self)
        .def(nb::self -= nb::self)
        .def(nb::self *= nb::self)
        .def(nb::self + Complex())
        .def(Complex() + nb::self)
        .def(nb::self - Complex())
        .def(Complex() - nb::self)
        .def(nb::self * Complex())
        .def(Complex() * nb::self)
        .def(nb::self / Complex())
        .def(nb::self += Complex())
        .def(nb::self -= Complex())
        .def(nb::self *= Complex())
        .def(nb::self /= Complex())
        .def(-nb::self)
        .def(nb::self == nb::self)
        .def(nb::self != nb::self);

    cls.def("__mul__", [](const Op& a, nb::handle s) { return scalar_mul(a, s); })
        .def("__rmul__", [](const Op& a, nb::handle s) { return scalar_mul(a, s); })
        .def("__add__",
             [](const Op& a, nb::handle s) { return scalar_add(a, s, false, false); })
        .def("__radd__",
             [](const Op& a, nb::handle s) { return scalar_add(a, s, false, false); })
        .def("__sub__",
             [](const Op& a, nb::handle s) { return scalar_add(a, s, false, true); })
        .def("__rsub__",
             [](const Op& a, nb::handle s) { return scalar_add(a, s, true, false); });

    cls.def("__pow__", &Op::pow, "exponent"_a);

    // openfermion operators are unhashable (__hash__ = None).
    cls.attr("__hash__") = nb::none();

    cls.def("isclose", &Op::isclose, "other"_a, "tol"_a = qarpx::ops::kEqTolerance)
        .def("hermitian_conjugated", &Op::hermitian_conjugated)
        .def("count_qubits", &Op::count_qubits)
        .def("compress", &Op::compress, "abs_tol"_a = qarpx::ops::kEqTolerance)
        .def("get_operators",
             [](const Op& self) {
                 std::vector<Op> out = self.get_operators();
                 return out;
             })
        .def("is_symbolic", &Op::is_symbolic)
        .def_prop_rw(
            "constant", [](const Op& self) { return self.constant(); },
            [](Op& self, nb::handle value) {
                self.set_constant(coefficient_to_complex(value));
            })
        .def_static("zero", []() { return Op(); })
        .def_static(
            "identity",
            [](nb::handle coefficient) {
                return Op::identity(coefficient_to_complex(coefficient));
            },
            "coefficient"_a = 1.0)
        .def("__str__", &Op::str)
        .def("__repr__", &Op::str)
        .def("__copy__", [](const Op& self) { return Op(self); })
        .def(
            "__deepcopy__", [](const Op& self, nb::handle) { return Op(self); },
            "memo"_a)
        .def("__getstate__",
             [](const Op& self) {
                 nb::list state;
                 self.visit([&state](const auto& eng) {
                     for (const auto& e : eng.terms)
                         state.append(nb::make_tuple(Glue::key_tuple(e.key),
                                                     coeff_to_python(e.coeff)));
                 });
                 return state;
             })
        .def("__setstate__", [](Op& self, nb::list state) {
            new (&self) Op();
            for (nb::handle item : state) {
                nb::sequence kv = nb::cast<nb::sequence>(item);
                const PyCoeff coeff = parse_py_coefficient(nb::object(kv[1]));
                set_single_term(self, Glue::single_term(nb::object(kv[0]), coeff));
            }
        });

#ifdef QARP_WITH_SYMENGINE
    cls.def(
           "substitute",
           [](const Op& self, nb::handle values) {
               std::vector<std::pair<std::string, Complex>> subs;
               nb::object items = nb::getattr(values, "items")();
               for (nb::handle item : items) {
                   nb::sequence kv = nb::cast<nb::sequence>(item);
                   std::string name;
                   nb::object key = nb::object(kv[0]);
                   if (!nb::try_cast<std::string>(key, name))
                       name = nb::cast<std::string>(nb::str(key));
                   subs.emplace_back(name, coefficient_to_complex(nb::object(kv[1])));
               }
               return self.substituted(subs);
           },
           "values"_a,
           "Substitute named symbols with numeric values; demotes to the "
           "numeric backend when no free symbols remain (qarpx extra).")
        .def("free_symbols", [](const Op& self) { return self.free_symbols(); },
             "Sorted names of the free symbols across all coefficients.");
#endif
}

}  // namespace

namespace qarpx::python {

void register_operator_bindings(nb::module_& m) {
    nb::class_<QubitOperator> qubit_cls(
        m, "QubitOperator", nb::dynamic_attr(),
        "openfermion-compatible QubitOperator backed by a packed binary-"
        "symplectic C++ core (PauliEngine design, arXiv:2601.02233).");
    bind_operator_common<QubitGlue>(qubit_cls);

    nb::class_<FermionOperator> fermion_cls(
        m, "FermionOperator", nb::dynamic_attr(),
        "openfermion-compatible FermionOperator backed by a packed ladder-"
        "sequence C++ core.");
    bind_operator_common<FermionGlue>(fermion_cls);

    m.def("hermitian_conjugated",
          [](const QubitOperator& op) { return op.hermitian_conjugated(); },
          "op"_a);
    m.def("hermitian_conjugated",
          [](const FermionOperator& op) { return op.hermitian_conjugated(); },
          "op"_a);
    m.def("count_qubits", [](const QubitOperator& op) { return op.count_qubits(); },
          "op"_a);
    m.def("count_qubits",
          [](const FermionOperator& op) { return op.count_qubits(); }, "op"_a);
    m.def("is_hermitian",
          [](const QubitOperator& op) { return op.is_hermitian(); }, "op"_a);
    m.def(
        "is_hermitian",
        [](const FermionOperator&) -> bool {
            PyErr_SetString(PyExc_NotImplementedError,
                            "is_hermitian(FermionOperator) requires normal "
                            "ordering, which qarpx does not implement; qarp "
                            "only uses the QubitOperator path.");
            throw nb::python_error();
        },
        "op"_a);

    // ── Fermion→qubit transforms (scalar + batch overloads; one Python
    //    crossing for Mapping.encode(list)) ──
    m.def("jordan_wigner",
          [](const FermionOperator& op) { return qarpx::ops::jordan_wigner(op); },
          "op"_a);
    m.def("jordan_wigner",
          [](const std::vector<FermionOperator>& ops) {
              std::vector<QubitOperator> out;
              out.reserve(ops.size());
              for (const auto& op : ops) out.push_back(qarpx::ops::jordan_wigner(op));
              return out;
          },
          "ops"_a);
    m.def("bravyi_kitaev",
          [](const FermionOperator& op, nb::handle n_qubits) {
              return qarpx::ops::bravyi_kitaev(
                  op, n_qubits.is_none() ? -1 : nb::cast<int>(n_qubits));
          },
          "op"_a, "n_qubits"_a = nb::none());
    m.def("bravyi_kitaev",
          [](const std::vector<FermionOperator>& ops, nb::handle n_qubits) {
              const int n = n_qubits.is_none() ? -1 : nb::cast<int>(n_qubits);
              std::vector<QubitOperator> out;
              out.reserve(ops.size());
              for (const auto& op : ops) out.push_back(qarpx::ops::bravyi_kitaev(op, n));
              return out;
          },
          "ops"_a, "n_qubits"_a = nb::none());
    m.def("parity_transform",
          [](const FermionOperator& op, int n_qubits) {
              return qarpx::ops::parity_transform(op, n_qubits);
          },
          "op"_a, "n_qubits"_a);
    m.def("parity_transform",
          [](const std::vector<FermionOperator>& ops, int n_qubits) {
              std::vector<QubitOperator> out;
              out.reserve(ops.size());
              for (const auto& op : ops)
                  out.push_back(qarpx::ops::parity_transform(op, n_qubits));
              return out;
          },
          "ops"_a, "n_qubits"_a);

    // ── Simulator observable ABI fast path: build the whole
    //    [(sparse [(qubit, 'X'|'Y'|'Z')], coeff)] list in C++ in one Python
    //    crossing (replaces the per-term Python loop over .terms) ──
    m.def("qubit_operator_to_observable",
          [](const QubitOperator& op) {
              std::vector<std::pair<std::vector<std::pair<uint32_t, char>>, Complex>> out;
              op.visit([&out](const auto& eng) { eng.to_observable(out); });
              return out;
          },
          "op"_a,
          "The qarpx simulator Pauli-sum format: list of (sparse "
          "[(qubit, letter)] term, complex coefficient) pairs.");

    // ── Sparse-matrix COO kernels (scipy assembly happens in
    //    qarp/operators/{qubit,fermion}_operator.py::sparse_matrix and, for
    //    the MSB openfermion layout, qarp/operators/compat.py) ──
    m.def("qubit_operator_coo",
          [](const QubitOperator& op, nb::handle n_qubits, bool msb) {
              return coo_to_python(qarpx::ops::qubit_operator_coo(
                  op, n_qubits.is_none() ? -1 : nb::cast<int>(n_qubits),
                  msb ? qarpx::ops::BitOrder::kMsb : qarpx::ops::BitOrder::kLsb));
          },
          "op"_a, "n_qubits"_a = nb::none(), "msb"_a = false,
          "COO triplets (data, rows, cols, dim) of the operator's matrix, one "
          "per non-zero cell.  Default is the qarpx LSB ordering "
          "(qubit q ↔ bit q); msb=True realizes openfermion's MSB qubit-0 "
          "layout for interop.");
    m.def("fermion_operator_coo",
          [](const FermionOperator& op, nb::handle n_qubits, bool msb) {
              return coo_to_python(qarpx::ops::fermion_operator_coo(
                  op, n_qubits.is_none() ? -1 : nb::cast<int>(n_qubits),
                  msb ? qarpx::ops::BitOrder::kMsb : qarpx::ops::BitOrder::kLsb));
          },
          "op"_a, "n_qubits"_a = nb::none(), "msb"_a = false);
}

}  // namespace qarpx::python
