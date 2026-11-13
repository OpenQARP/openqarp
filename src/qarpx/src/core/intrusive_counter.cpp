// Single translation unit providing the out-of-line definitions of nanobind's
// intrusive reference counter (intrusive_counter::inc_ref/dec_ref/set_self_py,
// the intrusive_inc_ref_py / intrusive_dec_ref_py function pointers, and
// nb::intrusive_init).
//
// counter.inl must be compiled in exactly ONE translation unit across the whole
// link.  The core qarpx library owns it so that:
//   * the C++ test binaries (which link qarpx but not the Python module) resolve
//     the symbols and run in pure-C++ atomic-refcount mode — intrusive_init is
//     never called there, so the Python function pointers stay null and are
//     never dereferenced; and
//   * the Python module (which links qarpx statically) reuses these symbols and
//     only needs to call nb::intrusive_init(...) once at import — see bindings.cpp.
//
// It depends on neither Python.h nor nanobind's runtime: PyObject appears only
// opaquely (forward-declared in counter.h) and the Py_INCREF/Py_DECREF wrappers
// are supplied at runtime via intrusive_init.
#include <nanobind/intrusive/counter.inl>
