"""Tests for symbolic (sympy → SymEngine) operator coefficients.

Most tests require a qarpx built with ``QARP_WITH_SYMENGINE=ON`` and skip
otherwise; the final test asserts the documented error path of a numeric-only
build.  Detection: the ``substitute`` method is bound only when the backend
is compiled in.
"""

import copy
import pickle

import pytest
import sympy

from qarp.operators import FermionOperator, QubitOperator
from qarp.operators.functions import (
    hermitian_conjugated,
    jordan_wigner,
)

SYMENGINE_ENABLED = hasattr(QubitOperator, "substitute")

needs_symengine = pytest.mark.skipif(
    not SYMENGINE_ENABLED,
    reason="qarpx built without SymEngine (QARP_WITH_SYMENGINE=OFF)",
)

a, b = sympy.symbols("a b")


def sym_equal(x, y):
    return sympy.simplify(sympy.sympify(x) - sympy.sympify(y)) == 0


@needs_symengine
def test_symbolic_construction_and_terms_round_trip():
    op = QubitOperator("X0 Z1", a) + QubitOperator("Y0", 2.0 * b) + QubitOperator("Z0", 0.5)
    assert op.is_symbolic()
    assert op.free_symbols() == ["a", "b"]
    terms = dict(op.terms)
    assert sym_equal(terms[((0, "X"), (1, "Z"))], a)
    assert sym_equal(terms[((0, "Y"),)], 2.0 * b)
    assert complex(terms[((0, "Z"),)]) == 0.5  # numeric coefficients stay complex


@needs_symengine
def test_promotion_on_mixed_arithmetic():
    numeric = QubitOperator("X0", 0.5)
    assert not numeric.is_symbolic()
    mixed = numeric + QubitOperator("Y0", a)
    assert mixed.is_symbolic()
    assert (QubitOperator("X0") * QubitOperator("Y0", a)).is_symbolic()

    # Scalar sympy dunders promote too.
    assert (a * QubitOperator("X0")).is_symbolic()
    assert (QubitOperator("X0") * a).is_symbolic()
    assert (QubitOperator("X0") + a).is_symbolic()
    assert (a - QubitOperator("X0")).is_symbolic()


@needs_symengine
def test_symbolic_products_expand_correctly():
    op = QubitOperator("X0", a) * QubitOperator("Y0", b)
    ((term, coeff),) = op.terms.items()
    assert term == ((0, "Z"),)
    assert sym_equal(coeff, sympy.I * a * b)


@needs_symengine
def test_hermitian_conjugated_conjugates_symbols():
    hc = hermitian_conjugated(QubitOperator("X0", a))
    ((_, coeff),) = hc.terms.items()
    assert sym_equal(coeff, sympy.conjugate(a))

    f = hermitian_conjugated(FermionOperator("2^ 1", a))
    ((key, coeff),) = f.terms.items()
    assert key == ((1, 1), (2, 0))
    assert sym_equal(coeff, sympy.conjugate(a))


@needs_symengine
def test_substitute_and_demotion():
    op = QubitOperator("X0", a) + QubitOperator("Y1", 2 * b) + QubitOperator("Z2", 0.5)

    partial = op.substitute({"a": 0.25})
    assert partial.is_symbolic()
    assert partial.free_symbols() == ["b"]

    full = partial.substitute({b: 1.5})  # sympy Symbol keys work too
    assert not full.is_symbolic()
    expected = QubitOperator("X0", 0.25) + QubitOperator("Y1", 3.0) + QubitOperator("Z2", 0.5)
    assert full == expected


@needs_symengine
def test_symbolic_jordan_wigner_matches_numeric_after_substitution():
    f = FermionOperator("2^ 1", a) + FermionOperator("0^", 2.0)
    jw_sym = jordan_wigner(f)
    assert jw_sym.is_symbolic()

    value = 0.75 - 0.5j
    jw_num = jordan_wigner(FermionOperator("2^ 1", value) + FermionOperator("0^", 2.0))
    assert jw_sym.substitute({"a": value}) == jw_num


@needs_symengine
def test_symbolic_coefficients_never_auto_erased():
    op = QubitOperator("X0", a)
    op += QubitOperator("X0", 1e-12)  # tiny numeric shift, still symbolic
    assert len(op.terms) == 1

    cancelled = QubitOperator("X0", a) - QubitOperator("X0", a)
    assert len(cancelled.terms) == 0  # exact cancellation simplifies to 0


@needs_symengine
def test_terms_setter_with_sympy_values():
    op = QubitOperator()
    op.terms = {((0, "X"),): a, (): 0.5}
    assert op.is_symbolic()
    assert sym_equal(op.terms[((0, "X"),)], a)
    assert complex(op.terms[()]) == 0.5


@needs_symengine
def test_copy_and_pickle_symbolic():
    op = QubitOperator("X0 Z1", 2 * a) + QubitOperator("", 1.5)
    for clone in [copy.deepcopy(op), pickle.loads(pickle.dumps(op))]:
        assert clone.is_symbolic()
        assert clone == op


@needs_symengine
def test_sparse_requires_numeric():
    op = QubitOperator("X0", a)
    with pytest.raises(ValueError, match="symbolic"):
        op.sparse_matrix()
    dense = op.substitute({"a": 2.0}).sparse_matrix().toarray()
    assert dense[0, 1] == 2.0


@needs_symengine
def test_mappings_accept_symbolic_fermion_operators():
    from qarp.operators import BravyiKitaev, JordanWigner, Parity

    f = FermionOperator("2^ 1", a)
    for encoded in [
        JordanWigner().encode_operator(f),
        BravyiKitaev().encode_operator(f),
        Parity(3).encode_operator(f),
    ]:
        assert encoded.is_symbolic()
        assert encoded.free_symbols() == ["a"]


@needs_symengine
def test_openfermion_differential_with_sympy_coefficients():
    openfermion = pytest.importorskip("openfermion")

    ours = QubitOperator("X0", a) * QubitOperator("Y0 Z1", 2 * b)
    theirs = openfermion.QubitOperator("X0", a) * openfermion.QubitOperator("Y0 Z1", 2 * b)
    assert list(ours.terms.keys()) == list(theirs.terms.keys())
    for key, coeff in ours.terms.items():
        assert sym_equal(coeff, theirs.terms[key])


@pytest.mark.skipif(SYMENGINE_ENABLED, reason="numeric-only build required")
def test_symbolic_raises_without_backend():
    with pytest.raises(RuntimeError, match="SymEngine"):
        QubitOperator("X0", a)
    with pytest.raises(RuntimeError, match="SymEngine"):
        FermionOperator("2^", a)
