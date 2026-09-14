"""Differential tests: qarp.operators.FermionOperator vs openfermion.

Same bar as test_qubit_operator.py: identical operation sequences on both
implementations, ``.terms`` compared exactly including insertion order.
The fermion-specific semantics under test: verbatim ladder sequences (no
normal ordering, ever), concatenating products, and dagger reverse+flip.

Shares the one deviation documented in test_qubit_operator.py — both operators
sit on the same term engine — so the sub-tolerance ``+=`` cases live in
``test_add_deviates_from_openfermion_below_eq_tolerance`` there, not in the
parity scripts here.
"""

import copy
import pickle

import numpy as np
import pytest

openfermion = pytest.importorskip("openfermion")

from qarp.operators import FermionOperator
from qarp.operators.functions import count_qubits, hermitian_conjugated

OFFermionOperator = openfermion.FermionOperator


def norm_terms(op):
    return [(key, complex(value)) for key, value in op.terms.items()]


def assert_same(ours, theirs):
    __tracebackhide__ = True
    assert norm_terms(ours) == norm_terms(theirs)


def both(*args, **kwargs):
    return FermionOperator(*args, **kwargs), OFFermionOperator(*args, **kwargs)


# ── Constructors ──────────────────────────────────────────────────────────

CTOR_CASES = [
    (),
    ("",),
    ("", 2.0),
    ("2^ 1",),
    ("0^", 0.0),  # zero coefficient is kept
    ("2^ 1 5^ 3", -0.5),
    ("1 2^",),  # annihilation before creation — stored verbatim
    ("0 0",),
    ("5^ 5^ ",),  # trailing whitespace (used in qarp tests)
    (((2, 1), (0, 0)), -1.0),
    ((2, 1),),  # single-factor shorthand
    ([(3, 1), (3, 0)], 0.25j),
    ("2^ 1", np.float64(0.5)),
    ("1.5 [2^ 3]", 2.0),  # bracketed string form
]


@pytest.mark.parametrize("args", CTOR_CASES, ids=[repr(c) for c in CTOR_CASES])
def test_constructors_match(args):
    ours, theirs = both(*args)
    assert_same(ours, theirs)


@pytest.mark.parametrize("bad", ["a", "^", "2^^", "2x", 5, ((0, 2),)])
def test_invalid_constructors_raise_valueerror_in_both(bad):
    with pytest.raises(ValueError):
        OFFermionOperator(bad)
    with pytest.raises(ValueError):
        FermionOperator(bad)


# ── Fermion-specific algebra ──────────────────────────────────────────────


def _script_products(cls):
    a = cls("2^ 1", 2.0) + cls("0^", -0.5)
    b = cls("3^ 0", 0.5) + cls("1", 1j)
    c = a * b  # concatenation, no reordering
    d = b * a
    e = a * a
    return [a, b, c, d, e]


def _script_add_sub(cls):
    a = cls("2^ 1", 0.75)
    a += cls("1 2^", -0.25)  # different key from "2^ 1"
    a -= cls("2^ 1", 0.75)  # exact cancellation
    d = cls("2^") + cls("2^ 1")
    e = d * (cls("1 3^") + cls("3^", -1.0))  # colliding concatenations → zero kept
    f = cls("2^ 1")
    f += 2.0
    f -= 2.0  # explicit zero constant survives
    return [a, d, e, f]


def test_add_deviation_matches_the_qubit_operator_engine():
    """FermionOperator shares ``TermOperator``, so it inherits the exact-zero
    ``+=`` rule; pinned here so a divergence between the two facades is caught.

    Oracle: the arithmetic — ``1.0 + (-1.0 + 5e-9)`` is ``5e-9``.
    """
    ours = FermionOperator("0^") + FermionOperator("0^", -1.0 + 5e-9)
    theirs = OFFermionOperator("0^") + OFFermionOperator("0^", -1.0 + 5e-9)
    assert [complex(v) for v in ours.terms.values()] == pytest.approx([5e-9 + 0j], rel=1e-7)
    assert list(theirs.terms.values()) == []

    ours.compress()
    assert list(ours.terms.values()) == []


def _script_scalars(cls):
    a = cls("2^ 1", 0.75) + cls("0", -2.0)
    return [a * 2, 2 * a, a * 0.5j, a / 2, a + 1.5, 1.5 - a, -a, a**0, a**2]


@pytest.mark.parametrize("script", [_script_products, _script_add_sub, _script_scalars])
def test_arithmetic_scripts_match(script):
    for ours, theirs in zip(script(FermionOperator), script(OFFermionOperator), strict=True):
        assert_same(ours, theirs)


def test_no_normal_ordering_ever():
    ours, theirs = both("1 2^")
    assert_same(ours, theirs)
    assert list(ours.terms) == [((1, 0), (2, 1))]

    # Products concatenate verbatim.
    prod_ours = FermionOperator("2^ 1") * FermionOperator("0^")
    prod_theirs = OFFermionOperator("2^ 1") * OFFermionOperator("0^")
    assert_same(prod_ours, prod_theirs)
    assert list(prod_ours.terms) == [((2, 1), (1, 0), (0, 1))]


# ── Dagger ────────────────────────────────────────────────────────────────


def test_hermitian_conjugated_matches():
    ours, theirs = both("2^ 1", 1.0 + 2.0j)
    ours += FermionOperator("3^ 0^ 1", -0.5j)
    theirs += OFFermionOperator("3^ 0^ 1", -0.5j)
    assert_same(hermitian_conjugated(ours), openfermion.hermitian_conjugated(theirs))


def test_dagger_round_trip():
    op = FermionOperator("3^ 1 2^", 0.5 - 0.25j) + FermionOperator("0^", 2.0)
    assert hermitian_conjugated(hermitian_conjugated(op)) == op


# ── Misc parity ───────────────────────────────────────────────────────────


def test_count_qubits_matches():
    for args in [(), ("", 2.0), ("0^",), ("2^ 1",), ("0^ 17",)]:
        ours, theirs = both(*args)
        assert count_qubits(ours) == openfermion.count_qubits(theirs)


def test_equality_matches():
    assert (FermionOperator("2^ 1") == FermionOperator("2^ 1", 1.0 + 5e-9)) == (
        OFFermionOperator("2^ 1") == OFFermionOperator("2^ 1", 1.0 + 5e-9)
    )
    assert (FermionOperator("2^ 1") == FermionOperator("1 2^")) == (
        OFFermionOperator("2^ 1") == OFFermionOperator("1 2^")
    )


def test_unhashable_and_type_errors():
    with pytest.raises(TypeError):
        hash(FermionOperator("2^"))
    with pytest.raises(TypeError):
        FermionOperator("2^") * "nope"


def test_terms_reassignment_and_get_operators():
    def rebuild(cls):
        source = cls("2^ 1", 0.5) + cls("0^", -1.0) + cls("", 3.0)
        op = cls()
        op.terms = {k: v for k, v in source.terms.items() if k != ()}
        return op, list(op.get_operators())

    ours, ours_singles = rebuild(FermionOperator)
    theirs, theirs_singles = rebuild(OFFermionOperator)
    assert_same(ours, theirs)
    for o, t in zip(ours_singles, theirs_singles, strict=True):
        assert_same(o, t)


def test_copy_deepcopy_pickle_round_trip():
    op = FermionOperator("2^ 1", 0.5 - 0.25j) + FermionOperator("0^", 2.0)
    for clone in [copy.copy(op), copy.deepcopy(op), pickle.loads(pickle.dumps(op))]:
        assert clone == op
        assert norm_terms(clone) == norm_terms(op)
        clone += FermionOperator("5^")
        assert clone != op
