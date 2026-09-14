"""Differential tests: qarp.operators.QubitOperator vs openfermion.

Every case builds the same operator with both implementations and compares
``.terms`` exactly — keys, coefficients (as complex) AND insertion order.
Insertion-order equality is the load-bearing bar: qarp derives circuit
structure (Trotter/LCU/grouping) from term order, and the golden mapping
tests assert it directly.

One deliberate deviation: ``+=`` erases only exact cancellations, where
openfermion erases anything below ``EQ_TOLERANCE``.  It is pinned by
``test_add_deviates_from_openfermion_below_eq_tolerance`` and the tests around
it rather than being quietly excluded.
"""

import copy
import pickle

import numpy as np
import pytest

openfermion = pytest.importorskip("openfermion")

from qarp.operators import QubitOperator
from qarp.operators.functions import (
    count_qubits,
    hermitian_conjugated,
    is_hermitian,
)

OFQubitOperator = openfermion.QubitOperator


def norm_terms(op):
    """(key, complex(value)) pairs in .terms iteration order."""
    return [(key, complex(value)) for key, value in op.terms.items()]


def assert_same(ours, theirs):
    __tracebackhide__ = True
    assert norm_terms(ours) == norm_terms(theirs)


def both(*args, **kwargs):
    return QubitOperator(*args, **kwargs), OFQubitOperator(*args, **kwargs)


# ── Constructors ──────────────────────────────────────────────────────────

CTOR_CASES = [
    (),
    ("",),
    ("", 2.0),
    ("X0",),
    ("X0", 0.0),  # zero coefficient is kept
    ("X0 Z2 Y3", 0.5),
    ("Z2 X0",),  # unsorted input
    ("X0 X0", 0.5),  # identity-carrier idiom (parity.py)
    ("X0 Y0", 1.0),  # repeated qubit with phase
    ("Y0 X0 Z1 X0", -0.25j),  # fold across interleaved factors
    (((0, "X"), (1, "Y")), 1j),
    ((0, "X"),),  # single-factor shorthand
    ([(3, "Z"), (0, "Y")], -2.0),
    ("X0", np.float64(0.5)),
    ("X0", np.complex128(0.25j)),
    ("1.5 [X0 Z1]", 2.0),  # bracketed string form
]


@pytest.mark.parametrize("args", CTOR_CASES, ids=[repr(c) for c in CTOR_CASES])
def test_constructors_match(args):
    ours, theirs = both(*args)
    assert_same(ours, theirs)


@pytest.mark.parametrize("bad", ["A0", "X", "X0Y1", "I0", 5, ((0, "I"),)])
def test_invalid_constructors_raise_valueerror_in_both(bad):
    with pytest.raises(ValueError):
        OFQubitOperator(bad)
    with pytest.raises(ValueError):
        QubitOperator(bad)


# ── Arithmetic scripts (identical operation sequences) ────────────────────


def _script_add_sub(cls):
    a = cls("X0", 0.75) + cls("Y1 Z2", -0.5j)
    a += cls("X0", 0.25)
    a -= cls("Z4", 1.5)
    b = a - cls("Y1 Z2", -0.5j)
    return [a, b, -a]


def _script_products(cls):
    a = cls("X0", 0.75) + cls("Y0 Z2", -0.5j) + cls("Z1")
    b = cls("Y0", 2.0) + cls("X1 Z2", 0.5)
    c = a * b
    d = b * a
    e = a * a
    f = c
    f *= a
    return [a, b, c, d, e, f]


def _script_scalars(cls):
    a = cls("X0", 0.75) + cls("Y1", -2.0)
    return [
        a * 2,
        2 * a,
        a * 0.5j,
        a / 2,
        a / 0.5j,
        a + 1.5,
        1.5 + a,
        a - 1.5,
        1.5 - a,
        a**0,
        a**3,
    ]


def _script_cancellation(cls):
    a = cls("X0")
    a += cls("X0", -1.0)  # exact cancellation → term erased
    d = cls("X0") + cls("Y0")
    e = d * (cls("X0") + cls("Y0", -1.0))  # product zero survives
    f = cls("X0", 1e-12)
    f *= 2.0  # scalar multiply never compacts
    g = cls("X0")
    g += 2.0  # scalar fold into constant
    g -= 2.0  # stays as explicit zero constant
    return [a, d, e, f, g]


# ── The one deviation from openfermion (see test_qubit_operator_add_deviation)
# openfermion's __iadd__ erases any resulting term below EQ_TOLERANCE.  qarp
# erases only exact cancellations, so these cases are deliberately NOT in the
# parity scripts above; they are pinned against the divergence instead.


@pytest.mark.parametrize(
    "script", [_script_add_sub, _script_products, _script_scalars, _script_cancellation]
)
def test_arithmetic_scripts_match(script):
    for ours, theirs in zip(script(QubitOperator), script(OFQubitOperator), strict=True):
        assert_same(ours, theirs)


# ── The += compaction deviation ───────────────────────────────────────────


def test_add_deviates_from_openfermion_below_eq_tolerance():
    """openfermion's ``__iadd__`` erases any resulting term below EQ_TOLERANCE;
    qarp erases only exact cancellations.  Pinned as a divergence, with both
    sides asserted, so the day either changes the test says which.

    Oracle: the arithmetic itself — ``1.0 + (-1.0 + 5e-9)`` is ``5e-9``, and a
    term equal to its own value is the only defensible result.  Truncating it is
    a policy, and policies belong where the caller can state a scale.
    """
    ours = QubitOperator("X0") + QubitOperator("X0", -1.0 + 5e-9)
    theirs = OFQubitOperator("X0") + OFQubitOperator("X0", -1.0 + 5e-9)
    # 1.0 + (-1.0 + 5e-9) is 4.9999999696e-9: the residual is what float
    # subtraction leaves, not the literal, which is the point of keeping it.
    assert [k for k, _ in norm_terms(ours)] == [((0, "X"),)]
    assert norm_terms(ours)[0][1] == pytest.approx(5e-9, rel=1e-7)
    assert norm_terms(theirs) == []

    ours_new = QubitOperator() + QubitOperator("X0", 5e-9)
    theirs_new = OFQubitOperator() + OFQubitOperator("X0", 5e-9)
    assert norm_terms(ours_new) == [(((0, "X"),), 5e-9 + 0j)]  # exact: no arithmetic
    assert norm_terms(theirs_new) == []

    # compress() is where truncation lives, and it still agrees with openfermion.
    ours.compress()
    assert norm_terms(ours) == []


def test_add_is_commutative():
    """The bug the deviation exists for.  The old rule tested the addend's terms
    and never the receiver's, so ``a + b`` dropped a sub-tolerance term that
    ``b + a`` kept — for the same two operands.

    Oracle: commutativity of addition.  openfermion fails this; that is the
    point, so it is asserted here too rather than left implied.
    """
    big, tiny = QubitOperator("Z0", 1.0), QubitOperator("X0", 1e-9)
    # Term order follows the receiver, so compare as mappings, not sequences.
    assert dict((big + tiny).terms) == dict((tiny + big).terms)
    assert len((big + tiny).terms) == 2

    of_big, of_tiny = OFQubitOperator("Z0", 1.0), OFQubitOperator("X0", 1e-9)
    assert dict((of_big + of_tiny).terms) != dict((of_tiny + of_big).terms)


def test_add_keeps_every_term_of_a_small_norm_operator():
    """An absolute cutoff annihilates an operator whose whole norm is small —
    a perturbation, a commutator that came out small, anything in atomic units —
    and rescaling afterwards cannot bring back a term already dropped.

    Oracle: the operator scaled up by 1e9 must equal the one built at unit scale.
    """
    small = QubitOperator("Z0", 1e-9) + QubitOperator("X0", -1e-9)
    assert len(small.terms) == 2
    assert small * 1e9 == QubitOperator("Z0", 1.0) - QubitOperator("X0", 1.0)

    of_small = OFQubitOperator("Z0", 1e-9) + OFQubitOperator("X0", -1e-9)
    assert len(of_small.terms) == 1  # openfermion lost the X0 term on assembly


def test_exact_cancellation_still_erases():
    """The narrowing is exact-zero, not no-compaction: a term that cancels
    exactly still goes, so transforms (JW/BK) do not accumulate dead terms.

    Oracle: ``X0 - X0`` is the zero operator.
    """
    assert norm_terms(QubitOperator("X0") - QubitOperator("X0")) == []
    assert norm_terms(QubitOperator("X0", 0.25) + QubitOperator("X0", -0.25)) == []


def test_inplace_returns_same_object():
    a = QubitOperator("X0")
    before = id(a)
    a += QubitOperator("Y0")
    a *= 2.0
    assert id(a) == before


# ── Equality (openfermion isclose semantics) ──────────────────────────────

# No case at |Δ| ≈ EQ_TOLERANCE: openfermion versions round the boundary
# differently, so a differential test must stay clear of it.
EQ_CASES = [
    ("X0", 1.0, "X0", 1.0 + 5e-9),
    ("X0", 1e6, "X0", 1e6 + 1e-3),  # tolerance scales with magnitude
    ("X0", 1.0, "Y0", 1.0),
    ("X0", 0.0, "", 0.0),  # zero coefficient vs empty-adjacent
]


@pytest.mark.parametrize("term_a,coeff_a,term_b,coeff_b", EQ_CASES)
def test_equality_matches_openfermion(term_a, coeff_a, term_b, coeff_b):
    ours = QubitOperator(term_a, coeff_a) == QubitOperator(term_b, coeff_b)
    theirs = OFQubitOperator(term_a, coeff_a) == OFQubitOperator(term_b, coeff_b)
    assert ours == theirs


def test_equality_one_sided_small_terms():
    ours = (QubitOperator("X0") + QubitOperator("Y1", 5e-9)) == QubitOperator("X0")
    theirs = (OFQubitOperator("X0") + OFQubitOperator("Y1", 5e-9)) == OFQubitOperator("X0")
    assert ours == theirs is True

    ours = (QubitOperator("X0") + QubitOperator("Y1", 1e-3)) == QubitOperator("X0")
    theirs = (OFQubitOperator("X0") + OFQubitOperator("Y1", 1e-3)) == OFQubitOperator("X0")
    assert ours == theirs is False


def test_unhashable_like_openfermion():
    with pytest.raises(TypeError):
        hash(OFQubitOperator("X0"))
    with pytest.raises(TypeError):
        hash(QubitOperator("X0"))


def test_cross_type_multiply_raises_typeerror():
    with pytest.raises(TypeError):
        QubitOperator("X0") * "nope"
    with pytest.raises(TypeError):
        OFQubitOperator("X0") * "nope"


def test_negative_power_raises_valueerror_in_both():
    with pytest.raises(ValueError):
        OFQubitOperator("X0") ** -1
    with pytest.raises(ValueError):
        QubitOperator("X0") ** -1


# ── .terms access patterns used across qarp ───────────────────────────────


def test_terms_read_patterns():
    ours, theirs = both("X0 Z2", 0.5)
    ours += QubitOperator("", 0.25)
    theirs += OFQubitOperator("", 0.25)

    key = ((0, "X"), (2, "Z"))
    assert complex(ours.terms[key]) == complex(theirs.terms[key])
    assert complex(ours.terms.get((), 0)) == complex(theirs.terms.get((), 0))
    assert (() in ours.terms) == (() in theirs.terms)
    assert max(i for term in ours.terms for i, _ in term) == max(
        i for term in theirs.terms for i, _ in term
    )
    assert dict(ours.terms).keys() == dict(theirs.terms).keys()
    missing = ((7, "Y"),)
    with pytest.raises(KeyError):
        _ = ours.terms[missing]
    with pytest.raises(KeyError):
        _ = theirs.terms[missing]


def test_terms_full_reassignment_trotter_pattern():
    # qarp/blocks/_primitives/trotter_block.py:296-301
    def strip_constant(cls):
        source = cls("X0", 0.5) + cls("Z1 Z2", -0.25) + cls("", 3.0)
        op = cls()
        op.terms = {k: v for k, v in source.terms.items() if k != ()}
        return op, list(op.get_operators())

    ours, ours_singles = strip_constant(QubitOperator)
    theirs, theirs_singles = strip_constant(OFQubitOperator)
    assert_same(ours, theirs)
    for o, t in zip(ours_singles, theirs_singles, strict=True):
        assert_same(o, t)


def test_terms_view_is_read_only_and_cached():
    op = QubitOperator("X0", 0.5)
    view = op.terms
    assert op.terms is view  # cached while unmutated
    with pytest.raises(TypeError):
        view[()] = 1.0  # loud failure instead of silent no-op
    op += QubitOperator("Y1")
    assert op.terms is not view  # mutation invalidates the cache
    assert ((1, "Y"),) in op.terms
    # dict(...) gives a mutable copy
    mutable = dict(op.terms)
    mutable[()] = 0.0
    assert () not in op.terms


def test_get_operators_matches():
    ours, theirs = both("X0", 0.5)
    ours += QubitOperator("Y1", 2.0) + QubitOperator("", 0.0)
    theirs += OFQubitOperator("Y1", 2.0) + OFQubitOperator("", 0.0)
    ours_singles = list(ours.get_operators())
    theirs_singles = list(theirs.get_operators())
    assert len(ours_singles) == len(theirs_singles)
    for o, t in zip(ours_singles, theirs_singles, strict=True):
        assert_same(o, t)


# ── Free functions ────────────────────────────────────────────────────────


def test_hermitian_conjugated_matches():
    ours, theirs = both("X0 Y1", 0.5 + 0.25j)
    ours += QubitOperator("Z2", -1j)
    theirs += OFQubitOperator("Z2", -1j)
    assert_same(hermitian_conjugated(ours), openfermion.hermitian_conjugated(theirs))


def test_count_qubits_matches():
    for args in [(), ("", 2.0), ("X0",), ("X3 Z7",)]:
        ours, theirs = both(*args)
        assert count_qubits(ours) == openfermion.count_qubits(theirs)


def test_is_hermitian_matches():
    for args, add in [(("X0", 0.5), ("Z1 Z2", -1.0)), (("X0", 1j), ("Y1", 0.5))]:
        ours = QubitOperator(*args) + QubitOperator(*add)
        theirs = OFQubitOperator(*args) + OFQubitOperator(*add)
        assert is_hermitian(ours) == openfermion.utils.is_hermitian(theirs)


# ── Copy / pickle ─────────────────────────────────────────────────────────


def test_copy_deepcopy_pickle_round_trip():
    op = QubitOperator("X0 Y1", 0.5 - 0.25j) + QubitOperator("", 2.0)
    for clone in [copy.copy(op), copy.deepcopy(op), pickle.loads(pickle.dumps(op))]:
        assert clone == op
        assert norm_terms(clone) == norm_terms(op)
        clone += QubitOperator("Z5")  # independent storage
        assert clone != op
