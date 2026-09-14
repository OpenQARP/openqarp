"""Randomized cross-validation of the qarpx C++ transform kernels against
openfermion (JW/BK) and a hand-rolled parity-mapping reference.

Complements the golden tests: random 1- and 2-body ladder terms with random
complex coefficients, n ≤ 12 modes, exact `.terms` equality including
insertion order.
"""

import numpy as np
import pytest

openfermion = pytest.importorskip("openfermion")

import qarpx as qx


def _reference_parity_encode(op, nqubits):
    """The hand-rolled parity mapping that lived in
    qarp's Python parity mapping (now qarp/operators/_mappings.py) before
    the C++ rewrite, verbatim,
    over openfermion operators — kept here as the golden reference."""
    Q = openfermion.QubitOperator

    def ladder(orbital, creation):
        qop = Q("X0 X0", 1.0 / 2)
        for k in range(orbital + 1, nqubits):
            qop *= Q((k, "X"), 1.0)
        sign = -1j if creation else 1j
        if orbital > 0:
            qop *= Q((orbital - 1, "Z"), 1.0) * Q((orbital, "X"), 1.0) + Q((orbital, "Y"), sign)
        else:
            qop *= Q((orbital, "X"), 1.0) + Q((orbital, "Y"), sign)
        return qop

    qop = Q()
    for term, coefficient in op.terms.items():
        qop_term = Q("X0 X0", coefficient)
        for orb_num, action in term:
            assert orb_num < nqubits
            qop_term *= ladder(orb_num, action == 1)
        qop += qop_term
    return qop


def norm_terms(op):
    return [(key, complex(value)) for key, value in op.terms.items()]


def random_ladder_terms(rng, n_modes, n_terms):
    """A FermionOperator-shaped list of (term_string, coefficient)."""
    entries = []
    for _ in range(n_terms):
        body = rng.choice([1, 2])
        modes = rng.integers(0, n_modes, size=2 * body)
        factors = []
        for i, mode in enumerate(modes):
            dagger = "^" if i < body else ""
            factors.append(f"{mode}{dagger}")
        coeff = complex(rng.normal(), rng.normal())
        entries.append((" ".join(factors), coeff))
    return entries


def build(cls, entries):
    op = cls()
    for term, coeff in entries:
        op += cls(term, coeff)
    return op


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("n_modes", [3, 6, 12])
def test_jordan_wigner_matches_openfermion(seed, n_modes):
    rng = np.random.default_rng(seed)
    entries = random_ladder_terms(rng, n_modes, n_terms=8)
    ours = qx.jordan_wigner(build(qx.FermionOperator, entries))
    theirs = openfermion.jordan_wigner(build(openfermion.FermionOperator, entries))
    assert norm_terms(ours) == norm_terms(theirs)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("n_modes", [3, 6, 12])
def test_bravyi_kitaev_matches_openfermion(seed, n_modes):
    rng = np.random.default_rng(seed)
    entries = random_ladder_terms(rng, n_modes, n_terms=8)
    ours = qx.bravyi_kitaev(build(qx.FermionOperator, entries))
    theirs = openfermion.bravyi_kitaev(build(openfermion.FermionOperator, entries))
    assert norm_terms(ours) == norm_terms(theirs)

    # Widened register overload.
    ours_wide = qx.bravyi_kitaev(build(qx.FermionOperator, entries), n_modes + 3)
    theirs_wide = openfermion.bravyi_kitaev(
        build(openfermion.FermionOperator, entries), n_modes + 3
    )
    assert norm_terms(ours_wide) == norm_terms(theirs_wide)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("n_modes", [3, 6])
def test_parity_matches_hand_rolled_reference(seed, n_modes):
    rng = np.random.default_rng(seed)
    entries = random_ladder_terms(rng, n_modes, n_terms=6)
    ours = qx.parity_transform(build(qx.FermionOperator, entries), n_modes)
    reference = _reference_parity_encode(build(openfermion.FermionOperator, entries), n_modes)
    assert norm_terms(ours) == norm_terms(reference)


def test_batch_overloads_match_scalar():
    entries_a = [("2^ 1", 0.5), ("0^", -1.0)]
    entries_b = [("1^ 3", 0.25j)]
    fops = [build(qx.FermionOperator, entries_a), build(qx.FermionOperator, entries_b)]

    for batch, single in [
        (qx.jordan_wigner(fops), [qx.jordan_wigner(f) for f in fops]),
        (qx.bravyi_kitaev(fops), [qx.bravyi_kitaev(f) for f in fops]),
        (qx.parity_transform(fops, 6), [qx.parity_transform(f, 6) for f in fops]),
    ]:
        assert len(batch) == len(single)
        for b, s in zip(batch, single, strict=True):
            assert norm_terms(b) == norm_terms(s)


def test_validation_errors():
    with pytest.raises(ValueError):
        qx.bravyi_kitaev(qx.FermionOperator("2^ 1"), 2)
    with pytest.raises(RuntimeError, match="Orbital number is larger"):
        qx.parity_transform(qx.FermionOperator("2^ 1"), 2)


# ── BravyiKitaev maps a list on one register width (P1.17) ───────────────


def _of_terms(op):
    return sorted((tuple(k), complex(v)) for k, v in op.terms.items())


def test_bravyi_kitaev_list_uses_one_width():
    """Per-operator inference gave '4^ 3' a 5-qubit BK image and '7^ 6' an
    8-qubit one, so the two did not compose.  A list now maps on
    max(count_qubits) — openfermion at n_qubits=8 is the oracle — and the
    single-operator form keeps inferring."""
    from qarp.operators import BravyiKitaev, FermionOperator

    strings = ["4^ 3", "7^ 6"]
    ours = BravyiKitaev().encode_operator([FermionOperator(s) for s in strings])
    theirs = [
        openfermion.bravyi_kitaev(openfermion.FermionOperator(s), n_qubits=8) for s in strings
    ]
    for o, t in zip(ours, theirs, strict=True):
        assert _of_terms(o) == _of_terms(t)

    # The row bites: the 5-qubit image of '4^ 3' is a different operator.
    narrow = BravyiKitaev().encode_operator(FermionOperator("4^ 3"))
    assert _of_terms(narrow) == _of_terms(
        openfermion.bravyi_kitaev(openfermion.FermionOperator("4^ 3"))
    )
    assert _of_terms(narrow) != _of_terms(ours[0])


def test_bravyi_kitaev_explicit_width():
    from qarp.operators import BravyiKitaev, FermionOperator

    ours = BravyiKitaev(n_qubits=10).encode_operator(FermionOperator("0^ 1"))
    theirs = openfermion.bravyi_kitaev(openfermion.FermionOperator("0^ 1"), n_qubits=10)
    assert _of_terms(ours) == _of_terms(theirs)
    ours_list = BravyiKitaev(n_qubits=10).encode_operator([FermionOperator("0^ 1")])
    assert _of_terms(ours_list[0]) == _of_terms(theirs)
    with pytest.raises(ValueError):
        BravyiKitaev(n_qubits=2).encode_operator(FermionOperator("3^ 0"))
    assert BravyiKitaev().encode_operator([]) == []
