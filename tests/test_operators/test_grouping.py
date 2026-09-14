"""Unit tests for qarp.operators._grouping strategies.

Partition validity, determinism, and the qubit_wise guarantee are the
load-bearing contracts — consumers (PauliAveraging, Trotter, cutting, PCE)
derive circuit identity from them.
"""

import random

import pytest

import qarpx as qx
from qarp.operators import QubitOperator


def commutator(a, b):
    """openfermion.utils.commutator equivalent."""
    return a * b - b * a


from qarp.operators import (
    FullyCommuting,
    GroupingStrategy,
    NoGrouping,
    QubitWiseCommuting,
    group_basis,
)
from qarp.operators._grouping import (
    canonical_term_order,
    diagonalise_group,
    greedy_first_fit,
    pauli_dict_to_string,
    qubit_wise_commute,
)

LETTERS = "XYZ"


def _random_terms(n_qubits, n_terms, seed):
    rng = random.Random(seed)
    terms = []
    for _ in range(n_terms):
        support = rng.sample(range(n_qubits), rng.randint(1, n_qubits))
        terms.append({q: rng.choice(LETTERS) for q in support})
    return terms


def _is_partition(groups, n_terms):
    flat = sorted(i for grp in groups for i in grp)
    return flat == list(range(n_terms))


@pytest.mark.parametrize("strategy", [NoGrouping(), QubitWiseCommuting(), FullyCommuting()])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_group_is_partition(strategy, seed):
    terms = _random_terms(n_qubits=6, n_terms=40, seed=seed)
    groups = strategy.group(terms, 6)
    assert _is_partition(groups, len(terms))


@pytest.mark.parametrize("strategy", [NoGrouping(), QubitWiseCommuting(), FullyCommuting()])
def test_group_is_deterministic(strategy):
    terms = _random_terms(n_qubits=5, n_terms=30, seed=7)
    assert strategy.group(terms, 5) == strategy.group(terms, 5)


def test_no_grouping_returns_singletons():
    terms = _random_terms(n_qubits=4, n_terms=10, seed=3)
    assert NoGrouping().group(terms, 4) == [[i] for i in range(10)]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_qwc_groups_pairwise_qubit_wise_commute(seed):
    terms = _random_terms(n_qubits=6, n_terms=40, seed=seed)
    for grp in QubitWiseCommuting().group(terms, 6):
        for i in grp:
            for j in grp:
                assert qubit_wise_commute(terms[i], terms[j])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fully_commuting_pairwise_commute(seed):
    n_qubits = 6
    terms = _random_terms(n_qubits, n_terms=40, seed=seed)
    strings = [pauli_dict_to_string(t, n_qubits) for t in terms]
    for grp in FullyCommuting().group(terms, n_qubits):
        for i in grp:
            for j in grp:
                assert qx.commutes(strings[i], strings[j])


def test_fully_commuting_is_superset_of_qwc():
    # XX and YY commute globally (two mismatches) but are not QWC.
    terms = [{0: "X", 1: "X"}, {0: "Y", 1: "Y"}]
    assert FullyCommuting().group(terms, 2) == [[0, 1]]
    assert QubitWiseCommuting().group(terms, 2) == [[0], [1]]
    # qubit_wise flag honesty: FullyCommuting produced a non-QWC group.
    assert not FullyCommuting.qubit_wise
    assert QubitWiseCommuting.qubit_wise
    assert NoGrouping.qubit_wise


def test_fully_commuting_vanishing_commutators():
    # Independent check via openfermion: every within-group pair of the
    # actual operators commutes as matrices would.
    n_qubits = 5
    terms = _random_terms(n_qubits, n_terms=25, seed=11)
    ops = [QubitOperator(tuple(sorted((q, p) for q, p in t.items()))) for t in terms]
    for grp in FullyCommuting().group(terms, n_qubits):
        for i in grp:
            for j in grp:
                assert commutator(ops[i], ops[j]) == QubitOperator()


def test_group_basis_consistent_on_qwc_groups():
    terms = _random_terms(n_qubits=6, n_terms=30, seed=5)
    for grp in QubitWiseCommuting().group(terms, 6):
        basis = group_basis(grp, terms)
        # Every term's letters must agree with the group basis recipe.
        for i in grp:
            for q, p in terms[i].items():
                assert basis[q] == p


def test_diagonalise_group_rejects_non_commuting():
    """A non-commuting group must raise: silently collapsing the Z row to
    identity would make PauliAveraging read its expectation as +1."""
    paulis = [qx.parse_pauli_string("X"), qx.parse_pauli_string("Z")]
    with pytest.raises(ValueError, match="does not commute"):
        diagonalise_group(paulis, 1)


def _as_term_groups(groups, terms):
    """Partition rendered as term signatures, so it can be compared across
    different input orderings (indices are meaningless between permutations)."""
    return sorted(sorted(tuple(sorted(terms[i].items())) for i in grp) for grp in groups)


@pytest.mark.parametrize("strategy", [QubitWiseCommuting(), FullyCommuting()])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_grouping_is_order_insensitive(strategy, seed):
    """Permuting the input must permute the partition, not change it.

    Greedy first-fit is order-sensitive by nature, so the same Hamiltonian
    written with its terms in a different order must not produce a different
    partition — that is silently different physics for symmetry-carrying operators
    (see ``test_grouped_trotter_conserves_particle_number``).
    """
    terms = _random_terms(n_qubits=5, n_terms=20, seed=99)
    expected = _as_term_groups(strategy.group(terms, 5), terms)

    shuffled = terms[:]
    random.Random(seed).shuffle(shuffled)
    assert _as_term_groups(strategy.group(shuffled, 5), shuffled) == expected


def test_no_grouping_stays_order_faithful():
    """``NoGrouping`` is deliberately exempt from canonical ordering — it makes
    no grouping decision, so it emits terms exactly as the caller wrote them."""
    terms = _random_terms(n_qubits=4, n_terms=6, seed=3)
    assert NoGrouping().group(terms, 4) == [[i] for i in range(6)]
    assert NoGrouping().group(terms[::-1], 4) == [[i] for i in range(6)]


def test_canonical_order_sorts_by_support_then_letters():
    """JW hopping partners share support, so support-major ordering is what
    keeps ``X_pX_q``/``Y_pY_q`` adjacent for the greedy pass."""
    terms = [{1: "Y", 2: "Y"}, {0: "Z"}, {0: "X", 1: "X"}, {1: "X", 2: "X"}, {0: "Y", 1: "Y"}]
    assert canonical_term_order(terms) == [1, 2, 4, 3, 0]


def test_greedy_first_fit_first_fit_order():
    # Item joins the FIRST compatible group, not the best one — pinned
    # because circuit identity depends on it.
    compat = {(1, 0): True, (2, 0): True, (2, 1): False}

    def compatible(i, j):
        return compat.get((i, j), compat.get((j, i), True))

    # 0 opens g0; 1 joins g0; 2 is compatible with 0 but not 1 → opens g1.
    assert greedy_first_fit(3, compatible) == [[0, 1], [2]]


def test_strategy_repr():
    assert repr(FullyCommuting()) == "FullyCommuting()"
    assert isinstance(QubitWiseCommuting(), GroupingStrategy)


def test_empty_input_returns_empty_partition():
    for strategy in (NoGrouping(), QubitWiseCommuting(), FullyCommuting()):
        assert strategy.group([], 3) == []
