"""qarp.graphs._utils helpers against hand-built graphs and combinatorics.

No hypernetx guard: SimplicialComplex carries no optional dependency, so these
run on a base install.
"""

import itertools
import math
import random

import pytest

from qarp.graphs import (
    Graph,
    community_vector_to_sets,
    generate_complete_simplicial_complex,
    generate_random_simplicial_complex,
    lists_to_tuples,
)


def _triangle():
    g = Graph()
    g.add_edges_from([(0, 1), (1, 2), (0, 2)])
    return g


def _closure(simplices) -> set:
    out: set = set()
    for s in simplices:
        s = tuple(sorted(s))
        for k in range(1, len(s) + 1):
            out.update(tuple(sorted(f)) for f in itertools.combinations(s, k))
    return out


def _is_closed(sc) -> bool:
    present = set(sc.get_simplices())
    return all(
        tuple(sorted(face)) in present
        for simplex in present
        for k in range(1, len(simplex))
        for face in itertools.combinations(simplex, k)
    )


def test_community_vector_to_sets_hand_example():
    out = community_vector_to_sets(_triangle(), [0, 1, 0])
    assert out == [{0, 2}, {1}]


def test_community_vector_length_mismatch_raises():
    with pytest.raises(ValueError, match="must match number of nodes"):
        community_vector_to_sets(_triangle(), [0, 1])


def test_lists_to_tuples():
    assert lists_to_tuples([[0, 1], [1, 2, 3]]) == [(0, 1), (1, 2, 3)]


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_complete_simplicial_complex_is_every_nonempty_subset(n):
    sc = generate_complete_simplicial_complex(n)
    assert set(sc.get_simplices()) == _closure([tuple(range(n))])
    assert len(sc) == 2**n - 1


def test_random_simplicial_complex_p_one_is_complete():
    random.seed(5)
    sc = generate_random_simplicial_complex(4, 1.0)
    assert set(sc.get_simplices()) == set(generate_complete_simplicial_complex(4).get_simplices())


def test_random_simplicial_complex_p_zero_is_empty():
    random.seed(5)
    sc = generate_random_simplicial_complex(4, 0.0)
    assert sc.get_simplices() == []


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_random_simplicial_complex_is_closed_under_faces(seed):
    random.seed(seed)
    sc = generate_random_simplicial_complex(5, 0.5)
    assert _is_closed(sc)
    assert all(len(s) <= 5 and set(s) <= set(range(5)) for s in sc.get_simplices())


def test_random_simplicial_complex_is_reproducible_under_the_global_seed():
    """The function draws from the global ``random`` module, not ``config.seed``."""
    random.seed(11)
    a = generate_random_simplicial_complex(5, 0.4).get_simplices()
    random.seed(11)
    b = generate_random_simplicial_complex(5, 0.4).get_simplices()
    assert a == b


def test_complete_complex_binomial_counts():
    sc = generate_complete_simplicial_complex(5)
    assert sc.f_vector() == [math.comb(5, k + 1) for k in range(5)]
