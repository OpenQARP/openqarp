"""SimplicialComplex: closure under faces, and the simplex renderer.

``hypernetx`` is deliberately NOT required here.  The class carries no
hypergraph dependency once ``plot`` stops routing through ``hnx.draw``, so it
imports on a base install and these tests run there too — pinned by
``test_module_has_no_hypernetx_dependency`` below.
"""

import ast
import inspect
import itertools
import math

import matplotlib
import networkx as nx
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  — must follow use("Agg")

from qarp import config  # noqa: E402
from qarp.graphs import (  # noqa: E402
    Graph,
    SimplicialComplex,
    generate_complete_simplicial_complex,  # noqa: E402
)

# ── Oracles: recomputed from the definitions, never read back from the class ──


def _is_closed(sc) -> bool:
    """True iff every non-empty proper face of every simplex is also present."""
    present = set(sc.get_simplices())
    return all(
        tuple(sorted(face)) in present
        for simplex in present
        for k in range(1, len(simplex))
        for face in itertools.combinations(simplex, k)
    )


def _euler_characteristic(sc) -> int:
    """chi = sum_k (-1)^k f_k, with f_k the number of k-simplices."""
    return sum((-1) ** (len(s) - 1) for s in sc.get_simplices())


def _closure(simplices) -> set:
    """Every non-empty face of every listed simplex, computed here from
    itertools rather than by asking the class."""
    out: set = set()
    for s in simplices:
        s = tuple(sorted(s))
        for k in range(1, len(s) + 1):
            out.update(tuple(sorted(f)) for f in itertools.combinations(s, k))
    return out


def _facets(simplices) -> list:
    """Maximal simplices of order >= 2, recomputed by set inclusion."""
    as_sets = [set(s) for s in simplices]
    return [tuple(sorted(s)) for s in as_sets if len(s) >= 3 and not any(s < t for t in as_sets)]


# ── Existing behaviour (unchanged) ───────────────────────────────────────


def test_initialization_and_faces():
    sc = SimplicialComplex([[1, 2, 3]])
    simplices = sc.get_simplices()
    expected = [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    assert sorted(simplices) == sorted(expected)


def test_add_simplex_and_faces():
    sc = SimplicialComplex([])
    sc.add_simplex((1, 2, 3))
    assert (1, 2, 3) in sc.get_simplices()
    assert (1, 2) in sc.get_simplices()
    assert (1,) in sc.get_simplices()


def test_remove_simplex():
    sc = SimplicialComplex([[1, 2]])
    sc.remove_simplex([1, 2])
    assert (1, 2) not in sc.get_simplices()
    # faces like (1,) or (2,) should still exist if they were never explicitly added
    assert (1,) in sc.get_simplices()
    assert (2,) in sc.get_simplices()


def test_neighbors():
    sc = SimplicialComplex([[1, 2, 3], [3, 4]])
    assert sc.neighbors(3) == [1, 2, 4]
    assert sc.neighbors(4) == [3]
    assert sc.neighbors(9) == []  # absent vertex: nothing shares a simplex with it


def test_max_dimension():
    sc = SimplicialComplex([[1, 2, 3], [4, 5]])
    assert sc.max_dimension() == 2


def test_str_repr():
    sc = SimplicialComplex([[1, 2]])
    assert "Simplicial complex" in str(sc)
    assert repr(sc) == str(sc)


def test_invalid_add_remove():
    sc = SimplicialComplex([])
    with pytest.raises(ValueError):
        sc.add_simplex("not a simplex")
    with pytest.raises(ValueError):
        sc.remove_simplex("not a simplex")


def test_complex_plot_instance():
    sc = SimplicialComplex([[1, 2, 3]])
    fig, ax = sc.plot(return_fig=True)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)  # Close the figure to avoid display in tests


# ── Closure under removal ────────────────────────────────────────────────

# The plan's worked example: two triangles sharing the edge (0, 1).
_INCIDENT = [(0,), (1,), (2,), (3,), (0, 1), (1, 2), (2, 3), (0, 1, 2), (0, 1, 3)]


def _incident_complex():
    return SimplicialComplex(_INCIDENT)


@pytest.mark.parametrize("target", sorted(_closure(_INCIDENT)))
def test_closure_holds_after_any_single_removal(target):
    """Exhaustive over every simplex present: removing any one of them leaves
    a complex that is still closed under faces.  The bare ``discard`` this
    replaces failed here for every simplex that had a coface."""
    sc = _incident_complex()
    sc.remove_simplex(target)
    assert _is_closed(sc), f"removing {target} left an unclosed complex"


def test_removal_drops_exactly_the_cofaces():
    """Oracle: the coface set computed from the input list by set inclusion,
    independent of the class."""
    sc = _incident_complex()
    before = _closure(_INCIDENT)
    sc.remove_simplex((0, 1))
    expected_removed = {s for s in before if {0, 1}.issubset(s)}
    assert set(sc.get_simplices()) == before - expected_removed
    # Both containing triangles went with the shared edge...
    assert (0, 1, 2) not in sc.get_simplices()
    assert (0, 1, 3) not in sc.get_simplices()
    # ...and the faces of the removed edge stayed.
    assert (0,) in sc.get_simplices()
    assert (1,) in sc.get_simplices()


def test_removing_a_vertex_removes_everything_containing_it():
    """The blast radius the docstring warns about, made explicit."""
    sc = _incident_complex()
    sc.remove_simplex((0,))
    assert all(0 not in s for s in sc.get_simplices())
    assert _is_closed(sc)


def test_remove_empty_simplex_raises():
    """The empty set is a subset of every simplex, so a naive cascade would
    silently wipe the whole complex.  Refuse it instead."""
    sc = _incident_complex()
    for empty in ((), []):
        with pytest.raises(ValueError, match="empty"):
            sc.remove_simplex(empty)
    assert set(sc.get_simplices()) == _closure(_INCIDENT)  # untouched


def test_removing_an_absent_simplex_is_a_noop():
    sc = _incident_complex()
    before = set(sc.get_simplices())
    sc.remove_simplex((7, 8))
    assert set(sc.get_simplices()) == before


# ── Euler characteristic: the load-bearing oracle ────────────────────────


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_complete_complex_counts_are_binomial(n):
    """A complete complex on n vertices has C(n, k+1) simplices of order k."""
    sc = generate_complete_simplicial_complex(n)
    for order in range(n):
        got = sum(1 for s in sc.get_simplices() if len(s) - 1 == order)
        assert got == math.comb(n, order + 1)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_complete_complex_is_contractible(n):
    """A full simplex is contractible, so chi = 1 — a published invariant."""
    assert _euler_characteristic(generate_complete_simplicial_complex(n)) == 1


def test_hollow_triangle_has_euler_characteristic_zero():
    """The boundary of a triangle is homotopy equivalent to a circle: chi = 0.
    Three vertices and three edges, and crucially NO 2-simplex."""
    sc = SimplicialComplex([(0, 1), (1, 2), (0, 2)])
    assert (0, 1, 2) not in sc.get_simplices()
    assert _euler_characteristic(sc) == 0


def test_filling_and_unfilling_a_sphere_moves_chi_between_published_values():
    """Add a simplex whose faces are all already present, then remove it.

    The boundary of a tetrahedron is a 2-sphere (chi = 2).  Filling it in makes
    a solid ball, which is contractible (chi = 1).  Removing the filling gives
    the sphere back.  Both endpoints are published invariants, and the round
    trip only closes if the removal takes the 3-simplex and nothing else.
    """
    sphere = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    sc = SimplicialComplex(sphere)
    assert _euler_characteristic(sc) == 2  # S^2
    assert (0, 1, 2, 3) not in sc.get_simplices()

    sc.add_simplex((0, 1, 2, 3))
    assert _euler_characteristic(sc) == 1  # solid ball, contractible

    sc.remove_simplex((0, 1, 2, 3))
    assert _euler_characteristic(sc) == 2
    assert set(sc.get_simplices()) == _closure(sphere)


def test_add_then_remove_is_not_an_inverse_when_faces_were_missing():
    """The semantics, stated: ``add_simplex`` closes downward and
    ``remove_simplex`` cascades upward, so they only invert each other when
    the added simplex brought no new faces with it.  Faces introduced by the
    add legitimately survive the remove — that is the same guarantee as
    ``test_remove_simplex``, one dimension up."""
    sc = _incident_complex()
    sc.add_simplex((0, 1, 2, 3))
    sc.remove_simplex((0, 1, 2, 3))
    assert (0, 1, 2, 3) not in sc.get_simplices()
    assert (0, 2, 3) in sc.get_simplices()  # introduced by the add, and kept
    assert set(sc.get_simplices()) > _closure(_INCIDENT)  # strictly larger
    assert _is_closed(sc)


# ── Renderer ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "simplices",
    [
        [(0, 1, 2)],
        [(0, 1, 2), (2, 3, 4, 5), (0, 5)],
        [(0,), (1,), (2,), (3,), (0, 1), (1, 2), (2, 3), (0, 1, 2)],
        [(0, 1), (1, 2), (0, 2)],
    ],
)
def test_plot_draws_one_artist_per_simplex(simplices):
    """One filled patch per facet, one line per edge, one labelled point per
    vertex.  Counts come from the simplex set, recomputed here."""
    sc = SimplicialComplex(simplices)
    present = sc.get_simplices()
    n_vertices = sum(1 for s in present if len(s) == 1)
    n_edges = sum(1 for s in present if len(s) == 2)
    n_facets = len(_facets(present))

    fig, ax = sc.plot(return_fig=True)
    try:
        assert len(ax.patches) == n_facets
        assert len(ax.lines) == n_edges
        assert len(ax.collections) == 1  # one scatter carrying every vertex
        assert len(ax.collections[0].get_offsets()) == n_vertices
        assert len(ax.texts) == n_vertices  # labelled
    finally:
        plt.close(fig)


def test_plot_fills_only_facets_not_their_faces():
    """A tetrahedron reads as one body, not four separate triangular faces:
    the four 2-faces are present in the complex but must not be filled."""
    sc = SimplicialComplex([(0, 1, 2, 3)])
    assert sum(1 for s in sc.get_simplices() if len(s) == 3) == 4  # faces are there
    fig, ax = sc.plot(return_fig=True)
    try:
        assert len(ax.patches) == 1
    finally:
        plt.close(fig)


def test_plot_hatches_order_three_but_not_order_two():
    """The mark that separates a 3-simplex indicator from a real 2-simplex:
    a plain triangle beside a hatched tetrahedron."""
    sc = SimplicialComplex([(0, 1, 2), (3, 4, 5, 6)])
    fig, ax = sc.plot(return_fig=True)
    try:
        hatches = sorted((p.get_hatch() or "") for p in ax.patches)
        assert hatches[0] == "", "the order-2 face must be plain"
        assert hatches[-1] != "", "the order-3 body must be hatched"
    finally:
        plt.close(fig)


def test_plot_refuses_above_max_order():
    """A 4-simplex has no faithful 2D drawing and no meaningful indicator."""
    from qarp.graphs._simplicial_complex import MAX_PLOT_ORDER

    sc = generate_complete_simplicial_complex(MAX_PLOT_ORDER + 2)
    assert sc.max_dimension() == MAX_PLOT_ORDER + 1
    with pytest.raises(ValueError, match="no faithful"):
        sc.plot()


def test_plot_accepts_exactly_max_order():
    """Boundary: order 3 draws, order 4 does not."""
    from qarp.graphs._simplicial_complex import MAX_PLOT_ORDER

    sc = generate_complete_simplicial_complex(MAX_PLOT_ORDER + 1)
    assert sc.max_dimension() == MAX_PLOT_ORDER
    fig, _ = sc.plot(return_fig=True)
    plt.close(fig)


def test_plot_handles_the_empty_complex():
    sc = SimplicialComplex([])
    fig, ax = sc.plot(return_fig=True)
    try:
        assert len(ax.patches) == 0
        assert len(ax.lines) == 0
    finally:
        plt.close(fig)


def test_seeded_layout_is_reproducible():
    """Round trip, additional to the oracles above and never the sole check
    (conventions section 18): the same seed must place vertices identically."""
    simplices = [(0, 1, 2), (2, 3, 4, 5), (0, 5)]
    previous = config.seed
    try:
        config.seed = 7
        fig_a, ax_a = SimplicialComplex(simplices).plot(return_fig=True)
        fig_b, ax_b = SimplicialComplex(simplices).plot(return_fig=True)
        assert (ax_a.collections[0].get_offsets() == ax_b.collections[0].get_offsets()).all()
        plt.close(fig_a)
        plt.close(fig_b)
    finally:
        config.seed = previous


# ── The hull helper, including the degeneracies its docstring promises ───


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        # Hand-checked geometry is the oracle; scipy's Qhull is not usable as
        # one here because it raises on the last two cases.
        ([(0, 0), (1, 0), (1, 1), (0, 1)], [(0, 0), (1, 0), (1, 1), (0, 1)]),
        ([(0, 0), (4, 0), (2, 4), (2, 1)], [(0, 0), (4, 0), (2, 4)]),  # interior dropped
        ([(0, 0), (1, 1), (2, 2), (3, 3)], [(0, 0), (3, 3)]),  # collinear
        ([(1, 1), (1, 1), (1, 1)], [(1, 1)]),  # coincident
        ([(0, 0), (1, 1)], [(0, 0), (1, 1)]),
        ([(5, 5)], [(5, 5)]),
    ],
)
def test_convex_hull(points, expected):
    from qarp.graphs._simplicial_complex import _convex_hull

    assert _convex_hull(points) == expected


# ── The dependency the class no longer has ───────────────────────────────


def test_module_has_no_hypernetx_dependency():
    """Pins the gate lift: with ``plot`` off hypernetx, the module must not
    import it at any level, so the class works on a base install.  AST rather
    than ``sys.modules``, because importing ``qarp.graphs`` still pulls
    hypernetx in for the (still gated) Hypergraph class."""
    import qarp.graphs._simplicial_complex as module

    tree = ast.parse(inspect.getsource(module))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "hypernetx" not in imported, f"simplicial_complex still imports hypernetx: {imported}"


# ── Accessors, counts and the container protocol (hand-worked complex) ───

# Two triangles sharing (1, 2), a pendant edge (3, 4), an isolated vertex 7.
_WORKED = [(0, 1, 2), (1, 2, 3), (3, 4), (7,)]


def _worked():
    return SimplicialComplex(_WORKED)


def test_vertices_are_sorted_labels_not_tuples():
    assert _worked().vertices() == [0, 1, 2, 3, 4, 7]
    assert SimplicialComplex([]).vertices() == []


def test_cofaces_is_the_closed_star_and_exactly_what_remove_simplex_drops():
    sc = _worked()
    assert sc.cofaces((1, 2)) == [(1, 2), (0, 1, 2), (1, 2, 3)]
    assert sc.cofaces([2, 1]) == sc.cofaces((1, 2))  # canonicalised like add_simplex
    assert sc.cofaces((0, 3)) == []  # absent simplex: no cofaces
    before = set(sc.get_simplices())
    dropped = set(sc.cofaces((1, 2)))
    sc.remove_simplex((1, 2))
    assert before - set(sc.get_simplices()) == dropped


def test_skeleton_keeps_simplices_up_to_order_k():
    sc = _worked()
    one = sc.skeleton(1)
    assert set(one.get_simplices()) == {s for s in _closure(_WORKED) if len(s) <= 2}
    assert _is_closed(one)
    zero = sc.skeleton(0)
    assert set(zero.get_simplices()) == {(v,) for v in sc.vertices()}
    # k >= max_dimension: a copy, not the same object.
    full = sc.skeleton(sc.max_dimension())
    assert set(full.get_simplices()) == set(sc.get_simplices())
    assert full is not sc and full.simplices is not sc.simplices


def test_num_simplices_and_f_vector_are_the_hand_counts():
    sc = _worked()
    # f_0 = 6 vertices; f_1 = (0,1),(0,2),(1,2),(1,3),(2,3),(3,4); f_2 = 2 triangles.
    assert sc.f_vector() == [6, 6, 2]
    assert sc.num_simplices(0) == 6 and sc.num_simplices(1) == 6 and sc.num_simplices(2) == 2
    assert sc.num_simplices(3) == 0
    assert sc.num_simplices() == 14 == len(sc)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_f_vector_of_the_complete_complex_is_binomial(n):
    assert generate_complete_simplicial_complex(n).f_vector() == [
        math.comb(n, k + 1) for k in range(n)
    ]


def test_container_protocol():
    sc = _worked()
    assert len(sc) == sc.num_simplices()
    assert list(sc) == sc.get_simplices()
    assert (1, 2) in sc and [2, 1] in sc and (0, 1, 2) in sc
    assert (0, 3) not in sc and (5,) not in sc
    assert "not a simplex" not in sc
    empty = SimplicialComplex([])
    assert len(empty) == 0 and list(empty) == []


def test_to_graph_is_the_one_skeleton_as_a_native_graph():
    g = _worked().to_graph()
    assert isinstance(g, Graph)
    assert list(g.nodes) == [0, 1, 2, 3, 4, 7]  # sorted vertices, so label order
    assert set(g.edges) == {(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)}


def test_plot_lays_out_through_to_graph(monkeypatch):
    calls = []
    original = SimplicialComplex.to_graph

    def spy(self):
        calls.append(1)
        return original(self)

    monkeypatch.setattr(SimplicialComplex, "to_graph", spy)
    fig, _ = _worked().plot(return_fig=True)
    plt.close(fig)
    assert calls == [1]


# ── Euler characteristic: the method against the published values ────────


def test_euler_characteristic_method_matches_published_values():
    """chi(hollow triangle) = 0 (circle), chi(filled triangle) = 1 (contractible),
    chi(boundary of a tetrahedron) = 2 (sphere).  The test-side helper above
    stays as the independent recomputation."""
    hollow = SimplicialComplex([(0, 1), (1, 2), (0, 2)])
    filled = SimplicialComplex([(0, 1, 2)])
    sphere = SimplicialComplex([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    assert hollow.euler_characteristic() == _euler_characteristic(hollow) == 0
    assert filled.euler_characteristic() == _euler_characteristic(filled) == 1
    assert sphere.euler_characteristic() == _euler_characteristic(sphere) == 2
    assert _worked().euler_characteristic() == _euler_characteristic(_worked()) == 6 - 6 + 2


def test_empty_complex_has_empty_f_vector_and_zero_chi():
    empty = SimplicialComplex([])
    assert empty.f_vector() == []
    assert empty.euler_characteristic() == 0
    assert empty.max_dimension() == -1


# ── Boundary matrices and Hodge Laplacians ───────────────────────────────


def test_boundary_one_of_the_hollow_triangle_is_hand_written():
    """Rows (0,), (1,), (2,); columns (0, 1), (0, 2), (1, 2).  Deleting the i-th
    vertex of a sorted edge gives its faces with sign (-1)^i: column (0, 1) is
    +(1,) - (0,)."""
    sc = SimplicialComplex([(0, 1), (1, 2), (0, 2)])
    expected = np.array([[-1, -1, 0], [1, 0, -1], [0, 1, 1]])
    assert np.array_equal(sc.boundary_matrix(1).toarray(), expected)


def test_boundary_two_of_the_filled_triangle_is_hand_written():
    """Column (0, 1, 2) against rows (0, 1), (0, 2), (1, 2): faces (1, 2) at i = 0,
    (0, 2) at i = 1, (0, 1) at i = 2 with signs +, -, +."""
    sc = SimplicialComplex([(0, 1, 2)])
    assert np.array_equal(sc.boundary_matrix(2).toarray(), np.array([[1], [-1], [1]]))


@pytest.mark.parametrize(
    "simplices",
    [
        [(0, 1, 2)],
        [(0, 1, 2, 3)],
        _WORKED,
        [(0, 1, 2), (2, 3, 4, 5), (0, 5)],
    ],
)
def test_boundary_of_a_boundary_is_zero(simplices):
    sc = SimplicialComplex(simplices)
    for k in range(1, sc.max_dimension() + 1):
        product = sc.boundary_matrix(k) @ sc.boundary_matrix(k + 1)
        assert product.shape == (sc.num_simplices(k - 1), sc.num_simplices(k + 1))
        assert product.count_nonzero() == 0


def test_boundary_matrix_end_cases_need_no_special_casing():
    sc = _worked()
    dim = sc.max_dimension()
    assert sc.boundary_matrix(0).shape == (0, sc.num_simplices(0))
    assert sc.boundary_matrix(dim + 1).shape == (sc.num_simplices(dim), 0)
    assert sc.boundary_matrix(dim + 2).shape == (0, 0)
    assert sc.hodge_laplacian(dim).shape == (sc.num_simplices(dim),) * 2


def test_hodge_laplacian_zero_is_the_graph_laplacian_of_the_one_skeleton():
    """Independent oracle: networkx's Laplacian of the 1-skeleton, with nodelist
    pinned to the vertex order (networkx otherwise orders by insertion)."""
    sc = SimplicialComplex([(0, 1, 2), (2, 3, 4, 5), (0, 5), (7,)])
    L = nx.laplacian_matrix(sc.to_graph(), nodelist=sc.vertices()).toarray()
    assert np.array_equal(sc.hodge_laplacian(0).toarray(), L)


@pytest.mark.parametrize("simplices", [_WORKED, [(0, 1, 2, 3)], [(0, 1, 2), (2, 3, 4, 5), (0, 5)]])
def test_hodge_laplacians_are_symmetric_positive_semidefinite(simplices):
    sc = SimplicialComplex(simplices)
    for k in range(sc.max_dimension() + 1):
        L = sc.hodge_laplacian(k).toarray()
        assert np.array_equal(L, L.T)
        assert np.linalg.eigvalsh(L.astype(float)).min() > -1e-12


def test_hodge_laplacian_one_of_the_filled_triangle():
    """Hand-checked: L_1 = d1^T d1 + d2 d2^T with the two matrices above."""
    sc = SimplicialComplex([(0, 1, 2)])
    d1 = np.array([[-1, -1, 0], [1, 0, -1], [0, 1, 1]])
    d2 = np.array([[1], [-1], [1]])
    assert np.array_equal(sc.hodge_laplacian(1).toarray(), d1.T @ d1 + d2 @ d2.T)
