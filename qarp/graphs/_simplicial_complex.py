from itertools import combinations
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_array

from .. import config
from ._graph import Graph

Simplex = Sequence[int]  # Union[List[int], Tuple[int, ...]]

# Above this order a simplex has no faithful 2D drawing, and no meaningful
# indicator either, so `plot` refuses rather than draw something misleading.
MAX_PLOT_ORDER = 3

# Hatching separates an order-3 body (an indicator of where the simplex is,
# not a depiction of it) from a real order-2 face.  Only order 3 is reachable
# while MAX_PLOT_ORDER is 3; raising the cap means extending this table.
_HATCH = {3: "//"}
# Fills stay pale on purpose: matplotlib draws hatching in the edge colour,
# and a dark fill swallows it.
_INK = "#1f3b63"
_VERTEX_INK = "#12263f"


def _convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Convex hull of a small planar point set (Andrew's monotone chain).

    Hand-rolled rather than scipy's Qhull, which raises on the degenerate
    (collinear or coincident) sets a spring layout can produce for a handful
    of vertices.
    """
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


class SimplicialComplex:
    def __init__(self, simplices: Sequence[Simplex]):
        """Simplicial complex for representing higher-order topological structures.

        SimplicialComplex represents a mathematical structure consisting of simplices (points,
        edges, triangles, tetrahedra, etc.) and all their faces. It automatically ensures closure
        by adding all lower-dimensional faces when a simplex is added, and by removing all
        higher-dimensional cofaces when one is removed. This structure is useful
        for topological data analysis, studying higher-order networks, and defining quantum states
        with complex correlation patterns beyond pairwise interactions.  Beyond the closure
        bookkeeping it offers accessors (``vertices``, ``neighbors``, ``cofaces``,
        ``skeleton``, counts and the container protocol), the Euler characteristic, and
        the sparse boundary matrices and combinatorial Hodge Laplacians; their spectra are
        left to the caller.

        Args:
            simplices: A sequence of simplices, where each simplex is a list or tuple of node indices.
                When a simplex is added, all its faces (lower-dimensional sub-simplices) are automatically
                included to maintain the closure property of simplicial complexes.
        """
        self.simplices: set[Simplex] = set()
        for simplex in simplices:
            self.add_simplex(simplex)

    def __str__(self):
        return f"Simplicial complex with {len(self.simplices)} simplices."

    def __repr__(self):
        return self.__str__()

    def add_simplex(self, simplex: Simplex) -> None:
        """
        Adds a simplex and all its faces to the simplicial complex.

        Args:
            simplex (Simplex): The simplex to add.
        """
        if not isinstance(simplex, (list, tuple)):
            raise ValueError("Simplex must be a list or tuple.")
        simplex = tuple(sorted(simplex))
        for k in range(1, len(simplex) + 1):
            for face in combinations(simplex, k):
                self.simplices.add(tuple(sorted(face)))

    def remove_simplex(self, simplex: Simplex) -> None:
        """
        Removes a simplex, and every simplex containing it, from the complex.

        Cofaces have to go too: a complex that kept (0, 1, 2) after (0, 1) was
        removed would no longer be closed under faces.  Faces of ``simplex``
        are left alone, so removing (0, 1) does not remove (0,) or (1,).

        The blast radius is the entire coface set, which can be most of the
        complex — removing a vertex removes every simplex it belongs to.

        Args:
            simplex (Simplex): The simplex to remove.

        Raises:
            ValueError: If ``simplex`` is not a list or tuple, or is empty.
                The empty simplex is a face of every simplex, so cascading on
                it would silently empty the complex.
        """
        if not isinstance(simplex, (list, tuple)):
            raise ValueError("Simplex must be a list or tuple.")
        if len(simplex) == 0:
            raise ValueError(
                "Cannot remove the empty simplex: it is a face of every simplex, "
                "so removing its cofaces would empty the complex."
            )
        target = set(simplex)
        self.simplices -= {s for s in self.simplices if target.issubset(s)}

    # ── container protocol ─────────────────────────────────────────────

    def __len__(self) -> int:
        """Number of simplices, all orders (``num_simplices()``)."""
        return len(self.simplices)

    def __iter__(self) -> Iterator[Tuple[int, ...]]:
        """Iterate over ``get_simplices()``: sorted by ``(order, vertices)``."""
        return iter(self.get_simplices())

    def __contains__(self, simplex) -> bool:
        """Membership, canonicalised like ``add_simplex`` (a list is accepted,
        vertex order is irrelevant); anything else is simply absent."""
        if not isinstance(simplex, (list, tuple)):
            return False
        return tuple(sorted(simplex)) in self.simplices

    # ── accessors ──────────────────────────────────────────────────────

    def vertices(self) -> List[int]:
        """The 0-simplices, as sorted vertex labels (not 1-tuples)."""
        return sorted(s[0] for s in self.simplices if len(s) == 1)

    def neighbors(self, node) -> List:
        """Vertices sharing a simplex with ``node``, sorted.  Under closure this
        is the 1-skeleton neighbourhood: every co-member shares an edge too."""
        neighbors: set[int] = set()
        for simplex in self.simplices:
            if node in simplex:
                neighbors.update(n for n in simplex if n != node)
        return sorted(neighbors)

    def cofaces(self, simplex: Simplex) -> List[Tuple[int, ...]]:
        """Every simplex having ``simplex`` as a face — its closed star, so a present
        ``simplex`` is included and an absent one has no cofaces.  Exactly the set
        ``remove_simplex`` deletes.  Sorted like ``get_simplices``."""
        target = set(simplex)
        return sorted(
            (tuple(s) for s in self.simplices if target.issubset(s)), key=lambda s: (len(s), s)
        )

    def skeleton(self, k: int) -> "SimplicialComplex":
        """The k-skeleton: a new complex of the simplices of order at most ``k``
        (closed automatically).  ``k >= max_dimension()`` gives a copy."""
        return SimplicialComplex([s for s in self.simplices if len(s) - 1 <= k])

    def num_simplices(self, order: Optional[int] = None) -> int:
        """Number of simplices of the given order, or of all orders when ``None``."""
        if order is None:
            return len(self.simplices)
        return sum(1 for s in self.simplices if len(s) - 1 == order)

    def f_vector(self) -> List[int]:
        """Simplex counts per order, ``[f_0, …, f_dim]``; ``[]`` for the empty complex."""
        return [self.num_simplices(k) for k in range(self.max_dimension() + 1)]

    def to_graph(self) -> Graph:
        """The 1-skeleton as a native ``Graph``: nodes are the vertices (sorted, so
        insertion order is label order), edges the 1-simplices."""
        graph = Graph()
        graph.add_nodes_from(self.vertices())
        graph.add_edges_from(s for s in self.get_simplices() if len(s) == 2)
        return graph

    # ── cheap topology: invariant and operators, no spectra ────────────

    def euler_characteristic(self) -> int:
        """``χ = Σ_k (−1)^k f_k``, the alternating sum of the f-vector; 0 when empty."""
        return sum((-1) ** k * f_k for k, f_k in enumerate(self.f_vector()))

    def boundary_matrix(self, k: int):
        """Signed boundary operator ``∂_k`` from k-simplices to (k−1)-simplices.

        Rows index the (k−1)-simplices and columns the k-simplices, both in
        ``get_simplices()`` order.  The entry for the face obtained by deleting the
        i-th vertex of a (sorted) k-simplex is ``(−1)^i``.  ``∂_0`` is ``0 × f_0``
        and ``∂_{dim+1}`` is ``f_dim × 0``, so ``hodge_laplacian`` needs no end cases.

        Returns:
            SciPy sparse array of ints. Append `.toarray()` for dense representation.
        """
        rows = [tuple(s) for s in self.get_simplices() if len(s) == k]
        cols = [tuple(s) for s in self.get_simplices() if len(s) == k + 1]
        row_index: dict = {s: i for i, s in enumerate(rows)}
        entries_r, entries_c, entries_v = [], [], []
        # k = 0: a vertex's only "face" is the empty simplex, which is not a row.
        for j, simplex in enumerate(cols if k > 0 else []):
            for i in range(len(simplex)):
                face = simplex[:i] + simplex[i + 1 :]
                entries_r.append(row_index[face])
                entries_c.append(j)
                entries_v.append((-1) ** i)
        return coo_array(
            (np.asarray(entries_v, dtype=np.int64), (entries_r, entries_c)),
            shape=(len(rows), len(cols)),
        ).tocsr()

    def hodge_laplacian(self, k: int):
        """Combinatorial Hodge Laplacian ``L_k = ∂_kᵀ ∂_k + ∂_{k+1} ∂_{k+1}ᵀ`` on the
        k-simplices (``f_k × f_k``).  ``L_0`` is the graph Laplacian of the
        1-skeleton.  The matrix only — its spectrum is the caller's business.

        Returns:
            SciPy sparse array of ints. Append `.toarray()` for dense representation.
        """
        lower = self.boundary_matrix(k)
        upper = self.boundary_matrix(k + 1)
        return lower.T @ lower + upper @ upper.T

    def get_simplices(self) -> List[Tuple[int]]:
        """
        Returns a sorted list of simplices in the complex.

        Returns:
            List[Tuple[int]]: The list of simplices.
        """
        return sorted(
            self.simplices,  # type: ignore[arg-type]  # mypy's false positive
            key=lambda s: (len(s), s),
        )

    def max_dimension(self) -> int:
        """
        Returns the dimension of the simplicial complex.

        Returns:
            int: The highest dimension (i.e., max(len(simplex) - 1)).
        """
        return max((len(s) - 1 for s in self.simplices), default=-1)

    def plot(self, figsize=(4, 3), return_fig=False):
        """
        Plots the simplicial complex as points, edges and filled simplices.

        Only *facets* (maximal simplices) are filled, so a tetrahedron reads as
        one body rather than four separate faces.  Faces are implied by the fill
        and never separately outlined, which is what makes a missing face
        visible: the body stays filled and the line along the absent edge is
        simply not drawn.  Order 3 is hatched, marking where the simplex is
        rather than claiming to depict it.

        Vertices are placed by a spring layout over the 1-skeleton, seeded from
        ``qarp.config.seed`` — which defaults to ``None``, so figures are
        reproducible only once the caller sets it.

        Args:
            figsize (tuple): Figure size.
            return_fig (bool): If True, return (fig, ax) for saving or further customization.

        Returns:
            (fig, ax) if return_fig is True, else None.

        Raises:
            ValueError: If the complex has order above ``MAX_PLOT_ORDER``.
        """
        order = self.max_dimension()
        if order > MAX_PLOT_ORDER:
            raise ValueError(
                f"Cannot plot a complex of order {order}: no faithful drawing "
                f"exists above order {MAX_PLOT_ORDER}. Summarise it instead "
                "with max_dimension(), simplex counts per order, or the Euler "
                "characteristic."
            )

        # Lazy: keep matplotlib out of headless imports (mirrors blocks S-F).
        import matplotlib.pyplot as plt
        import networkx as nx
        from matplotlib import colormaps
        from matplotlib.patches import Polygon

        simplices = self.get_simplices()
        vertices = self.vertices()
        edges = [s for s in simplices if len(s) == 2]
        as_sets = [set(s) for s in simplices]
        fills = [
            tuple(sorted(s)) for s in as_sets if len(s) >= 3 and not any(s < t for t in as_sets)
        ]

        # Layout over the 1-skeleton, so vertices sit where their edges pull them.
        pos = nx.spring_layout(self.to_graph(), seed=config.seed)

        orders = sorted({len(s) - 1 for s in fills})
        cmap = colormaps["Blues"]
        shade = {d: cmap(0.10 + 0.30 * (i / max(len(orders) - 1, 1))) for i, d in enumerate(orders)}

        fig, ax = plt.subplots(figsize=figsize)
        for simplex in fills:
            d = len(simplex) - 1
            ax.add_patch(
                Polygon(
                    _convex_hull([tuple(pos[v]) for v in simplex]),
                    closed=True,
                    facecolor=shade[d],
                    edgecolor=_INK,
                    hatch=_HATCH.get(d),
                    linewidth=0.0,
                    # Transparency washes the hatching out, so fills are opaque
                    # and higher orders sit behind lower ones instead.
                    alpha=1.0,
                    zorder=MAX_PLOT_ORDER - d,
                )
            )
        for a, b in edges:
            ax.plot(
                [pos[a][0], pos[b][0]],
                [pos[a][1], pos[b][1]],
                color=_INK,
                linewidth=1.6,
                solid_capstyle="round",
                zorder=MAX_PLOT_ORDER + 1,
            )
        ax.scatter(
            [pos[v][0] for v in vertices],
            [pos[v][1] for v in vertices],
            s=110,
            color=_VERTEX_INK,
            zorder=MAX_PLOT_ORDER + 2,
        )
        for v in vertices:
            ax.annotate(
                str(v),
                pos[v],
                color="white",
                fontsize=7,
                ha="center",
                va="center",
                zorder=MAX_PLOT_ORDER + 3,
            )

        ax.set_aspect("equal")
        ax.axis("off")
        fig.tight_layout()
        if return_fig:
            return fig, ax
        else:
            plt.show()
            return None
