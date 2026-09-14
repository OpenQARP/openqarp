"""Coupling maps, defined once and handed to every stack.

The single most likely way to make this track unfair is to give one compiler
an easier device than another, so the edge set lives here and every adapter is
built from *this* list — and every compiled circuit is checked back against it
in `verify`, not against whatever map the SDK thinks it was given.

Edges are undirected and stored as sorted pairs; directedness is out of scope
(all three compilers are given symmetric maps).
"""


def line(n: int) -> list[tuple[int, int]]:
    return [(q, q + 1) for q in range(n - 1)]


def ring(n: int) -> list[tuple[int, int]]:
    if n < 3:
        return line(n)
    return line(n) + [(0, n - 1)]


def grid(n: int) -> list[tuple[int, int]]:
    """Squarest rectangle holding n qubits; the leftover cells are unused."""
    rows = int(n**0.5)
    while rows > 1 and n % rows:
        rows -= 1
    cols = n // rows
    edges = []
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            if c + 1 < cols:
                edges.append((q, q + 1))
            if r + 1 < rows:
                edges.append((q, q + cols))
    return sorted(edges)


TOPOLOGIES = {"line": line, "ring": ring, "grid": grid}


def edges(topology: str, n: int) -> list[tuple[int, int]]:
    return sorted(tuple(sorted(e)) for e in TOPOLOGIES[topology](n))


def grid_dims(n: int) -> tuple[int, int]:
    rows = int(n**0.5)
    while rows > 1 and n % rows:
        rows -= 1
    return rows, n // rows


TWO_QUBIT = {"cx", "swap"}


def violations(ops, topology: str, n: int) -> int:
    """Two-qubit gates in a compiled circuit that straddle a non-edge."""
    allowed = set(edges(topology, n))
    return sum(
        1 for op in ops if op[0] in TWO_QUBIT and tuple(sorted((op[1], op[2]))) not in allowed
    )
