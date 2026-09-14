"""Compilation metrics and the equivalence check.

This track differs from the others: the headline is *quality* (two-qubit gate
count, depth, SWAPs) rather than time, and the stacks are supposed to produce
*different* circuits — so the cross-stack check cannot compare their outputs to
each other.  What every stack must agree on is that its circuit still computes
the same thing, so the check is the fingerprint of the state the compiled
circuit produces, with the routing permutation undone, compared against the
**uncompiled** fixture simulated by the numpy reference (§18).

Metrics are computed here from the canonical op list rather than read from each
SDK's own counters, so "depth" means one thing across the table.
"""

from benchmarks.common import agree  # noqa: F401  (re-exported per-track)
from benchmarks.compilation import architectures
from benchmarks.statevector import _np
from benchmarks.statevector import checks as sv_checks

TWO_QUBIT = architectures.TWO_QUBIT


def qubits_of(op) -> tuple[int, ...]:
    return (op[1], op[2]) if op[0] in TWO_QUBIT else (op[1],)


def depth(ops, n: int) -> int:
    """Circuit depth: gates pack into the earliest layer their qubits allow."""
    frontier = [0] * n
    for op in ops:
        qubits = qubits_of(op)
        layer = max(frontier[q] for q in qubits) + 1
        for q in qubits:
            frontier[q] = layer
    return max(frontier) if n else 0


def metrics(ops, n: int, topology: str) -> dict:
    """Quality metrics, all derived here so one definition covers the table.

    `cx_equivalent` is the comparable headline: a SWAP costs three CXs on
    hardware without a native SWAP, and the compilers disagree about whether
    to emit SWAP at all (qarp and pytket keep it, a rebase can expand it), so
    the raw two-qubit count is not comparable across columns on its own.
    """
    two_qubit = sum(1 for op in ops if op[0] in TWO_QUBIT)
    swaps = sum(1 for op in ops if op[0] == "swap")
    return {
        "gates": len(ops),
        "two_qubit": two_qubit,
        "swaps": swaps,
        "cx_equivalent": (two_qubit - swaps) + 3 * swaps,
        "depth": depth(ops, n),
        "coupling_violations": architectures.violations(ops, topology, n),
    }


def undo_layout(ops: list, layout) -> list:
    """Append SWAPs that return physical qubits to their logical positions.

    Routing leaves logical qubit i sitting at physical wire layout[i].  Rather
    than special-casing each SDK's layout object inside the comparison, the
    permutation is expressed as gates and simulated with everything else.
    """
    if layout is None:
        return list(ops)
    current = list(layout)
    out = list(ops)
    position = {physical: index for index, physical in enumerate(current)}
    for logical in range(len(current)):
        physical = current[logical]
        if physical == logical:
            continue
        other = position[logical]
        out.append(("swap", physical, logical))
        current[logical], current[other] = current[other], current[logical]
        position[current[logical]] = logical
        position[current[other]] = other
    return out


def equivalence_fingerprint(ops: list, n: int, layout=None) -> dict:
    """State fingerprint of a compiled circuit, routing permutation removed."""
    psi = _np.evolve(n, undo_layout(ops, layout))
    return sv_checks.state_fingerprint(psi, n)
