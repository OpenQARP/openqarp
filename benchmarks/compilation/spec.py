"""Workload registry for the compilation track.

A family is one circuit shape on one topology, because the runner sweeps a
single size axis and the device is not a size.  Sizes are qubit counts.

Unlike the other tracks the headline is *quality*, not time: fewer CX-equivalent
two-qubit gates and less depth is better, and compile wall time is secondary
(a compiler is allowed to be slower if it produces a materially better
circuit).  The stacks are meant to disagree, so the cross-stack check is not
their outputs but the state each compiled circuit produces — every column must
still compute what the `reference` column computes.
"""

ORACLE_STACK = "reference"
CHECK_RTOL = 1e-9

_ALL = ["reference", "qarpx", "qiskit1", "qiskit2", "qiskit3", "tket"]
_SHAPES = ("trotter", "hea", "qv", "mqt")
_TOPOLOGIES = ("line", "grid")

# Equivalence is verified by simulating the compiled circuit, so the ladder is
# capped where that stays cheap; routing quality is already well separated at
# these widths (the router benchmark settled its defaults at n <= 16).
SIZES = [8, 12, 14]
HEADROOM = [16, 18]

FAMILIES: dict[str, dict] = {
    f"{shape}_{topology}": {
        "sizes": SIZES,
        "headroom": HEADROOM,
        "smoke": 6,
        "stacks": _ALL,
        "shape": shape,
        "topology": topology,
    }
    for shape in _SHAPES
    for topology in _TOPOLOGIES
}
