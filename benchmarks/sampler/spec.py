"""Workload registry for the sampler track.

Deliberately smaller than the statevector track: shot loops, not the
statevector, are the cost here, so the ladder stops at n = 16 and the default
shot count is modest.  `shots_axis` holds n fixed and sweeps shots instead —
the two axes never multiply.

Circuits are the statevector track's, unchanged, so the two documents answer
"what does giving up exact amplitudes cost?" on the same workloads.

The oracle stack is `exact`: Born probabilities from the numpy reference, no
sampling at all.  Sampled rows are checked against exact physics with a
shot-noise tolerance (§18) — never against another sampler's draw.
"""

ORACLE_STACK = "exact"
CHECK_RTOL = 1e-9

SHOTS = 8192
SHOTS_AXIS_QUBITS = 12

SINGLE_PRECISION = {"qsim"}
SINGLE_PRECISION_RTOL = 1e-5

_ALL = ["exact", "qarpx", "qulacs", "aer", "lightning", "qsim"]


def rtol_for(stack: str) -> float:
    return SINGLE_PRECISION_RTOL if stack in SINGLE_PRECISION else CHECK_RTOL


FAMILIES: dict[str, dict] = {
    "brickwork_sample": {
        "sizes": [4, 8, 12, 16],
        "headroom": [20],
        "smoke": 4,
        "stacks": _ALL,
    },
    "trotter_sample": {
        "sizes": [4, 8, 12, 16],
        "headroom": [20],
        "smoke": 4,
        "stacks": _ALL,
    },
    "qpe_sample": {
        "sizes": [6, 10, 14],
        "headroom": [16],
        "smoke": 6,
        "stacks": _ALL,
    },
    # "size" is the shot count here, at a fixed SHOTS_AXIS_QUBITS-qubit circuit.
    "shots_axis": {
        "sizes": [1_024, 8_192, 65_536],
        "headroom": [524_288],
        "smoke": 256,
        "stacks": _ALL,
    },
}
