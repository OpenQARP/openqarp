"""Workload registry for the operators track.

One entry per workload family: published sizes, the tiny smoke size, and the
stacks that can express it (a stack absent here renders as n/a in the table,
never as a silent omission).  The oracle stack for every cross-stack check is
openfermion (§18) — each other stack's canonical summary must agree with it
at CHECK_RTOL.
"""

ORACLE_STACK = "openfermion"
CHECK_RTOL = 1e-6

_ALL = ["qarpx", "openfermion", "qiskit", "pennylane"]
# qiskit fermionic transforms live in qiskit-nature, which is not a [bench]
# dep — those cells are n/a by design, not missing implementations.
_NO_QISKIT = ["qarpx", "openfermion", "pennylane"]
# Symbolic coefficients: qiskit ParameterExpression cannot survive
# simplify()/product chains; pennylane PauliSentence coefficients are numeric.
_SYMBOLIC = ["qarpx", "openfermion"]

FAMILIES: dict[str, dict] = {
    "construct_string": {"sizes": [20_000, 100_000], "smoke": 200, "stacks": _ALL},
    "accumulate": {"sizes": [20_000, 100_000], "smoke": 200, "stacks": _ALL},
    "op_product": {"sizes": [100, 300], "smoke": 8, "stacks": _ALL},
    "commutator": {"sizes": [100, 300], "smoke": 8, "stacks": _ALL},
    "hermitian_conjugated": {"sizes": [100_000], "smoke": 200, "stacks": _ALL},
    "jw_molecular": {"sizes": [10, 20, 40], "smoke": 4, "stacks": _NO_QISKIT},
    "bk_molecular": {"sizes": [10, 20], "smoke": 4, "stacks": _NO_QISKIT},
    "parity_encode": {"sizes": [10, 16], "smoke": 4, "stacks": _NO_QISKIT},
    "sparse_construct": {"sizes": [12, 14], "smoke": 6, "stacks": _ALL},
    "symbolic_algebra": {"sizes": [500], "smoke": 12, "stacks": _SYMBOLIC},
    "memory_hold": {"sizes": [1_000_000], "smoke": 2_000, "stacks": _ALL},
}
