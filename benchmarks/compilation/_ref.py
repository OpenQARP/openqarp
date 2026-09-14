"""The uncompiled fixture — oracle and baseline column.

Compiles nothing: it reports the circuit as written, so its row is the
pre-routing cost every other column is trying to beat, and its fingerprint is
what every compiled circuit must reproduce (§18).  Its coupling violations are
expected and meaningless — an unrouted circuit is not claiming to fit the
device — so the family kernel marks this stack unrouted and the table renders
that cell as n/a rather than as a failure.
"""

LABEL = "reference"
ROUTED = False


def prepare(n: int, ops: list, topology: str):
    return list(ops)


def compile_circuit(n: int, prepared):
    return prepared


def to_ops(n: int, compiled) -> list:
    return list(compiled)


def layout_of(compiled) -> None:
    return None
