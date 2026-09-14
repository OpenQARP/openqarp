"""qiskit `transpile` — shared implementation for the O1/O2/O3 columns.

Given the same edge list and the same target basis as every other stack, so
the two-qubit counts are comparable.  `seed_transpiler` is pinned: qiskit's
layout and routing passes are stochastic above level 0, and an unpinned seed
would make the table unreproducible.
"""

from benchmarks.compilation import architectures, inputs

SEED_TRANSPILER = 20260824
BASIS = ["h", "rx", "ry", "rz", "cx", "swap"]


def prepare(n: int, ops: list, topology: str):
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(n)
    for op in ops:
        if op[0] == "h":
            circuit.h(op[1])
        elif op[0] in architectures.TWO_QUBIT:
            getattr(circuit, op[0])(op[1], op[2])
        else:
            getattr(circuit, op[0])(op[2], op[1])
    coupling = [list(e) for e in architectures.edges(topology, n)]
    coupling += [[b, a] for a, b in coupling]
    return circuit, coupling


def compile_at(n: int, prepared, level: int):
    from qiskit import transpile

    circuit, coupling = prepared
    return transpile(
        circuit,
        coupling_map=coupling,
        basis_gates=BASIS,
        optimization_level=level,
        seed_transpiler=SEED_TRANSPILER,
    )


def to_ops(n: int, compiled) -> list:
    return inputs.from_qiskit(compiled)


def layout_of(compiled) -> list | None:
    """Logical -> physical wire, matching qarp's `final_logical_to_physical`."""
    layout = getattr(compiled, "layout", None)
    if layout is None:
        return None
    return list(layout.final_index_layout())


def initial_layout_of(compiled) -> list:
    """Which physical wire each logical qubit starts on (see `_qk` docstring)."""
    layout = getattr(compiled, "layout", None)
    return None if layout is None else list(layout.initial_index_layout())
