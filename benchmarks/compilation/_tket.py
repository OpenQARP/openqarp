"""pytket adapter: FullPeepholeOptimise -> DefaultMappingPass -> rebase.

Two conventions differ from every other stack and are converted here, at the
boundary, never inside a measured comparison:

- **Angles are half-turns** (§2): pytket's `Rz(a)` is `exp(-i a pi Z / 2)`, so
  radians are divided by pi going in and multiplied coming out.
- **Routing permutations are implicit.**  pytket records them as a qubit
  relabelling on the circuit rather than as gates, so the layout is read from
  `implicit_qubit_permutation()`.
"""

import math

LABEL = "tket"


def _optypes():
    from pytket import OpType

    return {OpType.H, OpType.Rx, OpType.Ry, OpType.Rz, OpType.CX, OpType.SWAP}


def prepare(n: int, ops: list, topology: str):
    from pytket import Circuit
    from pytket.architecture import Architecture

    from benchmarks.compilation import architectures

    circuit = Circuit(n)
    for op in ops:
        if op[0] == "h":
            circuit.H(op[1])
        elif op[0] == "cx":
            circuit.CX(op[1], op[2])
        elif op[0] == "swap":
            circuit.SWAP(op[1], op[2])
        else:
            getattr(circuit, op[0].capitalize())(op[2] / math.pi, op[1])
    return circuit, Architecture(architectures.edges(topology, n))


def compile_circuit(n: int, prepared):
    """Compile through a CompilationUnit, which is what records the placement.

    Applying the pass straight to a Circuit throws away the initial mapping,
    leaving no way to undo the placement permutation — the compiled circuit
    then looks wrong even on fixtures that need no SWAPs at all.
    """
    from pytket.passes import AutoRebase, DefaultMappingPass, FullPeepholeOptimise, SequencePass
    from pytket.predicates import CompilationUnit

    circuit, arch = prepared
    unit = CompilationUnit(circuit.copy())
    SequencePass([FullPeepholeOptimise(), DefaultMappingPass(arch), AutoRebase(_optypes())]).apply(
        unit
    )
    return unit


def to_ops(n: int, compiled) -> list:
    circuit = compiled.circuit
    index = {qubit: qubit.index[0] for qubit in circuit.qubits}
    out: list = []
    for command in circuit:
        name = command.op.type.name.lower()
        qubits = [index[q] for q in command.qubits]
        params = [float(p) * math.pi for p in command.op.params]
        if name in ("cx", "swap"):
            out.append((name, qubits[0], qubits[1]))
        elif name == "h":
            out.append(("h", qubits[0]))
        elif name in ("rx", "ry", "rz"):
            out.append((name, qubits[0], params[0]))
        elif name in ("barrier",):
            continue
        else:
            raise ValueError(f"gate outside the target basis: {name}")
    return out


def layout_of(compiled) -> list:
    """Logical -> physical, matching qarp's `final_logical_to_physical`.

    `final_map` composes placement with the mapped wire labels, while SWAPs
    retained in the circuit contribute its implicit permutation as well.
    """
    layout = list(range(len(compiled.circuit.qubits)))
    implicit = compiled.circuit.implicit_qubit_permutation()
    for source, target in compiled.final_map.items():
        layout[source.index[0]] = implicit.get(target, target).index[0]
    return layout


def initial_layout_of(compiled) -> list:
    """Placement recorded by the CompilationUnit before routing."""
    layout = list(range(len(compiled.circuit.qubits)))
    for source, target in compiled.initial_map.items():
        layout[source.index[0]] = target.index[0]
    return layout
